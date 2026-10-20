"""Diffusion-model converters: W4A8 codebook and W6A8 (comfy-kitchen
AsymW4A8Int8Layout), replacing the old ratio-based "W4A8/W6A8 + INT8 Mixed"
nodes. No int8 ratio / ranking: every eligible layer gets the chosen format.

Per-layer rules
  exclusion list (MODEL_TYPE_EXCLUSIONS)        -> kept at source precision
  2D float weight, K % 256 == 0 and K % group   -> chosen grouped format
  other 2D float weight                         -> int8 ConvRot with the
                                                   largest group (256/64/16)
                                                   dividing K (rule-based
                                                   fallback, as Kijai's
                                                   checkpoints do)
  everything else                               -> copied unchanged
  prune_donor (MiniMax H3)                      -> curve-form adaln tensors
                                                   + config merged from donor

Output: Kijai-style single '_quantization_metadata' header block, streamed
to disk tensor-by-tensor (RAM ~ one tensor; needs ~2x output size of free
disk while writing).

Error measurement (report only):
  weights            per layer: Rel-RMSE, SNR, Outlier %, per-channel MSE
                     (MSE/MAE/max kept as local-check values in the JSON)
  weights+timesteps  + a short sampling run with the connected calib_*
                     inputs. Every model call runs twice — reference and
                     fake-quantized (decoded weights + int8 ConvRot
                     activations) — on the same input (teacher forcing).
                     Tracks the prediction-noise variance per timestep,
                     per-layer local error per step, and the diagonal
                     Hessian used to weight each layer's weight distance.

method = gptq (needs the calib_* inputs): sampling runs with the connected
model collect each layer's input Hessian, then GPTQ re-chooses the codes
with error feedback (same file format — see gptq_quant.py). Sequential by
default: later passes see the already-quantized layers. The experimental
video node (VideoGPTQSave) is the same pipeline with video defaults
(token subsampling, stronger dampening, held-out validation per layer).
"""

import json
import logging
import os
import re

import torch
import torch.nn.functional as F

import folder_paths
import comfy.utils
import comfy.model_management
import comfy.model_detection

from .int8_unet_loader import MODEL_TYPE_EXCLUSIONS, MODEL_TYPE_INT8_KEEP
from .grouped_quant import (
    FORMAT_W4A8, FORMAT_W6A8, GROUPED_BITS,
    kitchen_support, grouped_spec, int8_spec,
    SourceReader, StreamingSafetensorsWriter,
    print_plan, execute_plan, map_modules, FakeQuantSession, comfy_compiler_off,
    summarize_local, log_summary, write_report,
)
from .gptq_quant import run_gptq, estimate as gptq_estimate


_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _flat(t):
    if torch.is_tensor(t):
        return t.float().flatten()
    if isinstance(t, (list, tuple)):
        return torch.cat([_flat(x) for x in t])
    raise TypeError(f"unsupported model output type {type(t)}")


def _build_plan(reader, donor, model_type, fmt, format_name, group, label, int8_keep=False):
    """Plan entries for every tensor of the (normalized, donor-merged) model."""
    excluded = MODEL_TYPE_EXCLUSIONS.get(model_type, [])
    keep8 = MODEL_TYPE_INT8_KEEP.get(model_type, []) if int8_keep else []
    metadata = dict(reader.metadata)

    if any(k.endswith(".comfy_quant") for k in reader.keys) or "_quantization_metadata" in metadata:
        raise RuntimeError(f"{label}: source already contains quantized layers — "
                           "select the original bf16/fp16 checkpoint.")

    # Native key names (mirror of the loader's prefix strip).
    pairs = [(k, k) for k in reader.keys]
    try:
        prefix = comfy.model_detection.unet_prefix_from_state_dict(reader.meta_state_dict())
        if prefix:
            stripped = [(k[len(prefix):], k) for k in reader.keys if k.startswith(prefix)]
            if stripped:
                pairs = stripped
    except Exception as ex:
        logging.warning(f"{label}: prefix detection failed ({ex}); keeping key names as-is.")

    sources = [(out_key, reader, src_key, False) for out_key, src_key in pairs]

    if donor is not None:
        if not any("adaln_t_table" in k for k in donor.keys):
            raise RuntimeError(f"{label}: prune_donor has no 'adaln_t_table' — select a *pruned* checkpoint.")
        before = len(sources)
        sources = [s for s in sources if not any(p in s[0] for p in ("adaln_proj", "time_embedder"))]
        dropped = before - len(sources)
        taken = set()
        for k in donor.keys:
            if any(p in k for p in ("adaln_proj", "adaln_t_table", "time_embedder")):
                taken.add(k)
                if k.endswith(".weight"):
                    for side in (".comfy_quant", ".weight_scale"):
                        sk = k[:-len(".weight")] + side
                        if sk in donor.keys:
                            taken.add(sk)
        sources += [(k, donor, k, True) for k in sorted(taken)]
        metadata = {**metadata, **dict(donor.metadata)}
        metadata.pop("_quantization_metadata", None)
        table_key = next((k for k in taken if "adaln_t_table" in k), None)
        if "config" in metadata and table_key is not None:
            try:
                conf_json = json.loads(metadata["config"])
                tr = conf_json.get("transformer", {})
                tshape = donor.shape(table_key)
                tr["adaln_curve_grid"] = int(tshape[0])
                tr["time_embed_dim"] = int(tshape[1])
                for stale in ("timestep_input_dim", "time_embed_hidden_size"):
                    tr.pop(stale, None)
                conf_json["transformer"] = tr
                metadata["config"] = json.dumps(conf_json)
            except Exception:
                metadata.pop("config", None)
        logging.info(f"{label}: pruned-donor merge — dropped {dropped}, took {len(taken)} tensors "
                     f"+ metadata from the donor.")

    plan, n_fallback, fallback_examples = [], 0, []
    n_keep8 = 0
    for out_key, rdr, src_key, from_donor in sources:
        shape = rdr.shape(src_key)
        dtype = rdr.dtype(src_key)
        e = {"out_key": out_key, "reader": rdr, "src_key": src_key, "shape": shape,
             "itemsize": rdr.itemsize(src_key), "is_weight": out_key.endswith(".weight") and len(shape) >= 2,
             "action": "copy", "spec": None}
        is_lin = (out_key.endswith(".weight") and len(shape) == 2 and dtype in _FLOAT_DTYPES
                  and shape[0] > 1 and shape[1] > 1)
        if from_donor or not is_lin:
            plan.append(e)
            continue
        n, k = int(shape[0]), int(shape[1])
        if keep8 and any(p in out_key for p in keep8):
            # INT8-keep layers: int8 ConvRot instead of source precision.
            e["out_base"] = out_key[:-len(".weight")]
            e["action"] = "int8"
            e["spec"] = int8_spec(k, n)
            n_keep8 += 1
            plan.append(e)
            continue
        if any(ex in out_key for ex in excluded):
            e["action"] = "bf16"
            plan.append(e)
            continue
        e["out_base"] = out_key[:-len(".weight")]
        if k % 256 == 0 and k % int(group) == 0:
            e["action"] = "grouped"
            e["spec"] = grouped_spec(fmt, format_name, k, n, group)
        else:
            e["action"] = "int8"
            e["spec"] = int8_spec(k, n)
            n_fallback += 1
            if len(fallback_examples) < 4:
                fallback_examples.append(f"{e['out_base']} (K={k}, cg={e['spec']['cg']})")
        plan.append(e)
    if keep8:
        logging.info(f"{label}: int8_keep_layers — {n_keep8} layers matching {keep8} stored as "
                     f"INT8 ConvRot instead of source precision.")
    if n_fallback:
        logging.info(f"{label}: {n_fallback} layers with K not divisible by 256/group use the "
                     f"int8 ConvRot fallback, e.g. {fallback_examples}")
    return plan, metadata


_BLOCK_RE = re.compile(r"^(.*?)\.(\d+)\.")


def _apply_w6_tail(plan, n_last, group6, label):
    """Mixed W4A8/W6A8: switch the grouped layers of the last n_last blocks of
    the main block stack (the dotted prefix with the most numbered blocks,
    e.g. 'layers' / 'single_blocks') to W6A8. Returns the number switched."""
    if n_last <= 0:
        return 0
    if group6 < 16 or group6 % 16:
        raise RuntimeError(f"{label}: w6a8_group_size must be a multiple of 16.")
    stacks = {}
    for e in plan:
        if e["action"] == "grouped":
            m = _BLOCK_RE.match(e["out_base"])
            if m:
                stacks.setdefault(m.group(1), set()).add(int(m.group(2)))
    if not stacks:
        logging.warning(f"{label}: w6a8_last_blocks set but no numbered block stack found — all W4A8.")
        return 0
    prefix = max(stacks, key=lambda p: len(stacks[p]))
    idx = sorted(stacks[prefix])
    tail = set(idx[-n_last:])
    n6 = skipped = 0
    for e in plan:
        if e["action"] != "grouped":
            continue
        m = _BLOCK_RE.match(e["out_base"])
        if not (m and m.group(1) == prefix and int(m.group(2)) in tail):
            continue
        sp = e["spec"]
        if sp["k"] % group6:
            skipped += 1
            continue
        e["spec"] = grouped_spec(FORMAT_W6A8, FORMAT_W6A8, sp["k"], sp["n"], group6)
        n6 += 1
    logging.info(f"{label}: mixed layout — {n6} layers in '{prefix}' blocks {min(tail)}-{max(tail)} "
                 f"({len(tail)} of {len(idx)}) use W6A8 g{group6}, the rest W4A8"
                 + (f" ({skipped} kept W4A8: K not divisible by {group6})" if skipped else "") + ".")
    return n6


def _sample(model, positive, negative, latent, steps, cfg, seed):
    """One plain euler/simple sampling run (calibration / tracking driver)."""
    import comfy.sample
    latent_img = latent["samples"]
    if hasattr(comfy.sample, "fix_empty_latent_channels"):
        latent_img = comfy.sample.fix_empty_latent_channels(model, latent_img)
    noise = comfy.sample.prepare_noise(latent_img, seed)
    comfy.sample.sample(model, noise, steps, cfg, "euler", "simple", positive, negative,
                        latent_img, denoise=1.0, seed=seed, disable_pbar=True)


_PROMPT_SEP = re.compile(r"(?m)^[ \t]*-{3,}[ \t]*$")


def _split_prompts(text):
    """Prompts separated by a line containing only '---' (blank lines stay
    inside a prompt, so multi-paragraph prompts work)."""
    return [p.strip() for p in _PROMPT_SEP.split(text or "") if p.strip()]


def _cond_to_cpu(cond):
    out = []
    for c, d in cond:
        d2 = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()}
        out.append([c.cpu() if torch.is_tensor(c) else c, d2])
    return out


def _cond_tokens(cond):
    try:
        return int(cond[0][0].shape[1])
    except Exception:
        return -1


def _encode(clip, text):
    """Same as the stock CLIPTextEncode node; result kept on CPU."""
    tokens = clip.tokenize(text)
    if hasattr(clip, "encode_from_tokens_scheduled"):
        cond = clip.encode_from_tokens_scheduled(tokens)
    else:
        c, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        cond = [[c, {"pooled_output": pooled}]]
    return _cond_to_cpu(cond)


def _resolve_conds(label, clip, prompts, positive, negative, cfg):
    """-> (list of positive conditionings, negative). calib_positive (if
    connected) comes first, then every calib_prompts entry encoded with
    calib_clip. Missing negative: empty prompt via calib_clip."""
    positives = [positive] if positive is not None else []
    if prompts:
        if clip is None:
            logging.warning(f"{label}: calib_prompts given but calib_clip is not connected — "
                            "prompts ignored.")
        else:
            for p in prompts:
                positives.append(_encode(clip, p))
            logging.info(f"{label}: encoded {len(prompts)} calibration prompt(s), text tokens "
                         f"{[_cond_tokens(c) for c in positives]}.")
    if negative is None and positives:
        if clip is not None:
            negative = _encode(clip, "")
            n_tok = _cond_tokens(negative)
            logging.info(f"{label}: calib_negative not connected — using an empty prompt "
                         f"({n_tok} text tokens).")
            if n_tok == 0:
                negative = _encode(clip, " ")
        else:
            negative = positives[0]
            if cfg > 1.0:
                logging.warning(f"{label}: no calib_negative / calib_clip — using the positive as "
                                "negative (the CFG branch is then not representative).")
    return positives, negative


def _track_timesteps(session, specs2d, calib_model, positive, negative, latent,
                     steps, cfg, seed, label):
    """Teacher-forced sampling: every model call runs reference + fake-quant on
    the same input; returns per-call noise-variance records and the sigma map."""
    dm = calib_model.model.diffusion_model
    mapping = map_modules(dm, specs2d, want="linear")
    hooked = session.install_linear(mapping)
    logging.info(f"{label}: timestep tracking — {hooked}/{len(specs2d)} quantized layers matched "
                 f"in calib_model")
    if hooked == 0:
        session.remove()
        raise RuntimeError("no calib_model layers matched the converted file — connect the SAME "
                           "model loaded in high precision")
    records, sig_index = [], {}

    def wrapper(apply_model, args):
        x, t, c = args["input"], args["timestep"], args["c"]
        session.quant = False
        session.step = None
        ref = apply_model(x, t, **c)
        sig = round(float(t.flatten()[0]), 6)
        idx = sig_index.setdefault(sig, len(sig_index))
        session.step = idx
        session.quant = True
        try:
            q = apply_model(x, t, **c)
        finally:
            session.quant = False
        with torch.no_grad():
            r = _flat(ref)
            e = _flat(q) - r
            vr = r.var().clamp(min=1e-30)
            ve = e.var()
            records.append({
                "step": idx, "sigma": sig,
                "noise_var": float(ve), "ref_var": float(vr),
                "nsr": float(ve / vr),
                "snr_db": float(10.0 * torch.log10(vr / ve.clamp(min=1e-30))),
                "rel_rmse": float((e.pow(2).mean() / r.pow(2).mean().clamp(min=1e-30)).sqrt()),
                "cos": float(F.cosine_similarity(r + e, r, dim=0)),
            })
        return ref

    m = calib_model.clone()
    m.set_model_unet_function_wrapper(wrapper)
    try:
        with comfy_compiler_off():
            _sample(m, positive, negative, latent, steps, cfg, seed)
    finally:
        session.remove()
    return records, {v: k for k, v in sig_index.items()}


def _log_timesteps(label, records):
    by_step = {}
    for r in records:
        by_step.setdefault(r["step"], []).append(r)
    rows = []
    logging.info(f"{label}: time-step noise variance (quantized vs reference prediction, "
                 f"teacher-forced)")
    logging.info("  step     sigma     noise_var        NSR     SNR dB    rel-RMSE     cos")
    for step in sorted(by_step):
        rs = by_step[step]
        n = len(rs)
        row = {"step": step, "sigma": rs[0]["sigma"], "calls": n}
        for key in ("noise_var", "ref_var", "nsr", "snr_db", "rel_rmse", "cos"):
            row[key] = sum(r[key] for r in rs) / n
        rows.append(row)
        logging.info(f"  {step:4d}  {row['sigma']:8.4f}  {row['noise_var']:12.4e}  {row['nsr']:9.3e}  "
                     f"{row['snr_db']:7.2f}  {row['rel_rmse']:10.4f}  {row['cos']:.6f}")
    if len(rows) >= 2:
        n = len(rows)
        xm = (n - 1) / 2.0
        ym = sum(r["nsr"] for r in rows) / n
        den = sum((i - xm) ** 2 for i in range(n)) or 1.0
        slope = sum((i - xm) * (r["nsr"] - ym) for i, r in enumerate(rows)) / den
        logging.info(f"{label}: NSR trend {slope:+.3e} per step "
                     f"({'rising toward low noise' if slope > 0 else 'falling toward low noise'})")
    return rows


class _DiffusionGroupedBase:
    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "loaders"
    BITS = 4

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def _inputs(cls, group_choices, group_default, extra, default_format, default_prefix,
                video=False, mixed=False):
        req = {
            "unet_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "High-precision (bf16/fp16) source model."}),
            "model_type": (list(MODEL_TYPE_EXCLUSIONS.keys()), {"tooltip": "Sensitive layers kept at source precision."}),
            "prune_donor": (["None"] + folder_paths.get_filename_list("diffusion_models"), {"default": "None", "tooltip": "MiniMax H3 pruned conversion: existing *_pruned_* checkpoint supplying the adaln curve tensors + config metadata; the source's full-width modulation/time_embedder are dropped."}),
            "group_size": (group_choices, {"default": group_default}),
            "scale_dtype": (["fp8_e4m3", "fp32"], {"default": "fp8_e4m3", "tooltip": "Group-scale storage. fp8 = kitchen default and required for the fastest fused GEMM."}),
        }
        req.update(extra)
        req.update({
            "convrot_groupsize": (["256"], {"default": "256", "tooltip": "Fixed at 256 for the grouped format (kitchen fused kernels). Layers with K not divisible by 256 use int8 ConvRot with group 64/16."}),
            "format_name": ("STRING", {"default": default_format, "tooltip": "Per-layer 'format' string in _quantization_metadata (must match ComfyUI's QUANT_ALGOS)."}),
            "measure_error": (["weights", "off", "weights+timesteps"], {"default": "weights", "tooltip": "weights: per-layer weight error (no extra inputs). weights+timesteps: also a short teacher-forced sampling run with the calib_* inputs — per-step noise variance, per-layer per-step error, Hessian-weighted distances."}),
            "dry_run": ("BOOLEAN", {"default": False, "tooltip": "Print the per-layer plan and projected size, write nothing."}),
            "filename_prefix": ("STRING", {"default": default_prefix}),
            "calib_steps": ("INT", {"default": 8 if video else 4, "min": 1, "max": 50, "tooltip": "Sampling steps per calibration / tracking run (GPTQ collects inputs at every step)."}),
            "calib_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": "Use the model's normal CFG (e.g. MOSS base 3.0). CFG > 1 also calibrates the negative branch."}),
            "calib_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
        })
        if not video:
            req["method"] = (["rtn", "gptq"], {"default": "rtn", "tooltip": "rtn: round-to-nearest (no calibration). gptq: Hessian-aware error feedback (arXiv:2210.17323) — lower output error, same file format; needs calib_model/positive/negative/latent."})
        req.update({
            "gptq_damp": ("FLOAT", {"default": 0.03 if video else 0.01, "min": 0.0001, "max": 1.0, "step": 0.001, "tooltip": "Dampening as a fraction of the mean Hessian diagonal. Raise (0.03-0.1) if results look over-fitted."}),
            "gptq_act_order": ("BOOLEAN", {"default": True, "tooltip": "Quantize the most-used input columns first (static groups keep the format exact)."}),
            "gptq_sequential": ("BOOLEAN", {"default": True, "tooltip": "Later calibration passes run with the already-quantized layers (paper's sequential mode). Off = one pass when the budget allows, slightly less accurate."}),
            "gptq_max_tokens": ("INT", {"default": 2048 if video else 4096, "min": 128, "max": 65536, "step": 128, "tooltip": "Random tokens per layer per model call used for the Hessian (bounds memory/time for large latents)."}),
            "gptq_budget_gb": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 64.0, "step": 0.5, "tooltip": "VRAM for Hessians per calibration pass. Smaller = more passes. 16 GB cards: 3-4."}),
            "gptq_holdout": ("BOOLEAN", {"default": bool(video), "tooltip": "Alternate model calls form a held-out Hessian; a layer where GPTQ is worse than RTN on it keeps RTN (doubles Hessian memory)."}),
            "calib_samples": ("INT", {"default": 1, "min": 1, "max": 16, "tooltip": "GPTQ: sampling runs per prompt (seeds calib_seed, +1, ...)."}),
            "calib_prompts": ("STRING", {"default": "", "multiline": True, "tooltip": "Multi-prompt calibration (needs calib_clip). Separate prompts with a line containing only ---; blank lines stay inside a prompt. Each prompt runs calib_samples x calib_steps. calib_positive, if connected, is used as an extra first prompt. Timestep tracking uses the first prompt."}),
        })
        if mixed:
            req["w6a8_last_blocks"] = ("INT", {"default": 0, "min": 0, "max": 512, "tooltip": "Mixed W4A8/W6A8: the last N transformer blocks of the main block stack use W6A8 (0 = all W4A8). The last blocks carry most of the quantization error; e.g. MOSS (42 blocks): 12 roughly halves W4 output error for ~12% more file, 18 gets close to all-W6A8."})
            req["w6a8_group_size"] = (["16", "32", "64"], {"default": "16", "tooltip": "Group size of the W6A8 blocks (16 = 6.5 bits/weight, lowest error)."})
            req["int8_keep_layers"] = ("BOOLEAN", {"default": False, "tooltip": "Store the model's INT8-keep layers (MiniMax H3 token_refiner, MOSS text_embedder) as INT8 ConvRot instead of bf16."})
        calib = {
            "calib_model": ("MODEL", {"tooltip": "The SAME checkpoint (bf16, or fp8 weights for very large models). Needed for gptq and weights+timesteps."}),
            "calib_latent": ("LATENT", {"tooltip": "Empty latent at the resolution (and frame count) you will generate at."}),
        }
        cond = {
            "calib_positive": ("CONDITIONING", {"tooltip": "Optional single prompt (or use calib_prompts + calib_clip)."}),
            "calib_negative": ("CONDITIONING", {"tooltip": "Optional; defaults to an empty prompt encoded with calib_clip."}),
            "calib_clip": ("CLIP", {"tooltip": "Text encoder for calib_prompts (the one you generate with)."}),
        }
        if video:
            req.update(calib)
            return {"required": req, "optional": cond}
        return {"required": req, "optional": {**calib, **cond}}

    def _run(self, unet_name, model_type, prune_donor, group_size, scale_dtype, codebook,
             scale_search, format_name, measure_error, dry_run, filename_prefix,
             calib_steps, calib_cfg, calib_seed, calib_model, calib_positive,
             calib_negative, calib_latent, method="rtn", gptq_damp=0.01, gptq_act_order=True,
             gptq_sequential=True, gptq_max_tokens=4096, gptq_budget_gb=4.0, gptq_holdout=False,
             calib_samples=1, bits=None, label=None, calib_clip=None, calib_prompts="",
             w6a8_last_blocks=0, w6a8_group_size="16", int8_keep_layers=False):
        bits = bits or self.BITS
        fmt = FORMAT_W4A8 if bits == 4 else FORMAT_W6A8
        label = label or ("W4A8 Codebook" if bits == 4 else "W6A8")
        gptq = method == "gptq"
        group = int(group_size)
        if bits == 6 and (group < 16 or group % 16 != 0):
            raise RuntimeError(f"{label}: group_size must be a multiple of 16, got {group}.")
        s_dtype = torch.float8_e4m3fn if scale_dtype == "fp8_e4m3" else torch.float32
        device = comfy.model_management.get_torch_device()

        reader = SourceReader(folder_paths.get_full_path("diffusion_models", unet_name))
        donor = (SourceReader(folder_paths.get_full_path("diffusion_models", prune_donor))
                 if prune_donor and prune_donor != "None" else None)
        plan, metadata = _build_plan(reader, donor, model_type, fmt, format_name, group, label,
                                     int8_keep=bool(int8_keep_layers))

        n_w6 = _apply_w6_tail(plan, int(w6a8_last_blocks) if bits == 4 else 0,
                              int(w6a8_group_size), label)
        if n_w6:
            label = f"{label} + W6A8 tail"

        src_bytes = sum(reader.itemsize(k) * int(torch.Size(reader.shape(k)).numel()) for k in reader.keys)
        print_plan(plan, label, 1 if s_dtype != torch.float32 else 4, src_bytes)
        gptq_specs = {e["out_base"]: dict(e["spec"], src_key=e["src_key"], reader=e["reader"],
                                          shape=tuple(e["shape"]))
                      for e in plan if e["action"] in ("grouped", "int8") and len(e["shape"]) == 2}
        prompts = _split_prompts(calib_prompts)
        n_prompts = len(prompts) * (calib_clip is not None) + (calib_positive is not None)
        if gptq:
            tot, npass = gptq_estimate(gptq_specs, gptq_budget_gb, gptq_holdout)
            logging.info(f"{label}: GPTQ estimate — {len(gptq_specs)} layers, <= {tot / 2**30:.1f} GiB of "
                         f"Hessians, <= {npass} calibration pass(es), each {n_prompts} prompt(s) x "
                         f"{calib_samples} seed(s) x {calib_steps} steps, budget {gptq_budget_gb} GiB "
                         f"(shared q/k/v inputs lower this).")
        if dry_run:
            logging.info(f"{label}: dry run — nothing written.")
            return {}

        want_ts = measure_error == "weights+timesteps"
        positives, negative = [], None
        if (gptq or want_ts) and calib_model is not None and calib_latent is not None:
            positives, negative = _resolve_conds(label, calib_clip, prompts, calib_positive,
                                                 calib_negative, calib_cfg)
            if calib_clip is not None and prompts:
                # Text encoder done: unload it before the diffusion model is loaded.
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache()
        if gptq and not positives:
            raise RuntimeError(f"{label}: method=gptq needs calib_model, calib_latent and a prompt "
                               "(calib_positive, or calib_prompts + calib_clip).")
        if want_ts and not positives:
            logging.warning(f"{label}: weights+timesteps needs calib_model, calib_latent and a prompt — "
                            "falling back to weight error only.")
            want_ts = False
        kitchen = kitchen_support(6 if n_w6 else bits)

        full_folder, filename, counter, _sub, _pref = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir)
        out_path = os.path.join(full_folder, f"{filename}_{counter:05}_.safetensors")
        writer = StreamingSafetensorsWriter(out_path)
        gopts = {"scale_dtype": s_dtype, "codebook": codebook, "scale_search": scale_search}
        gptq_summary = None
        try:
            prequant, pre_dec = {}, None
            if gptq:
                mapping = map_modules(calib_model.model.diffusion_model, gptq_specs, want="linear")
                logging.info(f"{label}: GPTQ — {len(mapping)}/{len(gptq_specs)} layers matched in calib_model")
                if not mapping:
                    raise RuntimeError(f"{label}: no calib_model layers matched — connect the SAME model.")

                def run_pass():
                    for pos in positives:
                        for s in range(int(calib_samples)):
                            _sample(calib_model, pos, negative, calib_latent,
                                    calib_steps, calib_cfg, calib_seed + s)
                prequant, pre_dec, gptq_summary = run_gptq(
                    label, gptq_specs, mapping, run_pass, writer, kitchen, gopts, device,
                    damp=gptq_damp, act_order=gptq_act_order, sequential=gptq_sequential,
                    max_tokens=gptq_max_tokens, budget_gb=gptq_budget_gb, holdout=gptq_holdout,
                    seed=calib_seed, measure_weights=measure_error != "off")
            qmeta, records, specs, decoder_ok = execute_plan(
                plan, writer, device, label, conf_style="header", kitchen=kitchen,
                grouped_opts=gopts, measure_weights=measure_error != "off",
                prequant=prequant, decoder_ok=pre_dec)
            metadata["_quantization_metadata"] = json.dumps({"layers": dict(sorted(qmeta.items()))})
            writer.finalize(metadata)
        except BaseException:
            writer.abort()
            raise
        logging.info(f"{label}: wrote {out_path}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

        report = {"converter": label, "source": unet_name, "model_type": model_type,
                  "format": format_name, "group_size": group, "scale_dtype": scale_dtype,
                  "codebook": bool(codebook) if bits == 4 else None,
                  "scale_search": bool(scale_search) if bits == 6 else None,
                  "prune_donor": prune_donor if donor is not None else None,
                  "method": method, "gptq": gptq_summary,
                  "w6a8_last_blocks": int(w6a8_last_blocks) if n_w6 else 0,
                  "w6a8_layers": n_w6, "w6a8_group_size": int(w6a8_group_size) if n_w6 else None,
                  "int8_keep_layers": bool(int8_keep_layers),
                  "measure_error": measure_error, "layers": records}

        if want_ts:
            try:
                session = FakeQuantSession(out_path, specs, decoder_ok)
                specs2d = {b: s for b, s in specs.items() if len(s["shape"]) == 2}
                ts_records, step_sigma = _track_timesteps(
                    session, specs2d, calib_model, positives[0], negative, calib_latent,
                    calib_steps, calib_cfg, calib_seed, label)
                report["timesteps"] = _log_timesteps(label, ts_records)
                for base, s in summarize_local(session.local).items():
                    if s.get("worst_step") is not None:
                        s["worst_sigma"] = step_sigma.get(s["worst_step"])
                    records.setdefault(base, {})["activation"] = s
                for base, h in session.hessian_metrics(device).items():
                    records.setdefault(base, {})["hessian"] = h
            except Exception as ex:
                logging.warning(f"{label}: timestep tracking failed: {ex}")

        if measure_error != "off":
            log_summary(label, records, key="weight")
            if any(r.get("activation") for r in records.values()):
                log_summary(label, records, key="activation")
            write_report(out_path, report)
        return {}


class W4A8CodebookSave(_DiffusionGroupedBase):
    """bf16/fp16 diffusion model -> W4A8 codebook (asym_w4a8_int8)."""
    BITS = 4
    DESCRIPTION = ("Convert a diffusion model to W4A8 codebook (comfy-kitchen AsymW4A8Int8Layout): "
                   "4-bit storage at ~int8 speed. Kijai-style metadata, streamed to disk.")

    @classmethod
    def INPUT_TYPES(cls):
        return cls._inputs(["16", "32", "64"], "16", {
            "codebook": ("BOOLEAN", {"default": True, "tooltip": "Per-tensor Lloyd-Max 16-level codebook (recommended)."}),
        }, FORMAT_W4A8, "int4_models/W4A8", mixed=True)

    def convert(self, unet_name, model_type, prune_donor, group_size, scale_dtype, codebook,
                convrot_groupsize, format_name, measure_error, dry_run, filename_prefix,
                calib_steps=4, calib_cfg=1.0, calib_seed=0, calib_model=None,
                calib_positive=None, calib_negative=None, calib_latent=None, **gptq):
        return self._run(unet_name, model_type, prune_donor, group_size, scale_dtype, codebook,
                         True, format_name, measure_error, dry_run, filename_prefix,
                         calib_steps, calib_cfg, calib_seed, calib_model, calib_positive,
                         calib_negative, calib_latent, **gptq)


class W6A8Save(_DiffusionGroupedBase):
    """bf16/fp16 diffusion model -> W6A8 (w6a8_int8)."""
    BITS = 6
    DESCRIPTION = ("Convert a diffusion model to W6A8 (comfy-kitchen PR #191): ~3x lower weight "
                   "error than W4A8 at the same speed, ~0.78 bytes/param. Streamed to disk.")

    @classmethod
    def INPUT_TYPES(cls):
        return cls._inputs(["32", "16", "64"], "32", {
            "scale_search": ("BOOLEAN", {"default": True, "tooltip": "Kitchen's grid-aware group-scale search (better quality, slower conversion, no runtime cost)."}),
        }, FORMAT_W6A8, "int4_models/W6A8")

    def convert(self, unet_name, model_type, prune_donor, group_size, scale_dtype, scale_search,
                convrot_groupsize, format_name, measure_error, dry_run, filename_prefix,
                calib_steps=4, calib_cfg=1.0, calib_seed=0, calib_model=None,
                calib_positive=None, calib_negative=None, calib_latent=None, **gptq):
        return self._run(unet_name, model_type, prune_donor, group_size, scale_dtype, False,
                         scale_search, format_name, measure_error, dry_run, filename_prefix,
                         calib_steps, calib_cfg, calib_seed, calib_model, calib_positive,
                         calib_negative, calib_latent, **gptq)


class VideoGPTQSave(_DiffusionGroupedBase):
    """EXPERIMENTAL: GPTQ for video diffusion models (W4A8 codebook / W6A8)."""
    DESCRIPTION = ("EXPERIMENTAL GPTQ conversion for video models. Same file format as the W4A8 / W6A8 "
                   "nodes. Video defaults: token subsampling, stronger dampening, 8 calibration steps "
                   "and a held-out check that keeps RTN for any layer GPTQ makes worse. Calibrate at "
                   "the resolution and frame count you generate at. Multi-prompt: calib_prompts "
                   "separated by --- lines + calib_clip.")

    @classmethod
    def INPUT_TYPES(cls):
        return cls._inputs(["auto", "16", "32", "64"], "auto", {
            "format": (["w4a8_codebook", "w6a8"], {"default": "w4a8_codebook"}),
            "codebook": ("BOOLEAN", {"default": True, "tooltip": "W4A8 only: Lloyd-Max 16-level codebook."}),
            "scale_search": ("BOOLEAN", {"default": True, "tooltip": "W6A8 only: grid-aware group-scale search."}),
        }, "auto", "video_models/GPTQ", video=True, mixed=True)

    def convert(self, unet_name, model_type, prune_donor, group_size, scale_dtype, format, codebook,
                scale_search, convrot_groupsize, format_name, measure_error, dry_run,
                filename_prefix, calib_steps, calib_cfg, calib_seed, calib_model,
                calib_latent, calib_positive=None, calib_negative=None, **gptq):
        bits = 4 if format == "w4a8_codebook" else 6
        if group_size == "auto":
            group_size = "16" if bits == 4 else "32"
        if not format_name or format_name == "auto":
            format_name = FORMAT_W4A8 if bits == 4 else FORMAT_W6A8
        logging.warning("Video GPTQ is EXPERIMENTAL — compare against the RTN conversion before use.")
        return self._run(unet_name, model_type, prune_donor, group_size, scale_dtype,
                         codebook if bits == 4 else False, scale_search if bits == 6 else True,
                         format_name, measure_error, dry_run, filename_prefix,
                         calib_steps, calib_cfg, calib_seed, calib_model, calib_positive,
                         calib_negative, calib_latent, method="gptq", bits=bits,
                         label=f"Video GPTQ {'W4A8' if bits == 4 else 'W6A8'}", **gptq)
