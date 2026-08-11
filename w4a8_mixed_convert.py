"""W4A8 codebook + INT8 mixed converter (comfy-kitchen PR #90, AsymW4A8Int8Layout).

Replicates the recipe behind Kijai/MiniMax-H3-experimental's
minimax_h3_fl2va_pruned_w4a8_mixed.safetensors:

- Bulk linear layers -> ``AsymW4A8Int8Layout``: ConvRot-rotated weights,
  grouped int4 (default group 16) with a per-tensor Lloyd-Max 16-level
  codebook and fp8-e4m3 group scales (``s_rel``) + fp32 per-channel scales
  (``s_channel``). At runtime the int4 weights dequant to grouped int8 and run
  on the tuned INT8 GEMM (~1.09x int8 speed, ~0.56 B/param, weight relL2
  ~0.073 vs NVFP4 ~0.094). Calibration-free.
- Worst layers (by the selected ranking) -> ``int8_tensorwise`` (per-row INT8
  + ConvRot), the repo's proven W8A8 tier.
- MODEL_TYPE_EXCLUSIONS layers stay at source precision.
- ``prune_donor`` supports the MiniMax H3 pruned recipe (drop full-width
  modulation/time_embedder, merge adaln curve tensors + config metadata from
  an existing pruned checkpoint).

Tensors per w4a8 layer mirror the layout's ``state_dict_tensors``:
  weight            packed uint4 pairs as int8, [N, K/2]
  weight_s_rel      per-group relative scale, [N, K/G] (fp8-e4m3 or fp32)
  weight_s_channel  per-channel scale, [N] fp32
  weight_codebook   Lloyd-Max levels, [16] fp32 (codebook mode)

Quantization metadata is a single header block matching Kijai's checkpoints —
``_quantization_metadata`` = {"layers": {<base_key>: {"format":
"asym_w4a8_int8", "group_size": G, "convrot": true, "convrot_groupsize":
256}}} (int8-tier layers appear in the same block as "int8_tensorwise").
ComfyUI's convert_old_quants expands it to per-layer confs at load.

Status: kitchen PR #90 MERGED (aa1ab22, 2026-08-06) and ComfyUI PR #15308
"Support asym w4a8_int" MERGED (2026-08-07, native since ComfyUI v0.31.0).
The registered format string is ``asym_w4a8_int8`` (the ``format_name`` input
default; kept editable against future drift). Requires comfy-kitchen >= the
PR #90 merge. NOTE: the CUDA fused ConvRot kernels support rotation group 256
only, so layers whose K is not divisible by 256 pass through unquantized.
"""

import json
import logging
import os

import torch

import folder_paths
import comfy.utils
import comfy.model_management
import comfy.model_detection

from .convrot import build_hadamard, rotate_weight
from .int8_quant import quantize_int8_axiswise
from .int8_unet_loader import MODEL_TYPE_EXCLUSIONS
from .int4_mixed_convert import (
    _CALIB_PLUS_PRESETS,
    _is_linear_weight,
    _structural_scores,
    _w4a4_sensitivity,
    _calibrate_activation_errors,
    _calibrate_tier_metrics,
    _mean_and_drift,
)


def _w4a8_layout_available():
    """(QuantizedTensor, True) if the installed comfy-kitchen ships
    AsymW4A8Int8Layout (PR #90), else (None, False)."""
    try:
        from comfy.quant_ops import QuantizedTensor, get_layout_class
        if get_layout_class("AsymW4A8Int8Layout") is not None:
            return QuantizedTensor, True
    except Exception:
        pass
    return None, False


def _decide_tiers_w4a8(stats, candidates, preset, layer_params):
    """calibrated+ for the two-tier w4a8 mix.

    The measured 'int8mm' tier (group-64 uniform W4 + A8) is a conservative
    proxy for AsymW4A8Int8 (group-16 + codebook is strictly better), so a
    layer passing the int8mm gates is safe at w4a8; failures go full INT8.
    Includes the chrel channel gate, drift promotion and small-layer polish.
    """
    _rel44, _cos44, rel48, cos48 = _CALIB_PLUS_PRESETS[preset]
    ch48 = 2.5 * rel48
    int8_keys = set()
    total_params = max(1, sum(layer_params.values()))
    polish_pool = []

    logging.info(
        f"W4A8 Mixed Save: calibrated+ preset '{preset}' gates (w4a8 proxy=int8mm sim): "
        f"relRMSE<={rel48} cos>={cos48} chrel<={ch48:.3f}"
    )

    for key, _rot in sorted(candidates):
        rec = stats.get(key)
        if not rec or not rec["tiers"]["int8mm"]:
            int8_keys.add(key)
            logging.info(f"w4a8+  int8  {key}  (no activation samples — safe default)")
            continue
        m48 = _mean_and_drift(rec["tiers"]["int8mm"])
        m8 = _mean_and_drift(rec["tiers"]["int8"])
        n = rec["n"]
        drift = m48["dmse"] > 0 and m48["dmse"] * n > 0.5 * m48["mse"]
        ok = m48["rel"] <= rel48 and m48["cos"] >= cos48 and m48["chrel"] <= ch48 and not drift
        if ok:
            tier = "w4a8"
            if (layer_params.get(key, 0) / total_params < 0.003
                    and m48["rel"] > 0.5 * rel48):
                polish_pool.append((m48["rel"] / rel48, key))
        else:
            tier = "int8"
            int8_keys.add(key)
        note = " [drift-promoted]" if (tier == "int8" and drift) else ""
        logging.info(
            f"w4a8+  {tier:5s} {key}  rel {m48['rel']:.3f}/{m8['rel']:.3f}  "
            f"snr {m48['snr']:.1f}/{m8['snr']:.1f}dB  cos {m48['cos']:.4f}/{m8['cos']:.4f}  "
            f"chrel {m48['chrel']:.3f}  dMSE/step {m48['dmse']:+.2e}  (n={n}){note}"
        )

    polish_pool.sort(reverse=True)
    budget, spent = 0.02 * total_params, 0
    for _sev, key in polish_pool:
        cost = layer_params.get(key, 0)
        if spent + cost > budget:
            continue
        spent += cost
        int8_keys.add(key)
        logging.info(f"w4a8+  int8  {key}  [small-layer polish: near-free quality]")

    int8_params = sum(layer_params.get(k, 0) for k in int8_keys)
    logging.info(
        f"W4A8 Mixed Save: decisions — {len(candidates) - len(int8_keys)} w4a8, "
        f"{len(int8_keys)} int8 (int8_ratio {len(int8_keys)/max(1,len(candidates)):.2f} "
        f"by count, {int8_params/total_params:.2f} by params)"
    )
    return int8_keys


class W4A8MixedSave:
    """Offline converter: bf16/fp16 model -> AsymW4A8Int8 + int8_tensorwise
    mixed checkpoint (comfy-kitchen PR #90 format)."""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "High-precision (bf16/fp16) source model."}),
                "model_type": (list(MODEL_TYPE_EXCLUSIONS.keys()), {"tooltip": "Sensitive layers kept at source precision."}),
                "prune_donor": (["None"] + folder_paths.get_filename_list("diffusion_models"), {"default": "None", "tooltip": "MiniMax H3 pruned conversion: existing *_pruned_* checkpoint supplying the adaln curve tensors + config metadata; the source's full-width modulation/time_embedder are dropped."}),
                "int8_ratio": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Fraction of layers kept at full per-row INT8 (ranking modes). Ignored by calibrated+."}),
                "selection": (["structural", "calibrated", "calibrated+", "sensitivity", "random"], {"default": "structural", "tooltip": "Which layers get INT8 instead of w4a8. calibrated/calibrated+ need the calib_* inputs; calibrated+ decides per-layer from measured error and ignores int8_ratio."}),
                "calibrated_plus_preset": (["Balanced", "BalancedQ", "BalancedS", "Quality", "QualityQ", "QualityS"], {"default": "Balanced", "tooltip": "Threshold preset for calibrated+."}),
                "group_size": (["16", "32", "64"], {"default": "16", "tooltip": "W4A8 quant group along K. 16 = kitchen default/best quality. K must be divisible by the group."}),
                "scale_dtype": (["fp8_e4m3", "fp32"], {"default": "fp8_e4m3", "tooltip": "Group-scale storage. fp8 halves scale metadata (kitchen default, enables the chunked fused kernel); fp32 = max quality."}),
                "codebook": ("BOOLEAN", {"default": True, "tooltip": "Per-tensor Lloyd-Max 16-level codebook (recommended; ~14% better than uniform int4 at coarse groups)."}),
                "convrot_groupsize": (["256"], {"default": "256", "tooltip": "Fixed at 256: comfy-kitchen's CUDA fused ConvRot kernels implement group 256 only. Layers whose K is not divisible by 256 pass through at source precision."}),
                "int8_convrot": ("BOOLEAN", {"default": True, "tooltip": "ConvRot on the INT8 tier too (recommended)."}),
                "format_name": ("STRING", {"default": "asym_w4a8_int8", "tooltip": "comfy_quant 'format' string. 'asym_w4a8_int8' is the merged ComfyUI registration (PR #15308, native since v0.31.0) and matches Kijai's checkpoint."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "filename_prefix": ("STRING", {"default": "int4_models/W4A8_Mixed"}),
                "calib_steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "calib_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "calib_model": ("MODEL", {"tooltip": "Same checkpoint loaded in high precision (calibrated / calibrated+)."}),
                "calib_positive": ("CONDITIONING",),
                "calib_negative": ("CONDITIONING",),
                "calib_latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "loaders"
    DESCRIPTION = "Convert a model to W4A8 codebook + INT8 mixed precision (comfy-kitchen PR #90, AsymW4A8Int8Layout). 4-bit storage at ~int8 speed."

    def convert(self, unet_name, model_type, prune_donor, int8_ratio, selection,
                calibrated_plus_preset, group_size, scale_dtype, codebook,
                convrot_groupsize, int8_convrot, format_name, seed, filename_prefix,
                calib_steps=4, calib_cfg=1.0, calib_model=None, calib_positive=None,
                calib_negative=None, calib_latent=None):
        QuantizedTensor, ok = _w4a8_layout_available()
        if not ok:
            raise RuntimeError(
                "W4A8 Mixed Save: AsymW4A8Int8Layout not found. Update comfy-kitchen "
                "(PR #90 is merged upstream) and ComfyUI to >= v0.31.0."
            )

        g = int(group_size)
        requested_gs = int(convrot_groupsize)
        s_dtype = torch.float8_e4m3fn if scale_dtype == "fp8_e4m3" else torch.float32
        excluded = MODEL_TYPE_EXCLUSIONS.get(model_type, [])
        device = comfy.model_management.get_torch_device()

        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)

        # -- Normalize keys (no-op for native checkpoints) ----------------------
        prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
        if prefix:
            temp_sd = comfy.utils.state_dict_prefix_replace(sd, {prefix: ""}, filter_keys=True)
            if len(temp_sd) > 0:
                sd = temp_sd

        if any(k.endswith(".comfy_quant") for k in sd):
            raise RuntimeError("W4A8 Mixed Save: source already contains quantized layers.")

        # -- Pruned-donor merge (MiniMax H3), incl. curve-config metadata patch --
        if prune_donor and prune_donor != "None":
            donor_path = folder_paths.get_full_path("diffusion_models", prune_donor)
            donor_sd, donor_metadata = comfy.utils.load_torch_file(donor_path, return_metadata=True)
            if not any("adaln_t_table" in k for k in donor_sd):
                raise RuntimeError(
                    "W4A8 Mixed Save: prune_donor has no 'adaln_t_table' — select a *pruned* checkpoint.")
            dropped = [k for k in sd if any(p in k for p in ("adaln_proj", "time_embedder"))]
            for k in dropped:
                del sd[k]
            copied = 0
            for k in list(donor_sd.keys()):
                if not any(p in k for p in ("adaln_proj", "adaln_t_table", "time_embedder")):
                    continue
                sd[k] = donor_sd[k]
                copied += 1
                if k.endswith(".weight"):
                    base = k[:-len(".weight")]
                    for side in (".comfy_quant", ".weight_scale"):
                        if base + side in donor_sd:
                            sd[base + side] = donor_sd[base + side]
                            copied += 1
            del donor_sd
            if isinstance(donor_metadata, dict) and donor_metadata:
                metadata = {**(metadata if isinstance(metadata, dict) else {}), **donor_metadata}
            table_key = next((k for k in sd if "adaln_t_table" in k), None)
            if isinstance(metadata, dict) and "config" in metadata and table_key is not None:
                try:
                    conf_json = json.loads(metadata["config"])
                    tr = conf_json.get("transformer", {})
                    tshape = sd[table_key].shape
                    tr["adaln_curve_grid"] = int(tshape[0])
                    tr["time_embed_dim"] = int(tshape[1])
                    for stale in ("timestep_input_dim", "time_embed_hidden_size"):
                        tr.pop(stale, None)
                    conf_json["transformer"] = tr
                    metadata["config"] = json.dumps(conf_json)
                except Exception:
                    metadata.pop("config", None)
            logging.info(
                f"W4A8 Mixed Save: pruned-donor merge — dropped {len(dropped)}, copied {copied} "
                f"tensors + metadata from '{prune_donor}'.")

        # -- Candidates ----------------------------------------------------------
        # Kernel contract (kitchen, merged PR #90): K % 16 == 0, K % G == 0, and
        # the CUDA fused ConvRot kernels support group 256 ONLY — a smaller
        # per-layer rotation group would crash at sampling, so K % 256 != 0
        # layers pass through at source precision instead of falling back.
        candidates = []  # (key, rot_groupsize) — rot is always 256 here
        skipped_k = {}
        for key, tensor in sd.items():
            if not _is_linear_weight(key, tensor):
                continue
            if any(ex in key for ex in excluded):
                continue
            k_dim = tensor.shape[1]
            if k_dim % 256 != 0 or k_dim % g != 0:
                skipped_k[key] = k_dim
                continue
            candidates.append((key, 256))
        if not candidates:
            raise RuntimeError("W4A8 Mixed Save: no quantizable linear layers found.")
        if skipped_k:
            q_bytes = sum(sd[k].numel() for k, _ in candidates)
            s_bytes = sum(sd[k].numel() for k in skipped_k)
            frac = q_bytes / max(1, q_bytes + s_bytes)
            logging.warning(
                f"W4A8 Mixed Save: {len(skipped_k)} linear layers pass through at source "
                f"precision (K not divisible by 256/group): "
                f"{dict(list(skipped_k.items())[:5])}... "
                f"Quantized share of eligible linear params: {frac:.0%}."
            )

        # -- Tier assignment -----------------------------------------------------
        if selection == "calibrated+":
            if any(x is None for x in (calib_model, calib_positive, calib_negative, calib_latent)):
                raise RuntimeError("W4A8 Mixed Save: calibrated+ needs the calib_* inputs.")
            logging.info("W4A8 Mixed Save: calibrated+ active — int8_ratio ignored.")
            stats = _calibrate_tier_metrics(
                calib_model, calib_positive, calib_negative, calib_latent,
                calib_steps, calib_cfg, seed, dict(candidates))
            layer_params = {key: sd[key].numel() for key, _ in candidates}
            int8_keys = _decide_tiers_w4a8(stats, candidates, calibrated_plus_preset, layer_params)
        else:
            n_int8 = max(0, min(int(round(int8_ratio * len(candidates))), len(candidates)))
            if selection == "random":
                gen = torch.Generator().manual_seed(seed)
                order = torch.randperm(len(candidates), generator=gen).tolist()
                ranked_keys = [candidates[i][0] for i in order]
            elif selection == "structural":
                struct = _structural_scores(candidates)
                ranked_keys = [k for k, _ in sorted(candidates, key=lambda c: (-struct[c[0]], c[0]))]
            elif selection == "calibrated":
                if any(x is None for x in (calib_model, calib_positive, calib_negative, calib_latent)):
                    raise RuntimeError("W4A8 Mixed Save: 'calibrated' needs the calib_* inputs.")
                act_err = _calibrate_activation_errors(
                    calib_model, calib_positive, calib_negative, calib_latent,
                    calib_steps, calib_cfg, seed, dict(candidates))
                observed = sorted(act_err.values())
                fallback = observed[len(observed) // 2] if observed else 1.0
                scores = [(act_err.get(key, fallback) * _w4a4_sensitivity(sd[key], rot_gs, device), key)
                          for key, rot_gs in candidates]
                scores.sort(reverse=True)
                ranked_keys = [k for _, k in scores]
            else:  # sensitivity
                scores = [(_w4a4_sensitivity(sd[key], rot_gs, device), key)
                          for key, rot_gs in candidates]
                scores.sort(reverse=True)
                ranked_keys = [k for _, k in scores]
            int8_keys = set(ranked_keys[:n_int8])

        # -- Quantize ------------------------------------------------------------
        out_sd = {}
        candidate_map = dict(candidates)
        pbar = comfy.utils.ProgressBar(len(candidates))
        n_w4a8 = n_int8_done = 0
        # Single _quantization_metadata header block (kijai-style), instead of
        # per-layer comfy_quant sidecar tensors. ComfyUI's convert_old_quants
        # expands it into per-layer confs at load time.
        qmeta_layers = {}

        for key, tensor in sd.items():
            if key not in candidate_map:
                out_sd[key] = tensor
                continue
            base = key[:-len(".weight")]
            rot_gs = candidate_map[key]

            if key in int8_keys:
                w = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
                use_convrot = bool(int8_convrot)
                if use_convrot:
                    H = build_hadamard(rot_gs, device=w.device, dtype=w.dtype)
                    w = rotate_weight(w, H, group_size=rot_gs)
                q_weight, q_scale = quantize_int8_axiswise(w, dim=1)
                out_sd[key] = q_weight.cpu()
                out_sd[base + ".weight_scale"] = q_scale.cpu()
                quant_conf = {"format": "int8_tensorwise", "convrot": use_convrot}
                if use_convrot:
                    quant_conf["convrot_groupsize"] = rot_gs
                del w, q_weight, q_scale
                n_int8_done += 1
            else:
                w = tensor.to(device=device, non_blocking=True)
                if w.dtype == torch.float32:
                    w = w.to(torch.bfloat16)
                qt = QuantizedTensor.from_float(
                    w, "AsymW4A8Int8Layout",
                    group_size=g,
                    convrot_groupsize=rot_gs,
                    symmetric=True,
                    codebook=bool(codebook),
                    scale_dtype=s_dtype,
                )
                p = qt._params
                out_sd[key] = qt._qdata.cpu()
                out_sd[base + ".weight_s_rel"] = p.scale.cpu()
                out_sd[base + ".weight_s_channel"] = p.s_channel.cpu()
                if p.codebook is not None:
                    out_sd[base + ".weight_codebook"] = p.codebook.cpu()
                if p.correction is not None:
                    out_sd[base + ".weight_correction"] = p.correction.cpu()
                quant_conf = {"format": format_name, "group_size": g,
                              "convrot": True, "convrot_groupsize": rot_gs}
                del w, qt
                n_w4a8 += 1

            qmeta_layers[base] = quant_conf
            pbar.update(1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -- Save ----------------------------------------------------------------
        # Metadata matches kijai's checkpoints: source metadata preserved (incl.
        # the H3 curve config) + one _quantization_metadata block with sorted
        # layer keys. Tier decisions/settings are only logged, not embedded.
        out_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        out_metadata["_quantization_metadata"] = json.dumps(
            {"layers": dict(sorted(qmeta_layers.items()))})

        full_output_folder, filename, counter, _sub, _pref = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        output_checkpoint = os.path.join(full_output_folder, f"{filename}_{counter:05}_.safetensors")
        comfy.utils.save_torch_file(out_sd, output_checkpoint, metadata=out_metadata)
        logging.info(
            f"W4A8 Mixed Save: wrote {output_checkpoint} "
            f"({n_w4a8} w4a8 + {n_int8_done} int8_tensorwise layers, "
            f"{len(candidates) - n_w4a8 - n_int8_done} high-precision)."
        )
        return {}
