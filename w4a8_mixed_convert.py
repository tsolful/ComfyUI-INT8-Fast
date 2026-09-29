"""Grouped-INT8 converters: W4A8 codebook / W6A8 uniform + INT8 mixed.

Both formats live in comfy-kitchen's ``AsymW4A8Int8Layout`` and run on the
same tuned INT8 GEMM; the code width is implied by the packed row width.

W4A8 (kitchen PR #90, ComfyUI PR #15308 — format ``asym_w4a8_int8``)
  ConvRot-rotated weights, grouped int4 (default group 16), per-tensor
  Lloyd-Max 16-level codebook, fp8-e4m3 group scales. ~0.56 B/param,
  weight relL2 ~0.073. Replicates Kijai/MiniMax-H3-experimental's
  minimax_h3_fl2va_pruned_w4a8_mixed.safetensors.

W6A8 (kitchen PR #191, ComfyUI PR #16483 — format ``w6a8_int8``)
  Uniform symmetric 6-bit codes, no codebook, fp8 group scales, group a
  multiple of 16, optional grid-aware scale search at quantize time.
  ~0.78 B/param (g32), weight relL2 ~0.024 — ~3x lower error than W4A8 at
  the SAME speed (int8 parity at H3 token counts).

Tiers per converter: bulk layers -> the grouped format; the worst layers (by
the selected ranking, or measured per layer with calibrated+) -> per-row
``int8_tensorwise`` + ConvRot; MODEL_TYPE_EXCLUSIONS stay at source
precision. ``prune_donor`` supports the MiniMax H3 pruned recipe.

Tensors per grouped layer (layout ``state_dict_tensors``):
  weight            packed codes as int8: [N, K/2] (4-bit) or [N, 3K/4] (6-bit;
                    nibble plane then a K/4-byte plane of each code's top 2 bits)
  weight_s_rel      per-group relative scale, [N, K/G] (fp8-e4m3 or fp32)
  weight_s_channel  per-channel scale, [N] fp32
  weight_codebook   Lloyd-Max levels, [16] fp32 (4-bit codebook mode only)

Metadata: one ``_quantization_metadata`` header block, Kijai-style:
  {"layers": {<base>: {"format": "asym_w4a8_int8" | "w6a8_int8",
   "group_size": G, "convrot": true, "convrot_groupsize": 256}}}
ComfyUI's convert_old_quants expands it into per-layer confs at load.

Runtime speed (kitchen PR #191): the chunked W4A8/W6A8 linear now quantizes
activations with the 64-lane ConvRot kernel int8_linear uses (~20% faster
W4A8 at M=4096). It engages automatically after updating comfy-kitchen when
convrot_groupsize == 256, K % 256 == 0, M >= 512 and K <= the kernel's
fused-K limit — the converter enforces the first two and warns about layers
over the K limit. Existing W4A8 files get the speedup without reconverting.
Kitchen's fused CUDA requant is used for the default 4-bit config (symmetric,
codebook, group 16, fp8), so keeping those defaults also converts fastest.

NOTE: the CUDA fused ConvRot kernels support rotation group 256 only, so
layers whose K is not divisible by 256 pass through unquantized.
"""

import inspect
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

_PRESET_NAMES = ["Balanced", "BalancedQ", "BalancedS", "Quality", "QualityQ", "QualityS"]

# W6A8 calibrated+ gates: (rel_rmse, cos) the simulated W6+A8 output must meet
# to stay W6A8, else the layer goes full INT8. W6A8 error is ~3x below W4A8,
# so these sit well inside the W4A8 (int8mm-proxy) gates.
_W6_PRESETS = {
    "QualityQ":  (0.020, 0.9997),
    "Quality":   (0.028, 0.9995),
    "QualityS":  (0.036, 0.9992),
    "BalancedQ": (0.036, 0.9990),
    "Balanced":  (0.050, 0.9985),
    "BalancedS": (0.070, 0.9975),
}

_FORMAT_SPECS = {
    4: {"label": "W4A8", "tier": "w4a8", "default_format": "asym_w4a8_int8"},
    6: {"label": "W6A8", "tier": "w6a8", "default_format": "w6a8_int8"},
}


def _layout_support(bits):
    """(QuantizedTensor, ok, message) for the requested code width."""
    try:
        from comfy.quant_ops import QuantizedTensor, get_layout_class
        layout = get_layout_class("AsymW4A8Int8Layout")
    except Exception:
        layout = None
    if layout is None:
        return None, False, (
            "AsymW4A8Int8Layout not found. Update comfy-kitchen (PR #90, merged) "
            "and ComfyUI to >= v0.31.0.")
    if bits == 6:
        try:
            params = inspect.signature(layout.quantize).parameters
        except (TypeError, ValueError):
            params = {}
        if "bits" not in params:
            return None, False, (
                "This comfy-kitchen has no 6-bit support (AsymW4A8Int8Layout.quantize "
                "lacks 'bits'). Update comfy-kitchen to a build with PR #191.")
        try:
            from comfy.quant_ops import QUANT_ALGOS
            if "w6a8_int8" not in QUANT_ALGOS:
                logging.warning(
                    "W6A8 Mixed Save: this ComfyUI does not register 'w6a8_int8' yet "
                    "(ComfyUI PR #16483). Conversion works, but LOADING the output needs "
                    "a ComfyUI that includes that PR.")
        except Exception:
            pass
    return QuantizedTensor, True, ""


def _fused_act_k_limit():
    """Kitchen's fused-ConvRot K ceiling for the fast 64-lane activation path
    (PR #191), or None if this kitchen build doesn't expose it."""
    try:
        from comfy_kitchen.backends import cuda as ck_cuda
        v = getattr(ck_cuda, "_CONVROT_FUSED_MAX_K", None)
        return int(v) if v is not None else None
    except Exception:
        return None


def _decide_tiers_grouped(stats, candidates, preset, layer_params, bits):
    """calibrated+ for the two-tier grouped mix: grouped format vs full INT8.

    W4A8: gate on the measured 'int8mm' sim (group-64 uniform W4 + A8) — a
    conservative proxy (group-16 + codebook is strictly better).
    W6A8: gate on the measured 'w6a8' sim (uniform 6-bit + A8, no scale search
    — also slightly conservative).
    Includes the chrel channel gate, drift promotion and small-layer polish.
    """
    spec = _FORMAT_SPECS[bits]
    if bits == 6:
        rel_g, cos_g = _W6_PRESETS[preset]
        sim_key = "w6a8"
    else:
        _r44, _c44, rel_g, cos_g = _CALIB_PLUS_PRESETS[preset]
        sim_key = "int8mm"
    ch_g = 2.5 * rel_g
    tier_name = spec["tier"]
    int8_keys = set()
    total_params = max(1, sum(layer_params.values()))
    polish_pool = []

    logging.info(
        f"{spec['label']} Mixed Save: calibrated+ preset '{preset}' gates "
        f"({tier_name} proxy={sim_key} sim): relRMSE<={rel_g} cos>={cos_g} chrel<={ch_g:.3f}"
    )

    for key, _rot in sorted(candidates):
        rec = stats.get(key)
        if not rec or not rec["tiers"].get(sim_key):
            int8_keys.add(key)
            logging.info(f"{tier_name}+  int8  {key}  (no activation samples — safe default)")
            continue
        mg = _mean_and_drift(rec["tiers"][sim_key])
        m8 = _mean_and_drift(rec["tiers"]["int8"])
        n = rec["n"]
        drift = mg["dmse"] > 0 and mg["dmse"] * n > 0.5 * mg["mse"]
        ok = mg["rel"] <= rel_g and mg["cos"] >= cos_g and mg["chrel"] <= ch_g and not drift
        if ok:
            tier = tier_name
            if (layer_params.get(key, 0) / total_params < 0.003
                    and mg["rel"] > 0.5 * rel_g):
                polish_pool.append((mg["rel"] / rel_g, key))
        else:
            tier = "int8"
            int8_keys.add(key)
        note = " [drift-promoted]" if (tier == "int8" and drift) else ""
        logging.info(
            f"{tier_name}+  {tier:5s} {key}  rel {mg['rel']:.3f}/{m8['rel']:.3f}  "
            f"snr {mg['snr']:.1f}/{m8['snr']:.1f}dB  cos {mg['cos']:.4f}/{m8['cos']:.4f}  "
            f"chrel {mg['chrel']:.3f}  dMSE/step {mg['dmse']:+.2e}  (n={n}){note}"
        )

    polish_pool.sort(reverse=True)
    budget, spent = 0.02 * total_params, 0
    for _sev, key in polish_pool:
        cost = layer_params.get(key, 0)
        if spent + cost > budget:
            continue
        spent += cost
        int8_keys.add(key)
        logging.info(f"{tier_name}+  int8  {key}  [small-layer polish: near-free quality]")

    int8_params = sum(layer_params.get(k, 0) for k in int8_keys)
    logging.info(
        f"{spec['label']} Mixed Save: decisions — {len(candidates) - len(int8_keys)} {tier_name}, "
        f"{len(int8_keys)} int8 (int8_ratio {len(int8_keys)/max(1,len(candidates)):.2f} "
        f"by count, {int8_params/total_params:.2f} by params)"
    )
    return int8_keys


def _common_inputs(default_prefix, default_int8_ratio):
    return {
        "unet_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "High-precision (bf16/fp16) source model."}),
        "model_type": (list(MODEL_TYPE_EXCLUSIONS.keys()), {"tooltip": "Sensitive layers kept at source precision."}),
        "prune_donor": (["None"] + folder_paths.get_filename_list("diffusion_models"), {"default": "None", "tooltip": "MiniMax H3 pruned conversion: existing *_pruned_* checkpoint supplying the adaln curve tensors + config metadata; the source's full-width modulation/time_embedder are dropped."}),
        "int8_ratio": ("FLOAT", {"default": default_int8_ratio, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Fraction of layers kept at full per-row INT8 (ranking modes). Ignored by calibrated+. 0 = pure grouped format (Kijai-style)."}),
        "selection": (["structural", "calibrated", "calibrated+", "sensitivity", "random"], {"default": "structural", "tooltip": "Which layers get INT8 instead of the grouped format. calibrated/calibrated+ need the calib_* inputs; calibrated+ decides per-layer from measured error and ignores int8_ratio."}),
        "calibrated_plus_preset": (_PRESET_NAMES, {"default": "Balanced", "tooltip": "Threshold preset for calibrated+."}),
    }


def _tail_inputs(default_format, default_prefix):
    return {
        "convrot_groupsize": (["256"], {"default": "256", "tooltip": "Fixed at 256: comfy-kitchen's CUDA fused ConvRot kernels (incl. the PR #191 64-lane activation quantizer) implement group 256 only. Layers whose K is not divisible by 256 pass through at source precision."}),
        "int8_convrot": ("BOOLEAN", {"default": True, "tooltip": "ConvRot on the INT8 tier too (recommended)."}),
        "format_name": ("STRING", {"default": default_format, "tooltip": "Per-layer 'format' string in _quantization_metadata. Must match ComfyUI's QUANT_ALGOS registration."}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
        "filename_prefix": ("STRING", {"default": default_prefix}),
        "calib_steps": ("INT", {"default": 4, "min": 1, "max": 50}),
        "calib_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
    }


_OPTIONAL_CALIB = {
    "calib_model": ("MODEL", {"tooltip": "Same checkpoint loaded in high precision (calibrated / calibrated+)."}),
    "calib_positive": ("CONDITIONING",),
    "calib_negative": ("CONDITIONING",),
    "calib_latent": ("LATENT",),
}


class _GroupedInt8ConverterBase:
    """Shared conversion core for W4A8 / W6A8 (+ INT8 tier)."""

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "loaders"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    def _run(self, bits, unet_name, model_type, prune_donor, int8_ratio, selection,
             calibrated_plus_preset, group_size, scale_dtype, codebook, scale_search,
             convrot_groupsize, int8_convrot, format_name, seed, filename_prefix,
             calib_steps, calib_cfg, calib_model, calib_positive, calib_negative,
             calib_latent):
        spec = _FORMAT_SPECS[bits]
        label = f"{spec['label']} Mixed Save"
        QuantizedTensor, ok, msg = _layout_support(bits)
        if not ok:
            raise RuntimeError(f"{label}: {msg}")

        g = int(group_size)
        if bits == 6 and (g < 16 or g % 16 != 0):
            raise RuntimeError(f"{label}: W6A8 group_size must be a multiple of 16, got {g}.")
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

        if any(k.endswith(".comfy_quant") for k in sd) or (
                isinstance(metadata, dict) and "_quantization_metadata" in metadata):
            raise RuntimeError(f"{label}: source already contains quantized layers — "
                               "select the original bf16/fp16 checkpoint.")

        # -- Pruned-donor merge (MiniMax H3), incl. curve-config metadata patch --
        if prune_donor and prune_donor != "None":
            donor_path = folder_paths.get_full_path("diffusion_models", prune_donor)
            donor_sd, donor_metadata = comfy.utils.load_torch_file(donor_path, return_metadata=True)
            if not any("adaln_t_table" in k for k in donor_sd):
                raise RuntimeError(
                    f"{label}: prune_donor has no 'adaln_t_table' — select a *pruned* checkpoint.")
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
                # the donor's own quant block describes the donor, not this output
                metadata.pop("_quantization_metadata", None)
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
                f"{label}: pruned-donor merge — dropped {len(dropped)}, copied {copied} "
                f"tensors + metadata from '{prune_donor}'.")

        # -- Candidates ----------------------------------------------------------
        # Kernel contract: K % 16 == 0, K % G == 0 (6-bit also K % 32), and the
        # CUDA fused ConvRot kernels support group 256 ONLY -> K % 256 == 0,
        # which subsumes the rest for G <= 256.
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
            raise RuntimeError(f"{label}: no quantizable linear layers found.")
        if skipped_k:
            q_n = sum(sd[k].numel() for k, _ in candidates)
            s_n = sum(sd[k].numel() for k in skipped_k)
            logging.warning(
                f"{label}: {len(skipped_k)} linear layers pass through at source precision "
                f"(K not divisible by 256/group): {dict(list(skipped_k.items())[:5])}... "
                f"Quantized share of eligible linear params: {q_n / max(1, q_n + s_n):.0%}."
            )

        # PR #191 fast activation path awareness.
        k_limit = _fused_act_k_limit()
        if k_limit is not None:
            slow = [k for k, _ in candidates if sd[k].shape[1] > k_limit]
            if slow:
                logging.warning(
                    f"{label}: {len(slow)} layers have K > {k_limit} (kitchen fused-ConvRot "
                    f"limit) and will use the slower in-binding activation quantizer at "
                    f"runtime, e.g. {slow[:3]}. Consider them for the INT8 tier.")
            else:
                logging.info(f"{label}: all layers eligible for the 64-lane ConvRot "
                             f"activation fast path (K <= {k_limit}).")

        # -- Tier assignment -----------------------------------------------------
        if selection == "calibrated+":
            if any(x is None for x in (calib_model, calib_positive, calib_negative, calib_latent)):
                raise RuntimeError(f"{label}: calibrated+ needs the calib_* inputs.")
            logging.info(f"{label}: calibrated+ active — int8_ratio ignored.")
            stats = _calibrate_tier_metrics(
                calib_model, calib_positive, calib_negative, calib_latent,
                calib_steps, calib_cfg, seed, dict(candidates),
                w6_group=g if bits == 6 else None)
            layer_params = {key: sd[key].numel() for key, _ in candidates}
            int8_keys = _decide_tiers_grouped(
                stats, candidates, calibrated_plus_preset, layer_params, bits)
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
                    raise RuntimeError(f"{label}: 'calibrated' needs the calib_* inputs.")
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
        n_grouped = n_int8_done = 0
        qmeta_layers = {}
        qkwargs = {"symmetric": True, "scale_dtype": s_dtype}
        if bits == 6:
            qkwargs.update(bits=6, codebook=False, scale_search=bool(scale_search))
        else:
            qkwargs.update(codebook=bool(codebook))

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
                    group_size=g, convrot_groupsize=rot_gs, **qkwargs)
                p = qt._params
                qdata = qt._qdata
                expect_w = tensor.shape[1] * bits // 8
                if qdata.shape[1] != expect_w:
                    raise RuntimeError(
                        f"{label}: {key} packed width {qdata.shape[1]} != K*{bits}/8 = {expect_w}. "
                        "Your comfy-kitchen ignored the bit width — update it.")
                out_sd[key] = qdata.cpu()
                out_sd[base + ".weight_s_rel"] = p.scale.cpu()
                out_sd[base + ".weight_s_channel"] = p.s_channel.cpu()
                if p.codebook is not None:
                    out_sd[base + ".weight_codebook"] = p.codebook.cpu()
                if p.correction is not None:
                    out_sd[base + ".weight_correction"] = p.correction.cpu()
                quant_conf = {"format": format_name, "group_size": g,
                              "convrot": True, "convrot_groupsize": rot_gs}
                del w, qt, qdata
                n_grouped += 1

            qmeta_layers[base] = quant_conf
            pbar.update(1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -- Save (Kijai-style metadata) ------------------------------------------
        out_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        out_metadata["_quantization_metadata"] = json.dumps(
            {"layers": dict(sorted(qmeta_layers.items()))})

        full_output_folder, filename, counter, _sub, _pref = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        output_checkpoint = os.path.join(full_output_folder, f"{filename}_{counter:05}_.safetensors")
        comfy.utils.save_torch_file(out_sd, output_checkpoint, metadata=out_metadata)
        logging.info(
            f"{label}: wrote {output_checkpoint} "
            f"({n_grouped} {spec['tier']} + {n_int8_done} int8_tensorwise layers, "
            f"{len(candidates) - n_grouped - n_int8_done} high-precision candidates; "
            f"exclusion-list layers kept at source precision)."
        )
        return {}


class W4A8MixedSave(_GroupedInt8ConverterBase):
    """bf16/fp16 -> AsymW4A8Int8 (4-bit codebook) + int8_tensorwise mixed."""

    DESCRIPTION = ("Convert a model to W4A8 codebook + INT8 mixed precision (comfy-kitchen "
                   "AsymW4A8Int8Layout). 4-bit storage at ~int8 speed.")

    @classmethod
    def INPUT_TYPES(s):
        req = _common_inputs("int4_models/W4A8_Mixed", 0.15)
        req.update({
            "group_size": (["16", "32", "64"], {"default": "16", "tooltip": "W4A8 quant group along K. 16 = kitchen default/best quality and the only size using kitchen's fused CUDA requant (fastest conversion)."}),
            "scale_dtype": (["fp8_e4m3", "fp32"], {"default": "fp8_e4m3", "tooltip": "Group-scale storage. fp8 = kitchen default and required for the chunked fused GEMM (fastest runtime); fp32 = max quality."}),
            "codebook": ("BOOLEAN", {"default": True, "tooltip": "Per-tensor Lloyd-Max 16-level codebook (recommended; ~14% better than uniform int4)."}),
        })
        req.update(_tail_inputs("asym_w4a8_int8", "int4_models/W4A8_Mixed"))
        return {"required": req, "optional": dict(_OPTIONAL_CALIB)}

    def convert(self, unet_name, model_type, prune_donor, int8_ratio, selection,
                calibrated_plus_preset, group_size, scale_dtype, codebook,
                convrot_groupsize, int8_convrot, format_name, seed, filename_prefix,
                calib_steps=4, calib_cfg=1.0, calib_model=None, calib_positive=None,
                calib_negative=None, calib_latent=None):
        return self._run(
            4, unet_name, model_type, prune_donor, int8_ratio, selection,
            calibrated_plus_preset, group_size, scale_dtype, codebook, False,
            convrot_groupsize, int8_convrot, format_name, seed, filename_prefix,
            calib_steps, calib_cfg, calib_model, calib_positive, calib_negative,
            calib_latent)


class W6A8MixedSave(_GroupedInt8ConverterBase):
    """bf16/fp16 -> W6A8 (uniform 6-bit, kitchen PR #191) + int8_tensorwise mixed."""

    DESCRIPTION = ("Convert a model to W6A8 + INT8 mixed precision (comfy-kitchen PR #191, "
                   "format w6a8_int8). ~3x lower weight error than W4A8 at the same speed, "
                   "~0.78 bytes/param.")

    @classmethod
    def INPUT_TYPES(s):
        req = _common_inputs("int4_models/W6A8_Mixed", 0.05)
        req.update({
            "group_size": (["32", "16", "64"], {"default": "32", "tooltip": "W6A8 quant group along K (multiple of 16). Speed is identical across sizes. Kitchen PR #191 on real H3 weights: g16 0.81 B/w relL2 0.021 | g32 0.78 B/w 0.024 | g64 0.77 B/w 0.026."}),
            "scale_dtype": (["fp8_e4m3", "fp32"], {"default": "fp8_e4m3", "tooltip": "Group-scale storage. fp8 = kitchen default and required for the chunked fused GEMM (fastest runtime)."}),
            "scale_search": ("BOOLEAN", {"default": True, "tooltip": "Kitchen's grid-aware group-scale search at quantize time (6-bit only). Better quality, slower conversion; no runtime cost."}),
        })
        req.update(_tail_inputs("w6a8_int8", "int4_models/W6A8_Mixed"))
        return {"required": req, "optional": dict(_OPTIONAL_CALIB)}

    def convert(self, unet_name, model_type, prune_donor, int8_ratio, selection,
                calibrated_plus_preset, group_size, scale_dtype, scale_search,
                convrot_groupsize, int8_convrot, format_name, seed, filename_prefix,
                calib_steps=4, calib_cfg=1.0, calib_model=None, calib_positive=None,
                calib_negative=None, calib_latent=None):
        return self._run(
            6, unet_name, model_type, prune_donor, int8_ratio, selection,
            calibrated_plus_preset, group_size, scale_dtype, False, scale_search,
            convrot_groupsize, int8_convrot, format_name, seed, filename_prefix,
            calib_steps, calib_cfg, calib_model, calib_positive, calib_negative,
            calib_latent)
