"""Text-encoder / LLM converter: INT8 ConvRot, W4A8 codebook or W6A8.

Reproduces the Comfy-Org / Kijai text-encoder recipes (verified from the
headers of qwen3.5_9b_qwen_image_2.1_pe_t2i {int8_convrot, w4a8},
qwen3.8_27b_w4a8 and ming_image_0.1_ling_mini_2.0 {int8_convrot, w4a8}):

  dense (moe=False)
    decoder linears                    -> chosen format
    embed_tokens / lm_head             -> int8 ConvRot 256   (embeddings: auto)
    vision tower blocks                -> int8, ConvRot 256/64/16 (vision: auto)
    mtp.layers / connector layers      -> int8
  moe (moe=True)
    fused expert banks [E, N, K]       -> chosen format, "num_experts": E
    attention, dense & shared experts  -> int8
    lm_head -> int8; embed_tokens, vision tower -> kept (auto)
  always kept at source precision: routers (mlp.gate / image_gate), layers
  with < 256 outputs (e.g. in_proj_a/b), mergers / projectors (merger,
  mtp.fc, proj_in/out, linear_proj, proj_directvlm, patch/pos embed),
  norms, conv1d and every other non-2D tensor.
  Grouped formats need K % 256 == 0 (kitchen ConvRot); other layers fall
  back to int8 with the largest ConvRot group dividing K.

Output: per-layer 'comfy_quant' uint8-JSON tensors, original key names,
source metadata preserved — streamed to disk tensor-by-tensor.

Error measurement (report only):
  weights                 per layer: Rel-RMSE, SNR, Outlier %, per-channel
                          MSE (MSE/MAE/max as local-check values in JSON)
  weights+activations+ppl + with calib_clip (the SAME file loaded with the
                          stock CLIP loader) and calib_prompts: every prompt
                          is encoded twice — reference, then fake-quantized
                          (decoded weights + int8 ConvRot activations, int8
                          embedding rows, decoded expert banks). Reports
                          per-layer activation error, Hessian-weighted
                          weight distances (diagonal E[x^2] from the
                          reference pass) and perplexity of the prompts
                          through lm_head (reference vs quantized).

method = gptq (needs calib_clip): the calibration text (calib_prompts plus
an optional built-in multi-domain corpus) is encoded to collect each
layer's input Hessian, then GPTQ re-chooses the codes (gptq_quant.py).
Fused MoE expert banks and embeddings stay RTN.
"""

import logging
import math
import os
import re

import torch

import folder_paths
import comfy.model_management

from .grouped_quant import (
    FORMAT_INT8, FORMAT_W4A8, FORMAT_W6A8,
    kitchen_support, grouped_spec, int8_spec, rotate_rows,
    SourceReader, StreamingSafetensorsWriter,
    print_plan, execute_plan, map_modules, FakeQuantSession, comfy_compiler_off,
    summarize_local, log_summary, write_report,
)
from .gptq_quant import run_gptq, estimate as gptq_estimate

_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

_ROUTER_RE = re.compile(r"(^|\.)(gate|image_gate|router)(\.proj)?$")
_KEEP_RE = re.compile(r"(^|\.)(merger|proj_in|proj_out|linear_proj|proj_directvlm|patch_embed|"
                      r"pos_embed|multi_modal_projector|mm_projector)(\.|$)")
_MTP_KEEP_RE = re.compile(r"(^|\.)mtp\.fc$")
_EMBED_RE = re.compile(r"(^|\.)(embed_tokens|tok_embeddings|word_embeddings|wte)$")
_HEAD_RE = re.compile(r"(^|\.)lm_head$")
_VISION_RE = re.compile(r"(^|\.)(visual|vision|vision_tower|vision_model)(\.|$)")
_INT8_GROUP_RE = re.compile(r"(^|\.)mtp\.layers\.|^connector\.")
_SHARED_EXPERT_RE = re.compile(r"(^|\.)shared_experts?(\.|$)")
_EXPERT_RE = re.compile(r"(^|\.)experts(\.|$)")

_FORMATS = {"w4a8_codebook": FORMAT_W4A8, "w6a8": FORMAT_W6A8, "int8_convrot": FORMAT_INT8}

_DEFAULT_PROMPTS = (
    "A photograph of a red fox standing in fresh snow at sunrise, soft golden light.\n"
    "Explain in two sentences why the sky appears blue during the day.\n"
    "A minimalist poster with the words SUMMER SALE in bold white letters on teal."
)

# Built-in GPTQ calibration corpus (original text, mixed domains and prompt
# styles) so the Hessians see far more tokens than a few short prompts.
_GPTQ_CORPUS = (
    "A wide cinematic shot of a coastal town at dusk. Warm lights glow in the windows of white houses "
    "stacked on a steep hillside, fishing boats rock gently in the harbor, and a lighthouse sweeps its "
    "beam across low clouds tinted orange and violet. The water reflects every light in long, wavering "
    "streaks, and a narrow stone staircase winds from the docks up to a small chapel with a blue dome.",
    "Close-up portrait of an elderly woman with silver hair braided over one shoulder, deep laugh lines, "
    "and bright hazel eyes. She wears a hand-knitted green sweater and holds a chipped ceramic mug of "
    "tea. Soft window light from the left, shallow depth of field, 85mm lens, natural skin texture, "
    "muted background of bookshelves slightly out of focus.",
    "Isometric illustration of a tiny futuristic city on a floating island: glass towers, rooftop "
    "gardens, magnetic trains looping between platforms, waterfalls pouring off the edges into the "
    "clouds below. Clean vector style, pastel palette, crisp outlines, no text.",
    "Product photo of a matte black wireless headphone on a polished concrete surface, studio lighting "
    "with a soft key light and a thin rim light, subtle reflection, minimal composition, lots of "
    "negative space on the right for copy, high detail on the stitched leather ear cushions.",
    "Photosynthesis converts light energy into chemical energy. In the chloroplasts, chlorophyll absorbs "
    "mostly red and blue light, and the energy splits water molecules, releasing oxygen. The captured "
    "energy is stored in ATP and NADPH, which the Calvin cycle then uses to fix carbon dioxide into "
    "sugars that the plant uses for growth and respiration.",
    "def merge_sorted(a, b):\n    result, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n"
    "        if a[i] <= b[j]:\n            result.append(a[i]); i += 1\n        else:\n"
    "            result.append(b[j]); j += 1\n    return result + a[i:] + b[j:]\n"
    "# Merges two sorted lists in linear time; used as the combine step of merge sort.",
    "Le marché du samedi matin était plein de couleurs : des paniers de fraises, des fromages affinés, "
    "du pain encore chaud. Ein alter Mann spielte Akkordeon an der Ecke, und Kinder liefen zwischen den "
    "Ständen. 朝の市場はにぎやかで、魚屋の声が通りに響いていた。日落时，广场上的灯笼一盏一盏亮了起来。",
    "A vintage travel poster in art deco style: a sleek streamlined locomotive crossing a stone viaduct "
    "above a pine valley, bold geometric sun rays in the sky, large serif lettering at the top reading "
    "NORTHERN EXPRESS and smaller text at the bottom reading DAILY DEPARTURES, limited palette of navy, "
    "cream, and burnt orange.",
    "Quarterly summary: revenue grew 12 percent year over year to 48.6 million, driven by subscription "
    "renewals and two new enterprise contracts. Operating costs rose 7 percent, mainly from hiring in "
    "engineering. Cash on hand is 31.2 million, and the team expects margins to improve in the second "
    "half as infrastructure spending stabilizes.",
    "An underwater scene in a kelp forest: shafts of sunlight cut through green-gold water, a curious "
    "sea otter floats on its back cracking a shell on its chest, small silver fish dart between the "
    "swaying stalks, and particles drift in the light. Photorealistic, high dynamic range.",
    "Step by step: first preheat the oven to 200 degrees Celsius. Toss the diced potatoes with olive "
    "oil, salt, smoked paprika and crushed garlic. Spread them in a single layer, roast for 35 minutes, "
    "turning once halfway, until the edges are crisp and golden. Finish with chopped parsley and lemon.",
    "Anime-style key visual: a girl with short teal hair and a yellow raincoat stands on a rooftop in "
    "the rain, holding a transparent umbrella, neon signs of the city glowing below her, puddles "
    "reflecting pink and blue light, dynamic low angle, detailed background, cel shading.",
)


def _main_spec(fmt, k, n, group):
    if fmt != FORMAT_INT8 and k % 256 == 0 and k % group == 0:
        return "grouped", grouped_spec(fmt, fmt, k, n, group)
    return "int8", int8_spec(k, n)


def _classify(base, shape, moe, emb_mode, vis_mode, fmt, group):
    """-> (action, spec, reason)."""
    nd = len(shape)
    is_exp = bool(_EXPERT_RE.search(base)) and not _SHARED_EXPERT_RE.search(base)
    if nd == 3:
        if not is_exp:
            return "copy", None, "non-2D"
        if not moe:
            return "bf16", None, "expert bank (moe off)"
        e, n, k = (int(d) for d in shape)
        act, spec = _main_spec(fmt, k, e * n, group)
        spec["num_experts"] = e
        return act, spec, "experts"
    n, k = int(shape[0]), int(shape[1])
    if _ROUTER_RE.search(base):
        return "bf16", None, "router"
    if n < 256:
        return "bf16", None, "small (<256 outputs)"
    if _MTP_KEEP_RE.search(base) or _KEEP_RE.search(base):
        return "bf16", None, "merger/projector"
    if _EMBED_RE.search(base):
        mode = ("bf16" if moe else "int8") if emb_mode == "auto" else emb_mode
        return (("int8", int8_spec(k, n), "embedding") if mode == "int8"
                else ("bf16", None, "embedding"))
    if _HEAD_RE.search(base):
        mode = "int8" if emb_mode == "auto" else emb_mode
        return (("int8", int8_spec(k, n), "lm_head") if mode == "int8"
                else ("bf16", None, "lm_head"))
    if _VISION_RE.search(base):
        mode = ("bf16" if moe else "int8") if vis_mode == "auto" else vis_mode
        return (("int8", int8_spec(k, n), "vision") if mode == "int8"
                else ("bf16", None, "vision"))
    if _INT8_GROUP_RE.search(base):
        return "int8", int8_spec(k, n), "mtp/connector"
    if moe and is_exp:
        act, spec = _main_spec(fmt, k, n, group)
        return act, spec, "expert"
    if moe:
        return "int8", int8_spec(k, n), "moe non-expert"
    act, spec = _main_spec(fmt, k, n, group)
    return act, spec, "decoder"


def _build_plan(reader, moe, emb_mode, vis_mode, fmt, group, label):
    if any(k.endswith(".comfy_quant") for k in reader.keys) or "_quantization_metadata" in reader.metadata:
        raise RuntimeError(f"{label}: source already contains quantized layers — select the bf16 file.")
    plan = []
    n_banks = 0
    for key in reader.keys:
        shape = reader.shape(key)
        dtype = reader.dtype(key)
        e = {"out_key": key, "reader": reader, "src_key": key, "shape": shape,
             "itemsize": reader.itemsize(key), "action": "copy", "spec": None,
             "is_weight": key.endswith(".weight") and len(shape) >= 2}
        float_weight = key.endswith(".weight") and dtype in _FLOAT_DTYPES
        eligible = float_weight and (
            (len(shape) == 2 and shape[0] > 1 and shape[1] > 1) or len(shape) == 3)
        if not eligible:
            plan.append(e)
            continue
        base = key[:-len(".weight")]
        if len(shape) == 3 and _EXPERT_RE.search(base) and not _SHARED_EXPERT_RE.search(base):
            n_banks += 1
        act, spec, reason = _classify(base, shape, moe, emb_mode, vis_mode, fmt, group)
        e["action"] = act
        e["reason"] = reason
        if spec is not None:
            e["spec"] = spec
            e["out_base"] = base
        plan.append(e)
    if n_banks and not moe:
        logging.warning(f"{label}: found {n_banks} fused MoE expert banks but moe=false — they are "
                        "kept at source precision. Enable moe to quantize them.")
    if moe and not n_banks:
        logging.warning(f"{label}: moe=true but no fused expert banks were found — the MoE recipe "
                        "(non-expert layers int8) is still applied.")
    return plan


def _find_lm_body(root):
    best = None
    for name, mod in root.named_modules():
        emb = getattr(mod, "embed_tokens", None)
        layers = getattr(mod, "layers", None)
        norm = getattr(mod, "norm", None)
        if (isinstance(emb, torch.nn.Module) and isinstance(layers, torch.nn.ModuleList)
                and isinstance(norm, torch.nn.Module)):
            if best is None or len(layers) > len(best.layers):
                best = mod
    return best


@torch.no_grad()
def _nll_sum(h, tgt, chunks):
    t = h.shape[0]
    lse = torch.full((t,), float("-inf"), device=h.device)
    tl = torch.zeros(t, device=h.device)
    for v0, w in chunks():
        lg = h @ w.T
        lse = torch.logaddexp(lse, torch.logsumexp(lg, dim=1))
        m = (tgt >= v0) & (tgt < v0 + w.shape[0])
        if m.any():
            tl[m] = lg[m, tgt[m] - v0]
        del lg
    return float((lse - tl).sum()), t


def _head_sources(plan, specs, session, device):
    """Reference + quantized lm_head providers. Falls back to tied embeddings."""
    head = next((e for e in plan if e["src_key"].endswith("lm_head.weight")), None)
    tied = False
    if head is None:
        head = next((e for e in plan if _EMBED_RE.search(e["src_key"][:-len(".weight")])), None)
        tied = True
    if head is None:
        return None
    src = head["reader"].get(head["src_key"])
    base = head.get("out_base")
    spec = specs.get(base) if base else None
    rows = 16384

    def ref_chunks():
        for v0 in range(0, src.shape[0], rows):
            yield v0, src[v0:v0 + rows].to(device=device, dtype=torch.float32)

    if spec is not None and spec["format"] == FORMAT_INT8:
        p = session.parts(base)
        q, s, cg = p["weight"], p["weight_scale"], spec["cg"]

        def q_chunks():
            for v0 in range(0, q.shape[0], rows):
                yield v0, (q[v0:v0 + rows].to(device=device, dtype=torch.float32)
                           * s[v0:v0 + rows].to(device=device, dtype=torch.float32))

        def q_act(h):
            hr = rotate_rows(h, cg)
            a = (hr.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_(min=1e-30)
            return (hr / a).round_().clamp_(-128, 127).mul_(a)
        return {"ref": ref_chunks, "quant": q_chunks, "quant_act": q_act, "tied": tied,
                "key": head["src_key"], "quantized": True}
    return {"ref": ref_chunks, "quant": ref_chunks, "quant_act": lambda h: h, "tied": tied,
            "key": head["src_key"], "quantized": False}


def _te_calibration(session, specs, plan, clip, prompts, device, label):
    root = clip.cond_stage_model
    specs2d = {b: s for b, s in specs.items() if len(s["shape"]) == 2}
    n_lin = session.install_linear(map_modules(root, specs2d, want="linear"))
    n_emb = 0
    for base, (_name, mod) in map_modules(root, specs2d, want="embedding").items():
        n_emb += int(session.install_embedding(base, mod))
    exp_specs = {b: s for b, s in specs.items() if len(s["shape"]) == 3}
    n_exp = session.install_expert_swaps(map_modules(root, exp_specs, want="experts")) if exp_specs else 0
    logging.info(f"{label}: calibration hooks — {n_lin} linear, {n_emb} embedding, "
                 f"{n_exp}/{len(exp_specs)} expert banks of {len(specs)} quantized tensors")
    if n_lin + n_emb + n_exp == 0:
        session.remove()
        raise RuntimeError("no calib_clip modules matched the converted file — load the SAME "
                           "file with the stock CLIP loader")
    if exp_specs and n_exp < len(exp_specs):
        logging.warning(f"{label}: {len(exp_specs) - n_exp} expert banks could not be fake-quantized; "
                        "PPL then under-reports their error (their weight metrics still apply).")

    body = _find_lm_body(root)
    cap = {"ids": [], "h": []}
    cap_handles = []
    if body is not None:
        cap_handles.append(body.embed_tokens.register_forward_pre_hook(
            lambda m, inp: cap["ids"].append(inp[0].detach()) if inp and torch.is_tensor(inp[0]) else None))
        cap_handles.append(body.norm.register_forward_hook(
            lambda m, inp, out: cap["h"].append(out.detach().float()) if torch.is_tensor(out) else None))
    else:
        logging.warning(f"{label}: could not locate the LM body (embed_tokens/layers/norm) — PPL skipped.")

    pairs = []
    try:
        for i, text in enumerate(prompts):
            tokens = clip.tokenize(text)
            run = {}
            for quant in (False, True):
                cap["ids"].clear()
                cap["h"].clear()
                session.quant = quant
                session.step = i
                try:
                    with comfy_compiler_off():
                        clip.encode_from_tokens(tokens)
                finally:
                    session.quant = False
                run[quant] = (list(cap["ids"]), list(cap["h"]))
            pairs.append(run)
            logging.info(f"{label}: calibration prompt {i + 1}/{len(prompts)} done")
    finally:
        for h in cap_handles:
            h.remove()
        session.remove()

    ppl = None
    if body is not None:
        try:
            ppl = _perplexity(pairs, plan, specs, session, device, label)
        except Exception as ex:
            logging.warning(f"{label}: perplexity failed: {ex}")
    return ppl


def _perplexity(pairs, plan, specs, session, device, label):
    head = _head_sources(plan, specs, session, device)
    if head is None:
        logging.warning(f"{label}: no lm_head or embedding weights found — PPL skipped.")
        return None
    nll_r = nll_q = 0.0
    tokens = 0
    for run in pairs:
        (ids_r, h_r), (ids_q, h_q) = run[False], run[True]
        if not (ids_r and h_r and ids_q and h_q):
            continue
        ids = ids_r[0].reshape(-1, ids_r[0].shape[-1])[0].to(device).long()
        hr = h_r[-1].reshape(-1, h_r[-1].shape[-2], h_r[-1].shape[-1])[0].to(device)
        hq = h_q[-1].reshape(-1, h_q[-1].shape[-2], h_q[-1].shape[-1])[0].to(device)
        if hr.shape[0] != ids.shape[0] or hq.shape[0] != ids.shape[0] or ids.shape[0] < 2:
            continue
        tgt = ids[1:]
        a, n = _nll_sum(hr[:-1], tgt, head["ref"])
        b, _ = _nll_sum(head["quant_act"](hq[:-1]), tgt, head["quant"])
        nll_r += a
        nll_q += b
        tokens += n
    if tokens == 0:
        logging.warning(f"{label}: PPL unavailable — the encoder did not expose aligned token ids "
                        "and final hidden states (it may stop before the final norm).")
        return None
    pr, pq = math.exp(nll_r / tokens), math.exp(nll_q / tokens)
    res = {"tokens": tokens, "ppl_reference": pr, "ppl_quantized": pq,
           "delta_ppl": pq - pr, "ppl_ratio": pq / pr, "head": head["key"],
           "tied_embeddings": head["tied"], "head_quantized": head["quantized"]}
    logging.info(f"{label}: perplexity over {tokens} tokens — reference {pr:.3f}, quantized "
                 f"{pq:.3f} (delta {pq - pr:+.3f}, x{pq / pr:.4f})"
                 + (" [tied embeddings as head]" if head["tied"] else ""))
    return res


class TextEncoderQuantSave:
    """bf16 text encoder / LLM -> INT8 ConvRot, W4A8 codebook or W6A8."""

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "loaders"
    DESCRIPTION = ("Quantize a text encoder / LLM to INT8 ConvRot, W4A8 codebook or W6A8 using the "
                   "Comfy-Org recipes (dense or MoE). Streamed to disk; optional per-layer error, "
                   "Hessian-weighted distances and perplexity.")

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"), {"tooltip": "bf16/fp16 text encoder (.safetensors)."}),
                "format": (["w4a8_codebook", "w6a8", "int8_convrot"], {"default": "w4a8_codebook"}),
                "group_size": (["16", "32", "64"], {"default": "16", "tooltip": "Grouped formats only. W4A8 default 16; W6A8 any multiple of 16."}),
                "scale_dtype": (["fp8_e4m3", "fp32"], {"default": "fp8_e4m3"}),
                "codebook": ("BOOLEAN", {"default": True, "tooltip": "W4A8 only: Lloyd-Max 16-level codebook."}),
                "scale_search": ("BOOLEAN", {"default": True, "tooltip": "W6A8 only: grid-aware group-scale search."}),
                "moe": ("BOOLEAN", {"default": False, "tooltip": "MoE recipe: fused expert banks get the chosen format (+num_experts); attention / dense / shared experts become int8."}),
                "embeddings": (["auto", "int8", "bf16"], {"default": "auto", "tooltip": "embed_tokens / lm_head. auto = int8 both (dense) or lm_head int8 + embed_tokens kept (MoE), as Comfy-Org does."}),
                "vision_tower": (["auto", "int8", "bf16"], {"default": "auto", "tooltip": "auto = int8 (dense) or kept (MoE), as Comfy-Org does."}),
                "measure_error": (["weights", "off", "weights+activations+ppl"], {"default": "weights", "tooltip": "weights+activations+ppl needs calib_clip and calib_prompts."}),
                "dry_run": ("BOOLEAN", {"default": False, "tooltip": "Print the per-layer plan and projected size, write nothing."}),
                "filename_prefix": ("STRING", {"default": "text_encoders/TE_quant"}),
                "calib_prompts": ("STRING", {"default": _DEFAULT_PROMPTS, "multiline": True, "tooltip": "One prompt per line, used for activation error, Hessian and perplexity (and GPTQ calibration)."}),
                "method": (["rtn", "gptq"], {"default": "rtn", "tooltip": "gptq: Hessian-aware error feedback (arXiv:2210.17323), same file format; needs calib_clip. Expert banks / embeddings stay RTN."}),
                "gptq_damp": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 1.0, "step": 0.001}),
                "gptq_act_order": ("BOOLEAN", {"default": True}),
                "gptq_sequential": ("BOOLEAN", {"default": True, "tooltip": "Later passes run with the already-quantized layers."}),
                "gptq_max_tokens": ("INT", {"default": 4096, "min": 128, "max": 65536, "step": 128}),
                "gptq_budget_gb": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 64.0, "step": 0.5, "tooltip": "VRAM for Hessians per calibration pass."}),
                "gptq_holdout": ("BOOLEAN", {"default": False, "tooltip": "Alternate texts form a held-out Hessian; layers where GPTQ is worse keep RTN."}),
                "gptq_builtin_corpus": ("BOOLEAN", {"default": True, "tooltip": "Add a built-in multi-domain calibration corpus (~2k tokens) to calib_prompts for GPTQ. More text = better Hessians."}),
            },
            "optional": {
                "calib_clip": ("CLIP", {"tooltip": "The SAME bf16 file loaded with the stock CLIP loader (activation / PPL measurement only)."}),
            },
        }

    def convert(self, clip_name, format, group_size, scale_dtype, codebook, scale_search, moe,
                embeddings, vision_tower, measure_error, dry_run, filename_prefix, calib_prompts,
                calib_clip=None, method="rtn", gptq_damp=0.01, gptq_act_order=True,
                gptq_sequential=True, gptq_max_tokens=4096, gptq_budget_gb=4.0, gptq_holdout=False,
                gptq_builtin_corpus=True):
        label = "TE Quant"
        gptq = method == "gptq"
        fmt = _FORMATS[format]
        group = int(group_size)
        if fmt == FORMAT_W6A8 and (group < 16 or group % 16 != 0):
            raise RuntimeError(f"{label}: W6A8 group_size must be a multiple of 16, got {group}.")
        s_dtype = torch.float8_e4m3fn if scale_dtype == "fp8_e4m3" else torch.float32
        device = comfy.model_management.get_torch_device()

        reader = SourceReader(folder_paths.get_full_path("text_encoders", clip_name))
        plan = _build_plan(reader, moe, embeddings, vision_tower, fmt, group, label)
        src_bytes = sum(reader.itemsize(k) * int(torch.Size(reader.shape(k)).numel()) for k in reader.keys)
        print_plan(plan, label, 1 if s_dtype != torch.float32 else 4, src_bytes)
        gptq_specs = {e["out_base"]: dict(e["spec"], src_key=e["src_key"], reader=e["reader"],
                                          shape=tuple(e["shape"]))
                      for e in plan if e["action"] in ("grouped", "int8") and len(e["shape"]) == 2
                      and not _EMBED_RE.search(e["out_base"])}
        if gptq:
            tot, npass = gptq_estimate(gptq_specs, gptq_budget_gb, gptq_holdout)
            logging.info(f"{label}: GPTQ estimate — {len(gptq_specs)} layers, <= {tot / 2**30:.1f} GiB of "
                         f"Hessians, <= {npass} calibration pass(es) at budget {gptq_budget_gb} GiB.")
        if dry_run:
            logging.info(f"{label}: dry run — nothing written.")
            return {}
        if gptq and calib_clip is None:
            raise RuntimeError(f"{label}: method=gptq needs calib_clip (the SAME bf16 file loaded "
                               "with the stock CLIP loader).")

        has_grouped = any(e["action"] == "grouped" for e in plan)
        kitchen = kitchen_support(4 if fmt == FORMAT_W4A8 else 6) if has_grouped else None
        want_act = measure_error == "weights+activations+ppl"
        prompts = [p.strip() for p in (calib_prompts or "").splitlines() if p.strip()]
        if want_act and (calib_clip is None or not prompts):
            logging.warning(f"{label}: weights+activations+ppl needs calib_clip and calib_prompts — "
                            "falling back to weight error only.")
            want_act = False

        full_folder, filename, counter, _sub, _pref = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir)
        out_path = os.path.join(full_folder, f"{filename}_{counter:05}_.safetensors")
        writer = StreamingSafetensorsWriter(out_path)
        gopts = {"scale_dtype": s_dtype, "codebook": codebook, "scale_search": scale_search}
        gptq_summary = None
        try:
            prequant, pre_dec = {}, None
            if gptq:
                root = calib_clip.cond_stage_model
                mapping = map_modules(root, gptq_specs, want="linear")
                logging.info(f"{label}: GPTQ — {len(mapping)}/{len(gptq_specs)} layers matched in calib_clip")
                if not mapping:
                    raise RuntimeError(f"{label}: no calib_clip layers matched — load the SAME file.")
                texts = list(prompts) + (list(_GPTQ_CORPUS) if gptq_builtin_corpus else [])
                if not texts:
                    raise RuntimeError(f"{label}: GPTQ needs calib_prompts or gptq_builtin_corpus.")

                def run_pass():
                    for text in texts:
                        tok = calib_clip.tokenize(text)
                        calib_clip.encode_from_tokens(tok)
                prequant, pre_dec, gptq_summary = run_gptq(
                    label, gptq_specs, mapping, run_pass, writer, kitchen, gopts, device,
                    damp=gptq_damp, act_order=gptq_act_order, sequential=gptq_sequential,
                    max_tokens=gptq_max_tokens, budget_gb=gptq_budget_gb, holdout=gptq_holdout,
                    seed=0, measure_weights=measure_error != "off")
            _qmeta, records, specs, decoder_ok = execute_plan(
                plan, writer, device, label, conf_style="tensor", kitchen=kitchen,
                grouped_opts=gopts, measure_weights=measure_error != "off",
                prequant=prequant, decoder_ok=pre_dec)
            writer.finalize(dict(reader.metadata) if reader.metadata else None)
        except BaseException:
            writer.abort()
            raise
        logging.info(f"{label}: wrote {out_path}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

        report = {"converter": label, "source": clip_name, "format": format, "group_size": group,
                  "scale_dtype": scale_dtype, "moe": bool(moe), "embeddings": embeddings,
                  "vision_tower": vision_tower, "measure_error": measure_error,
                  "method": method, "gptq": gptq_summary,
                  "reasons": {e["src_key"]: e.get("reason") for e in plan if e.get("reason")},
                  "layers": records}

        if want_act:
            try:
                session = FakeQuantSession(out_path, specs, decoder_ok)
                report["perplexity"] = _te_calibration(session, specs, plan, calib_clip, prompts,
                                                       device, label)
                for base, s in summarize_local(session.local).items():
                    records.setdefault(base, {})["activation"] = s
                for base, h in session.hessian_metrics(device).items():
                    records.setdefault(base, {})["hessian"] = h
            except Exception as ex:
                logging.warning(f"{label}: activation / PPL measurement failed: {ex}")

        if measure_error != "off":
            log_summary(label, records, key="weight")
            if any(r.get("activation") for r in records.values()):
                log_summary(label, records, key="activation")
            write_report(out_path, report)
        return {}
