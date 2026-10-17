"""GPTQ (Frantar et al., ICLR 2023, arXiv:2210.17323) for the kitchen formats.

Instead of rounding every weight on its own (RTN), GPTQ quantizes a layer one
input column at a time and pushes each column's rounding error onto the
columns not yet quantized, weighted by the inverse Hessian H = X X^T of the
layer's real inputs. This minimizes the layer OUTPUT error ||W X - Q X||, so
plain weight Rel-RMSE rises slightly while output error drops (typically
30-40 % for 4-bit on image models).

Fit to our formats (file layout unchanged — loaders need nothing new):
  * Everything happens in the ConvRot basis: weights are rotated before
    quantization and H is collected from rotated activations.
  * Static groups: kitchen's normal RTN pass supplies s_rel / s_channel /
    codebook (W4A8) or the searched scales (W6A8); GPTQ only re-chooses the
    codes, picking per weight the nearest value the decoder can produce
    (round(clamp(level * s_rel)) * s_channel). Static groups make act-order
    (most-used columns first) safe.
  * int8 layers: per-row absmax scale fixed, codes chosen by GPTQ.
  * Fused MoE expert banks stay RTN: generic hooks cannot see the tokens
    routed to each expert inside a fused bank.

Calibration (generic, no per-architecture code):
  * HessianCollector hooks every not-yet-quantized Linear and allocates H
    lazily in execution order until the memory budget is reached, so each
    pass covers a contiguous prefix of the network. Layers reading the same
    input tensor (q/k/v, up/gate) share one H. Tokens are subsampled per call.
  * sequential=True: layers finished in earlier passes are fake-quantized
    (decoded from the spill file + int8 ConvRot activations) during later
    passes, so later layers see the quantized model's activations (paper's
    sequential mode). Passes ~ total H size / budget.
  * holdout=True: alternate model calls go to a second H used only to judge
    the result; a layer where GPTQ is worse than RTN on held-out data keeps
    its RTN codes (guards against over-fitting, e.g. video).
  * Report: per-layer output-proxy error tr(dW H dW^T) / tr(W H W^T) for RTN
    and GPTQ (the quantity GPTQ minimizes).
"""

import logging
import math

import torch

from .grouped_quant import (
    FORMAT_INT8, rotate_rows, quantize_grouped, quantize_int8_convrot,
    unpack_codes, pack_codes, decode_spec, decode_int8, verify_grouped_decoder,
    weight_error, log_weight_line, FakeQuantSession, comfy_compiler_off,
)

_GPTQ_CHUNK_BYTES = 256 << 20     # W chunk + code-value table per row chunk
_TOKEN_CHUNK = 4096               # tokens per H update matmul
_PROXY_ROWS = 256                 # rows sampled for the output-proxy metric


# =============================================================================
# Hessian collection
# =============================================================================

class HessianCollector:
    """Lazily-allocated full input Hessians (rotated basis) for Linear layers."""

    def __init__(self, specs, budget_bytes, max_tokens, seed, holdout, device):
        self.specs = specs
        self.budget = int(budget_bytes)
        self.max_tokens = int(max_tokens) if max_tokens else 0
        self.holdout = bool(holdout)
        self.device = device
        self.gen = torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF)
        self.H, self.Hv, self.n, self.nv, self.calls = {}, {}, {}, {}, {}
        self.owner, self.members, self.order = {}, {}, []
        self.bad = set()
        self.called = set()
        self.closed = False
        self.used = 0
        self.reserve = None
        self.handles = []
        self._last = None
        self._warned = False

    def install(self, mapping):
        for base, (_name, mod) in mapping.items():
            self.handles.append(mod.register_forward_hook(self._hook(base)))
        return len(self.handles)

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
        self._last = None
        self.reserve = None

    def _alloc(self, base, k):
        if self.closed:
            return False
        need = k * k * 4 * (2 if self.holdout else 1)
        if self.used and self.used + need > self.budget:
            self.closed = True
            return False
        self.reserve = None  # hand the reserved VRAM block to the allocator
        self.H[base] = torch.zeros((k, k), dtype=torch.float32, device=self.device)
        if self.holdout:
            self.Hv[base] = torch.zeros((k, k), dtype=torch.float32, device=self.device)
        self.n[base] = self.nv[base] = self.calls[base] = 0
        self.used += need
        self.order.append(base)
        self.members[base] = [base]
        return True

    def _accumulate(self, base, x, k, cg):
        x2 = x.detach().reshape(-1, k)
        t = x2.shape[0]
        if self.max_tokens and t > self.max_tokens:
            idx = torch.randperm(t, generator=self.gen)[:self.max_tokens].to(x2.device)
            x2 = x2.index_select(0, idx)
        c = self.calls[base]
        self.calls[base] = c + 1
        use_v = self.holdout and (c % 2 == 1)
        H = self.Hv[base] if use_v else self.H[base]
        for t0 in range(0, x2.shape[0], _TOKEN_CHUNK):
            xr = rotate_rows(x2[t0:t0 + _TOKEN_CHUNK].to(device=H.device, dtype=torch.float32), cg)
            if not torch.isfinite(xr).all():
                continue
            H.addmm_(xr.T, xr)
            if use_v:
                self.nv[base] += xr.shape[0]
            else:
                self.n[base] += xr.shape[0]

    def _hook(self, base):
        sp = self.specs[base]
        k, cg = sp["k"], sp["cg"]

        def hook(mod, inputs, output):
            x = inputs[0] if inputs else None
            if not torch.is_tensor(x) or x.shape[-1] != k or base in self.bad:
                return None
            self.called.add(base)
            with torch.no_grad():
                own = self.owner.get(base)
                last = self._last
                if own is None:
                    if (last is not None and last[0] is x and last[2] == (k, cg)
                            and last[1] in self.H):
                        own = last[1]
                        self.members[own].append(base)
                    elif self._alloc(base, k):
                        own = base
                    else:
                        return None
                    self.owner[base] = own
                if own != base:
                    if last is None or last[0] is not x:
                        # sharing pattern not stable -> this layer falls back to RTN
                        self.bad.add(base)
                        if base in self.members.get(own, []):
                            self.members[own].remove(base)
                        if not self._warned:
                            logging.warning(f"GPTQ: shared-input detection unstable for {base}; "
                                            "such layers keep RTN.")
                            self._warned = True
                    return None
                self._accumulate(base, x, k, cg)
                self._last = (x, base, (k, cg))
            return None
        return hook

    def finished(self):
        return [(o, list(self.members[o])) for o in self.order if self.n.get(o, 0) > 0]

    def seen(self):
        return set(self.owner) | self.bad

    def take(self, owner):
        H = self.H.pop(owner)
        Hv = self.Hv.pop(owner, None)
        return H, Hv, self.n.get(owner, 0), self.nv.get(owner, 0)

    def free(self):
        self.H.clear()
        self.Hv.clear()


# =============================================================================
# GPTQ solver
# =============================================================================

@torch.no_grad()
def prepare_hinv(H, damp, act_order):
    """Consumes H. Returns (U = upper Cholesky of H^-1, perm, dead, damp_used)."""
    diag = H.diagonal()
    dead = diag <= 0
    if dead.any():
        H[dead, dead] = 1.0
    perm = None
    if act_order:
        perm = torch.argsort(H.diagonal(), descending=True)
        H = H[perm][:, perm]
    mean = float(H.diagonal().mean())
    if not math.isfinite(mean) or mean <= 0:
        raise RuntimeError("degenerate Hessian")
    rel = float(damp)
    for _ in range(5):
        d = rel * mean
        H.diagonal().add_(d)
        L, info = torch.linalg.cholesky_ex(H)
        if int(info) == 0:
            Hinv = torch.cholesky_inverse(L)
            del L
            U, info2 = torch.linalg.cholesky_ex(Hinv, upper=True)
            del Hinv
            if int(info2) == 0 and torch.isfinite(U).all():
                del H
                return U, perm, dead, rel
            del U
        else:
            del L
        H.diagonal().sub_(d)
        rel *= 10.0
    raise RuntimeError("Cholesky failed even with strong dampening")


@torch.no_grad()
def gptq_solve(W, U, perm, perm_list, invperm, qcol, block=128):
    """W [r, k] fp32 (rotated basis, consumed). Returns codes int32 [r, k] in
    the original column order. qcol(w_col, orig_col) -> (values, codes)."""
    r, k = W.shape
    if perm is not None:
        W = W[:, perm]
    codes = torch.empty((r, k), dtype=torch.int32, device=W.device)
    for i1 in range(0, k, block):
        i2 = min(i1 + block, k)
        cnt = i2 - i1
        W1 = W[:, i1:i2].clone()
        Err1 = torch.empty((r, cnt), dtype=W.dtype, device=W.device)
        U1 = U[i1:i2, i1:i2]
        for i in range(cnt):
            w = W1[:, i]
            j = perm_list[i1 + i] if perm_list is not None else i1 + i
            q, c = qcol(w, j)
            codes[:, i1 + i] = c
            e = (w - q) / U1[i, i]
            if i + 1 < cnt:
                W1[:, i + 1:] -= e.unsqueeze(1).matmul(U1[i, i + 1:].unsqueeze(0))
            Err1[:, i] = e
        if i2 < k:
            W[:, i2:] -= Err1.matmul(U[i1:i2, i2:])
        del W1, Err1
    if invperm is not None:
        codes = codes[:, invperm]
    return codes


def _proxy(q_rot, w_rot, He):
    d = q_rot - w_rot
    return float(((d @ He) * d).sum())


@torch.no_grad()
def gptq_quantize_layer(src, spec, prep, h_eval, kitchen, opts, device, block=128,
                        check_pack=False, eval_name="calibration"):
    """GPTQ one 2D layer. Returns (parts, gptq_record)."""
    U, perm, dead, damp_used = prep
    k = spec["k"]
    shp = tuple(src.shape)
    w2 = src.reshape(-1, k)
    n = w2.shape[0]
    cg = spec["cg"]
    grouped = spec["format"] != FORMAT_INT8
    perm_list = perm.tolist() if perm is not None else None
    invperm = torch.argsort(perm) if perm is not None else None
    any_dead = bool(dead.any())

    if grouped:
        bits, group = spec["bits"], spec["group"]
        rtn = quantize_grouped(src, bits, group, opts["scale_dtype"], opts.get("codebook", True),
                               opts.get("scale_search", True), kitchen, device)
        qw = rtn["weight"].reshape(n, -1)
        u8 = qw.view(torch.uint8) if qw.dtype == torch.int8 else qw
        if check_pack and not torch.equal(pack_codes(unpack_codes(u8, k, bits), bits), u8):
            raise RuntimeError("GPTQ: code packing does not round-trip comfy-kitchen's layout — "
                               "aborting (use method=rtn).")
        s_rel = rtn["weight_s_rel"].reshape(n, -1)
        s_ch = rtn["weight_s_channel"].reshape(n)
        cb = rtn.get("weight_codebook")
        C = 1 << bits
        levels = (cb.to(device=device, dtype=torch.float32).reshape(-1) if (bits == 4 and cb is not None)
                  else torch.arange(C, device=device, dtype=torch.float32) - float(C // 2))
        G = k // group
        packed = torch.empty((n, k * bits // 8), dtype=torch.uint8)
        per_row = k + G * C
    else:
        rtn = None
        q_out = torch.empty((n, k), dtype=torch.int8)
        s_out = torch.empty((n, 1), dtype=torch.float32)
        per_row = 2 * k

    rows = max(1, min(n, _GPTQ_CHUNK_BYTES // (4 * per_row)))
    for r0 in range(0, n, rows):
        r1 = min(n, r0 + rows)
        W = rotate_rows(w2[r0:r1].to(device=device, dtype=torch.float32), cg)
        if grouped:
            sr = s_rel[r0:r1].to(device=device).to(torch.float32)
            sc = s_ch[r0:r1].to(device=device, dtype=torch.float32)
            table = (levels.view(1, 1, -1) * sr.unsqueeze(-1)).round_().clamp_(-127, 127)
            table.mul_(sc.view(-1, 1, 1))

            def qcol(w, j, _t=table, _g=group):
                vals = _t[:, j // _g, :]
                idx = (w.unsqueeze(1) - vals).abs_().argmin(dim=1)
                return vals.gather(1, idx.unsqueeze(1)).squeeze(1), idx
        else:
            s = (W.abs().amax(dim=1) / 127.0).clamp_(min=1e-30)

            def qcol(w, j, _s=s):
                q = (w / _s).round_().clamp_(-128, 127)
                return q * _s, q
        if any_dead:
            W[:, dead] = 0
        codes = gptq_solve(W, U, perm, perm_list, invperm, qcol, block)
        if grouped:
            packed[r0:r1] = pack_codes(codes, bits).cpu()
            del table
        else:
            q_out[r0:r1] = codes.to(torch.int8).cpu()
            s_out[r0:r1] = s.unsqueeze(1).cpu()
        del W, codes

    if grouped:
        parts = dict(rtn)
        parts["weight"] = packed.view(rtn["weight"].dtype).reshape(rtn["weight"].shape)
    else:
        parts = {"weight": q_out.reshape(shp), "weight_scale": s_out.reshape(*shp[:-1], 1)}

    # Output-proxy error on sampled rows: tr(dW H dW^T) / tr(W H W^T)
    sample = torch.linspace(0, n - 1, steps=min(n, _PROXY_ROWS)).round().long().unique()
    He = h_eval.to(device=device, dtype=torch.float32)
    w_rot = rotate_rows(w2.index_select(0, sample).to(device=device, dtype=torch.float32), cg)
    den = max(_proxy(torch.zeros_like(w_rot), w_rot, He), 1e-30)
    q_g = decode_spec(parts, spec, device, rotated=True, rows=sample)
    if grouped:
        q_r = decode_spec(rtn, spec, device, rotated=True, rows=sample)
    else:
        q_r = decode_int8(quantize_int8_convrot(w2.index_select(0, sample), cg, device), cg, device,
                          rotated=True)
    e_g = _proxy(q_g, w_rot, He) / den
    e_r = _proxy(q_r, w_rot, He) / den
    del He, w_rot, q_g, q_r

    method = "gptq"
    if e_g > e_r:
        method = "rtn_fallback"
        parts = rtn if grouped else quantize_int8_convrot(src, cg, device)

    def snr(x):
        return float(10.0 * math.log10(1.0 / max(x, 1e-30)))
    rec = {"method": method, "eval": eval_name, "damp": damp_used,
           "act_order": perm is not None,
           "out_rel_rtn": math.sqrt(max(e_r, 0.0)), "out_rel_gptq": math.sqrt(max(e_g, 0.0)),
           "out_snr_rtn_db": snr(e_r), "out_snr_gptq_db": snr(e_g),
           "improvement_pct": (1.0 - math.sqrt(max(e_g, 0.0)) / max(math.sqrt(max(e_r, 0.0)), 1e-30)) * 100.0}
    return parts, rec


# =============================================================================
# Driver
# =============================================================================

def estimate(specs2d, budget_gb, holdout):
    """Upper bound (no Hessian sharing): total H bytes and calibration passes."""
    total = sum(int(s["k"]) ** 2 * 4 for s in specs2d.values()) * (2 if holdout else 1)
    budget = max(1, int(float(budget_gb) * (1 << 30)))
    return total, max(1, math.ceil(total / budget))


def _free_vram(device):
    if device.type != "cuda":
        return
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        torch.cuda.empty_cache()


def _bump_reserved_vram(nbytes):
    try:
        import comfy.model_management as mm
        old = mm.EXTRA_RESERVED_VRAM
        mm.EXTRA_RESERVED_VRAM = old + int(nbytes)
        return old
    except Exception:
        return None


def _restore_reserved_vram(old):
    if old is None:
        return
    try:
        import comfy.model_management as mm
        mm.EXTRA_RESERVED_VRAM = old
    except Exception:
        pass


def run_gptq(label, specs, mapping, run_pass, writer, kitchen, grouped_opts, device,
             damp=0.01, act_order=True, sequential=True, max_tokens=4096, budget_gb=4.0,
             holdout=False, seed=0, measure_weights=True, block=128):
    """Calibrate + GPTQ every mapped 2D layer, writing parts to 'writer'.
    Returns (done {base: {"records": {...}}}, decoder_ok, summary)."""
    import comfy.utils
    budget = max(1 << 20, int(float(budget_gb) * (1 << 30)))
    remaining = [b for b in mapping if b in specs and len(specs[b]["shape"]) == 2]
    total_layers = len(remaining)
    session = FakeQuantSession(None, specs, True, part_source=writer) if sequential else None
    if session is not None:
        session.collect_local = False
    done, gptq_recs = {}, {}
    decoder_ok = None
    pack_checked = False
    passes = 0
    pbar = comfy.utils.ProgressBar(max(1, total_layers))
    est_total, est_passes = estimate({b: specs[b] for b in remaining}, budget_gb, holdout)
    logging.info(f"{label}: GPTQ over {total_layers} layers — up to {est_total / 2**30:.1f} GiB of "
                 f"Hessians, budget {budget_gb} GiB -> <= {est_passes} calibration pass(es)"
                 f"{' (sequential)' if sequential else ''}.")

    while remaining:
        passes += 1
        coll = HessianCollector(specs, budget, max_tokens, seed + passes, holdout, device)
        coll.install({b: mapping[b] for b in remaining})
        if session is not None and done:
            session.install_linear({b: mapping[b] for b in done if b in mapping})
            session.quant = True
        old_reserved = None
        if device.type == "cuda":
            # Unload, then raise ComfyUI's reserved-VRAM figure by the Hessian
            # budget so the (re)loaded model — including dynamic / aimdo VRAM
            # loading — leaves room for the Hessians. (A placeholder tensor is
            # not used: aimdo's allocator does not see torch's cache.)
            _free_vram(device)
            old_reserved = _bump_reserved_vram(budget + (256 << 20))
        try:
            with comfy_compiler_off():
                run_pass()
        finally:
            _restore_reserved_vram(old_reserved)
            coll.remove()
            if session is not None:
                session.quant = False
                session.remove()
        groups = coll.finished()
        if not groups:
            coll.free()
            if passes == 1:
                raise RuntimeError("GPTQ: no calibration inputs reached the quantized layers — "
                                   "connect the SAME model as the source")
            break
        _free_vram(device)
        n_pass = 0
        for owner, bases in groups:
            H, Hv, cnt, cntv = coll.take(owner)
            if cnt < spec_k(specs, owner):
                logging.info(f"{label}: {owner}: only {cnt} calibration tokens for K={spec_k(specs, owner)} "
                             "(dampening keeps it stable; more calibration data helps)")
            if Hv is not None and cntv > 0:
                h_eval, eval_name = Hv.div_(cntv), "holdout"
            else:
                h_eval, eval_name = H / max(1, cnt), "calibration"
            Hv = None
            try:
                prep = prepare_hinv(H, damp, act_order)
            except Exception as ex:
                logging.warning(f"{label}: {owner}: GPTQ skipped ({ex}); RTN used for {bases}")
                del H, h_eval
                continue
            del H
            for b in bases:
                sp = specs[b]
                src = sp["reader"].get(sp["src_key"])
                grouped = sp["format"] != FORMAT_INT8
                if grouped and decoder_ok is None:
                    decoder_ok, rel = verify_grouped_decoder(
                        src, sp, kitchen, grouped_opts["scale_dtype"], grouped_opts.get("codebook", True),
                        grouped_opts.get("scale_search", True), device)
                    if not decoder_ok:
                        raise RuntimeError(f"GPTQ: decoder disagrees with comfy-kitchen (rel {rel:.2e}); "
                                           "GPTQ needs an exact decoder — use method=rtn.")
                parts, g = gptq_quantize_layer(src, sp, prep, h_eval, kitchen, grouped_opts, device,
                                               block=block, check_pack=grouped and not pack_checked,
                                               eval_name=eval_name)
                pack_checked = pack_checked or grouped
                g["tokens"] = cnt
                g["shared_hessian"] = owner != b
                for suffix, t in parts.items():
                    writer.add(f"{b}.{suffix}", t)
                rec = {"gptq": g}
                if measure_weights:
                    try:
                        m = weight_error(src, parts, sp, device)
                        rec["weight"] = m
                    except Exception as ex:
                        logging.warning(f"{label}: weight error for {b} failed: {ex}")
                logging.info(f"  [gptq] {b}  out-err rtn {g['out_rel_rtn']:.4f} -> "
                             f"{g['out_rel_gptq']:.4f} ({-g['improvement_pct']:+.1f}%)"
                             + ("  KEPT RTN" if g["method"] != "gptq" else "")
                             + (f"  damp {g['damp']:.3g}" if g["damp"] != damp else ""))
                done[b] = {"records": rec}
                gptq_recs[b] = g
                n_pass += 1
                pbar.update(1)
                del src, parts
            del prep, h_eval
            if device.type == "cuda":
                torch.cuda.empty_cache()
        coll.free()
        seen = coll.seen()
        never = [b for b in remaining if b not in coll.called]
        if never:
            logging.info(f"{label}: {len(never)} layers never ran during calibration -> RTN "
                         f"(e.g. {never[:3]})")
        remaining = [b for b in remaining if b not in seen and b in coll.called]
        logging.info(f"{label}: GPTQ pass {passes} quantized {n_pass} layers "
                     f"({len(done)}/{total_layers} done, {len(remaining)} left).")
        del coll

    summary = _summary(label, gptq_recs, total_layers, passes, damp, act_order, sequential,
                       max_tokens, budget_gb, holdout)
    return done, decoder_ok, summary


def spec_k(specs, base):
    return int(specs[base]["k"])


def _summary(label, recs, total, passes, damp, act_order, sequential, max_tokens, budget_gb, holdout):
    used = [r for r in recs.values() if r["method"] == "gptq"]
    fb = len(recs) - len(used)
    s = {"layers_total": total, "layers_gptq": len(used), "layers_rtn_fallback": fb,
         "layers_rtn_uncalibrated": total - len(recs), "passes": passes, "damp": damp,
         "act_order": act_order, "sequential": sequential, "max_tokens": max_tokens,
         "budget_gb": budget_gb, "holdout": holdout}
    if recs:
        rr = sum(r["out_rel_rtn"] for r in recs.values()) / len(recs)
        gg = sum(r["out_rel_gptq"] if r["method"] == "gptq" else r["out_rel_rtn"]
                 for r in recs.values()) / len(recs)
        s.update({"mean_out_rel_rtn": rr, "mean_out_rel_final": gg,
                  "mean_improvement_pct": (1.0 - gg / max(rr, 1e-30)) * 100.0})
        logging.info(f"{label}: GPTQ summary — {len(used)} layers GPTQ, {fb} kept RTN, "
                     f"{total - len(recs)} never reached by calibration (RTN). Mean output-proxy "
                     f"error {rr:.4f} (RTN) -> {gg:.4f} ({-s['mean_improvement_pct']:+.1f}%).")
    return s
