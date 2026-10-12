#!/usr/bin/env python3
"""Training FLOP-share (Amdahl) analysis: which part of LLM *training* could a Fourier-optical
transform engine take over, and which of that work can sit on a static Fourier-plane mask?

Writes ``results.md`` and ``results.json`` beside this file. Every number in the FOptTrn decks
comes from here.

It extends the forward-pass ledger of LLM_Hub_Fourier_Optics_Inference/analysis/flop_share.py
(same model shape, same variants, same FLOP conventions) with the backward pass and the
optimizer step:

  forward       as the inference script's prefill, LM head on every token (training does).
  backward      dense and attention: 2x forward (dX and dW). Long convolution y = h * u by FFT:
                one FFT of the output gradient G, dX = IFFT(G . conj(H)), and the filter gradient
                dH = sum over the batch of G . conj(U), accumulated in the Fourier domain and
                inverse-transformed once per step. Block-circulant matrices: the same, per block.
  optimizer     AdamW element-wise work (not counted, as norms are not) and, optionally, DeMo's
                DCT of the momentum in chunks of 64 (decoupled momentum, Peng et al. 2024).

The key split is between *weight-masked* Fourier-domain products (one operand is a filter or
weight spectrum, fixed for the whole step: it can be a mask written once per step) and
*data-by-data* products (both operands depend on the sample, as in G . conj(U) for the weight
gradient: a mask would have to be rewritten for every sequence).

Model shape: Llama-3-8B (32 layers, d = 4096, 32 heads, 8 KV heads, SwiGLU d_ff = 14,336,
vocab 128,256), as in the inference series. FLOP conventions: matmul 2mnk; real FFT of length n
2.5 n log2 n and complex 5 n log2 n (FFTW's convention); complex pointwise multiply 6, complex
add 2. Causal long convolutions zero-pad to the next power of two >= 2L. Softmax, norms,
activations, recomputation (activation checkpointing) and the implicit filter's own small
network are not counted.
"""

from __future__ import annotations

import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# Llama-3-8B shape (the inference series takes it from Disaggregated_Inference_Sim)
D, L_LAYERS, FF, V = 4096, 32, 14336, 128256
N_HEADS, N_KV_HEADS = 32, 8
KV = N_KV_HEADS * (D // N_HEADS)
ORDER = 2                    # Hyena order N
SHORT = 3                    # short depthwise conv taps
CIRC_BLOCK = 256             # block-circulant block size (illustrative)
SEQS = [2048, 4096, 8192, 32768]
STEP_TOKENS = 1 << 20        # tokens per optimizer step (illustrative: about 1M, as Llama-scale runs)
DEMO_CHUNK = 64              # DeMo's DCT chunk size

# Illustrative mask device, as the inference series: a 2-megapixel DMD at 1,031 Hz (8-bit)
SLM_PIXELS = 2e6
SLM_HZ = 1031.0
# Illustrative cluster for step times: 64 GPUs at 400 TFLOP/s sustained each
GPUS, GPU_FLOPS = 64, 400e12


def rfft(n: int) -> float:
    return 2.5 * n * math.log2(n)


def pow2_at_least(n: int) -> int:
    return 1 << (n - 1).bit_length()


# ─────────────────────────────────────────────────────────── op ledger ──
@dataclass
class Ops:
    """FLOPs by class. ``transform`` = FFT/IFFT work. ``masked`` = Fourier-domain pointwise
    products whose second operand is a weight spectrum (static over a step: a mask can hold it).
    ``datadata`` = Fourier-domain products of two data-dependent operands (no static mask).
    ``dense`` = matmuls; ``attention`` = QK^T and AV; ``other`` = short convs, gating,
    Fourier-domain accumulation."""

    dense: float = 0.0
    attention: float = 0.0
    transform: float = 0.0
    masked: float = 0.0
    datadata: float = 0.0
    other: float = 0.0

    def __iadd__(self, o: "Ops") -> "Ops":
        for k in self.__dict__:
            setattr(self, k, getattr(self, k) + getattr(o, k))
        return self

    def __add__(self, o: "Ops") -> "Ops":
        r = Ops(**self.__dict__)
        r += o
        return r

    def scaled(self, f: float) -> "Ops":
        return Ops(**{k: v * f for k, v in self.__dict__.items()})

    @property
    def total(self) -> float:
        return self.dense + self.attention + self.transform + self.masked + self.datadata + self.other

    @property
    def optical(self) -> float:
        """What a transform engine could take: transforms plus all Fourier-domain products."""
        return self.transform + self.masked + self.datadata

    @property
    def optical_static(self) -> float:
        """What it could take with masks rewritten only once per optimizer step."""
        return self.transform + self.masked


# ─────────────────────────────────────────────── forward and backward ──
def matrix(m: int, n: int, tokens: int, circ: bool, seqs_per_step: float) -> tuple[Ops, Ops]:
    """y = W x, W of shape (m, n), on ``tokens`` vectors: (forward, backward)."""
    if not circ:
        f = Ops(dense=2.0 * m * n * tokens)
        return f, f.scaled(2.0)                       # dX = W^T G and dW = G X^T
    k = CIRC_BLOCK
    bins = k // 2 + 1
    p, q = m // k, n // k
    fwd = Ops(transform=tokens * (q + p) * rfft(k),               # FFT x blocks, IFFT y blocks
              masked=tokens * p * q * bins * 6.0,                 # times the block spectra
              other=tokens * p * (q - 1) * bins * 2.0)            # sum over input blocks
    bwd = Ops(transform=tokens * (p + q) * rfft(k),               # FFT G blocks, IFFT dX blocks
              masked=tokens * p * q * bins * 6.0,                 # conj(W) . G
              datadata=tokens * p * q * bins * 6.0,               # conj(X) . G for dW
              other=tokens * (q * (p - 1) + p * q) * bins * 2.0)  # dX block sums, dW accumulation
    # once per step, shared by the step's sequences: FFT of the weight blocks, IFFT of dW
    fwd.transform += p * q * rfft(k) / seqs_per_step
    bwd.transform += p * q * rfft(k) / seqs_per_step
    return fwd, bwd


def mlp(tokens: int, circ: bool, sps: float) -> tuple[Ops, Ops]:
    f, b = Ops(), Ops()
    for m, n in ((FF, D), (FF, D), (D, FF)):           # gate, up, down
        fo, bo = matrix(m, n, tokens, circ, sps)
        f += fo
        b += bo
    return f, b


def attention_block(s: int, sps: float) -> tuple[Ops, Ops]:
    f = Ops(dense=2.0 * s * (2 * D * D + 2 * D * KV), attention=2.0 * D * s * (s + 1))
    return f, f.scaled(2.0)                             # FlashAttention's recompute not counted


def hyena_block(s: int, circ: bool, sps: float) -> tuple[Ops, Ops]:
    f, b = Ops(), Ops()
    for m, n in (((ORDER + 1) * D, D), (D, D)):        # in- and out-projection
        fo, bo = matrix(m, n, s, circ, sps)
        f += fo
        b += bo
    f.other += s * 2.0 * SHORT * (ORDER + 1) * D + s * ORDER * D
    b.other += 2.0 * (s * 2.0 * SHORT * (ORDER + 1) * D + s * ORDER * D)
    n = pow2_at_least(2 * s)
    bins = n // 2 + 1
    ch = ORDER * D
    # forward, per channel: FFT(u), times the filter spectrum H, IFFT
    f.transform += ch * 2 * rfft(n)
    f.masked += ch * bins * 6.0
    # backward, per channel: FFT(G); dU = IFFT(G . conj(H)); dH += G . conj(U) (U kept from fwd)
    b.transform += ch * 2 * rfft(n)
    b.masked += ch * bins * 6.0
    b.datadata += ch * bins * 6.0
    b.other += ch * bins * 2.0
    # per step, shared by the step's sequences: FFT of each filter, IFFT of its gradient
    f.transform += ch * rfft(n) / sps
    b.transform += ch * rfft(n) / sps
    return f, b


def lm_head(tokens: int) -> tuple[Ops, Ops]:
    f = Ops(dense=2.0 * V * D * tokens)
    return f, f.scaled(2.0)


def step_per_sequence(variant: str, s: int) -> tuple[Ops, Ops]:
    """Forward and backward FLOPs for one training sequence of length ``s``."""
    sps = STEP_TOKENS / s
    circ = variant == "hyena_circ"
    f, b = Ops(), Ops()
    for layer in range(L_LAYERS):
        if variant == "transformer" or (variant == "hybrid" and layer % 4 == 0):
            fo, bo = attention_block(s, sps)
        else:
            fo, bo = hyena_block(s, circ, sps)
        f += fo
        b += bo
        fo, bo = mlp(s, circ, sps)
        f += fo
        b += bo
    fo, bo = lm_head(s)
    return f + fo, b + bo


VARIANTS = ["transformer", "hyena", "hybrid", "hyena_circ"]
LABEL = {"transformer": "Transformer (Llama-3-8B)", "hyena": "Hyena-2",
         "hybrid": "Hybrid 1:3 attention:Hyena", "hyena_circ": "Hyena-2 + block-circulant weights"}


def n_params() -> int:
    """Parameters of the transformer variant (embedding and head untied, norms ignored)."""
    per_layer = 2 * D * D + 2 * D * KV + 3 * D * FF
    return L_LAYERS * per_layer + 2 * V * D


# ─────────────────────────────────────────────────────── numeric checks ──
def causal_fft_conv(u: np.ndarray, h: np.ndarray) -> np.ndarray:
    s = len(u)
    n = pow2_at_least(2 * s)
    return np.fft.irfft(np.fft.rfft(u, n) * np.fft.rfft(h, n), n)[:s]


def fft_conv_grads(u: np.ndarray, h: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradients of L = <g, causal_conv(u, h)> by the spectral formulas used in the ledger."""
    s = len(u)
    n = pow2_at_least(2 * s)
    G = np.fft.rfft(g, n)
    du = np.fft.irfft(G * np.conj(np.fft.rfft(h, n)), n)[:s]
    dh = np.fft.irfft(G * np.conj(np.fft.rfft(u, n)), n)[:s]
    return du, dh


def direct_grads(u: np.ndarray, h: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The same gradients as direct correlations: du[j] = sum_i g[i] h[i-j], dh[k] = sum_i g[i] u[i-k]."""
    s = len(u)
    du = np.array([np.dot(g[j:], h[: s - j]) for j in range(s)])
    dh = np.array([np.dot(g[k:], u[: s - k]) for k in range(s)])
    return du, dh


def finite_difference(u, h, g, eps=1e-6, idx=(0, 7, 31)):
    out = []
    for j in idx:
        e = np.zeros_like(u)
        e[j] = eps
        out.append((np.dot(g, causal_fft_conv(u + e, h)) - np.dot(g, causal_fft_conv(u - e, h))) / (2 * eps))
    return np.array(out)


def checks(rng) -> dict:
    res = {}
    for s in (64, 1000):
        u, h, g = rng.standard_normal(s), rng.standard_normal(s), rng.standard_normal(s)
        du, dh = fft_conv_grads(u, h, g)
        du0, dh0 = direct_grads(u, h, g)
        fd = finite_difference(u, h, g)
        res[s] = dict(du_err=float(np.max(np.abs(du - du0))), dh_err=float(np.max(np.abs(dh - dh0))),
                      fd_err=float(np.max(np.abs(fd - du[[0, 7, 31]]))))
    return res


# ─────────────────────────────────────────────────────────── precision ──
def bf16(x: np.ndarray) -> np.ndarray:
    b = x.astype(np.float32).view(np.uint32)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.astype(np.uint32).view(np.float32).astype(np.float64)


def fp8(x: np.ndarray, man: int, fmax: float, emin: int) -> np.ndarray:
    """Round to an FP8 format with ``man`` mantissa bits, per-tensor scaled to ``fmax``."""
    scale = fmax / np.max(np.abs(x))
    y = x * scale
    e = np.floor(np.log2(np.maximum(np.abs(y), 2.0 ** emin)))
    q = 2.0 ** (e - man)
    return np.round(y / q) * q / scale


def rel_rms(a: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - ref) ** 2) / np.mean(ref ** 2)))


def precision(rng) -> dict:
    """ENOB an ADC needs to match each training number format's own rounding error, on the
    output of a forward long convolution and on its input gradient dU (both length 4,096)."""
    s = 4096
    u = rng.standard_normal(s)
    h = rng.standard_normal(s) * np.exp(-np.arange(s) / (s / 8))
    g = rng.standard_normal(s) * (1 + 3 * (rng.random(s) < 0.01))   # sparse spikes, as gradients have
    y = causal_fft_conv(u, h)
    du, dh = fft_conv_grads(u, h, g)
    out = {}
    for name, sig in (("forward y", y), ("gradient dU", du), ("gradient dH", dh)):
        crest = float(np.max(np.abs(sig)) / np.sqrt(np.mean(sig ** 2)))
        fmts = {"BF16": rel_rms(bf16(sig), sig),
                "FP8 E4M3 (fwd)": rel_rms(fp8(sig, 3, 448.0, -6), sig),
                "FP8 E5M2 (grad)": rel_rms(fp8(sig, 2, 57344.0, -14), sig)}
        enob = {k: math.log2(crest * 2 / (math.sqrt(12) * v)) for k, v in fmts.items()}
        out[name] = dict(crest=crest, format_err=fmts, enob_needed=enob)
    return out


# ───────────────────────────────────────────────────────── mask budget ──
def mask_budget(variant: str, s: int) -> dict:
    """Mask values and SLM frames per optimizer step, layer-major order (the whole step's
    sequences pass through one layer's mask before it is rewritten)."""
    n = pow2_at_least(2 * s)
    seqs = STEP_TOKENS // s
    weight_vals = L_LAYERS * ORDER * D * (n // 2 + 1)
    if variant == "hyena_circ":
        k = CIRC_BLOCK
        mats = [((ORDER + 1) * D, D), (D, D), (FF, D), (FF, D), (D, FF)]
        weight_vals += L_LAYERS * sum((m // k) * (nn // k) * (k // 2 + 1) for m, nn in mats)
    # forward (H) and backward (conj H) each need the weight spectra once per step
    static_frames = 2 * math.ceil(weight_vals / SLM_PIXELS)
    # data-by-data: conj(U) of every sequence (and every token's blocks, for circulant) is a mask
    dd_vals = L_LAYERS * ORDER * D * (n // 2 + 1) * seqs
    if variant == "hyena_circ":
        dd_vals += L_LAYERS * sum((nn // CIRC_BLOCK) * (CIRC_BLOCK // 2 + 1) for _, nn in mats) * STEP_TOKENS
    dd_frames = math.ceil(dd_vals / SLM_PIXELS)
    return dict(seqs=seqs, weight_values=weight_vals, static_frames=static_frames,
                static_s=static_frames / SLM_HZ, dd_values=dd_vals, dd_frames=dd_frames,
                dd_s=dd_frames / SLM_HZ)


# ────────────────────────────────────────────────────────────── report ──
def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def main() -> None:
    rng = np.random.default_rng(20261012)
    lines: list[str] = []
    w = lines.append
    res: dict = {"conventions": {"step_tokens": STEP_TOKENS, "circ_block": CIRC_BLOCK,
                                 "slm_pixels": SLM_PIXELS, "slm_hz": SLM_HZ}}

    w("# Training FLOP-share (Amdahl) analysis: Fourier optics in LLM training\n")
    w(f"Generated by `analysis/train_flop_share.py` on {date.today().isoformat()} ({platform.machine()}, "
      f"Python {platform.python_version()}, NumPy {np.__version__}). Every number in the FOptTrn decks "
      "comes from this file.\n")
    w("Conventions: as the inference series (matmul 2mnk; real FFT 2.5 n log2 n; complex pointwise "
      "multiply 6, add 2; causal long convolutions zero-padded to a power of two at least 2L). "
      "Backward of a matmul is 2x its forward. Softmax, norms, activations, activation recomputation "
      "and AdamW's element-wise work are not counted. Shape: Llama-3-8B. "
      f"An optimizer step is {STEP_TOKENS:,} tokens (illustrative).\n")

    # 1. sanity
    w("## 1. Cross-check: the 6N rule\n")
    w("For the transformer, training FLOPs per token should be close to 6N plus the attention term "
      "6 x (2 L d s) / 2 (Kaplan et al. 2020; Chinchilla's appendix), with N the parameters that "
      "multiply activations (the input embedding is a lookup, so it is excluded).\n")
    w("| Sequence | This script (GFLOP/token) | 6N + attention (GFLOP/token) | Ratio |")
    w("|---|---|---|---|")
    n_mat = n_params() - V * D
    res["sixN"] = {}
    for s in SEQS:
        f, b = step_per_sequence("transformer", s)
        mine = (f.total + b.total) / s
        rule = 6 * n_mat + 6 * L_LAYERS * D * (s + 1)
        res["sixN"][s] = dict(script=mine, rule=rule)
        w(f"| {s:,} | {mine / 1e9:,.2f} | {rule / 1e9:,.2f} | {mine / rule:.4f} |")
    w("")

    # 2. step shares
    w("## 2. A training step: what share of the FLOPs is transform work?\n")
    w("*Optical share* = FFTs plus every Fourier-domain product. *Static share* = FFTs plus only "
      "the products whose operand is a weight spectrum, the part a mask written once per step can "
      "hold. *Amdahl bound* = 1 / (1 - share).\n")
    w("| Variant | Sequence | Fwd share | Bwd share | Step share | Static share | Data-by-data part of optical | Amdahl (all) | Amdahl (static) |")
    w("|---|---|---|---|---|---|---|---|---|")
    res["shares"] = {}
    for v in VARIANTS:
        res["shares"][v] = {}
        for s in SEQS:
            f, b = step_per_sequence(v, s)
            t = f + b
            fs, bs, ts, ss = (f.optical / f.total if f.total else 0, b.optical / b.total,
                              t.optical / t.total, t.optical_static / t.total)
            dd = t.datadata / t.optical if t.optical else 0.0
            res["shares"][v][s] = dict(fwd=fs, bwd=bs, step=ts, static=ss, datadata=dd,
                                       gflop_per_token=t.total / s / 1e9,
                                       bwd_over_fwd=b.total / f.total)
            w(f"| {LABEL[v]} | {s:,} | {pct(fs)} | {pct(bs)} | {pct(ts)} | {pct(ss)} | {pct(dd)} | "
              f"{1 / (1 - ts):.3f}x | {1 / (1 - ss):.3f}x |")
    w("")
    hy = res["shares"]["hyena"][4096]
    hc = res["shares"]["hyena_circ"][4096]
    w(f"* Training does not rescue the transform share: Hyena-2 at 4,096 tokens is {pct(hy['step'])} "
      f"of a step, against {pct(hy['fwd'])} of a forward pass. The backward pass has as many FFTs as "
      "the forward, and the matmuls double too.")
    w(f"* With block-circulant weights the step share is {pct(hc['step'])} (Amdahl {1 / (1 - hc['step']):.3f}x), "
      f"but {pct(hc['datadata'])} of that optical work is data-by-data (the weight gradient), "
      f"so a once-per-step mask holds only {pct(hc['static'])} of the step.\n")

    # 3. optimizer
    w("## 3. The optimizer step and DeMo\n")
    P = n_params()
    demo = 2 * (P / DEMO_CHUNK) * rfft(DEMO_CHUNK)      # DCT of momentum and inverse DCT of the update
    step_f = res["shares"]["transformer"][4096]["gflop_per_token"] * 1e9 * STEP_TOKENS
    res["demo"] = dict(params=P, demo_flops=demo, step_flops=step_f, share=demo / step_f)
    w(f"DeMo transforms the momentum in chunks of {DEMO_CHUNK} with a DCT and sends only the largest "
      "coefficients. Counting a DCT and an inverse DCT per chunk at the real-FFT cost:\n")
    w("| Parameters | DeMo transform FLOPs per step | Step FLOPs (transformer, 4,096) | Share |")
    w("|---|---|---|---|")
    w(f"| {P:,} | {demo:,.3e} | {step_f:,.3e} | {demo / step_f:.2e} |")
    w("\nThe optimizer's transforms are negligible in FLOPs; their value is in bytes not sent, which is "
      "the Compute in Transit series' subject, not a compute engine's.\n")

    # 4. masks
    w("## 4. Holding the masks: SLM frames per optimizer step (illustrative)\n")
    w(f"Device as the inference series: {SLM_PIXELS / 1e6:.0f}-megapixel DMD at {SLM_HZ:,.0f} Hz (8-bit), "
      "one complex value per pixel (optimistic). Layer-major order: every sequence in the step passes "
      "through one layer's mask before it is rewritten. Weight spectra change every step (an "
      "inference engine writes them once); the weight gradient's conj(U) is a new mask for every "
      "sequence.\n")
    w("| Variant | Sequence | Sequences/step | Digital step time | Weight-mask frames (fwd + bwd) | Time | Data-by-data frames | Time |")
    w("|---|---|---|---|---|---|---|---|")
    res["mask"] = {}
    for v in ("hyena", "hyena_circ"):
        res["mask"][v] = {}
        for s in (2048, 8192):
            m = mask_budget(v, s)
            m["step_s"] = res["shares"][v][s]["gflop_per_token"] * 1e9 * STEP_TOKENS / (GPUS * GPU_FLOPS)
            res["mask"][v][s] = m
            w(f"| {LABEL[v]} | {s:,} | {m['seqs']:,} | {m['step_s']:,.2f} s | {m['static_frames']:,} | {m['static_s']:,.2f} s | "
              f"{m['dd_frames']:,} | {m['dd_s']:,.0f} s |")
    w(f"\n*Digital step time* = step FLOPs on an illustrative {GPUS} GPUs at {GPU_FLOPS / 1e12:,.0f} TFLOP/s "
      "sustained each. Rewriting the weight masks every step costs time comparable to a whole digital "
      "step or more; the data-by-data products would take minutes of mask writes per step, so they "
      "stay digital, and only the weight-masked part is a candidate.\n")

    # 5. precision
    w("## 5. Precision: what a training format asks of the ADC\n")
    w("ENOB needed so the analogue read-out adds no more error than the number format's own rounding "
      "(rule: rel = crest x 2^-ENOB x 2/sqrt(12)), on a 4,096-sample long convolution and its gradients. "
      "Gradients carry rare spikes (1% of samples x4 here), which raise the crest factor.\n")
    res["precision"] = precision(rng)
    w("| Signal | Crest | BF16 | FP8 E4M3 | FP8 E5M2 |")
    w("|---|---|---|---|---|")
    for name, r in res["precision"].items():
        e = r["enob_needed"]
        w(f"| {name} | {r['crest']:.2f} | {e['BF16']:.1f} | {e['FP8 E4M3 (fwd)']:.1f} | {e['FP8 E5M2 (grad)']:.1f} |")
    w("")

    # 6. checks
    w("## 6. Numeric checks of the backward formulas\n")
    res["checks"] = checks(rng)
    w("| Length | max abs error dU (spectral vs direct) | max abs error dH | dU vs finite difference |")
    w("|---|---|---|---|")
    for s, r in res["checks"].items():
        w(f"| {s:,} | {r['du_err']:.1e} | {r['dh_err']:.1e} | {r['fd_err']:.1e} |")
    w("")

    (HERE / "results.md").write_text("\n".join(lines) + "\n")
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
