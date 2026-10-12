// train_flop_model.js: JavaScript port of analysis/train_flop_share.py's training-step FLOP ledger,
// used live in FOptTrn deck 01. Same shape (Llama-3-8B), same four variants, same counting
// conventions and the same floating-point operation order, so the numbers match results.md
// (tested by analysis/test_js_parity.py, which runs this file in node).
(function (root) {
    const D = 4096, L = 32, FF = 14336, V = 128256, KV = 8 * (4096 / 32);   // Llama-3-8B
    const ORDER = 2, SHORT = 3, BLOCK = 256;
    const STEP_TOKENS = 1048576;                       // tokens per optimizer step (illustrative)
    const SLM_PIXELS = 2e6, SLM_HZ = 1031.0;           // illustrative mask device
    const GPUS = 64, GPU_FLOPS = 400e12;               // illustrative cluster
    const VARIANTS = ['transformer', 'hyena', 'hybrid', 'hyena_circ'];
    const LABEL = { transformer: 'Transformer (Llama-3-8B)', hyena: 'Hyena-2',
                    hybrid: 'Hybrid 1:3 attention:Hyena', hyena_circ: 'Hyena-2 + block-circulant weights' };
    const KEYS = ['dense', 'attention', 'transform', 'masked', 'datadata', 'other'];

    const rfft = n => 2.5 * n * Math.log2(n);           // FFTW's convention; n is a power of two
    const pow2AtLeast = n => { let p = 1; while (p < n) p *= 2; return p; };
    const div = (a, b) => Math.floor(a / b);
    const ops = (o = {}) => { const r = {}; for (const k of KEYS) r[k] = o[k] || 0; return r; };
    const add = (a, b) => { for (const k of KEYS) a[k] = a[k] + b[k]; return a; };
    const plus = (a, b) => add(ops(a), b);
    const scaled = (a, f) => { const r = {}; for (const k of KEYS) r[k] = a[k] * f; return r; };
    const total = o => o.dense + o.attention + o.transform + o.masked + o.datadata + o.other;
    const optical = o => o.transform + o.masked + o.datadata;
    const opticalStatic = o => o.transform + o.masked;

    // y = W x, W of shape (m, n), on `tokens` vectors: [forward, backward]
    function matrix(m, n, tokens, circ, sps) {
        if (!circ) { const f = ops({ dense: 2.0 * m * n * tokens }); return [f, scaled(f, 2.0)]; }
        const k = BLOCK, bins = div(k, 2) + 1, p = div(m, k), q = div(n, k);
        const f = ops({ transform: tokens * (q + p) * rfft(k), masked: tokens * p * q * bins * 6.0,
                        other: tokens * p * (q - 1) * bins * 2.0 });
        const b = ops({ transform: tokens * (p + q) * rfft(k), masked: tokens * p * q * bins * 6.0,
                        datadata: tokens * p * q * bins * 6.0,
                        other: tokens * (q * (p - 1) + p * q) * bins * 2.0 });
        f.transform = f.transform + p * q * rfft(k) / sps;   // once per step: FFT of the weight blocks
        b.transform = b.transform + p * q * rfft(k) / sps;   // once per step: IFFT of dW
        return [f, b];
    }
    function mlp(tokens, circ, sps) {
        const f = ops(), b = ops();
        for (const [m, n] of [[FF, D], [FF, D], [D, FF]]) { const [fo, bo] = matrix(m, n, tokens, circ, sps); add(f, fo); add(b, bo); }
        return [f, b];
    }
    function attentionBlock(s) {
        const f = ops({ dense: 2.0 * s * (2 * D * D + 2 * D * KV), attention: 2.0 * D * s * (s + 1) });
        return [f, scaled(f, 2.0)];
    }
    function hyenaBlock(s, circ, sps) {
        const f = ops(), b = ops();
        for (const [m, n] of [[(ORDER + 1) * D, D], [D, D]]) { const [fo, bo] = matrix(m, n, s, circ, sps); add(f, fo); add(b, bo); }
        f.other = f.other + (s * 2.0 * SHORT * (ORDER + 1) * D + s * ORDER * D);
        b.other = b.other + 2.0 * (s * 2.0 * SHORT * (ORDER + 1) * D + s * ORDER * D);
        const n = pow2AtLeast(2 * s), bins = div(n, 2) + 1, ch = ORDER * D;
        f.transform = f.transform + ch * 2 * rfft(n);         // FFT(u), IFFT(y)
        f.masked = f.masked + ch * bins * 6.0;                // U . H
        b.transform = b.transform + ch * 2 * rfft(n);         // FFT(G), IFFT(dU)
        b.masked = b.masked + ch * bins * 6.0;                // G . conj(H)
        b.datadata = b.datadata + ch * bins * 6.0;            // G . conj(U)
        b.other = b.other + ch * bins * 2.0;                  // accumulate dH spectrum
        f.transform = f.transform + ch * rfft(n) / sps;       // once per step: FFT of each filter
        b.transform = b.transform + ch * rfft(n) / sps;       // once per step: IFFT of its gradient
        return [f, b];
    }
    function lmHead(tokens) { const f = ops({ dense: 2.0 * V * D * tokens }); return [f, scaled(f, 2.0)]; }

    // Forward and backward FLOPs for one training sequence of length s: { fwd, bwd }
    function stepPerSequence(variant, s, stepTokens = STEP_TOKENS) {
        const sps = stepTokens / s, circ = variant === 'hyena_circ';
        const f = ops(), b = ops();
        for (let layer = 0; layer < L; layer++) {
            const [fo, bo] = (variant === 'transformer' || (variant === 'hybrid' && layer % 4 === 0))
                ? attentionBlock(s) : hyenaBlock(s, circ, sps);
            add(f, fo); add(b, bo);
            const [fm, bm] = mlp(s, circ, sps);
            add(f, fm); add(b, bm);
        }
        const [fh, bh] = lmHead(s);
        return { fwd: plus(f, fh), bwd: plus(b, bh) };
    }
    // The shares of results.md section 2
    function summary(variant, s, stepTokens = STEP_TOKENS) {
        const { fwd, bwd } = stepPerSequence(variant, s, stepTokens), t = plus(fwd, bwd);
        const ft = total(fwd), bt = total(bwd), tt = total(t);
        const r = { fwd, bwd, step: t, fwdTotal: ft, bwdTotal: bt, stepTotal: tt,
                    fwdShare: ft ? optical(fwd) / ft : 0, bwdShare: optical(bwd) / bt,
                    stepShare: optical(t) / tt, staticShare: opticalStatic(t) / tt,
                    datadataPart: optical(t) ? t.datadata / optical(t) : 0,
                    gflopPerToken: tt / s / 1e9, bwdOverFwd: bt / ft };
        r.amdahlAll = 1 / (1 - r.stepShare);
        r.amdahlStatic = 1 / (1 - r.staticShare);
        return r;
    }
    // SLM frames per optimizer step, layer-major order (results.md section 4; Hyena variants only)
    function maskBudget(variant, s, stepTokens = STEP_TOKENS) {
        const n = pow2AtLeast(2 * s), seqs = div(stepTokens, s);
        const mats = [[(ORDER + 1) * D, D], [D, D], [FF, D], [FF, D], [D, FF]];
        let weightVals = L * ORDER * D * (div(n, 2) + 1);
        if (variant === 'hyena_circ') {
            let a = 0; for (const [m, nn] of mats) a += div(m, BLOCK) * div(nn, BLOCK) * (div(BLOCK, 2) + 1);
            weightVals += L * a;
        }
        const staticFrames = 2 * Math.ceil(weightVals / SLM_PIXELS);
        let ddVals = L * ORDER * D * (div(n, 2) + 1) * seqs;
        if (variant === 'hyena_circ') {
            let a = 0; for (const [, nn] of mats) a += div(nn, BLOCK) * (div(BLOCK, 2) + 1);
            ddVals += L * a * stepTokens;
        }
        const ddFrames = Math.ceil(ddVals / SLM_PIXELS);
        const stepS = summary(variant, s, stepTokens).gflopPerToken * 1e9 * stepTokens / (GPUS * GPU_FLOPS);
        return { seqs, weightValues: weightVals, staticFrames, staticS: staticFrames / SLM_HZ,
                 ddValues: ddVals, ddFrames, ddS: ddFrames / SLM_HZ, stepS };
    }

    const api = { VARIANTS, LABEL, KEYS, STEP_TOKENS, stepPerSequence, summary, maskBudget, total, optical, opticalStatic };
    if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.TrainFlopModel = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
