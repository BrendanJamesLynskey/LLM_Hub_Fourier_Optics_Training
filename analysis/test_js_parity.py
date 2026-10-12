"""Parity: js/train_flop_model.js must reproduce train_flop_share.py's training-step ledger.

FOptTrn deck 01 runs an inline copy of the JS live. Run: pytest -q analysis   (needs node on PATH;
skipped without it).
"""
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

import train_flop_share as t

JS = Path(__file__).resolve().parent.parent / "js" / "train_flop_model.js"
SEQS = sorted(set(t.SEQS) | {512, 131072})
STEPS = [t.STEP_TOKENS, 1 << 17, 1 << 22]
CASES = [(v, s, st) for v in t.VARIANTS for s in SEQS for st in STEPS]
MASK_CASES = [(v, s, st) for v in ("hyena", "hyena_circ") for s in SEQS for st in STEPS]
KEYS = ("dense", "attention", "transform", "masked", "datadata", "other")
RTOL = 1e-12


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=0.0)


@pytest.fixture(scope="module")
def js():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    script = (f"const M = require({json.dumps(str(JS))});"
              "const [cases, mcases] = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
              "console.log(JSON.stringify([cases.map(([v, s, st]) => M.summary(v, s, st)),"
              " mcases.map(([v, s, st]) => M.maskBudget(v, s, st))]));")
    out = subprocess.run([node, "-e", script], input=json.dumps([CASES, MASK_CASES]),
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_step_ledger_matches(js, monkeypatch):
    for (v, s, st), r in zip(CASES, js[0]):
        monkeypatch.setattr(t, "STEP_TOKENS", st)
        f, b = t.step_per_sequence(v, s)
        for k in KEYS:
            assert close(r["fwd"][k], getattr(f, k)), (v, s, st, "fwd", k)
            assert close(r["bwd"][k], getattr(b, k)), (v, s, st, "bwd", k)
        tt = f + b
        assert close(r["stepTotal"], tt.total), (v, s, st)


def test_shares_match(js, monkeypatch):
    for (v, s, st), r in zip(CASES, js[0]):
        monkeypatch.setattr(t, "STEP_TOKENS", st)
        f, b = t.step_per_sequence(v, s)
        tt = f + b
        want = dict(fwdShare=f.optical / f.total, bwdShare=b.optical / b.total,
                    stepShare=tt.optical / tt.total, staticShare=tt.optical_static / tt.total,
                    datadataPart=tt.datadata / tt.optical if tt.optical else 0.0,
                    gflopPerToken=tt.total / s / 1e9, bwdOverFwd=b.total / f.total,
                    amdahlAll=1 / (1 - tt.optical / tt.total),
                    amdahlStatic=1 / (1 - tt.optical_static / tt.total))
        for k, w in want.items():
            assert close(r[k], w), (v, s, st, k, r[k], w)


def test_mask_budget_matches(js, monkeypatch):
    for (v, s, st), r in zip(MASK_CASES, js[1]):
        monkeypatch.setattr(t, "STEP_TOKENS", st)
        m = t.mask_budget(v, s)
        for k_js, k_py in (("seqs", "seqs"), ("weightValues", "weight_values"),
                           ("staticFrames", "static_frames"), ("ddValues", "dd_values"),
                           ("ddFrames", "dd_frames")):
            assert r[k_js] == m[k_py], (v, s, st, k_js)
        assert close(r["staticS"], m["static_s"]) and close(r["ddS"], m["dd_s"]), (v, s, st)
        f, b = t.step_per_sequence(v, s)
        step_s = (f + b).total / s / 1e9 * 1e9 * st / (t.GPUS * t.GPU_FLOPS)
        assert close(r["stepS"], step_s), (v, s, st)


def test_default_reproduces_results_md(js):
    """Default STEP_TOKENS rows equal results.json, which results.md is printed from."""
    res = json.loads((Path(__file__).parent / "results.json").read_text())
    for (v, s, st), r in zip(CASES, js[0]):
        if st != t.STEP_TOKENS or str(s) not in res["shares"][v]:
            continue
        want = res["shares"][v][str(s)]
        for k_js, k_py in (("fwdShare", "fwd"), ("bwdShare", "bwd"), ("stepShare", "step"),
                           ("staticShare", "static"), ("datadataPart", "datadata"),
                           ("gflopPerToken", "gflop_per_token")):
            assert close(r[k_js], want[k_py]), (v, s, k_js)
