"""pytest: the backward formulas and the 6N cross-check in train_flop_share.py."""

import numpy as np

import train_flop_share as t


def test_spectral_gradients_match_direct():
    rng = np.random.default_rng(1)
    for s in (17, 64, 300):
        u, h, g = rng.standard_normal(s), rng.standard_normal(s), rng.standard_normal(s)
        du, dh = t.fft_conv_grads(u, h, g)
        du0, dh0 = t.direct_grads(u, h, g)
        assert np.allclose(du, du0, atol=1e-10)
        assert np.allclose(dh, dh0, atol=1e-10)


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(2)
    u, h, g = rng.standard_normal(64), rng.standard_normal(64), rng.standard_normal(64)
    du, _ = t.fft_conv_grads(u, h, g)
    assert np.allclose(t.finite_difference(u, h, g), du[[0, 7, 31]], atol=1e-6)


def test_transformer_matches_six_n_rule():
    n_mat = t.n_params() - t.V * t.D
    for s in t.SEQS:
        f, b = t.step_per_sequence("transformer", s)
        rule = 6 * n_mat + 6 * t.L_LAYERS * t.D * (s + 1)
        assert abs((f.total + b.total) / s / rule - 1) < 1e-12


def test_backward_is_about_twice_forward():
    for v in t.VARIANTS:
        f, b = t.step_per_sequence(v, 4096)
        assert 1.6 < b.total / f.total < 2.4


def test_transformer_has_no_transforms():
    f, b = t.step_per_sequence("transformer", 4096)
    assert (f + b).optical == 0.0
