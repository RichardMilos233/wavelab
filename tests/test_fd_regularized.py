"""Regularized FD: the only way to march an ill-posed problem stably is to stop
solving it exactly. Spectral cut-off (keep modes k <= k_max) buys stability with
accuracy — the paper's Figure-7 trade-off, with a knob you can actually see.

All numbers below measured 2026-07-13 at N=101, dt=0.002, against branching MC
(n=40k) as ground truth.
"""
import cmath, math, warnings
import numpy as np
import pytest
from wavelab import WaveEquation, ExplicitFD, RegularizedFD
from wavelab.solvers.fd_regularized import sine_lowpass

SINE_CI = WaveEquation(dim=1, c=1j, f={1: -1, 3: 1},
                       phi=lambda z: cmath.sin(math.pi * z),
                       psi=lambda z: -cmath.sin(math.pi * z),
                       domain=((0.0, 1.0),), name="sine_ci_1d")

# branching-MC truth at x=0.5 (n=40k, seed=3)
TRUTH = {0.1: 0.948, 0.2: 0.992, 0.3: 1.140, 0.4: 1.418, 0.7: 4.559}


def test_lowpass_is_an_idempotent_projector():
    P = sine_lowpass(N=51, k_max=10)
    np.testing.assert_allclose(P @ P, P, atol=1e-10)      # P^2 = P
    np.testing.assert_allclose(P, P.T, atol=1e-10)        # orthogonal projector


def test_lowpass_keeps_low_modes_and_kills_high_ones():
    N, K = 51, 10
    P = sine_lowpass(N, K)
    x = np.linspace(0, 1, N)[1:-1]
    kept = np.sin(np.pi * 3 * x)                          # k=3 <= K: preserved
    killed = np.sin(np.pi * 25 * x)                       # k=25 >  K: removed
    np.testing.assert_allclose(P @ kept, kept, atol=1e-10)
    assert np.max(np.abs(P @ killed)) < 1e-10


def test_lowpass_rejects_bad_kmax():
    with pytest.raises(ValueError, match="k_max"):
        sine_lowpass(N=51, k_max=0)
    with pytest.raises(ValueError, match="k_max"):
        sine_lowpass(N=51, k_max=50)                      # > N-2


@pytest.mark.parametrize("t", [0.1, 0.2, 0.3, 0.4])
def test_tracks_mc_where_the_solution_is_still_tame(t):
    sol = RegularizedFD(N=101, dt=0.002, k_max=12).solve(SINE_CI, times=[t])
    assert sol.u[0][50].real == pytest.approx(TRUTH[t], abs=0.01)


def test_survives_far_past_the_explicit_blowup_and_stays_smooth():
    """Explicit dies at t=0.232 (N=101). The regularized scheme runs to t=0.7 —
    the range the paper's Figure 7 plots — and stays smooth."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = ExplicitFD(N=101, dt=0.002).solve(SINE_CI, times=[0.4])
    reg = RegularizedFD(N=101, dt=0.002, k_max=12).solve(SINE_CI, times=[0.7])
    assert np.all(np.isnan(exp.u[0].real))                # explicit: dead
    assert reg.meta["blowup_time"] is None                # regularized: alive
    u = reg.u[0].real
    assert np.all(np.isfinite(u))
    roughness = np.abs(u[2:] - 2 * u[1:-1] + u[:-2]).max()
    assert roughness < 0.1                                # no grid-scale oscillation


def test_stability_is_bought_with_accuracy():
    """The Figure-7 trade-off: smooth and stable at t=0.7, but visibly off — the
    error is ~30x the MC standard error, i.e. real bias, not noise."""
    reg = RegularizedFD(N=101, dt=0.002, k_max=12).solve(SINE_CI, times=[0.7])
    err = abs(reg.u[0][50].real - TRUTH[0.7])
    assert err > 0.05                                     # measurably inaccurate
    assert err < 0.5                                      # but still the right ballpark


def test_more_modes_kept_means_earlier_blowup():
    """The cut-off IS the regularization. Keep more modes and you recover the
    ill-posedness you were suppressing — the same fingerprint as refining the grid."""
    bt = {}
    for K in (12, 20, 30):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = RegularizedFD(N=101, dt=0.002, k_max=K).solve(SINE_CI, times=[0.7])
        bt[K] = sol.meta["blowup_time"]
    assert bt[12] is None                                 # k_max=12: survives to 0.7
    assert bt[20] == pytest.approx(0.64, abs=0.03)
    assert bt[30] == pytest.approx(0.438, abs=0.03)
    assert bt[20] > bt[30]                                # more modes -> dies sooner
