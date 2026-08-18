"""RegularizedFD in d=2 — the paper's §7.3 defocusing elliptic problem.

Paper (7.3): u_tt + Lap u + u + u^3 = 0 on [0,1]^2 with c = i, i.e. c=1j and
f = -u - u^3 (a_3 = -1, NOT the +1 of the focusing family). Figure 8 shows Monte
Carlo producing a smooth dome while finite differences produce +/-800 noise.

The interesting question this file answers: does the spectral cut-off rescue FD in
two dimensions the way it does in one? It does — and the sign of the cubic changes
the FAILURE MODE, which is why nothing here asserts a blow-up time.
"""
import math
import warnings
import numpy as np
import pytest

from wavelab import ExplicitFD, RegularizedFD, BranchingMC, library
from wavelab.solvers.fd_regularized import sine_lowpass

EQ = library.SINE_DEFOCUS_CI_2D          # §7.3
FOCUS = library.SINE_CI_2D               # same data, a_3 = +1


def test_tensor_product_cutoff_keeps_and_kills_the_right_modes():
    """In d=2 the projector is applied along both axes: U -> P U P. That keeps
    exactly the modes with k <= k_max AND m <= k_max."""
    N, K = 21, 4
    P = sine_lowpass(N, K)
    xi = np.linspace(0, 1, N)[1:-1]
    XI, YI = np.meshgrid(xi, xi, indexing="ij")
    for k, m in [(1, 1), (2, 3), (4, 4)]:
        V = np.sin(k * np.pi * XI) * np.sin(m * np.pi * YI)
        assert np.allclose(P @ V @ P, V, atol=1e-12)          # kept untouched
    for k, m in [(5, 1), (1, 5), (8, 8)]:
        V = np.sin(k * np.pi * XI) * np.sin(m * np.pi * YI)
        assert np.max(np.abs(P @ V @ P)) < 1e-12              # annihilated
    U = np.random.default_rng(0).standard_normal((N - 2, N - 2))
    assert np.allclose(P @ (P @ U @ P) @ P, P @ U @ P)        # idempotent


def test_solution_shape_and_meta():
    sol = RegularizedFD(N=21, dt=0.002, k_max=4).solve(EQ, times=[0.1, 0.2])
    assert sol.points.shape == (21 * 21, 2)
    assert sol.u.shape == (2, 21 * 21)
    assert sol.meta["shape"] == (21, 21)
    assert sol.meta["k_max"] == 4 and sol.params["k_max"] == 4
    assert sol.solver == "regularized_fd"


def test_dim3_still_rejected():
    with pytest.raises(NotImplementedError, match="dim"):
        RegularizedFD(N=11, dt=0.002, k_max=3).solve(library.SOLITON_3D, [0.1])


def test_regularized_2d_agrees_with_monte_carlo():
    """The headline: on §7.3 the regularized FD centre value matches the unbiased
    MC estimate, while the explicit scheme is off by two orders of magnitude."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mc = BranchingMC(lam=0.25, n=40_000, seed=0).solve(
            EQ, [0.5], points=np.array([[0.5, 0.5]]))
        reg = RegularizedFD(N=41, dt=0.002, k_max=6).solve(EQ, [0.5])
        exp = ExplicitFD(N=41, dt=0.002).solve(EQ, [0.5])
    ref, se = mc.u[0, 0].real, mc.meta["stderr"][0, 0]
    centre = reg.u[0].reshape(41, 41).real[20, 20]
    assert abs(centre - ref) < 4 * se                  # within MC error
    assert 2.5 < centre < 3.0                          # and near the paper's peak
    assert np.nanmax(np.abs(exp.u[0].real)) > 50       # explicit is garbage


def test_regularized_2d_is_grid_independent_where_explicit_is_not():
    """Refining the grid must not change a regularized answer (the cut-off, not the
    grid, sets the resolution) — while it makes the explicit answer worse."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vals, exps = [], []
        for N in (41, 61):
            r = RegularizedFD(N=N, dt=0.002, k_max=6).solve(EQ, [0.5])
            vals.append(r.u[0].reshape(N, N).real[N // 2, N // 2])
            e = ExplicitFD(N=N, dt=0.002).solve(EQ, [0.5])
            exps.append(np.nanmax(np.abs(e.u[0].real)))
    assert abs(vals[0] - vals[1]) < 1e-2               # regularized: converged
    assert exps[1] > exps[0]                           # explicit: worse when finer


def test_k_max_has_a_usable_window():
    """Unlike d=1, both ends of the trade-off are visible here: too FEW modes cannot
    resolve the true solution, too MANY let the ill-posedness back in.

    The d=1 analogue (test_more_modes_kept_means_earlier_blowup) judges by blow-up
    time. It cannot be used here: §7.3 is DEFOCUSING, -u^3 opposes growth, and
    blowup_time is always None (see test_defocusing_saturates_while_focusing_blows_up).
    So the judge is distance from an MC reference instead -- and it is RELATIVE, for
    the same reason: saturation bounds how far the k_max=20 run can drift, so no
    absolute threshold carried over from d=1 is meaningful. What the physics does
    guarantee is separation, and that is what is asserted.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mc = BranchingMC(lam=0.25, n=40_000, seed=1).solve(
            EQ, [0.5], points=np.array([[0.5, 0.5]]))
        ref = mc.u[0, 0].real
        noise = mc.meta["stderr"][0, 0].real
        got = {}
        for K in (1, 3, 6, 20):
            r = RegularizedFD(N=41, dt=0.002, k_max=K).solve(EQ, [0.5])
            got[K] = r.u[0].reshape(41, 41).real[20, 20]
    err = {K: abs(v - ref) for K, v in got.items()}

    assert err[1] > 0.2                  # under-resolved: only mode (1,1) survives
    assert err[3] < 0.15                 # inside the window (~0.046)
    assert err[6] < 0.15                 # inside the window (~0.014)

    # Outside the window: an order of magnitude worse than ANYWHERE inside it, and
    # far past MC noise -- i.e. real bias, not sampling error. Measured ~16x and ~20x.
    assert err[20] > 10 * max(err[3], err[6])
    assert err[20] > 10 * noise


def test_defocusing_saturates_while_focusing_blows_up():
    """The cubic's SIGN decides the failure mode, and §7.3 is the defocusing one.
    -u^3 opposes growth, so amplified round-off saturates: explicit FD returns
    bounded garbage with blowup_time=None (exactly the paper's Fig 8b), never NaN.
    +u^3 accelerates growth and does produce NaN. Nothing in the d=2 §7.3 study may
    therefore assert a blow-up time."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        defocus = ExplicitFD(N=41, dt=0.002).solve(EQ, [0.5])
        focus = ExplicitFD(N=41, dt=0.002).solve(FOCUS, [0.5])
    assert defocus.meta["blowup_time"] is None
    assert np.all(np.isfinite(defocus.u[0]))
    assert np.nanmax(np.abs(defocus.u[0].real)) > 50      # finite, and wrong
    assert focus.meta["blowup_time"] is not None
    assert np.all(np.isnan(focus.u[0].real))


def test_linear_growth_rate_sets_the_scale_in_2d():
    """Mode (1,1) of the d=2 problem grows at sqrt(2 pi^2 - 1), noticeably faster
    than d=1's sqrt(pi^2 - 1) — which is why the same t is harder here."""
    sig2 = math.sqrt(2 * math.pi**2 - 1)
    assert sig2 == pytest.approx(4.3289, abs=1e-3)
    a = math.cosh(sig2 * 0.5) - math.sinh(sig2 * 0.5) / sig2   # phi_11=1, psi_11=-1
    assert a == pytest.approx(3.42, abs=0.02)                  # paper Fig 8a peak ~3
