import cmath, math, warnings
import numpy as np
import pytest
from wavelab import WaveEquation, ExplicitFD, library

# well-posed 2-D control: c=1, f=0, phi=sin(pi x)sin(pi y), psi=0
# exact: u = sin(pi x) sin(pi y) cos(sqrt(2) pi t)
LINEAR_2D = WaveEquation(dim=2, c=1, f={},
                         phi=lambda z: cmath.sin(math.pi * z[0]) * cmath.sin(math.pi * z[1]),
                         psi=lambda z: 0j,
                         domain=((0.0, 1.0), (0.0, 1.0)), name="linear_2d")

# ill-posed 2-D (paper Simulation_07 setting): c=i, f=-u+u^3
SINE_CI_2D = WaveEquation(dim=2, c=1j, f={1: -1, 3: 1},
                          phi=lambda z: cmath.sin(math.pi * z[0]) * cmath.sin(math.pi * z[1]),
                          psi=lambda z: -cmath.sin(math.pi * z[0]) * cmath.sin(math.pi * z[1]),
                          domain=((0.0, 1.0), (0.0, 1.0)), name="sine_ci_2d")

def test_2d_linear_matches_exact():
    sol = ExplicitFD(N=41, dt=0.002).solve(LINEAR_2D, times=[0.2])
    x, y = sol.points[:, 0].real, sol.points[:, 1].real
    exact = np.sin(np.pi * x) * np.sin(np.pi * y) * math.cos(math.sqrt(2) * math.pi * 0.2)
    assert np.max(np.abs(sol.u[0] - exact)) < 5e-3
    assert sol.meta["blowup_time"] is None

def test_2d_shapes_and_meta():
    sol = ExplicitFD(N=21, dt=0.002).solve(LINEAR_2D, times=[0.1, 0.2])
    assert sol.points.shape == (21 * 21, 2)
    assert sol.u.shape == (2, 21 * 21)
    assert sol.meta["shape"] == (21, 21)
    assert np.all(np.isfinite(sol.u))

def test_2d_illposed_blows_up():
    # same ill-posedness as 1-D: the c=i case explodes in finite time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = ExplicitFD(N=41, dt=0.002).solve(SINE_CI_2D, times=[0.5])
    assert sol.meta["blowup_time"] is not None
    assert sol.meta["blowup_time"] < 0.5

def test_2d_boundary_is_clamped():
    sol = ExplicitFD(N=21, dt=0.002).solve(LINEAR_2D, times=[0.1])
    g = sol.u[0].reshape(21, 21)
    for edge in (g[0, :], g[-1, :], g[:, 0], g[:, -1]):
        np.testing.assert_allclose(edge, 0, atol=1e-12)

def test_clamp_holds_even_when_phi_is_nonzero_on_the_boundary():
    """The bug the shared-loop refactor could hide: if phi(boundary) != 0 then
    2u - u_prev leaks a nonzero boundary value unless we re-clamp every step.
    Sine data has phi(0)=0, so only a non-vanishing phi exposes it."""
    eq = WaveEquation(dim=1, c=1, f={}, phi=lambda z: 1 + 0j, psi=lambda z: 0j,
                      domain=((0.0, 1.0),), name="nonzero_boundary_phi")
    sol = ExplicitFD(N=21, dt=0.002).solve(eq, times=[0.05])
    assert sol.u[0][0] == 0 and sol.u[0][-1] == 0

def test_dim3_still_unsupported():
    eq = WaveEquation(dim=3, c=1, f={}, phi=lambda z: 0j, psi=lambda z: 0j,
                      domain=((0, 1), (0, 1), (0, 1)))
    with pytest.raises(NotImplementedError):
        ExplicitFD().solve(eq, times=[0.1])


# --- paper section 7.1: defocusing Klein-Gordon, c=1 (the well-posed control) -------
# SINE_DEFOCUS_C1_2D and SINE_DEFOCUS_CI_2D share data and f = -u - u^3 and differ
# ONLY in c. Everything the section-7.3 study blames on ill-posedness therefore has
# to disappear here, and it does.

def test_defocusing_c1_is_smooth_and_bounded_where_ci_is_grid_noise():
    """Same equation, c=1 instead of c=i. The §7.3 failure is entirely the operator."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = ExplicitFD(N=41, dt=0.002).solve(library.SINE_DEFOCUS_C1_2D, times=[0.5])
        bad = ExplicitFD(N=41, dt=0.002).solve(library.SINE_DEFOCUS_CI_2D, times=[0.5])

    assert ok.meta["blowup_time"] is None
    assert np.all(np.isfinite(ok.u[0]))
    assert np.abs(ok.u[0].real).max() < 1.0            # bounded oscillation, |u| ~ 0.82

    def roughness(sol):
        g = sol.u[0].real.reshape(41, 41)
        return np.abs(g[2:, 1:-1] - 2 * g[1:-1, 1:-1] + g[:-2, 1:-1]).max()

    assert roughness(ok) < 0.05                        # ~0.005: smooth
    assert roughness(bad) > 100                        # ~620: grid-scale noise
    assert np.abs(bad.u[0].real).max() > 100           # ~170, and bounded (defocusing)
    assert bad.meta["blowup_time"] is None             # -u^3 saturates: never NaN


def test_defocusing_c1_converges_under_refinement():
    """The well-posedness fingerprint, and the exact opposite of
    test_illposed_signature.py: refining the grid makes c=1 BETTER and c=i WORSE."""
    ok, bad = [], []
    for N in (21, 41, 61):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = ExplicitFD(N=N, dt=0.002).solve(library.SINE_DEFOCUS_C1_2D, times=[0.5])
            b = ExplicitFD(N=N, dt=0.002).solve(library.SINE_DEFOCUS_CI_2D, times=[0.5])
        ok.append(a.u[0].real.reshape(N, N)[N // 2, N // 2])
        bad.append(np.abs(b.u[0].real).max())

    d1, d2 = abs(ok[0] - ok[1]), abs(ok[1] - ok[2])
    assert d1 < 0.01 and d2 < d1                       # converging: 8.8e-4 -> 1.6e-4
    assert bad[0] < bad[1] < bad[2]                    # diverging: 2.8 -> 170 -> 273
