import cmath, math, warnings
import numpy as np
import pytest
from wavelab import WaveEquation, ExplicitFD

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
