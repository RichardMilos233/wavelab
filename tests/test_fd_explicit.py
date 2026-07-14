import cmath, math
import numpy as np
import pytest
from wavelab import WaveEquation, ExplicitFD

# well-posed control: c=1, f=0, phi=sin(pi x), psi=0 on [0,1] Dirichlet
# exact solution u = sin(pi x) cos(pi t)
LINEAR = WaveEquation(dim=1, c=1, f={},
                      phi=lambda z: cmath.sin(math.pi * z),
                      psi=lambda z: 0j,
                      domain=((0.0, 1.0),), name="linear_control")

# the Figure-6 problem: c=i, f = -u + u^3, sine data (spec §3.4 SINE_CI_1D)
SINE_CI = WaveEquation(dim=1, c=1j, f={1: -1, 3: 1},
                       phi=lambda z: cmath.sin(math.pi * z),
                       psi=lambda z: -cmath.sin(math.pi * z),
                       domain=((0.0, 1.0),), name="sine_ci_1d")

def test_linear_matches_exact():
    sol = ExplicitFD(N=101, dt=0.001).solve(LINEAR, times=[0.5])
    exact = np.sin(np.pi * sol.points.real) * math.cos(math.pi * 0.5)
    assert np.max(np.abs(sol.u[0] - exact)) < 1e-3
    assert sol.meta["blowup_time"] is None

def test_wellposed_refinement_improves():
    errs = []
    for N in (51, 101):
        sol = ExplicitFD(N=N, dt=0.0005).solve(LINEAR, times=[0.5])
        exact = np.sin(np.pi * sol.points.real) * math.cos(math.pi * 0.5)
        errs.append(np.max(np.abs(sol.u[0] - exact)))
    assert errs[1] < errs[0]      # finer grid -> smaller error (well-posed)

def test_illposed_blowup_time_regression():
    # verified in the fig6_study sandbox 2026-07-13: N=101, dt=0.002 -> t ~= 0.232
    sol = ExplicitFD(N=101, dt=0.002).solve(SINE_CI, times=[0.1, 0.4])
    assert sol.meta["blowup_time"] == pytest.approx(0.232, abs=0.01)
    assert np.max(np.abs(sol.u[0])) == pytest.approx(0.948, abs=0.01)  # t=0.1 fine
    assert np.all(np.isnan(sol.u[1].real))                             # t=0.4 after blow-up

def test_time_zero_snapshot_is_initial_data():
    sol = ExplicitFD(N=51, dt=0.002).solve(SINE_CI, times=[0.0])
    np.testing.assert_allclose(sol.u[0].real, np.sin(np.pi * sol.points.real), atol=1e-12)

def test_requires_domain():
    eq = WaveEquation(dim=1, c=1, f={}, phi=lambda z: 0j, psi=lambda z: 0j)
    with pytest.raises(ValueError, match="domain"):
        ExplicitFD().solve(eq, times=[0.1])

def test_rejects_offgrid_time():
    with pytest.raises(ValueError, match="multiple of dt"):
        ExplicitFD(N=51, dt=0.002).solve(SINE_CI, times=[0.1001])   # not a multiple of dt
