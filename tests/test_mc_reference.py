import cmath, math
import numpy as np
import pytest
from wavelab import WaveEquation, BranchingMC

# Simulation_01 (spec §3.4): d=1, c=1, f=u^2, closed form u = 6/(z + sqrt(2) t)^2
SIM01 = WaveEquation(dim=1, c=1, f={2: 1},
                     phi=lambda z: 6 / z**2,
                     psi=lambda z: -12 * math.sqrt(2) / z**3,
                     exact=lambda z, t: 6 / (z + math.sqrt(2) * t)**2,
                     name="sim01_quadratic")

SINE_CI = WaveEquation(dim=1, c=1j, f={1: -1, 3: 1},
                       phi=lambda z: cmath.sin(math.pi * z),
                       psi=lambda z: -cmath.sin(math.pi * z),
                       domain=((0.0, 1.0),), name="sine_ci_1d")

def test_sim01_matches_closed_form():
    mc = BranchingMC(lam=0.25, n=20_000, seed=1)
    sol = mc.solve(SIM01, times=[0.5], points=[3.0 + 0j])
    exact = SIM01.exact(3.0, 0.5)
    err, se = abs(sol.u[0, 0] - exact), sol.meta["stderr"][0, 0]
    assert err < 3 * se
    assert se < 0.05

def test_fig6_center_value_regression():
    # sandbox 2026-07-13: u(0.5, 0.5) ~= 1.91, Im ~ 0 (spec §6)
    sol = BranchingMC(lam=0.25, n=20_000, seed=2).solve(SINE_CI, times=[0.5], points=[0.5])
    assert sol.u[0, 0].real == pytest.approx(1.91, abs=0.06)
    assert abs(sol.u[0, 0].imag) < 0.02

def test_q_is_variance_only_not_mean():
    # offspring distribution q is a free parameter: changing it must not move the mean
    a = BranchingMC(n=20_000, seed=3).solve(SINE_CI, times=[0.4], points=[0.5])
    b = BranchingMC(n=20_000, seed=4, q={1: 0.7, 3: 0.3}).solve(SINE_CI, times=[0.4], points=[0.5])
    tol = 3 * (a.meta["stderr"][0, 0] + b.meta["stderr"][0, 0])
    assert abs(a.u[0, 0] - b.u[0, 0]) < tol

def test_seed_reproducible():
    r1 = BranchingMC(n=500, seed=7).solve(SINE_CI, times=[0.3], points=[0.5]).u
    r2 = BranchingMC(n=500, seed=7).solve(SINE_CI, times=[0.3], points=[0.5]).u
    np.testing.assert_array_equal(r1, r2)

def test_default_points_from_domain_and_complex_points_ok():
    sol = BranchingMC(n=200, seed=0).solve(SINE_CI, times=[0.1])
    assert len(sol.points) == 21                       # linspace over domain (N default)
    off = BranchingMC(n=200, seed=0).solve(SINE_CI, times=[0.1], points=[0.5 + 0.1j])
    assert np.isfinite(off.u).all()                    # off-axis evaluation works

def test_bad_q_rejected():
    with pytest.raises(ValueError, match="q"):
        BranchingMC(q={1: 1.0}).solve(SINE_CI, times=[0.1], points=[0.5])   # missing power 3

def test_points_required_without_domain():
    with pytest.raises(ValueError, match="points"):
        BranchingMC(n=10).solve(SIM01, times=[0.1])    # SIM01 has no domain


@pytest.mark.slow
def test_paper_scale_smoke():
    """1e6 samples at one point (~3 s in pure python) — the paper-scale run is
    feasible without a compiled backend. Locks u(0.5, 0.5) from spec §6."""
    sol = BranchingMC(n=1_000_000, seed=1).solve(SINE_CI, [0.5], points=[0.5])
    assert sol.u[0, 0].real == pytest.approx(1.91, abs=0.02)


def test_N_sets_the_default_grid_size():
    """N mirrors ExplicitFD's N: how many points the d=1 default grid has."""
    sol = BranchingMC(n=200, seed=0, N=9).solve(SINE_CI, times=[0.1])
    assert len(sol.points) == 9
    np.testing.assert_allclose(sol.points.real, np.linspace(0, 1, 9))
    assert sol.params["N"] == 9


def test_explicit_points_override_N():
    sol = BranchingMC(n=200, seed=0, N=9).solve(SINE_CI, times=[0.1], points=[0.25, 0.75])
    assert len(sol.points) == 2          # points wins; N is only the default builder


def test_bad_N_rejected():
    with pytest.raises(ValueError, match="N"):
        BranchingMC(N=1)
