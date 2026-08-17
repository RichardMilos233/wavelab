"""MC in d=2 and d=3 against the paper's closed forms.
SIM05 (d=2, c=1, f=u^2): u = 6/(z1+z2+sqrt(3) t)^2, evaluated at (4,4), lam=1.
SIM08 (d=3, c=1, f=u^2): u = 6/(z1+z2+z3+2t)^2,     evaluated at (4,4,4), lam=1.

Ground truth here is the closed forms, which is what these assertions use.

(Historical note, corrected 2026-08-17: this file used to say Simulation_07 was
unusable because it sets aJ = -1 for both J=1 and J=3 against its own README's
f = -u + u^3. The CODE is right and the README is wrong — aJ = -1 for both powers
is the paper's SS7.3, and that run produced Figure 8a. Its output is a valid
cross-check; see docs/agents/gotchas.md, where the comparison is recorded.)
"""
import math
import numpy as np
import pytest
from wavelab import WaveEquation, BranchingMC, library

S3 = math.sqrt(3.0)
SIM05 = WaveEquation(
    dim=2, c=1, f={2: 1},
    phi=lambda z: 6 / (z[0] + z[1])**2,
    psi=lambda z: -12 * S3 / (z[0] + z[1])**3,
    grad_phi=lambda z: (-12 / (z[0] + z[1])**3, -12 / (z[0] + z[1])**3),
    exact=lambda z, t: 6 / (z[0] + z[1] + S3 * t)**2,
    name="sim05_quadratic_2d")

SIM08 = WaveEquation(
    dim=3, c=1, f={2: 1},
    phi=lambda z: 6 / (z[0] + z[1] + z[2])**2,
    psi=lambda z: -24 / (z[0] + z[1] + z[2])**3,
    grad_phi=lambda z: tuple(-12 / (z[0] + z[1] + z[2])**3 for _ in range(3)),
    exact=lambda z, t: 6 / (z[0] + z[1] + z[2] + 2 * t)**2,
    name="sim08_quadratic_3d")

@pytest.mark.parametrize("t", [0.5, 1.0])
def test_sim05_d2_matches_closed_form(t):
    z = np.array([[4 + 0j, 4 + 0j]])
    sol = BranchingMC(lam=1.0, n=20_000, seed=int(t * 10)).solve(SIM05, [t], points=z)
    err = abs(sol.u[0, 0] - SIM05.exact([4, 4], t))
    assert err < 3 * sol.meta["stderr"][0, 0]

@pytest.mark.parametrize("t", [0.5, 1.0])
def test_sim08_d3_matches_closed_form(t):
    z = np.array([[4 + 0j, 4 + 0j, 4 + 0j]])
    sol = BranchingMC(lam=1.0, n=20_000, seed=int(t * 10)).solve(SIM08, [t], points=z)
    err = abs(sol.u[0, 0] - SIM08.exact([4, 4, 4], t))
    assert err < 3 * sol.meta["stderr"][0, 0]

def test_points_shape_preserved():
    z = np.array([[4 + 0j, 4 + 0j], [5 + 0j, 5 + 0j]])
    sol = BranchingMC(lam=1.0, n=500, seed=0).solve(SIM05, [0.5], points=z)
    assert sol.points.shape == (2, 2) and sol.u.shape == (1, 2)

def test_grad_phi_required_for_d2():
    no_grad = WaveEquation(dim=2, c=1, f={2: 1},
                           phi=lambda z: 6 / (z[0] + z[1])**2,
                           psi=lambda z: -12 * S3 / (z[0] + z[1])**3)
    with pytest.raises(ValueError, match="grad_phi"):
        BranchingMC(n=10).solve(no_grad, [0.5], points=np.array([[4 + 0j, 4 + 0j]]))

def test_points_required_for_d2():
    with pytest.raises(ValueError, match="points"):
        BranchingMC(n=10).solve(SIM05, [0.5])


def test_default_grid_is_N_per_axis_in_d2():
    """`N` means points PER AXIS in every dimension, exactly as in ExplicitFD, so
    the d=2 default grid is N x N and carries meta["shape"] for plotting."""
    sol = BranchingMC(n=200, seed=0, N=6).solve(library.SINE_DEFOCUS_CI_2D, [0.1])
    assert sol.points.shape == (36, 2)
    assert sol.meta["shape"] == (6, 6)
    # same convention as ExplicitFD: indexing="ij", row-major ravel
    ax = np.linspace(0.0, 1.0, 6)
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    np.testing.assert_allclose(sol.points[:, 0].real, X.ravel())
    np.testing.assert_allclose(sol.points[:, 1].real, Y.ravel())


def test_explicit_points_override_the_default_grid():
    pts = np.array([[0.5, 0.5], [0.25, 0.25]])
    sol = BranchingMC(n=200, seed=0, N=50).solve(
        library.SINE_DEFOCUS_CI_2D, [0.1], points=pts)
    assert len(sol.points) == 2               # not 50*50
    assert "shape" not in sol.meta            # we don't know how they're arranged


def test_d1_default_grid_unchanged():
    sol = BranchingMC(n=200, seed=0).solve(library.SINE_CI_1D, [0.1])
    assert len(sol.points) == 21
    assert "shape" not in sol.meta            # 1-D needs no reshape hint
