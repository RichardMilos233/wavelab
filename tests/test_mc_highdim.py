"""MC in d=2 and d=3 against the paper's closed forms.
SIM05 (d=2, c=1, f=u^2): u = 6/(z1+z2+sqrt(3) t)^2, evaluated at (4,4), lam=1.
SIM08 (d=3, c=1, f=u^2): u = 6/(z1+z2+z3+2t)^2,     evaluated at (4,4,4), lam=1.

NB: deliberately NOT validated against Simulation_07's output — that C++ sets
aJ = -1 for BOTH J=1 and J=3, although its own README states f = -u + u^3 (so
a_3 = +1). wavelab derives coefficients from eq.f, so it is right by construction;
the closed forms are the ground truth.
"""
import math
import numpy as np
import pytest
from wavelab import WaveEquation, BranchingMC

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
