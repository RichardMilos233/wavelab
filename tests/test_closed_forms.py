"""MC vs paper closed forms (../Nonlinear_Wave_simulations/README.md).
SIM01: u_tt - u_xx = u^2,  u = 6/(z + sqrt(2) t)^2.
SOLITON_1D (Sims 4/6/9, d=1): u_tt + u_xx = -u + u^3  (c=i),
  phi = tanh(i z / sqrt(6)), psi = -sqrt(2/3) sech^2(i z / sqrt(6)),
  exact u = tanh((i z - 2 t) / sqrt(6)),  evaluated at z = -1."""
import cmath, math
import pytest
from wavelab import WaveEquation, BranchingMC

S6 = math.sqrt(6.0)
SOLITON_1D = WaveEquation(
    dim=1, c=1j, f={1: -1, 3: 1},
    phi=lambda z: cmath.tanh(1j * z / S6),
    psi=lambda z: -math.sqrt(2.0 / 3.0) / cmath.cosh(1j * z / S6)**2,
    exact=lambda z, t: cmath.tanh((1j * z - 2 * t) / S6),
    name="soliton_1d")

SIM01 = WaveEquation(dim=1, c=1, f={2: 1},
                     phi=lambda z: 6 / z**2,
                     psi=lambda z: -12 * math.sqrt(2) / z**3,
                     exact=lambda z, t: 6 / (z + math.sqrt(2) * t)**2,
                     name="sim01_quadratic")

@pytest.mark.parametrize("t", [0.25, 0.5, 1.0])
def test_sim01_multiple_times(t):
    sol = BranchingMC(lam=0.25, n=20_000, seed=int(t * 100)).solve(
        SIM01, times=[t], points=[3.0 + 0j])
    err = abs(sol.u[0, 0] - SIM01.exact(3.0, t))
    assert err < 3 * sol.meta["stderr"][0, 0]

@pytest.mark.parametrize("t", [0.25, 0.5])
def test_soliton_1d(t):
    sol = BranchingMC(lam=0.25, n=20_000, seed=int(t * 100)).solve(
        SOLITON_1D, times=[t], points=[-1.0 + 0j])
    err = abs(sol.u[0, 0] - SOLITON_1D.exact(-1.0, t))
    assert err < 3 * sol.meta["stderr"][0, 0]

@pytest.mark.slow
def test_soliton_1d_longer_time():
    sol = BranchingMC(lam=0.25, n=200_000, seed=42).solve(
        SOLITON_1D, times=[1.0], points=[-1.0 + 0j])
    err = abs(sol.u[0, 0] - SOLITON_1D.exact(-1.0, 1.0))
    assert err < 3 * sol.meta["stderr"][0, 0]
