"""Ill-posedness fingerprint: refining the grid makes blow-up EARLIER (spec §6)."""
import cmath, math
import pytest
from wavelab import WaveEquation, ExplicitFD

SINE_CI = WaveEquation(dim=1, c=1j, f={1: -1, 3: 1},
                       phi=lambda z: cmath.sin(math.pi * z),
                       psi=lambda z: -cmath.sin(math.pi * z),
                       domain=((0.0, 1.0),), name="sine_ci_1d")

@pytest.mark.parametrize("N,expected", [(51, 0.44), (101, 0.232), (201, 0.128)])
def test_blowup_times(N, expected):
    # the solver only marches to max(times), so request t=0.5 to expose the blow-up
    sol = ExplicitFD(N=N, dt=0.002).solve(SINE_CI, times=[0.5])
    assert sol.meta["blowup_time"] == pytest.approx(expected, abs=0.02)

def test_finer_is_worse():
    bt = [ExplicitFD(N=N, dt=0.002).solve(SINE_CI, times=[0.5]).meta["blowup_time"]
          for N in (51, 101, 201)]
    assert bt[0] > bt[1] > bt[2]
