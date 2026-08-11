"""Acceptance gate for the fig6_study -> wavelab refactor (spec §6).
Baseline verified 2026-07-13 in the original `fig6_study` sandbox (outside this
repo) — do NOT edit the expected numbers to make this pass; a mismatch means the
port changed physics."""
import cmath, math
import numpy as np
import pytest
from wavelab import WaveEquation, ExplicitFD, BranchingMC

SINE_CI = WaveEquation(dim=1, c=1j, f={1: -1, 3: 1},
                       phi=lambda z: cmath.sin(math.pi * z),
                       psi=lambda z: -cmath.sin(math.pi * z),
                       domain=((0.0, 1.0),), name="sine_ci_1d")
TIMES = [0.1, 0.2, 0.3, 0.4]

def test_fd_side():
    fd = ExplicitFD(N=51, dt=0.002).solve(SINE_CI, TIMES)
    for i, expected in enumerate([0.948, 0.991, 1.137]):   # t = 0.1, 0.2, 0.3
        assert np.nanmax(np.abs(fd.u[i])) == pytest.approx(expected, abs=0.01)
    assert np.nanmax(np.abs(fd.u[3])) > 5                   # t=0.4: oscillations exploded

def test_mc_side_and_agreement():
    fd = ExplicitFD(N=51, dt=0.002).solve(SINE_CI, TIMES)
    mc = BranchingMC(lam=0.25, n=20_000, seed=5).solve(SINE_CI, TIMES, points=[0.1, 0.5])
    expect_center = {0.1: 0.948, 0.2: 0.990, 0.3: 1.135, 0.4: 1.412}
    for i, t in enumerate(TIMES):
        assert mc.u[i, 1].real == pytest.approx(expect_center[t], abs=0.06)
        assert abs(mc.u[i, 1].imag) < 0.02
    # where FD is still sane (t <= 0.3) the two methods agree at x=0.5
    for i in range(3):
        assert abs(fd.u[i, 25].real - mc.u[i, 1].real) < 0.05   # index 25 = x=0.5 on N=51
