"""Figure-6 side-by-side (FD unstable vs branching MC stable) via wavelab.
Replaces the hand-rolled ../fig6_study/fig6_side_by_side.py — same physics,
now ~15 lines. Run: <anaconda>/envs/wavelab/python.exe examples/fig6_side_by_side.py"""
import cmath, math
from wavelab import WaveEquation, ExplicitFD, BranchingMC, compare

SINE_CI = WaveEquation(dim=1, c=1j, f={1: -1, 3: 1},
                       phi=lambda z: cmath.sin(math.pi * z),
                       psi=lambda z: -cmath.sin(math.pi * z),
                       domain=((0.0, 1.0),), name="sine_ci_1d")

times = [0.1, 0.2, 0.3, 0.4]
fd = ExplicitFD(N=51, dt=0.002).solve(SINE_CI, times)
mc = BranchingMC(lam=0.25, n=40_000, seed=0).solve(SINE_CI, times)   # 21-pt default grid
cmp = compare(fd, mc)
print(cmp.table(probe_points=[0.1, 0.3, 0.5, 0.7, 0.9]))
cmp.plot("fig6_side_by_side.png")
print("saved fig6_side_by_side.png")
