import cmath, math
from wavelab import WaveEquation, ExplicitFD, RegularizedFD, BranchingMC, compare

# u_tt + Δu + u - u³ = 0   ==>   u_tt − (i)²Δu = −u + u³

SINE_CI = WaveEquation(dim=1, c=1j, f={1: -1, 3: 1},
                       phi=lambda z: cmath.sin(math.pi * z),
                       psi=lambda z: -cmath.sin(math.pi * z),
                       domain=((0.0, 1.0),), name="sine_ci_1d")

times = [0.1, 0.2, 0.3, 0.4]
fd = ExplicitFD(N=51, dt=0.002).solve(SINE_CI, times)
rfd = RegularizedFD(N=51, dt=0.002).solve(SINE_CI, times)
mc = BranchingMC(lam=0.25, n=40_000, seed=0, N=51).solve(SINE_CI, times)
compare(fd, rfd, mc).plot("fig6.png")
