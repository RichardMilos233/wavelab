import cmath, math
from wavelab import WaveEquation, ExplicitFD, RegularizedFD, BranchingMC
from wavelab.experiments import surfaces

# u_tt + Δu + u + u³ = 0   ==>   u_tt − (i)²Δu = −u − u³      (defocusing: a₃ = −1)

phi = lambda z: cmath.sin(math.pi * z[0]) * cmath.sin(math.pi * z[1])

SINE_DEFOCUS_CI = WaveEquation(
    dim=2, c=1j, f={1: -1, 3: -1},
    phi=phi, psi=lambda z: -phi(z),
    grad_phi=lambda z: (math.pi * cmath.cos(math.pi * z[0]) * cmath.sin(math.pi * z[1]),
                        math.pi * cmath.sin(math.pi * z[0]) * cmath.cos(math.pi * z[1])),
    domain=((0.0, 1.0), (0.0, 1.0)), name="sine_defocus_ci_2d")

times = [0.5]
fd = ExplicitFD(N=61, dt=0.002).solve(SINE_DEFOCUS_CI, times)
rfd = RegularizedFD(N=61, dt=0.002, k_max=6).solve(SINE_DEFOCUS_CI, times)
mc = BranchingMC(lam=0.25, n=10_000, seed=0, N=17).solve(SINE_DEFOCUS_CI, times)
surfaces([mc, fd, rfd], labels=["Monte Carlo", "Explicit FD", "Regularized FD"],
         path="fig8.png")
