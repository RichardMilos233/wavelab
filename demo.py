from wavelab import WaveEquation, ExplicitFD, RegularizedFD, BranchingMC, compare
import cmath, math
# u_tt - c^2 * u_xx = f

eq = WaveEquation(
    dim=1,                          # 1 space dimension
    c=1j,                          # wave speed (c=1 -> ordinary, well-posed)
    f={1:-1, 2:1},                           # no source term: u_tt = c² u_xx
    phi=lambda z: cmath.sin(math.pi * z),         # u(x, 0)
    psi=lambda z: -cmath.sin(math.pi * z),               # initial velocity, zero
    domain=((0.0, 1.0),),
    name="my_wave"
)


times = [0.0, 0.1, 0.2]
fd = ExplicitFD(N=101, dt=0.002).solve(eq, times)
rfd = RegularizedFD(N=101, dt=0.002, k_max=12).solve(eq, times)
# mc = BranchingMC(lam=0.25, n=20_000, seed=0).solve(eq, times)
compare(fd, rfd).plot("demo.png")