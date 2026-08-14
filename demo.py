from wavelab import WaveEquation, ExplicitFD, RegularizedFD, BranchingMC, compare
import cmath, math
# u_tt - c^2 * u_xx = f

eq = WaveEquation(
    dim=1,                                        # 1 space dimension
    c=1j,                                         # wave speed (c=1 -> well-posed, c=1j -> ill-posed)
    f={1:-1, 3:1},                                # outer force
    phi=lambda z: cmath.sin(math.pi * z),         # u(x, 0)
    psi=lambda z: -cmath.sin(math.pi * z),        # u_t(x, 0)
    domain=((0.0, 1.0),),
    name="my_wave"
)


times = [0.0, 0.1, 0.2, 0.5]
fd = ExplicitFD(N=101, dt=0.002).solve(eq, times)
rfd = RegularizedFD(N=101, dt=0.002, k_max=12).solve(eq, times)
mc = BranchingMC(lam=0.25, n=20_000, seed=0).solve(eq, times)
compare(fd, rfd, mc).plot("demo.png")