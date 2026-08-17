"""Regularized explicit FD: leapfrog + spectral low-pass filter (spec §3.3).

THE POINT. On the ill-posed (c=i) problem the mode sin(k pi x) of the TRUE solution
grows like exp(t*sqrt((k pi)^2 - 1)) ~ exp(k pi t). The grid-scale mode (k ~ N) thus
grows like exp(2t/dx) — unbounded as dx -> 0. Any CONSISTENT time-marching scheme
must reproduce that growth, so it must blow up once round-off (1e-16) in those modes
is amplified to O(1). This is not a defect of any particular scheme:

  * explicit leapfrog       -> blows up loudly (NaN at t ~ 0.232, N=101)
  * theta-scheme (implicit) -> energy-CONSERVING (roots g+ g- = 1, so one root is
                               always outside the unit circle) -> also blows up
  * (linearly) implicit Euler -> amplification 1/(1 -+ dt sqrt(omega)) has a POLE at
                               dt sqrt(omega) = 1; some mode always sits near it

The only way to march an ill-posed problem stably is to STOP SOLVING IT EXACTLY:
regularize by removing the modes that carry the blow-up. That is what this class
does, and it is why the paper's implicit curve is "more stable but exhibits a loss
of accuracy ... due to loss of energy conservation" — the stability is bought, not
earned.

Regularization: after every leapfrog step, project the interior state onto the first
`k_max` Dirichlet sine modes sin(k pi x), k = 1..k_max (a spectral cut-off, i.e.
quasi-reversibility / mode truncation). Modes above the cut-off are set to zero, so
their exponential growth can never be excited by round-off.

d=2 (paper §7.1/§7.3, the [0,1]^2 sine problems). The Dirichlet eigenfunctions of the
2-D discrete Laplacian are the TENSOR PRODUCTS sin(k pi x) sin(m pi y), so the cut-off
is separable and the projector never has to be formed as an (N-2)^2 square matrix:

    U  ->  P U P          (U = interior values as an (N-2)x(N-2) array, P symmetric)

which keeps modes with k <= k_max AND m <= k_max — a BOX in mode space, i.e. exactly
"the first k_max modes along each axis", the natural reading of the 1-D parameter.
Cost is two (N-2)^3 matmuls per step instead of an (N-2)^4 one.

CAUTION when carrying 1-D intuition across: mode (k, m) grows like
exp(t*sqrt((k^2+m^2) pi^2 - 1)), so the fastest mode a box cut-off retains is
(k_max, k_max), growing at sqrt(2)*k_max*pi — NOT k_max*pi. A 2-D run with k_max = K
is therefore roughly as aggressive as a 1-D run with sqrt(2)*K, and dies sooner than
the 1-D death-time formula would suggest. (A radial cut-off k^2 + m^2 <= k_max^2 would
match the 1-D rate exactly, but is not separable and so costs an order more; it is not
implemented.)

Trade-off, and it is the whole story: `k_max` small enough to kill the runaway modes,
large enough to keep the ones the true solution actually uses (the initial datum here
is pure k=1, and the cubic term feeds k=3,5,7,...). Push t out far enough and even the
retained modes need a bigger k_max than stability allows — the accuracy loss is
structural, not a tuning failure. Contrast with branching MC, which needs no
regularization at all because it never marches a coupled state forward.
"""
import warnings
import numpy as np

from wavelab.equation import WaveEquation
from wavelab.solution import Solution


def sine_lowpass(N: int, k_max: int) -> np.ndarray:
    """Projection matrix onto the first k_max Dirichlet sine modes (interior points).

    Interior nodes j = 1..N-2, modes k = 1..k_max, basis sin(pi k j / (N-1)).
    The DST-I basis is orthogonal with  sum_j sin(pi k j/(N-1)) sin(pi m j/(N-1))
    = (N-1)/2 * delta_km, so the projector is (2/(N-1)) * S_K @ S_K.T.

    In d=2 this same matrix is applied along each axis: U -> P U P.
    """
    n = N - 2
    if not (1 <= k_max <= n):
        raise ValueError(f"k_max must lie in [1, {n}] for N={N}, got {k_max}")
    j = np.arange(1, N - 1)[:, None]          # (n, 1) interior node indices
    k = np.arange(1, k_max + 1)[None, :]      # (1, K) retained mode numbers
    S = np.sin(np.pi * k * j / (N - 1))       # (n, K)
    return (2.0 / (N - 1)) * (S @ S.T)        # (n, n) projector, idempotent


class RegularizedFD:
    name = "regularized_fd"

    def __init__(self, N: int = 101, dt: float = 0.002, k_max: int = 12):
        self.N, self.dt, self.k_max = N, dt, k_max

    def _check(self, eq, times):
        if eq.bc != "dirichlet":
            raise NotImplementedError("RegularizedFD supports Dirichlet BC only")
        if eq.domain is None:
            raise ValueError("RegularizedFD requires eq.domain, e.g. ((0, 1),)")
        times = np.atleast_1d(np.asarray(times, dtype=float))
        steps_of = np.round(times / self.dt).astype(int)
        if np.any(np.abs(steps_of * self.dt - times) > 1e-9):
            raise ValueError(f"each requested time must be a multiple of dt={self.dt} "
                             f"(so snapshots are exact); got {times}")
        return times, steps_of

    def solve(self, eq: WaveEquation, times, points=None) -> Solution:
        if eq.dim == 1:
            return self._solve_1d(eq, *self._check(eq, times))
        if eq.dim == 2:
            return self._solve_2d(eq, *self._check(eq, times))
        raise NotImplementedError(
            f"RegularizedFD supports dim=1 and dim=2, got {eq.dim}")

    def _march(self, eq, times, steps_of, u_prev, u, rhs, filt, points, extra_meta):
        """Leapfrog with the spectral cut-off applied after every step."""
        dt, N, K = self.dt, self.N, self.k_max
        out = np.full((len(times), u.shape[0]), np.nan + 1j * np.nan,
                      dtype=np.complex128)
        blowup_time = None
        snaps = {0: u_prev.copy(), 1: u.copy()}
        for n in range(2, int(steps_of.max()) + 1):
            u_next = 2 * u - u_prev + dt**2 * rhs(u)
            filt(u_next)                                  # <- the regularization
            u_prev, u = u, u_next
            if not np.all(np.isfinite(u)):
                blowup_time = round(n * dt, 10)
                warnings.warn(f"{eq.name or 'equation'}: regularized FD blew up at "
                              f"t={blowup_time} (N={N}, dt={dt}, k_max={K}); "
                              f"later snapshots are NaN")
                break
            snaps[n] = u.copy()
        for i, s in enumerate(steps_of):
            if s in snaps:
                out[i] = snaps[s]
        meta = {"blowup_time": blowup_time, "dt": dt, "N": N, "k_max": K}
        meta.update(extra_meta)
        return Solution(eq=eq, solver=self.name,
                        params={"N": N, "dt": dt, "k_max": K},
                        times=times, points=points, u=out, meta=meta)

    def _solve_1d(self, eq, times, steps_of):
        dt, N, K = self.dt, self.N, self.k_max
        (a, b), = eq.domain
        x = np.linspace(a, b, N)
        dx = x[1] - x[0]
        c2 = complex(eq.c) ** 2
        f = eq.f_callable()
        P = sine_lowpass(N, K).astype(np.complex128)     # spectral cut-off projector

        def rhs(u):
            d2 = np.zeros_like(u)
            d2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
            r = c2 * d2 + f(u)
            r[0] = r[-1] = 0.0
            return r

        def filt(u):
            """Project the interior onto modes k <= k_max; boundary stays clamped."""
            u[1:-1] = P @ u[1:-1]
            u[0] = u[-1] = 0.0

        u_prev = np.array([complex(eq.phi(z)) for z in x], dtype=np.complex128)
        v0 = np.array([complex(eq.psi(z)) for z in x], dtype=np.complex128)
        filt(u_prev)
        u = u_prev + dt * v0 + 0.5 * dt**2 * rhs(u_prev)
        filt(u)
        return self._march(eq, times, steps_of, u_prev, u, rhs, filt,
                           x.astype(np.complex128), {"dx": dx})

    def _solve_2d(self, eq, times, steps_of):
        dt, N, K = self.dt, self.N, self.k_max
        (ax, bx), (ay, by) = eq.domain
        x = np.linspace(ax, bx, N)
        y = np.linspace(ay, by, N)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")
        c2 = complex(eq.c) ** 2
        f = eq.f_callable()
        P = sine_lowpass(N, K).astype(np.complex128)     # applied along BOTH axes

        def rhs(uf):
            u = uf.reshape(N, N)
            d2 = np.zeros_like(u)
            d2[1:-1, 1:-1] = (
                (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
                + (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
            )
            r = c2 * d2 + f(u)
            r[0, :] = r[-1, :] = 0.0
            r[:, 0] = r[:, -1] = 0.0
            return r.ravel()

        def filt(uf):
            """Separable tensor-product cut-off: U -> P U P, then re-clamp the edges."""
            g = uf.reshape(N, N)
            g[1:-1, 1:-1] = P @ g[1:-1, 1:-1] @ P
            g[0, :] = g[-1, :] = 0.0
            g[:, 0] = g[:, -1] = 0.0

        pts = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.complex128)
        u_prev = np.array([complex(eq.phi(p)) for p in pts], dtype=np.complex128)
        v0 = np.array([complex(eq.psi(p)) for p in pts], dtype=np.complex128)
        filt(u_prev)
        u = u_prev + dt * v0 + 0.5 * dt**2 * rhs(u_prev)
        filt(u)
        return self._march(eq, times, steps_of, u_prev, u, rhs, filt, pts,
                           {"dx": dx, "dy": dy, "shape": (N, N)})
