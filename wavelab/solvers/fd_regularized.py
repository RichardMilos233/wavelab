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

    Interior nodes j = 1..N-2, modes k = 1..N-2, basis sin(pi k j / (N-1)).
    The DST-I basis is orthogonal with  sum_j sin(pi k j/(N-1)) sin(pi m j/(N-1))
    = (N-1)/2 * delta_km, so the projector is (2/(N-1)) * S_K @ S_K.T.
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

    def solve(self, eq: WaveEquation, times, points=None) -> Solution:
        if eq.dim != 1:
            raise NotImplementedError("RegularizedFD supports dim=1 only")
        if eq.bc != "dirichlet":
            raise NotImplementedError("RegularizedFD supports Dirichlet BC only")
        if eq.domain is None:
            raise ValueError("RegularizedFD requires eq.domain, e.g. ((0, 1),)")
        dt, N, K = self.dt, self.N, self.k_max
        times = np.atleast_1d(np.asarray(times, dtype=float))
        steps_of = np.round(times / dt).astype(int)
        if np.any(np.abs(steps_of * dt - times) > 1e-9):
            raise ValueError(f"each requested time must be a multiple of dt={dt} "
                             f"(so snapshots are exact); got {times}")

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

        out = np.full((len(times), N), np.nan + 1j * np.nan, dtype=np.complex128)
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

        return Solution(eq=eq, solver=self.name,
                        params={"N": N, "dt": dt, "k_max": K},
                        times=times, points=x.astype(np.complex128), u=out,
                        meta={"blowup_time": blowup_time, "dt": dt, "N": N,
                              "dx": dx, "k_max": K})
