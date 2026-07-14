"""Explicit leapfrog FD for u_tt = c^2 u_xx + f(u), Dirichlet BC (spec §3.3).

Blow-up is data, not an exception: on the ill-posed c=i problem this solver
is EXPECTED to blow up in finite time; we record when and fill NaN after.
"""
import warnings
import numpy as np

from wavelab.equation import WaveEquation
from wavelab.solution import Solution


class ExplicitFD:
    name = "explicit_fd"

    def __init__(self, N: int = 101, dt: float = 0.002):
        self.N = N
        self.dt = dt

    def solve(self, eq: WaveEquation, times, points=None) -> Solution:
        if eq.dim != 1:
            raise NotImplementedError("ExplicitFD supports dim=1 only (M1)")
        if eq.bc != "dirichlet":
            raise NotImplementedError("ExplicitFD supports Dirichlet BC only")
        if eq.domain is None:
            raise ValueError("ExplicitFD requires eq.domain, e.g. ((0, 1),)")
        dt, N = self.dt, self.N
        times = np.atleast_1d(np.asarray(times, dtype=float))
        # map each requested time to a step index; must sit ON the time grid
        steps_of = np.round(times / dt).astype(int)
        if np.any(np.abs(steps_of * dt - times) > 1e-9):
            raise ValueError(f"each requested time must be a multiple of dt={dt} "
                             f"(so snapshots are exact); got {times}")

        (a, b), = eq.domain
        x = np.linspace(a, b, N)
        dx = x[1] - x[0]
        c2 = complex(eq.c) ** 2
        f = eq.f_callable()

        def rhs(u):
            d2 = np.zeros_like(u)
            d2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
            r = c2 * d2 + f(u)
            r[0] = r[-1] = 0.0
            return r

        u_prev = np.array([complex(eq.phi(z)) for z in x], dtype=np.complex128)
        v0 = np.array([complex(eq.psi(z)) for z in x], dtype=np.complex128)
        u = u_prev + dt * v0 + 0.5 * dt**2 * rhs(u_prev)   # first step (Taylor)
        u[0] = u[-1] = 0.0

        out = np.full((len(times), N), np.nan + 1j * np.nan, dtype=np.complex128)
        blowup_time = None
        snaps = {0: u_prev, 1: u}                     # step index -> state
        for n in range(2, int(steps_of.max()) + 1):   # u at step n = time n*dt
            u_next = 2 * u - u_prev + dt**2 * rhs(u)
            u_next[0] = u_next[-1] = 0.0
            u_prev, u = u, u_next
            if not np.all(np.isfinite(u)):
                blowup_time = round(n * dt, 10)
                warnings.warn(f"{eq.name or 'equation'}: explicit FD blew up at "
                              f"t={blowup_time} (N={N}, dt={dt}); later snapshots are NaN")
                break
            snaps[n] = u.copy()
        for i, s in enumerate(steps_of):
            if s in snaps:
                out[i] = snaps[s]

        return Solution(eq=eq, solver=self.name,
                        params={"N": N, "dt": dt},
                        times=times, points=x.astype(np.complex128), u=out,
                        meta={"blowup_time": blowup_time, "dt": dt, "N": N, "dx": dx})
