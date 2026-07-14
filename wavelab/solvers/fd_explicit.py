"""Explicit leapfrog FD for u_tt = c^2 Laplacian(u) + f(u), Dirichlet BC (spec §3.3).

Blow-up is data, not an exception: on the ill-posed c=i problem this solver
is EXPECTED to blow up in finite time; we record when and fill NaN after.
d=1: central 2nd difference.  d=2: five-point stencil.
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

    def _check(self, eq, times):
        if eq.bc != "dirichlet":
            raise NotImplementedError("ExplicitFD supports Dirichlet BC only")
        if eq.domain is None:
            raise ValueError("ExplicitFD requires eq.domain, e.g. ((0, 1),)")
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
        raise NotImplementedError(f"ExplicitFD supports dim=1 and dim=2, got {eq.dim}")

    def _march(self, eq, times, steps_of, u_prev, u, rhs, clamp, points, extra_meta):
        """Shared leapfrog loop: u^{n+1} = 2u^n - u^{n-1} + dt^2 rhs(u^n).

        `clamp(u)` re-imposes the Dirichlet zeros in place every step. This is NOT
        redundant: u_prev starts as phi on the grid, and phi(boundary) need not be 0,
        so 2u - u_prev would leak a nonzero boundary value for a general equation.
        (The sine data happens to have phi(0)=phi(1)=0, which would hide the bug.)
        """
        dt, N = self.dt, self.N
        P = u.shape[0]
        out = np.full((len(times), P), np.nan + 1j * np.nan, dtype=np.complex128)
        blowup_time = None
        snaps = {0: u_prev, 1: u}
        for n in range(2, int(steps_of.max()) + 1):
            u_next = 2 * u - u_prev + dt**2 * rhs(u)
            clamp(u_next)
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
        meta = {"blowup_time": blowup_time, "dt": dt, "N": N}
        meta.update(extra_meta)
        return Solution(eq=eq, solver=self.name, params={"N": N, "dt": dt},
                        times=times, points=points, u=out, meta=meta)

    def _solve_1d(self, eq, times, steps_of):
        dt, N = self.dt, self.N
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

        def clamp(u):
            u[0] = u[-1] = 0.0

        u_prev = np.array([complex(eq.phi(z)) for z in x], dtype=np.complex128)
        v0 = np.array([complex(eq.psi(z)) for z in x], dtype=np.complex128)
        u = u_prev + dt * v0 + 0.5 * dt**2 * rhs(u_prev)
        clamp(u)
        return self._march(eq, times, steps_of, u_prev, u, rhs, clamp,
                           x.astype(np.complex128), {"dx": dx})

    def _solve_2d(self, eq, times, steps_of):
        dt, N = self.dt, self.N
        (ax, bx), (ay, by) = eq.domain
        x = np.linspace(ax, bx, N)
        y = np.linspace(ay, by, N)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")
        c2 = complex(eq.c) ** 2
        f = eq.f_callable()

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

        def clamp(uf):
            g = uf.reshape(N, N)
            g[0, :] = g[-1, :] = 0.0
            g[:, 0] = g[:, -1] = 0.0

        pts = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.complex128)
        u_prev = np.array([complex(eq.phi(p)) for p in pts], dtype=np.complex128)
        v0 = np.array([complex(eq.psi(p)) for p in pts], dtype=np.complex128)
        u = u_prev + dt * v0 + 0.5 * dt**2 * rhs(u_prev)
        clamp(u)
        return self._march(eq, times, steps_of, u_prev, u, rhs, clamp, pts,
                           {"dx": dx, "dy": dy, "shape": (N, N)})
