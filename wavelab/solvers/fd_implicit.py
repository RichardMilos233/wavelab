"""Implicit theta-scheme FD for u_tt = c^2 u_xx + f(u), Dirichlet BC (spec §3.3).

Counterpart to the explicit scheme (paper Fig 7): far more stable on the
ill-posed c=i problem — it survives past the time the explicit scheme dies —
at the cost of accuracy (the implicit damping smooths the very modes that the
true, ill-posed dynamics amplify).

Step: solve G(w) = 0 for w = u^{n+1} by Newton, with
  R(u)  = c^2 L u + f(u)
  G(w)  = w - 2u^n + u^{n-1} - dt^2 [ th R(w) + (1-2 th) R(u^n) + th R(u^{n-1}) ]
  G'(w) = I - dt^2 th ( c^2 L + diag(f'(w)) )
theta=0 recovers the explicit leapfrog exactly; theta=0.5 is the stable default.
N <= 201 here, so a dense solve is plenty (and keeps scipy out of the deps).
"""
import warnings
import numpy as np

from wavelab.equation import WaveEquation
from wavelab.solution import Solution


class ImplicitFD:
    name = "implicit_fd"

    def __init__(self, N: int = 101, dt: float = 0.002, theta: float = 0.5,
                 newton_tol: float = 1e-10, newton_maxiter: int = 50):
        self.N, self.dt, self.theta = N, dt, theta
        self.newton_tol, self.newton_maxiter = newton_tol, newton_maxiter

    def solve(self, eq: WaveEquation, times, points=None) -> Solution:
        if eq.dim != 1:
            raise NotImplementedError("ImplicitFD supports dim=1 only")
        if eq.bc != "dirichlet":
            raise NotImplementedError("ImplicitFD supports Dirichlet BC only")
        if eq.domain is None:
            raise ValueError("ImplicitFD requires eq.domain, e.g. ((0, 1),)")
        if not (0.0 <= self.theta <= 0.5):
            raise ValueError(f"theta must lie in [0, 0.5], got {self.theta}")
        dt, N, th = self.dt, self.N, self.theta
        times = np.atleast_1d(np.asarray(times, dtype=float))
        steps_of = np.round(times / dt).astype(int)
        if np.any(np.abs(steps_of * dt - times) > 1e-9):
            raise ValueError(f"each requested time must be a multiple of dt={dt} "
                             f"(so snapshots are exact); got {times}")

        (a, b), = eq.domain
        x = np.linspace(a, b, N)
        dx = x[1] - x[0]
        c2 = complex(eq.c) ** 2
        f, fp = eq.f_callable(), eq.f_prime_callable()

        # second-difference matrix with zero rows at the Dirichlet ends
        L = np.zeros((N, N), dtype=np.complex128)
        idx = np.arange(1, N - 1)
        L[idx, idx - 1] = 1.0 / dx**2
        L[idx, idx] = -2.0 / dx**2
        L[idx, idx + 1] = 1.0 / dx**2

        def R(u):
            r = c2 * (L @ u) + f(u)
            r[0] = r[-1] = 0.0
            return r

        I = np.eye(N, dtype=np.complex128)

        u_prev = np.array([complex(eq.phi(z)) for z in x], dtype=np.complex128)
        v0 = np.array([complex(eq.psi(z)) for z in x], dtype=np.complex128)
        u = u_prev + dt * v0 + 0.5 * dt**2 * R(u_prev)     # first step (Taylor)
        u[0] = u[-1] = 0.0

        out = np.full((len(times), N), np.nan + 1j * np.nan, dtype=np.complex128)
        blowup_time = None
        newton_iters = []
        snaps = {0: u_prev, 1: u}
        for n in range(2, int(steps_of.max()) + 1):
            # everything in G that does not depend on w
            const = (-2 * u + u_prev
                     - dt**2 * ((1 - 2 * th) * R(u) + th * R(u_prev)))
            w = u.copy()                                   # Newton initial guess
            it = 0
            for it in range(1, self.newton_maxiter + 1):
                G = w + const - dt**2 * th * R(w)
                G[0] = w[0]
                G[-1] = w[-1]
                if np.max(np.abs(G)) < self.newton_tol:
                    break
                J = I - dt**2 * th * (c2 * L + np.diag(fp(w)))
                J[0, :] = 0.0
                J[0, 0] = 1.0
                J[-1, :] = 0.0
                J[-1, -1] = 1.0
                try:
                    w = w - np.linalg.solve(J, G)
                except np.linalg.LinAlgError:
                    w = np.full(N, np.nan + 1j * np.nan, dtype=np.complex128)
                    break
            newton_iters.append(it)
            w[0] = w[-1] = 0.0
            u_prev, u = u, w
            if not np.all(np.isfinite(u)):
                blowup_time = round(n * dt, 10)
                warnings.warn(f"{eq.name or 'equation'}: implicit FD blew up at "
                              f"t={blowup_time} (N={N}, dt={dt}, theta={th}); "
                              f"later snapshots are NaN")
                break
            snaps[n] = u.copy()
        for i, s in enumerate(steps_of):
            if s in snaps:
                out[i] = snaps[s]

        return Solution(eq=eq, solver=self.name,
                        params={"N": N, "dt": dt, "theta": th},
                        times=times, points=x.astype(np.complex128), u=out,
                        meta={"blowup_time": blowup_time, "dt": dt, "N": N,
                              "dx": dx, "theta": th, "newton_iters": newton_iters})
