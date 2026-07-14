"""Linearly Implicit Euler FD — the paper's Figure-7 scheme (spec §3.3).

The paper (§7.2) produces Figure 7 with Mathematica's
    NDSolveValue[..., Method -> "LinearlyImplicitEuler"]
and explains the result as:

    "the implicit scheme is more stable but exhibits a loss of accuracy
     compared to the explicit scheme DUE TO LOSS OF ENERGY CONSERVATION."

That sentence is the whole design brief. Stability here is *bought* with numerical
dissipation. Note what it rules out: a theta-scheme (see fd_implicit.py) is
energy-CONSERVING — its amplification roots satisfy g+ * g- = 1, so one of them
always sits outside the unit circle whenever they are real. It therefore cannot be
the stable scheme, and indeed it is not (it amplifies the ill-posed modes just like
the explicit scheme). Being implicit is not what buys stability; losing energy is.

Scheme. Write the wave equation as a first-order system and take an Euler step with
the LINEAR (stiff) part implicit and the nonlinear term explicit — "linearly
implicit", hence one linear solve per step and no Newton iteration:

    u_t = v,   v_t = c^2 u_xx + f(u)

    u^{n+1} = u^n + dt v^{n+1}
    v^{n+1} = v^n + dt ( c^2 L u^{n+1} + f(u^n) )       <- L implicit, f explicit
  =>  (I - dt^2 c^2 L) u^{n+1} = u^n + dt v^n + dt^2 f(u^n)
      v^{n+1} = (u^{n+1} - u^n) / dt

The system matrix is constant, so it is inverted once and reused every step.
Treating f explicitly also avoids the spurious far-away roots that a full Newton
solve on u^3 can jump to.

Damping regime: the high modes are damped only when dt^2 * mu_max > 2 (mu_max ~
4/dx^2 is the largest eigenvalue of -u_xx). With a small dt the scheme is
essentially explicit and buys you nothing — which is why the implicit runs want a
LARGER dt than the explicit ones. `stability_dt(eq, N)` reports that threshold.
"""
import warnings
import numpy as np

from wavelab.equation import WaveEquation
from wavelab.solution import Solution


def stability_dt(N: int, domain=((0.0, 1.0),)) -> float:
    """Smallest dt at which the highest grid mode is damped: dt^2 * mu_max > 2."""
    (a, b), = domain
    dx = (b - a) / (N - 1)
    mu_max = 4.0 / dx**2          # largest eigenvalue of -u_xx on the grid
    return float(np.sqrt(2.0 / mu_max))


class LinearlyImplicitFD:
    name = "linearly_implicit_fd"

    def __init__(self, N: int = 101, dt: float = 0.01):
        self.N, self.dt = N, dt

    def solve(self, eq: WaveEquation, times, points=None) -> Solution:
        if eq.dim != 1:
            raise NotImplementedError("LinearlyImplicitFD supports dim=1 only")
        if eq.bc != "dirichlet":
            raise NotImplementedError("LinearlyImplicitFD supports Dirichlet BC only")
        if eq.domain is None:
            raise ValueError("LinearlyImplicitFD requires eq.domain, e.g. ((0, 1),)")
        dt, N = self.dt, self.N
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

        L = np.zeros((N, N), dtype=np.complex128)
        idx = np.arange(1, N - 1)
        L[idx, idx - 1] = 1.0 / dx**2
        L[idx, idx] = -2.0 / dx**2
        L[idx, idx + 1] = 1.0 / dx**2

        # (I - dt^2 c^2 L), with Dirichlet rows pinned to identity. Constant -> invert once.
        M = np.eye(N, dtype=np.complex128) - dt**2 * c2 * L
        M[0, :] = 0.0
        M[0, 0] = 1.0
        M[-1, :] = 0.0
        M[-1, -1] = 1.0
        Minv = np.linalg.inv(M)

        u = np.array([complex(eq.phi(z)) for z in x], dtype=np.complex128)
        v = np.array([complex(eq.psi(z)) for z in x], dtype=np.complex128)
        u[0] = u[-1] = 0.0

        out = np.full((len(times), N), np.nan + 1j * np.nan, dtype=np.complex128)
        blowup_time = None
        snaps = {0: u.copy()}
        for n in range(1, int(steps_of.max()) + 1):
            fu = f(u)
            fu[0] = fu[-1] = 0.0
            rhs = u + dt * v + dt**2 * fu          # nonlinear term explicit
            rhs[0] = rhs[-1] = 0.0                 # Dirichlet
            u_next = Minv @ rhs
            u_next[0] = u_next[-1] = 0.0
            v = (u_next - u) / dt
            v[0] = v[-1] = 0.0
            u = u_next
            if not np.all(np.isfinite(u)):
                blowup_time = round(n * dt, 10)
                warnings.warn(f"{eq.name or 'equation'}: linearly-implicit FD blew up at "
                              f"t={blowup_time} (N={N}, dt={dt}); later snapshots are NaN")
                break
            snaps[n] = u.copy()
        for i, s in enumerate(steps_of):
            if s in snaps:
                out[i] = snaps[s]

        return Solution(eq=eq, solver=self.name,
                        params={"N": N, "dt": dt},
                        times=times, points=x.astype(np.complex128), u=out,
                        meta={"blowup_time": blowup_time, "dt": dt, "N": N, "dx": dx,
                              "damping_dt_threshold": stability_dt(N, eq.domain)})
