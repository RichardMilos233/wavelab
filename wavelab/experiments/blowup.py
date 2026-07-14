"""Instruments for the ill-posedness story (spec §4).

blowup_scan       : blow-up time vs grid spacing — refine and it gets WORSE.
mode_amplification: per-step growth factor of each Fourier mode of the
                    linearized leapfrog scheme — the highest modes explode.
Both are 1-D (the amplification argument is a 1-D Fourier computation).
"""
import warnings
import numpy as np

from wavelab.equation import WaveEquation


def blowup_scan(eq: WaveEquation, make_solver, Ns, dts, probe_time=0.5):
    """Run `make_solver(N, dt)` for each (N, dt) and record when it blew up.

    make_solver: (N, dt) -> solver, e.g. lambda N, dt: ExplicitFD(N=N, dt=dt).
    Works for any solver exposing meta["blowup_time"] (explicit, theta, regularized).
    Returns rows: {"N", "dt", "blowup_time" (float|None), "max_abs_u" (float)}.
    """
    rows = []
    for N in Ns:
        for dt in dts:
            solver = make_solver(N, dt)
            # probe a ladder of times, not just the endpoint: if the solver dies
            # before probe_time, the endpoint alone is all-NaN and max|u| would be
            # meaningless. The ladder records how big it got before it died.
            n_steps = max(1, int(round(probe_time / dt)))
            ladder = np.unique(np.round(
                np.linspace(0, n_steps, 11)[1:]).astype(int)) * dt
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")     # blow-ups are the point here
                sol = solver.solve(eq, times=ladder)
            finite = np.abs(sol.u[np.isfinite(sol.u)])
            rows.append({
                "N": N, "dt": dt,
                "blowup_time": sol.meta.get("blowup_time"),
                "max_abs_u": float(finite.max()) if finite.size else float("nan"),
            })
    return rows


def blowup_table(rows) -> str:
    lines = [f"{'N':>5}  {'dt':>7}  {'blowup t':>9}  {'max|u|':>12}"]
    for r in rows:
        bt = "none" if r["blowup_time"] is None else f"{r['blowup_time']:.3f}"
        lines.append(f"{r['N']:>5}  {r['dt']:>7}  {bt:>9}  {r['max_abs_u']:>12.3f}")
    return "\n".join(lines)


def mode_amplification(eq: WaveEquation, N: int, dt: float):
    """Per-step growth factor of each Fourier mode of the linearized leapfrog.

    Linearize about 0:  u_tt = (c^2 * (-mu_k) + f'(0)) u =: omega_k u,
    with mu_k = (2 - 2 cos(pi k dx)) / dx^2 >= 0 the eigenvalue of -u_xx.
    Leapfrog: g^2 - (2 + a) g + 1 = 0 with a = dt^2 omega_k; growth = max|g±|.
    A stable mode has both roots on the unit circle -> growth == 1.

    Note the roots always satisfy g+ * g- = 1 (the scheme is energy-conserving), so
    a mode is either neutral (both on the unit circle) or has one growing root.
    That is exactly why no theta-scheme can stabilize the ill-posed problem.
    """
    if eq.dim != 1:
        raise NotImplementedError("mode_amplification is a 1-D Fourier analysis")
    if eq.domain is None:
        raise ValueError("mode_amplification requires eq.domain, e.g. ((0, 1),)")
    (a0, b0), = eq.domain
    dx = (b0 - a0) / (N - 1)
    k = np.arange(1, N - 1)
    mu = (2 - 2 * np.cos(np.pi * k * dx)) / dx**2        # eigenvalues of -u_xx  (>= 0)
    c2 = complex(eq.c) ** 2
    fprime0 = complex(eq.f.get(1, 0))                    # d/du of sum a_j u^j at u=0
    omega = c2 * (-mu) + fprime0
    a = dt**2 * omega
    disc = np.sqrt((2 + a) ** 2 - 4 + 0j)                # complex sqrt: valid for any c
    g_plus = (2 + a + disc) / 2
    g_minus = (2 + a - disc) / 2
    growth = np.maximum(np.abs(g_plus), np.abs(g_minus))
    return {"k": k, "omega": omega, "growth": growth}
