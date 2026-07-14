"""MC variance profiling: where does the estimator's own short-time window end?

Branching MC is not magic either — it just fails in a completely different way from
a marching scheme. The branching representation is integrable only for short t. As t
grows the tree branches more and the weights (tau/lam)(a_k/q_k) multiply up, so the
VARIANCE explodes long before any bias appears. The estimator gets noisy, not wrong.

That is the honest counterweight to the Figure-6 story: FD dies of ill-posedness
(a deterministic, grid-coupled instability), MC dies of variance (a statistical,
pointwise one). This function plots MC's wall.
"""
import warnings
import numpy as np

from wavelab.experiments import plotting


def variance_profile(eq, mc, times, point):
    """Run `mc` at a single `point` for each t; report stderr and relative stderr."""
    rows = []
    for t in times:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # the high-stderr warning IS the signal
            sol = mc.solve(eq, [t], points=[point])
        u = sol.u[0, 0]
        se = float(sol.meta["stderr"][0, 0])
        rows.append({"t": float(t), "u": u, "stderr": se,
                     "rel_stderr": se / max(abs(u), 1e-300)})
    return rows


def variance_plot(rows, path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ts = [r["t"] for r in rows]
    ax.semilogy(ts, [r["stderr"] for r in rows], "-o", ms=4, label="stderr")
    ax.semilogy(ts, [r["rel_stderr"] for r in rows], "-s", ms=4, label="relative stderr")
    ax.set_xlabel("t")
    ax.set_ylabel("MC error (log scale)")
    ax.set_title("Branching MC variance vs t — the short-time window")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    if path:
        plotting.save(fig, path)
    return fig
