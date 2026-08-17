"""House style: every wavelab figure looks like family (spec §4).

Also holds the d=2 surface plotting used to reproduce the paper's Figures 2 and 8,
which compare a Monte Carlo surface against a finite-difference one on [0,1]^2.
"""
import numpy as np
import matplotlib.pyplot as plt

DPI = 140


def apply_style(ax):
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")


def _grid(sol, i, shape):
    """(X, Y, Z) arrays for a d=2 Solution at time index i.

    FD solutions carry meta["shape"]; MC solutions are evaluated at whatever points
    you asked for, so pass `shape=(M, M)` when those points form an MxM meshgrid
    (indexing="ij", the convention used everywhere in wavelab).
    """
    if sol.points.ndim != 2 or sol.points.shape[1] != 2:
        raise ValueError(
            f"surface() needs a d=2 solution; '{sol.solver}' has points of shape "
            f"{sol.points.shape}. Use compare(...).plot() for d=1.")
    shape = shape or sol.meta.get("shape")
    if shape is None:
        P = len(sol.points)
        m = int(round(np.sqrt(P)))
        if m * m != P:
            raise ValueError(
                f"cannot infer the grid shape for '{sol.solver}' ({P} points): pass "
                f"shape=(rows, cols) explicitly")
        shape = (m, m)
    X = sol.points[:, 0].real.reshape(shape)
    Y = sol.points[:, 1].real.reshape(shape)
    Z = sol.u[i].real.reshape(shape)
    return X, Y, Z


def surface(sol, i=0, shape=None, ax=None, title=None, zlim=None, cmap="plasma"):
    """3-D surface of Re u(x, y, t) for one time index of a d=2 Solution."""
    X, Y, Z = _grid(sol, i, shape)
    if ax is None:
        fig = plt.figure(figsize=(5.5, 4.2))
        ax = fig.add_subplot(111, projection="3d")
    finite = np.isfinite(Z)
    ax.plot_surface(X, Y, np.where(finite, Z, np.nan), cmap=cmap,
                    linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if zlim:
        ax.set_zlim(*zlim)
    rng = (f"[{np.nanmin(Z[finite]):.3g}, {np.nanmax(Z[finite]):.3g}]"
           if finite.any() else "all NaN")
    ax.set_title(title or f"{sol.solver}  t={sol.times[i]:g}\nrange {rng}", fontsize=10)
    return ax


def surfaces(sols, i=0, shapes=None, labels=None, path=None, zlim=None,
             suptitle=None):
    """Side-by-side d=2 surfaces — the paper's Figure 2 / Figure 8 layout.

    `shapes` is a per-solution grid shape (None to take meta["shape"] or infer).
    Pass `zlim` to force a shared vertical scale; leave it off to let each panel
    autoscale, which is usually what you want when one of them is garbage.
    """
    sols = list(sols)
    shapes = shapes or [None] * len(sols)
    labels = labels or [None] * len(sols)
    fig = plt.figure(figsize=(5.5 * len(sols), 4.2))
    for k, (s, sh, lab) in enumerate(zip(sols, shapes, labels), start=1):
        ax = fig.add_subplot(1, len(sols), k, projection="3d")
        surface(s, i=i, shape=sh, ax=ax, title=lab, zlim=zlim)
    if suptitle:
        fig.suptitle(suptitle)
    if path:
        save(fig, path)
    return fig
