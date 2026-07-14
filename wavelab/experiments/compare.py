"""Compare Solutions from different solvers: tables and side-by-side plots."""
from dataclasses import dataclass
import numpy as np

from wavelab.solution import Solution
from wavelab.experiments import plotting


@dataclass
class Comparison:
    solutions: tuple

    def _shared_times(self):
        shared = set(np.round(self.solutions[0].times, 9))
        for s in self.solutions[1:]:
            shared &= set(np.round(s.times, 9))
        return sorted(shared)

    def _check_1d(self):
        for s in self.solutions:
            if s.points.ndim > 1:
                raise NotImplementedError(
                    f"compare() is 1-D only; solution '{s.solver}' has "
                    f"dim={s.points.shape[1]} points. Compare d>=2 runs numerically "
                    f"(e.g. against eq.exact) instead of via compare().")

    def rows(self, probe_points=None):
        self._check_1d()
        if probe_points is None:
            probe_points = min((s.points for s in self.solutions), key=len)
        out = []
        for t in self._shared_times():
            for x in np.atleast_1d(probe_points):
                row = {"t": float(t), "x": complex(x)}
                for s in self.solutions:
                    ti = int(np.argmin(np.abs(s.times - t)))
                    pi = int(np.argmin(np.abs(s.points - x)))   # nearest own point
                    row[s.solver] = s.u[ti, pi]
                out.append(row)
        return out

    def table(self, probe_points=None):
        rows = self.rows(probe_points)
        names = [s.solver for s in self.solutions]
        lines = ["t      x          " + "  ".join(f"{n:>14}" for n in names)]
        for r in rows:
            vals = "  ".join(f"{r[n].real:+14.4f}" for n in names)
            lines.append(f"{r['t']:<5}  {r['x'].real:<9.4f}  {vals}")
        return "\n".join(lines)

    def plot(self, path=None):
        import matplotlib.pyplot as plt
        self._check_1d()
        n = len(self.solutions)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.4), squeeze=False)
        for ax, s in zip(axes[0], self.solutions):
            for i, t in enumerate(s.times):
                if np.all(np.isnan(s.u[i].real)):
                    continue
                ax.plot(s.points.real, s.u[i].real, "-o", ms=3, lw=1.6,
                        color=f"C{i}", label=f"t={t:g}")
            title = s.solver
            bt = s.meta.get("blowup_time")
            if bt is not None:
                title += f"  (blew up at t≈{bt:g})"
            ax.set_title(title)
            ax.set_xlabel("Re z")
            plotting.apply_style(ax)
            ax.legend(fontsize=9)
        axes[0][0].set_ylabel("Re u(z,t)")
        if path:
            plotting.save(fig, path)
        return fig


def compare(*solutions: Solution) -> Comparison:
    if not solutions:
        raise ValueError("compare() needs at least one Solution")
    return Comparison(tuple(solutions))
