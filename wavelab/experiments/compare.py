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
        """Only .plot() is 1-D only — .table() probes by nearest point in any dim."""
        for s in self.solutions:
            if s.points.ndim > 1:
                raise NotImplementedError(
                    f"compare(...).plot() is 1-D only; solution '{s.solver}' has "
                    f"dim={s.points.shape[1]} points. For d=2 use "
                    f"experiments.surfaces([...]) to draw, or .table() to compare "
                    f"numbers.")

    @staticmethod
    def _nearest(sol, x):
        """Index of the point of `sol` closest to x (scalar for d=1, vector for d>1)."""
        if sol.points.ndim == 1:
            return int(np.argmin(np.abs(sol.points - x)))
        return int(np.argmin(np.linalg.norm(sol.points - np.asarray(x), axis=1)))

    def rows(self, probe_points=None):
        """One row per (shared time, probe point). Works in any dimension: each
        solution is read at ITS OWN nearest point, never interpolated."""
        if probe_points is None:
            probe_points = min((s.points for s in self.solutions), key=len)
        probe_points = np.asarray(probe_points)
        if probe_points.ndim == 1:
            probe_points = np.atleast_1d(probe_points)
        out = []
        for t in self._shared_times():
            for x in probe_points:
                row = {"t": float(t), "x": x}
                for s in self.solutions:
                    ti = int(np.argmin(np.abs(s.times - t)))
                    row[s.solver] = s.u[ti, self._nearest(s, x)]
                out.append(row)
        return out

    def table(self, probe_points=None):
        rows = self.rows(probe_points)
        names = [s.solver for s in self.solutions]
        multi = bool(rows) and np.ndim(rows[0]["x"]) > 0
        lines = [f"t      {'point' if multi else 'x':<16} "
                 + "  ".join(f"{n:>14}" for n in names)]
        for r in rows:
            xs = ("(" + ", ".join(f"{v:.2f}" for v in np.real(r["x"])) + ")" if multi
                  else f"{float(np.real(r['x'])):.4f}")
            vals = "  ".join(f"{r[n].real:+14.4f}" for n in names)
            lines.append(f"{r['t']:<5}  {xs:<16} {vals}")
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
