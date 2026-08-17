"""BranchingMC: pointwise estimator of u(z,t)=E[H] from the branching representation."""
import math
import warnings
import numpy as np

from wavelab.equation import WaveEquation
from wavelab.solution import Solution
from wavelab.solvers.mc import reference


class BranchingMC:
    name = "branching_mc"

    def __init__(self, lam=0.25, n=10_000, q=None, seed=None, N=21):
        """CAUTION: `n` and `N` differ only in case and mean different things.
        `n` = Monte Carlo samples per point (10_000). `N` = points PER AXIS of the
        default grid (21), exactly as in ExplicitFD — so the default grid has N^dim
        points. Mixing them up does not raise: it silently gives a very noisy answer
        (n too small) or a very slow one (N too large). `points=` overrides `N`.

        Unlike FD — where one march produces the whole field — MC cost is LINEAR in
        the number of points, so the default grid costs N^dim point-evaluations:
        21 in d=1, 441 in d=2, 9261 in d=3. Pick N accordingly; it is your dial, and
        `points=` is there when a full grid is not what you want.
        """
        if int(N) < 2:
            raise ValueError(f"N (number of default grid points) must be >= 2, got {N}")
        self.lam, self.n, self.q, self.seed, self.N = lam, n, q, seed, int(N)

    def _offspring(self, eq):
        powers = np.array(sorted(eq.f.keys()), dtype=np.int64)
        coeffs = np.array([eq.f[int(k)] for k in powers], dtype=np.complex128)
        if self.q is None:
            probs = np.full(len(powers), 1.0 / len(powers))
        else:
            if set(self.q.keys()) != set(int(k) for k in powers):
                raise ValueError(f"q keys {set(self.q)} must equal f powers {set(eq.f)}")
            probs = np.array([self.q[int(k)] for k in powers], dtype=float)
            if np.any(probs <= 0) or not math.isclose(probs.sum(), 1.0, abs_tol=1e-9):
                raise ValueError(f"q probabilities must be positive and sum to 1, got {probs}")
        return powers, coeffs, probs

    def solve(self, eq: WaveEquation, times, points=None) -> Solution:
        times = np.atleast_1d(np.asarray(times, dtype=float))
        if eq.dim >= 2 and eq.grad_phi is None:
            raise ValueError(
                f"BranchingMC needs eq.grad_phi for dim={eq.dim}: the d>=2 boundary "
                f"functional carries a y.grad(phi)(z+y) term (see spec §3.3)")
        default_shape = None
        if points is None:
            if eq.domain is None:
                raise ValueError("points is required when eq.domain is None")
            axes = [np.linspace(a, b, self.N) for a, b in eq.domain]
            if eq.dim == 1:
                points = axes[0]
            else:
                # same convention as ExplicitFD: indexing="ij", row-major ravel
                mesh = np.meshgrid(*axes, indexing="ij")
                points = np.stack([m.ravel() for m in mesh], axis=1)
                default_shape = (self.N,) * eq.dim
        points = np.asarray(points, dtype=np.complex128)
        if eq.dim == 1:
            points = np.atleast_1d(points)
        elif points.ndim != 2 or points.shape[1] != eq.dim:
            raise ValueError(f"points must have shape (P, {eq.dim}) for dim={eq.dim}, "
                             f"got {points.shape}")
        powers, coeffs, probs = self._offspring(eq)

        rng = np.random.default_rng(self.seed)
        u = np.empty((len(times), len(points)), dtype=np.complex128)
        se = np.empty((len(times), len(points)), dtype=float)
        for i, t in enumerate(times):
            for j, z in enumerate(points):
                u[i, j], se[i, j] = reference.estimate(
                    eq, z, t, self.n, self.lam, powers, coeffs, probs, rng)

        rel = se / np.maximum(np.abs(u), 1e-300)
        if np.any(rel > 0.2):
            warnings.warn(f"{int((rel > 0.2).sum())} point(s) have relative stderr "
                          f"> 20% — t may be near the integrability window's edge; "
                          f"increase n or reduce t")
        return Solution(eq=eq, solver=self.name,
                        params={"lam": self.lam, "n": self.n, "seed": self.seed,
                                "N": self.N},
                        times=times, points=points, u=u,
                        meta={"stderr": se, "n": self.n, "lam": self.lam,
                              "seed": self.seed,
                              **({"shape": default_shape} if default_shape else {})})
