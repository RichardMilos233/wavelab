"""BranchingMC: pointwise estimator of u(z,t)=E[H]; dispatches on backend."""
import math
import warnings
import numpy as np

from wavelab.equation import WaveEquation
from wavelab.solution import Solution
from wavelab.solvers.mc import reference

_BACKENDS = ("python", "numba")


class BranchingMC:
    name = "branching_mc"

    def __init__(self, lam=0.25, n=10_000, q=None, backend="python",
                 seed=None, workers=1):
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
        self.lam, self.n, self.q = lam, n, q
        self.backend, self.seed, self.workers = backend, seed, workers

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
        if eq.dim != 1:
            raise NotImplementedError("BranchingMC supports dim=1 only (M1/M2)")
        times = np.atleast_1d(np.asarray(times, dtype=float))
        if points is None:
            if eq.domain is None:
                raise ValueError("points is required when eq.domain is None")
            (a, b), = eq.domain
            points = np.linspace(a, b, 21)
        points = np.atleast_1d(np.asarray(points, dtype=np.complex128))
        powers, coeffs, probs = self._offspring(eq)

        if self.backend == "numba":
            raise NotImplementedError("numba backend arrives in Task 8")
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
                        params={"lam": self.lam, "n": self.n, "backend": self.backend,
                                "seed": self.seed},
                        times=times, points=points, u=u,
                        meta={"stderr": se, "n": self.n, "lam": self.lam,
                              "seed": self.seed, "backend": self.backend})
