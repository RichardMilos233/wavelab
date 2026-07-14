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
        times = np.atleast_1d(np.asarray(times, dtype=float))
        if eq.dim >= 2 and eq.grad_phi is None:
            raise ValueError(
                f"BranchingMC needs eq.grad_phi for dim={eq.dim}: the d>=2 boundary "
                f"functional carries a y.grad(phi)(z+y) term (see spec §3.3)")
        if points is None:
            if eq.dim >= 2:
                raise ValueError(f"points is required for dim={eq.dim} "
                                 f"(shape (P, {eq.dim}) complex); no default grid")
            if eq.domain is None:
                raise ValueError("points is required when eq.domain is None")
            (a, b), = eq.domain
            points = np.linspace(a, b, 21)
        points = np.asarray(points, dtype=np.complex128)
        if eq.dim == 1:
            points = np.atleast_1d(points)
        elif points.ndim != 2 or points.shape[1] != eq.dim:
            raise ValueError(f"points must have shape (P, {eq.dim}) for dim={eq.dim}, "
                             f"got {points.shape}")
        powers, coeffs, probs = self._offspring(eq)

        if self.backend == "numba":
            if eq.dim > 1:
                raise NotImplementedError(
                    f"numba backend is dim=1 only; use backend='python' for dim={eq.dim}")
            try:
                from numba.core.dispatcher import Dispatcher
                from wavelab.solvers.mc import fast
            except ImportError as e:
                raise ImportError("numba backend needs numba: conda install numba") from e
            for fname, fn in (("phi", eq.phi), ("psi", eq.psi)):
                if not isinstance(fn, Dispatcher):
                    raise ValueError(
                        f"numba backend requires eq.{fname} to be numba.njit-compiled, "
                        f"e.g. {fname}=numba.njit(lambda z: cmath.sin(math.pi*z)); "
                        f"or use backend='python'")
            seed0 = self.seed if self.seed is not None else np.random.SeedSequence().entropy % (2**31)
            u = np.empty((len(times), len(points)), dtype=np.complex128)
            se = np.empty((len(times), len(points)), dtype=float)
            for i, t in enumerate(times):
                u[i], se[i] = fast.estimate_grid(
                    points, float(t), self.n, float(self.lam), complex(eq.c),
                    powers, coeffs, probs, int(seed0) + 100_003 * i, eq.phi, eq.psi)
        else:
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
