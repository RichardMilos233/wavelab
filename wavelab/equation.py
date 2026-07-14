"""WaveEquation: one immutable definition serving all solvers (spec §3.1)."""
from dataclasses import dataclass
from typing import Callable, Mapping, Optional
import numpy as np


@dataclass(frozen=True)
class WaveEquation:
    dim: int
    c: complex
    f: Mapping[int, complex]          # {k: a_k}, f(u) = sum a_k u^k
    phi: Callable                     # u(z, 0)
    psi: Callable                     # du/dt(z, 0)
    domain: Optional[tuple] = None    # per-axis intervals, e.g. ((0, 1),)
    bc: str = "dirichlet"
    exact: Optional[Callable] = None  # u(z, t) if known
    name: str = ""

    def __post_init__(self):
        if self.dim not in (1, 2, 3):
            raise ValueError(f"dim must be 1, 2 or 3, got {self.dim}")
        for k in self.f:
            if not isinstance(k, int) or isinstance(k, bool) or k < 0:
                raise ValueError(f"power {k!r} in f: powers must be non-negative ints")
        if self.domain is not None and len(self.domain) != self.dim:
            raise ValueError(f"domain has {len(self.domain)} axes but dim={self.dim}")
        z = 0.5 + 0j if self.dim == 1 else np.full(self.dim, 0.5 + 0j)
        for fname, fn in (("phi", self.phi), ("psi", self.psi)):
            try:
                complex(fn(z))
            except Exception as e:
                raise ValueError(f"{fname} must accept a complex argument "
                                 f"(dim={self.dim}): {e}") from e

    def f_callable(self) -> Callable:
        items = tuple(self.f.items())

        def f(u):
            u = np.asarray(u, dtype=np.complex128)
            out = np.zeros_like(u)
            for k, a in items:
                out = out + a * u**k
            return out

        return f
