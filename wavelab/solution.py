"""Solution: the common currency all solvers return (spec §3.2)."""
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class Solution:
    eq: Any
    solver: str
    params: dict
    times: np.ndarray
    points: np.ndarray
    u: np.ndarray                  # (T, P) complex128; NaN at/after FD blow-up
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.times = np.atleast_1d(np.asarray(self.times, dtype=float))
        self.points = np.atleast_1d(np.asarray(self.points))
        self.u = np.asarray(self.u, dtype=np.complex128)
        want = (len(self.times), len(self.points))
        if self.u.shape != want:
            raise ValueError(f"u has shape {self.u.shape}, expected {want} "
                             f"(times x points)")
