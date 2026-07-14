import numpy as np
import pytest
from wavelab import Solution

def test_construction_and_coercion():
    s = Solution(eq=None, solver="test", params={"n": 1},
                 times=[0.1, 0.2], points=[0.0, 0.5, 1.0],
                 u=[[1, 2, 3], [4, 5, 6]], meta={})
    assert s.u.shape == (2, 3) and s.u.dtype == np.complex128
    assert s.times.shape == (2,) and s.points.shape == (3,)

def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        Solution(eq=None, solver="test", params={}, times=[0.1],
                 points=[0.0, 1.0], u=[[1, 2], [3, 4]], meta={})
