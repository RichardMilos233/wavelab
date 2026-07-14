import cmath, math
import numpy as np
import pytest
numba = pytest.importorskip("numba")
from wavelab import WaveEquation, BranchingMC

phi_nb = numba.njit(lambda z: cmath.sin(math.pi * z))
psi_nb = numba.njit(lambda z: -cmath.sin(math.pi * z))
SINE_NB = WaveEquation(dim=1, c=1j, f={1: -1, 3: 1}, phi=phi_nb, psi=psi_nb,
                       domain=((0.0, 1.0),), name="sine_ci_1d_njit")

def test_backends_agree_within_stderr():
    py = BranchingMC(n=20_000, seed=11, backend="python").solve(SINE_NB, [0.5], points=[0.5])
    nb = BranchingMC(n=20_000, seed=11, backend="numba").solve(SINE_NB, [0.5], points=[0.5])
    tol = 3 * (py.meta["stderr"][0, 0] + nb.meta["stderr"][0, 0])
    assert abs(py.u[0, 0] - nb.u[0, 0]) < tol
    assert nb.meta["backend"] == "numba"

def test_numba_seed_reproducible():
    a = BranchingMC(n=2_000, seed=3, backend="numba").solve(SINE_NB, [0.4], points=[0.3, 0.7]).u
    b = BranchingMC(n=2_000, seed=3, backend="numba").solve(SINE_NB, [0.4], points=[0.3, 0.7]).u
    np.testing.assert_array_equal(a, b)

def test_numba_rejects_plain_python_phi():
    plain = WaveEquation(dim=1, c=1j, f={1: -1, 3: 1},
                         phi=lambda z: cmath.sin(math.pi * z),
                         psi=lambda z: -cmath.sin(math.pi * z), domain=((0.0, 1.0),))
    with pytest.raises(ValueError, match="njit"):
        BranchingMC(backend="numba", n=10).solve(plain, [0.1], points=[0.5])

@pytest.mark.slow
def test_paper_scale_smoke():
    # 1e6 samples at one point — proves paper-scale runs are feasible
    sol = BranchingMC(n=1_000_000, seed=1, backend="numba").solve(SINE_NB, [0.5], points=[0.5])
    assert sol.u[0, 0].real == pytest.approx(1.91, abs=0.02)
