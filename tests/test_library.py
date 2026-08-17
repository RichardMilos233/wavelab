"""The library doubles as the paper casebook and the test fixture set."""
import warnings
import numpy as np
import pytest
from wavelab import library, BranchingMC, ExplicitFD

def test_all_equations_present():
    assert set(library.ALL) == {
        "SIM01_QUADRATIC_1D", "SIM02_CUBIC_1D", "SIM03_MIXED_1D",
        "SIM05_QUADRATIC_2D", "SIM08_QUADRATIC_3D",
        "SOLITON_1D", "SOLITON_2D", "SOLITON_3D",
        "SINE_CI_1D", "SINE_CI_2D", "SINE_C1_2D",
        "SINE_DEFOCUS_C1_2D", "SINE_DEFOCUS_CI_2D"}

def test_with_exact_subset():
    assert set(library.WITH_EXACT) == {
        "SIM01_QUADRATIC_1D", "SIM02_CUBIC_1D", "SIM03_MIXED_1D",
        "SIM05_QUADRATIC_2D", "SIM08_QUADRATIC_3D",
        "SOLITON_1D", "SOLITON_2D", "SOLITON_3D"}
    assert all(eq.exact is not None for eq in library.WITH_EXACT.values())

def test_exact_solutions_satisfy_initial_data():
    """exact(z, 0) must equal phi(z) — catches transcription typos in either one."""
    for name, eq in library.WITH_EXACT.items():
        if name.startswith("SOLITON"):
            z = -1.0 + 0j if eq.dim == 1 else np.full(eq.dim, -1.0 + 0j)
        else:
            z = 3.0 + 0j if eq.dim == 1 else np.full(eq.dim, 4.0 + 0j)
        got, want = complex(eq.exact(z, 0.0)), complex(eq.phi(z))
        assert abs(got - want) < 1e-9, f"{name}: exact(z,0)={got} != phi(z)={want}"

# MC must reproduce every closed form in the library (3 sigma).
_MC_CASES = [
    ("SIM01_QUADRATIC_1D", 3.0 + 0j, 0.25, 0.5),
    ("SIM02_CUBIC_1D", 6.0 + 0j, 0.25, 0.5),
    ("SIM03_MIXED_1D", 9.0 + 0j, 1.0, 0.5),
    ("SOLITON_1D", -1.0 + 0j, 0.25, 0.5),
]

@pytest.mark.parametrize("name,z,lam,t", _MC_CASES)
def test_mc_matches_library_closed_forms_1d(name, z, lam, t):
    eq = library.ALL[name]
    sol = BranchingMC(lam=lam, n=20_000, seed=17).solve(eq, [t], points=[z])
    err = abs(sol.u[0, 0] - eq.exact(z, t))
    assert err < 3 * sol.meta["stderr"][0, 0]

@pytest.mark.slow
@pytest.mark.parametrize("name,zv,lam", [
    ("SIM05_QUADRATIC_2D", [4 + 0j, 4 + 0j], 1.0),
    ("SOLITON_2D", [-1 + 0j, -1 + 0j], 0.25),
    ("SIM08_QUADRATIC_3D", [4 + 0j, 4 + 0j, 4 + 0j], 1.0),
    ("SOLITON_3D", [-1 + 0j, -1 + 0j, -1 + 0j], 0.25),
])
def test_mc_matches_library_closed_forms_highdim(name, zv, lam):
    eq = library.ALL[name]
    z = np.array([zv], dtype=np.complex128)
    sol = BranchingMC(lam=lam, n=50_000, seed=23).solve(eq, [0.5], points=z)
    err = abs(sol.u[0, 0] - eq.exact(np.array(zv), 0.5))
    assert err < 3 * sol.meta["stderr"][0, 0]

def test_sine_ci_1d_is_the_fig6_equation():
    eq = library.SINE_CI_1D
    assert eq.dim == 1 and eq.c == 1j and eq.f == {1: -1, 3: 1}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = ExplicitFD(N=101, dt=0.002).solve(eq, times=[0.5])
    assert sol.meta["blowup_time"] == pytest.approx(0.232, abs=0.01)
