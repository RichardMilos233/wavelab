"""surface() / surfaces() — the d=2 plotting path (paper Figures 2 and 8).

This is the only output route for `examples/fig8.py`, and de92603 built a contract
underneath it: BranchingMC's default grid now reports meta["shape"], so that
`surfaces([mc, fd])` works with no shapes= argument. test_mc_highdim.py asserts the
shape is WRITTEN; this file asserts it is READ, which is the half that the figure
actually depends on.
"""
import warnings
import numpy as np
import pytest
import matplotlib.pyplot as plt

from wavelab import WaveEquation, ExplicitFD, BranchingMC, library
from wavelab.experiments import surface, surfaces
from wavelab.experiments.plotting import _grid

EQ = library.SINE_DEFOCUS_CI_2D          # §7.3 — bounded garbage, never NaN
FOCUS = library.SINE_CI_2D               # §7.2 in 2-D — does go NaN


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _fd(N=21):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ExplicitFD(N=N, dt=0.002).solve(EQ, [0.2])


def _mc(N=6, n=200):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BranchingMC(lam=0.25, n=n, seed=0, N=N).solve(EQ, [0.2])


def test_grid_reads_meta_shape_from_fd():
    """FD carries meta["shape"]; _grid must use it rather than guessing."""
    sol = _fd(N=21)
    X, Y, Z = _grid(sol, 0, None)
    assert X.shape == Y.shape == Z.shape == (21, 21)
    # ij ordering: x varies along axis 0, y along axis 1
    assert np.allclose(X[:, 0], np.linspace(0, 1, 21))
    assert np.allclose(Y[0, :], np.linspace(0, 1, 21))


def test_grid_reads_meta_shape_from_mc_default_grid():
    """The de92603 contract: MC's default grid reports its own shape, so callers
    never pass shapes= for it."""
    sol = _mc(N=6)
    assert sol.meta["shape"] == (6, 6)          # written (also covered elsewhere)
    _, _, Z = _grid(sol, 0, None)               # read
    assert Z.shape == (6, 6)


def test_surfaces_needs_no_shapes_argument_for_mc_plus_fd():
    """`surfaces([mc, fd])` — the call in examples/fig8.py — must just work."""
    fig = surfaces([_mc(N=6), _fd(N=21)], labels=["Monte Carlo", "Explicit FD"])
    assert len(fig.axes) == 2


def test_explicit_points_have_no_shape_but_square_counts_are_inferred():
    """points= overrides N and records no shape; _grid falls back to sqrt(P)."""
    pts = np.array([[x, y] for x in (0.25, 0.5, 0.75) for y in (0.25, 0.5, 0.75)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = BranchingMC(lam=0.25, n=200, seed=0).solve(EQ, [0.2], points=pts)
    assert "shape" not in sol.meta
    _, _, Z = _grid(sol, 0, None)
    assert Z.shape == (3, 3)


def test_non_square_point_count_asks_for_an_explicit_shape():
    pts = np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [0.4, 0.6], [0.6, 0.4]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = BranchingMC(lam=0.25, n=200, seed=0).solve(EQ, [0.2], points=pts)
    with pytest.raises(ValueError, match="cannot infer the grid shape"):
        _grid(sol, 0, None)


def test_explicit_shape_overrides_meta():
    sol = _fd(N=21)
    _, _, Z = _grid(sol, 0, (441, 1))
    assert Z.shape == (441, 1)


def test_d1_solution_is_rejected_with_a_pointer_to_compare():
    flat = WaveEquation(dim=1, c=1, f={}, phi=lambda z: 0j, psi=lambda z: 0j,
                        domain=((0.0, 1.0),), name="flat_1d")
    sol = ExplicitFD(N=21, dt=0.002).solve(flat, [0.1])
    with pytest.raises(ValueError, match="d=2 solution"):
        surface(sol)


def test_nan_panels_still_plot():
    """fig8's explicit-FD panel is garbage, and the focusing 2-D case is all-NaN.
    surface() must render it rather than raise — the failure IS the picture."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dead = ExplicitFD(N=41, dt=0.002).solve(FOCUS, [0.5])   # N=21/31 survive
    assert np.all(np.isnan(dead.u[0].real))
    fig = surfaces([dead], labels=["Explicit FD (dead)"])
    assert len(fig.axes) == 1


def test_surfaces_writes_the_file(tmp_path):
    out = tmp_path / "fig.png"
    surfaces([_mc(N=6), _fd(N=21)], labels=["MC", "FD"], path=str(out),
             suptitle="t = 0.2")
    assert out.exists() and out.stat().st_size > 0
