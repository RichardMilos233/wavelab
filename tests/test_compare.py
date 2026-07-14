import numpy as np
from wavelab import Solution, compare

def fake(solver, points, values_by_time):
    times = sorted(values_by_time)
    u = np.array([values_by_time[t] for t in times])
    return Solution(eq=None, solver=solver, params={}, times=times,
                    points=points, u=u, meta={})

A = fake("explicit_fd", np.linspace(0, 1, 11),
         {0.1: np.linspace(0, 1, 11), 0.2: 2 * np.linspace(0, 1, 11)})
B = fake("branching_mc", np.array([0.0, 0.52, 1.0]),
         {0.1: np.array([0, 0.5, 1]), 0.2: np.array([0, 1.0, 2]), 0.3: np.array([0, 9, 9])})

def test_rows_nearest_point_no_interpolation():
    rows = compare(A, B).rows()
    # probe = B's points (fewest); shared times only (0.1, 0.2)
    assert {r["t"] for r in rows} == {0.1, 0.2}
    r = next(r for r in rows if r["t"] == 0.1 and abs(r["x"] - 0.52) < 1e-9)
    assert r["explicit_fd"] == 0.5      # A's nearest grid point is x=0.5
    assert r["branching_mc"] == 0.5     # B's own value at 0.52

def test_table_mentions_solvers():
    s = compare(A, B).table()
    assert "explicit_fd" in s and "branching_mc" in s

def test_plot_writes_file(tmp_path):
    out = tmp_path / "cmp.png"
    fig = compare(A, B).plot(path=str(out))
    assert out.exists() and fig is not None
