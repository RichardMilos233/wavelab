import numpy as np
import pytest
from wavelab import library, BranchingMC
from wavelab.experiments import variance_profile, variance_plot

def test_variance_grows_with_time():
    rows = variance_profile(library.SINE_CI_1D,
                            BranchingMC(lam=0.25, n=4_000, seed=1),
                            times=[0.1, 0.3, 0.5], point=0.5)
    assert [r["t"] for r in rows] == [0.1, 0.3, 0.5]
    # MC's own limit: the estimator gets noisier as t grows
    assert rows[0]["stderr"] < rows[-1]["stderr"]
    assert all(r["rel_stderr"] >= 0 for r in rows)

def test_variance_plot_writes_file(tmp_path):
    rows = variance_profile(library.SINE_CI_1D,
                            BranchingMC(lam=0.25, n=1_000, seed=2),
                            times=[0.1, 0.3], point=0.5)
    out = tmp_path / "var.png"
    fig = variance_plot(rows, path=str(out))
    assert out.exists() and fig is not None
