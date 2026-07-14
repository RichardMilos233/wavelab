"""The ill-posedness instruments (spec §4)."""
import cmath, math
import numpy as np
import pytest
from wavelab import library, WaveEquation, ExplicitFD, RegularizedFD
from wavelab.experiments import blowup_scan, blowup_table, mode_amplification

SINE_CI = library.SINE_CI_1D

# well-posed 1-D control: c=1, f=0 -> no growing modes
WELLPOSED = WaveEquation(dim=1, c=1, f={},
                         phi=lambda z: cmath.sin(math.pi * z),
                         psi=lambda z: 0j,
                         domain=((0.0, 1.0),), name="wellposed")


def test_scan_reproduces_the_illposedness_fingerprint():
    rows = blowup_scan(SINE_CI, lambda N, dt: ExplicitFD(N=N, dt=dt),
                       Ns=(51, 101, 201), dts=(0.002,), probe_time=0.5)
    bt = {r["N"]: r["blowup_time"] for r in rows}
    assert bt[51] == pytest.approx(0.44, abs=0.02)
    assert bt[101] == pytest.approx(0.232, abs=0.02)
    assert bt[201] == pytest.approx(0.128, abs=0.02)
    assert bt[51] > bt[101] > bt[201]          # finer = worse = ill-posed


def test_scan_works_for_the_regularized_solver_too():
    """Same instrument, different knob: with the spectral cut-off the solver
    survives, and keeping more modes brings the blow-up back."""
    keep12 = blowup_scan(SINE_CI, lambda N, dt: RegularizedFD(N=N, dt=dt, k_max=12),
                         Ns=(101,), dts=(0.002,), probe_time=0.7)
    keep30 = blowup_scan(SINE_CI, lambda N, dt: RegularizedFD(N=N, dt=dt, k_max=30),
                         Ns=(101,), dts=(0.002,), probe_time=0.7)
    assert keep12[0]["blowup_time"] is None                       # survives
    assert keep30[0]["blowup_time"] == pytest.approx(0.438, abs=0.03)


def test_blowup_table_is_printable():
    rows = blowup_scan(SINE_CI, lambda N, dt: ExplicitFD(N=N, dt=dt),
                       Ns=(51,), dts=(0.002,), probe_time=0.3)
    s = blowup_table(rows)
    assert "N" in s and "blowup" in s.lower()


def test_mode_amplification_high_modes_grow_fastest():
    m = mode_amplification(SINE_CI, N=101, dt=0.002)
    g = m["growth"]
    assert g[0] == pytest.approx(1.006, abs=0.01)    # lowest mode: barely grows
    assert g[-1] > g[len(g) // 2] > g[0]             # monotone: highest mode worst
    assert g[-1] == pytest.approx(1.488, abs=0.01)   # fig6_study sandbox value


def test_wellposed_case_has_no_growing_modes():
    # c=1, f=0 (free wave): every mode sits on the unit circle under the CFL limit
    g = mode_amplification(WELLPOSED, N=101, dt=0.0005)["growth"]
    assert np.max(g) == pytest.approx(1.0, abs=1e-9)


def test_mode_amplification_is_1d_only():
    with pytest.raises(NotImplementedError):
        mode_amplification(library.SINE_CI_2D, N=21, dt=0.002)
