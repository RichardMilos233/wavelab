import cmath, math
import numpy as np
import pytest
from wavelab import WaveEquation

def sine_eq(**kw):
    args = dict(dim=1, c=1j, f={1: -1, 3: 1},
                phi=lambda z: cmath.sin(math.pi * z),
                psi=lambda z: -cmath.sin(math.pi * z),
                domain=((0.0, 1.0),), name="sine_ci_1d")
    args.update(kw)
    return WaveEquation(**args)

def test_construction_and_fields():
    eq = sine_eq()
    assert eq.dim == 1 and eq.c == 1j and eq.f == {1: -1, 3: 1}
    assert eq.bc == "dirichlet" and eq.exact is None

def test_f_callable_matches_polynomial():
    f = sine_eq().f_callable()
    u = np.array([0.5 + 0j, 2.0 + 1j])
    np.testing.assert_allclose(f(u), -u + u**3)
    assert f(u).dtype == np.complex128

def test_f_callable_constant_term():
    f = sine_eq(f={0: 2 + 1j, 2: 1}).f_callable()
    u = np.array([3.0 + 0j])
    np.testing.assert_allclose(f(u), (2 + 1j) + u**2)

@pytest.mark.parametrize("bad", [{-1: 1.0}, {1.5: 1.0}])
def test_rejects_bad_powers(bad):
    with pytest.raises(ValueError, match="power"):
        sine_eq(f=bad)

def test_rejects_bad_dim():
    with pytest.raises(ValueError, match="dim"):
        sine_eq(dim=4, domain=None)

def test_rejects_domain_dim_mismatch():
    with pytest.raises(ValueError, match="domain"):
        sine_eq(domain=((0, 1), (0, 1)))

def test_rejects_phi_not_accepting_complex():
    with pytest.raises(ValueError, match="phi"):
        sine_eq(phi=lambda z: math.sin(z))   # math.sin rejects complex

def test_frozen():
    with pytest.raises(Exception):
        sine_eq().dim = 2
