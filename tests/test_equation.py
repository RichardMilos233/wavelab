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

def test_f_prime_callable():
    f1 = sine_eq().f_prime_callable()          # f = -u + u^3  ->  f' = -1 + 3u^2
    u = np.array([0.5 + 0j, 2.0 + 1j])
    np.testing.assert_allclose(f1(u), -1 + 3 * u**2)
    assert f1(u).dtype == np.complex128

def test_f_prime_drops_constant_term():
    f1 = sine_eq(f={0: 5 + 0j, 2: 1}).f_prime_callable()   # f = 5 + u^2 -> f' = 2u
    u = np.array([3.0 + 0j])
    np.testing.assert_allclose(f1(u), 2 * u)

def test_grad_phi_defaults_none_and_is_accepted():
    assert sine_eq().grad_phi is None
    eq2 = WaveEquation(dim=2, c=1j, f={2: 1},
                       phi=lambda z: 6 / (z[0] + z[1])**2,
                       psi=lambda z: 0j,
                       grad_phi=lambda z: (-12 / (z[0] + z[1])**3,
                                           -12 / (z[0] + z[1])**3))
    g = eq2.grad_phi(np.array([2 + 0j, 2 + 0j]))
    assert len(g) == 2

def test_rejects_grad_phi_with_wrong_arity():
    with pytest.raises(ValueError, match="grad_phi"):
        WaveEquation(dim=2, c=1j, f={2: 1},
                     phi=lambda z: 6 / (z[0] + z[1])**2,
                     psi=lambda z: 0j,
                     grad_phi=lambda z: (1 + 0j,))     # returns 1 value, dim=2
