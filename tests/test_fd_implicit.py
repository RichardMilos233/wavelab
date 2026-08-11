"""The implicit schemes — and the honest finding that being implicit does NOT
rescue you from ill-posedness.

Paper §7.2 says of its Figure 7: "the implicit scheme is more stable but exhibits a
loss of accuracy compared to the explicit scheme DUE TO LOSS OF ENERGY CONSERVATION."
That sentence is the test oracle. A theta-scheme is energy-CONSERVING, so it must NOT
be stable here — and it isn't. Stability has to be bought with dissipation
(see test_fd_regularized.py).
"""
import cmath, math, warnings
import numpy as np
import pytest
from wavelab import WaveEquation, ExplicitFD, ImplicitFD, LinearlyImplicitFD

# well-posed control: c=1, f=0 -> exact u = sin(pi x) cos(pi t)
LINEAR = WaveEquation(dim=1, c=1, f={},
                      phi=lambda z: cmath.sin(math.pi * z),
                      psi=lambda z: 0j,
                      domain=((0.0, 1.0),), name="linear_control")

# the Figure-6/7 problem (ill-posed): c=i, f = -u + u^3
SINE_CI = WaveEquation(dim=1, c=1j, f={1: -1, 3: 1},
                       phi=lambda z: cmath.sin(math.pi * z),
                       psi=lambda z: -cmath.sin(math.pi * z),
                       domain=((0.0, 1.0),), name="sine_ci_1d")

# branching-MC truth at x=0.5 (spec §6, reconfirmed at n=40k)
TRUTH = {0.1: 0.948, 0.2: 0.992, 0.3: 1.140, 0.4: 1.418}


# ----------------------------------------------------------------- theta-scheme
def test_theta_linear_matches_exact():
    sol = ImplicitFD(N=101, dt=0.001).solve(LINEAR, times=[0.5])
    exact = np.sin(np.pi * sol.points.real) * math.cos(math.pi * 0.5)
    assert np.max(np.abs(sol.u[0] - exact)) < 1e-3
    assert sol.meta["blowup_time"] is None
    assert sol.solver == "implicit_fd"


def test_theta_zero_reproduces_explicit():
    # theta=0 IS the explicit leapfrog -> must agree to round-off
    imp = ImplicitFD(N=51, dt=0.002, theta=0.0).solve(SINE_CI, times=[0.2])
    exp = ExplicitFD(N=51, dt=0.002).solve(SINE_CI, times=[0.2])
    np.testing.assert_allclose(imp.u[0], exp.u[0], atol=1e-9)


def test_theta_scheme_is_accurate_only_at_small_t():
    """It solves the equation correctly (t=0.1 matches MC to 3 digits) ..."""
    sol = ImplicitFD(N=101, dt=0.002, theta=0.5).solve(SINE_CI, times=[0.1])
    assert sol.u[0][50].real == pytest.approx(TRUTH[0.1], abs=0.01)


def test_theta_scheme_does_not_cure_illposedness():
    """... and then it diverges anyway. Being implicit cures STIFFNESS, not
    ILL-POSEDNESS. Worse than explicit in one respect: it fails SILENTLY, emitting
    finite garbage instead of NaN, so `blowup_time` stays None."""
    sol = ImplicitFD(N=101, dt=0.002, theta=0.5).solve(SINE_CI, times=[0.3])
    centre = sol.u[0][50].real
    assert np.all(np.isfinite(sol.u[0]))          # no NaN: it "survives" ...
    assert sol.meta["blowup_time"] is None
    assert abs(centre - TRUTH[0.3]) > 100         # ... but the answer is garbage


def test_theta_amplification_has_a_growing_root_by_construction():
    """Why it cannot work: the theta-scheme roots satisfy g+ * g- = 1, so whenever
    they are real one of them is outside the unit circle. Energy conservation is
    exactly what forbids the damping that stability would require."""
    N, dt, th = 101, 0.002, 0.5
    dx = 1.0 / (N - 1)
    k = np.arange(1, N - 1)
    mu = (2 - 2 * np.cos(np.pi * k * dx)) / dx**2      # eigenvalues of -u_xx
    omega = mu - 1.0                                    # c=i => c^2 = -1, f'(0) = -1
    A = 1 - dt**2 * th * omega
    B = 2 + dt**2 * (1 - 2 * th) * omega
    disc = np.sqrt(B**2 - 4 * A**2 + 0j)
    g_plus, g_minus = (B + disc) / (2 * A), (B - disc) / (2 * A)
    np.testing.assert_allclose(np.abs(g_plus * g_minus), 1.0, rtol=1e-9)  # product = 1
    growth = np.maximum(np.abs(g_plus), np.abs(g_minus))
    assert growth.max() > 1.4                          # a strongly growing mode exists


def test_newton_converges_quickly():
    sol = ImplicitFD(N=51, dt=0.002).solve(SINE_CI, times=[0.2])
    assert max(sol.meta["newton_iters"]) <= 6          # quadratic convergence


def test_newton_convergence_is_reported_not_assumed():
    """Second stage of the failure: once the state is amplified round-off, Newton
    stops converging at all — it exits on newton_maxiter with a residual many orders
    above tol, so the returned values are not solutions of the theta-scheme either.
    That is the problem's doing, not a bug, but it must be VISIBLE: meta records it
    and the first occurrence warns, exactly as blow-up does. No specific magnitude is
    asserted — those are machine-dependent (see docs/agents/gotchas.md)."""
    with pytest.warns(UserWarning, match="Newton did not converge"):
        sol = ImplicitFD(N=101, dt=0.002, theta=0.5).solve(SINE_CI, times=[0.4])
    assert sol.meta["newton_failed_steps"] > 0
    assert sol.meta["newton_max_residual"] > 1.0                 # tol is 1e-10
    t_fail = sol.meta["newton_first_failure_time"]
    assert 0.15 < t_fail < 0.30        # after round-off onset (~0.18), well before 0.4
    assert sol.meta["blowup_time"] is None                       # still no NaN


def test_newton_reports_clean_convergence_on_the_well_posed_control():
    sol = ImplicitFD(N=101, dt=0.001).solve(LINEAR, times=[0.5])
    assert sol.meta["newton_failed_steps"] == 0
    assert sol.meta["newton_first_failure_time"] is None
    assert sol.meta["newton_max_residual"] == 0.0


def test_theta_validation():
    eq = WaveEquation(dim=1, c=1, f={}, phi=lambda z: 0j, psi=lambda z: 0j)
    with pytest.raises(ValueError, match="domain"):
        ImplicitFD().solve(eq, times=[0.1])
    with pytest.raises(ValueError, match="theta"):
        ImplicitFD(theta=0.9).solve(SINE_CI, times=[0.1])
    with pytest.raises(ValueError, match="multiple of dt"):
        ImplicitFD(N=51, dt=0.002).solve(SINE_CI, times=[0.1001])


# ------------------------------------------------- linearly-implicit Euler (paper's)
def test_linearly_implicit_is_accurate_at_small_t():
    sol = LinearlyImplicitFD(N=101, dt=0.002).solve(SINE_CI, times=[0.1])
    assert sol.u[0][50].real == pytest.approx(TRUTH[0.1], abs=0.01)


def test_linearly_implicit_still_blows_up():
    """The paper's named method (Mathematica's LinearlyImplicitEuler). Implemented
    faithfully on a fixed grid it ALSO blows up: its amplification factors are
    g± = 1/(1 -+ dt sqrt(omega)), which has a POLE at dt sqrt(omega) = 1. omega spans
    ~8.9 .. 40000 across the modes, so some mode always sits near the pole whatever
    dt you choose. (Mathematica's adaptive stepping/dissipation is what saves its
    curve; a plain fixed-step implementation has no such luck.)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = LinearlyImplicitFD(N=101, dt=0.002).solve(SINE_CI, times=[0.4])
    assert sol.meta["blowup_time"] is not None
    assert sol.meta["blowup_time"] < 0.4
    assert np.all(np.isnan(sol.u[0].real))
