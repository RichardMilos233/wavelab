"""Pure-Python branching MC recursion — the readable ground truth (spec §3.3).

u(z,t) = E[H] over a random branching tree: exponential clock Exp(rate lam),
spatial mark on the light cone z +- c*tau*(2p-1), splitting into k children for
the u^k term, with importance weights e^{lam*tau}, tau/lam, a_k/q_k.
Free parameters lam and q change variance only, never the mean.
"""
import math
import numpy as np


def _sample(z, t, c, lam, phi, psi, powers, coeffs, probs, rng):
    tau = rng.exponential(1.0 / lam)          # Exp(rate lam): mean 1/lam
    p = rng.random()
    if tau > t:                               # clock survives -> boundary functional
        return math.exp(lam * t) * (
            phi(z + c * t) / 2 + phi(z - c * t) / 2
            + t * psi(z + c * t * (2 * p - 1))
        )
    j = rng.choice(len(powers), p=probs)      # branch: pick power k with prob q_k
    k, a, q = int(powers[j]), coeffs[j], probs[j]
    znew = z + c * tau * (2 * p - 1)
    H = 1 + 0j
    for _ in range(k):                        # k = 0 -> empty product = 1
        H *= _sample(znew, t - tau, c, lam, phi, psi, powers, coeffs, probs, rng)
    return math.exp(lam * tau) * (tau / lam) * (a / q) * H


def _mark_nd(dim, s, c, p, theta):
    """Spatial mark on the light cone, radius scaled by s (= t at a leaf, tau at a branch).

    d=2 (Simulation_07): R = s*sqrt(1-(1-p)^2), y = c*R*(cos th, sin th)   [disc]
    d=3 (Simulation_08): alpha = arccos(1-2p),  y = c*s*(sin a cos th, sin a sin th, cos a)
                                                                            [sphere]
    """
    if dim == 2:
        R = s * math.sqrt(1.0 - (1.0 - p) ** 2)
        return np.array([c * R * math.cos(theta), c * R * math.sin(theta)],
                        dtype=np.complex128)
    alpha = math.acos(1.0 - 2.0 * p)
    sa = math.sin(alpha)
    return np.array([c * s * sa * math.cos(theta),
                     c * s * sa * math.sin(theta),
                     c * s * math.cos(alpha)], dtype=np.complex128)


def _sample_nd(z, t, dim, c, lam, phi, psi, grad_phi, powers, coeffs, probs, rng):
    """d>=2 tree functional. The leaf carries a gradient term y . grad(phi)(z+y)
    which has NO d=1 analogue (d=1 uses the d'Alembert form instead)."""
    tau = rng.exponential(1.0 / lam)
    p = rng.random()
    theta = rng.random() * 2.0 * math.pi
    if tau > t:                                   # leaf
        y = _mark_nd(dim, t, c, p, theta)
        zy = z + y
        g = grad_phi(zy)
        I2 = sum(y[i] * g[i] for i in range(dim))
        return math.exp(lam * t) * (phi(zy) + I2 + t * psi(zy))
    y = _mark_nd(dim, tau, c, p, theta)           # branch
    znew = z + y
    j = rng.choice(len(powers), p=probs)
    k, a, q = int(powers[j]), coeffs[j], probs[j]
    H = 1 + 0j
    for _ in range(k):
        H *= _sample_nd(znew, t - tau, dim, c, lam, phi, psi, grad_phi,
                        powers, coeffs, probs, rng)
    return math.exp(lam * tau) * (tau / lam) * (a / q) * H


def estimate(eq, z, t, n, lam, powers, coeffs, probs, rng):
    """Mean and stderr of n i.i.d. tree functionals at one (z, t)."""
    c = complex(eq.c)
    samples = np.empty(n, dtype=np.complex128)
    if eq.dim == 1:
        for i in range(n):
            samples[i] = _sample(complex(z), float(t), c, lam, eq.phi, eq.psi,
                                 powers, coeffs, probs, rng)
    else:
        z = np.asarray(z, dtype=np.complex128)
        for i in range(n):
            samples[i] = _sample_nd(z, float(t), eq.dim, c, lam, eq.phi, eq.psi,
                                    eq.grad_phi, powers, coeffs, probs, rng)
    mean = samples.mean()
    stderr = float(samples.std() / math.sqrt(n))   # complex std = sqrt(E|x-mean|^2)
    return mean, stderr
