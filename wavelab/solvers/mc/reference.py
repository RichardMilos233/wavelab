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


def estimate(eq, z, t, n, lam, powers, coeffs, probs, rng):
    """Mean and stderr of n i.i.d. tree functionals at one (z, t)."""
    c = complex(eq.c)
    samples = np.empty(n, dtype=np.complex128)
    for i in range(n):
        samples[i] = _sample(complex(z), float(t), c, lam, eq.phi, eq.psi,
                             powers, coeffs, probs, rng)
    mean = samples.mean()
    stderr = float(samples.std() / math.sqrt(n))   # complex std = sqrt(E|x-mean|^2)
    return mean, stderr
