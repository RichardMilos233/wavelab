"""Numba fast backend: same tree functional as reference.py, iterative form.

Key identity making iteration trivial: H is a PRODUCT of per-node weights
(leaf -> boundary functional, internal -> e^{lam tau} (tau/lam) (a_k/q_k)),
so we keep one running product and a stack of pending (z, t) nodes — no tree.
"""
import math
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _sample_one(z, t0, c, lam, powers, coeffs, probs, phi, psi):
    H = 1.0 + 0.0j
    cap = 256
    stack_z = np.empty(cap, dtype=np.complex128)
    stack_t = np.empty(cap, dtype=np.float64)
    stack_z[0] = z
    stack_t[0] = t0
    top = 1
    while top > 0:
        top -= 1
        z0 = stack_z[top]
        t = stack_t[top]
        tau = np.random.exponential(1.0 / lam)
        p = np.random.random()
        if tau > t:            # leaf: boundary functional
            H *= math.exp(lam * t) * (phi(z0 + c * t) / 2 + phi(z0 - c * t) / 2
                                      + t * psi(z0 + c * t * (2 * p - 1)))
        else:                  # branch: pick power k with prob q_k (inverse CDF)
            r = np.random.random()
            j = 0
            acc = probs[0]
            while r > acc and j < probs.shape[0] - 1:
                j += 1
                acc += probs[j]
            k = powers[j]
            znew = z0 + c * tau * (2 * p - 1)
            H *= math.exp(lam * tau) * (tau / lam) * (coeffs[j] / probs[j])
            if top + k > cap:  # grow the stack (rare: subcritical tree at short t)
                cap = cap * 2 + k
                nz = np.empty(cap, dtype=np.complex128)
                nt = np.empty(cap, dtype=np.float64)
                nz[:top] = stack_z[:top]
                nt[:top] = stack_t[:top]
                stack_z, stack_t = nz, nt
            for _ in range(k):
                stack_z[top] = znew
                stack_t[top] = t - tau
                top += 1
    return H


@njit(parallel=True, cache=True)
def estimate_grid(zs, t, n, lam, c, powers, coeffs, probs, seed, phi, psi):
    P = zs.shape[0]
    means = np.empty(P, dtype=np.complex128)
    errs = np.empty(P, dtype=np.float64)
    for i in prange(P):
        np.random.seed(seed + i)          # per-point seed -> deterministic regardless of threads
        acc = 0.0 + 0.0j
        acc2 = 0.0
        for _ in range(n):
            v = _sample_one(zs[i], t, c, lam, powers, coeffs, probs, phi, psi)
            acc += v
            acc2 += v.real * v.real + v.imag * v.imag
        m = acc / n
        var = acc2 / n - (m.real * m.real + m.imag * m.imag)
        if var < 0.0:
            var = 0.0
        means[i] = m
        errs[i] = math.sqrt(var / n)
    return means, errs
