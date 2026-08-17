"""Paper Figure 8 (§7.3): the defocusing elliptic problem in d = 2.

    u_tt + Lap u + u + u^3 = 0   on [0,1]^2,   c = i,   f(u) = -u - u^3
    phi = sin(pi x) sin(pi y),   psi = -phi,   t = 0.5

The paper shows Monte Carlo producing a smooth dome peaking near 3 (Fig 8a) and
finite differences producing +/-800 noise (Fig 8b). This script reproduces both, and
adds the third panel the paper does not have: the SAME finite-difference scheme with
a spectral cut-off, which lands back on the Monte Carlo answer.

Two things about §7.3 differ from the d=1 Figure-6 study and are worth stating:

  * The cubic is DEFOCUSING (a_3 = -1). Because -u^3 opposes growth, amplified
    round-off saturates rather than running away: explicit FD returns bounded
    garbage with blowup_time=None, never NaN. That is precisely what Fig 8b shows,
    and it means "blow-up time" is the wrong instrument here — accuracy is.
  * Mode (1,1) grows at sqrt(2 pi^2 - 1) = 4.33 in d=2 against sqrt(pi^2 - 1) = 2.98
    in d=1, so the same t is a harder problem, and MC's variance wall arrives much
    earlier (t ~ 0.6 here versus t ~ 1.2 in d=1).

Run: python examples/fig8_defocusing_2d.py     (~30 s, most of it Monte Carlo)
"""
import math
import time
import warnings

import numpy as np

from wavelab import ExplicitFD, RegularizedFD, BranchingMC, library
from wavelab.experiments import surfaces

EQ = library.SINE_DEFOCUS_CI_2D
T = 0.5
M = 17          # MC grid: MxM points, each an independent expectation
N = 61          # FD grid


def main():
    print(__doc__.split("Run:")[0].strip())
    print("\n" + "=" * 72)

    # ---- 1. Monte Carlo: the reference -----------------------------------
    g = np.linspace(0.0, 1.0, M)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([GX.ravel(), GY.ravel()], axis=1).astype(np.complex128)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # edge points have |u| ~ 0, so rel err ~ inf
        mc = BranchingMC(lam=0.25, n=10_000, seed=0).solve(EQ, [T], points=pts)
    Z = mc.u[0].real.reshape(M, M)
    print(f"1. BranchingMC  ({M}x{M} points, n=10000)   {time.perf_counter()-t0:5.1f}s")
    print(f"   range [{Z.min():+.3f}, {Z.max():+.3f}]   centre {Z[M//2, M//2]:+.4f}"
          f" +/- {mc.meta['stderr'][0].reshape(M, M)[M//2, M//2]:.4f}")
    sig = math.sqrt(2 * math.pi**2 - 1)
    print(f"   linear prediction at the centre: "
          f"{math.cosh(sig*T) - math.sinh(sig*T)/sig:.3f} "
          f"(defocusing cubic pulls it down)")

    # ---- 2. Explicit FD: the failure -------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fd = ExplicitFD(N=N, dt=0.002).solve(EQ, [T])
    F = fd.u[0].real.reshape(N, N)
    print(f"\n2. ExplicitFD   (N={N})")
    print(f"   range [{F.min():+.1f}, {F.max():+.1f}]   "
          f"blowup_time={fd.meta['blowup_time']}  <- finite, and meaningless")

    # ---- 3. Regularized FD: the fix --------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rg = RegularizedFD(N=N, dt=0.002, k_max=6).solve(EQ, [T])
    R = rg.u[0].real.reshape(N, N)
    print(f"\n3. RegularizedFD (N={N}, k_max=6)")
    print(f"   range [{R.min():+.3f}, {R.max():+.3f}]   centre {R[N//2, N//2]:+.4f}")

    # ---- 4. agreement ----------------------------------------------------
    # The surface MC above is deliberately cheap (many points, few samples), so it
    # is too noisy to judge accuracy by. Take one high-precision point for that.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_mc = BranchingMC(lam=0.25, n=200_000, seed=7).solve(
            EQ, [T], points=np.array([[0.5, 0.5]]))
    ref, ref_se = ref_mc.u[0, 0].real, ref_mc.meta["stderr"][0, 0]
    idx = np.round(np.linspace(0, N - 1, M)).astype(int)
    err = np.abs(R[np.ix_(idx, idx)] - Z)
    print(f"\n4. high-precision MC at the centre (n=200000): {ref:.4f} +/- {ref_se:.4f}")
    print(f"   regularized FD centre:                       {R[N//2, N//2]:.4f}"
          f"   ({abs(R[N//2, N//2]-ref)/ref_se:.1f} sigma)")
    print(f"   |regularized - coarse MC| over the {M}x{M} surface grid: "
          f"max {err.max():.4f}, mean {err.mean():.4f}")

    # ---- 5. the k_max window --------------------------------------------
    print(f"\n5. the k_max window at t=0.5 (centre value; MC says {ref:.4f}):")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for K in (1, 3, 6, 10, 20, 39):
            r = RegularizedFD(N=41, dt=0.002, k_max=K).solve(EQ, [T])
            c = r.u[0].real.reshape(41, 41)[20, 20]
            tag = ("OK" if abs(c - ref) < 0.15 else
                   "under-resolved" if K <= 2 else "contaminated")
            print(f"   k_max={K:>3}:  {c:>10.4f}   {tag}")
    print("   -> too few modes cannot represent the solution, too many let the")
    print("      ill-posedness back in. In d=1 only the upper end is visible.")

    surfaces([mc, fd, rg],
             shapes=[(M, M), None, None],
             labels=[f"Monte Carlo  (Fig 8a)\n{M}x{M} points, n=10000",
                     f"Explicit FD  (Fig 8b)\nN={N} — bounded garbage",
                     f"Regularized FD\nN={N}, k_max=6 — the fix"],
             path="fig8_defocusing_2d.png",
             suptitle=f"Paper §7.3: defocusing elliptic problem, d=2, t={T}")
    print("\nsaved fig8_defocusing_2d.png")


if __name__ == "__main__":
    main()
