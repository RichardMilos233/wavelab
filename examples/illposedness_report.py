"""The full Figure-6/7 argument in one script, using only the wavelab library.

Question: why does explicit FD blow up on the elliptic (c=i) problem while the
branching Monte Carlo stays stable — and is "use an implicit scheme" the answer?
(Short version: no. See section 3.)

Run:  <anaconda>/envs/wavelab/python.exe examples/illposedness_report.py
"""
import warnings

from wavelab import (library, ExplicitFD, ImplicitFD, RegularizedFD,
                     BranchingMC, compare)
from wavelab.experiments import (blowup_scan, blowup_table, mode_amplification,
                                 variance_profile, variance_plot)

warnings.simplefilter("ignore")          # blow-ups are the subject, not a surprise

eq = library.SINE_CI_1D
times = [0.1, 0.2, 0.3, 0.4]

print("=== 1. ill-posedness fingerprint: refine the grid, blow up SOONER ===")
print("    (a well-posed problem would IMPROVE under refinement)")
rows = blowup_scan(eq, lambda N, dt: ExplicitFD(N=N, dt=dt),
                   Ns=(51, 101, 201), dts=(0.004, 0.002, 0.001), probe_time=0.5)
print(blowup_table(rows))

print("\n=== 2. why: the highest Fourier modes amplify fastest ===")
m = mode_amplification(eq, N=101, dt=0.002)
g, k = m["growth"], m["k"]
print(f"  lowest  mode k={k[0]:<3}: growth/step = {g[0]:.4f}")
print(f"  middle  mode k={k[len(g)//2]:<3}: growth/step = {g[len(g)//2]:.4f}")
print(f"  highest mode k={k[-1]:<3}: growth/step = {g[-1]:.4f}   <-- grid scale")
print(f"  over 116 steps the highest mode amplifies round-off by {g[-1]**116:.1e}")

print("\n=== 3. is 'go implicit' the fix?  NO. ===")
print("  The paper's own words about its Fig 7: the implicit scheme is 'more stable")
print("  but exhibits a loss of accuracy ... DUE TO LOSS OF ENERGY CONSERVATION'.")
print("  Stability is BOUGHT with dissipation, not earned by implicitness:")
imp = ImplicitFD(N=101, dt=0.002, theta=0.5).solve(eq, [0.3])
print(f"    theta-scheme (energy-CONSERVING) at t=0.3: u(0.5) = {imp.u[0][50].real:.1f}")
print(f"      -> finite, so blowup_time is {imp.meta['blowup_time']} ... but the truth is ~1.14.")
print("      It fails SILENTLY. Its roots satisfy g+ * g- = 1, so a growing root")
print("      always exists -- conservation is exactly what forbids the damping.")

print("\n=== 4. what DOES work for FD: regularize (spectral cut-off) ===")
for k_max in (12, 20, 30):
    sol = RegularizedFD(N=101, dt=0.002, k_max=k_max).solve(eq, [0.7])
    bt = sol.meta["blowup_time"]
    state = "survives to t=0.7" if bt is None else f"blows up at t={bt}"
    print(f"    k_max={k_max:<3} -> {state}")
print("  Keep more modes and the ill-posedness you suppressed comes back:")
print("  the cut-off IS the regularization, and it costs accuracy (see section 5).")

print("\n=== 5. the methods side by side ===")
fd = ExplicitFD(N=51, dt=0.002).solve(eq, times)
reg = RegularizedFD(N=51, dt=0.002, k_max=12).solve(eq, times)
mc = BranchingMC(lam=0.25, n=40_000, seed=0).solve(eq, times)
cmp = compare(fd, reg, mc)
print(cmp.table(probe_points=[0.1, 0.3, 0.5, 0.7, 0.9]))
cmp.plot("illposedness_three_methods.png")
print("saved illposedness_three_methods.png")

print("\n=== 6. MC is not magic either: its own short-time window ===")
print("  FD dies of ill-posedness (deterministic, grid-coupled).")
print("  MC dies of VARIANCE (statistical, pointwise) -- noisy, not wrong.")
vp = variance_profile(eq, BranchingMC(lam=0.25, n=20_000, seed=1),
                      times=[0.1, 0.3, 0.5, 0.8, 1.2, 1.6], point=0.5)
for r in vp:
    print(f"  t={r['t']:<4} u={r['u'].real:+9.4f}  stderr={r['stderr']:9.4f}  "
          f"rel={r['rel_stderr']:7.1%}")
variance_plot(vp, "mc_variance_profile.png")
print("saved mc_variance_profile.png")
