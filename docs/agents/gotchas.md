# Gotchas — sharp edges, pins, and reference-code bugs

## Environment

- **numpy must stay <2.4**: numpy 2.4.x causes a NATIVE crash (0xc06d007f) inside
  matplotlib's compiled transforms on Windows, and numba 0.65.x doesn't support it.
  matplotlib pinned <3.11 for the same ABI reason. Working combo: numpy 2.3.5 +
  matplotlib 3.10.8. Both pinned in `environment.yml` — do not "upgrade to fix".
  (Platform-independent: the numba constraint bites everywhere.)
- Normal usage (any OS): `conda activate wavelab`, then `python` / `pytest` directly.

### Windows box (the author's original machine)

- `python`/`conda` are NOT on PATH. Use absolute paths:
  `C:\Users\LiaoTianrui\anaconda3\envs\wavelab\python.exe`;
  conda: `C:\Users\LiaoTianrui\anaconda3\Scripts\conda.exe`.
- PowerShell 5.1: no `&&`; heredocs via Bash tool are safer for multi-line commit
  messages; regex in Bash heredocs gets mangled — use the Write tool for scripts.

### macOS / Linux

- Plain `python`, `pytest`, `conda` work once the env is activated; no absolute paths.
- numba supports Apple Silicon (osx-arm64) via conda-forge — `environment.yml`
  resolves as-is. If a solver is slow, check you are on the `numba` backend, not the
  python reference backend.
- **Verified on Apple Silicon 2026-08-11** (macOS, osx-arm64): `conda env create -f
  environment.yml` resolves unchanged to python 3.12.13 / numpy 2.3.5 /
  matplotlib 3.10.9 / numba 0.66.0 (numba 0.66 is fine — the docs elsewhere say
  0.65.x only because that is what the Windows box had). `pytest -q` → 101 passed
  in ~12 s; every number in the regression lock below reproduced exactly.
- Regenerating `docs/tutorial/figures/` on macOS rewrites all five PNGs with
  different bytes — including `branching_tree.png`, which is a pure schematic with no
  computation in it. That is font/matplotlib-version rendering, not a numerical
  change. Don't commit the churn unless the plotted content actually changed.

## Semantics that look like bugs but aren't

- `ImplicitFD` returning huge finite values with `blowup_time=None` on SINE_CI_1D is
  the DOCUMENTED silent-failure mode of an energy-conserving scheme on an ill-posed
  problem (spec §3.3a). Do not "fix" it.
- Overflow RuntimeWarnings during FD blow-up are the expected path to NaN detection.
- MC results have `Im u ≈ 0` but not exactly 0 on real problems — that's MC noise;
  tests bound it, don't zero it.
- Statistical tests at 3σ fail ~0.3% of runs by chance: rerun once with another seed
  before investigating; twice-failing = real bug.

## Reference-code bugs (do NOT validate against these)

1. `../Nonlinear_Wave_simulations/Simulation_07` (d=2, c=i, sine): sets `aJ = -1`
   for BOTH J=1 and J=3, but its own README says f = −u + u³ (a₃ = +1). wavelab
   derives coefficients from `eq.f`, so validate d≥2 against closed forms
   (SIM05/SIM08/solitons), never against Sim_07 output.
2. `../wave_equation/finite_differences_wave_1D.ipynb` ("implicit" notebook):
   `theta = 0.0005` despite the comment claiming Crank–Nicolson θ=0.5, AND its
   spatial-operator sign solves the well-posed problem, not the elliptic one. Its
   apparent stability is an artifact. Unusable as a reference.

## Regression lock (never edit expected numbers)

SINE_CI_1D, dt=0.002: explicit blow-up 0.44 (N=51) / 0.232 (N=101) / 0.128 (N=201);
FD max|u| at t=0.1/0.2/0.3 (N=51): 0.948/0.991/1.137; MC u(0.5,t): 0.948/0.990/
1.135/1.412 (t=0.1..0.4), u(0.5,0.5)≈1.91; mode growth 1.006 (k=1) / 1.488 (k=99).
Source: `tests/test_fig6_regression.py`, `tests/test_illposed_signature.py`,
verified against the pre-refactor sandbox `../fig6_study/` on 2026-07-13.
All of the above re-verified on macOS/arm64 2026-08-11 — they are genuinely
machine-independent. The ImplicitFD magnitudes are NOT; see the next section.

## The ImplicitFD garbage values are NOT reproducible (2026-08-11)

`ImplicitFD(theta=0.5)` on SINE_CI_1D emits finite garbage for `t ≳ 0.2` (spec §3.3a).
The docs used to quote **specific** magnitudes for that garbage — `26.6` at t=0.2,
`1557` at t=0.3, `1988` at t=0.4. **Those were one machine's round-off, not
measurements**, and all five sites (README, `solvers.md`, tutorial 03 §3.3 / 05 §5.2,
spec §3.3a) were rewritten on 2026-08-11 to state order of magnitude + onset time
instead. **Rule: quote the time at which the output stops being meaningful, never the
value it prints there.** Do not treat a mismatch as a regression, and do not "fix" the
solver. The numbers below are kept as the *evidence* for that rule.

Evidence (N=101, dt=0.002; perturb φ by a *relative* ε, i.e. below double precision):

| t | quoted | macOS/arm64 | range over ε ∈ ±1e-15 |
|---|---|---|---|
| 0.1 | 0.948 | 0.9479 | 0.95 … 0.95 — rock stable |
| 0.2 | 26.6 | 22.58 | 7.74 … 32.08 |
| 0.3 | 1557.2 | 1293.11 | **−1519.52 … 1553.41** (sign flips) |
| 0.4 | 1988.2 | 3161.10 | −1654.91 … 3161.10 |

Why: the θ-scheme's highest grid mode is amplified by **g\* = 1.5129 per step** at N=101
(don't confuse this with 1.488 — that is the *explicit* scheme's k=99 figure in the
regression lock). A 1e-16 seed therefore reaches O(1) in `16·ln10/ln g* = 89` steps,
i.e. **t ≈ 0.178**, exactly where the computed curve departs; the same formula predicts
the observed onset at N=51 (g\*=1.223, t≈0.37) and N=201 (g\*=2.549, t≈0.08). From there
the value is *entirely* determined by the round-off pattern, which differs between
LAPACK builds. Runs on one machine are bit-identical; runs across machines are not.
Degradation is smooth and measurable: **about one significant digit lost per 10 steps**,
and past t≈0.19 not even the sign is determined.

Two corrections to the first version of this note (re-verified on Windows/OpenBLAS
2026-08-11): the linear amplification over 150 steps is 1.5129¹⁵⁰ = **9.4e26**, not 1e22
— but no quantity is ever literally amplified that far, because growth **saturates** near
|u|≈5e4 once the cubic term bites (~t=0.28). The load-bearing number is the **89 steps**
to reach O(1), not any total factor.

**Second stage — Newton stops converging.** Past t≈0.21 the Newton iteration exits on
`newton_maxiter` with residuals of 1e2–1e4 on ~96 of 199 steps, so the output is not
even a solution of the θ-scheme. This is the problem's doing, not a bug (raising the cap
to 500 still gives garbage, and −1782 at t=0.4), but it is now **reported**:
`meta["newton_failed_steps"]`, `["newton_first_failure_time"]`, `["newton_max_residual"]`,
plus a warning on first occurrence. It is not the *cause* of the machine-dependence —
iteration counts are identical across sub-ulp perturbations during the converged phase,
and thread count (1/2/4/8) changes nothing.

**Single-machine proof that it is round-off**, no second machine needed: swapping
`np.linalg.solve` for the mathematically equivalent `inv(J)@G`, `lstsq`, or a stencil
Laplacian gives t=0.3 values of 1557 / 1549 / 2782 / 1653 and **flips the sign** at
t=0.4, while t=0.1 stays 0.9479 to 5 digits in every route.

This **strengthens** §3.3a rather than weakening it: the θ-scheme's output is not
merely wrong, it is not even reproducible — its sign is not fixed. That is a sharper
statement of "silent garbage" than any single value.

Robust by contrast (unmoved by the same perturbation, and identical across machines):
all explicit-FD blow-up times, `LinearlyImplicitFD` blow-up 0.182–0.184 (docs say
"dies at 0.18"), every RegularizedFD and MC number. Blow-up times are integer step
counts, hence robust; amplified-round-off magnitudes are not. The test suite already
respects this — `test_theta_scheme_does_not_cure_illposedness` asserts
`abs(centre − truth) > 100`, never a specific value.
