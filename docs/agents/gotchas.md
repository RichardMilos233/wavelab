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
Several docs quote **specific** magnitudes for that garbage — `26.6` at t=0.2, `1557`
at t=0.3, `1988` at t=0.4 (README table, `docs/agents/solvers.md`, tutorial 03 §3.3
table and 05 §5.2 table, spec §3.3a). **Those are one machine's round-off, not
measurements.** Do not treat a mismatch as a regression, and do not "fix" the solver.

Evidence (N=101, dt=0.002; perturb φ by a *relative* ε, i.e. below double precision):

| t | quoted | macOS/arm64 | range over ε ∈ ±1e-15 |
|---|---|---|---|
| 0.1 | 0.948 | 0.9479 | 0.95 … 0.95 — rock stable |
| 0.2 | 26.6 | 22.58 | 7.74 … 32.08 |
| 0.3 | 1557.2 | 1293.11 | **−1519.52 … 1553.41** (sign flips) |
| 0.4 | 1988.2 | 3161.10 | −1654.91 … 3161.10 |

Why: the θ-scheme's per-step growth exceeds 1.4 on this problem, so over 150 steps
round-off at 1e-16 is amplified by ~1e22. The value is *entirely* determined by the
round-off pattern, which differs between LAPACK builds (Accelerate vs OpenBLAS/MKL).
Runs on one machine are bit-identical; runs across machines are not.

This **strengthens** §3.3a rather than weakening it: the θ-scheme's output is not
merely wrong, it is not even reproducible — its sign is not fixed. That is a sharper
statement of "silent garbage" than any single value.

Robust by contrast (unmoved by the same perturbation, and identical across machines):
all explicit-FD blow-up times, `LinearlyImplicitFD` blow-up 0.182–0.184 (docs say
"dies at 0.18"), every RegularizedFD and MC number. Blow-up times are integer step
counts, hence robust; amplified-round-off magnitudes are not. The test suite already
respects this — `test_theta_scheme_does_not_cure_illposedness` asserts
`abs(centre − truth) > 100`, never a specific value.
