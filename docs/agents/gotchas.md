# Gotchas — sharp edges, pins, and reference-code bugs

## Environment

- **numpy must stay <2.4**: numpy 2.4.x causes a NATIVE crash (0xc06d007f) inside
  matplotlib's compiled transforms on Windows. matplotlib pinned <3.11 for the same
  ABI reason. Working combo: numpy 2.3.5 + matplotlib 3.10.8. Both pinned in
  `environment.yml` — do not "upgrade to fix".
- Normal usage (any OS): `conda activate wavelab`, then `python` / `pytest` directly.

### Windows box (the author's original machine)

- `python`/`conda` are NOT on PATH. Use absolute paths:
  `C:\Users\LiaoTianrui\anaconda3\envs\wavelab\python.exe`;
  conda: `C:\Users\LiaoTianrui\anaconda3\Scripts\conda.exe`.
- PowerShell 5.1: no `&&`; heredocs via Bash tool are safer for multi-line commit
  messages; regex in Bash heredocs gets mangled — use the Write tool for scripts.

### macOS / Linux

- Plain `python`, `pytest`, `conda` work once the env is activated; no absolute paths.
- **Verified on Apple Silicon 2026-08-11** (macOS, osx-arm64): `conda env create -f
  environment.yml` resolves unchanged to python 3.12.13 / numpy 2.3.5 /
  matplotlib 3.10.9. `pytest -q` → all green in ~12 s; every number in the
  regression lock below reproduced exactly. (That run predates the numba removal,
  when the suite had 101 tests.)
- Regenerating `docs/tutorial/figures/` on macOS rewrites all five PNGs with
  different bytes — including `branching_tree.png`, which is a pure schematic with no
  computation in it. That is font/matplotlib-version rendering, not a numerical
  change. Don't commit the churn unless the plotted content actually changed.

## Removed on purpose — do not reintroduce

- **The numba backend (`mc/fast.py`) and BranchingMC's `backend` / `workers`
  parameters were deleted 2026-08-14** (owner's decision). wavelab demonstrates the
  algorithm; it is not a production solver, and pure Python is fast enough for every
  figure the FYP needs — ~3 s for 1e6 samples at one point, ~4.5 min for 1e6 samples
  over 101 points. `workers` had never been wired to anything: accepted, stored on
  `self`, never read. Cost of the removal: `test_backends_agree.py`, which
  cross-validated the recursive and iterative-stack implementations against each
  other; correctness now rests on the closed-form golden tests alone. If paper-scale
  2-D runs ever need it, recover it from git history rather than rewriting it.
- Consequence: `numpy<2.4` is still pinned, but now **only** for the matplotlib ABI
  crash on Windows — the numba constraint is gone. Don't "unpin because numba left".

## Semantics that look like bugs but aren't

- `ImplicitFD` returning huge finite values with `blowup_time=None` on SINE_CI_1D is
  the DOCUMENTED silent-failure mode of an energy-conserving scheme on an ill-posed
  problem (spec §3.3a). Do not "fix" it.
- Overflow RuntimeWarnings during FD blow-up are the expected path to NaN detection.
- MC results have `Im u ≈ 0` but not exactly 0 on real problems — that's MC noise;
  tests bound it, don't zero it.
- Statistical tests at 3σ fail ~0.3% of runs by chance: rerun once with another seed
  before investigating; twice-failing = real bug.

## d=2 and the defocusing problems (§7.1, §7.3) — different failure mode

- **`blowup_time` is the wrong instrument on a defocusing problem.** §7.1/§7.3 carry
  a₃ = −1, and −u³ *opposes* growth, so amplified round-off saturates instead of
  running away. Explicit FD on SINE_DEFOCUS_CI_2D returns bounded garbage
  (max|u| ≈ 175 at N=41, 274 at N=61) with `blowup_time=None` and no NaN — which is
  exactly the paper's Fig 8b. Judge these runs by ACCURACY against MC, not by when
  they die. The focusing counterpart (SINE_CI_2D, a₃=+1) does NaN, at t=0.398 (N=41).
- **MC's variance wall arrives much earlier in d=2**: mode (1,1) grows at
  √(2π²−1) = 4.33 versus d=1's √(π²−1) = 2.98. On SINE_DEFOCUS_CI_2D the relative
  stderr is still fine at t=0.5 but useless by t≈0.8 (stderr 66 on a value of 12),
  against t≈1.2 in d=1. Do not reuse the 1-D wall estimate.
- **MC in d=2 costs per point**: a 17×17 surface at n=10⁴ is ~23 s, 21×21 at n=2×10⁴
  is ~70 s. Split the two jobs: for a PICTURE use many points at small n (that is
  what `examples/fig8.py` does, N=17 at n=10⁴); for an ACCURACY claim use one point
  at large n — a coarse surface grid is far too noisy to quote as a reference.
- **`RegularizedFD` d=2 keeps a box, not a disc**: modes k ≤ k_max AND m ≤ k_max, so
  the fastest survivor is (k_max, k_max) growing at √2·k_max·π. A d=2 run at k_max=K
  is roughly as aggressive as d=1 at √2·K — the 1-D death-time formula under-predicts
  if applied naively.

## Reference-code bugs (do NOT validate against these)

1. ~~`Simulation_07` sets `aJ = -1` for both J=1 and J=3, contradicting its
   README~~ — **CORRECTED 2026-08-17. The code is right; the README is wrong.**
   `aJ = -1` for both powers means f = −u − u³, which is the paper's **§7.3**
   (defocusing elliptic); the README describes §7.2's f = −u + u³. Everything else
   in the code matches Figure 8 exactly: c = i, t = 0.5, 101×101 grid on [0,1]²,
   λ = 0.25, φ = sin(πz₁)sin(πz₂), ψ = −φ. So Simulation_07 **is the run that
   produced Figure 8a**, and its output IS a valid cross-check — see the
   verification note below. The earlier entry here cost us a free validation set;
   do not re-add it.
2. `../wave_equation/finite_differences_wave_1D.ipynb` ("implicit" notebook):
   `theta = 0.0005` despite the comment claiming Crank–Nicolson θ=0.5, AND its
   spatial-operator sign solves the well-posed problem, not the elliptic one. Its
   apparent stability is an artifact. Unusable as a reference.

## Verified against the authors' own §7.3 run (2026-08-17)

`Simulation_07/results/monte_carlo_real.csv` is the authors' 101×101 Monte Carlo
field for §7.3 at t=0.5 (external to this repo — see CLAUDE.md; absent on a fresh
clone). Compared against `RegularizedFD(N=101, dt=0.002, k_max=6)` on
`SINE_DEFOCUS_CI_2D`, over all 10 201 points:

    max |difference| = 0.0135      mean = 0.0009      at the centre = 0.0055

which sits inside the authors' own MC noise (their |Im u| reaches 0.0056 while the
true imaginary part is 0). Sample points, authors' MC vs our RegularizedFD:
(0.5,0.5) 2.74616 / 2.75166 · (0.25,0.25) 1.70502 / 1.70387 · (0.5,0.25) 2.35969 /
2.36020 · (0.1,0.5) 1.21042 / 1.21008. This is a one-time cross-check recorded here,
NOT a test — the CSV is not in this repo and a test would fail on any other machine.

## Don't judge a DEFOCUSING problem with an absolute threshold (fixed 2026-08-18)

`tests/test_fd_regularized_2d.py::test_k_max_has_a_usable_window` was RED from
`38f507f`, the commit that introduced it, until 2026-08-18. Re-running that commit in
isolation reproduced the identical value `0.7490911728582663`, so it was never a
regression — it was committed without the suite being green, and the messages of
`de92603` and `775454a` ("113 tests green") are wrong on this point. Worth knowing if
you are ever bisecting through that range.

The bug was in the test, not the solver. It is the d=2 analogue of
`test_more_modes_kept_means_earlier_blowup`, which in d=1 (focusing, a3=+1) judges by
`blowup_time`. §7.3 is DEFOCUSING, so `blowup_time` is always None (see the section
above) and it had to judge by distance from an MC reference instead — but it kept a
d=1-sized ABSOLUTE threshold:

    k_max=20 -> err > 1.0      actual 0.749

which the problem cannot reach, because `-u^3` opposes growth and the deviation
SATURATES. The qualitative claim was always true; only the magnitude was invented.

Now judged relatively, which is what the physics actually guarantees:

    err[20] > 10 * max(err[3], err[6])     # ~16x  -- outside vs inside the window
    err[20] > 10 * mc_stderr               # ~20x  -- real bias, not sampling noise

**The rule:** on §7.1/§7.3 (a3 = -1) never assert a blow-up time and never assert an
absolute error bound for a diverging run. Saturation bounds both. Assert SEPARATION.

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
