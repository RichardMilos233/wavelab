# wavelab — design spec

**Date:** 2026-07-13
**Status:** approved design, pre-implementation
**Repo:** `E:\NTU\fyp\wavelab\` (own git repo; the surrounding `fyp/` folder is not a git repo)

## 1. Purpose

A conda-isolated Python toolbox for nonlinear wave equations: define an equation
**once** as data, then solve it with **interchangeable methods** (explicit FD,
implicit FD, branching Monte Carlo) and compare the results — a small,
Mathematica-like workflow in plain Python.

It generalizes the ad-hoc `../fig6_study/` scripts and replaces the pattern of the
reference C++ repo (`../Nonlinear_Wave_simulations/`), which copy-pastes one MC
method into 14 folders, one per equation. Here the relationship is inverted:
equations are data, methods are reusable solvers.

Paper: Chan & Privault, *Probabilistic representation and classical solutions of
wave equations with complex polynomial nonlinearities* (2026) — `../m.pdf`.

### Scope (v1)

- Equations: `u_tt − c²Δu = f(u)` with polynomial `f(u) = Σ aₖuᵏ` (complex `aₖ`,
  integer `k ≥ 0`), complex `c`, complex-valued initial data, dimensions d = 1, 2, 3.
  This covers all 14 paper simulations (tanh solitons d=1..3, sine data d=1..2,
  `f=u²` closed form, etc.).
- Solvers: explicit FD (d=1 first, then d=2), implicit θ-scheme FD (d=1),
  branching MC (d=1 first, then d=2,3) with two backends (pure Python reference,
  numba fast path).
- Experiment utilities: method comparison, error vs exact solution, blow-up scans,
  linear mode-amplification analysis, MC variance profiling, house plotting style.

### Non-goals (v1)

- Other PDE families (heat, Schrödinger) — the abstraction is not designed for them.
- Non-polynomial nonlinearities, forcing terms, non-Dirichlet BCs.
- Calling the C++ or Mathematica reference code — those remain external cross-checks only.
- GUI / CLI / notebook framework — plain Python scripts are the entry point.

## 2. Architecture

Approach: **light protocol style** — "equations are data, solvers are classes,
`Solution` is the common currency."

```python
eq = WaveEquation(dim=1, c=1j, f={1: -1, 3: 1}, phi=..., psi=..., domain=((0, 1),))
sol_fd = ExplicitFD(N=101, dt=0.002).solve(eq, times=[0.1, 0.2, 0.3, 0.4])
sol_mc = BranchingMC(n=40_000, lam=0.25, backend="numba").solve(eq, times=[...], points=xs)
compare(sol_fd, sol_mc).plot("fig6.png")
```

Every solver implements `solve(eq, times, points=None) -> Solution`. The
experiments layer consumes only `Solution` objects and never inspects solver
internals. Adding a new method (e.g. a spectral solver later) = adding one class.

### Repo layout

```
wavelab/
├── environment.yml          # conda env "wavelab": python=3.12, numpy, matplotlib, numba, pytest
├── pyproject.toml           # pip install -e .
├── README.md
├── wavelab/
│   ├── __init__.py          # re-exports: WaveEquation, ExplicitFD, ImplicitFD, BranchingMC, compare
│   ├── equation.py          # WaveEquation dataclass + validation
│   ├── solution.py          # Solution result object
│   ├── library.py           # ready-made paper equations (with exact solutions)
│   ├── solvers/
│   │   ├── base.py          # Solver protocol
│   │   ├── fd_explicit.py   # leapfrog (d=1, then d=2 five-point)
│   │   ├── fd_implicit.py   # θ-scheme + Newton (paper Fig 7 counterpart)
│   │   └── mc/
│   │       ├── __init__.py  # BranchingMC — dispatches on backend
│   │       ├── reference.py # pure-Python recursion (readable ground truth)
│   │       └── fast.py      # numba njit + prange (paper-scale runs)
│   └── experiments/
│       ├── compare.py       # comparison tables / side-by-side plots
│       ├── blowup.py        # blowup_scan, mode_amplification
│       └── plotting.py      # house style
├── examples/
│   └── fig6_side_by_side.py # today's figure via the library (≈15 lines)
├── tests/                   # pytest, see §6
└── docs/superpowers/specs/  # this document
```

## 3. Core components

### 3.1 `WaveEquation` (equation.py)

Immutable dataclass; one definition serves both solver families.

```python
@dataclass(frozen=True)
class WaveEquation:
    dim: int                        # 1 / 2 / 3
    c: complex                      # wave speed: 1, 1j, ...
    f: Mapping[int, complex]        # nonlinearity {k: a_k}, f(u)=Σ aₖuᵏ, k ≥ 0 allowed
    phi: Callable                   # u(z,0); z complex (d=1) or complex vector (d>1)
    psi: Callable                   # ∂ₜu(z,0)
    domain: tuple | None = None     # per-axis intervals, e.g. ((0,1),); FD requires, MC ignores
    bc: str = "dirichlet"
    exact: Callable | None = None   # analytical u(z,t) if known → enables error tables
    name: str = ""
```

- Constructor validation (raise `ValueError` early): powers are non-negative
  integers; `phi`/`psi` accept a complex test argument of the right arity; `dim`
  matches `domain` length when given.
- `f_callable()` builds `u ↦ Σ aₖuᵏ` for the FD solvers from the same coefficient
  dict the MC uses for branching — the definition never forks.
- `domain`/`bc` live on the equation (it stays a complete problem statement);
  FD requires them, MC ignores them (whole-space Cauchy representation).

### 3.2 `Solution` (solution.py)

```python
@dataclass
class Solution:
    eq: WaveEquation
    solver: str                 # "explicit_fd" / "implicit_fd" / "branching_mc"
    params: dict                # snapshot of solver parameters (reproducibility)
    times: np.ndarray           # (T,)
    points: np.ndarray          # (P,) complex or (P, dim) complex
    u: np.ndarray               # (T, P) complex128; NaN at/after FD blow-up
    meta: dict                  # MC: stderr (T,P), n, lam, seed, backend
                                # FD: blowup_time, dt, N
```

Each solution reports on **its own point set** (FD on its grid, MC on the points
you asked for). Comparison and error tables evaluate per-solution — **no
interpolation**, so no extra error source.

### 3.3 Solvers (solvers/)

Protocol: `solve(eq, times, points=None) -> Solution`.

**`ExplicitFD(N, dt)`** — leapfrog in t, central 2nd differences in x, Dirichlet BC.
Builds its own grid from `eq.domain` (ignores `points`). Detects non-finite values
each step → records `meta["blowup_time"]`, fills later snapshots with NaN, issues a
warning — **blow-up is data, not an exception** (it is the ill-posedness
measurement). d=1 first; d=2 five-point stencil later.

**`ImplicitFD(N, dt, theta=0.5)`** — θ-scheme; each step solves the nonlinear system
by Newton iteration. d=1 only. **Not the Fig-7 counterpart** — see §3.3a.

**`LinearlyImplicitFD(N, dt)`** — the paper's *named* method (Mathematica's
`LinearlyImplicitEuler`): linear part implicit, nonlinear term explicit, one linear
solve per step. d=1 only. Blows up on a fixed grid — see §3.3a.

**`RegularizedFD(N, dt, k_max)`** — explicit leapfrog + spectral low-pass filter
(keep only Dirichlet sine modes `k ≤ k_max`). **This is the working Fig-7 analogue.**
d=1 only.

### 3.3a Finding (2026-07-13): implicit does NOT cure ill-posedness

The plan originally assumed "implicit FD = more stable" (paper Fig 7). Investigation
showed this is *half* true, and the missing half is the interesting part.

The paper's own sentence is the key: its Fig 7 is produced with Mathematica's
`LinearlyImplicitEuler`, and it says the implicit scheme "is more stable but exhibits
a loss of accuracy … **due to loss of energy conservation**." Stability is *bought
with dissipation*, not earned by implicitness. Consequences, all verified numerically:

| scheme | behaviour on the c=i problem (N=101, dt=0.002) | why |
|---|---|---|
| `ExplicitFD` | NaN at t=0.232 | consistent → must amplify the growing modes |
| `ImplicitFD` (θ=0.5) | **finite garbage** (26 at t=0.2, 1557 at t=0.3) | energy-**conserving**: roots satisfy `g₊g₋ = 1`, so one root is always outside the unit circle. Fails *silently* — worse than explicit, which at least reports NaN. |
| `LinearlyImplicitFD` | blows up at every dt tried | amplification `g± = 1/(1∓dt√ω)` has a **pole at `dt√ω = 1`**; ω spans 8.9→40000 across modes, so some mode always sits near it |
| `RegularizedFD` (k_max=12) | **stable to t=0.7, smooth, ~0.15 error** | modes above the cut-off are removed, so their growth can't be excited |

Root cause: the true solution's mode `k` grows like `exp(t·√((kπ)²−1)) ≈ exp(kπt)`, so
the grid-scale mode grows like `exp(2t/dx)` — unbounded as `dx→0`. **Any consistent
time-marching scheme must reproduce that growth**, hence must blow up once round-off
(1e-16) in those modes is amplified to O(1). A scheme only *looks* stable by damping
modes it should be growing — i.e. by solving a different, regularized problem.

`RegularizedFD` makes that bargain explicit and measurable. The cut-off is the
regularization, and keeping more modes recovers the ill-posedness you suppressed:

| `k_max` | 12 | 20 | 30 |
|---|---|---|---|
| blow-up time | none (survives t=0.7) | 0.64 | 0.438 |

This is the same fingerprint as "refine the grid → blow up sooner", now expressed in
the variable that actually causes it. **Branching MC needs none of this** — it never
marches a coupled state forward, so there is no mode to amplify. That contrast is the
sharpened thesis of the Figure-6 study.

**`BranchingMC(lam=0.25, n=10_000, q=None, backend="python", seed=None, workers=1)`**
— pointwise estimator of `u(z,t) = E[H]` from the paper's probabilistic
representation. Key properties:

- `points` may be **arbitrary complex points** (off-grid, off the real axis) — a
  capability FD does not have.
- `q` = offspring distribution over the support of `f` (free parameter; changes
  variance, not the mean). Default: uniform on the powers present, i.e. the paper's
  q₁ = q₃ = ½ for `f = −u + u³`. Importance weights `aₖ/qₖ`, `e^{λτ}`, `τ/λ` as in
  the paper; `λ` is a free parameter too.
- Tree ingredients: exponential clock `Exp(rate λ)`; spatial mark on the light cone
  (d=1: `z ± cτ(2p−1)`, i.e. uniform on the cone section; d=2,3: sampling per the
  paper's d-dimensional kernels, ported from C++ Simulations 4–10 as the reference);
  splitting into k children for the `uᵏ` term (k=0 → leaf with weight `a₀/q₀`).
- `meta["stderr"]` per point, always. Warn when relative stderr exceeds a threshold
  (default 20%) — signals `t` approaching the integrability window's edge.
- Backends:
  - `reference.py` — pure-Python recursion (current `fig6_study` code, cleaned up).
    Ground truth for correctness; fine for ~1e4–1e5 samples.
  - `fast.py` — numba `@njit`: recursion converted to an explicit work stack,
    `prange` over samples, per-thread RNG seeded from the user seed, complex128
    throughout. Target: paper-scale runs (101 points × 1e6–1e7 samples) on a laptop.
  - Both expose identical semantics; cross-validated by tests. If numba is missing,
    `backend="numba"` raises a clear error naming `conda install numba`;
    the reference backend never depends on it.

### 3.4 Equation library (library.py)

All paper simulations as ready-made `WaveEquation` instances with `exact` filled in
where known (from `../Nonlinear_Wave_simulations/README.md`):

| Name | Case | Exact solution |
|---|---|---|
| `SIM01_QUADRATIC` | d=1, c=1, f=u², φ=6z⁻² | `6/(z+√2 t)²` |
| `SOLITON_1D/2D/3D` | f=−u+u³ tanh data (Sims 4/6/9) | tanh closed forms |
| `SINE_CI_2D`, `SINE_C1_2D` | d=2 sine data, c=i / c=1 (Sims 7/10) | — |
| `SINE_CI_1D` | d=1, c=i, sine data (Sims 11–14; **the Figure-6 problem**) | — |

The library doubles as the test fixture set and as the FYP's casebook.

## 4. Experiments layer (experiments/)

Consumes `Solution` objects only:

- `compare(*solutions)` → `.table()` (values at shared times — the FD/MC agreement
  table) and `.plot(path)` (side-by-side figure; today's fig6 becomes ≈15 lines in
  `examples/`).
- `error_vs_exact(sol)` → sup / L² errors when `eq.exact` exists; for MC also
  error÷stderr (statistical consistency check).
- `blowup_scan(eq, make_solver, Ns, dts)` → blow-up time vs grid-spacing table
  (the ill-posedness fingerprint). Returns plain rows (list of dicts) + pretty
  printer; no pandas dependency.
- `mode_amplification(eq, N, dt)` → per-step linear amplification spectrum,
  generalized from `fig6_study/explicit_fd_1d.py` to any `c` and the linear part of `f`.
- `variance_profile(eq, mc, times)` → MC stderr growth vs t; locates MC's own
  short-time limit (CLAUDE.md experiment plan item 4).
- `plotting.py` → shared style (colors, dpi, titles) so every figure in the FYP
  report looks consistent.

## 5. Error handling

- **Blow-up is data**: FD never raises on blow-up; NaN + `meta["blowup_time"]` + warning.
- **MC reports uncertainty honestly**: stderr always present; relative-stderr
  warning near the integrability limit; `seed` explicit for reproducibility.
- **Fail fast at construction**: invalid equation definitions (negative powers,
  FD without `domain`, φ rejecting complex input) raise `ValueError` at
  `WaveEquation(...)` or `solve(...)` entry with actionable messages.
- Missing numba → clear ImportError with install hint; reference backend unaffected.

## 6. Testing (pytest)

| Test file | Verifies |
|---|---|
| `test_closed_forms.py` | MC vs closed forms: SIM01 `6/(z+√2t)²`, SOLITON_1D (2D/3D marked `slow`) — error < 3×stderr |
| `test_backends_agree.py` | python vs numba means agree within combined stderr |
| `test_fd_wellposed.py` | c=1 control: refinement **reduces** error (FD core is correct) |
| `test_illposed_signature.py` | c=i: refinement makes blow-up **earlier** — regression-locks t≈0.44 (N=51) / 0.23 (N=101) / 0.13 (N=201) at dt=0.002 |
| `test_fig6_regression.py` | library-based fig6 reproduces the verified numbers below |

**Verified baseline (2026-07-13, `fig6_study` sandbox, seeds fixed):**
FD (N=51, dt=0.002) max|u| = 0.948 / 0.991 / 1.137 / 20.14 at t = 0.1/0.2/0.3/0.4;
MC (n=40 000, λ=0.25) u(0.5) = 0.948 / 0.990 / 1.135 / 1.412 at the same times,
max|Im u| ≤ 0.006. FD and MC agree to 3 digits at t ≤ 0.3; FD blown up at t=0.4,
MC still smooth. The regression test is the acceptance criterion for the refactor:
same physics, new home.

## 7. Environment & migration

- `environment.yml`: env name `wavelab`; python=3.12, numpy, matplotlib, numba,
  pytest; then `pip install -e .`.
  (Note: base Anaconda is at `C:\Users\LiaoTianrui\anaconda3\`; `python` is not on PATH.)
- `wavelab/` is its own git repo (`git init` inside; `fyp/` root stays non-git).
- `../fig6_study/` is kept untouched as the historical sandbox; once
  `examples/fig6_side_by_side.py` + `test_fig6_regression.py` prove equivalence,
  it is retired naturally (no deletion required).
- Root `../CLAUDE.md` gets a short wavelab pointer (implementation phase).

## 8. Milestones

1. **M1 — walking skeleton:** equation.py, solution.py, ExplicitFD (1D),
   BranchingMC reference backend (1D), `examples/fig6_side_by_side.py` reproducing
   today's figure, `test_fig6_regression.py` + `test_illposed_signature.py` green.
2. **M2 — speed & validation:** numba backend, `test_closed_forms.py`,
   `test_backends_agree.py`.
3. **M3 — full method set:** ✅ done 2026-07-13. ImplicitFD (1D), **plus
   LinearlyImplicitFD and RegularizedFD** (added once §3.3a showed the θ-scheme
   cannot be the Fig-7 counterpart), MC d=2/3, ExplicitFD d=2, `library.py` with
   all 11 paper equations.
4. **M4 — experiment polish:** ✅ done 2026-07-13. variance_profile, generalized
   mode_amplification, blowup_scan, 2-D compare guard, `illposedness_report.py`.

**Status: M1–M4 complete, 101 tests green.** Remaining ideas, none started:
numba backend for d≥2 (needed only for paper-scale 2-D runs), implicit FD in d≥2,
2-D field visualisation, Deep Galerkin comparison (`../wave_equation/`).

Each milestone ends runnable with green tests. Implementation is planned for a
separate session (Opus) working from this spec plus a written implementation plan.
