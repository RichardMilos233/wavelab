# wavelab

Solver toolbox for nonlinear wave equations `u_tt − c²Δu = Σ aₖuᵏ` (complex `c`, complex data,
d = 1, 2, 3). Define an equation once as data, then solve it with interchangeable methods and
compare them. Design: `docs/superpowers/specs/2026-07-13-wavelab-design.md`.

## Setup
    conda env create -f environment.yml
    conda activate wavelab
    pip install -e .
    pytest -m "not slow"

## Use
```python
from wavelab import library, ExplicitFD, BranchingMC, compare

eq = library.SINE_CI_1D                      # the paper's Figure-6 problem (c=i)
times = [0.1, 0.2, 0.3, 0.4]
fd = ExplicitFD(N=51, dt=0.002).solve(eq, times)
mc = BranchingMC(lam=0.25, n=40_000, seed=0).solve(eq, times)
compare(fd, mc).plot("fig6.png")             # FD explodes; MC stays smooth
```

## What's in it
- **Solvers** (all share `solve(eq, times, points=None) -> Solution`):
  - `ExplicitFD` (d=1,2) — leapfrog. Blows up on the ill-posed problem, loudly.
  - `ImplicitFD` (d=1) — θ-scheme + Newton. **Does not cure ill-posedness** (see below).
  - `LinearlyImplicitFD` (d=1) — the paper's named method; blows up on a fixed grid.
  - `RegularizedFD` (d=1) — leapfrog + spectral cut-off. **The one that actually works.**
  - `BranchingMC` (d=1,2,3) — pointwise `E[H]`; `backend="python"` or `"numba"` (numba is d=1).
- **Equations**: `wavelab.library` — all 11 of the paper's simulations, 8 with closed forms.
- **Experiments**: `compare` · `blowup_scan` · `mode_amplification` · `variance_profile`

## The headline result

On the elliptic (`c=i`) problem, mode `k` of the **true** solution grows like `exp(kπt)`, so the
grid-scale mode grows like `exp(2t/dx)` — unbounded as `dx→0`. **Any consistent time-marching
scheme must reproduce that growth, hence must blow up.** Measured at N=101, dt=0.002:

| solver | result at t=0.3 | why |
|---|---|---|
| `ExplicitFD` | NaN from t=0.232 | consistent → must amplify |
| `ImplicitFD` (θ=0.5) | **1557** (truth: 1.14) | energy-**conserving**: roots satisfy `g₊g₋=1`, so a growing root always exists. Fails **silently** — arguably worse than the explicit scheme, which at least reports NaN. |
| `LinearlyImplicitFD` | blows up at every dt | `g± = 1/(1∓dt√ω)` has a **pole** at `dt√ω=1` |
| `RegularizedFD` (k_max=12) | 1.137 ✓, stable to t=0.7 | the runaway modes are simply removed |
| `BranchingMC` | 1.137 ✓ | never marches a coupled state — nothing to amplify |

"Go implicit" is **not** the fix. The paper says so itself: its Figure 7 is "more stable but
exhibits a loss of accuracy … **due to loss of energy conservation**". Stability is *bought* with
dissipation. `RegularizedFD` makes that bargain explicit — and keeping more modes brings the
blow-up straight back (k_max=12 survives to 0.7; 20 → dies at 0.64; 30 → dies at 0.438).

Branching MC needs no regularization at all. It has its own limit, though — **variance**, not
instability: by t≈1.2 its relative standard error exceeds 100% and the estimator is pure noise.
FD dies of ill-posedness; MC dies of variance. See `examples/illposedness_report.py`.

## Examples
- `examples/fig6_side_by_side.py` — the Figure-6 FD-vs-MC figure
- `examples/illposedness_report.py` — the full argument in six sections

## Notes
- Blow-up is **data**, not an error: solvers record `meta["blowup_time"]` and return NaN after it.
- MC always reports `meta["stderr"]`; pass `seed=` for reproducibility.
- `numpy<2.4` and `matplotlib<3.11` are pinned — numpy 2.4.x breaks matplotlib's compiled
  ABI on Windows and is unsupported by numba 0.65.x.
- The reference C++ `Simulation_07` (d=2) sets `a₃ = −1` though its README says `f = −u + u³`
  (`a₃ = +1`). wavelab derives coefficients from `eq.f`, so d≥2 is validated against the closed
  forms, not against that output.
