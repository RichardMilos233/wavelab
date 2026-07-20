# Solvers — API and behaviour

Summarizes `wavelab/solvers/`. All solvers implement
`solve(eq: WaveEquation, times, points=None) -> Solution` and return complex128
arrays of shape `(T, P)`. Requested `times` must be integer multiples of `dt`
(FD solvers), else `ValueError`.

## Quick chooser

| You want... | Use |
|---|---|
| The honest FD baseline (blows up on ill-posed problems, loudly) | `ExplicitFD` |
| A *stable* FD curve on the ill-posed problem | `RegularizedFD` (the only FD that works there) |
| The unbiased pointwise reference / complex evaluation points | `BranchingMC` |
| Paper-scale MC (1e6+ samples, d=1) | `BranchingMC(backend="numba")` (needs njit φ/ψ) |
| To demonstrate that implicit does NOT fix ill-posedness | `ImplicitFD`, `LinearlyImplicitFD` |

## ExplicitFD(N=101, dt=0.002) — `fd_explicit.py`

Leapfrog + central differences, Dirichlet. d=1 and d=2 (five-point stencil).
Ignores `points`; builds its grid from `eq.domain`. d=2: `Solution.points` is
`(N², 2)` (row-major meshgrid), `meta["shape"] = (N, N)` for reshaping.
`meta`: `blowup_time` (float|None), `dt`, `N`, `dx` (+ `dy`, `shape` in 2-D).
The per-step `clamp` re-imposing boundary zeros is load-bearing (φ(boundary) may be ≠0).

## ImplicitFD(N=101, dt=0.002, theta=0.5, newton_tol=1e-10, newton_maxiter=50) — `fd_implicit.py`

θ-scheme solved by dense Newton each step (`f_prime_callable` Jacobian). θ∈[0,0.5];
θ=0 reproduces ExplicitFD to round-off (tested). `meta` adds `theta`, `newton_iters`.
**Energy-conserving ⇒ cannot stabilize the ill-posed problem** — amplification roots
satisfy g₊g₋=1. On SINE_CI_1D it emits finite garbage (1557 at t=0.3, truth 1.14)
with `blowup_time=None` — silent failure by design of the mathematics, not a bug.
Keep it: it is the counterexample.

## LinearlyImplicitFD(N=101, dt=0.01) — `fd_implicit_linear.py`

The paper's named Fig-7 method (Mathematica `LinearlyImplicitEuler`): first-order
system, linear part implicit (matrix inverted once), nonlinearity explicit, no Newton.
On a fixed grid it blows up at every dt: amplification `1/(1∓dt√ω)` has a pole at
`dt√ω=1` and ω spans ~8.9→4·10⁴. `stability_dt(N, domain)` gives the damping
threshold. Keep it: faithful implementation of the named method, and the pole is the
teaching point.

## RegularizedFD(N=101, dt=0.002, k_max=12) — `fd_regularized.py`

Explicit leapfrog + per-step projection onto Dirichlet sine modes k ≤ k_max
(`sine_lowpass(N, k_max)` — idempotent DST-I projector, tested). The working Fig-7
analogue: k_max=12 survives to t=0.7 with err ~0.15 vs MC; k_max=20 dies at 0.64;
k_max=30 at 0.438 (more modes kept ⇒ earlier death — the regularization IS the
stability). d=1 only.

## BranchingMC(lam=0.25, n=10_000, q=None, backend="python", seed=None, workers=1) — `mc/`

Pointwise unbiased estimator of u(z,t)=E[H] from the paper's branching-tree
representation. d=1,2,3 (python backend); `points` may be arbitrary complex
(off-axis works). d≥2: `points` shape `(P, dim)` required, `eq.grad_phi` required
(leaf carries y·∇φ(z+y)); d=2 mark is a disc `R=s√(1−(1−p)²)`, d=3 the sphere
`α=arccos(1−2p)`. `q` maps power→probability (default uniform on `eq.f` keys);
changes variance only — tested. `meta`: `stderr` (T,P), `n`, `lam`, `seed`,
`backend`; warns when relative stderr >20%.
- `backend="python"` — `mc/reference.py`, readable recursion, ground truth.
- `backend="numba"` — `mc/fast.py`, d=1 only, iterative explicit stack (H is a
  product of per-node weights), `prange` over points, per-point seeding
  (deterministic across thread counts). Requires φ/ψ to be `numba.njit`-compiled,
  else ValueError. ~1e6 samples in ~4 s.
Failure mode is VARIANCE, not instability: stderr explodes with t (rel stderr >100%
by t≈1.2 on SINE_CI_1D). Use `variance_profile` to locate the wall.
