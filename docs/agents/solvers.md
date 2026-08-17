# Solvers — API and behaviour

Summarizes `wavelab/solvers/`. All solvers implement
`solve(eq: WaveEquation, times, points=None) -> Solution` and return complex128
arrays of shape `(T, P)`. Requested `times` must be integer multiples of `dt`
(FD solvers), else `ValueError`.

## Quick chooser

| You want... | Use |
|---|---|
| The honest FD baseline (blows up on ill-posed problems, loudly) | `ExplicitFD` |
| A *stable* FD curve or surface on the ill-posed problem | `RegularizedFD` (the only FD that works there; d=1 and d=2) |
| The unbiased pointwise reference / complex evaluation points | `BranchingMC` |
| To demonstrate that implicit does NOT fix ill-posedness | `ImplicitFD`, `LinearlyImplicitFD` |

## ExplicitFD(N=101, dt=0.002) — `fd_explicit.py`

Leapfrog + central differences, Dirichlet. d=1 and d=2 (five-point stencil).
Ignores `points`; builds its grid from `eq.domain`. d=2: `Solution.points` is
`(N², 2)` (row-major meshgrid), `meta["shape"] = (N, N)` for reshaping.
`meta`: `blowup_time` (float|None), `dt`, `N`, `dx` (+ `dy`, `shape` in 2-D).
The per-step `clamp` re-imposing boundary zeros is load-bearing (φ(boundary) may be ≠0).

## ImplicitFD(N=101, dt=0.002, theta=0.5, newton_tol=1e-10, newton_maxiter=50) — `fd_implicit.py`

θ-scheme solved by dense Newton each step (`f_prime_callable` Jacobian). θ∈[0,0.5];
θ=0 reproduces ExplicitFD to round-off (tested). `meta` adds `theta`, `newton_iters`,
`newton_failed_steps`, `newton_first_failure_time`, `newton_max_residual`.
**Energy-conserving ⇒ cannot stabilize the ill-posed problem** — amplification roots
satisfy g₊g₋=1 (N=101: g*≈1.513/step, so round-off reaches O(1) in ~89 steps, t≈0.18).
On SINE_CI_1D it emits finite garbage of order 10³ for t ≳ 0.2 (truth 1.14) with
`blowup_time=None` — silent failure by design of the mathematics, not a bug.
The magnitude is machine-dependent (it *is* amplified round-off) — never assert a
specific value, assert `abs(centre − truth) > 100` as `tests/test_fd_implicit.py` does.
Two-stage failure: past t≈0.21 Newton also stops converging (exits on `newton_maxiter`,
residual 1e2–1e4 on ~half the steps), so the output is not even a solution of the
θ-scheme; the solver records that in `meta` and warns once. Keep it: it is the
counterexample.

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
stability).

**d=2** (paper §7.1/§7.3): the 2-D Dirichlet eigenfunctions are tensor products
sin(kπx)sin(mπy), so the cut-off is separable — `U -> P U P` on the interior, two
(N−2)³ matmuls per step, never an (N−2)⁴ matrix. It keeps a BOX in mode space
(k ≤ k_max AND m ≤ k_max). **Carry 1-D intuition across carefully:** mode (k,m) grows
at √((k²+m²)π²−1), so the fastest retained mode is (k_max, k_max) at √2·k_max·π —
a d=2 run with k_max=K is about as aggressive as d=1 with √2·K. On SINE_DEFOCUS_CI_2D
at t=0.5 there is a usable **window**: k_max ≤ 2 is under-resolved, 3–10 matches MC to
<0.15, ≥20 is contaminated. In d=1 only the upper end of that window is visible.
`meta` adds `dy`, `shape` in 2-D. d=3 not implemented.

## BranchingMC(lam=0.25, n=10_000, q=None, seed=None, N=21) — `mc/`

Pointwise unbiased estimator of u(z,t)=E[H] from the paper's branching-tree
representation. **`n` = samples per point; `N` = points PER AXIS of the default grid**
(mirroring ExplicitFD's `N`), so the default grid is `N**dim` points and `meta["shape"]
= (N,)*dim` is set for plotting. `n` and `N` differ only in case and mixing them up does
not raise — it just gives a very noisy or a very slow answer. Explicit `points=`
overrides `N` entirely (and then no `shape` is recorded — the layout is unknown).
**Cost is LINEAR in the number of points** (unlike FD, where one march yields the whole
field): the default grid is 21 / 441 / 9261 evaluations in d=1/2/3, so choose `N` with
that in mind. d=1,2,3; `points` may be arbitrary complex (off-axis works).
d≥2: `points` shape `(P, dim)`, `eq.grad_phi` required
(leaf carries y·∇φ(z+y)); d=2 mark is a disc `R=s√(1−(1−p)²)`, d=3 the sphere
`α=arccos(1−2p)`. `q` maps power→probability (default uniform on `eq.f` keys);
changes variance only — tested. `meta`: `stderr` (T,P), `n`, `lam`, `seed`,
warns when relative stderr >20%. Single implementation: `mc/reference.py`, a readable
recursion — deliberately not optimised (see gotchas: the numba backend was removed).
Cost is ~3 s for 1e6 samples at one point, ~4.5 min for 1e6 samples over 101 points.
Failure mode is VARIANCE, not instability: stderr explodes with t (rel stderr >100%
by t≈1.2 on SINE_CI_1D). Use `variance_profile` to locate the wall.
