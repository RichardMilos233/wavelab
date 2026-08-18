# Experiments layer — analysis tools

Summarizes `wavelab/experiments/`. Everything consumes `Solution` objects only
(never solver internals), so any new solver gets these for free.

## compare(*solutions) → Comparison — `compare.py`

- `.rows(probe_points=None)` → list of dicts {t, x, <solver>: value}. Probe defaults
  to the point set of the solution with fewest points; each solution answers with its
  own NEAREST point — **no interpolation** (deliberate: no extra error source).
  Shared times only.
- `.table(probe_points=None)` → printable string (real parts).
- `.plot(path=None)` → one panel per solution, shared times overlaid, blow-up
  annotated in the title. House style from `plotting.py` (DPI 140).
- `.rows()`/`.table()` work in **any dimension** (nearest point by Euclidean distance
  for d≥2; the table prints `(x, y)` tuples). Only `.plot()` is 1-D — use
  `surfaces()` for d=2 pictures.

## blowup_scan(eq, make_solver, Ns, dts, probe_time=0.5) — `blowup.py`

`make_solver: (N, dt) -> solver` — works for explicit/theta/regularized (anything
with `meta["blowup_time"]`). Probes a ladder of times so `max_abs_u` records how big
the solution got before dying. Returns rows {N, dt, blowup_time, max_abs_u};
`blowup_table(rows)` pretty-prints. Suppresses blow-up warnings internally.
Canonical result on SINE_CI_1D (dt=0.002): N=51→0.44, 101→0.232, 201→0.128 —
finer grid dies sooner = the ill-posedness fingerprint.

## mode_amplification(eq, N, dt) — `blowup.py`

Linearized per-step growth of each Fourier mode under leapfrog. Returns
{k, omega, growth}. Math: μₖ = (2−2cos(πk·dx))/dx² (eigenvalue of −∂ₓₓ on the grid),
ω = c²(−μₖ) + f'(0) with f'(0)=`eq.f.get(1, 0)`; roots of g²−(2+dt²ω)g+1=0;
growth = max|g±|. Roots always satisfy g₊g₋=1 (energy conservation) — the reason no
θ-scheme can stabilize the elliptic case. 1-D only. Canonical numbers (SINE_CI_1D,
N=101, dt=0.002): g(k=1)=1.006, g(k=99)=1.488.

## variance_profile(eq, mc, times, point) / variance_plot(rows, path) — `variance.py`

Runs the given `BranchingMC` at one point across `times`; rows
{t, u, stderr, rel_stderr}. Locates MC's own short-time wall (variance explosion —
noisy, not wrong). On SINE_CI_1D at n=20k: rel stderr 0.1% at t=0.1, ~6% at 0.8,
>100% by t=1.2. Suppresses the high-stderr warning (it IS the signal).

## surface(sol, i=0, shape=None, ...) / surfaces(sols, ...) — `plotting.py`

d=2 surface plots of Re u(x,y,t) — the paper's Figure 2 / Figure 8 layout.
`surfaces([...], shapes=[...], labels=[...], path=...)` draws them side by side.
Both FD solutions and MC solutions **on the default grid** carry `meta["shape"]`, so
`surfaces([mc, fd])` just works. Only when you passed `points=` explicitly does MC not
know the layout — then give `shape=(M, M)` (a square count is inferred if you don't).
NaNs are masked rather than plotted (an all-NaN panel still renders — the failure
IS the picture).
Used by `examples/fig8.py`; covered by `tests/test_surfaces.py`,
which asserts the `meta["shape"]` contract is READ (test_mc_highdim only asserts
it is written).
