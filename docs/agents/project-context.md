# Project context — why this repo exists

Read this if you are an agent working in a fresh clone (e.g. the author's MacBook) and
need to know what the code is *for*. It is the FYP framing that used to live in a
`CLAUDE.md` one directory above the repo and does not travel with a clone.

## The project

Final-year project reproducing and extending **Chan & Privault, "Probabilistic
representation and classical solutions of wave equations with complex polynomial
nonlinearities" (2026)** (the PDF is `../m.pdf`, not in this repo).

The paper writes the solution of `u_tt − c²Δu = f(u)`, `f(u) = Σ aₖuᵏ`, as the
expectation of a random branching-tree functional, `u = E[H]`, and estimates it by Monte
Carlo. Three tree ingredients: an **exponential clock** (from the source's time
integral), a **spatial mark on the light cone** (from the wave kernel), and **splitting
into k children** (from the power `uᵏ`), carrying importance weights `aₖ/qₖ`, `e^{λτ}`,
`τ/λ`. `λ` and `qₖ` are **free parameters** — they change the variance, not the mean.
The representation is only valid for **short time** (integrability / possible blow-up),
which is what makes the variance question interesting.

## The research question

**Why does explicit finite differencing develop unstable edge oscillations (paper
Figure 6) while branching Monte Carlo stays smooth and usable?**

The test problem is the paper's §7.2 "sine data" (C++ Simulations 11–14), which is
`library.SINE_CI_1D` here: `c = i`, d = 1, `f(u) = −u + u³`, `φ = sin(πx)`,
`ψ = −sin(πx)`, `x ∈ [0,1]`, Dirichlet, `λ = 0.25`, look at `t ∈ [0, 0.5]`. Because
`c = i` the spatial term flips sign, so this is an **elliptic Cauchy problem** —
Hadamard ill-posed: the linearized mode `e^{ikx}` obeys `u_tt = (k²−1)u`, so high
frequencies grow fastest.

The answer the repo establishes is sharper than "explicit is unstable" — see spec §3.3a
and `README.md`: *any* consistent time-marching scheme must blow up, because the true
solution's grid-scale mode already grows like `exp(2t/dx)`. Implicit does not help;
only regularization does. Branching MC evaluates `u(x,t)` pointwise from the mild
representation, never marches a coupled state, and so has nothing to amplify — its own
limit is variance instead.

## What "done" looks like

A written FYP report backed by reproducible figures. The repo is the evidence layer:
`examples/fig6.py` is the Figure-6 comparison (§7.2, d=1) and `examples/fig8.py` the
Figure-8 one (§7.3, d=2); `tests/` locks every number the report quotes; and
`docs/tutorial/` (Chinese) carries the derivations for the report's method chapter.

One exception to "reproducible", and it matters whenever `ImplicitFD` is quoted: the
huge finite values it emits past t≈0.18 are amplified round-off, not measurements. They
differ between machines and their sign is not fixed — see `gotchas.md`. Quote the time
at which its output stops being meaningful, never the value. Everything else in this
repo is machine-independent.

## Conventions inherited from the FYP

- Build clean, instrumentable Python rather than porting the C++ — but keep results
  numerically comparable to the paper's setup (that is what the locked regression
  numbers in `tests/` guarantee).
- Blow-up is a measurement, not a crash.
- Two of the external reference implementations contain bugs; read
  `gotchas.md` before treating any of them as ground truth.
