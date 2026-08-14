# wavelab — agent guide (level 1: always loaded)

Python toolbox for nonlinear wave equations `u_tt − c²Δu = Σ aₖuᵏ` (complex `c`,
complex data, d = 1,2,3). Equations are immutable data (`WaveEquation`), solvers are
interchangeable classes sharing `solve(eq, times, points=None) -> Solution`, the
experiments layer consumes only `Solution`. Status: **M1–M4 complete, 99 tests green.**

## Run

Setup once: `conda env create -f environment.yml && conda activate wavelab && pip install -e .`

- Tests: `pytest -q -m "not slow"` (~9 s); with slow: `pytest -q` (~14 s)
- Examples: `python examples/fig6_side_by_side.py`, `python examples/illposedness_report.py`
- Figures: `python docs/tutorial/figures/make_figures.py`

Platform note: on the author's Windows box `python`/`conda` are not on PATH — use the
absolute interpreter path instead (see `docs/agents/gotchas.md`). On macOS/Linux with
the env activated, the plain commands above work.

## External references (NOT in this repo)

The paper being reproduced and the reference implementations live one directory up in
the author's FYP workspace and are **not** cloned with this repo: `../m.pdf` (the
paper), `../Nonlinear_Wave_simulations/` (C++ reference), `../wave_equation/`
(Deep Galerkin + FD notebooks), `../fig6_study/` (historical sandbox superseded by
this repo). Citations to them in docs/tests are provenance notes — everything needed
to run, test and understand wavelab is self-contained here. Do not try to read them
if they are absent; do not re-derive results from them without reading
`docs/agents/gotchas.md` first (two of them contain bugs).

## Hard rules

- Regression numbers (spec §6 / `tests/test_fig6_regression.py`) are LOCKED —
  never edit an expected number to make a test pass; investigate instead.
- Blow-up is DATA, not an exception: solvers set `meta["blowup_time"]`, NaN after, warn.
- Deps stay minimal: numpy `<2.4` (pinned — see gotchas), matplotlib `<3.11`, pytest.
  No scipy, no pandas, no numba — this is a demonstration repo, clarity over speed.
- All solution arrays complex128; MC always reports `meta["stderr"]`; seeds explicit.

## Deeper docs (level 2 — read on demand, don't preload)

| When you need... | Read |
|---|---|
| Why this repo exists — the paper, the research question, the deliverable | `docs/agents/project-context.md` |
| Solver APIs, params, meta keys, failure modes | `docs/agents/solvers.md` |
| WaveEquation fields / the 11 library equations | `docs/agents/equations.md` |
| compare / blowup_scan / mode_amplification / variance_profile | `docs/agents/experiments.md` |
| Environment pins, reference-code bugs, sharp edges | `docs/agents/gotchas.md` |
| The headline research finding (implicit ≠ stable) | spec §3.3a: `docs/superpowers/specs/2026-07-13-wavelab-design.md` |
| Human tutorial (Chinese, full derivations) | `docs/tutorial/00-overview.md` |

Design spec and implementation plans live in `docs/superpowers/`. The paper being
reproduced is `../m.pdf`; its C++ reference code is `../Nonlinear_Wave_simulations/`
(two known bugs there — see gotchas before trusting it).
