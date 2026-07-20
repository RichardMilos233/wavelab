# wavelab — agent guide (level 1: always loaded)

Python toolbox for nonlinear wave equations `u_tt − c²Δu = Σ aₖuᵏ` (complex `c`,
complex data, d = 1,2,3). Equations are immutable data (`WaveEquation`), solvers are
interchangeable classes sharing `solve(eq, times, points=None) -> Solution`, the
experiments layer consumes only `Solution`. Status: **M1–M4 complete, 101 tests green.**

## Run

- Interpreter (NOT on PATH): `C:\Users\LiaoTianrui\anaconda3\envs\wavelab\python.exe`
  (conda env `wavelab`; create with `conda env create -f environment.yml` + `pip install -e .`)
- Tests: `& $py -m pytest -q -m "not slow"` (~9 s); add slow: `& $py -m pytest -q` (~14 s)
- Examples: `& $py examples\fig6_side_by_side.py`, `& $py examples\illposedness_report.py`

## Hard rules

- Regression numbers (spec §6 / `tests/test_fig6_regression.py`) are LOCKED —
  never edit an expected number to make a test pass; investigate instead.
- Blow-up is DATA, not an exception: solvers set `meta["blowup_time"]`, NaN after, warn.
- Deps stay minimal: numpy `<2.4` (pinned — see gotchas), matplotlib `<3.11`, numba,
  pytest. No scipy, no pandas.
- All solution arrays complex128; MC always reports `meta["stderr"]`; seeds explicit.

## Deeper docs (level 2 — read on demand, don't preload)

| When you need... | Read |
|---|---|
| Solver APIs, params, meta keys, failure modes | `docs/agents/solvers.md` |
| WaveEquation fields / the 11 library equations | `docs/agents/equations.md` |
| compare / blowup_scan / mode_amplification / variance_profile | `docs/agents/experiments.md` |
| Environment pins, reference-code bugs, sharp edges | `docs/agents/gotchas.md` |
| The headline research finding (implicit ≠ stable) | spec §3.3a: `docs/superpowers/specs/2026-07-13-wavelab-design.md` |
| Human tutorial (Chinese, full derivations) | `docs/tutorial/00-overview.md` |

Design spec and implementation plans live in `docs/superpowers/`. The paper being
reproduced is `../m.pdf`; its C++ reference code is `../Nonlinear_Wave_simulations/`
(two known bugs there — see gotchas before trusting it).
