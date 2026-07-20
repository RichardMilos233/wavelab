# Gotchas — sharp edges, pins, and reference-code bugs

## Environment

- `python`/`conda` are NOT on PATH. Interpreter:
  `C:\Users\LiaoTianrui\anaconda3\envs\wavelab\python.exe`;
  conda: `C:\Users\LiaoTianrui\anaconda3\Scripts\conda.exe`.
- **numpy must stay <2.4**: numpy 2.4.x causes a NATIVE crash (0xc06d007f) inside
  matplotlib's compiled transforms on Windows, and numba 0.65.x doesn't support it.
  matplotlib pinned <3.11 for the same ABI reason. Working combo: numpy 2.3.5 +
  matplotlib 3.10.8. Both pinned in `environment.yml` — do not "upgrade to fix".
- PowerShell 5.1: no `&&`; heredocs via Bash tool are safer for multi-line commit
  messages; regex in Bash heredocs gets mangled — use the Write tool for scripts.

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
