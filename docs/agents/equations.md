# Equations — WaveEquation and the library

Summarizes `wavelab/equation.py` and `wavelab/library.py`.

## WaveEquation (frozen dataclass)

```python
WaveEquation(dim, c, f, phi, psi, grad_phi=None, domain=None, bc="dirichlet",
             exact=None, name="")
```

- `f: Mapping[int, complex]` — {k: aₖ}, f(u)=Σaₖuᵏ, k≥0 ints. One dict serves both
  families: FD builds `f_callable()` / `f_prime_callable()` from it, MC uses it for
  branch probabilities and weights. Definitions never fork.
- `phi/psi` take a complex scalar (d=1) or a length-d complex vector (d≥2); must
  accept complex input (validated at construction — `math.sin` fails, `cmath.sin` works).
- `grad_phi` (d≥2, MC only): returns d partials; arity validated.
- `domain`/`bc`: FD requires them; MC ignores (whole-space Cauchy representation).
- `exact(z, t)`: enables error tables and golden tests.
- Validation is fail-fast at construction: bad powers, dim/domain mismatch,
  non-complex-capable φ/ψ, wrong grad_phi arity → ValueError.

## library.py — the paper's 13 cases

`library.ALL: dict[name → WaveEquation]`; `library.WITH_EXACT` = the 8 with closed
forms. Transcription guard test asserts `exact(z,0) == phi(z)` for all of them.

| Name | d | c | f | exact? | role |
|---|---|---|---|---|---|
| SIM01_QUADRATIC_1D | 1 | 1 | u² | ✓ 6/(z+√2t)² | primary MC golden test |
| SIM02_CUBIC_1D | 1 | 1 | u³ | ✓ | golden test |
| SIM03_MIXED_1D | 1 | 1 | 1.5u²+2u³ | ✓ | multi-power f |
| SIM05_QUADRATIC_2D | 2 | 1 | u² | ✓ | d=2 validation |
| SIM08_QUADRATIC_3D | 3 | 1 | u² | ✓ | d=3 validation |
| SOLITON_1D/2D/3D | 1–3 | i | −u+u³ | ✓ tanh | elliptic with exact |
| SINE_CI_1D | 1 | i | −u+u³ | — | **the Figure-6/7 problem** |
| SINE_CI_2D | 2 | i | −u+u³ | — | 2-D ill-posed |
| SINE_C1_2D | 2 | 1 | −u+u³ | — | well-posed control |
| SINE_DEFOCUS_C1_2D | 2 | 1 | **−u−u³** | — | **paper §7.1** (defocusing Klein–Gordon) |
| SINE_DEFOCUS_CI_2D | 2 | i | **−u−u³** | — | **paper §7.3** (defocusing elliptic, Fig 8) |

**Watch the cubic's sign.** §7.1 and §7.3 have a₃ = **−1**; the soliton / SINE_CI
family has a₃ = **+1**. They are different equations, and the sign changes the
FAILURE MODE, not just the answer: −u³ opposes growth, so on §7.3 amplified
round-off *saturates* and explicit FD returns bounded garbage with
`blowup_time=None` instead of NaN — exactly the paper's Fig 8b. Never assert a
blow-up time on a defocusing problem; assert accuracy instead.

Evaluation points used by the paper/tests: SIM01 z=3, SIM02 z=6 (λ=0.25); SIM03 z=9,
SIM05 (4,4), SIM08 (4,4,4) (λ=1); solitons z=−1 per axis (λ=0.25).
