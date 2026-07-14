"""The paper's simulations as ready-made WaveEquation objects (spec §3.4).

Transcribed from ../Nonlinear_Wave_simulations/README.md. Doubles as the FYP
casebook and as the test fixture set: every equation with a closed form is a
golden test for the solvers.

Convention: the paper writes u_tt + Delta u = -u + u^3 for the soliton family,
i.e. u_tt - c^2 Delta u = f(u) with c^2 = -1 -> c = i, f = -u + u^3.
"""
import cmath
import math

from wavelab.equation import WaveEquation

S2 = math.sqrt(2.0)
S3 = math.sqrt(3.0)
S6 = math.sqrt(6.0)

# --------------------------------------------------------------- polynomial f, c=1
SIM01_QUADRATIC_1D = WaveEquation(
    dim=1, c=1, f={2: 1},
    phi=lambda z: 6 / z**2,
    psi=lambda z: -12 * S2 / z**3,
    exact=lambda z, t: 6 / (z + S2 * t)**2,
    name="SIM01_QUADRATIC_1D")

SIM02_CUBIC_1D = WaveEquation(
    dim=1, c=1, f={3: 1},
    phi=lambda z: S2 / z,
    psi=lambda z: -2 / z**2,
    exact=lambda z, t: S2 / (z + S2 * t),
    name="SIM02_CUBIC_1D")

SIM03_MIXED_1D = WaveEquation(
    dim=1, c=1, f={2: 1.5, 3: 2},
    phi=lambda z: 4 / (z**2 - 4),
    psi=lambda z: -8 * S2 * z / (z**2 - 4)**2,
    exact=lambda z, t: 4 / ((z + S2 * t)**2 - 4),
    name="SIM03_MIXED_1D")

SIM05_QUADRATIC_2D = WaveEquation(
    dim=2, c=1, f={2: 1},
    phi=lambda z: 6 / (z[0] + z[1])**2,
    psi=lambda z: -12 * S3 / (z[0] + z[1])**3,
    grad_phi=lambda z: (-12 / (z[0] + z[1])**3, -12 / (z[0] + z[1])**3),
    exact=lambda z, t: 6 / (z[0] + z[1] + S3 * t)**2,
    name="SIM05_QUADRATIC_2D")

SIM08_QUADRATIC_3D = WaveEquation(
    dim=3, c=1, f={2: 1},
    phi=lambda z: 6 / (z[0] + z[1] + z[2])**2,
    psi=lambda z: -24 / (z[0] + z[1] + z[2])**3,
    grad_phi=lambda z: tuple(-12 / (z[0] + z[1] + z[2])**3 for _ in range(3)),
    exact=lambda z, t: 6 / (z[0] + z[1] + z[2] + 2 * t)**2,
    name="SIM08_QUADRATIC_3D")

# --------------------------------------------------- tanh solitons, c=i, f=-u+u^3
# The scaled coordinate inside tanh: d=1 z/sqrt6, d=2 (z1+z2)/(2 sqrt3),
# d=3 (z1+z2+z3)/(3 sqrt2).  exact: tanh((i sum(z)/sqrt(d) - 2t)/sqrt6).
_SECH2 = lambda w: 1.0 / cmath.cosh(w)**2
_A = math.sqrt(2.0 / 3.0)

SOLITON_1D = WaveEquation(
    dim=1, c=1j, f={1: -1, 3: 1},
    phi=lambda z: cmath.tanh(1j * z / S6),
    psi=lambda z: -_A * _SECH2(1j * z / S6),
    exact=lambda z, t: cmath.tanh((1j * z - 2 * t) / S6),
    name="SOLITON_1D")

_K2 = 1.0 / (2 * S3)
SOLITON_2D = WaveEquation(
    dim=2, c=1j, f={1: -1, 3: 1},
    phi=lambda z: cmath.tanh(1j * (z[0] + z[1]) * _K2),
    psi=lambda z: -_A * _SECH2(1j * (z[0] + z[1]) * _K2),
    grad_phi=lambda z: (1j * _K2 * _SECH2(1j * (z[0] + z[1]) * _K2),) * 2,
    exact=lambda z, t: cmath.tanh((1j * (z[0] + z[1]) / S2 - 2 * t) / S6),
    name="SOLITON_2D")

_K3 = 1.0 / (3 * S2)
SOLITON_3D = WaveEquation(
    dim=3, c=1j, f={1: -1, 3: 1},
    phi=lambda z: cmath.tanh(1j * (z[0] + z[1] + z[2]) * _K3),
    psi=lambda z: -_A * _SECH2(1j * (z[0] + z[1] + z[2]) * _K3),
    grad_phi=lambda z: (1j * _K3 * _SECH2(1j * (z[0] + z[1] + z[2]) * _K3),) * 3,
    exact=lambda z, t: cmath.tanh((1j * (z[0] + z[1] + z[2]) / S3 - 2 * t) / S6),
    name="SOLITON_3D")

# ----------------------------------------- sine data on the unit box, f=-u+u^3
SINE_CI_1D = WaveEquation(
    dim=1, c=1j, f={1: -1, 3: 1},
    phi=lambda z: cmath.sin(math.pi * z),
    psi=lambda z: -cmath.sin(math.pi * z),
    domain=((0.0, 1.0),), name="SINE_CI_1D")

_sin2 = lambda z: cmath.sin(math.pi * z[0]) * cmath.sin(math.pi * z[1])
_grad_sin2 = lambda z: (
    math.pi * cmath.cos(math.pi * z[0]) * cmath.sin(math.pi * z[1]),
    math.pi * cmath.sin(math.pi * z[0]) * cmath.cos(math.pi * z[1]),
)

SINE_CI_2D = WaveEquation(
    dim=2, c=1j, f={1: -1, 3: 1},
    phi=_sin2, psi=lambda z: -_sin2(z), grad_phi=_grad_sin2,
    domain=((0.0, 1.0), (0.0, 1.0)), name="SINE_CI_2D")

SINE_C1_2D = WaveEquation(
    dim=2, c=1, f={1: -1, 3: 1},
    phi=_sin2, psi=lambda z: -_sin2(z), grad_phi=_grad_sin2,
    domain=((0.0, 1.0), (0.0, 1.0)), name="SINE_C1_2D")

ALL = {eq.name: eq for eq in (
    SIM01_QUADRATIC_1D, SIM02_CUBIC_1D, SIM03_MIXED_1D,
    SIM05_QUADRATIC_2D, SIM08_QUADRATIC_3D,
    SOLITON_1D, SOLITON_2D, SOLITON_3D,
    SINE_CI_1D, SINE_CI_2D, SINE_C1_2D,
)}
WITH_EXACT = {k: v for k, v in ALL.items() if v.exact is not None}
