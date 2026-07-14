# wavelab

Solver toolbox for nonlinear wave equations `u_tt − c²Δu = Σ aₖuᵏ` (complex c, complex data).
Define an equation once, solve with interchangeable methods (explicit/implicit FD, branching
Monte Carlo), compare results. Design: `docs/superpowers/specs/2026-07-13-wavelab-design.md`.

## Setup
    conda env create -f environment.yml
    conda activate wavelab
    pip install -e .
    pytest -m "not slow"
