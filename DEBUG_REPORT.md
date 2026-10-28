# BaryonForge Runtime Debug Report

## Scope and environment

- Worktree: `BaryonForge-check-debug`
- Branch: `check-debug`
- Clean starting revision: `fafc0f31f68a1b8a6da791afdf76a686b4473cf0` (`Improve runtime (#56)`)
- Python: 3.11.10
- NumPy: 1.26.4; SciPy: 1.14.1; Numba: 0.60.0; Healpy: 1.17.3; pyccl: 3.2; pytest: 9.1.1
- No tests were added.

## Reproduced runtime bugs and fixes

1. **Exact-zero profile table nodes returned NaN.** A `TabulatedProfile(GaussianProfile())` table node at `z=0.2`, `M=1e12`, and `r=12.655386133 Mpc` contained exactly zero, but lookup returned NaN. The cause was applying `log(0)`, then interpolating `-inf`. `TabulatedProfile` and `ParamTabulatedProfile` now use a finite log sentinel that exponentiates back to exactly zero at a zero-valued table node. The focused run's six divide-by-zero warnings from these table conversions disappeared after the change.

2. **Negative correlation samples returned NaN.** A `TabulatedCorrelation3D` table on 600 logarithmic radius samples from `1e-3` to `1e3 Mpc` had `xi=-9.62058015e-05` at `z=0`, `r=181.453190 Mpc`; lookup at that table node returned NaN because the signed correlation was log-transformed. The correlation table now interpolates raw signed values in linear space. Lookup at the reproduced node returns `-9.62058015e-05`.

Both issues were independently reproduced on the clean main baseline before editing. The fixes are in `BaryonForge/utils/Tabulate.py`.

## Validation

1. `../my_pyccl_env/bin/python -m pytest --collect-only -q tests/test_grid_runners.py tests/test_healpix.py tests/test_fast_edge_cases.py tests/test_profile_utilities.py` — 123 focused tests collected.
2. Focused baseline run, before fixes: `../my_pyccl_env/bin/python -m pytest -q tests/test_grid_runners.py tests/test_healpix.py tests/test_fast_edge_cases.py tests/test_profile_utilities.py` — 123 passed, 15 warnings in 87.42 s. The warnings included six `log(0)` warnings from tabulation.
3. Focused rerun after fixes, same command — 123 passed, 9 warnings in 92.38 s.
4. Full existing suite: `../my_pyccl_env/bin/python -m pytest -q tests/` — 453 passed, 53 warnings in 151.98 s. Output retained at `/tmp/baryonforge-clean-full.log`.

The runs were short test workloads, not production simulations. No Slurm job was submitted; the checkout has no Slurm test script, and the available client could not resolve its configured controller (`midway2-mgt2`).

## Remaining observations

The full suite has no failures. Existing warnings remain for synthetic edge-case profiles, convolution artifacts, and profile diagnostics. The center-pixel unit-vector calculation in `Map2DRunner` still emits an invalid-division warning at zero radius; the runner later zeros the non-finite displacement, and the relevant test passes. This warning was not changed in this pass.
