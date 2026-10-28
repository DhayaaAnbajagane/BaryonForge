# BaryonForge final code and documentation audit

## Scope and baseline

This final pass covers the clean-main `check-debug` worktree at base revision
`fafc0f31f68a1b8a6da791afdf76a686b4473cf0` and preserves the earlier
`DEBUG_REPORT.md`, `PROFILE_AUDIT_REPORT.md`, and `UTILITY_AUDIT_REPORT.md` work.
An AST inventory found 29 Python files, 130 top-level classes, and 85
top-level functions. The profile report has an explicit disposition for all 93
profile classes; the utility report inventories all 27 utility classes and
their methods plus all 35 utility functions. The remaining ten classes are the
runner classes listed below. No test files were added.

## File and API inventory

Counts are top-level `class` / `def` declarations from the 29 source files.
`P` refers to the profile class matrix in `PROFILE_AUDIT_REPORT.md`; `U` refers
to the routine-by-routine utility inventory in `UTILITY_AUDIT_REPORT.md`.

| File | Classes / functions | Disposition |
|---|---:|---|
| `Profiles/Arico20.py` | 21 / 0 | P: every class; constructors, component compositions, units, normalization, and boundaries reviewed. |
| `Profiles/BaryonCorrection.py` | 3 / 0 | P: every class; table generation, displacement readout, boundary behavior, and parameter restoration reviewed. |
| `Profiles/Base.py` | 1 / 0 | P: base API, units, projection, cutoff, and inherited methods reviewed. |
| `Profiles/Battaglia.py` | 3 / 0 | P: every class; pressure units, truncation edge, and electron-pressure ratio reviewed. |
| `Profiles/HyDif.py` | 4 / 0 | P: every class; gas fractions, mass normalization, aliases, and compositions reviewed. |
| `Profiles/Mead20.py` | 20 / 1 | P: every class; diffuse Fourier terms, pressure/temperature relations, and real/projected/Fourier paths reviewed; `Tagn2pars` reviewed. |
| `Profiles/Schneider19.py` | 10 / 0 | P: every class; shocked gas, relaxation, component fractions, and LSS composition reviewed. |
| `Profiles/Schneider25.py` | 11 / 0 | P: every class; gas decomposition, central limits, normalization, and CLM construction reviewed. |
| `Profiles/Thermodynamic.py` | 12 / 0 | P: every class; shared units, public profile APIs, pressure limits, tSZ, and X-ray compositions reviewed. |
| `Profiles/__init__.py` | 0 / 0 | Import order and submodule exposure inspected. |
| `Profiles/misc.py` | 8 / 1 | P: every class; operators, wrappers, truncation, and `Mdelta_to_Mtot` reviewed; `_has_custom_method` checked. |
| `Runners/HealpixRunner.py` | 4 / 5 | Fresh review of all four runner classes and five top-level geometry/regridding functions; empty inputs, redshift interpolation, p-key routing, and map sums checked. |
| `Runners/Map2DRunner.py` | 4 / 8 | Fresh review of all four runner classes and eight top-level regrid helpers; odd cutouts, periodic small grids, center direction, anisotropic keys, and conservation checked. |
| `Runners/SnapshotRunner.py` | 2 / 18 | Both classes and all cell-index, periodic-query, sorting, gather, and offset functions inspected; small-box paths compared against the KDTree reference. |
| `Runners/__init__.py` | 0 / 0 | Wildcard export order inspected and preserved. |
| `Runners/_chunks.py` | 0 / 15 | All chunk, reduction, threading, and batched-evaluation helpers inspected; empty input behavior checked. |
| `BaryonForge/__init__.py` | 0 / 1 | Root exports, version fallback, and commit parsing inspected. |
| `utils/Cache.py` | 4 / 1 | U1/U2/U9 retained; cache keys, dtype identity, copies, LRU, serialization, and collisions checked. |
| `utils/Parallelize.py` | 2 / 0 | U6 retained; empty work, split/join settings, and aggregation paths reviewed. |
| `utils/Pixel.py` | 4 / 0 | Pixel convolution/readout classes and shape/normalization contracts reviewed. |
| `utils/Tabulate.py` | 3 / 22 | Existing interpolation fixes retained; all classes/functions, bounds, NaNs, exact edges, parameter restoration, and batching reviewed. |
| `utils/Xray.py` | 1 / 0 | U3 retained; per-query scale factors, broadcasting, input validation, and interpolation bounds reviewed. |
| `utils/__init__.py` | 0 / 0 | Order-sensitive import sequence inspected and left intact. |
| `utils/concentration.py` | 7 / 0 | U4 retained; all concentration relations and translated-grid bounds reviewed. |
| `utils/constants.py` | 0 / 0 | Exported unit conversions and physical constants checked. |
| `utils/debug.py` | 0 / 1 | `log_time` callback and timing behavior reviewed. |
| `utils/halomodel.py` | 1 / 0 | U5 retained; normalization inputs and cache invalidation reviewed. |
| `utils/io.py` | 5 / 2 | U6 retained; cosmology checks, array shapes, catalogs, map containers, empty inputs, and snapshot binning reviewed. |
| `utils/misc.py` | 0 / 10 | U7/U8 retained; mass/radius helpers, dispatch, parameter validation, minimizer fallback, and operator errors reviewed. |

## Confirmed defects and corrections

### Empty HEALPix catalogs

Before the fix, `BaryonifyShell.process()` and `PaintProfilesShell.process()`
called `np.max(cat['z'])` for an empty catalog and raised
`ValueError: zero-size array to reduction operation maximum which has no
identity`. A direct Python reproduction used an empty
`HaloLightConeCatalog`, a nonzero NSIDE=1 shell, and a minimal profile.

- `BaryonifyShell` now returns the original map unchanged when there are no
  halos.
- `PaintProfilesShell` now returns a zero map with the input map's shape.
- `PaintProfilesAnisShell` includes the shell redshift when building its
  distance spline, so an empty catalog can still compute the shell background.
  With `background_val=2`, `global_tracer_fraction=0.25`, and
  `include_pixel_size=False`, the direct check returned exactly `0.5 * input`.

The generic `_chunk_bounds` / `_run_chunks` empty path was checked separately:
it returns one empty chunk, invokes the function once with zero items, and does
not itself fail.

### Anisotropic runner parameter keys

The anisotropic Map2D and HEALPix painters previously built `H['other']` from
the map model's `p_keys` only, then passed that same keyword set to the tracer
model. With an `alpha` map model and a `beta` tracer model, this failed on the
tracer's `beta` argument. Both runners now build the union of the two key lists
and filter the supplied parameter dictionary to each model's own keys before
calling it. Direct probes with distinct alpha/beta models returned finite
outputs in both geometries.

### Odd-grid full cutouts

The cutout cap used the largest even size below an odd grid dimension. On an
oversized cutout, a 5x5 grid therefore used 4x4 pixels and omitted the last row
and column. The cutout code now allows the exact odd full-grid size when the
requested cutout saturates that bound, while the existing `pick_indices`
contract remains unchanged. A custom clipped-index subclass was also exercised.
With `epsilon_max=1000`, large-cutout painting now touches every pixel for N=2
through 7; previously only 4/9, 16/25, and 36/49 cells were touched for odd
N=3, 5, and 7.

### Periodic regridding on small grids

The N<6 regrid fallback searched a fixed two-cell margin, then mutated loop
indices while wrapping them. For a 2x2 map, `BaryonifyGrid` failed its mass
conservation assertion (`sum(new_map)=27.4365911722987` versus `sum(old_map)=4`).
The fallback now computes the exact fractional overlap with the at-most two
neighbors per axis and wraps those unique cells. Both 2D and 3D code paths were
compared with an independent periodic interval-overlap reference for random
positions, signed values, and large offsets at N=1…8. End-to-end uniform-map
baryonification conserves mass for 2D and 3D N=2…6.

### Center direction in grid baryonification

When a halo lay exactly on a grid-pixel center, `g / r_grid` generated a
RuntimeWarning and NaN direction vectors. The downstream code discarded the
resulting non-finite offset, so the final map was usually unchanged, but the
helper returned invalid vectors. Compiled and Python cutout paths now use a
zero direction at zero radius. For `q_ell=1`, elliptical and circular
baryonification maps agree, the center vector is `[0, 0]`, and the direct
warning capture is empty.

### Wheel package discovery

The first local wheel contained `docs/source/conf.py` and all test modules
because setuptools namespace discovery searched every top-level directory.
`pyproject.toml` now limits package discovery to `BaryonForge*`. The rebuilt
wheel contains the package modules and dist-info only; docs and tests are not
shipped as namespace packages.

## Other runner and code-path checks

- Snapshot periodic wrapping was compared with the existing KDTree reference
  for small boxes L=1 and L=2, including a coordinate exactly at x=L. Results
  matched to absolute tolerance `1e-12`.
- Standard empty grid/snapshot behavior, split/join settings, batch versus
  per-halo readout, and `n_jobs` consistency are covered by existing runner
  tests and earlier reports. No separate `HaloModelRunner` module/class is
  present in this checkout.
- The existing profile findings F1–F23, C1–C2 and utility findings U1–U9 were
  retained; see the two detailed audit reports for their formulas, edge
  probes, and before/after values.
- Notebook JSON inspection covered all 20 example notebooks and 96 code cells
  referencing BaryonForge. A source-reference scan found 272 `bfg.*` expressions
  (98 unique expression paths); all resolve through the installed package
  exports. The notebooks were not executed because several use remote or local
  research data and long-running calculations.

## Documentation, exports, and packaging

- Public imports for the seven runner entry points, `SimpleArrayCache`,
  `TabulatedProfile`, and `HaloLightConeCatalog` were verified through
  `import BaryonForge as bfg`; the runner classes are available at the package
  root.
- The README and Sphinx quickstart Python examples compile. The quickstart
  placeholders are valid Python and define cosmology before use.
- A basic-theme Sphinx build completed after the local-path hook correction in
  `docs/source/conf.py`. It emitted 30 environment warnings: intersphinx was
  offline, system Sphinx used Python 3.8 without `pyccl`, and the configured
  RTD theme was not installed. Generated API `.rst` files were removed; no
  generated documentation artifacts remain in the worktree.
- `pip wheel --no-deps --no-build-isolation . --wheel-dir /tmp/bfg-final-wheel-clean`
  succeeded. The wheel was `BaryonForge-0.0.0-py3-none-any.whl` (185,323
  bytes); version `0.0.0` is the setuptools-scm fallback for this untagged
  checkout.

## Verification

All inline runtime probes used the existing environment at
`../my_pyccl_env/bin/python`; no test files were written. The first pytest
attempt, Slurm job `49612984`, exited 139 before collection because importing
the environment's `readline` extension segfaulted. The workaround below
successfully ran the focused set with the optional module disabled:

```sh
srun --partition=broadwl --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:30:00 \
  --output=/tmp/bfg-final-focused-postfix-%j.log --error=/tmp/bfg-final-focused-postfix-%j.err \
  ../my_pyccl_env/bin/python -c 'import sys; sys.modules["readline"] = None; import pytest; raise SystemExit(pytest.main(["-q", "tests/test_grid_runners.py", "tests/test_healpix.py", "tests/test_fast_edge_cases.py", "tests/test_io.py", "tests/test_profile_utilities.py"]))'
```

Slurm job `49613000` completed with **128 passed, 9 warnings** in 196.37
seconds; the `srun` client exit code was 0. Log:
`/tmp/bfg-final-focused-postfix-49613000.log`; paired stderr is empty. The
remaining warnings are SWIG deprecations, logarithms in the test fixture's
tabulated displacement reference, an overflow in the cutoff test fixture, a
profile convolution warning, and the expected `wa` default warning. The
previous Map2D center-direction warning is absent.

The final full-suite command was:

```sh
srun --partition=broadwl --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:45:00 \
  --output=/tmp/bfg-final-full-postfix-%j.log --error=/tmp/bfg-final-full-postfix-%j.err \
  ../my_pyccl_env/bin/python -c 'import sys; sys.modules["readline"] = None; import pytest; raise SystemExit(pytest.main(["-q", "tests"]))'
```

Slurm job `49613008` completed with **453 passed, 49 warnings** in 261.17
seconds; the `srun` client exit code was 0. Log:
`/tmp/bfg-final-full-postfix-49613008.log`; paired stderr is empty. The known
test-fixture logarithm warning remains; the Map2D center-direction warning is
gone. `python -m py_compile` over all 29 package Python files, README/quickstart
snippet compilation, and `git diff --check` passed after the runner changes.

## Remaining limits

- The normal `python -m pytest` startup segfaults because this environment's
  optional `readline` extension segfaults on import. Slurm test commands set
  `sys.modules['readline'] = None` before importing pytest; this allows pytest
  to skip the optional extension without changing project code.
- The Sphinx warnings and unexecuted notebook data workflows are environment
  limits, not confirmed source failures. Existing warnings from profile/table
  boundaries and test fixture logarithms are documented in the test logs.
