# BaryonForge Utilities Audit

## Scope and method

- Checkout: check-debug, based on fafc0f31f68a1b8a6da791afdf76a686b4473cf0.
- Scope: all 12 BaryonForge/utils modules, all top-level definitions, and their callers in profiles/runners/tests. The original audit inventory was 27 classes, 35 top-level functions, and 93 class methods; the follow-up adds private `_dtype_key`, for 27 classes, 36 top-level functions, and 93 class methods in the final source.
- Inspected public APIs and private helpers, with attention to shape/broadcast behavior, numerical bounds, key construction, units, parallel empty work, and order-sensitive imports. No tests were added.
- Preserved prior Tabulate.py, profile documentation, profile-composition fixes, DEBUG_REPORT.md, and PROFILE_AUDIT_REPORT.md.
- Cache baseline: upstream main at this checkout's base revision lacks a positional/keyword boundary and reproduces the alias described in U1. The separate original BaryonForge checkout already had an uncommitted _KEYWORD_BOUNDARY patch; I inspected that diff read-only and left the checkout untouched. This worktree's structured typed key preserves that separation and adds distinct type/object keys.

## Confirmed findings and fixes

| ID | Reproduction / impact | Fix |
|---|---|---|
| U1 | On the base implementation, a decorated variadic function returned the first result again for f('x', 1) followed by f(x=1): the flattened key is identical. | Decorated cache calls now encode positional and sorted keyword arguments in separate tagged fields. |
| U2 | The base array cache also aliased True with 1 (Python key equality), and distinct custom objects with the same str() could reuse one another's result. | Keys now include scalar/container/array type tags and use identity for arbitrary objects, retaining the object and a string snapshot to avoid id reuse and detect visible state changes. |
| U3 | With a training table at a=[0.5, 1], query a=[0.5, 1] returned both emissivities from the first scale factor ([1, 1] instead of [1, 4]). The old __call__ always used a[0]. | Scale factor now broadcasts to the query temperature shape and each point is interpolated with its own a. Incompatible shapes and negative/non-finite training emissivity raise ValueError. |
| U4 | For Duffy08 remapped from 200c to 200m, with M_in=[1e11,1e15], the translated target grid was approximately [1.2189649738e11, 1.3377509347e15]. The old check silently clamped a query below the translated lower bound and rejected valid target masses above M_in_hi. | Requests are checked against the translated target-mass grid; malformed input grids and non-increasing/non-finite translations fail explicitly. |
| U5 | FlexibleHMCalculator._mf0 and _mbf0 depend on rho0, but cached only by cosmology and scale factor. Directly changing rho0 at fixed (cosmo,a) left both normalizations unchanged. | Both cache predicates now include the last rho0. |
| U6 | SimpleParallel([]).process() set n_jobs=0 and raised from joblib. SplitJoinParallel divided by zero for an empty catalog. Stringifying empty HaloLightConeCatalog or HaloNDCatalog failed on min()/max(). | Empty simple work returns []; empty split-join work returns a zero map; empty catalog descriptions avoid reductions. Zero/invalid job counts raise actionable errors. |
| U7 | When PCHIP construction fails, the fallback correctly returns y[cen], the sample whose x is closest to zero, but the warning incorrectly claimed it used np.min(). Callers use this value as the estimated zero-crossing radius. | Corrected the diagnostic text; numerical fallback is unchanged. |
| U8 | Calling exported generate_operator_method() with an unsupported operation fell through to an uninitialized local and raised UnboundLocalError. | Unsupported operations now raise ValueError. |
| U9 | Arrays and NumPy scalars with identical raw bytes and `dtype.str` but different structured field names or layouts could share a cache entry. Dtype metadata was also absent from the old array key. | Cache keys now recursively encode dtype type/byte order/itemsize, ordered field names, titles, offsets, child and subarray dtypes/shapes, alignment, and supported metadata. Unsupported or cyclic metadata raises a clear `TypeError`, avoiding an ambiguous cache key. |

## Module-by-module definition inventory and disposition

### Cache.py

- Function `_dtype_key` — recursively encodes NumPy dtypes and safe metadata values for cache keys; opaque or cyclic values raise `TypeError`.
- Class _CachedFunction — methods audited: __init__, __call__.
- Class SimpleArrayCache — methods audited: __init__, _key, get, contains, lookup, set, __call__; its nested _IdentityKey helper was also checked for equality, hash stability, and pickling.
- Class CachedProfile — methods audited: __init__, __getattr__, __str_prf__, __str_par__.
- Class CachedHODProfile — methods audited: __init__.

Disposition: Reviewed LRU operations, wrapper copying, HOD forwarding, and all key paths. Fixes U1/U2 address argument-boundary, scalar-type, and arbitrary-object collisions; U9 separates structured dtypes and rejects unsupported metadata rather than allowing a cache collision.

### Parallelize.py

- Class SimpleParallel — methods audited: __init__, single_run, process.
- Class SplitJoinParallel — methods audited: __init__, split_run, single_run, process.

Disposition: Reviewed ordering, splitting, zero-work and job-count behavior, worker construction, and additive map joining. Empty SimpleParallel work now returns []; empty SplitJoin work returns a zero map.

### Pixel.py

- Class ConvolvedProfile — methods audited: __init__, __str_prf__, __str_par__, __getattr__, __getstate__, __setstate__, _real, projected, fourier.
- Class GridPixelApprox — methods audited: __init__, beam, beam_2D, real, projected.
- Class HealPixel — methods audited: __init__, real, projected.
- Class NoPix — methods audited: real, projected.

Disposition: Reviewed harmonic vs physical units, Fourier normalization factors, zero modes, shapes, and CCL wrapper delegation. HealPixel.real intentionally returns zero and is documented as unsupported; projected() is the supported path.

### Tabulate.py

- Function _log_profile_table — signature, validation, edge behavior, and callers audited.
- Function _find_interval — signature, validation, edge behavior, and callers audited.
- Function _linear_regular_grid — signature, validation, edge behavior, and callers audited.
- Function _find_interval_scaled — signature, validation, edge behavior, and callers audited.
- Function _axis_nodes — signature, validation, edge behavior, and callers audited.
- Function _inverse_spacings — signature, validation, edge behavior, and callers audited.
- Function _outside_or_nan — signature, validation, edge behavior, and callers audited.
- Function _linear_regular_grid_3d — signature, validation, edge behavior, and callers audited.
- Function _linear_regular_grid_4d — signature, validation, edge behavior, and callers audited.
- Function _linear_curves — signature, validation, edge behavior, and callers audited.
- Function _interpolate_curves — signature, validation, edge behavior, and callers audited.
- Function _interpolate — signature, validation, edge behavior, and callers audited.
- Function _pickle_without_Pk — signature, validation, edge behavior, and callers audited.
- Function _run_table_slices — signature, validation, edge behavior, and callers audited.
- Function _map_table_slices — signature, validation, edge behavior, and callers audited.
- Function _set_parameter — signature, validation, edge behavior, and callers audited.
- Function _set_parameter_recursive — signature, validation, edge behavior, and callers audited.
- Function _record_parameters — signature, validation, edge behavior, and callers audited.
- Function _record_parameter_recursive — signature, validation, edge behavior, and callers audited.
- Function _restore_parameters — signature, validation, edge behavior, and callers audited.
- Function _get_parameter — signature, validation, edge behavior, and callers audited.
- Function _get_parameter_recursive — signature, validation, edge behavior, and callers audited.
- Class TabulatedProfile — methods audited: __init__, __str_prf__, __str_par__, _table_slice, setup_interpolator, _readout, _readout_batch, _real_batch, _projected_batch, _real, _projected.
- Class ParamTabulatedProfile — methods audited: __init__, _table_slice, setup_interpolator, _readout, _readout_batch, _real_batch, _projected_batch, real, projected.
- Class TabulatedCorrelation3D — methods audited: __init__, setup_interpolator, __call__.

Disposition: Reviewed interpolation helpers, tabulation setup/readout, batching, parameter restoration, serialization, and callers. Existing Tabulate fixes from the prior runtime audit are preserved; no new Tabulate edits in this utilities pass.

### Xray.py

- Class EmissivityTable — methods audited: __init__, _build_regular_grid_interpolator, _assert_in_bounds, __call__.

Disposition: Reviewed regular/scattered grid construction, bounds, low-temperature padding, and query shapes. Fix U3 broadcasts per-query scale factors and rejects negative/non-finite emissivity training values.

### __init__.py

- No top-level classes or functions; module constants and import/export behavior inspected.

Disposition: Verified the existing import order is unchanged; the comment identifies it as necessary to avoid circular imports.

### concentration.py

- Class BaseGenericConcentration — methods audited: _concentration, _check_mass_def_strict.
- Class GenericConcentrationDuffy08 — methods audited: no methods.
- Class GenericConcentrationKlypin11 — methods audited: no methods.
- Class GenericConcentrationPrada12 — methods audited: no methods.
- Class GenericConcentrationDiemer15 — methods audited: no methods.
- Class GenericConcentrationBhattacharya13 — methods audited: no methods.
- Class GenericConcentrationIshiyama21 — methods audited: no methods.

Disposition: Reviewed all six remapper definitions and the scale-radius-preserving translation. Fix U4 validates the target-mass interval after translating the input grid.

### constants.py

- No top-level classes or functions; module constants and import/export behavior inspected.

Disposition: Reviewed the exported CCL-unit conversions and thermodynamic constants; no discrepancy was confirmed in this audit.

### debug.py

- Function log_time — signature, validation, edge behavior, and callers audited.

Disposition: Reviewed the timing decorator's injected callback and cumulative line timing; no runtime defect was confirmed.

### halomodel.py

- Class FlexibleHMCalculator — methods audited: __init__, _get_mass_function, _get_halo_bias.

Disposition: Reviewed the mass-function and bias normalization caches. Fix U5 includes rho0 in cache invalidation because it affects each normalization.

### io.py

- Function _check_cosmology — signature, validation, edge behavior, and callers audited.
- Function _as_1d_arrays — signature, validation, edge behavior, and callers audited.
- Class HaloLightConeCatalog — methods audited: __init__, data, cosmology, __getitem__, __str__.
- Class HaloNDCatalog — methods audited: __init__, data, cosmology, __getitem__, __str__, __repr__.
- Class LightconeShell — methods audited: __init__, data, cosmology.
- Class GriddedMap — methods audited: __init__, data, cosmology.
- Class ParticleSnapshot — methods audited: __init__, data, cosmology, make_map.

Disposition: Reviewed cosmology validation/copying, array shapes, slicing, map containers, empty catalogs, and snapshot binning. Fix U6 makes empty catalog descriptions safe.

### misc.py

- Function _default_mass_def — signature, validation, edge behavior, and callers audited.
- Function _runner_cosmology — signature, validation, edge behavior, and callers audited.
- Function _halo_radius — signature, validation, edge behavior, and callers audited.
- Function _batch_method — signature, validation, edge behavior, and callers audited.
- Function _check_p_keys — signature, validation, edge behavior, and callers audited.
- Function generate_operator_method — signature, validation, edge behavior, and callers audited.
- Function destory_Pk — signature, validation, edge behavior, and callers audited.
- Function build_cosmodict — signature, validation, edge behavior, and callers audited.
- Function safe_Pchip_minimize — signature, validation, edge behavior, and callers audited.
- Function combine_fftpars — signature, validation, edge behavior, and callers audited.

Disposition: Reviewed radius helpers, batch-method dispatch, runner parameter validation, cosmology conversion, PCHIP fallback, operator generation, and FFT-parameter merging. Fix U7 improves the PCHIP fallback diagnostic; U8 raises a clear error for unsupported operators.

## Existing tests and verification

- Direct runtime repros for U1-U8 completed before/after fixes; numerical assertions, shapes, exception paths, empty-work outputs, and rho0 cache changes were checked without adding tests.
- After the cache-key refactor, the positional/keyword, bool/int, and same-string-object checks passed again, and a populated CachedProfile remained pickleable.
- Dtype-key follow-up command: `../my_pyccl_env/bin/python - <<'PY'` with inline assertions against `SimpleArrayCache._key` and `_dtype_key` — equal-byte/equal-`dtype.str` mass-versus-radius structured arrays and `np.void` scalars, nested structures, subarray shapes, field titles and offsets, aligned versus unaligned layouts, and differing supported metadata all produced distinct keys. Repeated identical dtypes hit the cache, ordinary numeric arrays retained cache/copy behavior, opposite-endian dtypes stayed distinct, and an opaque `object()` in dtype metadata raised the clear `TypeError`. Output: `dtype collisions, metadata rejection, ordinary cache hit/copy checks: PASS`. The focused existing serialization test below also confirmed a populated `CachedProfile` remains pickleable.
- X-ray checks covered scalar and per-point scale factors, per-leading-axis scale factors for multidimensional T, broadcast errors, and negative emissivity rejection. Concentration checks covered both ends of the translated target grid, including a valid target mass above the original M_in_hi. Empty catalog formatting, SimpleParallel empty work, SplitJoin zero maps, PCHIP fallback diagnostic, unsupported operator errors, and rho0 cache invalidation were also checked directly.
- Slurm job 49611576: `srun --partition=broadwl --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:45:00 --output=/tmp/bfg-utility-focused-%j.log --error=/tmp/bfg-utility-focused-%j.err ../my_pyccl_env/bin/python -m pytest -q tests/test_fast_edge_cases.py tests/test_profile_utilities.py tests/test_io.py tests/test_grid_runners.py tests/test_healpix.py` — 128 passed, 10 warnings, 210.37 seconds; `srun` client exit code 0. Log: `/tmp/bfg-utility-focused-49611576.log`; paired stderr is empty.
- Full-suite Slurm job 49611644: `srun --partition=broadwl --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:45:00 --output=/tmp/bfg-utility-full-%j.log --error=/tmp/bfg-utility-full-%j.err ../my_pyccl_env/bin/python -m pytest -q tests` — 453 passed, 53 warnings, 263.96 seconds; `srun` client exit code 0. Log: `/tmp/bfg-utility-full-49611644.log`; paired stderr is empty. The separate `sacct` accounting query was unavailable because the controller host failed DNS resolution.
- Cache-focused follow-up Slurm job 49611866: `srun --partition=broadwl --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:20:00 --output=/tmp/bfg-cache-dtype-%j.log --error=/tmp/bfg-cache-dtype-%j.err ../my_pyccl_env/bin/python -m pytest -q tests/test_fast_edge_cases.py -k 'simple_array_cache or cached_profile or serialization'` — 9 passed, 33 deselected, 3 warnings in 27.62 seconds; `srun` client exit code 0. Log: `/tmp/bfg-cache-dtype-49611866.log`; paired stderr is empty. The earlier focused/full suite results above predate the dtype-key change.
- Final post-U9 full-suite Slurm job 49611901: `srun --partition=broadwl --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:45:00 --output=/tmp/bfg-utility-final-%j.log --error=/tmp/bfg-utility-final-%j.err ../my_pyccl_env/bin/python -m pytest -q tests` — 453 passed, 53 warnings in 265.85 seconds; `srun` client exit code 0. Log: `/tmp/bfg-utility-final-49611901.log`; paired stderr is empty. This is the final full-suite verification after the dtype-key change.
- `../my_pyccl_env/bin/python -m py_compile BaryonForge/utils/Cache.py BaryonForge/utils/Xray.py BaryonForge/utils/concentration.py BaryonForge/utils/halomodel.py BaryonForge/utils/Parallelize.py BaryonForge/utils/io.py BaryonForge/utils/misc.py` — passed before the dtype-key follow-up; `../my_pyccl_env/bin/python -m py_compile BaryonForge/utils/Cache.py` passed again after it.
- `git diff --check` — passed after the dtype-key follow-up.
- BaryonForge/utils/__init__.py import order was preserved.

## Scope limits

- HealPixel's real-space response intentionally remains zero and is documented as unsupported; use its projected response. No behavior change was made.
- No notebook or remote-data workflows were run.
- No new tests were authored.
