"""
Helpers shared by the HEALPix and grid runners: halos are processed in chunks (one model call per chunk, or
one per halo for models without a batched readout), optionally in threads, and their contributions are added
to the output in catalog order, so the result does not depend on the chunking or on the number of threads.
"""

import numpy as np
import joblib
from numba import njit
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from ..utils.misc import _batch_method


@njit(nogil = True)
def _add_at(target, index, values):
    """`target[index[i]] += values[i]` for every i, in order. `target` is modified in place."""

    for i in range(index.size):
        target[index[i]] += values[i]

    return target


@njit(nogil = True)
def _add_rows_at(target, index, values):
    """`target[index[i], :] += values[i, :]` for every i, in order. `target` is modified in place."""

    for i in range(index.size):
        for k in range(values.shape[1]):
            target[index[i], k] += values[i, k]

    return target


@njit(nogil = True)
def _add_rows_at_range(target, index, values, lo, hi):
    """`_add_rows_at` for the entries with `lo <= index[i] < hi` only (in the same order)."""

    for i in range(index.size):
        j = index[i]
        if (j >= lo) and (j < hi):
            for k in range(values.shape[1]):
                target[j, k] += values[i, k]

    return target


@njit(nogil = True)
def _fill_zeros(flat, lo, hi):
    """`flat[lo:hi] = 0`."""

    for i in range(lo, hi): flat[i] = 0.0
    return flat


def _zeros(shape, n_threads):
    """
    `np.zeros(shape)`, with the memory written by `n_threads` threads at once if `n_threads > 1`. Freshly
    allocated pages are otherwise mapped when first written, one page fault at a time, which for map-sized
    arrays filled pixel by pixel in the (serial) accumulation took a large part of the threaded runs.
    """

    if n_threads == 1: return np.zeros(shape)

    out    = np.empty(shape)
    flat   = out.reshape(-1)
    bounds = np.linspace(0, flat.size, n_threads + 1).astype(np.int64)
    with ThreadPoolExecutor(max_workers = n_threads) as pool:
        for f in [pool.submit(_fill_zeros, flat, bounds[t], bounds[t + 1]) for t in range(n_threads)]: f.result()

    return out


@njit(nogil = True)
def _all_close_to_zero(values):
    """`np.allclose(values, 0)` for a 1D array (|v| <= 1e-8 for all v, NaN counting as not close), stopping at
    the first value that is not, instead of building map-sized temporaries."""

    for v in values:
        if not (abs(v) <= 1e-8): return False
    return True


@njit(nogil = True)
def _count_nonzero(values, lo, hi):
    n = 0
    for i in range(lo, hi):
        if values[i] != 0: n += 1
    return n


@njit(nogil = True)
def _fill_nonzero(values, lo, hi, out, start):
    for i in range(lo, hi):
        if values[i] != 0:
            out[start] = i
            start += 1
    return out


def _nonzero(values, n_threads):
    """`np.flatnonzero(values)` for a 1D array, counted and filled in `n_threads` blocks at once (same result)."""

    bounds = np.linspace(0, values.size, n_threads + 1).astype(np.int64)
    blocks = list(zip(bounds[:-1], bounds[1:]))
    if n_threads == 1:
        counts = [_count_nonzero(values, lo, hi) for lo, hi in blocks]
    else:
        with ThreadPoolExecutor(max_workers = n_threads) as pool:
            counts = list(pool.map(lambda b: _count_nonzero(values, *b), blocks))
    out   = np.empty(int(sum(counts)), dtype = np.int64)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    if n_threads == 1:
        _fill_nonzero(values, 0, values.size, out, 0)
    else:
        with ThreadPoolExecutor(max_workers = n_threads) as pool:
            list(pool.map(lambda k: _fill_nonzero(values, blocks[k][0], blocks[k][1], out, start[k]), range(n_threads)))
    return out


def _sum(values, n_threads):
    """`np.sum(values)`; with `n_threads > 1`, the sum of the sums of `n_threads` blocks computed at once (which
    can differ from `np.sum` in the last bits; used for checks only)."""

    if n_threads == 1: return np.sum(values)
    blocks = np.array_split(values.ravel(), n_threads)
    with ThreadPoolExecutor(max_workers = n_threads) as pool:
        return sum(pool.map(np.sum, blocks))


@contextmanager
def _row_adder(n_threads):
    """
    Yields `add(target, index, values)`, the same as `_add_rows_at`, which with `n_threads > 1` splits the target's
    rows into `n_threads` ranges handled by separate threads (the GIL-free kernel scans the pairs for its range).
    Every row receives its contributions in the original order, so the result is exactly that of `_add_rows_at`.
    """

    if n_threads == 1:
        yield _add_rows_at
        return

    with ThreadPoolExecutor(max_workers = n_threads) as pool:
        def add(target, index, values):
            bounds  = np.linspace(0, target.shape[0], n_threads + 1).astype(np.int64)
            futures = [pool.submit(_add_rows_at_range, target, index, values, bounds[t], bounds[t + 1]) for t in range(n_threads)]
            for f in futures: f.result()
        yield add


def _n_threads(n_jobs):
    """Number of threads for `n_jobs`, following the joblib convention for negative values (-1 = all cores)."""

    n = 1 if n_jobs in (None, 0) else int(n_jobs)
    return max(1, joblib.cpu_count() + 1 + n) if n < 0 else n


def _chunk_bounds(work, max_work, n_threads):
    """
    Splits consecutive items with estimated costs `work` into chunks of at most ~`max_work` each, and (with
    several threads) into at least 8 chunks per thread for load balancing. Returns a list of index arrays.
    """

    N     = work.size
    size  = max(1, int(np.ceil(N / (8 * n_threads)))) if n_threads > 1 else max(N, 1)
    chunk = np.maximum(np.cumsum(work) // max_work, np.arange(N) // size).astype(int)
    chunk = np.maximum.accumulate(chunk)

    return np.split(np.arange(N), np.flatnonzero(np.diff(chunk)) + 1)


def _run_chunks(function, chunks, n_threads, verbose = False, desc = None):
    """
    Yields `function(chunk)` for every chunk, in chunk order, running up to `n_threads` chunks at once.
    At most two results per thread are held at any time. A progress bar is shown if `verbose` and `desc`.
    """

    with tqdm(total = sum(c.size for c in chunks), desc = desc, disable = (not verbose) or (desc is None)) as pbar:
        if n_threads == 1:
            for c in chunks:
                yield function(c)
                pbar.update(c.size)
            return

        with ThreadPoolExecutor(max_workers = n_threads) as executor:
            pending = [executor.submit(function, c) for c in chunks[:2*n_threads]]
            for i in range(len(chunks)):
                result = pending[i].result()
                if i + 2*n_threads < len(chunks): pending.append(executor.submit(function, chunks[i + 2*n_threads]))
                pending[i] = None
                yield result
                pbar.update(chunks[i].size)


def _evaluate_pairs(model, method, cosmo, r, hid, M, a, other):
    """
    `model.<method>` (`'displacement'`, `'projected'` or `'real'`) for many (halo, radius) pairs, where `r` are
    the comoving radii, `hid` the index of each pair's halo into `M`, `a` and the arrays in `other` (the extra
    tabulated properties), and the pairs of a halo are contiguous. Uses the model's batched readout if its class
    defines one (eg. `Baryonification2D`, `TabulatedProfile`), and otherwise calls the model once per halo.
    """

    batch = _batch_method(model, f'_{method}_batch', method, f'_{method}', '_readout')
    if batch is not None:
        return np.asarray(batch(r, hid, M, a, **other), dtype = float)

    out    = np.zeros(r.size)
    bounds = np.searchsorted(hid, np.arange(M.size + 1))
    for i in range(M.size):
        s, e = bounds[i], bounds[i + 1]
        if e == s: continue
        o_i = {k : v[i] for k, v in other.items()} #Other properties
        if method == 'displacement': out[s:e] = model.displacement(r[s:e], M[i], a[i], **o_i)
        else:                        out[s:e] = getattr(model, method)(cosmo, r[s:e], M[i], a[i], **o_i)

    return out
