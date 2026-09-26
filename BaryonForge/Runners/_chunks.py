"""
Helpers shared by the HEALPix and grid runners: halos are processed in chunks (one model call per chunk, or
one per halo for models without a batched readout), optionally in threads, and their contributions are added
to the output in catalog order, so the result does not depend on the chunking or on the number of threads.
"""

import numpy as np
import joblib
from numba import njit
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from ..utils.misc import _batch_method


@njit
def _add_at(target, index, values):
    """`target[index[i]] += values[i]` for every i, in order. `target` is modified in place."""

    for i in range(index.size):
        target[index[i]] += values[i]

    return target


@njit
def _add_rows_at(target, index, values):
    """`target[index[i], :] += values[i, :]` for every i, in order. `target` is modified in place."""

    for i in range(index.size):
        for k in range(values.shape[1]):
            target[index[i], k] += values[i, k]

    return target

#Compile once at import
_add_at(np.zeros(3), np.array([0, 2]), np.ones(2))
_add_rows_at(np.zeros([3, 3]), np.array([0, 2]), np.ones([2, 3]))


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
