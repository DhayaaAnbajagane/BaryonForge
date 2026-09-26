
import numpy as np
import pyccl as ccl
import pickle, warnings, joblib
from tqdm import tqdm
from itertools import product
from scipy import interpolate
from numba import njit
from concurrent.futures import ThreadPoolExecutor
from .misc import destory_Pk

__all__ = ['_set_parameter', '_get_parameter', 'TabulatedProfile', 'ParamTabulatedProfile']


@njit(nogil = True)
def _find_interval(grid, start, n, x):
    """Index i with grid[i] <= x < grid[i+1] (the last interval for x at or above the top node, and the
    first for x below the bottom one), as scipy's `find_interval_ascending` with extrapolation.
    The search starts from the interval a uniform grid would give (the tables are uniform in log radius
    and log mass), so it takes a step or two instead of a full bisection; any grid gives the same result."""

    lo, hi = grid[start], grid[start + n - 1]
    if not (lo <= x <= hi):
        return 0 if x < lo else n - 2
    if x == hi:
        return n - 2

    i = int((x - lo) / (hi - lo) * (n - 1))
    i = min(max(i, 0), n - 2)
    if (grid[start + i] <= x) and (x < grid[start + i + 1]):
        return i

    low, high = 0, n - 2
    while low < high:
        mid = (low + high) // 2
        if   x <  grid[start + mid]:     high = mid
        elif x >= grid[start + mid + 1]: low  = mid + 1
        else:
            low = mid
            break
    return low


@njit(nogil = True)
def _linear_regular_grid(grid, start, size, values, strides, points, fill, use_fill, two_d, out):
    """
    Linear interpolation on a rectilinear grid (all grids concatenated in `grid`, dimension k starting at
    `start[k]` with `size[k]` nodes), with the same arithmetic as scipy's `RegularGridInterpolator`: the same
    intervals and normalized distances, weights multiplied in dimension order, and corners summed in the
    same order. `two_d` selects the arithmetic of scipy's separate 2D path (`evaluate_linear_2d`).
    Points outside the grid get `fill` (if `use_fill`), and points with a NaN coordinate get NaN.
    """

    P, d  = points.shape
    lower = np.empty(d, dtype = np.int64)
    upper = np.empty(d, dtype = np.int64)
    y     = np.empty(d)
    ym    = np.empty(d)
    n_c   = 1 << d
    pw    = np.empty(n_c)                   #Weight of every corner
    fi    = np.empty(n_c, dtype = np.int64) #Flat index of every corner

    for p in range(P):
        has_nan = False
        outside = False
        for k in range(d):
            x = points[p, k]
            s = start[k]
            n = size[k]
            if x != x: has_nan = True
            if (x < grid[s]) or (x > grid[s + n - 1]): outside = True
            if n == 1:
                lower[k], upper[k], y[k] = 0, 0, 0.0 #Length-one axis: both "corners" are the single node
            else:
                i = _find_interval(grid, s, n, x)
                lower[k], upper[k] = i, i + 1
                y[k] = (x - grid[s + i]) / (grid[s + i + 1] - grid[s + i])
            ym[k] = 1 - y[k]

        if has_nan:
            out[p] = np.nan
            continue
        if outside and use_fill:
            out[p] = fill
            continue

        if two_d:
            s0, s1 = strides[0], strides[1]
            if size[1] == 1:
                #scipy interpolates along axis 0 only (and gives NaN if axis 0 also has length one)
                if size[0] == 1: out[p] = np.nan
                else: out[p] = values[lower[0]*s0]*ym[0] + values[upper[0]*s0]*y[0]
            elif size[0] == 1:
                out[p] = values[lower[1]*s1]*ym[1] + values[upper[1]*s1]*y[1]
            else:
                value = 0.0
                value = value + values[lower[0]*s0 + lower[1]*s1] * ym[0] * ym[1]
                value = value + values[lower[0]*s0 + upper[1]*s1] * ym[0] * y[1]
                value = value + values[upper[0]*s0 + lower[1]*s1] * y[0]  * ym[1]
                value = value + values[upper[0]*s0 + upper[1]*s1] * y[0]  * y[1]
                out[p] = value
            continue

        #The weights and flat indices of the corners, built one dimension at a time, so the products over the
        #leading dimensions are shared between corners. Corner c has the bit of dimension 0 as its most
        #significant bit (the order of itertools.product, as in scipy), and its weight is multiplied in
        #dimension order, ((1*w_0)*w_1)*..., so every weight, and the sum, are exactly those of scipy.
        pw[0], fi[0] = 1.0, 0
        for k in range(d):
            for c in range((1 << k) - 1, -1, -1): #Downwards, so that pw[c] and fi[c] are read before being replaced
                w, f = pw[c], fi[c]
                pw[2*c],     fi[2*c]     = w * ym[k], f + lower[k] * strides[k]
                pw[2*c + 1], fi[2*c + 1] = w * y[k],  f + upper[k] * strides[k]

        value = 0.0
        for c in range(n_c):
            value = value + values[fi[c]] * pw[c]
        out[p] = value

    return out


@njit(nogil = True)
def _find_interval_scaled(grid, s, n, x, inv):
    """`_find_interval`, with the uniform-grid guess computed with `inv = (n - 1)/(hi - lo)` (a multiplication
    instead of a division) and checked against the neighbouring intervals first. Same result for any grid."""

    lo, hi = grid[s], grid[s + n - 1]
    if not (lo <= x <= hi):
        return 0 if x < lo else n - 2
    if x == hi:
        return n - 2

    i = min(max(int((x - lo) * inv), 0), n - 2)
    if x < grid[s + i]:
        if (i > 0) and (x >= grid[s + i - 1]): return i - 1
    elif x < grid[s + i + 1]:
        return i
    elif (i < n - 2) and (x < grid[s + i + 2]):
        return i + 1
    return _find_interval(grid, s, n, x)


@njit(nogil = True)
def _axis_nodes(grid, s, n, x, stride, inv):
    """Flat offsets of the lower and upper node of `x` along one axis (both the single node for a length-one
    axis), and its normalised distance from the lower node, as in `_linear_regular_grid`."""

    if n == 1: return 0, 0, 0.0
    i = _find_interval_scaled(grid, s, n, x, inv)
    return i * stride, (i + 1) * stride, (x - grid[s + i]) / (grid[s + i + 1] - grid[s + i])


@njit(nogil = True)
def _inverse_spacings(grid, start, size):
    """(n - 1)/(hi - lo) of every axis (0 for length-one axes), for `_find_interval_scaled`."""

    inv = np.zeros(size.size)
    for k in range(size.size):
        if size[k] > 1: inv[k] = (size[k] - 1) / (grid[start[k] + size[k] - 1] - grid[start[k]])
    return inv


@njit(nogil = True)
def _outside_or_nan(grid, start, size, points, p):
    """(has a NaN coordinate, is outside the grid) for point `p`, as in `_linear_regular_grid`."""

    has_nan, outside = False, False
    for k in range(points.shape[1]):
        x = points[p, k]
        if x != x: has_nan = True
        if (x < grid[start[k]]) or (x > grid[start[k] + size[k] - 1]): outside = True
    return has_nan, outside


@njit(nogil = True)
def _linear_regular_grid_3d(grid, start, size, values, strides, points, fill, use_fill, out):
    """`_linear_regular_grid` for 3D tables, with the per-point work in registers. The corner weights are
    multiplied in dimension order and the corners summed in the same order, so the result is the same."""

    inv = _inverse_spacings(grid, start, size)
    s0, s1, s2 = start[0], start[1], start[2]
    n0, n1, n2 = size[0], size[1], size[2]
    lo0, lo1, lo2 = grid[s0], grid[s1], grid[s2]
    hi0, hi1, hi2 = grid[s0 + n0 - 1], grid[s1 + n1 - 1], grid[s2 + n2 - 1]

    for p in range(points.shape[0]):
        x0, x1, x2 = points[p, 0], points[p, 1], points[p, 2]
        if (x0 != x0) or (x1 != x1) or (x2 != x2):
            out[p] = np.nan
            continue
        if use_fill and ((x0 < lo0) or (x0 > hi0) or (x1 < lo1) or (x1 > hi1) or (x2 < lo2) or (x2 > hi2)):
            out[p] = fill
            continue

        L0, U0, y0 = _axis_nodes(grid, s0, n0, x0, strides[0], inv[0])
        L1, U1, y1 = _axis_nodes(grid, s1, n1, x1, strides[1], inv[1])
        L2, U2, y2 = _axis_nodes(grid, s2, n2, x2, strides[2], inv[2])
        m0, m1, m2 = 1 - y0, 1 - y1, 1 - y2
        a, b, c, d = m0 * m1, m0 * y1, y0 * m1, y0 * y1 #(1*w_0)*w_1 is exactly w_0*w_1

        #The 8 corners in the order of itertools.product (dimension 0 slowest)
        value = 0.0
        value = value + values[L0 + L1 + L2] * (a * m2)
        value = value + values[L0 + L1 + U2] * (a * y2)
        value = value + values[L0 + U1 + L2] * (b * m2)
        value = value + values[L0 + U1 + U2] * (b * y2)
        value = value + values[U0 + L1 + L2] * (c * m2)
        value = value + values[U0 + L1 + U2] * (c * y2)
        value = value + values[U0 + U1 + L2] * (d * m2)
        value = value + values[U0 + U1 + U2] * (d * y2)
        out[p] = value

    return out


@njit(nogil = True)
def _linear_regular_grid_4d(grid, start, size, values, strides, points, fill, use_fill, out):
    """`_linear_regular_grid` for 4D tables (eg. with one extra tabulated parameter); see `_linear_regular_grid_3d`."""

    inv = _inverse_spacings(grid, start, size)
    for p in range(points.shape[0]):
        has_nan, outside = _outside_or_nan(grid, start, size, points, p)
        if has_nan:
            out[p] = np.nan
            continue
        if outside and use_fill:
            out[p] = fill
            continue

        l0, u0, y0 = _axis_nodes(grid, start[0], size[0], points[p, 0], strides[0], inv[0])
        l1, u1, y1 = _axis_nodes(grid, start[1], size[1], points[p, 1], strides[1], inv[1])
        l2, u2, y2 = _axis_nodes(grid, start[2], size[2], points[p, 2], strides[2], inv[2])
        l3, u3, y3 = _axis_nodes(grid, start[3], size[3], points[p, 3], strides[3], inv[3])
        w0, w1, w2, w3 = (1 - y0, y0), (1 - y1, y1), (1 - y2, y2), (1 - y3, y3)
        f0, f1, f2, f3 = (l0, u0), (l1, u1), (l2, u2), (l3, u3)

        value = 0.0
        for b0 in range(2):
            for b1 in range(2):
                w01, f01 = w0[b0] * w1[b1], f0[b0] + f1[b1]
                for b2 in range(2):
                    w012, f012 = w01 * w2[b2], f01 + f2[b2]
                    for b3 in range(2):
                        value = value + values[f012 + f3[b3]] * (w012 * w3[b3])
        out[p] = value

    return out


@njit(nogil = True)
def _linear_curves(grid, start, size, values, strides, points, axis, fill, use_fill, out):
    """
    `_linear_regular_grid` (for tables of 3 or more dimensions) at every node of dimension `axis`: `out[p, i]`
    is the table at the coordinates `points[p]` (one per dimension other than `axis`, in order) with the
    coordinate of `axis` set to its i-th node. The intervals and weights of the other dimensions are found
    once per row, and so are, for every corner, the product of the weights of the dimensions before `axis`
    and the flat index of the other dimensions. The weights are still multiplied in dimension order and the
    corners summed in the same order as `_linear_regular_grid`, so the result is the same.
    """

    P      = points.shape[0]
    d      = points.shape[1] + 1
    s_ax   = start[axis]
    n_ax   = size[axis]
    n_c    = 1 << d
    n_post = d - 1 - axis
    lower  = np.empty(d, dtype = np.int64)
    upper  = np.empty(d, dtype = np.int64)
    y      = np.empty(d)
    ym     = np.empty(d)
    pre    = np.empty(n_c)                    #Product of the weights of dimensions before `axis`
    post   = np.empty((n_c, max(n_post, 1)))  #Weights of dimensions after `axis`
    base   = np.empty(n_c, dtype = np.int64)  #Flat index over the dimensions other than `axis`
    a_bit  = np.empty(n_c, dtype = np.int64)  #Whether the corner is at the upper node of `axis`

    for p in range(P):
        has_nan = False
        outside = False
        for k in range(d):
            if k == axis: continue
            x = points[p, k if k < axis else k - 1]
            s = start[k]
            n = size[k]
            if x != x: has_nan = True
            if (x < grid[s]) or (x > grid[s + n - 1]): outside = True
            if n == 1:
                lower[k], upper[k], y[k] = 0, 0, 0.0
            else:
                i = _find_interval(grid, s, n, x)
                lower[k], upper[k] = i, i + 1
                y[k] = (x - grid[s + i]) / (grid[s + i + 1] - grid[s + i])
            ym[k] = 1 - y[k]

        if has_nan or (outside and use_fill):
            for j in range(n_ax): out[p, j] = np.nan if has_nan else fill
            continue

        for c in range(n_c):
            weight = 1.0
            flat   = 0
            m      = 0
            for k in range(d):
                bit = (c >> (d - 1 - k)) & 1
                if k == axis:
                    a_bit[c] = bit
                    continue
                w     = y[k] if bit else ym[k]
                flat += (upper[k] if bit else lower[k]) * strides[k]
                if k < axis: weight = weight * w
                else:
                    post[c, m] = w
                    m += 1
            pre[c], base[c] = weight, flat

        for j in range(n_ax):
            #The node's interval, as _find_interval gives for x = grid[s_ax + j] (the last one for the top node)
            if n_ax == 1:
                i, y_ax = 0, 0.0
                i_up    = 0
            else:
                i    = min(j, n_ax - 2)
                i_up = i + 1
                y_ax = (grid[s_ax + j] - grid[s_ax + i]) / (grid[s_ax + i + 1] - grid[s_ax + i])
            ym_ax = 1 - y_ax

            value = 0.0
            for c in range(n_c):
                if a_bit[c]:
                    weight = pre[c] * y_ax
                    flat   = base[c] + i_up * strides[axis]
                else:
                    weight = pre[c] * ym_ax
                    flat   = base[c] + i * strides[axis]
                for m in range(n_post): weight = weight * post[c, m]
                value = value + values[flat] * weight
            out[p, j] = value

    return out


def _interpolate_curves(table, points, axis, n_threads = 1):
    """
    Evaluates `table` (a linear scipy `RegularGridInterpolator` of 3 or more dimensions) along all the nodes of
    dimension `axis`, for each row of `points` (the coordinates of the other dimensions, in order). Returns an
    array of shape (len(points), number of nodes of `axis`), equal to `_interpolate` at those points, or None
    if the table is not one that `_interpolate` evaluates itself. Rows are split over `n_threads` threads.
    """

    values = getattr(table, 'values', None)
    if not (isinstance(table, interpolate.RegularGridInterpolator) and (table.method == 'linear') and
            (not table.bounds_error) and (values is not None) and (values.dtype == np.float64) and
            (values.ndim == len(table.grid)) and (values.ndim >= 3)):
        return None

    pts    = np.ascontiguousarray(np.asarray(points, dtype = float).reshape(-1, len(table.grid) - 1))
    values = np.ascontiguousarray(values)
    grid   = np.concatenate([np.asarray(g, dtype = float) for g in table.grid])
    size   = np.array([len(g) for g in table.grid], dtype = np.int64)
    start  = np.concatenate([[0], np.cumsum(size)[:-1]]).astype(np.int64)
    fill   = np.nan if table.fill_value is None else float(table.fill_value)
    stride = np.array(values.strides, dtype = np.int64) // 8
    out    = np.empty((pts.shape[0], size[axis]))

    run    = lambda rows: _linear_curves(grid, start, size, values.ravel(), stride, pts[rows], axis, fill,
                                         table.fill_value is not None, out[rows])
    groups = [g for g in np.array_split(np.arange(pts.shape[0]), max(1, int(n_threads))) if g.size > 0]
    if len(groups) <= 1:
        run(slice(None))
    else:
        with ThreadPoolExecutor(max_workers = len(groups)) as executor: #The kernel releases the GIL
            list(executor.map(lambda g: run(slice(g[0], g[-1] + 1)), groups))

    return out


def _interpolate(table, points):
    """
    Evaluates `table` (a scipy `RegularGridInterpolator`) at `points`, a tuple with one array per dimension.
    Linear tables are evaluated with a numba kernel that reproduces scipy's result, without its per-call
    overhead and without holding the GIL (so runner threads can evaluate tables concurrently). Other tables
    are passed to scipy.
    """

    values = getattr(table, 'values', None)
    if not (isinstance(table, interpolate.RegularGridInterpolator) and (table.method == 'linear') and
            (not table.bounds_error) and (values is not None) and (values.dtype == np.float64) and
            (values.ndim == len(table.grid))):
        return table(points)

    pts    = np.stack(np.broadcast_arrays(*[np.asarray(p, dtype = float) for p in points]), axis = -1)
    shape  = pts.shape[:-1]
    pts    = np.ascontiguousarray(pts.reshape(-1, len(table.grid)))
    values = np.ascontiguousarray(values)
    grid   = np.concatenate([np.asarray(g, dtype = float) for g in table.grid])
    size   = np.array([len(g) for g in table.grid], dtype = np.int64)
    start  = np.concatenate([[0], np.cumsum(size)[:-1]]).astype(np.int64)
    fill   = np.nan if table.fill_value is None else float(table.fill_value)
    two_d  = (len(table.grid) == 2) and table.values.flags.writeable and (table.values.dtype.byteorder in ('=', '|'))
    stride = np.array(values.strides, dtype = np.int64) // 8
    use    = table.fill_value is not None
    out    = np.empty(pts.shape[0])
    if len(table.grid) == 3:   _linear_regular_grid_3d(grid, start, size, values.ravel(), stride, pts, fill, use, out)
    elif len(table.grid) == 4: _linear_regular_grid_4d(grid, start, size, values.ravel(), stride, pts, fill, use, out)
    else:                      _linear_regular_grid(grid, start, size, values.ravel(), stride, pts, fill, use, two_d, out)

    return out.reshape(shape)


def _pickle_without_Pk(obj, cosmologies):
    """
    Pickles `obj` while the P(k) caches of `cosmologies` (the CCL cosmologies it holds) are emptied, since
    those caches cannot be pickled, and puts the caches back afterwards. Worker processes recompute P(k)
    when needed, with the same result.
    """

    cosmologies = [c for c in cosmologies if c is not None]
    saved = [(c._pk_lin, c._pk_nl) for c in cosmologies]
    try:
        for c in cosmologies: destory_Pk(c)
        return pickle.dumps(obj)
    finally:
        for c, (lin, nl) in zip(cosmologies, saved): c._pk_lin, c._pk_nl = lin, nl


def _run_table_slices(payload, method, tasks):
    """
    Worker for `_map_table_slices`: unpickles the object and returns `[obj.<method>(*task) for task in tasks]`,
    with the warnings they raised (so the calling process can re-emit them).
    """

    obj = pickle.loads(payload)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter('always')
        out = [getattr(obj, method)(*task) for task in tasks]
    return out, [(str(w.message), w.category) for w in caught]


def _map_table_slices(obj, method, tasks, n_jobs, cosmologies, pbar = None):
    """
    `[obj.<method>(*task) for task in tasks]`, the slices of a table, in order. With `n_jobs > 1` they are
    computed by joblib (loky) worker processes, each on its own copy of `obj`, since computing a slice sets
    the model parameters. The copies are identical to `obj`, so the slices are the same as in serial.
    """

    n_jobs = 1 if n_jobs in (None, 0) else int(n_jobs)
    n_jobs = max(1, joblib.cpu_count() + 1 + n_jobs) if n_jobs < 0 else n_jobs
    n_jobs = min(n_jobs, len(tasks))

    if n_jobs <= 1:
        out = []
        for task in tasks:
            out.append(getattr(obj, method)(*task))
            if pbar is not None: pbar.update(1)
        return out

    try:
        payload = _pickle_without_Pk(obj, cosmologies)
    except Exception as e:
        warnings.warn(f"The table could not be sent to worker processes ({type(e).__name__}: {e}). "
                      "Building it serially instead.", UserWarning)
        return _map_table_slices(obj, method, tasks, 1, cosmologies, pbar)

    batches = [b for b in np.array_split(np.arange(len(tasks)), min(len(tasks), 4 * n_jobs)) if b.size > 0]
    results = joblib.Parallel(n_jobs = n_jobs, backend = 'loky', return_as = 'generator')(
                  joblib.delayed(_run_table_slices)(payload, method, [tasks[i] for i in b]) for b in batches)
    out = []
    for b, (values, caught) in zip(batches, results):
        out += values
        for message, category in caught: warnings.warn(message, category)
        if pbar is not None: pbar.update(b.size)
    return out


def _set_parameter(obj, key, value):
    """
    Recursively sets a parameter value for all attributes of an object that match a given key.

    The `_set_parameter` function is a utility to recursively search through all attributes of an object.
    If an attribute is a `HaloProfile` object and matches the specified key, this function sets its value
    to the provided value. This is particularly useful for updating configuration or parameter settings
    in complex objects with nested profiles.

    Parameters
    ----------
    obj : object
        The object whose attributes are to be searched. This object can contain nested attributes,
        some of which may be instances of `HaloProfile` or other objects.
    
    key : str
        The name of the attribute to search for within the object. If an attribute matches this name,
        its value will be set to the specified `value`.
    
    value : any
        The value to set for the attribute matching the `key`. This can be of any type, depending on
        the expected type of the attribute.

    Examples
    --------
    >>> class ExampleProfile:
    ...     def __init__(self):
    ...         self.param = 0
    ...         self.sub_profile = SomeHaloProfile()
    ...
    >>> profile = ExampleProfile()
    >>> _set_parameter(profile, 'param', 10)
    >>> print(profile.param)  # Output: 10

    Notes
    -----
    - This function checks all attributes of the given object. If an attribute matches the specified `key`,
      its value is updated. If an attribute is an instance of `HaloProfile`, the function calls itself
      recursively to check for the key in that profile.
    - The function uses the `setattr()` built-in function to set the attribute values dynamically.

    See Also
    --------
    `setattr` : Built-in function used to set the attribute of an object.

    """

    _set_parameter_recursive(obj, key, value, set())


def _set_parameter_recursive(obj, key, value, seen):

    #Some profiles hold references to themselves (eg. prof4params = self)
    #or share sub-profiles, so we track visited objects to avoid infinite recursion
    if id(obj) in seen: return
    seen.add(id(obj))

    obj_keys = dir(obj)

    for k in obj_keys:
        if k == key:
            setattr(obj, key, value)
        elif isinstance(getattr(obj, k), (ccl.halos.profiles.HaloProfile,)):
            _set_parameter_recursive(getattr(obj, k), key, value, seen)


def _record_parameters(obj, keys):
    """
    Records the current value of every attribute that `_set_parameter(obj, key, ...)` would modify,
    for each key in `keys`. Pass the output to `_restore_parameters` to undo those modifications.

    Returns
    -------
    records : list of (object, str, any)
        The (sub-)profile, the attribute name, and its current value.
    """

    records = []
    for key in keys: _record_parameter_recursive(obj, key, records, set())
    return records


def _record_parameter_recursive(obj, key, records, seen):

    #Same traversal as _set_parameter_recursive
    if id(obj) in seen: return
    seen.add(id(obj))

    for k in dir(obj):
        if k == key:
            records.append((obj, key, getattr(obj, key)))
        elif isinstance(getattr(obj, k), (ccl.halos.profiles.HaloProfile,)):
            _record_parameter_recursive(getattr(obj, k), key, records, seen)


def _restore_parameters(records):
    """Restores the attribute values saved by `_record_parameters`."""

    for obj, key, value in records: setattr(obj, key, value)


_NOT_FOUND = object()

def _get_parameter(obj, key):
    """
    Recursively searches an object to get the first instance of the entry with name "key". If
    there are multiple values then this function will not find them all.
    Parameters
    ----------
    obj : object
        The object whose attributes are to be searched. This object can contain nested attributes,
        some of which may be instances of `HaloProfile` or other objects.
    
    key : str
        The name of the attribute to search for within the object. If an attribute matches this name,
        its value will be returned.
    
    Notes
    -----
    - This function checks attributes of the given object. If the object itself has the attribute `key`, its value
      is returned. Otherwise, for every attribute that is an instance of `HaloProfile`, the function calls itself
      recursively to check for the key in that profile, and returns the first value found.
    See Also
    --------
    `getattr` : Built-in function used to get the attribute of an object.
    """

    res = _get_parameter_recursive(obj, key, set())
    return None if res is _NOT_FOUND else res


def _get_parameter_recursive(obj, key, seen):

    if id(obj) in seen: return _NOT_FOUND
    seen.add(id(obj))

    #The object's own attribute takes precedence over those of its sub-profiles. Otherwise
    #the alphabetical order of dir() decides, eg. DarkMatterBaryon.cutoff would be read from
    #its CollisionlessMatter's sub-profiles (which have their cutoff lifted to 1000).
    obj_keys = dir(obj)
    if key in obj_keys:
        return getattr(obj, key)
    for k in obj_keys:
        if isinstance(getattr(obj, k), (ccl.halos.profiles.HaloProfile,)):
            #Keep searching if this sub-profile does not have the key
            res = _get_parameter_recursive(getattr(obj, k), key, seen)
            if res is not _NOT_FOUND: return res

    return _NOT_FOUND

            
class TabulatedProfile(ccl.halos.profiles.HaloProfile):
    """
    A class for creating tabulated halo profiles from a given model.

    The `TabulatedProfile` class takes a profile model and generates tabulated profiles using the given cosmology
    and mass definition. It provides methods to set up interpolators for efficient profile evaluation across
    a range of redshifts, masses, and radii. This class is designed to handle both real-space and projected-space
    profiles.

    Parameters
    ----------
    model : object
        A profile model object that defines the real and projected halo profiles. This object should have `real()`
        and `projected()` methods for evaluating profiles.
    
    cosmo : object
        A `ccl.Cosmology` object representing the cosmological parameters.
    
    Attributes
    ----------    
    raw_input_3D : ndarray
        The raw 3D profile data used for setting up the interpolator.
    
    raw_input_2D : ndarray
        The raw 2D (projected) profile data used for setting up the interpolator.
    
    raw_input_z_range : ndarray
        The redshift range used in the interpolation, stored in log(1+z).
    
    raw_input_M_range : ndarray
        The mass range used in the interpolation, stored in log(M).
    
    raw_input_r_range : ndarray
        The radius range used in the interpolation, stored in log(r).
    
    interp3D : RegularGridInterpolator
        The interpolator for the 3D (real-space) profile.
    
    interp2D : RegularGridInterpolator
        The interpolator for the 2D (projected-space) profile.

    Methods
    -------
    setup_interpolator(z_min=1e-2, z_max=5, N_samples_z=30, z_linear_sampling=False,
                       M_min=1e12, M_max=1e16, N_samples_Mass=30,
                       R_min=1e-3, R_max=1e2, N_samples_R=100,
                       other_params={}, verbose=True)
        Sets up the interpolators for the 3D and 2D profiles based on the specified parameter ranges.
    
    _readout(r, M, a, table)
        Evaluates the profile from the interpolation table for given radii, masses, and scale factors.
    
    _real(cosmo, r, M, a)
        Computes the real-space profile using the tabulated interpolator.
    
    _projected(cosmo, r, M, a)
        Computes the projected-space profile using the tabulated interpolator.

    Examples
    --------
    >>> model = SomeProfileModel()
    >>> cosmo = ccl.Cosmology(...)
    >>> profile = TabulatedProfile(model, cosmo)
    >>> profile.setup_interpolator()
    >>> real_profile = profile.real(cosmo, r, M, a)
    >>> projected_profile = profile.projected(cosmo, r, M, a)

    Notes
    -----
    - The `setup_interpolator()` method must be called before using `real()` and `projected()` methods to
      initialize the interpolation tables.
    - The interpolators are set up using log-scaled grids for mass, radius, and redshift to efficiently handle
      a wide range of scales.
    - This class inherits from `ccl.halos.profiles.HaloProfile` and can be used in contexts where a halo profile
      is required.

    """

    def __init__(self, model, cosmo):

        self.model    = model
        self.cosmo    = cosmo #CCL cosmology instance

        #We just set this to the same as the inputted profile.
        super().__init__(mass_def = model.mass_def)

        self.update_precision_fftlog(**self.model.precision_fftlog.to_dict())
    
    def __str_prf__(self):

        return f"Tabulated[{self.model.__str_prf__()}"
    
    def __str_par__(self): return self.model.__str_par__()

    def _table_slice(self, r, M, a):
        """One redshift slice of the tables: the 3D and projected profiles at scale factor `a`."""

        return self.model.real(self.cosmo, r, M, a), self.model.projected(self.cosmo, r, M, a)


    def setup_interpolator(self, z_min = 1e-2, z_max = 5, N_samples_z = 30, z_linear_sampling = False,
                           M_min = 1e12, M_max = 1e16, N_samples_Mass = 30,
                           R_min = 1e-3, R_max = 1e2,  N_samples_R = 100,
                           other_params = {}, verbose = True, n_jobs = 1):

        """
        Sets up the interpolators for the 3D and 2D profiles based on the specified parameter ranges.

        This method generates tabulated profiles over specified ranges of redshift, mass, and radius.
        The profiles are stored in 3D and 2D interpolators for efficient profile evaluation. Can be
        read out using either the `_readout()` helper class, or the `real()` and `projected()` functions.

        Parameters
        ----------
        z_min : float, optional
            The minimum redshift value for the tabulation. Default is 1e-2.
        
        z_max : float, optional
            The maximum redshift value for the tabulation. Default is 5.
        
        N_samples_z : int, optional
            The number of redshift samples. Default is 30.
        
        z_linear_sampling : bool, optional
            If `True`, use linear sampling for redshift; otherwise, use logarithmic sampling. Default is `False`.
        
        M_min : float, optional
            The minimum mass value for the tabulation. Default is 1e12.
        
        M_max : float, optional
            The maximum mass value for the tabulation. Default is 1e16.
        
        N_samples_Mass : int, optional
            The number of mass samples. Default is 30.
        
        R_min : float, optional
            The minimum radius value for the tabulation. Default is 1e-3.
        
        R_max : float, optional
            The maximum radius value for the tabulation. Default is 1e2.
        
        N_samples_R : int, optional
            The number of radius samples. Default is 100.
        
        other_params : dict, optional
            Additional parameters for the profile model. Default is an empty dictionary.
        
        verbose : bool, optional
            If `True`, display a progress bar during the tabulation process. Default is `True`.

        n_jobs : int, optional
            Number of worker processes (joblib/loky) computing the redshift slices of the table. Default is 1
            (serial); -1 uses all available cores. The table does not depend on `n_jobs`.

        """

        M_range  = np.geomspace(M_min, M_max, N_samples_Mass)
        r        = np.geomspace(R_min, R_max, N_samples_R)
        z_range  = np.linspace(z_min, z_max, N_samples_z) if z_linear_sampling else np.geomspace(z_min, z_max, N_samples_z)

        interp3D = np.zeros([z_range.size, M_range.size, r.size])
        interp2D = np.zeros([z_range.size, M_range.size, r.size])

        with tqdm(total = z_range.size, desc = 'Building Table', disable = not verbose) as pbar:
            tasks  = [(r, M_range, 1/(1 + z_range[j])) for j in range(z_range.size)]
            slices = _map_table_slices(self, '_table_slice', tasks, n_jobs, [self.cosmo], pbar)
        for j, (prof3D, prof2D) in enumerate(slices):
            interp3D[j, :, :] = prof3D
            interp2D[j, :, :] = prof2D

        input_grid_1 = (np.log(1 + z_range), np.log(M_range), np.log(r))

        self.raw_input_3D = interp3D
        self.raw_input_2D = interp2D
        self.raw_input_z_range = np.log(1 + z_range)
        self.raw_input_M_range = np.log(M_range)
        self.raw_input_r_range = np.log(r)
        
        self.interp3D = interpolate.RegularGridInterpolator(input_grid_1, np.log(interp3D), bounds_error = False)
        self.interp2D = interpolate.RegularGridInterpolator(input_grid_1, np.log(interp2D), bounds_error = False)

        #Once all tabulation is done, we don't need to keep P(k) calculations in cosmology object.
        #This is good because the Pk class is not pickleable, so by destorying it here we
        #are able to keep this class pickleable.
        self.cosmo = destory_Pk(self.cosmo)


    def _readout(self, r, M, a, table):
        """
        Evaluates the profile from the interpolation table for given radii, masses, and scale factors.

        This method reads out values from a pre-computed interpolation table.

        Parameters
        ----------
        r : array_like
            The radii at which to evaluate the profile.
        
        M : array_like
            The masses for which to evaluate the profile.
        
        a : array_like
            The scale factors corresponding to the redshifts for profile evaluation.
        
        table : RegularGridInterpolator
            The interpolator object containing the tabulated profile data.

        Returns
        -------
        prof : ndarray
            The profile values evaluated at the given radii, masses, and scale factors.
        """
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        
        prof  = np.zeros([M_use.size, r_use.size])
        empty = np.ones_like(r_use)
        z_in  = np.log(1/a)*empty #This is log(1 + z)
        r_in  = np.log(r_use)
        
        for i in range(M_use.size):
            M_in  = np.log(M_use[i])*empty

            prof[i] = _interpolate(table, (z_in, M_in, r_in, ))
            prof[i] = np.exp(prof[i])
            
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
            
        return prof


    def _readout_batch(self, r, halo, M, a, table):
        """
        Profiles of many halos in a single read of `table`, as used by the runners.

        `M` and `a` hold one entry per halo, and `halo` gives, for every radius in `r`, the index of the halo
        it belongs to. The result has the shape of `r`, and each entry equals `_readout(r[i], M[halo[i]],
        a[halo[i]], table)` up to floating-point rounding.
        """

        if not (hasattr(self, 'interp3D') & hasattr(self, 'interp2D')):
            raise NameError("No Table created. Run setup_interpolator() method first")

        r    = np.asarray(r, dtype = float)
        halo = np.asarray(halo, dtype = int)
        M    = np.atleast_1d(np.asarray(M, dtype = float))
        a    = np.broadcast_to(np.asarray(a, dtype = float), M.shape)
        if r.size == 0: return np.zeros(0)

        return np.exp(_interpolate(table, (np.log(1/a)[halo], np.log(M)[halo], np.log(r))))


    def _real_batch(self, r, halo, M, a):      return self._readout_batch(r, halo, M, a, getattr(self, 'interp3D', None))
    def _projected_batch(self, r, halo, M, a): return self._readout_batch(r, halo, M, a, getattr(self, 'interp2D', None))


    def _real(self, cosmo, r, M, a):
        """
        Computes the real-space profile using the tabulated interpolator.

        Parameters
        ----------
        cosmo : object
            A `ccl.Cosmology` object representing the cosmological parameters.

        r : array_like
            The radii at which to compute the profile.

        M : float or array_like
            The mass of the halo.

        a : float or array_like
            The scale factor at which to compute the profile.

        Returns
        -------
        prof : ndarray
            The real-space profile values evaluated at the given radii, masses, and scale factors.
        """
        
        if not (hasattr(self, 'interp3D') & hasattr(self, 'interp2D')):
            raise NameError("No Table created. Run setup_interpolator() method first")

        prof = self._readout(r, M, a, self.interp3D)
        
        return prof
    
    
    def _projected(self, cosmo, r, M, a):
        """
        Computes the projected-space profile using the tabulated interpolator.

        Parameters
        ----------
        cosmo : object
            A `ccl.Cosmology` object representing the cosmological parameters.
        
        r : array_like
            The radii at which to compute the profile.
        
        M : float or array_like
            The mass of the halo.
        
        a : float or array_like
            The scale factor at which to compute the profile.
        
        Returns
        -------
        prof : ndarray
            The projected-space profile values evaluated at the given radii, masses, and scale factors.
        """
        
        if not (hasattr(self, 'interp3D') & hasattr(self, 'interp2D')):
            raise NameError("No Table created. Run setup_interpolator() method first")

        prof = self._readout(r, M, a, self.interp2D)
        
        return prof
    

    
class ParamTabulatedProfile(object):
    """
    A class for creating tabulated halo profiles that depend on additional parameters.

    The `ParamTabulatedProfile` class takes a profile model and tabulates its output as a function of
    halo mass, redshift, and additional parameters specified during initialization. This allows for
    flexible interpolation of profiles based on various physical properties of halos.

    Parameters
    ----------
    model : object
        A profile model object that defines the real and projected halo profiles. This object should have `real()`
        and `projected()` methods for evaluating profiles.
    
    cosmo : object
        A `ccl.Cosmology` object representing the cosmological parameters.

    Attributes
    ----------
    model : object
        The profile model used for generating tabulated profiles.

    cosmo : object
        The cosmology instance used for the profile calculations.

    mass_def : object
        The mass definition used for the profile calculations (taken from `model`).
    
    p_keys : list of str
        The list of parameter keys used in the profile model.
    
    raw_input_3D : ndarray
        The raw 3D profile data used for setting up the interpolator.
    
    raw_input_2D : ndarray
        The raw 2D (projected) profile data used for setting up the interpolator.
    
    interp3D : RegularGridInterpolator
        The interpolator for the 3D (real-space) profile.
    
    interp2D : RegularGridInterpolator
        The interpolator for the 2D (projected-space) profile.

    Methods
    -------
    setup_interpolator(z_min=1e-2, z_max=5, N_samples_z=30, z_linear_sampling=False,
                       M_min=1e12, M_max=1e16, N_samples_Mass=30,
                       R_min=1e-3, R_max=1e2, N_samples_R=100,
                       other_params={}, verbose=True)
        Sets up the interpolators for the 3D and 2D profiles based on the specified parameter ranges.
    
    _readout(r, M, a, table, **kwargs)
        Evaluates the profile from the interpolation table for given radii, masses, scale factors, and other parameters.
    
    real(cosmo, r, M, a, **kwargs)
        Computes the real-space profile using the tabulated interpolator.
    
    projected(cosmo, r, M, a, **kwargs)
        Computes the projected-space profile using the tabulated interpolator.

    Examples
    --------
    >>> model = SomeProfileModel()
    >>> cosmo = ccl.Cosmology(...)
    >>> profile = ParamTabulatedProfile(model, cosmo)
    >>> profile.setup_interpolator(other_params={'param1': np.array([0.1, 0.2, 0.3])})
    >>> real_profile = profile.real(cosmo, r, M, a, param1=0.2)
    >>> projected_profile = profile.projected(cosmo, r, M, a, param1=0.2)

    Notes
    -----
    - The `setup_interpolator()` method must be called before using `real()` and `projected()` methods to
      initialize the interpolation tables.
    - The class allows for parameterizing profiles over additional user-defined parameters (`other_params`).
    - This class is not compatible with `TabulatedProfile` objects; ensure that the input model is not an instance
      of `TabulatedProfile`.
    """

    
    def __init__(self, model, cosmo):
        """
        Initializes the ParamTabulatedProfile class with a given model, cosmology, and mass definition.

        Parameters
        ----------
        model : object
            A profile model object that defines the real and projected halo profiles. This object should have `real()`
            and `projected()` methods for evaluating profiles.
        
        cosmo : object
            A `ccl.Cosmology` object representing the cosmological parameters.
        """

        self.model    = model
        self.cosmo    = cosmo #CCL cosmology instance
        
        assert not isinstance(model, TabulatedProfile), "Input model cannot be 'TabulatedProfile' object."

        
    def _table_slice(self, r, M, a, keys, values):
        """One slice of the tables: the 3D and projected profiles at scale factor `a`, with the model's
        parameters `keys` set to `values`."""

        #Modify the model input params so that they are run with the right parameters
        for k, v in zip(keys, values): _set_parameter(self.model, k, v)

        return self.model.real(self.cosmo, r, M, a), self.model.projected(self.cosmo, r, M, a)


    def setup_interpolator(self, z_min = 1e-2, z_max = 5, N_samples_z = 30, z_linear_sampling = False,
                           M_min = 1e12, M_max = 1e16, N_samples_Mass = 30,
                           R_min = 1e-3, R_max = 1e2,  N_samples_R = 100,
                           other_params = {}, verbose = True, n_jobs = 1):
        """
        Sets up the interpolators for the 3D and 2D profiles based on the specified parameter ranges.

        This method generates tabulated profiles over specified ranges of redshift, mass, radius, and additional
        user-defined parameters. The profiles are stored in 3D and 2D interpolators for efficient profile evaluation.

        Parameters
        ----------
        z_min : float, optional
            The minimum redshift value for the tabulation. Default is 1e-2.
        
        z_max : float, optional
            The maximum redshift value for the tabulation. Default is 5.
        
        N_samples_z : int, optional
            The number of redshift samples. Default is 30.
        
        z_linear_sampling : bool, optional
            If `True`, use linear sampling for redshift; otherwise, use logarithmic sampling. Default is `False`.
        
        M_min : float, optional
            The minimum mass value for the tabulation. Default is 1e12.
        
        M_max : float, optional
            The maximum mass value for the tabulation. Default is 1e16.
        
        N_samples_Mass : int, optional
            The number of mass samples. Default is 30.
        
        R_min : float, optional
            The minimum radius value for the tabulation. Default is 1e-3.
        
        R_max : float, optional
            The maximum radius value for the tabulation. Default is 1e2.
        
        N_samples_R : int, optional
            The number of radius samples. Default is 100.
        
        other_params : dict, optional
            A dictionary of other parameters to be tabulated. The keys are parameter names, and the values are
            arrays (or lists) of parameter values. Default is an empty dictionary. The model's parameters are
            set to these values while tabulating, and restored to their original values afterwards.

        verbose : bool, optional
            If `True`, display a progress bar during the tabulation process. Default is `True`.

        n_jobs : int, optional
            Number of worker processes (joblib/loky) computing the (redshift, parameter) slices of the table.
            Default is 1 (serial); -1 uses all available cores. The table does not depend on `n_jobs`.

        """

        M_range  = np.geomspace(M_min, M_max, N_samples_Mass)
        r        = np.geomspace(R_min, R_max, N_samples_R)
        z_range  = np.linspace(z_min, z_max, N_samples_z) if z_linear_sampling else np.geomspace(z_min, z_max, N_samples_z)

        other_params = {k : np.atleast_1d(np.asarray(v, dtype = float)) for k, v in other_params.items()} #Allow lists/tuples
        p_keys   = list(other_params.keys()); setattr(self, 'p_keys', p_keys)
        interp3D = np.zeros([z_range.size, M_range.size, r.size] + [other_params[k].size for k in p_keys]) + np.nan
        interp2D = np.zeros([z_range.size, M_range.size, r.size] + [other_params[k].size for k in p_keys]) + np.nan

        #If other_params is empty then iterator will be empty and the code still works fine
        iterator = [p for p in product(*[np.arange(other_params[k].size) for k in p_keys])]

        #The slices below change the model's parameters. Save them, so the model is returned unchanged.
        original_params = _record_parameters(self.model, p_keys)

        #Loop over params to build table
        tasks = [(r, M_range, 1/(1 + z_range[j]), p_keys, [other_params[p_keys[k_i]][c[k_i]] for k_i in range(len(p_keys))])
                 for j in range(z_range.size) for c in iterator]
        with tqdm(total = interp3D.size//(M_range.size*r.size), desc = 'Building Table', disable = not verbose) as pbar:
            slices = _map_table_slices(self, '_table_slice', tasks, n_jobs, [self.cosmo], pbar)

        _restore_parameters(original_params)

        for (j, c), (prof3D, prof2D) in zip([(j, c) for j in range(z_range.size) for c in iterator], slices):
            index = tuple([j, slice(None), slice(None)] + list(c)) #Build a custom index into the array
            interp3D[index] = prof3D
            interp2D[index] = prof2D


        input_grid_1 = tuple([np.log(1 + z_range), np.log(M_range), np.log(r)] + [other_params[k] for k in p_keys])

        self.raw_input_3D = interp3D
        self.raw_input_2D = interp2D
        self.raw_input_z_range = np.log(1 + z_range)
        self.raw_input_M_range = np.log(M_range)
        self.raw_input_r_range = np.log(r)
        for k in other_params.keys(): setattr(self, 'raw_input_%s_range' % k, other_params[k]) #Save other raw inputs too
        
        self.interp3D = interpolate.RegularGridInterpolator(input_grid_1, np.log(interp3D), bounds_error = False)
        self.interp2D = interpolate.RegularGridInterpolator(input_grid_1, np.log(interp2D), bounds_error = False)

        #Once all tabulation is done, we don't need to keep P(k) calculations in cosmology object.
        #This is good because the Pk class is not pickleable, so by destorying it here we
        #are able to keep this class pickleable.
        self.cosmo = destory_Pk(self.cosmo)


    def _readout(self, r, M, a, table, **kwargs):
        """
        Evaluates the profile from the interpolation table for given radii, masses, scale factors, and other parameters.

        This method reads out values from a pre-computed interpolation table.

        Parameters
        ----------
        r : array_like
            The radii at which to evaluate the profile.
        
        M : array_like
            The masses for which to evaluate the profile.
        
        a : array_like
            The scale factors corresponding to the redshifts for profile evaluation.
        
        table : RegularGridInterpolator
            The interpolator object containing the tabulated profile data.
        
        **kwargs
            Additional parameters to be used in the profile evaluation.

        Returns
        -------
        prof : ndarray
            The profile values evaluated at the given radii, masses, scale factors, and other parameters.
        """
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        
        prof  = np.zeros([M_use.size, r_use.size])
        empty = np.ones_like(r_use)
        z_in  = np.log(1/a)*empty #This is log(1 + z)
        r_in  = np.log(r_use)
        extra = [k for k in kwargs.keys() if k not in self.p_keys]
        if len(extra) > 0:
            raise ValueError(f"Parameters {extra} were passed, but the table was only built with {self.p_keys}.")
        k_in  = [kwargs[k] * empty for k in self.p_keys] #Same order as the table axes, not the kwargs order
        
        for i in range(M_use.size):
            M_in  = np.log(M_use[i])*empty
            p_in  = tuple([z_in, M_in, r_in] + k_in)
            prof[i] = _interpolate(table, p_in)
            prof[i] = np.exp(prof[i])
            
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
            
        return prof


    def _readout_batch(self, r, halo, M, a, table, **kwargs):
        """
        Profiles of many halos in a single read of `table`, as used by the runners.

        `M`, `a` and the values of `kwargs` hold one entry per halo, and `halo` gives, for every radius in
        `r`, the index of the halo it belongs to. The result has the shape of `r`, and each entry equals
        `_readout(r[i], M[halo[i]], a[halo[i]], table, ...)` up to floating-point rounding.
        """

        if not (hasattr(self, 'interp3D') & hasattr(self, 'interp2D')):
            raise NameError("No Table created. Run setup_interpolator() method first")
        for k in self.p_keys:
            assert k in kwargs.keys(), "Need to provide %s as input. Table was built with this." % k
        extra = [k for k in kwargs.keys() if k not in self.p_keys]
        if len(extra) > 0:
            raise ValueError(f"Parameters {extra} were passed, but the table was only built with {self.p_keys}.")

        r    = np.asarray(r, dtype = float)
        halo = np.asarray(halo, dtype = int)
        M    = np.atleast_1d(np.asarray(M, dtype = float))
        a    = np.broadcast_to(np.asarray(a, dtype = float), M.shape)
        if r.size == 0: return np.zeros(0)

        k_in = [np.broadcast_to(np.asarray(kwargs[k], dtype = float), M.shape)[halo] for k in self.p_keys] #Same order as the table axes
        return np.exp(_interpolate(table, tuple([np.log(1/a)[halo], np.log(M)[halo], np.log(r)] + k_in)))


    def _real_batch(self, r, halo, M, a, **kwargs):
        return self._readout_batch(r, halo, M, a, getattr(self, 'interp3D', None), **kwargs)

    def _projected_batch(self, r, halo, M, a, **kwargs):
        return self._readout_batch(r, halo, M, a, getattr(self, 'interp2D', None), **kwargs)


    def real(self, cosmo, r, M, a, **kwargs):
        """
        Computes the real-space profile using the tabulated interpolator.

        Parameters
        ----------
        cosmo : object
            A `ccl.Cosmology` object representing the cosmological parameters.
            It's not actually used, but we allow it as input to have consistent
            API with the CCL profile methods.
        
        r : array_like
            The radii at which to compute the profile.
        
        M : float or array_like
            The mass of the halo.
        
        a : float or array_like
            The scale factor at which to compute the profile.
                
        **kwargs
            Additional parameters required for the profile evaluation.

        Returns
        -------
        prof : ndarray
            The real-space profile values evaluated at the given radii, masses, and scale factors.
        """
        
        if not (hasattr(self, 'interp3D') & hasattr(self, 'interp2D')):
            raise NameError("No Table created. Run setup_interpolator() method first")
        
        for k in self.p_keys:
            assert k in kwargs.keys(), "Need to provide %s as input into `real'. Table was built with this." % k
        
        prof = self._readout(r, M, a, self.interp3D, **kwargs)
        
        return prof
    
    
    def projected(self, cosmo, r, M, a, **kwargs):
        """
        Computes the projected-space profile using the tabulated interpolator.

        Parameters
        ----------
        cosmo : object
            A `ccl.Cosmology` object representing the cosmological parameters.
            It's not actually used, but we allow it as input to have consistent
            API with the CCL profile methods.
        
        r : array_like
            The radii at which to compute the profile.
        
        M : float or array_like
            The mass of the halo.
        
        a : float or array_like
            The scale factor at which to compute the profile.
                
        **kwargs
            Additional parameters required for the profile evaluation.

        Returns
        -------
        prof : ndarray
            The projected-space profile values evaluated at the given radii, masses, and scale factors.
        """
        
        if not (hasattr(self, 'interp3D') & hasattr(self, 'interp2D')):
            raise NameError("No Table created. Run setup_interpolator() method first")
        
        for k in self.p_keys:
            assert k in kwargs.keys(), "Need to provide %s as input into `projected'. Table was built with this." % k
        
        prof = self._readout(r, M, a, self.interp2D, **kwargs)
        
        return prof


class TabulatedCorrelation3D(object):

    
    def __init__(self, cosmo, R_range = [1e-3, 1e3], N_samples = 500):
        

        self.cosmo     = cosmo
        self.R_range   = R_range
        self.N_samples = N_samples
                
        
    def setup_interpolator(self, z_min = 0, z_max = 5, N_samples_z = 10, verbose = False):
        
        
        r    = np.geomspace(self.R_range[0], self.R_range[1], self.N_samples)
        z_range  = np.linspace(z_min, z_max, N_samples_z)
        
        interp3D = np.zeros([z_range.size, r.size]) + np.NaN
        
        #Loop over params to build table
        with tqdm(total = z_range.size, desc = 'Building Table', disable = not verbose) as pbar:
            for j in range(z_range.size):
                
                a = 1/(1 + z_range[j])
                interp3D[j, :] = ccl.correlation_3d(self.cosmo, r = r, a = a)
                
                pbar.update(1)
        
        input_grid_1 = (np.log(1 + z_range), np.log(r))

        self.raw_input_3D = interp3D
        self.raw_input_z_range = np.log(1 + z_range)
        self.raw_input_r_range = np.log(r)
        
        self.interp3D = interpolate.RegularGridInterpolator(input_grid_1, np.log(interp3D), bounds_error = False)
        
        
    def __call__(self, r, a):
        
        r_use = np.atleast_1d(r)
        
        empty = np.ones_like(r_use)
        z_in  = np.log(1/a)*empty #This is log(1 + z)
        r_in  = np.log(r_use)
        
        ln_xi = self.interp3D( (z_in, r_in) )
        xi    = np.exp(ln_xi)
        
        return xi
        
        