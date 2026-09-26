import numpy as np
import numba
from contextlib import contextmanager
from numba import njit, prange
from scipy.spatial import KDTree
from tqdm import tqdm
from ..utils.misc import _default_mass_def, _runner_cosmology, _check_p_keys, _halo_radius, _batch_method
from ..utils.Tabulate import _find_interval

__all__ = ['DefaultRunnerSnapshot', 'BaryonifySnapshot']


@njit(parallel = True)
def _cell_ids(x, y, z, L, n):
    """Cell `(ix*ny + iy)*nz + iz` of every particle, in a periodic grid of `n = (nx, ny, nz)` cells over [0, L)^3."""

    cell = np.empty(x.size, dtype = np.int64)
    for i in prange(x.size):
        ix = min(int((x[i] % L) / L * n[0]), n[0] - 1)
        iy = min(int((y[i] % L) / L * n[1]), n[1] - 1)
        iz = min(int((z[i] % L) / L * n[2]), n[2] - 1)
        cell[i] = (ix*n[1] + iy)*n[2] + iz

    return cell


@njit
def _counting_sort(cell, starts):
    """Particle indices sorted by cell (stable), given `starts`, the cumulative counts of the cells."""

    order = np.empty(cell.size, dtype = np.int64)
    fill  = starts[:-1].copy()
    for i in range(cell.size):
        order[fill[cell[i]]] = i
        fill[cell[i]] += 1

    return order


@njit(parallel = True)
def _gather(values, order):
    """`values[order]`, in parallel (`values` may be a strided view, eg. a field of a structured array)."""

    out = np.empty(order.size)
    for i in prange(order.size): out[i] = values[order[i]]
    return out


@njit(parallel = True)
def _shift_and_wrap(values, offsets, L, out):
    """`out = np.mod(values + offsets, L)`, in parallel (numba's float `%` matches np.mod)."""

    for i in prange(values.size): out[i] = (values[i] + offsets[i]) % L
    return out


@njit(parallel = True)
def _copy(values, out):
    """`out[:] = values`, in parallel (either may be a strided view)."""

    for i in prange(values.size): out[i] = values[i]
    return out


def _build_cell_index(x, y, z, L, n):
    """
    Sorts particles into a periodic grid of `n = (nx, ny, nz)` cells over the box [0, L)^3 (a stable counting
    sort). Returns `order`, the particle indices sorted by cell, and `starts`, such that the particles of cell
    `c = (ix*ny + iy)*nz + iz` are `order[starts[c]:starts[c+1]]`.
    """

    cell   = _cell_ids(x, y, z, L, n)
    starts = np.concatenate([[0], np.cumsum(np.bincount(cell, minlength = int(np.prod(n))))]).astype(np.int64)

    return _counting_sort(cell, starts), starts


@njit
def _cell_range(center, R, L, n):
    """
    First and last (unwrapped) cell overlapping [center - R, center + R], covering at most `n` cells. The range
    is padded by 1e-9 of a cell, so a particle on a cell boundary is found whichever way its cell was rounded.
    """

    lo = int(np.floor((center - R) / L * n - 1e-9))
    hi = int(np.floor((center + R) / L * n + 1e-9))
    if hi - lo + 1 > n: return 0, n - 1
    return lo, hi


@njit
def _periodic(dx, L):
    """Same as `DefaultRunnerSnapshot.enforce_periodicity`, for one value."""

    dx = dx - L if dx >  L/2 else dx
    dx = dx + L if dx < -L/2 else dx
    return dx


@njit(parallel = True)
def _snapshot_offsets(xs, ys, zs, order, starts, n, L, centers, R_q, R_cut, shift, nodes, curves,
                      plane_start, plane_halo, plane_order, out):
    """
    Adds the displacement of every halo to the particles within its query radius `R_q`. Particles are sorted by
    cell (`xs`, `ys`, `zs`, `starts`), and `out` is in the original particle order (`order` maps sorted positions
    to it). The work is split by planes of cells along x, each handled by one thread
    (`plane_halo[plane_start[ix]:plane_start[ix+1]]` are the halos overlapping plane `ix`). A thread only
    writes to the particles of its planes, and adds the halos in catalog order, so the result does not depend
    on the number of threads.

    The displacement of halo `h` at comoving distance `d` is the linear interpolation of `curves[h]` over
    `nodes` at `log(d) - shift[h]`, zero for `d >= R_cut[h]` and for non-finite values; it is applied along
    the (periodic) separation vector, as in `BaryonifySnapshot.process`.
    """

    nx, ny, nz = n[0], n[1], n[2]
    n_nodes    = nodes.size

    for k in prange(nx):
        ix = plane_order[k]
        for q in range(plane_start[ix], plane_start[ix + 1]):
            h  = plane_halo[q]
            cx, cy, cz = centers[h, 0], centers[h, 1], centers[h, 2]
            R  = R_q[h]
            y_lo, y_hi = _cell_range(cy, R, L, ny)
            z_lo, z_hi = _cell_range(cz, R, L, nz)

            for jy in range(y_lo, y_hi + 1):
                iy = jy % ny
                for jz in range(z_lo, z_hi + 1):
                    iz = jz % nz
                    c  = (ix*ny + iy)*nz + iz
                    for p in range(starts[c], starts[c + 1]):
                        dx = _periodic(xs[p] - cx, L)
                        dy = _periodic(ys[p] - cy, L)
                        dz = _periodic(zs[p] - cz, L)
                        d  = np.sqrt((dx*dx + dy*dy) + dz*dz)
                        if not (d <= R) or not (d < R_cut[h]) or not (d > 0): continue

                        u = np.log(d) - shift[h]
                        if not ((u >= nodes[0]) and (u <= nodes[n_nodes - 1])): continue #Outside the table: NaN -> 0
                        i    = _find_interval(nodes, 0, n_nodes, u)
                        t    = (u - nodes[i]) / (nodes[i + 1] - nodes[i])
                        disp = curves[h, i]*(1 - t) + curves[h, i + 1]*t
                        if not np.isfinite(disp): continue

                        o = order[p]
                        out[o, 0] += disp * (dx/d)
                        out[o, 1] += disp * (dy/d)
                        out[o, 2] += disp * (dz/d)

    return out


@njit
def _ball(xs, ys, zs, starts, n, L, cx, cy, cz, R):
    """
    Particles (positions in the cell-sorted arrays) within distance `R` of (cx, cy, cz), with their periodic
    separations and distances, in cell order.
    """

    nx, ny, nz = n[0], n[1], n[2]
    x_lo, x_hi = _cell_range(cx, R, L, nx)
    y_lo, y_hi = _cell_range(cy, R, L, ny)
    z_lo, z_hi = _cell_range(cz, R, L, nz)

    count = 0
    for sweep in range(2): #First count, then fill
        if sweep == 1:
            idx = np.empty(count, dtype = np.int64)
            sep = np.empty((count, 4))
            count = 0
        for jx in range(x_lo, x_hi + 1):
            for jy in range(y_lo, y_hi + 1):
                for jz in range(z_lo, z_hi + 1):
                    c = ((jx % nx)*ny + (jy % ny))*nz + (jz % nz)
                    for p in range(starts[c], starts[c + 1]):
                        dx = _periodic(xs[p] - cx, L)
                        dy = _periodic(ys[p] - cy, L)
                        dz = _periodic(zs[p] - cz, L)
                        d  = np.sqrt((dx*dx + dy*dy) + dz*dz)
                        if not (d <= R): continue
                        if sweep == 1:
                            idx[count] = p
                            sep[count, 0], sep[count, 1], sep[count, 2], sep[count, 3] = dx, dy, dz, d
                        count += 1

    return idx, sep


class DefaultRunnerSnapshot(object):
    """
    A utility class for handling input/output operations related to HaloNDCatalogs and particle snapshots.

    The `DefaultRunnerSnapshot` class provides methods to manage and process data associated with halo ND catalogs
    and particle snapshots, including distance calculations with periodic boundary conditions.

    Particles are found around each halo with a periodic grid of cells: the particles are sorted by cell once,
    on the first call to `process()` (a few seconds for ~10^8 particles), and reused by later calls, eg. when
    swapping in a different `model`.

    Parameters
    ----------
    HaloNDCatalog : object
        An instance of a `HaloNDCatalog`, containing data about halos and their properties. It must have a
        `cosmology` attribute to specify the cosmological parameters.

    ParticleSnapshot : object
        An instance representing a `ParticleSnapshot` containing the positions of particles
        in the simulation.

    epsilon_max : float
        A parameter specifying the maximum size, in units of halo radius, of cutouts made around
        each halo during painting/baryonification.

    model : object, optional
        An object that generates profiles or displacements. For example, see `Baryonification2D` or `Pressure`

    mass_def : object, optional
        An instance of a mass definition object from the CCL (Core Cosmology Library), specifying the
        mass definition to be used. Default is None, in which case the mass definition of `model` is used (or 200c, if `model` has none).

    verbose : bool, optional
        A flag to enable verbose output for logging or debugging purposes. Default is True.

    KDTree_kwargs : dict, optional
        Arguments for the `scipy.spatial.KDTree` of the particles, which is only built if the `tree`
        attribute is accessed. The runners themselves use the grid of cells described above.

    n_jobs : int, optional
        Number of threads used when the model is a tabulated displacement (eg. `Baryonification3D`).
        Default is 1 (serial). Use -1 for all available cores (-2 for all but one, and so on). The output
        does not depend on `n_jobs`: each thread handles its own particles, and applies the halos in
        catalog order.

    Attributes
    ----------
    HaloNDCatalog : object
        The halo ND catalog instance.

    ParticleSnapshot : object
        The particle snapshot instance.

    cosmo : object
        The cosmology object extracted from `HaloNDCatalog`.

    model : object
        The model used for baryonification or profile painting.

    epsilon_max : float
        The maximum radius, in halo radius units, of cutouts around halos.

    mass_def : object
        The mass definition object.

    verbose : bool
        Whether verbose output is enabled.

    tree : KDTree
        A KDTree built from the particle coordinates (built on first access; not used by the runners).

    Methods
    -------
    compute_distance(*args)
        Helper function that computes the Euclidean distance between points,
        accounting for periodic boundary conditions.

    enforce_periodicity(dx)
        Helper function that adjusts distances to enforce periodic boundary conditions,
        ensuring distances are within the box size.
    """

    def __init__(self, HaloNDCatalog, ParticleSnapshot, epsilon_max, model,
                 mass_def = None, verbose = True, KDTree_kwargs = {}, n_jobs = 1):

        self.HaloNDCatalog    = HaloNDCatalog
        self.ParticleSnapshot = ParticleSnapshot
        self.epsilon_max      = epsilon_max
        self.cosmo = HaloNDCatalog.cosmology
        self.model = model

        self.mass_def = _default_mass_def(model) if mass_def is None else mass_def
        self.verbose  = verbose
        self.n_jobs   = n_jobs

        self.KDTree_kwargs = KDTree_kwargs
        self._tree  = None
        self._index = None


    @property
    def tree(self):
        """A periodic `scipy.spatial.KDTree` of the particles, built on first access (the runners do not use it)."""

        if self._tree is None:
            Snap   = self.ParticleSnapshot
            coords = np.vstack([Snap.cat['x'], Snap.cat['y']] + ([] if Snap.is2D else [Snap.cat['z']])).T
            #Periodic KDTrees need data in [0, L). Snapshots stored on [0, L] can have x == L exactly.
            self._tree = KDTree(np.mod(coords, Snap.L), boxsize = Snap.L, **self.KDTree_kwargs)

        return self._tree


    def _particle_index(self):
        """
        The particles sorted into a periodic grid of cells (built once and cached): the number of cells per
        axis, the sort order, the start of every cell, and the particle coordinates in that order (as stored
        in the snapshot, ie. not wrapped into the box).
        """

        if self._index is None:
            Snap = self.ParticleSnapshot
            N    = Snap.cat.size
            #~16 particles per cell on average, which keeps the per-cell overhead small
            ndim = 2 if Snap.is2D else 3
            n1   = int(np.clip(np.round((N / 16)**(1/ndim)), 1, 4096 if Snap.is2D else 512))
            n    = np.array([n1, n1, 1 if Snap.is2D else n1], dtype = np.int64)

            x = Snap.cat['x'].astype(np.float64, copy = False)
            y = Snap.cat['y'].astype(np.float64, copy = False)
            z = np.zeros(N) if Snap.is2D else Snap.cat['z'].astype(np.float64, copy = False)
            with self._threads():
                order, starts = _build_cell_index(x, y, z, float(Snap.L), n)
                self._index = {'n' : n, 'order' : order, 'starts' : starts,
                               'xs' : _gather(x, order), 'ys' : _gather(y, order), 'zs' : _gather(z, order)}

        return self._index


    @contextmanager
    def _threads(self):
        """Runs the compiled (numba) loops with `_n_threads()` threads, restoring numba's setting afterwards."""

        previous = numba.get_num_threads()
        numba.set_num_threads(self._n_threads())
        try:
            yield
        finally:
            numba.set_num_threads(previous)


    def _n_threads(self):
        """Number of threads to use, following the joblib convention for negative `n_jobs`."""

        n = 1 if self.n_jobs in (None, 0) else int(self.n_jobs)
        n = numba.config.NUMBA_NUM_THREADS + 1 + n if n < 0 else n
        return int(np.clip(n, 1, numba.config.NUMBA_NUM_THREADS))


    def compute_distance(self, *args):
        """
        Computes the Euclidean distance between points, accounting for periodic boundary conditions.

        This method calculates the distance between points, ensuring that the computed distance takes into account
        the periodicity of the simulation box. It is designed to handle cases where distances might wrap around
        the edges of the box.

        Parameters
        ----------
        *args : list of ndarrays
            Arrays representing differences in each dimension (e.g., dx, dy, dz) between the points.

        Returns
        -------
        d : ndarray
            An array of distances computed for each pair of points, with periodicity accounted for.
        """

        return np.sqrt(sum(self.enforce_periodicity(dx)**2 for dx in args))


    def enforce_periodicity(self, dx):
        """
        Adjusts distances to enforce periodic boundary conditions.

        This method adjusts the input distances to ensure that they are within the box size, effectively
        enforcing periodic boundary conditions. It modifies the distances in place.

        Parameters
        ----------
        dx : ndarray
            An array of distances to be adjusted for periodic boundary conditions.

        Returns
        -------
        dx : ndarray
            The adjusted distances, with values wrapped around the box size if necessary.
        """

        L = self.ParticleSnapshot.L

        dx = np.where(dx > L/2,  dx - L, dx)
        dx = np.where(dx < -L/2, dx + L, dx)

        return dx



class BaryonifySnapshot(DefaultRunnerSnapshot):
    """
    A class to apply baryonification to a particle snapshot using a halo catalog.

    The `BaryonifySnapshot` class inherits from `DefaultRunnerSnapshot` and is designed to process a particle
    snapshot by applying baryonification techniques to adjust particle positions. It uses a halo catalog to
    determine the necessary adjustments based on halo properties, cosmological parameters, and a specified model.

    Methods
    -------
    process()
        Processes the particle snapshot by applying baryonification and returns the modified particle catalog.
    """

    def process(self):
        """
        Applies baryonification to the particle snapshot using the halo catalog.

        This method iterates over each halo in the `HaloNDCatalog`, calculating the necessary displacements
        for particles within a certain radius of each halo. The displacements are computed based on the halo's
        mass, position, and scale factor. The resulting offsets are applied to the particle positions, and the
        modified particle catalog is returned.

        Returns
        -------
        new_cat : ndarray
            A structured array representing the modified particle catalog after baryonification.

        Notes
        -----
        - This method supports both 2D and 3D particle snapshots.
        - Particles within the query radius of each halo are found with a periodic grid of cells (see
          `DefaultRunnerSnapshot`).
        - For tabulated displacement models (`Baryonification2D`/`3D`), the displacement of each halo is read
          from the table once, along the table's radial nodes, and the particles are displaced in a compiled
          loop (threaded if `n_jobs > 1`). Other models are called halo by halo.
        - Periodic boundary conditions are enforced to ensure particles remain within the simulation box.
        - The method assumes that the input catalog provides particle coordinates as 'x', 'y', and optionally 'z'.
        """

        cosmo = _runner_cosmology(self.cosmo)

        L    = float(self.ParticleSnapshot.L)
        axes = ['x', 'y'] if self.ParticleSnapshot.is2D else ['x', 'y', 'z']
        keys = _check_p_keys(self.model) #Names of extra (tabulated) model parameters
        cat  = self.HaloNDCatalog.cat

        M   = np.asarray(cat['M'], dtype = float)
        a   = 1/(1 + self.HaloNDCatalog.redshift)
        R   = _halo_radius(self.mass_def, cosmo, M, np.full(M.shape, a)) #in physical Mpc
        R_q = np.clip(self.epsilon_max * R/a, 0, L/2) #The radius for querying points, in comoving coords. Can't query distances more than half box-size.
        pos = np.zeros([M.size, 3])
        for i, ax in enumerate(axes): pos[:, i] = cat[ax] #CARTESIAN COORDINATES (z is not redshift)
        other = {key : np.asarray(cat[key]) for key in keys} #Other properties

        index   = self._particle_index()
        offsets = np.zeros([self.ParticleSnapshot.cat.size, 3]) #In the original particle order

        curves = None
        if _batch_method(self.model, '_displacement_curves', 'displacement', '_readout') is not None:
            curves = self.model._displacement_curves(M, np.full(M.shape, a), r = R_q, n_threads = self._n_threads(), **other)

        if curves is not None: self._offsets_tabulated(index, offsets, pos, R_q, curves, L)
        else:                  self._offsets_per_halo(index, offsets, pos, R_q, M, a, other, L)

        #Apply the offsets, and wrap into [0, L), so x == L maps to 0 (as periodic KDTrees expect).
        #Same as new_cat = cat.copy(); new_cat[ax] = np.mod(new_cat[ax] + offsets, L), but in parallel.
        cat     = self.ParticleSnapshot.cat
        new_cat = np.empty_like(cat)
        with self._threads():
            for field in cat.dtype.names:
                if field in axes: _shift_and_wrap(cat[field], offsets[:, axes.index(field)], L, new_cat[field])
                else:             _copy(cat[field], new_cat[field])

        return new_cat


    def _offsets_tabulated(self, index, offsets, pos, R_q, curves, L):
        """Offsets from a tabulated displacement model, with the compiled (optionally threaded) loop."""

        nodes, table, shift, R_cut = curves
        n = index['n']

        #The halos overlapping each plane of cells along x
        planes = [[] for _ in range(n[0])]
        for h in range(R_q.size):
            lo, hi = _cell_range(pos[h, 0], R_q[h], L, n[0])
            for j in range(lo, hi + 1): planes[j % n[0]].append(h)
        plane_start = np.concatenate([[0], np.cumsum([len(p) for p in planes])]).astype(np.int64)
        plane_halo  = np.array([h for p in planes for h in p], dtype = np.int64)
        plane_order = np.random.default_rng(0).permutation(n[0]).astype(np.int64) #Spread busy planes over the threads

        with self._threads(), tqdm(total = R_q.size, desc = 'Baryonifying matter', disable = not self.verbose) as pbar:
            _snapshot_offsets(index['xs'], index['ys'], index['zs'], index['order'], index['starts'], n, L,
                              pos, R_q, np.asarray(R_cut, dtype = float), np.asarray(shift, dtype = float),
                              nodes, np.ascontiguousarray(table, dtype = float),
                              plane_start, plane_halo, plane_order, offsets)
            pbar.update(R_q.size)


    def _offsets_per_halo(self, index, offsets, pos, R_q, M, a, other, L):
        """Offsets from any other displacement model, called once per halo."""

        for j in tqdm(range(M.size), desc = 'Baryonifying matter', disable = not self.verbose):

            o_j = {key : v[j] for key, v in other.items()} #Other properties
            idx, sep = _ball(index['xs'], index['ys'], index['zs'], index['starts'], index['n'], L,
                             pos[j, 0], pos[j, 1], pos[j, 2], R_q[j])
            d = sep[:, 3]

            #A particle exactly at the halo center has no direction. Give it zero displacement.
            with np.errstate(invalid = 'ignore', divide = 'ignore'):
                hats = [np.where(d > 0, sep[:, i]/d, 0) for i in range(3)]

            #Compute the displacement needed
            offset = self.model.displacement(d, M[j], a, **o_j)
            offset = np.where(np.isfinite(offset), offset, 0)
            offsets[index['order'][idx]] += np.vstack([offset*h for h in hats]).T
