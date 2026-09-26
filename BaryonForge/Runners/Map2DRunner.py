import numpy as np
import warnings

from tqdm import tqdm
from numba import njit
from ..utils.Tabulate import _get_parameter
from ..utils.misc import _default_mass_def, _runner_cosmology, _check_p_keys, _halo_radius
from ._chunks import _add_at, _add_rows_at, _n_threads, _chunk_bounds, _run_chunks, _evaluate_pairs

__all__ = ['DefaultRunnerGrid', 'BaryonifyGrid', 'PaintProfilesGrid', 'PaintProfilesAnisGrid',
           'regrid_pixels_2D', 'regrid_pixels_3D']

@njit
def _wrap_cell(i, N):
    """Periodic cell index, as in the regridding loops."""

    if i < 0: i += N
    if i + 1 > N: i = i % N
    return i


@njit
def _cell_overlap(j, start, end, N):
    """Overlap of [start, end) with cell j, trying the periodic images as in the regridding loops."""

    d = min(j + 1, end) - max(j, start)
    if d < 0: d = min(j + 1, end + N) - max(j, start + N)
    if d < 0: d = min(j + 1, end - N) - max(j, start - N)
    return d


@njit
def regrid_pixels_2D(grid, pix_positions, pix_values):

    """
    Redistributes pixel values onto a 2D (regular, square) grid considering periodic boundary conditions.

    A displaced (unit) pixel overlaps at most two cells along each axis, so only those are visited. The
    overlaps are computed exactly as in the reference loop `_regrid_pixels_2D_loop` (which also handles
    grids of fewer than 6 pixels a side), and the result is the same.

    Parameters
    ----------
    grid : ndarray
        A 2D numpy array representing the grid onto which pixel values will be redistributed.
        The grid is modified in place. Must be a square grid.

    pix_positions : ndarray of shape (N, 2)
        An array of pixel positions, where each position is given by (x, y) coordinates.
        These coordinates specify where the displaced pixels are located.

    pix_values : ndarray of shape (N,)
        An array of pixel values corresponding to each position in `pix_positions`.
        These values are redistributed across the grid based on the pixel's overlap
        with the grid.
    """

    N = grid.shape[0]
    if N < 6: return _regrid_pixels_2D_loop(grid, pix_positions, pix_values)

    for p in range(pix_positions.shape[0]):

        x_start, y_start = pix_positions[p, 0] % N, pix_positions[p, 1] % N #To handle edge-case where offset >> Lbox_sim
        x_end, y_end     = x_start + 1, y_start + 1
        x0, y0           = int(x_start), int(y_start)

        for di in range(2):
            i  = _wrap_cell(y0 + di, N)
            dy = _cell_overlap(i, y_start, y_end, N)
            if not (dy > 0): continue
            for dj in range(2):
                j  = _wrap_cell(x0 + dj, N)
                dx = _cell_overlap(j, x_start, x_end, N)
                if (dx > 0) & (dy > 0):
                    overlap_area = dx * dy
                    grid[i, j] += overlap_area * pix_values[p]


@njit
def regrid_pixels_3D(grid, pix_positions, pix_values):

    """
    Redistributes pixel values onto a 3D grid considering periodic boundary conditions.

    A displaced (unit) pixel overlaps at most two cells along each axis, so only those are visited. The
    overlaps are computed exactly as in the reference loop `_regrid_pixels_3D_loop` (which also handles
    grids of fewer than 6 pixels a side), and the result is the same.

    Parameters
    ----------
    grid : ndarray
        A 3D numpy array representing the grid onto which pixel values will be redistributed.
        The grid is modified in place. Must be a cubic grid.

    pix_positions : ndarray of shape (N, 3)
        An array of pixel positions, where each position is given by (x, y, z) coordinates.
        These coordinates specify where the displaced pixels are located.

    pix_values : ndarray of shape (N,)
        An array of pixel values corresponding to each position in `pix_positions`.
        These values are redistributed across the grid based on overlap.
    """

    N = grid.shape[0]
    if N < 6: return _regrid_pixels_3D_loop(grid, pix_positions, pix_values)

    for p in range(pix_positions.shape[0]):

        x_start, y_start, z_start = pix_positions[p, 0] % N, pix_positions[p, 1] % N, pix_positions[p, 2] % N
        x_end, y_end, z_end       = x_start + 1, y_start + 1, z_start + 1
        x0, y0, z0                = int(x_start), int(y_start), int(z_start)

        for di in range(2):
            i  = _wrap_cell(y0 + di, N)
            dy = _cell_overlap(i, y_start, y_end, N)
            if not (dy > 0): continue
            for dj in range(2):
                j  = _wrap_cell(x0 + dj, N)
                dx = _cell_overlap(j, x_start, x_end, N)
                if not (dx > 0): continue
                for dk in range(2):
                    k  = _wrap_cell(z0 + dk, N)
                    dz = _cell_overlap(k, z_start, z_end, N)
                    if (dx > 0) & (dy > 0) & (dz > 0):
                        overlap_vol = dx * dy * dz
                        grid[i, j, k] += overlap_vol * pix_values[p]


@njit
def _regrid_pixels_2D_loop(grid, pix_positions, pix_values):

    """
    Redistributes pixel values onto a 2D (regular, square) grid considering periodic boundary conditions.

    This function takes a list of pixel positions and their associated values, then redistributes these 
    values onto a specified 2D grid. It accounts for overlap and periodic boundary conditions, ensuring 
    proper handling of edge cases where offsets are significantly larger than the grid size.

    Parameters
    ----------
    grid : ndarray
        A 2D numpy array representing the grid onto which pixel values will be redistributed. 
        The grid is modified in place. Must be a square grid.

    pix_positions : ndarray of shape (N, 2)
        An array of pixel positions, where each position is given by (x, y) coordinates. 
        These coordinates specify where the displaced pixels are located.

    pix_values : ndarray of shape (N,)
        An array of pixel values corresponding to each position in `pix_positions`. 
        These values are redistributed across the grid based on the pixel's overlap
        with the grid.

    Notes
    -----
    - The function uses Numba's `@njit` decorator for just-in-time compilation, optimizing performance.
    - Periodic boundary conditions are handled explicitly to ensure proper wrapping around the grid edges.
    - This function assumes that both `grid` and `pix_positions` use a zero-based index system and that 
      the grid is square with shape `(N, N)`.

    """

    for pix_pos, pix_value in zip(pix_positions, pix_values):

        N = grid.shape[0]
        x_start, y_start = pix_pos
        x_start, y_start = x_start % N, y_start % N #To handle edge-case where offset >> Lbox_sim
        x_end, y_end     = x_start + 1, y_start + 1

        bound = 2
        
        x_min, x_max = int(x_start) - bound, int(x_end) + bound
        y_min, y_max = int(y_start) - bound, int(y_end) + bound
        
        for i in range(y_min, y_max):
            for j in range(x_min, x_max):

                if i < 0: i += N
                if i + 1 > N: i = i % N

                if j < 0: j += N
                if j + 1 > N: j = j % N

                #Find intersection length
                dx = min(j + 1, x_end) - max(j, x_start)
                dy = min(i + 1, y_end) - max(i, y_start)

                #Now account for periodic boundary conditions
                if dx < 0: dx = min(j + 1, x_end + N) - max(j, x_start + N)
                if dx < 0: dx = min(j + 1, x_end - N) - max(j, x_start - N)

                if dy < 0: dy = min(i + 1, y_end + N) - max(i, y_start + N)
                if dy < 0: dy = min(i + 1, y_end - N) - max(i, y_start - N)

                #If there is some intersection, then add
                if (dx > 0) & (dy > 0):
                    overlap_area = dx * dy
                    grid[i, j] += overlap_area * pix_value


@njit
def _regrid_pixels_3D_loop(grid, pix_positions, pix_values):

    """
    Redistributes pixel values onto a 3D grid considering periodic boundary conditions.

    This function takes a list of 3D pixel positions and their associated values, then redistributes 
    these values onto a specified 3D grid. It accounts for overlap and periodic boundary conditions, 
    ensuring proper handling of edge cases where offsets are significantly larger than the grid size.

    Parameters
    ----------
    grid : ndarray
        A 3D numpy array representing the grid onto which pixel values will be redistributed. 
        The grid is modified in place. Must be a cubic grid.

    pix_positions : ndarray of shape (N, 3)
        An array of pixel positions, where each position is given by (x, y, z) coordinates. 
        These coordinates specify where the displaced pixels are located.

    pix_values : ndarray of shape (N,)
        An array of pixel values corresponding to each position in `pix_positions`. 
        These values are redistributed across the grid based on overlap.

    Notes
    -----
    - The function uses Numba's `@njit` decorator for just-in-time compilation, optimizing performance.
    - Periodic boundary conditions are handled explicitly to ensure proper wrapping around the grid edges.
    - This function assumes that both `grid` and `pix_positions` use a zero-based index system and that 
      the grid is cubic with shape `(N, N, N)`.

    """

    for pix_pos, pix_value in zip(pix_positions, pix_values):

        N = grid.shape[0]
        x_start, y_start, z_start = pix_pos
        x_start, y_start, z_start = x_start % N, y_start % N, z_start % N #To handle edge-case where offset >> Lbox_sim
        x_end, y_end, z_end       = x_start + 1, y_start + 1, z_start + 1

        bound = 2
        
        x_min, x_max = int(x_start) - bound, int(x_end) + bound
        y_min, y_max = int(y_start) - bound, int(y_end) + bound
        z_min, z_max = int(z_start) - bound, int(z_end) + bound

        for i in range(y_min, y_max):
            for j in range(x_min, x_max):
                for k in range(z_min, z_max):
                    
                    if i < 0: i += N
                    if i + 1 > N: i = i % N

                    if j < 0: j += N
                    if j + 1 > N: j = j % N

                    if k < 0: k += N
                    if k + 1 > N: k = k % N
                    
                    #Find intersection length
                    dx = min(j + 1, x_end) - max(j, x_start)
                    dy = min(i + 1, y_end) - max(i, y_start)
                    dz = min(k + 1, z_end) - max(k, z_start)


                    if dx < 0: dx = min(j + 1, x_end + N) - max(j, x_start + N)
                    if dx < 0: dx = min(j + 1, x_end - N) - max(j, x_start - N)

                    if dy < 0: dy = min(i + 1, y_end + N) - max(i, y_start + N)
                    if dy < 0: dy = min(i + 1, y_end - N) - max(i, y_start - N)
                        
                    if dz < 0: dz = min(k + 1, z_end + N) - max(k, z_start + N)
                    if dz < 0: dz = min(k + 1, z_end - N) - max(k, z_start - N)


                    if (dx > 0) & (dy > 0) & (dz > 0):
                        overlap_vol = dx * dy * dz
                        grid[i, j, k] += overlap_vol * pix_value
                    

@njit(nogil = True)
def _wrap_index(i, Npix):
    """Periodic pixel index, as in `DefaultRunnerGrid.pick_indices`."""

    i = i + Npix if i < 0 else i
    i = i - Npix if i >= Npix else i
    return i


@njit(nogil = True, error_model = 'numpy') #numpy semantics: 0/0 = nan at the halo center, as before
def _circular_cutouts(cen, off, nsize, start, Npix, res, want_hats, inds, hid, r, hat):
    """
    Cutouts of many halos around their central pixels `cen`, with cutout widths `nsize` (even), written
    from position `start[h]` of the output arrays: the flat map index of each pixel, the index of its halo,
    its distance from the halo, and (if `want_hats`) the unit vector from the halo. The same pixels, order
    and arithmetic as `DefaultRunnerGrid._chunk_cutouts`, for cutouts without ellipticity.
    """

    ndim = cen.shape[1]
    for h in range(cen.shape[0]):
        n, s = nsize[h], start[h]
        w    = n // 2
        for a in range(n):
            ia = _wrap_index(cen[h, 0] - w + a, Npix)
            gx = (a - w) * res + off[h, 0]
            for b in range(n):
                ib = _wrap_index(cen[h, 1] - w + b, Npix)
                gy = (b - w) * res + off[h, 1]
                if ndim == 2:
                    p = s + a*n + b
                    inds[p], hid[p] = ia*Npix + ib, h
                    r[p] = np.sqrt(gx*gx + gy*gy)
                    if want_hats: hat[p, 0], hat[p, 1] = gx/r[p], gy/r[p]
                else:
                    for c in range(n):
                        ic = _wrap_index(cen[h, 2] - w + c, Npix)
                        gz = (c - w) * res + off[h, 2]
                        p  = s + (a*n + b)*n + c
                        inds[p], hid[p] = (ia*Npix + ib)*Npix + ic, h
                        r[p] = np.sqrt(gx*gx + gy*gy + gz*gz)
                        if want_hats: hat[p, 0], hat[p, 1], hat[p, 2] = gx/r[p], gy/r[p], gz/r[p]


#Quickly run the functions once so they compile and initialize
regrid_pixels_2D(np.zeros([5, 5]),    np.ones([2, 2]), np.ones(2))
regrid_pixels_3D(np.zeros([5, 5, 5]), np.ones([2, 3]), np.ones(2))
regrid_pixels_2D(np.zeros([8, 8]),    np.ones([2, 2]), np.ones(2))
regrid_pixels_3D(np.zeros([8, 8, 8]), np.ones([2, 3]), np.ones(2))

                        
class DefaultRunnerGrid(object):
    """
    A utility class for handling input/output operations related to halo ND catalogs and gridded maps.

    The `DefaultRunnerGrid` class provides methods to manage and process data associated with halo ND catalogs,
    including constructing rotation matrices and generating coordinate arrays. It supports operations in both
    2D and 3D contexts and handles optional ellipticity-based calculations.

    Parameters
    ----------
    HaloNDCatalog : object
        An instance of a `HaloNDCatalog`, containing data about halos and their properties. It must have a 
        `cosmology` attribute to specify the cosmological parameters.
    
    GriddedMap : object
        An instance representing a `GriddedMap`, either 2D or 3D, where halo information will be mapped.
    
    epsilon_max : float
        A parameter specifying the maximum size, in units of halo radius, of cutouts made around
        each halo during painting/baryonification.
    
    model : object, optional
        An object that generates profiles or displacements. For example, see `Baryonification2D` or `Pressure`
    
    use_ellipticity : bool, optional
        A flag indicating whether to use ellipticity in calculations. Default is False.
        Requires the 'q_ell' (axis ratio) and 'A_ell' (major-axis direction) catalog columns.
        Only supported for 2D maps.
    
    mass_def : object, optional
        An instance of a mass definition object from the CCL (Core Cosmology Library), specifying 
        the mass definition to be used. Default is None, in which case the mass definition of `model` is used (or 200c, if `model` has none).
    
    verbose : bool, optional
        A flag to enable verbose output for logging or debugging purposes. Default is True.

    Attributes
    ----------
    HaloNDCatalog : object
        The `HaloNDCatalog` instance.
    
    GriddedMap : object
        The `GriddedMap` instance.
    
    cosmo : object
        The cosmology object extracted from `HaloNDCatalog`.
    
    model : object
        The model used for baryonification or profile painting.
    
    epsilon_max : float
        The maximum radius, in halo radius units, of cutouts around halos.
    
    mass_def : object
        The mass definition object.
    
    verbose : bool, optional
        Whether verbose output is enabled. Defaults to True.
    
    include_pixel_size : bool, optional
        Only used when painting, not baryonifying.
        If True, then the returned map is multiplied by the area/volume of the pixel.
        Thus, painting with a density profile results in a Mass map. 
        Defaults to True.

    use_ellipticity : bool, optional
        Whether to use ellipticity in calculations. Defaults to False.

    Methods
    -------
    build_Rmat(A, q)
        Constructs a rotation matrix based on the input vector A and ellipticity parameter q.
    
    coord_array(*args)
        Flattens and stacks input arrays into a 2D array of coordinates.

    Raises
    ------
    AssertionError
        If `use_ellipticity` is True and required columns ('q_ell', 'A_ell') are missing in the
        `HaloNDCatalog`.
    NotImplementedError
        If attempting to use the 3D ellipticity method, which is not yet verified.
    """
    
    def __init__(self, HaloNDCatalog, GriddedMap, epsilon_max, model, use_ellipticity = False,
                 mass_def = None, include_pixel_size = True, verbose = True, n_jobs = 1):

        self.HaloNDCatalog = HaloNDCatalog
        self.GriddedMap    = GriddedMap
        self.cosmo = HaloNDCatalog.cosmology
        self.model = model


        self.epsilon_max = epsilon_max
        self.mass_def    = _default_mass_def(model) if mass_def is None else mass_def
        self.verbose     = verbose
        self.n_jobs      = n_jobs
        
        self.use_ellipticity    = use_ellipticity
        self.include_pixel_size = include_pixel_size
        
        #Assert that all the required quantities are in the input catalog
        if use_ellipticity:
            
            names = HaloNDCatalog.cat.dtype.names
            
            assert 'q_ell' in names, "The 'q_ell' column is missing, but you set use_ellipticity = True"
            assert 'A_ell' in names, "The 'A_ell' column is missing, but you set use_ellipticity = True"
    
    
    def build_Rmat(self, A, q):
        """
        Constructs a rotation matrix based on the input vector and ellipticity parameter.

        This method normalizes the input vector A and calculates the rotation matrix using
        the ellipticity parameter q. For 2D vectors, it uses the shear transformation. For 
        3D vectors, a not yet verified method is provided but raises a NotImplementedError.

        Parameters
        ----------
        A : ndarray
            A 1D array giving the direction (x, y) of the ellipse's major axis. It is normalized within the method.
            The painted/displaced profile is elongated along `A` and compressed perpendicular to it.

        q : float
            The axis ratio (minor/major) of the ellipse, used to compute the shear transformation.

        Returns
        -------
        Rmat : ndarray
            A 2x2 or 3x3 rotation matrix, depending on the dimensionality of the input vector.

        Raises
        ------
        ValueError
            If the input vector A is 1-dimensional.
        NotImplementedError
            If a 3D rotation is attempted, indicating that the method is not yet verified for 3D vectors.
        """

        A = np.asarray(A, dtype = float) / np.linalg.norm(A) #Not in-place, so the input is not modified

        if len(A) == 1:
            raise  ValueError("Can't rotate a 1-dimensional vector")

        elif len(A) == 2:

            #The 2D rotation is done using routines implemented in the galsim Shear class.
            #Use the signed angle of A (arccos would map A = (cos t, -sin t) onto (cos t, sin t)).
            #The transformation compresses the profile along beta, so beta is the minor axis,
            #perpendicular to the major axis A.

            beta = np.arctan2(A[1], A[0]) + np.pi/2
            eta  = -np.log(q) 
            
            if eta > 1e-4:
                eta2g = np.tanh(0.5*eta)/eta
            else:
                etasq = eta * eta
                eta2g = 0.5 + etasq*((-1/24) + etasq*(1/240))

            g   = eta2g * eta * np.exp(2j * beta)
            g1  = g.real
            g2  = g.imag

            det  = np.sqrt(1 - np.abs(g)**2)
            Rmat = np.array([[1 + g1, g2],
                             [g2, 1 - g1]]) / det
        
        elif len(A) == 3:
            
            raise NotImplementedError("This method has not yet been verified. Use 2D ellipticity method instead")

        return Rmat
        
    def coord_array(self, *args):
        """
        Flattens and stacks input arrays into a 2D array of coordinates.

        This method takes multiple input arrays, flattens each, and stacks them column-wise to
        create a single 2D array where each row represents a coordinate.

        Parameters
        ----------
        *args : list of ndarrays
            Arrays to be flattened and stacked. Each input array represents one dimension of the 
            coordinates. All arrays must have the same shape.

        Returns
        -------
        coords : ndarray
            A 2D array of shape (N, M) where N is the total number of elements (after flattening) and M
            is the number of input arrays. Each row represents a coordinate.
        """

        return np.vstack([a.flatten() for a in args]).T


    def pick_indices(self, center, width, Npix):
        """
        Selects and returns indices around a center point, accounting for periodic boundary conditions.

        This method takes a central index and a width and returns an array of indices around the center,
        wrapping around if the indices go beyond the boundaries of the grid. This is used to get
        cutouts around a given halo.

        Parameters
        ----------
        center : int
            The central index around which indices are selected.

        width : int
            The half-width of the selection range. The method selects indices from `center - width` to `center + width - 1`.

        Npix : int
            The total number of pixels along one dimension of the grid. Used to wrap indices for periodic boundary conditions.

        Returns
        -------
        inds : ndarray
            An array of selected indices, wrapped around the boundaries if necessary.
        """

        inds = np.arange(center - width, center + width)
        inds = np.where((inds) < 0,     inds + Npix, inds)
        inds = np.where((inds) >= Npix, inds - Npix, inds)

        return inds


    #Largest number of (halo, pixel) pairs handled in one chunk, as in the HEALPix runners
    _max_pairs_per_chunk = 1_000_000


    def _n_threads(self):
        """Number of threads to use, following the joblib convention for negative `n_jobs`."""

        return _n_threads(getattr(self, 'n_jobs', 1))


    def _halo_table(self, cosmo, keys):
        """
        Quantities of every halo in the catalog, computed once rather than inside the loop over halos: mass,
        scale factor, physical radius `R` (R_delta), the pixel nearest to the halo and the halo's offset from
        that pixel's center, the extra (tabulated) properties and, if used, the ellipticity.
        """

        cat  = self.HaloNDCatalog.cat
        bins = self.GriddedMap.bins
        res  = self.GriddedMap.res
        axes = ['x', 'y'] if self.GriddedMap.is2D else ['x', 'y', 'z']

        M = np.asarray(cat['M'], dtype = float)
        a = np.full(M.shape, 1/(1 + self.HaloNDCatalog.redshift))
        R = _halo_radius(self.mass_def, cosmo, M, a) #in physical Mpc

        #The pixel nearest to each halo, ie. np.argmin(np.abs(bins - x)) (ties go to the lower index)
        cen, off = [], []
        for ax in axes:
            x = np.asarray(cat[ax], dtype = float) #THIS IS A CARTESIAN COORDINATE, NOT REDSHIFT
            i = np.clip(np.searchsorted(bins, x), 1, bins.size - 1)
            i = np.where(np.abs(bins[i - 1] - x) <= np.abs(bins[i] - x), i - 1, i)
            cen.append(i)
            off.append(bins[i] - x) #Offsets between halo position and pixel center

        H = {'M' : M, 'a' : a, 'R' : R, 'cen' : np.stack(cen, axis = 1), 'off' : np.stack(off, axis = 1),
             'other' : {key : np.asarray(cat[key]) for key in keys}}
        if self.use_ellipticity: H['q'], H['A'] = cat['q_ell'], cat['A_ell']

        return H


    def _cutout_size(self, width):
        """Even cutout width in pixels for a cutout of `width` (comoving Mpc), at most the full box width."""

        Nsize = width / self.GriddedMap.res
        Nsize = int(Nsize // 2)*2 #Force it to be even
        return np.clip(Nsize, 2, 2*(self.GriddedMap.bins.size//2))


    def _halo_chunks(self, width):
        """Consecutive halos in chunks of at most ~`_max_pairs_per_chunk` (halo, pixel) pairs (the cutout sizes)."""

        ndim   = 2 if self.GriddedMap.is2D else 3
        n_pair = np.array([self._cutout_size(w) for w in width], dtype = float)**ndim
        return _chunk_bounds(n_pair, self._max_pairs_per_chunk, self._n_threads())


    def _chunk_cutouts(self, halos, H, width, hats = False):
        """
        The cutout of every halo in `halos` (full widths `width`, comoving Mpc): the flat map index of each
        pixel, the chunk-local halo index of each pixel, and its distance from the halo (elliptical, if used),
        plus, if `hats`, the unit vector from the halo to the pixel along each map axis (from the circular
        distance). The same cutouts, pixel order and arithmetic as the per-halo loop.
        """

        res, Npix = self.GriddedMap.res, self.GriddedMap.Npix
        is2D      = self.GriddedMap.is2D
        out       = {'inds' : [], 'hid' : [], 'r' : [], 'hat' : []}

        if not (self.use_ellipticity and is2D):
            #Without ellipticity, build all the cutouts in one compiled (GIL-free) call
            ndim  = 2 if is2D else 3
            nsize = np.array([self._cutout_size(width[j]) for j in halos], dtype = np.int64)
            off   = np.ascontiguousarray(H['off'][halos])
            if is2D and np.any(np.abs(off) > res):
                j = halos[np.flatnonzero(np.any(np.abs(off) > res, axis = 1))[0]]
                raise AssertionError("Halo offsets (%0.2f, %0.2f) are larger than res (%0.2f)" % (H['off'][j][0], H['off'][j][1], res))
            count = nsize**ndim
            start = np.concatenate([[0], np.cumsum(count)[:-1]]).astype(np.int64)
            P     = int(count.sum())
            C     = {'inds' : np.empty(P, dtype = np.int64), 'hid' : np.empty(P, dtype = np.int64), 'r' : np.empty(P),
                     'hat' : np.empty((P if hats else 0, ndim))}
            _circular_cutouts(np.ascontiguousarray(H['cen'][halos]), off, nsize, start, Npix, res, hats,
                              C['inds'], C['hid'], C['r'], C['hat'])
            if not hats: C.pop('hat')
            return C

        for i, j in enumerate(halos):
            Nsize = self._cutout_size(width[j])

            #Pixel-center offsets (from the central pixel) of the cutout. These match the
            #indices chosen by pick_indices, which run from center - width to center + width - 1.
            cutout_width = Nsize//2
            x    = (np.arange(Nsize) - cutout_width) * res
            inds = [self.pick_indices(c, cutout_width, Npix) for c in H['cen'][j]]
            d    = H['off'][j]

            if is2D:
                assert np.logical_and(np.abs(d[0]) <= res, np.abs(d[1]) <= res), "Halo offsets (%0.2f, %0.2f) are larger than res (%0.2f)" % (d[0], d[1], res)

                #Axis 0 of the cutout follows the halo x-coordinate and axis 1 the y-coordinate,
                #matching the (x_inds, y_inds) ordering of "inds" below.
                flat   = (inds[0][:, None]*Npix + inds[1][None, :]).flatten()
                grids  = np.meshgrid(x + d[0], x + d[1], indexing = 'ij')
                r_grid = np.sqrt(grids[0]**2 + grids[1]**2)
                if hats: out['hat'].append(np.stack([(g/r_grid).flatten() for g in grids], axis = 1))

                #If ellipticity exists, then account for it
                if self.use_ellipticity:
                    assert H['q'][j] > 0, "The axis ratio in halo %d is not positive" % j

                    Rmat = self.build_Rmat(H['A'][j], H['q'][j])
                    x_grid_ell, y_grid_ell = (self.coord_array(*grids) @ Rmat).T
                    r_grid = np.sqrt(x_grid_ell**2 + y_grid_ell**2).reshape(grids[0].shape)

            else:
                #Axes 0, 1, 2 of the cutout follow the halo x, y, z coordinates
                flat   = ((inds[0][:, None, None]*Npix + inds[1][None, :, None])*Npix + inds[2][None, None, :]).flatten()
                grids  = np.meshgrid(x + d[0], x + d[1], x + d[2], indexing = 'ij')
                r_grid = np.sqrt(grids[0]**2 + grids[1]**2 + grids[2]**2)
                if hats: out['hat'].append(np.stack([(g/r_grid).flatten() for g in grids], axis = 1))

            out['inds'].append(flat)
            out['hid'].append(np.full(flat.size, i))
            out['r'].append(r_grid.flatten())

        return {k : np.concatenate(v) for k, v in out.items() if len(v) > 0}


    def _evaluate(self, model, method, cosmo, r, hid, halos, H):
        """`model.<method>` for every (halo, pixel) pair of a chunk (see `_chunks._evaluate_pairs`)."""

        return _evaluate_pairs(model, method, cosmo, r, hid, H['M'][halos], H['a'][halos],
                               {k : v[halos] for k, v in H['other'].items()})



class BaryonifyGrid(DefaultRunnerGrid):

    """
    A class to apply baryonification to a gridded map using a halo catalog.

    The `BaryonifyGrid` class inherits from `DefaultRunnerGrid` and is designed to process a gridded map by
    applying baryonification techniques to adjust the matter distribution. It uses a halo catalog to determine
    the necessary adjustments based on halo properties, cosmological parameters, and a specified model.

    The inputted grid should be MASS grid rather than density grid. This is because the method uses
    pix = 0 to identify empty pixels.

    Methods
    -------
    process()
        Processes the gridded map by applying baryonification and returns the modified grid.

    pick_indices(center, width, Npix)
        Helper method that selects and returns indices around a center point, 
        accounting for periodic boundary conditions.

    """

    def process(self):
        """
        Applies baryonification to the gridded map using the halo catalog.

        This method iterates over each halo in the `HaloNDCatalog`, calculating the necessary
        displacements based on the halo's mass, position, and other properties. It uses the given model to
        compute the displacement, updates the gridded map accordingly, and ensures that the total mass
        remains conserved.

        Returns
        -------
        new_map : ndarray
            A 2D or 3D numpy array representing the modified grid after baryonification.

        Raises
        ------
        AssertionError
            If the sum of the new map values does not match the sum of the original map values, 
            indicating an error in pixel regridding.

        NotImplementedError
            If the 3D ellipticity method is attempted, which is currently not supported.

        Notes
        -----
        - This method supports both 2D and 3D gridded maps.
        - The `ParamTabulatedProfile` model is required if property keys are used in the model.
        - Non-finite displacement values are set to zero to avoid issues with map updates.
        """

        
        cosmo = _runner_cosmology(self.cosmo)

        orig_map = self.GriddedMap.map
        new_map  = np.zeros(orig_map.shape, dtype = np.float64)
        bins     = self.GriddedMap.bins
        

        orig_map_flat = orig_map.flatten()
        pix_offsets   = np.zeros([orig_map_flat.size, len(orig_map.shape)])
        keys = _check_p_keys(self.model) #Names of extra (tabulated) model parameters

        if self.use_ellipticity and not self.GriddedMap.is2D:
            raise NotImplementedError("Currently not able to ellipticities with 3D maps.")

        H     = self._halo_table(cosmo, keys)
        res   = self.GriddedMap.res
        R_q   = self.epsilon_max * H['R'] / H['a'] #Comoving cutout radius
        width = 2 * R_q #At most the full box width (radius L/2), see _cutout_size

        #In regrid_pixels_2D (3D), column 0 of pix_offsets shifts axis 1 of the map and column 1 shifts
        #axis 0 (and column 2 shifts axis 2). So column 0 takes the y unit vector and column 1 the x one.
        columns = [1, 0] if self.GriddedMap.is2D else [1, 0, 2]

        def chunk_offsets(halos):

            #Axes of the cutout follow the halo x, y (, z) coordinates. The unit vectors use the circular
            #distance, and the displacement the elliptical one (if ellipticity is used).
            C = self._chunk_cutouts(halos, H, width, hats = True)

            #Compute the (comoving) displacement needed. The 1/res makes sure the offset is in units of pixel widths.
            #Non-finite values (eg. a halo outside the table range, or the pixel exactly at the
            #halo center, where x_hat = 0/0) are zeroed per halo, so they cannot erase the
            #displacements of other halos in the same pixels.
            offset = self._evaluate(self.model, 'displacement', cosmo, C['r'], C['hid'], halos, H) / res
            d      = offset[:, None] * C['hat'][:, columns]
            return C['inds'], np.where(np.isfinite(d), d, 0)

        #Add the offsets of each halo, in catalog order
        for inds, d in _run_chunks(chunk_offsets, self._halo_chunks(width), self._n_threads(), self.verbose, 'Baryonifying matter'):
            _add_rows_at(pix_offsets, inds, d)


        #Now that pixels have all been offset, let's regrid the map
        N = orig_map.shape[0]
        x = np.arange(N)
        
        #Need to split 2D vs 3D since we have separate numba functions for each.
        #(pix_offsets are already finite, since each halo's offsets were cleaned above.)
        if self.GriddedMap.is2D:
            x_grid, y_grid = np.meshgrid(x, x, indexing = 'xy')

            pix_offsets[:, 0] += x_grid.flatten()
            pix_offsets[:, 1] += y_grid.flatten()
            
            #Add pixels to the array. Calculations happen in-place
            regrid_pixels_2D(new_map, pix_offsets, orig_map_flat)
            
        else:
            x_grid, y_grid, z_grid = np.meshgrid(x, x, x, indexing = 'xy')

            pix_offsets[:, 0] += x_grid.flatten()
            pix_offsets[:, 1] += y_grid.flatten()
            pix_offsets[:, 2] += z_grid.flatten()
            
            #Add pixels to the array. Calculations happen in-place
            regrid_pixels_3D(new_map, pix_offsets, orig_map_flat)
            
            
        #Do a quick check that the sum is the same
        new_sum = np.sum(new_map)
        old_sum = np.sum(orig_map_flat)
        assert np.isclose(new_sum, old_sum), "ERROR in pixel regridding, sum(new_map) [%0.14e] != sum(oldmap) [%0.14e]" % (new_sum, old_sum)
            
        return new_map


class PaintProfilesGrid(DefaultRunnerGrid):
    """
    A class to paint profiles onto a gridded map using a halo catalog.

    The `PaintProfilesGrid` class inherits from `DefaultRunnerGrid` and is designed to generated a grid
    of a given property (mass, temperature, pressure) by painting halo profiles. It uses a halo catalog to 
    determine the necessary profiles based on halo properties, cosmological parameters, and a specified model.

    The returned map by default includes an integration over pixel size. For example, 
    passing model = density_profile will return a map of the Mass = density * dV and not the density alone.

    Methods
    -------
    process()
        Processes the gridded map by painting baryonic profiles and returns the modified grid.
    pick_indices(center, width, Npix)
        Helper function that Selects and returns indices around a center point, 
        accounting for periodic boundary conditions.
    """


    def process(self):
        """
        Applies profile painting to the gridded map using the halo catalog.

        This method iterates over each halo in the `HaloNDCatalog`, calculating the profile
        contributions based on the halo's mass, position, and other properties. It uses the provided model to
        compute the profile and updates the gridded map accordingly.

        Returns
        -------
        new_map : ndarray
            A 2D or 3D numpy array representing the modified grid after painting the baryonic profiles.

        Raises
        ------
        AssertionError
            If a model is not provided or if the provided model is not an instance of `ParamTabulatedProfile`
            when property keys are used.

        ValueError
            If `use_ellipticity` is True and the 3D map painting method is attempted, which is currently not supported.

        Notes
        -----
        - This method supports both 2D and 3D gridded maps.
        - The `ParamTabulatedProfile` model is required if property keys are used in the model.
        - Non-finite profile values are set to zero to avoid issues with map updates.
        """

        cosmo = _runner_cosmology(self.cosmo)

        orig_map = self.GriddedMap.map
        new_map  = np.zeros(orig_map.size, dtype = np.float64)
        
        bins = self.GriddedMap.bins
        keys = _check_p_keys(self.model) #Names of extra (tabulated) model parameters

        dV = np.power(self.GriddedMap.res, 2 if self.GriddedMap.is2D else 3)

        if self.use_ellipticity and not self.GriddedMap.is2D:
            raise ValueError("use_ellipticity is not implemented for 3D maps")

        H      = self._halo_table(cosmo, keys)
        res    = self.GriddedMap.res
        R_j    = H['R'] / H['a'] #in comoving Mpc
        width  = 2 * self.epsilon_max * R_j #Even, at most the full box width, see _cutout_size. Can't skip small halos because we still must sum all contributions to a pixel
        method = 'projected' if self.GriddedMap.is2D else 'real'

        def chunk_paint(halos):

            #Axes of the cutout follow the halo x, y (, z) coordinates, and the distance is elliptical if used
            C = self._chunk_cutouts(halos, H, width)

            #A halo can sit exactly on a pixel center (r = 0), where profiles/projections are ill-defined.
            #Use a tiny floor on the radius. Use `ConvolvedProfile` for a proper pixel-averaged value.
            r_eval   = np.clip(C['r'], res * 1e-3, None)
            Painting = self._evaluate(self.model, method, cosmo, r_eval, C['hid'], halos, H)

            mask = np.isfinite(Painting) #Find which part of map cannot be modified due to out-of-bounds errors
            mask = mask & (C['r'] < R_j[halos][C['hid']]*self.epsilon_max)

            return C['inds'], np.where(mask, Painting, 0) #Set those tSZ values to 0

        #Add the profiles to the new map at the right indices, halo by halo in catalog order
        for inds, Painting in _run_chunks(chunk_paint, self._halo_chunks(width), self._n_threads(), self.verbose, 'Painting field'):
            _add_at(new_map, inds, Painting)
            
        #Add a factor of the map pixel area/volume if requested by the user.
        #This helps convert a density map to mass map (for example).
        if self.include_pixel_size: new_map *= dV

        new_map = new_map.reshape(orig_map.shape)
        
        return new_map
    


class PaintProfilesAnisGrid(PaintProfilesGrid):

    def __init__(self, HaloNDCatalog, GriddedMap, epsilon_max, model, Tracer_model, Mtot_model, 
                 background_val, global_tracer_fraction, 
                 mass_def = None,
                 include_pixel_size = True, use_ellipticity = False, verbose = True, n_jobs = 1):

        self.Tracer_model   = Tracer_model
        self.Mtot_model     = Mtot_model
        self.background_val = background_val
        self.global_tracer_fraction = global_tracer_fraction
        super().__init__(HaloNDCatalog, GriddedMap, epsilon_max, model, use_ellipticity, mass_def, include_pixel_size, verbose, n_jobs)
    
    
    def process(self):

        assert self.GriddedMap.is2D == True, "Can only paint tSZ on 2D maps. You have passed a 3D Map"

        cosmo = _runner_cosmology(self.cosmo)

        orig_map = self.GriddedMap.map
        new_map  = np.zeros(orig_map.size, dtype = np.float64)

        orig_map_flattened = orig_map.flatten()
        
        bins = self.GriddedMap.bins
        res  = self.GriddedMap.res
        keys = vars(self.model).get('p_keys', []) #Check if model has property keys

        #First we need to generate a model for the total mass distribution, according to the mass model
        Mtot_map = PaintProfilesGrid(include_pixel_size = False,
                                     HaloNDCatalog = self.HaloNDCatalog, GriddedMap = self.GriddedMap, 
                                     epsilon_max = self.epsilon_max, model = self.Mtot_model, 
                                     use_ellipticity = self.use_ellipticity,
                                     mass_def = self.mass_def, verbose = self.verbose, n_jobs = self.n_jobs).process()
        Mtot_map = Mtot_map.flatten() #Put it back in 1D array
        
        #The Mtot_map is painted without the pixel size, so it is a surface density.
        #The uniform background must be added in the same units: a density times the projection length.
        dL = (2 * _get_parameter(self.Mtot_model, 'proj_cutoff')) #Factor of 2 since proj_cutoff == Lproj/2
        rho_halos = np.average(Mtot_map) / dL

        #Now add the background contribution (we so far only have the halo contribution)
        #Force the background to be positive, incase the pasted density is larger than the box size.
        rho_m     = cosmo.rho_x(1/(self.HaloNDCatalog.redshift + 1), species = 'matter', is_comoving = True)
        drho_m    = np.clip(rho_m - rho_halos, 0, None)
        Mtot_map += dL * drho_m

        if self.verbose:
            print(f"Inputted halos contribute {100*(rho_halos/rho_m):0.2f}% of the total matter density.")
            print(f"Remaining density is assigned to a uniform background.")

        if rho_halos > rho_m:
            warnings.warn("Inputted halos contribute more mass than is available for this mean matter density."
                          "Your Mtot_model profiles are either too extended or you are using the wrong cosmology.")

        #We are ready to loop over halos now!
        H     = self._halo_table(cosmo, keys)
        R_j   = H['R'] / H['a'] #in comoving Mpc
        width = 2 * self.epsilon_max * R_j #Even, at most the full box width, see _cutout_size. Can't skip small halos because we still must sum all contributions to a pixel

        def chunk_paint(halos):

            #Axis 0 of the cutout follows the halo x-coordinate and axis 1 the y-coordinate
            C = self._chunk_cutouts(halos, H, width)

            #Tiny floor on the radius to avoid r = 0 (see PaintProfilesGrid)
            r_eval   = np.clip(C['r'], res * 1e-3, None)
            Painting = self._evaluate(self.model, 'projected', cosmo, r_eval, C['hid'], halos, H)
            Canvas   = self._evaluate(self.Tracer_model, 'projected', cosmo, r_eval, C['hid'], halos, H)
            Canvas   = np.where(np.isfinite(Canvas), Canvas, 0)
            Mtot     = Mtot_map[C['inds']]
            Mfrac    = np.divide(Canvas, Mtot, out = np.zeros_like(Canvas), where = Mtot > 0)
            Mfrac   *= orig_map_flattened[C['inds']]

            mask = np.isfinite(Painting) #Remove parts of map that have irregular values
            mask = mask & (C['r'] < R_j[halos][C['hid']]*self.epsilon_max)

            Painting = np.where(mask, Painting, 0) #Set bad regions of mask to 0

            #The profiles weighted by the mass fractions of the tracer particles
            return C['inds'], Painting * Mfrac

        #Add them to the new map at the right indices, halo by halo in catalog order
        for inds, values in _run_chunks(chunk_paint, self._halo_chunks(width), self._n_threads(), self.verbose, 'Painting field'):
            _add_at(new_map, inds, values)

        #Missing mass was assigned to uniform background. Here we account for that background's contribution
        #The Mtot_map here already has the contribution from dL * drho_m added to it.
        Mfrac    = np.divide(dL * drho_m, Mtot_map, out = np.zeros_like(Mtot_map), where = Mtot_map > 0)
        Mfrac   *= orig_map_flattened
        new_map += self.background_val * self.global_tracer_fraction * Mfrac
        new_map  = new_map.reshape(orig_map.shape)

        #Add a factor of the map pixel area if requested by the user.
        #This helps convert a density map to mass map (for example).
        if self.include_pixel_size:
            new_map *= self.GriddedMap.res**2

        
        return new_map    