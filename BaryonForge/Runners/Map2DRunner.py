import numpy as np
import pyccl as ccl
import warnings

from tqdm import tqdm
from numba import njit
from ..utils import ParamTabulatedProfile
from ..utils.Tabulate import _get_parameter

__all__ = ['DefaultRunnerGrid', 'BaryonifyGrid', 'PaintProfilesGrid', 'PaintProfilesAnisGrid',
           'regrid_pixels_2D', 'regrid_pixels_3D']

@njit
def regrid_pixels_2D(grid, pix_positions, pix_values):

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
def regrid_pixels_3D(grid, pix_positions, pix_values):

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
                    

#Quickly run the function once so it compiles and initializes
regrid_pixels_2D(np.zeros([5, 5]),    np.ones([2, 2]), np.ones(2))
regrid_pixels_3D(np.zeros([5, 5, 5]), np.ones([2, 3]), np.ones(2))

                        
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
        If set to True, a NotImplementedError is raised, as this mode has not yet been 
        implemented for curved, full-sky baryonification.
    
    mass_def : object, optional
        An instance of a mass definition object from the CCL (Core Cosmology Library), specifying 
        the mass definition to be used. Default is `ccl.halos.massdef.MassDef(200, 'critical')`.
    
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
        If `use_ellipticity` is True and required columns ('q_ell', 'c_ell', 'A_ell') are missing in the 
        `HaloNDCatalog`.
    NotImplementedError
        If attempting to use the 3D ellipticity method, which is not yet verified.
    """
    
    def __init__(self, HaloNDCatalog, GriddedMap, epsilon_max, model, use_ellipticity = False,
                 mass_def = ccl.halos.massdef.MassDef(200, 'critical'), include_pixel_size = True, verbose = True):

        self.HaloNDCatalog = HaloNDCatalog
        self.GriddedMap    = GriddedMap
        self.cosmo = HaloNDCatalog.cosmology
        self.model = model
        
        
        self.epsilon_max = epsilon_max
        self.mass_def    = mass_def
        self.verbose     = verbose
        
        self.use_ellipticity    = use_ellipticity
        self.include_pixel_size = include_pixel_size
        
        #Assert that all the required quantities are in the input catalog
        if use_ellipticity:
            
            names = HaloNDCatalog.cat.dtype.names
            
            assert 'q_ell' in names, "The 'q_ell' column is missing, but you set use_ellipticity = True"
            if not GriddedMap.is2D: assert 'c_ell' in names, "The 'c_ell' column is missing, but you set use_ellipticity = True"
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
            A 1D array representing the vector to be rotated. It will be normalized within the method.
        
        q : float
            The ellipticity parameter, used to compute the shear transformation.

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

        A /= np.linalg.norm(A)

        if len(A) == 1:
            raise  ValueError("Can't rotate a 1-dimensional vector")
        
        elif len(A) == 2:
            
            #The 2D rotation is done using routines implemented in the galsim Shear class
            
            ref  = np.array([1., 0.])
            beta = np.arccos(np.dot(A, ref))
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

            v = np.cross(A, ref)
            c = np.dot(A, ref)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], 
                             [v[2], 0, -v[0]], 
                             [-v[1], v[0], 0]])

            Rmat = np.eye(3) + kmat + np.dot(kmat, kmat) * ((1 - c) / (s ** 2))

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
            The half-width of the selection range. The method selects indices from `center - width` to `center + width`.

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

        
        cosmo = ccl.Cosmology(Omega_c = self.cosmo['Omega_m'] - self.cosmo['Omega_b'],
                              Omega_b = self.cosmo['Omega_b'], h = self.cosmo['h'],
                              sigma8  = self.cosmo['sigma8'],  n_s = self.cosmo['n_s'],
                              matter_power_spectrum = 'linear')
        cosmo.compute_sigma()

        orig_map = self.GriddedMap.map
        new_map  = np.zeros(orig_map.shape, dtype = np.float64)
        bins     = self.GriddedMap.bins
        

        orig_map_flat = orig_map.flatten()
        pix_offsets   = np.zeros([orig_map_flat.size, len(orig_map.shape)])
        keys          = vars(self.model).get('p_keys', []) #Check if model has property keys

        if len(keys) > 0:
            txt = (f"You asked to use {keys} properties in Baryonification. You must pass a ParamTabulatedProfile"
                   f"as the model. You have passed {type(self.model)} instead")
            assert isinstance(self.model, ParamTabulatedProfile), txt

        for j in tqdm(range(self.HaloNDCatalog.cat.size), desc = 'Baryonifying matter', disable = not self.verbose):

            M_j = self.HaloNDCatalog.cat['M'][j]
            x_j = self.HaloNDCatalog.cat['x'][j]
            y_j = self.HaloNDCatalog.cat['y'][j]
            z_j = self.HaloNDCatalog.cat['z'][j] #THIS IS A CARTESIAN COORDINATE, NOT REDSHIFT
            o_j = {key : self.HaloNDCatalog.cat[key][j] for key in keys} #Other properties

            a_j = 1/(1 + self.HaloNDCatalog.redshift)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) #in physical Mpc
            R_q = self.epsilon_max * R_j/a_j
            R_q = np.clip(R_q, 0, np.max(self.GriddedMap.bins)/2) #Can't query distances more than half box-size.
            
            if self.use_ellipticity:
                q_j = self.HaloNDCatalog.cat['q_ell'][j]
                A_j = self.HaloNDCatalog.cat['A_ell'][j]
                A_j = A_j/np.sqrt(np.sum(A_j**2))
            
            res    = self.GriddedMap.res
            Nsize  = 2 * R_q / res
            Nsize  = int(Nsize // 2)*2 #Force it to be even
            Nsize  = np.clip(Nsize, 2, bins.size//2)

            x  = np.linspace(-Nsize/2, Nsize/2, Nsize) * res
            cutout_width = Nsize//2
            
            if self.GriddedMap.is2D:

                shape = (Nsize, Nsize)
                
                x_cen  = np.argmin(np.abs(bins - x_j))
                y_cen  = np.argmin(np.abs(bins - y_j))
                x_inds = self.pick_indices(x_cen, cutout_width, self.GriddedMap.Npix)
                y_inds = self.pick_indices(y_cen, cutout_width, self.GriddedMap.Npix)
                inds   = self.GriddedMap.inds[x_inds, :][:, y_inds].flatten()
                
                #Get offsets between halo position and pixel center
                dx = bins[x_cen] - x_j
                dy = bins[y_cen] - y_j
                
                assert np.logical_and(dx <= res, dy <= res), "Halo offsets (%0.2f, %0.2f) are larger than res (%0.2f)" % (dx, dy, res)
                
                x_grid, y_grid = np.meshgrid(x, x, indexing = 'xy')
                r_grid = np.sqrt( (x_grid + dx)**2 +  (y_grid + dy)**2 )

                x_hat  = (x_grid + dx)/r_grid
                y_hat  = (y_grid + dy)/r_grid

                #If ellipticity exists, then account for it
                if self.use_ellipticity:
                    assert q_j > 0, "The axis ratio in halo %d is not positive" % j

                    Rmat = self.build_Rmat(A_j, q_j)
                    x_grid_ell, y_grid_ell = (self.coord_array(x_grid + dx, y_grid + dy) @ Rmat).T
                    r_grid = np.sqrt(x_grid_ell**2 + y_grid_ell**2).reshape(x_grid_ell.shape)

                #Compute the (comoving) displacement needed and add it to pixel offsets
                #The 1/res makes sure the offset is in units of pixel widths
                offset = self.model.displacement(r_grid.flatten(), M_j, a_j, **o_j) / res
                pix_offsets[inds, 0] += offset * x_hat.flatten()
                pix_offsets[inds, 1] += offset * y_hat.flatten()
                
            
            else:
                shape = (Nsize, Nsize, Nsize)

                x_cen  = np.argmin(np.abs(bins - x_j))
                y_cen  = np.argmin(np.abs(bins - y_j))
                z_cen  = np.argmin(np.abs(bins - z_j))
                x_inds = self.pick_indices(x_cen, cutout_width, self.GriddedMap.Npix)
                y_inds = self.pick_indices(y_cen, cutout_width, self.GriddedMap.Npix)
                z_inds = self.pick_indices(z_cen, cutout_width, self.GriddedMap.Npix)
                inds   = self.GriddedMap.inds[x_inds, ...][:, y_inds, :][..., z_inds].flatten()
                
                #Get offsets between halo position and pixel center
                dx = bins[x_cen] - x_j
                dy = bins[y_cen] - y_j
                dz = bins[z_cen] - z_j
                
                x_grid, y_grid, z_grid = np.meshgrid(x, x, x, indexing = 'xy')
                r_grid = np.sqrt( (x_grid + dx)**2 +  (y_grid + dy)**2 +  (z_grid + dz)**2 )

                x_hat  = (x_grid + dx)/r_grid
                y_hat  = (y_grid + dy)/r_grid
                z_hat  = (z_grid + dz)/r_grid
                
                #If ellipticity exists, then account for it
                if self.use_ellipticity:

                    raise NotImplementedError("Currently not able to ellipticities with 3D maps.")
                    assert q_j > 0, "The axis ratio in halo %d is zero" % j

                    Rmat = self.build_Rmat(A_j, np.array([0., 1., 0.]))
                    x_grid_ell, y_grid_ell, z_grid_ell = (self.coord_array(x_grid + dx, y_grid + dy, z_grid + dz) @ Rmat).T
                    r_grid = np.sqrt(x_grid_ell**2/ar_j**2 + 
                                     y_grid_ell**2/br_j**2 +
                                     z_grid_ell**2/cr_j**2).reshape(x_grid_ell.shape)

                
                #Compute the (comoving) displacement needed   
                #The 1/res makes sure the offset is in units of pixel widths 
                offset = self.model.displacement(r_grid.flatten(), M_j, a_j, **o_j) / res
                pix_offsets[inds, 0] += offset * x_hat.flatten()
                pix_offsets[inds, 1] += offset * y_hat.flatten()
                pix_offsets[inds, 2] += offset * z_hat.flatten()
            
            
        #Now that pixels have all been offset, let's regrid the map
        N = orig_map.shape[0]
        x = np.arange(N)
        
         #Need to split  2D vs 3D since we have separate numba functions for each
        if self.GriddedMap.is2D:
            x_grid, y_grid = np.meshgrid(x, x, indexing = 'xy')
        
            pix_offsets = np.where(np.isfinite(pix_offsets), pix_offsets, 0)
            pix_offsets[:, 0] += x_grid.flatten()
            pix_offsets[:, 1] += y_grid.flatten()
            
            #Add pixels to the array. Calculations happen in-place
            regrid_pixels_2D(new_map, pix_offsets, orig_map_flat)
            
        else:
            x_grid, y_grid, z_grid = np.meshgrid(x, x, x, indexing = 'xy')
        
            pix_offsets = np.where(np.isfinite(pix_offsets), pix_offsets, 0)
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

    
    def pick_indices(self, center, width, Npix):
        """
        Selects and returns indices around a center point, accounting for periodic boundary conditions.

        This method takes a central index and a width and returns an array of indices around the center,
        wrapping around if the indices go beyond the boundaries of the grid.

        Parameters
        ----------
        center : int
            The central index around which indices are selected.

        width : int
            The half-width of the selection range. The method selects indices from `center - width` to `center + width`.

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

        cosmo = ccl.Cosmology(Omega_c = self.cosmo['Omega_m'] - self.cosmo['Omega_b'],
                              Omega_b = self.cosmo['Omega_b'], h = self.cosmo['h'],
                              sigma8  = self.cosmo['sigma8'],  n_s = self.cosmo['n_s'],
                              matter_power_spectrum = 'linear')
        cosmo.compute_sigma()

        orig_map = self.GriddedMap.map
        new_map  = np.zeros(orig_map.size, dtype = np.float64)
        
        grid = self.GriddedMap.grid
        bins = self.GriddedMap.bins
        keys = vars(self.model).get('p_keys', []) #Check if model has property keys

        if len(keys) > 0:
            txt = (f"You asked to use {keys} properties in Baryonification. You must pass a ParamTabulatedProfile"
                   f"as the model. You have passed {type(self.model)} instead")
            assert isinstance(self.model, ParamTabulatedProfile), txt

        dV = np.power(self.GriddedMap.res, 2 if self.GriddedMap.is2D else 3)

        for j in tqdm(range(self.HaloNDCatalog.cat.size), desc = 'Painting field', disable = not self.verbose):

            M_j = self.HaloNDCatalog.cat['M'][j]
            x_j = self.HaloNDCatalog.cat['x'][j]
            y_j = self.HaloNDCatalog.cat['y'][j]
            z_j = self.HaloNDCatalog.cat['z'][j] #THIS IS A CARTESIAN COORDINATE, NOT REDSHIFT
            o_j = {key : self.HaloNDCatalog.cat[key][j] for key in keys} #Other properties
            
            a_j = 1/(1 + self.HaloNDCatalog.redshift)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) / a_j #in comoving Mpc

            if self.use_ellipticity:
                q_j = self.HaloNDCatalog.cat['q_ell'][j]
                A_j = self.HaloNDCatalog.cat['A_ell'][j]
                A_j = A_j/np.sqrt(np.sum(A_j**2))
            
            res    = self.GriddedMap.res
            Nsize  = 2 * self.epsilon_max * R_j / res
            Nsize  = int(Nsize // 2)*2 #Force it to be even
            Nsize  = np.clip(Nsize, 2, bins.size//2) #Can't skip small halos because we still must sum all contributions to a pixel

            x = np.linspace(-Nsize/2, Nsize/2, Nsize) * res
            cutout_width = Nsize//2

            if self.GriddedMap.is2D:
                
                x_cen  = np.argmin(np.abs(bins - x_j))
                y_cen  = np.argmin(np.abs(bins - y_j))
                x_inds = self.pick_indices(x_cen, cutout_width, self.GriddedMap.Npix)
                y_inds = self.pick_indices(y_cen, cutout_width, self.GriddedMap.Npix)
                inds   = self.GriddedMap.inds[x_inds, :][:, y_inds].flatten()
                
                #Get offsets between halo position and pixel center
                dx = bins[x_cen] - x_j
                dy = bins[y_cen] - y_j
                
                assert np.logical_and(dx <= res, dy <= res), "Halo offsets (%0.2f, %0.2f) are larger than res (%0.2f)" % (dx, dy, res)
                
                profile = self.model.projected

                x_grid, y_grid = np.meshgrid(x, x, indexing = 'xy')
                r_grid = np.sqrt( (x_grid + dx)**2 +  (y_grid + dy)**2 )

                #If ellipticity exists, then account for it
                if self.use_ellipticity:
                    assert q_j > 0, "The axis ratio in halo %d is zero" % j

                    Rmat = self.build_Rmat(A_j, q_j)
                    x_grid_ell, y_grid_ell = (self.coord_array(x_grid + dx, y_grid + dy) @ Rmat).T
                    r_grid = np.sqrt(x_grid_ell**2 + y_grid_ell**2).reshape(x_grid_ell.shape)
            
            else:
                
                shape  = (Nsize, Nsize, Nsize)
                x_cen  = np.argmin(np.abs(bins - x_j))
                y_cen  = np.argmin(np.abs(bins - y_j))
                z_cen  = np.argmin(np.abs(bins - z_j))
                x_inds = self.pick_indices(x_cen, cutout_width, self.GriddedMap.Npix)
                y_inds = self.pick_indices(y_cen, cutout_width, self.GriddedMap.Npix)
                z_inds = self.pick_indices(z_cen, cutout_width, self.GriddedMap.Npix)
                inds   = self.GriddedMap.inds[x_inds, ...][:, y_inds, :][..., z_inds].flatten()
                
                #Get offsets between halo position and pixel center
                dx = bins[x_cen] - x_j
                dy = bins[y_cen] - y_j
                dz = bins[z_cen] - z_j
                
                profile = self.model.real

                x_grid, y_grid, z_grid = np.meshgrid(x, x, x, indexing = 'xy')
                r_grid = np.sqrt( (x_grid + dx)**2 +  (y_grid + dy)**2 +  (z_grid + dz)**2 )
                

                #If ellipticity exists, then account for it
                if self.use_ellipticity:
                    
                    raise ValueError("use_ellipticity is not implemented for 3D maps")
                    
                    assert q_j > 0, "The axis ratio in halo %d is zero" % j

                    Rmat = self.build_Rmat(A_j, np.array([0., 1., 0.]))
                    x_grid_ell, y_grid_ell, z_grid_ell = (self.coord_array(x_grid + dx, y_grid + dy, z_grid + dz) @ Rmat).T
                    r_grid = np.sqrt(x_grid_ell**2/ar_j**2 + 
                                     y_grid_ell**2/br_j**2 +
                                     z_grid_ell**2/cr_j**2).reshape(x_grid_ell.shape)

        
            Painting = profile(cosmo, r_grid.flatten(), M_j, a_j, **o_j)
            
            mask = np.isfinite(Painting) #Find which part of map cannot be modified due to out-of-bounds errors
            mask = mask & (r_grid.flatten() < R_j*self.epsilon_max)
            if mask.sum() == 0: continue
                
            Painting = np.where(mask, Painting, 0) #Set those tSZ values to 0

            #Add the offsets to the new map at the right indices
            new_map[inds] += Painting
            
        #Add a factor of the map pixel area/volume if requested by the user.
        #This helps convert a density map to mass map (for example).
        if self.include_pixel_size: new_map *= dV

        new_map = new_map.reshape(orig_map.shape)
        
        return new_map
    


class PaintProfilesAnisGrid(PaintProfilesGrid):

    def __init__(self, HaloNDCatalog, GriddedMap, epsilon_max, model, Tracer_model, Mtot_model, 
                 background_val, global_tracer_fraction, 
                 mass_def = ccl.halos.massdef.MassDef(200, 'critical'), 
                 include_pixel_size = True, use_ellipticity = False, verbose = True):
        
        self.Tracer_model   = Tracer_model
        self.Mtot_model     = Mtot_model
        self.background_val = background_val
        self.global_tracer_fraction = global_tracer_fraction
        super().__init__(HaloNDCatalog, GriddedMap, epsilon_max, model, use_ellipticity, mass_def, include_pixel_size, verbose)
    
    
    def process(self):

        assert self.GriddedMap.is2D == True, "Can only paint tSZ on 2D maps. You have passed a 3D Map"

        cosmo = ccl.Cosmology(Omega_c = self.cosmo['Omega_m'] - self.cosmo['Omega_b'],
                              Omega_b = self.cosmo['Omega_b'], h = self.cosmo['h'],
                              sigma8  = self.cosmo['sigma8'],  n_s = self.cosmo['n_s'],
                              matter_power_spectrum = 'linear')
        cosmo.compute_sigma()

        orig_map = self.GriddedMap.map
        new_map  = np.zeros(orig_map.size, dtype = np.float64)

        orig_map_flattened = orig_map.flatten()
        
        grid = self.GriddedMap.grid
        bins = self.GriddedMap.bins
        res  = self.GriddedMap.res
        keys = vars(self.model).get('p_keys', []) #Check if model has property keys

        #First we need to generate a model for the total mass distribution, according to the mass model
        Mtot_map = PaintProfilesGrid(include_pixel_size = False,
                                     HaloNDCatalog = self.HaloNDCatalog, GriddedMap = self.GriddedMap, 
                                     epsilon_max = self.epsilon_max, model = self.Mtot_model, 
                                     use_ellipticity = self.use_ellipticity,
                                     mass_def = self.mass_def, verbose = self.verbose).process()
        Mtot_map = Mtot_map.flatten() #Put it back in 1D array
        
        #Volume of the cell is different depending on whether we've projected (or not) down one axis
        if self.GriddedMap.is2D:
            dL = (2 * _get_parameter(self.Mtot_model, 'proj_cutoff')) #Factor of 2 since proj_cutoff == Lproj/2
            dV = np.power(res, 2) * dL 
            rho_halos = np.average(Mtot_map) / dL
        else:
            dV = np.power(res, 3)
            rho_halos = np.average(Mtot_map)

        #Now add the background contribution (we so far only have the halo contribution)
        #Force the background to be positive, incase the pasted density is larger than the box size.
        rho_m     = cosmo.rho_x(1/(self.HaloNDCatalog.redshift + 1), species = 'matter', is_comoving = True)
        drho_m    = np.clip(rho_m - rho_halos, 0, None)
        Mtot_map += dV * drho_m

        if self.verbose:
            print(f"Inputted halos contribute {100*(rho_halos/rho_m):0.2f}% of the total matter density.")
            print(f"Remaining density is assigned to a uniform background.")

        if rho_halos > rho_m:
            warnings.warn("Inputted halos contribute more mass than is available for this mean matter density."
                          "Your Mtot_model profiles are either too extended or you are using the wrong cosmology.")

        #We are ready to loop over halos now!
        for j in tqdm(range(self.HaloNDCatalog.cat.size), desc = 'Painting field', disable = not self.verbose):

            M_j = self.HaloNDCatalog.cat['M'][j]
            x_j = self.HaloNDCatalog.cat['x'][j]
            y_j = self.HaloNDCatalog.cat['y'][j]
            z_j = self.HaloNDCatalog.cat['z'][j] #THIS IS A CARTESIAN COORDINATE, NOT REDSHIFT

            o_j = {key : self.HaloNDCatalog.cat[key][j] for key in keys} #Other properties
            
            a_j = 1/(1 + self.HaloNDCatalog.redshift)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) / a_j #in comoving Mpc
            
            res    = self.GriddedMap.res
            Nsize  = 2 * self.epsilon_max * R_j / res
            Nsize  = int(Nsize // 2)*2 #Force it to be even
            
            if self.use_ellipticity:
                q_j = self.HaloNDCatalog.cat['q_ell'][j]
                A_j = self.HaloNDCatalog.cat['A_ell'][j]
                A_j = A_j/np.sqrt(np.sum(A_j**2))
            
            Nsize = np.clip(Nsize, 2, bins.size//2) #Can't skip small halos because we still must sum all contributions to a pixel

            x = np.linspace(-Nsize/2, Nsize/2, Nsize) * res
            cutout_width = Nsize//2

            if self.GriddedMap.is2D:
                x_cen  = np.argmin(np.abs(bins - x_j))
                y_cen  = np.argmin(np.abs(bins - y_j))
                x_inds = self.pick_indices(x_cen, cutout_width, self.GriddedMap.Npix)
                y_inds = self.pick_indices(y_cen, cutout_width, self.GriddedMap.Npix)
                inds   = self.GriddedMap.inds[x_inds, :][:, y_inds].flatten()

                #Get offsets between halo position and pixel center
                dx = bins[x_cen] - x_j
                dy = bins[y_cen] - y_j

                assert np.logical_and(dx <= res, dy <= res), "Halo offsets (%0.2f, %0.2f) are larger than res (%0.2f)" % (dx, dy, res)

                Paint  = self.model.projected
                Tracer = self.Tracer_model.projected

                x_grid, y_grid = np.meshgrid(x, x, indexing = 'xy')
                r_grid = np.sqrt( (x_grid + dx)**2 +  (y_grid + dy)**2 )

                #If ellipticity exists, then account for it
                if self.use_ellipticity:
                    assert q_j > 0, "The axis ratio in halo %d is zero" % j

                    Rmat = self.build_Rmat(A_j, q_j)
                    x_grid_ell, y_grid_ell = (self.coord_array(x_grid + dx, y_grid + dy) @ Rmat).T
                    r_grid = np.sqrt(x_grid_ell**2 + y_grid_ell**2).reshape(x_grid_ell.shape)
            
            else:
                
                shape  = (Nsize, Nsize, Nsize)
                x_cen  = np.argmin(np.abs(bins - x_j))
                y_cen  = np.argmin(np.abs(bins - y_j))
                z_cen  = np.argmin(np.abs(bins - z_j))
                x_inds = self.pick_indices(x_cen, cutout_width, self.GriddedMap.Npix)
                y_inds = self.pick_indices(y_cen, cutout_width, self.GriddedMap.Npix)
                z_inds = self.pick_indices(z_cen, cutout_width, self.GriddedMap.Npix)
                inds   = self.GriddedMap.inds[x_inds, ...][:, y_inds, :][..., z_inds].flatten()

                #Get offsets between halo position and pixel center
                dx = bins[x_cen] - x_j
                dy = bins[y_cen] - y_j
                dz = bins[z_cen] - z_j

                Paint  = self.model.real
                Tracer = self.Tracer_model.real

                x_grid, y_grid, z_grid = np.meshgrid(x, x, x, indexing = 'xy')
                r_grid = np.sqrt( (x_grid + dx)**2 +  (y_grid + dy)**2 +  (z_grid + dz)**2 )

                #If ellipticity exists, then account for it
                if self.use_ellipticity:

                    raise ValueError("use_ellipticity is not implemented for 3D maps")

                    assert q_j > 0, "The axis ratio in halo %d is zero" % j

                    Rmat = self.build_Rmat(A_j, np.array([0., 1., 0.]))
                    x_grid_ell, y_grid_ell, z_grid_ell = (self.coord_array(x_grid + dx, y_grid + dy, z_grid + dz) @ Rmat).T
                    r_grid = np.sqrt(x_grid_ell**2/ar_j**2 + 
                                     y_grid_ell**2/br_j**2 +
                                     z_grid_ell**2/cr_j**2).reshape(x_grid_ell.shape)

            Painting = Paint(cosmo,  r_grid.flatten(), M_j, a_j, **o_j)
            Canvas   = Tracer(cosmo, r_grid.flatten(), M_j, a_j, **o_j)
            Canvas   = np.where(np.isfinite(Canvas) & np.invert(np.isnan(Canvas)), Canvas, 0)
            Mfrac    = np.divide(Canvas, Mtot_map[inds], out = np.zeros_like(Canvas), where = Mtot_map[inds] > 0)
            Mfrac   *= orig_map_flattened[inds]
                        
            mask = np.isfinite(Painting) & np.invert(np.isnan(Painting)) #Remove parts of map that have irregular values
            mask = mask & (r_grid.flatten() < R_j*self.epsilon_max)
            if mask.sum() == 0: continue
                
            Painting = np.where(mask, Painting, 0) #Set bad regions of mask to 0

            #Add the offsets to the new map at the right indices, 
            #and using the mass fractions of the tracer particles
            new_map[inds] += Painting * Mfrac

        #Missing mass was assigned to uniform background. Here we account for that background's contribution
        #The Mtot_map here already has the contribution from dV * drho_m added to it.
        Mfrac    = np.divide(dV * drho_m, Mtot_map, out = np.zeros_like(Mtot_map), where = Mtot_map > 0)
        Mfrac   *= orig_map_flattened
        new_map += self.background_val * self.global_tracer_fraction * Mfrac
        new_map  = new_map.reshape(orig_map.shape)

        #Add a factor of the map pixel area/volume if requested by the user.
        #This helps convert a density map to mass map (for example).
        if self.include_pixel_size: 
            new_map *= np.power(self.GriddedMap.res, 2 if self.GriddedMap.is2D else 3)

        
        return new_map    