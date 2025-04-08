
import numpy as np
import pyccl as ccl
import healpy as hp
from numba import njit
import warnings

from scipy import interpolate
from tqdm import tqdm
from ..utils import ParamTabulatedProfile
from ..utils.Tabulate import _get_parameter

__all__ = ['DefaultRunner', 'BaryonifyShell', 'PaintProfilesShell', 'PaintProfilesAnisShell',
           'regrid_pixels_hpix']

@njit
def regrid_pixels_hpix(hmap, parent_pix_vals, child_pix, child_weights):
    """
    Reassigns displaced HEALPix pixels back to the original map grid.

    This function modifies a HEALPix map (`hmap`) by redistributing pixel values
    from displaced pixels (`parent_pix_vals`) to their corresponding positions
    in the original grid. Each displaced pixel's value contributes to four 
    neighboring pixels, determined by `child_pix`, with contributions weighted 
    by `child_weights`.

    Parameters
    ----------
    hmap : ndarray
        The HEALPix map array to which the displaced pixel values will be assigned.
        This array will be modified in place by having values added to it.

    parent_pix_vals : ndarray of shape (N,)
        The array containing the values of displaced pixels. These are the values
        to be re-assigned to the original map grid.

    child_pix : ndarray of shape (N, 4)
        A 2D array where N is the number of displaced pixels. Each row contains the 
        indices of the four pixels in `hmap` to which the corresponding displaced 
        pixel contributes.

    child_weights : ndarray of shape (N, 4)
        A 2D array where N is the number of displaced pixels. Each row contains the 
        weights of the contributions of the corresponding displaced pixel to the four 
        pixels in `hmap`. The weights should sum to 1 for each displaced pixel.

    Returns
    -------
    hmap : ndarray
        The modified HEALPix map with the displaced pixel values assigned back
        to the original grid.

    Notes
    -----
    - This function utilizes Numba's `@njit` decorator for just-in-time compilation,
      optimizing performance.
    - The `child_pix` and `child_weights` arrays can be obtained using 
      `hp.interp_values()` from the HEALPix library.
    - The code runs this function once before import so that it's already compiled and
      ready for the njit speedup
    """
    
    for i in range(parent_pix_vals.size):

        for j in range(4):
            
            hmap[child_pix[i, j]] += child_weights[i, j] * parent_pix_vals[i]

    
    return hmap

#Quickly run the function once so it compiles and initializes
regrid_pixels_hpix(np.zeros(10), np.ones(5), np.ones([5, 4], dtype = int), np.ones([5, 4]) * 0.25)



class DefaultRunner(object):
    """
    A utility class for handling input/output operations related to halo lightcone catalogs and lightcone shells.

    This class provides methods for managing and processing data associated with halo lightcone catalogs,
    including constructing rotation matrices and generating coordinate arrays.

    Parameters
    ----------
    HaloLightConeCatalog : object
        An instance of a halo lightcone catalog, which contains data about the halos and their properties.
        It must have a `cosmology` attribute to specify the cosmological parameters.
    
    LightconeShell : object
        An instance of a lightcone shell, representing a thin shell in the lightcone where halos are located.
    
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
    HaloLightConeCatalog : object
        The halo lightcone catalog instance.
    
    LightconeShell : object
        The lightcone shell instance.
    
    cosmo : object
        The cosmology object extracted from `HaloLightConeCatalog`.
    
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
    build_Rmat(A, ref)
        Constructs a 2x2 rotation matrix to rotate vector A to align with the reference vector ref.
    
    coord_array(*args)
        Flattens and stacks input arrays into a 2D array of coordinates.

    Raises
    ------
    NotImplementedError
        If `use_ellipticity` is set to True.
    """
    
    def __init__(self, HaloLightConeCatalog, LightconeShell, epsilon_max, model, use_ellipticity = False,
                 mass_def = ccl.halos.massdef.MassDef(200, 'critical'), include_pixel_size = True, verbose = True):

        self.HaloLightConeCatalog = HaloLightConeCatalog
        self.LightconeShell       = LightconeShell
        self.cosmo = HaloLightConeCatalog.cosmology
        self.model = model
        
        
        self.epsilon_max = epsilon_max
        self.mass_def    = mass_def
        self.verbose     = verbose
        
        self.use_ellipticity    = use_ellipticity
        self.include_pixel_size = include_pixel_size

        if use_ellipticity:
            raise NotImplementedError("You have set use_ellipticity = True, but this not yet implemented for HealpixRunner")
    
    
    def build_Rmat(self, A, ref):
        """
        Constructs a 2x2 rotation matrix to rotate vector A to align with the reference vector `ref`.

        This method normalizes both input vectors and computes the rotation angle required to align
        vector A with the reference vector ref. It then constructs and returns the corresponding
        rotation matrix.

        Parameters
        ----------
        A : ndarray
            A 1D array representing the vector to be rotated. It will be normalized within the method.
        
        ref : ndarray
            A 1D array representing the reference vector to which A is to be aligned. It will also be normalized.

        Returns
        -------
        Rmat : ndarray
            A 2x2 rotation matrix that rotates vector A to align with vector ref.
        """

        A   /= np.linalg.norm(A)
        ref /= np.linalg.norm(ref)
    
        ang  = np.arccos(np.dot(A, ref))
        Rmat = np.array([[np.cos(ang), -np.sin(ang)], 
                         [np.sin(ang), np.cos(ang)]])
        
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
    

class BaryonifyShell(DefaultRunner):
    """
    A class to apply baryonification to lightcone shells using a halo catalog.

    The `BaryonifyShell` class inherits from `DefaultRunner` and is designed to process a lightcone shell
    by applying baryonification techniques to adjust the matter distribution. It uses a halo catalog to 
    determine the necessary adjustments based on halo properties, cosmological parameters, and a specified model.

    The input maps should be MASS maps rather than density maps. This is because the method uses
    pix = 0 to identify empty pixels.

    Methods
    -------
    process()
        Processes the lightcone shell by applying baryonification and returns the modified HEALPix map.
    """

    def process(self):
        """
        Applies baryonification to the lightcone shell using the halo catalog.

        This method iterates over each halo in the `HaloLightConeCatalog`, calculating the necessary
        displacements based on the halo's mass, redshift, and position. It uses the given `model` to
        compute the displacement, updates the HEALPix map accordingly, and ensures that the total mass
        remains consistent.

        Returns
        -------
        new_map : ndarray
            A 1D numpy array representing the modified HEALPix map after baryonification.

        Raises
        ------
        AssertionError
            If the sum of the new map values does not match the sum of the original map values, 
            indicating an error in pixel regridding.
        
        Notes
        -----
        - This method assumes flat cosmologies for distance calculations.
        - A `ParamTabulatedProfile` model is required if any property keys (eg. `cdelta`) are used in the model.
        - The method performs a quick check to ensure mass conservation by comparing the sum of the 
          new map with the original map.
        """

        cosmo = ccl.Cosmology(Omega_c = self.cosmo['Omega_m'] - self.cosmo['Omega_b'],
                              Omega_b = self.cosmo['Omega_b'], h   = self.cosmo['h'],
                              sigma8  = self.cosmo['sigma8'],  n_s = self.cosmo['n_s'],
                              w0      = self.cosmo['w0'],
                              matter_power_spectrum = 'linear')
        cosmo.compute_sigma()

        orig_map = self.LightconeShell.map
        NSIDE    = self.LightconeShell.NSIDE

        #Build interpolator between redshift and ang-diam-dist. Assume we never use z > 30
        z_t = np.linspace(0, 30, 1000)
        D_a = interpolate.CubicSpline(z_t, ccl.angular_diameter_distance(cosmo, 1/(1 + z_t)))
        
        keys = vars(self.model).get('p_keys', []) #Check if model has property keys

        if len(keys) > 0:
            txt = (f"You asked to use {keys} properties in Baryonification. You must pass a ParamTabulatedProfile"
                   f"as the model. You have passed {type(self.model)} instead")
            assert isinstance(self.model, ParamTabulatedProfile), txt
        
        pix_offsets = np.zeros([orig_map.size, 3]) 
        
        for j in tqdm(range(self.HaloLightConeCatalog.cat.size), desc = 'Baryonifying matter', disable = not self.verbose):

            M_j = self.HaloLightConeCatalog.cat['M'][j]
            z_j = self.HaloLightConeCatalog.cat['z'][j]
            a_j = 1/(1 + z_j)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) #in physical Mpc
            D_j = D_a(z_j)
            o_j = {key : self.HaloLightConeCatalog.cat[key][j] for key in keys} #Other properties

            #Now just ra and dec
            ra_j   = self.HaloLightConeCatalog.cat['ra'][j]
            dec_j  = self.HaloLightConeCatalog.cat['dec'][j]
            vec_j  = hp.ang2vec(ra_j, dec_j, lonlat = True)
            
            radius = R_j * self.epsilon_max / D_j
            pixind = hp.query_disc(self.LightconeShell.NSIDE, vec_j, radius, inclusive = False, nest = False)
            
            #If there are less than 4 particles, use the 4 nearest particles
            if pixind.size < 4:
                pixind = hp.get_interp_weights(NSIDE, ra_j, dec_j, lonlat = True)[0]
                
            vec    = np.stack(hp.pix2vec(nside = NSIDE, ipix = pixind), axis = 1) #We don't precompute/cache, in order to save memory
            
            pos_j  = vec_j * D_j #We assume flat cosmologies, where D_a is the right distance to use here
            pos    = vec   * D_j #In physical distance, since D_j is physical distance (not comoving)
            diff   = pos - pos_j
            r_sep  = np.sqrt(np.sum(diff**2, axis = 1))
            
            #Compute the displacement needed. Convert input distance from physical --> comoving.
            #Then convert the output from comoving --> physical since "pos" is in physical distance
            offset = self.model.displacement(r_sep/a_j, M_j, a_j, **o_j) * a_j
            offset = offset[:, None] * (diff/r_sep[:, None]) #Add direction
            offset = np.where(np.isfinite(offset), offset, 0) #If offset is weird, set it to 0
            
            #Now convert the 3D offset into a shift in the unit vector of the pixel
            nw_pos = pos + offset #New position
            nw_vec = nw_pos/np.sqrt(np.sum(nw_pos**2, axis = 1))[:, None] #Get unit vector of new position
            offset = nw_vec - vec #Subtract from it the pixel's original unit vector
            
            #Accumulate the offsets in the UNIT VECTORS of the hpixels
            pix_offsets[pixind, :] += offset
        
        new_vec = np.stack( hp.pix2vec(NSIDE, np.arange(orig_map.size)), axis = 1) + pix_offsets
        new_ang = np.stack( hp.vec2ang(new_vec, lonlat = True), axis = 1)
        p_pix   = np.where(orig_map > 0)[0] #Only select regions with positive mass. Zero mass pixels don't matter
        
        c_pix, c_weight = hp.get_interp_weights(NSIDE, new_ang[p_pix, 0], new_ang[p_pix, 1], lonlat = True)
        c_pix, c_weight = c_pix.T, c_weight.T
        
        new_map = np.zeros(orig_map.size, dtype = float)
        new_map = regrid_pixels_hpix(new_map, orig_map[p_pix], c_pix, c_weight)

        #Do a quick check that the sum is the same
        new_sum = np.sum(new_map)
        old_sum = np.sum(orig_map)
        assert np.isclose(new_sum, old_sum), "ERROR in pixel regridding, sum(new_map) [%0.14e] != sum(oldmap) [%0.14e]" % (new_sum, old_sum)
        
        
        return new_map
    

class PaintProfilesShell(DefaultRunner):

    """
    A class to apply profile painting to a lightcone shell using a halo catalog.

    The `PaintProfilesShell` class inherits from `DefaultRunner` and is designed to process a lightcone shell
    by painting profiles to the map based on a given model and halo properties.

    Methods
    -------
    process()
        Processes the lightcone shell by painting baryonic profiles and returns the modified HEALPix map.
    """

    def process(self):
        """
        Applies profile painting to the lightcone shell using the halo catalog.

        This method iterates over each halo in the `HaloLightConeCatalog`, calculating the profile
        contributions based on the halo's mass, redshift, and position. It uses the provided model to
        compute the profile and updates the HEALPix map accordingly.

        Returns
        -------
        new_map : ndarray
            A 1D numpy array representing the modified HEALPix map after painting the baryonic profiles.

        Raises
        ------
        AssertionError
            If a model is not provided or if the provided model is not an instance of `ParamTabulatedProfile`
            when property keys are used.

        Notes
        -----
        - This method assumes flat cosmologies for distance calculations.
        - The `ParamTabulatedProfile` model is required if property keys are used in the model.
        - Non-finite profile values are set to zero before adding profiles to the map.
        """

        cosmo = ccl.Cosmology(Omega_c = self.cosmo['Omega_m'] - self.cosmo['Omega_b'],
                              Omega_b = self.cosmo['Omega_b'], h   = self.cosmo['h'],
                              sigma8  = self.cosmo['sigma8'],  n_s = self.cosmo['n_s'],
                              w0      = self.cosmo['w0'],
                              matter_power_spectrum = 'linear')
        cosmo.compute_sigma()

        orig_map = self.LightconeShell.map
        new_map  = np.zeros_like(orig_map).astype(np.float64)
        NSIDE    = self.LightconeShell.NSIDE
        pixarea  = hp.nside2pixarea(NSIDE, degrees = False)

        #Build interpolator between redshift and ang-diam-dist. Assume we never use z > 30
        z_t = np.linspace(0, 30, 1000)
        D_a = interpolate.CubicSpline(z_t, ccl.angular_diameter_distance(cosmo, 1/(1 + z_t)))
        
        keys = vars(self.model).get('p_keys', []) #Check if model has property keys

        if len(keys) > 0:
            txt = (f"You asked to use {keys} properties in Baryonification. You must pass a ParamTabulatedProfile"
                   f"as the model. You have passed {type(self.model)} instead")
            assert isinstance(self.model, ParamTabulatedProfile), txt


        assert self.model is not None, "You must provide a model"
        Baryons  = self.model

        for j in tqdm(range(self.HaloLightConeCatalog.cat.size), desc = 'Painting Profile', disable = not self.verbose):

            M_j = self.HaloLightConeCatalog.cat['M'][j]
            z_j = self.HaloLightConeCatalog.cat['z'][j]
            a_j = 1/(1 + z_j)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) #in physical Mpc
            D_j = D_a(z_j) #also physical Mpc since Ang. Diam. Dist.
            o_j = {key : self.HaloLightConeCatalog.cat[key][j] for key in keys} #Other properties
            
            ra_j   = self.HaloLightConeCatalog.cat['ra'][j]
            dec_j  = self.HaloLightConeCatalog.cat['dec'][j]
            vec_j  = hp.ang2vec(ra_j, dec_j, lonlat = True)
            
            radius = R_j * self.epsilon_max / D_j
            pixind = hp.query_disc(self.LightconeShell.NSIDE, vec_j, radius, inclusive = False, nest = False)
            vec    = np.stack(hp.pix2vec(nside = NSIDE, ipix = pixind), axis = 1)
            
            pos_j  = vec_j * D_j #We assume flat cosmologies, where D_a is the right distance to use here
            pos    = vec   * D_j
            diff   = pos - pos_j
            r_sep  = np.sqrt(np.sum(diff**2, axis = 1))
            
            #Compute the painted map
            Paint  = Baryons.projected(cosmo, r_sep/a_j, M_j, a_j, **o_j)
            Paint  = np.where(np.isfinite(Paint), Paint, 0) #Set non-finite values to 0
            
            #Add the pixel area back to the maps if requested by user.
            #This factor is needed to get, eg., mass maps when inputting density profiles
            #Factor of D_j helps convert from radian^2 to physical Mpc^2
            if self.include_pixel_size: Paint = Paint * (pixarea * D_j**2)
            
            #Add the profiles to the new healpix map
            new_map[pixind] += Paint         

        return new_map
    


class PaintProfilesAnisShell(DefaultRunner):

    """
    A class to apply profile painting to a lightcone shell using a halo catalog.

    The `PaintProfilesShell` class inherits from `DefaultRunner` and is designed to process a lightcone shell
    by painting profiles to the map based on a given model and halo properties.

    Methods
    -------
    process()
        Processes the lightcone shell by painting baryonic profiles and returns the modified HEALPix map.
    """

    def __init__(self, HaloLightConeCatalog, LightConeShell, epsilon_max, model, Tracer_model, Mtot_model, 
                 background_val, global_tracer_fraction, 
                 mass_def = ccl.halos.massdef.MassDef(200, 'critical'), 
                 include_pixel_size = True, use_ellipticity = False, verbose = True):
        
        self.Tracer_model   = Tracer_model
        self.Mtot_model     = Mtot_model
        self.background_val = background_val
        self.global_tracer_fraction = global_tracer_fraction
        
        super().__init__(HaloLightConeCatalog, LightConeShell, epsilon_max, model, use_ellipticity, mass_def, include_pixel_size, verbose)
        
    def process(self):
        """
        Applies profile painting to the lightcone shell using the halo catalog.

        This method iterates over each halo in the `HaloLightConeCatalog`, calculating the profile
        contributions based on the halo's mass, redshift, and position. It uses the provided model to
        compute the profile and updates the HEALPix map accordingly.

        Returns
        -------
        new_map : ndarray
            A 1D numpy array representing the modified HEALPix map after painting the baryonic profiles.

        Raises
        ------
        AssertionError
            If a model is not provided or if the provided model is not an instance of `ParamTabulatedProfile`
            when property keys are used.

        Notes
        -----
        - This method assumes flat cosmologies for distance calculations.
        - The `ParamTabulatedProfile` model is required if property keys are used in the model.
        - Non-finite profile values are set to zero before adding profiles to the map.
        """

        cosmo = ccl.Cosmology(Omega_c = self.cosmo['Omega_m'] - self.cosmo['Omega_b'],
                              Omega_b = self.cosmo['Omega_b'], h   = self.cosmo['h'],
                              sigma8  = self.cosmo['sigma8'],  n_s = self.cosmo['n_s'],
                              w0      = self.cosmo['w0'],
                              matter_power_spectrum = 'linear')
        cosmo.compute_sigma()

        orig_map = self.LightconeShell.map
        new_map  = np.zeros_like(orig_map).astype(np.float64)
        NSIDE    = self.LightconeShell.NSIDE
        pixarea  = hp.nside2pixarea(NSIDE, degrees = False)

        #Build interpolator between redshift and ang-diam-dist. Assume we never use z > 30
        z_t = np.linspace(0, 30, 1000)
        D_a = interpolate.CubicSpline(z_t, ccl.angular_diameter_distance(cosmo, 1/(1 + z_t)))
        
        keys = vars(self.model).get('p_keys', []) #Check if model has property keys

        if len(keys) > 0:
            txt = (f"You asked to use {keys} properties in Baryonification. You must pass a ParamTabulatedProfile "
                   f"as the model. You have passed {type(self.model)} instead")
            assert isinstance(self.model, ParamTabulatedProfile), txt

        #First we need to generate a model for the total mass distribution, according to the mass model
        Mtot_map = PaintProfilesShell(HaloLightConeCatalog = self.HaloLightConeCatalog, 
                                      LightconeShell = self.LightconeShell, 
                                      epsilon_max = self.epsilon_max, model = self.Mtot_model, 
                                      use_ellipticity = self.use_ellipticity, 
                                      include_pixel_size = True,
                                      mass_def = self.mass_def, verbose = self.verbose).process()
        
        dL = (2 * _get_parameter(self.Mtot_model, 'proj_cutoff')) #Factor of 2 since proj_cutoff == Lproj/2
        dD = D_a(self.LightconeShell.redshift)
        dV = pixarea * ((dD + dL)**3 - dD**3)
        rho_halos = np.sum(Mtot_map) / (dV * Mtot_map.size)

        #Now add the background contribution (we so far only have the halo contribution)
        #Force the background to be positive, incase the pasted density is larger than the box size.
        rho_m     = cosmo.rho_x(1/(self.LightconeShell.redshift + 1), species = 'matter', is_comoving = False)
        drho_m    = np.clip(rho_m - rho_halos, 0, None)
        Mtot_map += dV * drho_m

        if self.verbose:
            print(f"Inputted halos contribute {100*(rho_halos/rho_m):0.2f}% of the total matter density.")
            print(f"Remaining density is assigned to a uniform background.")

        if rho_halos > rho_m:
            warnings.warn("Inputted halos contribute more mass than is available for this mean matter density."
                          "Your Mtot_model profiles are either too extended or you are using the wrong cosmology.")
            
        Paint  = self.model.projected
        Tracer = self.Tracer_model.projected

        for j in tqdm(range(self.HaloLightConeCatalog.cat.size), desc = 'Painting Profile', disable = not self.verbose):

            M_j = self.HaloLightConeCatalog.cat['M'][j]
            z_j = self.HaloLightConeCatalog.cat['z'][j]
            a_j = 1/(1 + z_j)
            R_j = self.mass_def.get_radius(cosmo, M_j, a_j) #in physical Mpc
            D_j = D_a(z_j) #also physical Mpc since Ang. Diam. Dist.
            o_j = {key : self.HaloLightConeCatalog.cat[key][j] for key in keys} #Other properties
            
            ra_j   = self.HaloLightConeCatalog.cat['ra'][j]
            dec_j  = self.HaloLightConeCatalog.cat['dec'][j]
            vec_j  = hp.ang2vec(ra_j, dec_j, lonlat = True)
            
            radius = R_j * self.epsilon_max / D_j
            pixind = hp.query_disc(self.LightconeShell.NSIDE, vec_j, radius, inclusive = False, nest = False)
            vec    = np.stack(hp.pix2vec(nside = NSIDE, ipix = pixind), axis = 1)
            
            pos_j  = vec_j * D_j #We assume flat cosmologies, where D_a is the right distance to use here
            pos    = vec   * D_j
            diff   = pos - pos_j
            r_sep  = np.sqrt(np.sum(diff**2, axis = 1))
            
            #Compute the painted map
            Painting = Paint(cosmo, r_sep/a_j, M_j, a_j, **o_j)
            Painting = np.where(np.isfinite(Painting), Painting, 0)
            Canvas   = Tracer(cosmo, r_sep/a_j, M_j, a_j, **o_j)
            Canvas   = np.where(np.isfinite(Canvas) & np.invert(np.isnan(Canvas)), Canvas, 0)
            Mfrac    = np.divide(Canvas, Mtot_map[pixind], out = np.zeros_like(Canvas), where = Mtot_map[pixind] > 0)
            Mfrac   *= orig_map[pixind]
            
            #Add the pixel area back to the maps if requested by user.
            #This factor is needed to get, eg., mass maps when inputting density profiles
            #Factor of D_j helps convert from radian^2 to physical Mpc^2
            if self.include_pixel_size: Painting = Painting * (pixarea * D_j**2) 

            #Add the profiles to the new healpix map
            new_map[pixind] += Painting * Mfrac   

        #Missing mass was assigned to uniform background. Here we account for that background's contribution
        Mfrac    = np.divide(dV * drho_m, Mtot_map, out = np.zeros_like(Mtot_map), where = Mtot_map > 0)
        Mfrac   *= orig_map
        new_map += self.background_val * self.global_tracer_fraction * Mfrac
        
        new_map  = new_map.reshape(orig_map.shape)      

        return new_map
