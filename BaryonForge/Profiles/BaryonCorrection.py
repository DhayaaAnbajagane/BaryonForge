
import numpy as np
import pyccl as ccl
from tqdm import tqdm
from scipy import interpolate, integrate
import warnings
import copy
from itertools import product

from ..utils.Tabulate import _set_parameter
from ..utils.misc     import destory_Pk

__all__ = ['BaryonificationClass', 'Baryonification3D', 'Baryonification2D']

class BaryonificationClass(object):
    """
    Base class for implementing displacement function models.

    It takes in various input profiles and cosmological parameters to calculate the 
    displacement of particles/cells that is needed to convertfrom one matter distribution to another. 
    The class provides methods to set up interpolation tables for quick calculations of mass displacements.

    Parameters
    ----------
    DMO : object
        An instance of a dark matter-only profile, see `DarkMatterOnly`.
    DMB : object
        An instance of a dark matter-baryon profile, see `DarkMatterBaryon`.
    cosmo : object
        A CCL cosmology instance containing the cosmological parameters used for calculations.
    epsilon_max : float, optional
        The maximum displacement factor for the mass profile, in units of halo radius. Default is 20.
    mass_def : object, optional
        Mass definition object from CCL, default is `MassDef(200, 'critical')`.

    Notes
    -----
    The `BaryonificationClass` generates a displacement function that specifies 
    how baryonic processes alter the distribution of matter.

    **Key Methods and Workflow:**

    1. **`displacement()`**: Main method, which provides the computed displacements at a given radius
       from a given halo, at a given scale factor. It checks if the interpolation table is set up and 
       reads out the displacement using `_readout()`. It verifies that all required parameters are 
       provided and calculates the displacement function based on the inputs.

    2. **`setup_interpolator()`**: This method constructs interpolation tables for the displacement 
       function over a range of redshifts, masses, and radii. It allows for efficient computation 
       of mass displacements by precomputing and storing results. The method iterates over possible 
       values of input parameters, checks for validity, and constructs interpolators using the 
       `PchipInterpolator`.

       .. math::

           d(r, M, a) = f(\\log(1 + z), \\log(M), \\log(r), \\text{other parameters})

       This function ensures that the computed profiles adhere to constraints such as monotonically 
       increasing mass profiles and differences between DMO and DMB profiles asymptotically converging
       to zero on large-enough scales.

    3. **`get_masses()`**: This is an abstract method that must be implemented in subclasses. It is 
       responsible for calculating the mass profiles given a model, radii, mass, scale factor, and 
       mass definition. This method is expected to be overridden to provide specific mass calculations 
       based on the profile models.

    
    4. **`_readout()`**: This helper method reads out the displacement from the precomputed 
       interpolation table. It ensures that displacements are set to zero for radii beyond 
       `epsilon_max` times the halo radius to avoid unphysical values.


    **Normalization and Cutoff Handling:**

    - The `cutoff` value for the DMO and DMB profiles (see `SchneiderProfiles`) is set to 1 Gpc by default, which assumes that the 
      profiles are negligible beyond this scale. This helps to prevent numerical divergences during 
      FFT calculations while ensuring asymptotic behavior at large scales.
    - Only the real-space cutoff is modified here to prevent divergence; the projected cutoff remains as specified by the user. 

    **Warnings:**

    - The class issues warnings if the mass profiles are nearly constant over most of the radius range, 
      which suggests potential numerical issues or negative densities. Users are advised to adjust FFT 
      precision parameters such as `padding_lo_fftlog`, `padding_hi_fftlog`, or `n_per_decade` 
      if such warnings occur.
    """


    def __init__(self, DMO, DMB, cosmo, epsilon_max = 20, mass_def = ccl.halos.massdef.MassDef(200, 'critical'),
                 r_min_int = 1e-6, r_max_int = 1000, N_int = 500):
        
        self.DMO = DMO
        self.DMB = DMB
        
        #Set cutoff to 1 Gpc for calculation, assuming profiles are negligible beyond that
        #Smaller cutoffs result in asymptotic value problems at large scales
        #Larger cutoffs lead to numerical divergence during FFTLogs
        #The user supplied cutoffs will be places when implementing cutoffs in data
        #NOTE: We have not altered the PROJECTED cutoff, only the real cutoff.
        #Projected cutoff must be specified to user input at all times.
        self.DMO.set_parameter('cutoff', 1000)
        self.DMB.set_parameter('cutoff', 1000)
        
        self.cosmo       = cosmo #CCL cosmology instance
        self.epsilon_max = epsilon_max
        self.mass_def    = mass_def


        self.r_min_int   = r_min_int
        self.r_max_int   = r_max_int
        self.N_int       = N_int


    def get_masses(self, model, r, M, a):
        """
        Abstract method for calculating mass profiles.

        This method is intended to be overridden by subclasses to provide specific mass profile 
        calculations. It should return the mass profile for a given model, radii, mass, scale factor, 
        and mass definition.

        Parameters
        ----------
        model : object
            The model instance used to calculate the mass profile (e.g., DMO or DMB).
        r : array_like
            Radii at which to evaluate the mass profile, in comoving Mpc.
        M : array_like
            Halo mass or array of halo masses, in solar masses.
        a : float
            Scale factor, related to redshift by `a = 1 / (1 + z)`.

        Raises
        ------
        NotImplementedError
            This method must be implemented in subclasses of `BaryonificationClass`.
        """

        raise NotImplementedError("Implement a get_masses() method first")


    def setup_interpolator(self, 
                           z_min = 1e-2, z_max = 5, N_samples_z = 30, z_linear_sampling = False, 
                           M_min = 1e12, M_max = 1e16, N_samples_Mass = 30, 
                           R_min = 1e-3, R_max = 1e2, N_samples_R = 100,
                           Rdelta_min = 1e-3, Rdelta_max = 10, Rdelta_sampling = False,
                           other_params = {}, verbose = True):
        
        """
        Sets up interpolation tables for the displacement function.

        This method constructs interpolation tables over a range of redshifts, halo masses, 
        and radii. It precomputes the displacement function values to facilitate efficient 
        calculations of mass displacements due to baryonic processes. User can compute it
        as a function of other inputs to `SchneiderProfiles` using the `other_params`
        function argument.

        Parameters
        ----------
        z_min : float, optional
            Minimum redshift for the interpolation. Default is 1e-2.
        z_max : float, optional
            Maximum redshift for the interpolation. Default is 5.
        N_samples_z : int, optional
            Number of redshift samples during tabulation. Default is 30.
        z_linear_sampling : bool, optional
            If True, use linear sampling for redshift; otherwise, use logarithmic sampling. Default is False.
            Useful if z_min = 0, as log spacing fails in that case.
        M_min : float, optional
            Minimum halo mass for the interpolation, in solar masses. Default is 1e12.
        M_max : float, optional
            Maximum halo mass for the interpolation, in solar masses. Default is 1e16.
        N_samples_Mass : int, optional
            Number of mass samples. Default is 30.
        R_min : float, optional
            Minimum radius for the interpolation, in comoving Mpc. Default is 1e-3.
        R_max : float, optional
            Maximum radius for the interpolation, in comoving Mpc. Default is 1e2.
        N_samples_R : int, optional
            Number of radius samples. Default is 100.
        Rdelta_min : float, optional
            Minimum radius (in units of halo radius) for the interpolation. Default is r/Rdelta = 1e-3.
        Rdelta_max : float, optional
            Maximum radius (in units of halo radius) for the interpolation. Default is r/Rdelta = 10.
        Rdelta_sampling : bool, optional
            If True, build (and sample) from a table where radius is r/Rdelta instead of r. This is important
            for any models that have sharp features (eg. Arico20) as a function of Rdelta. The model is built
            the same, but the interpolation table is organized differently as as to improve accuracy for such models.
        other_params : dict, optional
            Additional parameters for model customization. To be provided in the format `{key : [list-like of vals]}`. 
            The default is an empty dictionary.
        verbose : bool, optional
            If True, display progress information using `tqdm`. Default is True.


        Notes
        -----
        - Ensures that mass profiles are monotonic and valid across the specified parameter ranges.
        - Issues warnings if profiles are nearly constant over radius or if the interpolation 
        fails for specific halo masses. Warnings tend to be raised for the lowest masses,
        especially when using pixel window convolutions.
        """

        if z_min <= 0: 
            assert z_linear_sampling, f"Geometric series not possible for {z_min} < z < {z_max}. Set z_linear_sampling = True, or z_min > 0"

        M_range  = np.geomspace(M_min, M_max, N_samples_Mass)
        r        = np.geomspace(R_min, R_max, N_samples_R)
        z_range  = np.linspace(z_min, z_max, N_samples_z) if z_linear_sampling else np.geomspace(z_min, z_max, N_samples_z)
        a_range  = 1/(1 + z_range)
        p_keys   = list(other_params.keys()); setattr(self, 'p_keys', p_keys)
        d_interp = np.zeros([z_range.size, M_range.size, r.size] + [other_params[k].size for k in p_keys])

        if Rdelta_sampling: rdelta_range = np.geomspace(Rdelta_min, Rdelta_max, N_samples_R)
        
        #If other_params is empty then iterator will be empty and the code still works fine
        iterator = [p for p in product(*[np.arange(other_params[k].size) for k in p_keys])]
        
        with tqdm(total = d_interp.size//(M_range.size*r.size), desc = 'Building Table', disable = not verbose) as pbar:
            for j in range(z_range.size):
                
                for c in iterator:
                    
                    #Modify the model input params so that they are run with the right parameters
                    for k_i in range(len(p_keys)):
                        _set_parameter(self.DMO, p_keys[k_i], other_params[p_keys[k_i]][c[k_i]])
                        _set_parameter(self.DMB, p_keys[k_i], other_params[p_keys[k_i]][c[k_i]])
                    
                    M_DMO = self.get_masses(self.DMO, r, M_range, a_range[j])
                    M_DMB = self.get_masses(self.DMB, r, M_range, a_range[j])
                    
                    for i in range(M_range.size):
                        ln_DMB    = np.log(M_DMB[i])
                        ln_DMO    = np.log(M_DMO[i])
                        

                        #Require mass to always increase w/ radius, using iterative conditions
                        #And remove pts of DMO = DMB, improves large-scale convergence (while making
                        #sure it does not fail just because DMO is NaN values, which happens in some profs)
                        #And also require at least 1e-6 difference in DMB else the interpolator breaks :/
                        #We will handle DMO mask later separately
                        
                        min_diff  = -np.inf
                        diff_mask = np.ones_like(ln_DMB).astype(bool)
                        iterate   = 0
                        while (min_diff < 1e-5) & (diff_mask.sum() > 5):
                            
                            new_mask  = ( (np.diff(ln_DMB[diff_mask], prepend = 0) > 1e-5) & 
                                          ((np.abs(ln_DMB - ln_DMO)[diff_mask] > 1e-6) | np.isnan(ln_DMO)[diff_mask]) & 
                                          np.isfinite(ln_DMB)[diff_mask]
                                        )
                            
                            diff_mask[diff_mask] = new_mask
                            diff_mask[0] = True
                            
                            iterate += 1
                            
                            if iterate > 30:
                                diff_mask  = np.zeros_like(diff_mask).astype(bool) #Set everything to False and skip the building step next
                                warn_text  = (f"Mass profile of log10(M) = {np.log10(M_range[i])} is nearly constant over radius. " 
                                              "Suggests density is negative or zero for most of the range. If using convolutions,"
                                              "consider changing the fft precision params in the CCL profile:"
                                              "padding_lo_fftlog, padding_hi_fftlog, or n_per_decade. Otherwise, consider changing"
                                              "the grid spacing to be finer using the N_int parameter")
                                warnings.warn(warn_text, UserWarning)
                                break
                                
                            if diff_mask.sum() < 5: 
                                warn_text  = (f"Mass profile of log10(M) = {np.log10(M_range[i])} is nearly constant over radius. " 
                                              "Or it is broken. Less than 5 datapoints are usable.")
                                warnings.warn(warn_text, UserWarning)
                                break
                            
                            min_diff  = np.min(np.diff(ln_DMB[diff_mask], prepend = 0)[1:])                                                       
                            
                        #If we have enough usable mass values, then proceed as usual
                        #This generally breaks for very small halos, where projection
                        #can be catastrophicall broken (eg. only negative densities)
                        if diff_mask.sum() > 5:
                                
                            #Same mask as DMB but for DMO. No need for iterations, since
                            #the x-axis in interpolator is still radius, so the requirements are more lax.
                            fini_mask  = ( (np.diff(ln_DMO, prepend = 0) > 1e-5) & 
                                           ((np.abs(ln_DMB - ln_DMO) > 1e-6) | np.isnan(ln_DMB))& 
                                           np.isfinite(ln_DMO)
                                        )
                            interp_DMB = interpolate.PchipInterpolator(ln_DMB[diff_mask], np.log(r)[diff_mask], extrapolate = False)
                            interp_DMO = interpolate.PchipInterpolator(np.log(r)[fini_mask], ln_DMO[fini_mask], extrapolate = False)

                            offset = np.exp(interp_DMB(interp_DMO(np.log(r)))) - r
                            offset = np.where(np.isfinite(offset), offset, 0)

                            if Rdelta_sampling: 
                                Rdelta  = self.mass_def.get_radius(self.cosmo, M_range[i], a_range[j]) / a_range[j]
                                offset = np.interp(rdelta_range, r/Rdelta, offset)

                        #If broken, then these halos contribute nothing to the displacement function.
                        #Just provide a warning saying this is happening
                        else:
                            offset = np.zeros_like(r)
                            warn_text = (f"Displacement function for halo with log10(M) = {np.log10(M_range[i])} failed to compute." 
                                         "Defaulting to d = 0. If using convolutions, consider changing the fft precision "
                                         "params in the CCL profile: padding_lo_fftlog, padding_hi_fftlog, or n_per_decade")
                            warnings.warn(warn_text, UserWarning)
                        
                        #Build a custom index into the array
                        index = tuple([j, i, slice(None)] + list(c))
                        d_interp[index] = offset
                            
                    pbar.update(1)


        input_rad  = np.log(r) if not Rdelta_sampling else np.log(rdelta_range)
        input_grid = tuple([np.log(1 + z_range), np.log(M_range), input_rad] + [other_params[k] for k in p_keys])

        self.raw_input_d = d_interp
        self.raw_input_z_range = np.log(1 + z_range)
        self.raw_input_M_range = np.log(M_range)
        self.raw_input_r_range = input_rad
        for k in other_params.keys(): setattr(self, 'raw_input_%s_range' % k, other_params[k]) #Save other raw inputs too
            
        self.interp_d = interpolate.RegularGridInterpolator(input_grid, d_interp, bounds_error = False, fill_value = np.nan)    
        self.Rdelta_sampling = Rdelta_sampling #Set so we know what to do during readout

        #Once all tabulation is done, we don't need to keep P(k) calculations in cosmology object.
        #This is good because the Pk class is not pickleable, so by destorying it here we
        #are able to keep this class pickleable.
        self.cosmo = destory_Pk(self.cosmo)

    
    def _readout(self, r, M, a, **kwargs):

        """
        Read out the displacement from the interpolation table.

        This method retrieves the displacement function values from the precomputed 
        interpolation table for given radii, halo masses, and scale factor. It sets 
        displacement values to zero beyond `r > R * epsilon_max`, where `R` is the
        halo radius.

        Parameters
        ----------
        r : array_like
            Radii at which to evaluate the displacement, in comoving Mpc.
        M : array_like
            Halo mass or array of halo masses, in solar masses.
        a : float
            Scale factor, related to redshift by `a = 1 / (1 + z)`.
        **kwargs : dict
            Additional parameters required by the interpolation table.

        Returns
        -------
        displ : ndarray
            Displacement values corresponding to the input radii and masses.

        Notes
        -----
        - Ensures that displacements are set to zero for radii beyond `epsilon_max` 
        times the halo radius to avoid unphysical values.
        """
        
        table = self.interp_d #The interpolation table to use
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        a_use = np.atleast_1d(a)
        z_use = 1/a_use - 1
        
        displ = np.zeros([M_use.size, r_use.size])
        empty = np.ones_like(r_use)
        z_in  = np.log(1/a)*empty #This is log(1 + z)
        r_in  = np.log(r_use)
        k_in  = [kwargs[k] * empty for k in kwargs.keys()]

        #Get the ranges we used as input, so we can check if requested
        #ranges are contained within the input/tabulated ranges.
        #We saved log(1 + z) so converting back to z here...
        z_tab = np.exp(self.raw_input_z_range) - 1
        M_tab = np.exp(self.raw_input_M_range)
        r_tab = np.exp(self.raw_input_r_range)

        if (np.min(z_use) < np.min(z_tab)) | (np.max(z_use) > np.max(z_tab)):
            warn_text = f"Requested redshift range [{np.min(z_use)}, {np.max(z_use)}] outside table's range [{np.min(z_tab)}, {np.max(z_tab)}]"            
            warnings.warn(warn_text, UserWarning)
        
        if (np.min(M_use) < np.min(M_tab)) | (np.max(M_use) > np.max(M_tab)):
            warn_text = (f"Requested log_Mass range [{np.log10(np.min(M_use))}, {np.log10(np.max(M_use))}] outside "
                         f"table's range [{np.log10(np.min(M_tab))}, {np.log10(np.max(M_tab))}]")          
            warnings.warn(warn_text, UserWarning)

        if not self.Rdelta_sampling:
            if (np.min(r_use) < np.min(r_tab)) | (np.max(r_use) > np.max(r_tab)):
                warn_text = f"Requested Radius range [{np.min(r_use)}, {np.max(r_use)}] outside table's range [{np.min(r_tab)}, {np.max(r_tab)}]"            
                warnings.warn(warn_text, UserWarning)

        
        for i in range(M_use.size):
            M_in  = np.log(M_use[i])*empty
            R     = self.mass_def.get_radius(self.cosmo, M_use[i], a)/a #in comoving Mpc
            
            #If Rdelta sampling, the sample in r/Rdelta not r.
            #The table would have been constructed appropriately
            if not self.Rdelta_sampling:
                p_in     = tuple([z_in, M_in, r_in] + k_in)
                displ[i] = table(p_in)
            else:
                p_in     = tuple([z_in, M_in, r_in - np.log(R)] + k_in)
                displ[i] = table(p_in)
            
            inside   = (r < self.epsilon_max*R)
            displ[i] = np.where(inside, displ, 0) #Set large-scale displacements to 0
            
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            displ = np.squeeze(displ, axis=-1)
        if np.ndim(M) == 0:
            displ = np.squeeze(displ, axis=0)
            
        return displ

    
    def displacement(self, r, M, a, **kwargs):
        """
        Return the displacement needed to convert the matter distribution
        from the dmo profiles to the dmb profiles, for the given radii, halo 
        masses, and scale factor. It never does a calculation on-the-fly and
        instead reads out a precomputed table.

        Parameters
        ----------
        r : array_like
            Radii at which to evaluate the displacement, in comoving Mpc.
        M : array_like
            Halo mass or array of halo masses, in solar masses.
        a : float
            Scale factor, related to redshift by `a = 1 / (1 + z)`.
        **kwargs : dict
            Additional parameters required by the interpolation table.

        Returns
        -------
        displ : ndarray
            Displacement values corresponding to the input radii and masses.
            In comoving Mpc.

        Raises
        ------
        NameError
            If the interpolation table has not been set up using `setup_interpolator()`.
        AssertionError
            If required parameters are not provided in `kwargs`.
        """
        
        if not hasattr(self, 'interp_d'):
            raise NameError("No Table created. Run setup_interpolator() method first")
            
        for k in self.p_keys:
            assert k in kwargs.keys(), "Need to provide %s as input into `displacement'. Table was built with this." % k
        
        return self._readout(r, M, a, **kwargs)



class Baryonification3D(BaryonificationClass):
    """
    Class implementing a 3D baryonification model. It extends 
    `BaryonificationClass` to provide the displacement function 
    in three dimensions.

    Inherits from
    -------------
    BaryonificationClass : Base class for baryonification models.

    Notes
    -----
    The `Baryonification3D` class is used to compute the 3D enclosed mass profiles.
    It integrates the density profile to obtain the cumulative 
    enclosed mass as a function of radius:

    .. math::

        M_{\\text{enc}}(r) = 4\\pi \\int_0^r \\rho(r') r'^2 \\, d\\ln r'

    the displacement function is then,

    .. math::

        \Delta d (r) = M_{\rm dmb}^{-1}(M_{\rm dmo}(r)) - r,

    
    Methods
    -------
    displacement(r, M, a, **kwargs)
        Compute the displacement function for a given mass, radii, and scale factor
    get_masses(model, r, M, a)
        Computes the enclosed mass profile for a given model, radii, halo mass, and scale factor.
    """

    def get_masses(self, model, r, M, a):
        """
        Computes the enclosed mass profile for a given model.

        This method calculates the cumulative enclosed mass profile for a specified 
        model (e.g., dark matter-only or dark matter-baryon), radii, halo mass, and 
        scale factor.

        Parameters
        ----------
        model : object
            The model instance used to calculate the mass profile (e.g., DMO or DMB).
        r : array_like
            Radii at which to evaluate the mass profile, in comoving Mpc.
        M : float or array_like
            Halo mass or array of halo masses, in solar masses.
        a : float
            Scale factor, related to redshift by `a = 1 / (1 + z)`.

        Returns
        -------
        M_f : ndarray
            Enclosed mass profile corresponding to the input radii and halo mass. 
            The output shape is (len(M), len(r)), unless M is a scalar, in which 
            case the shape is (len(r),). Units of solar masses.

        Notes
        -----
        - The method adjusts the integration range to ensure that the minimum and 
        maximum radii do not disrupt the integral, adding a 20% buffer.
        - Negative densities are set to zero to avoid unphysical results.
        - The enclosed mass \( M_{\\text{enc}}(r) \) is calculated using:

        .. math::

            M_{\\text{enc}}(r) = 4\\pi \\int_0^r \\rho(r') r'^2 \\, d\\ln r'

        - `PchipInterpolator` is used to interpolate the mass profile, ensuring smoothness 
        and avoiding artifacts due to Fourier space ringing.
        - If `M` is a scalar, the output is squeezed to remove extra dimensions.

        Examples
        --------
        Compute the enclosed mass profile for a given model:

        >>> baryon_model = Baryonification3D(DMO=dark_matter_profile, DMB=dmb_profile, cosmo=my_cosmology)
        >>> r = np.logspace(-2, 1, 50)  # Radii in comoving Mpc
        >>> M = 1e14  # Halo mass in solar masses
        >>> a = 0.8  # Scale factor corresponding to redshift z
        >>> mass_profile = baryon_model.get_masses(baryon_model.DMO, r, M, a)
        """
        
        #Make sure the min/max does not mess up the integral
        #Adding some 20% buffer just in case
        r_min = np.min([np.min(r), self.r_min_int])
        r_max = np.max([np.max(r), self.r_max_int])
        r_int = np.geomspace(r_min/1.2, r_max*1.2, self.N_int)
        
        dlnr  = np.log(r_int[1]/r_int[0])
        rho   = model.real(self.cosmo, r_int, M, a)
        rho   = np.where(rho < 0, 0, rho) #Enforce non-zero densities
        
        if isinstance(M, (float, int) ): rho = rho[None, :]
            
        intgd = 4*np.pi*r_int**3 * rho * dlnr
        M_enc = integrate.cumulative_simpson(intgd, axis = -1, initial = 0) + intgd[:, [0]]
        lnr   = np.log(r)
        
        M_f   = np.zeros([M_enc.shape[0], r.size])
        
        #Remove datapoints in profile where rho == 0 and then just interpolate
        #across them. This helps deal with ringing profiles due to 
        #fourier space issues, where profile could go negative sometimes
        for M_i in range(M_enc.shape[0]):
            Mask     = (rho[M_i] > 0) & (np.isfinite(M_enc[M_i])) #Keep only finite points, and ones with increasing density
            M_f[M_i] = np.exp( interpolate.PchipInterpolator(np.log(r_int)[Mask], np.log(M_enc[M_i])[Mask], extrapolate = False)(lnr) )
        
        if isinstance(M, (float, int) ): M_f = np.squeeze(M_f, axis = 0)
            
        return M_f


class Baryonification2D(BaryonificationClass):

    """
    Class implementing a 2D baryonification model. It extends 
    `BaryonificationClass` to provide the displacement function 
    in two dimensions.

    Inherits from
    -------------
    BaryonificationClass : Base class for baryonification models.

    Notes
    -----
    The `Baryonification2D` class is used to compute the 2D mass profiles. 
    It integrates the projected surface density profile, \( \Sigma(r) \), to obtain 
    the cumulative enclosed mass as a function of radius. 

    The enclosed mass \( M_{\\text{enc}}(r) \) is calculated by integrating the projected 
    surface density profile \( \Sigma(r) \) over circular annuli:

    .. math::

        M_{\\text{enc}}(r) = 2\\pi \\int_0^r \Sigma(r') r' \, d\\ln r'

    the displacement function is then

    .. math::

        \Delta d (r_{\\rm p}) = M_{\\rm DMB, p}^{-1}(M_{\\rm DMO, p}(r_{\\rm p})) - r_{\\rm p},

    where :math:`r_{\\rm p}` is the projected radius.

    """

    def get_masses(self, model, r, M, a):
        """
        Computes the enclosed mass profile for a given model using 2D projection.

        This method calculates the cumulative enclosed mass profile for a specified 
        model (e.g., dark matter-only or dark matter-baryon), radii, halo mass, and 
        scale factor.

        Parameters
        ----------
        model : object
            The model instance used to calculate the mass profile (e.g., DMO or DMB).
        r : array_like
            Radii at which to evaluate the mass profile, in comoving Mpc.
        M : float or array_like
            Halo mass or array of halo masses, in solar masses.
        a : float
            Scale factor, related to redshift by `a = 1 / (1 + z)`.

        Returns
        -------
        M_f : ndarray
            Enclosed mass profile corresponding to the input radii and halo mass. 
            The output shape is (len(M), len(r)), unless M is a scalar, in which 
            case the shape is (len(r),). In units of solar masses.

        Notes
        -----
        - The method adjusts the integration range to ensure that the minimum and 
        maximum radii do not disrupt the integral, adding a 20% buffer.
        - Negative surface densities are set to zero to avoid unphysical results.
        - The enclosed mass \( M_{\\text{enc}}(r) \) is calculated using:

        .. math::

            M_{\\text{enc}}(r) = 2\\pi \\int_0^r \\Sigma(r') r' \\, d\\ln r'

        - `PchipInterpolator` is used to interpolate the mass profile, ensuring smoothness 
        and avoiding artifacts due to Fourier space ringing.
        - If `M` is a scalar, the output is squeezed to remove extra dimensions.

        Examples
        --------
        Compute the enclosed mass profile for a given model using 2D projection:

        >>> baryon_model = Baryonification2D(DMO=dark_matter_profile, DMB=dmb_profile, cosmo=my_cosmology)
        >>> r = np.logspace(-2, 1, 50)  # Radii in comoving Mpc
        >>> M = 1e14  # Halo mass in solar masses
        >>> a = 0.5  # Scale factor corresponding to redshift z
        >>> mass_profile = baryon_model.get_masses(baryon_model.DMO, r, M, a)
        """
        
        #Make sure the min/max does not mess up the integral
        #Adding some 20% buffer just in case
        r_min = np.min([np.min(r), self.r_min_int])
        r_max = np.max([np.max(r), self.r_max_int])
        r_int = np.geomspace(r_min/1.2, r_max*1.2, self.N_int)
        
        #The scale fac. is used in Sigma cause the projection in ccl is
        #done in comoving coords not physical coords
        dlnr  = np.log(r_int[1]/r_int[0])
        Sigma = model.projected(self.cosmo, r_int, M, a) * a 
        Sigma = np.where(Sigma < 0, 0, Sigma) #Enforce non-zero densities
        
        if isinstance(M, (float, int) ): Sigma = Sigma[None, :]
        
        intgd = 2*np.pi*r_int**2 * Sigma * dlnr
        M_enc = integrate.cumulative_simpson(intgd, axis = -1, initial = 0) + intgd[:, [0]]
        lnr   = np.log(r)
        
        M_f  = np.zeros([M_enc.shape[0], r.size])
        #Remove datapoints in profile where Sigma == 0 and then just interpolate
        #across them. This helps deal with ringing profiles due to 
        #fourier space issues, where profile could go negative sometimes
        for M_i in range(M_enc.shape[0]):
            Mask     = (Sigma[M_i] > 0) & (np.isfinite(M_enc[M_i])) #Keep only finite points, and ones with increasing density
            M_f[M_i] = np.exp( interpolate.PchipInterpolator(np.log(r_int)[Mask], np.log(M_enc[M_i])[Mask], extrapolate = False)(lnr) )
        
        if isinstance(M, (float, int) ): M_f = np.squeeze(M_f, axis = 0)
            
        return M_f