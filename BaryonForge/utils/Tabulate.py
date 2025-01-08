
import numpy as np
import pyccl as ccl
from tqdm import tqdm
from itertools import product
from scipy import interpolate

from ..utils.misc import destory_Pk

__all__ = ['_set_parameter', 'TabulatedProfile', 'ParamTabulatedProfile']

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

    obj_keys = dir(obj)
    
    for k in obj_keys:
        if k == key: 
            setattr(obj, key, value)
        elif isinstance(getattr(obj, k), ccl.halos.profiles.HaloProfile):
            _set_parameter(getattr(obj, k), key, value)

            
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
    
    mass_def : object, optional
        A `ccl.halos.massdef.MassDef` object that defines the mass definition. Default is `MassDef(200, 'critical')`.

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

    def __init__(self, model, cosmo, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        self.model    = model
        self.cosmo    = cosmo #CCL cosmology instance
        self.mass_def = mass_def

        #Get all the other params. Particularly those
        #needed for projecting profiles
        super().__init__(mass_def = mass_def)


    def setup_interpolator(self, z_min = 1e-2, z_max = 5, N_samples_z = 30, z_linear_sampling = False, 
                           M_min = 1e12, M_max = 1e16, N_samples_Mass = 30, 
                           R_min = 1e-3, R_max = 1e2,  N_samples_R = 100, 
                           other_params = {}, verbose = True):
        
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

        """

        M_range  = np.geomspace(M_min, M_max, N_samples_Mass)
        r        = np.geomspace(R_min, R_max, N_samples_R)
        z_range  = np.linspace(z_min, z_max, N_samples_z) if z_linear_sampling else np.geomspace(z_min, z_max, N_samples_z)
        dlnr     = np.log(r[1]) - np.log(r[0])

        interp3D = np.zeros([z_range.size, M_range.size, r.size])
        interp2D = np.zeros([z_range.size, M_range.size, r.size])
        
        with tqdm(total = z_range.size, desc = 'Building Table', disable = not verbose) as pbar:
            for j in range(z_range.size):                
                a_j = 1/(1 + z_range[j])

                #Extra factor of "a" accounts for projection in ccl being done in comoving, not physical units
                interp3D[j, :, :] = self.model.real(self.cosmo, r, M_range, a_j)
                interp2D[j, :, :] = self.model.projected(self.cosmo, r, M_range, a_j) * a_j
                pbar.update(1)

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
        a_use = np.atleast_1d(a)
        z_use = 1/a_use - 1
        
        prof  = np.zeros([M_use.size, r_use.size])
        empty = np.ones_like(r_use)
        z_in  = np.log(1/a)*empty #This is log(1 + z)
        r_in  = np.log(r_use)
        
        for i in range(M_use.size):
            M_in  = np.log(M_use[i])*empty

            prof[i] = table((z_in, M_in, r_in, ))
            prof[i] = np.exp(prof[i])
            
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
            
        return prof
            
        
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
    
    mass_def : object, optional
        A `ccl.halos.massdef.MassDef` object that defines the mass definition. Default is `MassDef(200, 'critical')`.

    Attributes
    ----------
    model : object
        The profile model used for generating tabulated profiles.
    
    cosmo : object
        The cosmology instance used for the profile calculations.
    
    mass_def : object
        The mass definition used for the profile calculations.
    
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

    
    def __init__(self, model, cosmo, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        """
        Initializes the ParamTabulatedProfile class with a given model, cosmology, and mass definition.

        Parameters
        ----------
        model : object
            A profile model object that defines the real and projected halo profiles. This object should have `real()`
            and `projected()` methods for evaluating profiles.
        
        cosmo : object
            A `ccl.Cosmology` object representing the cosmological parameters.
        
        mass_def : object, optional
            A `ccl.halos.massdef.MassDef` object that defines the mass definition. Default is `MassDef(200, 'critical')`.
        """

        self.model    = model
        self.cosmo    = cosmo #CCL cosmology instance
        self.mass_def = mass_def
        
        assert not isinstance(model, TabulatedProfile), "Input model cannot be 'TabulatedProfile' object."

        
    def setup_interpolator(self, z_min = 1e-2, z_max = 5, N_samples_z = 30, z_linear_sampling = False, 
                           M_min = 1e12, M_max = 1e16, N_samples_Mass = 30, 
                           R_min = 1e-3, R_max = 1e2,  N_samples_R = 100, 
                           other_params = {}, verbose = True):
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
            arrays of parameter values. Default is an empty dictionary.
        
        verbose : bool, optional
            If `True`, display a progress bar during the tabulation process. Default is `True`.

        """

        M_range  = np.geomspace(M_min, M_max, N_samples_Mass)
        r        = np.geomspace(R_min, R_max, N_samples_R)
        z_range  = np.linspace(z_min, z_max, N_samples_z) if z_linear_sampling else np.geomspace(z_min, z_max, N_samples_z)
        dlnr     = np.log(r[1]) - np.log(r[0])

        p_keys   = list(other_params.keys()); setattr(self, 'p_keys', p_keys)
        interp3D = np.zeros([z_range.size, M_range.size, r.size] + [other_params[k].size for k in p_keys]) + np.nan
        interp2D = np.zeros([z_range.size, M_range.size, r.size] + [other_params[k].size for k in p_keys]) + np.nan

        #If other_params is empty then iterator will be empty and the code still works fine
        iterator = [p for p in product(*[np.arange(other_params[k].size) for k in p_keys])]
        
        #Loop over params to build table
        with tqdm(total = interp3D.size//(M_range.size*r.size), desc = 'Building Table', disable = not verbose) as pbar:
            for j in range(z_range.size):                
                a_j = 1/(1 + z_range[j])
                
                for c in iterator:
                    
                    #Modify the model input params so that they are run with the right parameters
                    for k_i in range(len(p_keys)):
                        _set_parameter(self.model, p_keys[k_i], other_params[p_keys[k_i]][c[k_i]])
                    
                    #Build a custom index into the array
                    index = tuple([j, slice(None), slice(None)] + list(c))
                    
                    #Extra factor of "a" accounts for projection in ccl being done in comoving, not physical units
                    interp3D[index] = self.model.real(self.cosmo, r, M_range, a_j)
                    interp2D[index] = self.model.projected(self.cosmo, r, M_range, a_j) * a_j
                    pbar.update(1)
                    

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
        a_use = np.atleast_1d(a)
        z_use = 1/a_use - 1
        
        prof  = np.zeros([M_use.size, r_use.size])
        empty = np.ones_like(r_use)
        z_in  = np.log(1/a)*empty #This is log(1 + z)
        r_in  = np.log(r_use)
        k_in  = [kwargs[k] * empty for k in kwargs.keys()]
        
        for i in range(M_use.size):
            M_in  = np.log(M_use[i])*empty
            p_in  = tuple([z_in, M_in, r_in] + k_in)
            prof[i] = table(p_in)
            prof[i] = np.exp(prof[i])
            
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
            
        return prof
    
            
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
        dlnr = np.log(r[1]) - np.log(r[0])
        z_range  = np.linspace(z_min, z_max, N_samples_z)
        
        interp3D = np.zeros([z_range.size, r.size]) + np.NaN
        
        #Loop over params to build table
        with tqdm(total = z_range.size, desc = 'Building Table', disable = not verbose) as pbar:
            for j in range(z_range.size):
                
                a = 1/(1 + z_range[j])
                interp3D[j, :] = ccl.correlation_3d(self.cosmo, a, r)
                
                pbar.update(1)
        
        input_grid_1 = (np.log(1 + z_range), np.log(r))

        self.raw_input_3D = interp3D
        self.raw_input_z_range = np.log(1 + z_range)
        self.raw_input_r_range = np.log(r)
        
        self.interp3D = interpolate.RegularGridInterpolator(input_grid_1, np.log(interp3D), bounds_error = False)
        
        
    def __call__(self, r, a):
        
        r_use = np.atleast_1d(r)
        a_use = np.atleast_1d(a)
        z_use = 1/a_use - 1
        
        empty = np.ones_like(r_use)
        z_in  = np.log(1/a)*empty #This is log(1 + z)
        r_in  = np.log(r_use)
        
        ln_xi = self.interp3D( (z_in, r_in) )
        xi    = np.exp(ln_xi)
        
        return xi
        
        