import numpy as np
import pyccl as ccl
from operator import add, mul, sub, truediv, pow, neg, pos, abs
import warnings
from ..utils.Tabulate import _set_parameter

__all__ = ['BaseBFGProfiles', 'hyper_params']


hyper_params = ['mass_def', 'c_M_relation', 'use_fftlog_projection', 
                'padding_hi_proj', 'padding_hi_proj', 'n_per_decade_proj',
                'r_min_int', 'r_max_int', 'r_steps', 'xi_mm']

#The order matters, we search for the first, then the second etc.
#If a parameter matches two keys, it will only be assigned using the 1st match
DEFAULTS     = {'mu_' : 0, 'nu_' : 0, 'zeta_' : 0, 'M_' : 1e14}

class BaseBFGProfiles(ccl.halos.profiles.HaloProfile):
    """
    Base class for defining halo density profiles for any BaryonForge model.

    This class extends the `ccl.halos.profiles.HaloProfile` class and provides 
    additional functionality for handling different halo density profiles. It allows 
    for custom real-space projection methods, control over parameter initialization, 
    and adjustments to the Fourier transform settings to minimize artifacts.

    Parameters
    ----------
    use_fftlog_projection : bool, optional
        If True, the default FFTLog projection method is used for the `projected` method. 
        If False, a custom real-space projection is employed. Default is False.
    padding_lo_proj : float, optional
        The lower padding factor for the projection integral in real-space. Default is 0.1.
    padding_hi_proj : float, optional
        The upper padding factor for the projection integral in real-space. Default is 10.
    n_per_decade_proj : int, optional
        Number of integration points per decade in the real-space projection integral. Default is 10.
    xi_mm : callable, optional
        A function that returns the matter-matter correlation function at different radii.
        Default is None, in which case we use the CCL inbuilt model.
    **kwargs
        Additional keyword arguments for setting specific parameters of the profile.
    
    Attributes
    ----------
    model_params : dict
        A dictionary containing all model parameters and their values.
    precision_fftlog : dict
        Dictionary with precision settings for the FFTLog convolution. Can be modified 
        directly or using the update_precision_fftlog() method.

    Methods
    -------
    real(cosmo, r, M, a)
        Computes the real-space density profile.
    projected(cosmo, r, M, a)
        Computes the projected density profile.

    """

    #Define the params used in this model
    model_param_names = []
    hyper_param_names = []
    defaults_params   = DEFAULTS

    def __init__(self, mass_def = ccl.halos.massdef.MassDef200c, 
                 c_M_relation = None, use_fftlog_projection = False, 
                 padding_lo_proj = 0.1, padding_hi_proj = 10, n_per_decade_proj = 10, 
                 r_min_int = 1e-6, r_max_int = 1e3, r_steps = 500,
                 xi_mm = None, 
                 **kwargs):
        
        #Go through all input params, and assign Nones to ones that don't exist.
        #If mass/redshift/conc-dependence, then set to 1 if don't exist
        for m in self.model_param_names:

            if m in kwargs.keys():
                setattr(self, m, kwargs[m])
            
            elif any([k in m for k in self.defaults_params.keys()]):

                for k in self.defaults_params.keys():
                    if k in m:
                        setattr(self, m, self.defaults_params[k])
                        break
            
            else:
                setattr(self, m, None)

        #Let user specify their own c_M_relation as desired
        if c_M_relation is not None:
            self.c_M_relation = c_M_relation(mass_def = mass_def)
        else:
            self.c_M_relation = None

        #Also save the original input to propogate into profile ops.
        self._c_M_relation    = c_M_relation
                    
        #Some params for handling the realspace projection
        self.padding_lo_proj   = padding_lo_proj
        self.padding_hi_proj   = padding_hi_proj
        self.n_per_decade_proj = n_per_decade_proj 

        #Some params that control numerical integration
        self.r_min_int = r_min_int
        self.r_max_int = r_max_int
        self.r_steps   = r_steps
        
        #Import all other parameters from the base CCL Profile class
        ccl.halos.profiles.HaloProfile.__init__(self,mass_def = mass_def)

        #Function that returns correlation func at different radii
        self.xi_mm = xi_mm

        #Sets the cutoff scale of all profiles, in comoving Mpc. Prevents divergence in FFTLog
        #Also set cutoff of projection integral. Should be the box side length
        self.cutoff      = kwargs['cutoff'] if 'cutoff' in kwargs.keys() else 1e3 #1Gpc is a safe default choice
        self.proj_cutoff = kwargs['proj_cutoff'] if 'proj_cutoff' in kwargs.keys() else self.cutoff
        
        
        #This allows user to force usage of the default FFTlog projection, if needed.
        #Otherwise, we use the realspace integration, since that allows for specification
        #of a hard boundary on radius
        if not use_fftlog_projection:
            self._projected = self._projected_realspace
        else:
            text = ("You must set the same cutoff for 3D profile and projection profile if you want to use fftlog projection. "
                    f"You have cutoff = {self.cutoff} and proj_cutoff = {self.proj_cutoff}")
            assert self.cutoff == self.proj_cutoff, text

        #Save to propogate into profile-operated classes (+, -, %, *, etc.)
        self._use_fftlog_projection = use_fftlog_projection

        #Constant that helps with the fourier transform convolution integral.
        #This value minimized the ringing due to the transforms
        self.update_precision_fftlog(plaw_fourier = -2)

        #Need this to prevent projected profile from artificially cutting off
        self.update_precision_fftlog(padding_lo_fftlog = 1e-2, padding_hi_fftlog = 1e2,
                                     padding_lo_extra  = 1e-4, padding_hi_extra  = 1e4)
        
    
    @property
    def model_params(self):
        """
        Returns a dictionary containing all model parameters and their current values.

        Returns
        -------
        params : dict
            Dictionary of model parameters.
        """
        
        params = {k:v for k,v in vars(self).items() if k in self.model_param_names}
                  
        return params
    

    def update_precision_fftlog(self, **pars):
        """
        Updates the FFT parameters for the fourier method, and does so
        recursively for any and all BaryonForge (BFG) profiles that are
        held as attributes within a given class.
        """
        
        Haloprofile = ccl.halos.profiles.HaloProfile
        #Set precision for yourself
        Haloprofile.update_precision_fftlog(self, **pars)

        #Now check if you have any attributes that also need
        #to have their precision updated
        obj_keys = dir(self)
    
        for k in obj_keys:
            if isinstance(getattr(self, k), (ccl.halos.profiles.HaloProfile,)):
                BaseBFGProfiles.update_precision_fftlog(getattr(self, k), **pars)
                      

    @property
    def hyper_params(self):
        """
        Returns a dictionary containing all hyper parameters oof the calculation and their current values.

        Returns
        -------
        params : dict
            Dictionary of hyper parameters.
        """
        
        params = {k:v for k,v in vars(self).items() if k in self.hyper_param_names}
        params['c_M_relation']          = self._c_M_relation #Swap this one specifically
        params['use_fftlog_projection'] = self._use_fftlog_projection #This one isn't saved normally so do it here

        return params
    
        
    def _projected_realspace(self, cosmo, r, M, a):
        """
        Computes the projected profile using a custom real-space integration method. 
        Advantageous as it can avoid any hankel transform artifacts.

        Parameters
        ----------
        cosmo : object
            CCL cosmology object.
        r : array_like
            Radii at which to evaluate the profile.
        M : array_like
            Halo mass or array of halo masses.
        a : float
            Scale factor, related to redshift by `a = 1 / (1 + z)`.

        Returns
        -------
        proj_prof : ndarray
            Projected profile evaluated at the specified radii and masses.
        """

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        #Integral limits
        int_min = self.padding_lo_proj   * np.min(r_use)
        int_max = self.padding_hi_proj   * np.max(r_use)
        int_N   = self.n_per_decade_proj * np.int32(np.log10(int_max/int_min))
        
        #If proj_cutoff was passed, then use the largest of the two
        if self.proj_cutoff is not None: 
            int_max = np.max([self.proj_cutoff, int_max])

        r_integral  = np.geomspace(int_min, int_max, int_N)
        
        
        #Use proj_cutoff and if it is not passed then default to the regular cutoff
        if self.proj_cutoff is not None:
            r_max = self.proj_cutoff
        elif self.cutoff is not None:
            r_max = self.cutoff
        else:
            r_max = 1e4
            warnings.warn("WARNING: projected() profile requested without specifying proj_cutoff or cutoff. "
                          "Defaulting the integral upper limit to 10,000 (comoving) Mpc.")
            
        r_proj = np.geomspace(int_min, r_max, int_N)
        prof   = self._real(cosmo, r_integral, M, a)

        #The prof object is already "squeezed" in some way.
        #Code below removes that squeezing so rest of code can handle
        #passing multiple radii and masses.
        if np.ndim(r) == 0: prof = prof[:, None]
        if np.ndim(M) == 0: prof = prof[None, :]

        proj_prof = np.zeros([M_use.size, r_use.size])

        #This nested loop saves on memory, and vectorizing the calculation doesn't really
        #speed things up, so better to keep the loop this way.
        for i in range(M_use.size):
            for j in range(r_use.size):

                proj_prof[i, j] = 2*np.trapz(np.interp(np.sqrt(r_proj**2 + r_use[j]**2), r_integral, prof[i]), r_proj)

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            proj_prof = np.squeeze(proj_prof, axis=-1)
        if np.ndim(M) == 0:
            proj_prof = np.squeeze(proj_prof, axis=0)

        if np.any(proj_prof <= 0):
            warnings.warn("WARNING: Profile is zero/negative in some places."
                          "Likely a convolution artifact for objects smaller than the pixel scale")

        return proj_prof
    
    
    def __str_par__(self):
        '''
        String with all input params and their values
        '''
        
        string = f"("
        for m in self.model_param_names:
            string += f"{m} = {self.__dict__[m]}, "
        string = string[:-2] + ')'
        return string
        
    
    def __str_prf__(self):
        '''
        String with the class/profile name
        '''
        
        string = f"{self.__class__.__name__}"
        return string
        
    
    def __str__(self):
        
        string = self.__str_prf__() + self.__str_par__()
        return string 
    
    
    def __repr__(self):
        
        return self.__str__()
    
    
    #Add routines for consistently changing input params across all profiles
    def set_parameter(self, key, value): 
        """
        Sets a parameter value for the profile. It can do it recursively in
        case the profile contains other profiles as its attributes.

        Parameters
        ----------
        key : str
            Name of the parameter to set.
        value : any
            New value for the parameter.
        """
        _set_parameter(self, key, value)
    
    
    #Add routines for doing simple arithmetic operations with the classes
    from ..utils.misc import generate_operator_method
    
    __add__      = generate_operator_method(add)
    __mul__      = generate_operator_method(mul)
    __sub__      = generate_operator_method(sub)
    __truediv__  = generate_operator_method(truediv)
    __pow__      = generate_operator_method(pow)
    
    __radd__     = generate_operator_method(add, reflect = True)
    __rmul__     = generate_operator_method(mul, reflect = True)
    __rsub__     = generate_operator_method(sub, reflect = True)
    __rtruediv__ = generate_operator_method(truediv, reflect = True)
    
    __abs__      = generate_operator_method(abs)
    __pos__      = generate_operator_method(pos)
    __neg__      = generate_operator_method(neg)