import numpy as np
import pyccl as ccl
from operator import add, mul, sub, truediv, pow, neg, pos, abs
import warnings

from scipy import interpolate, integrate
from ..utils.Tabulate import _set_parameter

__all__ = ['model_params', 'SchneiderProfiles', 
           'DarkMatter', 'TwoHalo', 'Stars', 'Gas', 'ShockedGas', 'CollisionlessMatter',
           'DarkMatterOnly', 'DarkMatterBaryon']


model_params = ['cdelta', 'epsilon', 'a', 'n', #DM profle params
                'q', 'p', #Relaxation params
                'cutoff', 'proj_cutoff', #Cutoff parameters (numerical)
                
                'theta_ej', 'theta_co', 'M_c', 'gamma', 'delta', #Default gas profile param
                'mu_theta_ej', 'mu_theta_co', 'mu_beta', 'mu_gamma', 'mu_delta', #Mass dep
                'M_theta_ej',  'M_theta_co', 'M_gamma', 'M_delta', #Mass dep norm
                'nu_theta_ej', 'nu_theta_co', 'nu_M_c',  'nu_gamma', 'nu_delta', #Redshift  dep
                'zeta_theta_ej', 'zeta_theta_co', 'zeta_M_c', 'zeta_gamma', 'zeta_delta', #Concentration dep
                
                'A', 'M1', 'eta', 'eta_delta', 'tau', 'tau_delta', 'epsilon_h', #Star params
                
                'alpha_nt', 'nu_nt', 'gamma_nt', 'mean_molecular_weight' #Non-thermal pressure and gas density
               ]

class SchneiderProfiles(ccl.halos.profiles.HaloProfile):
    """
    Base class for defining halo density profiles based on Schneider et al. models.

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
        Additional keyword arguments for setting specific parameters of the profile. If a parameter 
        is not specified, defaults are assigned based on its type (e.g., mass/redshift/conc-dependence).

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
    model_param_names = model_params

    def __init__(self, mass_def = ccl.halos.massdef.MassDef200c, 
                 use_fftlog_projection = False, 
                 padding_lo_proj = 0.1, padding_hi_proj = 10, n_per_decade_proj = 10, 
                 r_min_int = 1e-6, r_max_int = 1e3, r_steps = 500,
                 xi_mm = None, 
                 **kwargs):
        
        #Go through all input params, and assign Nones to ones that don't exist.
        #If mass/redshift/conc-dependence, then set to 1 if don't exist
        for m in self.model_param_names:
            if m in kwargs.keys():
                setattr(self, m, kwargs[m])
            elif ('mu_' in m) or ('nu_' in m) or ('zeta_' in m): #Set mass/red/conc dependence
                setattr(self, m, 0)
            elif ('M_' in m): #Set mass normalization
                setattr(self, m, 1e14)
            else:
                setattr(self, m, None)
                    
        #Some params for handling the realspace projection
        self.padding_lo_proj   = padding_lo_proj
        self.padding_hi_proj   = padding_hi_proj
        self.n_per_decade_proj = n_per_decade_proj 

        #Some params that control numerical integration
        self.r_min_int = r_min_int
        self.r_max_int = r_max_int
        self.r_steps   = r_steps
        
        #Import all other parameters from the base CCL Profile class
        super().__init__(mass_def = mass_def)

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
        
        
    
    def _get_gas_params(self, M, z):
        """
        Computes gas-related parameters based on the mass and redshift.
        Will use concentration is cdelta is specified during Class initialization.
        Uses mass/redshift slopes provided during class initialization.

        Parameters
        ----------
        M : array_like
            Halo mass or array of halo masses.
        z : float
            Redshift.

        Returns
        -------
        beta : ndarray
            Small-scale gas slope.
        theta_ej : ndarray
            Ejection radius.
        theta_co : ndarray
            Core radius parameter.
        delta : ndarray
            Large-scale slope.
        gamma : ndarray
            Intermediate-scale slope.
        """
        
        cdelta   = 1 if self.cdelta is None else self.cdelta
        
        M_c      = self.M_c * (1 + z)**self.nu_M_c * cdelta**self.zeta_M_c
        beta     = 3*(M/M_c)**self.mu_beta / (1 + (M/M_c)**self.mu_beta)
        
        #Use M_c as the mass-normalization for simplicity sake
        theta_ej = self.theta_ej * (M/self.M_theta_ej)**self.mu_theta_ej * (1 + z)**self.nu_theta_ej * cdelta**self.zeta_theta_ej
        theta_co = self.theta_co * (M/self.M_theta_co)**self.mu_theta_co * (1 + z)**self.nu_theta_co * cdelta**self.zeta_theta_co
        delta    = self.delta    * (M/self.M_delta)**self.mu_delta       * (1 + z)**self.nu_delta    * cdelta**self.zeta_delta
        gamma    = self.gamma    * (M/self.M_gamma)**self.mu_gamma       * (1 + z)**self.nu_gamma    * cdelta**self.zeta_gamma
        
        beta     = beta[:, None]
        theta_ej = theta_ej[:, None]
        theta_co = theta_co[:, None]
        delta    = delta[:, None]
        gamma    = gamma[:, None]
        
        return beta, theta_ej, theta_co, delta, gamma
        
        
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
        
        #If proj_cutoff was passed, then rewrite the integral max limit
        if self.proj_cutoff is not None:
            int_max = self.proj_cutoff

        r_integral = np.geomspace(int_min, int_max, int_N)

        prof = self._real(cosmo, r_integral, M, a)

        #The prof object is already "squeezed" in some way.
        #Code below removes that squeezing so rest of code can handle
        #passing multiple radii and masses.
        if np.ndim(r) == 0:
            prof = prof[:, None]
        if np.ndim(M) == 0:
            prof = prof[None, :]

        proj_prof = np.zeros([M_use.size, r_use.size])

        #This nested loop saves on memory, and vectorizing the calculation doesn't really
        #speed things up, so better to keep the loop this way.
        for i in range(M_use.size):
            for j in range(r_use.size):

                proj_prof[i, j] = 2*np.trapz(np.interp(np.sqrt(r_integral**2 + r_use[j]**2), r_integral, prof[i]), r_integral)

        
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



class DarkMatter(SchneiderProfiles):
    """
    Class representing the total Dark Matter (DM) profile using the NFW (Navarro-Frenk-White) profile.

    This class is derived from the `SchneiderProfiles` class and provides an implementation of the 
    dark matter profile based on the NFW model. It includes a custom `_real` method for calculating 
    the real-space dark matter density profile, considering factors like the concentration-mass 
    relation and truncation radius.

    See `SchneiderProfiles` for more docstring details.

    Notes
    -----
    The `DarkMatter` class calculates the dark matter density profile using the NFW model with a 
    modification for truncation at a specified radius set by `epsilon`. This profile accounts for the concentration-mass 
    relation, which can be provided as `cdelta` during class init. If none is provided,
    we use the `ConcentrationDiemer15` model.

    The profile also includes an additional exponential cutoff to prevent numerical overflow and 
    artifacts at large radii.

    The dark matter density profile is given by:

    .. math::

        \\rho_{\\text{DM}}(r) = \\frac{\\rho_c}{\\frac{r}{r_s} \\left(1 + \\frac{r}{r_s}\\right)^2} 
        \\cdot \\frac{1}{\\left(1 + \\frac{r}{r_t}\\right)^2}

    where:

    - :math:`\\rho_c` is the characteristic density of the halo.
    - :math:`r_s` is the scale radius of the halo, defined as :math:`r_s = R/c`.
    - :math:`r_t = \\epsilon \\cdot R` is the truncation radius, controlled by the parameter `epsilon`.
    - :math:`r` is the radial distance.


    Examples
    --------
    Create a `DarkMatter` profile and compute the density at specific radii:

    >>> dm_profile = DarkMatter(**parameters)
    >>> cosmo = ...  # Define or load a cosmology object
    >>> r = np.logspace(-2, 1, 50)  # Radii in comoving Mpc
    >>> M = 1e14  # Halo mass in solar masses
    >>> a = 0.5  # Scale factor corresponding to redshift z
    >>> density_profile = dm_profile.real(cosmo, r, M, a)
    """

    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        if self.cdelta is None:
            c_M_relation = ccl.halos.concentration.ConcentrationDiemer15(mass_def = self.mass_def) #Use the diemer calibration
            
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mass_def = self.mass_def)
            #c_M_relation = ccl.halos.concentration.ConcentrationConstant(7, mdef = self.mass_def) #needed to get Schneider result
            
        c   = c_M_relation(cosmo, M_use, a)
        R   = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        r_s = R/c
        r_t = R*self.epsilon
        
        r_s, r_t = r_s[:, None], r_t[:, None]

        
        #Get the normalization (rho_c) numerically
        #The analytic integral doesn't work since we have a truncation radii now.
        #We loop over every halo, instead of vectorizing, since the integral limits
        #now depend on the halo radius. 
        Normalization = np.zeros_like(M_use)
        for m_i in range(M_use.size):
            r_integral     = np.geomspace(self.r_min_int, R[m_i], self.r_steps)
            prof_integral  = 1/(r_integral/r_s[m_i] * (1 + r_integral/r_s[m_i])**2) * 1/(1 + (r_integral/r_t[m_i])**2)**2
            Normalization[m_i] = np.trapz(4*np.pi*r_integral**2 * prof_integral, r_integral)
        
        rho_c = M_use/Normalization
        rho_c = rho_c[:, None]

        arg  = (r_use[None, :] - self.cutoff)
        arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof = rho_c/(r_use/r_s * (1 + r_use/r_s)**2) * 1/(1 + (r_use/r_t)**2)**2 * kfac
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)


        return prof


class TwoHalo(SchneiderProfiles):
    """
    Class representing the two-halo term profile.

    This class is derived from the `SchneiderProfiles` class and provides an implementation 
    of the two-halo term profile. It utilizes the 2-point correlation function directly, rather 
    than employing the full halo model. 

    See `SchneiderProfiles` for more docstring details.

    Notes
    -----
    The `TwoHalo` class calculates the two-halo term profile using the linear matter power spectrum 
    to ensure the correct large-scale clustering behavior. The profile is defined using the matter-matter 
    correlation function, :math:`\\xi_{\\text{mm}}(r)`, and a mass-dependent bias term.

    The two-halo term density profile is given by:

    .. math::

        \\rho_{\\text{2h}}(r) = \\left(1 + b(M) \\cdot \\xi_{\\text{mm}}(r)\\right) \\cdot \\rho_{\\text{m}}(a) \\cdot \\text{kfac}

    where:

    - :math:`b(M)` is the linear halo bias, defined as:

      .. math::

          b(M) = 1 + \\frac{q \\nu_M^2 - 1}{\\delta_c} + \\frac{2p}{\\delta_c \\left(1 + (q \\nu_M^2)^p\\right)}

    - :math:`\\nu_M` is the peak height parameter, :math:`\\nu_M = \\delta_c / \\sigma(M)`.
    - :math:`\\delta_c` is the critical density for spherical collapse.
    - :math:`\\xi_{\\text{mm}}(r)` is the matter-matter correlation function.
    - :math:`\\rho_{\\text{m}}(a)` is the mean matter density at scale factor `a`.
    - :math:`\\text{kfac}` is an additional exponential cutoff factor to prevent numerical overflow.

    See `Sheth & Tormen 1999 <https://arxiv.org/pdf/astro-ph/9901122>`_ for more details on the bias prescription.

    The two-halo term is only valid when the cosmology object's matter power spectrum is set 
    to 'linear'. An assertion check is included to ensure this.

    Examples
    --------
    Create a `TwoHalo` profile and compute the density at specific radii:

    >>> two_halo_profile = TwoHalo(**parameters)
    >>> cosmo = ...  # Define or load a cosmology object with linear matter power spectrum
    >>> r = np.logspace(-2, 1, 50)  # Radii in comoving Mpc
    >>> M = 1e14  # Halo mass in solar masses
    >>> a = 0.5  # Scale factor corresponding to redshift z
    >>> density_profile = two_halo_profile.real(cosmo, r, M, a)
    """

    def _real(self, cosmo, r, M, a):

        #Need it to be linear if we're doing two halo term
        assert cosmo._config_init_kwargs['matter_power_spectrum'] == 'linear', "Must use matter_power_spectrum = linear for 2-halo term"

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        R   = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        z = 1/a - 1

        if self.xi_mm is None:
            xi_mm   = ccl.correlation_3d(cosmo, r = r_use, a = a)
        else:
            xi_mm   = self.xi_mm(r_use, a)

        delta_c = 1.686/ccl.growth_factor(cosmo, a)
        nu_M    = delta_c / ccl.sigmaM(cosmo, M_use, a)
        bias_M  = 1 + (self.q*nu_M**2 - 1)/delta_c + 2*self.p/delta_c/(1 + (self.q*nu_M**2)**self.p)

        bias_M  = bias_M[:, None]
        prof    = (1 + bias_M * xi_mm)*ccl.rho_x(cosmo, a, species = 'matter', is_comoving = True)

        #Need this truncation so the fourier space integral isnt infinity
        arg  = (r_use[None, :] - self.cutoff)
        arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof = prof * kfac

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof


class Stars(SchneiderProfiles):
    """
    Class representing the exponential stellar mass profile.

    This class is derived from the `SchneiderProfiles` class and provides an implementation 
    of an exponential stellar mass profile. It calculates the real-space stellar mass 
    density profile, using parameters to account for factors like stellar mass fraction 
    and halo radius.

    See `SchneiderProfiles` for more docstring details.

    Notes
    -----
    The `Stars` class models the stellar mass distribution with an exponential profile, 
    modulated by parameters such as `eta`, `tau`, `A`, and `M1`. These parameters 
    adjust the stellar mass fraction as a function of halo mass. The profile also applies 
    an exponential cutoff controlled by the `epsilon_h` parameter to define the 
    characteristic radius of the stellar distribution.

    The stellar mass density profile is given by:

    .. math::

        \\rho_\\star(r) = \\frac{f_{\\text{cga}} M_{\\text{tot}}}{4 \\pi^{3/2} R_h} \\frac{1}{r^2} 
                          \\exp\\left(-\\frac{r^2}{4 R_h^2}\\right) 

    where:

    - :math:`f_{\\text{cga}}` is the stellar mass fraction, defined as:

      .. math::

          f_{\\text{cga}} = 2 A \\left(\\left(\\frac{M}{M_1}\\right)^{\\tau + \\tau_\\delta} 
          + \\left(\\frac{M}{M_1}\\right)^{\\eta + \\eta_\\delta}\\right)^{-1}

    - :math:`M_{\\text{tot}}` is the total halo mass.
    - :math:`R_h = \\epsilon_h R` is the characteristic scale radius of the stellar distribution.
    - :math:`r` is the radial distance.

    The class overrides specific `precision_fftlog` settings to prevent ringing artifacts 
    in the profiles. This is achieved by setting extreme padding values.

    An additional exponential cutoff is included to prevent numerical overflow and artifacts 
    at large radii.

    Examples
    --------
    Create a `Stars` profile and compute the density at specific radii:

    >>> stars_profile = Stars(**parameters)
    >>> cosmo = ...  # Define or load a cosmology object
    >>> r = np.logspace(-2, 1, 50)  # Radii in comoving Mpc
    >>> M = 1e14  # Halo mass in solar masses
    >>> a = 0.5  # Scale factor corresponding to redshift z
    >>> density_profile = stars_profile.real(cosmo, r, M, a)
    """
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        #For some reason, we need to make this extreme in order
        #to prevent ringing in the profiles. Haven't figured out
        #why this is the case
        self.update_precision_fftlog(padding_lo_fftlog = 1e-5, padding_hi_fftlog = 1e5)

    
    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R   = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        eta_cga = self.eta + self.eta_delta
        tau_cga = self.tau + self.tau_delta
        
        f_cga  = 2 * self.A * ((M_use/self.M1)**tau_cga  + (M_use/self.M1)**eta_cga)**-1

        R_h   = self.epsilon_h * R

        f_cga, R_h = f_cga[:, None], R_h[:, None]

        r_integral = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)
        DM    = DarkMatter(**self.model_params); setattr(DM, 'cutoff', 1e3) #Set large cutoff just for normalization calculation
        rho   = DM.real(cosmo, r_integral, M_use, a)
        M_tot = np.trapz(4*np.pi*r_integral**2 * rho, r_integral, axis = -1)
        M_tot = np.atleast_1d(M_tot)[:, None]
        
        arg  = (r_use[None, :] - self.cutoff)
        arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof = f_cga*M_tot / (4*np.pi**(3/2)*R_h) * 1/r_use**2 * np.exp(-(r_use/2/R_h)**2) * kfac
                
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof


class Gas(SchneiderProfiles):

    """
    Class representing the gas density profile.

    This class is derived from the `SchneiderProfiles` class and provides an implementation 
    of a gas density profile. It calculates the real-space gas density profile, using
    the general NFW (GNFW) model of `Nagai, Kravtsov & Vikhlinin 2009 <https://arxiv.org/pdf/astro-ph/0703661>`_.

    See `SchneiderProfiles` for more docstring details.

    Notes
    -----
    The `Gas` class models the gas distribution in halos by considering the gas fraction, 
    which is computed based on the total baryonic fraction minus the stellar fraction. 
    The gas density profile is defined using parameters such as `beta`, `delta`, `gamma`, 
    `theta_co`, and `theta_ej`. These parameters characterize the core and ejection properties 
    of the gas distribution.

    The gas density profile is given by:

    .. math::

        \\rho_{\\text{gas}}(r) = \\frac{f_{\\text{gas}} M_{\\text{tot}}}{N} \\cdot 
        \\frac{1}{(1 + u)^{\\beta}} \\cdot \\frac{1}{(1 + v)^{(\\delta - \\beta)/\\gamma}}

    where:

    - :math:`f_{\\text{gas}} = f_{\\text{bar}} - f_{\\star}` is the gas fraction.
    - :math:`f_{\\text{bar}}` is the cosmic baryon fraction.
    - :math:`f_{\\star}` is the stellar mass fraction, defined as:

      .. math::

          f_{\\star} = 2A \\left(\\left(\\frac{M}{M_1}\\right)^{\\tau} + \\left(\\frac{M}{M_1}\\right)^{\\eta}\\right)^{-1}

    - :math:`M_{\\text{tot}}` is the total halo mass.
    - :math:`N` is the normalization factor to ensure mass conservation.
    - :math:`u = \\frac{r}{R_{\\text{co}}}` and :math:`v = \\frac{r}{R_{\\text{ej}}}` are dimensionless radii.
    - :math:`\\beta` is the power-law slope for :math:`R_{\\text{co}} \lesssim r \lesssim R_{\\text{ej}}`
    - :math:`\\delta` is the power-law slope at :math:`r \sim \lesssim R_{\\text{ej}}`
    - :math:`\\gamma` is the power-law slope for :math:`r \gg R_{\\text{ej}}`
    - :math:`R_{\\text{co}} = \\theta_{\\text{co}} R` is the core radius.
    - :math:`R_{\\text{ej}} = \\theta_{\\text{ej}} R` is the ejection radius.
    - :math:`r` is the radial distance.

    Examples
    --------
    Create a `Gas` profile and compute the density at specific radii:

    >>> gas_profile = Gas(**parameters)
    >>> cosmo = ...  # Define or load a cosmology object
    >>> r = np.logspace(-2, 1, 50)  # Radii in comoving Mpc
    >>> M = 1e14  # Halo mass in solar masses
    >>> a = 0.5  # Scale factor corresponding to redshift z
    >>> density_profile = gas_profile.real(cosmo, r, M, a)
    """

    def _real(self, cosmo, r, M, a):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_star = 2 * self.A * ((M_use/self.M1)**self.tau + (M_use/self.M1)**self.eta)**-1
        f_bar  = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        f_gas  = f_bar - f_star
        f_gas  = f_gas[:, None]
        
        #Get gas params
        beta, theta_ej, theta_co, delta, gamma = self._get_gas_params(M_use, z)
        R_co = theta_co*R[:, None]
        R_ej = theta_ej*R[:, None]
        
        u = r_use/R_co
        v = r_use/R_ej
        
        
        #Integrate over wider region in radii to get normalization of gas profile
        r_integral = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)
        u_integral = r_integral/R_co
        v_integral = r_integral/R_ej
        

        prof_integral = 1/(1 + u_integral)**beta / (1 + v_integral**gamma)**( (delta - beta)/gamma )
        Normalization = np.trapz(4 * np.pi * r_integral**2 * prof_integral, r_integral, axis = -1)[:, None]

        del u_integral, v_integral, prof_integral

        DM    = DarkMatter(**self.model_params); setattr(DM, 'cutoff', 1e3) #Set large cutoff just for normalization calculation
        rho   = DM.real(cosmo, r_integral, M_use, a)
        M_tot = np.trapz(4*np.pi*r_integral**2 * rho, r_integral, axis = -1)
        M_tot = np.atleast_1d(M_tot)[:, None]
        
        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = 1/(1 + u)**beta / (1 + v**gamma)**( (delta - beta)/gamma ) * kfac
        prof *= f_gas*M_tot/Normalization
        

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof
    
    
class ShockedGas(Gas):
    """
    Class representing a shocked gas profile.

    This class is derived from the `Gas` class and provides an implementation 
    of a shocked gas profile, assuming Rankine-Hugoniot conditions. It models the 
    effect of a high Mach-number shock, leading to a density suppression by a factor of 4. 
    This suppression is implemented using a logistic function based on the radial distance.

    Parameters
    ----------
    epsilon_shock : float
        A scaling factor that sets the shock radius as a fraction of the halo radius.
    width_shock : float
        The width of the shock transition, controlling how sharply the gas density changes 
        across the shock front.
    **kwargs
        Additional keyword arguments passed to the `Gas` class.

    Notes
    -----
    The `ShockedGas` class modifies the gas density profile inherited from the `Gas` class 
    by applying a shock model. The shock is characterized by the `epsilon_shock` and `width_shock` 
    parameters. The density is reduced by a factor of 4 at the shock front, a result that 
    assumes a high Mach-number shock under Rankine-Hugoniot conditions.

    The gas density profile is calculated as:

    .. math::

        \\rho_{\\text{shocked}}(r) = \\rho_{\\text{gas}}(r) \\cdot 
        \\left[ \\frac{1 - 0.25}{1 + \\exp\\left(\\frac{\\log(r) - \\log(\\epsilon_{\\text{shock}} R)}{\\text{width}_{\\text{shock}}}\\right)} + 0.25 \\right]

    where:

    - :math:`\\rho_{\\text{gas}}(r)` is the gas density profile from the `Gas` class.
    - :math:`\\epsilon_{\\text{shock}}` sets the location of the shock.
    - :math:`\\text{width}_{\\text{shock}}` determines the sharpness of the transition.
    - :math:`r` is the radial distance from the halo center.
    - The factor of 0.25 represents the maximum possible density drop due to the shock.

    See the `Gas` class for more details on the base gas profile and additional parameters.
    """
    
    def __init__(self, epsilon_shock, width_shock, **kwargs):
        
        self.epsilon_shock = epsilon_shock
        self.width_shock   = width_shock
        
        super().__init__(**kwargs)

    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        #Minimum is 0.25 since a factor of 4x drop is the maximum possible for a shock
        rho_gas = super()._real(cosmo, r, M, a)
        g_arg   = 1/self.width_shock*(np.log(r_use) - np.log(self.epsilon_shock*R)[:, None])
        g_arg   = np.where(g_arg > 1e2, np.inf, g_arg) #To prevent overflows when doing exp
        factor  = (1 - 0.25)/(1 + np.exp(g_arg)) + 0.25
        
        #Get the right size for rho_gas
        if M_use.size == 1: rho_gas = rho_gas[None, :]
            
        prof = rho_gas * factor
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        
        return prof


class CollisionlessMatter(SchneiderProfiles):

    """
    Class representing the collisionless matter density profile.

    This class is derived from the `SchneiderProfiles` class and provides an implementation 
    for the collisionless matter density profile. It combines contributions from gas, stars, 
    and dark matter to compute the total density profile, using an iterative method to solve
    for the collisionless matter (dark matter and galaxies) after adiabatic relaxation.

    Parameters
    ----------
    gas : Gas, optional
        An instance of the `Gas` class defining the gas profile. If not provided, a default 
        `Gas` object is created using `kwargs`.
    stars : Stars, optional
        An instance of the `Stars` class defining the stellar profile. If not provided, a default 
        `Stars` object is created using `kwargs`.
    darkmatter : DarkMatter, optional
        An instance of the `DarkMatter` class defining the dark matter profile. If not provided, 
        a default `DarkMatter` object is created using `kwargs`.
    max_iter : int, optional
        Maximum number of iterations for the relaxation method. Default is 10.
    reltol : float, optional
        Relative tolerance for convergence in the relaxation method. Default is 1e-2.
    r_min_int : float, optional
        Minimum radius for integration during the iterative relaxation. Default is 1e-8.
    r_max_int : float, optional
        Maximum radius for integration during the iterative relaxation. Default is 1e5.
    r_steps : int, optional
        Number of steps in the radius for integration. Default is 5000.
    **kwargs
        Additional keyword arguments passed to initialize the `Gas`, `Stars`, and `DarkMatter` 
        profiles, as well as other parameters from `SchneiderProfiles`.

    
    Notes
    -----
    The `CollisionlessMatter` class computes the total density profile by combining the 
    contributions from gas, stars, and dark matter profiles. The relaxation method iteratively 
    adjusts these profiles to achieve equilibrium, ensuring mass conservation. This approach 
    accounts for different physical components and their interactions within the halo.

    **Calculation Steps:**

    1. **Initial Profiles**: The class starts by calculating the individual density profiles for dark matter, gas, and stars:

       .. math::

           \\rho_{\\text{DM}}(r), \\; \\rho_{\\text{gas}}(r), \\; \\rho_{\\text{stars}}(r)

    2. **Cumulative Mass Profiles**: The cumulative mass profiles are calculated by integrating the density profiles:

       .. math::

           M_{\\text{DM}}(r) = 4\\pi \\int_0^r \\rho_{\\text{DM}}(r') r'^2 dr'

       .. math::

           M_{\\text{gas}}(r) = 4\\pi \\int_0^r \\rho_{\\text{gas}}(r') r'^2 dr'

       .. math::

           M_{\\text{stars}}(r) = 4\\pi \\int_0^r \\rho_{\\text{stars}}(r') r'^2 dr'

    3. **Relaxation Iteration**: The relaxation method iteratively adjusts the mass profile to achieve equilibrium. The adjusted mass profile is calculated as:

       .. math::

           M_{\\text{CLM}}(r) = f_{\\text{clm}}(M_{\\text{DM}} + M_{\\text{gas}} + M_{\\text{stars}})

       where :math:`f_{\\text{clm}}` is calculated using:

       .. math::

           f_{\\text{clm}} = 1 - \\frac{\\Omega_b}{\\Omega_m} + f_{\\text{sga}}

       Here, :math:`f_{\\text{sga}}` is the satellite galaxy mass fraction

    4. **Relaxation Factor Update**: During each iteration, the relaxation factor \( \zeta \) is updated using:

       .. math::

           \\zeta_{\\text{new}} = a \\left( \\left(\\frac{M_{\\text{DM}}}{M_{\\text{CLM}}}\\right)^n - 1 \\right) + 1

       This equation ensures that the mass distribution relaxes towards equilibrium over successive iterations, where \( a \) and \( n \) are parameters controlling the relaxation process.

    5. **Density Profile Calculation**: The final collisionless matter density profile is derived from the adjusted cumulative mass:

       .. math::

           \\rho_{\\text{CLM}}(r) = \\frac{1}{4\\pi r^2} \\frac{d}{dr} M_{\\text{CLM}}(r)


    **Integration Range and Convergence**: The relaxation method uses a logarithmic integration range defined by `r_min_int`, `r_max_int`, and `r_steps`. The method iterates until the relative difference falls below `reltol` or the maximum number of iterations (`max_iter`) is reached.

    See `SchneiderProfiles` and associated classes (`Gas`, `Stars`, `DarkMatter`) for more details on 
    the underlying profiles and parameters.

    Warnings
    --------
    The method checks if the provided radius values fall within the integration limits. Warnings are 
    issued if adjustments to `r_min_int` or `r_max_int` are recommended to cover the full range of 
    the input radii. Note that sometimes warnings occur because the FFTlog asks for a ridiculously
    high/low radius, and the profile calculation will just return 0s there. In this case the
    warning is benign and can be safely ignored
    """
    
    def __init__(self, gas = None, stars = None, darkmatter = None, max_iter = 10, reltol = 1e-2, r_min_int = 1e-8, r_max_int = 1e5, r_steps = 5000, **kwargs):
        
        self.Gas   = gas
        self.Stars = stars
        self.DarkMatter = darkmatter
        
        if self.Gas is None: self.Gas = Gas(**kwargs)          
        if self.Stars is None: self.Stars = Stars(**kwargs)
        if self.DarkMatter is None: self.DarkMatter = DarkMatter(**kwargs)
            
        #Stop any artificially cutoffs when doing the relaxation.
        #The profile will be cutoff at the very last step instead
        self.Gas.set_parameter('cutoff', 1000)
        self.Stars.set_parameter('cutoff', 1000)
        self.DarkMatter.set_parameter('cutoff', 1000)
            
        self.max_iter   = max_iter
        self.reltol     = reltol

        self.r_min_int  = r_min_int
        self.r_max_int  = r_max_int
        self.r_steps    = r_steps
        
        super().__init__(**kwargs, r_min_int = r_min_int, r_max_int = r_max_int, r_steps = r_steps)
        

    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        if np.min(r) < self.r_min_int: 
            warnings.warn(f"Decrease integral lower limit, r_min_int ({self.r_min_int}) < minimum radius ({np.min(r)})", UserWarning)
        if np.max(r) > self.r_max_int: 
            warnings.warn(f"Increase integral upper limit, r_max_int ({self.r_max_int}) < maximum radius ({np.max(r)})", UserWarning)

        #Def radius sampling for doing iteration.
        #And don't check iteration near the boundaries, since we can have numerical errors
        #due to the finite width oof the profile during iteration.
        #Radius boundary is very large, I found that worked best without throwing edgecases
        #especially when doing FFTlog transforms
        r_integral = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)
        safe_range = (r_integral > 2 * np.min(r_integral) ) & (r_integral < 1/2 * np.max(r_integral) )
        
        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        eta_cga = self.eta + self.eta_delta
        tau_cga = self.tau + self.tau_delta
        
        f_star = 2 * self.A * ((M_use/self.M1)**self.tau + (M_use/self.M1)**self.eta)**-1
        f_cga  = 2 * self.A * ((M_use/self.M1)**tau_cga  + (M_use/self.M1)**eta_cga)**-1
        f_star = f_star[:, None]
        f_cga  = f_cga[:, None]
        f_sga  = f_star - f_cga
        f_clm  = 1 - cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m + f_sga
        
        
        rho_i      = self.DarkMatter.real(cosmo, r_integral, M_use, a)
        rho_cga    = self.Stars.real(cosmo, r_integral, M_use, a)
        rho_gas    = self.Gas.real(cosmo, r_integral, M_use, a)

        dlnr  = np.log(r_integral[1]) - np.log(r_integral[0])
        dV    = r_integral**3 * dlnr
        M_i   = 4 * np.pi * integrate.cumulative_simpson(dV * rho_i  , axis = -1, initial = dV[0] * rho_i[:, [0]])
        M_cga = 4 * np.pi * integrate.cumulative_simpson(dV * rho_cga, axis = -1, initial = dV[0] * rho_cga[:, [0]])
        M_gas = 4 * np.pi * integrate.cumulative_simpson(dV * rho_gas, axis = -1, initial = dV[0] * rho_gas[:, [0]])
        
        #We intentionally set Extrapolate = True. This is to handle behavior at extreme small-scales (due to stellar profile)
        #and radius limits at largest scales. Using extrapolate=True does not introduce numerical artifacts into predictions
        ln_M_NFW = [interpolate.PchipInterpolator(np.log(r_integral), np.log(M_i[m_i]),   extrapolate = True) for m_i in range(M_i.shape[0])]
        ln_M_cga = [interpolate.PchipInterpolator(np.log(r_integral), np.log(M_cga[m_i]), extrapolate = True) for m_i in range(M_i.shape[0])]
        ln_M_gas = [interpolate.PchipInterpolator(np.log(r_integral), np.log(M_gas[m_i]), extrapolate = True) for m_i in range(M_i.shape[0])]

        del M_cga, M_gas, rho_i, rho_cga, rho_gas

        relaxation_fraction = np.ones_like(M_i)

        for m_i in range(M_i.shape[0]):
            
            counter  = 0
            max_rel_diff = np.inf #Initializing variable at infinity
            
            while max_rel_diff > self.reltol:

                with np.errstate(over = 'ignore'):
                    r_f  = r_integral*relaxation_fraction[m_i]
                    M_f  = f_clm[m_i]*M_i[m_i] + np.exp(ln_M_cga[m_i](np.log(r_f))) + np.exp(ln_M_gas[m_i](np.log(r_f)))

                relaxation_fraction_new = self.a*( (M_i[m_i]/M_f)**self.n - 1 ) + 1

                diff     = relaxation_fraction_new/relaxation_fraction[m_i] - 1
                abs_diff = np.abs(diff)
                
                max_rel_diff = np.max(abs_diff[safe_range])
                
                relaxation_fraction[m_i] = relaxation_fraction_new

                counter += 1

                #Though we do a while loop, we break it off after 10 tries
                #this seems to work well enough. The loop converges
                #after two or three iterations.
                if (counter >= self.max_iter) & (max_rel_diff > self.reltol): 
                    
                    med_rel_diff = np.max(abs_diff[safe_range])
                    warn_text = ("Profile of halo index %d did not converge after %d tries. " % (m_i, counter) +
                                 "Max_diff = %0.5f, Median_diff = %0.5f. Try increasing max_iter." % (max_rel_diff, med_rel_diff)
                                )
                    
                    warnings.warn(warn_text, UserWarning)
                    break

        ln_M_clm = np.vstack([np.log(f_clm[m_i]) + ln_M_NFW[m_i](np.log(r_integral/relaxation_fraction[m_i])) for m_i in range(M_i.shape[0])])
        ln_M_clm = interpolate.CubicSpline(np.log(r_integral), ln_M_clm, axis = -1, extrapolate = False)
        log_der  = ln_M_clm.derivative(nu = 1)(np.log(r_use))
        lin_der  = log_der * np.exp(ln_M_clm(np.log(r_use))) / r_use
        prof     = 1/(4*np.pi*r_use**2) * lin_der
        
        arg  = (r_use[None, :] - self.cutoff)
        arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof = np.where(np.isnan(prof), 0, prof) * kfac

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof


class DarkMatterOnly(SchneiderProfiles):

    """
    Class representing a combined dark matter profile using the NFW profile and the two-halo term.

    This class is derived from the `SchneiderProfiles` class and provides an implementation 
    that combines the contributions from the Navarro-Frenk-White (NFW) profile (representing 
    dark matter within the halo) and the two-halo term (representing the contribution of 
    neighboring halos). This approach models the total dark matter distribution by considering 
    both the one-halo and two-halo terms.

    Parameters
    ----------
    darkmatter : DarkMatter, optional
        An instance of the `DarkMatter` class defining the NFW profile for dark matter within 
        a halo. If not provided, a default `DarkMatter` object is created using `kwargs`.
    twohalo : TwoHalo, optional
        An instance of the `TwoHalo` class defining the two-halo term profile, representing 
        the contribution from neighboring halos. If not provided, a default `TwoHalo` object 
        is created using `kwargs`.
    **kwargs
        Additional keyword arguments passed to initialize the `DarkMatter` and `TwoHalo` 
        profiles, as well as other parameters from `SchneiderProfiles`.

    Notes
    -----
    The `DarkMatterOnly` class models the total dark matter density profile by summing 
    the contributions from a one-halo term (using the NFW profile) and a two-halo term. 
    This provides a more complete description of the dark matter distribution, accounting 
    for both the mass within individual halos and the influence of surrounding structure.

    The total dark matter density profile is calculated as:

    .. math::

        \\rho_{\\text{DMO}}(r) = \\rho_{\\text{NFW}}(r) + \\rho_{\\text{2h}}(r)

    where:

    - :math:`\\rho_{\\text{NFW}}(r)` is the NFW profile for the dark matter halo.
    - :math:`\\rho_{\\text{2h}}(r)` is the two-halo term representing contributions from 
      neighboring halos.
    - :math:`r` is the radial distance from the center of the halo.

    This class provides a way to model dark matter distribution that includes the impact 
    of both the immediate halo and the larger-scale structure, which is important for 
    understanding clustering and cosmic structure formation.

    See the `DarkMatter` and `TwoHalo` classes for more details on the underlying profiles 
    and their parameters.
    """

    def __init__(self, darkmatter = None, twohalo = None, **kwargs):
        
        self.DarkMatter = darkmatter
        self.TwoHalo    = twohalo
        
        if self.TwoHalo is None: self.TwoHalo = TwoHalo(**kwargs)
        if self.DarkMatter is None: self.DarkMatter = DarkMatter(**kwargs)
            
        super().__init__(**kwargs)
        
    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        prof = (self.DarkMatter.real(cosmo, r, M, a) +
                self.TwoHalo.real(cosmo, r, M, a)
               )

        return prof


class DarkMatterBaryon(SchneiderProfiles):

    """
    Class representing a combined dark matter and baryonic matter profile.

    This class is derived from the `SchneiderProfiles` class and provides an implementation 
    that combines the contributions from dark matter, gas, stars, and collisionless matter 
    to compute the total density profile. It includes both one-halo and two-halo terms, 
    ensuring mass conservation and accounting for both dark matter and baryonic components.

    Parameters
    ----------
    gas : Gas, optional
        An instance of the `Gas` class defining the gas profile. If not provided, a default 
        `Gas` object is created using `kwargs`.
    stars : Stars, optional
        An instance of the `Stars` class defining the stellar profile. If not provided, a default 
        `Stars` object is created using `kwargs`.
    collisionlessmatter : CollisionlessMatter, optional
        An instance of the `CollisionlessMatter` class defining the profile that combines dark matter, 
        gas, and stars. If not provided, a default `CollisionlessMatter` object is created using `kwargs`.
    darkmatter : DarkMatter, optional
        An instance of the `DarkMatter` class defining the NFW profile for dark matter. If not provided, 
        a default `DarkMatter` object is created using `kwargs`.
    twohalo : TwoHalo, optional
        An instance of the `TwoHalo` class defining the two-halo term profile, representing 
        the contribution of neighboring halos. If not provided, a default `TwoHalo` object is created using `kwargs`.
    **kwargs
        Additional keyword arguments passed to initialize the `Gas`, `Stars`, `CollisionlessMatter`, 
        `DarkMatter`, and `TwoHalo` profiles, as well as other parameters from `SchneiderProfiles`.

    Notes
    -----
    The `DarkMatterBaryon` class models the total matter density profile by combining 
    contributions from collisionless matter, gas, stars, dark matter, and the two-halo term. 
    This comprehensive approach accounts for the interaction and distribution of both dark 
    matter and baryonic matter within halos and across neighboring halos.

    **Calculation Steps:**

    1. **Normalization of Dark Matter**: To ensure mass conservation, the one-halo term is 
       normalized so that the dark matter-only profile matches the dark matter-baryon 
       profile at large radii. The normalization factor is calculated as:

       .. math::

           \\text{Factor} = \\frac{M_{\\text{DMO}}}{M_{\\text{DMB}}}

       where:

       - :math:`M_{\\text{DMO}}` is the total mass from the dark matter-only profile.
       - :math:`M_{\\text{DMB}}` is the total mass from the combined dark matter and baryon profile.

    2. **Total Density Profile**: The total density profile is computed by summing the contributions 
       from the collisionless matter, stars, gas, and two-halo term, scaled by the normalization factor:

       .. math::

           \\rho_{\\text{total}}(r) = \\rho_{\\text{CLM}}(r) \\cdot \\text{Factor} + \\rho_{\\text{stars}}(r) \\cdot \\text{Factor} + \\rho_{\\text{gas}}(r) \\cdot \\text{Factor} + \\rho_{\\text{2h}}(r)

       where:

       - :math:`\\rho_{\\text{CLM}}(r)` is the density from the collisionless matter profile.
       - :math:`\\rho_{\\text{stars}}(r)` is the stellar density profile.
       - :math:`\\rho_{\\text{gas}}(r)` is the gas density profile.
       - :math:`\\rho_{\\text{2h}}(r)` is the two-halo term density profile.

    This method ensures that both dark matter and baryonic matter are accounted for, 
    providing a realistic representation of the total matter distribution.

    See `SchneiderProfiles`, `Gas`, `Stars`, `CollisionlessMatter`, `DarkMatter`, and `TwoHalo` 
    classes for more details on the underlying profiles and parameters.
    """

    def __init__(self, gas = None, stars = None, collisionlessmatter = None, darkmatter = None, twohalo = None, 
                 r_min_int = 1e-5, r_max_int = 100, r_steps = 500, **kwargs):
        
        self.Gas   = gas
        self.Stars = stars
        self.TwoHalo    = twohalo
        self.DarkMatter = darkmatter
        self.CollisionlessMatter = collisionlessmatter
        
        if self.Gas is None: self.Gas = Gas(**kwargs)          
        if self.Stars is None: self.Stars = Stars(**kwargs)
        if self.TwoHalo is None: self.TwoHalo = TwoHalo(**kwargs)
        if self.DarkMatter is None: self.DarkMatter = DarkMatter(**kwargs)
        if self.CollisionlessMatter is None: self.CollisionlessMatter = CollisionlessMatter(**kwargs)
            
        super().__init__(**kwargs, r_min_int = r_min_int, r_max_int = r_max_int, r_steps = r_steps)
        
    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        #Need DMO for normalization
        #Makes sure that M_DMO(<r) = M_DMB(<r) for the limit r --> infinity
        #This is just for the onehalo term
        r_integral = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)

        rho   = self.DarkMatter.real(cosmo, r_integral, M, a)
        M_tot = np.trapz(4*np.pi*r_integral**2 * rho, r_integral)

        rho   = (self.CollisionlessMatter.real(cosmo, r_integral, M, a) +
                 self.Stars.real(cosmo, r_integral, M, a) +
                 self.Gas.real(cosmo, r_integral, M, a))

        M_tot_dmb = np.trapz(4*np.pi*r_integral**2 * rho, r_integral, axis = -1)

        Factor = M_tot/M_tot_dmb
        
        if np.ndim(Factor) == 1:
            Factor = Factor[:, None]

        prof = (self.CollisionlessMatter.real(cosmo, r, M, a) * Factor +
                self.Stars.real(cosmo, r, M, a) * Factor +
                self.Gas.real(cosmo, r, M, a) * Factor +
                self.TwoHalo.real(cosmo, r, M, a))

        return prof