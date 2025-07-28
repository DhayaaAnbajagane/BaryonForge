import numpy as np
import pyccl as ccl
import warnings

from scipy import interpolate, integrate
from . import Schneider19 as S19

__all__ = ['model_params', 'SchneiderProfiles', 
           'DarkMatter', 'TwoHalo', 'Stars', 'SatelliteStars', 
           'Gas', 'ShockedGas', 'CollisionlessMatter',
           'DarkMatterOnly', 'DarkMatterBaryon']


model_params = ['cdelta', 'epsilon0', 'epsilon1', 'alpha_excl', 'q', 'p', #DM profle params
                'cutoff', 'proj_cutoff', #Cutoff parameters (numerical)

                'q0', 'q1', 'q2', 'nu_q0', 'nu_q1', 'nu_q2', 'nstep', #Relaxation params
                
                'theta_c', 'M_c', 'gamma', 'delta', 'alpha',  #Default gas profile param
                'mu_theta_c', 'mu_beta', 'mu_gamma', 'mu_delta', 'mu_alpha', #Mass dep
                'M_theta_c', 'M_gamma', 'M_delta', 'M_alpha', #Mass dep norm
                'nu_theta_c', 'nu_M_c',  'nu_gamma', 'nu_delta', 'nu_alpha', #Redshift  dep
                'zeta_theta_c', 'zeta_M_c', 'zeta_gamma', 'zeta_delta',  'zeta_alpha', #Concentration dep
                'c_iga', 'nu_c_iga', #proportionality constant for inner gas fraction
                
                'Nstar', 'Mstar', 'eta', 'eta_delta', 'tau', 'tau_delta', 'epsilon_cga', #Star params
                
                'alpha_nt', 'nu_nt', 'gamma_nt', 'mean_molecular_weight' #Non-thermal pressure and gas density
               ]

class Schneider25Profiles(S19.SchneiderProfiles):

    #Define the new param names
    model_param_names = model_params

    #Use a smaller r_max, since most profiles are truncated at R200c now.
    def __init__(self, r_max_int = 10, **kwargs):
        
        super().__init__(**kwargs, r_max_int = r_max_int)
        
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


        #Sets the cutoff scale of all profiles, in comoving Mpc. Prevents divergence in FFTLog
        #Also set cutoff of projection integral. Should be the box side length
        self.cutoff      = kwargs['cutoff'] if 'cutoff' in kwargs.keys() else 1e3 #1Gpc is a safe default choice
        self.proj_cutoff = kwargs['proj_cutoff'] if 'proj_cutoff' in kwargs.keys() else self.cutoff


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
        theta_c : ndarray
            Core radius parameter.
        delta : ndarray
            Large-scale slope.
        gamma : ndarray
            Intermediate-scale slope.
        alpha : ndarray
            core slope.
        """
        
        cdelta   = 1 if self.cdelta is None else self.cdelta
        
        M_c      = self.M_c * (1 + z)**self.nu_M_c * cdelta**self.zeta_M_c
        beta     = 3*(M/M_c)**self.mu_beta / (1 + (M/M_c)**self.mu_beta)
        
        #Use M_c as the mass-normalization for simplicity sake
        theta_c  = self.theta_c  * (M/self.M_theta_c)**self.mu_theta_c   * (1 + z)**self.nu_theta_c  * cdelta**self.zeta_theta_c 
        delta    = self.delta    * (M/self.M_delta)**self.mu_delta       * (1 + z)**self.nu_delta    * cdelta**self.zeta_delta
        gamma    = self.gamma    * (M/self.M_gamma)**self.mu_gamma       * (1 + z)**self.nu_gamma    * cdelta**self.zeta_gamma
        alpha    = self.alpha    * (M/self.M_alpha)**self.mu_alpha       * (1 + z)**self.nu_alpha    * cdelta**self.zeta_alpha
        
        beta     = beta[:, None]
        theta_c  = theta_c [:, None]
        delta    = delta[:, None]
        gamma    = gamma[:, None]
        alpha    = alpha[:, None]
        
        return beta, theta_c , delta, gamma, alpha


class Schneider25Fractions:
    
    def _get_star_frac(self, M_use, a, cosmo):
        
        """
        Compute the fractional mass components of stars in the full halo, central galaxy (cga), 
        and satellite galaxy (sga) for a given set of halo masses and scale factor.

        Parameters
        ----------
        M_use : ndarray
            Array of halo masses for which to compute the baryonic fractions.
        a : float
            Scale factor at which the computation is performed (unused in this function but
            may be relevant for future extensions).
        cosmo : object
            Cosmology object containing cosmological parameters, specifically Omega_b and Omega_m.

        Returns
        -------
        f_star : ndarray
            Stellar mass fraction for each halo, clipped to be between 1e-10 and the cosmic 
            baryon fraction.
        f_cga : ndarray
            Central galaxy mass fraction, clipped to be between 1e-10 and the stellar fraction.
        f_sga : ndarray
            Satellite galaxy mass fraction, defined as `f_star - f_cga`, and clipped to be at 
            least 1e-10 to avoid issues in log calculations.

        Notes
        -----
        The function ensures numerical stability by enforcing lower bounds of 1e-10 on all 
        returned fractions and upper bounds that respect physical limits on baryon content.
        """
            
        eta_cga = self.eta + self.eta_delta
        tau_cga = self.tau + self.tau_delta
        
        f_bar  = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        f_star = self.Nstar / ((M_use/self.Mstar)**self.tau + (M_use/self.Mstar)**self.eta)
        f_cga  = self.Nstar / ((M_use/self.Mstar)**tau_cga  + (M_use/self.Mstar)**eta_cga)
        
        #Star frac cannot be larger than baryon fraction. If it is 0 then the code fails
        #when taking logs of profiles. So give it a super small value instead.
        #Similarly, the cga fraction cannot be larger than the star fraction.
        f_star = np.clip(f_star, 1e-10, f_bar)
        f_cga  = np.clip(f_cga,  1e-10, f_star)
        
        f_star = f_star[:, None]
        f_cga  = f_cga[:, None]
        
        f_sga  = np.clip(f_star - f_cga, 1e-10, None) 
        
        return f_star, f_cga, f_sga
    
    
    def _get_gas_frac(self, M_use, a, cosmo):

        """
        Compute the fractional mass components of hot gas (hga) and inner gas (iga) 
        for a given set of halo masses and scale factor.

        Parameters
        ----------
        M_use : ndarray
            Array of halo masses for which to compute the gas fractions.
        a : float
            Scale factor at which the computation is performed.
        cosmo : object
            Cosmology object containing cosmological parameters, specifically Omega_b and Omega_m.

        Returns
        -------
        f_hga : ndarray
            Hot gas mass fraction for each halo, defined as the remaining baryon fraction 
            after accounting for stars and inner gas. Clipped to be at least 1e-10.
        f_iga : ndarray
            Inner gas mass fraction for each halo, computed from the hot gas fraction 
            with a redshift-dependent suppression factor. Clipped to be between 1e-10 and 
            `f_bar - f_star`.

        Notes
        -----
        The function internally calls `_get_star_frac` to obtain the stellar 
        components. All outputs are clipped to avoid zero values, which would cause issues 
        in downstream logarithmic calculations.
        """

        
        f_star, f_cga, f_sga = self._get_star_frac(M_use, a, cosmo)
        f_bar  = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        f_iga  = f_cga * self.c_iga * np.power(a, -self.nu_c_iga) #-ve sign since we do a^nu instead of (1 + z)^nu
        f_iga  = np.clip(f_iga, 1e-10, f_bar - f_star)
        f_hga  = np.clip(f_bar - f_star - f_iga, 1e-10, f_bar) #Cannot let the fraction be identically 0.        
        
        return f_hga, f_iga
        
        
        
class DarkMatter(Schneider25Profiles):
    """
    Class representing the dark matter (DM) density profile using a truncated Navarro-Frenk-White (NFW) model.

    This class extends `Schneider25Profiles` to implement a real-space DM density profile that includes:
    - A standard NFW profile,
    - A truncation factor based on a halo-specific radius,
    - An additional exponential cutoff to prevent numerical overflow at large radii.

    The truncation radius is defined as:

    .. math::

        r_t = \\epsilon(\\nu) \\cdot R,

    where:

    .. math::

        \\epsilon(\\nu) = \\epsilon_0 + \\epsilon_1 \\nu,

    and \\( \\nu \\) is the peak height of the halo.

    The normalization of the profile is computed numerically by integrating over each halo’s radius from 
    a fixed minimum value to its virial radius. This ensures the total mass is preserved while accounting 
    for truncation.

    The final density profile is given by:

    .. math::

        \\rho(r) = \\frac{\\rho_c}{(r/r_s)(1 + r/r_s)^2} 
        \\cdot \\frac{1}{\\left(1 + (r/r_t)^2\\right)^2}
        \\cdot \\frac{1}{1 + \\exp\\left[2(r - r_\\text{cutoff})\\right]},

    where:
    - \\( \\rho_c \\) is the normalization to match the total halo mass,
    - \\( r_s = R / c \\) is the scale radius from the concentration–mass relation,
    - \\( r_t \\) is the truncation radius as defined above,
    - \\( r_\\text{cutoff} \\) is a user-defined soft cutoff radius to suppress unphysical densities at large \\( r \\).

    Notes
    -----
    If no concentration–mass relation is provided via `cdelta` or `c_M_relation`, the default 
    Diemer & Kravtsov (2015) relation is used. The method handles both scalar and array inputs 
    for radii and halo masses.

    See `Schneider25Profiles` for base class documentation.
    """

    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        if (self.cdelta is None) and (self.c_M_relation is None):
            c_M_relation = ccl.halos.concentration.ConcentrationDiemer15(mass_def = self.mass_def) #Use the diemer calibration
        elif self.c_M_relation is not None:
            c_M_relation = self.c_M_relation
        else:
            assert self.cdelta is not None, "Either provide cdelta or a c_M_relation input"
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mass_def = self.mass_def)
            
        c   = c_M_relation(cosmo, M_use, a)
        c   = np.where(np.isfinite(c), c, 1) #Set default to r_s = R200c if c200c broken (normally for low mass obj in some cosmologies)
        R   = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        r_s = R/c
        nu  = 1.686/ccl.sigmaM(cosmo, M_use, a)
        eps = self.epsilon0 + self.epsilon1 * nu
        r_t = R*eps
        
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


class TwoHalo(Schneider25Profiles):
    """
    Class representing the two-halo term profile.

    This class is derived from the `Schneider25Profiles` class and provides an implementation 
    of the two-halo term profile. It utilizes the 2-point correlation function directly, rather 
    than employing the full halo model. 

    See `Schneider25Profiles` for more docstring details.

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
        f_excl  = 1 - np.exp(-self.alpha_excl * np.clip(r_use / R[:, None], 0, 30)) #Clip to avoid overflow

        bias_M  = bias_M[:, None]
        prof    = f_excl * (1 + bias_M * xi_mm)*ccl.rho_x(cosmo, a, species = 'matter', is_comoving = True)

        #Need this truncation so the fourier space integral isnt infinity
        arg  = (r_use[None, :] - self.cutoff)
        arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof = prof * kfac

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof


class Stars(Schneider25Profiles, Schneider25Fractions):
    """
    Class representing the two-halo term density profile based on linear theory.

    This class extends `Schneider25Profiles` and implements the large-scale clustering 
    contribution to the halo density profile (the two-halo term) using the linear matter 
    correlation function and a Sheth-Tormen-type halo bias prescription.

    The real-space profile is defined as:

    .. math::

        \\rho_{\\mathrm{2h}}(r) = \\left[1 + b(M)\\,\\xi_{\\mathrm{mm}}(r)\\right] \\cdot \\bar{\\rho}_m(a) 
        \\cdot f_{\\mathrm{excl}}(r, R) \\cdot k_{\mathrm{cut}}(r),

    where:
    - \\( b(M) \\) is the halo bias, given by:

      .. math::

          b(M) = 1 + \\frac{q \\nu^2 - 1}{\\delta_c} + \\frac{2p}{\\delta_c\\left[1 + (q \\nu^2)^p\\right]},

    - \\( \\nu = \\delta_c / \\sigma(M) \\) is the peak height,
    - \\( \\delta_c \\) is the critical overdensity for collapse (rescaled by growth factor),
    - \\( \\xi_{\\mathrm{mm}}(r) \\) is the linear matter correlation function,
    - \\( \\bar{\\rho}_m(a) \\) is the mean matter density at scale factor \\( a \\),
    - \\( f_{\\mathrm{excl}}(r, R) = 1 - \\exp\\left[-\\alpha_{\\mathrm{excl}} \\cdot (r / R)\\right] \\) 
      suppresses the profile at small radii,
    - \\( k_{\\mathrm{cut}}(r) = [1 + \\exp(2(r - r_{\\mathrm{cutoff}}))]^{-1} \\) imposes an 
      exponential cutoff at large radii to ensure convergence in Fourier space.

    Notes
    -----
    - The two-halo term is valid only when the cosmology object's matter power spectrum is linear.
      An assertion enforces this requirement.
    - If `xi_mm` is not provided at initialization, it is computed using 
      `pyccl.correlation_3d`.
    - Scalar and vector inputs for mass and radius are supported and mirrored in the output shape.

    See also
    --------
    Sheth & Tormen (1999), https://arxiv.org/abs/astro-ph/9901122
    `Schneider25Profiles` base class for interface and shared attributes.

    Examples
    --------
    >>> profile = TwoHalo(**parameters)
    >>> cosmo = ...  # cosmology with linear power spectrum
    >>> r = np.logspace(-2, 1, 100)
    >>> M = 1e14
    >>> a = 0.5
    >>> rho_2h = profile.real(cosmo, r, M, a)
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

        f_cga  = self._get_star_frac(M_use, a, cosmo)[1]
        R_cga  = self.epsilon_cga * R[:, None]

        r_integral = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)
        DM    = DarkMatter(**self.model_params); setattr(DM, 'cutoff', 1e3) #Set large cutoff just for normalization calculation
        rho   = DM.real(cosmo, r_integral, M_use, a)
        M_tot = np.trapz(4*np.pi*r_integral**2 * rho, r_integral, axis = -1)
        M_tot = np.atleast_1d(M_tot)[:, None]


        #Integrate over wider region in radii to get normalization of star profile
        prof_integral = 1 / np.power(r_integral, 2) * np.exp(-r_integral/R_cga)
        Normalization = np.trapz(4 * np.pi * r_integral**2 * prof_integral, r_integral, axis = -1)[:, None]
        
        arg  = (r_use[None, :] - self.cutoff)
        arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof = 1/r_use**2 * np.exp(-r_use/R_cga) * kfac
        prof = prof * f_cga*M_tot/Normalization
                
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof


class HotGas(Schneider25Profiles, Schneider25Fractions):

    """
    Class representing the hot gas density profile in galaxy halos using a generalized NFW form.

    This class extends both `Schneider25Profiles` and `Schneider25Fractions` and implements the 
    real-space hot gas density profile following the GNFW parameterization from 
    Nagai, Kravtsov & Vikhlinin (2007). The profile accounts for feedback-driven redistribution 
    of gas using mass- and redshift-dependent core and ejection radii.

    The real-space profile is given by:

    .. math::

        \\rho_{\\mathrm{gas}}(r) = \\frac{f_{\\mathrm{hga}} M_{\\mathrm{tot}}}{N} 
        \\cdot \\frac{1}{\\left(1 + (r/R_{\\mathrm{co}})^\\alpha\\right)^{\\beta/\\alpha}} 
        \\cdot \\frac{1}{\\left(1 + (r/R_{\\mathrm{ej}})^\\gamma\\right)^{\\delta/\\gamma}} 
        \\cdot k_{\\mathrm{cut}}(r),

    where:
    - \\( f_{\\mathrm{hga}} \\) is the hot gas fraction computed from the total baryon budget,
    - \\( M_{\\mathrm{tot}} \\) is the total halo mass from a dark matter profile,
    - \\( N \\) is a normalization factor from integrating the GNFW profile,
    - \\( R_{\\mathrm{co}} = \\theta_{\\mathrm{co}} R \\) is the core radius,
    - \\( R_{\\mathrm{ej}} = \\epsilon(\\nu) R \\) is the ejection radius,
    - \\( \\alpha, \\beta, \\gamma, \\delta \\) are slope parameters characterizing the gas profile,
    - \\( k_{\\mathrm{cut}}(r) = [1 + \\exp(2(r - r_{\\mathrm{cutoff}}))]^{-1} \\) is an exponential 
      cutoff ensuring profile suppression at large radii.

    Notes
    -----
    - The hot gas fraction \\( f_{\\mathrm{hga}} \\) is derived from the cosmic baryon fraction minus the 
      stellar and infalling gas fractions, using internal methods from `Schneider25Fractions`.
    - The profile is normalized by integrating over a wide radial range, with the total mass obtained 
      from an associated `DarkMatter` profile.
    - All components are computed in comoving units, with shape-preserving handling of scalar and 
      array inputs.

    References
    ----------
    - Nagai, Kravtsov & Vikhlinin (2007), https://arxiv.org/abs/astro-ph/0703661

    Examples
    --------
    >>> gas_profile = HotGas(**parameters)
    >>> cosmo = ...  # Cosmology object
    >>> r = np.logspace(-2, 1, 50)  # Radii in comoving Mpc
    >>> M = 1e14  # Halo mass in solar masses
    >>> a = 0.5  # Scale factor
    >>> rho_gas = gas_profile.real(cosmo, r, M, a)
    """


    def _real(self, cosmo, r, M, a):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_hga, f_iga  = self._get_gas_frac(M_use, a, cosmo)
        
        #Get gas params
        beta, theta_c, delta, gamma, alpha = self._get_gas_params(M_use, z)
        R_c = theta_c*R[:, None]
        nu  = 1.686/ccl.sigmaM(cosmo, M_use, a)[:, None]
        eps = self.epsilon0 + self.epsilon1 * nu
        R_t = eps * R[:, None]
        
        u = r_use/R_c
        v = r_use/R_t
        
        
        #Integrate over wider region in radii to get normalization of gas profile
        r_integral = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)
        u_integral = r_integral/R_c
        v_integral = r_integral/R_t
        

        prof_integral = 1/(1 + np.power(u_integral, alpha))**(beta/alpha) / (1 + v_integral**gamma)**(delta/gamma)
        Normalization = np.trapz(4 * np.pi * r_integral**2 * prof_integral, r_integral, axis = -1)[:, None]

        del u_integral, v_integral, prof_integral

        DM    = DarkMatter(**self.model_params); setattr(DM, 'cutoff', 1e3) #Set large cutoff just for normalization calculation
        rho   = DM.real(cosmo, r_integral, M_use, a)
        M_tot = np.trapz(4*np.pi*r_integral**2 * rho, r_integral, axis = -1)
        M_tot = np.atleast_1d(M_tot)[:, None]
        
        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = 1/(1 + np.power(u, alpha))**(beta/alpha) / (1 + v**gamma)**(delta/gamma) * kfac
        prof *= f_hga*M_tot/Normalization
        

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof
    

class InnerGas(Schneider25Profiles, Schneider25Fractions):

    """
    Class representing the inner gas density profile in halos.

    This class extends both `Schneider25Profiles` and `Schneider25Fractions` to implement 
    a real-space profile for the centrally concentrated inner gas component. The profile 
    is designed to capture steep gas distributions that dominate at small radii.

    The real-space profile is given by:

    .. math::

        \\rho_{\\mathrm{inner}}(r) = \\frac{f_{\\mathrm{iga}} M_{\\mathrm{tot}}}{N} 
        \\cdot r^{-2} \\cdot e^{-r / R} \\cdot k_{\\mathrm{cut}}(r),

    where:
    - \\( f_{\\mathrm{iga}} \\) is the inner gas fraction computed from the baryon budget,
    - \\( M_{\\mathrm{tot}} \\) is the total halo mass obtained by integrating a `DarkMatter` profile,
    - \\( N \\) is a normalization factor ensuring mass conservation over the integration range,
    - \\( R \\) is the halo radius from the mass definition,
    - \\( k_{\\mathrm{cut}}(r) = [1 + \\exp(2(r - r_{\\mathrm{cutoff}}))]^{-1} \\) is an exponential cutoff 
      applied at large radii for numerical stability.

    Notes
    -----
    - The inner gas fraction \\( f_{\\mathrm{iga}} \\) is computed using `_get_gas_frac()`, 
      which also returns the hot gas component.
    - The profile normalization is computed numerically over a wide radial range to match the 
      total inner gas mass.
    - Scalar and array inputs for both radius and halo mass are supported.
    """


    def _real(self, cosmo, r, M, a):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_hga, f_iga  = self._get_gas_frac(M_use, a, cosmo)
        
        #Integrate over wider region in radii to get normalization of gas profile
        r_integral = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)
        
        prof_integral = np.power(r_integral, -2) * np.exp(-r_integral/R[:, None])
        Normalization = np.trapz(4 * np.pi * r_integral**2 * prof_integral, r_integral, axis = -1)[:, None]

        DM    = DarkMatter(**self.model_params); setattr(DM, 'cutoff', 1e3) #Set large cutoff just for normalization calculation
        rho   = DM.real(cosmo, r_integral, M_use, a)
        M_tot = np.trapz(4*np.pi*r_integral**2 * rho, r_integral, axis = -1)
        M_tot = np.atleast_1d(M_tot)[:, None]
        
        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = np.power(r_use, -2) * np.exp(-r_use/R[:, None]) * kfac
        prof *= f_iga*M_tot/Normalization
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof


class Gas(Schneider25Profiles, Schneider25Fractions):
    """
    Convenience class for combining gas components in halos.

    The `Gas` class provides a unified interface for modeling the total gas profile in halos. 
    It combines contributions from the following components:
    - `HotGas`: Represents the hot gas component within halos.
    - `InnerGas`: Represents gas in the inner core of the halo. Generally not a notable fraction of the gas

    This class simplifies calculations by leveraging the logic and methods of these individual 
    gas components and combining their profiles into a single representation.
    """

    def __init__(self, **kwargs): self.myprof = HotGas(**kwargs) + InnerGas(**kwargs)
    def __getattr__(self, name):  return getattr(self.myprof, name)
    
    #Need to explicitly set these two methods (to enable pickling)
    #since otherwise the getattr call above leads to infinite recursions.
    def __getstate__(self): self.__dict__.copy()    
    def __setstate__(self, state): self.__dict__.update(state)


class CollisionlessMatter(Schneider25Profiles, Schneider25Fractions):

    """
    Class representing the collisionless matter density profile after adiabatic relaxation.

    This class extends `Schneider25Profiles` and `Schneider25Fractions` to compute the final 
    density profile of collisionless matter (dark matter + stars) after accounting for 
    the effects of baryonic components such as hot and inner gas. The computation is 
    performed using a (non-iterative) relaxation method based on the formalism in Schneider et al. (2025).

    The final profile reflects how baryons gravitationally reshape the collisionless component, 
    conserving total mass and enforcing physical consistency between baryonic and dark matter 
    distributions.

    Parameters
    ----------
    hotgas : HotGas, optional
        Instance of the `HotGas` class. If not provided, one is created from `kwargs`.
    innergas : InnerGas, optional
        Instance of the `InnerGas` class. If not provided, one is created from `kwargs`.
    stars : Stars, optional
        Instance of the `Stars` class. If not provided, one is created from `kwargs`.
    darkmatter : DarkMatter, optional
        Instance of the `DarkMatter` class. If not provided, one is created from `kwargs`.
    r_min_int : float, optional
        Minimum radius for internal integrations (default: 1e-8 Mpc).
    r_max_int : float, optional
        Maximum radius for internal integrations (default: 1e5 Mpc).
    r_steps : int, optional
        Number of radial steps used for integration (default: 5000).
    **kwargs : dict
        Additional keyword arguments passed to initialize subcomponents and the base class.

    Notes
    -----
    The relaxation process proceeds as follows:

    1. Compute individual density profiles for dark matter, hot gas, inner gas, and stars.
    2. Integrate each profile to obtain cumulative mass profiles.
    3. Compute the collisionless matter mass profile:

       .. math::

           M_{\\mathrm{CLM}}(r) = f_{\\mathrm{clm}} \\cdot M_{\\mathrm{tot}}(r')

       where \\( f_{\\mathrm{clm}} = 1 - \\Omega_b / \\Omega_m + f_{\\mathrm{sga}} \\) 
       and \\( r' = r \\cdot \\zeta \\) with the relaxation factor \\( \\zeta \\) defined as:

       .. math::

           \\zeta = Q_0 / (1 + (r/r_s)^n) + Q_1 f_{\\mathrm{cga}} \\left( \\frac{M_{\\mathrm{stars}}}{M_{\\mathrm{DM}}} - 1 \\right)
                 + Q_1 f_{\\mathrm{iga}} \\left( \\frac{M_{\\mathrm{inner}}}{M_{\\mathrm{DM}}} - 1 \\right)
                 + Q_2 f_{\\mathrm{hga}} \\left( \\frac{M_{\\mathrm{hot}}}{M_{\\mathrm{DM}}} - 1 \\right) + 1

    4. Differentiate the relaxed mass profile to obtain the final density:

       .. math::

           \\rho_{\\mathrm{CLM}}(r) = \\frac{1}{4\\pi r^2} \\frac{dM_{\\mathrm{CLM}}}{dr}

    An additional exponential cutoff is applied at large radii for numerical stability.

    Warnings
    --------
    A warning is issued if the requested radii fall outside the integration bounds. These warnings 
    are often benign and can be ignored if they arise from extreme FFTlog sampling.

    Examples
    --------
    >>> clm = CollisionlessMatter(**parameters)
    >>> cosmo = ...  # Cosmology object
    >>> r = np.logspace(-2, 1, 100)
    >>> M = 1e14
    >>> a = 0.5
    >>> rho_clm = clm.real(cosmo, r, M, a)
    """

    
    def __init__(self, hotgas = None, innergas = None, stars = None, darkmatter = None, r_min_int = 1e-8, r_max_int = 1e5, r_steps = 5000, **kwargs):
        
        self.HotGas     = hotgas
        self.InnerGas   = innergas
        self.Stars      = stars
        self.DarkMatter = darkmatter
        
        if self.HotGas is None:     self.HotGas     = HotGas(**kwargs)
        if self.InnerGas is None:   self.InnerGas   = InnerGas(**kwargs)      
        if self.Stars is None:      self.Stars      = Stars(**kwargs)
        if self.DarkMatter is None: self.DarkMatter = DarkMatter(**kwargs)

        #Stop any artificially cutoffs when doing the relaxation.
        #The profile will be cutoff at the very last step instead
        self.Stars.set_parameter('cutoff', 1000)
        self.HotGas.set_parameter('cutoff', 1000)
        self.InnerGas.set_parameter('cutoff', 1000)
        self.DarkMatter.set_parameter('cutoff', 1000)
            
        self.r_min_int  = r_min_int
        self.r_max_int  = r_max_int
        self.r_steps    = r_steps
        
        super().__init__(**kwargs, r_min_int = r_min_int, r_max_int = r_max_int, r_steps = r_steps)
        

    def _get_Qis(self, M, a, cosmo):

        z  = 1/a - 1
        Q0 = self.q0 * np.power(1 + z, self.nu_q0)
        Q1 = self.q1 * np.power(1 + z, self.nu_q1)
        Q2 = self.q2 * np.power(1 + z, self.nu_q2)

        return Q0, Q1, Q2
    
        
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

        f_star, f_cga, f_sga = self._get_star_frac(M_use, a, cosmo)
        f_hga,  f_iga        = self._get_gas_frac(M_use, a, cosmo)
        Q0, Q1, Q2           = self._get_Qis(M_use, a, cosmo)

        f_clm      = 1 - cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m + f_sga
        nu         = 1.686/ccl.sigmaM(cosmo, M_use, a)[:, None]
        eps        = self.epsilon0 + self.epsilon1 * nu
        rstep      = eps / self.epsilon0
        
        rho_i      = self.DarkMatter.real(cosmo, r_integral, M_use, a)
        rho_cga    = self.Stars.real(cosmo, r_integral, M_use, a)
        rho_hga    = self.HotGas.real(cosmo, r_integral, M_use, a)
        rho_iga    = self.InnerGas.real(cosmo, r_integral, M_use, a)
        

        #Need to add the offset manually now since scipy deprecates initial != 0
        #Offset required so that the integrated array has the same size as the profile array
        dlnr  = np.log(r_integral[1]) - np.log(r_integral[0])
        dV    = 4 * np.pi * r_integral**3 * dlnr
        M_i   = integrate.cumulative_simpson(dV * rho_i  , axis = -1, initial = 0) + dV[0] * rho_i[:, [0]]
        M_cga = integrate.cumulative_simpson(dV * rho_cga, axis = -1, initial = 0) + dV[0] * rho_cga[:, [0]]
        M_hga = integrate.cumulative_simpson(dV * rho_hga, axis = -1, initial = 0) + dV[0] * rho_hga[:, [0]]
        M_iga = integrate.cumulative_simpson(dV * rho_iga, axis = -1, initial = 0) + dV[0] * rho_iga[:, [0]]

        #We intentionally set Extrapolate = True. This is to handle behavior at extreme small-scales (due to stellar profile)
        #and radius limits at largest scales. Using extrapolate=True does not introduce numerical artifacts into predictions
        ln_M_NFW = [interpolate.PchipInterpolator(np.log(r_integral), np.log(M_i[m_i]),   extrapolate = True) for m_i in range(M_i.shape[0])]
        ln_M_clm = np.ones_like(M_i)

        for m_i in range(M_i.shape[0]):
            
            with np.errstate(over = 'ignore'):
                 
                xi0  = Q0 / (1 + np.power(r_integral/rstep, self.nstep))
                xi1  = Q1 * f_cga * (M_cga[m_i] / M_i[m_i] - 1)
                xi2  = Q1 * f_iga * (M_iga[m_i] / M_i[m_i] - 1)
                xi3  = Q2 * f_hga * (M_hga[m_i] / M_i[m_i] - 1)
                relaxation_fraction = xi0 + xi1 + xi2 + xi3 + 1

                #Schneider+25 defines relaxation fraction as r_i/r_f so the bottom should indeed be multiplied,
                #and not divided like we do in Schneider+19, where the definition was r_f/r_i.
                ln_M_clm[m_i] = np.log(f_clm[m_i]) + ln_M_NFW[m_i](np.log(r_integral * relaxation_fraction[m_i]))

        ln_M_clm = interpolate.CubicSpline(np.log(r_integral), ln_M_clm, axis = -1, extrapolate = False)
        log_der  = ln_M_clm.derivative(nu = 1)(np.log(r_use))
        lin_der  = log_der * np.exp(ln_M_clm(np.log(r_use))) / r_use
        prof     = 1/(4*np.pi*r_use**2) * lin_der
        prof     = np.clip(prof, 0, None) #If prof < 0 due to interpolation errors, then force it to 0.
        
        arg  = (r_use[None, :] - self.cutoff)
        arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof = np.where(np.isfinite(prof), prof, 0) * kfac

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof
    

class SatelliteStars(CollisionlessMatter, Schneider25Fractions):

    """
    Class representing the matter density profile of stars in satellites.

    It uses the `CollisionlessMatter` profiles with a simple rescaling to
    get just the SG (satellite galaxies) term alone. See that class for
    more details.
    """
    
    def _real(self, cosmo, r, M, a):

        M_use = np.atleast_1d(M)

        f_sga  = self._get_star_frac(M_use, a, cosmo)[2]
        f_clm  = 1 - cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m + f_sga
        
        if np.ndim(M) == 0: 
            f_clm = np.squeeze(f_clm, axis = 0)
            f_sga = np.squeeze(f_sga, axis = 0)

        prof   = super()._real(cosmo, r, M, a) * (f_sga/f_clm)
        
        return prof


class DarkMatterOnly(Schneider25Profiles):

    """
    Class representing a combined dark matter profile using the NFW profile and the two-halo term.

    This class is derived from the `Schneider25Profiles` class and provides an implementation 
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
        profiles, as well as other parameters from `Schneider25Profiles`.

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


class DarkMatterBaryon(Schneider25Profiles, Schneider25Fractions):

    """
    Class representing a combined dark matter and baryonic matter profile.

    This class is derived from the `Schneider25Profiles` class and provides an implementation 
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
        `DarkMatter`, and `TwoHalo` profiles, as well as other parameters from `Schneider25Profiles`.

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

    See `Schneider25Profiles`, `Gas`, `Stars`, `CollisionlessMatter`, `DarkMatter`, and `TwoHalo` 
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