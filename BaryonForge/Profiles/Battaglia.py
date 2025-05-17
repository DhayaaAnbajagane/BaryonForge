import numpy as np
import pyccl as ccl
from .Thermodynamic import (Msun_to_Kg, Mpc_to_m, G, Y, Pth_to_Pe, Pressure_at_infinity)


__all__ = ['Pressure', 'ElectronPressure', 'GasDensity']


class Pressure(ccl.halos.profiles.HaloProfile):
    """
    Class for implementing the Battaglia pressure profile using CCL's halo profile framework.

    This class computes the pressure profile of halos using the `Battaglia et al. (2012) <https://arxiv.org/pdf/1109.3711>`_ model. 
    The model is based on numerical simulations and provides a way to calculate the electron 
    pressure profile in galaxy clusters, which is useful for studying the thermal Sunyaev-Zel'dovich 
    effect and other astrophysical phenomena. The final profile is in units of comoving
    volume. Use a factor of 1/a^3 (not 1/a^4) to convert to physical pressure.

    Inherits from
    -------------
    ccl.halos.profiles.HaloProfile : Base class for halo profiles in CCL.

    Parameters
    ----------
    Model_def : str
        Specifies the model calibration to use from Battaglia et al. (2012). Options are:
        - '200_AGN': Calibrated using AGN feedback and a 200c overdensity mass definition.
        - '500_AGN': Calibrated using AGN feedback and a 500c overdensity mass definition.
        - '500_SH': Calibrated without AGN feedback and a 500c overdensity mass definition.
    truncate : float, optional
        Radius (in units of \( R / R_{\text{def}} \), where \( R_{\text{def}} \) is the halo 
        radius defined via the chosen spherical overdensity) at which to truncate the profiles 
        and set them to zero. Default is `False`, meaning no truncation is applied.

    Notes
    -----
    - The Battaglia pressure profile is parameterized based on simulations and provides 
    different calibrations depending on the inclusion of AGN feedback and the mass 
    definition used (200c or 500c overdensities). This class supports switching 
    between these calibrations using the `Model_def` parameter.

    - The profile has limits for validity, including redshift, halo mass, and distance 
    from the halo center. The `truncate` parameter can be used to enforce a distance 
    cutoff, setting the profile to zero beyond the specified radius.

    - The pressure profile is computed using the following parameters from Battaglia et al. (2012):

    .. math::

        P(r) = P_\\delta \\cdot P_0 \\cdot \\left( \\frac{x}{x_c} \\right)^\\gamma \\cdot 
        \\left( 1 + \\left( \\frac{x}{x_c} \\right)^\\alpha \\right)^{-\\beta}

    where:
        - \( x = \\frac{r}{R} \) is the radial distance normalized by the halo radius.
        - \( P_\\delta \) is the self-similar expectation for pressure.
        - \( P_0, x_c, \\beta \) are model parameters depending on the chosen `Model_def`.
        - \( \\alpha, \\gamma \) are set to 1 and -0.3, respectively.

    - Cosmological and halo parameters such as \( \\Omega_m \), \( \\Omega_b \), \( \\Omega_g \), 
    and \( h \) are obtained from the `cosmo` object.
    - If the `truncate` parameter is set, the profile is truncated to zero beyond the specified radius.
    - Units are converted to CGS for the final profile values.

    Examples
    --------
    Compute the Battaglia pressure profile for a given cosmology and halo:

    >>> battaglia_model = BattagliaPressure(Model_def='200_AGN', truncate=1.5)
    >>> r = np.logspace(-2, 1, 50)  # Radii in comoving Mpc
    >>> M = 1e14  # Halo mass in solar masses
    >>> a = 0.5  # Scale factor corresponding to redshift z
    >>> pressure_profile = battaglia_model._real(my_cosmology, r, M, a)
    """

    def __init__(self, Model_def, mass_def = ccl.halos.massdef.MassDef200c, truncate = False):

        #Set mass definition using the input Model_def
        if Model_def == '200_AGN':
            self.mdef = ccl.halos.massdef.MassDef(200, 'critical')

        elif Model_def == '500_AGN':
            self.mdef = ccl.halos.massdef.MassDef(500, 'critical')

        elif Model_def == '500_SH':
            self.mdef = ccl.halos.massdef.MassDef(500, 'critical')

        else:

            raise ValueError("Input Model_def not valid. Select one of: 200_AGN, 500_AGN, 500_SH")

        self.Model_def = Model_def
        self.truncate  = truncate

        #Import all other parameters from the base CCL Profile class
        super(Pressure, self).__init__(mass_def = mass_def)

        #Constant that helps with the fourier transform convolution integral.
        #This value minimized the ringing due to the transforms
        self.update_precision_fftlog(plaw_fourier = -2)

        #Need this to prevent projected profile from artificially cutting off
        self.update_precision_fftlog(padding_lo_fftlog = 1e-4, padding_hi_fftlog = 1e4)

    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1
        mass_def = self.mdef

        #Setup parameters as they were calibrated in Battaglia+ 2012
        if self.Model_def == '200_AGN':

            P_0  = 18.1  * (M_use/1e14)**0.154    * (1 + z)**-0.758
            x_c  = 0.497 * (M_use/1e14)**-0.00865 * (1 + z)**0.731
            beta = 4.35  * (M_use/1e14)**0.0393   * (1 + z)**0.415

        elif self.Model_def == '500_AGN':

            P_0  = 7.49  * (M_use/1e14)**0.226   * (1 + z)**-0.957
            x_c  = 0.710 * (M_use/1e14)**-0.0833 * (1 + z)**0.853
            beta = 4.19  * (M_use/1e14)**0.0480  * (1 + z)**0.615

        elif self.Model_def == '500_SH':

            P_0  = 20.7  * (M_use/1e14)**-0.074 * (1 + z)**-0.743
            x_c  = 0.428 * (M_use/1e14)**0.011  * (1 + z)**1.01
            beta = 3.82  * (M_use/1e14)**0.0375 * (1 + z)**0.535


        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        x = r_use[None, :]/R[:, None]

        #The overdensity constrast related to the mass definition
        Delta    = mass_def.get_Delta(cosmo, a)

        #Cosmological parameters
        Omega_m  = cosmo.cosmo.params.Omega_m
        Omega_b  = cosmo.cosmo.params.Omega_b
        Omega_g  = cosmo.cosmo.params.Omega_g
        h        = cosmo.cosmo.params.h

        #We start with critical density in physical coordinates, in Msun/Mpc^3
        #Then, we scale it to the right redshift using H(z)/H_0 factor.
        #Finally, we convert it to comoving coordinates using the a^3 factor.        
        RHO_CRIT = ccl.physical_constants.RHO_CRITICAL*h**2 * ccl.background.h_over_h0(cosmo, a)**2 
        RHO_CRIT = RHO_CRIT * a**3

        # The self-similar expectation for Pressure
        # Need R*a to convert comoving Mpc to physical
        P_delta = Delta*RHO_CRIT * Omega_b/Omega_m * G * (M_use)/(2*R*a)
        alpha, gamma = 1, -0.3

        P_delta, P_0, beta, x_c = P_delta[:, None], P_0[:, None], beta[:, None], x_c[:, None]
        prof = P_delta * P_0 * (x/x_c)**gamma * (1 + (x/x_c)**alpha)**-beta
        
        #Convert to CGS
        prof = prof * (Msun_to_Kg * 1e3) / (Mpc_to_m * 1e2)

        # Battaglia profile has validity limits for redshift, mass, and distance from halo center.
        # Here, we enforce the distance limit at R/R_Delta > X, where X is input by user
        if self.truncate:
            prof[x > self.truncate] = 0
            
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        
        return prof
    
    
class ElectronPressure(Pressure):
    """
    Computes the electron pressure profile based on the Battaglia et al. (2012) model.

    This class extends `BattagliaPressure` by scaling the gas pressure profile 
    to electron pressure using a predefined conversion factor.

    Inherits from
    -------------
    BattagliaPressure : Base class for computing thermal pressure profiles.

    Notes
    -----
    The electron pressure is calculated as a scaled version of the thermal pressure 
    using:

    .. math::

        P_{\\text{e}}(r) = P_{\\text{gas-to-e}} \\times P_{\\text{gas}}(r)

    where \( P_{\\text{gas-to-e}} \) is a constant conversion factor.

    Methods
    -------
    _real(cosmo, r, M, a)
        Computes the electron pressure profile using the scaled gas pressure.
    """
    
    def _real(self, cosmo, r, M, a):
        
        prof = Pth_to_Pe * super()._real(cosmo, r, M, a)
        
        return prof
    
    
class GasDensity(ccl.halos.profiles.HaloProfile):
    """
    Computes the gas density profile based on the Battaglia et al. (2012) model.

    This class implements the gas density profile using the Battaglia model, 
    allowing for different calibrations and optional truncation at specified radii.
    The mass-definition is forced to be 200c.

    Inherits from
    -------------
    ccl.halos.profiles.HaloProfile : Base class for halo profiles in CCL.

    Parameters
    ----------
    Model_def : str
        Specifies the calibration model to use ('200_AGN', '200_SH'). These options 
        determine the parameters based on different feedback scenarios and mass definitions.
    truncate : float, optional
        Radius (in units of \( R / R_{\\text{def}} \)) at which to truncate the profile. 
        Default is `False`, meaning no truncation is applied.

    Notes
    -----
    The gas density profile is calibrated using simulations with and without AGN feedback, 
    depending on the selected `Model_def`.
    """

    def __init__(self, Model_def, truncate = False):

        self.mdef = ccl.halos.massdef.MassDef(200, 'critical')

        self.Model_def = Model_def
        self.truncate  = truncate

        #Import all other parameters from the base CCL Profile class
        super().__init__(mass_def = self.mdef)

        #Constant that helps with the fourier transform convolution integral.
        #This value minimized the ringing due to the transforms
        self.update_precision_fftlog(plaw_fourier = -2)

        #Need this to prevent projected profile from artificially cutting off
        self.update_precision_fftlog(padding_lo_fftlog = 1e-4, padding_hi_fftlog = 1e4)


    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1
        mass_def = self.mdef

        #These two are fixed parameters
        x_c   = 0.5
        gamma = -0.2

        #Setup parameters as they were calibrated in Battaglia+ 2012
        if self.Model_def == '200_AGN':

            rho_0  = 4e3   * (M_use/1e14)**0.29   * (1 + z)**-0.66
            alpha  = 0.88  * (M_use/1e14)**-0.03  * (1 + z)**0.19
            beta   = 3.83  * (M_use/1e14)**0.04   * (1 + z)**-0.025

        elif self.Model_def == '200_SH':

            rho_0  = 1.9e4 * (M_use/1e14)**0.09   * (1 + z)**-0.95
            alpha  = 0.70 * (M_use/1e14)**-0.017  * (1 + z)**0.27
            beta   = 4.43 * (M_use/1e14)**0.005   * (1 + z)**0.037


        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        x = r_use[None, :]/R[:, None]

        #Cosmological parameters
        h        = cosmo.cosmo.params.h
        Omega_m  = cosmo.cosmo.params.Omega_m
        Omega_b  = cosmo.cosmo.params.Omega_b
        fb       = Omega_b/Omega_m
        
        #We start with critical density in physical coordinates, in Msun/Mpc^3
        #Then, we scale it to the right redshift using H(z)/H_0 factor.
        #Finally, we convert it to comoving coordinates using the a^3 factor.        
        RHO_CRIT = ccl.physical_constants.RHO_CRITICAL*h**2 * ccl.background.h_over_h0(cosmo, a)**2 
        RHO_CRIT = RHO_CRIT * a**3

        rho_0, alpha, beta = rho_0[:, None], alpha[:, None], beta[:, None]
        prof = RHO_CRIT * fb * rho_0 * (x/x_c)**gamma * (1 + (x/x_c)**alpha)**-((beta - gamma)/alpha)

        # Battaglia profile has validity limits for redshift, mass, and distance from halo center.
        # Here, we enforce the distance limit at R/R_Delta > X, where X is input by user
        if self.truncate:
            prof[x > self.truncate] = 0
            
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        
        return prof