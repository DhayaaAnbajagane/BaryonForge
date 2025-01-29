
import numpy as np
import pyccl as ccl
from scipy import interpolate, integrate

from ..Profiles.Schneider19 import model_params, SchneiderProfiles, Gas, DarkMatterBaryon, TwoHalo


#Define relevant physical constants
Msun_to_Kg = ccl.physical_constants.SOLAR_MASS
Mpc_to_m   = ccl.physical_constants.MPC_TO_METER
G          = ccl.physical_constants.GNEWT / Mpc_to_m**3 * Msun_to_Kg
m_to_cm    = 1e2
kb_cgs     = ccl.physical_constants.KBOLTZ * 1e7 
K_to_kev   = ccl.physical_constants.KBOLTZ / ccl.physical_constants.EV_IN_J * 1e-3

#Just define some useful conversions/constants
sigma_T = 6.652458e-29 / Mpc_to_m**2
m_e     = 9.10938e-31 / Msun_to_Kg
m_p     = 1.67262e-27 / Msun_to_Kg
c       = 2.99792458e8 / Mpc_to_m

#CGS units of everything, to use in thermalSZ
sigma_T_cgs = 6.652458e-29 * m_to_cm**2 #m^2 -> cm^2
m_e_cgs     = 9.10938e-31 * 1e3 #Kg -> g
m_p_cgs     = 1.67262e-27 * 1e3 #Kg -> g
c_cgs       = 2.99792458e8 * m_to_cm #m/s -> cm/s


#Thermodynamic/abundance quantities
Y         = 0.24 #Helium mass ratio
Pth_to_Pe = (4 - 2*Y)/(8 - 5*Y) #Factor to convert gas temp. to electron temp


#Technically P(r -> infty) is zero, but we  may need finite
#value for numerical reasons (interpolator). This is a
#computatational constant.
Pressure_at_infinity = 1e-200


__all__ = ['Pressure', 'NonThermalFrac', 'NonThermalFracGreen20',
           'Temperature', 'ThermalSZ', 'ElectronPressure', 'GasNumberDensity']


class BaseThermodynamicProfile(SchneiderProfiles):

    def __init__(self, mass_def = ccl.halos.massdef.MassDef200c, 
                 use_fftlog_projection = False, 
                 padding_lo_proj = 0.1, padding_hi_proj = 10, n_per_decade_proj = 10,
                 r_min_int = 1e-6, r_max_int = 1e3, r_steps = 500,
                 **kwargs):
        
        #Go through all input params, and assign Nones to ones that don't exist.
        #If mass/redshift/conc-dependence, then set to 1 if don't exist
        for m in self.model_param_names:
            if m in kwargs.keys():
                setattr(self, m, kwargs[m])
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
        

class Pressure(BaseThermodynamicProfile):
    """
    Class for computing the gas pressure profile in halos.

    This class extends `SchneiderProfiles` to compute the gas pressure profile within halos 
    under the assumption of hydrostatic equilibrium. The gas pressure is derived using a 
    total mass profile and a gas density profile. We define a pressure gradient from the
    assumption of hydrostatic equilibrium, and integrate to obtain the pressure.

    This gives only the *total gas pressure*. If you want the electron pressure
    see `ElectronPressure`, and if you want to only the thermal/non-thermal
    pressure see `NonThermalFrac`.

    Inherits from
    -------------
    SchneiderProfiles : Base class for halo profiles.

    Parameters
    ----------
    gas : Gas, optional
        An instance of the `Gas` class defining the gas density profile. If not provided, 
        a default `Gas` object is created using `kwargs`.
    darkmatterbaryon : DarkMatterBaryon, optional
        An instance of the `DarkMatterBaryon` class defining the combined dark matter 
        and baryonic mass profile. If not provided, a default `DarkMatterBaryon` object 
        is created using `kwargs`.
    nonthermal_model : object, optional
        An instance defining a model for non-thermal pressure contributions. Default is None.
    **kwargs
        Additional keyword arguments passed to initialize the `Gas`, `DarkMatterBaryon`, 
        and other parameters from `SchneiderProfiles`.

    Notes
    -----
    - This class calculates the pressure assuming hydrostatic equilibrium, which gives:

      .. math::

          \\frac{dP}{dr} = -\\frac{GM(<r)\\rho_{\\text{gas}}(r)}{r^2}

      where:
        - \( G \) is the gravitational constant.
        - \( M(<r) \) is the cumulative mass within radius \( r \).
        - \( \\rho_{\\text{gas}}(r) \) is the gas density profile.

    - The gas pressure \( P(r) \) is then obtained by integrating \( dP/dr \):

      .. math::

          P(r) = \\int_r^{\\infty} -\\frac{GM(<r')\\rho_{\\text{gas}}(r')}{r'^2} r' d\\ln r'

    - The integration is performed numerically, and the result is converted to CGS units 
      for practical applications.
    - An exponential cutoff is applied to the profile to prevent numerical overflow at large radii.

    Methods
    -------
    _real(cosmo, r, M, a)
        Computes the gas pressure profile based on the given cosmology, radii, mass, 
        scale factor, and mass definition.
    """
    
    def __init__(self, gas = None, darkmatterbaryon = None, **kwargs):
        
        self.Gas = gas
        self.DarkMatterBaryon = darkmatterbaryon
        
        #The subtraction in DMB case is so we only have the 1halo term
        if self.Gas is None: self.Gas = Gas(**kwargs)
        if self.DarkMatterBaryon is None: self.DarkMatterBaryon = DarkMatterBaryon(**kwargs) - TwoHalo(**kwargs)
            
        #Now make sure the cutoff is sufficiently high
        #We don't want small cutoff of 1-halo term when computing the TRUE pressure profile.
        #The cutoff is reapplied to the derives pressure profiles in _real()
        self.Gas.set_parameter('cutoff', 1000)
        self.DarkMatterBaryon.set_parameter('cutoff', 1000)
            
        super().__init__(**kwargs)
        
    
    def _real(self, cosmo, r, M, a):
        
        """
        Computes the gas pressure profile using hydrostatic equilibrium.

        This method calculates the gas pressure profile for a specified cosmology, radii, 
        halo mass, and scale factor. The pressure is computed by integrating the pressure 
        gradient derived from the hydrostatic equilibrium condition.

        Parameters
        ----------
        cosmo : object
            A CCL cosmology instance containing the cosmological parameters used for calculations.
        r : array_like
            Radii at which to evaluate the pressure profile, in comoving Mpc.
        M : float or array_like
            Halo mass or array of halo masses, in solar masses.
        a : float
            Scale factor, related to redshift by `a = 1 / (1 + z)`.

        Returns
        -------
        prof : ndarray
            Pressure profile corresponding to the input radii and halo mass, in CGS units.

        Notes
        -----
        - The pressure gradient is calculated using the formula:

        .. math::

            \\frac{dP}{dr} = -\\frac{GM(<r)\\rho_{\\text{gas}}(r)}{r^2}

        where:
            - \( G \) is the gravitational constant.
            - \( M(<r) \) is the cumulative mass within radius \( r \), calculated from the 
            total density profile.
            - \( \\rho_{\\text{gas}}(r) \) is the gas density profile.

        - The pressure profile \( P(r) \) is obtained by numerically integrating the pressure gradient:

        .. math::

            P(r) = \\int_r^{\\infty} -\\frac{GM(<r')\\rho_{\\text{gas}}(r')}{r'^2} r' d\\ln r'

        - The integration is performed from large radii towards smaller radii to satisfy the boundary 
        condition \( P(r \\to \\infty) = 0 \). 
        - The profile is converted to CGS units using appropriate conversion factors.

        Examples
        --------
        Compute the gas pressure profile for a given cosmology and halo:

        >>> pressure_model = Pressure(gas=gas_profile, darkmatterbaryon=dmb_profile, cosmo=my_cosmology)
        >>> r = np.logspace(-2, 1, 50)  # Radii in comoving Mpc
        >>> M = 1e14  # Halo mass in solar masses
        >>> a = 0.5  # Scale factor corresponding to redshift z
        >>> pressure_profile = pressure_model._real(my_cosmology, r, M, a)
        """
            
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1
        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        r_integral = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)
        rho_total  = self.DarkMatterBaryon.real(cosmo, r_integral, M, a)
        rho_gas    = self.Gas.real(cosmo, r_integral, M, a)

        #The real() routine squeezes out dimension if len(M) = 1
        #Add it back so the integration below works out
        if M_use.size == 1: 
            rho_total = rho_total[None, :]
        
        #Integrate total density profile to get the cumulative mass distribution
        dlnr    = np.log(r_integral[1]) - np.log(r_integral[0])
        dV      = r_integral**3 * dlnr
        M_total = 4 * np.pi * integrate.cumulative_simpson(dV * rho_total, axis = -1, initial = dV[0] * rho_total[:, [0]])

        #Assuming hydrostatic equilibrium to get dP/dr = -G*M(<r)*rho(r)/r^2
        dP_dr = - G * M_total * rho_gas / r_integral**2
        
        #Make it have the right shape that ccl expects (size(M), size(r)) 
        if len(dP_dr.shape) < 2:
            dP_dr = dP_dr[np.newaxis, :]

        #integrate to get actual pressure, P(r). Boundary condition is P(r -> infty) = 0.
        #So we start from the boundary and integrate inwards. We reverse array once to
        #flip the integral direction, and flip it second time so P(r) goes from r = 0 to r = infty
        #We use trapezoid rule here because simpson was causing odd oscillatory errors because
        #Some profiles have sharp transitions in their pressure/gas profiles (eg. Mead)
        intgr = (dP_dr * r_integral)[:, ::-1] * dlnr
        prof  = -np.array([integrate.cumulative_trapezoid(intgr[i], initial = intgr[i, 0])[::-1] for i in range(intgr.shape[0])])
        
        prof  = interpolate.PchipInterpolator(np.log(r_integral), np.log(prof + Pressure_at_infinity), axis = 1, extrapolate = False)
        prof  = np.exp(prof(np.log(r_use))) - Pressure_at_infinity
        prof  = np.where(np.isfinite(prof), prof, 0) #Get rid of pesky NaN and inf values if they exist! They break CCL spline interpolator
        
        #Convert to CGS. Using only one factor of Mpc_to_m is correct here!
        prof  = prof * (Msun_to_Kg * 1e3) / (Mpc_to_m * 1e2)
        
        #Now do cutoff
        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = prof * kfac
        

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof
    


class NonThermalFrac(BaseThermodynamicProfile):
    
    """
    Class for computing the non-thermal pressure fraction profile in halos.

    This class extends `SchneiderProfiles` to compute the fraction of pressure that is 
    non-thermal within halos. The non-thermal fraction is modelled in a redshift, and radius
    dependent manner, following Equations 15/16 in `Pandey et. al 2025 <https://arxiv.org/pdf/2401.18072>`_. 
    This can model be applied to any profiles using simple multiplication of the initialized classes,

    `ThermalPressure = Pressure(**kwargs) * (1 - NonThermalFrac(**kwargs))`

    The `real()` method of `ThermalPressure` will then account for non-thermal pressure
    effects as well.

    Inherits from
    -------------
    SchneiderProfiles : Base class for halo profiles.

    Parameters
    ----------
    alpha_nt : float
        Normalization factor for the non-thermal pressure fraction.
    nu_nt : float
        Parameter controlling the redshift dependence of the non-thermal fraction.
    gamma_nt : float
        Parameter controlling the radial dependence of the non-thermal fraction.
    **kwargs
        Additional keyword arguments passed to initialize other parameters from `SchneiderProfiles`.

    Notes
    -----
    The `NonThermalFrac` class is used to compute the non-thermal pressure fraction, 
    which represents the fraction of total pressure that is non-thermal (e.g., due to 
    turbulence or magnetic fields) as a function of radius and redshift. This fraction 
    is used to modify the total pressure profile, accounting for contributions that are 
    not purely thermal.

    The non-thermal pressure fraction \( f_{\\text{nt}}(r, z) \) is calculated using:

    .. math::

        f_{\\text{nt}}(r, z) = \\alpha_{\\text{nt}} \\times f_z \\times \\left( \\frac{r}{R} \\right)^{\\gamma_{\\text{nt}}}

    where:
        - \( \\alpha_{\\text{nt}} \) is the normalization factor.
        - \( f_z \) is the redshift-dependent factor, defined as:

          .. math::

              f_z = \\min\\left[(1 + z)^{\\nu_{\\text{nt}}}, \\left(f_{\\text{max}} - 1\\right) \\tanh\\left(\\nu_{\\text{nt}} z\\right) + 1\\right]

        - \( R \) is the halo radius based on the mass definition.
        - \( \\gamma_{\\text{nt}} \) controls the radial dependence.
    """

    def __init__(self, alpha_nt, nu_nt, gamma_nt, **kwargs):
        
        super().__init__(**kwargs)
        
        self.alpha_nt = alpha_nt
        self.nu_nt    = nu_nt
        self.gamma_nt = gamma_nt
    
    
    def _real(self, cosmo, r, M, a):
        
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_max = 6**-self.gamma_nt/self.alpha_nt
        f_z   = np.min([(1 + z)**self.nu_nt, (f_max - 1)*np.tanh(self.nu_nt * z) + 1])
        f_nt  = self.alpha_nt * f_z * (r_use/R[:, None])**self.gamma_nt
        f_nt  = np.clip(f_nt, 0, 1) #Enforce 0 < f_nt < 1
        prof  = f_nt #Rename just for consistency sake
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof
    
    

class NonThermalFracGreen20(BaseThermodynamicProfile):
    
    """
    Class for computing the non-thermal pressure fraction profile using the Green et al. (2020) model.

    
    Notes
    -----
    The model is based on parameters calibrated to simulations and is specifically defined 
    with respect to \( R_{200m} \), the radius within which the mean density is 200 times 
    the mean matter density of the universe.

    The non-thermal pressure fraction \( f_{\\text{nt}}(r) \) is calculated using:

    .. math::

        f_{\\text{nt}}(r) = 1 - a \\left(1 + \\exp\\left(-\\left(\\frac{x}{b}\\right)^c\\right)\\right) 
                            \\left(\\frac{\\nu_M}{4.1}\\right)^{\\frac{d}{1 + \\left(\\frac{x}{e}\\right)^f}}

    where:
        - \( x = \\frac{r}{R_{200m}} \)
        - \( \\nu_M = \\frac{1.686}{\\sigma(M_{200m})} \) is the peak height parameter.
        - \( a, b, c, d, e, f \) are model parameters calibrated to fit simulation data.

    There are no free parameters in this model; it is completely specified by the halo mass and redshift.
    """

    def _real(self, cosmo, r, M, a):
        
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        
        #They define the model with R200m, so gotta use that redefinition here.
        mdef  = ccl.halos.massdef.MassDef(200, 'matter')
        cnvrt = ccl.halos.mass_translator(mass_in = self.mass_def, mass_out = mdef, concentration = self.mass_def.concentration)
        M200m = cnvrt(cosmo, M_use, a)
        R200m = mdef.get_radius(cosmo, M_use, a)/a #in comoving distance

        x = r_use/R200m[:, None]

        nu_M = 1.686/ccl.sigmaM(cosmo, M200m, a)
        nu_M = nu_M[:, None]
        
        A, b, c, d, e, f = 0.495, 0.719, 1.417,-0.166, 0.265, -2.116
        nth  = 1 - A * (1 + np.exp(-(x/b)**c)) * (nu_M/4.1)**(d/(1 + (x/e)**f))
        nth  = np.clip(nth, 0, 1)
        prof = nth #Rename just for consistency sake
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof

    

class ElectronPressure(Pressure):
    """
    Class for computing the electron pressure profile in halos.

    This class extends the `Pressure` class to compute the electron pressure 
    profile from the total gas pressure. The conversion factor is 
    \( P_{\\text{e}} = P_{\\text{th}} \\times P_{\\text{th-to-Pe}} \), where 
    \( P_{\\text{th-to-Pe}} = (4 - 2Y)/(8 - 5Y), with Y = 0.24\).


    Inherits from
    -------------
    Pressure : Base class for computing gas pressure profiles in halos.

    Notes
    -----
    The `ElectronPressure` class is used to compute the electron pressure profile 
    within halos, which is relevant for understanding the thermal Sunyaev-Zel'dovich 
    (tSZ) effect. The conversion is done assuming pressure equilibrium between electrons
    and protons in a system dominated by hydrogen and helium species (hence the use of Y).

    """
    def _real(self, cosmo, r, M, a):
        
        prof = Pth_to_Pe * super()._real(cosmo, r, M, a)
        
        return prof


class GasNumberDensity(BaseThermodynamicProfile):
    """
    Class for computing the gas number density profile in halos.

    This class extends `SchneiderProfiles` to compute the gas number density profile 
    within halos. The number density is derived from the gas density profile by dividing 
    by the mean molecular weight and the mass of the proton, and then converting to 
    proper CGS units.

    Inherits from
    -------------
    SchneiderProfiles : Base class for halo profiles.

    Parameters
    ----------
    gas : Gas, optional
        An instance of the `Gas` class defining the gas density profile. If not provided, 
        a default `Gas` object is created using `kwargs`.
    mean_molecular_weight : float, optional
        Mean molecular weight of the gas. Default is 1.15, which is typical for ionized 
        hydrogen with a small fraction of helium.
    **kwargs
        Additional keyword arguments passed to initialize the `Gas` profile and other 
        parameters from `SchneiderProfiles`.

    Notes
    -----
    The `GasNumberDensity` class is used to compute the number density of gas particles 
    in halos, which is relevant for understanding the baryonic content of halos and for 
    modeling various astrophysical processes, such as cooling and star formation.

    The gas number density \( n_{\\text{gas}} \) is calculated by dividing the gas density 
    profile \( \\rho_{\\text{gas}} \) by the mean molecular weight and the mass of the proton:

    .. math::

        n_{\\text{gas}}(r) = \\frac{\\rho_{\\text{gas}}(r)}{\\mu \\cdot m_p}

    where:
        - \( \\mu \) is the mean molecular weight of the gas.
        - \( m_p \) is the mass of the proton.
        - The result is converted to the proper units (number per cubic centimeter).
    """
    
    def __init__(self, gas = None, mean_molecular_weight = 1.15, **kwargs):
        
        self.Gas = gas
        if self.Gas is None: self.Gas = Gas(**kwargs)
        
        super().__init__(**kwargs)
        
        self.mean_molecular_weight = mean_molecular_weight
        
        
    
    def _real(self, cosmo, r, M, a):
        
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        rho  = self.Gas
        rho  = rho.real(cosmo, r_use, M, a)
        prof = rho / (self.mean_molecular_weight * m_p) / (Mpc_to_m * m_to_cm)**3
        
        return prof
    
    
class Temperature(BaseThermodynamicProfile):
    """
    Class for computing the temperature profile in halos.

    The temperature is derived from the thermal pressure and the number density profiles, 
    of a species using the ideal gas law. The temperature profile is important for understanding 
    the thermal state of the intracluster medium and its impact on various astrophysical processes.

    For this model to be correct, the input pressure must be the *thermal pressure*, i.e. the
    non-thermal pressure must have already been accounted for in the model passed to this class.


    Parameters
    ----------
    pressure : Pressure, optional
        An instance of the `Pressure` class defining the thermal gas pressure profile. 
        If non-thermal pressure is relevant for your problem, it must be included in this
        profile; see `Pressure` or `NonThermalFrac` for more details.
        If this parameter is not provided, a default `Pressure` object is created using `kwargs`.
    gasnumberdensity : GasNumberDensity, optional
        An instance of the `GasNumberDensity` class defining the gas number density profile. 
        If not provided, a default `GasNumberDensity` object is created using `kwargs`.
    **kwargs
        Additional keyword arguments passed to initialize the `Pressure`, `GasNumberDensity`, 
        and other parameters from `SchneiderProfiles`.

    Notes
    -----
    The `Temperature` class computes the temperature profile of the gas in halos by dividing 
    the gas pressure by the gas number density and the Boltzmann constant. This calculation 
    assumes the ideal gas law, which relates pressure, number density, and temperature.

    The gas temperature \( T \) is calculated using:

    .. math::

        T(r) = \\frac{P}(r)}{n(r) \\cdot k_B}

    where:
        - \( P(r) \) is the Thermal pressure profile of a species.
        - \( n(r) \) is the number density profile of a species.
        - \( k_B \) is the Boltzmann constant (in eV).
    """
    
    def __init__(self, pressure = None, gasnumberdensity = None, **kwargs):
        
        self.Pressure = pressure
        self.GasNumberDensity = gasnumberdensity
        
        if self.Pressure is None: self.Pressure = Pressure(**kwargs) * (1 - NonThermalFrac(**kwargs))
        if self.GasNumberDensity is None: self.GasNumberDensity = GasNumberDensity(**kwargs)
            
        super().__init__(**kwargs)
        
    
    def _real(self, cosmo, r, M, a):
        
        P   = self.Pressure.real(cosmo, r, M, a)
        n   = self.GasNumberDensity.real(cosmo, r, M, a)
        
        #We'll have instances of n == 0, which isn't a problem so let's ignore
        #warnings of divide errors, because we know they happen here.
        #Instead we will fix them by replacing the temperature with 0s,
        #since there is no gas in those regions to use anyway.
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            prof = P/(n * kb_cgs)
            prof = np.where(n == 0, 0, prof)
        
        assert np.all(prof >= 0), "Something went wrong. Temperature is negative in some places"

        return prof
    
    
    def projected(self, cosmo, r, M, a):
        """
        Compute the projected temperature profile along the line of sight.

        This method calculates the "average temperature" along the line of sight, 
        which is a physically meaningful quantity for comparing with observations 
        such as X-ray or Sunyaev-Zel'dovich measurements. It differs from the 
        "integrated temperature," which lacks physical relevance in most astrophysical contexts.

        Parameters
        ----------
        cosmo : Cosmology
            The cosmology object containing cosmological parameters.
        r : array_like
            The projected radial distances at which to compute the profile, in units of Mpc/h.
        M : float
            The halo mass, in units of solar masses.
        a : float
            The scale factor of the Universe.

        Returns
        -------
        prof : array_like
            The projected average temperature profile, in units of eV.

        Notes
        -----
        The projected temperature is computed using the ideal gas law:

        .. math::

            T_{\text{proj}}(r) = \\frac{P_{\text{proj}}(r)}{n_{\text{proj}}(r) \\cdot k_B}

        where:
            - \( P_{\text{proj}}(r) \) is the projected thermal pressure profile.
            - \( n_{\text{proj}}(r) \) is the projected number density profile.
            - \( k_B \) is the Boltzmann constant (in eV).

        Regions with zero gas density (\( n_{\text{proj}}(r) = 0 \)) are assigned a temperature of 0 
        to avoid division errors, as these regions lack gas to support a meaningful temperature.
        """

        P   = self.Pressure.projected(cosmo, r, M, a)
        n   = self.GasNumberDensity.projected(cosmo, r, M, a)

        #We'll have instances of n == 0, which isn't a problem so let's ignore
        #warnings of divide errors, because we know they happen here.
        #Instead we will fix them by replacing the temperature with 0s,
        #since there is no gas in those regions to use anyway.
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            prof = P/(n * kb_cgs)
            prof = np.where(n == 0, 0, prof)

        return prof
    
    

class ThermalSZ(BaseThermodynamicProfile):
    """
    Class for computing the thermal Sunyaev-Zel'dovich (tSZ) effect profile in halos.

    This class extends `SchneiderProfiles` to compute the tSZ effect, which is caused 
    by the inverse Compton scattering of cosmic microwave background (CMB) photons 
    off hot electrons in the intracluster medium of galaxy clusters. The tSZ effect 
    is represented by the Compton-y parameter, which is proportional to the line-of-sight 
    integral of the electron pressure.

    In practice, this scale uses the `projected` method of the input `pressure` object.
    It accounts for the right units, to provide a dimensionless compton-y parameter.

    Inherits from
    -------------
    SchneiderProfiles : Base class for halo profiles.

    Parameters
    ----------
    pressure : Pressure, optional
        An instance of the `Pressure` class defining the thermal gas pressure profile. 
        If not provided, a default `Pressure` object is created using `kwargs`.
    **kwargs
        Additional keyword arguments passed to initialize the `Pressure` profile and other 
        parameters from `SchneiderProfiles`.

    Notes
    -----
    The `ThermalSZ` class computes the tSZ effect by calculating the projected electron 
    pressure profile along the line of sight. 

    - The tSZ effect is computed by projecting the electron pressure along the line of sight:

    .. math::

        y(r) = \\frac{\\sigma_T}{m_e c^2} \\int P_{\\text{e}}(r') \\, dr'

    where \( P_{\\text{e}}(r') \) is the electron pressure profile.
    
    Methods
    -------
    Pgas_to_Pe(cosmo, r, M, a)
        Returns the conversion factor from gas pressure to electron pressure.
    
    projected(cosmo, r, M, a)
        Computes the projected tSZ profile (Compton-y parameter) based on the given 
        cosmology, radii, mass, scale factor, and mass definition.
    
    real(cosmo, r, M, a)
        This is not to be used, as SZ is a projected quantity. However, the method
        still returns a sentinel value of -99, needed for consistency in other parts
         of the pipeline (eg. Tabulation, see `TabulatedProfile`).
    """
    
    
    def __init__(self, pressure = None, **kwargs):
        
        self.pressure = pressure
        if self.pressure is None: self.pressure = Pressure(**kwargs)

        super().__init__(**kwargs)
        
    
    def Pgas_to_Pe(self, cosmo, r, M, a):
        
        """
        Returns the precomputed conversion factor from gas pressure to electron pressure.
        Can be redefined by user if they wish to use a different (mass-dependent) value
        for this quantity.
        """
            
        return Pth_to_Pe
    
    
    def projected(self, cosmo, r, M, a):        

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z     = 1/a - 1
        R     = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        #Now a series of units changes to the projected profile.
        prof  = self.pressure.projected(cosmo, r_use, M_use, a) #generate profile
        prof  = prof * a * (Mpc_to_m * 1e2) #Line-of-sight integral is done in comoving Mpc, we want physical cm
        prof  = prof * sigma_T_cgs/(m_e_cgs*c_cgs**2) #Convert to SZ (dimensionless units)
        prof  = prof * self.Pgas_to_Pe(cosmo, r_use, M_use, a) #Then convert from gas pressure to electron pressure
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)
        
        return prof
    
    
    def real(self, cosmo, r, M, a):
        
        #Don't raise ValueError because then we can't pass this object in a TabulatedProfile class
        #Instead just output sentinel value of -99

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        shape = (M_use.size, r_use.size)
        prof  = np.ones(shape) * -99
        
        return prof
    

    #Have dummy methods because CCL asserts that these must exist.
    #Hacky because I want to keep SchneiderProfiles as base class
    #in order to get __init__ to be simple, but then we have to follow
    #the CCL HaloProfile base class API. 
    def _real(self): return np.nan
    def _projected(self): return np.nan
    
    
    
class XrayLuminosity(BaseThermodynamicProfile):
    
    
    def __init__(self, temperature = None, gasnumberdensity = None, **kwargs):

        raise NotImplementedError("Dhayaa: I have not yet worked on this profile properly. "
                                  "So don't use this yet! It's missing cooling factor calibrations.")

        self.Temperature      = temperature
        self.GasNumberDensity = gasnumberdensity
        
        if self.Temperature is None: self.Temperature = Temperature(**kwargs)
        if self.GasNumberDensity is None: self.GasNumberDensity = GasNumberDensity(**kwargs)
        
        super().__init__()
        
    
    def _real(self, cosmo, r, M, a):
        
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        T   = self.Temperature.real(cosmo, r_use, M, a)
        n   = self.GasNumberDensity.real(cosmo, r_use, M, a)
        
        prof = n**2*T
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)
        
        return prof