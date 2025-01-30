import numpy as np
import pyccl as ccl
from operator import add, mul, sub, truediv, pow, neg, pos, abs
import warnings

from scipy import interpolate, special
from ..utils import _set_parameter, safe_Pchip_minimize
from .misc import Zeros
from . import Schneider19 as S19, Arico20 as A20
from .Thermodynamic import (G, Msun_to_Kg, Mpc_to_m, kb_cgs, m_p, m_to_cm)

__all__ = ['model_params', 'MeadProfiles', 
           'DarkMatter', 'TwoHalo', 'Stars', 'Gas', 'BoundGas', 'EjectedGas', 'ReaccretedGas', 'CollisionlessMatter',
           'DarkMatterOnly', 'DarkMatterBaryon']

model_params = ['cdelta', 'eps1', 'nu_eps1', 'eps2', #DM profle param and relaxation params
                'cutoff', 'proj_cutoff', #Cutoff parameters (numerical)
                'p', 'q',  #Two halo terms
                
                'M_0', 'beta', 'Gamma', 'nu_Gamma', 'eta_b', #Default gas profile param

                'A_star', 'nu_A_star', 'M_star', 'nu_M_star', 'sigma_star', 'epsilon_h', 'eta', #Star params

                'T_w', 'nu_T_w', #Temperature params
                'mean_molecular_weight', #Gas number density params
                'alpha', #Pressure params
               ]


class MeadProfiles(A20.AricoProfiles):
    __doc__ = A20.AricoProfiles.__doc__.replace('Arico', 'Mead')

    #Define the new param names
    model_param_names = model_params


    def _get_star_frac(self, M_use, a, cosmo):
        """
        Compute the stellar fraction and its components for given halo masses and scale factor.

        This method calculates the total stellar fraction (\( f_{\star} \)), 
        as well as its central (\( f_{\\text{cen}} \)) and satellite (\( f_{\\text{sat}} \)) 
        components, based on a parametric model that evolves with redshift.

        Parameters
        ----------
        M_use : array_like
            Halo masses, in units of solar masses.
        a : float
            Scale factor of the Universe, where \( a = 1 \) corresponds to the present day.

        Returns
        -------
        f_str : array_like
            Total stellar fraction for each input halo mass.
        f_cen : array_like
            Central stellar fraction for each input halo mass.
        f_sat : array_like
            Satellite stellar fraction for each input halo mass.

        Notes
        -----
        The stellar fraction is modeled using a Gaussian function centered on a characteristic 
        mass (\( M_{\star} \)) with redshift evolution. The model also includes separate 
        prescriptions for the central and satellite stellar fractions based on the relationship 
        between the halo mass and the characteristic mass.

        The total stellar fraction (\( f_{\star} \)) is given by:

        .. math::

            f_{\\star} = A_{\\star} \exp \\left( - \\frac{ \\left( \\log_{10} M_{\\text{use}} - \\log_{10} M_{\star} \\right)^2 }{ 2 \\sigma_{\star}^2 } \\right)

        For \( M_{\\text{use}} > M_{\star} \), \( f_{\star} \) is limited to a fraction of \( A_{\star} \).

        The central and satellite components are calculated as:

        .. math::

            f_{\\text{cen}} = f_{\star} \\cdot \\begin{cases} 
                1 & M_{\\text{use}} < M_{\star} \\\\ 
                \\left( \\frac{M_{\\text{use}}}{M_{\star}} \\right)^{\\eta} & M_{\\text{use}} > M_{\star}
            \\end{cases}

            f_{\\text{sat}} = f_{\star} \\cdot \\begin{cases} 
                0 & M_{\\text{use}} < M_{\star} \\\\ 
                1 - \\left( \\frac{M_{\\text{use}}}{M_{\star}} \\right)^{\\eta} & M_{\\text{use}} > M_{\star}
            \\end{cases}
        """
        
        z     = 1/a - 1
        Astr  = self.A_star + self.nu_A_star * z
        Mstr  = self.M_star * np.exp(z * self.nu_M_star)
        f_str = Astr * np.exp(-np.power(np.log10(M_use/Mstr)/self.sigma_star, 2)/2)
        f_str = np.where(M_use > Mstr, np.max([f_str, Astr/3 * np.ones_like(f_str)], axis = 0), f_str)

        #Compute bound gas fraction to check a summation condition
        f_bar = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        f_bnd = f_bar * np.power(M_use/self.M_0, self.beta) / (1 + np.power(M_use/self.M_0, self.beta))
        f_sum = f_bnd + f_str
        f_str = np.where(f_sum > f_bar, f_str - (f_sum - f_bar), f_str)

        assert np.all(f_str >= 0), "Chosen parameter range (and mass range) produces negative star fraction"

        #Finally, now get the central and satellite masses
        f_cen = f_str * np.where(M_use < Mstr, 1, np.power(M_use/Mstr, self.eta))
        f_sat = f_str * np.where(M_use < Mstr, 0, 1 - np.power(M_use/Mstr, self.eta))
        
        return f_str, f_cen, f_sat    

    def _get_gas_params(self): return self.M0, self.beta

    def _get_gas_frac(self, M_use, a, cosmo):

        f_str, f_cen, f_sat = self._get_star_frac(M_use, a, cosmo)
        f_bar = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        f_bnd = f_bar * np.power(M_use/self.M_0, self.beta) / (1 + np.power(M_use/self.M_0, self.beta))
        f_ej  = ((f_bar - f_str) - f_bnd)
        
        return f_bnd, f_ej


class DarkMatter(MeadProfiles):
    """
    Class for modeling the dark matter density profile using the NFW (Navarro-Frenk-White) framework.

    This class extends `MeadProfiles` to compute the dark matter density profile, incorporating 
    flexible concentration-mass relations and a truncated profile at the spherical overdensity radius.

    Parameters
    ----------
    None (inherits all parameters from `MeadProfiles`).

    Notes
    -----
    The dark matter profile is calculated using the NFW formula, which depends on the halo's 
    mass and concentration. The normalization is determined analytically to ensure that the 
    total mass within the virial radius matches the input halo mass.

    The density profile is given by:

    .. math::

        \\rho(r) = 
        \\begin{cases} 
        \\frac{\\rho_c}{(r/r_s)(1 + r/r_s)^2}, & r \\leq R_{200c} \\\\ 
        0, & r > R_{200c}
        \\end{cases}


    where:
    - \( \\rho_c \) is the characteristic density, computed using the halo mass.
    - \( r_s \) is the scale radius, defined as \( R/c \), where \( R \) is the virial radius and \( c \) 
    is the concentration parameter.
    """

    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        if self.cdelta is None:
            #Use the Duffy08 calibration following Equation 33 in https://arxiv.org/pdf/2005.00009
            #Can in principle swap this around as you want
            c_M_relation = ccl.halos.concentration.ConcentrationDuffy08(mass_def = self.mass_def)
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mass_def = self.mass_def)
            
        #No modification of DMO concentration here
        c   = c_M_relation(cosmo, M_use, a)
        R   = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        r_s = R/c

        #Get the normalization (rho_c) analytically since we don't have a truncation radii like S19 does
        Norm  = 4*np.pi*r_s**3 * (np.log(1 + c) - c/(1 + c))
        rho_c = M_use/Norm

        r_s, c, rho_c = r_s[:, None], c[:, None], rho_c[:, None]
        r_use, R      = r_use[None, :], R[:, None]

        arg  = (r_use - self.cutoff)
        arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof = rho_c/(r_use/r_s * (1 + r_use/r_s)**2) * kfac
        prof = np.where(r_use <= R, prof, 0)
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof
    

class TwoHalo(S19.TwoHalo, MeadProfiles):
    __doc__ = S19.TwoHalo.__doc__.replace('Schneider', 'Mead')


class CentralStars(MeadProfiles):
    """
    Class for modeling the stellar density profile of central galaxy in halos.

    This class extends `MeadProfiles` to compute the stellar density profile of the central
    galaxy. The stellar fraction is a simple Gaussian in halo mass. While Mead20 uses a
    delta function for their star profile, we use a simple exponential, following Arico and Schneider.


    Notes
    -----
    - The stellar profile is computed based on the fraction of stars in the halo (\( f_{\star} \)), 
      divided into central and satellite components.
    - The central component is modeled using a Gaussian distribution centered on the halo center, 
      characterized by the scale radius \( R_h \).
    
    The density profile is given by:

    .. math::

        \\rho_{\star}(r) = \\frac{f_{\\text{cen}} M}{4 \\pi^{3/2} R_h} 
        \\cdot \\frac{1}{r^2} \\cdot \\exp\\left(- \\frac{r^2}{4 R_h^2}\\right)

    where:
    - \( f_{\\text{cen}} \) is the fraction of stars in the central component.
    - \( M \) is the halo mass.
    - \( R_h \) is the characteristic scale radius, proportional to the halo virial radius.
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

        R     = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        f_str, f_cen, f_sat = self._get_star_frac(M_use, a, cosmo)
        f_cen = f_cen[:, None]
        R_h   = self.epsilon_h * R[:, None]

        #Final profile. No truncation needed since exponential cutoff already does that for us
        prof = f_cen*M_use[:, None] / (4*np.pi**(3/2)*R_h) * 1/r_use**2 * np.exp(-(r_use/2/R_h)**2)

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof
    

class SatelliteStars(DarkMatter):

    """
    Class for modeling the stellar density profile of satellite galaxies in halos.

    This is simply an NFW profile (without baryon-corrected concentrations) with the
    normalization set by the satellite star fraction of the halo.
    """
    
    def _real(self, cosmo, r, M, a):

        f_str, f_cen, f_sat = self._get_star_frac(M, a, cosmo)
        
        if np.atleast_1d(f_sat).size > 1:
            f_sat = f_sat[:, None]

        prof = super()._real(cosmo, r, M, a) * f_sat 

        return prof


class Stars(MeadProfiles):
    """
    Convenience class for combining central and satellite star components.

    This class serves as a unified interface for gas profiles in halos, combining the contributions 
    from the central galaxy (`CentralStars`) and satellite galaxies (`SatelliteStars`). It simplifies calculations where 
    the total gas profile is required, leveraging the underlying logic and methods of the individual 
    star components.
    """

    def __init__(self, **kwargs): self.myprof = CentralStars(**kwargs) + SatelliteStars(**kwargs)
    def __getattr__(self, name):  return getattr(self.myprof, name)
    
    #Need to explicitly set these two methods (to enable pickling)
    #since otherwise the getattr call above leads to infinite recursions.
    def __getstate__(self): self.__dict__.copy()    
    def __setstate__(self, state): self.__dict__.update(state)


class DeltaStars(MeadProfiles):
    """
    The exact model for stars used in Mead2020

    The star profile is modelled as a delta function, delta(r). In fourier space, this
    is a simple constant profile, which is what we implement. Given the nature of the
    profile, it not advised to use the real() routine. If you need a real-space profile,
    considering using the default `Stars` class instead of this `DeltaStars` class.

    Notes
    -----
    
    The density profile is given by:

    .. math::

        \\rho_{\star}(k) = f_{\\text{cen}} M

    where:
    - \( f_{\\text{cen}} \) is the fraction of stars in the central component.
    - \( M \) is the halo mass.
    """

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        #For some reason, we need to make this extreme in order
        #to prevent ringing in the profiles. Haven't figured out
        #why this is the case
        self.update_precision_fftlog(padding_lo_fftlog = 1e-5, padding_hi_fftlog = 1e5)

    
    def _fourier(self, cosmo, k, M, a):

        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)

        f_str, f_cen, f_sat = self._get_star_frac(M_use, a, cosmo)
        f_cen = f_cen[:, None]
        
        #Final profile. No truncation needed since exponential cutoff already does that for us
        prof = f_cen*M_use[:, None] * np.ones_like(k_use)

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(k) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof
    

    #Need this so the __add__ etc. calls work okay (they add the real-space models together)
    def _real(self, cosmo, r, M, a):

        return self._fftlog_wrap(cosmo, r, M, a, fourier_out=False)
    

class BoundGas(MeadProfiles):
    """
    Class for modeling the bound gas density profile in halos.

    This class extends `MeadProfiles` to compute the density profile of bound gas in halos. The bound gas profile accounts 
    depends on the baryon fraction, stellar fraction, and halo properties such as mass and concentration.


    Notes
    -----
    - The bound gas fraction is calculated as the difference between the total baryon fraction and the stellar fraction, 
      scaled by a parametric model that includes mass dependence.
    - The radial density profile is normalized on a per-halo basis to ensure physical consistency, integrating the profile 
      within the virial radius.
    - The profile follows a parametric form, which depends on the concentration parameter and redshift-dependent factors 
      like the effective gamma \( \Gamma \).

    The density profile is given by:

    .. math::

        \\rho_{\\text{gas}}(r) = f_{\\text{bnd}} M 
        \\cdot \\frac{\\left[\\ln(1 + x) / x\\right]^{1 / (\\Gamma - 1)}}{N}

    where:
    - \( f_{\\text{bnd}} \) is the bound gas fraction, determined from the baryon and stellar fractions.
    - \( M \) is the halo mass.
    - \( x = r / r_s \), where \( r_s = R / c \) is the scale radius.
    - \( N \) is the normalization factor ensuring the profile integrates to the bound gas mass within the virial radius.
    - \( \\Gamma \) is a redshift-dependent parameter that modifies the profile shape.
    """

    def _real(self, cosmo, r, M, a):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        if self.cdelta is None:
            #Use the Duffy08 calibration following Equation 33 in https://arxiv.org/pdf/2005.00009
            #Can in principle swap this around as you want
            c_M_relation = ccl.halos.concentration.ConcentrationDuffy08(mass_def = self.mass_def)
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mass_def = self.mass_def)

        z     = 1/a - 1
        c     = c_M_relation(cosmo, M_use, a)
        R     = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        r_s   = R/c
        r_s   = r_s[:, None]
        Geff  = self.Gamma + self.nu_Gamma * z

        f_bnd, f_ej = self._get_gas_frac(M_use, a, cosmo)
        f_bnd       = f_bnd[:, None]
        
        #Do normalization halo-by-halo, since we want custom radial ranges.
        #This way, we can handle sharp transition at R200c without needing
        #super fine resolution in the grid.
        Normalization = np.ones_like(M_use)
        for m_i in range(M_use.shape[0]):
            r_integral    = np.geomspace(self.r_min_int, R[m_i], self.r_steps)
            x_integral    = r_integral/r_s[m_i]
            prof_integral = np.power(np.log(1 + x_integral) / x_integral, 1/(Geff - 1))
            Normalization[m_i] = np.trapz(4 * np.pi * r_integral**2 * prof_integral, r_integral)
        Normalization = Normalization[:, None]

        del prof_integral, x_integral

        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        x_use = r_use / r_s
        prof  = np.power(np.log(1 + x_use) / x_use, 1/(Geff - 1))
        prof  = np.where(r_use[None, :] <= R[:, None], prof, 0)
        prof *= f_bnd*M_use[:, None]/Normalization

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)


        return prof
    

class EjectedGas(MeadProfiles):
    """
    Class for modeling the density profile of ejected gas in halos.

    This class extends `MeadProfiles` to compute the density profile of gas that has been ejected 
    from halos due to feedback processes. The profile accounts for the escape radius, redshift-dependent 
    parameters, and the baryon fraction of the halo. This follows Omori+23 (https://arxiv.org/pdf/2212.07420)
    who use the methods in Schneider & Teyssier 2015. In Mead20, the Ejected Gas is included as an
    addition to the two-halo term, which is not the approach used here.

    Notes
    -----
    - The ejected gas fraction (\( f_{\\text{ej}} \)) is calculated as the difference between the 
      total baryon fraction and the sum of the stellar fraction and bound gas fraction.
    - The profile includes an escape radius (\( R_{\\text{esc}} \)) derived from the halo's escape velocity and 
      cosmological parameters, which limits the spatial extent of the ejected gas.
    - The radial distribution of ejected gas is modeled as a Gaussian profile, normalized by the total ejected mass.

    The density profile is given by:

    .. math::

        \\rho_{\\text{ej}}(r) = \\frac{f_{\\text{ej}} M}{(2\\pi R_{\\text{ej}}^2)^{3/2}} 
        \\cdot \\exp\\left(-\\frac{r^2}{2R_{\\text{ej}}^2}\\right)

    where:
    - \( f_{\\text{ej}} \) is the ejected gas fraction.
    - \( M \) is the halo mass.
    - \( R_{\\text{ej}} \) is the ejection radius, determined according to the ejection fraction and 
      a maxwellian velocity distribution for the gas.
    """

    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1
        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_bar = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        f_bnd, f_ej = self._get_gas_frac(M_use, a, cosmo)
        f_ej        = f_ej[:, None]

        #Now use the escape radius, which is r_esc = v_esc * t_hubble
        #and this reduces down to just 1/2 * sqrt(Delta) * R_Delta
        R_esc = 1/2 * np.sqrt(200) * R[:, None]
        rgrid = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)
        Term1 = 1 - special.erf(self.eta_b * R_esc / np.sqrt(2) / rgrid)
        Term2 = np.sqrt(2/np.pi) * self.eta_b * R_esc / rgrid * np.exp(-np.power(self.eta_b*R_esc/rgrid, 2)/2)
        Diff  = Term1 + Term2 - 1/f_bar * f_ej

        R_ej  = np.zeros([Diff.shape[0], 1])
        for i in range(R_ej.size):

            #If ejected fraction is 0 then we don not need to set a ejected radii. Set it to inf.
            if f_ej[i] > 0:
                R_ej[i] = np.exp(safe_Pchip_minimize(Diff[i], np.log(rgrid)))
            else:
                R_ej[i] = np.inf
        
        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = f_ej * M_use[:, None] / np.power(2*np.pi*R_ej**2, 3/2) * np.exp(-np.power(r_use/R_ej, 2)/2) * kfac

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof


class Gas(MeadProfiles):
    """
    Convenience class for combining bound and ejected gas components.

    This class serves as a unified interface for gas profiles in halos, combining the contributions 
    from bound gas (`BoundGas`) and ejected gas (`EjectedGas`). It simplifies calculations where 
    the total gas profile is required, leveraging the underlying logic and methods of the individual 
    gas components.
    """

    def __init__(self, **kwargs): self.myprof = BoundGas(**kwargs) + EjectedGas(**kwargs)
    def __getattr__(self, name):  return getattr(self.myprof, name)
    
    #Need to explicitly set these two methods (to enable pickling)
    #since otherwise the getattr call above leads to infinite recursions.
    def __getstate__(self): self.__dict__.copy()    
    def __setstate__(self, state): self.__dict__.update(state)


class CollisionlessMatter(MeadProfiles):
    """
    Class for modeling the density profile of collisionless matter in halos.

    This class extends `MeadProfiles` to compute the density profile of collisionless matter, 
    including dark matter and satellite components. The profile accounts for modifications to 
    the concentration parameter based on baryonic feedback effects and redshift evolution.

    Notes
    -----
    - The concentration parameter is adjusted using a feedback-dependent factor that depends on 
      the bound gas fraction (\( f_{\\text{bnd}} \)) and redshift.
    - The density profile follows the NFW formula, with normalization based on the halo mass 
      and concentration.
    - The fraction of baryons and stars (\( f_{\\text{bar}} \) and \( f_{\\text{sat}} \)) is 
      incorporated to rescale the characteristic density, ensuring proper mass accounting.

    The density profile is given by:

    .. math::

        \\rho(r) = \\frac{\\rho_c}{(r/r_s)(1 + r/r_s)^2}

    where:
    - \( \\rho_c \) is the characteristic density, adjusted for baryonic effects and normalized 
      by the halo mass.
    - \( r_s = R / c \) is the scale radius, with \( c \) being the concentration parameter.

    The concentration parameter is modified as:

    .. math::

        c = c_{\\text{original}} \\cdot \\left( 1 + \\epsilon_1 + 
        (\\epsilon_2 - \\epsilon_1) \\frac{f_{\\text{bnd}}}{f_{\\text{bar}}} \\right)

    where:
    - \( \\epsilon_1 \) and \( \\epsilon_2 \) are redshift-dependent parameters.
    - \( f_{\\text{bnd}} \) is the bound gas fraction.
    - \( f_{\\text{bar}} \) is the total baryon fraction.
    """

    def _modify_concentration(self, cosmo, c, M, a):

        z      = 1/a - 1
        f_bar  = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        f_bnd  = self._get_gas_frac(M, a, cosmo)[0]
        eps1   = self.eps1 + z * self.nu_eps1
        factor = (1 + eps1 + (self.eps2 - eps1) * f_bnd / f_bar)

        return c * factor
    
    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        if self.cdelta is None:
            #Use the Duffy08 calibration following Equation 33 in https://arxiv.org/pdf/2005.00009
            #Can in principle swap this around as you want
            c_M_relation = ccl.halos.concentration.ConcentrationDuffy08(mass_def = self.mass_def)
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mass_def = self.mass_def)
            
        c   = c_M_relation(cosmo, M_use, a)
        c   = self._modify_concentration(cosmo, c, M_use, a)
        R   = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        r_s = R/c

        #Get the normalization (rho_c) analytically since we don't have a truncation radii like S19 does
        Norm  = 4*np.pi*r_s**3 * (np.log(1 + c) - c/(1 + c))
        rho_c = M_use/Norm
        f_str, f_cen, f_sat = self._get_star_frac(M_use, a, cosmo)
        f_bar = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        rho_c = rho_c * (1 - f_bar) #Rescale to correct fraction of mass

        r_s, c, rho_c = r_s[:, None], c[:, None], rho_c[:, None]
        r_use, R      = r_use[None, :], R[:, None]

        arg  = (r_use - self.cutoff)
        arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof = rho_c/(r_use/r_s * (1 + r_use/r_s)**2) * kfac
        prof = np.where(r_use <= R, prof, 0)
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof


class DarkMatterOnly(DarkMatter):
    """
    For Mead20, the DarkMatterOnly model includes just an NFW profile.
    There is no two-halo term. This class is simply a copy of the `DarkMatter` class.
    See that class for more details
    """


class DarkMatterBaryon(S19.DarkMatterBaryon, MeadProfiles):

    """
    Class representing a combined dark matter and baryonic matter profile.

    This class is derived from the `MeadProfiles` class and provides an implementation 
    that combines the contributions from dark matter, gas, stars, and collisionless matter 
    to compute the total density profile. It ensures mass conservation and accounts for both 
    dark matter and baryonic components. It does not include a two-halo term. 
    See `DarkMatterBaryonwithLSS` for a convenience class that includes the TwoHalo. 

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

    1. **Normalization of Dark Matter Baryon**: To ensure mass conservation, the one-halo term is 
       normalized so that the dark matter-baryon matches the dark matter-only profile
         at large radii. The normalization factor is calculated as:

       .. math::

           \\text{Factor} = \\frac{M_{\\text{DMO}}}{M_{\\text{DMB}}}

       where:

       - :math:`M_{\\text{DMO}}` is the total mass from the dark matter-only profile.
       - :math:`M_{\\text{DMB}}` is the total mass from the combined dark matter and baryon profile.

    2. **Total Density Profile**: The total density profile is computed by summing the contributions 
       from the collisionless matter, stars, and gas, scaled by the normalization factor:

       .. math::

           \\rho_{\\text{total}}(r) = \\rho_{\\text{CLM}}(r) \\cdot \\text{Factor} + \\rho_{\\text{stars}}(r) \\cdot \\text{Factor} + \\rho_{\\text{gas}}(r) \\cdot \\text{Factor}

       where:

       - :math:`\\rho_{\\text{CLM}}(r)` is the density from the collisionless matter profile.
       - :math:`\\rho_{\\text{stars}}(r)` is the stellar density profile.
       - :math:`\\rho_{\\text{gas}}(r)` is the gas density profile.

    This method ensures that both dark matter and baryonic matter are accounted for, 
    providing a realistic representation of the total matter distribution.

    See `SchneiderProfiles`, `Gas`, `Stars`, `CollisionlessMatter`, `DarkMatter`
    classes for more details on the underlying profiles and parameters.
    """

    def __init__(self, gas = None, stars = None, collisionlessmatter = None, darkmatter = None, **kwargs):
        
        self.Gas   = gas
        self.Stars = stars
        self.TwoHalo    = Zeros() #Should not add 2-halo in Mead method
        self.DarkMatter = darkmatter
        self.CollisionlessMatter = collisionlessmatter
        
        if self.Gas is None:        self.Gas        = Gas(**kwargs)        
        if self.Stars is None:      self.Stars      = Stars(**kwargs)
        if self.DarkMatter is None: self.DarkMatter = DarkMatter(**kwargs)
        if self.CollisionlessMatter is None: self.CollisionlessMatter = CollisionlessMatter(**kwargs)

        MeadProfiles.__init__(self, **kwargs)


class DarkMatterOnlywithLSS(S19.DarkMatterOnly, MeadProfiles):

    __doc__ = S19.DarkMatterOnly.__doc__.replace('Schneider', 'Mead')

    def __init__(self, darkmatter = None, twohalo = None, **kwargs):
        
        self.DarkMatter = darkmatter
        self.TwoHalo    = twohalo

        if self.DarkMatter is None: self.DarkMatter = DarkMatter(**kwargs)
        if self.TwoHalo is None: self.TwoHalo = TwoHalo(**kwargs)

        MeadProfiles.__init__(self, **kwargs)


class DarkMatterBaryonwithLSS(S19.DarkMatterBaryon, MeadProfiles):

    __doc__ = S19.DarkMatterBaryon.__doc__.replace('Schneider', 'Mead')

    def __init__(self, gas = None, stars = None, collisionlessmatter = None, darkmatter = None, twohalo = None, **kwargs):
        
        self.Gas   = gas
        self.Stars = stars
        self.TwoHalo    = twohalo
        self.DarkMatter = darkmatter
        self.CollisionlessMatter = collisionlessmatter
        
        if self.Gas is None:        self.Gas        = Gas(**kwargs)        
        if self.Stars is None:      self.Stars      = Stars(**kwargs)
        if self.TwoHalo is None:    self.TwoHalo    = TwoHalo(**kwargs)
        if self.DarkMatter is None: self.DarkMatter = DarkMatter(**kwargs)
        if self.CollisionlessMatter is None: self.CollisionlessMatter = CollisionlessMatter(**kwargs)

        MeadProfiles.__init__(self, **kwargs)


class Temperature(MeadProfiles):
    """
    Class for modeling the temperature profile of halos.

    This class extends `MeadProfiles` to compute the temperature profile of gas in halos, 
    based on the gravitational potential and the mean molecular weight of the gas. 

    Notes
    -----
    - The real-space temperature profile is derived from the characteristic energy scale, 
      assuming hydrostatic equilibrium.

    The real-space temperature profile is given by:

    .. math::

        T(r) = T_0 \\cdot \\frac{\\ln(1 + r/r_s)}{r/r_s}

    where:
    - \( T_0 \) is the characteristic temperature scale, proportional to the gravitational 
      potential of the halo:
      
      .. math::

          T_0 = \\frac{G M \\mu m_p}{R} \\cdot \\frac{1}{k_B}

      \( G \) is the gravitational constant, \( M \) is the halo mass, \( \mu \) is the mean molecular weight, 
      \( m_p \) is the proton mass, and \( k_B \) is the Boltzmann constant.
    - \( r_s \) is the scale radius, defined as \( R / c \), where \( R \) is the virial radius and \( c \) is 
      the concentration parameter.
    """

    
    def _real(self, cosmo, r, M, a):
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1
        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        
        if self.cdelta is None:
            #Use the Duffy08 calibration following Equation 33 in https://arxiv.org/pdf/2005.00009
            #Can in principle swap this around as you want
            c_M_relation = ccl.halos.concentration.ConcentrationDuffy08(mass_def = self.mass_def)
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mass_def = self.mass_def)
            
        c    = c_M_relation(cosmo, M_use, a)
        R    = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        r_s  = (R/c)[:, None]
    
        #Characteristic energy scale, convert to cgs and then convert to temperature (Kelvin)
        E0   = G * M_use * m_p * self.mean_molecular_weight / (a * R) * (Msun_to_Kg * 1e3) * (Mpc_to_m * 1e2)**2 
        T0   = self.alpha * E0 / (3/2 * kb_cgs)
        prof = T0[:, None] * np.log(1 + r_use/r_s) / (r_use/r_s)

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof
    
    
    def projected(self, cosmo, r, M, a):

        r_max = self.padding_hi_proj * np.max(r)
        if self.proj_cutoff is not None: r_max = self.proj_cutoff

        return super().projected(cosmo, r, M, a) / (2 * r_max)
    


class Pressure(MeadProfiles):
    """
    Class for modeling the pressure profile of gas in halos.

    This class extends `MeadProfiles` to compute the pressure profile, incorporating contributions 
    from both bound and ejected gas components. The pressure is calculated as the product of the 
    gas density and temperature, with separate terms for the bound and ejected gas phases.

    Parameters
    ----------
    boundgas : BoundGas, optional
        Instance of the `BoundGas` class representing the bound gas component.
        If not provided, a default `BoundGas` object is created.
    ejectedgas : EjectedGas, optional
        Instance of the `EjectedGas` class representing the ejected gas component.
        If not provided, a default `EjectedGas` object is created.
    temperature : Temperature, optional
        Instance of the `Temperature` class representing the gas temperature.
        If not provided, a default `Temperature` object is created.
    **kwargs
        Additional arguments passed to initialize the parent `MeadProfiles` class and associated components.

    Notes
    -----
    - The pressure profile is computed as the sum of two components:
        1. The bound gas component, which depends on the real-space gas density and temperature.
        2. The ejected gas component, which uses a redshift-dependent temperature normalization.
    - The gas number density is derived from the mass density by dividing by the mean molecular weight and proton mass.

    The pressure profile is given by:

    .. math::

        P(r) = P_{\\text{bound}}(r) + P_{\\text{ejected}}(r)

    where:
    - \( P_{\\text{bound}}(r) = n_{\\text{bound}}(r) \\cdot T_{\\text{bound}}(r) \\cdot k_B \)
    - \( P_{\\text{ejected}}(r) = n_{\\text{ejected}}(r) \\cdot T_{\\text{ejected}}(r) \\cdot k_B \)
    - \( n(r) \) is the number density of gas.
    - \( T(r) \) is the temperature of the gas.
    - \( k_B \) is the Boltzmann constant.
    """

    def __init__(self, boundgas = None, ejectedgas = None, temperature = None, **kwargs):
        
        self.BoundGas    = boundgas
        self.EjectedGas  = ejectedgas
        self.Temperature = temperature
        if self.BoundGas is None:    self.BoundGas    = BoundGas(**kwargs)
        if self.EjectedGas is None:  self.EjectedGas  = EjectedGas(**kwargs)
        if self.Temperature is None: self.Temperature = Temperature(**kwargs)        

        super().__init__(**kwargs)


    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        #The first "bound" component
        T    = self.Temperature.real(cosmo, r_use, M, a)
        n    = self.BoundGas.real(cosmo, r_use, M, a) / (self.mean_molecular_weight * m_p) / (Mpc_to_m * m_to_cm)**3
        P1   = T * n * kb_cgs

        #The second, "ejected" component
        T    = self.T_w * np.exp(self.nu_T_w * z)
        n    = self.EjectedGas.real(cosmo, r_use, M, a) / (self.mean_molecular_weight * m_p) / (Mpc_to_m * m_to_cm)**3
        P2   = T * n * kb_cgs

        prof = P1 + P2

        return prof
    

#Default params, provided in Table 2 of https://arxiv.org/pdf/2005.00009
#We convert from Msun/h --> Msun assuming h = 0.7
Params_TAGN_7p6_All =  {'A_star' : 0.0346, 'nu_A_star' : -0.0092, 'M_star' : np.power(10, 12.5506)/0.7, 'nu_M_star' : -0.4615,
                        'eta' : -0.4970, 'eps1' : 0.4021, 'nu_eps1' : 0.0435, 'Gamma' : 1.2763, 'nu_Gamma' : -0.0554, 
                        'M_0' : np.power(10, 13.0978)/0.7, 'T_w' : np.power(10, 6.6762), 'nu_T_w' : -0.5566,
                        'eps2' : 0, 'mean_molecular_weight' : 0.59, 'eta_b' : 0.5, 'sigma_star' : 1.2, 'beta' : 0.6,
                        'epsilon_h' : 0.015, 'p' : 0.3, 'q' : 0.707, 'alpha' : 1}

Params_TAGN_7p8_All =  {'A_star' : 0.0342, 'nu_A_star' : -0.0105, 'M_star' : np.power(10, 12.3715)/0.7, 'nu_M_star' : 0.0149,
                        'eta' : -0.4052, 'eps1' : 0.1236, 'nu_eps1' : -0.0187, 'Gamma' : 1.2956, 'nu_Gamma' : -0.0937, 
                        'M_0' : np.power(10, 13.4854)/0.7, 'T_w' : np.power(10, 6.6545), 'nu_T_w' : -0.3652,
                        'eps2' : 0, 'mean_molecular_weight' : 0.59, 'eta_b' : 0.5, 'sigma_star' : 1.2, 'beta' : 0.6,
                        'epsilon_h' : 0.015, 'p' : 0.3, 'q' : 0.707, 'alpha' : 1}

Params_TAGN_8p0_All =  {'A_star' : 0.0321, 'nu_A_star' : -0.0094, 'M_star' : np.power(10, 12.3032)/0.7, 'nu_M_star' : -0.0817,
                        'eta' : -0.3443, 'eps1' : -0.1158, 'nu_eps1' : 0.1408, 'Gamma' : 1.2861, 'nu_Gamma' : -0.1382, 
                        'M_0' : np.power(10, 14.1254)/0.7, 'T_w' : np.power(10, 6.6615), 'nu_T_w' : -0.0617,
                        'eps2' : 0, 'mean_molecular_weight' : 0.59, 'eta_b' : 0.5, 'sigma_star' : 1.2, 'beta' : 0.6,
                        'epsilon_h' : 0.015, 'p' : 0.3, 'q' : 0.707, 'alpha' : 1}


Params_TAGN_7p6_MPr =  {'A_star' : 0.0348, 'nu_A_star' : -0.0093, 'M_star' : np.power(10, 12.462)/0.7, 'nu_M_star' : -0.3664,
                        'eta' : -0.3428, 'eps1' : -0.10017, 'nu_eps1' : -0.04559, 'Gamma' : 1.16468, 'nu_Gamma' : 0.0, 
                        'M_0' : np.power(10, 13.19486)/0.7, 'T_w' : np.power(10, 6.67618), 'nu_T_w' : -0.55659,
                        'eps2' : 0, 'mean_molecular_weight' : 0.59, 'eta_b' : 0.5, 'sigma_star' : 1.2, 'beta' : 0.6,
                        'epsilon_h' : 0.015, 'p' : 0.3, 'q' : 0.707, 'alpha' : 0.7642}

Params_TAGN_7p8_MPr =  {'A_star' : 0.0330, 'nu_A_star' : -0.0088, 'M_star' : np.power(10, 12.4479)/0.7, 'nu_M_star' : -0.3521,
                        'eta' : -0.3556, 'eps1' : -0.1065, 'nu_eps1' : -0.1073, 'Gamma' : 1.17702, 'nu_Gamma' : 0.0, 
                        'M_0' : np.power(10, 13.59369)/0.7, 'T_w' : np.power(10, 6.65445), 'nu_T_w' : -0.36515,
                        'eps2' : 0, 'mean_molecular_weight' : 0.59, 'eta_b' : 0.5, 'sigma_star' : 1.2, 'beta' : 0.6,
                        'epsilon_h' : 0.015, 'p' : 0.3, 'q' : 0.707, 'alpha' : 0.8471}

Params_TAGN_8p0_MPr =  {'A_star' : 0.0309, 'nu_A_star' : -0.0082, 'M_star' : np.power(10, 12.3923)/0.7, 'nu_M_star' : -0.3073,
                        'eta' : -0.3505, 'eps1' : -0.12533, 'nu_eps1' : -0.01107, 'Gamma' : 1.19657, 'nu_Gamma' : 0.0, 
                        'M_0' : np.power(10, 14.24798)/0.7, 'T_w' : np.power(10, 6.66146), 'nu_T_w' : -0.06167,
                        'eps2' : 0, 'mean_molecular_weight' : 0.59, 'eta_b' : 0.5, 'sigma_star' : 1.2, 'beta' : 0.6,
                        'epsilon_h' : 0.015, 'p' : 0.3, 'q' : 0.707, 'alpha' : 1.0314}