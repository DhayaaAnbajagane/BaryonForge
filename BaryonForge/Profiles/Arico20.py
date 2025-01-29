import numpy as np
import pyccl as ccl
from operator import add, mul, sub, truediv, pow, neg, pos, abs
import warnings

from scipy import interpolate, special, integrate
from ..utils import _set_parameter, safe_Pchip_minimize
from .misc import Zeros
from . import Schneider19 as S19
from .Thermodynamic import (G, Msun_to_Kg, Mpc_to_m, kb_cgs, m_p, m_to_cm)

__all__ = ['model_params', 'AricoProfiles', 
           'DarkMatter', 'TwoHalo', 'Stars', 'Gas', 'BoundGas', 'EjectedGas', 'ReaccretedGas', 'CollisionlessMatter',
           'DarkMatterOnly', 'DarkMatterBaryon', 'Pressure', 'NonThermalFrac', 'Temperature']


model_params = ['cdelta', 'a', 'n', #DM profle params and relaxation params
                'q', 'p', #Two Halo
                'cutoff', 'proj_cutoff', #Cutoff parameters (numerical)
                
                'theta_out', 'theta_inn', 'M_inn', 'M_c', 'mu', 'beta', 
                'M_r', 'beta_r', 'eta', 'theta_rg', 'sigma_rg', 'epsilon_hydro', #Default gas profile param

                'alpha_sat', 'M1_0', 'alpha_g', 'epsilon_h', #Star params

                'A_nt', 'alpha_nt', #Pressure params
                'mean_molecular_weight', #Gas number density params
               ]


class AricoProfiles(S19.SchneiderProfiles):
    __doc__ = S19.SchneiderProfiles.__doc__.replace('Schneider', 'Arico')

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
            else:
                setattr(self, m, None)

        #Sets the cutoff scale of all profiles, in comoving Mpc. Prevents divergence in FFTLog
        #Also set cutoff of projection integral. Should be the box side length
        self.cutoff      = kwargs['cutoff'] if 'cutoff' in kwargs.keys() else 1e3 #1Gpc is a safe default choice
        self.proj_cutoff = kwargs['proj_cutoff'] if 'proj_cutoff' in kwargs.keys() else self.cutoff
                
    
    def _get_gas_params(self, M, z):
        """
        Compute gas parameters based on halo mass and redshift.

        This method calculates key gas parameters, including the slope (\( \\beta \)) and the 
        inner and outer radius ratios (\( \\theta_{\\text{inn}} \) and \( \\theta_{\\text{out}} \)), 
        which are used to model the gas profile in halos.

        Parameters
        ----------
        M : array_like
            Halo masses, in units of solar masses.
        z : array_like
            Redshift values corresponding to the input halo masses.

        Returns
        -------
        beta : array_like
            The slope parameter for each halo mass, defined as:

            .. math::

                \\beta = 3 - \\left(\\frac{M_{\\text{inn}}}{M}\\right)^{\mu}

            where \( M_{\\text{inn}} \) and \( \mu \) are model parameters.
        theta_out : array_like
            The outer temperature ratio, set to a constant value for all masses.
        theta_inn : array_like
            The inner temperature ratio, set to a constant value for all masses.
        """
        
        beta = 3 - np.power(self.M_inn/M, self.mu) * np.ones_like(M)
        beta = np.clip(beta, -1, None)
        
        #Use M_c as the mass-normalization for simplicity sake
        theta_out = self.theta_out * np.ones_like(M) 
        theta_inn = self.theta_inn * np.ones_like(M)
        
        beta     = beta[:, None]
        theta_out = theta_out[:, None]
        theta_inn = theta_inn[:, None]
        
        return beta, theta_out, theta_inn
    

    def _get_star_frac(self, M, z, satellite = False):
        """
        Compute the stellar fraction as a function of halo mass and redshift.

        This method calculates the stellar fraction, \( f_{\\text{CG}} \), using a parametric model 
        based on fitting functions from Behroozi et al. (2013) and param values from Kravtsov et al. (2018). 
        The model accounts for redshift evolution and includes an optional modification for satellite galaxies.

        Parameters
        ----------
        M : array_like
            Halo masses, in units of solar masses.
        z : array_like
            Redshift values corresponding to the input halo masses.
        satellite : bool, optional
            If True, modifies the stellar fraction parameters for satellite galaxies. 
            Default is False.

        Returns
        -------
        fCG : array_like
            The computed stellar fraction for each input halo mass and redshift.

        Notes
        -----
        - The model parameters are derived from the fitting functions in Behroozi et al. (2013) 
        and include terms for redshift evolution and halo mass dependence.
        - For satellite galaxies, all parameters are adjusted using a scaling factor, `alpha_sat`.
        - The stellar fraction is computed as:

        .. math::

            f_{\\text{CG}} = \epsilon \\cdot \frac{M_1}{M} 
            \\cdot 10^{g(x) - g(0)}

        where:
        - \( x = \log_{10}(M / M_1) \)
        - \( g(x) \) is a complex function of \( x \), \(\alpha\), \(\delta\), and \(\gamma\).
        - \(\epsilon\), \(M_1\), \(\alpha\), \(\delta\), and \(\gamma\) are redshift-dependent parameters.
        """

        #Based on fitting function of Behroozi+2013 and data from Kravtsov+2018
        #see Eq A16-17 in https://arxiv.org/pdf/1911.08471
        M1_a    = -1.793
        M1_z    = -0.251
        eps_0   = np.log10(0.023)
        eps_a   = -0.006
        eps_a2  = -0.119
        alpha_0 = -1.779
        alpha_a = 0.731
        delta_0 = 4.394
        delta_a = 2.608
        delta_z = -0.043
        gamma_0 = 0.547
        gamma_a = 1.319
        gamma_z = 0.279

        a   = 1/(1 + z)
        nu  = np.exp(-4*a**2)
        M1  = self.M1_0 * np.power(10, (M1_a*(a - 1) + M1_z * z)*nu)
        eps = np.power(10, eps_0 + nu*(eps_a*(a - 1)) + eps_a2 * (a - 1))
        alpha = alpha_0 + nu*(alpha_a*(a - 1))
        delta = delta_0 + nu*(delta_a*(a - 1) + delta_z*z)
        gamma = gamma_0 + nu*(gamma_a*(a - 1) + gamma_z*z)

        if satellite:

            M1     = np.power(M1,  self.alpha_sat)
            eps    = np.power(eps, self.alpha_sat)
            alpha *= self.alpha_sat
            delta *= self.alpha_sat
            gamma *= self.alpha_sat

        x   = np.log10(M/M1)
        g_x = -np.log10(np.power(10, alpha * x) + 1) + delta * np.power(np.log10(1 + np.exp(x)), gamma)/(1 + np.exp(np.clip(10**-x, None, 30)))
        g_0 = -np.log10(np.power(10, alpha * 0) + 1) + delta * np.power(np.log10(1 + np.exp(0)), gamma)/(1 + np.exp(10**-0))
        fCG = eps * (M1/M) * np.power(10, g_x - g_0)

        return fCG    


    def __str_par__(self):
        '''
        String with all input params and their values
        '''
        
        string = f"("
        for m in self.model_param_names:
            string += f"{m} = {self.__dict__[m]}, "
        string = string[:-2] + ')'
        return string


class DarkMatter(AricoProfiles):
    """
    Class for modeling the dark matter density profile using the NFW (Navarro-Frenk-White) framework.

    This class extends `AricoProfiles` to compute the dark matter density profile, incorporating 
    flexible concentration-mass relations and a truncated profile at large radii. 

    Notes
    -----
    The dark matter profile is calculated using the NFW formula, which depends on the halo's 
    mass and concentration. The normalization is determined analytically to ensure that the 
    total mass within the virial radius matches the input halo mass.

    The density profile is given by:

    .. math::

        \\rho(r) = 
        \\begin{cases} 
        \\frac{\\rho_c}{(r/r_s)(1 + r/r_s)^2}, & r \\leq R \\\\ 
        0, & r > R
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
            c_M_relation = ccl.halos.concentration.ConcentrationDiemer15(mass_def = self.mass_def) #Use the diemer calibration
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mass_def = self.mass_def)
            
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


class TwoHalo(S19.TwoHalo, AricoProfiles):
    __doc__ = S19.TwoHalo.__doc__.replace('SchneiderProfiles', 'AricoProfiles')


class Stars(AricoProfiles):
    """
    Class for modeling the stellar density profile in halos.

    This class extends `AricoProfiles` to compute the density profile of stars within halos. 
    The profile accounts for the stellar fraction, redshift evolution, and a normalized 
    radial distribution.

    Notes
    -----
    - The radial profile is modeled with a power-law dependence and an exponential cutoff, 
      normalized to integrate to the stellar mass within the halo.
    - The stellar fraction is computed using the `_get_star_frac` method, which depends on 
      halo mass and redshift.
    
    The stellar density profile is given by:

    .. math::

        \\rho_{\\star}(r) = \\frac{f_{\\text{cga}} M}{R_h r^{\\alpha_g}} 
        \\cdot \\exp\\left(-\\frac{r^2}{4 R_h^2}\\right) 
        \\cdot \\frac{1}{N}

    where:
    - \( f_{\\text{cga}} \) is the stellar fraction at a given halo mass and redshift.
    - \( M \) is the halo mass.
    - \( R_h \) is the scale radius, proportional to the halo virial radius.
    - \( \\alpha_g \) is the power-law slope parameter.
    - \( N \) is the numerically computed normalization factor, computed to ensure mass conservation.
    """

    def __init__(self, r_min_int = 1e-6, r_max_int = 5, **kwargs):
        
        super().__init__(r_min_int = r_min_int, r_max_int = r_max_int, **kwargs)
        
        #For some reason, we need to make this extreme in order
        #to prevent ringing in the profiles. Haven't figured out
        #why this is the case. We also change the plaw to be close to -3.
        #If exactly -3 we get CCL spline error, and being close to -3 results in
        #convergent predictions for alpha_g >> 2
        self.update_precision_fftlog(padding_lo_fftlog = 1e-5, padding_hi_fftlog = 1e5)
        self.update_precision_fftlog(plaw_fourier = -3 + 1e-4)

    
    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        R     = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        z     = 1/a - 1

        f_cga = self._get_star_frac(M_use, z)[:, None]
        R_h   = self.epsilon_h * R[:, None]

        #Integrate over wider region in radii to get normalization of star profile
        #There's no way the profile has any support than 5Mpc. So use a narrower range.
        r_integral    = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)
        prof_integral = 1 / R_h / np.power(r_integral, self.alpha_g) * np.exp(-np.power(r_integral/2/R_h, 2))
        Normalization = np.trapz(4 * np.pi * r_integral**2 * prof_integral, r_integral, axis = -1)[:, None]
        
        #Final profile. No truncation needed since exponential cutoff already does that for us
        prof = f_cga*M_use[:, None] / R_h / np.power(r_use, self.alpha_g) * np.exp(-np.power(r_use/2/R_h, 2)) / Normalization

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof


class BoundGas(AricoProfiles):
    """
    Class for modeling the bound gas density profile in halos.

    This class extends `AricoProfiles` to compute the density profile of bound gas within halos.
    This follows the updated model from https://arxiv.org/pdf/2009.14225 rather than the original
    model.

    Parameters
    ----------
    None (inherits all parameters from `AricoProfiles`).

    Notes
    -----
    - Radial dependence is modeled using two scale radii:
        1. \( R_{\\text{co}} \): Core radius, controlling the central density slope.
        2. \( R_{\\text{ej}} \): Outer radius, controlling the cutoff.
    - The bound gas fraction (\( f_{\\text{bg}} \)) is derived by subtracting the contributions of 
      stellar and ejected gas fractions from the total baryon fraction.

    The density profile is given by:

    .. math::

        \\rho_{\\text{bg}}(r) = 
        \\begin{cases} 
        \\frac{f_{\\text{bg}} M}{N} 
        \\cdot \\frac{1}{(1 + u)^{\\beta}} 
        \\cdot \\frac{1}{(1 + v^2)^2}, & r \\leq R \\\\ 
        0, & r > R
        \\end{cases}


    where:
    - \( u = r / R_{\\text{co}} \), \( v = r / R_{\\text{ej}} \)
    - \( R_{\\text{co}} = \\theta_{\\text{inn}} R \), \( R_{\\text{ej}} = \\theta_{\\text{out}} R \)
    - \( \\beta \) is the slope parameter, and \( N \) is the normalization factor.
    - R is the spherical overdensity radius of the halo
    """

    def _real(self, cosmo, r, M, a):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_cg  = self._get_star_frac(M_use, z)
        f_sg  = self._get_star_frac(M_use, z, satellite = True)
        f_bar = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        beta  = self._get_gas_params(M_use, z)[0]
        f_gas = f_bar - f_cg - f_sg
        f_hg  = f_gas / (1 + np.power(self.M_c/M_use, self.beta))
        f_rg  = (f_gas - f_hg) / (1 + np.power(self.M_r/M_use, self.beta_r))
        f_rg  = np.clip(f_rg, None, f_hg) #Reaccreted gas cannot be more than halo gas 
        f_bg  = f_hg - f_rg

        #Get gas params
        beta, theta_out, theta_inn = self._get_gas_params(M_use, z)
        R_co = theta_inn*R[:, None]
        R_ej = theta_out*R[:, None]
        f_bg = f_bg[:, None]
        
        u = r_use/R_co
        v = r_use/R_ej
        
        #Do normalization halo-by-halo, since we want custom radial ranges.
        #This way, we can handle sharp transition at R200c without needing
        #super fine resolution in the grid.
        Normalization = np.ones_like(M_use)
        for m_i in range(M_use.shape[0]):
            r_integral = np.geomspace(self.r_min_int, R[m_i], self.r_steps)
            u_integral = r_integral/R_co[m_i]
            v_integral = r_integral/R_ej[m_i]        

            prof_integral = 1/(1 + u_integral)**beta[m_i] / (1 + v_integral**2)**2
            prof_integral = np.where(r_integral < R[m_i], prof_integral, 0)
            Normalization[m_i] = np.trapz(4 * np.pi * r_integral**2 * prof_integral, r_integral)
        Normalization = Normalization[:, None]

        del u_integral, v_integral, prof_integral

        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = 1/(1 + u)**beta / (1 + v**2)**2 * kfac
        prof *= f_bg*M_use[:, None]/Normalization #This profile is allowed to go beyond R200c!

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)


        return prof
    

class EjectedGas(AricoProfiles):
    """
    Class for modeling the ejected gas density profile in halos.

    This class extends `AricoProfiles` to compute the density profile of gas that has been 
    ejected from halos due to feedback processes, such as supernovae or AGN activity. 
    The profile is a simple Gaussian, with a scale set by the escape radius.

    Notes
    -----
    - The ejected gas fraction (\( f_{\\text{eg}} \)) is calculated as the remainder of the 
      baryonic fraction after subtracting the stellar and bound gas components.
    - The ejection radius (\( R_{\\text{ej}} \)) is derived from the escape radius, scaled 
      by a parameter \( \eta \).
    - The radial density profile follows a Gaussian distribution, normalized to integrate 
      to the total ejected gas mass.

    The density profile is given by:

    .. math::

        \\rho_{\\text{eg}}(r) = \\frac{f_{\\text{eg}} M}{(2 \\pi R_{\\text{ej}}^2)^{3/2}} 
        \\cdot \\exp\\left(-\\frac{r^2}{2 R_{\\text{ej}}^2}\\right)

    where:
    - \( f_{\\text{eg}} \) is the ejected gas fraction.
    - \( M \) is the halo mass.
    - \( R_{\\text{ej}} = \\eta \\cdot R_{\\text{esc}} \), where \( R_{\\text{esc}} \) is 
      the escape radius calculated from the halo's escape velocity.
    """

    def _real(self, cosmo, r, M, a):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1
        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_cg  = self._get_star_frac(M_use, z)
        f_sg  = self._get_star_frac(M_use, z, satellite = True)
        f_bar = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        beta  = self._get_gas_params(M_use, z)[0]
        f_gas = f_bar - f_cg - f_sg
        f_hg  = f_gas / (1 + np.power(self.M_c/M_use, self.beta))
        f_eg  = f_gas - f_hg #By definition f_hg <= f_gas
        f_eg  = f_eg[:, None]

        #Now use the escape radius, which is r_esc = v_esc * t_hubble
        #and this reduces down to just 1/2 * sqrt(Delta) * R_Delta
        assert self.mass_def.name[-1] == 'c', f"Escape radius cannot be calculated for mass_def = {self.mass_def.name}. Use critical overdensity."
        R_esc = 1/2 * np.sqrt(self.mass_def.Delta) * R
        R_ej  = self.eta * 0.75 * R_esc
        R_ej  = R_ej[:, None]

        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = f_eg * M_use[:, None] / np.power(2*np.pi*R_ej**2, 3/2) * np.exp(-np.power(r_use/R_ej, 2)/2) * kfac

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof



class ReaccretedGas(AricoProfiles):
    """
    Class for modeling the reaccreted gas density profile in halos.

    This class extends `AricoProfiles` to compute the density profile of gas that has been 
    reaccreted onto halos after being ejected, incorporating redshift evolution and mass dependence.

    Notes
    -----
    - The reaccreted gas fraction (\( f_{\\text{rg}} \)) is derived by subtracting the contributions 
      of stellar, ejected, and bound gas fractions from the total baryon fraction.
    - The radial profile is modeled as a Gaussian distribution with a peak radius (\( R_{\\text{rg}} \)) 
      and width (\( \sigma_{\\text{rg}} \)).
    - The profile is normalized analytically to ensure that the total reaccreted gas mass integrates correctly.

    The density profile is given by:

    .. math::

        \\rho_{\\text{rg}}(r) = \\frac{f_{\\text{rg}} M}{N} 
        \\cdot \\frac{1}{\\sqrt{2 \\pi \\sigma_{\\text{rg}}^2}} 
        \\cdot \\exp\\left(-\\frac{(r - R_{\\text{rg}})^2}{2 \\sigma_{\\text{rg}}^2}\\right)

    where:
    - \( f_{\\text{rg}} \) is the reaccreted gas fraction.
    - \( M \) is the halo mass.
    - \( R_{\\text{rg}} = \\theta_{\\text{rg}} R \) is the characteristic radius for reaccreted gas.
    - \( \sigma_{\\text{rg}} = \\sigma_{\\text{rg}} R \) is the standard deviation of the Gaussian profile.
    - \( N \) is the normalization factor, computed analytically to ensure mass conservation.
    """

    def _real(self, cosmo, r, M, a):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_cg  = self._get_star_frac(M_use, z)
        f_sg  = self._get_star_frac(M_use, z, satellite = True)
        f_bar = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        beta  = self._get_gas_params(M_use, z)[0]
        f_gas = f_bar - f_cg - f_sg
        f_hg  = f_gas / (1 + np.power(self.M_c/M_use, self.beta))
        f_rg  = (f_gas - f_hg) / (1 + np.power(self.M_r/M_use, self.beta_r))
        f_rg  = np.clip(f_rg, None, f_hg) #Reaccreted gas cannot be more than halo gas 
        
        #Get gas params
        R_rg = self.theta_rg*R[:, None]
        S_rg = self.sigma_rg*R[:, None]
        f_rg = f_rg[:, None]
        R    = R[:, None]
        
        #Can get normalization analytically
        t1   =  2 * np.sqrt(2 * np.pi) * (np.exp(-R_rg**2 / (2 * S_rg**2)) * R_rg - np.exp(-(R_rg - R)**2 / (2 * S_rg**2)) * (R_rg + R))
        t2   =  2 * np.pi * (R_rg**2 + S_rg**2) * special.erf(R_rg / (np.sqrt(2) * S_rg))
        t3   = -2 * np.pi * (R_rg**2 + S_rg**2) * special.erf((R_rg - R) / (np.sqrt(2) * S_rg))
        Norm = t1 * S_rg + t2 + t3

        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = 1/np.sqrt(2*np.pi*S_rg**2) * np.exp(-np.power((r_use - R_rg)/S_rg, 2)/2)
        prof *= f_rg*M_use[:, None]/Norm
        prof  = np.where(r_use[None, :] <= R, prof, 0)

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)


        return prof
    

class Gas(AricoProfiles):
    """
    Convenience class for combining gas components in halos.

    The `Gas` class provides a unified interface for modeling the total gas profile in halos. 
    It combines contributions from the following components:
    - `BoundGas`: Represents the bound gas component within halos.
    - `EjectedGas`: Represents gas that has been ejected from halos due to feedback processes.
    - `ReaccretedGas`: Represents gas that has been reaccreted onto halos after ejection.

    This class simplifies calculations by leveraging the logic and methods of these individual 
    gas components and combining their profiles into a single representation.
    """

    def __init__(self, **kwargs): self.myprof = BoundGas(**kwargs) + EjectedGas(**kwargs) + ReaccretedGas(**kwargs)
    def __getattr__(self, name):  return getattr(self.myprof, name)
    
    #Need to explicitly set these two methods (to enable pickling)
    #since otherwise the getattr call above leads to infinite recursions.
    def __getstate__(self): self.__dict__.copy()    
    def __setstate__(self, state): self.__dict__.update(state)


class ModifiedDarkMatter(AricoProfiles):
    """
    Class for modeling the modified dark matter density profile in halos.

    This class extends `AricoProfiles` to compute a dark matter profile modified by baryonic effects, 
    such as the influence of gas on the gravitational potential. It uses both 
    a gravity-only dark matter profile (`DarkMatter`) and a bound gas profile (`BoundGas`) to 
    account for the redistribution of dark matter mass within halos.

    Parameters
    ----------
    gas : BoundGas, optional
        Instance of the `BoundGas` class representing the bound gas component. 
        If not provided, a default `BoundGas` object is created.
    gravityonly : DarkMatter, optional
        Instance of the `DarkMatter` class representing the gravity-only dark matter component.
        If not provided, a default `DarkMatter` object is created.
    **kwargs
        Additional arguments passed to initialize the parent `AricoProfiles` class and associated components.

    Notes
    -----
    - The modified profile accounts for the interplay between gas and dark matter through 
      a redistribution of mass, ensuring physical consistency.
    - A minimization routine is used to solve a key equation for the 
      characteristic radius \( r_p \).
    - The final profile is normalized using the characteristic density (\( \rho_c \)) derived 
      from analytical relations.

    The modified dark matter density profile is given by:

    .. math::

        \\rho_{\\text{DM}}(r) = 
        \\begin{cases} 
        \\frac{\\rho_c}{(r/r_s)(1 + r/r_s)^2}, & r < r_p \\\\ 
        p_{\\text{Gro}} - p_{\\text{BG}}, & r \\geq r_p, r \\leq R \\\\ 
        0, & r > R
        \\end{cases}

    where:
    - \( \\rho_c \) is the characteristic density derived from the gravitational and gas components.
    - \( r_s = R / c \) is the scale radius.
    - \( r_p \) is the radius obtained by solving the balance equation for dark matter and gas.
    """

    def __init__(self, gas = None, gravityonly = None, **kwargs):
        
        self.Gas   = gas
        self.GravityOnly = gravityonly
        
        if self.Gas is None: self.Gas = BoundGas(**kwargs) 
        if self.GravityOnly is None: self.GravityOnly = DarkMatter(**kwargs)

        super().__init__(**kwargs)
    
    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        if self.cdelta is None:
            c_M_relation = ccl.halos.concentration.ConcentrationDiemer15(mass_def = self.mass_def) #Use the diemer calibration
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mass_def = self.mass_def)
            
        c   = c_M_relation(cosmo, M_use, a)
        R   = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        r_s = R/c        
        r_s = r_s[:, None]
        fDM = 1 - cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m

        #Solving equation A10 of https://arxiv.org/pdf/1911.08471 through minimization
        rp    = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)
        pGro  = np.array([self.GravityOnly.real(cosmo, r, m, a) for r, m in zip(R, M_use)])[:, None]
        pBG   = np.array([self.Gas.real(cosmo, r, m, a) for r, m in zip(R, M_use)])[:, None]
        LHS   = rp * np.power(rp + r_s, 2) * (pGro - pBG) * (np.log(1 + rp/r_s) - 1/(1 + r_s/rp)) + (pGro - pBG)/3 * (R**3 - rp**3)
        RHS   = fDM * M_use[None, :] / (4*np.pi)
        rp    = np.exp([safe_Pchip_minimize((LHS - RHS)[m_i], np.log(rp)) for m_i in range(LHS.shape[0])])[:, None]
        
        #Get the normalization based on equation A8 of https://arxiv.org/pdf/1911.08471
        rho_c = (pGro - pBG) * (rp/r_s) * np.power(1 + rp/r_s, 2)

        #Now the final profile
        prof  = rho_c / (r_use/r_s) / np.power(1 + r_use/r_s, 2)
        prof  = np.where(r_use[None, :] < rp, prof, (pGro - pBG))
        
        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = prof * kfac
        prof  = np.where(r_use[None, :] <= R, prof, 0)

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)
        
        return prof


class CollisionlessMatter(AricoProfiles):
    __doc__ = S19.CollisionlessMatter.__doc__.replace('Schneider', 'Arico')
    
    def __init__(self, gas = None, stars = None, darkmatter = None, max_iter = 10, reltol = 1e-2, r_min_int = 1e-8, r_max_int = 1e1, r_steps = 5000, **kwargs):
        
        self.Gas   = gas
        self.Stars = stars
        self.DarkMatter = darkmatter
        
        if self.Gas is None: self.Gas = Gas(**kwargs) 
        if self.Stars is None: self.Stars = Stars(**kwargs)
        if self.DarkMatter is None: self.DarkMatter = ModifiedDarkMatter(**kwargs) #Arico uses modified DM as default
            
        #Stop any artificially cutoffs when doing the relaxation.
        #The profile will be cutoff at the very last step instead
        self.Gas.set_parameter('cutoff', 1000)
        self.Stars.set_parameter('cutoff', 1000)
        self.DarkMatter.set_parameter('cutoff', 1000)
            
        self.max_iter   = max_iter
        self.reltol     = reltol
        
        super().__init__(**kwargs, r_min_int = 1e-8, r_max_int = 1e5, r_steps = 5000)

    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        if np.min(r) < self.r_min_int: 
            warnings.warn(f"Decrease integral lower limit, r_min_int ({self.r_min_int}) < minimum radius ({np.min(r)})", UserWarning)
        if np.max(r) > self.r_max_int: 
            warnings.warn(f"Increase integral upper limit, r_max_int ({self.r_max_int}) < maximum radius ({np.max(r)})", UserWarning)

        z = 1/a - 1
        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_sg   = self._get_star_frac(M_use, z, satellite = True)
        f_dm   = 1 - cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        f_clm  = f_dm + f_sg
        
        rho_clm = np.ones([M_use.shape[0], r_use.shape[0]], dtype = float)
        for m_i in range(M_use.shape[0]):

            #Def radius sampling for doing iteration.
            #This is different from Schneider version, because we have to do everything
            #halo by halo due to the sharp truncation radius (which induces oscillations,
            #in the cubic interpolations otherwise). Make sure the lower bound is always
            #sufficiently wide though
            r_integral = np.geomspace(self.r_min_int, R[m_i], self.r_steps)
            safe_range = (r_integral > 2 * np.min(r_integral) )

            #The DarkMatter profile may already have the f_dm normalization, but
            #this doesn't matter since we anyway renormalize the profiles later so it
            #gets to M_clm(<R200c) = M_tot * f_clm
            rho_i      = self.DarkMatter.real(cosmo, r_integral, M_use[m_i], a)
            rho_cga    = self.Stars.real(cosmo, r_integral, M_use[m_i], a)
            rho_gas    = self.Gas.real(cosmo, r_integral, M_use[m_i], a)

            dlnr  = np.log(r_integral[1]) - np.log(r_integral[0])
            dV    = r_integral**3 * dlnr
            M_i   = 4 * np.pi * integrate.cumulative_simpson(dV * rho_i  , axis = -1, initial = dV[0] * rho_i[0])
            M_cga = 4 * np.pi * integrate.cumulative_simpson(dV * rho_cga, axis = -1, initial = dV[0] * rho_cga[0])
            M_gas = 4 * np.pi * integrate.cumulative_simpson(dV * rho_gas, axis = -1, initial = dV[0] * rho_gas[0])

            #Assume extrapolation is used only for r > r_max. In this case, the extrapolation
            #coefficients are just the integrated mass at r_max. Our r_min is sufficientyly
            #low that we will not suffer extrapolation errors there (and even if we do it
            #should not matter at all given the infinitesimal volume element)
            M_i_max   = M_i[-1]
            M_cga_max = M_cga[-1]
            M_gas_max = M_gas[-1]

            #Set Extrapolate = False. We only need to extrapolate if r > R200c, where profile should be 0
            #and mass should be M(<R200c) so we'll just set it to that
            ln_M_NFW = interpolate.PchipInterpolator(np.log(r_integral), np.log(M_i),   extrapolate = False)
            ln_M_cga = interpolate.PchipInterpolator(np.log(r_integral), np.log(M_cga), extrapolate = False)
            ln_M_gas = interpolate.PchipInterpolator(np.log(r_integral), np.log(M_gas), extrapolate = False)

            del M_cga, M_gas, rho_i, rho_cga, rho_gas

            relaxation_fraction = np.ones_like(M_i)            
            counter      = 0
            max_rel_diff = np.inf #Initializing variable at infinity
            
            while max_rel_diff > self.reltol:

                with np.errstate(over = 'ignore'):
                    r_f  = r_integral*relaxation_fraction
                    M_f1 = f_clm[m_i]*M_i
                    M_f2 = np.exp(ln_M_cga(np.log(r_f)))
                    M_f3 = np.exp(ln_M_gas(np.log(r_f)))
                    M_f  = (np.where(np.isfinite(M_f1), M_f1, M_i_max) + 
                            np.where(np.isfinite(M_f2), M_f2, M_cga_max) + 
                            np.where(np.isfinite(M_f3), M_f3, M_gas_max)
                            )

                #Solve for the relaxation fraction following Equation A11 in https://arxiv.org/pdf/1911.08471
                relaxation_fraction_new = 1 + self.a*(np.power(M_i/M_f, self.n) - 1)

                #Normalize so the relaxation is at 1 at R200c
                #then make sure no r_f is greater than R200c
                norm = np.interp(R[m_i], r_integral, relaxation_fraction_new)
                relaxation_fraction_new /= norm

                diff     = relaxation_fraction_new/relaxation_fraction - 1
                abs_diff = np.abs(diff)
                
                max_rel_diff = np.max(abs_diff[safe_range])
                
                relaxation_fraction = relaxation_fraction_new * 1 #Multiple to avoid pointer assignment

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

            
            #Compute the relaxed DM profile, and the normalize so it 
            #has the right mass fraction within R200c.
            ln_M_clm  = np.log(f_clm[m_i]) + ln_M_NFW(np.log(r_integral/relaxation_fraction))
            ln_M_clm += np.log(f_clm[m_i] * M_use[m_i]) - np.interp(np.log(R[m_i]), np.log(r_integral), ln_M_clm)
            

            log_M    = interpolate.CubicSpline(np.log(r_integral), ln_M_clm, extrapolate = False)
            log_der  = log_M.derivative(nu = 1)(np.log(r_integral))
            lin_der  = log_der * np.exp(ln_M_clm) / r_integral
            prof     = 1/(4*np.pi*r_integral**2) * lin_der
            prof     = interpolate.PchipInterpolator(np.log(r_integral), prof, extrapolate = False)(np.log(r_use))
            
            arg  = (r_use - self.cutoff)
            arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
            kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
            prof = np.where(np.isnan(prof), 0, prof) * kfac
            prof = np.where(r_use <= R[m_i], prof, 0)

            rho_clm[m_i] = prof
        
        prof = rho_clm #Pointer just so naming is all consistent
        

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof


class DarkMatterOnly(DarkMatter):
    """
    For Arico20, the DarkMatterOnly model includes just an NFW profile.
    There is no two-halo term. This class is simply a copy of the `DarkMatter` class.
    See that class for more details
    """

class DarkMatterBaryon(Gas):

    __doc__ = S19.DarkMatterBaryon.__doc__.replace('SchneiderProfiles', 'AricoProfiles')

    def __init__(self, gas = None, stars = None, collisionlessmatter = None, **kwargs):
        
        
        self.Gas   = gas
        self.Stars = stars
        self.CollisionlessMatter = collisionlessmatter
        
        if self.Gas is None:        self.Gas        = Gas(**kwargs)        
        if self.Stars is None:      self.Stars      = Stars(**kwargs)
        if self.CollisionlessMatter is None: self.CollisionlessMatter = CollisionlessMatter(**kwargs)

        self.myprof = self.Gas + self.Stars + self.CollisionlessMatter
        

class DarkMatterOnlywithLSS(S19.DarkMatterOnly, AricoProfiles):

    __doc__ = S19.DarkMatterOnly.__doc__.replace('SchneiderProfiles', 'AricoProfiles')

    def __init__(self, darkmatter = None, twohalo = None, **kwargs):
        
        self.DarkMatter = darkmatter
        self.TwoHalo    = twohalo

        if self.DarkMatter is None: self.DarkMatter = DarkMatter(**kwargs)
        if self.TwoHalo is None: self.TwoHalo = TwoHalo(**kwargs)

        AricoProfiles.__init__(self, **kwargs)


class DarkMatterBaryonwithLSS(DarkMatterBaryon):

    __doc__ = S19.DarkMatterBaryon.__doc__.replace('SchneiderProfiles', 'AricoProfiles')

    def __init__(self, gas = None, stars = None, collisionlessmatter = None, darkmatter = None, twohalo = None, **kwargs):
        
        self.Gas   = gas
        self.Stars = stars
        self.TwoHalo    = twohalo
        self.CollisionlessMatter = collisionlessmatter
        
        if self.Gas is None:        self.Gas        = Gas(**kwargs)        
        if self.Stars is None:      self.Stars      = Stars(**kwargs)
        if self.TwoHalo is None:    self.TwoHalo    = TwoHalo(**kwargs)
        if self.CollisionlessMatter is None: self.CollisionlessMatter = CollisionlessMatter(**kwargs)

        self.myprof = self.Gas + self.Stars + self.CollisionlessMatter + self.TwoHalo
    

class Pressure(AricoProfiles):
    """
    Class for modeling the gas pressure profile in halos.

    This class extends `AricoProfiles` to compute the pressure profile of gas within halos. 
    Contrary to the  `Thermodynamic.Pressure` class in BaryonForge, this class here follows
    Arico20 and computes the pressure through a polytropic equation of state.

    Parameters
    ----------
    gas : BoundGas, optional
        Instance of the `BoundGas` class representing the bound gas component.
        If not provided, a default `BoundGas` object is created.
    **kwargs
        Additional arguments passed to initialize the parent `AricoProfiles` class and associated components.

    Notes
    -----
    - The pressure profile is derived from the bound gas density profile and the effective 
      equation of state (\( \Gamma_{\\text{eff}} \)).
    - The profile normalization is calculated based on the gravitational potential, halo 
      properties, and the gas density at the halo center.
    
    The pressure profile is given by:

    .. math::

        P(r) = P_0 \\cdot \\rho_{\\text{BG}}^{\\Gamma_{\\text{eff}}}

    where:
    - \( P_0 \) is the normalization constant, computed as:

      .. math::

          P_0 = 4 \\pi G \\cdot \\frac{\\rho_c r_s^2}{\\rho_0^{\\Gamma_{\\text{eff}} - 1}} \\cdot (1 - \\frac{1}{\\Gamma_{\\text{eff}}})

      \( \\rho_c \) is the characteristic density, \( r_s \) is the scale radius, and 
      \( \\rho_0 \) is the gas density at the center of the halo.
    - \( \\Gamma_{\\text{eff}} \) is the effective equation of state, derived from the gas 
      distribution parameters.
    - \( \\rho_{\\text{BG}}(r) \) is the bound gas density profile.
    """

    def __init__(self, gas = None, **kwargs):
        
        self.Gas = gas
        if self.Gas is None: self.Gas = BoundGas(**kwargs)        

        super().__init__(**kwargs)

    def _real(self, cosmo, r, M, a):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        if self.cdelta is None:
            c_M_relation = ccl.halos.concentration.ConcentrationDiemer15(mass_def = self.mass_def) #Use the diemer calibration
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mass_def = self.mass_def)

        #Get concentration values, and the effective equation of state, Gamma    
        c    = c_M_relation(cosmo, M_use, a)[:, None]
        r_s  = R[:, None]/c
        Norm = 4*np.pi*r_s**3 * (np.log(1 + c) - c/(1 + c))
        rhoc = M_use[:, None]/Norm
        xp   = c * self.theta_out
        Geff = 1 + ((1 + xp)*np.log(1 + xp) - xp) / ((1 + 3*xp) * np.log(1 + xp))
        
        #Normalization from Equation 5 in https://arxiv.org/pdf/2406.01672v1
        rho0  = self.Gas.real(cosmo, np.atleast_1d([0]), M_use, a) #To get normalization of gas profile
        P0    = 4*np.pi*G * (rhoc * r_s**2)/np.power(rho0, Geff - 1) * (1 - 1/Geff) 
        P0    = P0 * (Msun_to_Kg * 1e3) / (Mpc_to_m * 1e2) #Convert to CGS. Using only one factor of Mpc_to_m is correct!

        #Now compute the final profile
        rhoBG = self.Gas.real(cosmo, r_use, M_use, a)
        prof  = P0 * np.power(rhoBG, Geff)
        prof  = np.where(np.isfinite(prof), prof, 0) #Really happens when f_BG = 0 because of weird param space
        
        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = prof * kfac

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof
    

class NonThermalFrac(AricoProfiles):
    """
    Class for modeling the non-thermal pressure fraction profile in halos.

    This class extends `AricoProfiles` to compute the fraction of pressure in halos that 
    arises from non-thermal sources, such as turbulence and bulk motions. The profile is 
    based on a parametric model calibrated to simulations from Green+20, with two
    degrees of freedom.

    Notes
    -----
    - The non-thermal pressure fraction (\( f_{\\text{nt}} \)) is modeled as a function of 
      radius and halo mass, with redshift-dependent scaling.
    - The model is defined using the radius \( R_{200m} \), corresponding to the halo 
      boundary defined with respect to the matter overdensity.
    - The parameters of the model are calibrated to simulation results (e.g., Green et al. 2020).

    The non-thermal pressure fraction is given by:

    .. math::

        f_{\\text{nt}}(r) = 1 - A_{\\text{nt}} (1 + \\exp(-(x/b)^c)) 
        \\cdot \\left(\\frac{\\nu_M}{4.1}\\right)^{d / (1 + (x/e)^f)}

    where:
    - \( x = r / R_{200m} \)
    - \( A_{\\text{nt}} \) is a normalization factor that scales with redshift: 
      \( A_{\\text{nt}} = a (1 + z)^{\\alpha_{\\text{nt}}} \)
    - \( \\nu_M = 1.686 / \\sigma(M_{200m}) \) is the peak height of the halo.
    - \( b, c, d, e, f \) are model constants derived in Green et al. (2020).
    """

    def _real(self, cosmo, r, M, a):
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        
        #They define the model with R200m, so gotta use that redefinition here.
        mdef  = ccl.halos.massdef.MassDef(200, 'matter')
        cnvrt = ccl.halos.mass_translator(mass_in = self.mass_def, mass_out = mdef, concentration = 'Diemer15')
        M200m = cnvrt(cosmo, M_use, a)
        R200m = mdef.get_radius(cosmo, M_use, a)/a #in comoving distance

        x = r_use/R200m[:, None]

        nu_M = 1.686/ccl.sigmaM(cosmo, M200m, a)
        nu_M = nu_M[:, None]

        #Using "A" so no conflict with scale factor above, "a".
        #We assign A the fiducial value from Green for reference, but
        #this gets rewriten later with the custom model from Arico for the amplitude
        A, b, c, d, e, f = 0.495, 0.719, 1.417,-0.166, 0.265, -2.116 #Values from Green20
        A    = self.A_nt * np.power(1 + z, self.alpha_nt) #We override the "a" param alone for more flexibility.
        nth  = 1 - A * (1 + np.exp(-(x/b)**c)) * (nu_M/4.1)**(d/(1 + (x/e)**f))
        nth  = np.clip(nth, 0, 1)
        prof = nth #Rename just for consistency sake
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof
    
class ThermalPressure(Gas):
    """
    Convenience class for combining Pressure and NonthermalFraction.

    This class simplifies calculations by leveraging the logic and methods of these individual 
    pressure and Nth components and combining their profiles into a single representation.
    """

    def __init__(self, **kwargs): self.myprof = Pressure(**kwargs) * (1 - NonThermalFrac(**kwargs))


class Temperature(AricoProfiles):
    """
    Class for modeling the temperature profile of gas in halos.

    This class extends `AricoProfiles` to compute the temperature profile of gas, 
    based on the ideal gas law, using the pressure and gas density profiles.

    Parameters
    ----------
    pressure : Pressure, optional
        Instance of the `Pressure` class representing the gas pressure profile. 
        If not provided, a default `Pressure` object is created, accounting for 
        non-thermal pressure using `NonThermalFrac`.
    gas : BoundGas, optional
        Instance of the `BoundGas` class representing the gas density profile. 
        If not provided, a default `BoundGas` object is created.
    **kwargs
        Additional arguments passed to initialize the parent `AricoProfiles` class 
        and associated components.

    Notes
    -----
    - The real-space temperature profile is calculated using the ideal gas law:

      .. math::

          T(r) = \\frac{P(r)}{n(r) \\cdot k_B}

      where:
      - \( P(r) \) is the gas pressure profile.
      - \( n(r) \) is the gas number density, derived from the mass density and mean molecular weight.
      - \( k_B \) is the Boltzmann constant.

    - The projected temperature profile computes the average temperature along the line of sight, 
      normalizing by the number density. Thus, the projected result is still in units of temperature
      and not in units of temperature * distance.
    """

    def __init__(self, pressure = None, gas = None, **kwargs):
        
        self.Pressure = pressure
        self.Gas      = gas
        
        if self.Pressure is None: self.Pressure = Pressure(**kwargs) * (1 - NonThermalFrac(**kwargs))
        if self.Gas is None:      self.Gas      = BoundGas(**kwargs)
            
        super().__init__(**kwargs)
        
    
    def _real(self, cosmo, r, M, a):
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        P   = self.Pressure.real(cosmo, r_use, M, a)
        n   = self.Gas.real(cosmo, r_use, M, a) / (self.mean_molecular_weight * m_p) / (Mpc_to_m * m_to_cm)**3
        
        #We'll have instances of n == 0, which isn't a problem so let's ignore
        #warnings of divide errors, because we know they happen here.
        #Instead we will fix them by replacing the temperature with 0s,
        #since there is no gas in those regions to use anyway.
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            prof = P/(n * kb_cgs)
            prof = np.where(n == 0, 0, prof)
        
        return prof
    

    def projected(self, cosmo, r, M, a):
        
        P   = self.Pressure.projected(cosmo, r, M, a)
        n   = self.Gas.projected(cosmo, r, M, a) / (self.mean_molecular_weight * m_p) / (Mpc_to_m * m_to_cm)**3

        #We'll have instances of n == 0, which isn't a problem so let's ignore
        #warnings of divide errors, because we know they happen here.
        #Instead we will fix them by replacing the temperature with 0s,
        #since there is no gas in those regions to use anyway.
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            prof = P/(n * kb_cgs)
            prof = np.where(n == 0, 0, prof)

        return prof
    

class ExtendedBoundGas(AricoProfiles):
    """
    Class for modeling the extended bound gas density profile in halos.

    This class follows `BoundGas` but enforces an additional criteria that
    the profile follows an NFW form for r > r_out.

    Notes
    -----
    - The bound gas fraction (\( f_{\\text{bg}} \)) is calculated by accounting for the contributions 
      of stellar and reaccreted gas fractions, as well as the total baryon fraction.
    - The radial profile transitions from a power-law form with two characteristic radii (\( R_{\\text{inn}} \) and \( R_{\\text{out}} \)) 
      to an NFW-like behavior at large radii.
    - The profile is normalized to ensure consistency with the total bound gas mass.

    The density profile is given by:

    .. math::

        \\rho_{\\text{bg}}(r) = 
        \\begin{cases} 
        \\frac{1}{(1 + u)^\\beta (1 + v^2)^2}, & v \\leq 1 \\\\ 
        \\frac{y_1}{x (1 + x)^2}, & v > 1
        \\end{cases}

    where:
    - \( u = r / R_{\\text{inn}} \), \( v = r / R_{\\text{out}} \), and \( x = r / r_s \).
    - \( \\beta \) is the slope parameter.
    - \( R_{\\text{inn}} = \\theta_{\\text{inn}} R \), \( R_{\\text{out}} = \\theta_{\\text{out}} R \).
    - \( y_1 \) is a normalization factor for the large-scale NFW-like behavior, set by matching profiles at R_out.
    - \( r_s = R / c \) is the scale radius.
    - The profile is normalized to get the correct gas fraction within the overdensity radius, R.
    """

    def _real(self, cosmo, r, M, a):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_cg  = self._get_star_frac(M_use, z)
        f_sg  = self._get_star_frac(M_use, z, satellite = True)
        f_bar = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        beta  = self._get_gas_params(M_use, z)[0]
        f_hg  = (f_bar - f_cg - f_sg) / (1 + np.power(self.M_c/M_use, self.beta))
        f_rg  = (f_bar - f_cg - f_sg - f_hg) / (1 + np.power(self.M_r/M_use, self.beta_r))
        f_bg  = (f_hg - f_rg)

        #Get gas params
        beta, theta_out, theta_inn = self._get_gas_params(M_use, z)
        R_inn = theta_inn*R[:, None]
        R_out = theta_out*R[:, None]
        f_bg = f_bg[:, None]
        
        u = r_use/R_inn
        v = r_use/R_out

        #Now compute the large-scale behavior (which is an NFW profile)
        if self.cdelta is None:
            c_M_relation = ccl.halos.concentration.ConcentrationDiemer15(mass_def = self.mass_def) #Use the diemer calibration
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mass_def = self.mass_def)
            
        c     = c_M_relation(cosmo, M_use, a)
        r_s   = (R/c)[:, None]
        x     = r_use / r_s
        y1    = np.power(1 + R_out/R_inn, -beta)/4 * (R_out/r_s) * np.power(1 + R_out/r_s, 2)
        
        #Integrate over wider region in radii to get normalization of gas profile
        #Using a number narrower range than Schneider cause we only need to go to R200c
        r_integral = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)
        u_integral = r_integral/R_inn
        v_integral = r_integral/R_out

        prof_integral = 1/(1 + u_integral)**beta / (1 + v_integral**2)**2
        prof_integral = np.where(r_integral[None, :] < R[:, None], prof_integral, 0)
        Normalization = np.trapz(4 * np.pi * r_integral**2 * prof_integral, r_integral, axis = -1)[:, None]

        del u_integral, v_integral, prof_integral

        prof  = 1/(1 + u)**beta / (1 + v**2)**2
        nfw   = y1 / x / np.power(1 + x, 2)
        prof  = np.where(v <= 1, prof, nfw) / Normalization
        prof *= f_bg*M_use[:, None] #This profile is allowed to go beyond R200c!
        
        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = prof * kfac
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)


        return prof
    

class BoundGasDeprecated(AricoProfiles):
    """
    Deprecated class for modeling the bound gas density profile in halos.

    This class extends `AricoProfiles` to compute the density profile of bound gas within halos, 
    based on an earlier formulation. The profile incorporates both hydrostatic equilibrium 
    and a cutoff at the virial radius. It is a deprecated model in the Arico framework, and newer 
    implementations such as `BoundGas` should be used.

    Notes
    -----
    - The bound gas fraction (\( f_{\\text{bg}} \)) is calculated by accounting for the stellar fraction 
      and the total baryon fraction, scaled by a mass-dependent factor.
    - The profile transitions between a hydrostatic regime (defined by \( r / \\sqrt{\epsilon_{\\text{hydro}}} \)) 
      and an outer profile proportional to \( (1 + x)^{-2} \), ensuring continuity at the transition radius.
    - The normalization is computed by integrating the profile within the virial radius.

    The density profile is given by:

    .. math::

        \\rho_{\\text{bg}}(r) = 
        \\begin{cases} 
        \\frac{\\ln(1 + x)}{x}^{\\Gamma_{\\text{eff}}}, & r < R / \\epsilon \\\\ 
        y_1 \\cdot \\frac{1}{x (1 + x)^2}, & r \\geq R / \\epsilon \\\\ 
        0, & r > R
        \\end{cases}

    where:
    - \( x = r / r_s \), and \( r_s = R / c \) is the scale radius.
    - \( \\Gamma_{\\text{eff}} \) is the effective equation of state.
    - \( y_1 \) is a normalization factor ensuring continuity at \( R / \\epsilon \).
    - The profile is normalized by integrating over a range of radii to ensure mass conservation.

    Deprecation Warning
    --------------------
    This class represents an old version of the bound gas profile from Arico 2020. 
    Please use the latest Arico model, found in `BoundGas`.
    """

    def _real(self, cosmo, r, M, a):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1
        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_cg  = self._get_star_frac(M_use, z)
        f_bar = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        f_bg  = (f_bar - f_cg) / (1 + np.power(self.M_c/M_use, self.beta))
        f_bg  = f_bg[:, None]
        
        if self.cdelta is None:
            c_M_relation = ccl.halos.concentration.ConcentrationDiemer15(mdef = self.mass_def) #Use the diemer calibration
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mdef = self.mass_def)
            
        c    = c_M_relation.get_concentration(cosmo, M_use, a)
        r_s  = (R/c)[:, None]
        eps  = self.epsilon_hydro
        e5   = c[:, None] / eps
        Geff = (1 + 3*c/eps) * np.log(1 + c/eps) / ((1 + c/eps)*np.log(1 + c/eps) - c/eps)
        y1   = np.power(np.log(1 + e5)/e5, Geff) * (e5*(1 + e5)**2) #Set y1 based on continuity
        
        #Integrate over wider region in radii to get normalization of gas profile
        #Only go till 10Mpc since profile is cut at R200c
        r_integral = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)
        x_integral = r_integral / r_s

        u_integral = np.power(np.log(1 + x_integral)/x_integral, Geff)
        v_integral = y1 * np.power(1 + x_integral, -2)/x_integral
        y_integral = np.where(r_integral < R/eps, u_integral, v_integral)
        y_integral = np.where(r_integral > R, 0, y_integral)
        Norm       = np.trapz(4 * np.pi * r_integral**2 * y_integral, r_integral, axis = -1)[:, None]

        del r_integral, x_integral, u_integral, v_integral, y_integral

        #Now define the actual profile
        x = r_use / r_s
        u = np.power(np.log(1 + x)/x, Geff)
        v = y1 * np.power(1 + x, -2)/x
        
        prof  = np.where(r_use < R/eps, u, v)
        prof  = np.where(r_use > R, 0, prof)
        prof  = f_bg * M_use * prof / Norm

        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof *= kfac

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof