r"""Schneider et al. (2025) baryonification density profiles.

The module implements truncated dark matter, central and satellite stars,
hot and inner gas, relaxed collisionless matter, and combined one- and
two-halo profiles. Equations in the class docstrings describe the operations
performed by the current implementation, including its finite integration
ranges, clipping, interpolation, and soft cutoff conventions.
"""

import numpy as np
import pyccl as ccl
import warnings

from scipy import interpolate, integrate
from . import Schneider19 as S19
from .Base import BaseBFGProfiles, hyper_params

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
                'c_iga', 'nu_c_iga', 'r_min_iga', #Amplitudes and inner radii for inner gas fraction
                
                'Nstar', 'Mstar', 'eta', 'eta_delta', 'tau', 'tau_delta', 'epsilon_cga', #Star params
                
                'alpha_nt', 'nu_nt', 'gamma_nt', 'mean_molecular_weight' #Non-thermal pressure and gas density
               ]


class Schneider25Profiles(BaseBFGProfiles):
    r"""Shared parameterization for the Schneider et al. (2025) profiles.

    Parameters
    ----------
    r_max_int : float, optional
        Upper radius used by internal normalizations, in comoving Mpc.
        The default is 10.
    **kwargs
        Model and numerical parameters forwarded to
        :class:`BaseBFGProfiles`.
    """

    #Define the new param names
    model_param_names = model_params
    hyper_param_names = hyper_params

    #Use a smaller r_max, since most profiles are truncated at R200c now.
    def __init__(self, r_max_int = 10, **kwargs):
        
        super().__init__(**kwargs, r_max_int = r_max_int)


    def _get_gas_params(self, M, z):
        r"""Return the mass-dependent gas-profile parameters.

        For ``c = cdelta`` when a fixed concentration is supplied, and
        ``c = 1`` otherwise, the characteristic mass is

        .. math::

            M_c(z,c) = M_c\,(1+z)^{\nu_{M_c}} c^{\zeta_{M_c}},

        and the inner gas slope is

        .. math::

            \beta(M,z,c) = 3\,\frac{(M/M_c)^{\mu_\beta}}
            {1 + (M/M_c)^{\mu_\beta}}.

        The remaining dimensionless parameters are evaluated as

        .. math::

            x(M,z,c) = x_0
            \left(\frac{M}{M_x}\right)^{\mu_x}
            (1+z)^{\nu_x} c^{\zeta_x},

        for :math:`x \in \{\theta_c,\delta,\gamma,\alpha\}`.

        Parameters
        ----------
        M : array_like
            Halo masses in solar masses.
        z : float
            Redshift.

        Returns
        -------
        beta, theta_c, delta, gamma, alpha : ndarray
            Parameter arrays with shape ``(n_mass, 1)``. ``theta_c`` is
            converted to the physical core scale through
            :math:`R_c=\theta_c R` in :class:`HotGas`.
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
    
    def _get_star_frac(self, M_use, a, cosmo):
        
        r"""Compute total, central, and satellite stellar fractions.

        The total stellar and central-galaxy fractions are

        .. math::

            f_\star(M) = \frac{N_\star}
            {(M/M_\star)^\tau + (M/M_\star)^\eta},

        .. math::

            f_{\rm cga}(M) = \frac{N_\star}
            {(M/M_\star)^{\tau+\tau_\Delta} +
             (M/M_\star)^{\eta+\eta_\Delta}}.

        The code clips :math:`f_\star` to ``[1e-10, f_bar]``, clips
        :math:`f_{\rm cga}` to ``[1e-10, f_star]``, and then sets

        .. math::

            f_{\rm sga} = \max(f_\star-f_{\rm cga},10^{-10}),

        where :math:`f_{\rm bar}=\Omega_b/\Omega_m`.

        Parameters
        ----------
        M_use : array_like
            Halo masses in solar masses.
        a : float
            Scale factor. It is accepted for a common profile interface but
            does not enter these equations.
        cosmo : object
            Cosmology wrapper providing ``Omega_b`` and ``Omega_m``.

        Returns
        -------
        f_star, f_cga, f_sga : ndarray
            Total stellar, central-galaxy, and satellite-galaxy mass
            fractions.
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
        
        f_sga  = np.clip(f_star - f_cga, 1e-10, None) 
        
        return f_star, f_cga, f_sga
    
    
    def get_f_star(self, M_use, a, cosmo):
        return self._get_star_frac(M_use, a, cosmo)[0]
    
    def get_f_star_cen(self, M_use, a, cosmo):
        return self._get_star_frac(M_use, a, cosmo)[1]
    
    def get_f_star_sat(self, M_use, a, cosmo):
        return self._get_star_frac(M_use, a, cosmo)[2] 
    

    def _get_gas_frac(self, M_use, a, cosmo):

        r"""Compute the hot- and inner-gas mass fractions.

        The inner-gas fraction is tied to the central stellar fraction,

        .. math::

            f_{\rm iga}(M,a) = f_{\rm cga}(M)\,c_{\rm iga}\,
            a^{-\nu_{c_{\rm iga}}},

        and is clipped to ``[1e-10, f_bar - f_star]``. The remaining hot-gas
        fraction is

        .. math::

            f_{\rm hga} = f_{\rm bar} - f_\star - f_{\rm iga},

        clipped to ``[1e-10, f_bar]``.

        Parameters
        ----------
        M_use : array_like
            Halo masses in solar masses.
        a : float
            Scale factor.
        cosmo : object
            Cosmology wrapper providing ``Omega_b`` and ``Omega_m``.

        Returns
        -------
        f_hga, f_iga : ndarray
            Hot- and inner-gas mass fractions.
        """

        
        f_star = self.get_f_star(M_use, a, cosmo)
        f_cga  = self.get_f_star_cen(M_use, a, cosmo)

        f_bar  = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        f_iga  = f_cga * self.c_iga * np.power(a, -self.nu_c_iga) #-ve sign since we do a^nu instead of (1 + z)^nu
        f_iga  = np.clip(f_iga, 1e-10, f_bar - f_star)
        f_hga  = np.clip(f_bar - f_star - f_iga, 1e-10, f_bar) #Cannot let the fraction be identically 0.        
        
        return f_hga, f_iga
        

    def get_f_gas(self, M, a, cosmo):
        f = self._get_gas_frac(self, M, a, cosmo)
        return f[0] + f[1]


    def _get_dm_eps(self, M_use, a, cosmo):
        r"""Return ``max(epsilon0 - epsilon1 * nu, 1e-3)``.

        Here :math:`\nu=1.686/\sigma(M,a)`. The floor prevents a vanishing
        truncation radius.
        """

        nu  = 1.686/ccl.sigmaM(cosmo, M_use, a)
        eps = self.epsilon0 - self.epsilon1 * nu #The R200c version of Eqn 6 in https://arxiv.org/pdf/1401.1216. 
                                                    #Eqn 2.8 in https://arxiv.org/pdf/2507.07892 has the wrong sign
        eps = np.clip(eps, 1e-3, None) #Guard against r_t == 0.

        return eps
        
        
class DarkMatter(Schneider25Profiles):
    r"""Truncated one-halo dark-matter density profile.

    For halo radius :math:`R`, concentration :math:`c`, and peak height
    :math:`\nu=1.686/\sigma(M,a)`, the code defines

    .. math::

        r_s = \frac{R}{c}, \qquad
        \epsilon(\nu)=\max(\epsilon_0-\epsilon_1\nu,10^{-3}),
        \qquad r_t=\epsilon R.

    The unnormalized radial shape is

    .. math::

        u_{\rm dm}(r) =
        \frac{1}{(r/r_s)(1+r/r_s)^2}
        \frac{1}{\left[1+(r/r_t)^2\right]^2}.

    For each mass, the normalization is evaluated only up to the halo radius,

    .. math::

        N_{\rm dm}(M) = \int_{r_{\min}}^R 4\pi r^2
        u_{\rm dm}(r)\,dr,

    and :math:`\rho_c=M/N_{\rm dm}`. The returned profile is

    .. math::

        \rho_{\rm dm}(r) = \rho_c u_{\rm dm}(r)
        \left[1+\exp\{2(r-r_{\rm cutoff})\}\right]^{-1}.

    If neither ``cdelta`` nor ``c_M_relation`` is supplied, the
    Diemer--Kravtsov (2015) concentration relation is used. Non-finite
    concentrations are replaced by one.
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
        eps = self._get_dm_eps(M_use, a, cosmo)
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
    r"""Two-halo density term with small-scale exclusion.

    The profile is evaluated from the linear matter correlation function as

    .. math::

        \rho_{2h}(r|M) = f_{\rm excl}(r|M)
        \left[1+b(M)\xi_{mm}(r)\right]\bar\rho_m
        k_{\rm cut}(r),

    where

    .. math::

        f_{\rm excl}(r|M) = 1-
        \exp\left[-\alpha_{\rm excl}
        \operatorname{clip}\left(\frac{r}{R},0,30\right)\right],

    .. math::

        b(M)=1+\frac{q\nu^2-1}{\delta_c}
        +\frac{2p}{\delta_c\left[1+(q\nu^2)^p\right]},
        \qquad \nu=\frac{\delta_c}{\sigma(M,a)},
        \qquad \delta_c=1.686,

    and

    .. math::

        k_{\rm cut}(r)=
        \left[1+\exp\{2(r-r_{\rm cutoff})\}\right]^{-1}.

    ``xi_mm`` may be supplied as a callable; otherwise
    :func:`pyccl.correlation_3d` is used. The CCL cosmology must be configured
    with ``matter_power_spectrum='linear'``.
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

        #Bias via Eqn 12 in https://arxiv.org/pdf/astro-ph/9901122
        delta_c = 1.686
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


class Stars(Schneider25Profiles):
    r"""Central-galaxy stellar density profile.

    The central scale radius is

    .. math::

        R_{\rm cga}=\epsilon_{\rm cga}R,

    and the unnormalized stellar shape is

    .. math::

        u_{\rm cga}(r)=r^{-2}\exp(-r/R_{\rm cga}).

    The reference halo mass :math:`M_{\rm tot}` is obtained by integrating a
    :class:`DarkMatter` profile over ``[r_min_int, r_max_int]`` with its
    numerical cutoff moved to a very large radius. With

    .. math::

        N_{\rm cga}=\int_{r_{\min}}^{r_{\max}}
        4\pi r^2 u_{\rm cga}(r)\,dr,

    the returned profile is

    .. math::

        \rho_{\rm cga}(r)=
        \frac{f_{\rm cga}M_{\rm tot}}{N_{\rm cga}}
        u_{\rm cga}(r)k_{\rm cut}(r).

    The normalization is computed before applying the final soft cutoff. The
    class also uses broad FFTLog padding to reduce ringing from the compact
    stellar profile.
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

        f_cga  = self.get_f_star_cen(M_use, a, cosmo)[:, None]
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


class HotGas(Schneider25Profiles):

    r"""Hot-gas density profile.

    For :math:`R_c=\theta_cR` and :math:`R_t=\epsilon R`, define

    .. math::

        u=\frac{r}{R_c}, \qquad v=\frac{r}{R_t}.

    The generalized gas shape implemented here is

    .. math::

        u_{\rm hga}(r)=
        \left(1+u^\alpha\right)^{-\beta/\alpha}
        \left(1+v^\gamma\right)^{-\delta/\gamma}.

    The dark-matter reference mass :math:`M_{\rm tot}` is obtained by
    integrating :class:`DarkMatter` over the internal radial range. With

    .. math::

        N_{\rm hga}=\int_{r_{\min}}^{r_{\max}}
        4\pi r^2u_{\rm hga}(r)\,dr,

    the returned density is

    .. math::

        \rho_{\rm hga}(r)=
        \frac{f_{\rm hga}M_{\rm tot}}{N_{\rm hga}}
        u_{\rm hga}(r)k_{\rm cut}(r).

    The mass-, redshift-, and concentration-dependent values of
    :math:`\beta,\theta_c,\delta,\gamma`, and :math:`\alpha` are supplied by
    :meth:`Schneider25Profiles._get_gas_params`.
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
        eps = self._get_dm_eps(M_use, a, cosmo)[:, None]
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
        prof *= f_hga[:, None]*M_tot/Normalization
        

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof
    

class InnerGas(Schneider25Profiles):

    r"""Centrally concentrated, cooled inner-gas density profile.

    The large-radius shape is formally

    .. math::

        u_{\rm iga}^{\rm outer}(r)=r^{-3}\exp(-r/R),

    which is ultraviolet divergent. The implementation regularizes it around
    ``r_min_iga`` using

    .. math::

        w(r)=\frac{1}{2}\left[1+\tanh\left(
        \frac{\log_{10}(r/r_{\min,\rm iga})}{0.02}\right)\right]

    and

    .. math::

        u_{\rm iga}(r)=(1-w)h^3+w\,r^{-3}\exp(-r/R).

    This uncut shape is used for the normalization

    .. math::

        N_{\rm iga}=\int_{r_{\min}}^{r_{\max}}
        4\pi r^2u_{\rm iga}(r)\,dr.

    In the returned profile the soft cutoff multiplies only the outer branch,

    .. math::

        \rho_{\rm iga}(r)=\frac{f_{\rm iga}M_{\rm tot}}{N_{\rm iga}}
        \left[(1-w)h^3+w\,r^{-3}e^{-r/R}k_{\rm cut}(r)\right].

    ``h`` is read from the CCL cosmology, and :math:`M_{\rm tot}` is obtained
    by integrating the corresponding :class:`DarkMatter` profile.
    """


    def _real(self, cosmo, r, M, a):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1
        h = cosmo['h']
        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_hga, f_iga  = self._get_gas_frac(M_use, a, cosmo)
        
        #Integrate over wider region in radii to get normalization of gas profile
        r_integral = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)
        
        #This profile is formally UV divergent. The enclosed mass goes to
        #infinity if you integrate down to 0. If you set a finite minimum radius
        #then the profile depends critically on this minimum radius. We set this
        #as a free parameter, but its value is generally 5kpc. This is the
        #choice made in Schneider+2025 (private comm.)
        #
        #Below this scale, I need to set the profile to be h^3. This is an odd choice
        #to me but is the chosen definition of Schneider+25, so I mimic it here
        #in the interest of reproducability of the original results.
        prof_integral = np.power(r_integral, -3) * np.exp(-r_integral/R[:, None])
        weight        = 0.5 * (1 + np.tanh(np.log10(r_integral/self.r_min_iga)/0.02))
        prof_integral = (1 - weight)*h**3 + weight*prof_integral
        Normalization = np.trapz(4 * np.pi * r_integral**2 * prof_integral, r_integral, axis = -1)[:, None]

        DM    = DarkMatter(**self.model_params); setattr(DM, 'cutoff', 1e3) #Set large cutoff just for normalization calculation
        rho   = DM.real(cosmo, r_integral, M_use, a)
        M_tot = np.trapz(4*np.pi*r_integral**2 * rho, r_integral, axis = -1)
        M_tot = np.atleast_1d(M_tot)[:, None]
        
        arg   = (r_use[None, :] - self.cutoff)
        arg   = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac  = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof  = np.power(r_use, -3) * np.exp(-r_use/R[:, None]) * kfac
        wgt   = 0.5 * (1 + np.tanh(np.log10(r_use/self.r_min_iga)/0.02))
        prof  = (1 - wgt)*h**3 + wgt*prof
        prof *= f_iga[:, None]*M_tot/Normalization
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof


class Gas(Schneider25Profiles):
    r"""Combined Schneider25 gas profile.

    This convenience wrapper returns the sum

    .. math::

        \rho_{\rm gas}(r)=\rho_{\rm hga}(r)+\rho_{\rm iga}(r),

    using :class:`HotGas` and :class:`InnerGas` initialized from the same
    keyword arguments. Attribute access is delegated to the summed profile.
    """

    def __init__(self, **kwargs): self.myprof = HotGas(**kwargs) + InnerGas(**kwargs)
    def __getattr__(self, name):  return getattr(self.myprof, name)
    
    #Need to explicitly set these two methods (to enable pickling)
    #since otherwise the getattr call above leads to infinite recursions.
    def __getstate__(self): self.__dict__.copy()    
    def __setstate__(self, state): self.__dict__.update(state)


class CollisionlessMatter(Schneider25Profiles):

    r"""Relaxed collisionless-matter density profile.

    The collisionless fraction is

    .. math::

        f_{\rm clm}=1-\frac{\Omega_b}{\Omega_m}+f_{\rm sga}.

    For the initial dark-matter cumulative mass :math:`M_i(r)` and cumulative
    component masses :math:`M_{\rm cga}`, :math:`M_{\rm iga}`, and
    :math:`M_{\rm hga}`, the code defines

    .. math::

        \mathcal{R}(r)=1+\xi_0+\xi_1+\xi_2+\xi_3,

    with

    .. math::

        \xi_0=\frac{Q_0}{1+(r/r_{\rm step})^{n_{\rm step}}},
        \qquad r_{\rm step}=\frac{\epsilon}{\epsilon_0}R,

    .. math::

        \xi_1=Q_1\left(\frac{M_{\rm cga}}{M_i}-f_{\rm cga}\right),
        \qquad
        \xi_2=Q_1\left(\frac{M_{\rm iga}}{M_i}-f_{\rm iga}\right),

    .. math::

        \xi_3=Q_2\left(\frac{M_{\rm hga}}{M_i}-f_{\rm hga}\right).

    The Schneider25 convention is :math:`\mathcal{R}=r_i/r_f`, so the final
    cumulative collisionless mass at radius :math:`r_f` is

    .. math::

        M_{\rm clm}(r_f)=f_{\rm clm}
        M_i\!\left(r_f\mathcal{R}(r_f)\right).

    The cumulative component masses are obtained by integrating their density
    profiles on a logarithmic radial grid. A cubic spline is then constructed
    for :math:`\ln M_{\rm clm}` as a function of :math:`\ln r`, and the density
    is recovered through

    .. math::

        \rho_{\rm clm}(r)=\frac{1}{4\pi r^2}
        \frac{dM_{\rm clm}}{dr}.

    Negative values introduced by interpolation are clipped to zero, and the
    final soft cutoff is applied. A warning is emitted when requested radii
    lie outside the internal integration range.

    Parameters
    ----------
    hotgas, innergas, stars, darkmatter : object, optional
        Component profiles. Missing components are constructed from ``kwargs``.
    r_min_int, r_max_int : float, optional
        Internal radial integration limits in comoving Mpc.
    r_steps : int, optional
        Number of logarithmically spaced integration samples.
    **kwargs
        Shared model and numerical parameters.
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
        r"""Return the redshift-dependent relaxation amplitudes.

        The implemented relations are :math:`Q_i(z)=Q_i+\nu_{Q_i}z` for
        ``i = 0, 1, 2``. ``M`` and ``cosmo`` are accepted for interface
        consistency but do not enter these equations.
        """

        z  = 1/a - 1
        Q0 = self.q0 + self.nu_q0 * z
        Q1 = self.q1 + self.nu_q1 * z
        Q2 = self.q2 + self.nu_q2 * z

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

        f_cga, f_sga  = self.get_f_star_cen(M_use, a, cosmo), self.get_f_star_sat(M_use, a, cosmo)
        f_hga, f_iga  = self._get_gas_frac(M_use, a, cosmo)
        Q0, Q1, Q2    = self._get_Qis(M_use, a, cosmo)

        f_clm      = 1 - cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m + f_sga[:, None]
        eps        = self._get_dm_eps(M_use, a, cosmo)
        rstep      = eps / self.epsilon0 * R
        
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

                #The masses (M_cga, M_iga) already contain the factor of f_cga, so we shouldn't remultiply
                xi0  = Q0 / (1 + np.power(r_integral/rstep[m_i], self.nstep))
                xi1  = Q1 * (M_cga[m_i] / M_i[m_i] - f_cga[m_i])
                xi2  = Q1 * (M_iga[m_i] / M_i[m_i] - f_iga[m_i])
                xi3  = Q2 * (M_hga[m_i] / M_i[m_i] - f_hga[m_i])
                relaxation_fraction = xi0 + xi1 + xi2 + xi3 + 1

                #Schneider+25 defines relaxation fraction as r_i/r_f so the bottom should indeed be multiplied,
                #and not divided like we do in Schneider+19, where the definition was r_f/r_i.
                ln_M_clm[m_i] = np.log(f_clm[m_i]) + ln_M_NFW[m_i](np.log(r_integral * relaxation_fraction))

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
    

class SatelliteStars(CollisionlessMatter):

    r"""Satellite-galaxy stellar density profile.

    Satellite stars are assumed to trace the relaxed collisionless component.
    The profile is therefore

    .. math::

        \rho_{\rm sga}(r)=\rho_{\rm clm}(r)
        \frac{f_{\rm sga}}{f_{\rm clm}},
        \qquad
        f_{\rm clm}=1-\frac{\Omega_b}{\Omega_m}+f_{\rm sga}.
    """
    
    def _real(self, cosmo, r, M, a):

        M_use = np.atleast_1d(M)

        f_sga  = self.get_f_star_sat(M_use, a, cosmo)[:, None]
        f_clm  = 1 - cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m + f_sga
        
        if np.ndim(M) == 0: 
            f_clm = np.squeeze(f_clm, axis = 0)
            f_sga = np.squeeze(f_sga, axis = 0)

        prof   = super()._real(cosmo, r, M, a) * (f_sga/f_clm)
        
        return prof


class DarkMatterOnly(Schneider25Profiles):

    r"""Dark-matter-only profile including the two-halo term.

    The returned density is the direct sum

    .. math::

        \rho_{\rm DMO}(r)=\rho_{\rm dm}(r)+\rho_{2h}(r),

    where :class:`DarkMatter` supplies the truncated one-halo profile and
    :class:`TwoHalo` supplies the excluded large-scale contribution.

    Parameters
    ----------
    darkmatter : DarkMatter, optional
        One-halo dark-matter profile. Constructed from ``kwargs`` when absent.
    twohalo : TwoHalo, optional
        Two-halo density profile. Constructed from ``kwargs`` when absent.
    **kwargs
        Shared model and numerical parameters.
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


class DarkMatterBaryon(Schneider25Profiles):

    r"""Total dark-matter-plus-baryon density profile.

    The unnormalized one-halo baryonified density is

    .. math::

        \rho_{1h}^{\rm raw}(r)=
        \rho_{\rm clm}(r)+\rho_{\rm cga}(r)+\rho_{\rm gas}(r).

    Over the internal normalization interval, the code computes

    .. math::

        M_{\rm dm}=\int 4\pi r^2\rho_{\rm dm}(r)\,dr,
        \qquad
        M_{\rm dmb}^{\rm raw}=\int 4\pi r^2
        \rho_{1h}^{\rm raw}(r)\,dr,

    and rescales all one-halo baryonified components by

    .. math::

        A_M=\frac{M_{\rm dm}}{M_{\rm dmb}^{\rm raw}}.

    The final profile is

    .. math::

        \rho_{\rm DMB}(r)=A_M\left[
        \rho_{\rm clm}(r)+\rho_{\rm cga}(r)+\rho_{\rm gas}(r)\right]
        +\rho_{2h}(r).

    Thus the normalization is applied only to the one-halo contribution; the
    two-halo term is added afterward.

    Parameters
    ----------
    gas : Gas, optional
        Combined hot- and inner-gas profile.
    stars : Stars, optional
        Central-galaxy stellar profile.
    collisionlessmatter : CollisionlessMatter, optional
        Relaxed collisionless-matter profile.
    darkmatter : DarkMatter, optional
        Reference one-halo dark-matter profile used for mass normalization.
    twohalo : TwoHalo, optional
        Two-halo density term.
    r_min_int, r_max_int : float, optional
        Radial limits used to determine the one-halo normalization.
    r_steps : int, optional
        Number of logarithmically spaced normalization samples.
    **kwargs
        Shared model and numerical parameters.
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