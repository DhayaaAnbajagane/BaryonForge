import numpy as np
import pyccl as ccl

from .Base import BaseBFGProfiles, hyper_params

__all__ = ['model_params', 'HyDifProfiles', 'HydrostaticGas', 'DiffuseGas', 'Gas']


model_params = ['DM', 'Base', #Input profiles used to source f_gas (Base) and total mass (DM)
                'M_c', 'mu', 'theta_c', 'gamma', #Component-fraction and shared shape params
                'beta_h', 'delta_h', 'theta_h', #Hydrostatic-component params
                'theta_d', 'delta_d', 'M_d', 'mu_d', #Diffuse-component params
                'cutoff', 'proj_cutoff', #Cutoff parameters (numerical)
               ]


def _gnfw_shape(r, r_c, r_x, beta, gamma, delta):
    """
    Generic two-scale GNFW-like shape shared by the hydrostatic and diffuse
    HyDif components:

    .. math::

        \\left(1 + \\frac{r}{r_c}\\right)^{-\\beta}
        \\left(1 + \\left(\\frac{r}{r_x}\\right)^{\\gamma}\\right)^{-(\\delta-\\beta)/\\gamma}

    All inputs must already be broadcastable against each other (typically
    `r_c`/`r_x`/`beta`/`delta` have shape (N, 1) and `r` has shape (1, M)).
    """

    return (1 + r/r_c)**(-beta) * (1 + (r/r_x)**gamma)**(-(delta - beta)/gamma)


def _get_beta_d(M_use, M_d, mu_d):
    """
    Mass-dependent outer slope of the diffuse component (Eq. 9 of Shavelle et al. 2026).
    """

    x = (M_use/M_d)**mu_d
    return 3*x / (1 + x)


class HyDifProfiles(BaseBFGProfiles):
    """
    Shared base class for the HyDif two-component hot-gas model of
    `Shavelle et al. 2026 <https://arxiv.org/abs/2609.21861>`_.

    HyDif decomposes the usual single-component baryonification hot-gas profile into an
    approximately hydrostatic inner component (`HydrostaticGas`) and a diffuse, extended
    outer component (`DiffuseGas`), motivated by the complementary sensitivity of X-ray
    (density-squared, inner halo) and kSZ (density, outer halo) observations. See
    `HydrostaticGas` and `DiffuseGas` for the individual profile shapes, and `Gas` for the
    combined profile.

    This class is deliberately agnostic to which baryonification model (Schneider19,
    Schneider25, or otherwise) it is used alongside. It does not import or depend on any
    other `BaryonForge.Profiles` module. Instead, it takes two external profile objects:

    Parameters
    ----------
    DM : ccl.halos.profiles.HaloProfile
        A dark-matter profile object (e.g. `Schneider19.DarkMatter` or `Schneider25.DarkMatter`)
        used to compute the total halo mass, via numerical integration of `DM.real(...)`, that
        the hydrostatic/diffuse components are normalized against. Required.
    Base : ccl.halos.profiles.HaloProfile, optional
        An optional profile object whose `get_f_gas(M, a, cosmo)` method is used as the total
        hot-gas mass fraction, :math:`f_{\\rm hga}`, that gets split between the hydrostatic and
        diffuse components (Eq. 5). If not provided, `DM.get_f_gas(...)` is used instead --
        this works for any `DarkMatter` class in this package, since those inherit `get_f_gas`
        from their parent `SchneiderProfiles`/`Schneider25Profiles`-style base class. Providing
        `Base` explicitly is useful if you want the gas fraction sourced from a different
        model/definition than whatever `DM` itself provides.
    **kwargs
        Additional keyword arguments; see `SchneiderProfiles`-family free parameters below,
        and `BaseBFGProfiles` for hyperparameters (`mass_def`, `r_min_int`, etc.).

    Notes
    -----
    Free/shape parameters (all set via `model_params`, following the `f_comp`/`beta_d(M)`
    formulas of Eqs. 4 and 9):

    - `M_c`, `mu` : set the component fraction, :math:`f_{\\rm comp}(M) = (M/M_c)^\\mu / (1 + (M/M_c)^\\mu)`
      (Eq. 4), i.e. the fraction of the hot gas budget in the hydrostatic component.
    - `theta_c` : core radius parameter, shared by both components, :math:`r_c = \\theta_c R_{200}`.
    - `gamma` : shared intermediate-to-outer transition sharpness, used by both components.
    - `beta_h`, `delta_h`, `theta_h` : hydrostatic-component shape (Eq. 7).
    - `theta_d`, `delta_d`, `M_d`, `mu_d` : diffuse-component shape and its mass-dependent
      inner slope, :math:`\\beta_d(M)` (Eq. 8, 9).
    """

    model_param_names = model_params
    hyper_param_names = hyper_params

    def _get_f_comp(self, M_use):
        """
        Fraction of the total hot-gas mass residing in the hydrostatic component (Eq. 4).

        Parameters
        ----------
        M_use : ndarray, shape (N,)
            Halo mass.

        Returns
        -------
        f_comp : ndarray, shape (N,)
        """

        x = (M_use/self.M_c)**self.mu
        return x / (1 + x)

    def _get_f_hga(self, M_use, a, cosmo):
        """
        Total hot-gas mass fraction, :math:`f_{\\rm hga}`, used to normalize the combined
        hydrostatic + diffuse profile (Eq. 5).

        Uses `self.Base.get_f_gas(...)` if `Base` was provided at initialization, otherwise
        falls back to `self.DM.get_f_gas(...)`.
        """

        assert self.DM is not None, "Must provide a `DM` (DarkMatter-like) profile object to HyDif profiles."

        source = self.Base if self.Base is not None else self.DM
        return source.get_f_gas(M_use, a, cosmo)

    def _get_M_tot(self, cosmo, r_integral, M_use, a):
        """
        Total halo mass, obtained by numerically integrating `self.DM`'s density profile.

        Temporarily widens `self.DM`'s cutoff so the mass integral is not truncated by
        whatever (potentially small, e.g. FFTlog-motivated) cutoff the caller set on it,
        then restores the original value. `DM` may be an externally-supplied object shared
        elsewhere in the user's pipeline (e.g. also passed as `darkmatter=` to a
        `CollisionlessMatter`), so we must not mutate it permanently. Not thread-safe if
        `DM` is evaluated concurrently elsewhere during this call.
        """

        old_cutoff = self.DM.cutoff
        try:
            self.DM.cutoff = 1e3
            rho = self.DM.real(cosmo, r_integral, M_use, a)
        finally:
            self.DM.cutoff = old_cutoff

        M_tot = np.atleast_1d(np.trapz(4*np.pi*r_integral**2 * rho, r_integral, axis = -1))[:, None]
        return M_tot


class HydrostaticGas(HyDifProfiles):
    """
    Hydrostatic (inner) component of the HyDif hot-gas decomposition.

    See `HyDifProfiles` for shared parameters/methods.

    Notes
    -----
    The hydrostatic component shape is fixed (not mass-dependent), following the fiducial
    values of Giri & Schneider (2021) adopted by Shavelle et al. (2026):

    .. math::

        \\rho_h(r) \\propto \\left(1 + \\frac{r}{r_c}\\right)^{-\\beta_h}
        \\left[1 + \\left(\\frac{r}{r_h}\\right)^{\\gamma}\\right]^{-(\\delta_h - \\beta_h)/\\gamma}

    where :math:`r_c = \\theta_c R_{200}` and :math:`r_h = \\theta_h R_{200}`. The profile is
    normalized so that its integrated mass equals :math:`f_h M_{\\rm tot} = f_{\\rm hga} f_{\\rm comp} M_{\\rm tot}`
    (Eqs. 4, 5), with :math:`M_{\\rm tot}` obtained by integrating `DM`.

    Examples
    --------
    >>> DM  = Schneider19.DarkMatter(**bpar_S19)
    >>> gas = HydrostaticGas(DM=DM, **bpar_HyDif)
    >>> rho_h = gas.real(cosmo, r, M, a)
    """

    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_hga  = self._get_f_hga(M_use, a, cosmo)
        f_comp = self._get_f_comp(M_use)
        f_h    = (f_hga * f_comp)[:, None]

        R_c = self.theta_c * R[:, None]
        R_h = self.theta_h * R[:, None]

        #Integrate over wider region in radii to get normalization of the profile
        r_integral    = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)
        prof_integral = _gnfw_shape(r_integral[None, :], R_c, R_h, self.beta_h, self.gamma, self.delta_h)
        Normalization = np.trapz(4*np.pi*r_integral**2 * prof_integral, r_integral, axis = -1)[:, None]

        M_tot = self._get_M_tot(cosmo, r_integral, M_use, a)

        arg  = (r_use[None, :] - self.cutoff)
        arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof = _gnfw_shape(r_use[None, :], R_c, R_h, self.beta_h, self.gamma, self.delta_h) * kfac
        prof = prof * f_h*M_tot/Normalization

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof


class DiffuseGas(HyDifProfiles):
    """
    Diffuse (outer) component of the HyDif hot-gas decomposition.

    See `HyDifProfiles` for shared parameters/methods.

    Notes
    -----
    Unlike `HydrostaticGas`, the diffuse component's inner slope is mass-dependent:

    .. math::

        \\rho_d(r) \\propto \\left(1 + \\frac{r}{r_c}\\right)^{-\\beta_d(M)}
        \\left[1 + \\left(\\frac{r}{r_d}\\right)^{\\gamma}\\right]^{-(\\delta_d - \\beta_d(M))/\\gamma}

    .. math::

        \\beta_d(M) = \\frac{3(M/M_d)^{\\mu_d}}{1 + (M/M_d)^{\\mu_d}}

    where :math:`r_c = \\theta_c R_{200}` (shared with `HydrostaticGas`) and
    :math:`r_d = \\theta_d R_{200}`. The profile is normalized so that its integrated mass
    equals :math:`f_d M_{\\rm tot} = (f_{\\rm hga} - f_h) M_{\\rm tot}` (Eq. 5).

    Examples
    --------
    >>> DM  = Schneider19.DarkMatter(**bpar_S19)
    >>> gas = DiffuseGas(DM=DM, **bpar_HyDif)
    >>> rho_d = gas.real(cosmo, r, M, a)
    """

    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_hga  = self._get_f_hga(M_use, a, cosmo)
        f_comp = self._get_f_comp(M_use)
        f_d    = (f_hga - f_hga*f_comp)[:, None]

        beta_d = _get_beta_d(M_use, self.M_d, self.mu_d)[:, None]

        R_c = self.theta_c * R[:, None]
        R_d = self.theta_d * R[:, None]

        #Integrate over wider region in radii to get normalization of the profile
        r_integral    = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)
        prof_integral = _gnfw_shape(r_integral[None, :], R_c, R_d, beta_d, self.gamma, self.delta_d)
        Normalization = np.trapz(4*np.pi*r_integral**2 * prof_integral, r_integral, axis = -1)[:, None]

        M_tot = self._get_M_tot(cosmo, r_integral, M_use, a)

        arg  = (r_use[None, :] - self.cutoff)
        arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof = _gnfw_shape(r_use[None, :], R_c, R_d, beta_d, self.gamma, self.delta_d) * kfac
        prof = prof * f_d*M_tot/Normalization

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof


class Gas(HyDifProfiles):
    """
    Convenience class combining the hydrostatic and diffuse HyDif gas components into the
    total hot-gas profile, `Gas = HydrostaticGas + DiffuseGas`.

    This is the class most users want: it self-consistently returns the sum of both
    components, conserving the total hot-gas mass fraction `f_hga` (Eq. 5, 6 of
    Shavelle et al. 2026). See `HydrostaticGas`/`DiffuseGas` if you want either component
    on its own.
    """

    def __init__(self, **kwargs):
        self.myprof = HydrostaticGas(**kwargs) + DiffuseGas(**kwargs)

    def __getattr__(self, name):
        return getattr(self.myprof, name)

    #Need to explicitly set these two methods (to enable pickling)
    #since otherwise the getattr call above leads to infinite recursions.
    def __getstate__(self): return self.__dict__.copy()
    def __setstate__(self, state): return self.__dict__.update(state)
