import numpy as np

from .Base import BaseBFGProfiles, hyper_params

__all__ = ['model_params', 'HyDifProfiles', 'HydrostaticGas', 'DiffuseGas', 'Gas']


model_params = ['darkmatter', 'Base', #Input profiles used to source f_gas (Base) and total mass (darkmatter)
                'M_c', 'mu', 'theta_c', 'gamma', #Component-fraction and shared shape params
                'beta_h', 'delta_h', 'theta_h', #Hydrostatic-component params
                'theta_d', 'delta_d', 'M_d', 'mu_d', #Diffuse-component params
                'cutoff', 'proj_cutoff', #Cutoff parameters (numerical)
               ]


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
    darkmatter : ccl.halos.profiles.HaloProfile
        A one-halo dark-matter profile object (e.g. `Schneider19.DarkMatter` or
        `Schneider25.DarkMatter`) used to compute the total halo mass, via numerical integration
        of `darkmatter.real(...)`, that the hydrostatic/diffuse components are normalized
        against. Required. Do not pass a `DarkMatterOnly` profile, since that includes a two-halo
        term and is not a finite halo-mass profile.
    Base : ccl.halos.profiles.HaloProfile, optional
        An optional profile object whose `get_f_gas(M, a, cosmo)` method is used as the total
        hot-gas mass fraction, :math:`f_{\\rm hga}`, that gets split between the hydrostatic and
        diffuse components (Eq. 5). If not provided, `darkmatter.get_f_gas(...)` is used instead --
        this works for any `DarkMatter` class in this package, since those inherit `get_f_gas`
        from their parent `SchneiderProfiles`/`Schneider25Profiles`-style base class. Providing
        `Base` explicitly is useful if you want the gas fraction sourced from a different
        model/definition than whatever `darkmatter` itself provides.
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

    `DM` is accepted as a backwards-compatible alias for `darkmatter`, but new code should use
    the explicit `darkmatter` name.
    """

    model_param_names = model_params
    hyper_param_names = hyper_params

    @classmethod
    def _gnfw(cls, r, r_c, r_x, beta, gamma, delta):
        """
        Evaluate the two-scale GNFW-like shape used by both HyDif components.

        .. math::

            \\left(1 + \\frac{r}{r_c}\\right)^{-\\beta}
            \\left(1 + \\left(\\frac{r}{r_x}\\right)^{\\gamma}\\right)^{-(\\delta-\\beta)/\\gamma}

        All inputs must already be broadcastable against each other (typically
        `r_c`/`r_x`/`beta`/`delta` have shape (N, 1) and `r` has shape (1, M)).
        This is a class method so that users can evaluate the shared shape
        without constructing a profile instance.
        """

        return (1 + r/r_c)**(-beta) * (1 + (r/r_x)**gamma)**(-(delta - beta)/gamma)

    @classmethod
    def _get_beta_d(cls, M_use, M_d, mu_d):
        """Return the mass-dependent inner slope of the diffuse component."""

        x = (M_use/M_d)**mu_d
        return 3*x / (1 + x)

    def __init__(self, darkmatter=None, Base=None, DM=None, **kwargs):
        """Initialize a HyDif profile with a one-halo dark-matter source.

        ``DM`` remains accepted as a compatibility alias for older code. The explicit
        ``darkmatter`` name is preferred because a two-halo ``DarkMatterOnly`` profile must not
        be used for HyDif's halo-mass normalization.
        """

        if darkmatter is not None and DM is not None and darkmatter is not DM:
            raise TypeError("Pass only one of `darkmatter` and the deprecated `DM` alias.")
        if darkmatter is None:
            darkmatter = DM
        if darkmatter is None:
            raise ValueError("HyDif requires a one-halo `darkmatter` profile.")

        # DarkMatterOnly-style profiles expose their two-halo component as `TwoHalo`. Reject
        # them early rather than silently normalizing the gas against the large-scale background.
        if getattr(darkmatter, 'TwoHalo', None) is not None:
            raise ValueError(
                "HyDif requires a one-halo `darkmatter` profile; pass the nested one-halo "
                "profile instead of a `DarkMatterOnly`/two-halo composite."
            )

        required_methods = ('real', 'set_parameter')
        if any(not hasattr(darkmatter, name) for name in required_methods) or not hasattr(darkmatter, 'cutoff'):
            raise TypeError(
                "HyDif `darkmatter` must provide `real()`, `cutoff`, and recursive "
                "`set_parameter()` methods."
            )
        if Base is not None and not hasattr(Base, 'get_f_gas'):
            raise TypeError("HyDif `Base` must provide a `get_f_gas()` method.")
        if Base is None and not hasattr(darkmatter, 'get_f_gas'):
            raise TypeError("HyDif `darkmatter` must provide `get_f_gas()` when `Base` is omitted.")

        super().__init__(darkmatter=darkmatter, Base=Base, **kwargs)

    @property
    def DM(self):
        """Backwards-compatible view of the canonical ``darkmatter`` attribute."""

        return self.darkmatter

    @DM.setter
    def DM(self, value):
        self.darkmatter = value

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
        falls back to `self.darkmatter.get_f_gas(...)`.
        """

        source = self.Base if self.Base is not None else self.darkmatter
        return source.get_f_gas(M_use, a, cosmo)

    def _get_gas_frac(self, M_use, a, cosmo):
        """Return the total gas fraction assigned to HyDif.

        HyDif treats the supplied total gas fraction as its hot-gas budget;
        the budget is then split between the hydrostatic and diffuse
        components by ``_get_f_comp``.
        """

        return self._get_f_hga(M_use, a, cosmo)

    def get_f_gas(self, M, a, cosmo):
        """Return the total gas fraction using the standard BaryonForge API."""

        fraction = self._get_gas_frac(np.atleast_1d(M), a, cosmo)
        if np.ndim(M) == 0:
            fraction = np.squeeze(fraction, axis=0)
        return fraction

    def _get_M_tot(self, cosmo, r_integral, M_use, a):
        """
        Total halo mass, obtained by numerically integrating `self.darkmatter`'s density profile.

        Temporarily widens `self.darkmatter`'s cutoff so the mass integral is not truncated by
        whatever (potentially small, e.g. FFTlog-motivated) cutoff the caller set on it,
        then restores the original value. `darkmatter` may be an externally-supplied object
        shared elsewhere in the user's pipeline, so we must not mutate it permanently. The
        temporary mutation is not thread-safe if `darkmatter` is evaluated concurrently
        elsewhere during this call.
        """

        old_cutoff = self.darkmatter.cutoff
        try:
            self.darkmatter.set_parameter('cutoff', 1e3)
            rho = self.darkmatter.real(cosmo, r_integral, M_use, a)
        finally:
            self.darkmatter.set_parameter('cutoff', old_cutoff)

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
    (Eqs. 4, 5), with :math:`M_{\\rm tot}` obtained by integrating `darkmatter`.

    Examples
    --------
    >>> darkmatter = Schneider19.DarkMatter(**bpar_S19)
    >>> gas = HydrostaticGas(darkmatter=darkmatter, **bpar_HyDif)
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
        prof_integral = self._gnfw(r_integral[None, :], R_c, R_h, self.beta_h, self.gamma, self.delta_h)
        Normalization = np.trapz(4*np.pi*r_integral**2 * prof_integral, r_integral, axis = -1)[:, None]

        M_tot = self._get_M_tot(cosmo, r_integral, M_use, a)

        arg  = (r_use[None, :] - self.cutoff)
        arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof = self._gnfw(r_use[None, :], R_c, R_h, self.beta_h, self.gamma, self.delta_h) * kfac
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
    >>> darkmatter = Schneider19.DarkMatter(**bpar_S19)
    >>> gas = DiffuseGas(darkmatter=darkmatter, **bpar_HyDif)
    >>> rho_d = gas.real(cosmo, r, M, a)
    """

    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        f_hga  = self._get_f_hga(M_use, a, cosmo)
        f_comp = self._get_f_comp(M_use)
        f_d    = (f_hga - f_hga*f_comp)[:, None]

        beta_d = self._get_beta_d(M_use, self.M_d, self.mu_d)[:, None]

        R_c = self.theta_c * R[:, None]
        R_d = self.theta_d * R[:, None]

        #Integrate over wider region in radii to get normalization of the profile
        r_integral    = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)
        prof_integral = self._gnfw(r_integral[None, :], R_c, R_d, beta_d, self.gamma, self.delta_d)
        Normalization = np.trapz(4*np.pi*r_integral**2 * prof_integral, r_integral, axis = -1)[:, None]

        M_tot = self._get_M_tot(cosmo, r_integral, M_use, a)

        arg  = (r_use[None, :] - self.cutoff)
        arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof = self._gnfw(r_use[None, :], R_c, R_d, beta_d, self.gamma, self.delta_d) * kfac
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

    def __init__(self, darkmatter=None, Base=None, DM=None, **kwargs):
        # Initialize this object as a real BFG profile so its parameters, precision settings,
        # and inherited projection/Fourier methods are present on the object being serialized.
        super().__init__(darkmatter=darkmatter, Base=Base, DM=DM, **kwargs)

        child_kwargs = {**self.model_params, **self.hyper_params}
        self.HydrostaticGas = HydrostaticGas(**child_kwargs)
        self.DiffuseGas = DiffuseGas(**child_kwargs)

    def _real(self, cosmo, r, M, a):
        """Return the sum of the hydrostatic and diffuse components."""

        return (self.HydrostaticGas.real(cosmo, r, M, a) +
                self.DiffuseGas.real(cosmo, r, M, a))
