import numpy as np
import pyccl as ccl
from operator import add, mul, sub, truediv, pow, neg, pos, abs
from .Base import BaseBFGProfiles, hyper_params
from scipy import interpolate
from ..utils.Tabulate import _set_parameter, _get_parameter
from ..utils.misc import combine_fftpars
from pyccl.pyutils import resample_array, _fftlog_transform
fftlog = _fftlog_transform

__all__ = ['Truncation', 'Identity', 'Zeros', 'ComovingToPhysical', 'Mdelta_to_Mtot', 'CombinedProfile']


#CCL implementation hooks. These must never be delegated from a container/composite
#profile to one of its inputs, since CCL uses their presence to decide how to compute
#real/fourier/projected profiles.
_CCL_HOOKS = ('_real', '_fourier', '_projected', '_cumul2d', '_fftlog_wrap', '_projected_fftlog_wrap')


def _has_custom_method(profile, method):
    """
    Returns True if the `projected` or `fourier` method of a profile does something other than
    the default: the real-space projection of `BaseBFGProfiles`, or the FFTLog of the real profile.
    """

    Haloprofile = ccl.halos.profiles.HaloProfile

    #Public method has been overriden (eg. ComovingToPhysical, ConvolvedProfile, Temperature)
    public = getattr(profile, method, None)
    if getattr(public, '__func__', None) is not getattr(Haloprofile, method): return True

    if method == 'projected':
        hook = getattr(profile, '_projected', None)
        return (hook is not None) and (getattr(hook, '__func__', None) is not BaseBFGProfiles._projected_realspace)
    else:
        return getattr(profile, '_fourier', None) is not None


class CombinedProfile(BaseBFGProfiles):
    """
    Profile obtained from an arithmetic operation between profiles, or between a profile and a number.

    This class is what the arithmetic operators (``+``, ``-``, ``*``, ``/``, ``**``, unary ``-``, ``+``
    and ``abs``) of every BaryonForge profile return. It is rarely constructed directly.

    The input profiles are stored as attributes (``Profile1``, ``Profile2``), so that
    `set_parameter`, `update_precision_fftlog`, tabulation over parameters, and pickling all
    reach them. Attributes that are not found on the combined profile (eg. model parameters,
    or methods like ``get_f_gas``) are looked up on ``Profile1``, the profile the operator was
    called on.

    Parameters
    ----------
    op : callable
        One of ``operator.add, sub, mul, truediv, pow, neg, pos, abs``.
    Profile1 : ccl.halos.profiles.HaloProfile
        The profile whose operator was called.
    Profile2 : ccl.halos.profiles.HaloProfile, int, or float, optional
        The second operand. Not needed for unary operators.
    reflect : bool, optional
        If True, the operation is ``op(Profile2, Profile1)`` (eg. ``2 - profile``).

    Notes
    -----
    - The real-space profile is always ``op`` applied to the ``real()`` outputs of the inputs.
    - For scaling by a number (and negation), the projected and Fourier profiles are the scaled
      ``projected()`` and ``fourier()`` outputs of the input. This is exact, and respects any
      custom projection that the input implements (eg. `ComovingToPhysical`, `ConvolvedProfile`).
    - For sums/differences of profiles, the same is done if either input has a custom projection
      (or Fourier transform). Otherwise, the combined real profile is projected (or FFTLog'd)
      directly, which avoids numerical cancellation between large terms (eg. ``DMB - TwoHalo``).
    - For other (non-linear) operations, the projected profile is the real-space projection of
      the combined real profile, and the Fourier profile is the FFTLog of it.
    - Numerical settings (``mass_def``, ``cutoff``, ``proj_cutoff``, projection precision) are
      inherited from ``Profile1``. The FFTLog precision is a superset of both inputs.
    """

    def __init__(self, op, Profile1, Profile2 = None, reflect = False):

        self.op       = op
        self.Profile1 = Profile1
        self.Profile2 = Profile2
        self.reflect  = reflect

        is_prof2 = isinstance(Profile2, ccl.halos.profiles.HaloProfile)
        is_unary = op in (neg, pos, abs)

        #Inherit numerical settings from the first profile
        settings = {}
        for k in ['cutoff', 'proj_cutoff', 'padding_lo_proj', 'padding_hi_proj', 'n_per_decade_proj']:
            v = _get_parameter(Profile1, k)
            if (v is None) and is_prof2: v = _get_parameter(Profile2, k)
            if v is not None: settings[k] = v
        use_fftlog = bool(getattr(Profile1, '_use_fftlog_projection', False))
        if use_fftlog and ('cutoff' in settings): settings['proj_cutoff'] = settings['cutoff']

        super().__init__(mass_def = Profile1.mass_def, use_fftlog_projection = use_fftlog, **settings)

        fft_pars = Profile1.precision_fftlog.to_dict()
        if is_prof2: fft_pars = combine_fftpars(fft_pars, Profile2.precision_fftlog.to_dict())
        ccl.halos.profiles.HaloProfile.update_precision_fftlog(self, **fft_pars)

        #Scaling by a number (or negation) commutes with projection and Fourier transforms, so
        #we apply it directly to the projected/fourier profiles of the input.
        scaling = (is_unary and (op is not abs)) or \
                  ((op is mul) and (not is_prof2)) or \
                  ((op is truediv) and (not is_prof2) and (not reflect))

        #Sums/differences of profiles are also linear. If any input has a custom projection
        #(eg. ComovingToPhysical, ConvolvedProfile) or Fourier transform, we must combine the
        #inputs' projected/fourier profiles to respect it. Otherwise, we project the combined
        #real-space profile, which avoids numerical cancellation between large, similar terms
        #(eg. DarkMatterBaryon - TwoHalo).
        summed  = (op in (add, sub)) and is_prof2

        if scaling or (summed and any(_has_custom_method(p, 'projected') for p in (Profile1, Profile2))):
            self._projected = self._projected_linear
        if scaling or (summed and any(_has_custom_method(p, 'fourier') for p in (Profile1, Profile2))):
            self._fourier   = self._fourier_linear


    def __getattr__(self, name):

        #Delegate unknown attributes to the first profile. Guard against lookups
        #before Profile1 exists (eg. during unpickling), and never delegate dunder
        #methods or the CCL implementation hooks.
        if (name.startswith('__') and name.endswith('__')) or (name in _CCL_HOOKS):
            raise AttributeError(name)
        try:
            Profile1 = object.__getattribute__(self, 'Profile1')
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(Profile1, name)


    def _apply(self, method, cosmo, r, M, a):

        A = getattr(self.Profile1, method)(cosmo, r, M, a)
        if self.op in (neg, pos, abs): return self.op(A)

        if isinstance(self.Profile2, ccl.halos.profiles.HaloProfile):
            B = getattr(self.Profile2, method)(cosmo, r, M, a)
        else:
            B = self.Profile2

        return self.op(B, A) if self.reflect else self.op(A, B)

    def _real(self, cosmo, r, M, a):             return self._apply('real', cosmo, r, M, a)
    def _projected_linear(self, cosmo, r, M, a): return self._apply('projected', cosmo, r, M, a)
    def _fourier_linear(self, cosmo, k, M, a):   return self._apply('fourier', cosmo, k, M, a)


    @property
    def model_params(self):
        params = {}
        for p in (self.Profile2, self.Profile1): #So that Profile1 takes precedence
            if isinstance(p, ccl.halos.profiles.HaloProfile): params.update(getattr(p, 'model_params', {}))
        return params

    @property
    def hyper_params(self):
        return getattr(self.Profile1, 'hyper_params', BaseBFGProfiles.hyper_params.fget(self))


    def __str_prf__(self):

        def name(p):
            if isinstance(p, ccl.halos.profiles.HaloProfile):
                return p.__str_prf__() if hasattr(p, '__str_prf__') else p.__class__.__name__
            return p

        op_name = self.op.__name__
        if self.op in (neg, pos, abs): return f"{op_name}[{name(self.Profile1)}]"
        if self.reflect: return f"{op_name}[{name(self.Profile2)}, {name(self.Profile1)}]"
        return f"{op_name}[{name(self.Profile1)}, {name(self.Profile2)}]"

    def __str_par__(self):
        return self.Profile1.__str_par__() if hasattr(self.Profile1, '__str_par__') else "()"


class WrappedProfile(object):
    """
    Mixin for convenience classes that are defined as a combination of other profiles,
    stored in the attribute ``myprof`` (eg. ``Gas = BoundGas + EjectedGas``).

    All attribute lookups, model/hyper parameters, string representations, and pickling
    are forwarded to ``myprof``. It must be placed before the profile base class in the
    inheritance list.
    """

    def __getattr__(self, name):

        #Guard against lookups before myprof exists (eg. during unpickling)
        try:
            myprof = object.__getattribute__(self, 'myprof')
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(myprof, name)

    #Need to explicitly set these two methods (to enable pickling)
    #since otherwise the getattr call above leads to infinite recursions.
    def __getstate__(self): return self.__dict__.copy()
    def __setstate__(self, state): self.__dict__.update(state)

    @property
    def model_params(self): return self.myprof.model_params

    @property
    def hyper_params(self): return self.myprof.hyper_params

    def __str_prf__(self): return f"{self.__class__.__name__}"
    def __str_par__(self): return self.myprof.__str_par__()

class Truncation(BaseBFGProfiles):
    """
    Class for truncating profiles conveniently.

    The `Truncation` profile imposes a cutoff on any profile beyond a specified 
    fraction of the halo's virial radius. The profile is used by modify existing 
    halo profiles, ensuring that contributions are zeroed out beyond the truncation radius.

    Parameters
    ----------
    epsilon : float
        The truncation parameter, representing the fraction of the virial radius 
        \( R_{200c} \) at which the profile is truncated. For example, an `epsilon` of 1 
        implies truncation at the virial radius, while a value < 1 truncates at a smaller radius.
    mass_def : ccl.halos.massdef.MassDef, optional
        The mass definition for the halo. By default, this is set to `MassDef200c`, which 
        defines the virial radius \( R_{200c} \) as the radius where the average density is 
        200 times the critical density.

    Notes
    -----
    
    The truncation condition is defined as:

    .. math::

        \\rho_{\\text{trunc}}(r) = 
        \\begin{cases} 
        1, & r < \\epsilon \\cdot R_{200c} \\\\ 
        0, & r \\geq \\epsilon \\cdot R_{200c}
        \\end{cases}

    where:
    - \( \\epsilon \) is the truncation fraction.
    - \( R_{200c} \) is the virial radius for the given mass definition.

    Examples
    --------
    Create a truncation profile and apply it to a given halo:

    >>> truncation_profile = Truncation(epsilon=0.8)
    >>> other_bfg_profile  = Profile(...)
    >>> truncated_profiled = other_bfg_profile * Truncation
    >>> r = np.logspace(-2, 1, 50)  # Radii in comoving Mpc
    >>> M = 1e14  # Halo mass in solar masses
    >>> a = 0.8  # Scale factor
    >>> truncated = other_bfg_profile.real(cosmo, r, M, a)
    """

    hyper_param_names = hyper_params + ['epsilon_trunc']
    def __init__(self, epsilon_trunc, mass_def = ccl.halos.massdef.MassDef200c, **kwargs):

        self.epsilon_trunc = epsilon_trunc
        super().__init__(mass_def = mass_def, **kwargs)


    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        R     = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        
        prof  = r_use[None, :] < R[:, None] * self.epsilon_trunc
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof
    

    def __str_prf__(self): return "Truncation"
    def __str_par__(self): return  f"(epsilon_trunc = {self.epsilon_trunc})"
    

class Identity(BaseBFGProfiles):
    """
    Class for the identity profile.

    The `Identity` profile is a simple profile that returns 1 for all radii, masses,
    and cosmologies. It is useful just for testing.

    Parameters
    ----------
    mass_def : ccl.halos.massdef.MassDef, optional
        The mass definition for the halo. By default, this is set to `MassDef200c`, 
        which defines the virial radius \( R_{200c} \) as the radius where the average 
        density is 200 times the critical density.

    """
    def __init__(self, mass_def = ccl.halos.massdef.MassDef200c, **kwargs):

        super().__init__(mass_def = mass_def, **kwargs)
        self._projected = self._real
        self._fourier = self._real

    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        prof  = np.ones([M_use.size, r_use.size])
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof
    
    _projected = _real
    _fourier   = _real

    def __str_prf__(self): return "Identity"
    def __str_par__(self): return  f"()"
    

class Zeros(BaseBFGProfiles):
    """
    Class for the zeros profile.

    The `Zeros` profile is a ccl profile class that returns 0 for all radii, masses,
    and cosmologies. It is useful just for testing, or evaluating inherited classes
    with certain components nulled out (eg. evaluating DMB profiles with no 2-halo)

    Parameters
    ----------
    mass_def : ccl.halos.massdef.MassDef, optional
        The mass definition for the halo. By default, this is set to `MassDef200c`, 
        which defines the virial radius \( R_{200c} \) as the radius where the average 
        density is 200 times the critical density.

    """
    def __init__(self, mass_def = ccl.halos.massdef.MassDef200c, **kwargs):

        super().__init__(mass_def = mass_def, **kwargs)
        self._projected = self._real
        self._fourier = self._real

    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        prof  = np.zeros([M_use.size, r_use.size])
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof
    
    _projected = _real
    _fourier   = _real

    def __str_prf__(self): return "Zeros"
    def __str_par__(self): return  f"()"
    


class TruncatedFourier(object):
    """
    Class for performing FFTLog transforms on profiles with sharp real-space truncations.
    The class sets the profile to zero outside the truncation radii, per halo, and places a
    node of the FFTLog grid exactly at the outer truncation radius, so the transform does not
    depend on where the sharp edge falls on the grid. All other methods and attributes are
    those of the input profile.

    You can set both a maximum and a minimum radii for the integration, though there is no
    known use-case where setting minimum-radii !=0 is reasonable.

    Parameters
    ----------
    Profile : ccl.halos.profiles.HaloProfile
        The profile to transform. Its mass definition sets the halo radius, R.
    epsilon_max : float
        The outer truncation radius, in units of R.
    epsilon_min : float, optional
        The inner truncation radius, in units of R. Default is None (no inner truncation).

    Notes
    -----
    The accuracy is set by the FFTLog precision of `Profile` (`n_per_decade`), as for any
    CCL Fourier profile: about 2% at low k for the default of 100, and about 0.2% for 1000.
    """

    def __init__(self, Profile, epsilon_max, epsilon_min = None, **kwargs):

        self.Profile     = Profile
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.fft_par     = Profile.precision_fftlog

    def __getattr__(self, name):
        '''
        Use the Profile's inbuilt methods for all routines EXCEPT the fourier
        routine, where we instead substitute with our method below
        '''
        #Guard against lookups before Profile exists (eg. during unpickling)
        try:
            Profile = object.__getattribute__(self, 'Profile')
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(Profile, name)

    def fourier(self, cosmo, k, M, a):

        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)
        prof  = np.zeros([M_use.size, k_use.size])
        R     = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        kprof = np.zeros([M_use.size, k_use.size])
        for M_i in range(M_use.size):

            #Setup r_min and r_max the same way CCL internal methods do for FFTlog transforms.
            #The profile is set to zero outside the truncation radii (rather than ending the grid there),
            #since otherwise the low-k modes (k < 1/R) are only extrapolated and the mass is overestimated.
            r_hi  = R[M_i] * self.epsilon_max #The halo has a sharp truncation at Rdelta * epsilon
            r_lo  = R[M_i] * self.epsilon_min if self.epsilon_min is not None else 0
            r_min = np.min([1/(np.max(k_use) * self.fft_par['padding_hi_fftlog']), r_hi/10])
            r_max = np.max([1/(np.min(k_use) * self.fft_par['padding_lo_fftlog']), r_hi*10])

            #Log-spaced grid with a node exactly at the truncation radius, which gets half the weight
            #(as in the trapezoid rule). Otherwise the result depends on where the edge falls on the grid.
            dlnr  = np.log(10) / self.fft_par['n_per_decade']
            n_lo  = int(np.ceil(np.log(r_hi/r_min)/dlnr))
            n_hi  = int(np.ceil(np.log(r_max/r_hi)/dlnr))
            r_fft = r_hi * np.exp(dlnr * np.arange(-n_lo, n_hi + 1))

            #Generate the real-space profile, sampled at the points defined above.
            prof  = self.Profile.real(cosmo, r_fft, M_use[M_i], a)
            prof  = np.where((r_fft < r_hi) & (r_fft >= r_lo), prof, 0)
            prof[n_lo] = 0.5 * self.Profile.real(cosmo, r_hi * (1 - 1e-10), M_use[M_i], a)
            
            #Now convert it to fourier space, apply the window function, and transform back
            k_out, Pk  = fftlog(r_fft, prof, 3, 0, self.fft_par['plaw_fourier'])
            
            prof       = resample_array(k_out, Pk, k_use, self.fft_par['extrapol'], self.fft_par['extrapol'], 0, 0)
            kprof[M_i] = np.where(np.isnan(prof), 0, prof) * (2*np.pi)**3 #(2\pi)^3 is from the fourier transforms.

        if np.ndim(k) == 0: kprof = np.squeeze(kprof, axis=-1)
        if np.ndim(M) == 0: kprof = np.squeeze(kprof, axis=0)

        return kprof
    

class ComovingToPhysical(BaseBFGProfiles):
    """
    Converts a given profile from comoving to physical units by applying
    a user-specified scale factor (`a`) correction. The projected profile is rescaled
    by one less power of `a` since one factor cancels in the projection integral.

    This is the step that takes a profile off the BaryonForge comoving ladder and turns
    it into a physical observable; see the "Units and redshift conventions" section of
    `BaseBFGProfiles` for the full contract.

    Parameters
    ----------
    profile : ccl.halo.HaloProfile object
        A CCL profile object (of any kind)
    factor : float
        The power of the scale factor `a` applied to convert the profile
        from comoving to physical units. Use -3 for anything on the standard BaryonForge
        comoving ladder: density profiles, pressure profiles, and the observable classes
        `ThermalSZ` and `XraySkyCounts`. For the observable classes, wrap the observable
        object itself, never its input profile.

    Returns
    -------
    ccl.halo.HaloProfile object
        A halo profile class with `real`, `projected`, and `fourier` routines that have been
        rescaled by scale factor `a` to the appropriate power. The Fourier profile is rescaled
        by ``a^(factor + 3)``, since it is an integral over (comoving) volume, and is evaluated
        at comoving wavenumbers.
    """

    hyper_param_names = hyper_params + ['profile', 'factor']
    def __init__(self, profile, factor, **kwargs):

        self.profile = profile
        self.factor  = factor

        #Remove mass_def from kwargs if provided because we need to use the
        #mass_def from the input profile instead
        kwargs.pop('mass_def', None)

        #The cutoffs of this wrapper are those of the input profile
        for k in ['cutoff', 'proj_cutoff']:
            v = _get_parameter(profile, k)
            if (k not in kwargs) and (v is not None): kwargs[k] = v

        #We just set this to the same as the inputted profile.
        super().__init__(mass_def = profile.mass_def, **kwargs)


    def real(self, cosmo, r, M, a):      return self.profile.real(cosmo, r, M, a)      * np.power(a, self.factor)
    def projected(self, cosmo, r, M, a): return self.profile.projected(cosmo, r, M, a) * np.power(a, self.factor + 1)
    def fourier(self, cosmo, k, M, a):   return self.profile.fourier(cosmo, k, M, a)   * np.power(a, self.factor + 3)

    def set_parameter(self, key, value): _set_parameter(self, key, value)

    #CCL asserts that at least one of these methods exist. They simply
    #forward to the public methods above, which do not use them.
    def _real(self, cosmo, r, M, a):      return self.real(cosmo, r, M, a)
    def _projected(self, cosmo, r, M, a): return self.projected(cosmo, r, M, a)
    

class Mdelta_to_Mtot(object):
    """
    Computes the total mass of a halo by integrating its density profile over a specified radial range.

    Parameters
    ----------
    profile : object
        A density profile object that provides the `real(cosmo, r, M, a)` method,
        returning the density at a given radius `r` for mass `M` and scale factor `a`.
    r_min : float, optional
        The minimum radius for integration, in the same units as `r`. Default is `1e-3`.
    r_max : float, optional
        The maximum radius for integration, in the same units as `r`. Default is `1e2`.
    N_int : int, optional
        The number of integration points between `r_min` and `r_max`. Default is `1000`.

    Methods
    -------
    __call__(cosmo, M, a)
        Computes the total mass by integrating the density profile over the radial range.

    Returns
    -------
    M_tot : float or array-like
        The total mass of the halo, computed as the integral of the density profile.
        If `M` is a scalar, returns a scalar; if `M` is an array, returns an array of the same shape.
    """
    
    def __init__(self, profile, r_min = 1e-3, r_max = 1e2, N_int = 1000):
        
        self.profile = profile
        self.r_min   = r_min
        self.r_max   = r_max
        self.N_int   = N_int
    
    def __call__(self, cosmo, M, a):

        M_use = np.atleast_1d(M)
        r     = np.geomspace(self.r_min, self.r_max, self.N_int)
        prof  = self.profile.real(cosmo, r, M_use, a)

        dV    = 4*np.pi*r**2
        M_tot = np.trapz(dV * prof, r, axis = 1)

        if np.ndim(M) == 0: M_tot = np.squeeze(M_tot, axis=0)

        return M_tot
