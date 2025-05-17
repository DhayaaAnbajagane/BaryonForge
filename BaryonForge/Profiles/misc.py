import numpy as np
import pyccl as ccl
from .Schneider19 import SchneiderProfiles
from scipy import interpolate
from ..utils.Tabulate import _set_parameter
from pyccl.pyutils import resample_array, _fftlog_transform
fftlog = _fftlog_transform

__all__ = ['Truncation', 'Identity', 'Zeros', 'Comoving_to_Physical', 'Mdelta_to_Mtot']

class Truncation(SchneiderProfiles):
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

    def __init__(self, epsilon, mass_def = ccl.halos.massdef.MassDef200c):

        self.epsilon = epsilon
        ccl.halos.profiles.HaloProfile.__init__(self, mass_def = mass_def)


    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)
        R     = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        
        prof  = r_use[None, :] < R[:, None] * self.epsilon
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

        return prof
    

    def __str_prf__(self): return "Truncation"
    def __str_par__(self): return  f"(epsilon = {self.epsilon})"
    

class Identity(SchneiderProfiles):
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
    def __init__(self, mass_def = ccl.halos.massdef.MassDef200c):

        ccl.halos.profiles.HaloProfile.__init__(self, mass_def = mass_def)

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
    

class Zeros(SchneiderProfiles):
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
    def __init__(self, mass_def = ccl.halos.massdef.MassDef200c):

        ccl.halos.profiles.HaloProfile.__init__(self, mass_def = mass_def)

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
    The class simply limits the `fourier` method integration limits, per halo, to account
    for this truncation.

    You can set both a maximum and a minimum radii for the integration, though there is no
    known use-case where setting minimum-radii !=0 is reasonable.

    Parameters
    ----------
    mass_def : ccl.halos.massdef.MassDef, optional
        The mass definition for the halo. By default, this is set to `MassDef200c`, 
        which defines the virial radius \( R_{200c} \) as the radius where the average 
        density is 200 times the critical density.

    """
    def __init__(self, Profile, epsilon_max, epsilon_min = None): 
        self.Profile     = Profile
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.fft_par     = Profile.precision_fftlog

    def __getattr__(self, name):  
        '''
        Use the Profile's inbuilt methods for all routines EXCEPT the fourier
        routine, where we instead substitute with our method below
        '''
        if name != 'fourier':
            return getattr(self.Profile, name)
        else:
            return self.fourier

    def fourier(self, cosmo, k, M, a):

        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)
        prof  = np.zeros([M_use.size, k_use.size])
        R     = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        kprof = np.zeros([M_use.size, k_use.size])
        for M_i in range(M_use.size):

            #Setup r_min and r_max the same way CCL internal methods do for FFTlog transforms.
            #We set minimum and maximum radii here to make sure the transform uses sufficiently
            #wide range in radii. It helps prevent ringing in transformed profiles.
            r_min = R[M_i] * self.epsilon_min if self.epsilon_min is not None else (np.min(k) * self.fft_par['padding_lo_fftlog'])
            r_max = R[M_i] * self.epsilon_max #The halo has a sharp truncation at Rdelta * epsilon, so we always set that as the max.
            n     = self.fft_par['n_per_decade'] * np.int32(np.log10(r_max/r_min))
            
            #Generate the real-space profile, sampled at the points defined above.
            r_fft = np.geomspace(r_min, r_max, n)
            prof  = self.Profile.real(cosmo, r_fft, M_use[M_i], a)
            
            #Now convert it to fourier space, apply the window function, and transform back
            k_out, Pk  = fftlog(r_fft, prof, 3, 0, self.fft_par['plaw_fourier'])
            
            prof       = resample_array(k_out, Pk, k_use, self.precision_fftlog['extrapol'], self.precision_fftlog['extrapol'], 0, 0)
            kprof[M_i] = np.where(np.isnan(prof), 0, prof) * (2*np.pi)**3 #(2\pi)^3 is from the fourier transforms.

        if np.ndim(k) == 0: kprof = np.squeeze(kprof, axis=-1)
        if np.ndim(M) == 0: kprof = np.squeeze(kprof, axis=0)

        return kprof
    

class ComovingToPhysical(ccl.halos.profiles.HaloProfile):
    """
    Converts a given profile from comoving to physical units by applying 
    a user-specified scale factor (`a`) correction. The projected profile is rescaled
    by one less power of `a`.

    Parameters
    ----------
    profile : ccl.halo.HaloProfile object
        A CCL profile object (of any kind)
    factor : float
        The power of the scale factor `a` applied to convert the profile 
        from comoving to physical units. Should use -3 for density profiles AND for pressure profiles in BaryonForge. 

    Returns
    -------
    ccl.halo.HaloProfile object
        A halo profile class with `_real` and `projected` routines that have been rescaled by
        scale factor `a` to the appropriate power.
    """
    
    def __init__(self, profile, factor):
        
        self.profile = profile
        self.factor  = factor

        #We just set this to the same as the inputted profile.
        super().__init__(mass_def = profile.mass_def)
        
    def real(self, cosmo, r, M, a):      return self.profile.real(cosmo, r, M, a)      * np.power(a, self.factor)
    def projected(self, cosmo, r, M, a): return self.profile.projected(cosmo, r, M, a) * np.power(a, self.factor + 1)

    def set_parameter(self, key, value): _set_parameter(self, key, value)

    #Have dummy methods because CCL asserts that these must exist.
    #Hacky because I want to keep SchneiderProfiles as base class
    #in order to get __init__ to be simple, but then we have to follow
    #the CCL HaloProfile base class API. 
    def _real(self): return np.nan
    def _projected(self): return np.nan
    

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