import numpy as np
import pyccl as ccl
from .Schneider19 import SchneiderProfiles

__all__ = ['Truncation', 'Identity']

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