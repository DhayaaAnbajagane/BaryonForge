import numpy as np
import pyccl as ccl
from .Schneider19 import SchneiderProfiles
from scipy import interpolate
fftlog = ccl.pyutils._fftlog_transform

__all__ = ['Truncation', 'Identity', 'Zeros']

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
    


class TruncatedFourier(object):
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
    def __init__(self, CLASS, epsilon = 1): 
        self.CLASS   = CLASS
        self.epsilon = epsilon

    def __getattr__(self, name):  
        if name != 'fourier':
            return getattr(self.CLASS, name)
        else:
            return self.fourier

    def fourier(self, cosmo, k, M, a):

        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)
        prof  = np.zeros([M_use.size, k_use.size])
        R     = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        for M_i in range(M_use.size):

            #Setup r_min and r_max the same way CCL internal methods do for FFTlog transforms.
            #We set minimum and maximum radii here to make sure the transform uses sufficiently
            #wide range in radii. It helps prevent ringing in transformed profiles.
            r_min = np.min([np.min(r) * self.fft_par['padding_lo_fftlog'], 1e-8])
            r_max = R[M_i] * self.epsilon #The halo has a sharp truncation at Rdelta * epsilon.
            n     = self.fft_par['n_per_decade'] * np.int32(np.log10(r_max/r_min))
            
            #Generate the real-space profile, sampled at the points defined above.
            r_fft = np.geomspace(r_min, r_max, n)
            prof  = self.Profile.real(cosmo, r_fft, M, a)
            
            #Now convert it to fourier space, apply the window function, and transform back
            k_out, Pk   = fftlog(r_fft, prof, 3, 0, self.fft_par['plaw_fourier'])
            r_out, prof = fftlog(k_out, Pk * self.Pixel.real(k_out), 3, 0, self.fft_par['plaw_fourier'] + 1)
            
            #Below the pixel scale, the profile will be constant. However, numerical issues can cause ringing.
            #So below pixel_size/5, we set the profile value to r = pixel_size. What happens at five times below
            #the pixel-scale should never matter for your analysis. But doing this will help avoid edge-case errors
            #later on (eg. in defining enclosed masses) so we do this
            r    = np.clip(r, self.Pixel.size / 5, None) #Set minimum radius according to pixel, to prevent ringing on small-scale outputs
            prof = interpolate.PchipInterpolator(np.log(r_out), prof, extrapolate = False, axis = -1)(np.log(r))
            prof = np.where(np.isnan(prof), 0, prof) * (2*np.pi)**3 #(2\pi)^3 is from the fourier transforms.
        
        return prof