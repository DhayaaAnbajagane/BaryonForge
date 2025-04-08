import pyccl as ccl
import numpy as np, healpy as hp
from scipy import interpolate, special
from .Tabulate import _set_parameter

__all__ = ['ConvolvedProfile', 'GridPixelApprox', 'HealPixel']

#Define a shorthand to use everywhere
fftlog = ccl.pyutils._fftlog_transform

class ConvolvedProfile(ccl.halos.profiles.HaloProfile):
    """
    A class to compute profiles convolved with a pixel window function.

    The `ConvolvedProfile` class takes in a `HaloProfile` object from `CCL` and a pixel window function object (eg. `HealPixel`)
    and returns a new object that computes profiles while including the pixel window function of the map that we paint into.
    It is assumed that the window function is isotropic. This class supports both real-space and projected-space profiles.

    Parameters
    ----------
    Profile : object
        A `ccl.halos.profiles.HaloProfile` object that defines the halo profile to be convolved with the pixel window function.
    
    Pixel : object
        An object that defines the pixel window function. This object must provide `real()` and `projected()` methods
        to return the window function in real space and harmonic space, respectively. It must also have an `isHarmonic` attribute,
        to specify where the pixel scale is defined in angular space or physical space.

    Attributes
    ----------
    Profile : object
        The original profile object to be convolved with the pixel window function.
    
    Pixel : object
        The pixel window function object used to convolve the profile.
    
    fft_par : dict
        A dictionary of FFT log parameters used for Fourier transforms, derived from the `Profile` object.
    
    isHarmonic : bool
        A boolean indicating whether the pixel window function is defined in harmonic space.

    Methods
    -------
    real(cosmo, r, M, a)
        Computes the real-space profile convolved with the pixel window function.
    
    projected(cosmo, r, M, a)
        Computes the projected-space profile convolved with the pixel window function, accounting for harmonic space if applicable.

    Examples
    --------
    >>> profile = SomeHaloProfile()
    >>> pixel = SomePixelWindowFunction()
    >>> convolved_profile = ConvolvedProfile(profile, pixel)
    >>> result = convolved_profile.real(cosmo, r, M, a)
    >>> projected_result = convolved_profile.projected(cosmo, r, M, a)

    Notes
    -----
    - This class uses FFTLog for computing convolutions in Fourier space.
    - The `real` and `projected` methods perform clipping on the radius `r` to prevent artifacts like ringing at small scales.
    - The convolved profile is scaled by appropriate powers of \(2\pi\) to account for the transformations.

    """

    
    def __init__(self, Profile, Pixel):

        self.Profile    = Profile
        self.Pixel      = Pixel
        self.fft_par    = Profile.precision_fftlog
        
        self.isHarmonic = Pixel.isHarmonic

        #We just set this to the same as the inputted profile.
        super().__init__(mass_def = Profile.mass_def)
        
        
    def __getattr__(self, name):
        """
        Delegate attribute and method access to the Profile object passed to the class,
        but only if the attribute or method is not already found in the class.

        Parameters
        ----------
        name : str
            The name of the attribute or method to access.

        Returns
        -------
        object
            The attribute or method from the Profile object.

        """
        
        try:
            return super().__getattribute__(name)
        
        except AttributeError:
            return getattr(self.Profile, name)


    #Need to explicitly set these two methods (to enable pickling)
    #since otherwise the getattr call above leads to infinite recursions.
    def __getstate__(self): self.__dict__.copy()    
    def __setstate__(self, state): self.__dict__.update(state)

    def set_parameter(self, key, value): _set_parameter(self, key, value)
    
    
    def _real(self, cosmo, r, M, a):
        """
        Computes the real-space profile convolved with the pixel window function.

        This method convolves the real-space profile with the pixel window function using FFTLog. The profile
        is first transformed into Fourier space, multiplied by the window function, and then transformed back
        to real space.

        Parameters
        ----------
        cosmo : object
            A `ccl.Cosmology` object representing the cosmological parameters.
        
        r : ndarray
            An array of radii at which to compute the convolved profile. In comoving Mpc.
        
        M : float
            The mass of the halo, in solar masses.
        
        a : float
            The scale factor at which to compute the profile.
        
        Returns
        -------
        prof : ndarray
            The convolved real-space profile evaluated at the input radii `r`.
        """

        #Setup r_min and r_max the same way CCL internal methods do for FFTlog transforms.
        #We set minimum and maximum radii here to make sure the transform uses sufficiently
        #wide range in radii. It helps prevent ringing in transformed profiles.
        r_min = np.min([np.min(r) * self.fft_par['padding_lo_fftlog'], 1e-8])
        r_max = np.max([np.max(r) * self.fft_par['padding_hi_fftlog'], 1e3])
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
    
    
    def projected(self, cosmo, r, M, a):
        """
        Computes the projected-space profile convolved with the pixel window function.

        This method convolves the projected-space profile with the pixel window function using FFTLog. It can
        account for harmonic space considerations if the pixel window function is harmonic.

        Parameters
        ----------
        cosmo : object
            A `ccl.Cosmology` object representing the cosmological parameters.
        
        r : ndarray
            An array of radii at which to compute the convolved profile. In comoving Mpc.
        
        M : float
            The mass of the halo, in solar masses.
        
        a : float
            The scale factor at which to compute the profile.
        
        Returns
        -------
        prof : ndarray
            The convolved projected-space profile evaluated at the input radii `r`.
        """

        #If the pixel is in harmonic space (angular space), then we will need to use the
        #angular diameter distance, so calculate it
        if self.isHarmonic:    
            assert a < 1, f"You cannot set a = 1, z = 0 when computing harmonic sky projections"
            D_A = ccl.comoving_angular_distance(cosmo, a)
            
        #Setup r_min and r_max the same way CCL internal methods do for FFTlog transforms.
        #We set minimum and maximum radii here to make sure the transform uses sufficiently
        #wide range in radii. It helps prevent ringing in transformed profiles.
        r_min = np.min([np.min(r) * self.fft_par['padding_lo_fftlog'], 1e-8])
        r_max = np.max([np.max(r) * self.fft_par['padding_hi_fftlog'], 1e3])
        n     = self.fft_par['n_per_decade'] * np.int32(np.log10(r_max/r_min))
        
        #Generate the real-space profile, sampled at the points defined above.
        r_fft = np.geomspace(r_min, r_max, n)
        prof  = self.Profile.projected(cosmo, r_fft, M, a)
        
        #If we want harmonic space, then r_fft shouldn't be a distance, but an angle.
        if self.isHarmonic: r_fft = r_fft / D_A
        
        #Now convert it to fourier space, apply the window function, and transform back
        k_out, Pk   = fftlog(r_fft, prof, 2, 0, self.fft_par['plaw_fourier'] + 1)
        r_out, prof = fftlog(k_out, Pk * self.Pixel.projected(k_out), 2, 0, self.fft_par['plaw_fourier'] + 1)
        
        #Below the pixel scale, the profile will be constant. However, numerical issues can cause ringing.
        #So below pixel_size/5, we set the profile value to r = pixel_size. What happens at five times below
        #the pixel-scale should never matter for your analysis. But doing this will help avoid edge-case errors
        #later on (eg. in defining enclosed masses) so we do this.
        if self.isHarmonic: 
            r_out = r_out * D_A
            r     = np.clip(r, self.Pixel.size / 5 * D_A, None)
        else:
            r     = np.clip(r, self.Pixel.size / 5, None)
            
        prof = interpolate.PchipInterpolator(np.log(r_out), prof, extrapolate = False, axis = -1)(np.log(r))
        prof = np.where(np.isnan(prof), 0, prof) * (2*np.pi)**2
        
        return prof
    

    def fourier(self, cosmo, k, M, a):
        """
        Computes the fourier-space profile convolved with the pixel window function.

        Parameters
        ----------
        cosmo : object
            A `ccl.Cosmology` object representing the cosmological parameters.
        
        k : ndarray
            An array of wavenumbers at which to compute the convolved profile. In comoving 1/Mpc.
        
        M : float
            The mass of the halo, in solar masses.
        
        a : float
            The scale factor at which to compute the profile.
        
        Returns
        -------
        prof : ndarray
            The convolved real-space profile evaluated at the input radii `r`.
        """

        return self.Profile.fourier(cosmo, k, M, a) * self.Pixel.real(k)
    
    
    
    
class GridPixelApprox(object):
    """
    A class for approximating the window function of a grid's pixel as a circular tophat.

    The `GridPixelApprox` class provides methods to approximate the window function of a grid's pixel.
    In this approximation, the pixel window function is considered as a circular tophat in real space.
    This approximation is used to compute the effective window function in both real and projected spaces,
    for use in Fourier transforms.

    Parameters
    ----------
    size : float
        The size of the grid's square pixel, in comoving Mpc. 
        This size is used to compute the radius of the tophat approximation.

    Methods
    -------
    beam(k, R)
        Computes the beam function given a wavenumber `k` and a radius `R`.
    
    real(k)
        Computes the real-space approximation of the pixel window function for given wavenumbers `k`.
    
    projected(k)
        Computes the projected-space approximation of the pixel window function for given wavenumbers `k`.

    Examples
    --------
    >>> pixel_approx = GridPixelApprox(size=0.1)
    >>> k = np.linspace(0.01, 1.0, 100)
    >>> real_space_beam = pixel_approx.real(k)
    >>> projected_space_beam = pixel_approx.projected(k)

    Notes
    -----
    - The `beam()` method uses the spherical Bessel function of the first kind (`spherical_jn`) to compute the beam function.
    - The `real()` and `projected()` methods compute the effective radius from the given pixel size and use this
      radius in the `beam()` function.
    """
    
    isHarmonic = False

    def __init__(self, size):  
        """
        Initializes the GridPixelApprox class with a specified pixel size.

        Parameters
        ----------
        size : float
            The size of the grid's pixel. This size is used to compute the radius for the tophat approximation.
        """
        
        self.size = size
        
    
    def beam(self, k, R):
        """
        Computes the beam function for given wavenumbers and a radius.

        The beam function represents the response of a circular tophat window function and is derived from 
        the spherical Bessel function of the first kind, \( j_1 \). The beam function \( B(k) \) is calculated 
        as:

        .. math::

            B(k) = \\frac{3j_1(kr)}{kr}

        where:
        
        - \( k \) is the wavenumber.
        - \( r = 2R \) is the diameter of the pixel (not the radius).
        - \( j_1 \) is the spherical Bessel function of the first kind of order one.

        The factor of 2 in the radius calculation arises because the window function is defined for the 
        diameter rather than the radius.

        Parameters
        ----------
        k : ndarray
            An array of wavenumbers at which to evaluate the beam function.
        
        R : float
            The effective radius of the pixel, which depends on the size of the grid's pixel.

        Returns
        -------
        beam : ndarray
            An array of the beam function values corresponding to the input wavenumbers. The output is 
            calculated as \( \\frac{3j_1(kr)}{kr} \), with special handling to avoid division by zero when 
            \( kr = 0 \).

        """
        
        kr = k * (2*R) #Factor of 2 because the window function needs diameter, not radius

        with np.errstate(invalid = 'ignore', divide = 'ignore'):
            beam = np.where(kr > 0, 3*special.spherical_jn(1, kr)/kr, 1)
            
        return beam
        
        
    def real(self, k):
        """
        Computes the real-space approximation of the pixel window function.

        This method approximates the pixel window function using a circular tophat in real space. The effective 
        radius \( R \) is calculated based on the volume-equivalent size of the grid's pixel, assuming a spherical 
        shape. The real-space window function is then computed using this radius.

        The effective radius \( R \) of the tophat is given by:

        .. math::

            R = \left( \\frac{\\text{size}^3}{\\frac{4}{3} \pi} \\right)^{\\frac{1}{3}}

        where `size` is the linear size of the pixel.

        Parameters
        ----------
        k : ndarray
            An array of wavenumbers at which to evaluate the real-space window function.

        Returns
        -------
        beam : ndarray
            An array of the real-space window function values corresponding to the input wavenumbers, computed 
            using the `beam` function with the effective radius.

        Notes
        -----
        - This function approximates the real-space window function by using a circular tophat model, which 
        simplifies the computation while capturing the essential behavior of the pixel's effect in real space.
        """
        
        R = np.cbrt(self.size**3 / (4/3 * np.pi) )
        
        return self.beam(k, R)
        
        
    
    def projected(self, k):
        """
        Computes the projected-space approximation of the pixel window function.

        This method approximates the pixel window function using a circular tophat in projected space. 
        The effective radius \( R \) is calculated based on the area-equivalent size of the grid's pixel, 
        assuming a circular shape. The projected-space window function is then computed using this radius.

        The effective radius \( R \) is given by:

        .. math::

            R = \sqrt{\\frac{\\text{size}^2}{\pi}}

        where `size` is the linear size of the pixel.

        Parameters
        ----------
        k : ndarray
            An array of wavenumbers at which to evaluate the projected-space window function.

        Returns
        -------
        beam : ndarray
            An array of the projected-space window function values corresponding to the input wavenumbers, 
            computed using the `beam` function with the effective radius.

        Notes
        -----
        - This function approximates the projected-space window function by using a circular tophat model, 
        which simplifies the computation while capturing the essential behavior of the pixel's effect 
        in projected space.
        - The beam function is calculated by calling the `self.beam()` method, which computes the response 
        using the spherical Bessel function of the first kind.
        """
        
        R = np.sqrt(self.size**2 / np.pi)
        
        return self.beam(k, R)
            
            

class HealPixel(object):
    """
    A class for approximating the window function of a HEALPix pixel using a Gaussian beam.

    The `HealPixel` class provides methods to approximate the window function of a HEALPix pixel. Instead of
    using the built-in HEALPix pixel window function, which is limited to `3 * NSIDE - 1`, this class uses a Gaussian
    beam approximation. The Gaussian beam, with a full width at half maximum (FWHM) that is \( \frac{1}{\sqrt{2}} \)
    smaller than the HEALPix pixel, provides a smooth window function for stable FFTLogs over a wide range of multipoles.

    The Gaussian beam \( B(k) \) is computed as:

    .. math::

        B(k) = \exp\left(-\\frac{k(k+1)\sigma^2}{2}\\right)

    where \( \sigma \) is the standard deviation of the Gaussian beam, calculated as:

    .. math::

        \sigma = \\frac{\\text{size}}{\sqrt{8 \log(2)} \cdot \sqrt{2}}

    Attributes
    ----------
    isHarmonic : bool
        A class-level attribute indicating whether the pixel window function is defined in harmonic space.
        For `HealPixel`, this is always `True`.
    
    NSIDE : int
        The NSIDE parameter of the HEALPix map, which determines the resolution.
    
    size : float
        The size (resolution) of a single pixel in the HEALPix grid, in radians.

    Methods
    -------
    real(k)
        Returns a zero array, indicating that the real-space representation of the pixel is not supported.
    
    projected(k)
        Computes the projected-space approximation of the pixel window function using a Gaussian beam.

    Examples
    --------
    >>> heal_pixel = HealPixel(NSIDE=64)
    >>> k = np.linspace(0.01, 1.0, 100)
    >>> real_space_response = heal_pixel.real(k)  # Will return zeros
    >>> projected_space_beam = heal_pixel.projected(k)

    Notes
    -----
    - The Gaussian beam used for approximation is designed to match the HEALPix pixel window function closely,
      with <0.1% deviation for most scales and about 1% at the smallest scales.
    - This approximation is suitable for stable FFTLogs, ensuring smooth behavior at large multipoles.
    """
    
    isHarmonic = True

    def __init__(self, NSIDE):
        """
        Initializes the HealPixel class with a specified NSIDE value.

        Parameters
        ----------
        NSIDE : int
            The NSIDE parameter of the HEALPix map, which determines the resolution of the map.
        """
        
        self.NSIDE = NSIDE
        self.size  = hp.nside2resol(NSIDE)
        
    
    def real(self, k):
        """
        Returns a zero array for the real-space window function.

        This method indicates that the real-space representation of the HEALPix pixel window function
        is not supported. It returns a zero array, which will propagate through calculations and help
        to throw errors when attempting to use real-space profiles.

        Parameters
        ----------
        k : ndarray
            An array of wavenumbers at which to evaluate the real-space window function.

        Returns
        -------
        zeros : ndarray
            An array of zeros, indicating that real-space representation is not supported.
        """

        #Can't use healpix pixel for real-space, so just make the beam 0.
        #That way the real-space profile will also be 0 and throw errors. 
        return np.zeros_like(k)
        
    
    def projected(self, k):
        """
        Computes the projected-space approximation of the HEALPix pixel window function.

        This method uses a Gaussian beam to approximate the pixel window function in harmonic space.
        The Gaussian beam is chosen to have a full width at half maximum (FWHM) that is \( \frac{1}{\sqrt{2}} \)
        smaller than the pixel size, providing a close approximation to the actual HEALPix pixel window function.

        The Gaussian beam \( B(k) \) is computed as:

        .. math::

            B(k) = \exp\left(-\\frac{k(k+1)\sigma^2}{2}\\right)

        where \( \sigma \) is the standard deviation of the Gaussian beam, given by:

        .. math::

            \sigma = \\frac{\\text{size}}{\sqrt{8 \log(2)} \cdot \sqrt{2}}

        Parameters
        ----------
        k : ndarray
            An array of wavenumbers (or multipole moments) at which to evaluate the projected-space window function.

        Returns
        -------
        beam : ndarray
            An array representing the Gaussian beam values corresponding to the input wavenumbers.
        """
        
        sig  = hp.nside2resol(self.NSIDE) / np.sqrt(8 * np.log(2)) / np.sqrt(2)
        beam = np.exp(-k*(1 + k)/2 * sig**2)
        
        return beam
       
 
class NoPix(object):
    """
    A class representing no pixel effect, i.e., no convolution or smoothing.

    The `NoPix` class is used to simulate a scenario where there is no pixel window function applied,
    effectively representing a situation with no smoothing or convolution. This class is primarily
    used for testing purposes, allowing the evaluation of profiles without the influence of a pixel window
    function.

    Methods
    -------
    real(k)
        Returns an array of ones, representing no effect of a pixel window function in real space.
    
    projected(k)
        Returns an array of ones, representing no effect of a pixel window function in projected space.

    Examples
    --------
    >>> nopix = NoPix()
    >>> k = np.linspace(0.01, 1.0, 100)
    >>> real_space_response = nopix.real(k)  # Returns an array of ones
    >>> projected_space_response = nopix.projected(k)  # Returns an array of ones

    Notes
    -----
    - This class is useful for testing and comparison purposes, where the impact of pixel window functions
      needs to be isolated or excluded.
    - The `real()` and `projected()` methods return arrays of ones, indicating that there is no attenuation
      or modification of the input profile due to a pixel window function.
    """
    
    def __init__(self):
        pass
        
    def real(self, k):
        return np.ones_like(k)
                
    def projected(self, k):
        return np.ones_like(k)       
