
import numpy as np
import pyccl as ccl
from collections import OrderedDict
from ..Profiles.Base import BaseBFGProfiles

__all__ = ['SimpleArrayCache', 'CachedProfile']

class SimpleArrayCache:
    """
    A lightweight LRU-style cache designed for functions whose inputs include
    NumPy arrays. Unlike ``functools.lru_cache``, this cache supports
    unhashable arguments (e.g. ``numpy.ndarray``) by constructing a stable
    byte-based key from the array contents.

    The cache stores results keyed by a tuple consisting of:
        - ``str(cosmo)``: a string representation of the cosmology object;
        - ``a``: the scale factor as a float;
        - For each array argument (``R`` and ``M``):
            * its shape,
            * its dtype,
            * its raw byte buffer from ``.tobytes()``.

    When used as a decorator, the cache wraps a function of the form
    ``func(cosmo, R, M, a)`` and automatically caches its return value based
    on these arguments. Repeated calls with identical inputs return the
    cached result without re-evaluating the function. For fourier functions.
    R is simply the wavenumber rather than radius.

    Parameters
    ----------
    maxsize : int, optional
        Maximum number of cached entries to store. The cache evicts the
        least recently used (LRU) entry when the limit is exceeded.
        Default is 64

    Notes
    -----
    - This cache treats the *contents* of ``R`` and ``M`` as part of the key,
      so even small differences in floating-point values produce distinct cache
      entries.
    - ``cosmo`` is converted to a string and used verbatim. Two cosmology
      objects that print identically will collide; two identical cosmologies
      with different string representations will not.
    - The cache is implemented using ``collections.OrderedDict`` and maintains
      LRU behavior manually.

    Examples
    --------
    >>> cached_func = SimpleArrayCache(maxsize=64)(func)
    """
    def __init__(self, maxsize = 32):
        self.maxsize = maxsize
        self._store = OrderedDict()

    def _key(self, cosmo, R, M, a):
        R = np.atleast_1d(R)
        M = np.atleast_1d(M)
        return (
            str(cosmo),
            float(a),
            R.shape, R.dtype.str, R.tobytes(),
            M.shape, M.dtype.str, M.tobytes(),
        )

    def get(self, cosmo, R, M, a):
        k = self._key(cosmo, R, M, a)
        if k in self._store:
            self._store.move_to_end(k)
            return self._store[k]
        return None

    def set(self, cosmo, R, M, a, value):
        k = self._key(cosmo, R, M, a)
        self._store[k] = value
        self._store.move_to_end(k)
        if len(self._store) > self.maxsize:
            self._store.popitem(last=False)


    def __call__(self, func):

        def cached_func(cosmo, R, M, a):
            cached = self.get(cosmo, R, M, a)
            if cached is not None:
                return cached
            val = func(cosmo, R, M, a)
            self.set(cosmo, R, M, a, val)
            return val
        
        return cached_func
        

class CachedProfile(BaseBFGProfiles):
    """
    A class that will cache the profile evaluations for the real, projected, and fourier methods.

    This class will take in a BaryonForge (BFG) class and cache its results. It is
    useful for halo model P(k) calculations, where the same masses, redshifts, wavenumbers/radii
    are evaluated many times. See also `TabulatedProfile` if you want to only store a sparser grid.

    Parameters
    ----------
    Profile : object
        A profile that we want to cache. Can either be a vanilla CCL profile or a BaryonForge Profile.

    """

    def __init__(self, Profile, maxsize = 64):
        
        self.Profile  = Profile
        self.maxsize  = maxsize

        self.real      = SimpleArrayCache(self.maxsize)(self.Profile.real)
        self.projected = SimpleArrayCache(self.maxsize)(self.Profile.projected)
        self.fourier   = SimpleArrayCache(self.maxsize)(self.Profile.fourier)
        
        #We just set this to the same as the inputted profile.
        super().__init__(mass_def = self.Profile.mass_def)

        self.update_precision_fftlog(**self.Profile.precision_fftlog.to_dict())


    def __getattr__(self, key):

        safe_keys = ['real', 'projected', 'fourier', 'Profile', 'maxsize']

        if key in safe_keys:
            return object.__getattribute__(self, key)
        else:
            return getattr(object.__getattribute__(self, 'Profile'), key)
        

    def __str_prf__(self):

        return f"Cached[{self.Profile.__str_prf__()}]"
    
    def __str_par__(self): return self.Profile.__str_par__()