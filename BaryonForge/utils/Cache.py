
import numpy as np
import pyccl as ccl
from collections import OrderedDict
from ..Profiles.Base import BaseBFGProfiles
import functools

__all__ = ['SimpleArrayCache', 'CachedProfile', 'CachedHODProfile']

class SimpleArrayCache:
    """
    A lightweight LRU-style cache designed for functions whose inputs include
    NumPy arrays. Unlike ``functools.lru_cache``, this cache supports
    unhashable arguments (e.g. ``numpy.ndarray``) by constructing a stable
    byte-based key from the array contents.

    The cache stores results keyed by a tuple that is dynamically generated
    according to the function being cached:
        - floats, ints, and strings are saved as single values
        - For each array argument, we store:
            * its shape,
            * its dtype,
            * its raw byte buffer from ``.tobytes()``.
        - Any lists and tuples are converted to arrays and follow the above
        - Other objects (custom classes) are converted to string representations

    When used as a decorator, the cache wraps a function of the form
    ``func(*args)`` and automatically caches its return value based
    on these arguments. Repeated calls with identical inputs return the
    cached result without re-evaluating the function.

    Parameters
    ----------
    maxsize : int, optional
        Maximum number of cached entries to store. The cache evicts the
        least recently used (LRU) entry when the limit is exceeded.
        Default is 64

    Notes
    -----
    - This cache treats the *contents* of arrays as part of the key,
      so even small differences in floating-point values produce distinct cache
      entries.
    - Custom classes are converted to a string and used verbatim. Two custom
      objects that print identically will collide.
    - The cache is implemented using ``collections.OrderedDict`` and maintains
      LRU behavior manually.

    Examples
    --------
    >>> cached_func = SimpleArrayCache(maxsize=64)(func)
    """
    def __init__(self, maxsize = 32):
        self.maxsize = maxsize
        self._store = OrderedDict()

    def _key(self, *args, **kwargs):

        key    = []
        inputs = list(args) + [kwargs[k] for k in kwargs.keys()]

        for a in inputs:

            if isinstance(a, (int, float, str)):
                key.append(a)

            elif isinstance(a, (list, tuple)):
                a = np.array(a)
                key.append(a.shape)
                key.append(a.dtype.str)
                key.append(a.tobytes())

            elif isinstance(a, (np.ndarray)):
                key.append(a.shape)
                key.append(a.dtype.str)
                key.append(a.tobytes())

            else:
                key.append(str(a))

        return tuple(key)
    

    def get(self, *args, **kwargs):
        k = self._key(*args, **kwargs)
        if k in self._store:
            self._store.move_to_end(k)
            return self._store[k]
        return None

    def set(self, value, *args, **kwargs):
        k = self._key(*args, **kwargs)
        self._store[k] = value
        self._store.move_to_end(k)
        if len(self._store) > self.maxsize:
            self._store.popitem(last=False)


    def __call__(self, func):

        functools.wraps(func)
        def cached_func(*args, **kwargs):
            cached = self.get(*args, **kwargs)
            
            if cached is not None:
                return cached
            
            val = func(*args, **kwargs)
            self.set(val, *args, **kwargs)

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

    def __init__(self, Profile, maxsize = 64, methods = ['real', 'projected', 'fourier']):
        
        assert isinstance(methods, list), f"You passed methods = {methods}, but we need a list of strings"

        self.Profile   = Profile
        self.maxsize   = maxsize
        self.methods   = methods

        for m in self.methods:
            setattr(self, m, SimpleArrayCache(self.maxsize)(getattr(self.Profile, m)))
        
        #We just set this to the same as the inputted profile.
        super().__init__(mass_def = self.Profile.mass_def)

        self.update_precision_fftlog(**self.Profile.precision_fftlog.to_dict())


    def __getattr__(self, key):

        safe_keys = self.methods + ['Profile', 'maxsize']

        if key in safe_keys:
            return object.__getattribute__(self, key)
        else:
            return getattr(object.__getattribute__(self, 'Profile'), key)
        

    def __str_prf__(self):

        return f"Cached[{self.Profile.__str_prf__()}]"
    
    def __str_par__(self): return self.Profile.__str_par__()


class CachedHODProfile(CachedProfile, ccl.halos.profiles.hod.HaloProfileHOD):

    def __init__(self, Profile, maxsize = 64, 
                 methods = ['get_normalization', '_fourier_variance', '_usat_fourier', '_fourier',  'fourier', 'real']):

        self.Profile   = Profile
        self.maxsize   = maxsize
        self.methods   = methods

        for m in self.methods:
            setattr(self, m, SimpleArrayCache(self.maxsize)(getattr(self.Profile, m)))
        
        #We just set this to the same as the inputted profile.
        BaseBFGProfiles.__init__(self, mass_def = self.Profile.mass_def)

        self.update_precision_fftlog(**self.Profile.precision_fftlog.to_dict())
