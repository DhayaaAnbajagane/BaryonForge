
import copy
import struct
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
import pyccl as ccl
from ..Profiles.Base import BaseBFGProfiles
from .Tabulate import _get_parameter

__all__ = ['SimpleArrayCache', 'CachedProfile']

_MISSING = object() #Marks a key absent from the cache


def _dtype_key(value, _seen = None):
    """Build an immutable, recursive key for NumPy dtypes and dtype metadata."""
    if _seen is None: _seen = set()

    def enter(obj):
        marker = id(obj)
        if marker in _seen:
            raise TypeError("Cyclic dtype metadata cannot be used in a cache key.")
        _seen.add(marker)
        return marker

    if isinstance(value, np.dtype):
        marker = enter(value)
        try:
            names = value.names
            fields = None
            if names is not None:
                field_keys = []
                for name in names:
                    field = value.fields[name]
                    field_dtype, offset = field[:2]
                    title = field[2] if len(field) > 2 else None
                    field_keys.append((name, _dtype_key(field_dtype, _seen), int(offset),
                                       None if title is None else _dtype_key(title, _seen)))
                fields = tuple(field_keys)

            subdtype = value.subdtype
            subarray = (None if subdtype is None else
                        (_dtype_key(subdtype[0], _seen), tuple(int(x) for x in subdtype[1])))
            metadata = _dtype_key(value.metadata, _seen)
            return (
                "dtype", value.type, value.kind, value.char, value.byteorder,
                int(value.itemsize), int(value.alignment), bool(value.hasobject),
                bool(value.isalignedstruct), names, fields, subarray, metadata,
            )
        finally:
            _seen.remove(marker)

    if value is None: return ("none",)
    if type(value) is bool: return ("bool", value)
    if type(value) is int: return ("int", value)
    if type(value) is float: return ("float", struct.pack("!d", value))
    if type(value) is complex: return ("complex", struct.pack("!dd", value.real, value.imag))
    if type(value) is str: return ("str", value)
    if type(value) is bytes: return ("bytes", value)

    if isinstance(value, np.generic):
        dtype_key = _dtype_key(value.dtype, _seen)
        array = np.asarray(value)
        if value.dtype.hasobject:
            value_key = _dtype_key(value.item(), _seen)
        else:
            value_key = array.tobytes()
        return ("numpy-scalar", dtype_key, value_key)

    if isinstance(value, np.ndarray):
        marker = enter(value)
        try:
            dtype_key = _dtype_key(value.dtype, _seen)
            values = (tuple(_dtype_key(item, _seen) for item in value.flat)
                      if value.dtype.hasobject else value.tobytes())
            return ("ndarray", type(value), tuple(value.shape), dtype_key, values)
        finally:
            _seen.remove(marker)

    if isinstance(value, Mapping):
        marker = enter(value)
        try:
            items = ((_dtype_key(key, _seen), _dtype_key(item, _seen))
                     for key, item in value.items())
            return ("mapping", tuple(sorted(items, key=repr)))
        finally:
            _seen.remove(marker)

    if type(value) is list:
        marker = enter(value)
        try: return ("list", tuple(_dtype_key(item, _seen) for item in value))
        finally: _seen.remove(marker)
    if type(value) is tuple:
        marker = enter(value)
        try: return ("tuple", tuple(_dtype_key(item, _seen) for item in value))
        finally: _seen.remove(marker)
    if type(value) in (set, frozenset):
        marker = enter(value)
        try:
            items = tuple(sorted((_dtype_key(item, _seen) for item in value), key=repr))
            return (type(value).__name__, items)
        finally: _seen.remove(marker)

    raise TypeError(
        "Unsupported dtype metadata value of type "
        f"{type(value).__module__}.{type(value).__qualname__}; use immutable primitive/container metadata."
    )


class _CachedFunction:
    """Pickleable callable used by :class:`SimpleArrayCache`."""

    def __init__(self, cache, func):
        self.cache = cache
        self.func = func

    def __call__(self, *args, **kwargs):
        #Keep positional and keyword arguments in separate, tagged fields. A
        #flat concatenation aliases func("x", 1) with func(x=1).
        key = (("args", args), ("kwargs", tuple(sorted(kwargs.items()))))

        #Always hand out copies, so that callers modifying the output in-place
        #cannot corrupt the cached value
        found, value = self.cache.lookup(key)
        if found:
            return copy.deepcopy(value)

        value = self.func(*args, **kwargs)
        self.cache.set(copy.deepcopy(value), key)
        return value


class SimpleArrayCache:
    """
    A lightweight LRU-style cache designed for functions whose inputs include
    NumPy arrays. Unlike ``functools.lru_cache``, this cache supports
    unhashable arguments (e.g. ``numpy.ndarray``) by constructing a stable
    byte-based key from the array contents.

    The cache stores results using tagged, structured keys. Array keys contain
    their type, shape, dtype, and values; lists, tuples, and mappings retain
    their boundaries and types. Arbitrary objects use identity so distinct
    objects with the same string representation cannot alias.

    When used as a decorator, the cache wraps a function of the form
    ``func(*args)`` and automatically caches its return value based
    on these arguments. Repeated calls with identical inputs return (a copy of)
    the cached result without re-evaluating the function. Copies are returned so
    that in-place modifications of the output cannot corrupt the cache.

    Parameters
    ----------
    maxsize : int, optional
        Maximum number of cached entries to store. The cache evicts the
        least recently used (LRU) entry when the limit is exceeded.
        Default is 32

    Notes
    -----
    - This cache treats the *contents* of arrays as part of the key,
      so even small differences in floating-point values produce distinct cache
      entries.
    - Arbitrary objects are retained by the key and keyed by identity. Their
      string representation is also snapshotted, so visible state changes
      invalidate the prior key.
    - The cache is implemented using ``collections.OrderedDict`` and maintains
      LRU behavior manually.

    Examples
    --------
    >>> cached_func = SimpleArrayCache(maxsize=64)(func)
    """

    class _IdentityKey:
        """Hash an arbitrary input by identity while retaining it against id reuse."""

        __slots__ = ("value", "text", "_hash")

        def __init__(self, value):
            self.value = value
            self.text = str(value)
            self._hash = hash((type(value), id(value), self.text))

        def __hash__(self):
            return self._hash

        def __eq__(self, other):
            return (isinstance(other, SimpleArrayCache._IdentityKey)
                    and self.value is other.value
                    and self.text == other.text)

        def __repr__(self):
            return f"_IdentityKey(type={type(self.value)!r}, id={id(self.value)}, text={self.text!r})"

    def __init__(self, maxsize = 32):
        self.maxsize = maxsize
        self._store = OrderedDict()

    def _key(self, *args):

        def key_arg(arg):
            """Return a hashable, type-tagged snapshot of one cache argument."""
            if isinstance(arg, np.ndarray):
                if arg.dtype.hasobject:
                    values = tuple(key_arg(value) for value in arg.flat)
                else:
                    values = arg.tobytes()
                return ("ndarray", type(arg), arg.shape, _dtype_key(arg.dtype), values)

            if isinstance(arg, np.generic):
                array = np.asarray(arg)
                value = (key_arg(arg.item()) if array.dtype.hasobject
                         else array.tobytes())
                return ("numpy-scalar", type(arg), _dtype_key(array.dtype), value)

            if arg is None: return ("none",)
            if type(arg) is bool: return ("bool", arg)
            if type(arg) is int: return ("int", arg)
            if type(arg) is float: return ("float", struct.pack("!d", arg))
            if type(arg) is complex:
                return ("complex", struct.pack("!dd", arg.real, arg.imag))
            if type(arg) is str: return ("str", arg)
            if type(arg) is bytes: return ("bytes", arg)

            if type(arg) is list:
                return ("list", tuple(key_arg(value) for value in arg))
            if type(arg) is tuple:
                return ("tuple", tuple(key_arg(value) for value in arg))
            if type(arg) is dict:
                items = ((key_arg(key), key_arg(value)) for key, value in arg.items())
                return ("dict", tuple(sorted(items, key=repr)))
            if type(arg) in (set, frozenset):
                values = tuple(sorted((key_arg(value) for value in arg), key=repr))
                return (type(arg).__name__, values)

            return ("object", type(arg), self._IdentityKey(arg))

        return tuple(key_arg(arg) for arg in args)
    

    def get(self, *args):
        k = self._key(*args)
        if k in self._store:
            self._store.move_to_end(k)
            return self._store[k]
        return None

    def contains(self, *args):
        """Return whether an entry exists, including entries valued ``None``."""
        return self._key(*args) in self._store

    def lookup(self, *args):
        """
        Return ``(True, value)`` if an entry exists (including entries valued ``None``), else ``(False, None)``.
        The entry is read in a single step, so this is safe when runner threads (``n_jobs > 1``) share the
        cache, unlike ``contains`` followed by ``get``, between which another thread may evict the entry.
        """
        k = self._key(*args)
        value = self._store.get(k, _MISSING)
        if value is _MISSING: return False, None
        try:
            self._store.move_to_end(k)
        except KeyError:
            pass #Evicted by another thread in the meantime
        return True, value

    def set(self, value, *args):
        k = self._key(*args)
        self._store[k] = value
        try:
            self._store.move_to_end(k)
            if len(self._store) > self.maxsize:
                self._store.popitem(last=False)
        except KeyError:
            pass #Another thread evicted/emptied entries in the meantime


    def __call__(self, func):
        return _CachedFunction(self, func)
        

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

        #Profile methods that are not cached are forwarded to the input profile directly.
        #Otherwise, they would be computed by this wrapper with its own (default) settings.
        for m in ['real', 'projected', 'fourier']:
            if m not in self.methods: setattr(self, m, getattr(self.Profile, m))

        #We just set this to the same as the inputted profile.
        cutoffs = {k : _get_parameter(self.Profile, k) for k in ['cutoff', 'proj_cutoff']}
        cutoffs = {k : v for k, v in cutoffs.items() if v is not None}
        super().__init__(mass_def = self.Profile.mass_def, **cutoffs)

        ccl.halos.profiles.HaloProfile.update_precision_fftlog(self, **self.Profile.precision_fftlog.to_dict())


    def __getattr__(self, key):

        # During unpickling, attributes may be requested before ``methods``
        # has been restored.  Read it through object.__getattribute__ so the
        # delegation path cannot recurse indefinitely.
        try:
            methods = object.__getattribute__(self, 'methods')
        except AttributeError:
            raise AttributeError(key) from None

        safe_keys = methods + ['Profile', 'maxsize']

        if key in safe_keys:
            return object.__getattribute__(self, key)
        else:
            return getattr(object.__getattribute__(self, 'Profile'), key)
        

    def __str_prf__(self):

        return f"Cached[{self.Profile.__str_prf__()}]"
    
    def __str_par__(self): return self.Profile.__str_par__()


class CachedHODProfile(CachedProfile, ccl.halos.profiles.hod.HaloProfileHOD):

    def __init__(self, Profile, maxsize = 64, 
                 methods = ['get_normalization', '_fourier_variance', '_fourier', 'fourier', 'real']):

        self.Profile   = Profile
        self.maxsize   = maxsize
        self.methods   = methods

        for m in self.methods:
            setattr(self, m, SimpleArrayCache(self.maxsize)(getattr(self.Profile, m)))
        
        #BaseBFGProfiles.__init__ is not run for this class, so set the attributes its properties use
        self._c_M_relation = None
        self._use_fftlog_projection = False

        #We just set this to the same as the inputted profile (HOD profiles also need a concentration).
        ccl.halos.profiles.hod.HaloProfileHOD.__init__(self, mass_def = self.Profile.mass_def,
                                                       concentration = self.Profile.concentration)

        ccl.halos.profiles.HaloProfile.update_precision_fftlog(self, **self.Profile.precision_fftlog.to_dict())
