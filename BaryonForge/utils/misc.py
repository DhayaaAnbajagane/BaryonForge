import numpy as np
import pyccl as ccl
from operator import add, mul, sub, truediv, pow, abs, neg, pos
import warnings
from scipy import interpolate

__all__ = ['generate_operator_method', 'destory_Pk', 'build_cosmodict', 'safe_Pchip_minimize', 'combine_fftpars']


def _default_mass_def(model, default = ccl.halos.massdef.MassDef200c):
    """
    Returns the mass definition of a model: a profile, a `BaryonificationClass`, or a tabulated
    profile (which holds the profile in its `model` attribute). Falls back to `default` (200c)
    if no mass definition can be found, eg. if `model` is None.
    """

    for obj in (model, getattr(model, 'model', None)):
        mass_def = getattr(obj, 'mass_def', None)
        if mass_def is not None: return mass_def

    return default


def generate_operator_method(op, reflect = False):
    """
    Defines a method for generating simple arithmetic operations for the Profile classes.

    The `generate_operator_method` function dynamically creates methods that can be used to perform
    arithmetic operations (such as addition, subtraction, multiplication, etc.) on instances of the
    `HaloProfile` classes or similar. The operation returns a `CombinedProfile`, whose `real()` is
    equivalent to computing the `real()` of the individual classes and then performing the specified
    arithmetic operation. See `CombinedProfile` for how the projected and Fourier profiles are computed.

    Parameters
    ----------
    op : function
        The arithmetic operation to be applied. It should be one of the Python arithmetic operators like
        `add`, `mul`, `sub`, `pow`, `truediv`, `abs`, `neg`, or `pos` from the `operator` module.
    
    reflect : bool, optional
        If `True`, the order of the operands is reversed (i.e., the operation is performed with the
        second operand as the first). Used to handle right-handed/left-handed operations. Default is `False`.

    Returns
    -------
    operator_method : function
        A function that defines how the arithmetic operation should be performed on the `HaloProfile`
        classes. This function can handle operations with other `HaloProfile` instances, integers, or
        floats.

    Examples
    --------
    To use `generate_operator_method` for addition, you might do:

    >>> from operator import add
    >>> add_method = generate_operator_method(add)
    >>> profile_sum = add_method(profile1, profile2)
    
    Notes
    -----
    - The method returned can handle both unary and binary operations depending on the operator specified.
    """

    if op in [add, mul, sub, pow, truediv]:
        def operator_method(self, other):

            assert isinstance(other, (int, float, np.number, ccl.halos.profiles.HaloProfile)), \
                   f"Object must be int/float/HaloProfile but is type '{type(other).__name__}'."

            #Imported here to avoid a circular import (Profiles.Base imports this module)
            from ..Profiles.misc import CombinedProfile

            return CombinedProfile(op, self, other, reflect = reflect)

    #For some operators we don't need a second input, so rewrite function for that
    elif op in [abs, neg, pos]:

        def operator_method(self):

            from ..Profiles.misc import CombinedProfile

            return CombinedProfile(op, self)

    return operator_method


def destory_Pk(cosmo):
    """
    Removes some SwigPyObject from cosmo that are unable to be pickled
    and therefore caused the multiprocessing to break.

    Parameters
    ----------
    cosmo : object
        A ccl Cosmology object
    
    Returns
    -------
    cosmo : object
        The ccl Cosmology object but with some attributes (that are necessarily
        SwigPyObjects) destoryed in order to allow pickling.

    Notes
    -----
    - For efficiency in pipeline, this function should be used on a Cosmology object
    only once the main profile/cosmology operations are finished. Otherwise this
    forces the recalculation of the power spectrum (which may not be an issue depending
    on your use-case, but something to be aware of).
    """

    cosmo._pk_lin = {}
    cosmo._pk_nl  = {}

    return cosmo


def build_cosmodict(cosmo):
    """
    Generate a dictionary containing a subset of cosmological parameters from a pyccl Cosmology object.
    
    This function extracts key cosmological parameter values from a pyccl Cosmology object and stores 
    them in a dictionary. If the `sigma8` parameter is not already computed (NaN), it triggers its computation.

    Parameters
    ----------
    cosmo : pyccl.core.Cosmology
        A pyccl Cosmology object containing the cosmological model parameters.
    
    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'Omega_m' : float
            The total matter density parameter.
        - 'Omega_b' : float
            The baryonic matter density parameter.
        - 'sigma8' : float
            The normalization of the power spectrum.
        - 'h' : float
            The dimensionless Hubble constant.
        - 'n_s' : float
            The spectral index of the primordial power spectrum.
        - 'w0' : float
            The equation-of-state parameter for dark energy (constant term).
        - 'wa' : float
            The equation-of-state parameter for dark energy (time-dependent term).
    
    Notes
    -----
    If `sigma8` is not set in the Cosmology object (eg. it was initialized with `A_s`), this
    function computes it with `pyccl.sigma8(cosmo)` before returning the dictionary.
    """
    
    cdict = {'Omega_m' : cosmo.cosmo.params.Omega_m,
             'Omega_b' : cosmo.cosmo.params.Omega_b,
             'sigma8'  : cosmo.cosmo.params.sigma8,
             'h'       : cosmo.cosmo.params.h,
             'n_s'     : cosmo.cosmo.params.n_s,
             'w0'      : cosmo.cosmo.params.w0,
             'wa'      : cosmo.cosmo.params.wa,
            }
    
    if np.isnan(cdict['sigma8']):
        cdict['sigma8'] = float(ccl.sigma8(cosmo))
        
    return cdict

def safe_Pchip_minimize(x, y):

    if (np.min(x) > 0) | (np.max(x) < 0):
        warnings.warn(f"Cannot minimize. Range {np.min(x)} < LHS - RHS < {np.max(x)} does not include zero!"
                        "Setting R_ej = inf. This error can occur if f_ej = 0.", UserWarning)
        return np.inf
    
    cen = np.argmin(np.abs(x - 0)) #Find the point around which we should search for minima
    buf = 5 #Large enough (one-sided) buffer in case any weird interpolator effects from using too few points
    ind = slice(max(cen - buf, 0), min(cen + buf, len(x)))

    #Many reasons why Pchip could fail
    try:
        return interpolate.PchipInterpolator(x[ind], y[ind])(0)
    
    except ValueError:
        warnings.warn(f"Cannot minimize. Interpolator crashed. Reverting to using np.min()", UserWarning)
        return y[cen]
    

#FFT params and the rules for determining a 
#superset of parameter values between two profiles
_fft_precision_logic = {
    'padding_lo_fftlog' : min,
    'padding_hi_fftlog' : max,
    'n_per_decade'      : max,
    'extrapol'          : None,
    'padding_lo_extra'  : min,
    'padding_hi_extra'  : max,
    'large_padding_2D'  : None,
    'plaw_fourier'      : None,
    'plaw_projected'    : None
}


def combine_fftpars(setA, setB):
    """
    Combine two dictionaries of FFT-related parameters using predefined
    merge rules.

    This function merges two parameter dictionaries (``setA`` and ``setB``)
    by iterating over a global list of merge rules (``_fft_precision_logic``).
    Each entry in ``_fft_precision_logic`` specifies:
    
    - a key ``k`` expected in both dictionaries, and  
    - a merge rule ``rule`` which is either a callable or ``None``.

    If ``rule`` is a callable, it is applied to the corresponding entries
    ``setA[k]`` and ``setB[k]`` to compute the merged value.

    If ``rule`` is ``None`` and the values in ``setA`` and ``setB`` differ,
    the value from ``setA`` is taken as the default and a warning is issued.
    If the values match, that value is used.

    Parameters
    ----------
    setA, setB : dict
        First and second dictionary of FFT parameter values. All keys appearing in
        ``_fft_precision_logic`` must be present.

    Returns
    -------
    dict
        A new dictionary containing the merged FFT parameter values.

    Warns
    -----
    UserWarning
        If a merge rule is ``None`` and the values for a parameter differ
        between ``setA`` and ``setB``. In this case, the value from ``setA``
        is used.
    """

    new_set = {}

    #Loop over all parameters defined by CCL's FFT precision
    for k, rule in _fft_precision_logic.items():

        A, B = setA[k], setB[k]

        #If we said there is no merging rule, it means we have no
        #way of automatically setting a value to the combined profile
        #so just pick the value in the first dict and call it a day (with a warning).
        if (rule == None):

            new_set[k] = A

            if A != B:
                warnings.warn(f"Value of parameter {k} is inconsistent between two profiles you are combining ({A}, {B}). "
                              f"We will use {A} as the default value")
            
        #Otherwise, we have clear rules for setting params
        #such that you encompass the requirements of
        #both profiles.
        else:
            new_set[k] = rule(A, B)

    return new_set
