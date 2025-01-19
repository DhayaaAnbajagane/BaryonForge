import numpy as np
import pyccl as ccl
from operator import add, mul, sub, truediv, pow, abs, neg, pos
import warnings
from scipy import interpolate

__all__ = ['generate_operator_method', 'destory_Pk', 'build_cosmodict', 'safe_Pchip_minimize']

def generate_operator_method(op, reflect = False):
    """
    Defines a method for generating simple arithmetic operations for the Profile classes.

    The `generate_operator_method` function dynamically creates methods that can be used to perform
    arithmetic operations (such as addition, subtraction, multiplication, etc.) on instances of the
    `HaloProfile` classes or similar. This function alters the `_real()` routine in the `HaloProfile`
    classes so that the new result is equivalent to computing the `_real()` of the individual classes
    and then performing the specified arithmetic operation.

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

            assert isinstance(other, (int, float, ccl.halos.profiles.HaloProfile)), f"Object must be int/float/SchneiderProfile but is type '{type(other).__name__}'."

            Combined = self.__class__(**self.model_params, xi_mm = self.xi_mm, 
                                      padding_lo_proj   = self.padding_lo_proj, 
                                      padding_hi_proj   = self.padding_hi_proj, 
                                      n_per_decade_proj = self.n_per_decade_proj,
                                      mass_def = self.mass_def)

            def __tmp_real__(cosmo, r, M, a):

                A = self._real(cosmo, r, M, a)

                if isinstance(other, ccl.halos.profiles.HaloProfile):
                    B = other._real(cosmo, r, M, a)
                else:
                    B = other

                if not reflect:
                    return op(A, B)
                else:
                    return op(B, A)

            def __str_prf__():

                op_name = op.__name__
                cl_name = self.__str_prf__()
                
                #Check if it has cutom string output already
                #If not then use the class name. Or if int/float just use number
                if isinstance(other, ccl.halos.profiles.HaloProfile):
                    if hasattr(other, '__str_prf__'):
                        ot_name = other.__str_prf__()
                    else:
                        ot_name = other.__class__.__name__ 
                else: 
                    ot_name = other

                
                if not reflect:
                    return f"{op_name}[{cl_name}, {ot_name}]"
                else:
                    return f"{op_name}[{ot_name}, {cl_name}]"

            Combined._real = __tmp_real__
            Combined.__str_prf__ = __str_prf__

            return Combined

    #For some operators we don't need a second input, so rewrite function for that
    elif op in [abs, neg, pos]:

        def operator_method(self):
            
            Combined = self.__class__(**self.model_params, xi_mm = self.xi_mm, 
                                      padding_lo_proj = self.padding_lo_proj, 
                                      padding_hi_proj = self.padding_hi_proj, 
                                      n_per_decade_proj = self.n_per_decade_proj,
                                      mass_def = self.mass_def)

            def __tmp_real__(cosmo, r, M, a):

                A = self._real(cosmo, r, M, a)

                return op(A)

            def __str_prf__():

                op_name = op.__name__
                cl_name = self.__str_prf__()

                return f"{op_name}[{cl_name}]"

            Combined._real = __tmp_real__
            Combined.__str_prf__ = __str_prf__

            return Combined

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
    If `sigma8` is not already computed in the Cosmology object, this function invokes 
    `cosmo.compute_sigma()` to compute and update its value before returning the dictionary.
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
        cosmo.compute_sigma()
        cdict['sigma'] = cosmo.cosmo.params.sigma8
        
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