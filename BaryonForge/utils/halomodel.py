import pyccl as ccl
import numpy as np

from pyccl.halos.halo_model import HMCalculator
from pyccl import unlock_instance

__all__ = ['FlexibleHMCalculator']

class FlexibleHMCalculator(HMCalculator):

    """This a modified class that implements a set of methods for
    computing various halo model quantities. A lot of these quantities
    will involve integrals of the sort:

    .. math::
       \\int dM\\,n(M,a)\\,f(M,k,a),

    where :math:`n(M,a)` is the halo mass function, and :math:`f` is
    an arbitrary function of mass, scale factor and Fourier scales.

    We have modified the class to also allow for M_delta, the spherical overdensity
    mass used when computing the mass function or bias, to be different from M_tot,
    the actual total mass of the halo if we integrate the density profile to infinity.
    This is an important effect when profiles are not truncated at R > R_delta.

    Parameters
    ----------
    mass_function : str or :class:`~pyccl.halos.halo_model_base.MassFunc`
        the mass function to use
    halo_bias : str or :class:`~pyccl.halos.halo_model_base.HaloBias`
        the halo bias function to use
    halo_m_to_mtot : function
        A function that converts M_delta mass to M_tot (latter integrated to r = infinity).
        Use an instance of the `~baryonforge.utils.Mdelta_to_Mtot` class.
    mass_def : str or :class:`~pyccl.halos.massdef.MassDef`
        the halo mass definition to use
    log10M_min : float
        lower bound of the mass integration range (base-10 logarithmic).
    log10M_max : float 
        lower bound of the mass integration range (base-10 logarithmic).
    nM : int 
        number of uniformly-spaced samples in :math:`\\log_{10}(M)`to be used in the mass integrals.
    integration_method_M : str 
        integration method to use in the mass integrals. Options: "simpson" and "spline".
    """ # noqa
    
    def __init__(self, *, mass_function, halo_bias, halo_m_to_mtot, mass_def = None,
                 log10M_min = 8., log10M_max = 16., nM = 128,
                 integration_method_M = 'simpson',):

        self.halo_m_to_mtot = halo_m_to_mtot
        super().__init__(mass_function = mass_function, 
                         halo_bias = halo_bias, 
                         mass_def  = mass_def,
                         log10M_min = log10M_min, log10M_max = log10M_max, nM = nM,
                         integration_method_M = integration_method_M)

    @unlock_instance(mutate=False)
    def _get_mass_function(self, cosmo, a, rho0):
        # Compute the mass function at this cosmo and a.
        if a != self._a_mf or cosmo != self._cosmo_mf:
            self._mf = self.mass_function(cosmo, self._mass, a)
            self._mtot  = self.halo_m_to_mtot(cosmo, self._mass, a)
            self._mtot0 = self._mtot[0]
            integ = self._integrator(self._mf*self._mtot, self._lmass)
            self._mf0 = (rho0 - integ) / self._mtot0
            self._cosmo_mf, self._a_mf = cosmo, a  # cache

    @unlock_instance(mutate=False)
    def _get_halo_bias(self, cosmo, a, rho0):
        # Compute the halo bias at this cosmo and a.
        if a != self._a_bf or cosmo != self._cosmo_bf:
            self._bf = self.halo_bias(cosmo, self._mass, a)
            integ = self._integrator(self._mf*self._bf*self._mtot, self._lmass)
            self._mbf0 = (rho0 - integ) / self._mtot0
            self._cosmo_bf, self._a_bf = cosmo, a  # cache