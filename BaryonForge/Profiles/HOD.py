"""
Adapted from code written by pranjalrs,
model from Shivam Pandey
"""

import numpy as np
import pyccl as ccl
from operator import add, mul, sub, truediv, pow, neg, pos, abs
import warnings

from scipy import interpolate, special, integrate
from ..utils import _set_parameter, safe_Pchip_minimize
from .misc import Zeros, Truncation
from . import Base

__all__ = ['model_params', 'HODTransformerPandey25']


model_params = ['Profile', 'halo_m_to_mtot', 'Mstar_th', 'M1_fshmr', 'log10M1_a_fshmr', 
				'Mstar0_fshmr', 'log10Mstar0_a_fshmr', 'beta_fshmr', 'beta_a_fshmr', 'delta_shmr',
				'delta_a_shmr', 'gamma_fshmr', 'gamma_a_fshmr', 'siglogMstar_Ncen', 
				'Bsat_Nsat', 'betasat_Nsat', 'Bcut_Nsat', 'betacut_Nsat', 'alphasat_Nsat']

class HODTransformerPandey25(Base.BaseBFGProfiles):

	model_param_names = model_params

	def __init__(self, Profile, halo_m_to_mtot, **kwargs):
		
		Base.BaseBFGProfiles.__init__(self, Profile = Profile, halo_m_to_mtot = halo_m_to_mtot, **kwargs)

		#Make sure profile's stellar fractions derive from the
		#HODTransformer, and not from something else. This recursively sets this
		#for all objects in the profile
		_set_parameter(self.Profile, 'get_f_star_cen' , self.get_f_star_cen)
		_set_parameter(self.Profile, 'get_f_star_sat' , self.get_f_star_sat)
		_set_parameter(self.Profile, 'get_f_star', 		self.get_f_star)
    
	def Mhalo_Mstar(self, Mstar, a, cosmo):
		"""
		Returns halo mass given stellar mass using the inverse of the stellar-to-halo mass relation
		"""
		beta_SHMR  = self.beta_fshmr  + self.beta_a_fshmr  * (a-1)
		delta_SHMR = self.delta_shmr  + self.delta_a_shmr  * (a-1)
		gamma_SHMR = self.gamma_fshmr + self.gamma_a_fshmr * (a-1)

		M1         = self.M1_fshmr     * np.power(10, self.log10M1_a_fshmr     * (a - 1))
		Mstar0     = self.Mstar0_fshmr * np.power(10, self.log10Mstar0_a_fshmr * (a - 1))

		x = Mstar/Mstar0
		log10_inv_fSHMR = np.log10(M1) + beta_SHMR* np.log10(x) + (x)**delta_SHMR/(1 + x**-gamma_SHMR) - 0.5

		return np.power(10, log10_inv_fSHMR)
	

	def Mstar_Mhalo(self, M, a, cosmo):
		"""
		Returns stellar mass given halo mass using the (numerically) inverted inverse stellar-to-halo mass relation.
		"""

		Mstar = np.geomspace(1e4, 1e24, 101)
		Mhalo = self.Mhalo_Mstar(Mstar, a, cosmo)

		Mout  = np.power(10, np.interp(np.log10(M), np.log10(Mhalo), np.log10(Mstar)))
		
		return Mout
	

	def get_N_cen(self, M, a, cosmo, Mstar_th):
		"""
		Mean number of central galaxies
		<Ncen(M|Mstar_th)> = 0.5 * [1 - erf((log10(Mstar_th) - log10(fSHMR(M))) / (sqrt(2) * sigma_logMstar))]
		"""

		numerator = np.log10(Mstar_th) - np.log10(self.Mstar_Mhalo(M, a, cosmo))
		denominator = np.sqrt(2) * self.siglogMstar_Ncen
		Nc = 0.5 * (1 - special.erf(numerator / denominator))

		return Nc


	def get_N_sat(self, M, a, cosmo, Mstar_th):
		"""
		Mean number of satellite galaxies
		<Nsat(M|Mstar_th)> = <Ncen(M|Mstar_th)> * (M/Msat)^alpha_sat * exp(-Mcut/M)
		"""

		Mhalo = self.Mhalo_Mstar(Mstar_th, a, cosmo)

		Msat = self.Bsat_Nsat * (Mhalo/1e12)**self.betasat_Nsat * 1e12
		Mcut = self.Bcut_Nsat * (Mhalo/1e12)**self.betacut_Nsat * 1e12

		return self.get_N_cen(M, a, cosmo, Mstar_th) * (M/Msat)**self.alphasat_Nsat * np.exp(-Mcut/M)

	
	def _integrator(self, M, a, cosmo, Ngal):
		'''Eq. 10'''

		M = np.atleast_1d(M)

		if self.halo_m_to_mtot is None:
			M_tot = M
		else:
			M_tot = self.halo_m_to_mtot(cosmo, M, a)

		results	  = np.zeros_like(M, dtype=float)
		Mstar_min = self.Mstar_th
		Mstar_max = 1e16
		n_points  = 64

		Mstars = np.geomspace(Mstar_min, Mstar_max, n_points)


		for i, this_Mhalo in enumerate(M):
			# Discrete numerical integration (log-spaced) instead of quad
			integ = Ngal(this_Mhalo, a, cosmo, Mstars)
			term1 =  (Ngal(this_Mhalo, a, cosmo, Mstar_max)  	* Mstar_max - 
			 		  Ngal(this_Mhalo, a, cosmo, self.Mstar_th) * self.Mstar_th)

			term2 = np.trapz(integ * Mstars, x = np.log10(Mstars))

			results[i] = (term2 - term1) / M_tot[i]
		

		return results


	def get_f_star_cen(self, M, a, cosmo):
		f     = self._integrator(M, a, cosmo, self.get_N_cen)
		f_bar = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
		f 	  = np.clip(f, 1e-10, f_bar)

		return f

	def get_f_star_sat(self, M, a, cosmo):
		f = self._integrator(M, a, cosmo, self.get_N_sat)
		f = np.clip(f, 1e-10, None)

		f_cen = self.get_f_star_cen(M, a, cosmo)
		f_bar = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
		f_tot = f + f_cen

		f	  = f - np.clip(f_tot - f_bar, 0, f)
		
		return f
	
	
	def get_f_star(self, M, a, cosmo):
		return self.get_f_star_cen(M, a, cosmo) + self.get_f_star_sat(M, a, cosmo)
	

	def __getattr__(self, key):

		safe_keys = ['Mhalo_Mstar', 'Mstar_Mhalo', 'get_N_cen', 'get_N_sat', '_integrator']
		
		if key in safe_keys:
			return object.__getattribute__(self, key)
		else:
			return getattr(object.__getattribute__(self, 'Profile'), key)
		

	def __str_prf__(self):

		return f"{self.__class__.__name__}[{self.Profile.__str_prf__()}]"
		

class HODGalaxiesPandey25(HODTransformerPandey25, ccl.halos.profiles.hod.HaloProfileHOD):

	model_param_names = model_params

	def __init__(self, satellitestars, halo_m_to_mtot, **kwargs):
		self.SatelliteStars = satellitestars
		HODTransformerPandey25.__init__(self, Profile = None, halo_m_to_mtot = halo_m_to_mtot, **kwargs)

	def update_parameters(self, **kwargs):
		raise NotImplementedError("BFG profiles don't support in-place parameter updating. Please initialize a new class instance instead.")
	
	def _real(self, cosmo, r, M, a):

		return self._fftlog_wrap(cosmo, r, M, a, fourier_out=False)
	

	def __getattr__(self, key): return object.__getattribute__(self, key)
		
	
	def _fourier(self, cosmo, k, M, a):
		"""
		Interface with Fourier-space
		"""
		Ncen = self.get_N_cen(M, a, cosmo, self.Mstar_th)
		Nsat = self.get_N_sat(M, a, cosmo, self.Mstar_th)
		
		if len(Ncen) > 1:
			Ncen, Nsat = Ncen[:, None], Nsat[:, None]
		
		return Ncen + Nsat * self._usat_fourier(cosmo, k, M, a)


	def _usat_fourier(self, cosmo, k, M, a):
		"""
		Interface with Fourier-space for satellites
		"""
		#Should return a normalized profile i.e., u(k|M,a) should go to 1 on large scales
		
		Norm = self.halo_m_to_mtot(cosmo, M, a) * self.get_f_star_sat(M, a, cosmo)
		if len(Norm) > 1:
			Norm = Norm[:, None]

		#Handle case where frac == 0, so we don't want Norm = 0 and 1/Norm = NaN
		#in this case. So instead, just force 1/Norm = 0.
		Norm = np.where(Norm == 0, np.inf, Norm)			

		return self.SatelliteStars.fourier(cosmo, k, M, a) / Norm
	

	def _fourier_variance(self, cosmo, k, M, a):
		
		Ncen = self.get_N_cen(M, a, cosmo, self.Mstar_th)
		Nsat = self.get_N_sat(M, a, cosmo, self.Mstar_th)
		
		if len(Ncen) > 1:
			Ncen, Nsat = Ncen[:, None], Nsat[:, None]

		# NFW profile
		uk = self._usat_fourier(cosmo, k, M, a)

		prof = 2*Ncen*Nsat*uk + (Nsat*uk)**2

		if np.ndim(k) == 0: prof = np.squeeze(prof, axis=-1)
		if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)

		return prof


	def get_normalization(self, cosmo, a, hmc):
		"""Returns the normalization of this profile, which is the
		mean galaxy number density.

		Args:
			cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology
				object.
			a (:obj:`float`): scale factor.
			hmc (:class:`~pyccl.halos.halo_model.HMCalculator`): a halo
				model calculator object.

		Returns:
			:obj:`float`: normalization factor of this profile.
		"""

		def integ(M):
			Nc = self.get_N_cen(M, a, cosmo, self.Mstar_th)
			Ns = self.get_N_sat(M, a, cosmo, self.Mstar_th)
			return Nc + Ns
		
		return hmc.integrate_over_massfunc(integ, cosmo, a)