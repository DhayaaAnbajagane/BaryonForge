
import numpy as np
import pyccl as ccl

from scipy import interpolate
from astropy.cosmology import z_at_value, FlatLambdaCDM, FlatwCDM
from astropy import units as u

from .Profiles.Schneider19Profiles import DarkMatterOnly, DarkMatterBaryon

class BaryonificationClass(object):

    def __init__(self, DMO, DMB, ccl_cosmo, R_range = [1e-5, 40], N_samples = 500, epsilon_max = 4,
                 mass_def = ccl.halos.massdef.MassDef(200, 'critical')):


        self.DMO = DMO
        self.DMB = DMB
        self.ccl_cosmo   = ccl_cosmo #CCL cosmology instance
        self.R_range     = R_range
        self.epsilon_max = epsilon_max
        self.N_samples   = N_samples
        self.mass_def    = mass_def

        self.setup_interpolator()


    def get_masses(self, model, r, M, a, mass_def):

        raise NotImplementedError("Implement a get_masses() method first")


    def setup_interpolator(self, z_min = 1e-2, z_max = 5, M_min = 1e12, M_max = 1e16, N_samples_Mass = 30, N_samples_z = 30):

        M_range = np.geomspace(M_min, M_max, N_samples_Mass)
        z_range = np.geomspace(z_min, z_max, N_samples_z)
        r    = np.geomspace(self.R_range[0], self.R_range[1], self.N_samples)
        dlnr = np.log(r[1]) - np.log(r[0])

        M_DMO_interp = np.zeros([z_range.size, M_range.size, r.size])
        M_DMB_interp = np.zeros([z_range.size, M_range.size, r.size])

        M_DMB_range_interp = np.geomspace(1e5, 1e18, self.N_samples)
        log_r_new_interp   = np.zeros([z_range.size, M_range.size, M_DMB_range_interp.size])

        for i in range(M_range.size):
            for j in range(z_range.size):

                #Extra factor of "a" accounts for projection in ccl being done in comoving, not physical units
                M_DMO_interp[j, i, :] = self.get_masses(self.DMO, r, M_range[i], 1/(1 + z_range[j]), mass_def = self.mass_def)
                M_DMB_interp[j, i, :] = self.get_masses(self.DMB, r, M_range[i], 1/(1 + z_range[j]), mass_def = self.mass_def)

                log_r_new_interp[j, i, :] = np.interp(np.log(M_DMB_range_interp), np.log(M_DMB_interp[j, i]), np.log(r))

        input_grid_1 = (np.log(1 + z_range), np.log(M_range), np.log(r))
        input_grid_2 = (np.log(1 + z_range), np.log(M_range), np.log(M_DMB_range_interp))

        self.interp_DMO = interpolate.RegularGridInterpolator(input_grid_1, np.log(M_DMO_interp), bounds_error = False)
        self.interp_DMB = interpolate.RegularGridInterpolator(input_grid_2, log_r_new_interp, bounds_error = False) #Reverse needed for practical application
        

        return 0


    def displacements(self, x, M, a):

        r    = np.geomspace(self.R_range[0], self.R_range[1], self.N_samples)
        dlnr = np.log(r[1]) - np.log(r[0])

        z = 1/a - 1

        M_use = np.atleast_1d(M)
        R     = self.mass_def.get_radius(self.ccl_cosmo, M_use, a)/a #in comoving Mpc

        offset = np.zeros_like(x)
        inside = (x > self.R_range[0]) & (x < self.epsilon_max*R)

        x = x[inside]

        empty = np.ones_like(x)
        z_in  = np.log(1 + z)*empty
        M_in  = np.log(M)*empty

        one = self.interp_DMO((z_in, M_in, np.log(x), ))
        two = self.interp_DMB((z_in, M_in, one, ))

        offset[inside] = np.exp(two) - x

        return offset


class Baryonification3D(BaryonificationClass):

    def get_masses(self, model, r, M, a, mass_def):

        dlnr = np.log(r[1]/r[0])
        rho  = model.real(self.ccl_cosmo, r, M, a, mass_def = mass_def)
        M    = np.cumsum(4*np.pi*r**3 * rho * dlnr)

        return M


class Baryonification2D(BaryonificationClass):

    def get_masses(self, model, r, M, a, mass_def):

        dlnr  = np.log(r[1]/r[0])
        Sigma = model.projected(self.ccl_cosmo, r, M, a, mass_def = mass_def) * a #scale fac. needed because ccl projection done in comoving, not physical, units
        M     = np.cumsum(2*np.pi*r**2 * Sigma * dlnr)

        return M