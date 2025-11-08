import BaryonForge as bfg
import numpy as np, healpy as hp
import pyccl as ccl

from defaults import bpar_S19, bpar_A20, ccl_dict, h
cosmo = ccl.Cosmology(**ccl_dict)
cosmo.compute_growth()

M = np.geomspace(1e11, 1e16, 5)
R = np.geomspace(1e-3, 1e3, 10)
k = np.geomspace(1e-3, 1e3, 10)

def test_profile2profile():

    DMB = bfg.Profiles.Schneider19.DarkMatterBaryon(**bpar_S19)
    THL = bfg.Profiles.Schneider19.TwoHalo(**bpar_S19)
    
    SUB = DMB - THL
    ZER = bfg.Profiles.misc.Zeros()
    MOD = bfg.Profiles.Schneider19.DarkMatterBaryon(**bpar_S19, twohalo = ZER)

    for a in [0.1, 0.5, 1]:
        ProfA = SUB.real(cosmo, R, M, a)
        ProfB = MOD.real(cosmo, R, M, a)
        np.testing.assert_allclose(ProfA, ProfB, rtol = 1e-6, atol = 1e-8)

        ProfA = SUB.projected(cosmo, R, M, a)
        ProfB = MOD.projected(cosmo, R, M, a)
        np.testing.assert_allclose(ProfA, ProfB, rtol = 1e-6, atol = 1e-8)

if __name__ == '__main__':
    pass