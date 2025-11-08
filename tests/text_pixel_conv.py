import BaryonForge as bfg
import numpy as np, healpy as hp
import pyccl as ccl

from defaults import bpar_S19, bpar_A20, ccl_dict, h
cosmo = ccl.Cosmology(**ccl_dict)
cosmo.compute_growth()

M = np.geomspace(1e11, 1e16, 5)
R = np.geomspace(1e-3, 1e3, 10)
k = np.geomspace(1e-3, 1e3, 10)

def test_nobeam():

    DMB  = bfg.Profiles.Schneider19.DarkMatterBaryon(**bpar_S19)
    PIX  = bfg.utils.NoPix()
    DMBP = bfg.utils.ConvolvedProfile(DMB, PIX)
    
    for a in [0.1, 0.5, 1]:
        ProfA = DMB.real(cosmo, R, M, a)
        ProfB = DMBP.real(cosmo, R, M, a)
        np.testing.assert_allclose(ProfA, ProfB, rtol = 1e-3, atol = 1e-8)

        ProfA = DMB.projected(cosmo, R, M, a)
        ProfB = DMBP.projected(cosmo, R, M, a)
        np.testing.assert_allclose(ProfA, ProfB, rtol = 1e-3, atol = 1e-8)


if __name__ == '__main__':
    
    test_nobeam()