import BaryonForge as bfg
import numpy as np, healpy as hp
import pyccl as ccl

from defaults import bpar_S19, bpar_A20, cosmo, h

M = np.geomspace(1e11, 1e16, 5)
R = np.geomspace(1e-3, 1e3, 10)
k = np.geomspace(1e-3, 1e3, 10)

def test_schneider19():

    for a in [0.1, 0.5, 1]:
        bfg.Profiles.Schneider19.Stars(**bpar_S19).real(cosmo, M, R, a)
        bfg.Profiles.Schneider19.Stars(**bpar_S19).projected(cosmo, M, R, a)
        bfg.Profiles.Schneider19.Stars(**bpar_S19).fourier(cosmo, M, k, a)

def test_arico20():

    for a in [0.1, 0.5, 1]:

        bfg.Profiles.Arico20.Stars(**bpar_A20).real(cosmo, M, R, a)
        bfg.Profiles.Arico20.Stars(**bpar_A20).projected(cosmo, M, R, a)
        bfg.Profiles.Arico20.Stars(**bpar_A20).fourier(cosmo, M, k, a)

def test_mead20():

    bpar_M20 = bfg.Profiles.Mead20.Params_TAGN_7p6_All

    for a in [0.1, 0.5, 1]:

        bfg.Profiles.Mead20.Stars(**bpar_M20).real(cosmo, M, R, a)
        bfg.Profiles.Mead20.Stars(**bpar_M20).projected(cosmo, M, R, a)
        bfg.Profiles.Mead20.Stars(**bpar_M20).fourier(cosmo, M, k, a)


if __name__ == '__main__':

    test_schneider19()
    test_arico20()
    test_mead20()