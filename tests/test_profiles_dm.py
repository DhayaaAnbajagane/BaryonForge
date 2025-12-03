import BaryonForge as bfg
import numpy as np, healpy as hp
import pyccl as ccl

from defaults import bpar_S19, bpar_S25, bpar_A20, ccl_dict, h
cosmo = ccl.Cosmology(**ccl_dict)
cosmo.compute_growth()

M = np.geomspace(1e11, 1e16, 5)
R = np.geomspace(1e-3, 1e3, 10)
k = np.geomspace(1e-3, 1e3, 10)

def test_schneider19():

    for a in [0.1, 0.5, 1]:
        bfg.Profiles.Schneider19.DarkMatter(**bpar_S19).real(cosmo, R, M, a)
        bfg.Profiles.Schneider19.DarkMatter(**bpar_S19).projected(cosmo, R, M, a)
        bfg.Profiles.Schneider19.DarkMatter(**bpar_S19).fourier(cosmo, k, M, a)

        bfg.Profiles.Schneider19.DarkMatterOnly(**bpar_S19).real(cosmo, R, M, a)
        bfg.Profiles.Schneider19.DarkMatterOnly(**bpar_S19).projected(cosmo, R, M, a)
        bfg.Profiles.Schneider19.DarkMatterOnly(**bpar_S19).fourier(cosmo, k, M, a)
    

    X = bfg.Profiles.Schneider19.DarkMatter(**bpar_S19).real(cosmo, R, M[0], a);        assert len(X.shape) == 1
    X = bfg.Profiles.Schneider19.DarkMatter(**bpar_S19).real(cosmo, R[0], M[0], a);     assert np.isscalar(X)
    X = bfg.Profiles.Schneider19.DarkMatterOnly(**bpar_S19).real(cosmo, R, M[0], a);    assert len(X.shape) == 1
    X = bfg.Profiles.Schneider19.DarkMatterOnly(**bpar_S19).real(cosmo, R[0], M[0], a); assert np.isscalar(X)


def test_schneider25():

    for a in [0.1, 0.5, 1]:
        bfg.Profiles.Schneider25.DarkMatter(**bpar_S25).real(cosmo, R, M, a)
        bfg.Profiles.Schneider25.DarkMatter(**bpar_S25).projected(cosmo, R, M, a)
        bfg.Profiles.Schneider25.DarkMatter(**bpar_S25).fourier(cosmo, k, M, a)

        bfg.Profiles.Schneider25.DarkMatterOnly(**bpar_S25).real(cosmo, R, M, a)
        bfg.Profiles.Schneider25.DarkMatterOnly(**bpar_S25).projected(cosmo, R, M, a)
        bfg.Profiles.Schneider25.DarkMatterOnly(**bpar_S25).fourier(cosmo, k, M, a)

    X = bfg.Profiles.Schneider25.DarkMatter(**bpar_S25).real(cosmo, R, M[0], a);        assert len(X.shape) == 1
    X = bfg.Profiles.Schneider25.DarkMatter(**bpar_S25).real(cosmo, R[0], M[0], a);     assert np.isscalar(X)
    X = bfg.Profiles.Schneider25.DarkMatterOnly(**bpar_S25).real(cosmo, R, M[0], a);    assert len(X.shape) == 1
    X = bfg.Profiles.Schneider25.DarkMatterOnly(**bpar_S25).real(cosmo, R[0], M[0], a); assert np.isscalar(X)


def test_arico20():

    for a in [0.1, 0.5, 1]:
        bfg.Profiles.Arico20.DarkMatter(**bpar_A20).real(cosmo, R, M, a)
        bfg.Profiles.Arico20.DarkMatter(**bpar_A20).projected(cosmo, R, M, a)
        bfg.Profiles.Arico20.DarkMatter(**bpar_A20).fourier(cosmo, k, M, a)

        bfg.Profiles.Arico20.DarkMatterOnly(**bpar_A20).real(cosmo, R, M, a)
        bfg.Profiles.Arico20.DarkMatterOnly(**bpar_A20).projected(cosmo, R, M, a)
        bfg.Profiles.Arico20.DarkMatterOnly(**bpar_A20).fourier(cosmo, k, M, a)


def test_mead20():

    bpar_M20 = bfg.Profiles.Mead20.Params_TAGN_7p6_All

    for a in [0.1, 0.5, 1]:
        bfg.Profiles.Mead20.DarkMatter(**bpar_M20).real(cosmo, R, M, a)
        bfg.Profiles.Mead20.DarkMatter(**bpar_M20).projected(cosmo, R, M, a)
        bfg.Profiles.Mead20.DarkMatter(**bpar_M20).fourier(cosmo, k, M, a)

        bfg.Profiles.Mead20.DarkMatterOnly(**bpar_M20).real(cosmo, R, M, a)
        bfg.Profiles.Mead20.DarkMatterOnly(**bpar_M20).projected(cosmo, R, M, a)
        bfg.Profiles.Mead20.DarkMatterOnly(**bpar_M20).fourier(cosmo, k, M, a)


if __name__ == '__main__':
    test_schneider19()
    test_arico20()
    test_mead20()