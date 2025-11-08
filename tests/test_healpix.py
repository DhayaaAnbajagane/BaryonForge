import BaryonForge as bfg
import numpy as np, healpy as hp
import pyccl as ccl

from defaults import bpar_S19, bpar_A20, ccl_dict, h
cosmo = ccl.Cosmology(**ccl_dict)
cosmo.compute_growth()


def _sample_sky(n, degrees=True, seed=None):
    """
    Uniformly sample n points on the sphere, returning (ra, dec).
    Correctly accounts for the Jacobian dΩ = cos(dec) ddec dra by sampling sin(dec) ~ U[-1,1].
    """
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.0, 2*np.pi, size=n)                # RA ~ U[0, 2π)
    s = rng.uniform(-1.0, 1.0, size=n)                    # sin(dec) ~ U[-1, 1]
    dec = np.arcsin(s)                                    # dec = arcsin(s)

    ra = np.degrees(ra)
    dec = np.degrees(dec) - 90.0
    ra %= 360.0

    return ra, dec


def test_baryonification():

    N = 100
    ra, dec = _sample_sky(N, seed = 42)
    M = np.power(10, np.random.uniform(12, 15.5, N))
    z = np.random.uniform(0.4, 0.5, N)

    Cat  = bfg.HaloLightConeCatalog(ra, dec, M, z, cosmo)

    bpar_S19['proj_cutoff'] = 50/2
    
    DMB  = bfg.Profiles.Schneider19.DarkMatterBaryon(**bpar_S19)
    DMO  = bfg.Profiles.Schneider19.DarkMatterOnly(**bpar_S19)
    
    for nside in [256, 512, 1024]:
        
        PIX  = bfg.utils.HealPixel(NSIDE = nside)
        DMBP = bfg.utils.ConvolvedProfile(DMB, PIX)
        DMOP = bfg.utils.ConvolvedProfile(DMO, PIX)
        Disp = bfg.Baryonification2D(DMOP, DMBP, cosmo, epsilon_max = 20)
        Disp.setup_interpolator(z_min = 0.01, z_max = 1, N_samples_z = 10)

        Map  = np.random.uniform(0, 10, hp.nside2npix(nside))
        Shel = bfg.LightconeShell(Map, cosmo = bfg.utils.build_cosmodict(cosmo))
        Bmap = bfg.BaryonifyShell(Cat, Shel, epsilon_max = 10, model = Disp, verbose = True)

        Map  = np.random.uniform(-10, 10, hp.nside2npix(nside))
        Shel = bfg.LightconeShell(Map, cosmo = bfg.utils.build_cosmodict(cosmo))
        Bmap = bfg.BaryonifyShell(Cat, Shel, epsilon_max = 10, model = Disp, verbose = True)
            

if __name__ == '__main__':
    
    test_baryonification()