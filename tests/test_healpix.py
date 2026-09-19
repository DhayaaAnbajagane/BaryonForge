"""Fast HEALPix runner tests.

Test index:
    test_baryonification_returns_zero_map_unchanged: checks zero-map shortcut.
"""

import numpy as np
import healpy as hp
import pyccl as ccl

import BaryonForge as bfg


def test_baryonification_returns_zero_map_unchanged():
    """A zero input map should bypass profile calculations."""
    cosmology = ccl.Cosmology(
        Omega_c=0.26,
        Omega_b=0.04,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        matter_power_spectrum="linear",
    )
    cosmology_parameters = bfg.utils.build_cosmodict(cosmology)
    catalog = bfg.HaloLightConeCatalog(
        [0.0], [0.0], [1.0e14], [0.5], cosmology_parameters.copy()
    )
    shell = bfg.LightconeShell(
        np.zeros(hp.nside2npix(1)),
        cosmo=cosmology_parameters.copy(),
    )

    runner = bfg.BaryonifyShell(
        catalog, shell, epsilon_max=10, model=None, verbose=False
    )
    result = runner.process()

    np.testing.assert_array_equal(result, shell.map)
