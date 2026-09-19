"""Regression test for profile composition identities.

Test index:
    test_profile2profile: checks two-halo subtraction equivalence.
"""

import numpy as np
import pyccl as ccl
import BaryonForge as bfg

from defaults import bpar_S19, ccl_dict

FAST_SETTINGS = {
    "r_steps": 64,
    "n_per_decade_proj": 4,
    "cutoff": 20,
    "proj_cutoff": 20,
}


def test_profile2profile():
    """Subtracting the two-halo term matches an explicit zero term."""
    cosmo = ccl.Cosmology(**ccl_dict)
    cosmo.compute_growth()
    masses = np.array([1.0e13, 1.0e14])
    radii = np.array([0.05, 0.5, 5.0])
    parameters = {**bpar_S19, **FAST_SETTINGS}

    dark_matter_baryon = bfg.Profiles.Schneider19.DarkMatterBaryon(**parameters)
    two_halo = bfg.Profiles.Schneider19.TwoHalo(**parameters)
    subtract = dark_matter_baryon - two_halo
    explicit_zero = bfg.Profiles.Schneider19.DarkMatterBaryon(
        **parameters, twohalo=bfg.Profiles.misc.Zeros()
    )

    for method in ("real", "projected"):
        left = getattr(subtract, method)(cosmo, radii, masses, 0.8)
        right = getattr(explicit_zero, method)(cosmo, radii, masses, 0.8)
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=3e-3)
