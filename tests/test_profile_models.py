"""Small smoke tests for the supported physical profile families.

Test index:
    test_profile_methods_return_finite_arrays: evaluates each model family.
"""

from collections.abc import Callable

import numpy as np
import pyccl as ccl
import pytest

import BaryonForge as bfg

from defaults import bpar_A20, bpar_HyDif, bpar_S19, bpar_S25, ccl_dict


MASS_GRID = np.array([1.0e13, 1.0e14])
RADIUS_GRID = np.array([0.05, 0.5, 5.0])
WAVENUMBER_GRID = np.array([0.02, 0.2, 2.0])
SCALE_FACTOR = 0.8

FAST_SETTINGS = {
    "r_steps": 64,
    "n_per_decade_proj": 4,
    "cutoff": 20,
    "proj_cutoff": 20,
}


def _fast(parameters):
    return {**parameters, **FAST_SETTINGS}


PROFILE_CASES: tuple[tuple[str, Callable[[], object]], ...] = (
    # Matter profiles
    (
        "schneider19_matter_baryon",
        lambda: bfg.Profiles.Schneider19.DarkMatterBaryon(**_fast(bpar_S19)),
    ),
    (
        "schneider19_matter_only",
        lambda: bfg.Profiles.Schneider19.DarkMatterOnly(**_fast(bpar_S19)),
    ),
    (
        "arico20_matter_baryon",
        lambda: bfg.Profiles.Arico20.DarkMatterBaryon(**_fast(bpar_A20)),
    ),
    (
        "arico20_matter_only",
        lambda: bfg.Profiles.Arico20.DarkMatterOnly(**_fast(bpar_A20)),
    ),
    (
        "mead20_matter_baryon",
        lambda: bfg.Profiles.Mead20.DarkMatterBaryon(
            **_fast(bfg.Profiles.Mead20.Params_TAGN_7p6_All)
        ),
    ),
    (
        "mead20_matter_only",
        lambda: bfg.Profiles.Mead20.DarkMatterOnly(
            **_fast(bfg.Profiles.Mead20.Params_TAGN_7p6_All)
        ),
    ),
    # Dark matter profiles
    (
        "schneider19_dark_matter",
        lambda: bfg.Profiles.Schneider19.DarkMatter(**_fast(bpar_S19)),
    ),
    (
        "schneider19_dark_matter_only",
        lambda: bfg.Profiles.Schneider19.DarkMatterOnly(**_fast(bpar_S19)),
    ),
    (
        "schneider25_dark_matter",
        lambda: bfg.Profiles.Schneider25.DarkMatter(**_fast(bpar_S25)),
    ),
    (
        "schneider25_dark_matter_only",
        lambda: bfg.Profiles.Schneider25.DarkMatterOnly(**_fast(bpar_S25)),
    ),
    (
        "arico20_dark_matter",
        lambda: bfg.Profiles.Arico20.DarkMatter(**_fast(bpar_A20)),
    ),
    (
        "arico20_dark_matter_only",
        lambda: bfg.Profiles.Arico20.DarkMatterOnly(**_fast(bpar_A20)),
    ),
    (
        "mead20_dark_matter",
        lambda: bfg.Profiles.Mead20.DarkMatter(
            **_fast(bfg.Profiles.Mead20.Params_TAGN_7p6_All)
        ),
    ),
    (
        "mead20_dark_matter_only",
        lambda: bfg.Profiles.Mead20.DarkMatterOnly(
            **_fast(bfg.Profiles.Mead20.Params_TAGN_7p6_All)
        ),
    ),
    # Gas profiles
    (
        "schneider19_gas",
        lambda: bfg.Profiles.Schneider19.Gas(**_fast(bpar_S19)),
    ),
    (
        "schneider25_gas",
        lambda: bfg.Profiles.Schneider25.Gas(**_fast(bpar_S25)),
    ),
    (
        "arico20_gas",
        lambda: bfg.Profiles.Arico20.Gas(**_fast(bpar_A20)),
    ),
    (
        "mead20_gas",
        lambda: bfg.Profiles.Mead20.Gas(
            **_fast(bfg.Profiles.Mead20.Params_TAGN_7p6_All)
        ),
    ),
    (
        "hydif_gas",
        lambda: bfg.Profiles.HyDif.Gas(
            DM=bfg.Profiles.Schneider19.DarkMatter(**_fast(bpar_S19)),
            **_fast(bpar_HyDif),
        ),
    ),
    (
        "hydif_hydrostatic_gas",
        lambda: bfg.Profiles.HyDif.HydrostaticGas(
            DM=bfg.Profiles.Schneider19.DarkMatter(**_fast(bpar_S19)),
            **_fast(bpar_HyDif),
        ),
    ),
    (
        "hydif_diffuse_gas",
        lambda: bfg.Profiles.HyDif.DiffuseGas(
            DM=bfg.Profiles.Schneider19.DarkMatter(**_fast(bpar_S19)),
            **_fast(bpar_HyDif),
        ),
    ),
    # Stellar profiles
    (
        "schneider19_stars",
        lambda: bfg.Profiles.Schneider19.Stars(**_fast(bpar_S19)),
    ),
    (
        "schneider25_stars",
        lambda: bfg.Profiles.Schneider25.Stars(**_fast(bpar_S25)),
    ),
    (
        "arico20_stars",
        lambda: bfg.Profiles.Arico20.Stars(**_fast(bpar_A20)),
    ),
    (
        "mead20_stars",
        lambda: bfg.Profiles.Mead20.Stars(
            **_fast(bfg.Profiles.Mead20.Params_TAGN_7p6_All)
        ),
    ),
    # Two-halo profiles
    (
        "schneider19_two_halo",
        lambda: bfg.Profiles.Schneider19.TwoHalo(**_fast(bpar_S19)),
    ),
    (
        "arico20_two_halo",
        lambda: bfg.Profiles.Arico20.TwoHalo(**_fast(bpar_A20)),
    ),
    (
        "mead20_two_halo",
        lambda: bfg.Profiles.Mead20.TwoHalo(
            **_fast(bfg.Profiles.Mead20.Params_TAGN_7p6_All)
        ),
    ),
    # Collisionless matter profiles
    (
        "schneider19_collisionless_matter",
        lambda: bfg.Profiles.Schneider19.CollisionlessMatter(**_fast(bpar_S19)),
    ),
    (
        "arico20_collisionless_matter",
        lambda: bfg.Profiles.Arico20.CollisionlessMatter(**_fast(bpar_A20)),
    ),
    (
        "mead20_collisionless_matter",
        lambda: bfg.Profiles.Mead20.CollisionlessMatter(
            **_fast(bfg.Profiles.Mead20.Params_TAGN_7p6_All)
        ),
    ),
)


@pytest.fixture(scope="module")
def cosmo():
    cosmology = ccl.Cosmology(**ccl_dict)
    cosmology.compute_growth()
    return cosmology


@pytest.mark.parametrize(
    "case", PROFILE_CASES, ids=[case[0] for case in PROFILE_CASES]
)
def test_profile_methods_return_finite_arrays(cosmo, case):
    """Check real, projected, and Fourier model smoke behavior."""
    _, factory = case
    profile = factory()

    results = (
        profile.real(cosmo, RADIUS_GRID, MASS_GRID, SCALE_FACTOR),
        profile.projected(cosmo, RADIUS_GRID, MASS_GRID, SCALE_FACTOR),
        profile.fourier(cosmo, WAVENUMBER_GRID, MASS_GRID, SCALE_FACTOR),
    )

    for result in results:
        assert np.asarray(result).shape == (2, 3)
        assert np.all(np.isfinite(result))
