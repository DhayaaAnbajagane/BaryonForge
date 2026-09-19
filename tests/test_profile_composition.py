"""Regression tests for profile composition identities.

Test index:
    test_profile2profile: checks two-halo subtraction equivalence.
    test_arithmetic_composes_different_model_profiles: checks cross-model arithmetic.
"""

import numpy as np
import pyccl as ccl
import pytest
import BaryonForge as bfg

from defaults import bpar_A20, bpar_S19, bpar_S25, ccl_dict

FAST_SETTINGS = {
    "r_steps": 64,
    "n_per_decade_proj": 4,
    "cutoff": 20,
    "proj_cutoff": 20,
}


def _fast(parameters):
    return {**parameters, **FAST_SETTINGS}


def _profile_pairs():
    profiles = (
        (
            "schneider19",
            bfg.Profiles.Schneider19.DarkMatter(**_fast(bpar_S19)),
        ),
        (
            "schneider25",
            bfg.Profiles.Schneider25.DarkMatter(**_fast(bpar_S25)),
        ),
        (
            "arico20",
            bfg.Profiles.Arico20.DarkMatter(**_fast(bpar_A20)),
        ),
        (
            "mead20",
            bfg.Profiles.Mead20.DarkMatter(
                **_fast(bfg.Profiles.Mead20.Params_TAGN_7p6_All)
            ),
        ),
    )
    return tuple(
        (f"{left_name}_with_{right_name}", left, right)
        for index, (left_name, left) in enumerate(profiles)
        for right_name, right in profiles[index + 1:]
    )


PROFILE_PAIRS = _profile_pairs()


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


@pytest.mark.parametrize(
    "operation", ("add", "subtract", "multiply", "divide")
)
@pytest.mark.parametrize(
    "case", PROFILE_PAIRS, ids=[case[0] for case in PROFILE_PAIRS]
)
def test_arithmetic_composes_different_model_profiles(operation, case):
    """Check pointwise arithmetic between different model implementations."""
    _, left, right = case
    operations = {
        "add": (lambda a, b: a + b, np.add),
        "subtract": (lambda a, b: a - b, np.subtract),
        "multiply": (lambda a, b: a * b, np.multiply),
        "divide": (lambda a, b: a / b, np.divide),
    }
    compose, expected_operation = operations[operation]

    cosmology = ccl.Cosmology(**ccl_dict)
    cosmology.compute_growth()
    masses = np.array([1.0e14])
    radii = np.array([0.2, 1.0])
    result = compose(left, right).real(cosmology, radii, masses, 0.8)
    expected = expected_operation(
        left.real(cosmology, radii, masses, 0.8),
        right.real(cosmology, radii, masses, 0.8),
    )
    np.testing.assert_allclose(result, expected)
