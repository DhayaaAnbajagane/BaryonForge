"""Input-shape and profile-composition regression tests.

These tests intentionally use small input grids.  They are smoke tests for the
public profile API rather than numerical-accuracy tests, and are therefore
suited to running for every supported Python version in GitHub Actions.
"""

from collections.abc import Callable

import numpy as np
import pyccl as ccl
import pytest

import BaryonForge as bfg

from defaults import bpar_A20, bpar_S19, bpar_S25, ccl_dict


MASSES = (1.0e13, 1.0e14)
RADII = (0.05, 0.5)
SCALE_FACTORS = (0.5, np.float64(0.8), 1.0)

# Keep numerical integrations small enough for a smoke-test matrix while still
# spanning several radial decades.
FAST_SETTINGS = {
    "r_steps": 64,
    "n_per_decade_proj": 4,
    "cutoff": 20,
    "proj_cutoff": 20,
}


def _with_fast_settings(parameters):
    """Copy profile parameters and override only numerical test settings."""
    return {**parameters, **FAST_SETTINGS}


@pytest.fixture(scope="module")
def cosmo():
    cosmology = ccl.Cosmology(**ccl_dict)
    cosmology.compute_growth()
    return cosmology


PROFILE_FACTORIES: tuple[tuple[str, Callable[[], object]], ...] = (
    (
        "schneider19",
        lambda: bfg.Profiles.Schneider19.DarkMatter(
            **_with_fast_settings(bpar_S19)
        ),
    ),
    (
        "schneider25",
        lambda: bfg.Profiles.Schneider25.DarkMatter(
            **_with_fast_settings(bpar_S25)
        ),
    ),
    (
        "arico20",
        lambda: bfg.Profiles.Arico20.DarkMatter(
            **_with_fast_settings(bpar_A20)
        ),
    ),
    (
        "mead20",
        lambda: bfg.Profiles.Mead20.DarkMatter(
            **_with_fast_settings(bfg.Profiles.Mead20.Params_TAGN_7p6_All)
        ),
    ),
)


@pytest.fixture(params=PROFILE_FACTORIES, ids=lambda item: item[0])
def profile(request):
    return request.param[1]()


def _assert_valid_result(result, expected_shape):
    result = np.asarray(result)
    assert result.shape == expected_shape
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize(
    ("mass", "expected_shape"),
    (
        pytest.param(MASSES[0], (len(RADII),), id="scalar"),
        pytest.param(list(MASSES), (len(MASSES), len(RADII)), id="list"),
        pytest.param(np.asarray(MASSES), (len(MASSES), len(RADII)), id="array"),
    ),
)
def test_real_accepts_mass_input_types(cosmo, profile, mass, expected_shape):
    result = profile.real(cosmo, np.asarray(RADII), mass, 0.8)
    _assert_valid_result(result, expected_shape)


@pytest.mark.parametrize(
    ("radius", "expected_shape"),
    (
        pytest.param(RADII[0], (len(MASSES),), id="scalar"),
        pytest.param(list(RADII), (len(MASSES), len(RADII)), id="list"),
        pytest.param(np.asarray(RADII), (len(MASSES), len(RADII)), id="array"),
    ),
)
def test_real_accepts_radius_input_types(cosmo, profile, radius, expected_shape):
    result = profile.real(cosmo, radius, np.asarray(MASSES), 0.8)
    _assert_valid_result(result, expected_shape)


@pytest.mark.parametrize("scale_factor", SCALE_FACTORS)
def test_real_accepts_scale_factor_values(cosmo, profile, scale_factor):
    result = profile.real(
        cosmo, np.asarray(RADII), np.asarray(MASSES), scale_factor
    )
    _assert_valid_result(result, (len(MASSES), len(RADII)))


@pytest.mark.parametrize("method", ("real", "projected", "fourier"))
def test_profile_methods_accept_list_inputs(cosmo, profile, method):
    result = getattr(profile, method)(cosmo, list(RADII), list(MASSES), 0.8)
    _assert_valid_result(result, (len(MASSES), len(RADII)))


def _composition_profiles():
    dark_matter = bfg.Profiles.Schneider19.DarkMatter(
        **_with_fast_settings(bpar_S19)
    )
    truncation = bfg.Profiles.misc.Truncation(
        epsilon_trunc=2, **FAST_SETTINGS
    )
    return dark_matter, truncation


@pytest.mark.parametrize(
    ("operation", "expected"),
    (
        pytest.param(lambda left, right: left + right, np.add, id="add"),
        pytest.param(lambda left, right: left - right, np.subtract, id="subtract"),
        pytest.param(lambda left, right: left * right, np.multiply, id="multiply"),
    ),
)
def test_profile_composition_is_pointwise_in_real_space(
    cosmo, operation, expected
):
    left, right = _composition_profiles()
    composed = operation(left, right)

    left_result = left.real(cosmo, list(RADII), list(MASSES), 0.8)
    right_result = right.real(cosmo, list(RADII), list(MASSES), 0.8)
    result = composed.real(cosmo, list(RADII), list(MASSES), 0.8)

    np.testing.assert_allclose(result, expected(left_result, right_result))


@pytest.mark.parametrize("method", ("real", "projected", "fourier"))
@pytest.mark.parametrize(
    ("mass", "radius", "expected_shape"),
    (
        pytest.param(MASSES[0], RADII[0], (), id="scalars"),
        pytest.param(list(MASSES), list(RADII), (2, 2), id="lists"),
        pytest.param(
            np.asarray(MASSES), np.asarray(RADII), (2, 2), id="arrays"
        ),
    ),
)
def test_nested_composition_runs_for_all_profile_methods(
    cosmo, method, mass, radius, expected_shape
):
    dark_matter, truncation = _composition_profiles()
    composed = (dark_matter * truncation + dark_matter) / 2

    result = getattr(composed, method)(cosmo, radius, mass, 0.8)
    _assert_valid_result(result, expected_shape)
