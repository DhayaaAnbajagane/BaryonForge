"""Regression tests for profile composition identities.

Test index:
    test_profile2profile: checks two-halo subtraction equivalence.
    test_arithmetic_composes_different_model_profiles: checks cross-model arithmetic.
    test_all_input_profiles_compose_with_identity: checks arithmetic for all profiles.
    test_fftlog_fallback_profiles_compose: checks FFTLog composition fallback.
    test_all_input_profiles_compose_as_left_operand: checks arithmetic called on every profile.
    test_scaling_profiles_scales_projection: checks scaled projections for wrapped/custom profiles.
    test_set_parameter_reaches_composed_operands: checks parameter updates of composed profiles.
    test_parameter_tabulation_of_composed_profile: checks tabulating a composed profile over a parameter.
    test_arico_dark_matter_baryon_projection_includes_all_components: checks Arico DMB projection.
    test_composed_profiles_inherit_numerical_settings: checks mass_def and projection settings.
    test_comoving_to_physical_composes: checks arithmetic with ComovingToPhysical.
    test_composed_and_wrapper_profiles_pickle: checks serialization of composed profiles.
"""

import pickle

import numpy as np
import pyccl as ccl
import pytest
import BaryonForge as bfg

from defaults import bpar_A20, bpar_S19, bpar_S25, ccl_dict
from profile_cases import PROFILE_CASES

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


@pytest.fixture(scope="module")
def cosmo():
    cosmology = ccl.Cosmology(**ccl_dict)
    cosmology.compute_growth()
    return cosmology


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


@pytest.mark.parametrize(
    "case", PROFILE_CASES, ids=[case[0] for case in PROFILE_CASES]
)
def test_all_input_profiles_compose_with_identity(cosmo, case):
    """Check all input profiles against an analytic identity profile."""
    _, factory = case
    profile = factory()
    identity = bfg.Profiles.misc.Identity()
    masses = np.array([1.0e14])
    radii = np.array([0.2, 1.0])
    operations = {
        "add": (lambda a, b: a + b, np.add),
        "subtract": (lambda a, b: a - b, np.subtract),
        "multiply": (lambda a, b: a * b, np.multiply),
        "divide": (lambda a, b: a / b, np.divide),
    }

    profile._real = _deterministic_real
    profile_result = profile.real(cosmo, radii, masses, 0.8)
    identity_result = identity.real(cosmo, radii, masses, 0.8)
    for compose, expected_operation in operations.values():
        composed = compose(identity, profile)
        result = composed.real(cosmo, radii, masses, 0.8)
        expected = expected_operation(identity_result, profile_result)
        np.testing.assert_allclose(result, expected)


def _deterministic_real(cosmo, r, M, a):
    """Return cheap, shape-preserving values for composition tests."""
    radii = np.atleast_1d(r)
    masses = np.atleast_1d(M)
    result = masses[:, None] / 1.0e14 + radii[None, :] + a
    if np.ndim(r) == 0:
        result = np.squeeze(result, axis=-1)
    if np.ndim(M) == 0:
        result = np.squeeze(result, axis=0)
    return result


def test_fftlog_fallback_profiles_compose(cosmo, monkeypatch):
    """Check arithmetic uses a profile's FFTLog real-space fallback."""
    _, factory = next(
        case for case in PROFILE_CASES
        if case[0] == "mead20_pressure_add_diffuse"
    )
    profile = factory()
    monkeypatch.setattr(profile, "_fftlog_wrap", _deterministic_fftlog)

    masses = np.array([1.0e14])
    radii = np.array([0.2, 1.0])
    result = (bfg.Profiles.misc.Identity() + profile).real(
        cosmo, radii, masses, 0.8
    )
    expected = 1 + _deterministic_real(cosmo, radii, masses, 0.8)
    np.testing.assert_allclose(result, expected)


def _deterministic_fftlog(cosmo, r, M, a, fourier_out=False):
    """Provide the public real-space fallback used by CCL profiles."""
    assert not fourier_out
    return _deterministic_real(cosmo, r, M, a)


@pytest.mark.parametrize(
    "case", PROFILE_CASES, ids=[case[0] for case in PROFILE_CASES]
)
def test_all_input_profiles_compose_as_left_operand(cosmo, case):
    """Arithmetic must work when called on any profile, not just Identity."""
    _, factory = case
    profile = factory()
    profile._real = _deterministic_real
    identity = bfg.Profiles.misc.Identity()
    masses = np.array([1.0e14])
    radii = np.array([0.2, 1.0])
    expected = profile.real(cosmo, radii, masses, 0.8)

    np.testing.assert_allclose((profile * 2).real(cosmo, radii, masses, 0.8), 2 * expected)
    np.testing.assert_allclose((profile + identity).real(cosmo, radii, masses, 0.8), expected + 1)
    np.testing.assert_allclose((1 - profile).real(cosmo, radii, masses, 0.8), 1 - expected)
    np.testing.assert_allclose((-profile).real(cosmo, radii, masses, 0.8), -expected)


def _scaling_cases():
    parameters = _fast(bpar_S19)
    gas = bfg.Profiles.Schneider19.Gas(**parameters)
    return (
        ("schneider19_gas", lambda: bfg.Profiles.Schneider19.Gas(**parameters)),
        ("shocked_gas", lambda: bfg.Profiles.Schneider19.ShockedGas(0.1, 0.2, **parameters)),
        ("comoving_to_physical", lambda: bfg.Profiles.misc.ComovingToPhysical(gas, factor=-3)),
        ("convolved", lambda: bfg.utils.ConvolvedProfile(gas, bfg.utils.NoPix())),
        ("gas_number_density", lambda: bfg.Profiles.Thermodynamic.GasNumberDensity(
            gas=gas, mean_molecular_weight=0.59)),
        ("arico20_gas", lambda: bfg.Profiles.Arico20.Gas(**_fast(bpar_A20))),
        ("schneider25_gas", lambda: bfg.Profiles.Schneider25.Gas(**_fast(bpar_S25))),
    )


@pytest.mark.parametrize(
    "case", _scaling_cases(), ids=[case[0] for case in _scaling_cases()]
)
def test_scaling_profiles_scales_projection(cosmo, case):
    _, factory = case
    profile = factory()
    masses = np.array([1.0e14])
    radii = np.array([0.2, 1.0])
    for method in ("real", "projected"):
        expected = getattr(profile, method)(cosmo, radii, masses, 0.8)
        np.testing.assert_allclose(
            getattr(profile * 2, method)(cosmo, radii, masses, 0.8), 2 * expected
        )
        np.testing.assert_allclose(
            getattr(profile / 4, method)(cosmo, radii, masses, 0.8), expected / 4
        )


def test_set_parameter_reaches_composed_operands(cosmo):
    parameters = _fast(bpar_S19)
    combined = (bfg.Profiles.Schneider19.Gas(**parameters) +
                bfg.Profiles.Schneider19.Stars(**parameters))
    radii = np.array([0.2, 1.0])

    before = combined.real(cosmo, radii, 1.0e14, 0.8)
    combined.set_parameter("theta_ej", 8)
    after = combined.real(cosmo, radii, 1.0e14, 0.8)

    assert combined.Profile1.theta_ej == 8
    assert not np.allclose(before, after, rtol=1e-3, atol=0)
    np.testing.assert_allclose(
        after, bfg.Profiles.Schneider19.Gas(**{**parameters, "theta_ej": 8}).real(cosmo, radii, 1.0e14, 0.8) +
               bfg.Profiles.Schneider19.Stars(**parameters).real(cosmo, radii, 1.0e14, 0.8)
    )


def test_parameter_tabulation_of_composed_profile(cosmo):
    parameters = _fast(bpar_S19)
    fraction = bfg.Profiles.Thermodynamic.NonThermalFrac(
        alpha_nt=0.1, nu_nt=0.5, gamma_nt=0.8, **parameters
    )
    thermal = bfg.Profiles.misc.Identity() * (1 - fraction)
    table = bfg.utils.ParamTabulatedProfile(thermal, cosmo)
    table.setup_interpolator(
        z_min=0.25, z_max=0.25, N_samples_z=1,
        M_min=1.0e14, M_max=2.0e14, N_samples_Mass=2,
        R_min=0.1, R_max=1.0, N_samples_R=3,
        other_params={"alpha_nt": np.array([0.05, 0.2])}, verbose=False,
    )

    for index, alpha in enumerate((0.05, 0.2)):
        fraction.alpha_nt = alpha
        np.testing.assert_allclose(
            table.raw_input_3D[0, ..., index],
            1 - fraction.real(cosmo, np.geomspace(0.1, 1, 3), np.geomspace(1e14, 2e14, 2), 0.8),
        )


def test_arico_dark_matter_baryon_projection_includes_all_components(cosmo):
    parameters = {**_fast(bpar_A20), "n_per_decade_proj": 30}
    gas = bfg.Profiles.Arico20.Gas(**parameters)
    stars = bfg.Profiles.Arico20.Stars(**parameters)
    clm = bfg.Profiles.Arico20.CollisionlessMatter(**parameters)
    dmb = bfg.Profiles.Arico20.DarkMatterBaryon(
        gas=gas, stars=stars, collisionlessmatter=clm, **parameters
    )
    radii = np.array([0.1, 0.5, 2.0])
    expected = sum(p.projected(cosmo, radii, 1.0e14, 0.8) for p in (gas, stars, clm))
    np.testing.assert_allclose(
        dmb.projected(cosmo, radii, 1.0e14, 0.8), expected, rtol=0.05
    )


def test_composed_profiles_inherit_numerical_settings():
    mass_def = ccl.halos.massdef.MassDef500c
    arico_gas = bfg.Profiles.Arico20.Gas(**_fast(bpar_A20), mass_def=mass_def)
    assert arico_gas.mass_def.name == "500c"
    mead_gas = bfg.Profiles.Mead20.Gas(
        **_fast(bfg.Profiles.Mead20.Params_TAGN_7p6_All), mass_def=mass_def
    )
    assert mead_gas.mass_def.name == "500c"

    gas = bfg.Profiles.Schneider19.Gas(**{**_fast(bpar_S19), "padding_lo_proj": 0.01})
    stars = bfg.Profiles.Arico20.Stars(**{**_fast(bpar_A20), "n_per_decade_proj": 7})
    assert (gas * 2).padding_lo_proj == 0.01
    assert (gas * 2).proj_cutoff == gas.proj_cutoff
    assert (stars * 2).n_per_decade_proj == 7


def test_comoving_to_physical_composes():
    physical = bfg.Profiles.misc.ComovingToPhysical(
        bfg.Profiles.misc.Identity(), factor=-3
    )
    radii = np.array([0.1, 1.0])
    masses = np.array([1.0e13, 1.0e14])
    np.testing.assert_allclose((physical * 2).real(None, radii, masses, 0.5), 16)
    np.testing.assert_allclose((physical * 2).projected(None, radii, masses, 0.5), 8)
    np.testing.assert_allclose(
        (bfg.Profiles.misc.Identity() + physical).real(None, radii, masses, 0.5), 9
    )


def test_composed_and_wrapper_profiles_pickle(cosmo):
    mead = bfg.Profiles.Mead20.Params_TAGN_7p6_All
    profiles = (
        bfg.Profiles.Schneider19.Gas(**_fast(bpar_S19)) * 2,
        bfg.Profiles.Arico20.Gas(**_fast(bpar_A20)),
        bfg.Profiles.Schneider25.Gas(**_fast(bpar_S25)),
        bfg.Profiles.Mead20.Gas(**_fast(mead)),
        bfg.Profiles.Mead20.Stars(**_fast(mead)),
    )
    radii = np.array([0.2, 1.0])
    for profile in profiles:
        restored = pickle.loads(pickle.dumps(profile))
        np.testing.assert_allclose(
            restored.real(cosmo, radii, 1.0e14, 0.8),
            profile.real(cosmo, radii, 1.0e14, 0.8),
        )
        assert repr(restored) == repr(profile)
