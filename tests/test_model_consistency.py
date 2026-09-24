"""Consistency checks for model-specific profile implementations.

Test index:
    test_schneider25_normalization_follows_mass_definition: checks S25 gas normalization for 500c.
    test_schneider25_follows_the_validated_model: checks S25 truncation, relaxation and inner-gas mass.
    test_arico_bound_gas_truncates_at_its_own_radius: checks the A20 truncation mass definition.
    test_arico_modified_dark_matter_matches_at_boundary: checks the A20 DM/bound-gas matching at R.
    test_battaglia_warns_on_inconsistent_mass_definition: checks the Battaglia mass_def warning.
    test_model_modules_support_star_imports: checks every model module's ``__all__``.
    test_wrapper_profiles_have_string_representations: checks repr of convenience classes.
    test_shocked_gas_preserves_input_shapes: checks ShockedGas output shapes.
    test_composite_profiles_preserve_input_shapes: checks scalar/array shapes of composite profiles.
    test_mead_diffuse_fourier_profiles_preserve_input_shapes: checks scalar/array k and projection of the Mead diffuse terms.
    test_lss_classes_use_their_own_model_fractions: checks Mead/Arico LSS classes use their own fractions.
"""

import warnings

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


@pytest.fixture(scope="module")
def cosmo():
    cosmology = ccl.Cosmology(**ccl_dict)
    cosmology.compute_growth()
    return cosmology


def test_schneider25_normalization_follows_mass_definition(cosmo):
    """The hot gas mass must be f_hga times the total mass of the matching DarkMatter profile."""
    parameters = {
        **_fast(bpar_S25),
        "mass_def": ccl.halos.massdef.MassDef500c,
        "c_M_relation": bfg.utils.GenericConcentrationDuffy08,
    }
    masses, scale_factor = np.array([1.0e14]), 0.8
    radii = np.geomspace(1e-6, 10, 4000)

    hot_gas = bfg.Profiles.Schneider25.HotGas(**parameters)
    gas_mass = np.trapz(4 * np.pi * radii**2 * hot_gas.real(cosmo, radii, masses, scale_factor), radii)
    f_hga, _ = hot_gas._get_gas_frac(masses, scale_factor, cosmo)

    dark_matter = bfg.Profiles.Schneider25.DarkMatter(**{**parameters, "r_steps": 500})
    dark_matter.cutoff = 1e3
    total_mass = np.trapz(4 * np.pi * radii**2 * dark_matter.real(cosmo, radii, masses, scale_factor), radii)

    np.testing.assert_allclose(gas_mass, f_hga * total_mass, rtol=2e-2)


def test_schneider25_follows_the_validated_model(cosmo):
    """Pins features of the S25 model cross-checked against the reference code (example 15)."""
    model = bfg.Profiles.Schneider25.DarkMatter(**_fast(bpar_S25))
    masses = np.geomspace(1e12, 1e15, 4)

    # Truncation radius parameter shrinks with peak height: epsilon0 - epsilon1 * nu
    assert np.all(np.diff(model._get_dm_eps(masses, 0.8, cosmo)) < 0)

    # Relaxation amplitudes evolve linearly with redshift: q_i + nu_qi * z
    relaxed = bfg.Profiles.Schneider25.CollisionlessMatter(**_fast(bpar_S25))
    q0, q1, q2 = relaxed._get_Qis(masses, 0.5, cosmo)
    assert (q0, q1, q2) == pytest.approx((bpar_S25["q0"], bpar_S25["q1"] + bpar_S25["nu_q1"], bpar_S25["q2"]))

    # The inner gas contains exactly f_iga times the total mass over the integration range
    inner_gas = bfg.Profiles.Schneider25.InnerGas(**{**_fast(bpar_S25), "r_steps": 4000})
    radii = np.geomspace(inner_gas.r_min_int, inner_gas.r_max_int, 4000)
    gas_mass = np.trapz(4 * np.pi * radii**2 * inner_gas.real(cosmo, radii, masses, 0.8), radii)
    _, f_iga = inner_gas._get_gas_frac(masses, 0.8, cosmo)
    dark_matter = bfg.Profiles.Schneider25.DarkMatter(**_fast(bpar_S25))
    dark_matter.cutoff = 1e3
    total_mass = np.trapz(4 * np.pi * radii**2 * dark_matter.real(cosmo, radii, masses, 0.8), radii)
    np.testing.assert_allclose(gas_mass, f_iga * total_mass, rtol=2e-2)


def test_arico_bound_gas_truncates_at_its_own_radius(cosmo):
    mass_def = ccl.halos.massdef.MassDef500c
    bound_gas = bfg.Profiles.Arico20.BoundGas(**_fast(bpar_A20), mass_def=mass_def, cdelta=5)
    radius = mass_def.get_radius(cosmo, 1.0e14, 0.8) / 0.8
    result = bound_gas.real(cosmo, np.array([0.9, 1.1, 1.3]) * radius, 1.0e14, 0.8)
    assert result[0] > 0
    np.testing.assert_array_equal(result[1:], 0)


def test_arico_modified_dark_matter_matches_at_boundary(cosmo):
    """Just inside R, the modified DM density is the gravity-only minus bound-gas density."""
    modified = bfg.Profiles.Arico20.ModifiedDarkMatter(**_fast(bpar_A20))
    masses, scale_factor = np.array([1.0e13, 1.0e14]), 0.8
    radius = modified.mass_def.get_radius(cosmo, masses, scale_factor) / scale_factor * (1 - 1e-6)
    for mass, r in zip(masses, radius):
        gravity_only = modified.GravityOnly.real(cosmo, r, mass, scale_factor)
        bound_gas = modified.Gas.real(cosmo, r, mass, scale_factor)
        assert bound_gas > 0
        assert modified.real(cosmo, r, mass, scale_factor) == pytest.approx(gravity_only - bound_gas, rel=1e-6)


def test_battaglia_warns_on_inconsistent_mass_definition():
    with pytest.warns(UserWarning, match="500c"):
        bfg.Profiles.Battaglia.Pressure("500_AGN", mass_def=ccl.halos.massdef.MassDef200c)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bfg.Profiles.Battaglia.Pressure("500_AGN", mass_def=ccl.halos.massdef.MassDef500c)
        bfg.Profiles.Battaglia.Pressure("200_AGN")

    with pytest.raises(ValueError, match="Model_def"):
        bfg.Profiles.Battaglia.GasDensity("500_AGN")


@pytest.mark.parametrize(
    "module", ("Schneider19", "Schneider25", "Arico20", "Mead20", "HyDif",
               "Battaglia", "Thermodynamic", "BaryonCorrection", "misc")
)
def test_model_modules_support_star_imports(module):
    namespace = {}
    exec(f"from BaryonForge.Profiles.{module} import *", namespace)
    module_object = getattr(bfg.Profiles, module)
    assert all(name in namespace for name in module_object.__all__)


def test_wrapper_profiles_have_string_representations():
    mead = bfg.Profiles.Mead20.Params_TAGN_7p6_All
    profiles = (
        bfg.Profiles.Arico20.Gas(**_fast(bpar_A20)),
        bfg.Profiles.Arico20.DarkMatterBaryon(**_fast(bpar_A20)),
        bfg.Profiles.Schneider25.Gas(**_fast(bpar_S25)),
        bfg.Profiles.Mead20.Gas(**_fast(mead)),
        bfg.Profiles.Mead20.Stars(**_fast(mead)),
    )
    for profile in profiles:
        text = repr(profile)
        assert text.startswith(profile.__class__.__name__)
        assert len(profile.model_params) > 0


def test_shocked_gas_preserves_input_shapes(cosmo):
    shocked = bfg.Profiles.Schneider19.ShockedGas(0.1, 0.2, **_fast(bpar_S19))
    radii = np.array([0.05, 0.5, 5.0])
    assert shocked.real(cosmo, radii, np.array([1.0e14]), 0.8).shape == (1, 3)
    assert shocked.real(cosmo, radii, 1.0e14, 0.8).shape == (3,)
    assert shocked.real(cosmo, 0.5, np.array([1.0e13, 1.0e14]), 0.8).shape == (2,)
    assert np.ndim(shocked.real(cosmo, 0.5, 1.0e14, 0.8)) == 0


def _composite_profiles():
    mead = _fast(bfg.Profiles.Mead20.Params_TAGN_7p6_All)
    return {
        "S19 DarkMatterBaryon": bfg.Profiles.Schneider19.DarkMatterBaryon(**_fast(bpar_S19)),
        "S25 DarkMatterBaryon": bfg.Profiles.Schneider25.DarkMatterBaryon(**_fast(bpar_S25)),
        "S19 SatelliteStars": bfg.Profiles.Schneider19.SatelliteStars(**_fast(bpar_S19)),
        "S25 SatelliteStars": bfg.Profiles.Schneider25.SatelliteStars(**_fast(bpar_S25)),
        "A20 SatelliteStars": bfg.Profiles.Arico20.SatelliteStars(**_fast(bpar_A20)),
        "M20 Pressure": bfg.Profiles.Mead20.Pressure(**mead),
        "ThermalSZ": bfg.Profiles.ThermalSZ(thermalpressure=bfg.Profiles.Mead20.Pressure(**mead), **mead),
    }


@pytest.mark.parametrize("name", list(_composite_profiles().keys()))
def test_composite_profiles_preserve_input_shapes(cosmo, name):
    profile = _composite_profiles()[name]
    radii, masses = np.array([0.05, 0.5]), np.array([1.0e13, 1.0e14, 1.0e15])
    assert profile.real(cosmo, 0.5, masses, 0.8).shape == (3,)
    assert profile.real(cosmo, radii, 1.0e14, 0.8).shape == (2,)
    assert np.ndim(profile.real(cosmo, 0.5, 1.0e14, 0.8)) == 0
    np.testing.assert_allclose(profile.real(cosmo, 0.5, masses, 0.8), profile.real(cosmo, radii, masses, 0.8)[:, 1])


def test_mead_diffuse_fourier_profiles_preserve_input_shapes(cosmo):
    mead = _fast(bfg.Profiles.Mead20.Params_TAGN_7p6_All)
    masses, k = np.array([1.0e13, 1.0e14]), np.array([0.1, 1.0])
    for profile in (bfg.Profiles.Mead20.GasAddDiffuse(**mead), bfg.Profiles.Mead20.PressureAddDiffuse(**mead)):
        assert profile.fourier(cosmo, 1.0, masses, 0.8).shape == (2,)
        assert profile.fourier(cosmo, k, 1.0e14, 0.8).shape == (2,)
        assert np.ndim(profile.fourier(cosmo, 1.0, 1.0e14, 0.8)) == 0
        # The real-space profile (from FFTLog) must also be usable for projection
        assert np.all(np.isfinite(profile.projected(cosmo, np.array([0.1, 1.0]), masses, 0.8)))
        # FFTLog output depends slightly on the requested k range, hence the tolerance
        np.testing.assert_allclose(profile.fourier(cosmo, 1.0, masses, 0.8), profile.fourier(cosmo, k, masses, 0.8)[:, 1],
                                   rtol=1e-2)


def test_lss_classes_use_their_own_model_fractions(cosmo):
    mead = _fast(bfg.Profiles.Mead20.Params_TAGN_7p6_All)
    masses = np.array([1.0e13, 1.0e14])
    reference = bfg.Profiles.Mead20.CentralStars(**mead)
    for profile in (bfg.Profiles.Mead20.DarkMatterBaryon(**mead),
                    bfg.Profiles.Mead20.DarkMatterBaryonwithLSS(twohalo=bfg.Profiles.misc.Zeros(), **mead)):
        np.testing.assert_allclose(
            profile.get_f_star(masses, 0.8, cosmo), reference.get_f_star(masses, 0.8, cosmo)
        )
        assert "M_0" in profile.model_params

    arico = _fast(bpar_A20)
    dmo = bfg.Profiles.Arico20.DarkMatterOnlywithLSS(twohalo=bfg.Profiles.misc.Zeros(), **arico)
    assert isinstance(dmo.DarkMatter, bfg.Profiles.Arico20.DarkMatter)
    np.testing.assert_allclose(
        dmo.get_f_gas(masses, 0.8, cosmo),
        bfg.Profiles.Arico20.BoundGas(**arico).get_f_gas(masses, 0.8, cosmo),
    )
