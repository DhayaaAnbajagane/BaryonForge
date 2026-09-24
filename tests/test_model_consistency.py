"""Consistency checks for model-specific profile implementations.

Test index:
    test_schneider25_normalization_follows_mass_definition: checks S25 gas normalization for 500c.
    test_arico_bound_gas_truncates_at_its_own_radius: checks the A20 truncation mass definition.
    test_model_modules_support_star_imports: checks every model module's ``__all__``.
    test_wrapper_profiles_have_string_representations: checks repr of convenience classes.
    test_shocked_gas_preserves_input_shapes: checks ShockedGas output shapes.
    test_lss_classes_use_their_own_model_fractions: checks Mead/Arico LSS classes use their own fractions.
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


def test_arico_bound_gas_truncates_at_its_own_radius(cosmo):
    mass_def = ccl.halos.massdef.MassDef500c
    bound_gas = bfg.Profiles.Arico20.BoundGas(**_fast(bpar_A20), mass_def=mass_def, cdelta=5)
    radius = mass_def.get_radius(cosmo, 1.0e14, 0.8) / 0.8
    result = bound_gas.real(cosmo, np.array([0.9, 1.1, 1.3]) * radius, 1.0e14, 0.8)
    assert result[0] > 0
    np.testing.assert_array_equal(result[1:], 0)


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
