"""Regression tests for the HyDif composite profile."""

import pickle

import numpy as np
import pyccl as ccl
import pytest

import BaryonForge as bfg

from BaryonForge.Profiles import Schneider19 as S19
from BaryonForge.Profiles import HyDif
from BaryonForge.Profiles import Thermodynamic as thermo
from defaults import bpar_HyDif, bpar_S19, ccl_dict


FAST_SETTINGS = {
    "r_steps": 64,
    "n_per_decade_proj": 3,
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


@pytest.fixture
def hydif_profile():
    darkmatter = bfg.Profiles.Schneider19.DarkMatter(**_fast(bpar_S19))
    gas = bfg.Profiles.HyDif.Gas(
        darkmatter=darkmatter,
        **_fast(bpar_HyDif),
    )
    return darkmatter, gas


def test_gas_is_a_regular_picklable_profile(cosmo, hydif_profile):
    darkmatter, gas = hydif_profile
    radii = np.array([0.05, 0.5, 5.0])
    masses = np.array([1.0e13, 1.0e14])

    before = gas.real(cosmo, radii, masses, 0.8)
    restored = pickle.loads(pickle.dumps(gas))
    after = restored.real(cosmo, radii, masses, 0.8)

    np.testing.assert_allclose(after, before)
    assert set(restored.model_params) == set(gas.model_params)
    assert restored.HydrostaticGas is not restored.DiffuseGas
    assert restored.HydrostaticGas.darkmatter is restored.DiffuseGas.darkmatter
    assert darkmatter.cutoff == FAST_SETTINGS["cutoff"]


def test_gas_matches_the_explicit_components(cosmo, hydif_profile):
    _, gas = hydif_profile
    radii = np.array([0.05, 0.5, 5.0])
    masses = np.array([1.0e13, 1.0e14])

    expected = (
        gas.HydrostaticGas.real(cosmo, radii, masses, 0.8)
        + gas.DiffuseGas.real(cosmo, radii, masses, 0.8)
    )
    np.testing.assert_allclose(gas.real(cosmo, radii, masses, 0.8), expected)


def test_mass_normalization_temporarily_updates_and_restores_cutoff(cosmo, hydif_profile):
    darkmatter, gas = hydif_profile
    old_cutoff = darkmatter.cutoff

    gas.real(cosmo, np.array([0.05, 0.5]), np.array([1.0e14]), 0.8)

    assert darkmatter.cutoff == old_cutoff


def test_two_halo_darkmatter_is_rejected():
    darkmatter = bfg.Profiles.Schneider19.DarkMatter(**_fast(bpar_S19))
    dmo = bfg.Profiles.Schneider19.DarkMatterOnly(
        darkmatter=darkmatter,
        twohalo=bfg.Profiles.misc.Zeros(),
        **_fast(bpar_S19),
    )

    with pytest.raises(ValueError, match="one-halo"):
        bfg.Profiles.HyDif.Gas(darkmatter=dmo, **_fast(bpar_HyDif))


def test_deprecated_dm_alias_remains_compatible():
    darkmatter = bfg.Profiles.Schneider19.DarkMatter(**_fast(bpar_S19))
    gas = bfg.Profiles.HyDif.Gas(DM=darkmatter, **_fast(bpar_HyDif))

    assert gas.darkmatter is darkmatter


def test_shared_hydif_helpers_are_class_methods():
    assert isinstance(HyDif.HyDifProfiles.__dict__["_gnfw"], classmethod)
    assert isinstance(HyDif.HyDifProfiles.__dict__["_get_beta_d"], classmethod)

    radii = np.array([0.1, 1.0, 10.0])
    shape = HyDif.HyDifProfiles._gnfw(radii, 0.5, 2.0, 1.2, 2.0, 4.0)
    expected_shape = ((1 + radii/0.5)**(-1.2)
                     * (1 + (radii/2.0)**2.0)**(-(4.0 - 1.2)/2.0))
    np.testing.assert_allclose(shape, expected_shape)

    masses = np.array([1.0e13, 1.0e14])
    expected_beta = 3*(masses/1.0e14)**0.5 / (1 + (masses/1.0e14)**0.5)
    np.testing.assert_allclose(
        HyDif.DiffuseGas._get_beta_d(masses, 1.0e14, 0.5),
        expected_beta,
    )


def test_hydif_exposes_the_standard_gas_fraction_api(cosmo, hydif_profile):
    darkmatter, gas = hydif_profile
    masses = np.array([1.0e13, 1.0e14])
    expected = darkmatter.get_f_gas(masses, 0.8, cosmo)

    for profile in (gas, gas.HydrostaticGas, gas.DiffuseGas):
        np.testing.assert_allclose(profile._get_gas_frac(masses, 0.8, cosmo), expected)
        np.testing.assert_allclose(profile.get_f_gas(masses, 0.8, cosmo), expected)

    expected_scalar = darkmatter.get_f_gas(1.0e14, 0.8, cosmo)
    np.testing.assert_allclose(gas.get_f_gas(1.0e14, 0.8, cosmo), expected_scalar)


def test_hydif_thermodynamic_profiles_evaluate(cosmo):
    parameters = _fast({**bpar_S19, "mean_molecular_weight": 0.59})
    darkmatter = S19.DarkMatter(**parameters)
    gas = bfg.Profiles.HyDif.Gas(
        darkmatter=darkmatter,
        **_fast(bpar_HyDif),
    )
    dmb = S19.DarkMatterBaryon(
        gas=gas,
        twohalo=bfg.Profiles.misc.Zeros(),
        **parameters,
    )
    pressure = thermo.Pressure(gas=gas, darkmatterbaryon=dmb, **parameters)
    density = thermo.GasNumberDensity(gas=gas, **parameters)
    temperature = thermo.Temperature(
        thermalpressure=pressure,
        gasnumberdensity=density,
        **parameters,
    )

    radii = np.array([0.05, 0.5, 5.0])
    masses = np.array([1.0e13, 1.0e14])
    for profile in (pressure, density, temperature):
        result = profile.real(cosmo, radii, masses, 0.8)
        assert result.shape == (2, 3)
        assert np.all(np.isfinite(result))
