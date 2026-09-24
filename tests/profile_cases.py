"""Shared fast profile fixtures for input and composition tests."""

from collections.abc import Callable

import numpy as np

import BaryonForge as bfg

from defaults import bpar_A20, bpar_HyDif, bpar_S19, bpar_S25


MASSES = np.array([1.0e13, 1.0e14])
RADII = np.array([0.05, 0.5, 5.0])
SCALE_FACTOR = 0.8

FAST_SETTINGS = {
    "r_steps": 64,
    "n_per_decade_proj": 3,
    "cutoff": 20,
    "proj_cutoff": 20,
}


def _fast(parameters):
    return {**parameters, **FAST_SETTINGS}


def _s19(name):
    parameters = _fast(bpar_S19)
    module = bfg.Profiles.Schneider19
    if name == "dark_matter":
        return module.DarkMatter(**parameters)
    if name == "two_halo":
        return module.TwoHalo(**parameters)
    if name == "stars":
        return module.Stars(**parameters)
    if name == "gas":
        return module.Gas(**parameters)
    if name == "shocked_gas":
        return module.ShockedGas(0.1, 0.2, **parameters)
    if name == "collisionless_matter":
        return module.CollisionlessMatter(**parameters)
    if name == "satellite_stars":
        return module.SatelliteStars(**parameters)
    if name == "dark_matter_only":
        return module.DarkMatterOnly(twohalo=bfg.Profiles.misc.Zeros(), **parameters)
    if name == "dark_matter_baryon":
        return module.DarkMatterBaryon(twohalo=bfg.Profiles.misc.Zeros(), **parameters)
    raise KeyError(name)


def _s25(name):
    parameters = _fast(bpar_S25)
    module = bfg.Profiles.Schneider25
    if name == "dark_matter":
        return module.DarkMatter(**parameters)
    if name == "two_halo":
        return module.TwoHalo(**parameters)
    if name == "stars":
        return module.Stars(**parameters)
    if name == "hot_gas":
        return module.HotGas(**parameters)
    if name == "inner_gas":
        return module.InnerGas(**parameters)
    if name == "gas":
        return module.Gas(**parameters)
    if name == "collisionless_matter":
        return module.CollisionlessMatter(**parameters)
    if name == "satellite_stars":
        return module.SatelliteStars(**parameters)
    if name == "dark_matter_only":
        return module.DarkMatterOnly(twohalo=bfg.Profiles.misc.Zeros(), **parameters)
    if name == "dark_matter_baryon":
        return module.DarkMatterBaryon(twohalo=bfg.Profiles.misc.Zeros(), **parameters)
    raise KeyError(name)


def _a20(name):
    parameters = _fast(bpar_A20)
    module = bfg.Profiles.Arico20
    if name == "dark_matter":
        return module.DarkMatter(**parameters)
    if name == "two_halo":
        return module.TwoHalo(**parameters)
    if name == "stars":
        return module.Stars(**parameters)
    if name == "bound_gas_untruncated":
        return module.BoundGasUntruncated(**parameters)
    if name == "bound_gas":
        return module.BoundGas(**parameters)
    if name == "ejected_gas":
        return module.EjectedGas(**parameters)
    if name == "reaccreted_gas":
        return module.ReaccretedGas(**parameters)
    if name == "gas":
        return module.Gas(**parameters)
    if name == "modified_dark_matter":
        return module.ModifiedDarkMatter(**parameters)
    if name == "collisionless_matter":
        return module.CollisionlessMatter(**parameters)
    if name == "satellite_stars":
        return module.SatelliteStars(**parameters)
    if name == "dark_matter_only":
        return module.DarkMatterOnly(**parameters)
    if name == "dark_matter_baryon":
        return module.DarkMatterBaryon(**parameters)
    if name == "dark_matter_only_with_lss":
        return module.DarkMatterOnlywithLSS(
            twohalo=bfg.Profiles.misc.Zeros(), **parameters
        )
    if name == "dark_matter_baryon_with_lss":
        return module.DarkMatterBaryonwithLSS(
            twohalo=bfg.Profiles.misc.Zeros(), **parameters
        )
    if name == "pressure":
        return module.Pressure(**parameters)
    if name == "nonthermal_fraction":
        return module.NonThermalFrac(**parameters)
    if name == "thermal_pressure":
        return module.ThermalPressure(**parameters)
    if name == "temperature":
        return module.Temperature(**parameters)
    if name == "bound_gas_deprecated":
        return module.BoundGasDeprecated(**parameters)
    raise KeyError(name)


def _m20(name):
    parameters = _fast(bfg.Profiles.Mead20.Params_TAGN_7p6_All)
    module = bfg.Profiles.Mead20
    if name == "dark_matter":
        return module.DarkMatter(**parameters)
    if name == "two_halo":
        return module.TwoHalo(**parameters)
    if name == "central_stars":
        return module.CentralStars(**parameters)
    if name == "satellite_stars":
        return module.SatelliteStars(**parameters)
    if name == "stars":
        return module.Stars(**parameters)
    if name == "delta_stars":
        return module.DeltaStars(**parameters)
    if name == "bound_gas":
        return module.BoundGas(**parameters)
    if name == "ejected_gas":
        return module.EjectedGas(**parameters)
    if name == "gas":
        return module.Gas(**parameters)
    if name == "gas_add_diffuse":
        return module.GasAddDiffuse(**parameters)
    if name == "collisionless_matter":
        return module.CollisionlessMatter(**parameters)
    if name == "dark_matter_only":
        return module.DarkMatterOnly(**parameters)
    if name == "dark_matter_baryon":
        return module.DarkMatterBaryon(**parameters)
    if name == "dark_matter_baryon_add_diffuse":
        return module.DarkMatterBaryonAddDiffuse(**parameters)
    if name == "dark_matter_only_with_lss":
        return module.DarkMatterOnlywithLSS(
            twohalo=bfg.Profiles.misc.Zeros(), **parameters
        )
    if name == "dark_matter_baryon_with_lss":
        return module.DarkMatterBaryonwithLSS(
            twohalo=bfg.Profiles.misc.Zeros(), **parameters
        )
    if name == "temperature":
        return module.Temperature(**parameters)
    if name == "pressure":
        return module.Pressure(**parameters)
    if name == "pressure_add_diffuse":
        return module.PressureAddDiffuse(**parameters)
    raise KeyError(name)


def _hydif(name):
    dm_parameters = _fast(bpar_S19)
    dm = bfg.Profiles.Schneider19.DarkMatter(**dm_parameters)

    parameters = {**_fast(bpar_HyDif), "darkmatter": dm}
    module = bfg.Profiles.HyDif
    if name == "hydrostatic_gas":
        return module.HydrostaticGas(**parameters)
    if name == "diffuse_gas":
        return module.DiffuseGas(**parameters)
    if name == "gas":
        return module.Gas(**parameters)
    raise KeyError(name)


def _thermodynamic(name):
    parameters = _fast({**bpar_S19, "mean_molecular_weight": 0.59})
    gas = bfg.Profiles.Schneider19.Gas(**parameters)
    dmb = bfg.Profiles.Schneider19.DarkMatterBaryon(
        twohalo=bfg.Profiles.misc.Zeros(), **parameters
    )
    pressure = bfg.Profiles.Thermodynamic.Pressure(
        gas=gas, darkmatterbaryon=dmb, **parameters
    )
    density = bfg.Profiles.Thermodynamic.GasNumberDensity(
        gas=gas, **parameters
    )
    temperature = bfg.Profiles.Thermodynamic.Temperature(
        thermalpressure=pressure,
        gasnumberdensity=density,
        **parameters,
    )
    if name == "pressure":
        return pressure
    if name == "electron_pressure":
        return bfg.Profiles.Thermodynamic.ElectronPressure(
            gas=gas, darkmatterbaryon=dmb, **parameters
        )
    if name == "nonthermal_fraction":
        return bfg.Profiles.Thermodynamic.NonThermalFrac(
            alpha_nt=0.1, nu_nt=0.5, gamma_nt=0.8, **parameters
        )
    if name == "green20_nonthermal_fraction":
        return bfg.Profiles.Thermodynamic.NonThermalFracGreen20(**parameters)
    if name == "gas_number_density":
        return density
    if name == "temperature":
        return temperature
    if name == "thermal_sz":
        return bfg.Profiles.Thermodynamic.ThermalSZ(
            thermalpressure=pressure, **parameters
        )
    if name == "metallicity":
        return bfg.Profiles.Thermodynamic.Metallicity(
            Z_out=0.01,
            Z_core=0.1,
            gamma_Z_core=1,
            theta_Z_core=0.1,
            **parameters,
        )
    if name == "emissivity":
        metallicity = _thermodynamic("metallicity")
        return bfg.Profiles.Thermodynamic.Emissivity(
            lambda temperature, metallicity, a: metallicity,
            temperature=temperature,
            metallicity=metallicity,
            **parameters,
        )
    if name == "xray_counts":
        emissivity = _thermodynamic("emissivity")
        return bfg.Profiles.Thermodynamic.XrayCounts(
            emissivity=emissivity,
            electronnumberdensity=density,
            hydrogennumberdensity=density,
            **parameters,
        )
    if name == "xray_sky_counts":
        return bfg.Profiles.Thermodynamic.XraySkyCounts(
            xraycounts=_thermodynamic("xray_counts"), **parameters
        )
    raise KeyError(name)


PROFILE_CASES: tuple[tuple[str, Callable[[], object]], ...] = tuple(
    (f"schneider19_{name}", lambda name=name: _s19(name))
    for name in (
        "dark_matter", "two_halo", "stars", "gas", "shocked_gas",
        "collisionless_matter", "satellite_stars", "dark_matter_only",
        "dark_matter_baryon",
    )
) + tuple(
    (f"schneider25_{name}", lambda name=name: _s25(name))
    for name in (
        "dark_matter", "two_halo", "stars", "hot_gas", "inner_gas", "gas",
        "collisionless_matter", "satellite_stars", "dark_matter_only",
        "dark_matter_baryon",
    )
) + tuple(
    (f"arico20_{name}", lambda name=name: _a20(name))
    for name in (
        "dark_matter", "two_halo", "stars", "bound_gas_untruncated",
        "bound_gas", "ejected_gas", "reaccreted_gas", "gas",
        "modified_dark_matter", "collisionless_matter", "satellite_stars",
        "dark_matter_only", "dark_matter_baryon", "dark_matter_only_with_lss",
        "dark_matter_baryon_with_lss", "pressure", "nonthermal_fraction",
        "thermal_pressure", "temperature", "bound_gas_deprecated",
    )
) + tuple(
    (f"mead20_{name}", lambda name=name: _m20(name))
    for name in (
        "dark_matter", "two_halo", "central_stars", "satellite_stars", "stars",
        "delta_stars", "bound_gas", "ejected_gas", "gas", "gas_add_diffuse",
        "collisionless_matter", "dark_matter_only", "dark_matter_baryon",
        "dark_matter_baryon_add_diffuse", "dark_matter_only_with_lss",
        "dark_matter_baryon_with_lss", "temperature", "pressure",
        "pressure_add_diffuse",
    )
) + tuple(
    (f"hydif_{name}", lambda name=name: _hydif(name))
    for name in ("hydrostatic_gas", "diffuse_gas", "gas")
) + tuple(
    (f"thermodynamic_{name}", lambda name=name: _thermodynamic(name))
    for name in (
        "pressure", "electron_pressure", "nonthermal_fraction",
        "green20_nonthermal_fraction", "gas_number_density", "temperature",
        "thermal_sz", "metallicity", "emissivity", "xray_counts",
        "xray_sky_counts",
    )
)
