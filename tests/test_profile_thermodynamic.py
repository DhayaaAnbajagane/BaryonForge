"""Fast construction tests for model-specific thermodynamic profiles.

The Schneider models use the generic thermodynamic pressure and temperature
classes.  Arico and Mead provide their own implementations.  Every fixture
uses explicit gas components and a zero two-halo term where a DMB profile is
needed; no full tabulated profile is generated here.

Test index:
    test_pressure_temperature_and_dmb_construct: checks pressure/temperature wiring.
    test_temperature_rejects_pressure_keyword: checks Temperature and ThermalSZ only take ``thermalpressure``.
"""

import pytest

import BaryonForge as bfg

from BaryonForge.Profiles import Arico20 as A20
from BaryonForge.Profiles import HyDif
from BaryonForge.Profiles import Mead20 as M20
from BaryonForge.Profiles import Schneider19 as S19
from BaryonForge.Profiles import Schneider25 as S25
from BaryonForge.Profiles import Thermodynamic as thermo

from defaults import bpar_A20, bpar_HyDif, bpar_S19, bpar_S25


FAST_SETTINGS = {
    "r_steps": 32,
    "n_per_decade_proj": 3,
    "cutoff": 20,
    "proj_cutoff": 20,
}


def _fast(parameters):
    return {**parameters, **FAST_SETTINGS}


def _schneider19():
    parameters = _fast({**bpar_S19, "mean_molecular_weight": 0.59})
    gas = S19.Gas(**parameters)
    dmb = S19.DarkMatterBaryon(
        twohalo=bfg.Profiles.misc.Zeros(), **parameters
    )
    pressure = thermo.Pressure(
        gas=gas, darkmatterbaryon=dmb, **parameters
    )
    density = thermo.GasNumberDensity(gas=gas, **parameters)
    temperature = thermo.Temperature(
        thermalpressure=pressure,
        gasnumberdensity=density,
        **parameters,
    )
    return pressure, temperature, dmb


def _schneider25():
    parameters = _fast(bpar_S25)
    # HotGas excludes the cold inner-gas component from pressure.
    gas = S25.HotGas(**parameters)
    dmb = S25.DarkMatterBaryon(
        twohalo=bfg.Profiles.misc.Zeros(), **parameters
    )
    pressure = thermo.Pressure(
        gas=gas, darkmatterbaryon=dmb, **parameters
    )
    density = thermo.GasNumberDensity(gas=gas, **parameters)
    temperature = thermo.Temperature(
        thermalpressure=pressure,
        gasnumberdensity=density,
        **parameters,
    )
    return pressure, temperature, dmb


def _hydif():
    parameters = _fast({**bpar_S19, "mean_molecular_weight": 0.59})
    # HyDif's Gas plugs into Schneider19's own DarkMatterBaryon (Stars, CollisionlessMatter,
    # one-halo darkmatter unchanged), matching the pattern used in examples/22_Plot_Profiles_HyDif.ipynb.
    dm = S19.DarkMatter(**parameters)
    gas = HyDif.Gas(darkmatter=dm, **_fast(bpar_HyDif))
    dmb = S19.DarkMatterBaryon(
        gas=gas, twohalo=bfg.Profiles.misc.Zeros(), **parameters
    )
    pressure = thermo.Pressure(
        gas=gas, darkmatterbaryon=dmb, **parameters
    )
    density = thermo.GasNumberDensity(gas=gas, **parameters)
    temperature = thermo.Temperature(
        thermalpressure=pressure,
        gasnumberdensity=density,
        **parameters,
    )
    return pressure, temperature, dmb


def _arico20():
    parameters = _fast(bpar_A20)
    bound_gas = A20.BoundGasUntruncated(**parameters)
    gas = A20.Gas(**parameters)
    pressure = A20.Pressure(
        bound_gas_untruncated=bound_gas,
        gas=gas,
        **parameters,
    )
    temperature = A20.Temperature(pressure=pressure, gas=gas, **parameters)
    dmb = A20.DarkMatterBaryonwithLSS(
        twohalo=bfg.Profiles.misc.Zeros(), **parameters
    )
    return pressure, temperature, dmb


def _mead20():
    parameters = _fast(M20.Params_TAGN_7p6_All)
    bound_gas = M20.BoundGas(**parameters)
    ejected_gas = M20.EjectedGas(**parameters)
    temperature = M20.Temperature(**parameters)
    pressure = M20.Pressure(
        boundgas=bound_gas,
        ejectedgas=ejected_gas,
        temperature=temperature,
        **parameters,
    )
    dmb = M20.DarkMatterBaryonwithLSS(
        twohalo=bfg.Profiles.misc.Zeros(), **parameters
    )
    return pressure, temperature, dmb


THERMO_CASES = (
    ("schneider19", _schneider19),
    ("schneider25", _schneider25),
    ("hydif", _hydif),
    ("arico20", _arico20),
    ("mead20", _mead20),
)


@pytest.mark.parametrize(
    "name, factory", THERMO_CASES, ids=[name for name, _ in THERMO_CASES]
)
def test_pressure_temperature_and_dmb_construct(name, factory):
    """Construct thermodynamic profiles with a zero two-halo DMB."""
    pressure, temperature, dmb = factory()
    assert pressure is not None, name
    assert temperature is not None, name
    assert pressure.mass_def == temperature.mass_def
    assert isinstance(dmb.TwoHalo, bfg.Profiles.misc.Zeros)


def test_temperature_rejects_pressure_keyword():
    """``pressure=`` would otherwise be swallowed by **kwargs and silently replaced by a default."""
    pressure, temperature, _ = _schneider19()
    assert temperature.Pressure is pressure
    with pytest.raises(TypeError, match="thermalpressure"):
        thermo.Temperature(pressure=pressure, gasnumberdensity=temperature.GasNumberDensity)

    # Same for ThermalSZ, whose input is also the thermal (gas) pressure
    assert thermo.ThermalSZ(thermalpressure=pressure).Pressure is pressure
    with pytest.raises(TypeError, match="thermalpressure"):
        thermo.ThermalSZ(pressure=pressure)
