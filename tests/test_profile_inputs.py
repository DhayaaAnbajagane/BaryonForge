"""Input smoke tests for physical and thermodynamic profile families.

Test index:
    test_profile_constructs_and_accepts_inputs: checks all model input shapes.
"""

import numpy as np
import pyccl as ccl
import pytest

from defaults import ccl_dict
from profile_cases import MASSES, PROFILE_CASES, RADII, SCALE_FACTOR


@pytest.fixture(scope="module")
def cosmo():
    cosmology = ccl.Cosmology(**ccl_dict)
    cosmology.compute_growth()
    return cosmology


@pytest.mark.parametrize(
    "case", PROFILE_CASES, ids=[case[0] for case in PROFILE_CASES]
)
def test_profile_constructs_and_accepts_inputs(cosmo, case):
    """Evaluate each profile on one small array input."""
    _, factory = case
    result = factory().real(cosmo, RADII, MASSES, SCALE_FACTOR)
    assert np.asarray(result).shape == (2, 3)
    assert np.all(np.isfinite(result))
