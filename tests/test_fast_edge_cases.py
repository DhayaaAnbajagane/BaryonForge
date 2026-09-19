"""Small, deterministic tests for profile infrastructure and edge cases.

These tests deliberately use analytic profiles and tiny numerical grids.  The
physical profile smoke tests cover model construction; this module checks the
contracts implemented by the shared wrappers and utilities.
"""

from collections import Counter

import numpy as np
import pyccl as ccl
import pytest

import BaryonForge as bfg
from BaryonForge.Profiles.Base import BaseBFGProfiles
from BaryonForge.utils.Cache import CachedProfile, SimpleArrayCache
from BaryonForge.utils.Tabulate import _get_parameter
from BaryonForge.utils.misc import build_cosmodict, combine_fftpars, safe_Pchip_minimize

from defaults import ccl_dict


@pytest.fixture(scope="module")
def cosmo():
    return ccl.Cosmology(**ccl_dict)


class LinearProfile(BaseBFGProfiles):
    """Cheap profile with a known value for every input shape."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._projected = self._real
        self._fourier = self._real

    def _real(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        m_use = np.atleast_1d(M)
        result = m_use[:, None] / 1.0e14 + r_use[None, :] + a
        if np.ndim(r) == 0:
            result = np.squeeze(result, axis=-1)
        if np.ndim(M) == 0:
            result = np.squeeze(result, axis=0)
        return result


class ConstantProfile(BaseBFGProfiles):
    def _real(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        m_use = np.atleast_1d(M)
        result = np.ones((m_use.size, r_use.size))
        if np.ndim(r) == 0:
            result = np.squeeze(result, axis=-1)
        if np.ndim(M) == 0:
            result = np.squeeze(result, axis=0)
        return result


def test_identity_and_zero_profiles_preserve_all_public_shapes():
    radii = np.array([0.1, 1.0])
    masses = np.array([1.0e13, 1.0e14])

    for profile, expected_value in ((bfg.Profiles.misc.Identity(), 1),
                                    (bfg.Profiles.misc.Zeros(), 0)):
        for method in ("real", "projected", "fourier"):
            result = getattr(profile, method)(None, radii, masses, 0.8)
            assert result.shape == (2, 2)
            np.testing.assert_array_equal(result, expected_value)

        assert np.ndim(profile.real(None, radii[0], masses[0], 0.8)) == 0


def test_projection_handles_scalar_radius_with_multiple_masses(cosmo):
    profile = ConstantProfile(
        cutoff=2,
        proj_cutoff=2,
        n_per_decade_proj=8,
    )

    result = profile.projected(cosmo, 0.2, [1.0e13, 1.0e14], 0.8)
    assert result.shape == (2,)
    assert np.all(np.isfinite(result))


def test_truncation_is_zero_at_the_boundary(cosmo):
    profile = bfg.Profiles.misc.Truncation(
        epsilon_trunc=1,
        mass_def=ccl.halos.massdef.MassDef200c,
    )
    masses = np.array([1.0e13, 1.0e14])
    scale_factor = 0.8
    boundary = profile.mass_def.get_radius(cosmo, masses, scale_factor) / scale_factor

    below = np.array([
        profile.real(cosmo, radius * (1 - 1e-8), mass, scale_factor)
        for radius, mass in zip(boundary, masses)
    ])
    at_boundary = np.array([
        profile.real(cosmo, radius, mass, scale_factor)
        for radius, mass in zip(boundary, masses)
    ])
    above = np.array([
        profile.real(cosmo, radius * (1 + 1e-8), mass, scale_factor)
        for radius, mass in zip(boundary, masses)
    ])

    np.testing.assert_array_equal(below, np.ones(2))
    np.testing.assert_array_equal(at_boundary, np.zeros(2))
    np.testing.assert_array_equal(above, np.zeros(2))


def test_profile_arithmetic_matches_analytic_values():
    profile = LinearProfile()
    radii = np.array([0.2, 0.5])
    masses = np.array([1.0e13, 2.0e14])
    expected = profile.real(None, radii, masses, 0.75)

    checks = (
        (profile + 2, expected + 2),
        (2 + profile, 2 + expected),
        (profile - 2, expected - 2),
        (2 - profile, 2 - expected),
        (profile * 3, expected * 3),
        (profile / 2, expected / 2),
        (-profile, -expected),
        (abs(-profile), abs(-expected)),
    )
    for composed, expected_result in checks:
        np.testing.assert_allclose(
            composed.real(None, radii, masses, 0.75), expected_result
        )


def test_comoving_conversion_and_mass_integration_are_shape_safe():
    converted = bfg.Profiles.misc.ComovingToPhysical(
        bfg.Profiles.misc.Identity(), factor=-2
    )
    np.testing.assert_allclose(
        converted.real(None, [0.1, 1.0], [1.0e13, 1.0e14], 0.5), 4
    )
    np.testing.assert_allclose(
        converted.projected(None, [0.1, 1.0], [1.0e13, 1.0e14], 0.5), 2
    )

    r_min, r_max = 0.1, 2.0
    integrator = bfg.Profiles.misc.Mdelta_to_Mtot(
        ConstantProfile(), r_min=r_min, r_max=r_max, N_int=128
    )
    expected = 4 * np.pi / 3 * (r_max**3 - r_min**3)
    result = integrator(None, [1.0e13, 1.0e14], 0.8)
    np.testing.assert_allclose(result, expected, rtol=2e-3)


def test_simple_array_cache_supports_arrays_and_lru_behavior():
    calls = Counter()

    def evaluate(values):
        calls["evaluate"] += 1
        return np.asarray(values).sum()

    cached = SimpleArrayCache(maxsize=2)(evaluate)
    cached(np.array([1.0, 2.0]))
    cached([1, 2])
    assert calls["evaluate"] == 2  # the integer and float dtypes differ

    cached(np.array([1.0, 2.0]))
    assert calls["evaluate"] == 2

    cached((3.0,))
    cached(np.array([4.0]))
    cached([1, 2])
    assert calls["evaluate"] == 5


def test_cached_profile_caches_only_selected_methods():
    profile = LinearProfile()
    calls = Counter()
    original_real = profile.real

    def counted_real(*args):
        calls["real"] += 1
        return original_real(*args)

    profile.real = counted_real
    cached = CachedProfile(profile, methods=["real"])
    args = (None, np.array([0.2]), np.array([1.0e14]), 0.8)
    cached.real(*args)
    cached.real(*args)
    assert calls["real"] == 1
    assert cached.Profile is profile


def test_tiny_tabulation_has_expected_grids_and_boundary_readout(cosmo):
    model = LinearProfile()
    table = bfg.utils.TabulatedProfile(model, cosmo)

    with pytest.raises(NameError, match="No Table created"):
        table.real(cosmo, 0.1, 1.0e14, 0.8)

    table.setup_interpolator(
        z_min=0.1,
        z_max=1.0,
        N_samples_z=2,
        z_linear_sampling=True,
        M_min=1.0e13,
        M_max=1.0e14,
        N_samples_Mass=2,
        R_min=0.1,
        R_max=1.0,
        N_samples_R=4,
        verbose=False,
    )

    assert table.raw_input_3D.shape == (2, 2, 4)
    assert table.raw_input_2D.shape == (2, 2, 4)
    assert table.cosmo._pk_lin == {}
    assert table.cosmo._pk_nl == {}

    radii = np.array([0.1, 1.0])
    masses = np.array([1.0e13, 1.0e14])
    scale_factor = 1 / 1.1
    result = table.real(cosmo, radii, masses, scale_factor)
    np.testing.assert_allclose(result, model.real(cosmo, radii, masses, scale_factor))


def test_parameter_helpers_reach_nested_profiles(cosmo):
    inner = LinearProfile()
    outer = bfg.Profiles.misc.ComovingToPhysical(inner, factor=-3)
    outer.set_parameter("r_steps", 7)
    assert inner.r_steps == 7
    assert _get_parameter(outer, "r_steps") == 7

    cosmology_parameters = build_cosmodict(cosmo)
    assert set(cosmology_parameters) == {
        "Omega_m", "Omega_b", "sigma8", "h", "n_s", "w0", "wa"
    }
    assert cosmology_parameters["Omega_m"] == pytest.approx(ccl_dict["Omega_c"] + ccl_dict["Omega_b"])


def test_fft_parameter_merging_and_pchip_edge_cases():
    profile = LinearProfile()
    first = profile.precision_fftlog.to_dict()
    second = first.copy()
    second["n_per_decade"] = first["n_per_decade"] + 1
    second["plaw_fourier"] = first["plaw_fourier"] + 1

    with pytest.warns(UserWarning, match="plaw_fourier"):
        merged = combine_fftpars(first, second)
    assert merged["n_per_decade"] == second["n_per_decade"]
    with pytest.warns(UserWarning, match="plaw_fourier"):
        assert combine_fftpars(first, second)["plaw_fourier"] == first["plaw_fourier"]

    x = np.array([-1.0, 0.0, 1.0])
    y = x**2
    assert safe_Pchip_minimize(x, y) == pytest.approx(0)
    with pytest.warns(UserWarning, match="Cannot minimize"):
        assert safe_Pchip_minimize(np.array([1.0, 2.0]), y[:2]) == np.inf
