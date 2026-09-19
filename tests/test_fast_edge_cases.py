"""Small, deterministic tests for profile infrastructure and edge cases.

These tests deliberately use analytic profiles and tiny numerical grids.  The
physical profile smoke tests cover model construction; this module checks the
contracts implemented by the shared wrappers and utilities.

Test index:
    test_identity_and_zero_profiles_preserve_all_public_shapes: checks identity and zero shapes.
    test_projection_handles_scalar_radius_with_multiple_masses: checks scalar-radius projection shapes.
    test_truncation_is_zero_at_the_boundary: checks truncation boundary behavior.
    test_profile_arithmetic_matches_analytic_values: checks profile arithmetic results.
    test_comoving_conversion_and_mass_integration_are_shape_safe: checks conversion and mass integration.
    test_simple_array_cache_supports_arrays_and_lru_behavior: checks array cache eviction.
    test_simple_array_cache_caches_none_results: checks caching of None results.
    test_cached_profile_caches_only_selected_methods: checks selected-method caching.
    test_tiny_tabulation_has_expected_grids_and_boundary_readout: checks tiny table readout.
    test_parameterized_tiny_tabulation_interpolates_extra_parameters: checks parameterized tables.
    test_harmonic_convolution_rejects_zero_redshift: checks harmonic projection validation.
    test_regrid_pixels_preserves_weighted_total: checks HEALPix regridding conservation.
    test_default_runner_coordinate_and_rotation_helpers: checks runner geometry helpers.
    test_generic_concentration_remapping_is_finite_and_bounded: checks concentration remapping.
    test_serialization_preserves_tiny_tabulated_profile: checks table serialization.
    test_serialization_preserves_cached_and_convolved_profiles: checks wrapper serialization.
    test_parameter_helpers_reach_nested_profiles: checks nested parameter helpers.
    test_fft_parameter_merging_and_pchip_edge_cases: checks numerical utility boundaries.
"""

from collections import Counter
from types import SimpleNamespace
import pickle

import numpy as np
import pyccl as ccl
import pytest

import BaryonForge as bfg
from BaryonForge.Profiles.Base import BaseBFGProfiles
from BaryonForge.utils.Cache import CachedProfile, SimpleArrayCache
from BaryonForge.utils.Pixel import ConvolvedProfile, HealPixel
from BaryonForge.utils.Tabulate import _get_parameter
from BaryonForge.utils.misc import build_cosmodict, combine_fftpars, safe_Pchip_minimize
from BaryonForge.Runners.HealpixRunner import DefaultRunner, regrid_pixels_hpix

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


class ParameterProfile(BaseBFGProfiles):
    model_param_names = ["amplitude"]

    def __init__(self, amplitude=1, **kwargs):
        super().__init__(amplitude=amplitude, **kwargs)
        self._projected = self._real
        self._fourier = self._real

    def _real(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        m_use = np.atleast_1d(M)
        result = self.amplitude * (m_use[:, None] / 1.0e14) * r_use[None, :]
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


def test_simple_array_cache_caches_none_results():
    calls = Counter()

    def evaluate(value):
        calls["evaluate"] += 1
        return None

    cached = SimpleArrayCache()(evaluate)
    assert cached("input") is None
    assert cached("input") is None
    assert calls["evaluate"] == 1


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

    outside = table.real(cosmo, 2.0, 1.0e14, scale_factor)
    assert np.isnan(outside)


def test_parameterized_tiny_tabulation_interpolates_extra_parameters(cosmo):
    model = ParameterProfile()
    table = bfg.utils.ParamTabulatedProfile(model, cosmo)
    table.setup_interpolator(
        z_min=0.1,
        z_max=0.1,
        N_samples_z=1,
        M_min=1.0e13,
        M_max=1.0e14,
        N_samples_Mass=2,
        R_min=0.1,
        R_max=1.0,
        N_samples_R=2,
        other_params={"amplitude": np.array([1.0, 4.0])},
        verbose=False,
    )

    with pytest.raises(AssertionError, match="amplitude"):
        table.real(cosmo, 0.5, 1.0e14, 1 / 1.1)

    result = table.real(
        cosmo, 0.5, 1.0e14, 1 / 1.1, amplitude=2.5
    )
    np.testing.assert_allclose(result, 1.0)


def test_harmonic_convolution_rejects_zero_redshift(cosmo):
    convolved = ConvolvedProfile(LinearProfile(), HealPixel(NSIDE=8))
    with pytest.raises(AssertionError, match="a = 1"):
        convolved.projected(cosmo, 0.1, 1.0e14, 1.0)


def test_regrid_pixels_preserves_weighted_total():
    hmap = np.zeros(5)
    parent_values = np.array([2.0, 3.0])
    child_pixels = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    child_weights = np.array([[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]])

    result = regrid_pixels_hpix(
        hmap, parent_values, child_pixels, child_weights
    )
    np.testing.assert_allclose(result, [0.5, 0.8, 1.1, 1.4, 1.2])
    assert result.sum() == pytest.approx(parent_values.sum())


def test_default_runner_coordinate_and_rotation_helpers():
    runner = DefaultRunner(
        SimpleNamespace(cosmology={}),
        SimpleNamespace(),
        epsilon_max=1,
        model=None,
        verbose=False,
    )
    coordinates = runner.coord_array(np.array([[1, 2]]), np.array([[3, 4]]))
    np.testing.assert_array_equal(coordinates, [[1, 3], [2, 4]])

    rotation = runner.build_Rmat(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    np.testing.assert_allclose(
        rotation @ np.array([1.0, 0.0]), [0.0, 1.0], atol=1e-15
    )

    with pytest.raises(NotImplementedError, match="ellipticity"):
        DefaultRunner(
            SimpleNamespace(cosmology={}),
            SimpleNamespace(),
            epsilon_max=1,
            model=None,
            use_ellipticity=True,
            verbose=False,
        )


def test_generic_concentration_remapping_is_finite_and_bounded(cosmo):
    concentration = bfg.utils.GenericConcentrationDuffy08(
        mass_def=ccl.halos.massdef.MassDef200m
    )
    concentration.M_in_lo = 1.0e11
    concentration.M_in_hi = 1.0e15
    concentration.M_in_N = 12
    masses = np.array([1.0e12, 1.0e14])

    result = concentration(cosmo, masses, 0.8)
    assert result.shape == masses.shape
    assert np.all(np.isfinite(result))
    assert np.all(result > 0)

    with pytest.raises(AssertionError):
        concentration(cosmo, concentration.M_in_lo, 0.8)


def test_serialization_preserves_tiny_tabulated_profile(cosmo):
    model = LinearProfile()
    table = bfg.utils.TabulatedProfile(model, cosmo)
    table.setup_interpolator(
        z_min=0.1,
        z_max=0.1,
        N_samples_z=1,
        M_min=1.0e13,
        M_max=1.0e14,
        N_samples_Mass=2,
        R_min=0.1,
        R_max=1.0,
        N_samples_R=2,
        verbose=False,
    )
    restored = pickle.loads(pickle.dumps(table))
    expected = table.real(cosmo, 0.5, 1.0e14, 1 / 1.1)
    np.testing.assert_allclose(
        restored.real(cosmo, 0.5, 1.0e14, 1 / 1.1), expected
    )


def test_serialization_preserves_cached_and_convolved_profiles():
    arguments = (None, np.array([0.5]), np.array([1.0e14]), 0.8)

    cached = CachedProfile(LinearProfile(), methods=["real"])
    expected_cached = cached.real(*arguments)
    restored_cached = pickle.loads(pickle.dumps(cached))
    np.testing.assert_allclose(restored_cached.real(*arguments), expected_cached)

    convolved = ConvolvedProfile(LinearProfile(), bfg.utils.NoPix())
    restored_convolved = pickle.loads(pickle.dumps(convolved))
    assert restored_convolved.Pixel.__class__ is bfg.utils.NoPix
    assert restored_convolved.Profile.__class__ is LinearProfile


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
