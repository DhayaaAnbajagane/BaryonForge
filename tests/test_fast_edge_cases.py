"""Small, deterministic tests for profile infrastructure and edge cases.

These tests deliberately use analytic profiles and tiny numerical grids.  The
physical profile smoke tests cover model construction; this module checks the
contracts implemented by the shared wrappers and utilities.

Test index:
    test_identity_and_zero_profiles_preserve_all_public_shapes: checks identity and zero shapes.
    test_projection_handles_scalar_radius_with_multiple_masses: checks scalar-radius projection shapes.
    test_truncation_is_zero_at_the_boundary: checks truncation boundary behavior and mask arithmetic.
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
    test_get_parameter_searches_all_nested_profiles: checks nested lookup past unrelated profiles.
    test_get_parameter_prefers_own_attributes: checks wrappers inherit the outer profile's cutoff.
    test_self_referencing_profiles_do_not_recurse_forever: checks tSZ wrappers of non-S19 pressures.
    test_wrapping_preserves_inner_fft_precision: checks wrappers keep custom FFTLog settings.
    test_comoving_to_physical_fourier_scaling: checks the scale-factor power of the Fourier profile.
    test_integer_masses_match_float_masses: checks integer mass inputs.
    test_cosmodict_computes_sigma8_from_As: checks sigma8 for A_s cosmologies.
    test_tabulated_correlation_function_evaluates: checks the xi_mm tabulator.
    test_cached_profile_forwards_uncached_methods: checks uncached methods use the input profile.
    test_projection_does_not_depend_on_requested_radii: checks real-space projection convergence.
    test_emissivity_table_accepts_documented_inputs: checks point lists, ordering, and 2D queries.
    test_grid_pixel_window_matches_pixel_average: checks the grid pixel window against direct averaging.
    test_runners_default_to_the_model_mass_definition: checks the default mass_def of runners/baryonification.
    test_parameter_tables_match_parameters_by_name: checks tabulated parameters are matched by name and restored.
    test_displacement_table_matches_parameters_by_name: checks displacement parameters are matched by name and restored.
    test_simple_array_cache_returns_copies: checks cached outputs cannot be modified in place.
    test_truncated_fourier_matches_direct_transform: checks the Fourier transform of truncated profiles.
    test_projection_at_zero_radius: checks projected profiles at r = 0.
    test_projection_resolves_the_cutoff_edge: checks background terms projected up to the 3D cutoff.
    test_legacy_projection_is_retained: checks the original real-space projection is still available.
    test_concentration_classes_and_instances: checks c_M_relation accepts classes and instances.
    test_cached_profiles_in_halo_model: checks cached (HOD) profiles inside CCL halo-model calls.
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

from defaults import bpar_A20, bpar_S19, ccl_dict


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

    # Masks must support arithmetic, eg. an annulus between two truncation radii
    outer = bfg.Profiles.misc.Truncation(epsilon_trunc=2, mass_def=ccl.halos.massdef.MassDef200c)
    radii = np.array([0.5, 1.5, 2.5]) * boundary[1]
    np.testing.assert_array_equal((outer - profile).real(cosmo, radii, masses[1], scale_factor), [0, 1, 0])
    np.testing.assert_array_equal((-profile).real(cosmo, radii, masses[1], scale_factor), [-1, 0, 0])


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
    # Clockwise rotations too, and the inputs must not be modified in place
    A, ref = np.array([0.0, 2.0]), np.array([3.0, 0.0])
    rotation = runner.build_Rmat(A, ref)
    np.testing.assert_allclose(rotation @ np.array([0.0, 1.0]), [1.0, 0.0], atol=1e-15)
    np.testing.assert_array_equal(A, [0.0, 2.0])

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


class _Container(BaseBFGProfiles):
    """Profile holding two sub-profiles, where only the second has ``amplitude``."""

    def __init__(self, first, second, **kwargs):
        self.Alpha = first
        self.Beta = second
        super().__init__(**kwargs)

    def _real(self, cosmo, r, M, a):
        return self.Beta.real(cosmo, r, M, a)


def test_get_parameter_searches_all_nested_profiles():
    container = _Container(LinearProfile(), ParameterProfile(amplitude=3))
    assert _get_parameter(container, "amplitude") == 3
    assert _get_parameter(container, "not_a_parameter") is None


def test_get_parameter_prefers_own_attributes():
    # A sub-profile that sorts first alphabetically must not override the outer value
    container = _Container(ParameterProfile(amplitude=3), LinearProfile(), cutoff=20)
    container.Alpha.cutoff = 1000
    assert _get_parameter(container, "cutoff") == 20

    parameters = {**bpar_S19, "r_steps": 64, "cutoff": 20, "proj_cutoff": 20}
    dark_matter_baryon = bfg.Profiles.Schneider19.DarkMatterBaryon(**parameters)
    two_halo = bfg.Profiles.Schneider19.TwoHalo(**parameters)
    assert (dark_matter_baryon - two_halo).cutoff == 20
    assert ConvolvedProfile(dark_matter_baryon, HealPixel(64)).cutoff == 20


def test_self_referencing_profiles_do_not_recurse_forever(cosmo):
    # Pressure models without a `prof4params` attribute make ThermalSZ point
    # `prof4params` at itself. Wrapping and setting parameters must still work.
    pressure = bfg.Profiles.Battaglia.Pressure("200_AGN")
    tsz = bfg.Profiles.ThermalSZ(pressure, cutoff=20, proj_cutoff=20)
    assert tsz.prof4params is tsz

    compton_y = bfg.Profiles.misc.ComovingToPhysical(tsz, factor=-3)
    compton_y.set_parameter("r_steps", 7)
    result = compton_y.projected(cosmo, np.array([0.1, 1.0]), 1.0e14, 0.8)
    assert tsz.r_steps == 7
    assert np.all(np.isfinite(result)) and np.all(result > 0)


def test_wrapping_preserves_inner_fft_precision():
    parameters = {**bpar_S19, "r_steps": 64, "cutoff": 20, "proj_cutoff": 20}
    stars = bfg.Profiles.Schneider19.Stars(**parameters)
    assert stars.precision_fftlog["padding_lo_fftlog"] == 1e-5

    bfg.Profiles.misc.ComovingToPhysical(stars, factor=-3)
    ConvolvedProfile(stars, bfg.utils.NoPix())
    CachedProfile(stars)
    assert stars.precision_fftlog["padding_lo_fftlog"] == 1e-5

    dmb = bfg.Profiles.Schneider19.DarkMatterBaryon(**parameters)
    assert dmb.Stars.precision_fftlog["padding_lo_fftlog"] == 1e-5


def test_comoving_to_physical_fourier_scaling():
    inner = LinearProfile()
    converted = bfg.Profiles.misc.ComovingToPhysical(inner, factor=-2)
    k = np.array([0.1, 1.0])
    masses = np.array([1.0e13, 1.0e14])
    np.testing.assert_allclose(
        converted.fourier(None, k, masses, 0.5),
        inner.fourier(None, k, masses, 0.5) * 0.5,
    )


def test_integer_masses_match_float_masses(cosmo):
    parameters = {**bpar_S19, "r_steps": 64, "cutoff": 20, "proj_cutoff": 20}
    radii = np.array([0.1, 1.0])
    for profile in (bfg.Profiles.Schneider19.DarkMatter(**parameters),
                    bfg.Profiles.Arico20.BoundGas(**{**bpar_A20, "r_steps": 64})):
        np.testing.assert_allclose(
            profile.real(cosmo, radii, 10**14, 0.8),
            profile.real(cosmo, radii, 1.0e14, 0.8),
        )
        np.testing.assert_allclose(
            profile.real(cosmo, radii, np.array([10**13, 10**14]), 0.8),
            profile.real(cosmo, radii, np.array([1.0e13, 1.0e14]), 0.8),
        )


def test_cosmodict_computes_sigma8_from_As():
    cosmology = ccl.Cosmology(
        Omega_c=0.26, Omega_b=0.04, h=0.7, A_s=2.1e-9, n_s=0.96
    )
    parameters = build_cosmodict(cosmology)
    assert set(parameters) == {
        "Omega_m", "Omega_b", "sigma8", "h", "n_s", "w0", "wa"
    }
    assert parameters["sigma8"] == pytest.approx(ccl.sigma8(cosmology))


def test_tabulated_correlation_function_evaluates(cosmo):
    from BaryonForge.utils.Tabulate import TabulatedCorrelation3D

    table = TabulatedCorrelation3D(cosmo, R_range=[1.0, 10.0], N_samples=16)
    table.setup_interpolator(z_min=0, z_max=0.5, N_samples_z=2)
    radii = np.array([2.0, 5.0])
    np.testing.assert_allclose(
        table(radii, 1.0), ccl.correlation_3d(cosmo, r=radii, a=1.0), rtol=1e-2
    )


def test_cached_profile_forwards_uncached_methods(cosmo):
    parameters = {**bpar_S19, "r_steps": 64, "cutoff": 20, "proj_cutoff": 20}
    gas = bfg.Profiles.Schneider19.Gas(**parameters)
    cached = CachedProfile(gas, methods=["real"])
    radii = np.array([0.1, 1.0])

    assert cached.proj_cutoff == gas.proj_cutoff
    np.testing.assert_allclose(
        cached.projected(cosmo, radii, 1.0e14, 0.8),
        gas.projected(cosmo, radii, 1.0e14, 0.8),
    )


def test_projection_does_not_depend_on_requested_radii(cosmo):
    parameters = {**bpar_S19, "cutoff": 1000, "proj_cutoff": 1000}
    gas = bfg.Profiles.Schneider19.Gas(**parameters)
    reference = bfg.Profiles.Schneider19.Gas(
        **parameters, n_per_decade_proj=100
    ).projected(cosmo, np.array([1e-3, 0.3, 10]), 1.0e14, 0.8)[1]

    scalar = gas.projected(cosmo, 0.3, 1.0e14, 0.8)
    wide = gas.projected(cosmo, np.array([1e-3, 0.3, 10]), 1.0e14, 0.8)[1]
    assert scalar == pytest.approx(reference, rel=1e-2)
    assert wide == pytest.approx(reference, rel=1e-2)


def test_emissivity_table_accepts_documented_inputs():
    temperatures = np.geomspace(1e5, 1e9, 5)
    metallicities = np.linspace(0, 1, 3)
    redshifts = np.array([0.0, 1.0])  # Scale factor is then *descending*
    T, Z, A = np.meshgrid(
        temperatures, metallicities, 1 / (1 + redshifts), indexing="ij"
    )
    emissivity = np.sqrt(T) * (1 + Z) * A**2

    def expected(t, z, a):
        return np.sqrt(t) * (1 + z) * a**2

    grid_table = bfg.utils.EmissivityTable(T, Z, A, emissivity)
    point_table = bfg.utils.EmissivityTable(
        T.ravel(), Z.ravel(), A.ravel(), emissivity.ravel()
    )
    strict_table = bfg.utils.EmissivityTable(
        T, Z, A, emissivity, pad_low_T=False
    )

    query_T = np.array([[1e5, 1e7], [1e9, 1e6]])
    query_Z = np.full_like(query_T, 0.5)
    for table in (grid_table, point_table, strict_table):
        for a in (0.5, 1.0):
            result = table(query_T, query_Z, a)
            assert result.shape == query_T.shape
            np.testing.assert_allclose(
                result, expected(query_T, query_Z, a), rtol=0.3
            )
            # Exactly on the tabulated grid points
            np.testing.assert_allclose(
                table(query_T, query_Z * 0, a)[0, 0], expected(1e5, 0, a)
            )


class _Gaussian(BaseBFGProfiles):
    """Gaussian of width 0.3 Mpc, with its analytic projection."""

    def _real(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        m_use = np.atleast_1d(M)
        result = np.exp(-r_use[None, :]**2 / 2 / 0.3**2) * np.ones((m_use.size, 1))
        if np.ndim(r) == 0:
            result = np.squeeze(result, axis=-1)
        if np.ndim(M) == 0:
            result = np.squeeze(result, axis=0)
        return result

    def projected(self, cosmo, r, M, a):
        return np.sqrt(2 * np.pi) * 0.3 * self._real(cosmo, r, M, a)


def test_grid_pixel_window_matches_pixel_average():
    size = 0.5
    radii = np.array([1e-4, 0.3, 0.6])
    convolved = ConvolvedProfile(_Gaussian(), bfg.utils.GridPixelApprox(size=size))

    # Direct average of the projected profile over a square pixel centered at each radius
    offsets = np.linspace(-size / 2, size / 2, 201)
    dx, dy = np.meshgrid(offsets, offsets)
    direct = np.array([
        np.mean(np.sqrt(2 * np.pi) * 0.3 * np.exp(-((r + dx)**2 + dy**2) / 2 / 0.3**2))
        for r in radii
    ])
    np.testing.assert_allclose(
        convolved.projected(None, radii, 1.0e14, 0.8), direct, rtol=0.05
    )

    # Same in 3D, averaging the real-space profile over a cubic pixel
    offsets = np.linspace(-size / 2, size / 2, 31)
    dx, dy, dz = np.meshgrid(offsets, offsets, offsets)
    direct = np.array([
        np.mean(np.exp(-((r + dx)**2 + dy**2 + dz**2) / 2 / 0.3**2)) for r in radii
    ])
    np.testing.assert_allclose(
        convolved.real(None, radii, 1.0e14, 0.8), direct, rtol=0.05
    )


def test_runners_default_to_the_model_mass_definition(cosmo):
    mass_def = ccl.halos.massdef.MassDef500c
    model = LinearProfile(mass_def=mass_def)
    catalog, shell = SimpleNamespace(cosmology={}), SimpleNamespace()

    assert DefaultRunner(catalog, shell, epsilon_max=1, model=model).mass_def is model.mass_def
    assert DefaultRunner(catalog, shell, epsilon_max=1, model=None).mass_def.name == "200c"
    explicit = ccl.halos.massdef.MassDef200m
    assert DefaultRunner(catalog, shell, epsilon_max=1, model=model, mass_def=explicit).mass_def is explicit

    table = bfg.utils.ParamTabulatedProfile(model, cosmo)  # Has no mass_def itself
    assert DefaultRunner(catalog, shell, epsilon_max=1, model=table).mass_def is model.mass_def

    baryonification = bfg.Profiles.Baryonification3D(model, LinearProfile(mass_def=mass_def), cosmo)
    assert baryonification.mass_def is model.mass_def


class TwoParameterProfile(BaseBFGProfiles):
    model_param_names = ["alpha", "beta"]

    def __init__(self, alpha=1, beta=1, **kwargs):
        super().__init__(alpha=alpha, beta=beta, **kwargs)
        self._projected = self._real

    def _real(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        m_use = np.atleast_1d(M)
        result = (m_use[:, None] / 1.0e14) * (self.alpha + 10 * self.beta) * r_use[None, :]
        if np.ndim(r) == 0:
            result = np.squeeze(result, axis=-1)
        if np.ndim(M) == 0:
            result = np.squeeze(result, axis=0)
        return result


def test_parameter_tables_match_parameters_by_name(cosmo):
    table = bfg.utils.ParamTabulatedProfile(TwoParameterProfile(), cosmo)
    table.setup_interpolator(
        z_min=0.1, z_max=0.1, N_samples_z=1,
        M_min=1.0e13, M_max=1.0e14, N_samples_Mass=2,
        R_min=0.1, R_max=1.0, N_samples_R=2,
        other_params={"alpha": [1.0, 2.0], "beta": (1.0, 3.0)},  # Lists/tuples are accepted
        verbose=False,
    )
    # Evaluate on the table's grid points, where the interpolation is exact
    expected = TwoParameterProfile(alpha=2.0, beta=3.0).real(cosmo, 0.1, 1.0e14, 1 / 1.1)
    for kwargs in ({"alpha": 2.0, "beta": 3.0}, {"beta": 3.0, "alpha": 2.0}):
        np.testing.assert_allclose(table.real(cosmo, 0.1, 1.0e14, 1 / 1.1, **kwargs), expected)

    with pytest.raises(ValueError, match="gamma"):
        table.real(cosmo, 0.1, 1.0e14, 1 / 1.1, alpha=2.0, beta=3.0, gamma=1)

    # Tabulating must leave the model's parameters (including those of sub-profiles) unchanged
    assert (table.model.alpha, table.model.beta) == (1, 1)
    composite = TwoParameterProfile(alpha=4) + TwoParameterProfile(alpha=6)
    bfg.utils.ParamTabulatedProfile(composite, cosmo).setup_interpolator(
        z_min=0.1, z_max=0.1, N_samples_z=1, M_min=1.0e13, M_max=1.0e14, N_samples_Mass=2,
        R_min=0.1, R_max=1.0, N_samples_R=2, other_params={"alpha": [1.0, 2.0]}, verbose=False,
    )
    assert (composite.Profile1.alpha, composite.Profile2.alpha) == (4, 6)


class _ToyBaryonification(bfg.Profiles.BaryonificationClass):
    """Analytic enclosed masses: the DMB mass is rescaled in radius by (1 + alpha + 10 beta)."""

    def get_masses(self, model, r, M, a):
        scale = 1 if model is self.DMO else (1 + model.alpha + 10 * model.beta)
        return M[:, None] * (1 - np.exp(-r[None, :] * scale))


def test_displacement_table_matches_parameters_by_name(cosmo):
    toy = _ToyBaryonification(TwoParameterProfile(), TwoParameterProfile(), cosmo)
    toy.setup_interpolator(
        z_min=0.1, z_max=0.2, N_samples_z=2,
        M_min=1.0e13, M_max=1.0e14, N_samples_Mass=2,
        R_min=0.05, R_max=1.0, N_samples_R=40,
        other_params={"alpha": [0.1, 0.2], "beta": [0.01, 0.05]}, verbose=False,
    )
    # Evaluate on the table's parameter grid points, where the interpolation is exact
    radii = np.array([0.1, 0.3])
    forward = toy.displacement(radii, 1.0e14, 1 / 1.15, alpha=0.2, beta=0.05)
    reverse = toy.displacement(radii, 1.0e14, 1 / 1.15, beta=0.05, alpha=0.2)
    np.testing.assert_allclose(forward, reverse)

    # For these masses, r_DMB = r / (1 + alpha + 10 beta), so the displacement is analytic
    np.testing.assert_allclose(forward, radii / (1 + 0.2 + 10 * 0.05) - radii, rtol=1e-2)

    with pytest.raises(ValueError, match="gamma"):
        toy.displacement(radii, 1.0e14, 1 / 1.15, alpha=0.2, beta=0.05, gamma=1)

    # Tabulating must leave the DMO/DMB parameters unchanged
    for profile in (toy.DMO, toy.DMB):
        assert (profile.alpha, profile.beta) == (1, 1)


def test_simple_array_cache_returns_copies():
    cached = SimpleArrayCache()(lambda values: np.asarray(values) * 2.0)
    first = cached(np.array([1.0, 2.0]))
    first *= 100  # Modifying the output in place must not change the cache
    np.testing.assert_array_equal(cached(np.array([1.0, 2.0])), [2.0, 4.0])
    second = cached(np.array([1.0, 2.0]))
    second[:] = 0
    np.testing.assert_array_equal(cached(np.array([1.0, 2.0])), [2.0, 4.0])


def test_truncated_fourier_matches_direct_transform(cosmo):
    from BaryonForge.Profiles.misc import TruncatedFourier

    parameters = {**bpar_S19, "r_steps": 64, "cutoff": 1000, "proj_cutoff": 1000}
    profile = bfg.Profiles.Schneider19.DarkMatter(**parameters)
    profile.update_precision_fftlog(n_per_decade=1000)
    mass, scale_factor = 1.0e14, 0.8
    radius = profile.mass_def.get_radius(cosmo, mass, scale_factor) / scale_factor
    k = np.array([0.05, 0.5, 2.0, 5.0]) / radius

    # Direct transform of the profile truncated at the halo radius
    r = np.geomspace(1e-6, radius, 20000)
    rho = profile.real(cosmo, r, mass, scale_factor)
    direct = np.array([np.trapz(4 * np.pi * r**2 * rho * np.sinc(ki * r / np.pi), r) for ki in k])

    truncated = TruncatedFourier(profile, epsilon_max=1)
    # 2% is the FFTLog accuracy at the lowest k; before the fix this was 10% at low k
    np.testing.assert_allclose(truncated.fourier(cosmo, k, mass, scale_factor), direct, rtol=2e-2)
    assert truncated.fourier(cosmo, k, np.array([1.0e13, 1.0e14]), scale_factor).shape == (2, 4)
    assert truncated.mass_def is profile.mass_def
    restored = pickle.loads(pickle.dumps(truncated))
    np.testing.assert_allclose(restored.fourier(cosmo, k, mass, scale_factor),
                               truncated.fourier(cosmo, k, mass, scale_factor))


def test_projection_at_zero_radius(cosmo):
    # A halo can sit exactly on a pixel center; the (cored) projection there must be finite
    parameters = {**bpar_S19, "r_steps": 64, "cutoff": 20, "proj_cutoff": 20}
    gas = bfg.Profiles.Schneider19.Gas(**parameters)
    result = gas.projected(cosmo, np.array([0.0, 1e-4, 0.1]), 1.0e14, 0.8)
    assert np.all(np.isfinite(result))
    assert result[0] == pytest.approx(result[1], rel=1e-3)


class _BackgroundWithEdge(BaseBFGProfiles):
    """Constant density with the (0.5 Mpc wide) exponential cutoff used by the two-halo terms."""

    def _real(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        m_use = np.atleast_1d(M)
        result = np.ones((m_use.size, 1)) / (1 + np.exp(2 * (r_use[None, :] - self.cutoff)))
        if np.ndim(r) == 0:
            result = np.squeeze(result, axis=-1)
        if np.ndim(M) == 0:
            result = np.squeeze(result, axis=0)
        return result


def test_projection_resolves_the_cutoff_edge(cosmo):
    # For terms with a mean-density background, the line-of-sight integral is set by the edge
    # at the cutoff, so the result must not depend on how the grid falls relative to that edge
    profile = _BackgroundWithEdge(cutoff=250, proj_cutoff=250)
    radius = 16.0
    l = np.linspace(0, 250, 500001)
    direct = 2 * np.trapz(profile._real(cosmo, np.sqrt(l**2 + radius**2), 1.0e14, 0.8), l)

    for radii in (np.geomspace(1e-3, 53.6, 10), np.geomspace(1e-3, 120, 10)):
        radii = np.sort(np.append(radii, radius))
        result = profile.projected(cosmo, radii, 1.0e14, 0.8)[radii == radius][0]
        assert result == pytest.approx(direct, rel=1e-2)


def test_legacy_projection_is_retained(cosmo):
    parameters = {**bpar_S19, "r_steps": 64, "cutoff": 20, "proj_cutoff": 20}
    gas = bfg.Profiles.Schneider19.Gas(**parameters)
    radii = np.array([0.01, 0.03, 0.1])
    legacy = gas._projected_realspace_legacy(cosmo, radii, 1.0e14, 0.8)
    np.testing.assert_allclose(legacy, gas.projected(cosmo, radii, 1.0e14, 0.8), rtol=5e-2)


def test_concentration_classes_and_instances(cosmo):
    parameters = {**bpar_S19, "r_steps": 64, "cutoff": 20, "proj_cutoff": 20}
    mass_def = ccl.halos.massdef.MassDef200c
    from_class = bfg.Profiles.Schneider19.DarkMatter(**parameters, c_M_relation=ccl.halos.ConcentrationDuffy08)
    from_instance = bfg.Profiles.Schneider19.DarkMatter(
        **parameters, c_M_relation=ccl.halos.ConcentrationDuffy08(mass_def=mass_def)
    )
    radii = np.array([0.1, 1.0])
    np.testing.assert_allclose(from_instance.real(cosmo, radii, 1.0e14, 0.8),
                               from_class.real(cosmo, radii, 1.0e14, 0.8))

    with pytest.warns(UserWarning, match="200m"):
        bfg.Profiles.Schneider19.DarkMatter(
            **parameters, c_M_relation=ccl.halos.ConcentrationDuffy08(mass_def=ccl.halos.massdef.MassDef200m)
        )


def test_cached_profiles_in_halo_model(cosmo):
    from BaryonForge.utils.Cache import CachedHODProfile

    mass_def = ccl.halos.massdef.MassDef200c
    hmc = ccl.halos.HMCalculator(mass_function="Tinker08", halo_bias="Tinker10", mass_def=mass_def, nM=16)
    k = np.array([0.1, 1.0])

    parameters = {**bpar_S19, "r_steps": 64, "cutoff": 1000}
    dark_matter = bfg.Profiles.Schneider19.DarkMatter(**parameters)
    np.testing.assert_allclose(
        ccl.halos.halomod_power_spectrum(cosmo, hmc, k, 0.8, CachedProfile(dark_matter)),
        ccl.halos.halomod_power_spectrum(cosmo, hmc, k, 0.8, dark_matter),
    )

    hod = ccl.halos.HaloProfileHOD(mass_def=mass_def, concentration=ccl.halos.ConcentrationDuffy08(mass_def=mass_def))
    cached_hod = CachedHODProfile(hod)
    np.testing.assert_allclose(
        ccl.halos.halomod_power_spectrum(cosmo, hmc, k, 0.8, cached_hod, prof_2pt=ccl.halos.Profile2ptHOD()),
        ccl.halos.halomod_power_spectrum(cosmo, hmc, k, 0.8, hod, prof_2pt=ccl.halos.Profile2ptHOD()),
    )
