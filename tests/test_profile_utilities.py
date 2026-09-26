"""Fast unit tests for profile wrappers and interpolation utilities.

Test index:
    test_unit_pixel_window_leaves_profile_unchanged: checks no-pixel convolution.
    test_no_pixel_window_is_public_and_returns_ones: checks identity pixel window.
    test_pixel_windows_have_correct_zero_mode_and_shapes: checks pixel window outputs.
    test_comoving_to_physical_applies_expected_scale_factor_powers: checks scale-factor conversion.
    test_tabulated_profile_uses_a_tiny_analytic_grid: checks tiny analytic tabulation.
    test_table_interpolation_matches_scipy_exactly: checks the numba table readout against scipy.
    test_table_curves_match_the_table_readout_exactly: checks the readout along all nodes of one axis.
    test_batched_table_readout_matches_per_halo_readout: checks the runners' batched readouts.
    test_wrappers_do_not_expose_the_batched_readout_of_their_input: checks the batched-readout guard.
    test_tables_built_in_worker_processes_are_identical: checks setup_interpolator(n_jobs = 2).
"""

import numpy as np
import pyccl as ccl
import pytest

import BaryonForge as bfg
from BaryonForge.Profiles.Base import BaseBFGProfiles

from defaults import ccl_dict


class GaussianProfile(BaseBFGProfiles):
    """Smooth analytic profile for testing numerical FFTLog round trips."""

    @staticmethod
    def _values(coordinate, mass, normalization, exponent_scale):
        coordinate_use = np.atleast_1d(coordinate)
        mass_use = np.atleast_1d(mass)
        radial = normalization * np.exp(
            exponent_scale * coordinate_use**2
        )
        values = np.broadcast_to(radial, (mass_use.size, coordinate_use.size))

        if np.ndim(coordinate) == 0:
            values = np.squeeze(values, axis=-1)
        if np.ndim(mass) == 0:
            values = np.squeeze(values, axis=0)
        return values

    def _real(self, cosmo, r, M, a):
        return self._values(r, M, normalization=1, exponent_scale=-1)

    def projected(self, cosmo, r, M, a):
        return self._values(
            r, M, normalization=np.sqrt(np.pi), exponent_scale=-1
        )

    def _fourier(self, cosmo, k, M, a):
        return self._values(
            k, M, normalization=np.pi**1.5, exponent_scale=-0.25
        )


class LogLinearProfile(BaseBFGProfiles):
    """Positive profile whose logarithm is linear in the table coordinates."""

    @staticmethod
    def _values(r, M, a, normalization, radius_power):
        r_use = np.atleast_1d(r)
        mass_use = np.atleast_1d(M)
        values = (
            normalization
            * (mass_use[:, None] / 1.0e14)
            * r_use[None, :] ** radius_power
            * a**-2
        )

        if np.ndim(r) == 0:
            values = np.squeeze(values, axis=-1)
        if np.ndim(M) == 0:
            values = np.squeeze(values, axis=0)
        return values

    def _real(self, cosmo, r, M, a):
        return self._values(r, M, a, normalization=1, radius_power=2)

    def projected(self, cosmo, r, M, a):
        return self._values(r, M, a, normalization=2, radius_power=1)


@pytest.mark.parametrize("method", ("real", "projected", "fourier"))
def test_unit_pixel_window_leaves_profile_unchanged(method):
    profile = GaussianProfile()
    profile.update_precision_fftlog(n_per_decade=64)
    convolved = bfg.utils.ConvolvedProfile(profile, bfg.utils.NoPix())
    coordinate = np.geomspace(0.01, 2, 10)
    masses = [1.0e13, 1.0e14]

    expected = getattr(profile, method)(None, coordinate, masses, 0.8)
    result = getattr(convolved, method)(None, coordinate, masses, 0.8)

    # The Fourier result is algebraically exact. Real and projected space make
    # an FFTLog round trip, so allow its small numerical interpolation error.
    rtol = 0 if method == "fourier" else 2.0e-4
    np.testing.assert_allclose(result, expected, rtol=rtol, atol=0)


def test_no_pixel_window_is_public_and_returns_ones():
    window = bfg.utils.NoPix()
    k = np.array([0.0, 0.1, 10.0])

    assert window.isHarmonic is False
    assert window.size == 0
    np.testing.assert_array_equal(window.real(k), np.ones_like(k))
    np.testing.assert_array_equal(window.projected(k), np.ones_like(k))


def test_pixel_windows_have_correct_zero_mode_and_shapes():
    k = np.array([0.0, 0.1, 1.0, 10.0])
    grid = bfg.utils.GridPixelApprox(size=0.5)
    healpix = bfg.utils.HealPixel(NSIDE=32)

    for result in (grid.real(k), grid.projected(k), healpix.projected(k)):
        assert result.shape == k.shape
        assert np.all(np.isfinite(result))
        assert result[0] == pytest.approx(1)

    np.testing.assert_array_equal(healpix.real(k), np.zeros_like(k))


def test_comoving_to_physical_applies_expected_scale_factor_powers():
    converted = bfg.Profiles.misc.ComovingToPhysical(
        bfg.Profiles.misc.Identity(), factor=-3
    )
    radii = [0.1, 1.0]
    masses = [1.0e13, 1.0e14]

    np.testing.assert_allclose(
        converted.real(None, radii, masses, 0.5), np.full((2, 2), 8.0)
    )
    np.testing.assert_allclose(
        converted.projected(None, radii, masses, 0.5),
        np.full((2, 2), 4.0),
    )


def test_tabulated_profile_uses_a_tiny_analytic_grid():
    """Exercise tabulation without putting a physical-model table in CI."""
    cosmo = ccl.Cosmology(**ccl_dict)
    model = LogLinearProfile()
    tabulated = bfg.utils.TabulatedProfile(model, cosmo)

    with pytest.raises(NameError, match="No Table created"):
        tabulated.real(cosmo, 0.1, 1.0e14, 0.8)

    tabulated.setup_interpolator(
        z_min=0.2,
        z_max=1.0,
        N_samples_z=3,
        M_min=1.0e13,
        M_max=1.0e15,
        N_samples_Mass=3,
        R_min=0.01,
        R_max=10,
        N_samples_R=8,
        verbose=False,
    )

    radius = np.array([0.03, 0.3, 3.0])
    mass = np.array([3.0e13, 3.0e14])
    scale_factor = 1 / 1.5

    np.testing.assert_allclose(
        tabulated.real(cosmo, radius, mass, scale_factor),
        model.real(cosmo, radius, mass, scale_factor),
        rtol=1.0e-12,
    )
    #The tabulator stores the model's own projected profile, with no extra factor of
    #"a". Converting to physical units is the caller's job, via ComovingToPhysical.
    np.testing.assert_allclose(
        tabulated.projected(cosmo, radius, mass, scale_factor),
        model.projected(cosmo, radius, mass, scale_factor),
        rtol=1.0e-12,
    )


@pytest.mark.parametrize("sizes", ([5, 2], [1, 6], [4, 1], [3, 5, 7], [2, 1, 6], [3, 2, 4, 5]))
@pytest.mark.parametrize("fill_value", (np.nan, None, 0.0))
def test_table_interpolation_matches_scipy_exactly(sizes, fill_value):
    """The numba readout used by the tables reproduces scipy's RegularGridInterpolator bit for bit."""
    from scipy.interpolate import RegularGridInterpolator
    from BaryonForge.utils.Tabulate import _interpolate

    rng = np.random.default_rng(len(sizes) * 10 + sizes[0])
    grids = tuple(np.sort(rng.uniform(-2, 2, n)) if n > 1 else np.array([0.3]) for n in sizes)
    table = RegularGridInterpolator(grids, rng.normal(size=sizes), bounds_error=False, fill_value=fill_value)

    points = []
    for g in grids:
        c = rng.uniform(g[0] - 1, g[-1] + 1, 300)  # inside and outside
        c[:50] = rng.choice(g, 50)  # on the nodes
        c[50:55] = g[-1]  # on the top edge
        c[55:58] = np.nan
        rng.shuffle(c)
        points.append(c)

    np.testing.assert_array_equal(_interpolate(table, tuple(points)), table(tuple(points)))


@pytest.mark.parametrize("sizes, axis", (([3, 5, 7], 2), ([2, 1, 6], 2), ([3, 2, 4, 5], 2), ([4, 3, 1], 1), ([3, 1, 4], 1),
                                         ([2, 3, 4, 2, 3], 0)))
@pytest.mark.parametrize("fill_value", (np.nan, None, 0.0))
def test_table_curves_match_the_table_readout_exactly(sizes, axis, fill_value):
    """The readout along all nodes of one axis (the snapshot runner's displacement curves) matches the table
    readout at those points bit for bit, including non-finite table values and coordinates."""
    from scipy.interpolate import RegularGridInterpolator
    from BaryonForge.utils.Tabulate import _interpolate, _interpolate_curves

    rng = np.random.default_rng(len(sizes) * 10 + axis)
    grids = tuple(np.sort(rng.uniform(-2, 2, n)) if n > 1 else np.array([0.3]) for n in sizes)
    values = rng.normal(size=sizes)
    values.flat[rng.integers(values.size, size=2)] = (np.nan, np.inf)
    table = RegularGridInterpolator(grids, values, bounds_error=False, fill_value=fill_value)

    rows = np.column_stack([rng.uniform(g[0] - 0.5, g[-1] + 0.5, 60) for k, g in enumerate(grids) if k != axis])
    rows[:10] = [[rng.choice(g) for k, g in enumerate(grids) if k != axis] for _ in range(10)]  # on the nodes
    rows[10, 0] = np.nan
    nodes = grids[axis]
    points = [np.repeat(rows[:, k if k < axis else k - 1], nodes.size) if k != axis else np.tile(nodes, len(rows))
              for k in range(len(sizes))]
    expected = _interpolate(table, tuple(points)).reshape(len(rows), nodes.size)
    for n_threads in (1, 3):
        np.testing.assert_array_equal(_interpolate_curves(table, rows, axis, n_threads=n_threads), expected)


def _tiny_table(cls=None, **kwargs):
    cosmo = ccl.Cosmology(**ccl_dict)
    tabulated = (cls or bfg.utils.TabulatedProfile)(LogLinearProfile(), cosmo)
    tabulated.setup_interpolator(
        z_min=0.2, z_max=1.0, N_samples_z=3, M_min=1.0e13, M_max=1.0e15, N_samples_Mass=4,
        R_min=0.01, R_max=10, N_samples_R=16, verbose=False, **kwargs
    )
    return cosmo, tabulated


def test_batched_table_readout_matches_per_halo_readout():
    """`_real_batch`/`_projected_batch` give exactly what calling the table halo by halo gives."""
    cosmo, tabulated = _tiny_table()
    masses = np.array([2.0e13, 3.0e14, 7.0e14])
    scale_factors = np.array([0.9, 0.7, 0.55])
    radii = [np.geomspace(0.02, 5, 7), np.array([0.5]), np.geomspace(0.011, 9, 4)]
    halo = np.repeat(np.arange(3), [r.size for r in radii])
    r = np.concatenate(radii)

    for batch, method in ((tabulated._real_batch, tabulated.real), (tabulated._projected_batch, tabulated.projected)):
        expected = np.concatenate([method(cosmo, radii[i], masses[i], scale_factors[i]) for i in range(3)])
        np.testing.assert_array_equal(batch(r, halo, masses, scale_factors), expected)

    #Through ComovingToPhysical, which rescales by the scale factor of each halo
    physical = bfg.Profiles.misc.ComovingToPhysical(tabulated, factor=-3)
    expected = np.concatenate([physical.projected(cosmo, radii[i], masses[i], scale_factors[i]) for i in range(3)])
    np.testing.assert_allclose(physical._projected_batch(r, halo, masses, scale_factors), expected, rtol=1e-15)

    #And with an extra tabulated parameter
    cosmo, param_table = _tiny_table(bfg.utils.ParamTabulatedProfile, other_params={"cutoff": [50.0, 100.0]})
    cutoffs = np.array([60.0, 75.0, 99.0])
    expected = np.concatenate([param_table.projected(cosmo, radii[i], masses[i], scale_factors[i], cutoff=cutoffs[i])
                               for i in range(3)])
    np.testing.assert_array_equal(param_table._projected_batch(r, halo, masses, scale_factors, cutoff=cutoffs), expected)


def test_wrappers_do_not_expose_the_batched_readout_of_their_input():
    """A convolved (or combined) table must be evaluated through the wrapper, not its input's table."""
    from BaryonForge.utils.misc import _batch_method

    cosmo, tabulated = _tiny_table()
    assert _batch_method(tabulated, "_projected_batch") is not None
    convolved = bfg.utils.ConvolvedProfile(tabulated, bfg.utils.GridPixelApprox(0.1))
    assert _batch_method(convolved, "_projected_batch") is None
    assert _batch_method(bfg.Profiles.misc.ComovingToPhysical(convolved, factor=-3), "_projected_batch") is None
    #A CombinedProfile forwards unknown attributes to its first input, which here has a batched readout
    scaled = bfg.Profiles.misc.ComovingToPhysical(tabulated, factor=0)
    assert _batch_method(scaled, "_projected_batch") is not None
    assert _batch_method(scaled * 2, "_projected_batch") is None

    #A subclass that overrides the per-halo method must not have it bypassed by the inherited batched readout
    class Doubled(bfg.utils.TabulatedProfile):
        def _projected(self, cosmo, r, M, a):
            return 2 * super()._projected(cosmo, r, M, a)

    doubled = Doubled(LogLinearProfile(), cosmo)
    assert _batch_method(doubled, "_projected_batch", "projected", "_projected", "_readout") is None
    assert _batch_method(doubled, "_real_batch", "real", "_real", "_readout") is not None
    assert _batch_method(bfg.Profiles.misc.ComovingToPhysical(doubled, factor=0), "_projected_batch") is None


def test_tables_built_in_worker_processes_are_identical():
    """setup_interpolator(n_jobs = 2) builds the same tables as in serial, and restores the model parameters."""
    from defaults import bpar_S19

    cosmo = ccl.Cosmology(**ccl_dict)
    fast = dict(r_steps=64, cutoff=20, proj_cutoff=20, n_per_decade_proj=4)
    grid = dict(z_min=0.1, z_max=0.5, N_samples_z=2, M_min=1e13, M_max=1e15, N_samples_Mass=3, verbose=False)

    tables = []
    for n_jobs in (1, 2):
        gas = bfg.Profiles.Schneider19.Gas(**bpar_S19, **fast)
        table = bfg.utils.ParamTabulatedProfile(gas, cosmo)
        table.setup_interpolator(R_min=0.01, R_max=10, N_samples_R=16, other_params={"theta_ej": [3.0, 5.0]}, n_jobs=n_jobs, **grid)
        assert gas.theta_ej == bpar_S19["theta_ej"]

        B3 = bfg.Baryonification3D(bfg.Profiles.Schneider19.DarkMatterOnly(**bpar_S19, **fast),
                                   bfg.Profiles.Schneider19.DarkMatterBaryon(**bpar_S19, **fast), cosmo, N_int=200)
        B3.setup_interpolator(R_min=1e-3, R_max=20, N_samples_R=50, other_params={"theta_ej": [3.0, 5.0]}, n_jobs=n_jobs, **grid)
        tables.append((table.raw_input_3D, table.raw_input_2D, B3.raw_input_d))

    for serial, parallel in zip(*tables):
        np.testing.assert_array_equal(parallel, serial)
