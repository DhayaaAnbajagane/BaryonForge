"""Fast unit tests for profile wrappers and interpolation utilities.

Test index:
    test_unit_pixel_window_leaves_profile_unchanged: checks no-pixel convolution.
    test_no_pixel_window_is_public_and_returns_ones: checks identity pixel window.
    test_pixel_windows_have_correct_zero_mode_and_shapes: checks pixel window outputs.
    test_comoving_to_physical_applies_expected_scale_factor_powers: checks scale-factor conversion.
    test_tabulated_profile_uses_a_tiny_analytic_grid: checks tiny analytic tabulation.
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
    np.testing.assert_allclose(
        tabulated.projected(cosmo, radius, mass, scale_factor),
        model.projected(cosmo, radius, mass, scale_factor) * scale_factor,
        rtol=1.0e-12,
    )
