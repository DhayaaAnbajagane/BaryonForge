"""Fast HEALPix runner tests.

Test index:
    test_baryonification_returns_zero_map_unchanged: checks zero-map shortcut.
    test_painting_skips_halos_smaller_than_a_pixel: checks halos with no pixels in their cutout.
    test_split_join_preserves_runner_settings: checks split runners inherit the painting settings.
    test_anisotropic_painting_assigns_all_tracer_to_single_halo: checks tracer/mass units in PaintProfilesAnisShell.
    test_baryonification_moves_mass_inward_and_conserves_it: checks BaryonifyShell end to end.
"""

import warnings

import numpy as np
import healpy as hp
import pyccl as ccl
import pytest

import BaryonForge as bfg
from BaryonForge.Profiles.Base import BaseBFGProfiles


def _cosmology():
    return ccl.Cosmology(
        Omega_c=0.26,
        Omega_b=0.04,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        matter_power_spectrum="linear",
    )


class GaussianProfile(BaseBFGProfiles):
    """Gaussian of width 0.3 Mpc, with an analytic projection."""

    def _real(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        m_use = np.atleast_1d(M)
        result = (m_use[:, None] / 1.0e14) * np.exp(-r_use[None, :]**2 / 2 / 0.3**2)
        if np.ndim(r) == 0:
            result = np.squeeze(result, axis=-1)
        if np.ndim(M) == 0:
            result = np.squeeze(result, axis=0)
        return result

    def projected(self, cosmo, r, M, a):
        return np.sqrt(2 * np.pi) * 0.3 * self._real(cosmo, r, M, a)


def test_baryonification_returns_zero_map_unchanged():
    """A zero input map should bypass profile calculations."""
    cosmology = _cosmology()
    cosmology_parameters = bfg.utils.build_cosmodict(cosmology)
    catalog = bfg.HaloLightConeCatalog(
        [0.0], [0.0], [1.0e14], [0.5], cosmology_parameters.copy()
    )
    shell = bfg.LightconeShell(
        np.zeros(hp.nside2npix(1)),
        cosmo=cosmology_parameters.copy(),
    )

    runner = bfg.BaryonifyShell(
        catalog, shell, epsilon_max=10, model=None, verbose=False
    )
    result = runner.process()

    np.testing.assert_array_equal(result, shell.map)


def test_painting_skips_halos_smaller_than_a_pixel():
    """A halo whose cutout contains no pixel centers contributes nothing (and must not crash)."""
    cosmology_parameters = bfg.utils.build_cosmodict(_cosmology())
    catalog = bfg.HaloLightConeCatalog(
        np.array([10.0]), np.array([10.0]), np.array([1.0e12]), np.array([1.0]), cosmology_parameters.copy()
    )
    shell = bfg.LightconeShell(
        np.zeros(hp.nside2npix(8)), cosmo=cosmology_parameters.copy(), redshift=1.0
    )
    gas = bfg.Profiles.Schneider19.Gas(
        theta_ej=4, theta_co=0.1, M_c=1e14, mu_beta=0.4, eta=0.3, eta_delta=0.3, tau=-1.5, tau_delta=0,
        A=0.045, M1=3e11, epsilon_h=0.015, a=0.3, n=2, epsilon=4, p=0.3, q=0.707, gamma=2, delta=7,
        r_steps=64, cutoff=20, proj_cutoff=20,
    )
    result = bfg.PaintProfilesShell(catalog, shell, epsilon_max=1, model=gas, verbose=False).process()
    np.testing.assert_array_equal(result, 0)


def test_split_join_preserves_runner_settings():
    cosmology_parameters = bfg.utils.build_cosmodict(_cosmology())
    catalog = bfg.HaloLightConeCatalog(
        np.array([10.0, 50.0]), np.array([10.0, -20.0]), np.array([1.0e14, 1.0e14]),
        np.array([0.3, 0.3]), cosmology_parameters.copy()
    )
    shell = bfg.LightconeShell(
        np.zeros(hp.nside2npix(16)), cosmo=cosmology_parameters.copy(), redshift=0.3
    )
    runner = bfg.PaintProfilesShell(
        catalog, shell, epsilon_max=20, model=GaussianProfile(), include_pixel_size=True, verbose=False
    )
    split = bfg.utils.SplitJoinParallel(runner, njobs=2)

    for sub_runner in split.Runner_list:
        assert sub_runner.include_pixel_size
        assert sub_runner.LightconeShell.redshift == shell.redshift

    joined = np.sum([sub_runner.process() for sub_runner in split.Runner_list], axis=0)
    np.testing.assert_allclose(joined, runner.process())


def test_anisotropic_painting_assigns_all_tracer_to_single_halo():
    """With one halo that dominates the mass budget, every pixel's tracer belongs to that halo."""
    cosmology_parameters = bfg.utils.build_cosmodict(_cosmology())
    catalog = bfg.HaloLightConeCatalog(
        np.array([10.0]), np.array([10.0]), np.array([1.0e14]), np.array([0.3]), cosmology_parameters.copy()
    )
    NSIDE = 64
    shell = bfg.LightconeShell(
        np.ones(hp.nside2npix(NSIDE)), cosmo=cosmology_parameters.copy(), redshift=0.3
    )
    #Huge mass normalization, so the halo exceeds the mean density and no background is assigned
    tracer = bfg.Profiles.misc.ComovingToPhysical(GaussianProfile(proj_cutoff=10) * 1e30, factor=-3)
    runner = bfg.PaintProfilesAnisShell(
        catalog, shell, epsilon_max=5, model=bfg.Profiles.misc.Identity(),
        Tracer_model=tracer, Mtot_model=tracer, background_val=1, global_tracer_fraction=1,
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = runner.process()

    vec = hp.ang2vec(10.0, 10.0, lonlat=True)
    center = hp.vec2pix(NSIDE, *vec)
    assert result[center] == pytest.approx(1, rel=1e-6)


class InwardDisplacement:
    """Comoving displacement of -1.5 Mpc within 6 comoving Mpc of the halo."""

    def displacement(self, r, M, a):
        return np.where(np.atleast_1d(r) < 6, -1.5, 0.0)


def test_baryonification_moves_mass_inward_and_conserves_it():
    cosmology_parameters = bfg.utils.build_cosmodict(_cosmology())
    NSIDE = 1024  # ~1.2 comoving Mpc pixels at z = 0.3
    catalog = bfg.HaloLightConeCatalog([30.0], [10.0], [1e15], [0.3], cosmology_parameters.copy())
    original = np.ones(hp.nside2npix(NSIDE))
    shell = bfg.LightconeShell(map=original, cosmo=cosmology_parameters.copy(), redshift=0.3)
    result = bfg.BaryonifyShell(catalog, shell, epsilon_max=5, model=InwardDisplacement(), verbose=False).process()

    assert result.sum() == pytest.approx(original.sum())
    disc = hp.query_disc(NSIDE, hp.ang2vec(30.0, 10.0, lonlat=True), np.radians(8 / 60))
    assert result[disc].sum() > 1.5 * original[disc].sum()
    far = hp.query_disc(NSIDE, hp.ang2vec(60.0, -20.0, lonlat=True), np.radians(8 / 60))
    np.testing.assert_allclose(result[far], original[far])
