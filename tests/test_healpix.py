"""Fast HEALPix runner tests.

Test index:
    test_baryonification_returns_zero_map_unchanged: checks zero-map shortcut.
    test_painting_skips_halos_smaller_than_a_pixel: checks halos with no pixels in their cutout.
    test_split_join_preserves_runner_settings: checks split runners inherit the painting settings.
    test_anisotropic_painting_assigns_all_tracer_to_single_halo: checks tracer/mass units in PaintProfilesAnisShell.
    test_baryonification_moves_mass_inward_and_conserves_it: checks BaryonifyShell end to end (undisplaced pixels exact).
    test_split_join_does_not_create_empty_splits: checks catalogs that do not fill every job.
    test_anisotropic_painting_background_includes_pixel_size: checks the background's pixel-area factor.
    test_runners_do_not_depend_on_batching_or_n_jobs: checks batched/per-halo evaluation and threads agree exactly.
    test_map_scans_match_numpy: checks the compiled zero-map check, non-zero pixel search and moved-pixel split.
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


def test_split_join_does_not_create_empty_splits():
    cosmology_parameters = bfg.utils.build_cosmodict(_cosmology())
    rng = np.random.default_rng(4)
    catalog = bfg.HaloLightConeCatalog(
        rng.uniform(0, 90, 5), rng.uniform(-30, 30, 5), np.full(5, 1.0e14), np.full(5, 0.3),
        cosmology_parameters.copy()
    )
    shell = bfg.LightconeShell(np.zeros(hp.nside2npix(16)), cosmo=cosmology_parameters.copy(), redshift=0.3)
    runner = bfg.PaintProfilesShell(catalog, shell, epsilon_max=20, model=GaussianProfile(), verbose=False)

    # 5 halos over 4 jobs: 2 halos per split, so only 3 splits hold halos
    split = bfg.utils.SplitJoinParallel(runner, njobs=4)
    assert all(len(sub_runner.HaloLightConeCatalog.cat) > 0 for sub_runner in split.Runner_list)
    joined = np.sum([sub_runner.process() for sub_runner in split.Runner_list], axis=0)
    np.testing.assert_allclose(joined, runner.process())


def test_anisotropic_painting_background_includes_pixel_size():
    """The background term gets the same pixel-area factor as the halo terms."""
    cosmology = _cosmology()
    cosmology_parameters = bfg.utils.build_cosmodict(cosmology)
    catalog = bfg.HaloLightConeCatalog(
        np.array([10.0]), np.array([10.0]), np.array([1.0e14]), np.array([0.3]), cosmology_parameters.copy()
    )
    NSIDE = 64
    tracer = bfg.Profiles.misc.ComovingToPhysical(GaussianProfile(proj_cutoff=10) * 1e10, factor=-3)
    results = {}
    for include_pixel_size in (False, True):
        shell = bfg.LightconeShell(np.ones(hp.nside2npix(NSIDE)), cosmo=cosmology_parameters.copy(), redshift=0.3)
        runner = bfg.PaintProfilesAnisShell(
            catalog, shell, epsilon_max=5, model=bfg.Profiles.misc.Identity(),
            Tracer_model=tracer, Mtot_model=tracer, background_val=1, global_tracer_fraction=1,
            include_pixel_size=include_pixel_size, verbose=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results[include_pixel_size] = runner.process()

    far = hp.vec2pix(NSIDE, *hp.ang2vec(100.0, -40.0, lonlat=True))
    D_A = ccl.angular_diameter_distance(cosmology, 1 / 1.3)
    assert results[False][far] > 0
    assert results[True][far] / results[False][far] == pytest.approx(hp.nside2pixarea(NSIDE) * D_A**2, rel=1e-6)


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
    np.testing.assert_array_equal(result[far], original[far])  # Pixels no halo displaces keep their values exactly


class ScalingDisplacement:
    """Displacement that depends on radius, mass and scale factor, evaluated halo by halo."""

    def displacement(self, r, M, a):
        return -0.5 * (M / 1e14)**(1 / 3) * np.exp(-np.atleast_1d(r) / (2 * a))


class ScalingDisplacementBatched(ScalingDisplacement):
    """The same displacement through the batched readout that tabulated models provide (each halo's radii
    are evaluated with the per-halo formula, so the runners' bookkeeping can be checked exactly)."""

    def _displacement_batch(self, r, halo, M, a):
        out = np.empty(r.size)
        for h in np.unique(halo):
            out[halo == h] = self.displacement(r[halo == h], M[h], a[h])
        return out


class PerHaloOnly:
    """Hides the batched readout of a profile, so the runners call it halo by halo."""

    def __init__(self, profile):
        self.profile = profile
        self.mass_def = profile.mass_def

    def projected(self, cosmo, r, M, a):
        return self.profile.projected(cosmo, r, M, a)


def test_runners_do_not_depend_on_batching_or_n_jobs():
    """Batched and per-halo model evaluation, and any number of threads, give bit-identical maps."""
    cosmology = _cosmology()
    cosmology_parameters = bfg.utils.build_cosmodict(cosmology)
    rng = np.random.default_rng(3)
    N, NSIDE = 40, 64
    catalog = bfg.HaloLightConeCatalog(rng.uniform(0, 360, N), rng.uniform(-60, 60, N), 10**rng.uniform(13, 15, N),
                                       rng.uniform(0.2, 0.5, N), cosmology_parameters.copy())
    counts = rng.poisson(5, hp.nside2npix(NSIDE)).astype(float)

    def baryonify(model, n_jobs):
        shell = bfg.LightconeShell(map=counts, cosmo=cosmology_parameters.copy())
        return bfg.BaryonifyShell(catalog, shell, epsilon_max=5, model=model, verbose=False, n_jobs=n_jobs).process()

    reference = baryonify(ScalingDisplacement(), 1)
    np.testing.assert_array_equal(baryonify(ScalingDisplacementBatched(), 1), reference)
    np.testing.assert_array_equal(baryonify(ScalingDisplacementBatched(), 3), reference)
    assert reference.sum() == pytest.approx(counts.sum())

    table = bfg.utils.TabulatedProfile(GaussianProfile(), cosmology)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table.setup_interpolator(z_min=0.1, z_max=0.6, N_samples_z=4, M_min=1e12, M_max=1e16, N_samples_Mass=8,
                                 R_min=1e-3, R_max=50, N_samples_R=64, verbose=False)

    def paint(model, n_jobs):
        shell = bfg.LightconeShell(map=np.zeros_like(counts), cosmo=cosmology_parameters.copy())
        return bfg.PaintProfilesShell(catalog, shell, epsilon_max=5, model=model, verbose=False,
                                      include_pixel_size=True, n_jobs=n_jobs).process()

    reference = paint(PerHaloOnly(table), 1)
    assert reference.sum() > 0
    np.testing.assert_array_equal(paint(table, 1), reference)
    np.testing.assert_array_equal(paint(table, 3), reference)

    def paint_anisotropic(model, n_jobs):
        shell = bfg.LightconeShell(map=counts, cosmo=cosmology_parameters.copy(), redshift=0.35)
        return bfg.PaintProfilesAnisShell(catalog, shell, epsilon_max=5, model=model, Tracer_model=model,
                                          Mtot_model=table, background_val=1.0, global_tracer_fraction=0.1,
                                          verbose=False, n_jobs=n_jobs).process()

    reference = paint_anisotropic(PerHaloOnly(table), 1)
    np.testing.assert_array_equal(paint_anisotropic(table, 1), reference)
    np.testing.assert_array_equal(paint_anisotropic(table, 3), reference)


def test_map_scans_match_numpy():
    """The compiled map scans of BaryonifyShell give exactly the numpy expressions they replace."""
    from BaryonForge.Runners._chunks import _all_close_to_zero, _nonzero
    from BaryonForge.Runners.HealpixRunner import _split_moved

    for values in (np.zeros(50), np.full(50, 1e-9), np.r_[np.zeros(49), 2e-8], np.r_[np.zeros(49), np.nan],
                   np.r_[-1e-8, np.zeros(49)], np.r_[np.inf, np.zeros(49)], np.arange(50.0)):
        assert _all_close_to_zero(values) == np.allclose(values, 0)

    rng = np.random.default_rng(7)
    values = rng.normal(size=10_001) * (rng.random(10_001) < 0.3)
    values[[3, 7]] = np.nan, -0.0
    for n_threads in (1, 3, 8):
        np.testing.assert_array_equal(_nonzero(values, n_threads), np.flatnonzero(values))
    np.testing.assert_array_equal(_nonzero(np.zeros(5), 3), np.zeros(0, dtype=np.int64))

    offsets = rng.normal(size=(2000, 3)) * (rng.random((2000, 1)) < 0.5)
    offsets[5] = (0.0, np.nan, 0.0)
    counts = rng.poisson(3, 2000).astype(np.float32)
    pix = np.sort(rng.choice(2000, 700, replace=False))
    moved, off, moved_values, fixed, fixed_values = _split_moved(pix, offsets, counts)
    mask = np.any(offsets[pix] != 0, axis=1)
    for got, expected in ((moved, pix[mask]), (off, offsets[pix][mask]), (moved_values, counts[pix[mask]]),
                          (fixed, pix[~mask]), (fixed_values, counts[pix[~mask]])):
        np.testing.assert_array_equal(got, expected)
        assert got.dtype == expected.dtype
