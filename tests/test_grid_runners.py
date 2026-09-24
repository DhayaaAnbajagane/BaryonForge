"""Fast tests for the gridded-map and particle-snapshot runners.

Test index:
    test_painted_halo_is_centered_on_its_position: checks painted-map centroids and axis ordering.
    test_baryonified_grid_is_symmetric_about_the_halo: checks displacement geometry and mass conservation.
    test_anisotropic_painting_background_units: checks the halo/background split of PaintProfilesAnisGrid.
    test_snapshot_particle_at_halo_center_stays_finite: checks the zero-separation particle.
    test_large_halos_on_grids_with_odd_half_size: checks cutouts larger than a quarter box on N = 50 grids.
    test_3D_grid_runners_are_centered_and_symmetric: checks 3D painting centroids and baryonification symmetry.
    test_gridded_map_coordinates_follow_axis_convention: checks GriddedMap.grid matches the map axes.
"""

import numpy as np
import pyccl as ccl
import pytest

import BaryonForge as bfg
from BaryonForge.Profiles.Base import BaseBFGProfiles

from defaults import ccl_dict


N_PIX, BOX = 40, 20.0
RES  = BOX / N_PIX
BINS = (np.arange(N_PIX) + 0.5) * RES
REDSHIFT = 0.25


@pytest.fixture(scope="module")
def cosmology_parameters():
    return bfg.utils.build_cosmodict(ccl.Cosmology(**ccl_dict))


class GaussianProfile(BaseBFGProfiles):
    """Gaussian of width 0.3 Mpc, with an analytic projection."""

    width = 0.3

    def _real(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        m_use = np.atleast_1d(M)
        result = (m_use[:, None] / 1.0e14) * np.exp(-r_use[None, :]**2 / 2 / self.width**2)
        if np.ndim(r) == 0:
            result = np.squeeze(result, axis=-1)
        if np.ndim(M) == 0:
            result = np.squeeze(result, axis=0)
        return result

    def projected(self, cosmo, r, M, a):
        return np.sqrt(2 * np.pi) * self.width * self._real(cosmo, r, M, a)


class InwardDisplacement:
    """Radial displacement model: every pixel within 2 Mpc moves 0.3 pixels inward."""

    def displacement(self, r, M, a):
        r = np.atleast_1d(r)
        return np.where(r < 2, -0.3 * RES, 0.0)


def _catalog(x, y, parameters, **kwargs):
    return bfg.HaloNDCatalog(x=np.array([x]), y=np.array([y]), M=np.array([1.0e14]),
                             redshift=REDSHIFT, cosmo=dict(parameters), **kwargs)


def _grid(parameters, values=None):
    values = np.zeros((N_PIX, N_PIX)) if values is None else values
    return bfg.GriddedMap(map=values, bins=BINS, redshift=REDSHIFT, cosmo=dict(parameters))


@pytest.mark.parametrize("offset", ((0.0, 0.0), (0.2, -0.3)))
def test_painted_halo_is_centered_on_its_position(cosmology_parameters, offset):
    x0 = BINS[20] + offset[0] * RES
    y0 = BINS[12] + offset[1] * RES
    runner = bfg.PaintProfilesGrid(
        _catalog(x0, y0, cosmology_parameters), _grid(cosmology_parameters),
        epsilon_max=5, model=GaussianProfile(), verbose=False,
    )
    painted = runner.process()

    # Axis 0 of the map follows x, axis 1 follows y (as in ParticleSnapshot.make_map)
    total = painted.sum()
    centroid_x = (painted.sum(axis=1) * BINS).sum() / total
    centroid_y = (painted.sum(axis=0) * BINS).sum() / total
    assert centroid_x == pytest.approx(x0, abs=0.02 * RES)
    assert centroid_y == pytest.approx(y0, abs=0.02 * RES)

    # Total painted mass = integral of the projected profile over the pixels
    assert total == pytest.approx(2 * np.pi * GaussianProfile.width**3 * np.sqrt(2 * np.pi), rel=1e-2)


def test_baryonified_grid_is_symmetric_about_the_halo(cosmology_parameters):
    density = np.ones((N_PIX, N_PIX))
    runner = bfg.BaryonifyGrid(
        _catalog(BINS[20], BINS[12], cosmology_parameters), _grid(cosmology_parameters, density),
        epsilon_max=5, model=InwardDisplacement(), verbose=False,
    )
    new_map = runner.process()

    assert new_map.sum() == pytest.approx(density.sum())
    # Mass should pile up at the halo pixel, and the map must be point-symmetric about it
    assert np.unravel_index(np.argmax(new_map), new_map.shape) == (20, 12)
    window = new_map[20 - 6: 20 + 7, 12 - 6: 12 + 7]
    np.testing.assert_allclose(window, window[::-1, ::-1], rtol=1e-10)


def test_anisotropic_painting_background_units(cosmology_parameters):
    cosmo = ccl.Cosmology(**ccl_dict)
    tracer = GaussianProfile(proj_cutoff=10) * 1.0e12
    catalog = _catalog(BINS[20], BINS[20], cosmology_parameters)
    runner = bfg.PaintProfilesAnisGrid(
        catalog, _grid(cosmology_parameters, np.ones((N_PIX, N_PIX))),
        epsilon_max=5, model=bfg.Profiles.misc.Identity(), Tracer_model=tracer, Mtot_model=tracer,
        background_val=0, global_tracer_fraction=1, include_pixel_size=False, verbose=False,
    )
    new_map = runner.process()

    # Expected halo share of the tracer: Sigma_halo / (Sigma_halo + L * (rho_m - <Sigma_halo> / L))
    sigma = bfg.PaintProfilesGrid(catalog, _grid(cosmology_parameters), epsilon_max=5, model=tracer,
                                  include_pixel_size=False, verbose=False).process()
    length = 2 * 10
    rho_m = ccl.rho_x(cosmo, 1 / (1 + REDSHIFT), "matter", is_comoving=True)
    background = length * (rho_m - sigma.mean() / length)
    expected = sigma / (sigma + background)
    np.testing.assert_allclose(new_map[20, 20], expected[20, 20], rtol=1e-6)


def test_snapshot_particle_at_halo_center_stays_finite(cosmology_parameters):
    rng = np.random.default_rng(1)
    x, y, z = rng.uniform(0, BOX, (3, 500))
    x[0], y[0], z[0] = 10, 10, 10
    snapshot = bfg.ParticleSnapshot(x=x, y=y, z=z, M=np.ones(500), L=BOX,
                                    redshift=REDSHIFT, cosmo=dict(cosmology_parameters))
    catalog = _catalog(10.0, 10.0, cosmology_parameters, z=np.array([10.0]))
    new_catalog = bfg.BaryonifySnapshot(catalog, snapshot, epsilon_max=5,
                                        model=InwardDisplacement(), verbose=False).process()

    for axis in ("x", "y", "z"):
        assert np.all(np.isfinite(new_catalog[axis]))
    assert new_catalog["x"][0] == pytest.approx(10)


class WideGaussian(GaussianProfile):
    width = 3.0


def test_large_halos_on_grids_with_odd_half_size(cosmology_parameters):
    """N = 50 (so N/2 is odd), with cutouts (15 R ~ 36 Mpc) larger than a quarter of the box."""
    n_pix, box = 50, 50.0
    bins = (np.arange(n_pix) + 0.5) * box / n_pix
    catalog = bfg.HaloNDCatalog(x=[25.5], y=[25.5], M=[1e15], redshift=REDSHIFT, cosmo=dict(cosmology_parameters))

    def grid(values):
        return bfg.GriddedMap(map=values, bins=bins, redshift=REDSHIFT, cosmo=dict(cosmology_parameters))

    painted = bfg.PaintProfilesGrid(catalog, grid(np.zeros((n_pix, n_pix))), epsilon_max=15,
                                    model=WideGaussian(), verbose=False).process()
    #Integral of the projected Gaussian (amplitude M / 1e14 = 10) out to 15 R (the mask radius)
    radius = 15 * ccl.halos.MassDef200c.get_radius(ccl.Cosmology(**ccl_dict), 1e15, 1 / (1 + REDSHIFT)) * (1 + REDSHIFT)
    width = WideGaussian.width
    expected = 10 * 2 * np.pi * width**3 * np.sqrt(2 * np.pi) * (1 - np.exp(-radius**2 / 2 / width**2))
    assert painted.sum() == pytest.approx(expected, rel=1e-2)

    baryonified = bfg.BaryonifyGrid(catalog, grid(np.ones((n_pix, n_pix))), epsilon_max=15,
                                    model=InwardDisplacement(), verbose=False).process()
    assert baryonified.sum() == pytest.approx(n_pix**2)


def test_3D_grid_runners_are_centered_and_symmetric(cosmology_parameters):
    n_pix, box = 24, 12.0
    res  = box / n_pix
    bins = (np.arange(n_pix) + 0.5) * res

    def grid(values):
        return bfg.GriddedMap(map=values, bins=bins, redshift=REDSHIFT, cosmo=dict(cosmology_parameters))

    #Painting (3D uses the real-space profile)
    position = (bins[12] + 0.1 * res, bins[7] - 0.2 * res, bins[15] + 0.3 * res)
    catalog  = bfg.HaloNDCatalog(x=[position[0]], y=[position[1]], z=[position[2]], M=[1e14],
                                 redshift=REDSHIFT, cosmo=dict(cosmology_parameters))
    painted  = bfg.PaintProfilesGrid(catalog, grid(np.zeros((n_pix,) * 3)), epsilon_max=5,
                                     model=GaussianProfile(), verbose=False).process()
    for axis in range(3):
        others   = tuple(j for j in range(3) if j != axis)
        centroid = (painted.sum(axis=others) * bins).sum() / painted.sum()
        assert centroid == pytest.approx(position[axis], abs=0.02 * res)

    #Baryonification of a uniform field around a halo at a pixel center
    catalog = bfg.HaloNDCatalog(x=[bins[12]], y=[bins[7]], z=[bins[15]], M=[1e14],
                                redshift=REDSHIFT, cosmo=dict(cosmology_parameters))
    new_map = bfg.BaryonifyGrid(catalog, grid(np.ones((n_pix,) * 3)), epsilon_max=5,
                                model=InwardDisplacement(), verbose=False).process()
    assert new_map.sum() == pytest.approx(n_pix**3)
    assert np.unravel_index(np.argmax(new_map), new_map.shape) == (12, 7, 15)
    window = new_map[12 - 5: 12 + 6, 7 - 5: 7 + 6, 15 - 5: 15 + 6]
    np.testing.assert_allclose(window, window[::-1, ::-1, ::-1], rtol=1e-10)


def test_gridded_map_coordinates_follow_axis_convention(cosmology_parameters):
    gridded = _grid(cosmology_parameters)
    x, y = gridded.grid
    np.testing.assert_array_equal(x[:, 0], BINS)  #x varies along axis 0
    np.testing.assert_array_equal(y[0, :], BINS)  #y varies along axis 1

    #A painted halo peaks at the pixel whose grid coordinates are the halo position
    painted = bfg.PaintProfilesGrid(_catalog(BINS[20], BINS[12], cosmology_parameters), gridded,
                                    epsilon_max=5, model=GaussianProfile(), verbose=False).process()
    peak = np.unravel_index(np.argmax(painted), painted.shape)
    assert (x[peak], y[peak]) == (BINS[20], BINS[12])
