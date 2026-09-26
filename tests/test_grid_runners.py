"""Fast tests for the gridded-map and particle-snapshot runners.

Test index:
    test_painted_halo_is_centered_on_its_position: checks painted-map centroids and axis ordering.
    test_baryonified_grid_is_symmetric_about_the_halo: checks displacement geometry and mass conservation.
    test_anisotropic_painting_background_units: checks the halo/background split of PaintProfilesAnisGrid.
    test_snapshot_particle_at_halo_center_stays_finite: checks the zero-separation particle.
    test_large_halos_on_grids_with_odd_half_size: checks cutouts larger than a quarter box on N = 50 grids.
    test_3D_grid_runners_are_centered_and_symmetric: checks 3D painting centroids and baryonification symmetry.
    test_gridded_map_coordinates_follow_axis_convention: checks GriddedMap.grid matches the map axes.
    test_elliptical_painting_orientation: checks ellipse orientation and axis ratio for +/- angles.
    test_baryonify_grid_isolates_non_finite_displacements: checks a NaN halo leaves other halos unchanged.
    test_baryonify_grid_accepts_parameterized_baryonification: checks BaryonificationClass with p_keys.
    test_baryonify_grid_does_not_depend_on_bin_origin: checks large cutouts for bins centered on zero.
    test_snapshot_output_is_wrapped_into_the_box: checks outputs lie in [0, L) and can be reused.
    test_snapshot_paths_match_the_kdtree_algorithm: checks the cell index, tabulated/per-halo paths and n_jobs (2D, 3D).
    test_grid_runners_do_not_depend_on_batching_or_n_jobs: checks batched/per-halo evaluation and threads agree exactly.
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


class OneMpcGaussian(GaussianProfile):
    width = 1.0


@pytest.mark.parametrize("angle", (30.0, -30.0, 75.0))
def test_elliptical_painting_orientation(cosmology_parameters, angle):
    n_pix, box = 80, 40.0
    bins = (np.arange(n_pix) + 0.5) * box / n_pix
    theta = np.radians(angle)
    catalog = bfg.HaloNDCatalog(x=[bins[40]], y=[bins[40]], M=[1e14], redshift=REDSHIFT, cosmo=dict(cosmology_parameters),
                                q_ell=[0.5], A_ell=np.array([[np.cos(theta), np.sin(theta)]]))
    gridded = bfg.GriddedMap(map=np.zeros((n_pix, n_pix)), bins=bins, redshift=REDSHIFT, cosmo=dict(cosmology_parameters))
    painted = bfg.PaintProfilesGrid(catalog, gridded, epsilon_max=5, model=OneMpcGaussian(),
                                    use_ellipticity=True, verbose=False).process()

    x, y = gridded.grid
    w = painted / painted.sum()
    dx, dy = x - bins[40], y - bins[40]
    Q = np.array([[(w*dx*dx).sum(), (w*dx*dy).sum()], [(w*dx*dy).sum(), (w*dy*dy).sum()]])
    evals, evecs = np.linalg.eigh(Q)
    assert np.sqrt(evals[0] / evals[1]) == pytest.approx(0.5, rel=1e-2)
    # A_ell is the major axis, see DefaultRunnerGrid.build_Rmat
    major = evecs[:, 1]
    assert abs(np.dot(major, [np.cos(theta), np.sin(theta)])) == pytest.approx(1, abs=1e-3)


class TableEdgeDisplacement(InwardDisplacement):
    """Like InwardDisplacement, but NaN for halos below 1e13 (as a table outside its mass range)."""

    def displacement(self, r, M, a):
        d = super().displacement(r, M, a)
        return d * np.nan if M < 1.0e13 else d


def test_baryonify_grid_isolates_non_finite_displacements(cosmology_parameters):
    density = np.ones((N_PIX, N_PIX))
    single = bfg.BaryonifyGrid(
        _catalog(BINS[20], BINS[12], cosmology_parameters), _grid(cosmology_parameters, density),
        epsilon_max=5, model=TableEdgeDisplacement(), verbose=False,
    ).process()

    # Add a small halo whose displacement is NaN, two pixels away from the first one
    pair = bfg.HaloNDCatalog(x=np.array([BINS[20], BINS[22]]), y=np.array([BINS[12], BINS[12]]),
                             M=np.array([1.0e14, 1.0e12]), redshift=REDSHIFT, cosmo=dict(cosmology_parameters))
    both = bfg.BaryonifyGrid(pair, _grid(cosmology_parameters, density), epsilon_max=5,
                             model=TableEdgeDisplacement(), verbose=False).process()
    np.testing.assert_allclose(both, single, rtol=1e-12)


class ParameterizedDisplacement(bfg.Profiles.BaryonificationClass):
    """Minimal BaryonificationClass with a tabulated extra parameter (``cdelta``)."""

    def __init__(self):
        self.p_keys = ["cdelta"]

    def displacement(self, r, M, a, cdelta):
        return InwardDisplacement().displacement(r, M, a)


def test_baryonify_grid_accepts_parameterized_baryonification(cosmology_parameters):
    catalog = _catalog(BINS[20], BINS[12], cosmology_parameters, cdelta=np.array([4.0]))
    density = np.ones((N_PIX, N_PIX))
    result = bfg.BaryonifyGrid(catalog, _grid(cosmology_parameters, density), epsilon_max=5,
                               model=ParameterizedDisplacement(), verbose=False).process()
    expected = bfg.BaryonifyGrid(_catalog(BINS[20], BINS[12], cosmology_parameters),
                                 _grid(cosmology_parameters, density), epsilon_max=5,
                                 model=InwardDisplacement(), verbose=False).process()
    np.testing.assert_allclose(result, expected)


class WideInwardDisplacement:
    """Every pixel within 7 Mpc moves 0.3 pixels inward."""

    def displacement(self, r, M, a):
        r = np.atleast_1d(r)
        return np.where(r < 7, -0.3 * RES, 0.0)


def test_baryonify_grid_does_not_depend_on_bin_origin(cosmology_parameters):
    density = np.ones((N_PIX, N_PIX))
    maps = []
    for origin in (0.0, -BOX / 2):
        bins = BINS + origin
        catalog = _catalog(bins[20], bins[20], cosmology_parameters)
        grid = bfg.GriddedMap(map=density, bins=bins, redshift=REDSHIFT, cosmo=dict(cosmology_parameters))
        maps.append(bfg.BaryonifyGrid(catalog, grid, epsilon_max=10, model=WideInwardDisplacement(),
                                      verbose=False).process())
    np.testing.assert_allclose(maps[1], maps[0], rtol=1e-12)


class OutwardDisplacement:
    """Every particle within 2 Mpc moves 0.3 Mpc outward."""

    def displacement(self, r, M, a):
        r = np.atleast_1d(r)
        return np.where(r < 2, 0.3, 0.0)


def test_snapshot_output_is_wrapped_into_the_box(cosmology_parameters):
    rng = np.random.default_rng(2)
    x, y, z = rng.uniform(0, BOX, (3, 500))
    x[:50] = rng.uniform(BOX - 1, BOX, 50)   # Particles near the edge get pushed across it
    x[50] = BOX                               # Snapshots stored on [0, L] can have x == L
    snapshot = bfg.ParticleSnapshot(x=x, y=y, z=z, M=np.ones(500), L=BOX,
                                    redshift=REDSHIFT, cosmo=dict(cosmology_parameters))
    catalog = _catalog(BOX - 0.5, 10.0, cosmology_parameters, z=np.array([10.0]))
    new_catalog = bfg.BaryonifySnapshot(catalog, snapshot, epsilon_max=5,
                                        model=OutwardDisplacement(), verbose=False).process()

    for axis in ("x", "y", "z"):
        assert np.all((new_catalog[axis] >= 0) & (new_catalog[axis] < BOX))

    # The output must be usable as an input again
    again = bfg.ParticleSnapshot(x=new_catalog["x"], y=new_catalog["y"], z=new_catalog["z"], M=np.ones(500),
                                 L=BOX, redshift=REDSHIFT, cosmo=dict(cosmology_parameters))
    bfg.BaryonifySnapshot(catalog, again, epsilon_max=5, model=OutwardDisplacement(), verbose=False).process()


class PiecewiseLogDisplacement:
    """Inward displacement that is linear in log(r) between nodes (like a table), and zero beyond 3 Mpc."""

    nodes = np.linspace(np.log(1e-3), np.log(10), 400)

    def _curve(self, M):
        return -0.3 * (M / 1e14)**(1 / 3) * np.exp(-np.exp(self.nodes) / 1.5)

    def displacement(self, r, M, a):
        u = np.log(np.atleast_1d(r))
        d = np.interp(u, self.nodes, self._curve(M))
        d = np.where((u >= self.nodes[0]) & (u <= self.nodes[-1]), d, np.nan)
        return np.where(np.atleast_1d(r) < 3, d, 0)


class PiecewiseLogDisplacementTabulated(PiecewiseLogDisplacement):
    """The same displacement, also offering the per-halo curves that tabulated models provide."""

    def _displacement_curves(self, M, a, r=None, n_threads=1):
        M = np.atleast_1d(M)
        return self.nodes, np.stack([self._curve(m) for m in M]), np.zeros(M.size), np.full(M.size, 3.0)


def _kdtree_baryonification(catalog, snapshot, epsilon_max, model):
    """The KDTree algorithm the snapshot runner used before the cell index, as a reference."""
    from scipy.spatial import KDTree
    from BaryonForge.utils.misc import _runner_cosmology

    cosmo, L = _runner_cosmology(catalog.cosmology), snapshot.L
    axes = ["x", "y"] if snapshot.is2D else ["x", "y", "z"]
    tree = KDTree(np.mod(np.vstack([snapshot.cat[ax] for ax in axes]).T, L), boxsize=L)
    wrap = lambda dx: np.where(np.where(dx > L / 2, dx - L, dx) < -L / 2, np.where(dx > L / 2, dx - L, dx) + L,
                               np.where(dx > L / 2, dx - L, dx))
    offsets, a = np.zeros([snapshot.cat.size, len(axes)]), 1 / (1 + catalog.redshift)
    for j in range(catalog.cat.size):
        M = catalog.cat["M"][j]
        R_q = np.clip(epsilon_max * ccl.halos.MassDef200c.get_radius(cosmo, M, a) / a, 0, L / 2)
        pos = [catalog.cat[ax][j] for ax in axes]
        inds = tree.query_ball_point(pos, R_q)
        dxs = [wrap(snapshot.cat[ax][inds] - p) for ax, p in zip(axes, pos)]
        d = np.sqrt(sum(dx**2 for dx in dxs))
        with np.errstate(invalid="ignore", divide="ignore"):
            hats = [np.where(d > 0, dx / d, 0) for dx in dxs]
        offset = model.displacement(d, M, a)
        offset = np.where(np.isfinite(offset), offset, 0)
        offsets[inds] += np.vstack([offset * h for h in hats]).T
    return [np.mod(snapshot.cat[ax] + offsets[:, i], L) for i, ax in enumerate(axes)]


@pytest.mark.parametrize("is2D", (False, True))
def test_snapshot_paths_match_the_kdtree_algorithm(cosmology_parameters, is2D):
    """The cell-index runner (compiled tabulated path, per-halo path, any n_jobs) matches the KDTree algorithm."""
    rng = np.random.default_rng(4)
    N, N_halo = 20000, 25
    x, y, z = rng.uniform(0, BOX, (3, N))
    x[:200] = rng.uniform(BOX - 0.5, BOX, 200)  # Near the edge
    x[200] = BOX                                # Snapshots stored on [0, L] can have x == L
    halo = np.column_stack([rng.uniform(0, BOX, (N_halo, 3))])
    halo[0] = [BOX - 0.2, 0.1, 10.0]           # Cutout wrapping across two edges
    x[201], y[201], z[201] = halo[3]            # A particle exactly at a halo center
    snapshot = bfg.ParticleSnapshot(x=x, y=y, z=None if is2D else z, M=np.ones(N), L=BOX,
                                    redshift=REDSHIFT, cosmo=dict(cosmology_parameters))
    catalog = bfg.HaloNDCatalog(x=halo[:, 0], y=halo[:, 1], z=None if is2D else halo[:, 2],
                                M=10**rng.uniform(13, 14.5, N_halo), redshift=REDSHIFT, cosmo=dict(cosmology_parameters))
    axes = ["x", "y"] if is2D else ["x", "y", "z"]

    reference = _kdtree_baryonification(catalog, snapshot, 5, PiecewiseLogDisplacement())
    runs = {}
    for label, model, n_jobs in (("per halo", PiecewiseLogDisplacement(), 1),
                                 ("tabulated", PiecewiseLogDisplacementTabulated(), 1),
                                 ("tabulated, 3 threads", PiecewiseLogDisplacementTabulated(), 3)):
        new = bfg.BaryonifySnapshot(catalog, snapshot, epsilon_max=5, model=model, verbose=False, n_jobs=n_jobs).process()
        runs[label] = new
        for i, ax in enumerate(axes):
            moved = np.abs(reference[i] - snapshot.cat[ax])
            assert moved.max() > 0.05  # The halos do move particles
            np.testing.assert_allclose(new[ax], reference[i], rtol=0, atol=1e-12, err_msg=f"{label}, axis {ax}")
            assert np.all((new[ax] >= 0) & (new[ax] < BOX))
    for ax in axes:
        np.testing.assert_array_equal(runs["tabulated, 3 threads"][ax], runs["tabulated"][ax])
    np.testing.assert_array_equal(runs["tabulated"]["M"], snapshot.cat["M"])


class ScalingDisplacement:
    """Displacement depending on radius and mass, evaluated halo by halo."""

    def displacement(self, r, M, a):
        return -0.4 * (M / 1e14)**(1 / 3) * np.exp(-np.atleast_1d(r) / 1.5)


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
        self.profile, self.mass_def = profile, profile.mass_def

    def real(self, cosmo, r, M, a):      return self.profile.real(cosmo, r, M, a)
    def projected(self, cosmo, r, M, a): return self.profile.projected(cosmo, r, M, a)


@pytest.mark.parametrize("dim, elliptical", ((2, False), (2, True), (3, False)))
def test_grid_runners_do_not_depend_on_batching_or_n_jobs(cosmology_parameters, dim, elliptical):
    """Batched and per-halo model evaluation, and any number of threads, give bit-identical maps."""
    rng = np.random.default_rng(5)
    N_halo = 12
    pos = rng.uniform(0, BOX, (N_halo, 3))
    pos[0, :] = BINS[7]  # A halo exactly on a pixel center
    extra = dict(q_ell=rng.uniform(0.5, 1, N_halo), A_ell=rng.normal(size=(N_halo, 2))) if elliptical else {}
    catalog = bfg.HaloNDCatalog(x=pos[:, 0], y=pos[:, 1], z=None if dim == 2 else pos[:, 2], M=10**rng.uniform(13, 14.5, N_halo),
                                redshift=REDSHIFT, cosmo=dict(cosmology_parameters), **extra)
    mass = rng.uniform(1, 2, (N_PIX,) * dim)
    grid = lambda values: bfg.GriddedMap(map=values, bins=BINS, redshift=REDSHIFT, cosmo=dict(cosmology_parameters))

    table = bfg.utils.TabulatedProfile(GaussianProfile(), ccl.Cosmology(**ccl_dict))
    table.setup_interpolator(z_min=0.2, z_max=0.3, N_samples_z=2, M_min=1e12, M_max=1e16, N_samples_Mass=8,
                             R_min=1e-3, R_max=50, N_samples_R=64, verbose=False)

    def run(runner, model, n_jobs, **kwargs):
        return runner(catalog, grid(mass if runner is bfg.BaryonifyGrid else np.zeros_like(mass)), epsilon_max=5,
                      model=model, verbose=False, use_ellipticity=elliptical, n_jobs=n_jobs, **kwargs).process()

    reference = run(bfg.BaryonifyGrid, ScalingDisplacement(), 1)
    assert reference.sum() == pytest.approx(mass.sum())
    np.testing.assert_array_equal(run(bfg.BaryonifyGrid, ScalingDisplacementBatched(), 1), reference)
    np.testing.assert_array_equal(run(bfg.BaryonifyGrid, ScalingDisplacementBatched(), 3), reference)

    reference = run(bfg.PaintProfilesGrid, PerHaloOnly(table), 1)
    assert reference.sum() > 0
    np.testing.assert_array_equal(run(bfg.PaintProfilesGrid, table, 1), reference)
    np.testing.assert_array_equal(run(bfg.PaintProfilesGrid, table, 3), reference)

    if dim == 2:
        anis = lambda model, n_jobs: bfg.PaintProfilesAnisGrid(
            catalog, grid(mass), epsilon_max=5, model=model, Tracer_model=model, Mtot_model=bfg.Profiles.misc.ComovingToPhysical(table, 0),
            background_val=1.0, global_tracer_fraction=0.1, use_ellipticity=elliptical, verbose=False, n_jobs=n_jobs).process()
        reference = anis(PerHaloOnly(table), 1)
        np.testing.assert_array_equal(anis(table, 1), reference)
        np.testing.assert_array_equal(anis(table, 3), reference)
