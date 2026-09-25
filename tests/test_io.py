"""Fast tests for the catalog and map containers in ``utils/io.py``.

Test index:
    test_catalogs_accept_scalars_lists_and_arrays: checks input types and float64 storage.
    test_catalogs_reject_inconsistent_inputs: checks informative errors for bad shapes/lengths.
    test_catalog_indexing_returns_catalogs: checks integer and slice indexing.
    test_containers_do_not_modify_the_cosmology_dictionary: checks the user's dict is copied.
    test_maps_and_snapshots_accept_lists: checks LightconeShell, GriddedMap, and ParticleSnapshot inputs.
"""

import numpy as np
import healpy as hp
import pytest

import BaryonForge as bfg


COSMO = {"Omega_m": 0.3, "Omega_b": 0.04, "h": 0.7, "sigma8": 0.8, "n_s": 0.96, "w0": -1.0, "wa": 0.0}


def test_catalogs_accept_scalars_lists_and_arrays():
    for ra, dec, M, z in ((10.0, 20.0, 1e14, 0.3), ([10.0], [20.0], [1e14], [0.3]),
                          ((10.0,), (20.0,), (1e14,), (0.3,)), (np.array([10.0]), 20, 1e14, 0.3)):
        catalog = bfg.HaloLightConeCatalog(ra, dec, M, z, COSMO, cdelta=5)
        assert catalog.cat.size == 1
        assert catalog.cat["cdelta"][0] == 5

    catalog = bfg.HaloNDCatalog(x=[1.0, 2.0], y=[3.0, 4.0], M=[1e14, 2e14], redshift=0.2, cosmo=COSMO,
                                A_ell=np.array([[1.0, 0.0], [0.0, 1.0]]))
    for name in ("M", "x", "y", "z", "A_ell"):
        assert catalog.cat.dtype[name].base == np.float64
    assert catalog.cat["A_ell"].shape == (2, 2)
    np.testing.assert_array_equal(catalog.cat["z"], 0)

    #Precision is kept, eg. for positions in large boxes
    catalog = bfg.HaloNDCatalog(x=[1000.123456789], y=[0.0], M=[1e14], redshift=0.2, cosmo=COSMO)
    assert catalog.cat["x"][0] == 1000.123456789

    #Declinations at the poles are offset, including for list inputs
    with pytest.warns(UserWarning, match="poles"):
        catalog = bfg.HaloLightConeCatalog([0.0, 1.0], [90, -90], [1e14, 1e14], [0.3, 0.3], COSMO)
    assert np.all(np.abs(catalog.cat["dec"]) < 90)


def test_catalogs_reject_inconsistent_inputs():
    with pytest.raises(ValueError, match="same length"):
        bfg.HaloLightConeCatalog([0.0, 1.0], [0.0], [1e14, 1e14], [0.3, 0.3], COSMO)
    with pytest.raises(ValueError, match="1D"):
        bfg.HaloLightConeCatalog(np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2)), np.ones((2, 2)), COSMO)
    with pytest.raises(ValueError, match="same length"):
        bfg.HaloNDCatalog(x=[1.0, 2.0], y=[3.0], M=[1e14, 1e14], redshift=0.2, cosmo=COSMO)
    with pytest.raises(ValueError, match="rows"):
        bfg.HaloNDCatalog(x=[1.0, 2.0], y=[3.0, 4.0], M=[1e14, 1e14], redshift=0.2, cosmo=COSMO,
                          A_ell=np.ones((3, 2)))
    with pytest.raises(ValueError, match="missing"):
        bfg.HaloNDCatalog(x=[1.0], y=[3.0], M=[1e14], redshift=0.2, cosmo={"Omega_m": 0.3})


def test_catalog_indexing_returns_catalogs():
    lightcone = bfg.HaloLightConeCatalog([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [1e13, 1e14, 1e15], [0.1, 0.2, 0.3],
                                         COSMO, cdelta=[4, 5, 6])
    single = lightcone[1]
    assert single.cat.size == 1 and single.cat["cdelta"][0] == 5
    assert lightcone[1:].cat.size == 2

    grid = bfg.HaloNDCatalog(x=[1.0, 2.0], y=[3.0, 4.0], M=[1e14, 2e14], redshift=0.2, cosmo=COSMO,
                             A_ell=np.array([[1.0, 0.0], [0.0, 1.0]]))
    single = grid[-1]
    assert single.cat.size == 1
    np.testing.assert_array_equal(single.cat["A_ell"][0], [0.0, 1.0])


def test_containers_do_not_modify_the_cosmology_dictionary():
    cosmo = {k: v for k, v in COSMO.items() if k != "wa"}
    with pytest.warns(UserWarning, match="wa"):
        catalog = bfg.HaloNDCatalog(x=[1.0], y=[3.0], M=[1e14], redshift=0.2, cosmo=cosmo)
    assert "wa" not in cosmo
    assert catalog.cosmology["wa"] == 0.0


def test_maps_and_snapshots_accept_lists():
    shell = bfg.LightconeShell(map=list(np.zeros(hp.nside2npix(1))), cosmo=COSMO)
    assert isinstance(shell.map, np.ndarray) and shell.NSIDE == 1
    with pytest.raises(ValueError, match="1D"):
        bfg.LightconeShell(map=np.zeros((2, 6)), cosmo=COSMO)

    grid = bfg.GriddedMap(map=[[0.0, 1.0], [2.0, 3.0]], bins=[0.5, 1.5], redshift=0.2, cosmo=COSMO)
    assert grid.res == 1.0 and grid.is2D

    snapshot = bfg.ParticleSnapshot(x=[1.0, 2.0], y=[1.0, 2.0], M=1e10, L=4.0, redshift=0.2, cosmo=COSMO)
    np.testing.assert_array_equal(snapshot.cat["M"], 1e10)
    assert snapshot.make_map(2).sum() == pytest.approx(2e10)
    without_mass = bfg.ParticleSnapshot(x=[1.0], y=[1.0], L=4.0, redshift=0.2, cosmo=COSMO)
    with pytest.raises(AssertionError, match="particle mass"):
        without_mass.make_map(2)
