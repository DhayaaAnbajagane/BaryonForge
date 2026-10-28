Quickstart
============

Run this template from the repository root. Replace the map and catalog placeholders with your data.

.. code-block:: python

    import numpy as np
    import BaryonForge as bfg
    import pyccl as ccl

    # Create the cosmology before building objects that need its parameter dictionary.
    cosmo = ccl.Cosmology(
        Omega_c=0.26, Omega_b=0.04, h=0.7, sigma8=0.8, n_s=0.96,
        matter_power_spectrum="linear",
    )
    cosmo_dict = bfg.utils.build_cosmodict(cosmo)

    # Replace these valid Python placeholders with your map and catalog arrays.
    HealpixMap = ...
    ra = dec = M200c = z = c200c = ...
    Shell = bfg.utils.LightconeShell(map=HealpixMap, cosmo=cosmo_dict)
    Catalog = bfg.utils.HaloLightConeCatalog(
        ra=ra, dec=dec, M=M200c, z=z, cosmo=cosmo_dict, cdelta=c200c,
    )

    # Load the source-tree example's Schneider19 parameter dictionary.
    profile_params = np.load("examples/default_config.npy", allow_pickle=True)[()]
    profile_params["M_c"] = 1e14
    profile_params["proj_cutoff"] = 100

    DMO = bfg.Profiles.DarkMatterOnly(**profile_params)
    DMB = bfg.Profiles.DarkMatterBaryon(**profile_params)
    model = bfg.Profiles.Baryonification2D(DMO, DMB, cosmo)
    model.setup_interpolator(
        z_min=Catalog.cat["z"].min(), z_max=Catalog.cat["z"].max(),
        N_samples_z=10, M_min=Catalog.cat["M"].min(),
        M_max=Catalog.cat["M"].max(), N_samples_Mass=10,
        R_min=1e-3, R_max=3e2, N_samples_R=500,
    )

    # Build a pressure profile, convert it to Compton-y, convolve, and tabulate it.
    gas = bfg.Profiles.Gas(**profile_params)
    pressure_dmb = bfg.Profiles.DarkMatterBaryon(
        **profile_params, twohalo=bfg.Profiles.misc.Zeros(),
    )
    pressure = bfg.Profiles.Pressure(gas=gas, darkmatterbaryon=pressure_dmb)
    tsz = bfg.Profiles.ThermalSZ(thermalpressure=pressure)
    tsz = bfg.Profiles.misc.ComovingToPhysical(tsz, factor=-3)
    pixel = bfg.utils.HealPixel(NSIDE=1024)
    profile = bfg.utils.ConvolvedProfile(tsz, pixel)
    profile = bfg.utils.TabulatedProfile(profile, cosmo)
    profile.setup_interpolator(
        z_min=Catalog.cat["z"].min(), z_max=Catalog.cat["z"].max(),
        N_samples_z=10, M_min=Catalog.cat["M"].min(),
        M_max=Catalog.cat["M"].max(), N_samples_Mass=10,
        R_min=1e-3, R_max=3e2, N_samples_R=500,
    )

    baryonified_map = bfg.Runners.BaryonifyShell(
        Catalog, Shell, model=model, epsilon_max=20,
    ).process()
    painted_map = bfg.Runners.PaintProfilesShell(
        Catalog, Shell, model=profile, epsilon_max=20,
    ).process()

The profile parameters are loaded from ``examples/default_config.npy``; the examples folder contains other notebooks for model-specific workflows.
