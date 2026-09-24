Quickstart
============

.. code-block:: python
    
    import BaryonForge as bfg
    import pyccl as ccl

    #Add the healpix map and the lightcone halo catalog into the respective data objects
    Shell   = bfg.utils.LightconeShell(map = HealpixMap, cosmo = cosmo_dict)
    Catalog = bfg.utils.HaloLightConeCatalog(ra = ra, dec = dec, M = M200c, z = z, cosmo = cosmo_dict, cdelta = c200c)

    #Define a cosmology object, to be used in all profile calculations
    cosmo   = ccl.Cosmology(Omega_c = 0.26, h = 0.7, Omega_b = 0.04, sigma8 = 0.8, n_s = 0.96)

    #Define the DMO and DMB model which are the root of the baryonification routine
    #The model params can be specified during initialization of the class.
    #The Baryonification 2D class generates the offsets of density field.
    #We setup an interpolator to speed up the calculations.
    DMO     = bfg.Profiles.DarkMatterOnly(M_c = 1e14, proj_cutoff = 100)
    DMB     = bfg.Profiles.DarkMatterBaryon(M_c = 1e14, proj_cutoff = 100)
    model   = bfg.Profiles.Baryonification2D(DMO, DMB, cosmo)
    model.setup_interpolator(z_min = Catalog.cat['z'].min(), z_max = Catalog.cat['z'].max(), N_samples_z = 10,
                            M_min = Catalog.cat['M'].min(), M_max = Catalog.cat['M'].max(), N_samples_Mass = 10,
                            R_min = 1e-3, R_max = 3e2, N_samples_R = 500,)

    #The halo pressure profile as well. This is converted to Compton-y, convolved with a
    #Healpix window function, and then tabulated for speedup
    PRESS   = bfg.Profiles.Pressure(theta_ej = 8, theta_co = 0.1, mu_theta_ej = 0.1, ...)
    PRESS   = bfg.Profiles.ThermalSZ(PRESS)
    PRESS   = bfg.Profiles.misc.ComovingToPhysical(PRESS, factor = -3)
    Pixel   = bfg.utils.HealPixel(NSIDE = 1024)
    PRESS   = bfg.utils.ConvolvedProfile(PRESS, Pixel)
    PRESS   = bfg.utils.TabulatedProfile(PRESS, cosmo)
    PRESS.setup_interpolator(z_min = Catalog.cat['z'].min(), z_max = Catalog.cat['z'].max(), N_samples_z = 10,
                             M_min = Catalog.cat['M'].min(), M_max = Catalog.cat['M'].max(), N_samples_Mass = 10,
                             R_min = 1e-3, R_max = 3e2, N_samples_R = 500,)

    #Run the baryonification on this one shell
    Runner  = bfg.Runners.BaryonifyShell(Catalog, Shell, model = model, epsilon_max = 20)
    New_map = Runner.process()

    #Run the profile painting on this one shell
    Runner  = bfg.Runners.PaintProfilesShell(Catalog, Shell, model = PRESS, epsilon_max = 20)
    New_map = Runner.process()

See the `/examples <https://github.com/DhayaaAnbajagane/BaryonForge/blob/main/examples/>`_ folder in the Github repo for more notebooks demonstrating how to use the code for different applications.
