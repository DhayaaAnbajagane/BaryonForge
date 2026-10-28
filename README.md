<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/DhayaaAnbajagane/BaryonForge/main/docs/source/LOGO_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/DhayaaAnbajagane/BaryonForge/main/docs/source/LOGO_light.png">
  <img alt="Logo" src="https://raw.githubusercontent.com/DhayaaAnbajagane/BaryonForge/main/docs/source/LOGO_dark.png" title="Logo">
</picture>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://img.shields.io/readthedocs/baryonforge?color=blue)](https://baryonforge.readthedocs.io/en/latest)

## Overview

A pipeline for *Baryonifying* N-body simulations, by adding baryon-induced corrections to the density field and/or adding thermodynamic fields such as the gas pressure, temperature etc. The entire modelling pipeline is built out of the Core Cosmology Library (CCL). The profile classes can also be used with CCL tools to compute analytic, halo model-based predictions for different power spectra.

## Features

- **Baryonification & Painting**: Modify density fields from N-body simulations and/or paint any field that has a halo profile associated with it


- **Maps, grids, and snapshots**: Work directly with 2D fields (eg. HealPix maps or 2D grids) but can also use 3D grids or full particle snapshots


- **Parallelized**: Painting and baryonification is parallelized under joblib.


A detailed documentation is available at [readthedocs](https://baryonforge.readthedocs.io/en/latest).

## Environment

The ```BaryonForge``` pipeline is designed to have minimal dependencies on external packages. The ```environment.yaml``` file can be used to create an environment that contains all necessary packages prior to your installation. We only specify three hard version requirements, which is ```pyccl>=3.1.2``` and ```numpy==1.*```, to avoid API-breaking changes, and ```scipy>=1.12``` for a cumulative simpson integration routine.


## Installation

The package can be installed through PyPi

```bash
pip install BaryonForge
```

You can also install directly from source, by running the following command:

```bash
pip install git+https://github.com/DhayaaAnbajagane/BaryonForge.git
```

or alternatively you can download the repo yourself and set it up,

```bash
git clone https://github.com/DhayaaAnbajagane/BaryonForge.git
cd BaryonForge
pip install -e .
```

The latter will keep the source files in the location you git clone'd from, and is useful if you are developing on the pipeline and are making frequent edits to it.


## Quickstart

Run this template from the repository root. Replace the map and catalog placeholders with your data.

```python
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
```

The profile parameters are loaded from `examples/default_config.npy`; the examples folder contains other notebooks for model-specific workflows.

## Attribution

If you use this code or derivatives of it, please cite [Anbajagane, Pandey & Chang 2024](https://arxiv.org/abs/2409.03822).

```bibtex
@ARTICLE{Anbajagane:2024:Baryonification,
       author = {{Anbajagane}, Dhayaa and {Pandey}, Shivam and {Chang}, Chihway},
        title = "{Map-level baryonification: Efficient modelling of higher-order correlations in the weak lensing and thermal Sunyaev-Zeldovich fields}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = 2024,
        month = sep,
          eid = {arXiv:2409.03822},
        pages = {arXiv:2409.03822},
archivePrefix = {arXiv},
       eprint = {2409.03822},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240903822A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Contact

Please contant Dhayaa Anbajagane (dhayaa at uchicago dot edu) for any questions on the pipeline (or bugs that you find!)
