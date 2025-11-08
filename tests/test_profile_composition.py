import BaryonForge as bfg
import numpy as np, healpy as hp
import pyccl as ccl

from defaults import bpar_S19, bpar_A20, ccl_dict, h
cosmo = ccl.Cosmology(**ccl_dict)
cosmo.compute_power()

if __name__ == '__main__':
    pass