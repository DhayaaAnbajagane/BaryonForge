import numpy as np
import pyccl as ccl

#Define relevant physical constants
Msun_to_Kg = ccl.physical_constants.SOLAR_MASS
Mpc_to_m   = ccl.physical_constants.MPC_TO_METER
G          = ccl.physical_constants.GNEWT / Mpc_to_m**3 * Msun_to_Kg
m_to_cm    = 1e2
kb_cgs     = ccl.physical_constants.KBOLTZ * 1e7 
K_to_kev   = ccl.physical_constants.KBOLTZ / ccl.physical_constants.EV_IN_J * 1e-3

#Just define some useful conversions/constants
sigma_T = 6.652458e-29 / Mpc_to_m**2
m_e     = 9.10938e-31 / Msun_to_Kg
m_p     = 1.67262e-27 / Msun_to_Kg
c       = 2.99792458e8 / Mpc_to_m

#CGS units of everything, to use in thermalSZ
sigma_T_cgs = 6.652458e-29 * m_to_cm**2 #m^2 -> cm^2
m_e_cgs     = 9.10938e-31 * 1e3 #Kg -> g
m_p_cgs     = 1.67262e-27 * 1e3 #Kg -> g
c_cgs       = 2.99792458e8 * m_to_cm #m/s -> cm/s

#Thermodynamic/abundance quantities
Y         = 0.24 #Helium mass ratio
Pth_to_Pe = (4 - 2*Y)/(8 - 5*Y) #Factor to convert gas temp. to electron temp