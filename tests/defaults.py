import pyccl as ccl, numpy as np

#Load cosmology object from CCL. Linear P(k) is needed since we use it for 2-halo term.
#We don't use P(k) anywhere else in this model, so it's ok to use linear P(k) throughout
ccl_dict = dict(Omega_c = 0.26, Omega_b = 0.04, h = 0.7, sigma8 = 0.8, n_s = 0.96, matter_power_spectrum='linear')
h        = ccl_dict['h']

#Config params. Can change as you need. I store these as a dict and then unpack.
bpar_S19 = dict(theta_ej = 4, theta_co = 0.1, M_c = 1e14/h, mu_beta = 0.4,
                eta = 0.3, eta_delta = 0.3, tau = -1.5, tau_delta = 0, #Must use tau here since we go down to low mass
                A = 0.09/2, M1 = 2.5e11/h, epsilon_h = 0.015, 
                a = 0.3, n = 2, epsilon = 4, p = 0.3, q = 0.707, gamma = 2, delta = 7)

bpar_S25 = dict(epsilon0 = 4, epsilon1 = 0.5, alpha_excl = 0.4, p = 0.3, q = 0.707, M_c = 1e15, mu = 0.8, 
                q0 = 0.075, q1 = 0.25, q2 = 0.7, nu_q0 = 0, nu_q1 = 1, nu_q2 = 0, nstep = 3/2,
                theta_c = 0.3, nu_theta_c = 1/2, c_iga = 0.1, nu_c_iga = 3/2, r_min_iga = 1e-3, alpha = 1, gamma = 3/2, delta = 7,
                tau = -1.376, tau_delta = 0, Mstar = 3e11, Nstar = 0.03, eta = 0.1, eta_delta = 0.22, epsilon_cga = 0.03,
                alpha_nt = 0.1, nu_nt = 0.5, gamma_nt = 0.8, mean_molecular_weight = 0.6125)

bpar_A20 = dict(alpha_g = 2, epsilon_h = 0.015, M1_0 = 2.2e11/h, 
                alpha_fsat = 1, M1_fsat = 1, delta_fsat = 1, gamma_fsat = 1, eps_fsat = 1,
                M_c = 1.2e14/h, eta = 0.6, mu = 0.31, beta = 0.6, epsilon_hydro = np.sqrt(5),
                M_inn = 3.3e13/h, M_r = 1e16, beta_r = 2, theta_inn = 0.1, theta_out = 3,
                theta_rg = 0.3, sigma_rg = 0.1, a = 0.3, n = 2, p = 0.3, q = 0.707,
                A_nt = 0.495, alpha_nt = 0.1,
                mean_molecular_weight = 0.59)