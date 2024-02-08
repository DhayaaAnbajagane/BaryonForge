
import numpy as np
import pyccl as ccl
from operator import add, mul, sub, truediv, pow, neg, pos, abs
import warnings

from scipy import interpolate
from astropy.cosmology import z_at_value, FlatLambdaCDM, FlatwCDM
from astropy import units as u

model_params = ['cdelta', 'epsilon', 'a', 'n', 
                'theta_ej', 'theta_co', 'M_c', 'mu', 'gamma', 'delta',
                'A', 'M1', 'eta', 'eta_delta', 'beta', 'beta_delta', 'epsilon_h',
                'q', 'p', 'cutoff']


#All profiles are exponentially suppressed after R = CUTOFF*R200c.
#This is just to prevent infinite density issues
CUTOFF = 10


class SchneiderProfiles(ccl.halos.profiles.HaloProfile):

    def __init__(self,
                 cdelta = None,
                 epsilon = None, a = None, n = None,
                 theta_ej = None, theta_co = None, M_c = None, mu = None, gamma = None, delta = None,
                 A = None, M1 = None, eta = None, eta_delta = None, beta = None, beta_delta = None, epsilon_h = None,
                 q = None, p = None, cutoff = None, xi_mm = None, R_range = [1e-10, 1e10], use_fftlog_projection = False):


        self.epsilon    = epsilon
        self.a          = a
        self.n          = n
        self.theta_ej   = theta_ej
        self.theta_co   = theta_co
        self.M_c        = M_c
        self.mu         = mu
        self.gamma      = gamma
        self.delta      = delta
        self.A          = A
        self.M1         = M1
        self.eta        = eta
        self.eta_delta  = eta_delta
        self.beta       = beta
        self.beta_delta = beta_delta 
        self.epsilon_h  = epsilon_h
        self.q          = q
        self.p          = p
        self.cdelta     = cdelta

        #Import all other parameters from the base CCL Profile class
        super(SchneiderProfiles, self).__init__()

        #Function that returns correlation func at different radii
        self.xi_mm     = xi_mm

        #Sets the range that we compute profiles too (if we need to do any numerical stuff)
        self.R_range = R_range
        
        #Sets the cutoff scale of all profiles, in comoving Mpc
        #This should be the box side length
        self.cutoff  = cutoff
        
        
        #This allows user to force usage of the default FFTlog projection, if needed.
        #Otherwise, we use the realspace integration, since that allows for specification
        #of a hard boundary on radius
        if not use_fftlog_projection:
            self._projected = self._projected_realspace

        #Constant that helps with the fourier transform convolution integral.
        #This value minimized the ringing due to the transforms
        self.precision_fftlog['plaw_fourier'] = -2

        #Need this to prevent projected profile from artificially cutting off
        self.precision_fftlog['padding_lo_fftlog'] = 1e-2
        self.precision_fftlog['padding_hi_fftlog'] = 1e2

        self.precision_fftlog['padding_lo_extra'] = 1e-4
        self.precision_fftlog['padding_hi_extra'] = 1e4
        
    
    @property
    def model_params(self):
        
        params = {k:v for k,v in vars(self).items() if k in model_params}
                  
        return params
        
        
    def _projected_realspace(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):
        '''
        Custom method for projection where we do it all in real-space. Not that slow and
        can avoid any hankel transform features.
        '''

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        #Integral limits
        int_min = self.precision_fftlog['padding_lo_fftlog']*np.min(r_use)
        int_max = self.precision_fftlog['padding_hi_fftlog']*np.max(r_use)
        int_N   = self.precision_fftlog['n_per_decade'] * np.int32(np.log10(int_max/int_min))
        
        #If cutoff was passed, then rewrite the integral max limit
        if self.cutoff is not None:
            int_max = self.cutoff

        r_integral = np.geomspace(int_min, int_max, int_N)

        prof = self._real(cosmo, r_integral, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical'))

        #The prof object is already "squeezed" in some way.
        #Code below removes that squeezing so rest of code can handle
        #passing multiple radii and masses.
        if np.ndim(r) == 0:
            prof = prof[:, None]
        if np.ndim(M) == 0:
            prof = prof[None, :]

        proj_prof = np.zeros([M_use.size, r_use.size])

        for i in range(M_use.size):
            for j in range(r_use.size):

                proj_prof[i, j] = 2*np.trapz(np.interp(np.sqrt(r_integral**2 + r_use[j]**2), r_integral, prof[i]), r_integral)

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            proj_prof = np.squeeze(proj_prof, axis=-1)
        if np.ndim(M) == 0:
            proj_prof = np.squeeze(proj_prof, axis=0)

        assert np.all(proj_prof >= 0), "Something went wrong. Profile is negative in some places"

        return proj_prof
    
    
    def __str_par__(self):
        
        string = f"("
        for m in model_params:
            string += f"{m} = {self.__dict__[m]}, "
        string += f"xi_mm = {self.xi_mm}, R_range = {self.R_range})"
        return string
        
    def __str_prf__(self):
        
        string = f"{self.__class__.__name__}"
        return string
        
    
    def __str__(self):
        
        string = self.__str_prf__() + self.__str_par__()
        return string 
    
    
    def __repr__(self):
        
        return self.__str__()
    
    
    
    from ..utils.misc import generate_operator_method
    
    #Add routines for doing simple arithmetic operations with the classes
    __add__     = generate_operator_method(add)
    __mul__     = generate_operator_method(mul)
    __sub__     = generate_operator_method(sub)
    __pow__     = generate_operator_method(pow)
    __truediv__ = generate_operator_method(truediv)
    
    __abs__     = generate_operator_method(abs)
    __pos__     = generate_operator_method(pos)
    __neg__     = generate_operator_method(neg)    



class DarkMatter(SchneiderProfiles):
    '''
    Total DM profile, which is just NFW
    '''

    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        if self.cdelta is None:
            c_M_relation = ccl.halos.concentration.ConcentrationDiemer15(mdef = mass_def) #Use the diemer calibration
            
        else:
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mdef = mass_def)
            #c_M_relation = ccl.halos.concentration.ConcentrationConstant(7, mdef = mass_def) #needed to get Schneider result
            
        c   = c_M_relation.get_concentration(cosmo, M_use, a)
        R   = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        r_s = R/c
        r_t = R*self.epsilon
        
        r_s, r_t = r_s[:, None], r_t[:, None]

        
        #Get the normalization (rho_c) numerically
        #The analytic integral doesn't work since we have a truncation radii now.
        r_integral = np.geomspace(1e-6, 100, 500)

        prof_integral  = 1/(r_integral/r_s * (1 + r_integral/r_s)**2) * 1/(1 + (r_integral/r_t)**2)**2
        Normalization  = [interpolate.CubicSpline(np.log(r_integral), 4 * np.pi * r_integral**3 * p) for p in prof_integral]
        Normalization  = np.array([N_i.integrate(np.log(r_integral[0]), np.log(R_i)) for N_i, R_i in zip(Normalization, R)])
        
        rho_c = M_use/Normalization
        rho_c = rho_c[:, None]

        kfac = np.exp( - (r_use[None, :]/self.cutoff)**2) #Extra exponential cutoff
        prof = rho_c/(r_use/r_s * (1 + r_use/r_s)**2) * 1/(1 + (r_use/r_t)**2)**2 * kfac
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)


        return prof


class TwoHalo(SchneiderProfiles):
    '''
    Simple two halo term (uses 2pt corr func, not halo model)
    '''

    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        #Need it to be linear if we're doing two halo term
        assert cosmo._config_init_kwargs['matter_power_spectrum'] == 'linear', "Must use matter_power_spectrum = linear for 2-halo term"

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        R   = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        z = 1/a - 1

        if self.xi_mm is None:
            xi_mm   = ccl.correlation_3d(cosmo, a, r_use)
        else:
            xi_mm   = self.xi_mm(r_use,)

        delta_c = 1.686/ccl.growth_factor(cosmo, a)
        nu_M    = delta_c / ccl.sigmaM(cosmo, M_use, a)
        bias_M  = 1 + (self.q*nu_M**2 - 1)/delta_c + 2*self.p/delta_c/(1 + (self.q*nu_M**2)**self.p)

        bias_M  = bias_M[:, None]
        prof    = (1 + bias_M * xi_mm)*ccl.rho_x(cosmo, a, species = 'matter', is_comoving = True)

        #Need this truncation so the fourier space integral isnt infinity
        kfac = np.exp( - (r_use[None, :]/self.cutoff)**2)
        prof = prof * kfac

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof


class Stars(SchneiderProfiles):
    '''
    Exponential stellar mass profile
    '''
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        #For some reason, we need to make this extreme in order
        #to prevent ringing in the profiles. Haven't figured out
        #why this is the case
        self.precision_fftlog['padding_lo_fftlog'] = 1e-5
        self.precision_fftlog['padding_hi_fftlog'] = 1e5

    
    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R   = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        eta_cga  = self.eta  + self.eta_delta
        beta_cga = self.beta + self.beta_delta
        
        f_cga = self.A * ((M_use/self.M1)**beta_cga + (M_use/self.M1)**eta_cga)**-1

        R_h   = self.epsilon_h * R

        f_cga, R_h = f_cga[:, None], R_h[:, None]

        r_integral = np.geomspace(1e-3, 100, 500)
        rho   = DarkMatter(**self.model_params).real(cosmo, r_integral, M_use, a, mass_def)
        M_tot = np.trapz(4*np.pi*r_integral**2 * rho, r_integral, axis = -1)
        M_tot = np.atleast_1d(M_tot)[:, None]
        
        kfac = np.exp( - (r_use[None, :]/self.cutoff)**2)
        prof = f_cga*M_tot / (4*np.pi**(3/2)*R_h) * 1/r_use**2 * np.exp(-(r_use/2/R_h)**2) * kfac
                
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof


class Gas(SchneiderProfiles):

    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):


        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        u = r_use/(self.theta_co*R)[:, None]
        v = r_use/(self.theta_ej*R)[:, None]

        f_star = self.A * ((M_use/self.M1)**self.beta + (M_use/self.M1)**self.eta)**-1
        f_bar  = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
        f_gas  = f_bar - f_star

        beta   = 3*(M_use/self.M_c)**self.mu / (1 + (M_use/self.M_c)**self.mu)

        f_gas, beta = f_gas[:, None], beta[:, None]

        #Integrate over wider region in radii to get normalization of gas profile
        r_integral = np.geomspace(1e-5, 100, 500)

        u_integral = r_integral/(self.theta_co*R)[:, None]
        v_integral = r_integral/(self.theta_ej*R)[:, None]

        prof_integral  = 1/(1 + u_integral)**beta / (1 + v_integral**self.gamma)**((self.delta - beta)/self.gamma)

        Normalization  = interpolate.CubicSpline(np.log(r_integral), 4 * np.pi * r_integral**3 * prof_integral, axis = -1)
        Normalization  = Normalization.integrate(np.log(r_integral[0]), np.log(r_integral[-1]))
        Normalization  = Normalization[:, None]

        del u_integral, v_integral, prof_integral

        rho   = DarkMatter(**self.model_params).real(cosmo, r_integral, M, a, mass_def)
        M_tot = np.trapz(4*np.pi*r_integral**2 * rho, r_integral, axis = -1)
        M_tot = np.atleast_1d(M_tot)[:, None]
        
        kfac = np.exp( - (r_use[None, :]/self.cutoff)**2)
        prof  = 1/(1 + u)**beta / (1 + v**self.gamma)**((self.delta - beta)/self.gamma) * kfac
        prof *= f_gas*M_tot/Normalization
        

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)


        return prof
    
    
class ShockedGas(Gas):
    '''
    Implements shocked gas profile, assuming a Rankine-Hugonoit conditions.
    To simplify, we assume a high mach-number shock, and so the 
    density is suppressed by a factor of 4.
    '''
    
    def __init__(self, epsilon_shock, width_shock, **kwargs):
        
        self.epsilon_shock = epsilon_shock
        self.width_shock   = width_shock
        
        super().__init__(**kwargs)

    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        #Minimum is 0.25 since a factor of 4x drop is the maximum possible for a shock
        rho_gas = super()._real(cosmo, r, M, a, mass_def)
        g_arg   = 1/self.width_shock*(np.log(r_use) - np.log(self.epsilon_shock*R)[:, None])
        g_arg   = np.where(g_arg > 1e2, np.inf, g_arg) #To prevent overflows when doing exp
        factor  = (1 - 0.25)/(1 + np.exp(g_arg)) + 0.25
        
        #Get the right size for rho_gas
        if M_use.size == 1: rho_gas = rho_gas[None, :]
            
        prof = rho_gas * factor
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        
        return prof


class CollisionlessMatter(SchneiderProfiles):
    
    def __init__(self, gas = None, stars = None, darkmatter = None, max_iter = 10, reltol = 1e-2,**kwargs):
        
        self.Gas   = gas
        self.Stars = stars
        self.DarkMatter = darkmatter
        
        if self.Gas is None: self.Gas = Gas(**kwargs)          
        if self.Stars is None: self.Stars = Stars(**kwargs)
        if self.DarkMatter is None: self.DarkMatter = DarkMatter(**kwargs)
            
        self.max_iter   = max_iter
        self.reltol     = reltol
        
        super().__init__(**kwargs)
        

    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        #Def radius sampling for doing iteration.
        #And don't check iteration near the boundaries, since we can have numerical errors
        #due to the finite width oof the profile during iteration.
        r_integral = np.geomspace(1e-5, 300, 500)
        safe_range = (r_integral > 2 * np.min(r_integral) ) & (r_integral < 1/2 * np.max(r_integral) )
        
        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        eta_cga  = self.eta  + self.eta_delta
        beta_cga = self.beta + self.beta_delta
        
        f_star = self.A * ((M_use/self.M1)**self.beta + (M_use/self.M1)**self.eta)**-1
        f_cga  = self.A * ((M_use/self.M1)**beta_cga  + (M_use/self.M1)**eta_cga)**-1
        f_star = f_star[:, None]
        f_cga  = f_cga[:, None]
        f_sga  = f_star - f_cga
        f_clm  = 1 - cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m + f_sga
        
        
        rho_i      = self.DarkMatter.real(cosmo, r_integral, M, a, mass_def)
        rho_cga    = self.Stars.real(cosmo, r_integral, M, a, mass_def)
        rho_gas    = self.Gas.real(cosmo, r_integral, M, a, mass_def)

        #The ccl profile class removes the dimension of size 1
        #we're adding it back in here in order to keep code general
        if M_use.size == 1:
            rho_i   = rho_i[None, :]
            rho_cga = rho_cga[None, :]
            rho_gas = rho_gas[None, :]
            
        dlnr  = np.log(r_integral[1]) - np.log(r_integral[0])
        M_i   = 4 * np.pi * np.cumsum(r_integral**3 * rho_i   * dlnr, axis = -1)
        M_cga = 4 * np.pi * np.cumsum(r_integral**3 * rho_cga * dlnr, axis = -1)
        M_gas = 4 * np.pi * np.cumsum(r_integral**3 * rho_gas * dlnr, axis = -1)
        
        ln_M_NFW = [interpolate.CubicSpline(np.log(r_integral), np.log(M_i[m_i]), axis = -1) for m_i in range(M_i.shape[0])]
        ln_M_cga = [interpolate.CubicSpline(np.log(r_integral), np.log(M_cga[m_i]), axis = -1) for m_i in range(M_i.shape[0])]
        ln_M_gas = [interpolate.CubicSpline(np.log(r_integral), np.log(M_gas[m_i]), axis = -1) for m_i in range(M_i.shape[0])]

        del M_cga, M_gas, rho_i, rho_cga, rho_gas

        relaxation_fraction = np.ones_like(M_i)

        for m_i in range(M_i.shape[0]):
            
            counter  = 0
            max_rel_diff = np.inf #Initializing variable at infinity
            
            while max_rel_diff > self.reltol:

                r_f  = r_integral*relaxation_fraction[m_i]
                M_f  = f_clm[m_i]*M_i[m_i] + np.exp(ln_M_cga[m_i](np.log(r_f))) + np.exp(ln_M_gas[m_i](np.log(r_f)))

                relaxation_fraction_new = self.a*((M_i[m_i]/M_f)**self.n - 1) + 1

                diff     = relaxation_fraction_new/relaxation_fraction[m_i] - 1
                abs_diff = np.abs(diff)
                
                max_rel_diff = np.max(abs_diff[safe_range])
                
                relaxation_fraction[m_i] = relaxation_fraction_new

                counter += 1

                #Though we do a while loop, we break it off after 10 tries
                #this seems to work well enough. The loop converges
                #after two or three iterations.
                if (counter >= self.max_iter) & (max_rel_diff > self.reltol): 
                    
                    med_rel_diff = np.max(abs_diff[safe_range])
                    warn_text = ("Profile of halo index %d did not converge after %d tries." % (m_i, counter) +
                                 "Max_diff = %0.5f, Median_diff = %0.5f. Try increasing max_iter." % (max_rel_diff, med_rel_diff)
                                )
                    
                    warnings.warn(warn_text, UserWarning)
                    break

        ln_M_clm = np.vstack([np.log(f_clm[m_i]) + 
                              ln_M_NFW[m_i](np.log(r_integral/relaxation_fraction[m_i])) for m_i in range(M_i.shape[0])])
        ln_M_clm = interpolate.CubicSpline(np.log(r_integral), ln_M_clm, axis = -1, extrapolate = False)
        log_der  = ln_M_clm.derivative(nu = 1)(np.log(r_use))
        lin_der  = log_der * np.exp(ln_M_clm(np.log(r_use))) / r_use
        prof     = 1/(4*np.pi*r_use**2) * lin_der
        
        kfac = np.exp( - (r_use[None, :]/self.cutoff)**2)
        prof = np.where(np.isnan(prof), 0, prof) * kfac

        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof


class DarkMatterOnly(SchneiderProfiles):

    def __init__(self, darkmatter = None, twohalo = None, **kwargs):
        
        self.DarkMatter = darkmatter
        self.TwoHalo    = twohalo
        
        if self.TwoHalo is None: self.TwoHalo = TwoHalo(**kwargs)
        if self.DarkMatter is None: self.DarkMatter = DarkMatter(**kwargs)
            
        super().__init__(**kwargs)
        
    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        prof = (self.DarkMatter.real(cosmo, r, M, a, mass_def) +
                self.TwoHalo.real(cosmo, r, M, a, mass_def)
               )

        return prof


class DarkMatterBaryon(SchneiderProfiles):

    def __init__(self, gas = None, stars = None, collisionlessmatter = None, darkmatter = None, twohalo = None, **kwargs):
        
        self.Gas   = gas
        self.Stars = stars
        self.TwoHalo    = twohalo
        self.DarkMatter = darkmatter
        self.CollisionlessMatter = collisionlessmatter
        
        if self.Gas is None: self.Gas = Gas(**kwargs)          
        if self.Stars is None: self.Stars = Stars(**kwargs)
        if self.TwoHalo is None: self.TwoHalo = TwoHalo(**kwargs)
        if self.DarkMatter is None: self.DarkMatter = DarkMatter(**kwargs)
        if self.CollisionlessMatter is None: self.CollisionlessMatter = CollisionlessMatter(**kwargs)
            
        super().__init__(**kwargs)
        
    def _real(self, cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        R = mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

        #Need DMO for normalization
        #Makes sure that M_DMO(<r) = M_DMB(<r) for the limit r --> infinity
        #This is just for the onehalo term
        r_integral = np.geomspace(1e-5, 100, 500)

        rho   = self.DarkMatter.real(cosmo, r_integral, M, a, mass_def)
        M_tot = np.trapz(4*np.pi*r_integral**2 * rho, r_integral)

        rho   = (self.CollisionlessMatter.real(cosmo, r_integral, M, a, mass_def) +
                 self.Stars.real(cosmo, r_integral, M, a, mass_def) +
                 self.Gas.real(cosmo, r_integral, M, a, mass_def))

        M_tot_dmb = np.trapz(4*np.pi*r_integral**2 * rho, r_integral, axis = -1)

        Factor = M_tot/M_tot_dmb
        
        if np.ndim(Factor) == 1:
            Factor = Factor[:, None]

        prof = (self.CollisionlessMatter.real(cosmo, r, M, a, mass_def) * Factor +
                self.Stars.real(cosmo, r, M, a, mass_def) * Factor +
                self.Gas.real(cosmo, r, M, a, mass_def) * Factor +
                self.TwoHalo.real(cosmo, r, M, a, mass_def))

        return prof
