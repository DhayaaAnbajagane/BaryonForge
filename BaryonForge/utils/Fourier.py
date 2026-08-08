import numpy as np
import pyccl as ccl
import healpy as hp

class BaseKernel:

    def __init__(self, profile1, profile2, HaloModelCalculator):
        
        self.profile1 = profile1
        self.profile2 = profile2
        self.HMCalc   = HaloModelCalculator
    
    def interpolated(self, cosmo, k, a, kinterp):

        # assert kinterp[0]  <= np.min(k), "The interpolation array is narrower than the target array"
        # assert kinterp[-1] >= np.max(k), "The interpolation array is narrower than the target array"

        x    = np.log(kinterp)
        y    = np.log(self.process(cosmo, kinterp, a))

        yout = np.exp(np.interp(np.log(k), x, y))

        return yout
    
    def __call__(self, cosmo, k, a):
        return self.process(cosmo, k, a)
    

class Kernel3D(BaseKernel):

    def process(self, cosmo, k, a):

        P11 = ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, self.HMCalc, k, a, self.profile1, get_1h = False, get_2h = True)
        P22 = ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, self.HMCalc, k, a, self.profile2, get_1h = False, get_2h = True)

        return np.sqrt(P11/P22)
    

class Kernel2D(BaseKernel):

    def __init__(self, profile1, profile2, HaloModelCalculator, proj_cutoff, padding_lo_proj, padding_hi_proj, n_per_decade_proj,):
        
        self.proj_cutoff        = proj_cutoff
        self.padding_lo_proj    = padding_lo_proj
        self.padding_hi_proj    = padding_hi_proj
        self.n_per_decade_proj  = n_per_decade_proj

        BaseKernel.__init__(self, profile1, profile2, HaloModelCalculator)


    def process(self, cosmo, k, a):

        kmn = np.min(k) * self.padding_lo_proj
        kmx = np.max(k) * self.padding_hi_proj
        Nk  = int(np.log10(kmx/kmn)) * self.n_per_decade_proj
        kin = np.geomspace(kmn, kmx, Nk)

        P11 = ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, self.HMCalc, kin, a, self.profile1, get_1h = False, get_2h = True)
        P22 = ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, self.HMCalc, kin, a, self.profile2, get_1h = False, get_2h = True)
        
        P11_proj = np.zeros(k.size)
        P22_proj = np.zeros(k.size)

        #This nested loop saves on memory, and vectorizing the calculation doesn't really
        #speed things up, so better to keep the loop this way.
        for j in range(k.size):

            Wprojkernel = self.proj_cutoff * np.sinc(kin * self.proj_cutoff)
            knew        = np.sqrt(kin**2 + k[j]**2)
            P11_proj[j] = 2*np.trapz(np.interp(knew, kin, P11 * Wprojkernel**2, left = 0, right = 0), knew)
            P22_proj[j] = 2*np.trapz(np.interp(knew, kin, P22 * Wprojkernel**2, left = 0, right = 0), knew)

        return np.sqrt(P11_proj/P22_proj)


class KernelHarmonic(BaseKernel):

    def __init__(self, profile1, profile2, HaloModelCalculator, z_min, z_max, Nz = 10):
        
        self.z_min              = z_min
        self.z_max              = z_max
        self.Nz                 = Nz

        assert self.z_max > self.z_min, "z_max must be larger than z_min"

        BaseKernel.__init__(self, profile1, profile2, HaloModelCalculator)


    def _power3D(self, cosmo, k, a):

        P11 = ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, self.HMCalc, k, a, self.profile1, get_1h = False, get_2h = True)
        P22 = ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, self.HMCalc, k, a, self.profile2, get_1h = False, get_2h = True)

        return P11, P22

    def process(self, cosmo, ell, a = None):

        ell     = np.atleast_1d(ell)
        z_edges = np.linspace(self.z_min, self.z_max, self.Nz + 1)
        z       = 0.5 * (z_edges[1:] + z_edges[:-1])
        dz      = z_edges[1] - z_edges[0]
        a       = 1/(1 + z)

        chi       = ccl.comoving_radial_distance(cosmo, a)
        chi_edges = ccl.comoving_radial_distance(cosmo, 1/(1 + z_edges))
        dchi_dz   = np.diff(chi_edges) / dz

        #Normalized top-hat density kernel over z_min < z < z_max.
        #Its overall normalization cancels in the transfer-function ratio.
        nz  = np.ones_like(z) / (self.z_max - self.z_min)
        C11 = np.zeros(ell.size)
        C22 = np.zeros(ell.size)

        for i in range(z.size):

            k = (ell + 0.5) / chi[i]

            P11, P22 = self._power3D(cosmo, k, a[i])
            weight = nz[i]**2 / (chi[i]**2 * dchi_dz[i])
            C11   += weight * P11 * dz
            C22   += weight * P22 * dz

        kernel = np.sqrt(np.divide(C11, C22, out = np.ones_like(C11), where = C22 > 0))

        return kernel



class BaseFilter:

    def __init__(self, Filter):

        self.Filter = Filter

    def process3D(self, cosmo, Map, Nk_interp = 100):
        
        res = Map.res
        Min = Map.data
        a   = 1/(1 + Map.redshift)

        klin   = np.fft.fftfreq(Min.shape[0], res / (2*np.pi))
        k      = np.sqrt(klin[:, None, None]**2 + klin[None, None, :]**2 + klin[None, :, None]**2).flatten()
        kinter = np.geomspace(np.min(k[k > 0]), np.max(k), Nk_interp)
        kernel = self.Filter.interpolated(cosmo, k, a, kinter).reshape(Min.shape)

        MapK   = np.fft.ifftn(np.fft.fftn(Min) * kernel).real

        return MapK


    def process2D(self, cosmo, Map, Nk_interp = 100):
        
        res = Map.res
        Min = Map.data
        a   = 1/(1 + Map.redshift)

        klin   = np.fft.fftfreq(Min.shape[0], res / (2*np.pi))
        k      = np.sqrt(klin[:, None]**2 + klin[None, :]**2).flatten()
        kinter = np.geomspace(np.min(k[k > 0]), np.max(k), Nk_interp)
        kernel = self.Filter.interpolated(cosmo, k, a, kinter).reshape(Min.shape)

        MapK   = np.fft.ifftn(np.fft.fftn(Min) * kernel).real

        return MapK


    def processHarmonic(self, cosmo, Map, Nk_interp = 100):

        Min   = Map.data
        nside = hp.get_nside(Min)
        lmax  = 3*nside - 1

        ell        = np.arange(1, lmax + 1)
        ellinter   = np.geomspace(1, lmax, min(Nk_interp, lmax))
        kernel     = np.ones(lmax + 1)
        kernel[1:] = self.Filter.interpolated(cosmo, ell, None, ellinter)

        alm  = hp.map2alm(Min, lmax = lmax)
        alm  = hp.almxfl(alm, kernel)
        MapK = hp.alm2map(alm, nside, lmax = lmax)

        return MapK