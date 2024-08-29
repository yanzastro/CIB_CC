from pyccl.halos import HaloProfile
import pyccl.halos
from pyccl.halos import Profile2pt, Concentration, HaloProfileNFW
from scipy.special import sici, erf
from astropy.modeling import models
from astropy import units as u
from astropy import constants
from astropy.io import fits
from scipy.integrate import simps
from scipy.interpolate import interp1d
import pyccl as ccl
import numpy as np
from scipy.integrate import simps
from scipy.special import lambertw
from scipy.integrate import simpson


class cib_tracer:
    def __init__(self, cosmo, cib_zsample):
        '''
        This super class contains function needed to generating CCL tracer for CIB given different models.
        '''
        self.cosmo = cosmo
        self.cc = [1.097, 1.068, 0.995]
        self.cib_zsample = cib_zsample
        self.cib_asample = 1/(1+cib_zsample)
        self.cib_chisample = ccl.comoving_radial_distance(cosmo, 1/(1+cib_zsample))
        self.freq_dict = {353: 0, 545: 1, 857: 2}

    def get_snu(self, freq):
        return 
    
    def get_ckernel_pref(self):
        return 
    
    def get_cib_kernel(self, freq):
        ckernel_pref = self.get_ckernel_pref()
        cib_kernel = cib_kernel_pref * self.get_snu(freq) * self.cc[self.freq_dict[freq]]
        cib_tracer = ccl.Tracer()
        cib_tracer.add_tracer(cosmo, kernel=(cib_chisample, cib_kernels))
        return cib_tracer


class cib_tracer_Y23(cib_tracer):
    def __init__(self, cosmo, cib_zsample, L0, T0, alpha, beta, gamma):
        '''
        This class defines the CIB kernel for the Y23 model.
        '''
        super().__init__(cosmo, cib_zsample)
        self.L0 = L0
        self.T0 = T0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def get_snu(self, freq):
        '''
        Greybody effective SED for the dust.
        '''
        # h*nu_GHZ / k_B / Td_K
        a = self.cib_asample
        h_GHz_o_kB_K = 0.0479924466
        nu = freq / a
        Td = self.T0 / a ** self.alpha
        x = h_GHz_o_kB_K * nu / Td

        # Find nu_0
        q = self.beta+3+self.gamma
        x0 = q+np.real(lambertw(-q*np.exp(-q), k=0))
    
        def mBB(x):
            ex = np.exp(x)
            return x**(3+beta)/(ex-1)

        mBB0 = mBB(x0)

        def plaw(x):
            return mBB0*(x0/x)**gamma
        return np.piecewise(x, [x <= x0], [mBB, plaw])/mBB0
    
    def get_ckernel_pref(self, freq):
        f = self.freq_dict[freq]
        zs = self.cib_zsample
        Omm = self.cosmo['Omega_c'] + self.cosmo['Omega_b']
        Omb = self.cosmo['Omega_b']
        ckernel_pref = (1+1.11*zs) * np.sqrt(Omm*(1+zs)**3+(1-Omm))*Omb/Omm * self.L0 * 1/(1+zs)
        return ckernel_pref

        
class cib_tracer_S12(cib_tracer):
    def __init__(self, cosmo, cib_zsample, L0, T0, alpha, beta, gamma, delta):
        super().__init__(cosmo, cib_zsample)
        self.L0 = L0
        self.T0 = T0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
    def get_snu(self, freq):
        '''
        Greybody effective SED for the dust.
        '''
        # h*nu_GHZ / k_B / Td_K
        a = self.cib_asample
        h_GHz_o_kB_K = 0.0479924466
        nu = freq / a
        Td = self.T0 / a ** self.alpha
        x = h_GHz_o_kB_K * nu / Td

        # Find nu_0
        q = self.beta+3+self.gamma
        x0 = q+np.real(lambertw(-q*np.exp(-q), k=0))
    
        def mBB(x):
            ex = np.exp(x)
            return x**(3+beta)/(ex-1)

        mBB0 = mBB(x0)

        def plaw(x):
            return mBB0*(x0/x)**gamma
        return np.piecewise(x, [x <= x0], [mBB, plaw])/mBB0    
    
    def get_ckernel_pref(self, freq):
        f = self.freq_dict[freq]
        sed = self.get_snu(freq)
        prefactor = self.L0 * self.cib_asample * self.cib_asample ** (-self.delta)
        return prefactor
    
class cib_tracer_M21(cib_tracer):
    def __init__(self, cosmo, cib_zsample, snu_file):
        
        super().__init__(cosmo, cib_zsample)
        hdulist = fits.open(snu_file)
        self.snu_redshifts = hdulist[1].data 
        self.snu_eff_pl = hdulist[0].data  # in Jy/Lsun
        self.K = 1.0e-10
        
    def get_snu(self, freq):
        f = self.freq_dict[freq]
        snu_func = interp1d(self.snu_redshifts, self.snu_eff_pl[3+f], fill_value='extrapolate')
        return snu_func(self.cib_zsample)
    
    def get_ckernel_pref(self):
        f = self.freq_dict[freq]
        zs = self.cib_zsample
        Omm = self.cosmo['Omega_c'] + self.cosmo['Omega_b']
        Omb = self.cosmo['Omega_b']
        ckernel_pref = self.cib_chisample ** 2 / self.K * 46.1 * (1+1.11*self.cib_zsample) * np.sqrt(Omm*(1+self.cib_zsample)**3+(1-Omm))*Omb/Omm        
        return ckernel_pref
