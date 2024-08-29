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


class HaloProfileCIB(HaloProfile):
    """ 
    This is the superclass of CIB halo profiles based on the format given by the PYCCL package.
    """
    name = 'CIB'
    _one_over_4pi = 0.07957747154

    def __init__(self, c_M_relation,alpha=0.36, T0=24.4, beta=1.75,
                 gamma=1.7, s_z=3.6, log10meff=12.6, lMmin_sub=10,
                 L0=6.4E-8, lMmin_0=12., lMmin_p=0., siglM_0=0.4,
                 siglM_p=0., lM0_0=7., lM0_p=0.,
                 lM1_0=13.3, lM1_p=0., alpha_0=1.,
                 alpha_p=0., a_pivot=1.,sed=None, z_sed=None, Om0=0.2589+0.0486,
                Ob0=0.0486,Ode0=1-(0.2589+0.0486),eta_max=0.42,
                lMpeak=12.94,lMpeak_p=0,sigma_M0=1.75,tau=0.9,z_c=1.5):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")

        self.alpha = alpha
        self.T0 = T0
        self.beta = beta
        self.gamma = gamma
        self.s_z = s_z
        self.l10meff = log10meff
        self.Mmin = 10**lMmin_sub
        self.L0 = L0
        self.pNFW = HaloProfileNFW(c_M_relation)
        self.lMmin_0 = lMmin_0
        self.lMmin_p = lMmin_p
        self.lM0_0 = lM0_0
        self.lM0_p = lM0_p
        self.lM1_0 = lM1_0
        self.lM1_p = lM1_p
        self.siglM_0 = siglM_0
        self.siglM_p = siglM_p
        self.alpha_0 = alpha_0
        self.alpha_p = alpha_p
        self.a_pivot = a_pivot
        self.sed = sed
        self.z_sed = z_sed
        self.eta_max = eta_max
        self.lMpeak = lMpeak
        self.lMpeak_p = lMpeak_p
        self.sigma_M0 = sigma_M0
        self.tau = tau
        self.z_c = z_c
        self.fsub = 0.0
        super(HaloProfileCIB, self).__init__()

    def dNsub_dlnM_TinkerWetzel10(self, Msub, Mparent):
        """Subhalo mass function of `Tinker & Wetzel 2010
        <https://arxiv.org/abs/0909.1325>`_. Number of subhalos
        per (natural) logarithmic interval of mass.

        Args:
            Msub (:obj:`float` or `array`): sub-halo mass (in solar masses).
            Mparent (:obj:`float`): parent halo mass (in solar masses).

        Returns:
            (:obj:`float` or `array`): average number of subhalos.
        """
        return 0.30*(Msub/Mparent)**(-0.7)*np.exp(-9.9*(Msub/Mparent)**2.5)

    def update_parameters(self,
                          alpha=None, T0=None, beta=None, gamma=None,
                          s_z=None, log10meff=None, sigLM=None,
                          lMmin_sub=None, L0=None, 
                          lMmin_0=None, lMmin_p=None,
                          siglM_0=None, siglM_p=None,
                          lM0_0=None, lM0_p=None,
                          lM1_0=None, lM1_p=None,
                          alpha_0=None, alpha_p=None,
                          eta_max=None,
                          lMpeak=None,lMpeak_p=None,sigma_M0=None,tau=None,z_c=None):
        """ Update any of the parameters associated with
        this profile. Any parameter set to `None` won't be updated.
        Args:
            nu_GHz (float): frequency in GHz.
            alpha (float): dust temperature evolution parameter.
            T0 (float): dust temperature at :math:`z=0` in Kelvin.
            beta (float): dust spectral index.
            gamma (float): high frequency slope.
            s_z (float): luminosity evolution slope.
            log10meff (float): log10 of the most efficient mass.
            sigLM (float): logarithmic scatter in mass.
            Mmin (float): minimum subhalo mass.
            L0 (float): luminosity scale (in
                :math:`{\\rm Jy}\\,{\\rm Mpc}^2\\,M_\\odot^{-1}`).
        """

        if alpha is not None:
            self.alpha = alpha
        if T0 is not None:
            self.T0 = T0
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if s_z is not None:
            self.s_z = s_z
        if log10meff is not None:
            self.l10meff = log10meff
        if sigLM is not None:
            self.sigLM = sigLM
        if lMmin_sub is not None:
            #print(np.float64(lMmin_sub))
            self.Mmin = 10**np.float64(lMmin_sub)
        if L0 is not None:
            self.L0 = L0    
        if lMmin_0 is not None:
            self.lMmin_0 = lMmin_0
        if lMmin_p is not None:
            self.lMmin_p = lMmin_p
        if lM0_0 is not None:
            self.lM0_0 = lM0_0
        if lM0_p is not None:
            self.lM0_p = lM0_p
        if lM1_0 is not None:
            self.lM1_0 = lM1_0
        if lM1_p is not None:
            self.lM1_p = lM1_p
        if siglM_0 is not None:
            self.siglM_0 = siglM_0
        if siglM_p is not None:
            self.siglM_p = siglM_p
        if alpha_0 is not None:
            self.alpha_0 = alpha_0
        if alpha_p is not None:
            self.alpha_p = alpha_p
        if eta_max is not None:
            self.eta_max = eta_max
        if lMpeak is not None:
            self.lMpeak = lMpeak
        if lMpeak_p is not None:
            self.lMpeak_p = lMpeak_p
        if sigma_M0 is not None:
            self.sigma_M0 = sigma_M0
        if tau is not None:
            self.tau = tau
        if z_c is not None:
            self.z_c = z_c

    def BAR(self, M, a):
        return (M / 1.0e12)**1.1
    
    def eta(self, M, a):        
        return 
    
    def _sfr(self, M, a):
        return
    
    def _sfrcen(self, M, a):
        fsub = self.fsub
        M_ = M * (1-fsub)
        return self._Nc(M, a)*self._sfr(M_, a) 
    
    def _sfrsat(self, M, a, nmsub=40):
        fsub = self.fsub
        
        Mmin = 1.e5 #self.Mmin
        M_ = M * (1-fsub)
        Ms = np.ones((M.size, nmsub)) * M[:,None] 
        msubs = np.geomspace(Mmin, M, nmsub).T  # integrated variables. If M<Mmin, this generates a decreasing array
        sfr_sub = np.zeros((M.size, msubs.size))
        
        sfr_sub1 = self._sfr(msubs, a)
        sfr_sub2 = self._sfr(M_, a)[:, None] * (msubs.T / M_).T
        
        #sfr_sub = self._sfr(msubs, a).reshape(-1)#np.minimum(sfr_sub1, sfr_sub2).reshape(-1)
        sfr_sub = np.minimum(sfr_sub1, sfr_sub2).reshape(-1)
        nmsubs = (self.dNsub_dlnM_TinkerWetzel10(msubs.reshape(-1), Ms.reshape(-1))*sfr_sub).reshape(M.size, nmsub)
        sfrsat  = simps(nmsubs, np.log(msubs), axis=1)
        sfrsat[sfrsat<0] = 0 # if M<Mmin, the decreasing integrated variable will yield a negative Lumsat and we can just change them to be zero.
        return sfrsat
    

    def _Lumcen(self, M, a):
        #Lum = self._sfrcen(M, a)
        #Lumcen = self._Nc(M, a)*Lum
        return self._sfrcen(M, a)
    
    
    def _Lumsat(self, M, a):
        Lum = self._sfrsat(M, a)
        #Lumcen = self._Ns(M, a)*Lum
        return Lum
    
    def _real(self, cosmo, r, M, a, mass_def):
        M_use = np.atleast_1d(M)
        r_use = np.atleast_1d(r)

        # (redshifted) Frequency dependence

        Ls = self._Lumsat(M_use, a)
        ur = self.pNFW._real(cosmo, r_use, M_use,
                             a, mass_def)/M_use[:, None]
        prof = Ls[:, None]*ur

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        # (redshifted) Frequency dependence

        Lc = self._Lumcen(M_use, a)
        Ls = self._Lumsat(M_use, a)
        uk = self.pNFW._fourier(cosmo, k_use, M_use,
                                a, mass_def)/M_use[:, None]

        prof = (Lc[:, None]+Ls[:, None]*uk)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_variance(self, cosmo, k, M, a, mass_def, nu_other=None):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        Lc = self._Lumcen(M_use, a)
        Ls = self._Lumsat(M_use, a)
        uk = self.pNFW._fourier(cosmo, k_use, M_use,
                                a, mass_def)/M_use[:, None]

        prof = (Ls * Lc * 2)[:, None] * uk + (Ls * Ls)[:, None] * uk ** 2

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
    
    def _fourier_variance_withhod(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        Lc = self._Lumcen(M_use, a)
        Ls = self._Lumsat(M_use, a)
        uk = self.pNFW._fourier(cosmo, k_use, M_use,
                                a, mass_def)/M_use[:, None]
        
        Nc = self._Nc(M_use, a) 
        Ns = self._Ns(M_use, a) * Nc

        prof = (Ls * Nc + Lc * Ns)[:, None] * uk + (Ls * Ns)[:, None] * uk ** 2
        #prof *= spec_nu1*self._one_over_4pi

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
    
    def _fourier_variance_poisson(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        Lc = self._Lumcen(M_use, a)
        Ls = self._Lumsat(M_use, a)
        
        Nc = self._Nc(M_use, a) 
        Ns = self._Ns(M_use, a) * Nc

        prof = (Ls)[:, None] * (Ns)[:, None] + (Lc)[:, None] * (Nc)[:, None]
        #prof *= spec_nu1*self._one_over_4pi

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
    
    def _Nc(self, M, a):
        # Number of centrals

        Mmin = 10.**(self.lMmin_0 + self.lMmin_p * (a - self.a_pivot))
        siglM = self.siglM_0 + self.siglM_p * (a - self.a_pivot)
        return 0.5 * (1 + erf(np.log(M/Mmin)/siglM)) 

    def _Ns(self, M, a):
        # Number of satellites
        M0 = 10.**(self.lM0_0 + self.lM0_p * (a - self.a_pivot))
        M1 = 10.**(self.lM1_0 + self.lM1_p * (a - self.a_pivot))
        alpha = self.alpha_0 + self.alpha_p * (a - self.a_pivot)
        return np.heaviside(M-M0, 1) * (np.fabs(M-M0) / M1)**alpha 
    

class Profile2ptCIB(Profile2pt):
    """ This class implements the Fourier-space 1-halo 2-point
    correlator for the CIB profile. It follows closely the
    implementation of the equivalent HOD quantity
    (see :class:`~pyccl.halos.profiles_2pt.Profile2ptHOD`
    and Eq. 15 of McCarthy & Madhavacheril (2021PhRvD.103j3515M)).
    """
    def fourier_2pt(self, prof, cosmo, k, M, a,
                    prof2=None, mass_def=None):
        """ Returns the Fourier-space two-point moment for the CIB
        profile.
        Args:
            prof (:class:`HaloProfileCIBShang12`):
                halo profile for which the second-order moment
                is desired.
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            k (float or array_like): comoving wavenumber in Mpc^-1.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            prof2 (:class:`HaloProfileCIBShang12`):
                second halo profile for which the second-order moment
                is desired. If `None`, the assumption is that you want
                an auto-correlation. Note that only auto-correlations
                are allowed in this case.
            mass_def (:obj:`~pyccl.halos.massdef.MassDef`): a mass
                definition object.
        Returns:
            float or array_like: second-order Fourier-space
            moment. The shape of the output will be `(N_M, N_k)`
            where `N_k` and `N_m` are the sizes of `k` and `M`
            respectively. If `k` or `M` are scalars, the
            corresponding dimension will be squeezed out on output.
        """
        return prof._fourier_variance(cosmo, k, M, a, mass_def)
    
    
class HaloProfileCIBM21(HaloProfileCIB):
    """ 
    This is the subclass of CIB halo profiles for the  M21 model.
    """
    name = 'CIBM21'

    def __init__(self, c_M_relation):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")

        super(HaloProfileCIBM21, self).__init__(c_M_relation=c_M_relation)
    
    def eta(self, M, a):
        lM = np.log(M)
        z_c = self.z_c
        z = 1./a - 1
        eta_max = self.eta_max        
        lMpeak_ = self.lMpeak + self.lMpeak_p  * (self.a_pivot-a)
        lMpeak = np.log(10**lMpeak_)
        
        sigma_M0 = self.sigma_M0
        tau = self.tau
        
        sigma_M = sigma_M0*(1 - tau / z_c * np.maximum(0, z_c-z))#np.heaviside(z_c-z, 0) * (z_c-z)
        #if sigma_M<0:
        #    return np.ones_like(lM) * 1e20
        #sigma_M = sigma_M0 - tau*np.heaviside(z_c-z, 0) * (z_c-z)
        #print((lM-lMpeak)**2/2/sigma_M**2)
        def eta_below(xlM):
            return eta_max*np.exp(-(xlM-lMpeak)**2/(2*sigma_M0**2))
        def eta_above(xlM):
            return eta_max*np.exp(-(xlM-lMpeak)**2/(2*sigma_M**2))
        
        return np.piecewise(lM, [lM <= lMpeak, lM > lMpeak], [eta_below, eta_above])
    
    def _sfr(self, M, a):
        return self.eta(M, a) * self.BAR(M, a)
    
class HaloProfileCIBY23(HaloProfileCIB):
    """ 
    This is the subclass of CIB halo profiles for the  Y23 model.
    """
    name = 'CIBY23'

    def __init__(self, c_M_relation,):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")

        super(HaloProfileCIBY23, self).__init__(c_M_relation)
    
    def eta(self, M, a):
        lM = np.log(M)
        z_c = self.z_c
        z = 1./a - 1
        eta_max = 1
        lMpeak_ = self.lMpeak + self.lMpeak_p  * (1-a)
        lMpeak = np.log(10**lMpeak_)
        
        sigma_M0 = self.sigma_M0
        tau = self.tau
        
        sigma_M = sigma_M0*(1 - tau / z_c * np.maximum(0, z_c-z))#np.heaviside(z_c-z, 0) * (z_c-z)
        #if sigma_M<0:
        #    return np.ones_like(lM) * 1e20
        #sigma_M = sigma_M0 - tau*np.heaviside(z_c-z, 0) * (z_c-z)
        #print((lM-lMpeak)**2/2/sigma_M**2)
        def eta_below(xlM):
            return eta_max*np.exp(-(xlM-lMpeak)**2/(2*sigma_M0**2))
        def eta_above(xlM):
            return eta_max*np.exp(-(xlM-lMpeak)**2/(2*sigma_M**2))
        
        return np.piecewise(lM, [lM <= lMpeak, lM > lMpeak], [eta_below, eta_above])
    
    def _sfr(self, M, a):
        return self.eta(M, a) * self.BAR(M, a)
    
    def _Lumcen(self, M, a):
        #Lum = self._sfrcen(M, a)
        #Lumcen = self._Nc(M, a)*Lum
        return self._sfrcen(M, a)

class HaloProfileCIBS12(HaloProfileCIB):
    """ 
    This is the subclass of CIB halo profiles for the S12 model.
    """
    name = 'CIBS12'

    def __init__(self, c_M_relation):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")

        super(HaloProfileCIBS12, self).__init__(c_M_relation)
    
    def eta(self, M, a):
        lM = np.log(M)
        lMpeak_ = self.lMpeak + self.lMpeak_p  * (1-a)
        lMpeak = np.log(10**lMpeak_)
        
        sigma_M0 = self.sigma_M0

        return M / sigma_M0 * np.exp(-(lM-lMpeak)**2/(2*sigma_M0**2))
    
    def _sfr(self, M, a):
        return self.eta(M, a)
    
class Profile2ptCIB_HOD(Profile2pt):
    """ This class implements the Fourier-space 1-halo 2-point
    correlator for the CIB profile. It follows closely the
    implementation of the equivalent HOD quantity
    (see :class:`~pyccl.halos.profiles_2pt.Profile2ptHOD`
    and Eq. 15 of McCarthy & Madhavacheril (2021PhRvD.103j3515M)).
    """
    def fourier_2pt(self, prof, cosmo, k, M, a,
                    prof2=None, mass_def=None):
        try:
            return prof._fourier_variance_withhod(cosmo, k, M, a, mass_def)
        except:
            return prof2._fourier_variance_withhod(cosmo, k, M, a, mass_def)
    
class Profile2ptCIB_poisson(Profile2pt):
    """ This class implements the Fourier-space 1-halo 2-point
    correlator for the CIB profile. It follows closely the
    implementation of the equivalent HOD quantity
    (see :class:`~pyccl.halos.profiles_2pt.Profile2ptHOD`
    and Eq. 15 of McCarthy & Madhavacheril (2021PhRvD.103j3515M)).
    """
    def fourier_2pt(self, prof, cosmo, k, M, a,
                    prof2=None, mass_def=None):

        return prof._fourier_variance_poisson(cosmo, k, M, a, mass_def)
    
    
