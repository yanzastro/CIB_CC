import gc
import healpy as hp
import numpy as np
import os
import pymaster as nmt

class covmaster:
    def __init__(self, maps2cls_obj, nside=None):
        
        '''
        This is a class to calculate Gaussian covariance matrix for pseudo-Cl.
        The input is a maps2cls object.
        
        '''
        self.field_names = maps2cls_obj.field_names
        self.name_pairs = maps2cls_obj.name_pairs
        self.masks = maps2cls_obj.masks
        self.maps = maps2cls_obj.maps
        self.spins = maps2cls_obj.field_spins
        self.fields = maps2cls_obj.field
        self.field_res = maps2cls_obj.field_res
        self.wsp = maps2cls_obj.wsp

    def coupled_cell(self, fn_1, fn_2):
        f_1 = self.fields[fn_1]
        f_2 = self.fields[fn_2]
        mask1 = f_1.get_mask()
        mask2 = f_2.get_mask()
        nside = hp.npix2nside(mask2.size)
        fsky = np.mean(mask1*mask2)
        if self.spins[fn_1] + self.spins[fn_2] > 0:
            cl = nmt.compute_coupled_cell(f_1, f_2)[0]
            lmax = cl.size
            cl /= self.get_beam(fn_1, fn_2, lmax, nside)
            return cl.reshape(1,cl.size) / fsky
        else:
            cl = nmt.compute_coupled_cell(f_1, f_2)[0]
            lmax = cl.size
            cl /= self.get_beam(fn_1, fn_2, lmax, nside)
            return cl / fsky

    def get_beam(self, fn_1, fn_2, lmax, nside):
        res_1 = self.field_res[fn_1]
        res_2 = self.field_res[fn_2]
        beam_1 = hp.gauss_beam(np.deg2rad(res_1/60.), lmax=lmax-1)
        beam_2 = hp.gauss_beam(np.deg2rad(res_2/60.), lmax=lmax-1)
        beam_pix = hp.pixwin(nside, lmax=lmax-1)
        return beam_1 * beam_2 * beam_pix ** 2

        
    def covmaster(self, b, name_pairs=None):
        if name_pairs is None:
            name_pairs = self.name_pairs
        else:
            pass
        nb = b.get_n_bands()
        fullcov = np.zeros((len(name_pairs)*nb, len(name_pairs)*nb))
        print('Calculating the full covariance matrix...')
        self.blockcov = {}
        self.fullcov = np.zeros((len(name_pairs)*nb, len(name_pairs)*nb))
        for ia, namepair_a in enumerate(name_pairs):
            wa = self.wsp[namepair_a[0]+'x'+namepair_a[1]]
            n_a1 = namepair_a[0]
            n_a2 = namepair_a[1]
            fa1 = self.fields[n_a1]  # fields to calculate coupled C_ell
            fa2 = self.fields[n_a2]
            for ib, namepair_b in enumerate(name_pairs):
                
                if ib > ia:
                    continue
                
                wb = self.wsp[namepair_b[0]+'x'+namepair_b[1]]
                n_b1 = namepair_b[0]
                n_b2 = namepair_b[1]
                print('Calculating the covariance matrix of ('+n_a1+'x'+n_a2+', '+n_b1+'x'+n_b2+')...')
                fb1 = self.fields[n_b1]  # fields to calculate coupled C_ell
                fb2 = self.fields[n_b2]
               
                cw = nmt.NmtCovarianceWorkspace()
                cw.compute_coupling_coefficients(fa1, fa2, fb1, fb2)
                
                cl_a1b1 = self.coupled_cell(n_a1, n_b1)
                cl_a1b2 = self.coupled_cell(n_a1, n_b2)
                cl_a2b1 = self.coupled_cell(n_a2, n_b1)
                cl_a2b2 = self.coupled_cell(n_a2, n_b2)
                
                block = nmt.gaussian_covariance(cw,
                                      0, 0, 0, 0,  # Spins of the 4 fields
                                      [cl_a1b1],  # TT
                                      [cl_a1b2],  # TT
                                      [cl_a2b1],  # TT
                                      [cl_a2b2],  # TT
                                      wa, wb).reshape([nb, nb])
                
                self.blockcov[n_a1+'x'+n_a2+', '+n_b1+'x'+n_b2] = block
                self.blockcov[n_b1+'x'+n_b2+', '+n_a1+'x'+n_a2] = block.T
                                
                self.fullcov[ia*nb:ia*nb+nb, ib*nb:ib*nb+nb] = block
                self.fullcov[ib*nb:ib*nb+nb, ia*nb:ia*nb+nb] = block.T
                                
        return fullcov
    
    def savecov(self, path):
        np.save(path+'fullcov_gaussian.npy', self.fullcov)
        np.save(path+'blockcov_gaussian.npy', self.blockcov)
