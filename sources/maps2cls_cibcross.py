import numpy as np
import pymaster as nmt
import healpy as hp
import os
import pickle
import gc
import yaml

#os.environ['OMP_NUM_THREADS'] = '1'
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['MKL_NUM_THREADS'] = '1'

class maps2cls:
    
    '''
    This class is a wrapper to calculate nx2pt power spectra.
    To initialize it, one needs to produce names of the fields, and corresponding spins
    (0 for galaxy overdensity and 2 for shear), path to the healpix maps and masks.
    The basic input of this class is
    field_names: a list containing the names (as strings) of each field to correlate;
    field_spins: a dict specifying the spins of each field. The keys should agree
                field_names. Spin can only be 0 (for a scalar map like galacy density, 
                CMB temperature, tSZ y map, etc) or 2 (for shear maps or CMB polarization 
                maps);
    field_maps_path: a dict containing the path to the maps of the fields;
    field_masks_path: a dict containing the path to the masks of the fields;
    nside: None or an integer to ud_grade maps and masks into a same resolution.
    field_pairs: None or an array of name pairs set by the user.
                The array should be like array([[name11, name12],[name21, name22],...]), 
                where 'names' should be included in field_names;
    field_res: None or a dict containing beam resolution of corresponding field map (in arcmin)
    '''
    
    def __init__(self, field_names, field_spins, field_maps_path, field_masks_path, field_res, nside=None, field_pairs=None):
        
        """
        This function initializes all the information needed for cross-correlations between
        multiple fields.
        """
        
        self.field_names = field_names
        self.field_spins = field_spins
        self.field_maps_path = field_maps_path
        self.field_masks_path = field_masks_path
        self.maps = {}
        self.masks = {}
        self.cls = {}
        if field_pairs is None:
            self._make_field_pairs()
        else:
            self.name_pairs = field_pairs
            self._validate_names_in_pairs()
        self.field_res = field_res
        self._load_maps()
        
        self._get_fields(nside)
        self.wsp = None
        
    def _make_field_pairs(self):
        
        '''
        Combine all the n fields to make C(n,2) = n*(n+1)/2 pairs of fields. 
        '''
        
        for t in range(len(self.field_names)):
            name1 = self.field_names[t]
            for i in range(len(self.field_names)):
                if t > i:
                    continue
                name2 = self.field_names[i]
                #name_pair_text = name1+'_'+name2
            
                name_pair = np.array([name1, name2]) 
            
                if i == 0 and t == 0:
                    self.name_pairs = name_pair
                else:
                    self.name_pairs = np.vstack([self.name_pairs, name_pair])
            if self.name_pairs.ndim == 1:
                self.name_pairs = self.name_pairs[None,:]
    
    def _validate_names_in_pairs(self):
        for name_pair in self.name_pairs:
            if (name_pair[0] not in self.field_names) or (name_pair[1] not in self.field_names):
                raise ValueError("Field names in name pairs should be from field_names: "+self.field_names)
    
    def _load_maps(self):
        
        '''
        Load maps of all the fields. If the corresponding spin is zero, then only the
        first 'field' of the healpix map is loaded; if it is two, then the second and 
        third 'fields' are loaded. Note that a single map triplet can make both a spin
        2 and a spin 0 field. If both information are needed in the cross-correlation,
        then one needs to have two fields with the same map path but different spins.          
        '''
        
        for field_name in self.field_names:
            
            field_spin = self.field_spins[field_name]
            if field_spin == 0:
                map0 = hp.read_map(self.field_maps_path[field_name], field=0)
                map0[map0==hp.UNSEEN] = 0
                self.maps[field_name] = [map0]
            elif field_spin == 2:
                map1, map2 = hp.read_map(self.field_maps_path[field_name], field=[1, 2])
                map1[map1==hp.UNSEEN] = 0
                map2[map2==hp.UNSEEN] = 0
                self.maps[field_name] = [map1, map2]
            else:
                raise ValueError("Field spin can only be either 0 or 2!")
                                         
            self.masks[field_name] = hp.read_map(self.field_masks_path[field_name])
            self.masks[field_name][self.masks[field_name]==hp.UNSEEN] = 0
                        
            
    def _get_fields(self, nside):
        
        '''
        Combine loaded maps, masks and spins to make MntField objects.
        '''
        
        self.field = {}
        for field_name in self.field_names:
            map = self.maps[field_name]            
            mask = self.masks[field_name]
            if nside is None:
                if mask.size != map[0].size:
                    print('Warning: size of mask and map of field '+field_name+' do not match, unified to that of map.')
                    ns = hp.npix2nside(map[0].size)
                    mask = hp.ud_grade(mask, ns)
            else:
                mask = hp.ud_grade(mask, nside)
                map = hp.ud_grade(map, nside)                
            
            ns = hp.npix2nside(mask.size)
            pixwin = hp.pixwin(ns, lmax=3*ns)
            if self.field_res is None:
                self.field[field_name] = nmt.NmtField(mask, map, beam=pixwin)
            else:
                beam = hp.gauss_beam(np.deg2rad(self.field_res[field_name]/60), lmax=3*ns)
                self.field[field_name] = nmt.NmtField(mask, map, beam=beam*pixwin)
        self.nside = ns
            
    def calculate_wsp(self, b, name_pairs):
        
        '''
        This function initializes NmtWorkspace objects for each pair of fields and calculate
        their coupling matrix.
        b: a nmt.NmtBin object that specifies the binning scheme.
        '''
        
        self.wsp = {}
        for name_pair in name_pairs:
            print('Initializing NmtWorkspace for '+name_pair[0]+'x'+name_pair[1])
           
            f_1 = self.field[name_pair[0]]
            f_2 = self.field[name_pair[1]]
            wsp = nmt.NmtWorkspace()
            wsp.compute_coupling_matrix(f_1, f_2, b)
            self.wsp[name_pair[0]+'x'+name_pair[1]] = wsp
            del wsp, f_1, f_2
            gc.collect()
    
    def save_wsp(self, wsp_path):
        
        '''
        This function saves the NmtWorkspaces that have been initialized.
        wsp_path: string, the path to save the workspaces
        '''
        
        if ~os.path.exists(wsp_path):
            os.system('mkdir -p '+wsp_path)
        name_pairs_keys = list(self.wsp.keys())
        for name_pair in name_pairs_keys:
            filename = os.path.join(wsp_path, 'wsp_'+name_pair+'.pkl')
            self.wsp[name_pair].write_to(filename)
    
    def load_wsp(self, wsp_path):
        
        '''
        This function loads pre-initialized NmtWorkspaces.
        '''
        
        self.wsp = {}
        for name_pair in self.name_pairs:
            filename = os.path.join(wsp_path, 'wsp_'+name_pair[0]+'x'+name_pair[1]+'.pkl')
            print('Loading external Workspace from '+filename)
            wsp = nmt.NmtWorkspace()
            wsp.read_from(filename)
            self.wsp[name_pair[0]+'x'+name_pair[1]] = wsp
    
    def cls_from_map(self, name_pair, b, 
                     bmode=False, noise_bias=None):

        '''
        Calculate a single C_ell.
        '''
            
        f_1 = self.field[name_pair[0]]
        f_2 = self.field[name_pair[1]]
        
        print('Calculating coupled C_ell for '+name_pair[0]+'x'+name_pair[1]+' and correcting for the window function...') 
        cl_coupled = nmt.compute_coupled_cell(f_1, f_2)
                
        print('Decoupling C_ell '+name_pair[0]+'x'+name_pair[1]+'...')
        wsp = self.wsp[name_pair[0]+'x'+name_pair[1]]
        
        if noise_bias is None:
            noise_bias = 0
        
        cls = wsp.decouple_cell(cl_coupled, cl_noise=np.ones_like(cl_coupled)*noise_bias)
        
        if bmode == True and len(map1) == 2 and len(map2) == 2:
            return cls[3]
        else:
            return cls[0]
    
    
    def run(self, b, bmode=False, external_wsp=None, noise_bias=None, name_pairs=None):
        
        '''
        The main function to calculate the power spectra. 
        A NaMaster Nmtbin object b needs to be specified.
        b: a nmt.NmtBin object;
        bmode: bool, calculate the B-mode instead of E-mode for cross-correlation power 
        spectra of spin-2 fields;
        external_wsp: None or string pointing to the path of pre-saved workspace. This will
        speed up the calculation;
        noise_bias: None or a float of the amplitude of theoretical noise_bias.
        '''
        
        self.cls['ell'] = b.get_effective_ells()
        
        if name_pairs is None:
            name_pairs = self.name_pairs
            
        if external_wsp is not None:
            self.load_wsp(external_wsp)
        elif self.wsp is None:
            self.calculate_wsp(b, name_pairs)
        else:
            pass
        
        for name_pair in name_pairs:
            map1 = self.maps[name_pair[0]]
            map2 = self.maps[name_pair[1]]
            
            mask1 = self.masks[name_pair[0]]
            mask2 = self.masks[name_pair[1]]
            if name_pair[0] == name_pair[1] and self.field_spins[name_pair[0]] == 2 and noise_bias is not None:
                nb = noise_bias[name_pair[0]]
                self.cls[name_pair[0]+'x'+name_pair[1]] = self.cls_from_map(name_pair, b, bmode, nb)
            else:
                self.cls[name_pair[0]+'x'+name_pair[1]] = self.cls_from_map(name_pair, b, bmode)
                
    def save_cls(self, path):
        if ~os.path.exists(path):
            os.system('mkdir -p '+path)        
        np.save(path+'cls.npy', self.cls)

    def concatenate_cls(self, name_pairs=None):
        if name_pairs == None:
            name_pairs = self.name_pairs
            
        realcell = np.array([])
        for name_pair in name_pairs:
            realcell = np.hstack([realcell, (cls[name_pair[0]+'x'+name_pair[1]])])
        return realcell