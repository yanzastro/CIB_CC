# This file defines general log-likelihood class for MCMC fitting
# and functions to setup and call emcee to conduct the fitting.

import pyccl as ccl
import numpy as np
import emcee
import os
from multiprocessing import Pool
from getdist import * 

os.environ["OMP_NUM_THREADS"] = "1"

class lnlikelihood:

    def __init__(self, data_x, data_y, cov, model, params_range,
                 paramsname_free, paramsdict_fixed=None, prior_only=False, kwargs=None):
        '''
        This class defines the log-likelihood function for MCMC fitting.
        Inputs:
        data_x: x data  # not very important though...
        data_y: y data  # as a function of x
        cov: covariance matrix of the data
        model: model function to fit the data, the output should be the same format as data_y
        params_range: the range of the parameter prior
        paramsname_free: the name list of the free parameters
        paramsdict_fixed: the dictionary of the fixed parameters
        prior_only: if True, only the prior is calculated
        kwargs: the keyword arguments for the model function
        '''
        self.data_x = data_x
        self.data_y = data_y
        self.model = model
        self.cov = cov
        self.params_range = params_range
        self.nparams = params_range.shape[0]
        self.paramsdict_fixed = paramsdict_fixed
        self.paramsname_free = paramsname_free
        self.invcov = np.linalg.inv(cov)
        self.prior_only = prior_only
        self.kwargs = kwargs
        
    def lnprior(self, params):
        for i, p in enumerate(params):
            if not(self.params_range[i][0] <= p <= self.params_range[i][1]):
                return -np.inf
            else:
                continue
        return 0


    def prior_transform(self, utheta):
        prange = (self.params_range.T[1] - self.params_range.T[0]).T

        pt = prange * utheta + self.params_range.T[0]
            
        return pt
    
    def lnposterior(self, params):
        '''
        This function returns the log-posterior of the model given the data.
        '''
        prior = self.lnprior(params)
        if self.prior_only:
            return prior
        if prior == -np.inf:
            return -np.inf
        if self.kwargs != None:
            model_y = self.model(params, self.paramsname_free,
                             self.paramsdict_fixed, **self.kwargs)
        else:
            model_y = self.model(params, self.paramsname_free,
                             self.paramsdict_fixed)
        #cov, model_y = self.model(params)

        if (model_y is None) or (self.cov is None):
            return -np.inf
        elif  (np.isnan(model_y).sum() != 0) or (np.isnan(self.cov).sum() != 0):
            return -np.inf
        else:

            diff = model_y - self.data_y
            chi2_dev_2 = diff @ self.invcov @ diff / 2
            if np.isnan(chi2_dev_2):
                return -np.inf                
            if chi2_dev_2 < 0:
                return -np.inf
            return -chi2_dev_2
        

def runmcmc(pos_0, ndim, nstep, nwalkers, lnposterior, pool=None, burnin=None, thread=20, **kwargs):
    '''
    This function defines emcee sampler and run the MCMC chain.
    Inputs:
    pos_0: the initial position of the walkers
    ndim: the number of parameters
    nstep: the number of steps for the MCMC chain
    nwalkers: the number of walkers
    lnposterior: the log-posterior function
    pool: the pool for multiprocessing
    burnin: the number of steps for burn-in
    thread: the number of threads
    kwargs: the keyword arguments for the emcee sampler
    '''

    #ndim = params0.size
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior, threads=60)
    if pool is None:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior, threads=thread, **kwargs)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior, pool=pool, **kwargs)
        
    ###
    #print(params_range.shape)
    #pos = [np.random.rand(ndim)*(params_range[:, 1]-params_range[:, 0])+params_range[:, 0] for i in range(nwalkers)]
    
    #pos = [np.asarray(params0) + lstep * np.random.rand(ndim) for i in range(nwalkers)]

    if burnin is None:
        print('MCMC chain running...')
        pos, prob, state = sampler.run_mcmc(pos_0, nstep, progress=True)
        return sampler.flatchain, sampler.flatlnprobability
    else:
        print('Burning in...')
        state = sampler.run_mcmc(pos_0, burnin, progress=True)
        sampler.reset()
        print('MCMC chain running...')
        state = sampler.run_mcmc(state, nstep, progress=True)
        return sampler.flatchain, sampler.flatlnprobability
    
def make_trangle_plot(g, plot_wrapper, kwargs={}):
    '''
    This function makes a triangle plot of the MCMC chain using Getdist.
    '''
    samples_list = []
    line_args_list = []
    contour_args_list = []
    filled_list = []
    for key in plot_wrapper.keys():
        samples_list.append(plot_wrapper[key]['sampler'])
        filled_list.append(plot_wrapper[key]['filled'])
        line_args_list.append({'color':plot_wrapper[key]['color'],'alpha': plot_wrapper[key]['alpha'], 'lw': plot_wrapper[key]['lw'], 'zorder': plot_wrapper[key]['zorder']})
        contour_args_list.append({'color':plot_wrapper[key]['color'], 'alpha': plot_wrapper[key]['alpha'], 'zorder': plot_wrapper[key]['zorder']})
    g.triangle_plot(samples_list, filled = filled_list, legend_loc='upper right', line_args=line_args_list, contour_args=contour_args_list, **kwargs)

    
def funcsample(chain, x, func):
    '''
    This function samples the model function given an MCMC chain, then calculates the mean, upper and lower limits of the model function given the MCMC chain.
    Inputs:
    chain: the MCMC chain (as an ndarray)
    x: the x values
    func: the model function
    '''
    x = np.atleast_1d(x)

    funcchain = np.zeros((chain.shape[0], x.size))
    
    test_res = func(chain[-1])
    data_size = test_res.size
    
    lower = np.zeros(data_size)
    upper = np.zeros(data_size)
    mean = np.zeros(data_size)
    #for i in range(chain.shape[0]):
    #    funcchain[i] = sfrd(zs=x, params=chain[i])
    with Pool(50) as pool:
        results = pool.map(func, chain)    
    funcchain = np.array(results)
    funcchain = funcchain[~np.isnan(funcchain).any(axis=1)]
    #print(np.where(funcchain[0]<0))
    lower, upper = np.percentile(funcchain, [50-34,50+34], axis=0)
    mean = np.percentile(funcchain, 50, axis=0)
    return mean, upper, lower