'''

module for different priors used in the hodalpt project

'''
import os
from functools import lru_cache
import numpy as np

# best-fit nmean (fit to fiducial HOD)
nmean_bf = np.array([[9.8537e-05, 4.2518e-05, 3.3140e-06],
                     [5.7186e-05, 1.5255e-04, 3.7179e-05],
                     [3.7550e-06, 1.6143e-05, 1.3051e-05]])



def sample_bias(seed, model='nonlocal2'):
    ''' sample comsic web classification bias model based on best-fit to
    Quijote fiducial + HOD 


    returns
    -------
    dict with alpha, beta, nmean, and rsd parameters
    function to write pm 10x ALPT nlb priors centered on quijote fiducial best fit.
    alpha, beta, nmean are arrays of cenral best fit values (16,), width is desired prior width (percentile)
    returns dictionaries for alpha, beta, nmean 
    '''
    rng = np.random.default_rng(seed)
    
    if model == 'nonlocal2': 
        # sample nmean (ranges set based on best-fit and spanning 10x)
        sample_nmean = np.zeros((4,4))
        # knots
        sample_nmean[0,0] = 10**rng.uniform(np.log10(3e-5), np.log10(3e-4)) 
        sample_nmean[0,1] = 10**rng.uniform(np.log10(1e-5), np.log10(1e-4)) 
        sample_nmean[0,2] = 10**rng.uniform(np.log10(1e-6), np.log10(1e-5)) 
        # filaments
        sample_nmean[1,0] = 10**rng.uniform(np.log10(1e-5), np.log10(1e-4)) 
        sample_nmean[1,1] = 10**rng.uniform(np.log10(5e-5), np.log10(5e-4)) 
        sample_nmean[1,2] = 10**rng.uniform(np.log10(1e-5), np.log10(1e-4)) 
        # sheets 
        sample_nmean[2,0] = 10**rng.uniform(np.log10(1e-6), np.log10(1e-5)) 
        sample_nmean[2,1] = 10**rng.uniform(np.log10(5e-6), np.log10(5e-5)) 
        sample_nmean[2,2] = 10**rng.uniform(np.log10(5e-6), np.log10(5e-5)) 

        # sample alpha
        sample_alpha = np.zeros((4,4))
        sample_alpha[:3,:3] = rng.uniform(0.01, 3, size=(3,3))

        # sample beta 
        sample_beta = np.zeros((4,4))
        sample_beta[:3,:3] = rng.uniform(0.1, 100, size=(3,3))

        # sample rhoeps 
        sample_rhoeps = np.zeros((4,4))
        sample_rhoeps[:3,:3] = rng.uniform(0., 20, size=(3,3))

        # sample eps
        sample_eps = np.zeros((4,4))
        sample_eps[:3,:3] = rng.uniform(0., 4, size=(3,3))
        
        theta = {'nmean': sample_nmean, 
                 'alpha': sample_alpha, 
                 'beta': sample_beta, 
                 'rhoeps': sample_rhoeps, 
                 'eps': sample_eps} 
    else: 
        raise NotImplementedError

    # best-fit for reference 
    # theta_rsd = { 'bv': 0.7289, 'bb': 1.1652, 'betarsd': 1.3136, 'gamma': 0.4944}
    theta_rsd = {} 
    theta_rsd['bv'] = rng.uniform(0., 2.)       # linear velocity bias --- no RSD to double RSD
    theta_rsd['bb'] = rng.uniform(0., 2.)       # FoG sigma linear factor --- no FoG to double FoG
    theta_rsd['betarsd'] = rng.uniform(0., 2.)  # FoG (1+delta)**betarsd
    theta_rsd['gamma'] = rng.uniform(0., 1.)    # fog deviation from Gaussian

    return theta, theta_rsd

def sample_bias_realspace(seed, model='nonlocal2'):
    ''' sample comsic web classification bias model based on best-fit to
    Quijote fiducial + HOD 


    returns
    -------
    dict with alpha, beta, nmean, and rsd parameters
    function to write pm 10x ALPT nlb priors centered on quijote fiducial best fit.
    alpha, beta, nmean are arrays of cenral best fit values (16,), width is desired prior width (percentile)
    returns dictionaries for alpha, beta, nmean 

    modifications from redshift-space priors:
    * linear-uniform prior on nmean
    * no theta-rsd dictionary produced
    '''
    rng = np.random.default_rng(seed)
    
    if model == 'nonlocal2': 
        # sample nmean (ranges set based on best-fit and spanning 10x)
        sample_nmean = np.zeros((4,4))
        # knots
        sample_nmean[0,0] = 10**rng.uniform(3e-5, 3e-4) 
        sample_nmean[0,1] = 10**rng.uniform(1e-5, 1e-4) 
        sample_nmean[0,2] = 10**rng.uniform(1e-6, 1e-5) 
        # filaments
        sample_nmean[1,0] = 10**rng.uniform(1e-5, 1e-4) 
        sample_nmean[1,1] = 10**rng.uniform(5e-5, 5e-4) 
        sample_nmean[1,2] = 10**rng.uniform(1e-5, 1e-4) 
        # sheets 
        sample_nmean[2,0] = 10**rng.uniform(1e-6, 1e-5) 
        sample_nmean[2,1] = 10**rng.uniform(5e-6, 5e-5) 
        sample_nmean[2,2] = 10**rng.uniform(5e-6, 5e-5) 

        # sample alpha
        sample_alpha = np.zeros((4,4))
        sample_alpha[:3,:3] = rng.uniform(0.01, 3, size=(3,3))

        # sample beta 
        sample_beta = np.zeros((4,4))
        sample_beta[:3,:3] = rng.uniform(0.1, 100, size=(3,3))

        # sample rhoeps 
        sample_rhoeps = np.zeros((4,4))
        sample_rhoeps[:3,:3] = rng.uniform(0., 20, size=(3,3))

        # sample eps
        sample_eps = np.zeros((4,4))
        sample_eps[:3,:3] = rng.uniform(0., 4, size=(3,3))
        
        theta = {'nmean': sample_nmean, 
                 'alpha': sample_alpha, 
                 'beta': sample_beta, 
                 'rhoeps': sample_rhoeps, 
                 'eps': sample_eps} 
    else: 
        raise NotImplementedError
    return theta

def sample_HOD(seed): 
    ''' sample HOD parameters from Gaussian priors set around SIMBIG CMASS
    constraints  
    '''
    rng = np.random.default_rng(seed)

    hod = {
        'logMmin': rng.normal(12.97, 0.11),
        'sigma_logM': max(rng.normal(0.40, 0.1), 1e-3),
        'logM0': rng.normal(13.67, 0.3),
        'logM1': rng.normal(13.68, 0.31),
        'alpha': max(rng.normal(0.79, 0.26), 1e-3),
        'Abias': rng.normal(0.01, 0.16),
        'eta_conc': max(rng.normal(1.11,0.40), 1e-3),
        'eta_cen': max(rng.normal(0.31, 0.13), 1e-3),
        'eta_sat': max(rng.normal(0.85, 0.27), 1e-3)
        }
    return hod

