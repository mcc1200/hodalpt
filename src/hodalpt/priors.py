'''

module for different priors used in the hodalpt project

'''
import os
from functools import lru_cache
import numpy as np
import os
from scipy.special import erf, erfinv
from pathlib import Path
import torch
import sbi

# best-fit nmean (fit to fiducial HOD)
nmean_bf = np.array([[9.8537e-05, 4.2518e-05, 3.3140e-06],
                     [5.7186e-05, 1.5255e-04, 3.7179e-05],
                     [3.7550e-06, 1.6143e-05, 1.3051e-05]])

def cdf_transform(x, bounds):
    """standard-normal x (..., D) -> uniform on [lo, hi]; bounds shape (2, D)"""
    lo, hi = bounds
    assert lo.shape == hi.shape == x.shape[-1:], (bounds.shape, x.shape)
    return lo + 0.5 * (1 + erf(x / np.sqrt(2))) * (hi - lo)

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
        sample_nmean[0,0] = rng.uniform(3e-5, 3e-4)
        sample_nmean[0,1] = rng.uniform(1e-5, 1e-4)
        sample_nmean[0,2] = rng.uniform(1e-6, 1e-5)
        # filaments
        sample_nmean[1,0] = rng.uniform(1e-5, 1e-4)
        sample_nmean[1,1] = rng.uniform(5e-5, 5e-4)
        sample_nmean[1,2] = rng.uniform(1e-5, 1e-4)
        # sheets
        sample_nmean[2,0] = rng.uniform(1e-6, 1e-5)
        sample_nmean[2,1] = rng.uniform(5e-6, 5e-5)
        sample_nmean[2,2] = rng.uniform(5e-6, 5e-5)

        # sample alpha
        sample_alpha = np.zeros((4,4))
        sample_alpha[:3,:3] = rng.uniform(0.01, 3, size=(3,3))

        # sample beta 
        sample_beta = np.zeros((4,4))
        sample_beta[:3,:3] = rng.uniform(0.1, 100, size=(3,3))

        # sample rhoeps 
        sample_rhoeps = np.zeros((4,4))
        sample_rhoeps[:3,:3] = rng.uniform(2.0, 20, size=(3,3)) # modified

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

def sample_bias_realspace_conservative(seed, model='nonlocal2'):
    ''' sample comsic web classification bias model based on best-fit to
    Quijote fiducial + HOD 


    returns
    -------
    dict with alpha, beta, nmean, and rsd parameters
    function to write pm 10x ALPT nlb priors centered on quijote fiducial best fit.
    alpha, beta, nmean are arrays of cenral best fit values (16,), width is desired prior width (percentile)
    returns dictionaries for alpha, beta, nmean 

    modifications from redshift-space priors:
    * no theta-rsd dictionary produced
    * alpha FF allowed to reach 4 (amplitude)
    * capped floor on rhoeps at 1 (avoid pathological spectra)
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

        sample_alpha[1,1] = rng.uniform(0.01, 4) # allow alpha ff to raise amplitude
    

        # sample beta 
        sample_beta = np.zeros((4,4))
        sample_beta[:3,:3] = rng.uniform(0.1, 100, size=(3,3))

        # sample rhoeps 
        sample_rhoeps = np.zeros((4,4))
        sample_rhoeps[:3,:3] = rng.uniform(1.0, 20, size=(3,3)) # modified

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

def sample_HOD_realspace(seed): 
    ''' sample HOD parameters from Gaussian priors set around SIMBIG CMASS
    constraints, real space 
    modification: multivariate gaussian prior from CMASS bispectrum npe posteriors, widened 5%
    '''

    rng = np.random.default_rng(seed)
    CHAIN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sims", "dat", "mc.simbig.cmass.b0k_kmax0p5.npy")
    chain_allpars = np.load(CHAIN_FILE)
    chain = chain_allpars[:, 5:11]

    mu = np.mean(chain, axis=0)
    cov = np.cov(chain.T)

    s = 1.05                      # widen stds by 5%
    cov_puffy = s**2 * cov

    t = rng.multivariate_normal(mu, cov_puffy, size=1)[0]

    hod = {
        'logMmin': t[0],
        'sigma_logM': max(t[1],1e-3),
        'logM0': t[2],
        'logM1': t[3],
        'alpha': max(t[4], 1e-3),
        'Abias': t[5],
        'eta_conc':1.0, # etas not used in real-space, avoids nonetype err
        'eta_cen': 0.0, 
        'eta_sat': 1.0
        }
    return hod

def sample_HOD_realspace_array(seed=None, size=None, widen=1.05):
    ''' Draw HOD params from a multivariate Gaussian fit to the SimBIG CMASS
    bispectrum NPE posterior, stds widened by `widen`. Returns array of shape
    (6,) if size is None, else (size, 6), columns ordered as HOD_KEYS. O

    output of this form is required for sampling theta'_NLB from the emulator

    HOD_KEYS = ('logMmin', 'sigma_logM', 'logM0', 'logM1', 'alpha', 'Abias')
    HOD_FIXED_REALSPACE = {'eta_conc': 1.0, 'eta_cen': 0.0, 'eta_sat': 1.0}

    `seed` may be an int, SeedSequence, or np.random.Generator.
    '''
    rng = np.random.default_rng(seed)

    CHAIN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sims", "dat", "mc.simbig.cmass.b0k_kmax0p5.npy")
    chain_allpars = np.load(CHAIN_FILE)
    chain = chain_allpars[:, 5:11]

    mu = np.mean(chain, axis=0)
    cov = np.cov(chain.T)
    cov_puffy = widen**2 * cov
    
    t = rng.multivariate_normal(mu, cov_puffy, size=size)
    t[..., 1] = np.maximum(t[..., 1], 1e-3)   # sigma_logM clip to positive
    t[..., 4] = np.maximum(t[..., 4], 1e-3)   # ditto for alpha
    return t

def pip_NLB(seed, size_hod_arr, n_nlb_samp=1):
    """ Physically informed prior on nonlocal bias:
        theta_hod ~ p(theta_hod),  theta'_nlb ~ q_phi(theta_nlb | theta_hod)

    Returns
    -------
    theta_hod : (size_hod_arr, 6), columns HOD_KEYS
    theta_nlb : (n_nlb_samp, size_hod_arr, 80); theta_nlb[s, i] is conditioned
                on theta_hod[i]
    """
    ss_hod, ss_nlb = np.random.SeedSequence(seed).spawn(2)
    
    NPE_DIR   = Path('/work/11053/mcasas/ls6/hodalpt/bin/npe/studies/' # TACC
    #NPE_DIR    = Path('/Users/mcc3842/CosmicSim2025/hodalpt/bin/npe/studies/'
                 'qphi_hod2nlb.exp20.maf.pdf+p0+q123.0.4.emb.noRSD.n45.d100.emb')
    NPE_TAG   = NPE_DIR.name
    qphi_path = NPE_DIR / f'{NPE_TAG}.36.pt'
    norm_path = NPE_DIR / f'{NPE_TAG}.hod_norm.npz'

    qphi = torch.load(qphi_path, weights_only=False)
    norm = np.load(norm_path)

    bounds, aidx = norm['bounds_keep'], norm['active_idx']
    hod_mean, hod_std = norm['hod_mean'], norm['hod_std']

    theta_hod = sample_HOD_realspace_array(ss_hod, size_hod_arr)
    x_t = (theta_hod - hod_mean) / (hod_std + 1e-8)

    x = torch.tensor(x_t, dtype=torch.float32)
    with torch.random.fork_rng():
        torch.manual_seed(int(ss_nlb.generate_state(1)[0]))
        y = qphi.sample_batched((n_nlb_samp,), x=x, show_progress_bars=False).cpu().numpy()
    assert y.shape == (n_nlb_samp, size_hod_arr, len(aidx)), y.shape

    theta_nlb = np.zeros((n_nlb_samp, size_hod_arr, 80))
    theta_nlb[..., aidx] = cdf_transform(y, bounds)


    return theta_hod, theta_nlb


