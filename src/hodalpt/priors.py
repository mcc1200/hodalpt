'''

module for different priors usd in the hodalpt project

'''
import os
from functools import lru_cache
import numpy as np

# best-fit nmean (fit to fiducial HOD)
nmean_bf = np.array([[9.8537e-05, 4.2518e-05, 3.3140e-06],
                     [5.7186e-05, 1.5255e-04, 3.7179e-05],
                     [3.7550e-06, 1.6143e-05, 1.3051e-05]])


# ---------------------------------------------------------------------------
# real-space (no RSD) NLB sampling
#
# The full NLB theta vector is 5 blocks (alpha, beta, nmean, rhoeps, eps),
# each a flattened 4x4 matrix over cosmic web type x density-bin type
# (K=knots, F=filaments, S=sheets, plus an always-zero 4th "void" row/col),
# followed by 4 RSD params (bv, bb, betarsd, gamma) -- 84 entries total. This
# matches bin/npe/utils.get_pruned_indices()/PARAM_NAMES and the layout of
# cmassfid_closest_NLBpars.txt.
#
# _CONSTRAINED_ACTIVE_IDX below indexes into the *pruned* 45-entry non-RSD
# vector (5 blocks x 9 active K/F/S x K/F/S cells each, row-major, void
# row/col dropped) -- i.e. the same index space as bin/npe/utils.
# _CONSTRAINED_ACTIVE_IDX / get_constrained_indices() before RSD indices
# 45-48 are appended. Keep this in sync with that module.
# ---------------------------------------------------------------------------

_NLB_BLOCKS = ['alpha', 'beta', 'nmean', 'rhoeps', 'eps']

_CONSTRAINED_ACTIVE_IDX = np.array([
     0,   # alpha_KK
     1,   # alpha_KF
     2,   # alpha_KS
     3,   # alpha_FK
     4,   # alpha_FF
     5,   # alpha_FS
    18,   # nmean_KK
    19,   # nmean_KF
    20,   # nmean_KS
    21,   # nmean_FK
    22,   # nmean_FF
    23,   # nmean_FS
    24,   # nmean_SK
    25,   # nmean_SF
    26,   # nmean_SS
    27,   # rhoeps_KK
    28,   # rhoeps_KF
    31,   # rhoeps_FF
    33,   # rhoeps_SK
    36,   # eps_KK
    37,   # eps_KF
    41,   # eps_FS
], dtype=int)

_NLB_FID_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'bin', 'npe', 'data',
    'cmassfid_closest_NLBpars.txt'))


def _compact_loc(idx):
    ''' map an index into the pruned 45-entry (5 blocks x 9 K/F/S cells,
    no RSD) NLB vector to (block_name, row, col) into a 4x4 block matrix
    '''
    if not (0 <= idx < 9 * len(_NLB_BLOCKS)):
        raise ValueError('idx %i is outside the 45-dim non-RSD active vector' % idx)
    block = _NLB_BLOCKS[idx // 9]
    row, col = divmod(idx % 9, 3)
    return block, row, col


@lru_cache(maxsize=4)
def _load_nlb_fiducial_vec(path):
    return np.loadtxt(path)


def _load_nlb_fiducial(path):
    ''' load the 84-dim padded NLB fiducial vector and split it into the 5
    4x4 (alpha, beta, nmean, rhoeps, eps) matrices. the trailing 4 RSD
    entries are dropped -- irrelevant in real space.
    '''
    vec = _load_nlb_fiducial_vec(path)
    return {name: vec[i * 16:(i + 1) * 16].reshape(4, 4).copy()
            for i, name in enumerate(_NLB_BLOCKS)}


# nmean priors are linear-uniform (NOT log-uniform, unlike sample_bias()) --
# range is specific to each (row, col) K/F/S cell. linear-uniform matches
# the Uniform assumption baked into bin/npe/utils.cdf_transform() /
# inv_cdf_transform(), which every other NLB parameter here already satisfies
# and log-uniform nmean did not -- that mismatch is a likely source of the
# floor/ceiling posterior bunching seen in closure tests.
_NMEAN_RANGES = {
    (0, 0): (3e-5, 3e-4),
    (0, 1): (1e-5, 1e-4),
    (0, 2): (1e-6, 1e-5),
    (1, 0): (1e-5, 1e-4),
    (1, 1): (5e-5, 5e-4),
    (1, 2): (1e-5, 1e-4),
    (2, 0): (1e-6, 1e-5),
    (2, 1): (5e-6, 5e-5),
    (2, 2): (5e-6, 5e-5),
}

# alpha, beta, rhoeps, eps priors are uniform over the same range for every
# K/F/S cell -- same ranges as sample_bias()
_UNIFORM_RANGES = {
    'alpha': (0.01, 3),
    'beta': (0.1, 100),
    'rhoeps': (0., 20),
    'eps': (0., 4),
}


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

def sample_bias_realspace(seed, model='nonlocal2', active_idx=_CONSTRAINED_ACTIVE_IDX,
                           fiducial_path=_NLB_FID_PATH):
    ''' sample cosmic web classification bias model for real-space (no RSD)
    training sets.

    Only the parameters listed in `active_idx` (indices into the pruned
    45-entry non-RSD NLB vector -- see bin/npe/utils.get_pruned_indices() /
    _CONSTRAINED_ACTIVE_IDX) are drawn, using the same ranges as
    `sample_bias` -- except nmean, which is sampled linear-uniform here
    instead of log-uniform (see _NMEAN_RANGES). Every other
    alpha/beta/nmean/rhoeps/eps entry is fixed to the fiducial best-fit value
    loaded from `fiducial_path`. RSD parameters (bv, bb, betarsd, gamma) are
    not sampled or returned -- they don't apply in real space.

    returns
    -------
    dict with alpha, beta, nmean, rhoeps, eps (4x4 arrays; the 4th "void"
    row/col is unused and stays 0, matching the padded NLB convention)
    '''
    if model != 'nonlocal2':
        raise NotImplementedError

    rng   = np.random.default_rng(seed)
    theta = _load_nlb_fiducial(fiducial_path)

    for idx in active_idx:
        block, row, col = _compact_loc(int(idx))
        if block == 'nmean':
            lo, hi = _NMEAN_RANGES[(row, col)]
        else:
            lo, hi = _UNIFORM_RANGES[block]
        theta[block][row, col] = rng.uniform(lo, hi)

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

