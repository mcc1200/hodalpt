'''
gen_thetas.py: script uses emulator funciton pip_NLB to 
generate theta_nlb from the theta_hod prior 
(5% widened SimBIG CMASS b(k) posterior)

writes json file of theta_gals, theta_hod dicts 
along with i_LHC (for idx in lhc_idx)
'''
import sbi
import h5py
import numpy as np
import torch
from scipy.special import erf, erfinv
import os
from pathlib import Path
import hodalpt.priors as s
import json

cuda = torch.cuda.is_available()
device = ("cuda:0" if cuda else "cpu")

seed = 20261029

from hodalpt.priors import pip_NLB



HOD_KEYS = ('logMmin', 'sigma_logM', 'logM0', 'logM1', 'alpha', 'Abias')
HOD_FIXED_REALSPACE = {'eta_conc': 1.0, 'eta_cen': 0.0, 'eta_sat': 1.0}
NLB_KEYS = ('alpha', 'beta', 'nmean', 'rhoeps', 'eps')
N_NLB = 80

lhc_idx = np.load('./95p_lhc_idx.npy') # run from bin/lhc_alpt directory

def theta_nlb_dict(pars):
    return dict(zip(NLB_KEYS, np.asarray(pars).reshape(5, 4, 4)))
def theta_hod_dict(pars):
    return dict(zip(HOD_KEYS, np.asarray(pars)))
def nlb_to_vec(theta_gal):
    return np.concatenate([theta_gal[k].ravel() for k in NLB_KEYS])


theta_hod, theta_nlb = pip_NLB(seed, size_hod_arr=len(lhc_idx), n_nlb_samp=1)
theta_HOD_dict = [{**theta_hod_dict(p), **HOD_FIXED_REALSPACE} for p in theta_hod]

theta_NLB_dict = [theta_nlb_dict(p) for p in theta_nlb[0]]  
# for n_nlb_samp > 1:  theta_nlb.reshape(-1, N_NLB) is sample-major,
# so row k pairs with theta_hod[k % size_hod_arr]


def _jsonable(d):
    return {k: np.asarray(v).tolist() for k, v in d.items()}

# row k of the pip_NLB output pairs with lhc_idx[k]
assert len(lhc_idx) == len(theta_NLB_dict) == len(theta_HOD_dict)
out = {
    str(i): {'theta_NLB': _jsonable(nlb), 'theta_HOD': _jsonable(hod)}
    for i, nlb, hod in zip(lhc_idx, theta_NLB_dict, theta_HOD_dict)
}
with open('theta_NLB_PIP.json', 'w') as f:
    json.dump(out, f)

