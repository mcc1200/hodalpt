
import pickle
import numpy as np
import torch
from scipy.special import erf

SEED = 20261106
N_SAMP = 9000

extra_names  = ["Ob", "h", "ns"]
extra_bounds = np.array([(0.03, 0.07),   # Ob
                         (0.50, 0.90),   # h
                         (0.80, 1.20)])  # ns
def cdf_transform(x, bounds):
    return 0.5 * (1 + erf(x / np.sqrt(2))) * (bounds[1] - bounds[0]) + bounds[0]

def sample_reproducible(posterior, x, n, seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return posterior.sample((n,), x=x, show_progress_bars=False)
    
def sample_cosmoprior_ngal(n_samp, seed):

    with open("qphi_cosmo_ngal.pkl", "rb") as f:
        ckpt = pickle.load(f)

    posterior = ckpt['posterior']
    bounds = ckpt['bounds']

    ng_fid = 444682
    x_o = torch.tensor([[np.log1p(ng_fid)]], dtype=torch.float32)
    samples = sample_reproducible(posterior, x_o, n_samp, seed=seed)               # Gaussianized space
    samples_phys_fid = cdf_transform(samples.numpy(), bounds)       # (Om, s8)
    return samples_phys_fid

def sample_extra(n_samp, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(extra_bounds[:, 0], extra_bounds[:, 1], size=(n_samp, len(extra_bounds)))

    
samples = sample_cosmoprior_ngal(N_SAMP, SEED)
extra = sample_extra(N_SAMP, seed=SEED)            # columns: Ob, h, ns

cosmo = np.column_stack([
    samples[:, 0],   # Om
    extra[:, 0],              # Ob
    extra[:, 1],              # h
    extra[:, 2],              # ns
    samples[:, 1],   # s8
])

np.save('cosmo_samples.npy', cosmo)

