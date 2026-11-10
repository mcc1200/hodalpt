import sbi
import h5py
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from scipy.special import erf, erfinv
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

seed = 12387
device = "cpu"   # small MAF + 1D x: CPU is fine and deterministic

# ... cdf_transform / inv_cdf_transform as before ...
def cdf_transform(x, bounds):
    return 0.5 * (1 + erf(x / np.sqrt(2))) * (bounds[1] - bounds[0]) + bounds[0]

def _inv_gaussian_cdf(x, mu, sigma):
    """ Inverse CDF of a Gaussian distribution.

    :math:`F^{-1}(x) = \\mu + \\sigma \\sqrt{2} \\text{erfinv}(2 \\times x - 1)`

    """
    return mu + sigma * np.sqrt(2) * erfinv(2 * x - 1)
def inv_cdf_transform(x, bounds):
    """ Transform from a Uniform with bounds (which is x) to a Gaussian
    """
    return _inv_gaussian_cdf((x - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)

trainfn = '/Users/mcc3842/CosmicSim2025/hodalpt/bin/npe/data/LHC_cosmo_ngal_trainset.hdf5'

with h5py.File(trainfn, "r") as f:
    LHC_idx = f["LHC_idx"][...]
    Om = f["Om"][...]
    s8 = f["s8"][...]
    ngs = f["ngal"][...]

X_t = np.log1p(ngs.reshape(-1, 1))
theta = np.vstack([Om, s8]).T
bounds = np.array([(0.1, 0.5), (0.6, 1.0)]).T
theta_t = inv_cdf_transform(theta, bounds)
assert np.all(np.isfinite(theta_t)), "parameter on/outside bounds -> inf after Gaussianization"

# --- reproducible split (use the LHC_idx group split above if cosmologies repeat) ---
rng = np.random.default_rng(seed)
ishuffle = rng.permutation(len(theta_t))
n_train = int(0.9 * len(theta_t))
itrain, itest = ishuffle[:n_train], ishuffle[n_train:]

to_t = lambda a: torch.tensor(a, dtype=torch.float32)
y_train, x_train = to_t(theta_t[itrain]), to_t(X_t[itrain])
y_test,  x_test  = to_t(theta_t[itest]),  to_t(X_t[itest])

# Uniform LHC in physical space -> standard normal in Gaussianized space
prior = MultivariateNormal(torch.zeros(2), torch.eye(2))

# --- small architecture sweep, each run seeded ---
configs = [dict(hidden_features=h, num_transforms=t)
           for h in (32, 64, 128) for t in (3, 5, 8)]

results = []
for i, cfg in enumerate(configs):
    torch.manual_seed(seed + i)
    net_builder = posterior_nn(model="maf", **cfg)
    inference = NPE(prior=prior, density_estimator=net_builder, device=device)
    inference.append_simulations(y_train, x_train).train(
        training_batch_size=128,
        learning_rate=5e-4,
        validation_fraction=0.1,
        stop_after_epochs=30,
        max_num_epochs=3000,
    )
    post = inference.build_posterior()

    # held-out score: mean log q(theta_i | x_i) over test pairs (higher = better)
    with torch.no_grad():
        lp = post.log_prob_batched(y_test.unsqueeze(0), x=x_test)  # (1, n_test)
    score = lp.mean().item()
    results.append((score, cfg, post))
    print(f"{cfg}: test log prob = {score:.4f}")

results.sort(key=lambda r: r[0], reverse=True)
best_score, best_cfg, posterior = results[0]
print("best:", best_cfg, best_score)

import pickle
import sbi

fn = 'qphi_cosmo_ngal.pkl'

with open(fn, "wb") as f:
    pickle.dump({"posterior": posterior,
                 "bounds": bounds,
                 "sbi_version": sbi.__version__}, f)


import tarp

test_samples = posterior.sample_batched((10000,), x=x_test, show_progress_bars=True)
y_test_phys = cdf_transform(y_test.numpy(), bounds)
samples_phys = cdf_transform(test_samples.cpu().numpy(), bounds)

ecp, alpha = tarp.get_tarp_coverage(samples_phys, y_test_phys)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot([0, 1], [0, 1], 'k--', label='ideal')
ax.plot(alpha, ecp, label='TARP')
ax.set_xlabel('Credibility Level')
ax.set_ylabel('Expected Coverage')
ax.set_title(r'TARP: $q_{\phi}(\Omega_m, \sigma_8| \bar{n}_g$)')
ax.legend()
fig.tight_layout()
fig.savefig('cosmoprior_tarp.png')

import corner
ng_fid = 444682
x_o = torch.tensor([[np.log1p(ng_fid)]], dtype=torch.float32)   # shape (1, 1), log1p ≈ 13.0
labels = [r'$\Omega_m$', r'$\sigma_8$']
samples = posterior.sample((10000,), x=x_o)                     # (10000, 2), still in Gaussian space
samples_phys_fid = cdf_transform(samples.cpu().numpy(), bounds) # back to (Om, s8)
truths= [0.3175, 0.834]
fig = corner.corner(samples_phys_fid, labels=labels, truths=truths,
                    range=[(0.1, 0.5), (0.6, 1.0)],
                    quantiles=[0.16, 0.5, 0.84], show_titles=True)
fig.suptitle(r'$q_{\phi}(\Omega_m, \sigma_8 |n_{\mathrm{fiducial}}= 444682)$', y=1.05)
fig.savefig('cosmoprior_corner.png')