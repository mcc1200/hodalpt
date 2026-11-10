# npe_utils.py (formerly utils.py)
from sbi import utils as Ut
import numpy as np
import torch
import h5py
from scipy.special import erf, erfinv

def signed_log(x, floor=1e-10):
    return np.sign(x) * np.log(np.abs(x) + floor)

def training_data(kmax, infn, sumstat, kmin=0.01):
    with h5py.File(infn, 'r') as f:
        k     = f['k'][:]
        p0    = f['p0'][:]
        p2    = f['p2'][:]
        theta = f['theta'][:]
        mask  = (k <= kmax) & (k >= kmin)
        if sumstat == 'p0':
            x = np.log(np.clip(p0[:,mask], 1e-10, None))
        elif sumstat == 'p2':
            x = signed_log(p2[:,mask])
        elif sumstat == 'pk_all':
            x = np.concatenate([
                np.log(np.clip(p0[:,mask], 1e-10, None)),
                signed_log(p2[:,mask])
            ], axis=1)
        # elif sumstat == 'b+q':
        #     b123 = f['b123'][:]
        #     q123 = f['q123'][:]
        #     klim = f['klim'][:]
        # elif sumstat =='b123':

    return theta, x
# def training_data(kmax, infn, sumstat):
#     with h5py.File(infn, 'r') as f:
#         k     = f['k'][:]
#         p0    = f['p0'][:]
#         p2    = f['p2'][:]
#         theta = f['theta'][:]
#         mask  = k <= kmax
#         if sumstat == 'p0':
#             x = p0[:,mask]
#         elif sumstat == 'p2':
#             x = p2[:,mask]
#         elif sumstat == 'pk_all':
#             x = np.concatenate([p0[:,mask], p2[:,mask]], axis=1)
#     return theta, x

def train_val_split(theta, x, seed, n_test=100, save_path=None):
    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(len(x))

    x_all = x[shuffled_idx]
    y_all = theta[shuffled_idx]

    test_idx = shuffled_idx[:n_test]
    if save_path is not None:
        np.savez(save_path, 
                 test_idx=test_idx,
                 x_test=x_all[:n_test],
                 y_test=y_all[:n_test])

    x_train = torch.tensor(x_all[n_test:],       dtype=torch.float32)
    y_train = torch.tensor(y_all[n_test:],       dtype=torch.float32)

    return x_train, y_train

def load_data(f, kmin, kmax, has_p2=True, has_pdf=True):
    ''' has_p2=False for real-space (noRSD) files -- save_spectrum() in
    bias_fiducial_noRSD.py never writes 'p2' (no quadrupole without RSD), so
    p2 is returned as None rather than read.
    '''
    theta = f['theta'][...]
    k = f['k'][...]
    p0 = f['p0'][...]
    p2 = f['p2'][...] if has_p2 else None
    b123 = f['b123'][...]
    q123 = f['q123'][...]
    pdf =  f['cic_pdf'][...] if has_pdf else None
    i_k1 = f['i_k1'][...]
    i_k2 = f['i_k2'][...]
    i_k3 = f['i_k3'][...]
    ngs = f['ngs'][()]

    klim_pk = ((k >= kmin) & (k <= kmax))
    p0 = p0[:, klim_pk]
    if has_p2:
        p2 = p2[:, klim_pk]

    k_fund = 2 * np.pi / 1000.
    # i_k1/i_k2/i_k3 are (N_sims, N_triangles); triangle configs are identical across sims
    k1 = i_k1[0] * k_fund
    k2 = i_k2[0] * k_fund
    k3 = i_k3[0] * k_fund



    klim = (k1 > kmin) & (k2 > kmin) & (k3 > kmin) & \
            (k1 <= kmax)  & (k2 <= kmax)  & (k3 <= kmax)

    b123 = b123[:, klim]   # compress to valid triangles only; shape (N, N_valid)
    q123 = q123[:, klim]   # avoids masked fill-value corruption downstream
    return theta, ngs, p0, p2, q123, pdf, klim

def get_nlb_bounds():
    """
    Return (84, 2) prior bounds for NLB theta derived from sample_bias()
    in hodalpt/src/hodalpt/priors.py (model='nonlocal2').

    Canonical ordering (must match theta vectorization):
        alpha(16), beta(16), nmean(16), rhoeps(16), eps(16), rsd(4)
    Each of the first five groups is a flattened 4×4 block where rows 0-2 /
    cols 0-2 are the active knot/filament/sheet × density subgrid; the void
    row (3) and 4th column are always zero.
    """
    _EPS = 1e-6   # tight bracket for always-zero parameters

    # active mask: rows 0-2, cols 0-2 of each 4×4 block
    active = np.zeros((4, 4), dtype=bool)
    active[:3, :3] = True
    active = active.ravel()

    def _block(lo, hi):
        """(16, 2) bounds for a parameter with the same range at every active cell."""
        b = np.full((16, 2), [0., _EPS])
        b[active] = [lo, hi]
        return b

    # alpha: uniform [0.01, 3]
    alpha_b = _block(0.01, 3.)

    # beta: uniform [0.1, 100]
    beta_b = _block(0.1, 100.)

    # nmean: log-uniform; bounds vary per cell (row = tweb, col = density bin)
    nmean_b = np.full((4, 4, 2), [0., _EPS])
    nmean_b[0, 0] = [3e-5, 3e-4]   # knots,     density 0
    nmean_b[0, 1] = [1e-5, 1e-4]   # knots,     density 1
    nmean_b[0, 2] = [1e-6, 1e-5]   # knots,     density 2
    nmean_b[1, 0] = [1e-5, 1e-4]   # filaments, density 0
    nmean_b[1, 1] = [5e-5, 5e-4]   # filaments, density 1
    nmean_b[1, 2] = [1e-5, 1e-4]   # filaments, density 2
    nmean_b[2, 0] = [1e-6, 1e-5]   # sheets,    density 0
    nmean_b[2, 1] = [5e-6, 5e-5]   # sheets,    density 1
    nmean_b[2, 2] = [5e-6, 5e-5]   # sheets,    density 2
    nmean_b = nmean_b.reshape(16, 2)

    # rhoeps: uniform [0, 20]
    rhoeps_b = _block(0., 20.)

    # eps: uniform [0, 4]
    eps_b = _block(0., 4.)

    # rsd in order _RSD_KEYS = ['bv', 'bb', 'betarsd', 'gamma']
    rsd_b = np.array([
        [0., 2.],   # bv
        [0., 2.],   # bb
        [0., 2.],   # betarsd
        [0., 1.],   # gamma
    ])

    return np.vstack([alpha_b, beta_b, nmean_b, rhoeps_b, eps_b, rsd_b])  # (84, 2)


def get_nlb_bounds_conservative():
    """
    Return (80, 2) prior bounds for NLB theta derived from
    sample_bias_realspace_conservative() in hodalpt/src/hodalpt/priors.py
    (model='nonlocal2'). No RSD block -- sample_bias_realspace_conservative
    is real-space only and returns no theta_rsd.

    Canonical ordering (must match theta vectorization):
        alpha(16), beta(16), nmean(16), rhoeps(16), eps(16)
    Each group is a flattened 4×4 block where rows 0-2 / cols 0-2 are the
    active knot/filament/sheet × density subgrid; the void row (3) and 4th
    column are always zero.
    """
    _EPS = 1e-6   # tight bracket for always-zero parameters

    # active mask: rows 0-2, cols 0-2 of each 4×4 block
    active = np.zeros((4, 4), dtype=bool)
    active[:3, :3] = True
    active = active.ravel()

    def _block(lo, hi):
        """(16, 2) bounds for a parameter with the same range at every active cell."""
        b = np.full((16, 2), [0., _EPS])
        b[active] = [lo, hi]
        return b

    # alpha: uniform [0.01, 3], except kk/fk/ff which are widened to [0.01, 5]
    alpha_b = _block(0.01, 3.).reshape(4, 4, 2)
    alpha_b[1, 1] = [0.01, 4.]   # ff
    alpha_b = alpha_b.reshape(16, 2)

    # beta: uniform [0.1, 100]
    beta_b = _block(0.1, 100.)

    # nmean: linear-uniform; bounds vary per cell (row = tweb, col = density bin)
    nmean_b = np.full((4, 4, 2), [0., _EPS])
    nmean_b[0, 0] = [3e-5, 3e-4]   # knots,     density 0 
    nmean_b[0, 1] = [1e-5, 1e-4]   # knots,     density 1 
    nmean_b[0, 2] = [1e-6, 1e-5]   # knots,     density 2
    nmean_b[1, 0] = [1e-5, 1e-4]   # filaments, density 0
    nmean_b[1, 1] = [5e-5, 5e-4]   # filaments, density 1 
    nmean_b[1, 2] = [1e-5, 1e-4]   # filaments, density 2
    nmean_b[2, 0] = [1e-6, 1e-5]   # sheets,    density 0
    nmean_b[2, 1] = [5e-6, 5e-5]   # sheets,    density 1
    nmean_b[2, 2] = [5e-6, 5e-5]   # sheets,    density 2
    nmean_b = nmean_b.reshape(16, 2)

    # rhoeps: uniform [1.0, 20]
    rhoeps_b = _block(1.0, 20.)

    # eps: uniform [0, 4]
    eps_b = _block(0., 4.)

    return np.vstack([alpha_b, beta_b, nmean_b, rhoeps_b, eps_b])  # (80, 2)


def get_prior_bounds():
    # from sobol
    cosmo_lower = torch.tensor([0.1, 0.03, 0.5, 0.8, 0.6])
    cosmo_upper = torch.tensor([0.5, 0.07, 0.9, 1.2, 1.0])
    # central from parslist_quijote_bestfit, 50% bounds
    alpt_lower = torch.tensor([3.8557e-01, 6.8758e-01, 5.0618e-01, 1.3292e+00, 5.5732e-01, 9.3604e-01,
            1.3567e+00, 1.4422e+00, 4.8722e-01, 1.6048e+00, 2.1101e+00, 1.4812e+00,
            4.2917e-01, 4.0281e-01, 4.0621e-01, 2.3636e-01, 1.2361e+00, 2.2217e+00,
            2.9492e-01, 2.4092e+01, 4.4946e+01, 4.6910e+00, 3.3169e+01, 3.5682e+01,
            8.2116e+00, 3.2208e+00, 4.9426e+01, 2.3394e+01, 5.2163e+01, 4.0580e+00,
            5.0913e+01, 2.1794e+01, 7.5953e-05, 3.2730e-05, 2.5717e-06, 7.5000e-09,
            4.4805e-05, 1.2004e-04, 2.9497e-05, 3.7650e-07, 3.1065e-06, 1.3229e-05,
            1.0483e-05, 2.9475e-07, 1.2750e-08, 5.0250e-08, 5.4000e-08, 3.0000e-09,
            5.4668e-01, 8.7390e-01, 9.8520e-01, 3.7080e-01])
    alpt_upper = torch.tensor([6.4262e-01, 1.1460e+00, 8.4363e-01, 2.2153e+00, 9.2887e-01, 1.5601e+00,
            2.2612e+00, 2.4036e+00, 8.1204e-01, 2.6747e+00, 3.5169e+00, 2.4687e+00,
            7.1528e-01, 6.7135e-01, 6.7702e-01, 3.9393e-01, 2.0601e+00, 3.7028e+00,
            4.9153e-01, 4.0153e+01, 7.4911e+01, 7.8184e+00, 5.5282e+01, 5.9470e+01,
            1.3686e+01, 5.3680e+00, 8.2376e+01, 3.8990e+01, 8.6938e+01, 6.7634e+00,
            8.4856e+01, 3.6324e+01, 1.2659e-04, 5.4550e-05, 4.2862e-06, 1.2500e-08,
            7.4675e-05, 2.0007e-04, 4.9162e-05, 6.2750e-07, 5.1775e-06, 2.2049e-05,
            1.7471e-05, 4.9125e-07, 2.1250e-08, 8.3750e-08, 9.0000e-08, 5.0000e-09,
            9.1113e-01, 1.4565e+00, 1.6420e+00, 6.1800e-01])

    lower_bounds = torch.cat([cosmo_lower, alpt_lower])  # (57,)
    upper_bounds = torch.cat([cosmo_upper, alpt_upper])  # (57,)
    bounds = np.stack([lower_bounds.numpy(), upper_bounds.numpy()], axis=1)
    return bounds #lower_bounds, upper_bounds

def get_prior():
    lower_bounds, upper_bounds = get_prior_bounds()
    return Ut.BoxUniform(low=lower_bounds, high=upper_bounds)

def cdf_transform(x, bounds):
    """ Transform from a Gaussian (which is x) to a Uniform with bounds as input.
    """
    return _gaussian_cdf(x, 0, 1) * (bounds[1] - bounds[0]) + bounds[0]


def inv_cdf_transform(x, bounds):
    """ Transform from a Uniform with bounds (which is x) to a Gaussian
    """
    return _inv_gaussian_cdf((x - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)

def _gaussian_cdf(x, mu, sigma):
    """ CDF of a Gaussian distribution.

    :math:`F(x) = \\frac{1}{2}(1 + erf(\\frac{x - \\mu}{\\sigma}))`
    """
    return 0.5 * (1 + erf((x - mu) / (np.sqrt(2) * sigma)))


def _inv_gaussian_cdf(x, mu, sigma):
    """ Inverse CDF of a Gaussian distribution.

    :math:`F^{-1}(x) = \\mu + \\sigma \\sqrt{2} \\text{erfinv}(2 \\times x - 1)`

    """
    return mu + sigma * np.sqrt(2) * erfinv(2 * x - 1)


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
    # 45,   # b_v
    # 46,   # b_b
    # 47,   # beta_rsd
    # 48,   # gamma
], dtype=int)


def get_constrained_indices():
    """Indices into the 49-dim active theta vector for well-constrained parameters.

    22 parameters selected empirically from exp4/exp5 TARP checks.
    Used when marg_unconstrained=True in npe_emulator.py.
    """
    return _CONSTRAINED_ACTIVE_IDX.copy()


# plain-text names for the 49-dim active theta vector (same order as
# study_corners.PARAM_LABELS, without the LaTeX markup)
_T = ['K', 'F', 'S']   # tweb: knots, filaments, sheets
_D = ['K', 'F', 'S']   # density bins, labeled by their tweb-type environment

def _plain_block(name):
    return ['%s_%s%s' % (name, t, d) for t in _T for d in _D]

PARAM_NAMES = (
    _plain_block('alpha')    +   # [0:9]
    _plain_block('beta')     +   # [9:18]
    _plain_block('nmean')    +   # [18:27]
    _plain_block('rhoeps')   +   # [27:36]
    _plain_block('eps')      +   # [36:45]
    ['bv', 'bb', 'betarsd', 'gamma']  # [45:49]
)


def save_active_params_txt(path, indices, names=PARAM_NAMES):
    """Write a human-readable index/name reference for the active theta params.

    `indices` are positions into the 49-dim active theta vector (e.g. the
    output of get_constrained_indices(), or np.arange(49) when running
    without marginalization).
    """
    with open(path, 'w') as fp:
        fp.write('# idx  name\n')
        for i in indices:
            fp.write('%-4d  %s\n' % (int(i), names[int(i)]))


def load_active_params_txt(path):
    """Inverse of save_active_params_txt: read back the array of active-49-space
    indices from an active_parameters.txt (first whitespace-delimited column of
    each non-comment line; the name column is ignored)."""
    idx = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            idx.append(int(line.split()[0]))
    return np.array(idx, dtype=int)


def get_pruned_indices():
    """Return (pruned_idx, active_idx) for the 84-dim NLB theta vector.

    Each of alpha, beta, nmean, rhoeps, eps is a flattened 4×4 block.
    Only rows 0-2, cols 0-2 are active (9 cells); the void row (row 3) and
    4th column are always zero — 7 pruned per group, 35 total.
    The 4 rsd params at the end are all active, giving 49 active total.

    Flat 4×4 index layout per block:
        col:  0   1   2   3
        row 0: 0   1   2   3*
        row 1: 4   5   6   7*
        row 2: 8   9  10  11*
        row 3:12* 13* 14* 15*
    (* = pruned)

    Returns
    -------
    pruned_idx : np.ndarray, shape (35,)  — column indices to drop
    active_idx : np.ndarray, shape (49,)  — column indices to keep
    """
    # 4th column (col 3 of rows 0-2) + entire void row (row 3)
    inactive_in_block = np.array([3, 7, 11, 12, 13, 14, 15])
    pruned = np.concatenate([inactive_in_block + i * 16 for i in range(5)])
    active = np.setdiff1d(np.arange(84), pruned)
    return pruned.astype(int), active.astype(int)


def prune_theta(theta):
    """Remove always-zero columns from a (N, 84) NLB theta array.

    Returns (N, 49) with the 35 void-row / 4th-col entries stripped.
    Also accepts a 1-D (84,) array and returns (49,).
    """
    _, active = get_pruned_indices()
    if theta.ndim == 1:
        return theta[active]
    return theta[:, active]


def unprune_theta(theta_pruned, fill=0.0, no_rsd=False):
    """Re-insert pruned (zero) columns into a compact NLB theta array.

    Use before cdf_transform / NLB evaluation to recover the full 84-dim (or,
    no_rsd=True, 80-dim) parameter vector from a 49-dim (or 45-dim) NPE
    posterior sample.

    Parameters
    ----------
    theta_pruned : (N, 49) or (49,) array -- (N, 45) or (45,) if no_rsd
    fill : scalar placed in the pruned slots (default 0.0, i.e. lower-bound
           value in transformed space — maps back to ~0 in physical space)
    no_rsd : if True, drop the 4 trailing RSD columns from the active set and
             target width 80 instead of 84 -- for real-space (rsd=False)
             studies, whose theta has no RSD block at all.

    Returns
    -------
    (N, 84) or (84,) array -- (N, 80) or (80,) if no_rsd
    """
    _, active = get_pruned_indices()
    n_dim = 84
    if no_rsd:
        active = active[active < 80]   # drop the 4 RSD columns (80-83)
        n_dim = 80
    scalar = theta_pruned.ndim == 1
    if scalar:
        theta_pruned = theta_pruned[np.newaxis]
    out = np.full((theta_pruned.shape[0], n_dim), fill, dtype=theta_pruned.dtype)
    out[:, active] = theta_pruned
    return out[0] if scalar else out

def pars_to_theta(pars):
    return (
        {'alpha':  pars[:16].reshape((4, 4)),
         'beta':   pars[16:32].reshape((4, 4)),
         'nmean':  pars[32:48].reshape((4, 4)),
         'rhoeps': pars[48:64].reshape((4, 4)),
         'eps':    pars[64:80].reshape((4, 4))},
        {'bv': float(pars[80]), 'bb': float(pars[81]),
         'betarsd': float(pars[82]), 'gamma': float(pars[83])},
    )