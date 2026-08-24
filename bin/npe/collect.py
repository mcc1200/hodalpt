'''
Collect training datasets for NPE.

Usage:
    python collect.py [bias_fid_hod] [bias_fid_nlb] [bias_fid_hod_norsd] \\
                       [bias_fid_nlb_norsd] [bias_lhc_hod] [bias_sobol_alpt]

With no arguments, all six of the above datasets are collected.

bias_fid_*_norsd reads the real-space (rsd=False) spec.noRSD.i.h5 runs from
bias_fiducial_noRSD.py (spectra live alongside the RSD spec.i.h5 files in
the same NLB_bias_dir/HOD_bias_dir, but are collected separately and written
to bias_fid_*_noRSD_data.hdf5 so as not to overwrite the RSD datasets). They
have no 'p2' -- no quadrupole in real space.

Opt-in only (not part of the no-arg default -- request by name):
    python collect.py bias_fid_nlb_norsd_positions bias_fid_hod_norsd_positions

These archive real-space galaxy positions ('xyz', keyed by sample 'idx') to
corral -- large, and only useful for a possible future apply-RSD-in-post-
processing step, so kept out of the routine training-data collection.
'''
import argparse
import h5py
import numpy as np
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Paths ---
WORK = os.environ.get('WORK', '/work/11053/mcasas/ls6')

NLB_bias_dir  = '/corral/utexas/AST25023/simbig/quijote/fiducial_HR/0/bias/NLB'
HOD_bias_dir  = '/corral/utexas/AST25023/simbig/quijote/fiducial_HR/0/bias/HOD'
LHC_base      = '/corral/utexas/AST25023/simbig/quijote/latinhypercube_hr'
sobol_fn      = f'{WORK}/hodalpt/bin/sense/cosmology_alpt_sobol2048.dat'
alpt_prior_fn = f'{WORK}/hodalpt/bin/sense/alpt_nlb_priors_50p.hdf5'
sobol_base    = '/corral/utexas/AST25023/simbig/alpt/sobol'

out_dir      = f'{WORK}/hodalpt/bin/npe'

# real-space galaxy position archives -- corral (bulk storage), not out_dir,
# since these are large and kept only for a possible future apply-RSD-in-
# post-processing use case, not for routine NPE training
NLB_positions_fn = '/corral/utexas/AST25023/simbig/quijote/fiducial_HR/0/bias/NLB_noRSD_positions.hdf5'
HOD_positions_fn = '/corral/utexas/AST25023/simbig/quijote/fiducial_HR/0/bias/HOD_noRSD_positions.hdf5'

kmax         = 1.0
kmin_bispec  = 0.0
k_fund       = 2 * np.pi / 1000.   # fundamental wavenumber for Lbox=1000 Mpc/h
N_LHC     = 400
N_SOBOL   = 2048
N_WORKERS = 16


# ---------------------------------------------------------------------------
# bias_fid_HOD  /  bias_fid_NLB
# ---------------------------------------------------------------------------

def _load_bias_fid(args):
    fn, kmax, kmin_b = args
    with h5py.File(fn, 'r') as f:
        k     = f['k'][:]
        p0    = f['p0'][:]
        p2    = f['p2'][:]
        ngs   = f['ngs'][()]
        theta = f['theta'][:]
        mask  = k <= kmax
        if 'b123' in f:
            i_k1 = f['i_k1'][:] #* k_fund
            i_k2 = f['i_k2'][:] #* k_fund
            i_k3 = f['i_k3'][:] #* k_fund
            k1, k2, k3 = i_k1 * k_fund, i_k2 * k_fund, i_k3 * k_fund
            klim = (k1 > kmin_b) & (k2 > kmin_b) & (k3 > kmin_b) & \
                   (k1 <= kmax)  & (k2 <= kmax)  & (k3 <= kmax)
            b123 = f['b123'][:][klim]
            q123 = f['q123'][:][klim]
        else:
            b123, q123 = None, None
    return k[mask], p0[mask], p2[mask], theta, b123, q123, klim, i_k1, i_k2, i_k3, ngs


def collect_bias_fid(bias_dir, out_fn, label):
    available = sorted(
        [fn for fn in os.listdir(bias_dir) if fn.startswith('spec.') and fn.endswith('.h5')],
        key=lambda fn: int(fn.split('.')[1])
    )
    n = len(available)
    print(f'[{label}] Found {n} spectra in {bias_dir}')

    paths = [(os.path.join(bias_dir, fn), kmax, kmin_bispec) for fn in available]
    results = [None] * n
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_load_bias_fid, args): i for i, args in enumerate(paths)}
        done = 0
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            done += 1
            if done % 500 == 0 or done == n:
                print(f'  {done}/{n}  ({time.time()-t0:.0f}s)')

    k         = results[0][0]
    all_p0    = np.array([r[1] for r in results])
    all_p2    = np.array([r[2] for r in results])
    all_theta = np.array([r[3] for r in results])
    ngs       = np.array([r[10] for r in results])
    n_with_bispec = sum(1 for r in results if r[4] is not None)
    print(f'[{label}] {n_with_bispec}/{n} samples have bispectrum')

    with h5py.File(out_fn, 'w') as f:
        f.create_dataset('theta', data=all_theta)
        f.create_dataset('p0',    data=all_p0)
        f.create_dataset('p2',    data=all_p2)
        f.create_dataset('k',     data=k)
        f.create_dataset('ngs',   data=ngs)
        if n_with_bispec == n:
            f.create_dataset('b123', data=np.array([r[4] for r in results]))
            f.create_dataset('q123', data=np.array([r[5] for r in results]))
            f.create_dataset('klim', data=np.array([r[6] for r in results]))
            f.create_dataset('i_k1', data=np.array([r[7] for r in results]))
            f.create_dataset('i_k2', data=np.array([r[8] for r in results]))
            f.create_dataset('i_k3', data=np.array([r[9] for r in results]))

    print(f'[{label}] Wrote {out_fn}: theta {all_theta.shape}, p0 {all_p0.shape}')


# ---------------------------------------------------------------------------
# bias_fid_HOD_noRSD  /  bias_fid_NLB_noRSD
#
# real-space (rsd=False) runs from bias_fiducial_noRSD.py -- filenames are
# spec.noRSD.i.h5 (not spec.i.h5) so they coexist with the RSD spectra in
# the same NLB_bias_dir/HOD_bias_dir, and save_spectrum() there never writes
# 'p2' (no quadrupole in real space), so it's dropped here rather than read.
# ---------------------------------------------------------------------------

def _load_bias_fid_norsd(args):
    fn, kmax, kmin_b = args
    with h5py.File(fn, 'r') as f:
        k     = f['k'][:]
        p0    = f['p0'][:]
        ngs   = f['ngs'][()]
        theta = f['theta'][:]
        mask  = k <= kmax
        if 'b123' in f:
            i_k1 = f['i_k1'][:] #* k_fund
            i_k2 = f['i_k2'][:] #* k_fund
            i_k3 = f['i_k3'][:] #* k_fund
            k1, k2, k3 = i_k1 * k_fund, i_k2 * k_fund, i_k3 * k_fund
            klim = (k1 > kmin_b) & (k2 > kmin_b) & (k3 > kmin_b) & \
                   (k1 <= kmax)  & (k2 <= kmax)  & (k3 <= kmax)
            b123 = f['b123'][:][klim]
            q123 = f['q123'][:][klim]
        else:
            b123, q123 = None, None
    return k[mask], p0[mask], theta, b123, q123, klim, i_k1, i_k2, i_k3, ngs


def collect_bias_fid_norsd(bias_dir, out_fn, label):
    available = sorted(
        [fn for fn in os.listdir(bias_dir) if fn.startswith('spec.noRSD.') and fn.endswith('.h5')],
        key=lambda fn: int(fn.split('.')[2])
    )
    n = len(available)
    print(f'[{label}] Found {n} spectra in {bias_dir}')

    paths = [(os.path.join(bias_dir, fn), kmax, kmin_bispec) for fn in available]
    results = [None] * n
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_load_bias_fid_norsd, args): i for i, args in enumerate(paths)}
        done = 0
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            done += 1
            if done % 500 == 0 or done == n:
                print(f'  {done}/{n}  ({time.time()-t0:.0f}s)')

    k         = results[0][0]
    all_p0    = np.array([r[1] for r in results])
    all_theta = np.array([r[2] for r in results])
    ngs       = np.array([r[9] for r in results])
    n_with_bispec = sum(1 for r in results if r[3] is not None)
    print(f'[{label}] {n_with_bispec}/{n} samples have bispectrum')

    with h5py.File(out_fn, 'w') as f:
        f.create_dataset('theta', data=all_theta)
        f.create_dataset('p0',    data=all_p0)
        f.create_dataset('k',     data=k)
        f.create_dataset('ngs',   data=ngs)
        if n_with_bispec == n:
            f.create_dataset('b123', data=np.array([r[3] for r in results]))
            f.create_dataset('q123', data=np.array([r[4] for r in results]))
            f.create_dataset('klim', data=np.array([r[5] for r in results]))
            f.create_dataset('i_k1', data=np.array([r[6] for r in results]))
            f.create_dataset('i_k2', data=np.array([r[7] for r in results]))
            f.create_dataset('i_k3', data=np.array([r[8] for r in results]))

    print(f'[{label}] Wrote {out_fn}: theta {all_theta.shape}, p0 {all_p0.shape}')


def _load_xyz_norsd(args):
    fn, i = args
    with h5py.File(fn, 'r') as f:
        xyz = f['xyz'][:]
        ngs = f['ngs'][()]
    return i, ngs, xyz


def collect_positions_norsd(bias_dir, out_fn, label):
    '''Archive real-space galaxy positions on corral, explicitly keyed by
    sample index (not just concatenation order), for applying RSD in
    post-processing later without re-running the pipeline. Kept separate
    from collect_bias_fid_norsd's summary-stat file since positions are
    much larger and are only ever needed for that hypothetical use case --
    not part of the default no-arg collect.py run, request by name.
    '''
    available = sorted(
        [fn for fn in os.listdir(bias_dir) if fn.startswith('spec.noRSD.') and fn.endswith('.h5')],
        key=lambda fn: int(fn.split('.')[2])
    )
    idx = [int(fn.split('.')[2]) for fn in available]
    n = len(available)
    print(f'[{label}] Found {n} spectra in {bias_dir}')

    paths = [(os.path.join(bias_dir, fn), i) for fn, i in zip(available, idx)]
    results = [None] * n
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_load_xyz_norsd, args): k for k, args in enumerate(paths)}
        done = 0
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            done += 1
            if done % 500 == 0 or done == n:
                print(f'  {done}/{n}  ({time.time()-t0:.0f}s)')

    all_idx = np.array([r[0] for r in results])
    ngs     = np.array([r[1] for r in results])
    # ngs varies per sample, so xyz can't stack into one fixed-shape array --
    # concatenate into one flat (sum(ngs), 3) array; reconstruct sample i's
    # positions via offsets = np.concatenate([[0], np.cumsum(ngs)]) matched
    # against all_idx (NOT positional order -- indices can have gaps if run
    # against a partial/in-progress data directory), i.e.
    # xyz[offsets[j]:offsets[j+1]] for all_idx[j] == i.
    all_xyz = np.concatenate([r[2] for r in results], axis=0)

    with h5py.File(out_fn, 'w') as f:
        f.create_dataset('idx', data=all_idx)
        f.create_dataset('ngs', data=ngs)
        f.create_dataset('xyz', data=all_xyz)

    print(f'[{label}] Wrote {out_fn}: idx {all_idx.shape}, ngs {ngs.shape}, xyz {all_xyz.shape}')


# ---------------------------------------------------------------------------
# bias_lhc_hod
# ---------------------------------------------------------------------------

def _load_lhc_hod(args):
    idx, base, kmax = args
    spec_fn = f'{base}/{idx}/spec.hdf5'
    with h5py.File(spec_fn, 'r') as f:
        k     = f['k'][:]
        p0    = f['p0k'][:]
        p2    = f['p2k'][:]
        theta = np.concatenate([f['cosmo'][:], f['hod'][:]])  # (5,) + (9,) -> (14,)
        mask  = k <= kmax
    return idx, k[mask], p0[mask], p2[mask], theta


def collect_lhc_hod(out_fn):
    label = 'bias_lhc_hod'
    print(f'[{label}] Collecting {N_LHC} LHC spectra from {LHC_base}')

    results = {}
    skipped = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_load_lhc_hod, (idx, LHC_base, kmax)): idx for idx in range(N_LHC)}
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                res = fut.result()
                results[res[0]] = res
            except Exception as e:
                print(f'  Skipping idx={idx}: {e}')
                skipped.append(idx)
            done += 1
            if done % 50 == 0 or done == N_LHC:
                print(f'  {done}/{N_LHC}  ({time.time()-t0:.0f}s)')

    good = sorted(results.keys())
    k         = results[good[0]][1]
    all_theta = np.array([results[i][4] for i in good])
    all_p0    = np.array([results[i][2] for i in good])
    all_p2    = np.array([results[i][3] for i in good])

    with h5py.File(out_fn, 'w') as f:
        f.create_dataset('theta', data=all_theta)
        f.create_dataset('p0',    data=all_p0)
        f.create_dataset('p2',    data=all_p2)
        f.create_dataset('k',     data=k)

    print(f'[{label}] Wrote {out_fn}: theta {all_theta.shape}, p0 {all_p0.shape}')
    if skipped:
        print(f'[{label}] Skipped: {sorted(skipped)}')


# ---------------------------------------------------------------------------
# bias_sobol_alpt
# ---------------------------------------------------------------------------

def _load_sobol_pair(args):
    i_sobol, sobol_base, alpt_prior_fn, cosmopars_row, kmax = args

    def get_alpt_prior(i_alpt):
        with h5py.File(alpt_prior_fn, 'r') as f:
            alpha = f[f'samples/{i_alpt}/alpha'][()].flatten()
            beta  = f[f'samples/{i_alpt}/beta'][()].flatten()
            nmean = f[f'samples/{i_alpt}/nmean'][()].flatten()
            theta_rsd_grp = f[f'samples/{i_alpt}/theta_rsd']
            rsd_keys = ['bb', 'betarsd', 'bv', 'gamma']
            theta_rsd = np.array([theta_rsd_grp[key][()] for key in rsd_keys])
        return np.concatenate([alpha, beta, nmean, theta_rsd])

    spec0 = f'{sobol_base}/{i_sobol}/bias/0/spec.hdf5'
    spec1 = f'{sobol_base}/{i_sobol}/bias/1/spec.hdf5'

    theta_alpt0 = get_alpt_prior(2 * i_sobol)
    theta_alpt1 = get_alpt_prior(2 * i_sobol + 1)
    theta0 = np.concatenate([cosmopars_row, theta_alpt0])
    theta1 = np.concatenate([cosmopars_row, theta_alpt1])

    def read_spec(fn):
        with h5py.File(fn, 'r') as f:
            k  = f['k'][:]
            p0 = f['p0k'][:]
            p2 = f['p2k'][:]
            mask = k <= kmax
        return k[mask], p0[mask], p2[mask]

    k, p0_0, p2_0 = read_spec(spec0)
    _, p0_1, p2_1 = read_spec(spec1)

    return i_sobol, k, (theta0, p0_0, p2_0), (theta1, p0_1, p2_1)


def collect_sobol_alpt(out_fn):
    label = 'bias_sobol_alpt'
    cosmopars = np.loadtxt(sobol_fn)
    print(f'[{label}] Collecting {N_SOBOL}×2 sobol spectra')

    all_theta, all_p0, all_p2 = [], [], []
    skipped = []
    t0 = time.time()

    for i_sobol in range(N_SOBOL):
        try:
            _, k, (th0, p0_0, p2_0), (th1, p0_1, p2_1) = _load_sobol_pair(
                (i_sobol, sobol_base, alpt_prior_fn, cosmopars[i_sobol], kmax)
            )
            all_theta.extend([th0, th1])
            all_p0.extend([p0_0, p0_1])
            all_p2.extend([p2_0, p2_1])
        except Exception as e:
            print(f'  Skipping i_sobol={i_sobol}: {e}')
            skipped.append(i_sobol)
        if (i_sobol + 1) % 200 == 0 or i_sobol + 1 == N_SOBOL:
            print(f'  {i_sobol+1}/{N_SOBOL}  ({time.time()-t0:.0f}s)')

    all_theta = np.array(all_theta)
    all_p0    = np.array(all_p0)
    all_p2    = np.array(all_p2)

    with h5py.File(out_fn, 'w') as f:
        f.create_dataset('theta', data=all_theta)
        f.create_dataset('p0',    data=all_p0)
        f.create_dataset('p2',    data=all_p2)
        f.create_dataset('k',     data=k)

    print(f'[{label}] Wrote {out_fn}: theta {all_theta.shape}, p0 {all_p0.shape}')
    if skipped:
        print(f'[{label}] Skipped i_sobol: {sorted(skipped)}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_COLLECTORS = {
    'bias_fid_hod':        lambda: collect_bias_fid(HOD_bias_dir,  f'{out_dir}/bias_fid_HOD_data.hdf5',  'bias_fid_hod'),
    'bias_fid_nlb':        lambda: collect_bias_fid(NLB_bias_dir,  f'{out_dir}/bias_fid_NLB_data.hdf5',  'bias_fid_nlb'),
    'bias_fid_hod_norsd':  lambda: collect_bias_fid_norsd(HOD_bias_dir, f'{out_dir}/bias_fid_HOD_noRSD_data.hdf5', 'bias_fid_hod_norsd'),
    'bias_fid_nlb_norsd':  lambda: collect_bias_fid_norsd(NLB_bias_dir, f'{out_dir}/bias_fid_NLB_noRSD_data.hdf5', 'bias_fid_nlb_norsd'),
    'bias_lhc_hod':        lambda: collect_lhc_hod(               f'{out_dir}/bias_lhc_hod_data.hdf5'),
    'bias_sobol_alpt':     lambda: collect_sobol_alpt(            f'{out_dir}/bias_sobol_alpt_data.hdf5'),
}

# opt-in only -- large corral archives for the just-in-case apply-RSD-later
# path, not needed for routine NPE training, so excluded from the default
# no-arg run (must be requested by name)
OPTIN_COLLECTORS = {
    'bias_fid_nlb_norsd_positions': lambda: collect_positions_norsd(NLB_bias_dir, NLB_positions_fn, 'bias_fid_nlb_norsd_positions'),
    'bias_fid_hod_norsd_positions': lambda: collect_positions_norsd(HOD_bias_dir, HOD_positions_fn, 'bias_fid_hod_norsd_positions'),
}

COLLECTORS = {**DEFAULT_COLLECTORS, **OPTIN_COLLECTORS}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect NPE training datasets.')
    parser.add_argument('datasets', nargs='*', choices=list(COLLECTORS), default=list(DEFAULT_COLLECTORS),
                        help='Which datasets to collect (default: all except the opt-in position archives)')
    args = parser.parse_args()

    for name in args.datasets:
        print(f'\n=== {name} ===')
        COLLECTORS[name]()
