'''
Collect the LHC real-space (noRSD) HOD galaxy spectra into one dataset.

Usage:
    python collect.py

Reads <LHC_base>/<i_LHC>/bias/HOD/spec.noRSD.<i_LHC>.h5 (written by
rockstar_hod_LHC.py) for every LHC index that
has one (indices are not sequential) and writes bias_LHC_HOD_data.hdf5 next
to this script. Rows are self-consistent: a file that fails to read is dropped
entirely, so theta / p0 / b123 / cic_pdf / idx always line up row-for-row.
Row order follows ascending 'idx'.
'''
import glob
import h5py
import numpy as np
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

LHC_base = '/corral/utexas/AST25023/simbig/quijote/latinhypercube_hr'
out_dir  = os.path.dirname(os.path.abspath(__file__))
out_fn   = f'{out_dir}/bias_LHC_HOD_data.hdf5'

kmax        = 1.0
kmin_bispec = 0.0
k_fund      = 2 * np.pi / 1000.   # fundamental wavenumber for Lbox=1000 Mpc/h
N_WORKERS   = 16


def _lhc_idx(fn):
    return int(os.path.basename(fn).split('.')[2])


def _load(args):
    fn, kmax, kmin_b = args
    with h5py.File(fn, 'r') as f:
        k     = f['k'][:]
        p0    = f['p0'][:]
        ngs   = f['ngs'][()]
        theta = f['theta'][:]
        cosmo = f['cosmo'][:]
        nmodes    = f['nmodes'][:]
        shotnoise = f['shotnoise'][()]
        mask  = k <= kmax
        if 'b123' in f:
            i_k1 = f['i_k1'][:]
            i_k2 = f['i_k2'][:]
            i_k3 = f['i_k3'][:]
            k1, k2, k3 = i_k1 * k_fund, i_k2 * k_fund, i_k3 * k_fund
            klim = (k1 > kmin_b) & (k2 > kmin_b) & (k3 > kmin_b) & \
                   (k1 <= kmax)  & (k2 <= kmax)  & (k3 <= kmax)
            b123 = f['b123'][:][klim]
            q123 = f['q123'][:][klim]
        else:
            b123, q123, klim, i_k1, i_k2, i_k3 = None, None, None, None, None, None
        if 'cic_pdf' in f:
            cic_pdf       = f['cic_pdf'][:]
            cic_pdf_edges = f['cic_pdf_edges'][:]
        else:
            cic_pdf, cic_pdf_edges = None, None
    return (k[mask], p0[mask], theta, b123, q123, klim, i_k1, i_k2, i_k3, ngs, cic_pdf, cic_pdf_edges,
            cosmo, nmodes[mask], shotnoise)


def collect_lhc(out_fn, label='bias_LHC_HOD'):
    files = sorted(glob.glob(f'{LHC_base}/*/bias/HOD/spec.noRSD.*.h5'), key=_lhc_idx)
    idx = [_lhc_idx(fn) for fn in files]
    n = len(files)
    print(f'[{label}] Found {n} spectra in {LHC_base}')

    paths = [(fn, kmax, kmin_bispec) for fn in files]
    results = [None] * n
    t0 = time.time()

    skipped = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_load, args): i for i, args in enumerate(paths)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                print(f'  Skipping idx={idx[i]}: {e}')
                skipped.append(i)
            done += 1
            if done % 100 == 0 or done == n:
                print(f'  {done}/{n}  ({time.time()-t0:.0f}s, {len(skipped)} skipped)')

    # drop failed files from every array at once so rows stay aligned
    good = [i for i in range(n) if results[i] is not None]
    results = [results[i] for i in good]
    all_idx   = np.array([idx[i] for i in good])
    k         = results[0][0]
    all_p0    = np.array([r[1] for r in results])
    all_theta = np.array([r[2] for r in results])
    ngs       = np.array([r[9] for r in results])
    all_cosmo     = np.array([r[12] for r in results])
    all_nmodes    = np.array([r[13] for r in results])
    all_shotnoise = np.array([r[14] for r in results])
    n_with_bispec = sum(1 for r in results if r[3] is not None)
    n_with_cic    = sum(1 for r in results if r[10] is not None)
    print(f'[{label}] {n_with_bispec}/{len(results)} samples have bispectrum')
    print(f'[{label}] {n_with_cic}/{len(results)} samples have cic_pdf')
    if skipped:
        print(f'[{label}] Skipped idx: {sorted(idx[i] for i in skipped)}')

    with h5py.File(out_fn, 'w') as f:
        f.create_dataset('idx',   data=all_idx)
        f.create_dataset('theta', data=all_theta)
        f.create_dataset('cosmo', data=all_cosmo)
        f.create_dataset('p0',    data=all_p0)
        f.create_dataset('k',     data=k)
        f.create_dataset('nmodes',    data=all_nmodes)
        f.create_dataset('shotnoise', data=all_shotnoise)
        f.create_dataset('ngs',   data=ngs)
        if n_with_bispec == len(results):
            f.create_dataset('b123', data=np.array([r[3] for r in results]))
            f.create_dataset('q123', data=np.array([r[4] for r in results]))
            f.create_dataset('klim', data=np.array([r[5] for r in results]))
            f.create_dataset('i_k1', data=np.array([r[6] for r in results]))
            f.create_dataset('i_k2', data=np.array([r[7] for r in results]))
            f.create_dataset('i_k3', data=np.array([r[8] for r in results]))
        else:
            print(f'[{label}] Skipping bispectrum datasets ({n_with_bispec}/{len(results)} present)')
        if n_with_cic == len(results):
            f.create_dataset('cic_pdf',       data=np.array([r[10] for r in results]))
            f.create_dataset('cic_pdf_edges', data=results[0][11])
        else:
            print(f'[{label}] Skipping cic_pdf dataset ({n_with_cic}/{len(results)} present)')

    print(f'[{label}] Wrote {out_fn}: theta {all_theta.shape}, p0 {all_p0.shape}')


if __name__ == '__main__':
    collect_lhc(out_fn)
