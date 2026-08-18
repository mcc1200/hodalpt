'''
check_spectra_noRSD.py

Scan bias_fiducial_noRSD.py output for missing, corrupt, or truncated h5
files -- e.g. after a pylauncher job hit its SLURM walltime mid-run and left
some spec.noRSD.i.h5 files partially written.

Classifies every expected index (NLB: [i0, i1); HOD: [i0, min(i1, 1000)))
as one of:
    ok        -- opens fine, has every key save_spectrum() writes
    missing   -- file does not exist
    corrupt   -- exists but h5py can't open it (killed mid-write, truncated
                 HDF5 container)
    truncated -- opens fine but missing one or more expected keys (killed
                 after the file was created but before all datasets/attrs
                 were written)

Usage:
    python check_spectra_noRSD.py 0 1000
    python check_spectra_noRSD.py 0 1000 --write-bad-idx     # also write
        nlb_bad_idx_0_1000.txt / hod_bad_idx_0_1000.txt listing indices
        that need a re-run (missing + corrupt + truncated)

Note: bias_fiducial_noRSD_pylauncher.py's --skip-done already re-checks
every index this same way before resubmitting, so you don't strictly need
to run this before relaunching -- it's here so you can see the scope of the
damage and sanity-check --skip-done's behavior.
'''
import os
import sys
import h5py

NLB_DIR = '/corral/utexas/AST25023/simbig/quijote/fiducial_HR/0/bias/NLB'
HOD_DIR = '/corral/utexas/AST25023/simbig/quijote/fiducial_HR/0/bias/HOD'
N_HOD   = 1000

# every key save_spectrum() writes, in write order
SPEC_KEYS = ['theta', 'ngs', 'xyz', 'k', 'p0', 'nmodes', 'shotnoise',
             'i_k1', 'i_k2', 'i_k3', 'b123', 'q123']


def classify(fn):
    '''Return one of 'ok', 'missing', 'corrupt', 'truncated' for a spec file.'''
    if not os.path.exists(fn):
        return 'missing'
    try:
        with h5py.File(fn, 'r') as f:
            missing_keys = [k for k in SPEC_KEYS if k not in f]
            if missing_keys:
                return 'truncated'
            return 'ok'
    except Exception:
        return 'corrupt'


def scan(dirpath, indices):
    results = {'ok': [], 'missing': [], 'corrupt': [], 'truncated': []}
    for i in indices:
        fn = os.path.join(dirpath, 'spec.noRSD.%i.h5' % i)
        results[classify(fn)].append(i)
    return results


def _summarize(name, results, n_total):
    print('%s: %i/%i ok' % (name, len(results['ok']), n_total))
    for status in ('missing', 'corrupt', 'truncated'):
        idx = results[status]
        if idx:
            preview = idx[:10] + (['...'] if len(idx) > 10 else [])
            print('  %-9s %4i  e.g. %s' % (status, len(idx), preview))


if __name__ == '__main__':
    i0 = int(sys.argv[1])
    i1 = int(sys.argv[2])
    write_bad_idx = '--write-bad-idx' in sys.argv

    nlb_indices = list(range(i0, i1))
    hod_indices = list(range(i0, min(i1, N_HOD)))

    nlb_results = scan(NLB_DIR, nlb_indices)
    hod_results = scan(HOD_DIR, hod_indices) if hod_indices else None

    _summarize('NLB', nlb_results, len(nlb_indices))
    if hod_results is not None:
        _summarize('HOD', hod_results, len(hod_indices))

    if write_bad_idx:
        nlb_bad = sorted(nlb_results['missing'] + nlb_results['corrupt'] + nlb_results['truncated'])
        with open('nlb_bad_idx_%i_%i.txt' % (i0, i1), 'w') as f:
            f.write('\n'.join(map(str, nlb_bad)))
        print('wrote %i bad NLB indices to nlb_bad_idx_%i_%i.txt' % (len(nlb_bad), i0, i1))

        if hod_results is not None:
            hod_bad = sorted(hod_results['missing'] + hod_results['corrupt'] + hod_results['truncated'])
            with open('hod_bad_idx_%i_%i.txt' % (i0, i1), 'w') as f:
                f.write('\n'.join(map(str, hod_bad)))
            print('wrote %i bad HOD indices to hod_bad_idx_%i_%i.txt' % (len(hod_bad), i0, i1))
