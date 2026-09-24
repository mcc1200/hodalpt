'''
strip_xyz.py

Reclaim corral quota from already-written spec.[noRSD.]<i>.h5 files: 'xyz'
(the galaxy position array) is ~95% of every file's size and, as of the
save_spectrum() fix in bias_fiducial_noRSD.py / bias_fiducial.py, is no
longer written to new files -- this strips it out of files that predate
that fix.

Each file is rewritten to a temp path with every dataset/attr EXCEPT 'xyz',
then os.replace()'d over the original (atomic, and the only extra headroom
needed at any moment is one small temp file, not a full extra copy).
Deleting the 'xyz' dataset in place instead (h5py `del f['xyz']`) would NOT
reclaim space -- HDF5 doesn't shrink a file on dataset deletion without a
separate repack, so a full rewrite is the correct approach here.

Files that error out (corrupt/truncated -- the ones already being tracked
in *.skipped.txt) are left untouched and reported, not deleted.

Usage:
    python strip_xyz.py <dir> [<dir> ...] [--pattern spec.] [--workers 16]
    python strip_xyz.py <dir> --dry-run     # report reclaimable space only

Typical:
    python strip_xyz.py \\
        /corral/utexas/AST25023/simbig/quijote/fiducial_HR/0/bias/NLB \\
        /corral/utexas/AST25023/simbig/quijote/fiducial_HR/0/bias/HOD \\
        --dry-run
'''
import os
import sys
import time
import argparse
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed

N_WORKERS = 16


def _has_xyz(fn):
    '''Cheap check for dry-run: does fn contain 'xyz'? Returns (has_xyz, size) or None on error.'''
    try:
        with h5py.File(fn, 'r') as f:
            has = 'xyz' in f
        return (has, os.path.getsize(fn))
    except Exception:
        return None


def _strip_one(fn):
    '''Rewrite fn without 'xyz'. Returns (status, bytes_freed).
    status is one of 'stripped', 'no_xyz', or an exception message.'''
    try:
        with h5py.File(fn, 'r') as f:
            if 'xyz' not in f:
                return ('no_xyz', 0)
            data  = {k: f[k][()] for k in f.keys() if k != 'xyz'}
            attrs = dict(f.attrs)
        orig_size = os.path.getsize(fn)

        tmp_fn = fn + '.striptmp'
        with h5py.File(tmp_fn, 'w') as f:
            for k, v in data.items():
                f[k] = v
            for k, v in attrs.items():
                f.attrs[k] = v
        os.replace(tmp_fn, fn)
        return ('stripped', orig_size - os.path.getsize(fn))
    except Exception as e:
        tmp_fn = fn + '.striptmp'
        if os.path.exists(tmp_fn):
            os.remove(tmp_fn)
        return (str(e), 0)


def _find_files(dirs, pattern):
    files = []
    for d in dirs:
        files += [os.path.join(d, fn) for fn in os.listdir(d)
                  if fn.startswith(pattern) and fn.endswith('.h5')]
    return files


def dry_run(files):
    print(f'checking {len(files)} files for xyz...')
    t0 = time.time()
    n_xyz, bytes_with_xyz, n_err = 0, 0, 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = [pool.submit(_has_xyz, fn) for fn in files]
        for i, fut in enumerate(as_completed(futures)):
            res = fut.result()
            if res is None:
                n_err += 1
            else:
                has, size = res
                if has:
                    n_xyz += 1
                    bytes_with_xyz += size
            if (i + 1) % 2000 == 0 or i + 1 == len(files):
                print(f'  {i+1}/{len(files)}  ({time.time()-t0:.0f}s)')
    print(f'{n_xyz}/{len(files)} files still have xyz  '
          f'(~{bytes_with_xyz/1e9:.1f} GB on disk -- reclaimable is close to '
          f'this since xyz dominates file size)')
    print(f'{n_err} files could not be opened (left untouched either way)')


def strip(files):
    print(f'stripping xyz from {len(files)} files...')
    t0 = time.time()
    n_stripped, n_no_xyz, bytes_freed = 0, 0, 0
    errors = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_strip_one, fn): fn for fn in files}
        for i, fut in enumerate(as_completed(futures)):
            status, freed = fut.result()
            if status == 'stripped':
                n_stripped += 1
                bytes_freed += freed
            elif status == 'no_xyz':
                n_no_xyz += 1
            else:
                errors.append((futures[fut], status))
            if (i + 1) % 2000 == 0 or i + 1 == len(files):
                print(f'  {i+1}/{len(files)}  ({time.time()-t0:.0f}s, '
                      f'{bytes_freed/1e9:.1f} GB freed so far)')

    print(f'stripped {n_stripped} files, freed {bytes_freed/1e9:.1f} GB')
    print(f'{n_no_xyz} files already had no xyz (untouched)')
    if errors:
        log_fn = 'strip_xyz_errors.txt'
        with open(log_fn, 'w') as f:
            for fn, msg in errors:
                f.write(f'{fn}\t{msg}\n')
        print(f'{len(errors)} files errored and were left untouched -- see {log_fn}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dirs', nargs='+', help='directories to scan (e.g. the NLB and HOD bias dirs)')
    p.add_argument('--pattern', default='spec.',
                    help="filename prefix to match (default 'spec.' -- catches both "
                         "spec.<i>.h5 and spec.noRSD.<i>.h5)")
    p.add_argument('--workers', type=int, default=N_WORKERS)
    p.add_argument('--dry-run', action='store_true',
                    help='report how many files still have xyz and roughly how much '
                         'space is reclaimable, without touching anything')
    args = p.parse_args()
    N_WORKERS = args.workers

    files = _find_files(args.dirs, args.pattern)
    if not files:
        print('no matching files found')
        sys.exit(0)

    if args.dry_run:
        dry_run(files)
    else:
        strip(files)
