'''
delete_bad_specs.py

Delete unopenable/corrupt spec.[noRSD.]<i>.h5 files ahead of a --skip-done
resubmission -- not required (bias_fiducial_noRSD.py's h5py.File(fname, 'w')
overwrites unconditionally, and --skip-done already re-checks completeness
itself), but useful for reclaiming whatever block space they hold and for a
clean directory listing.

Takes a plain-text list of full file paths, one per line -- e.g. the
*.skipped.txt collect.py writes (bias_fid_NLB_noRSD_data.hdf5.skipped.txt),
or a bad-idx file from check_spectra_noRSD.py --write-bad-idx (in which case
pass --dir and --pattern so indices can be turned into paths).

Usage:
    python delete_bad_specs.py bias_fid_NLB_noRSD_data.hdf5.skipped.txt
        # dry run -- lists what would be deleted

    python delete_bad_specs.py bias_fid_NLB_noRSD_data.hdf5.skipped.txt --force
        # actually deletes

    python delete_bad_specs.py nlb_bad_idx_9820_20000.txt \\
        --dir /corral/utexas/AST25023/simbig/quijote/fiducial_HR/0/bias/NLB \\
        --pattern spec.noRSD. --force
        # same, but input file is bare indices (one per line) rather than paths
'''
import os
import sys
import argparse

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('list_file', help='file of full paths, or bare indices with --dir/--pattern')
    p.add_argument('--dir', default=None, help='bias dir, if list_file contains bare indices')
    p.add_argument('--pattern', default='spec.noRSD.', help='filename prefix, used with --dir')
    p.add_argument('--force', action='store_true', help='actually delete (default: dry run)')
    args = p.parse_args()

    with open(args.list_file) as f:
        lines = [line.strip() for line in f if line.strip()]

    if args.dir:
        targets = [os.path.join(args.dir, f'{args.pattern}{i}.h5') for i in lines]
    else:
        targets = lines

    existing = [fn for fn in targets if os.path.exists(fn)]
    print(f'{len(existing)} / {len(targets)} listed files exist')

    if not args.force:
        print('dry run -- rerun with --force to delete')
        for fn in existing[:10]:
            print(f'  would delete: {fn}')
        if len(existing) > 10:
            print(f'  ... and {len(existing) - 10} more')
        sys.exit(0)

    for fn in existing:
        os.remove(fn)
    print(f'deleted {len(existing)} files')
