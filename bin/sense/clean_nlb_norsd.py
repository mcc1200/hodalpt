'''
clean_nlb_norsd.py

Run this ON TACC before resubmitting bias_fiducial_noRSD_pylauncher.py, to
clear out old NLB spec.noRSD.i.h5 files sampled under the previous (partial)
priors.sample_bias_realspace, so the rerun can't be mistaken for a
--skip-done "already complete" file.

HOD files are untouched -- HOD spectra don't depend on the bias priors and
are not being regenerated (see bias_fiducial_noRSD.py).

Usage:
    python clean_nlb_norsd.py 0 10000            # dry run, prints what it would delete
    python clean_nlb_norsd.py 0 10000 --force    # actually deletes
'''
import os
import sys

NLB_DIR = '/corral/utexas/AST25023/simbig/quijote/fiducial_HR/0/bias/NLB'

if __name__ == '__main__':
    i0 = int(sys.argv[1])
    i1 = int(sys.argv[2])
    force = '--force' in sys.argv

    targets = [os.path.join(NLB_DIR, 'spec.noRSD.%i.h5' % i) for i in range(i0, i1)]
    existing = [fn for fn in targets if os.path.exists(fn)]

    print('%i / %i files exist in [%i, %i)' % (len(existing), len(targets), i0, i1))

    if not force:
        print('dry run -- rerun with --force to delete')
        sys.exit(0)

    for fn in existing:
        os.remove(fn)
    print('deleted %i files' % len(existing))
