'''
alpt_sense.py: run ALPT (DM field, Tweb, velocities, subgrid positions) for one
Quijote LHC realization from its ICs. Launched in parallel by
alpt_sense_pylauncher.py, which also uses the completeness checks below.

    python alpt_sense.py <i_LHC>

Writes to <LHC_ROOT>/<i_LHC>/alpt/. Exits non-zero (so failures are visible in
the logs) if the ICs are missing or the ALPT outputs are incomplete afterward.
'''
import os
import sys

LHC_ROOT = '/corral/utexas/AST25023/simbig/quijote/latinhypercube_hr'
NGRID = 256
FILE_BYTES = NGRID ** 3 * 4      # every ALPT .DAT/.dat field is float32 on the grid

DELTA_IC = 'Quijote_ICs_delta_z127_n256_CIC.DAT'

# every file CSbox_galaxy() reads for subgrid=True, zsnap=0.5, plus 2LPT.param
# (copied into alpt/ by CSbox_alpt_Q). Field files must be exactly FILE_BYTES,
# which catches runs killed mid-write.
DM_FILES = [
    'super_deltaBOX.dat', 'Tweb_.dat', 'TwebDelta_.dat',
    'VExEULz0.500.dat', 'VEyEULz0.500.dat', 'VEzEULz0.500.dat',
    'super_BOXposx.dat', 'super_BOXposy.dat', 'super_BOXposz.dat',
]
PARAM_FILE = '2LPT.param'

# Quijote LHC ICs: 8 hdf5 files of ~1.9 GB each; anything far below that is a
# truncated transfer. (Sized from LHC 0.)
N_ICS = 8
MIN_ICS_BYTES = int(1.5e9)


def _size_ok(path, nbytes=None):
    try:
        s = os.path.getsize(path)
    except OSError:
        return False
    return s == nbytes if nbytes is not None else s > 0


def alpt_missing(i_lhc, root=LHC_ROOT):
    '''Names of ALPT outputs that are missing or the wrong size ([] if complete).'''
    d = os.path.join(root, str(i_lhc), 'alpt')
    bad = [fn for fn in DM_FILES if not _size_ok(os.path.join(d, fn), FILE_BYTES)]
    if not _size_ok(os.path.join(d, PARAM_FILE)):
        bad.append(PARAM_FILE)
    return bad


def alpt_complete(i_lhc, root=LHC_ROOT):
    return not alpt_missing(i_lhc, root)


def delta_ok(i_lhc, root=LHC_ROOT):
    '''True if the gridded z=127 delta IC is present, so the ICs are not needed.'''
    return _size_ok(os.path.join(root, str(i_lhc), 'alpt', DELTA_IC), FILE_BYTES)


def ics_missing(i_lhc, root=LHC_ROOT):
    '''Problems with the inputs needed to run ALPT from the ICs ([] if all present).'''
    d = os.path.join(root, str(i_lhc))
    bad = ['ICs/ics.%i.hdf5' % j for j in range(N_ICS)
           if not _size_ok(os.path.join(d, 'ICs', 'ics.%i.hdf5' % j), None)
           or os.path.getsize(os.path.join(d, 'ICs', 'ics.%i.hdf5' % j)) < MIN_ICS_BYTES]
    for fn in ('ICs/' + PARAM_FILE, 'Cosmo_params.dat'):
        if not _size_ok(os.path.join(d, fn)):
            bad.append(fn)
    return bad


if __name__ == '__main__':
    from hodalpt.sims import alpt as CS

    i_lhc = int(sys.argv[1])
    ic_path = os.path.join(LHC_ROOT, str(i_lhc), 'ICs')
    outdir = os.path.join(LHC_ROOT, str(i_lhc), 'alpt')

    print('ALPT running for LHC%i' % i_lhc)
    if alpt_complete(i_lhc):
        print('LHC%i ALPT outputs already complete, skipping' % i_lhc)
        sys.exit(0)

    # grid the ICs only if the delta IC is not already there (needs the heavy ICs)
    make_ics = not delta_ok(i_lhc)
    if make_ics:
        bad = ics_missing(i_lhc)
        if bad:
            print('LHC%i missing/truncated inputs: %s' % (i_lhc, bad))
            sys.exit(1)

    CS.CSbox_alpt_Q(ic_path, outdir, seed=0, dgrowth_short=5., ngrid=NGRID,
                    make_ics=make_ics, subgrid=True, return_pos=False, silent=True)

    # webonx failures do not raise in CSbox_alpt_Q, so verify the outputs here
    bad = alpt_missing(i_lhc)
    if bad:
        print('LHC%i ALPT incomplete, missing/wrong size: %s' % (i_lhc, bad))
        sys.exit(1)
    print('LHC%i ALPT done' % i_lhc)
