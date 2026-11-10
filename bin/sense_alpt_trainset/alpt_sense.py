'''
alpt_sense.py: run ALPT (DM field, Tweb, velocities, subgrid positions) for a 
given cosmology sampled from q_phi(cosmo|ngal). Launched in parallel by
alpt_sense_pylauncher.py, which also uses the completeness checks below.

    python alpt_sense.py <idx_cosmo> [--dm | --bias] [--overwrite]

idx_cosmo is a row of cosmo_samples.npy. Writes to <SENSE_ROOT>/<idx_cosmo>/alpt/.
Exits non-zero (so failures are visible in the logs) if the ALPT outputs are
incomplete after the DM stage.
'''
import os
import sys
from hodalpt.sims import alpt as CS
from hodalpt import stats
import numpy as np
from pathlib import Path
import json, h5py
from nbodykit.lab import ArrayCatalog, FFTPower

SENSE_ROOT = Path('/corral/utexas/AST25023/simbig/alpt/sense')
SCRIPTDIR = Path(__file__).resolve().parent

COSMOFN = SCRIPTDIR / 'cosmo_samples.npy'  # rows: Om, Ob, h, n_s, s8; row index = idx_cosmo

SEED = 20261110  # IC seed base; realization idx_cosmo uses SEED + idx_cosmo
NGRID, LBOX, ZSNAP = 256, 1000., 0.5


DM_FILES = [
    'super_deltaBOX.dat', 'Tweb_.dat', 'TwebDelta_.dat',
    'VExEULz0.500.dat', 'VEyEULz0.500.dat', 'VEzEULz0.500.dat',
    'super_BOXposx.dat', 'super_BOXposy.dat', 'super_BOXposz.dat',
]
NLB_KEYS = ('alpha', 'beta', 'nmean', 'rhoeps', 'eps')

def alpt_sense_paths(idx_cosmo, root=SENSE_ROOT):
    """All per-realization paths for a given LHC index."""
    d = root / str(idx_cosmo)
    return dict(
        dm_dir     = d / 'alpt',
        outdir_NLB = d / 'bias',
        fname_NLB  = d / 'bias' / f'spec.noRSD.{idx_cosmo}.h5',
    )

def alpt_missing(idx_cosmo, root=SENSE_ROOT):
    """DM_FILES that are absent or the wrong size in the ALPT output dir (empty list = complete).

    The gridded fields (delta, Tweb, TwebDelta, velocities) must be exactly
    NGRID^3 float32. The subgrid position files only need to be non-empty float32
    arrays, since their length is set by the subgrid model."""
    dm_dir = alpt_sense_paths(idx_cosmo, root)['dm_dir']
    ngrid_bytes = NGRID**3 * 4
    bad = []
    for fn in DM_FILES:
        f = dm_dir / fn
        if not f.is_file():
            bad.append(fn)
            continue
        size = f.stat().st_size
        if fn.startswith('super_BOXpos'):
            ok = size > 0 and size % 4 == 0
        else:
            ok = size == ngrid_bytes
        if not ok:
            bad.append(f'{fn} (bad size {size})')
    return bad

def alpt_complete(idx_cosmo, root=SENSE_ROOT):
    return not alpt_missing(idx_cosmo, root)

def n_cosmo():
    """Number of cosmologies in cosmo_samples.npy."""
    return np.load(COSMOFN, mmap_mode='r').shape[0]

def save_spectrum(fname, xyz, theta, cosmo):
    """Save FFTPower multipoles to HDF5."""

    ######## P(k) #################################
    cat = ArrayCatalog({'Position': xyz}, BoxSize=1000.)
    r   = FFTPower(cat, mode='2d', Nmesh=256, dk=0.005, kmin=0.008,
               Nmu=10, los=[0,0,1], poles=[0])

    poles = r.poles
    k      = poles['k']
    p0     = poles['power_0'].real - poles.attrs['shotnoise']
    # p2     = poles['power_2'].real
    nmodes = poles['modes']
    ######## PDF ###################################
    Nmesh = 40
    BoxSize=1000.
    bins = np.arange(-0.5, 85.5, 1)

    mesh = cat.to_mesh(Nmesh=Nmesh, BoxSize=BoxSize, resampler='cic',
                        compensated=False, interlaced=False)
    field = mesh.paint(mode='real')
    mean_n = cat.csize / (BoxSize ** 3)
    cell_vol = (BoxSize / Nmesh) ** 3
    counts = field.value.ravel() * mean_n * cell_vol
    pdf, edges = np.histogram(counts, bins=bins, density=False)
    pdf = pdf / pdf.sum()

    ######### B(k) #################################
    bispec = stats.B0_periodic(xyz.T, w=None, Lbox=1000., fft='pyfftw', silent=True)

    with h5py.File(fname, 'w') as f:
        f['theta']    = theta
        f['cosmo']    = cosmo
        f['ngs']      = xyz.shape[0]
        f['k']        = k
        f['p0']       = p0
        f['nmodes']   = nmodes
        f['shotnoise'] = poles.attrs['shotnoise']
        f['cic_pdf']  = pdf
        f['cic_pdf_edges'] = edges
        f['i_k1']     = bispec['i_k1']
        f['i_k2']     = bispec['i_k2']
        f['i_k3']     = bispec['i_k3']
        f['b123']     = bispec['b123']
        f['q123']     = bispec['q123']
        # save useful metadata
        f.attrs['N']       = poles.attrs['N1']
        f.attrs['BoxSize'] = 1000.
        f.attrs['Nmesh']   = 256
        f.attrs['kmin']    = 0.008
        f.attrs['dk']      = 0.005
        f.attrs['cic_pdf_Nmesh'] = Nmesh
        f.attrs['cic_pdf_BoxSize'] = BoxSize

def run_alpt_dm(idx_cosmo, cosmo_dict, dm_dir):
    """Generate ICs from cosmo_dict and run ALPT, writing the DM fields to dm_dir."""
    print('ALPT running for cosmo %i' % idx_cosmo, flush=True)
    CS.CSbox_alpt(cosmo_dict, dm_dir, seed=SEED + idx_cosmo, dgrowth_short=5.,
                  ngrid=NGRID, lbox=LBOX, zsnap=ZSNAP, lambdath_tweb=0.,
                  lambdath_twebdelta=0., Nmesh_ic=512, Nsample_ic=256,
                  subgrid=True, return_pos=False, silent=True)

def run_bias(idx_cosmo, theta_gal, cosmo, dm_dir, fname_NLB):
    print('NLB running for cosmo %i' % idx_cosmo, flush=True)
    # CSbox_galaxy builds the velocity filenames with `dm_dir + 'VEx...'`, so dm_dir needs a trailing slash
    xyz_nlb = CS.CSbox_galaxy(theta_gal, None, os.path.join(dm_dir, ''), Ngrid=NGRID,
                              Lbox=LBOX, zsnap=ZSNAP, bias_model='nonlocal2',
                              subgrid=True, silent=True, rsd=False)
    save_spectrum(fname_NLB, xyz_nlb, nlb_to_vec(theta_gal), cosmo)


def load_thetas(idx_cosmo, thetas):
    t = thetas[str(idx_cosmo)]
    theta_gal = {k: np.array(v) for k, v in t['theta_NLB'].items()}  # (4,4) arrays
    return theta_gal

def nlb_to_vec(theta_gal):
    return np.concatenate([theta_gal[k].ravel() for k in NLB_KEYS])


if __name__ == '__main__':


    idx_cosmo = int(sys.argv[1])
    overwrite = '--overwrite' in sys.argv[2:]

    # --dm / --bias select a stage; with neither, run both
    do_dm   = '--dm' in sys.argv[2:] or '--bias' not in sys.argv[2:]
    do_bias = '--bias' in sys.argv[2:] or '--dm' not in sys.argv[2:]

    cosmo_all = np.load(COSMOFN)
    cosmo_vals = cosmo_all[idx_cosmo, :] # Om, Ob, h, n_s, s8
    cosmo_keys = ['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma_8']
    cosmo_dict = {k: float(v) for k, v in zip(cosmo_keys, cosmo_vals)}

    p = alpt_sense_paths(idx_cosmo)
    # hodalpt.sims.alpt does string ops on these, so pass str, not Path
    dm_dir     = str(p['dm_dir'])
    fname_NLB  = str(p['fname_NLB'])

    if do_bias and p['fname_NLB'].exists() and not overwrite:
        print(f'{fname_NLB} exists, skipping'); raise SystemExit

    # get PIP NLB
    with open(SCRIPTDIR / 'theta_NLB_PIP.json') as f:
        thetas = json.load(f)
    theta_gal = load_thetas(idx_cosmo, thetas)

    p['dm_dir'].mkdir(exist_ok=True, parents=True)
    p['outdir_NLB'].mkdir(exist_ok=True, parents=True)

    if do_dm:
        run_alpt_dm(idx_cosmo, cosmo_dict, dm_dir)

    # fail loudly (and before the expensive bias step) if the ALPT outputs are incomplete
    missing = alpt_missing(idx_cosmo)
    if missing:
        print(f'ALPT outputs incomplete for cosmology {idx_cosmo}, missing: {missing}', flush=True)
        sys.exit(1)

    if do_bias:
        run_bias(idx_cosmo, theta_gal, cosmo_vals, dm_dir, fname_NLB)

    print('ALPT done for cosmology %i' % idx_cosmo)
