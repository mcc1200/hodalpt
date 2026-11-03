import os, sys
import numpy as np
from pathlib import Path
from hodalpt import stats
from hodalpt.sims import alpt as CS
import json, h5py
from nbodykit.lab import ArrayCatalog, FFTPower


LHC_ROOT  = Path('/corral/utexas/AST25023/simbig/quijote/latinhypercube_hr')
SCRIPTDIR = Path(__file__).resolve().parent
NLB_KEYS = ('alpha', 'beta', 'nmean', 'rhoeps', 'eps')

SEED = 20261029  # IC seed base; realization i_LHC uses SEED + i_LHC
NGRID, LBOX, ZSNAP = 256, 1000., 0.5


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


def lhc_paths(i_LHC, root=LHC_ROOT):
    """All per-realization paths for a given LHC index."""
    d = root / str(i_LHC)
    return dict(
        dm_dir     = d / 'alpt',
        cosmofn    = d / 'Cosmo_params.dat',
        outdir_NLB = d / 'bias',
        fname_NLB  = d / 'bias' / f'spec.noRSD.{i_LHC}.h5',
    )

def nlb_to_vec(theta_gal):
    return np.concatenate([theta_gal[k].ravel() for k in NLB_KEYS])


def load_thetas(i_LHC, thetas):
    t = thetas[str(i_LHC)]
    theta_gal = {k: np.array(v) for k, v in t['theta_NLB'].items()}  # (4,4) arrays
    return theta_gal


def run_alpt_dm(i_LHC, cosmo_dict, dm_dir):
    """Generate ICs from cosmo_dict and run ALPT, writing the DM fields to dm_dir."""
    print('ALPT running for LHC%i' % i_LHC, flush=True)
    CS.CSbox_alpt(cosmo_dict, dm_dir, seed=SEED + i_LHC, dgrowth_short=5.,
                  ngrid=NGRID, lbox=LBOX, zsnap=ZSNAP, lambdath_tweb=0.,
                  lambdath_twebdelta=0., Nmesh_ic=512, Nsample_ic=256,
                  subgrid=True, return_pos=False, silent=True)


def run_bias(i_LHC, theta_gal, cosmo, dm_dir, fname_NLB):
    print('NLB running for LHC%i' % i_LHC, flush=True)
    # CSbox_galaxy builds the velocity filenames with `dm_dir + 'VEx...'`, so dm_dir needs a trailing slash
    xyz_nlb = CS.CSbox_galaxy(theta_gal, None, os.path.join(dm_dir, ''), Ngrid=NGRID,
                              Lbox=LBOX, zsnap=ZSNAP, bias_model='nonlocal2',
                              subgrid=True, silent=True, rsd=False)
    save_spectrum(fname_NLB, xyz_nlb, nlb_to_vec(theta_gal), cosmo)


if __name__ == '__main__':
    i_LHC = int(sys.argv[1]) # LHC realization
    overwrite = '--overwrite' in sys.argv[2:]
    # --dm / --bias select a stage; with neither, run both
    do_dm   = '--dm' in sys.argv[2:] or '--bias' not in sys.argv[2:]
    do_bias = '--bias' in sys.argv[2:] or '--dm' not in sys.argv[2:]

    p = lhc_paths(i_LHC)
    # hodalpt.sims.alpt does string ops on these, so pass str, not Path
    dm_dir     = str(p['dm_dir'])
    fname_NLB  = str(p['fname_NLB'])

    if do_bias and p['fname_NLB'].exists() and not overwrite:
        print(f'{fname_NLB} exists, skipping'); raise SystemExit
    assert p['cosmofn'].exists(), f'missing cosmofn: {p["cosmofn"]}'

    # load everything before the expensive DM step so a bad input fails fast
    cosmo = np.loadtxt(p['cosmofn']) # om, ob, h, ns, s8
    cosmo_labels = ['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma_8']
    cosmo_dict = {k: float(v) for k, v in zip(cosmo_labels, cosmo)}

    with open(SCRIPTDIR / 'theta_NLB_PIP.json') as f:
        thetas = json.load(f)
    theta_gal = load_thetas(i_LHC, thetas)

    p['dm_dir'].mkdir(exist_ok=True)
    p['outdir_NLB'].mkdir(exist_ok=True)

    if do_dm:
        run_alpt_dm(i_LHC, cosmo_dict, dm_dir)
    if do_bias:
        run_bias(i_LHC, theta_gal, cosmo, dm_dir, fname_NLB)
