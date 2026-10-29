import os, sys
import glob
import numpy as np
from pathlib import Path
from hodalpt import stats
from hodalpt.sims import quijote as Q
from hodalpt.sims import alpt as CS
import json, h5py
from nbodykit.lab import ArrayCatalog, FFTPower
import time
import pyfftw

i_LHC = int(sys.argv[1]) # LHC realization
overwrite = True
seed = 20261029

LHC_ROOT  = Path('/corral/utexas/AST25023/simbig/quijote/latinhypercube_hr')
SCRIPTDIR = Path(__file__).resolve().parent
NLB_KEYS = ('alpha', 'beta', 'nmean', 'rhoeps', 'eps')

theta_fn  = SCRIPTDIR / 'theta_NLB_PIP.json'

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
        ic_path    = d / 'ICs',
        dm_dir     = d / 'alpt',
        cosmofn    = d / 'Cosmo_params.dat',
        outdir_NLB = d / 'bias',
        fname_NLB  = d / 'bias' / f'spec.noRSD.{i_LHC}.h5',
    )

def nlb_to_vec(theta_gal):
    return np.concatenate([theta_gal[k].ravel() for k in NLB_KEYS])

p = lhc_paths(i_LHC)

if p['fname_NLB'].exists() and overwrite==False:
    print(f'{p["fname_NLB"]} exists, skipping'); raise SystemExit
for k in ('ic_path', 'cosmofn'):
    assert p[k].exists(), f'missing {k}: {p[k]}'
p['dm_dir'].mkdir(exist_ok=True)
p['outdir_NLB'].mkdir(exist_ok=True)

# define paths
ic_path     = str(p['ic_path'])  # hodalpt.sims.alpt does string ops (.split) on these, so not Path
dm_dir      = str(p['dm_dir'])
cosmofn     = str(p['cosmofn'])
outdir_NLB  = str(p['outdir_NLB'])
fname_NLB   = str(p['fname_NLB'])

with open(theta_fn) as f:
    thetas = json.load(f)

def load_thetas(i_LHC):
    t = thetas[str(i_LHC)]
    theta_gal = {k: np.array(v) for k, v in t['theta_NLB'].items()}  # (4,4) arrays
    return theta_gal

theta_gal = load_thetas(i_LHC)

# run the DM field computation
print('ALPT running for LHC%i' % i_LHC) 
if not os.path.isfile(os.path.join(dm_dir, 'Quijote_ICs_delta_z127_n256_CIC.DAT')): 
    CS.CSbox_alpt_Q(ic_path, dm_dir, seed=seed, dgrowth_short=5., make_ics=True,
            subgrid=True, return_pos=False, silent=True)
else: 
    # run subgrid on 
    CS.CSbox_alpt_Q(ic_path, dm_dir, seed=seed, dgrowth_short=5., make_ics=False,
            subgrid=True, return_pos=False, silent=True)


# for bias computation
cosmo = np.loadtxt(cosmofn) # om, ob, h, ns, s8
xyz_nlb = CS.CSbox_galaxy(theta_gal, None, dm_dir, bias_model='nonlocal2', 
                    subgrid=True, silent=True, rsd=False)

save_spectrum(fname_NLB, xyz_nlb, nlb_to_vec(theta_gal), cosmo)



