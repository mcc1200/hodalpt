import os, sys
import numpy as np
from pathlib import Path
from hodalpt import stats
from hodalpt.sims import quijote as Q
import json, h5py
from nbodykit.lab import ArrayCatalog, FFTPower

LHC_ROOT  = Path('/corral/utexas/AST25023/simbig/quijote/latinhypercube_hr')
SCRIPTDIR = Path(__file__).resolve().parent

NGRID, LBOX, ZSNAP = 256, 1000., 0.5

_HOD_KEYS = ['logMmin', 'sigma_logM', 'logM0', 'logM1', 'alpha',
             'Abias', 'eta_conc', 'eta_cen', 'eta_sat']

def hod_to_vec(hod):
    """Flatten HOD dict to 1-D array in canonical order (_HOD_KEYS)."""
    return np.array([hod[k] for k in _HOD_KEYS])

def load_hod(i_LHC, thetas):
    t = thetas[str(i_LHC)]
    theta_hod = t['theta_HOD']  # dict with HOD_KEYS 
    return theta_hod

def lhc_paths(i_LHC, root=LHC_ROOT):
    """All per-realization paths for a given LHC index."""
    d = root / str(i_LHC)
    return dict(
        path_quij  = d,
        cosmofn    = d / 'Cosmo_params.dat',
        outdir_HOD = d / 'bias' / 'HOD',
        fname_HOD  = d / 'bias' / 'HOD' / f'spec.noRSD.{i_LHC}.h5',
    )

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

if __name__ == '__main__':
    i_LHC = int(sys.argv[1]) # LHC realization
    overwrite = '--overwrite' in sys.argv[2:]

    p = lhc_paths(i_LHC)
        # hodalpt.sims.alpt does string ops on these, so pass str, not Path
    path_quij  = str(p['path_quij'])
    cosmofn    = str(p['cosmofn'])
    outdir_HOD = str(p['outdir_HOD'])
    fname_HOD  = str(p['fname_HOD'])
    os.makedirs(outdir_HOD, exist_ok=True)

    if  p['fname_HOD'].exists() and not overwrite:
        print(f'{fname_HOD} exists, skipping'); raise SystemExit

    cosmo = np.loadtxt(cosmofn) # om, ob, h, ns, s8

    with open(SCRIPTDIR / 'theta_NLB_PIP.json') as f:
            thetas = json.load(f)
    hod = load_hod(i_LHC, thetas)


    gals = Q.HODgalaxies(hod, path_quij, z=ZSNAP)
    xyz = np.array(gals['Position'])

    save_spectrum(fname_HOD, xyz, hod_to_vec(hod), cosmo)

    
    