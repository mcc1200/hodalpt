import os, sys
import warnings
import numpy as np
import h5py
from hodalpt import priors
from hodalpt.sims import quijote as Q  
from nbodykit.lab import ArrayCatalog, FFTPower
import time
import pyfftw
pyfftw.config.NUM_THREADS = 1

from hodalpt import stats

warnings.filterwarnings('ignore', message='You have selected 18 bins')

t0 = time.time()

i0       = int(sys.argv[1])
i1       = int(sys.argv[2])



outdir = '/corral/utexas/AST25023/simbig/quijote/fiducial_HR/0/bias'
path_quij = '/corral/utexas/AST25023/simbig/quijote/fiducial_HR/0'
outdir_HOD =  os.path.join(outdir,'HOD')
os.makedirs(outdir_HOD, exist_ok=True)

def save_spectrum(fname, xyz, theta):
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
        f['ngs']      = xyz.shape[0]
        # xyz itself is NOT stored here -- it dwarfs every other dataset in
        # this file (10s of MB vs a few hundred floats) and isn't read by
        # the routine NPE collection pipeline. It's a deterministic function
        # of seed=i, so it's reproducible later via CS.CSbox_galaxy if ever
        # actually needed (e.g. for the opt-in position archive, which only
        # covers indices generated before this change).
        f['k']        = k
        f['p0']       = p0
        # f['p2']       = p2
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

_HOD_KEYS = ['logMmin', 'sigma_logM', 'logM0', 'logM1', 'alpha',
             'Abias', 'eta_conc', 'eta_cen', 'eta_sat']

def hod_to_vec(hod):
    """Flatten HOD dict to 1-D array in canonical order (_HOD_KEYS)."""
    return np.array([hod[k] for k in _HOD_KEYS])

for i in range(i0, i1):
    t_i = time.time()
    fname_HOD = outdir_HOD+'/spec.noRSD.%i.h5' % i
    hod = priors.sample_HOD_realspace(seed=i)
    gals =  Q.HODgalaxies(hod, path_quij, z=0.5)
    xyz_hod = np.array(gals['Position'])
    save_spectrum(fname_HOD, xyz_hod, hod_to_vec(hod))

dt = time.time() - t_i
print('[%i/%i] sample %i done in %.1f s (total %.1f min)' % (i - i0 + 1, i1 - i0, i, dt, (time.time() - t0) / 60.), flush=True)
