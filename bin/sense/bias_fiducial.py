import os, sys
import warnings
import numpy as np
import h5py
from hodalpt import priors
from hodalpt.sims import alpt as CS
from hodalpt.sims import quijote as Q
from nbodykit.lab import ArrayCatalog, FFTPower
import time
# import pyfftw
# pyfftw.config.NUM_THREADS = 1

from hodalpt import stats

warnings.filterwarnings('ignore', message='You have selected 18 bins')

t0 = time.time()
'''
for one fiducial_hr realization:
    read in samples
    generate galaxy catalog from theta_bias
        save positions xyz_g
    compute power spectra
        save k p0k p2k
'''
i0 = int(sys.argv[1])
i1 = int(sys.argv[2])

n_hod = int(1000)
dm_dir = '/corral/utexas/AST25023/simbig/quijote/fiducial_HR/0/alpt/'
outdir = '/corral/utexas/AST25023/simbig/quijote/fiducial_HR/0/bias'
path_quij = '/corral/utexas/AST25023/simbig/quijote/fiducial_HR/0'
outdir_NLB = os.path.join(outdir,'NLB')
outdir_HOD =  os.path.join(outdir,'HOD')
os.makedirs(outdir_NLB, exist_ok=True)
os.makedirs(outdir_HOD, exist_ok=True)

def save_spectrum(fname, xyz, theta):
    """Save FFTPower multipoles to HDF5."""
    
    ######## P(k) #################################
    cat = ArrayCatalog({'Position': xyz}, BoxSize=1000.) 
    r   = FFTPower(cat, mode='2d', Nmesh=256, dk=0.005, kmin=0.008,
               Nmu=10, los=[0,0,1], poles=[0, 2])

    poles = r.poles
    k      = poles['k']
    p0     = poles['power_0'].real - poles.attrs['shotnoise']
    p2     = poles['power_2'].real
    nmodes = poles['modes']
    ######### B(k) #################################
    bispec = stats.B0_periodic(xyz.T, w=None, Lbox=1000., fft='pyfftw', silent=True)
    
    with h5py.File(fname, 'w') as f:
        f['theta']    = theta
        f['ngs']      = xyz.shape[0]
        f['xyz']      = xyz
        f['k']        = k
        f['p0']       = p0
        f['p2']       = p2
        f['nmodes']   = nmodes
        f['shotnoise'] = poles.attrs['shotnoise']
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

_HOD_KEYS = ['logMmin', 'sigma_logM', 'logM0', 'logM1', 'alpha',
             'Abias', 'eta_conc', 'eta_cen', 'eta_sat']

_RSD_KEYS = ['bv', 'bb', 'betarsd', 'gamma']

def hod_to_vec(hod):
    """Flatten HOD dict to 1-D array in canonical order (_HOD_KEYS)."""
    return np.array([hod[k] for k in _HOD_KEYS])

def nlb_to_vec(theta_gal, theta_rsd):
    """Flatten NLB dicts to 1-D array: alpha(16), beta(16), nmean(16), rsd(4)."""
    return np.concatenate([
        theta_gal['alpha'].ravel(),
        theta_gal['beta'].ravel(),
        theta_gal['nmean'].ravel(),
        theta_gal['rhoeps'].ravel(),
        theta_gal['eps'].ravel(),
        [theta_rsd[k] for k in _RSD_KEYS],
    ])



for i in range(i0, i1):
    t_i = time.time()
    print('[%i/%i] computing sample %i' % (i - i0 + 1, i1 - i0, i), flush=True)

    fname_NLB = outdir_NLB+'/spec.%i.h5' % i
    seed = i
    theta_gal, theta_rsd = priors.sample_bias(seed=seed, model='nonlocal2')
    xyz_nlb = CS.CSbox_galaxy(theta_gal, theta_rsd, dm_dir, bias_model='nonlocal2', subgrid=True, silent=True)
    save_spectrum(fname_NLB, xyz_nlb, nlb_to_vec(theta_gal, theta_rsd))

    if i < n_hod:
        fname_HOD = outdir_HOD+'/spec.%i.h5' % i
        hod = priors.sample_HOD(seed=i)
        gals =  Q.HODgalaxies(hod, path_quij, z=0.5)
        xyz_hod = Q.Box_RSD(gals, LOS=[0,0,1], Lbox=1000.)
        save_spectrum(fname_HOD, xyz_hod, hod_to_vec(hod))

    dt = time.time() - t_i
    print('[%i/%i] sample %i done in %.1f s (total %.1f min)' % (
          i - i0 + 1, i1 - i0, i, dt, (time.time() - t0) / 60.), flush=True)

    

        