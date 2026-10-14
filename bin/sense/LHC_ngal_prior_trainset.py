import os
from glob import glob
import h5py
import numpy as np
from hodalpt.sims import quijote as Q

# LHC_DIR = '/Users/mcc3842/CosmicSim2025/data/quijote/latin_hypercube_HR' # local
LHC_DIR = '/corral/utexas/AST25023/simbig/quijote/latinhypercube_hr' # TACC


paths = sorted([p for p in glob(os.path.join(LHC_DIR, '[0-9]*')) if int(os.path.basename(p)) < 400], key=lambda p: int(os.path.basename(p)))
LHC_idx = [int(os.path.basename(p)) for p in paths]

FID_HOD = {
    'logMmin': 12.97,
    'sigma_logM': 0.40,
    'logM0': 13.67,
    'logM1': 13.68,
    'alpha': 0.79,
    'Abias': 0.01,
    'eta_conc': 1.11,
    'eta_cen': 0.31,
    'eta_sat': 0.85,
}
Oms, s8s, ngs = [], [], []
for path in paths:
    cat = Q.HODgalaxies(FID_HOD, path)
    xyz = np.array(cat['Position'])
    ng = xyz.shape[0]
    # look up cosmology
    setup = path.split('/')[-2]
    ireal = int(path.split('/')[-1])
    Om, Ob, h, ns, s8 = Q._cosmo_lookup(setup, ireal)
    Oms.append(Om)
    s8s.append(s8)
    ngs.append(ng)

outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LHC_cosmo_ngal_trainset.hdf5')
with h5py.File(outfile, 'w') as f:
    f.create_dataset('LHC_idx', data=np.array(LHC_idx))
    f.create_dataset('Om', data=np.array(Oms))
    f.create_dataset('s8', data=np.array(s8s))
    f.create_dataset('ngal', data=np.array(ngs))
print('wrote', outfile)
