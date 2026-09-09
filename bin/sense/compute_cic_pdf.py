import os, sys
import time
import glob
import numpy as np
import h5py
import multiprocessing as mp
from tqdm import tqdm
from nbodykit.lab import ArrayCatalog

'''
Append the 1D CIC PDF to existing spec.noRSD.*.h5 files (NLB or HOD).
Idempotent -- skips files that already have 'cic_pdf' unless --overwrite.

Usage (run on an idev node):
    python compute_cic_pdf.py NLB 0 100      # small batch first, to check timing/I-O on corral
    python compute_cic_pdf.py NLB            # full NLB set (10,000 files)
    python compute_cic_pdf.py HOD            # full HOD set (1,000 files)
    python compute_cic_pdf.py NLB 0 -1 --overwrite

Safe to re-run / resubmit if the idev session runs out of walltime -- already
processed files are skipped.
'''

BoxSize = 1000.
Nmesh = 40
bins = np.arange(-0.5, 85.5, 1)

files_NLB = sorted(glob.glob(
    '/corral/utexas/AST25023/simbig/quijote/fiducial_HR/0/bias/NLB/spec.noRSD.*.h5')) # 10,000 catalogs
files_HOD = sorted(glob.glob(
    '/corral/utexas/AST25023/simbig/quijote/fiducial_HR/0/bias/HOD/spec.noRSD.*.h5')) # 1000 catalogs

def cic_pdf(cat, BoxSize=BoxSize, Nmesh=Nmesh, bins=bins):
    mesh = cat.to_mesh(Nmesh=Nmesh, BoxSize=BoxSize, resampler='cic',
                        compensated=False, interlaced=False)
    field = mesh.paint(mode='real')
    mean_n = cat.csize / (BoxSize ** 3)
    cell_vol = (BoxSize / Nmesh) ** 3
    counts = field.value.ravel() * mean_n * cell_vol
    pdf, edges = np.histogram(counts, bins=bins, density=False)
    pdf = pdf / pdf.sum()
    return pdf, edges

def process_file(args):
    fpath, overwrite = args
    t0 = time.time()
    try:
        with h5py.File(fpath, 'a') as f:
            if 'cic_pdf' in f and not overwrite:
                return (fpath, 'skipped', time.time() - t0, None)
            xyz = f['xyz'][:]
            cat = ArrayCatalog({'Position': xyz}, BoxSize=BoxSize)
            pdf, edges = cic_pdf(cat)

            for name in ('cic_pdf', 'cic_pdf_edges'):
                if name in f:
                    del f[name]   # overwrite cleanly if rerunning
            f['cic_pdf'] = pdf
            f['cic_pdf_edges'] = edges
            f.attrs['cic_pdf_Nmesh'] = Nmesh
            f.attrs['cic_pdf_BoxSize'] = BoxSize
        return (fpath, 'done', time.time() - t0, None)
    except Exception as e:
        return (fpath, 'failed', time.time() - t0, str(e))

if __name__ == '__main__':
    dataset   = sys.argv[1] if len(sys.argv) > 1 else 'NLB'   # 'NLB' or 'HOD'
    i0        = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    i1        = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != '-1' else None
    overwrite = '--overwrite' in sys.argv
    nworkers  = int(os.environ.get('SLURM_CPUS_ON_NODE', os.cpu_count() or 4))

    files = files_NLB if dataset == 'NLB' else files_HOD
    files = files[i0:i1]

    print('%s: %i files, %i workers' % (dataset, len(files), nworkers), flush=True)

    # nbodykit imports mpi4py and inits MPI at import time -- forking a pool
    # after that is unsafe, so use 'spawn' (each worker gets a fresh interpreter
    # and does its own MPI_Init).
    ctx = mp.get_context('spawn')
    tasks = [(fp, overwrite) for fp in files]
    n_done = n_skip = n_fail = 0
    t_start = time.time()

    with ctx.Pool(nworkers) as pool:
        pbar = tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks),
                    desc=dataset, unit='file', mininterval=1.0)
        for fpath, status, dt, err in pbar:
            if status == 'done':
                n_done += 1
            elif status == 'skipped':
                n_skip += 1
            else:
                n_fail += 1
                tqdm.write('FAILED %s: %s' % (fpath, err))
            pbar.set_postfix(done=n_done, skip=n_skip, fail=n_fail, last='%.2fs' % dt)

    print('finished %s in %.1f min: %i done, %i skipped, %i failed' % (
        dataset, (time.time() - t_start) / 60., n_done, n_skip, n_fail), flush=True)
