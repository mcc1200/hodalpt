import os
import numpy as np

'''
Submit compute_cic_pdf.py as a regular (non-interactive) SLURM job, rather
than waiting on the idev queue -- runs on a single node, using all its cores
via the multiprocessing pool in compute_cic_pdf.py (nworkers picked up from
SLURM_CPUS_ON_NODE).

Idempotent -- safe to resubmit if a job doesn't finish in time (already
processed files are skipped).
'''

def submit_cic_pdf(dataset='NLB', i0=0, i1=-1, overwrite=False,
                    time=2, queue='normal'):
    scriptdir = os.path.dirname(__file__)
    hr = int(np.floor(time))
    mn = int((time * 60) % 60)
    os.makedirs('o', exist_ok=True)

    a = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -J cic.%s' % dataset,
        '#SBATCH -o o/cic.%s.%%j.out' % dataset,
        '#SBATCH -p %s' % queue,
        '#SBATCH -N 1',
        '#SBATCH -n 1',
        '#SBATCH --time=%s:%s:00' % (str(hr).zfill(2), str(mn).zfill(2)),
        '#SBATCH -A AST25023',
        '',
        'module purge',
        'module load intel',
        'module load impi',
        'module load fftw3/3.3.10',
        'module load gsl',
        '',
        'unset PYTHONPATH',
        'source ~/.bashrc',
        '',
        'conda activate simbig',
        '',
        ''])

    a += 'python %s/compute_cic_pdf.py %s %i %i%s\n' % (
        scriptdir, dataset, i0, i1, ' --overwrite' if overwrite else '')

    f = open(os.path.join(os.environ['WORK'], 'script.slurm'), 'w')
    f.write(a)
    f.close()
    os.system('sbatch %s' % os.path.join(os.environ['WORK'], 'script.slurm'))
    return None

if __name__ == '__main__':
    import sys
    # smoke test: python submit_cic_pdf.py 0 50  -> first 50 of both NLB and HOD
    i0 = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    i1 = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    queue = 'development' if i1 != -1 else 'normal'
    time  = 0.5 if i1 != -1 else 2

    submit_cic_pdf(dataset='NLB', i0=i0, i1=i1, time=time, queue=queue)
    submit_cic_pdf(dataset='HOD', i0=i0, i1=i1, time=(0.5 if i1 != -1 else 1), queue=queue)
