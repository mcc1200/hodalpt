'''
pip_train_pylauncher.py

Parallelizes the full pipeline (bias sampling + P(k) + B(k)) for fiducial_HR
NLB and HOD samples across TACC nodes using pylauncher.  Each commandline
invokes bias_fiducial.py for a single sample index; pylauncher saturates all
available cores.  Existing spectra files are overwritten by default.

    NLB: 10,000 samples (all i),    written to bias/NLB/spec.i.h5
    HOD:  1,000 samples (i < 1000), written to bias/HOD/spec.i.h5
    bias_fiducial.py handles the split internally.

Scale (rough estimates, 9 cores/task → ~113 slots per 8 nodes on Lonestar6):
    batch 1 (i < 1000):  NLB + HOD ≈ 1300 s/sample → 1000×1300/113 ≈ 3.2 h on 8 nodes
    batch 2 (i >= 1000): NLB only  ≈  650 s/sample → 9000×650/113  ≈ 14 h on 8 nodes
                                                     → 9000×650/450  ≈  3.6 h on 32 nodes

Recommended usage (submit both batches separately):
    python pip_train_pylauncher.py 0 1000    --nodes 8  --time 4   # NLB + HOD
    python pip_train_pylauncher.py 1000 10000 --nodes 32 --time 4  # NLB only

Other options:
    python pip_train_pylauncher.py 0 1000 --skip-done              # skip already-complete samples
    python pip_train_pylauncher.py resume 0 1000                   # restart interrupted job

Do NOT call the 'launch' / 'resume-run' subcommands directly;
they are invoked by the SLURM script.
'''
import os
import sys
import numpy as np


NLB_DIR = '/corral/utexas/AST25023/simbig/quijote/fiducial_HR/0/bias/NLB'
HOD_DIR = '/corral/utexas/AST25023/simbig/quijote/fiducial_HR/0/bias/HOD'
N_HOD   = 1000


def _spec_complete(fn):
    '''Return True if fn is an existing h5 file with both p0 and b123.'''
    import h5py
    try:
        with h5py.File(fn, 'r') as f:
            return 'p0' in f and 'b123' in f
    except Exception:
        return False


def _pending_indices(i0, i1, skip_done=False):
    '''Return list of sample indices that still need full pipeline computed.'''
    indices = list(range(i0, i1))
    if not skip_done:
        return indices

    pending = []
    for i in indices:
        nlb_ok = _spec_complete(os.path.join(NLB_DIR, 'spec.%i.h5' % i))
        hod_ok = _spec_complete(os.path.join(HOD_DIR, 'spec.%i.h5' % i)) if i < N_HOD else True
        if not (nlb_ok and hod_ok):
            pending.append(i)
    n_done = len(indices) - len(pending)
    if n_done:
        print('skipping %i already-complete samples' % n_done)
    return pending


def run_pip_train_pylauncher(i0, i1, nodes=8, time=4, queue='normal',
                             skip_done=False):
    '''Submit full pipeline (bias + P(k) + B(k)) for fiducial_HR samples via pylauncher.

    Each task calls bias_fiducial.py for a single index, which samples bias
    parameters, generates galaxy catalogs, and writes P(k) and B(k) to HDF5,
    overwriting any existing file.

    Parameters
    ----------
    i0, i1 : int
        Sample index range [i0, i1).
    nodes : int
        Number of compute nodes.
    time : float
        Requested wall-clock hours.
    queue : str
        SLURM partition ('normal', 'development').
    skip_done : bool
        Skip samples whose NLB spec file already contains both p0 and b123.
    '''
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    workdir   = os.environ.get('WORK', os.getcwd())
    scratch   = os.environ.get('SCRATCH', workdir)

    hr = int(np.floor(time))
    mn = int((time * 60) % 60)

    indices = _pending_indices(i0, i1, skip_done=skip_done)
    if not indices:
        print('all samples already complete, nothing to submit')
        return None
    print('%i samples to process' % len(indices))

    logdir = os.path.join(scratch, 'pip_train_logs_%i_%i' % (i0, i1))
    os.makedirs(logdir, exist_ok=True)

    cmdfile = os.path.join(scratch, 'pip_train_cmds_%i_%i.txt' % (i0, i1))
    with open(cmdfile, 'w') as f:
        for i in indices:
            logfile = os.path.join(logdir, 'sample_%i.log' % i)
            f.write('python %s/bias_fiducial.py %i %i > %s 2>&1\n'
                    % (scriptdir, i, i + 1, logfile))
    print('wrote %i commandlines to %s' % (len(indices), cmdfile))

    pyl_workdir_base = os.path.join(scratch, 'pylauncher_pip_train_%i_%i' % (i0, i1))

    slurm = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -J pip.train.pyl.%i_%i' % (i0, i1),
        '#SBATCH -o %s/pip_train.pyl.%i_%i.%%j.out' % (workdir, i0, i1),
        '#SBATCH -e %s/pip_train.pyl.%i_%i.%%j.err' % (workdir, i0, i1),
        '#SBATCH -p %s' % queue,
        '#SBATCH -N %i' % nodes,
        '#SBATCH --time=%s:%s:00' % (str(hr).zfill(2), str(mn).zfill(2)),
        '#SBATCH -A AST25022',
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
        'module load pylauncher',
        '',
        'export OMP_NUM_THREADS=1',
        'export MKL_NUM_THREADS=1',
        'export OPENBLAS_NUM_THREADS=1',
        '',
        'python %s launch %s %s_${SLURM_JOB_ID}' % (os.path.abspath(__file__), cmdfile, pyl_workdir_base),
        '',
    ])

    slurmfile = os.path.join(workdir, 'pip_train_pylauncher_%i_%i.slurm' % (i0, i1))
    with open(slurmfile, 'w') as f:
        f.write(slurm)
    print('submitting %s' % slurmfile)
    os.system('sbatch %s' % slurmfile)
    return None


def resume_pip_train_pylauncher(i0, i1, jobid, nodes=8, time=2, queue='normal'):
    '''Resume an interrupted pylauncher pip_train job.'''
    workdir = os.environ.get('WORK', os.getcwd())
    scratch = os.environ.get('SCRATCH', workdir)

    hr = int(np.floor(time))
    mn = int((time * 60) % 60)

    pyl_workdir = os.path.join(scratch, 'pylauncher_pip_train_%i_%i_%s' % (i0, i1, jobid))
    queuestate  = os.path.join(pyl_workdir, 'queuestate')

    if not os.path.exists(queuestate):
        raise FileNotFoundError(
            'no queuestate at %s — has this job been run before?' % queuestate)

    slurm = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -J pip.train.pyl.resume.%i_%i' % (i0, i1),
        '#SBATCH -o %s/pip_train.pyl.resume.%i_%i.%%j.out' % (workdir, i0, i1),
        '#SBATCH -e %s/pip_train.pyl.resume.%i_%i.%%j.err' % (workdir, i0, i1),
        '#SBATCH -p %s' % queue,
        '#SBATCH -N %i' % nodes,
        '#SBATCH --time=%s:%s:00' % (str(hr).zfill(2), str(mn).zfill(2)),
        '#SBATCH -A AST25022',
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
        'module load pylauncher',
        '',
        'export OMP_NUM_THREADS=1',
        'export MKL_NUM_THREADS=1',
        'export OPENBLAS_NUM_THREADS=1',
        '',
        'python %s resume-run %s' % (os.path.abspath(__file__), queuestate),
        '',
    ])

    slurmfile = os.path.join(workdir, 'pip_train_pylauncher_resume_%i_%i.slurm' % (i0, i1))
    with open(slurmfile, 'w') as f:
        f.write(slurm)
    print('resuming from %s' % queuestate)
    os.system('sbatch %s' % slurmfile)
    return None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    subcmd = sys.argv[1]

    if subcmd == 'launch':
        import pylauncher
        cmdfile     = sys.argv[2]
        pyl_workdir = sys.argv[3]
        pylauncher.ClassicLauncher(cmdfile, workdir=pyl_workdir, debug='job', delay=0.01, cores=32)

    elif subcmd == 'resume-run':
        import pylauncher
        queuestate = sys.argv[2]
        pylauncher.ResumeClassicLauncher(queuestate, debug='job', delay=0.01, cores=32)

    elif subcmd == 'resume':
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument('subcmd')
        p.add_argument('i0', type=int)
        p.add_argument('i1', type=int)
        p.add_argument('--jobid', type=str,   required=True)
        p.add_argument('--nodes', type=int,   default=8)
        p.add_argument('--time',  type=float, default=2.0)
        p.add_argument('--queue', type=str,   default='normal')
        args = p.parse_args()
        resume_pip_train_pylauncher(
            args.i0, args.i1, args.jobid, nodes=args.nodes, time=args.time, queue=args.queue)

    else:
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument('i0',          type=int)
        p.add_argument('i1',          type=int)
        p.add_argument('--nodes',     type=int,   default=8)
        p.add_argument('--time',      type=float, default=4.0)
        p.add_argument('--queue',     type=str,   default='normal')
        p.add_argument('--skip-done', action='store_true')
        args = p.parse_args()
        run_pip_train_pylauncher(
            args.i0, args.i1,
            nodes=args.nodes, time=args.time, queue=args.queue,
            skip_done=args.skip_done)
