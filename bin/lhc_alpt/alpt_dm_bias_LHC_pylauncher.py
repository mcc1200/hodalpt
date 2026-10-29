'''
alpt_dm_bias_LHC_pylauncher.py

Parallelizes alpt_dm_bias_LHC.py (ALPT DM field + real-space bias sampling +
P(k), CIC PDF, B(k)) over the Quijote LHC realizations in theta_NLB_PIP.json
across TACC nodes using pylauncher. Each commandline invokes
alpt_dm_bias_LHC.py for a single i_LHC; pylauncher saturates all cores.

i0, i1 index POSITIONS in theta_NLB_PIP.json (i.e. in 95p_lhc_idx.npy, same
order written by gen_thetas.py), NOT i_LHC values themselves; the launcher
looks up i_LHC for each position, so every i_LHC submitted is guaranteed to
have a theta_NLB entry.

    output: latinhypercube_hr/<i_LHC>/bias/spec.noRSD.<i_LHC>.h5

Memory: each task runs the ALPT DM step (reads the ICs) and then the bias +
FFTPower + B(k), so peak memory is higher than the fiducial bias-only pipeline.
pylauncher gives each task --cores-per-task cores, so tasks/node =
128 // cores-per-task and memory/task ~ (node mem) / (tasks/node). Default 12
(10 tasks/node, ~1/10 of node memory each); calibrate with seff / MaxRSS from
a --queue development smoke test, then raise or lower.

Usage:
    python alpt_dm_bias_LHC_pylauncher.py 0 10 --nodes 1 --time 1 --queue development  # smoke test
    python alpt_dm_bias_LHC_pylauncher.py 0 1000 --nodes 9 --time 4

Other options:
    python alpt_dm_bias_LHC_pylauncher.py 0 1000 --skip-done   # skip already-complete spectra
    python alpt_dm_bias_LHC_pylauncher.py resume 0 1000 --jobid <id>  # restart interrupted job

Do NOT call the 'launch' / 'resume-run' subcommands directly;
they are invoked by the SLURM script.
'''
import os
import sys
import json
import numpy as np


LHC_ROOT = '/corral/utexas/AST25023/simbig/quijote/latinhypercube_hr'
SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
THETA_FN = os.path.join(SCRIPTDIR, 'theta_NLB_PIP.json')

# every key save_spectrum() in alpt_dm_bias_LHC.py writes; 'q123' is written
# last, so a file killed mid-write is missing it.
_SPEC_KEYS = ['theta', 'cosmo', 'ngs', 'k', 'p0', 'nmodes', 'shotnoise',
              'cic_pdf', 'cic_pdf_edges', 'i_k1', 'i_k2', 'i_k3', 'b123', 'q123']


def _lhc_indices():
    '''i_LHC values in theta_NLB_PIP.json, in the order gen_thetas.py wrote them.'''
    with open(THETA_FN) as f:
        return [int(k) for k in json.load(f)]


def _spec_complete(fn):
    '''True if fn is an openable h5 file with every key save_spectrum() writes.'''
    import h5py
    try:
        with h5py.File(fn, 'r') as f:
            return all(k in f for k in _SPEC_KEYS)
    except Exception:
        return False


def _pending(i0, i1, skip_done=False):
    '''i_LHC values at positions [i0, i1) that still need computing.'''
    lhc = _lhc_indices()[i0:i1]
    if not skip_done:
        return lhc
    pending = [i for i in lhc if not _spec_complete(
        os.path.join(LHC_ROOT, str(i), 'bias', 'spec.noRSD.%i.h5' % i))]
    if len(lhc) - len(pending):
        print('skipping %i already-complete realizations' % (len(lhc) - len(pending)))
    return pending


def _header(name, i0, i1, nodes, time, queue, workdir):
    hr = int(np.floor(time))
    mn = int((time * 60) % 60)
    return [
        '#!/bin/bash',
        '#SBATCH -J %s.%i_%i' % (name, i0, i1),
        '#SBATCH -o %s/%s.%i_%i.%%j.out' % (workdir, name, i0, i1),
        '#SBATCH -e %s/%s.%i_%i.%%j.err' % (workdir, name, i0, i1),
        '#SBATCH -p %s' % queue,
        '#SBATCH -N %i' % nodes,
        '#SBATCH -n %i' % (nodes * 128),
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
        'module load pylauncher',
        '',
        'export OMP_NUM_THREADS=1',
        'export MKL_NUM_THREADS=1',
        'export OPENBLAS_NUM_THREADS=1',
        'export HDF5_USE_FILE_LOCKING=FALSE',
        '',
    ]


def run_alpt_dm_bias_LHC_pylauncher(i0, i1, nodes=9, time=4, queue='normal',
                                    cores=12, skip_done=False):
    '''Submit alpt_dm_bias_LHC.py for LHC realizations at positions [i0, i1).

    Parameters
    ----------
    i0, i1 : int
        Position range [i0, i1) in theta_NLB_PIP.json.
    nodes : int
        Number of compute nodes.
    time : float
        Requested wall-clock hours.
    queue : str
        SLURM partition ('normal', 'development').
    cores : int
        Cores pylauncher reserves per task (sets tasks/node, hence memory/task).
    skip_done : bool
        Skip realizations whose spec.noRSD.<i>.h5 already has every key.
    '''
    workdir = os.environ.get('WORK', os.getcwd())
    scratch = os.environ.get('SCRATCH', workdir)

    indices = _pending(i0, i1, skip_done=skip_done)
    if not indices:
        print('nothing to submit')
        return None
    print('%i realizations to process' % len(indices))

    logdir = os.path.join(scratch, 'alpt_dm_bias_LHC_logs_%i_%i' % (i0, i1))
    os.makedirs(logdir, exist_ok=True)

    cmdfile = os.path.join(scratch, 'alpt_dm_bias_LHC_cmds_%i_%i.txt' % (i0, i1))
    with open(cmdfile, 'w') as f:
        for i in indices:
            f.write('python %s/alpt_dm_bias_LHC.py %i > %s/lhc_%i.log 2>&1\n'
                    % (SCRIPTDIR, i, logdir, i))
    print('wrote %i commandlines to %s' % (len(indices), cmdfile))

    pyl_workdir_base = os.path.join(scratch, 'pylauncher_alpt_dm_bias_LHC_%i_%i' % (i0, i1))

    slurm = '\n'.join(_header('alpt_dm_bias_LHC.pyl', i0, i1, nodes, time, queue, workdir) + [
        'python %s launch %s %s_${SLURM_JOB_ID} %i'
        % (os.path.abspath(__file__), cmdfile, pyl_workdir_base, cores),
        '',
    ])

    slurmfile = os.path.join(workdir, 'alpt_dm_bias_LHC_pylauncher_%i_%i.slurm' % (i0, i1))
    with open(slurmfile, 'w') as f:
        f.write(slurm)
    print('submitting %s' % slurmfile)
    os.system('sbatch %s' % slurmfile)
    return None


def resume_alpt_dm_bias_LHC_pylauncher(i0, i1, jobid, nodes=9, time=2, queue='normal',
                                       cores=12):
    '''Resume an interrupted pylauncher alpt_dm_bias_LHC job.'''
    workdir = os.environ.get('WORK', os.getcwd())
    scratch = os.environ.get('SCRATCH', workdir)

    pyl_workdir = os.path.join(scratch, 'pylauncher_alpt_dm_bias_LHC_%i_%i_%s' % (i0, i1, jobid))
    queuestate = os.path.join(pyl_workdir, 'queuestate')
    if not os.path.exists(queuestate):
        raise FileNotFoundError(
            'no queuestate at %s — has this job been run before?' % queuestate)

    slurm = '\n'.join(_header('alpt_dm_bias_LHC.pyl.resume', i0, i1, nodes, time, queue, workdir) + [
        'python %s resume-run %s %i' % (os.path.abspath(__file__), queuestate, cores),
        '',
    ])

    slurmfile = os.path.join(workdir, 'alpt_dm_bias_LHC_pylauncher_resume_%i_%i.slurm' % (i0, i1))
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
        cmdfile, pyl_workdir, cores = sys.argv[2], sys.argv[3], int(sys.argv[4])
        pylauncher.ClassicLauncher(cmdfile, workdir=pyl_workdir, debug='job', delay=0.01, cores=cores)

    elif subcmd == 'resume-run':
        import pylauncher
        queuestate, cores = sys.argv[2], int(sys.argv[3])
        pylauncher.ResumeClassicLauncher(queuestate, debug='job', delay=0.01, cores=cores)

    elif subcmd == 'resume':
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument('subcmd')
        p.add_argument('i0', type=int)
        p.add_argument('i1', type=int)
        p.add_argument('--jobid', type=str, required=True)
        p.add_argument('--nodes', type=int, default=9)
        p.add_argument('--time', type=float, default=2.0)
        p.add_argument('--queue', type=str, default='normal')
        p.add_argument('--cores-per-task', type=int, default=12)
        args = p.parse_args()
        resume_alpt_dm_bias_LHC_pylauncher(
            args.i0, args.i1, args.jobid, nodes=args.nodes, time=args.time,
            queue=args.queue, cores=args.cores_per_task)

    else:
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument('i0', type=int)
        p.add_argument('i1', type=int)
        p.add_argument('--nodes', type=int, default=9)
        p.add_argument('--time', type=float, default=4.0)
        p.add_argument('--queue', type=str, default='normal')
        p.add_argument('--cores-per-task', type=int, default=12)
        p.add_argument('--skip-done', action='store_true')
        args = p.parse_args()
        run_alpt_dm_bias_LHC_pylauncher(
            args.i0, args.i1, nodes=args.nodes, time=args.time, queue=args.queue,
            cores=args.cores_per_task, skip_done=args.skip_done)
