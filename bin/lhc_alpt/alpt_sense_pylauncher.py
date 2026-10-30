'''
alpt_sense_pylauncher.py

Runs ALPT (alpt_sense.py) for batches of the Quijote LHC realizations in
95p_lhc_idx.npy across TACC nodes with pylauncher, and checks/cleans the heavy
ICs afterward. i0, i1 are POSITIONS in 95p_lhc_idx.npy (idx[i0:i1]), not LHC
numbers. Per batch (e.g. 280 realizations at a time, ~15.7 GB of ICs each):

    1. transfer the batch's ICs (Globus)
    2. python alpt_sense_pylauncher.py check 0 280            # ICs/ALPT status
    3. python alpt_sense_pylauncher.py run   0 280 --nodes 4 --time 1.5
    4. python alpt_sense_pylauncher.py check 0 280            # all ALPT complete?
    5. python alpt_sense_pylauncher.py clean 0 280            # dry run
       python alpt_sense_pylauncher.py clean 0 280 --delete   # remove ICs

`run` skips realizations whose ALPT outputs are already complete, and ones with
neither ICs nor a gridded delta IC (reported). `clean` only deletes the ICs of
realizations whose ALPT outputs are complete (all files CSbox_galaxy reads, at
the right size).

Each task reads ~15.7 GB of ICs, so memory per task limits tasks/node
(128 // cores-per-task). Default 21 -> 6 tasks/node (guess; calibrate memory
and wall time with a small smoke test first, e.g.
`run 0 3 --nodes 1 --time 1 --queue development`, then `sacct` MaxRSS).

Do NOT call the 'launch' subcommand directly; the SLURM script invokes it.
'''
import os
import sys
import glob
import numpy as np

from alpt_sense import (LHC_ROOT, N_ICS, alpt_missing, alpt_complete,
                        delta_ok, ics_missing)

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(SCRIPTDIR, 'alpt_sense.py')


def lhc_batch(i0, i1):
    '''LHC realization numbers at positions [i0, i1) of 95p_lhc_idx.npy.'''
    idx = np.load(os.path.join(SCRIPTDIR, '95p_lhc_idx.npy'))
    return [int(i) for i in idx[i0:i1]]


def run_alpt_sense_pylauncher(idx_batch, tag, nodes=4, time=1.5, queue='normal', cores=21):
    '''Submit alpt_sense.py for each LHC realization in idx_batch via pylauncher.'''
    workdir = os.environ.get('WORK', os.getcwd())
    scratch = os.environ.get('SCRATCH', workdir)

    todo, n_done, no_ics = [], 0, []
    for i in idx_batch:
        if alpt_complete(i):
            n_done += 1
        elif not delta_ok(i) and ics_missing(i):
            no_ics.append(i)
        else:
            todo.append(i)
    print('%i in batch: %i already complete, %i missing ICs, %i to run'
          % (len(idx_batch), n_done, len(no_ics), len(todo)))
    if no_ics:
        print('missing ICs (first 10): %s' % no_ics[:10])
    if not todo:
        print('nothing to submit')
        return None

    hr = int(np.floor(time))
    mn = int((time * 60) % 60)

    logdir = os.path.join(scratch, 'alpt_sense_logs_%s' % tag)
    os.makedirs(logdir, exist_ok=True)
    cmdfile = os.path.join(scratch, 'alpt_sense_cmds_%s.txt' % tag)
    with open(cmdfile, 'w') as f:
        for i in todo:
            f.write('python %s %i > %s/lhc_%i.log 2>&1\n' % (WORKER, i, logdir, i))
    print('wrote %i commandlines to %s' % (len(todo), cmdfile))

    pyl_workdir_base = os.path.join(scratch, 'pylauncher_alpt_sense_%s' % tag)

    slurm = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -J alpt.sense.%s' % tag,
        '#SBATCH -o %s/alpt_sense.%s.%%j.out' % (workdir, tag),
        '#SBATCH -e %s/alpt_sense.%s.%%j.err' % (workdir, tag),
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
        '# the cores pylauncher reserves per task are this task\'s own, so let',
        '# webonx (if OpenMP) and numpy use them',
        'export OMP_NUM_THREADS=%i' % cores,
        'export HDF5_USE_FILE_LOCKING=FALSE',
        '',
        'python %s launch %s %s_${SLURM_JOB_ID} %i'
        % (os.path.abspath(__file__), cmdfile, pyl_workdir_base, cores),
        '',
    ])

    slurmfile = os.path.join(workdir, 'alpt_sense_pylauncher_%s.slurm' % tag)
    with open(slurmfile, 'w') as f:
        f.write(slurm)
    print('submitting %s' % slurmfile)
    os.system('sbatch %s' % slurmfile)
    return None


def check_alpt_sense(idx_batch):
    '''Print ICs / ALPT status counts for idx_batch.'''
    done, ready, no_ics, partial = [], [], [], []
    for i in idx_batch:
        if alpt_complete(i):
            done.append(i)
        elif not delta_ok(i) and ics_missing(i):
            no_ics.append(i)
        else:
            ready.append(i)
    for i in no_ics[:10]:
        print('LHC %i: cannot run, missing %s' % (i, ics_missing(i)))
    for i in [j for j in ready if len(alpt_missing(j)) < 10 and glob.glob(
            os.path.join(LHC_ROOT, str(j), 'alpt', 'super_*'))][:10]:
        print('LHC %i: partial ALPT outputs, missing %s' % (i, alpt_missing(i)))
    print('%i in batch: %i ALPT complete, %i ready to run (ICs ok), %i missing ICs'
          % (len(idx_batch), len(done), len(ready), len(no_ics)))
    return None


def clean_ics(idx_batch, delete=False):
    '''Remove ics.*.hdf5 for realizations whose ALPT outputs are complete.'''
    nbytes, n_clean, n_keep = 0, 0, 0
    for i in idx_batch:
        files = glob.glob(os.path.join(LHC_ROOT, str(i), 'ICs', 'ics.*.hdf5'))
        if not files:
            continue
        bad = alpt_missing(i)
        if bad:
            print('keeping LHC %i ICs: ALPT incomplete (%s)' % (i, bad))
            n_keep += 1
            continue
        nbytes += sum(os.path.getsize(f) for f in files)
        n_clean += 1
        if delete:
            for f in files:
                os.remove(f)
    print('%s %.2f TB of ICs from %i realizations; kept %i with incomplete ALPT'
          % ('removed' if delete else 'would remove (dry run, pass --delete)',
             nbytes / 1e12, n_clean, n_keep))
    return None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == 'launch':
        import pylauncher
        cmdfile, pyl_workdir, cores = sys.argv[2], sys.argv[3], int(sys.argv[4])
        pylauncher.ClassicLauncher(cmdfile, workdir=pyl_workdir, debug='job', delay=0.01, cores=cores)
        sys.exit(0)

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['run', 'check', 'clean'])
    p.add_argument('i0', type=int)
    p.add_argument('i1', type=int)
    p.add_argument('--nodes', type=int, default=4)
    p.add_argument('--time', type=float, default=1.5)
    p.add_argument('--queue', type=str, default='normal')
    p.add_argument('--cores-per-task', type=int, default=21)
    p.add_argument('--delete', action='store_true')
    args = p.parse_args()

    batch = lhc_batch(args.i0, args.i1)
    if args.cmd == 'run':
        run_alpt_sense_pylauncher(batch, '%i_%i' % (args.i0, args.i1), nodes=args.nodes,
                                  time=args.time, queue=args.queue, cores=args.cores_per_task)
    elif args.cmd == 'check':
        check_alpt_sense(batch)
    else:
        clean_ics(batch, delete=args.delete)
