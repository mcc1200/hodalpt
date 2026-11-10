'''
alpt_sense_pylauncher.py

Runs ALPT (alpt_sense.py) for batches of the cosmologies in cosmo_samples.npy
across TACC nodes with pylauncher. i0, i1 are ROW indices (idx_cosmo in
[i0, i1), clipped to the number of rows). Per batch:

    1. python alpt_sense_pylauncher.py check 0 280            # ALPT status
    2. python alpt_sense_pylauncher.py run   0 280 --nodes 4 --time 1.5
    3. python alpt_sense_pylauncher.py check 0 280            # all ALPT complete?

`run` skips realizations whose bias spectrum (spec.noRSD.<i>.h5) is already
written. If the ALPT outputs are complete (all files CSbox_galaxy reads, at the
right size) but the spectrum is missing, only the bias stage is rerun (--bias).

Memory per task limits tasks/node (128 // cores-per-task). Default 10 -> 12
tasks/node, the setting that worked for 1000 alpt_dm_bias_LHC.py realizations
(9 nodes, 3.5 h); these tasks run the same stages. For a quick check, e.g.
`run 0 3 --nodes 1 --time 1 --queue development --measure`.

Do NOT call the 'launch' subcommand directly; the SLURM script invokes it.
'''
import os
import sys
import numpy as np

from alpt_sense import DM_FILES, alpt_missing, alpt_complete, bias_complete, n_cosmo

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(SCRIPTDIR, 'alpt_sense.py')


def cosmo_batch(i0, i1):
    '''Cosmology indices [i0, i1) (rows of cosmo_samples.npy), clipped to its length.'''
    return list(range(i0, min(i1, n_cosmo())))


def run_alpt_sense_pylauncher(idx_batch, tag, nodes=4, time=1.5, queue='normal', cores=10,
                              measure=False):
    '''Submit alpt_sense.py for each cosmology index in idx_batch via pylauncher.

    measure : wrap each task in `/usr/bin/time -v`, so its log ends with the
        task's peak memory ("Maximum resident set size") and wall time. sacct
        does not see pylauncher tasks, so this is the only way to get them.'''
    workdir = os.environ.get('WORK', os.getcwd())
    scratch = os.environ.get('SCRATCH', workdir)

    # a realization is done once its bias spectrum is written; if only the DM stage
    # is done, run just the bias stage (--bias) instead of redoing ALPT
    todo = [i for i in idx_batch if not bias_complete(i)]
    n_bias_only = sum(alpt_complete(i) for i in todo)
    print('%i in batch: %i already complete, %i to run (%i bias stage only)'
          % (len(idx_batch), len(idx_batch) - len(todo), len(todo), n_bias_only))
    if not todo:
        print('nothing to submit')
        return None

    hr = int(np.floor(time))
    mn = int((time * 60) % 60)

    logdir = os.path.join(scratch, 'alpt_sense_logs_%s' % tag)
    os.makedirs(logdir, exist_ok=True)
    cmdfile = os.path.join(scratch, 'alpt_sense_cmds_%s.txt' % tag)
    with open(cmdfile, 'w') as f:
        prefix = '/usr/bin/time -v ' if measure else ''
        for i in todo:
            stage = ' --bias' if alpt_complete(i) else ''
            f.write('%spython %s %i%s > %s/cosmo_%i.log 2>&1\n' % (prefix, WORKER, i, stage, logdir, i))
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
    '''Print ALPT status counts for idx_batch.'''
    done, bias_todo, partial, todo = [], [], [], []
    for i in idx_batch:
        bad = alpt_missing(i)
        if not bad and bias_complete(i):
            done.append(i)
        elif not bad:
            bias_todo.append(i)
        elif len(bad) < len(DM_FILES):  # some outputs written, some not
            partial.append(i)
            if len(partial) <= 10:
                print('cosmo %i: partial ALPT outputs, missing %s' % (i, bad))
        else:
            todo.append(i)
    print('%i in batch: %i fully complete, %i ALPT done but no bias spectrum, '
          '%i partial ALPT, %i not started'
          % (len(idx_batch), len(done), len(bias_todo), len(partial), len(todo)))
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
    p.add_argument('cmd', choices=['run', 'check'])
    p.add_argument('i0', type=int)
    p.add_argument('i1', type=int)
    p.add_argument('--nodes', type=int, default=4)
    p.add_argument('--time', type=float, default=1.5)
    p.add_argument('--queue', type=str, default='normal')
    p.add_argument('--cores-per-task', type=int, default=10)
    p.add_argument('--measure', action='store_true',
                   help='log per-task peak memory and wall time via /usr/bin/time -v')
    args = p.parse_args()

    batch = cosmo_batch(args.i0, args.i1)
    if args.cmd == 'run':
        run_alpt_sense_pylauncher(batch, '%i_%i' % (args.i0, args.i1), nodes=args.nodes,
                                  time=args.time, queue=args.queue, cores=args.cores_per_task,
                                  measure=args.measure)
    else:
        check_alpt_sense(batch)
