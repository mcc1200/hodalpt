import os, sys
import multiprocessing as mp
import numpy as np
from scipy.special import erf, erfinv
import sbi
import h5py
import torch
import tarp
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hodalpt.sims import alpt as CS
from nbodykit.lab import ArrayCatalog, FFTPower

def cdf_transform(x, bounds):
    return _gaussian_cdf(x, 0, 1) * (bounds[1] - bounds[0]) + bounds[0]

def inv_cdf_transform(x, bounds):
    return _inv_gaussian_cdf((x - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)

def _gaussian_cdf(x, mu, sigma):
    return 0.5 * (1 + erf((x - mu) / (np.sqrt(2) * sigma)))

def _inv_gaussian_cdf(x, mu, sigma):
    return mu + sigma * np.sqrt(2) * erfinv(2 * x - 1)

cuda   = torch.cuda.is_available()
device = 'cuda:0' if cuda else 'cpu'

study_name = 'qphi_hod2nlb.exp20.maf.pdf+p0+q123.0.4.emb.noRSD.n45.d100.emb'

# paths: use the TACC locations when they exist, else the local ones
_TACC_NPE = '/work/11053/mcasas/ls6/hodalpt/bin/npe'
_TACC_DM  = '/corral/utexas/AST25023/simbig/quijote/fiducial_HR/0/alpt/'
NPE_DIR   = _TACC_NPE if os.path.isdir(_TACC_NPE) else '/Users/mcc3842/CosmicSim2025/hodalpt/bin/npe'
DM_DIR    = _TACC_DM  if os.path.isdir(_TACC_DM)  else '/Users/mcc3842/CosmicSim2025/data/quijote/fiducial_HR/0/alpt/'
# the pylauncher SLURM script stages the ALPT files to $SCRATCH and exports this,
# so every task reads local copies instead of corral (same as bias_fiducial_noRSD.py)
DM_DIR    = os.environ.get('HODALPT_DM_DIR', DM_DIR)

study_dir = os.path.join(NPE_DIR, 'studies', study_name)
TRIAL     = 36   # best optuna trial for this study (hardcoded; no .db needed)
HOD_FILE  = os.path.join(NPE_DIR, 'data', 'bias_fid_HOD_noRSD_data.hdf5')
# CMASS_FIL = os.path.join(NPE_DIR, 'data', 'CMASS_fid_HOD_noRSD.h5')
CIC_BoxSize = 1000.
CIC_Nmesh   = 40
CIC_bins    = np.arange(-0.5, 85.5, 1)
CIC_bin_centers = 0.5 * (CIC_bins[1:] + CIC_bins[:-1])




def pick_test_hods(study_dir, study_name, seed=0, fid_tol=0.10, n_fid_fallback=5):
    """Randomly pick one held-out HOD sim from the lower ngs quartile, upper ngs
    quartile, and near the CMASS fiducial ngs. Quartiles are over the held-out
    (test) sims only. Fiducial-ish = |ngs - ngs_cmass| / ngs_cmass <= fid_tol;
    if none qualify, falls back to the n_fid_fallback closest test sims.

    Returns {label: dict} with the sim's row in HOD_FILE ('sim_idx'), raw
    'theta_hod' (6,), z-scored 'x_hod' (6,) ready for qphi, and the x_obs
    pieces 'k', 'p0', 'cic_pdf' to overplot against the PPC."""
    rng  = np.random.default_rng(seed)
    norm = np.load(os.path.join(study_dir, study_name + '.hod_norm.npz'), allow_pickle=True)
    test_idx = np.asarray(norm['test_sims_idx'])

    with h5py.File(HOD_FILE) as f:
        ngs_all   = np.atleast_1d(f['ngs'][()])
        theta_hod = f['theta'][...][:, :6]
        k         = f['k'][...]
        # h5py fancy indexing needs sorted idx; we index in-memory instead
        p0_all    = f['p0'][...]
        pdf_all   = f['cic_pdf'][...]
        q123_all  = f['q123'][...] if 'q123' in f else None
    # with h5py.File(CMASS_FIL) as f:
        # ngs_cmass = float(np.atleast_1d(f['ngs'][()])[0])
    ngs_cmass = 444682.0
    ngs_test = ngs_all[test_idx]
    q25, q75 = np.percentile(ngs_test, [25, 75])
    fid_mask = np.abs(ngs_test - ngs_cmass) / ngs_cmass <= fid_tol
    if not fid_mask.any():
        fid_mask = np.zeros_like(fid_mask)
        fid_mask[np.argsort(np.abs(ngs_test - ngs_cmass))[:n_fid_fallback]] = True
        print('no test sims within %.0f%% of CMASS ngs; using %i closest' % (100*fid_tol, n_fid_fallback))

    pools = {'lower_quartile': ngs_test <= q25,
             'fiducial':       fid_mask,
             'upper_quartile': ngs_test >= q75}

    out = {}
    for label, m in pools.items():
        sim = int(rng.choice(test_idx[m]))
        th  = theta_hod[sim]
        out[label] = dict(
            sim_idx=sim, ngs=float(ngs_all[sim]), ngs_cmass=ngs_cmass,
            theta_hod=th, x_hod=(th - norm['hod_mean']) / (norm['hod_std'] + 1e-8),
            k=k, p0=p0_all[sim], cic_pdf=pdf_all[sim],
            q123=None if q123_all is None else q123_all[sim],
        )
    return out


def sample_nlb(qphi, sel, norm_path, n=50, device='cpu'):
    """Draw n theta_NLB from qphi(nlb | hod) for one entry of pick_test_hods().
    Returns (n, 80) physical-space theta (zeros in pruned slots), ready for the
    forward model (noRSD: no RSD block)."""
    norm   = np.load(norm_path, allow_pickle=True)
    bounds = norm['bounds_keep']                   # (2, n_idx)
    aidx   = norm['active_idx']
    x = torch.tensor(sel['x_hod'], dtype=torch.float32, device=device)
    z = qphi.sample((n,), x=x, show_progress_bars=False).cpu().numpy()   # (n, n_idx) Gaussian
    phys = cdf_transform(z, bounds)
    theta80 = np.zeros((n, 80))
    theta80[:, aidx] = phys
    return theta80

def cic_pdf(cat, BoxSize=CIC_BoxSize, Nmesh=CIC_Nmesh, bins=CIC_bins):

    mesh = cat.to_mesh(Nmesh=Nmesh, BoxSize=BoxSize, resampler='cic',
                        compensated=False, interlaced=False)
    field = mesh.paint(mode='real')
    mean_n = cat.csize / (BoxSize ** 3)
    cell_vol = (BoxSize / Nmesh) ** 3
    counts = field.value.ravel() * mean_n * cell_vol
    pdf, edges = np.histogram(counts, bins=bins, density=False)
    pdf = pdf / pdf.sum()
    return pdf, edges


def alpt_pk(theta_gal, theta_rsd, bias_model='nonlocal2', DM_DIR=DM_DIR, compute_bispec=False, rsd=True):


    xyz  = CS.CSbox_galaxy(theta_gal, theta_rsd, DM_DIR,
                           bias_model=bias_model, rsd=rsd, subgrid=True,
                           silent=True)
    _cat = ArrayCatalog({'Position': xyz}, BoxSize=1000.)
    poles = [0, 2] if rsd else [0]
    r    = FFTPower(_cat, mode='1d', Nmesh=256, dk=0.005, kmin=0.008,
                    los=[0, 0, 1], poles=poles)

    bispec = None
    if compute_bispec:
        from hodalpt.stats import B0_periodic
        bk = B0_periodic(xyz.T, w=None, Lbox=1000., fft='pyfftw', silent=True)
        bispec = (bk['i_k1'], bk['i_k2'], bk['i_k3'], bk['b123'], bk['q123'])

    
    pdf, _pdf_edges = cic_pdf(_cat)

    return (r.poles['k'],
            r.poles['power_0'].real - r.poles.attrs['shotnoise'],
            r.poles['power_2'].real if rsd else None,
            xyz.shape[0],
            bispec,
            pdf)

def pars_to_theta(pars):
    """80-dim noRSD NLB vector -> theta_gal dict for CSbox_galaxy (no theta_rsd)."""
    return {
        'alpha':  pars[:16].reshape((4, 4)),
        'beta':   pars[16:32].reshape((4, 4)),
        'nmean':  pars[32:48].reshape((4, 4)),
        'rhoeps': pars[48:64].reshape((4, 4)),
        'eps':    pars[64:80].reshape((4, 4)),
    }


def _alpt_worker(pars):
    """Top-level (module scope) so multiprocessing's spawn context can pickle it."""
    return alpt_pk(pars_to_theta(pars), None, rsd=False)


def run_alpt(theta80, n_workers=1, desc='ALPT'):
    """Forward model each row of theta80 (n, 80) with alpt_pk(rsd=False).
    Returns k_pk, p0 (n, n_k), ng (n,), pdf (n, n_bins)."""
    if n_workers > 1:
        ctx = mp.get_context('spawn')
        with ctx.Pool(n_workers) as pool:
            results = list(tqdm(pool.imap(_alpt_worker, list(theta80)),
                                total=len(theta80), desc=desc, unit='sim', ncols=72))
    else:
        results = [_alpt_worker(p) for p in tqdm(theta80, desc=desc, unit='sim', ncols=72)]
    k_pk = results[0][0]
    p0   = np.array([r[1] for r in results])
    ng   = np.array([r[3] for r in results])
    pdf  = np.array([r[5] for r in results])
    return k_pk, p0, ng, pdf


def _align_obs(k_src, obs, k_target):
    if len(k_src) == len(k_target) and np.allclose(k_src, k_target, rtol=1e-4):
        return obs
    return np.interp(k_target, k_src, obs)


def make_figure(sels, ppc, out_path, kmax_train=0.4):
    """Row 1: P0 (3 HODs); row 2: P0 fractional residuals; row 3: CIC pdf;
    row 4: CIC pdf fractional residuals (empty obs bins masked). Residuals are
    (ALPT - HOD sim) / HOD sim, ylim (-1, 1), median + 16-84% band."""
    labels = list(sels)
    fig = plt.figure(figsize=(14, 11))
    gs  = fig.add_gridspec(4, len(labels), height_ratios=[3, 1, 3, 1], hspace=0.08, wspace=0.28)

    for c, label in enumerate(labels):
        s = sels[label]
        k_pk, p0, ng, pdf = ppc[label]
        good  = np.isfinite(k_pk)
        k_pk, p0 = k_pk[good], p0[:, good]

        # ── P0 ──
        ax = fig.add_subplot(gs[0, c])
        for sp in p0:
            ax.plot(k_pk, sp, color='steelblue', lw=0.5, alpha=0.5)
        ax.plot(k_pk, np.median(p0, axis=0), color='steelblue', lw=1.5, label='emulator+ALPT median')
        ax.plot(s['k'], s['p0'], color='black', lw=2, zorder=3, label='HOD sim %i' % s['sim_idx'])
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.axvline(kmax_train, color='k', ls='--', alpha=0.4)
        ax.set_title('%s\n$n_g^{\\rm obs}$=%.2e  $n_g^{\\rm pred}$=%.2e$\\pm$%.1e'
                     % (label, s['ngs'], ng.mean(), ng.std()), fontsize=9)
        if c == 0: ax.set_ylabel(r'$P_0(k)\;[(\mathrm{Mpc}/h)^3]$')
        ax.legend(fontsize=7)
        plt.setp(ax.get_xticklabels(), visible=False)

        axr = fig.add_subplot(gs[1, c], sharex=ax)
        obs = _align_obs(s['k'], s['p0'], k_pk)
        res = (p0 - obs[None, :]) / obs[None, :]
        lo, hi = np.percentile(res, [16, 84], axis=0)
        axr.fill_between(k_pk, lo, hi, color='steelblue', alpha=0.25, lw=0)
        axr.plot(k_pk, np.median(res, axis=0), color='steelblue', lw=1.2)
        axr.axhline(0, color='k', lw=0.8, ls=':')
        axr.axvline(kmax_train, color='k', ls='--', alpha=0.4)
        axr.set_ylim(-1, 1)
        axr.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
        if c == 0: axr.set_ylabel(r'$\Delta P_0/P_0$')

        # ── CIC pdf ──
        axp = fig.add_subplot(gs[2, c])
        for p in pdf:
            axp.plot(CIC_bin_centers, p, color='seagreen', lw=0.5, alpha=0.3, drawstyle='steps-mid')
        axp.plot(CIC_bin_centers, np.median(pdf, axis=0), color='seagreen', lw=1.5,
                 drawstyle='steps-mid', label='emulator+ALPT median')
        axp.plot(CIC_bin_centers, s['cic_pdf'], color='black', lw=2, zorder=3,
                 drawstyle='steps-mid', label='HOD sim %i' % s['sim_idx'])
        axp.set_yscale('log')
        if c == 0: axp.set_ylabel('CIC pdf')
        axp.legend(fontsize=7)
        plt.setp(axp.get_xticklabels(), visible=False)

        axpr = fig.add_subplot(gs[3, c], sharex=axp)
        pobs = s['cic_pdf']
        m    = pobs > 0
        resp = (pdf[:, m] - pobs[None, m]) / pobs[None, m]
        lo, hi = np.percentile(resp, [16, 84], axis=0)
        axpr.fill_between(CIC_bin_centers[m], lo, hi, color='seagreen', alpha=0.25, lw=0, step='mid')
        axpr.plot(CIC_bin_centers[m], np.median(resp, axis=0), color='seagreen', lw=1.2, drawstyle='steps-mid')
        axpr.axhline(0, color='k', lw=0.8, ls=':')
        axpr.set_ylim(-1, 1)
        axpr.set_xlabel('counts in cell')
        if c == 0: axpr.set_ylabel(r'$\Delta{\rm pdf}/{\rm pdf}$')

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved', out_path)


# ══════════════════════════════════════════════════════════════════════════════
# Bispectrum run: N_BISPEC draws per HOD, one pylauncher task per draw
#   bk_launch : draw theta_NLB from the emulator -> bk/draws.npz, write the
#               pylauncher command file + SLURM script, sbatch
#   bk_draw   : (internal, one pylauncher task) forward model one draw with
#               compute_bispec=True -> bk/<label>.<i>.npz
#   bk_plot   : gather bk/*.npz -> Q123 figure (+ P0/pdf figure from same draws)
# Every task writes its OWN file (never a shared one), atomically via
# tmp + os.replace, so there is no HDF5/file locking to worry about and a task
# killed at walltime can't leave a truncated file that looks finished.
# ══════════════════════════════════════════════════════════════════════════════
N_BISPEC   = 50          # draws per HOD
BK_SEED    = 1           # pick_test_hods seed (must match between launch and plot)
BK_NODES   = 2
BK_TIME    = 2         # hours
BK_QUEUE   = 'normal'
BK_ACCOUNT = 'AST25023'
PYL_CORES  = 9           # cores per task -> 128 // 9 = 14 concurrent draws per node
SKIP_DONE  = True        # skip draws whose output file already exists
Q_KMIN, Q_KMAX = 0.01, 0.4

BK_DIR     = os.path.join(study_dir, 'bk')
DRAWS_FILE = os.path.join(BK_DIR, 'draws.npz')
LABELS     = ['lower_quartile', 'fiducial', 'upper_quartile']

# ALPT density-field files CSbox_galaxy() reads (subgrid=True, nonlocal2, zsnap=0.5);
# staged to $SCRATCH once per job, same list as bias_fiducial_noRSD_pylauncher.py
ALPT_SRC_DIR = '/corral/utexas/AST25023/simbig/quijote/fiducial_HR/0/alpt'
ALPT_FILES = [
    'super_deltaBOX.dat', 'Tweb_.dat', 'TwebDelta_.dat',
    'VExEULz0.500.dat', 'VEyEULz0.500.dat', 'VEzEULz0.500.dat',
    'super_BOXposx.dat', 'super_BOXposy.dat', 'super_BOXposz.dat',
    '2LPT.param',
]


def bk_path(label, i):
    return os.path.join(BK_DIR, '%s.%i.npz' % (label, i))


def bk_draw(label, i):
    """One pylauncher task: forward model draw i of HOD `label` incl. bispectrum."""
    out = bk_path(label, i)
    if SKIP_DONE and os.path.exists(out):
        print('already done:', out); return
    theta80 = np.load(DRAWS_FILE)['theta_' + label][i]
    k_pk, p0, _, ng, bispec, pdf = alpt_pk(pars_to_theta(theta80), None,
                                           compute_bispec=True, rsd=False)
    i_k1, i_k2, i_k3, b123, q123 = bispec
    tmp = out[:-len('.npz')] + '.tmp.npz'
    np.savez(tmp, k=k_pk, p0=p0, ng=ng, pdf=pdf,
             i_k1=i_k1, i_k2=i_k2, i_k3=i_k3, b123=b123, q123=q123)
    os.replace(tmp, out)
    print('saved', out)


def bk_launch():
    os.makedirs(BK_DIR, exist_ok=True)
    workdir = os.environ.get('WORK', os.getcwd())
    scratch = os.environ.get('SCRATCH', workdir)

    # ── 1. sample theta_NLB once (fixed seeds) so every task sees the same draws.
    # Never resample if draws.npz exists: with SKIP_DONE a relaunch would
    # otherwise mix finished outputs from old draws with new ones.
    if os.path.exists(DRAWS_FILE):
        print('using existing', DRAWS_FILE, '(delete it, and bk/*.npz, to resample)')
    else:
        sels = pick_test_hods(study_dir, study_name, seed=BK_SEED)
        qphi = torch.load(os.path.join(study_dir, '%s.%i.pt' % (study_name, TRIAL)),
                          map_location=device, weights_only=False)
        norm_path = os.path.join(study_dir, study_name + '.hod_norm.npz')
        torch.manual_seed(BK_SEED)
        arrs = {}
        for label, s in sels.items():
            arrs['theta_' + label]   = sample_nlb(qphi, s, norm_path, n=N_BISPEC, device=device)
            arrs['sim_idx_' + label] = s['sim_idx']
            print('%s: sim %i  ngs=%.3e' % (label, s['sim_idx'], s['ngs']))
        np.savez(DRAWS_FILE, **arrs)

    # ── 2. command file: one line per draw ──
    logdir = os.path.join(scratch, 'emu_bk_logs')
    os.makedirs(logdir, exist_ok=True)
    cmdfile = os.path.join(scratch, 'emu_bk_cmds.txt')
    n = 0
    with open(cmdfile, 'w') as f:
        for label in LABELS:
            for i in range(N_BISPEC):
                if SKIP_DONE and os.path.exists(bk_path(label, i)):
                    continue
                f.write('python %s bk_draw %s %i > %s/%s.%i.log 2>&1\n'
                        % (os.path.abspath(__file__), label, i, logdir, label, i))
                n += 1
    print('wrote %i commandlines to %s' % (n, cmdfile))
    if n == 0:
        print('all draws already complete, nothing to submit'); return

    hr, mn = int(np.floor(BK_TIME)), int((BK_TIME * 60) % 60)
    pyl_workdir_base = os.path.join(scratch, 'pylauncher_emu_bk')
    slurm = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -J emu.bk.pyl',
        '#SBATCH -o %s/emu_bk.pyl.%%j.out' % workdir,
        '#SBATCH -e %s/emu_bk.pyl.%%j.err' % workdir,
        '#SBATCH -p %s' % BK_QUEUE,
        '#SBATCH -N %i' % BK_NODES,
        '#SBATCH -n %i' % (BK_NODES * 128),
        '#SBATCH --time=%s:%s:00' % (str(hr).zfill(2), str(mn).zfill(2)),
        '#SBATCH -A %s' % BK_ACCOUNT,
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
        '# stage the ALPT files to $SCRATCH once per job (see bias_fiducial_noRSD_pylauncher.py)',
        'export HODALPT_DM_DIR=$SCRATCH/alpt_cache_${SLURM_JOB_ID}/',
        'mkdir -p $HODALPT_DM_DIR',
        'cp %s $HODALPT_DM_DIR' % ' \\\n    '.join(
            [os.path.join(ALPT_SRC_DIR, fn) for fn in ALPT_FILES]),
        '',
        'python %s bk_pyl %s %s_${SLURM_JOB_ID}' % (os.path.abspath(__file__), cmdfile, pyl_workdir_base),
        '',
        'rm -rf $HODALPT_DM_DIR',
        '',
    ])
    slurmfile = os.path.join(workdir, 'emu_bk_pylauncher.slurm')
    with open(slurmfile, 'w') as f:
        f.write(slurm)
    print('submitting %s' % slurmfile)
    os.system('sbatch %s' % slurmfile)


def _load_bk(label):
    """Gather finished draws for one HOD. Returns dict of stacked arrays, or None."""
    files = [bk_path(label, i) for i in range(N_BISPEC) if os.path.exists(bk_path(label, i))]
    if not files:
        return None
    ds = [np.load(f) for f in files]
    print('%s: %i / %i draws found' % (label, len(ds), N_BISPEC))
    return dict(k=ds[0]['k'], p0=np.array([d['p0'] for d in ds]),
                ng=np.array([d['ng'] for d in ds]), pdf=np.array([d['pdf'] for d in ds]),
                i_k1=ds[0]['i_k1'], i_k2=ds[0]['i_k2'], i_k3=ds[0]['i_k3'],
                q123=np.array([d['q123'] for d in ds]))


def make_bk_figure(sels, bk, out_path):
    """One column, one (Q123, residual) pair per HOD. Triangles sorted by largest
    side and cut to [Q_KMIN, Q_KMAX] as in tarp_ppc.py; residual is absolute
    (Q123_ALPT - Q123_HOD sim), ylim (-1, 1), median + 16-84% band."""
    labels = [l for l in LABELS if bk.get(l) is not None]
    fig = plt.figure(figsize=(12, 3.6 * len(labels)))
    gs  = fig.add_gridspec(2 * len(labels), 1, height_ratios=[3, 1] * len(labels), hspace=0.12)
    k_fund = 2 * np.pi / 1000.

    for r, label in enumerate(labels):
        s, b = sels[label], bk[label]
        order = np.lexsort((b['i_k3'], b['i_k2'], b['i_k1']))
        i1, i2, i3 = b['i_k1'][order], b['i_k2'][order], b['i_k3'][order]
        q123 = b['q123'][:, order]
        klim = ((i1 * k_fund > Q_KMIN) & (i2 * k_fund > Q_KMIN) & (i3 * k_fund > Q_KMIN) &
                (i1 * k_fund <= Q_KMAX) & (i2 * k_fund <= Q_KMAX) & (i3 * k_fund <= Q_KMAX))
        q123 = q123[:, klim]
        tri  = np.arange(klim.sum())

        qobs = None
        if s.get('q123') is not None and len(s['q123']) == len(order):
            qobs = s['q123'][order][klim]
        else:
            print('%s: Q123 obs missing or triangle-count mismatch -- skipping obs overlay' % label)

        ax = fig.add_subplot(gs[2 * r])
        ax.plot(tri, q123.T, color='darkorange', alpha=0.1, lw=0.7)
        ax.plot([], [], color='darkorange', alpha=0.5, label='emulator+ALPT (n=%i)' % len(q123))
        if qobs is not None:
            ax.plot(tri, qobs, color='black', lw=2, zorder=10, label='HOD sim %i' % s['sim_idx'])
        ax.set_ylabel(r'$Q_{123}$')
        ax.set_title('%s   $n_g^{\\rm obs}$=%.2e  $n_g^{\\rm pred}$=%.2e$\\pm$%.1e'
                     % (label, s['ngs'], b['ng'].mean(), b['ng'].std()), fontsize=9, loc='left')
        ax.legend(fontsize=7)
        plt.setp(ax.get_xticklabels(), visible=False)

        axr = fig.add_subplot(gs[2 * r + 1], sharex=ax)
        if qobs is not None:
            res = q123 - qobs[None, :]
            lo, hi = np.percentile(res, [16, 84], axis=0)
            axr.fill_between(tri, lo, hi, color='darkorange', alpha=0.25, lw=0)
            axr.plot(tri, np.median(res, axis=0), color='darkorange', lw=1.2)
        axr.axhline(0, color='k', lw=0.8, ls=':')
        axr.set_ylim(-1, 1)
        axr.set_ylabel(r'$\Delta Q_{123}$')
        if r == len(labels) - 1:
            axr.set_xlabel('triangle configuration index, largest-side order')

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved', out_path)


def bk_plot():
    sels = pick_test_hods(study_dir, study_name, seed=BK_SEED)
    saved = np.load(DRAWS_FILE)
    for l in LABELS:
        assert int(saved['sim_idx_' + l]) == sels[l]['sim_idx'], \
            'HOD pick for %s changed since bk_launch (seed / test split?)' % l
    bk = {l: _load_bk(l) for l in LABELS}
    make_bk_figure(sels, bk, os.path.join(BK_DIR, '%s.%i.emu_bk_q123.png' % (study_name, TRIAL)))
    # the same catalogs give P0 / pdf for free: reuse the P0 + CIC figure
    ppc = {l: (b['k'], b['p0'], b['ng'], b['pdf']) for l, b in bk.items() if b is not None}
    make_figure({l: sels[l] for l in ppc}, ppc,
                os.path.join(BK_DIR, '%s.%i.emu_bk_p0pdf.png' % (study_name, TRIAL)))


def run_ppc_small():
    """Original quick P0 + CIC pdf check (N_ALPT draws per HOD, single node, mp.Pool)."""
    N_ALPT    = 10   # ALPT realizations per HOD (small for now, for time)
    N_WORKERS = 4    # parallel ALPT workers (each holds a full ALPT catalog in RAM)

    fqphi = os.path.join(study_dir, '%s.%i.pt' % (study_name, TRIAL))
    qphi  = torch.load(fqphi, map_location=device, weights_only=False)
    print('loaded:', fqphi)

    norm_path = os.path.join(study_dir, study_name + '.hod_norm.npz')
    sels = pick_test_hods(study_dir, study_name, seed=1)
    ppc  = {}
    for label, s in sels.items():
        print('%s: sim %i  ngs=%.3e' % (label, s['sim_idx'], s['ngs']))
        theta80 = sample_nlb(qphi, s, norm_path, n=N_ALPT, device=device)
        ppc[label] = run_alpt(theta80, N_WORKERS, desc='ALPT %s' % label)

    prefix = os.path.join(study_dir, '%s.%i.emu_ppc' % (study_name, TRIAL))
    # save raw ALPT outputs so the figure can be re-made without re-running the forward model
    np.savez(prefix + '.npz', **{'%s_%s' % (l, n): v for l, r in ppc.items()
                                 for n, v in zip(('k', 'p0', 'ng', 'pdf'), r)},
             **{'%s_sim_idx' % l: s['sim_idx'] for l, s in sels.items()})
    make_figure(sels, ppc, prefix + '.png')


MODE = 'bk_launch'    # 'ppc' (small local P0/pdf check) | 'bk_launch' (submit) | 'bk_plot'

if __name__ == '__main__':
    # internal subcommands, invoked by the SLURM script / pylauncher, not by hand
    if len(sys.argv) > 1 and sys.argv[1] == 'bk_draw':
        bk_draw(sys.argv[2], int(sys.argv[3]))
    elif len(sys.argv) > 1 and sys.argv[1] == 'bk_pyl':
        import pylauncher
        pylauncher.ClassicLauncher(sys.argv[2], workdir=sys.argv[3], debug='job',
                                   delay=0.01, cores=PYL_CORES)
    elif MODE == 'ppc':
        run_ppc_small()
    elif MODE == 'bk_launch':
        bk_launch()
    elif MODE == 'bk_plot':
        bk_plot()
    else:
        raise ValueError('unknown MODE %r' % MODE)
