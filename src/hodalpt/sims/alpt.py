'''

module to construct CosmicSignal ALPT galaxy mocks 


author(s):
    * Francesco Sinigaglia 
    * ChangHoon Hahn: minor modifications 

'''
import numpy as np 
import subprocess
import os, sys, glob 

import configparser
from numba import njit, prange

from . import cwc as C
from . import quijote as Q


def CSbox_galaxy(theta_gal, theta_rsd, dm_dir, Ngrid=256, Lbox=1000.,
                 zsnap=0.5, lambdath_tweb=0.0, lambdath_twebdelta = 0.0,
                 seed=123456, silent=True): 
    ''' construct CosmicSignal galaxy mock given DM box. Applies the bias model
    in hodalpt.sims.cwc to specified ALPT DM output 

    .. code-block:: python
        
       from hodalpt.sims import alpt as Alpt
       theta_gal = {'alpha': 1.9230, 'beta': 2.0253, 'dth': -0.7889, 'rhoeps':
                    14.6874, 'eps': 0.5616, 'nmean': 3.3e-4}
       theta_rsd = {'bv': 0.7289, 'bb': 1.1652, 'betarsd': 1.3136, 'gamma':
                    0.4944}

       xyz = Alpt.CSbox_galaxy(theta_gal, theta_rsd, '/Users/hahnchanghoon/data/simbig/quijote/fiducial/0/alpt/', silent=False)


    paramters
    ---------
    theta_gal : dict
        Dictionary specify the galaxy bias parameters: 'alpha', 'beta', 'dth',
        'rhoeps', 'eps'. For some default values use: `theta_gal = { 'alpha':
        1.9230, 'beta': 2.0253, 'dth': -0.7889, 'rhoeps': 14.6874, 'eps':
        0.5616, 'nmean': 3.3e-4}`
        
    theta_rsd : dict
        Dictionary specifying the RSD parameters: bv, bb, betarsd, gamma. For
        some default values use `theta_rsd = {'bv': 0.7289, 'bb': 1.1652,
        'betarsd': 1.3136, 'gamma': 0.4944}`

    dm_dir : str
        Directory with the ALPT DM files 


    return 
    ------
    xyz : array 
        Ngal x 3 array specifying the x, y, z position of galaxies.
    '''
    np.random.seed(seed)

    assert os.path.isdir(dm_dir), "specify correct directory for the DM files"

    # find file suffix and check for consistency 
    suffix = 'OM'+(glob.glob(os.path.join(dm_dir, 'deltaBOXOM*'))[0].split('deltaBOXOM')[-1]).split('.gz')[0]
    omega_m = float(suffix.split('OM')[1].split('OL')[0])
    if not silent: print(suffix)
    if not silent: print('Omega_m %f' % omega_m) 
    def Fname(prefix): return os.path.join(dm_dir, '%s%s' % (prefix, suffix))

    dm_filename         = Fname('deltaBOX')
    tweb_filename       = Fname('Tweb_')
    twebdelta_filename  = Fname('TwebDelta_')

    vx_filename = dm_dir + 'VExEULz%3.3f.dat' % zsnap
    vy_filename = dm_dir + 'VEyEULz%3.3f.dat' % zsnap
    vz_filename = dm_dir + 'VEzEULz%3.3f.dat' % zsnap

    posx_filename = Fname('BOXposx')
    posy_filename =	Fname('BOXposy')
    posz_filename = Fname('BOXposz')
    
    assert os.path.isfile(dm_filename), 'missing %s' % dm_filename
    assert os.path.isfile(tweb_filename), 'missing %s' % tweb_filename
    assert os.path.isfile(twebdelta_filename), 'missing %s' % twebdelta_filename
    assert os.path.isfile(vx_filename), 'missing %s' % vx_filename
    assert os.path.isfile(vy_filename), 'missing %s' % vy_filename 
    assert os.path.isfile(vz_filename), 'missing %s' % vz_filename 
    assert os.path.isfile(posx_filename), 'missing %s' % posx_filename
    assert os.path.isfile(posy_filename), 'missing %s' % posy_filename
    assert os.path.isfile(posz_filename), 'missing %s' % posz_filename

    # parse galaxy bias parameters 
    alpha   = theta_gal['alpha']  
    beta    = theta_gal['beta']
    dth     = theta_gal['dth'] 
    rhoeps  = theta_gal['rhoeps']
    eps     = theta_gal['eps']     
    rhoepsprime = 0.
    epsprime    = 0.
    nmean   = theta_gal['nmean'] # yes, this is weird but lets not overthink it for now 


    # parse rsd parameters
    bv      = theta_rsd['bv'] 
    bb      = theta_rsd['bb']
    betarsd = theta_rsd['betarsd']
    gamma   = theta_rsd['gamma'] 
    
    # Observer positions            
    obspos = [Lbox/2., Lbox/2., Lbox/2.]

    lcell = Lbox/Ngrid

    xobs = obspos[0]
    yobs = obspos[1]
    zobs = obspos[2]
    
    if not silent: print('Reading input ...')
    # read inputs 
    delta = np.fromfile(dm_filename, dtype=np.float32)  # In real space
    delta = np.reshape(delta, (Ngrid,Ngrid,Ngrid))

    tweb = np.fromfile(tweb_filename, dtype=np.float32)  # In real space
    twebdelta = np.fromfile(twebdelta_filename, dtype=np.float32)  # In real space 

    # Positions
    posx = np.fromfile(posx_filename, dtype=np.float32)   
    posy = np.fromfile(posy_filename, dtype=np.float32) 
    posz = np.fromfile(posz_filename, dtype=np.float32)

    # Now they are velocity vectors
    vx = np.fromfile(vx_filename, dtype=np.float32)  
    vy = np.fromfile(vy_filename, dtype=np.float32) 
    vz = np.fromfile(vz_filename, dtype=np.float32) 

    # Reshape arrays from 1D to 3D --> reshape only arrays which have mesh structure, e.g. NOT positions
    #delta = np.reshape(delta, (Ngrid,Ngrid,Ngrid))
    tweb = np.reshape(tweb, (Ngrid,Ngrid,Ngrid))
    twebdelta = np.reshape(twebdelta, (Ngrid,Ngrid,Ngrid))

    vx = np.reshape(vx, (Ngrid,Ngrid,Ngrid))
    vy = np.reshape(vy, (Ngrid,Ngrid,Ngrid))
    vz = np.reshape(vz, (Ngrid,Ngrid,Ngrid))

    # Apply the bias and get halo/galaxy number counts
    if not silent: print('Getting number counts via parametric bias ...')
    ncounts = C.biasmodel_local_box(Ngrid, Lbox, delta,  nmean, alpha, beta, dth, rhoeps, eps, rhoepsprime, epsprime, xobs, yobs, zobs)
    ncountstot = np.sum(ncounts) # total number of objects
    if not silent: print('Number counts diagnostics (min, max, mean): ', np.amin(ncounts), np.amax(ncounts), np.mean(ncounts))

    ncounts = np.reshape(ncounts, (Ngrid, Ngrid, Ngrid))

    # Now assign positions
    if not silent: print('Preparing galaxy positions ...')
    posxarr_prep, posyarr_prep, poszarr_prep = C.prepare_indices_array(posx, posy, posz, Ngrid, Lbox)
    if not silent: print('Sampling galaxy positions ...')
    posx, posy, posz = C.sample_galaxies(Lbox, Ngrid, posxarr_prep, posyarr_prep, poszarr_prep, ncounts)


    if not silent: print('apply RSD ...')
    posx, posy, posz = C.real_to_redshift_space_local_box(delta, tweb, posx,
                                                          posy, posz, vx, vy,
                                                          vz, Ngrid, Lbox,
                                                          xobs, yobs, zobs, bv,
                                                          bb, betarsd, gamma,
                                                          zsnap, omega_m) 
    
    return np.vstack([posx, posy, posz]).T


def CSbox_alpt(config_file, outdir, seed=0, make_ics=True, return_pos=True, silent=True): 
    ''' wrapper for CosmicSignal ALPT code 
    '''
    # parse config file 
    config = configparser.ConfigParser()
    config.read(config_file)

    ngrid = int(config['SETUP']['ngrid'])
    lbox = config['SETUP']['lbox']

    zsnap = config['SETUP']['zsnap'].split(' ')
    assert len(zsnap) == 1
    zmin = zsnap[0] # hardcoded
    zmax = zsnap[0] # hardcoded

    lambdath_tweb = config['SETUP']['lambdath_tweb']
    lambdath_twebdelta = config['SETUP']['lambdath_twebdelta']

    # ICs
    ic_path = os.path.join(config['ICs']['quijote_path'], 'ICs')
    ic_paramfile = config['ICs']['ic_paramfile']

    # webonx executable should be in same directory as this script
    scriptdir = os.path.dirname(__file__)
    webonx = os.path.join(scriptdir, 'webonx')
    assert os.path.isfile(webonx), "webonx executable not found"

    os.makedirs(outdir, exist_ok=True) # make output directory in case it doesn't exist 
    os.system('cp %s %s' % (config_file, outdir)) # copy config file for clarity 

    # Set cosmological parameters for this run from Quijote IC  
    omega_m, omega_b, w0, n_s, wa, sigma8, hh = _read_cosmo_pars_from_config(ic_path, ic_paramfile)

    # Write input redshift snapshots file
    _write_z_input_file(zsnap, outdir)

    # Write cosmological parameters file 
    _write_cosmology_par_input_file(omega_m, omega_b, w0, n_s, wa, sigma8, hh, outdir)

    # write input parameter file 
    _write_input_par_file(ngrid, lbox, seed, 1, lambdath_tweb, lambdath_twebdelta, omega_m, zmin, zmax, outdir)
    
    if make_ics: 
        if not silent: print(f'Computing and writing out delta IC')
        # grid ICs and get delta
        delta = _make_ics_quijote(ic_path, float(lbox), ngrid)
        # write delta to outdir 
        delta.astype('float32').tofile(os.path.join(outdir, 'Quijote_ICs_delta_z127_n256_CIC.DAT'))

    os.chdir(outdir)

    # Compute displacement fields at different redshifts
    if not silent: print(f'Computing displacement fields at z=%s' % zsnap[0])
    sys.stdout.flush()
    if not silent: subprocess.run([webonx,])
    else: subprocess.run([webonx,], stdout=subprocess.DEVNULL)
    sys.stdout.flush()

    if not return_pos: 
        return None 
    else: 
        try: 
            suffix = 'OM'+(glob.glob(os.path.join(outdir, 'deltaBOXOM*'))[0].split('deltaBOXOM')[-1]).split('.gz')[0]
        except IndexError:
            # let webonx finish running 
            import time 
            print('sleeping to let webonx finish') 
            time.sleep(15)
            suffix = 'OM'+(glob.glob(os.path.join(outdir, 'deltaBOXOM*'))[0].split('deltaBOXOM')[-1]).split('.gz')[0]

        posx = np.fromfile(os.path.join(outdir, 'BOXposx%s' % suffix), dtype=np.float32)
        posy = np.fromfile(os.path.join(outdir, 'BOXposy%s' % suffix), dtype=np.float32)
        posz = np.fromfile(os.path.join(outdir, 'BOXposz%s' % suffix), dtype=np.float32)
        
        return np.vstack([posx, posy, posz]).T

        
def _write_input_par_file(ngrid, lbox, seed, sfmodel, lambdath_tweb, lambdath_twebdelta, omegam, zmin, zmax, outdir):
    ff = open(os.path.join(outdir, 'input.par'), 'w')

    omegal = 1. - omegam
    
    ff.write('#---------------------------------------------------------------------\n')
    ff.write('# BOX: parameter file\n')
    ff.write('#---------------------------------------------------------------------\n')
    ff.write('seed = %d\n' %seed)
    ff.write('#---------------------------------------------------------------------\n')
    ff.write('Nx = %d # Number of pixels x\n' %ngrid)
    ff.write('#---------------------------------------------------------------------\n')
    ff.write('Lx = %s # Box length in x-direction [Mpc/h]\n' %str(lbox))
    ff.write('#---------------------------------------------------------------------\n')
    ff.write('fnameIC = Quijote_ICs_delta_z127_n256_CIC.DAT# Attention no space after = unless you give a name\n')
    ff.write('fnameDM = deltaBOXOM%3.3fOL%3.3fG%dV%s.dat\n' %(omegam, omegal, ngrid, str(round(float(lbox),1))))
    ff.write('sfmodel = %s\n' %str(sfmodel))
    ff.write('filter = 1\n')
    ff.write('dgrowth_short = 5.\n')
    ff.write('rsml = 50.0\n')
    ff.write('dtol = 0.005\n')
    ff.write('curlfrac = 0\n')
    ff.write('write_box = true\n')
    ff.write('check_calibration = false\n')
    ff.write('#---------------------------------------------------------------------\n')
    ff.write('z = 0.\n')
    ff.write('zref = 127.\n')
    ff.write('slength = 5\n')
    ff.write('#---------------------------------------------------------------------\n')
    ff.write('mpro = 1\n')
    ff.write('nminperc = 20\n')
    ff.write('tcw = 4\n')
    ff.write('tcwD = 1234\n')
    ff.write('deltathL = -.5\n')
    ff.write('deltathH = -.5\n')
    ff.write('lth = %s # lambdath phi-web\n' %lambdath_tweb)
    ff.write('lthD = %s # lambdath delta-web\n' %lambdath_twebdelta)
    ff.write('fridge = 0.1\n')
    ff.write('boostk = 1.7\n')
    ff.write('boostf = 1.7\n')
    ff.write('boosts = 1.7\n')
    ff.write('alpha = 0.5\n')
    ff.write('#---------------------------------------------------------------------\n')
    ff.write('NsnapOMP = 1\n')
    ff.write('zsnapsmin = %s\n' % zmin)
    ff.write('zsnapsmax = %s\n' % zmax)
    ff.write('#---------------------------------------------------------------------\n')
    ff.write('xllc = %s\n' %str(-np.float32(lbox)/2))
    ff.write('yllc = %s\n' %str(-np.float32(lbox)/2))
    ff.write('zllc = %s\n' %str(-np.float32(lbox)/2))

    ff.close()
    return None 


def _write_z_input_file(zsnap, outdir):
    ''' write z input file used by webonx 
    '''
    ff = open(os.path.join(outdir, 'z_input.par'), 'w')

    for ii in range(len(zsnap)):
        ff.write(zsnap[ii] + '\n')
    ff.close()
    return None 


def _write_cosmology_par_input_file(omega_m, omega_b, w0, n_s, wa, sigma8, hpar, outdir):
    ''' write cosmology input file for webonx
    '''
    ff = open(os.path.join(outdir, 'cosmology.par'), 'w')

    ff.write('#---------------------------------------------------------------------\n')
    ff.write('# CosmicSignal: cosmology parameter file\n')
    ff.write('#---------------------------------------------------------------------\n')
    ff.write('omega_m = %s\n' %str(omega_m))
    ff.write('omega_b = %s\n' %str(omega_b))
    ff.write('wpar    = %s\n' %str(w0))
    ff.write('n_s     = %s\n' %str(n_s))
    ff.write('wprime  = %s\n' %str(wa))
    ff.write('sigma8  = %s\n' %str(sigma8))
    ff.write('rsmooth = 8.0\n')
    ff.write('hpar    = %s\n' %str(hpar))
    ff.write('betapar = 1.5\n')
    ff.write('#---------------------------------------------------------------------\n')
    ff.write('readPS = true\n')
    ff.write('#---------------------------------------------------------------------\n')
    ff.write('fnamePS = linear_power_z0.DAT\n')
    ff.write('#---------------------------------------------------------------------\n')
    ff.write('#---------------------------------------------------------------------\n')

    ff.close()
    return None 


def _read_cosmo_pars_from_config(ic_path, ic_paramfile):
    ''' read cosmological parameters from config file 
    '''
    fn = os.path.join(ic_path, ic_paramfile)

    raw = np.genfromtxt(fn, comments='%')

    omega_m = raw[7,1]
    omega_b = raw[9,1]
    hh = raw[11,1]
    #zz = raw[12,1]
    sigma8 = raw[13,1]
    n_s = raw[19,1]
    w0 = -1
    wa = 0.

    return omega_m, omega_b, w0, n_s, wa, sigma8, hh


def _make_ics_quijote(ic_path, lbox, ngrid):
    ''' grid quijote initial conditions 
    '''
    ics = Q.IC(ic_path.split('ICs')[0])
    posx = ics.pos[:,0] 
    posy = ics.pos[:,1] 
    posz = ics.pos[:,2] 
    
    weight = np.ones(len(posx))                                                                                                         

    delta = get_cic(posx, posy, posz, weight, float(lbox), ngrid)
    delta = delta.flatten()
    delta = delta/np.mean(delta)-1.

    return delta 


@njit(parallel=False, cache=True, fastmath=True)
def get_cic(posx, posy, posz, weight, lbox, ngrid):
    ''' cloud in cell gridding of x, y, z positions 
    '''
    lcell = lbox/ngrid

    delta = np.zeros((ngrid,ngrid,ngrid))

    for ii in prange(len(posx)):
        xx = posx[ii]
        yy = posy[ii]
        zz = posz[ii]

        if xx<0:
            xx += lbox
        if xx>=lbox:
            xx -= lbox

        if yy<0:
            yy += lbox
        if yy>=lbox:
            yy -= lbox

        if zz<0:
            zz += lbox
        if zz>=lbox:
            zz -= lbox


        indxc = int(xx/lcell)
        indyc = int(yy/lcell)
        indzc = int(zz/lcell)

        wxc = xx/lcell - indxc
        wyc = yy/lcell - indyc
        wzc = zz/lcell - indzc

        if wxc <=0.5:
            indxl = indxc - 1
            if indxl<0:
                indxl += ngrid
            wxc += 0.5
            wxl = 1 - wxc
        elif wxc >0.5:
            indxl = indxc + 1
            if indxl>=ngrid:
                indxl -= ngrid
            wxl = wxc - 0.5
            wxc = 1 - wxl

        if wyc <=0.5:
            indyl = indyc - 1
            if indyl<0:
                indyl += ngrid
            wyc += 0.5
            wyl = 1 - wyc
        elif wyc >0.5:
            indyl = indyc + 1
            if indyl>=ngrid:
                indyl -= ngrid
            wyl = wyc - 0.5
            wyc = 1 - wyl

        if wzc <=0.5:
            indzl = indzc - 1
            if indzl<0:
                indzl += ngrid
            wzc += 0.5
            wzl = 1 - wzc
        elif wzc >0.5:
            indzl = indzc + 1
            if indzl>=0:
                indzl -= ngrid
            wzl = wzc - 0.5
            wzc = 1 - wzl                                                                                                 

        ww = weight[ii]

        delta[indxc,indyc,indzc] += ww * wxc*wyc*wzc
        delta[indxl,indyc,indzc] += ww * wxl*wyc*wzc
        delta[indxc,indyl,indzc] += ww * wxc*wyl*wzc
        delta[indxc,indyc,indzl] += ww * wxc*wyc*wzl
        delta[indxl,indyl,indzc] += ww * wxl*wyl*wzc
        delta[indxc,indyl,indzl] += ww * wxc*wyl*wzl
        delta[indxl,indyc,indzl] += ww * wxl*wyc*wzl
        delta[indxl,indyl,indzl] += ww * wxl*wyl*wzl

    return delta
