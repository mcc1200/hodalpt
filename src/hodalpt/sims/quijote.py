'''


module for reading in Quijote data. This module assumes that you've downloaded
the Quijote data in the following way

QUIJOTE/
    fiducial/ 
        0/
            ICS/ 
            Halos/
        1/
        2/
        ...
    latin_hypercube_HR/
        0/
        1/
        ...

'''
import os
import h5py 
import hdf5plugin
import numpy as np 

import nbodykit.lab as NBlab
from simbig import galaxies as G 


quijote_zsnap_dict = {0.: 4, 0.5: 3, 1.:2, 2.: 1, 3.: 0}


def HODgalaxies(theta_hod, _dir, z=0.5): 
    ''' Galaxies populated using HOD on Quijote Rockstar halos. By default, it
    uses the `simbig` training data HOD, which requires 

    .. code-block:: python
        theta_hod = {
            'logMmin': ,
            'sigma_logM': ,
            'logM0': ,
            'logM1': ,
            'alpha': ,
            'Abias':, 
            'eta_conc': , 
            'eta_cen': , 
            'eta_sat': }

    '''
    # read halo catalog 
    halos = Halos(_dir, z=z) 
    
    # parse HOD parameters
    _theta = {}
    _theta['logMmin']       = theta_hod['logMmin']
    _theta['sigma_logM']    = theta_hod['sigma_logM']
    _theta['logM0']         = theta_hod['logM0']
    _theta['logM1']         = theta_hod['logM1']
    _theta['alpha']         = theta_hod['alpha']
    _theta['mean_occupation_centrals_assembias_param1'] = theta_hod['Abias']
    _theta['mean_occupation_satellites_assembias_param1'] = theta_hod['Abias']
    _theta['conc_gal_bias.satellites']  = theta_hod['eta_conc']
    _theta['eta_vb.centrals']           = theta_hod['eta_cen']
    _theta['eta_vb.satellites']         = theta_hod['eta_sat']

    # populate with HOD
    _Z07AB = G.VelAssembiasZheng07Model() # default simbig HOD model 

    Z07AB = _Z07AB.to_halotools(
            halos.cosmo, 
            halos.attrs['redshift'],
            halos.attrs['mdef'],
            sec_haloprop_key='halo_nfw_conc')
    hod = halos.populate(Z07AB, **_theta)
    return hod 


def Halos(_dir, z=0.5, silent=True): 
    ''' read in Quijote Rockstar halos given the folder and store it as
    a nbodykit HaloCatalog object. The HaloCatalog object is convenient for
    populating with galaxies and etc.


    Parameters
    ----------
    _dir : string
        directory that contains the snapshots, ICs, and halos e.g. 
        /$QUIJOTE/latin_hypercube_HR/0/

    Return
    ------
    cat : nbodykit.lab.HaloCatalog
        Quijote halo catalog
    '''
    # redshift snapshot
    assert z in quijote_zsnap_dict.keys(), 'snapshots are available at z=0, 0.5, 1, 2, 3'
    snapnum = quijote_zsnap_dict[z]

    # look up cosmology 
    setup = _dir.split('/')[-2]
    ireal = int(_dir.split('/')[-1])

    Om, Ob, h, ns, s8 = _cosmo_lookup(setup, ireal)

    # define cosmology; caution: we don't match sigma8 here
    cosmo = NBlab.cosmology.Planck15.clone(
            h=h,
            Omega0_b=Ob,
            Omega0_cdm=Om - Ob,
            m_ncdm=None,
            n_s=ns)
    Ol = 1.  - Om
    Hz = 100.0 * np.sqrt(Om * (1. + z)**3 + Ol) # km/s/(Mpc/h)

    rsd_factor = (1. + z) / Hz

    # rockstar file columns: ID DescID Mvir Vmax Vrms Rvir
    # Rs Np X Y Z VX VY VZ JX JY JZ Spin rs_klypin Mvir_all
    # M200b M200c M500c M2500c Xoff Voff spin_bullock
    # b_to_a c_to_a A[x] A[y] A[z] b_to_a(500c)
    # c_to_a(500c) A[x](500c) A[y](500c) A[z](500c) T/|U|
    # M_pe_Behroozi M_pe_Diemer Halfmass_Radius
    # read in columns: Mvir, Vmax, Vrms, Rvir, Rs, Np,
    # X, Y, Z, VX, VY, VZ, parent id
    _rstar = np.loadtxt(os.path.join(_dir, 'Rockstar', 'out_%i_pid.list' % snapnum), usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1])

    # select only halos
    is_halo = (_rstar[:,-1] == -1)
    rstar = _rstar[is_halo]

    # calculate concentration Rvir/Rs
    conc = rstar[:,3] / rstar[:,4]

    group_data = {}
    group_data['Length']    = rstar[:,5].astype(int)
    group_data['Position']  = rstar[:,6:9]
    group_data['Velocity']  = rstar[:,9:12]  # km/s * (1 + z)
    group_data['Mass']      = rstar[:,0]

    # calculate velocity offset
    group_data['VelocityOffset'] = group_data['Velocity'] * rsd_factor

    # save to ArryCatalog for consistency
    cat = NBlab.ArrayCatalog(group_data, BoxSize=np.array([1000., 1000., 1000.]))
    cat = NBlab.HaloCatalog(cat, cosmo=cosmo, redshift=z, mdef='vir')
    cat['Length'] = group_data['Length']
    cat['Concentration'] = conc # default concentration is using Dutton & Maccio (2014), which is based only on halo mass.

    cat.attrs['Om'] = Om
    cat.attrs['Ob'] = Ob
    cat.attrs['Ol'] = Ol
    cat.attrs['h'] = h
    cat.attrs['ns'] = ns
    cat.attrs['s8'] = s8
    cat.attrs['Hz'] = Hz # km/s/(Mpc/h)A
    cat.attrs['rsd_factor'] = rsd_factor
    return cat


def Nbody(_dir, z=0.5): 
    ''' read CDM particles from Quijote n-body simulation output snapshot 

    Parameters
    ----------
    _dir : string
        directory that contains all the snapshots e.g. 
        /$QUIJOTE/latin_hypercube_HR/0/

    z : float
        redshift of the snapshot you want read 
    '''

    # redshift snapshot 
    assert z in quijote_zsnap_dict.keys(), 'snapshots are available at z=0, 0.5, 1, 2, 3'
    snapnum = quijote_zsnap_dict[z]

    snapshot = os.path.join(_dir, 'snapdir_%s' % str(snapnum).zfill(3), 'snap_%s' % str(snapnum).zfill(3))
    return _read_snap(snapshot)


def IC(_dir): 
    ''' read initial conditions of CDM particles from Quijote. The IC is
    generated at z=127. 

    Parameters
    ----------
    _dir : string
        directory that contains the snapshots and IC files e.g.
        /$QUIJOTE/latin_hypercube_HR/0/
    '''
    return _read_snap(os.path.join(_dir, 'ICs', 'ICs'))


class Snap(object): 
    ''' class object for particle snapshots 
    '''
    def __init__(self): 
        self.BoxSize  = None #Mpc/h
        self.Nall     = None #Total number of particles
        self.Masses   = None #Masses of the particles in Msun/h
        self.Omega_m  = None #value of Omega_m
        self.Omega_l  = None #value of Omega_l
        self.h        = None #value of h
        self.redshift = None #redshift of the snapshot
        self.Hubble   = None #Value of H(z) in km/s/(Mpc/h)

        self.pos      = None #positions in Mpc/h
        self.vel      = None #peculiar velocities in km/s

    def _read_quijote_header(self, header): 
        ''' given Quijote simulation header, 
        '''
        self.BoxSize  = header['boxsize']/1e3  #Mpc/h
        self.Nall     = header['nall']         #Total number of particles
        self.Masses   = header['massarr']*1e10 #Masses of the particles in Msun/h
        self.Omega_m  = header['omega_m']      #value of Omega_m
        self.Omega_l  = header['omega_l']      #value of Omega_l
        self.h        = header['hubble']       #value of h
        self.redshift = header['redshift']     #redshift of the snapshot
        self.Hubble   = 100.0*np.sqrt(self.Omega_m*(1.0+self.redshift)**3+self.Omega_l)#Value of H(z) in km/s/(Mpc/h)
        return None 

    def save_cs(self):
        ''' save snapshot to binary format that can be ready by CosmicSignals 
        '''
        return None 


def _cosmo_lookup(setup, ireal): 
    ''' look up cosmology for quijote realization 
    '''
    if setup not in ['fiducial', 'latin_hypercube_HR']: 
        raise NotImplementedError("non-fiducial cosmologies not yet implemented")

    if 'fiducial' in setup: 
        Om = 0.3175
        Ob = 0.049
        h  = 0.6711
        ns = 0.9624
        s8 = 0.834
    elif 'latin_hypercube' in setup:  
        fcosmo = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat',
                'quijote_lhc_cosmo.txt')

        # Omega_m, Omega_l, h, ns, s8
        cosmo = np.loadtxt(fcosmo, unpack=True, usecols=range(5)) 

        Om = cosmo[0][ireal]
        Ob = cosmo[1][ireal]
        h  = cosmo[2][ireal]
        ns = cosmo[3][ireal]
        s8 = cosmo[4][ireal]

    return Om, Ob, h, ns, s8


def _read_snap(snapshot):
    ''' read snapshot in hdf5 format. This is streamlined and modified version
    of readgadget from pylians. 
    '''
    ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)

    # read header
    header = _read_header(snapshot)
    snap = Snap()
    snap._read_quijote_header(header)

    # read positions, velocities and IDs of the particles
    pos = _read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
    vel = _read_block(snapshot, "VEL ", ptype)     #peculiar velocities in km/s
    #ids = readgadget.read_block(snapshot, "ID  ", ptype)-1   #IDs starting from 0

    snap.pos = pos 
    snap.vel = vel
    return snap 


def _read_header(snapshot): 
    ''' read header of snapshot file 
    '''
    filename, fformat = _fname_format(snapshot)

    hdr = {} 

    f = h5py.File(filename, 'r')
    hdr['time']     = f['Header'].attrs[u'Time']
    hdr['redshift'] = f['Header'].attrs[u'Redshift']
    hdr['boxsize']  = f['Header'].attrs[u'BoxSize']
    hdr['filenum']  = f['Header'].attrs[u'NumFilesPerSnapshot']
    hdr['omega_m']  = f['Header'].attrs[u'Omega0']
    hdr['omega_l']  = f['Header'].attrs[u'OmegaLambda']
    hdr['hubble']   = f['Header'].attrs[u'HubbleParam']
    hdr['massarr']  = f['Header'].attrs[u'MassTable']
    hdr['npart']    = f['Header'].attrs[u'NumPart_ThisFile']
    hdr['nall']     = f['Header'].attrs[u'NumPart_Total']
    hdr['cooling']  = f['Header'].attrs[u'Flag_Cooling']
    hdr['format']   = 'hdf5'
    f.close()

    return hdr 


# This function reads a block from an entire gadget snapshot (all files)
# it can read several particle types at the same time. 
# ptype has to be a list. E.g. ptype=[1], ptype=[1,2], ptype=[0,1,2,3,4,5]
def _read_block(snapshot, block, ptype, verbose=False):

    # find the format of the file and read header
    filename, fformat = _fname_format(snapshot)
    head    = _read_header(filename)    
    Nall    = head['nall']
    filenum = head['filenum']

    # find the total number of particles to read
    Ntotal = 0
    for i in ptype:
        Ntotal += Nall[i]

    # find the dtype of the block
    if   block=="POS ":  dtype=np.dtype((np.float32,3))
    elif block=="VEL ":  dtype=np.dtype((np.float32,3))
    elif block=="MASS":  dtype=np.float32
    elif block=="ID  ":  dtype=_read_field(filename, block, ptype[0]).dtype
    else: raise Exception('block not implemented in readgadget!')

    # define the array containing the data
    array = np.zeros(Ntotal, dtype=dtype)


    # do a loop over the different particle types
    offset = 0
    for pt in ptype:

        if filenum==1:
            array[offset:offset+Nall[pt]] = _read_field(snapshot, block, pt)
            offset += Nall[pt]

        # multi-file hdf5 snapshot
        else:

            # do a loop over the different files
            for i in range(filenum):
                
                # find the name of the file to read
                filename = '%s.%d.hdf5'%(snapshot,i)

                # read number of particles in the file and read the data
                npart = _read_header(filename)['npart'][pt]
                array[offset:offset+npart] = _read_field(filename, block, pt)
                offset += npart   

    if offset!=Ntotal:  raise Exception('not all particles read!!!!')
            
    return array


# This function reads a block of an individual file of a gadget snapshot
def _read_field(snapshot, block, ptype):

    filename, fformat = _fname_format(snapshot)
    head              = _read_header(filename)

    prefix = 'PartType%d/'%ptype
    f = h5py.File(filename, 'r')
    if   block=="POS ":  suffix = "Coordinates"
    elif block=="MASS":  suffix = "Masses"
    elif block=="ID  ":  suffix = "ParticleIDs"
    elif block=="VEL ":  suffix = "Velocities"
    else: raise Exception('block not implemented in readgadget!')
    array = f[prefix+suffix][:]
    f.close()

    if block=="VEL ":  array *= np.sqrt(head['time'])
    if block=="POS " and array.dtype==np.float64:
        array = array.astype(np.float32)
    return array


def _fname_format(snapshot):
    # find snapshot name and format
    if os.path.exists(snapshot):
        if snapshot[-4:]=='hdf5':  filename, fformat = snapshot, 'hdf5'
        else:                      filename, fformat = snapshot, 'binary'
    elif os.path.exists(snapshot+'.0'):
        filename, fformat = snapshot+'.0', 'binary'
    elif os.path.exists(snapshot+'.hdf5'):
        filename, fformat = snapshot+'.hdf5', 'hdf5'
    elif os.path.exists(snapshot+'.0.hdf5'):
        filename, fformat = snapshot+'.0.hdf5', 'hdf5'
    else:  
        raise Exception('File (%s) not found!' % snapshot)
    return filename,fformat

def Box_RSD(cat, LOS=[0,0,1], Lbox=1000.):
    ''' Given a halo/galaxy catalog in a periodic box, apply redshift space
    distortion specified LOS along LOS

    Parameters
    ----------
    cat : CatalogBase
        nbodykit.Catalog object
    LOS : array_like
        3 element list specifying the direction of the line-of-sight
    Lbox : float
        box size in Mpc/h
    '''
    pos = np.array(cat['Position']) + np.array(cat['VelocityOffset']) * LOS

    # impose periodic boundary conditions for particles outside the box
    i_rsd = np.arange(3)[np.array(LOS).astype(bool)][0]
    rsd_pos = pos[:,i_rsd] % Lbox
    pos[:,i_rsd] = np.array(rsd_pos)
    return pos