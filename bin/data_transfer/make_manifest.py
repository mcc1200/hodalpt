#!/bin/python
import os 
import numpy as np 

ff = open('transfer_manifest.txt', 'w')
for i in range(400, 500):
    ff.write('Snapshots/latin_hypercube_HR/%i/CAMB.params %i/CAMB.params\n' % (i, i))
    ff.write('Snapshots/latin_hypercube_HR/%i/Cosmo_params.dat %i/Cosmo_params.dat\n' % (i, i))

    # initial conditions 
    ff.write('Snapshots/latin_hypercube_HR/%i/ICs/2LPT.param %i/ICs/2LPT.param\n' % (i,i))
    ff.write('Snapshots/latin_hypercube_HR/%i/ICs/ics.0.hdf5 %i/ICs/ics.0.hdf5\n' % (i,i))
    ff.write('Snapshots/latin_hypercube_HR/%i/ICs/ics.1.hdf5 %i/ICs/ics.1.hdf5\n' % (i,i))
    ff.write('Snapshots/latin_hypercube_HR/%i/ICs/ics.2.hdf5 %i/ICs/ics.2.hdf5\n' % (i,i))
    ff.write('Snapshots/latin_hypercube_HR/%i/ICs/ics.3.hdf5 %i/ICs/ics.3.hdf5\n' % (i,i))
    ff.write('Snapshots/latin_hypercube_HR/%i/ICs/ics.4.hdf5 %i/ICs/ics.4.hdf5\n' % (i,i))
    ff.write('Snapshots/latin_hypercube_HR/%i/ICs/ics.5.hdf5 %i/ICs/ics.5.hdf5\n' % (i,i))
    ff.write('Snapshots/latin_hypercube_HR/%i/ICs/ics.6.hdf5 %i/ICs/ics.6.hdf5\n' % (i,i))
    ff.write('Snapshots/latin_hypercube_HR/%i/ICs/ics.7.hdf5 %i/ICs/ics.7.hdf5\n' % (i,i))
    ff.write('Snapshots/latin_hypercube_HR/%i/ICs/inputspec_ics.txt %i/ICs/inputspec_ics.txt\n' % (i,i))
    ff.write('Snapshots/latin_hypercube_HR/%i/ICs/Pk_mm_z=0.000.txt %i/ICs/Pk_mm_z=0.000.txt\n' % (i,i))
    
    # rockstar halo catalogs
    #ff.write('Halos/Rockstar/latin_hypercube_HR//%i/out_4_pid.list %i/Rockstar/out_4_pid.list\n' % (i, i))
    ff.write('Halos/Rockstar/latin_hypercube_HR/%i/out_3_pid.list %i/Rockstar/out_3_pid.list\n' % (i, i))
    #ff.write('Halos/Rockstar/latin_hypercube_HR//%i/out_2_pid.list %i/Rockstar/out_2_pid.list\n' % (i, i))
    #ff.write('Halos/Rockstar/latin_hypercube_HR//%i/out_1_pid.list %i/Rockstar/out_1_pid.list\n' % (i, i))
    #ff.write('Halos/Rockstar/latin_hypercube_HR//%i/out_0_pid.list %i/Rockstar/out_0_pid.list\n' % (i, i))

ff.close() 

