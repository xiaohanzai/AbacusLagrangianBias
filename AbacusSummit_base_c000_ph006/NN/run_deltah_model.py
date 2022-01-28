import numpy as np
#from mpi4py import MPI
from nbodykit.lab import *
from nbodykit import setup_logging
#from nbodykit import CurrentMPIComm
import matplotlib.pyplot as plt
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import sys
sys.path.append('../matrixA')
from run_matrixA import load_and_process_particles
import os
from NNB_NN import *
from particle_functions import *

_path = '/mnt/marvin2/'
if not os.path.exists(_path):
    _path = '/mnt/store2/'

def main():
    sim, z, Nmesh = sys.argv[1:]
    # redshift, smoothing scale, mesh size
    z = float(z)
    Nmesh = int(Nmesh)

    interp_method = 'nnb'

    Rfs = [3, 4]
    qs = ['dnG', 'dnG']

    # where the model is stored and where we should output
    folder = ''
    for i in range(len(Rfs)):
        folder += 'Rf%s%s_' % (Rfs[i], qs[i])
    folder = folder[:-1]
    outpath = _path+'/xwu/AbacusSummit/small/AbacusSummit_small_c000_ph3100/NN/'+folder

    # load model
    n_inputs = 0
    for q in qs:
        n_inputs += len(q)
    model = MyNetwork(n_inputs, 50, 3, F.gelu)
    model.load_state_dict(torch.load(outpath+'/model_N150_Nmesh100_20part_5real_5epochs.pt'))

    # calculate mesh and save
    mesh = calc_deltah_model_nbodykit(model, sim, z, Rfs, qs, Nmesh, interp_method)
    fname = '/deltah_model_N150_%d_%s_%s' % (Nmesh, interp_method, sim)
    FieldMesh(mesh).save(outpath+fname+'.bigfile')

if __name__ == '__main__':
    main()

