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
    sim_sol, sim_app, z, Nmesh = sys.argv[1:]
    # redshift, smoothing scale, mesh size
    z = float(z)
    Nmesh = int(Nmesh)

    boxsize = 500.
    interp_method = 'cic'

    Rfs = [2, 2.83, 4]
    qs = ['dnG', 'dnG', 'dnG']

    # where the models are stored and where we should output
    folder = ''
    for i in range(len(Rfs)):
        folder += 'Rf%s%s_' % (Rfs[i], qs[i])
    folder = folder[:-1]
    path = _path+'/xwu/AbacusSummit/%s/NN/' % sim_sol + folder + '/haloN150_Nmesh100_subsample0.1/'
    outpath = path + '%s' % sim_app
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # load particle features
    # training set
    pos, features = load_particle_features(sim_sol, z, 0, Rfs, qs)
    eigvals, eigvecs, features = decorrelate_features(features, Rfs, qs)
    ii = choose_decorrelated_features(eigvals, Rfs, qs)
    # the other sim
    if sim_sol != sim_app:
        pos, features = load_particle_features(sim_app, z, 0, Rfs, qs)
        features = decorrelate_features(features, Rfs, qs, eigvecs=eigvecs, eigvals=eigvals)[-1]
    features = features[ii]
    n_feature = features.shape[0]
    fs_ave = 0.

    for i in range(5):
        fname = 'model_5x64_lr2e-03_gamma0.9_50-70epochs_0%d' % i
        try:
            # load model
            model = MyNetwork(n_feature, 64, 5, F.gelu)
            model.load_state_dict(torch.load(path+'/'+fname+'.pt'))
        except:
            continue

        # calculate mesh and save
        deltah_model, fs = calc_deltah_model_nbodykit_(model, pos, features, boxsize, Nmesh, interp_method, return_fs=True)
        fs_ave += fs
        FieldMesh(deltah_model).save(outpath+'/deltah_'+fname+'_%d_%s.bigfile' % (Nmesh, interp_method))
    # calculate mesh using averaged f
    deltah_model = ArrayCatalog({'Position': pos, 'Value': fs_ave/5}).to_mesh(Nmesh=Nmesh, resampler=interp_method, BoxSize=boxsize).compute()
    FieldMesh(deltah_model).save(outpath+'/deltah_'+fname[:-3]+'_ave_%d_%s.bigfile' % (Nmesh, interp_method))

if __name__ == '__main__':
    main()

