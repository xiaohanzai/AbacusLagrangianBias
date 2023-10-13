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

    #Rfs = [2, 2.83, 4, 5.66, 8, 11.3]
    Rfs = [2, 2.83, 4]
    qs = ['dnG', 'dnG', 'dnG']#, 'dnG', 'dnG', 'dnG']
    # the first few PCs of grad f
    #vec_PCs = np.array([[ 8.95454546e-02, -6.66394959e-01, -4.87251723e-01,
    #          1.06250633e-01,  2.04382515e-01,  3.03278670e-01,
    #          8.24247116e-02, -4.01010502e-01,  4.02268938e-02,
    #          2.18092295e-02,  1.13407954e-01, -5.21411794e-02,
    #          7.73827767e-02, -9.69223376e-02, -7.94673766e-03,
    #          2.22830793e-02,  2.19769078e-02,  2.45066107e-02],
    #        [ 2.12948081e-01,  1.31374692e+00,  3.85733695e-01,
    #          7.60598861e-03, -1.21737537e+00, -1.98088559e+00,
    #          3.30545206e-02,  1.07579031e+00,  2.16587710e+00,
    #          2.12008104e-01, -6.83917574e-01, -1.75330954e+00,
    #         -2.79381108e-02,  6.33552015e-02,  8.52787794e-01,
    #          1.99802680e-01,  5.73530348e-02, -1.26144557e-01],
    #        [-1.01868275e-02, -3.30646080e-01,  3.99797763e+00,
    #          3.94168638e-02,  3.96519592e-01, -6.89866328e+00,
    #          1.57378379e-01,  9.45220000e-02,  6.67743459e+00,
    #          1.91367891e-01, -1.58166790e-01, -5.55504915e+00,
    #          1.23638913e-01, -2.94515976e-01,  2.54164367e+00,
    #          2.32677599e-01,  7.61297925e-01, -1.44141330e-01]])
    # vec_PCs = np.array([[-0.53164768,  0.56337627,  0.09695874],
    #         [-1.05876138, -1.09275604,  0.43997749]])
    vec_PCs = np.array([[ 2.03716382e-01, -5.16206211e-01, -5.43930512e-01,
              1.47488589e-01, -7.65245945e-02,  4.31894185e-01,
              1.80170069e-01, -2.44588406e-02, -8.07901814e-02],
            [ 2.81888189e-01,  1.05837854e+00,  3.40635792e-02,
              3.70584859e-01, -5.51052982e-01, -7.37042245e-01,
              4.10980038e-01,  6.38605334e-01,  2.92332227e-01]])

    # where the models are stored and where we should output
    folder = ''
    for i in range(len(Rfs)):
        folder += 'Rf%s%s_' % (Rfs[i], qs[i])
    folder = folder[:-1]
    path = _path+'/xwu/AbacusSummit/%s/NN/' % sim_sol + folder + '/haloN150_Nmesh100_subsample0.1/gradf_PCs/%dPCs/' % vec_PCs.shape[0]
    outpath = path + '%s' % sim_app
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # load particle features
    pos, features = load_particle_features(sim_app, z, 0, Rfs, qs)
    features = vec_PCs.dot(features).astype(np.float32)
    n_feature = vec_PCs.shape[0]
    fs_ave = 0.

    for i in range(5):
        fname = 'model_5x64_lr2e-03_gamma0.9_50-70epochs_0%d' % i
        # load model
        model = MyNetwork(n_feature, 64, 5, F.gelu)
        model.load_state_dict(torch.load(path+'/'+fname+'.pt'))

        # calculate mesh and save
        deltah_model, fs = calc_deltah_model_nbodykit_(model, pos, features, boxsize, Nmesh, interp_method, return_fs=True)
        fs_ave += fs
        FieldMesh(deltah_model).save(outpath+'/deltah_'+fname+'_%d_%s.bigfile' % (Nmesh, interp_method))
    # calculate mesh using averaged f
    deltah_model = ArrayCatalog({'Position': pos, 'Value': fs_ave/5}).to_mesh(Nmesh=Nmesh, resampler=interp_method, BoxSize=boxsize).compute()
    FieldMesh(deltah_model).save(outpath+'/deltah_'+fname[:-3]+'_ave_%d_%s.bigfile' % (Nmesh, interp_method))

if __name__ == '__main__':
    main()

