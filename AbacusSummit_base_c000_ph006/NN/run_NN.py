import numpy as np
import pyfftw
from nbodykit.lab import *
from nbodykit import setup_logging, style
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import warnings
warnings.filterwarnings("ignore")
import time
import os
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../matrixA/')
from cosmology_functions import *
from particle_functions import *
from NNB_functions import *
from NNB_NN import *

_path = '/mnt/store2/'
if not os.path.exists(_path):
    _path = '/mnt/marvin2/'

def main():
    sim = 'small/AbacusSummit_small_c000_ph3100'
    boxsize = 500.
    islab = 0
    z = 0.5
    N = 576
    N0 = 1728
    ic_path = _path+'/xwu/AbacusSummit/%s/ic_%d/' % (sim, N)

    Rfs = [3, 4]
    qs = ['dnG', 'dnG']

    # load in particle position and features
    pos, features = load_particle_features(sim, z, islab, Rfs, qs)

    # load catalog
    cat = CompaSOHaloCatalog(_path+'/bigsims/AbacusSummit/%s/halos/z%.3f/' % (sim, z),
                             fields=['N', 'x_com', 'r100_com'])
    Nthres = 150
    ii_h = (cat.halos['N'] > Nthres)

    # prepare for the integral constraint
    ind = np.random.choice(len(pos), int(1e5))
    inputs_integral = torch.tensor(features[:,ind].T.reshape(1,len(ind),-1))

    Nmesh = 100
    # distribute particles
    part_list_in_cell = gen_part_list_in_cell(pos, boxsize, Nmesh)

    # halo field
    deltah_true = ArrayCatalog({'Position': cat.halos[ii_h]['x_com'], 'Value': cat.halos[ii_h]['N']}).to_mesh(
        Nmesh=Nmesh, BoxSize=boxsize, resampler='nnb').compute()
    deltah_true = np.asarray(deltah_true/np.mean(deltah_true)-1.)

    # start training
    Npart_per_cell = 20
    N_realizations = 5
    data = MyDataset(part_list_in_cell, boxsize, Nmesh, deltah_true, Npart_per_cell, N_realizations, features)
    data_train = DataLoader(dataset=data, batch_size=32, shuffle=False)
    del features

    logf = True
    nonnegative_contraint = True
    integral_constraint = True
    f_tol = lambda epoch: 1e-4#max(10**(-epoch-2), 1e-4)

    # instantiate the model
    model = MyNetwork(data.inputs.shape[-1], 50, 3, F.gelu)
    # model.to(device)

    # create a stochastic gradient descent optimizer
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # start training
    N_epoch = 5
    for epoch in range(N_epoch):
        if integral_constraint and epoch == 1:
            data_train = DataLoader(dataset=data, batch_size=128, shuffle=False)
        t0 = time.time()
        epoch_loss = 0
        for b_idx, batch in enumerate(data_train):
            b_inputs, b_facs, b_deltah_true = batch
            #b_inputs.to(device)
            #b_facs.to(device)
            #b_deltah_true.to(device)
            loss = train(model, b_inputs, b_facs, b_deltah_true, optimizer, logf=logf,
                nonnegative_contraint=nonnegative_contraint,
                integral_constraint=integral_constraint*epoch,
                fac=1., tol=f_tol(epoch), inputs_integral=inputs_integral)
            epoch_loss += loss
        print('Epoch {} Loss : {}'.format((epoch+1),epoch_loss), 'time=%.2f' % (time.time()-t0))

    folder = ''
    for i in range(len(Rfs)):
        folder += 'Rf%s%s_' % (Rfs[i], qs[i])
    folder = folder[:-1]
    path = _path+'/xwu/AbacusSummit/small/AbacusSummit_small_c000_ph3100/NN/'+folder
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), path+'/model_N%d_Nmesh%d_%dpart_%dreal_%depochs.pt' % (Nthres, Nmesh, Npart_per_cell, N_realizations, N_epoch))

if __name__ == '__main__':
    main()

