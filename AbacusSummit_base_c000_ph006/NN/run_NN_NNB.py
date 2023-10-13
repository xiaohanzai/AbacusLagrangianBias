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
from particle_functions import *
from NNB_NN import *

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'

nc = ''
no_decorrelation = False

# play with a small box sim first
sim = 'small/AbacusSummit_small_c000_ph3100'
boxsize = 500.
islab = 0
z = 0.5
N = 576
N0 = 1728
ic_path = '/mnt/marvin2/xwu/AbacusSummit/%s/ic_%d/' % (sim, N)

Rfs = [2, 2.83, 4]#, 5.66, 8, 11.3]
qs = ['dnG', 'dnG', 'dnG']#, 'dnG', 'dnG', 'dnG']

Nmesh = 100

Nthres = 150

# training set
pos, features = load_particle_features(sim, z, islab, Rfs, qs)
eigvals = None
eigvecs = None
if not no_decorrelation:
    eigvals, eigvecs, features = decorrelate_features(features, Rfs, qs)
    print(eigvals.dtype, eigvecs.dtype, features.dtype)

cat = CompaSOHaloCatalog('/mnt/marvin2/bigsims/AbacusSummit/%s/halos/z%.3f/' % (sim, z),
                         fields=['N', 'x_com', 'r100_com'])
ii_h = (cat.halos['N'] > Nthres)
deltah_true = ArrayCatalog({'Position': cat.halos[ii_h]['x_com'], 'Value': cat.halos[ii_h]['N']}).to_mesh(
    Nmesh=Nmesh, BoxSize=boxsize, resampler='nnb').compute()
deltah_true = np.asarray(deltah_true/np.mean(deltah_true)-1.)

# validation set
sim_val = 'small/AbacusSummit_small_c000_ph3102'
pos_val, features_val = load_particle_features(sim_val, z, islab, Rfs, qs)
pos_val = pos_val[::10]
features_val = features_val[:,::10]
if not no_decorrelation:
    features_val = decorrelate_features(features_val, Rfs, qs, eigvecs=eigvecs, eigvals=eigvals)[-1]
    print(features_val.dtype)

# choose a subset of features
# keep all G2's, but only keep the first n+1 delta and nabla, where
#   n is the number of smoothing scales
if not no_decorrelation:
    ii = choose_decorrelated_features(eigvals, Rfs, qs)
    features = features[ii]
    features_val = features_val[ii]
    print('%d features left' % features.shape[0])

cat = CompaSOHaloCatalog('/mnt/marvin2/bigsims/AbacusSummit/%s/halos/z%.3f/' % (sim_val, z),
                         fields=['N', 'x_com', 'r100_com'])
ii_h = (cat.halos['N'] > Nthres)
deltah_val = ArrayCatalog({'Position': cat.halos[ii_h]['x_com'], 'Value': cat.halos[ii_h]['N']}).to_mesh(
    Nmesh=Nmesh, BoxSize=boxsize, resampler='nnb').compute()
deltah_val = np.asarray(deltah_val/np.mean(deltah_val)-1.)

# build dataset and dataloader
part_list_in_cell = gen_part_list_in_cell(pos, boxsize, Nmesh)
#data = MyDataset(part_list_in_cell, deltah_true, 0.1, features)
#bs = Nmesh**2
#data_train = DataLoader(dataset=data, batch_size=bs, shuffle=True, collate_fn=collate_fn)

# randomly sample a similar number of particles for the integral constraint
inds = np.random.choice(len(pos), int(1e5))
inputs_integral = torch.tensor(features[:,inds].T.reshape(1,len(inds),-1)).to(device)

n_epoch = 70
f_tol = lambda epoch: max(10**(-1-3*np.log10(epoch)), 1e-4)
n_neuron = 64
n_layer = 5
gamma = 0.9
lr = 2e-3
bs = 10000
subsample_fac = 0.1

folder = ''
for i in range(len(Rfs)):
    folder += 'Rf%s%s_' % (Rfs[i], qs[i])
folder = folder[:-1]
folder += '/haloN%d_Nmesh%d_subsample%.2g/' % (Nthres, Nmesh, subsample_fac)
if no_decorrelation:
    folder += 'no_decorrelation/'
path = '/mnt/marvin2/xwu/AbacusSummit/small/AbacusSummit_small_c000_ph3100/NN/'+folder
if not os.path.exists(path):
    os.makedirs(path)

# for model_ind in range(5):
model_ind = int(sys.argv[1])

# build dataset
data = MyDataset(part_list_in_cell, deltah_true, subsample_fac, features)
data_train = DataLoader(dataset=data, batch_size=bs, shuffle=True, collate_fn=collate_fn)

# instantiate the model
model = MyNetwork(data.inputs.shape[-1], n_neuron, n_layer, F.gelu)
model = model.to(device)

# create a stochastic gradient descent optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = None
if gamma > 1e-6:
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

train_losses = np.zeros(n_epoch)
val_losses = np.zeros(n_epoch)
for epoch in range(n_epoch):
    epoch_loss = 0
    t0 = time.time()
    for b_idx, batch in enumerate(data_train):
        b_inputs, b_facs, b_deltahs, locs = batch
        #if torch.cuda.is_available():
        b_inputs = b_inputs.to(device)
        b_facs = b_facs.to(device)
        b_deltahs = b_deltahs.to(device)
        locs = torch.IntTensor(locs).to(device)
        loss = train(model, b_inputs, b_facs, b_deltahs, locs, optimizer,
            integral_constraint=bool(max(0,epoch-1)),
            fac=1., tol=f_tol(epoch), inputs_integral=inputs_integral)
        epoch_loss += loss

    deltah_model = calc_deltah_model_nbodykit_(model.cpu(), pos_val, features_val, boxsize, Nmesh, 'nnb', minus1=True)
    loss = np.sum( (deltah_model - deltah_val)**2 )
    model = model.to(device)
    print('Epoch {} TrainLoss {} ValLoss {}'.format((epoch+1),epoch_loss,loss), 'time=%.3g' % (time.time()-t0))

    if gamma > 1e-6:# and lr*gamma**epoch > 5e-5:
        scheduler.step()

    train_losses[epoch] = epoch_loss
    val_losses[epoch] = loss

    if epoch in [19,29,39,49,n_epoch-1]:
        fname = 'model%s_%dx%d_lr%.0e_gamma%.2g_%d-%depochs_0%d' % (nc, n_layer, n_neuron, lr, gamma, (epoch+1), n_epoch, model_ind)
        deltah_model = calc_deltah_model_nbodykit_(model.cpu(), pos, features, boxsize, Nmesh, 'nnb')
        FieldMesh(deltah_model).save(path+'/deltah_'+fname+'.bigfile')
        torch.save(model.state_dict(), path+'/'+fname+'.pt')
        model = model.to(device)
        np.save(path+'/losses_'+fname, [train_losses[:epoch+1], val_losses[:epoch+1]])

