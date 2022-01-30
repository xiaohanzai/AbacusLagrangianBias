import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from tqdm import tqdm
from nbodykit.lab import *
from NNB_functions import *
from particle_functions import *


class MyDataset(Dataset):
    def __init__(self, arg1, boxsize, Nmesh, deltah_true, Npart_per_cell, N_realizations, features, subsample=False):
        '''
        arg1 is either the particle positions or the particle list in cells.
        All the particle features should be input at the end, where features is a list or np array, where if the latter, the number of rows = number of features.
        '''
        if type(arg1) == dict:
            part_list_in_cell = arg1
        else:
            pos = arg1
            part_list_in_cell = gen_part_list_in_cell(pos, boxsize, Nmesh)
        # randomly choose an equal number of particles for each cell
        self.inputs = np.zeros((Npart_per_cell*Nmesh**3, N_realizations, len(features)), dtype=np.float32)
        self.part_list_in_cell = {}
        fac = float(Nmesh**3)/len(features[0])
        self.facs = {}
        for i in range(Nmesh**3):
            self.facs[i] = len(part_list_in_cell[i])/float(Npart_per_cell)*fac
            ind_ = np.random.choice(part_list_in_cell[i], Npart_per_cell*N_realizations, replace=True)
            self.part_list_in_cell[i] = np.arange(i*Npart_per_cell, (i+1)*Npart_per_cell, dtype=int).tolist()
            for n in range(N_realizations):
                for j, q in enumerate(features):
                    self.inputs[i*Npart_per_cell:(i+1)*Npart_per_cell,n,j] = q[
                        ind_[n*Npart_per_cell:(n+1)*Npart_per_cell]]

        self.deltah_true = deltah_true.reshape(-1)
        self.cell_inds = np.arange(0, Nmesh**3)
        if subsample:
            self.cell_inds = self.cell_inds[::8]
        self.cell_inds = self.cell_inds.tolist()

    def __len__(self):
        return len(self.cell_inds) # the number of cells

    def __getitem__(self, i):
        '''
        Get the particle data within the cell specified by inx, and the corresponding true deltah of that cell.
        '''
        j = self.cell_inds[i]
        inds = self.part_list_in_cell[j]
        return torch.Tensor(self.inputs[inds]), torch.Tensor([self.facs[j]]), torch.Tensor([self.deltah_true[j]])

# class MyDataset(Dataset):
#     def __init__(self, part_list_in_cell, boxsize, Nmesh, deltah_true, subsample_fac, pos, features):
#         '''
#         All the particle features should be input at the end.
#         '''
#         N_particles = features.shape[1]
#
#         ## randomly select a subsample of particles
#         #inds = np.random.choice(N_particles, int(N_particles*subsample_fac), replace=False)
#         ## generate new list
#         #self.part_list_in_cell = gen_part_list_in_cell(pos[inds], boxsize, Nmesh)
#         ## inputs array
#         #self.inputs = features.T[inds,:]
#
#         self.part_list_in_cell = {}
#         self.inputs = features.T*0.
#         n_particles = 0
#         for i in range(Nmesh**3):
#             n = len(part_list_in_cell[i])
#             n_new = max(int(n*subsample_fac),1)
#             self.part_list_in_cell[i] = np.arange(n_particles, n_particles+n_new, dtype=int).tolist()
#             inds = np.array(part_list_in_cell[i])[np.random.choice(n, n_new)]
#             self.inputs[n_particles:n_particles+n_new] = features.T[inds,:]
#             n_particles += n_new
#         self.inputs = self.inputs[:n_particles]
#
#         # correction factors
#         fac = float(Nmesh**3)/N_particles
#         self.facs = {}
#         for i in range(Nmesh**3):
#             self.facs[i] = len(part_list_in_cell[i])/len(self.part_list_in_cell[i])*fac
#
#         # true halo field
#         self.deltah_true = deltah_true.reshape(-1)
#
#     def __len__(self):
#         return len(self.part_list_in_cell.keys()) # the number of cells
#
#     def __getitem__(self, i):
#         '''
#         Get the particle data within the cell specified by inx, and the corresponding true deltah of that cell.
#         '''
#         inds = self.part_list_in_cell[i]
#         return torch.Tensor(self.inputs[inds]), torch.Tensor([self.facs[i]]), torch.Tensor([self.deltah_true[i]])


class MyNetwork(nn.Module):
    def __init__(self, n_input, n_neurons, n_layers, f_activation):
        '''
        n_input specifies how many features to use for a particle.
        '''
        # call constructor from superclass
        super().__init__()

        # define the layers
        # TODO: structure of the NN?
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        self.fc1 = nn.Linear(n_input, self.n_neurons)
        self.fc2s = nn.ModuleList([nn.Linear(self.n_neurons, self.n_neurons) for _ in range(n_layers-1)])
        self.fc3 = nn.Linear(n_neurons, 1)

        self.f_activation = f_activation
    
    def forward(self, x):
        # forward pass
        x = self.f_activation(self.fc1(x))
        for i in range(self.n_layers-1):
            x = self.f_activation(self.fc2s[i](x))
        x = self.fc3(x)
        return x


# functions for training
def calc_output_NN(model, inputs):
    '''
    Calculate and return the f or logf values for all realizations in a batch.
    '''
    N_realizations = inputs.shape[2]
    ys = torch.zeros((inputs.shape[0], inputs.shape[1], N_realizations, 1))
    for n in range(N_realizations):
        ys[:,:,n,:] = model(inputs[:,:,n,:]) # run all particles in a batch through the NN
    return ys

def criterion_nonnegative(ys, logf=True):
    '''
    The non-negativity constraint.  logf=False doesn't work very well.
    '''
    if not logf:
        thres = 0.
        l0 = ys**2
    else:
        thres = -5.
        l0 = (ys-thres)**4
    l0[ys > thres] = 0.
    return l0.sum()

def criterion_integral(ys, logf=True, fac=1., tol=1e-3):
    '''
    The f's should be evaluated on a bunch of pre-selected points.
    \int f x PDF(input) = 1 -> sum of f divided by number of points = 1.
    '''
    if not logf:
        return ((ys.mean()-1.)/tol)**2*fac
    return (((10**ys).mean()-1.)/tol)**2*fac

def criterion_squaredloss(ys, deltah_true, facs, logf=True):
    '''
    TODO: this assumes that, if batch size > 1, each cell should have an equal number of particles
    f_delta1s should be the f values of particles within a cell
    facs should be ncells / nparticles with a correction for the number of particles in each cell
    In the case of multiple realizations, f_deltas should have shape
      batch_size x Npart_per_cell x N_realizations x 1
    '''
    if logf:
        deltah_model = (10**ys).mean(axis=2).sum(axis=1) * facs - 1.
    else:
        deltah_model = ys.mean(axis=2).sum(axis=1) * facs - 1.
    return ((deltah_model - deltah_true)**2).sum()

# def criterion_squaredloss(fs, facs, deltahs, locs, logf=True):
#     if logf:
#         fs = 10**fs
#     deltahs_ = deltahs*0.
#     for i in range(len(deltahs_)):
#         deltahs_[i] = fs[locs[i]:locs[i+1]].sum()*facs[i]-1.
#     return ((deltahs_ - deltahs)**2).sum()

def train(model, inputs, facs, deltah_true, optimizer, logf=True,
          nonnegative_contraint=True,
          integral_constraint=True, fac=1., tol=1e-3, inputs_integral=None):
    model.zero_grad()
    # In the case of multiple realizations, f_deltas should have shape
    #   batch_size x Npart_per_cell x N_realizations x 1
    ys = calc_output_NN(model, inputs)
    loss = criterion_squaredloss(ys, deltah_true, facs, logf=logf)
    if nonnegative_contraint:
        loss += criterion_nonnegative(ys, logf=logf)
    if integral_constraint:
        loss += criterion_integral(model(inputs_integral), logf=logf, fac=fac, tol=tol)
    loss.backward()
    optimizer.step()
    return loss.item()


# functions for calculating the loss and deltah_model
def calc_deltah_model_nbodykit_(model, pos, features, boxsize, Nmesh, interp_method, logf=True, fac=1., minus1=False):
    '''
    Similar to calc_deltah_model_NN, but using nbodykit (this is a lot faster).
    Can be used in calc_deltah_model_nbodykit or used separately.
    If small box, fac=1, minus1=True.
    '''
    fs = np.zeros(len(pos))
    i = 0
    while i<len(pos):
        j = min(len(pos), i+100000)
        fs[i:j] = model(torch.tensor(features.T[np.newaxis,i:j,:])).detach().numpy().reshape(-1)
        i = j
    if logf:
        fs = 10**fs

    # convert to mesh
    mesh = ArrayCatalog({'Position': pos, 'Value': fs}).to_mesh(Nmesh=Nmesh, resampler=interp_method, BoxSize=boxsize).compute()*fac
    if minus1:
        mesh -= 1.
    return mesh

def calc_deltah_model_nbodykit(model, sim, z, Rfs, qs, Nmesh, interp_method, logf=True):
    '''
    Similar to calc_deltah_model_NN, but using nbodykit (this is a lot faster).
    '''
    boxsize = 2000.
    Nfiles = 34
    N0 = 6912
    if 'small' in sim:
        boxsize = 500.
        Nfiles = 1
        N0 = 1728

    # create and run mesh
    mesh = ArrayMesh(np.zeros((Nmesh,Nmesh,Nmesh)), BoxSize=boxsize).compute()
    Nparticles = 0
    for i in range(Nfiles):
        pos, features = load_particle_features(sim, z, i, Rfs, qs)
        mesh += calc_deltah_model_nbodykit_(model, pos, features, boxsize, Nmesh, interp_method, logf=logf, fac=len(pos)/(0.03*N0**3))
    mesh -= 1.

    return mesh

def calc_deltah_model_NN(model, part_list_in_cell, features, logf=True):
    '''
    Loop over the cells to calculate deltah_model.
    features should be a list or numpy array.
    '''
    Nparticles = len(features[0])
    Ncells = len(part_list_in_cell.keys())
    deltah_model = np.zeros(Ncells)
    inputs = features
    if type(features) == list:
        inputs = np.asarray(features)
    inputs = inputs.T
    for i in part_list_in_cell:
        inds = part_list_in_cell[i]
        ys = model(torch.tensor(inputs[np.newaxis,inds,:])).detach().numpy().reshape(-1)
        if logf:
            deltah_model[i] = (10**ys).sum()
        else:
            deltah_model[i] = ys.sum()
    deltah_model *= Ncells/Nparticles
    Nmesh = int(np.round(Ncells**(1/3)))
    return deltah_model.reshape(Nmesh,Nmesh,Nmesh)-1.

def calc_loss_NN(model, part_list_in_cell, deltah_true, features, logf=True):
    '''
    Calculate deltah_model and return the squared loss.
    features should be a list or numpy array.
    '''
    deltah_model = calc_deltah_model_NN(model, part_list_in_cell, features, logf=logf)
    return deltah_model, ((deltah_model-deltah_true)**2).sum()


# class MyNN():
#     def __init__(self, model):
#         self.model = model
#
#     def calc_output_NN(self, inputs):
#         '''
#         Calculate and return the f or logf values for all realizations in a batch.
#         '''
#         N_realizations = inputs.shape[2]
#         ys = torch.zeros((inputs.shape[0], inputs.shape[1], N_realizations, 1))
#         for n in range(N_realizations):
#             ys[:,:,n,:] = self.model(inputs[:,:,n,:]) # run all particles in a batch through the NN
#         return ys
#
#     def criterion_nonnegative(self, ys, logf=True):
#         '''
#         The f>=0 constraint.  logf=False is not a good choice
#         '''
#         if not logf:
#             thres = 0.
#             l0 = ys**2
#         else:
#             thres = -5.
#             l0 = (ys-thres)**4
#         l0[ys > thres] = 0.
#         return l0.sum()
#
#     def criterion_integral(self, ys, logf=True, fac=1., tol=1e-3):
#         '''
#         The f's should be evaluated on a bunch of pre-selected points.
#         \int f x PDF(input) = 1 -> sum of f divided by number of points = 1.
#         '''
#         if not logf:
#             return ((ys.mean()-1.)/tol)**2*fac
#         return (((10**ys).mean()-1.)/tol)**2*fac
#
#     def criterion_squaredloss(self, ys, deltah_true, facs, logf=True):
#         '''
#         TODO: this assumes that, if batch size > 1, each cell should have an equal number of particles
#         f_delta1s should be the f values of particles within a cell
#         facs should be ncells / nparticles with a correction for the number of particles in each cell
#         In the case of multiple realizations, f_deltas should have shape
#           batch_size x Npart_per_cell x N_realizations x 1
#         '''
#         if logf:
#             deltah_model = (10**ys).mean(axis=2).sum(axis=1) * facs - 1.
#         else:
#             deltah_model = ys.mean(axis=2).sum(axis=1) * facs - 1.
#         return ((deltah_model - deltah_true)**2).sum()
#
#
#     def train_batch(self, inputs, facs, deltah_true, optimizer, logf=True,
#               nonnegative_contraint=True,
#               integral_constraint=True, fac=1., tol=1e-4, inputs_integral=None):
#         self.model.zero_grad()
#         # In the case of multiple realizations, f_deltas should have shape
#         #   batch_size x Npart_per_cell x N_realizations x 1
#         ys = calc_output_NN(self.model, inputs)
#         loss = criterion_squaredloss(ys, deltah_true, facs, logf=logf)
#         if nonnegative_contraint:
#             loss += criterion_nonnegative(ys, logf=logf)
#         if integral_constraint:
#             loss += criterion_integral(self.model(inputs_integral), logf=logf, fac=fac, tol=tol)
#         loss.backward()
#         optimizer.step()
#         return loss.item()
#
#     def train_epoch(self, data_train, optimizer, logf=True, nonnegative_contraint=True, integral_constraint=True, fac=1., tol=1e-4, inputs_integral=None, verbose=True):
#         epoch_loss = 0
#         t0 = time.time()
#         for b_idx, batch in enumerate(data_train):
#             b_inputs, b_facs, b_deltah_true = batch
#             #b_inputs.to(device)
#             #b_facs.to(device)
#             #b_deltah_true.to(device)
#             loss = train_batch(self.model, b_inputs, b_facs, b_deltah_true, optimizer, logf=logf,
#                 nonnegative_contraint=nonnegative_contraint,
#                 integral_constraint=integral_constraint,
#                 fac=fac, tol=tol, inputs_integral=inputs_integral)
#             epoch_loss += loss
#         if verbose:
#             print('epoch loss = %.2g, time = %.2g' % (epoch_loss, time.time()-t0))
#         return epoch_loss
#
#     def calc_deltah_model_NN(self, part_list_in_cell, logf=True, *args):
#         Nparticles = len(args[0])
#         inputs = np.zeros((Nparticles, len(args)), dtype=np.float32)
#         Ncells = len(part_list_in_cell.keys())
#         deltah_model = np.zeros(Ncells)
#         for i in range(len(args)):
#             inputs[:,i] = args[i]
#         for i in part_list_in_cell:
#             inds = part_list_in_cell[i]
#             ys = self.model(torch.tensor(inputs[np.newaxis,inds,:])).detach().numpy().reshape(-1)
#             if logf:
#                 deltah_model[i] = (10**ys).sum()
#             else:
#                 deltah_model[i] = ys.sum()
#         deltah_model *= Ncells/Nparticles
#         Nmesh = int(np.round(Ncells**(1/3)))
#         return deltah_model.reshape(Nmesh,Nmesh,Nmesh)-1.
#
#     def calc_loss_NN(self, part_list_in_cell, deltah_true, logf=True, *args):
#         deltah_model = calc_deltah_model_NN(self.model, part_list_in_cell, logf, *args)
#         return deltah_model, ((deltah_model-deltah_true)**2).sum()
#
