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
from NNB_NN import * # but a few things will be rewritten
from CIC_functions import *


def find_nearest(xs_desired, xs):
    '''
    Find the indices of the xs array where the values are closest to xs_desired.
    '''
    inds = np.zeros(len(xs_desired), dtype=int)
    for i,x in enumerate(xs_desired):
        inds[i] = np.argmin( np.abs(xs - x) )
    return inds


class MyDataset(Dataset):
    def __init__(self, part_list_in_cell, part_ws_of_cell, boxsize, Nmesh, deltah_true, subsample_fac, features):
        '''
        Particle positions and features are input as pos and features.  features.shape[0] = number of features, shape[1] = number of particles.
        subsample_fac should be a float, e.g. 0.5, 0.25.  We subsample this fraction of particles around each grid point.
        Note that deltah_true should be smoothed properly too (CIC interpolated with the corresponding kernel size).
        '''
        N_particles = features.shape[1]

        # true halo field
        self.deltah_true = deltah_true.reshape(-1)

        # correction factors will be incorporated into the weights
        fac = float(Nmesh**3)/N_particles

        if abs(subsample_fac - 1) < 1e-6:
            self.part_list_in_cell = part_list_in_cell
            self.part_ws_of_cell = {}
            for i in part_list_in_cell:
                self.part_ws_of_cell[i] = np.asarray(part_ws_of_cell[i])*fac
            self.inputs = np.asarray(features.T).astype(np.float32)
            self.cell_inds = list(self.part_list_in_cell.keys())
            return

        ciccdf = CICCDF(3)

        self.part_list_in_cell = {} # stores the indices of the sampled particles
        self.part_ws_of_cell = {} # stores the CIC weights of the sampled particles
        ii_all = np.zeros(N_particles, dtype=bool)
        for i in part_list_in_cell:
            # get the indices and weights of particles contributing to cell i
            inds, ws = np.asarray(part_list_in_cell[i]), np.asarray(part_ws_of_cell[i])

            # subsample particles according to their weights
            n_new = max(int(len(ws)*subsample_fac),1)
            # ps = np.random.rand(n_new)
            # ws_ps = ciccdf.f_invcdf(ps)
            # inds_new = find_nearest(ws_ps, ws)
            inds_new = np.random.choice(len(ws), n_new)
            ws_new = ws[inds_new]
            inds_new = inds[inds_new]

            # put into dict
            self.part_list_in_cell[i] = inds_new
            self.part_ws_of_cell[i] = ws_new * ws.sum()/ws_new.sum()*fac # incorporate the correction factor into the weights
            ii_all[inds_new] = True

        # build inputs
        self.inputs = features.T[ii_all,:]

        # correct for the particle indices
        inds = np.where(ii_all>0)[0]
        f_inds_new = {}
        for i, j in enumerate(inds):
            f_inds_new[j] = i
        for i in self.part_list_in_cell:
            for j in range(len(self.part_list_in_cell[i])):
                self.part_list_in_cell[i][j] = f_inds_new[self.part_list_in_cell[i][j]]

        self.cell_inds = list(self.part_list_in_cell.keys())

    def __len__(self):
        return len(self.cell_inds) # the number of cells

    def __getitem__(self, i_):
        '''
        Get the particle data within the cell specified by inx, and the corresponding true deltah of that cell.
        '''
        i = self.cell_inds[i_]
        inds = self.part_list_in_cell[i]
        return torch.Tensor(self.inputs[inds]), torch.Tensor(self.part_ws_of_cell[i]), torch.Tensor([self.deltah_true[i]])


def collate_fn(batch):
    '''
    Custom dataloader.  Concatenate cells in a batch together.
    Return locs as an indicator of where to separate the particles of different cells.
    '''
    locs = [0]
    n = 0 # stores the number of particles in total
    for item in batch:
        #inputs, ws, deltah_true = item
        n_ = n+len(item[0]) # number of particles in a cell
        locs.append(n_)
        n = n_
    b_inputs = torch.zeros((n, batch[0][0].shape[1])) # shape = number of particles, number of features
    b_ws = torch.zeros(n)
    b_deltahs = torch.zeros(len(batch)) # length = number of cells
    for i in range(len(batch)):
        b_inputs[locs[i]:locs[i+1]] = batch[i][0]
        b_ws[locs[i]:locs[i+1]] = batch[i][1]
        b_deltahs[i] = batch[i][2]
    return b_inputs, b_ws, b_deltahs, locs


def criterion_squaredloss(fs, ws, deltahs, locs, logf=True):
    if logf:
        fs = 10**fs
    deltahs_ = deltahs*0.
    for i in range(len(deltahs_)):
        deltahs_[i] = (fs[locs[i]:locs[i+1]].reshape(-1)*ws[locs[i]:locs[i+1]]).sum()-1.
    return ((deltahs_ - deltahs)**2).sum()


def train(model, inputs, ws, deltahs, locs, optimizer, logf=True,
          nonnegative_contraint=True,
          integral_constraint=True, fac=1., tol=1e-3, inputs_integral=None):
    model.zero_grad()
    # In the case of multiple realizations, f_deltas should have shape
    #   batch_size x Npart_per_cell x N_realizations x 1
    # ys = calc_output_NN(model, inputs)
    ys = model(inputs)
    loss = criterion_squaredloss(ys, ws, deltahs, locs, logf=logf)
    if nonnegative_contraint:
        loss += criterion_nonnegative(ys, logf=logf)
    if integral_constraint:
        loss += criterion_integral(model(inputs_integral), logf=logf, fac=fac, tol=tol)
    loss.backward()
    optimizer.step()
    return loss.item()

