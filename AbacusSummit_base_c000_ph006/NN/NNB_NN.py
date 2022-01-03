import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from NNB_functions import *


class MyDataset(Dataset):
    def __init__(self, arg1, boxsize, Nmesh, deltah_true, Npart_per_cell, N_realizations, *args):
        '''
        arg1 is either the particle positions or the particle list in cells.
        All the particle features should be input at the end.
        '''
        if type(arg1) == dict:
            part_list_in_cell = arg1
        else:
            pos = arg1
            part_list_in_cell = gen_part_list_in_cell(pos, boxsize, Nmesh)
        # randomly choose an equal number of particles for each cell
        self.inputs = np.zeros((Npart_per_cell*Nmesh**3, N_realizations, len(args)))
        self.part_list_in_cell = {}
        fac = float(Nmesh**3)/len(args[0])
        self.facs = {}
        for i in range(Nmesh**3):
            self.facs[i] = len(part_list_in_cell[i])/float(Npart_per_cell)*fac
            ind_ = np.random.choice(part_list_in_cell[i], Npart_per_cell*N_realizations, replace=True)
            self.part_list_in_cell[i] = np.arange(i*Npart_per_cell, (i+1)*Npart_per_cell, dtype=int).tolist()
            for n in range(N_realizations):
                for j, q in enumerate(args):
                    self.inputs[i*Npart_per_cell:(i+1)*Npart_per_cell,n,j] = q[
                        ind_[n*Npart_per_cell:(n+1)*Npart_per_cell]]

        self.deltah_true = deltah_true.reshape(-1)
        self.cell_inds = list(self.part_list_in_cell.keys()) # TODO: we might want to use ordered dict?

    def __len__(self):
        return len(self.part_list_in_cell.keys()) # the number of cells

    def __getitem__(self, i):
        '''
        Get the particle data within the cell specified by inx, and the corresponding true deltah of that cell.
        '''
        inds = self.part_list_in_cell[i]
        return torch.Tensor(self.inputs[inds]), torch.Tensor([self.facs[i]]), torch.Tensor([self.deltah_true[i]])


def criterion(f_delta1s, deltah_true, fac):
    '''
    TODO: this assumes that, if batch size > 1, each cell should have an equal number of particles
    f_delta1s should be the f values of particles within a cell
    fac should be ncells / nparticles
    In the case of multiple realizations, f_deltas should have shape
      batch_size x Npart_per_cell x N_realizations x 1
    '''
    deltah_model = f_delta1s.mean(axis=2).sum(axis=1) * fac - 1.
    return ((deltah_model - deltah_true)**2).sum()


def train(model, inputs, deltah_true, optimizer, criterion, fac):
    '''
    see this simplest example:
    https://towardsdatascience.com/how-to-code-a-simple-neural-network-in-pytorch-for-absolute-beginners-8f5209c50fdd
    '''
    model.zero_grad()
    # In the case of multiple realizations, f_deltas should have shape
    #   batch_size x Npart_per_cell x N_realizations x 1
    f_delta1s = calc_f_NN(model, inputs)
    loss = criterion(f_delta1s, deltah_true, fac)
    loss.backward()
    optimizer.step()
    return loss


def calc_f_NN(model, inputs):
    '''
    Calculate and return the f values for all realizations in a batch.
    '''
    N_realizations = inputs.shape[2]
    f_delta1s = torch.zeros((inputs.shape[0], inputs.shape[1], N_realizations, 1))
    for n in range(N_realizations):
        f_delta1s[:,:,n,:] = model(inputs[:,:,n,:]) # run all particles in a batch through the NN
    return f_delta1s


