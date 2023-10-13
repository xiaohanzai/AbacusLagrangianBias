import numpy as np


# functions for nearest neighbor interpolation
def calc_nn_cell(pos, boxsize, Nmesh):
    '''
    Given the particle positions, calculate the cell indices the particles fall in.
    '''
    dl = boxsize/Nmesh
    if len(np.shape(pos)) == 1:
        pos = pos.reshape(-1,3)
    inds_ = pos/dl
    inds = np.floor(inds_).astype(int)
    ii = inds_ - inds >= 0.5 # just to be consistent with nbodykit
    inds[ii] += 1
    return inds%Nmesh

def gen_part_list_in_cell(pos, boxsize, Nmesh):
    '''
    Given the particle positions, return a dict containing the list of particle indices within each cell.
    '''
    inds = calc_nn_cell(pos, boxsize, Nmesh)
    inds = inds[:,0]*Nmesh**2 + inds[:,1]*Nmesh + inds[:,2]
    ind_part = np.linspace(0, len(pos)-1, len(pos), dtype=int)
    part_list_in_cell = {}
    for i in range(len(pos)):
        if inds[i] in part_list_in_cell:
            part_list_in_cell[inds[i]].append(i)
        else:
            part_list_in_cell[inds[i]] = [i]
    return part_list_in_cell

# do nearest neighbor interpolation
def distribute_ws(part_list_in_cell, Nmesh=None, vals=None, pytorch=False):
    '''
    Given the lists of particle indices in the cells, do nearest neighbor interpolation to calculate
      \sum_i w_ij f_i, where i indicates the particle index, j denote the cell index, and w_ij = 0 or 1
      depending on whether particle is in cell.
    If Nmesh is not None, create a whole grid of deltah_model.  Otherwise we are only constructing
      deltah_model_j for a certain number of j.
    The f values should be given by the vals array.  If not given, assume 1.
    '''
    if Nmesh is not None:
        # construct a full grid
        mesh = np.zeros(Nmesh**3)
    else:
        # only return a list of cell values
        mesh = np.zeros(len(part_list_in_cell.key()))
    if pytorch:
        mesh = torch.Tensor(mesh) # pytorch needs this
    for i,j in enumerate(part_list_in_cell):
        inds = part_list_in_cell[j] # particle indices in cell j
        if Nmesh is not None:
            k = j
        else:
            k = i
        if vals is None:
            mesh[k] = len(inds)
        else:
            mesh[k] = vals[inds].sum()
    if Nmesh is not None:
        mesh = mesh.reshape(Nmesh,Nmesh,Nmesh)
    return mesh

