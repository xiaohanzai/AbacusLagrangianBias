import numpy as np
from particle_functions import f_ind2ijk, f_ijk2ind


class CICCDF():
    def __init__(self, dim):
        '''
        This class will give the CDF of CIC weights, given the dimensions.
        '''
        self.dim = dim
        xs = np.random.rand(int(1e6), dim)
        ws = np.prod(xs, axis=1)
        Ns, bin_edges = np.histogram(ws, 100, (0,1))
        # bin centers
        self.bins = (bin_edges[1:]+bin_edges[:-1])/2.
        self.bins[-1] = 1.
        self.bins = np.append([0], self.bins)
        # pdf
        self.pdf = Ns/float(len(xs))
        # cdf values
        self.cdf = np.cumsum(Ns)/float(len(xs))
        self.cdf = np.append([0], self.cdf)

    def f_invcdf(self, ps):
        '''
        Given the cumulative probabilities, return the corresponding weights.
        '''
        return np.interp(ps, self.cdf, self.bins)


def gen_part_ws_of_cell(pos, boxsize, Nmesh, cic_fac):
    '''
    Given particle positions, calculate the weights and indices of particles contributing to each cell.
    '''
    dl = boxsize/Nmesh

    # construct cell_ws
    cic_fac_ = int( np.ceil(cic_fac) )
    tmp = np.arange(-cic_fac_+1, cic_fac_+1, dtype=int)
    is_ = tmp.reshape(1,-1,1,1)
    js_ = tmp.reshape(1,1,-1,1)
    ks_ = tmp.reshape(1,1,1,-1)

    tmp = pos/dl
    inds = np.floor(tmp).astype(int) # note that the indices of cells starts from negative
    dxs = tmp - inds

    cell_ws = (
        (1.-np.abs(dxs[:,0].reshape(-1,1,1,1)-is_)/cic_fac).clip(0) * \
        (1.-np.abs(dxs[:,1].reshape(-1,1,1,1)-js_)/cic_fac).clip(0) * \
        (1.-np.abs(dxs[:,2].reshape(-1,1,1,1)-ks_)/cic_fac).clip(0)
        ).reshape(len(pos),-1)

    # construct part_list_in_cell
    inds = (inds+(-cic_fac_+1))%Nmesh # the leftmost cell which a particle has influence n
    inds = f_ijk2ind( inds[:,0], inds[:,1], inds[:,2], Nmesh )

    part_list_in_cell = {}
    for i in range(len(pos)):
        i_cell = inds[i]
        if i_cell in part_list_in_cell:
            part_list_in_cell[i_cell].append(i)
        else:
            part_list_in_cell[i_cell] = [i]

    return part_list_in_cell, cell_ws


def get_part_list_and_ws(part_list_in_cell_, cell_ws, Nmesh):
    '''
    Get the particle indices and weights for a given cell i_cell.
    '''
    cic_fac_ = int( np.round(cell_ws.shape[1]**(1/3) / 2) )

    # construct indices to access neighbors of a cell
    # note that this is different from corresponding lines in gen_part_ws_of_cell
    tmp = np.arange(-cic_fac_, cic_fac_, dtype=int)
    is_ = tmp.reshape(-1,1,1)
    js_ = tmp.reshape(1,-1,1)
    ks_ = tmp.reshape(1,1,-1)

    # the corresponding columns of cell_ws
    n = 2*cic_fac_
    i_cells = f_ijk2ind(is_%n, js_%n, ks_%n, n).reshape(-1)

    part_list_in_cell = {}
    part_ws_of_cell = {}
    for i_cell in range(Nmesh**3):
        part_inds = np.array([], dtype=int)
        ws_inds = np.array([], dtype=int)
        # the neighboring cells needed to take into account
        ii, ij, ik = f_ind2ijk(i_cell, Nmesh)
        i_cells_ = f_ijk2ind((ii+is_)%Nmesh, (ij+js_)%Nmesh, (ik+ks_)%Nmesh, Nmesh).reshape(-1)
        # sum up the particles in them with their weights
        for i_, n_ in zip(i_cells_, i_cells):
            if i_ not in part_list_in_cell_:
                continue
            part_inds = np.append(part_inds, part_list_in_cell_[i_])
            ws_inds = np.append(ws_inds, np.ones(len(part_list_in_cell_[i_]), dtype=int)*n_)
        if len(part_inds) == 0:
            continue
        part_list_in_cell[i_cell] = part_inds
        part_ws_of_cell[i_cell] = cell_ws[(part_inds, ws_inds)]

    return part_list_in_cell, part_ws_of_cell


# do CIC interpolation
def distribute_ws(part_ws_of_cell, Nmesh=None, part_list_in_cell=None, vals=None, pytorch=False):
    '''
    Given the weights of particles (and lists of particle indices in the cells), do CIC interpolation to calculate
      \sum_i w_ij f_i, where i indicates the particle index, j denote the cell index, and w_ij is the CIC weight.
    The f values should be given by the vals array.  If not given, assume 1.
    '''
    if Nmesh is None:
        Nmesh = int(np.round( len(part_ws_of_cell.key())**(1/3) ))
    mesh = np.zeros(Nmesh**3)
    if pytorch:
        mesh = torch.Tensor(mesh) # pytorch needs this
    for i in part_ws_of_cell:
        ws = part_ws_of_cell[i]
        if vals is not None and part_list_in_cell is not None:
            mesh[i] = (vals[part_list_in_cell[i]]*ws).sum()
        else:
            mesh[i] = np.sum(ws)
    return mesh.reshape(Nmesh,Nmesh,Nmesh)


# def distribute_ws_(part_list_in_cell, cell_ws, cic_fac, Nmesh, vals=None, pytorch=False):
#     '''
#     This function is built for test purposes.  part_list_in_cell should be the output of gen_part_ws_of_cell.
#     '''
#     mesh = np.zeros(Nmesh**3)
#     if pytorch:
#         mesh = torch.Tensor(mesh) # pytorch needs this
#
#     cic_fac_ = int( np.ceil(cic_fac) )
#
#     # construct indices to access neighbors of a cell
#     # note that this is different from corresponding lines in gen_part_ws_of_cell
#     tmp = np.arange(-cic_fac_, cic_fac_, dtype=int)
#     is_ = tmp.reshape(-1,1,1)
#     js_ = tmp.reshape(1,-1,1)
#     ks_ = tmp.reshape(1,1,-1)
#
#     # the corresponding columns of cell_ws
#     n = 2*cic_fac_
#     i_cells = f_ijk2ind(is_%n, js_%n, ks_%n, n).reshape(-1)
#
#     for i in range(Nmesh**3):
#         # the neighboring cells needed to take into account
#         ii, ij, ik = f_ind2ijk(i, Nmesh)
#         i_cells_ = f_ijk2ind((ii+is_)%Nmesh, (ij+js_)%Nmesh, (ik+ks_)%Nmesh, Nmesh).reshape(-1)
#         # sum up the particles in them with their weights
#         for i_, n_ in zip(i_cells_, i_cells):
#             if i_ not in part_list_in_cell:
#                 continue
#             if vals is None:
#                 mesh[i] += cell_ws[part_list_in_cell[i_], n_].sum()
#             else:
#                 mesh[i] += (cell_ws[part_list_in_cell[i_], n_]*vals[part_list_in_cell[i_]]).sum()
#
#     return mesh.reshape(Nmesh,Nmesh,Nmesh)

