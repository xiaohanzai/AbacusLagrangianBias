import numpy as np
import sys
import os
from nbodykit.lab import *
from nbodykit import setup_logging, style

interp_method = 'cic'

def _downsampleA(A, ind_slab, Nmesh, Nmesh_new, interp_method):
    tmp = np.linspace(0, Nmesh-1, Nmesh, dtype=int)
    x_ = tmp+0.
    x_[Nmesh//2:] -= Nmesh

    indx, indy, indz = np.meshgrid(ind_slab,tmp,tmp, indexing='ij')
    pos = np.zeros((A.shape[1],3))
    pos[:,0] = x_[indx.reshape(-1)]
    pos[:,1] = x_[indy.reshape(-1)]
    pos[:,2] = x_[indz.reshape(-1)]

    s = Nmesh/Nmesh_new
    ind_slab_new = np.array([int(ind_slab[0]/s)], dtype=int)
    for i in ind_slab[1:]:
        ind = int(i/s)
        if ind != ind_slab_new[-1]:
            ind_slab_new = np.append(ind_slab_new, ind)
    ind = int(np.ceil(ind_slab[-1]/s))
    if ind != ind_slab_new[-1]:
        ind_slab_new = np.append(ind_slab_new, ind)
    ncell_new = Nmesh_new**2 * len(ind_slab_new)
    nbins = A.shape[0]
    A_new = np.zeros((nbins, ncell_new))

    for m in range(nbins):
        Am = ArrayCatalog({'Position': pos, 'Value': A[m]}).to_mesh(
            Nmesh=Nmesh_new, BoxSize=Nmesh, resampler=interp_method).compute()
        A_new[m] = Am[ind_slab_new].reshape(-1)
        if np.sum(Am)>0.:
            A_new[m] = A_new[m]/np.sum(Am)*np.sum(A[m]) # preserve sum
    return A_new, ind_slab_new

def load_matrixA_slab(islab, path, Nmesh, interp_method='cic', remove_overlaps=(False, False), direct_load=False, kmax=None, Nmesh_new=None, **kwargs):
    '''
    Load the matrix A for a single slab.  Take care of the boundaries.
    If sum up the nabla2d1 or G2 bins, get the matrix A assuming f(delta1).
    If downsample, I assume the no-overlapping has been done already.
    '''
    # to be compatible with previous versions
    sum_q_bins = False
    nbins_q = None
    if 'sum_nabla2d1_bins' in kwargs:
        sum_q_bins = kwargs['sum_nabla2d1_bins']
        nbins_q = kwargs['nbins_nabla2d1']
    if 'sum_G2_bins' in kwargs:
        sum_q_bins = kwargs['sum_G2_bins']
        nbins_q = kwargs['nbins_G2']

    Nmesh2 = Nmesh**2
    fname = path + '/matrixA_slab%d_Nmesh%d_%s.npz' % (islab, Nmesh, interp_method)
    if kmax: # direct load only
        fname = path + '/matrixA_slab%d_Nmesh%d_%s_kmax%.2f.npz' % (islab, Nmesh, interp_method, kmax)
    with np.load(fname) as tmp:
        A_ = tmp['A']
        ind_slab = tmp['ind_slab']
    if sum_q_bins:
        A = np.zeros((A_.shape[0]//nbins_q, A_.shape[1]))
        for i in range(nbins_q):
            A += A_[i::nbins_q]
        del A_
    else:
        A = A_
    if direct_load:
        if Nmesh_new:
            A, ind_slab = _downsampleA(A, ind_slab, Nmesh, Nmesh_new, interp_method)
        return A, ind_slab

    # left and right
    imin = 0
    imax = len(ind_slab)
    if 'small' not in path:
        for l in range(2):
            islab_ = (islab-(-1)**l)%Nfiles
            remove = remove_overlaps[l]
            with np.load(path + '/matrixA_slab%d_Nmesh%d_%s.npz' % (islab_, Nmesh, interp_method)) as tmp:
                A1_ = tmp['A']
                ind_slab1 = tmp['ind_slab']
            if sum_q_bins:
                A1 = np.zeros_like(A)
                for i in range(nbins_q):
                    A1 += A1_[i::nbins_q]
                del A1_
            else:
                A1 = A1_
            for i,j in enumerate(ind_slab1):
                # find the overlaps
                ind = np.where(ind_slab == j)[0]
                if len(ind) > 0:
                    ind = ind[0]
                    if not remove:
                        A[:,ind*Nmesh2:(ind+1)*Nmesh2] += A1[:,i*Nmesh2:(i+1)*Nmesh2]
                    else:
                        if l == 0: # left
                            imin = ind+1
                        else: # right
                            if imax > ind:
                                imax = ind
    return A[:, imin*Nmesh2:imax*Nmesh2], ind_slab[imin:imax]

def save_nooverlap(path, Nmesh):
    outpath = path+'/nooverlap'
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    # load each A and save to disk
    for i in range(Nfiles):
        A, ind_slab = load_matrixA_slab(i, path, Nmesh, interp_method, remove_overlaps=(False, True))
        np.savez_compressed(outpath+'/matrixA_slab%d_Nmesh%d_%s' % (i, Nmesh, interp_method), A=A, ind_slab=ind_slab)

def save_downsample(path, Nmesh, Nmesh_new):
    for i in range(Nfiles):
        A, ind_slab = load_matrixA_slab(i, path, Nmesh, interp_method, direct_load=True, Nmesh_new=Nmesh_new)
        np.savez_compressed(path+'/matrixA_slab%d_Nmesh%d_%s' % (i, Nmesh_new, interp_method), A=A, ind_slab=ind_slab)

def main():
    sim, z, Rf, Nmesh, qname = sys.argv[1:6]
    z = float(z)
    Rf = float(Rf)
    Nmesh = int(Nmesh)
    try:
        Nmesh_new = int(sys.argv[6])
    except:
        Nmesh_new = None

    global Nfiles
    Nfiles = 34
    if 'small' in sim:
        Nfiles = 1
        sim = 'small/'+sim

    # saved to which folder
    folder = 'matrixA'
    if qname in ['nabla2d1', 'G2']:
        folder += '_'+qname

    path = '/mnt/store2/xwu/AbacusSummit/%s/z%s_tilde_operators_nbody/Rf%.3g/' % (sim, str(z), Rf) + folder

    if Nmesh_new:
        save_downsample(path, Nmesh, Nmesh_new)
        save_nooverlap(path, Nmesh_new)
    else:
        save_nooverlap(path, Nmesh)

if __name__ == '__main__':
    main()

