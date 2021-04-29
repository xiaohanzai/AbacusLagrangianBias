import numpy as np
import sys
import os

# parameters about the sim
boxsize = 2000.
k0 = 2*np.pi/boxsize
N0 = 6912
N = 1152
Nfiles = 34

use_nabla2d1 = True

interp_method = 'cic'

def load_matrixA_slab(islab, path, Nmesh, interp_method='cic', remove_overlaps=(False, False), direct_load=False, sum_nabla2d1_bins=False, nbins_nabla2d1=None):
    '''
    Load the matrix A for a single slab.  Take care of the boundaries.
    If sum up the nabla2d1 bins, get the matrix A assuming f(delta1).
    '''
    Nmesh2 = Nmesh**2
    with np.load(path + '/matrixA_slab%d_Nmesh%d_%s.npz' % (islab, Nmesh, interp_method)) as tmp:
        A_ = tmp['A']
        ind_slab = tmp['ind_slab']
    if sum_nabla2d1_bins:
        A = np.zeros((A_.shape[0]//nbins_nabla2d1, A_.shape[1]))
        for i in range(nbins_nabla2d1):
            A += A_[i::nbins_nabla2d1]
        del A_
    else:
        A = A_
    if direct_load:
        return A, ind_slab

    # left and right
    imin = 0
    imax = len(ind_slab)
    for l in range(2):
        islab_ = (islab-(-1)**l)%Nfiles
        remove = remove_overlaps[l]
        with np.load(path + '/matrixA_slab%d_Nmesh%d_%s.npz' % (islab_, Nmesh, interp_method)) as tmp:
            A1_ = tmp['A']
            ind_slab1 = tmp['ind_slab']
        if sum_nabla2d1_bins:
            A1 = np.zeros_like(A)
            for i in range(nbins_nabla2d1):
                A1 += A1_[i::nbins_nabla2d1]
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

def main():
    sim, z, Rf, Nmesh = sys.argv[1:]
    z = float(z)
    Rf = float(Rf)
    Nmesh = int(Nmesh)

    # saved to which folder
    folder = 'matrixA'
    if use_nabla2d1:
        folder += '_nabla2d1'

    path = '/mnt/store2/xwu/AbacusSummit/%s/z%s_tilde_operators_nbody/Rf%.3g/' % (sim, str(z), Rf) + folder

    outpath = path+'/nooverlap'
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    # load each A and save to disk
    for i in range(Nfiles):
        A, ind_slab = load_matrixA_slab(i, path, Nmesh, interp_method, remove_overlaps=(False, True))
        np.savez_compressed(outpath+'/matrixA_slab%d_Nmesh%d_%s' % (i, Nmesh, interp_method), A=A, ind_slab=ind_slab)

if __name__ == '__main__':
    main()

