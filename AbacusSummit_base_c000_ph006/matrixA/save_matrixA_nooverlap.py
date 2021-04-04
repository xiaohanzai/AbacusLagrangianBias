import numpy as np
import sys
import os

# parameters about the sim
boxsize = 2000.
k0 = 2*np.pi/boxsize
N0 = 6912
N = 1728
Nfiles = 34

use_nabla2d1 = True

interp_method = 'cic'

def load_matrixA_slab(islab, path, Nmesh, interp_method, remove_overlaps=(False, False), direct_load=False):
    '''
    Load the matrix A for a single slab.  Take care of the boundaries.
    '''
    Nmesh2 = Nmesh**2
    with np.load(path + '/matrixA_slab%d_Nmesh%d_%s.npz' % (islab, Nmesh, interp_method)) as tmp:
        A = tmp['A']
        ind_slab = tmp['ind_slab']
    if direct_load:
        return A, ind_slab

    # left and right
    imin = 0
    imax = len(ind_slab)
    for l in range(2):
        islab_ = (islab-(-1)**l)%Nfiles
        remove = remove_overlaps[l]
        with np.load(path + '/matrixA_slab%d_Nmesh%d_%s.npz' % (islab_, Nmesh, interp_method)) as tmp:
            A1 = tmp['A']
            ind_slab1 = tmp['ind_slab']
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
    z, Rf, Nmesh = sys.argv[1:]
    z = float(z)
    Rf = float(Rf)
    Nmesh = int(Nmesh)

    # saved to which folder
    folder = 'matrixA'
    if use_nabla2d1:
        folder += '_nabla2d1'

    path = '/mnt/store2/xwu/AbacusSummit/base_c000_ph006/z%s_tilde_operators_nbody/Rf%.3g/' % (str(z), Rf) + folder

    outpath = path+'/nooverlap'
    os.mkdir(outpath)

    # load each A and save to disk
    for i in range(Nfiles):
        A, ind_slab = load_matrixA_slab(i, path, Nmesh, interp_method, remove_overlaps=(False, True))
        np.savez_compressed(outpath+'/matrixA_slab%d_Nmesh%d_%s' % (i, Nmesh, interp_method), A=A, ind_slab=ind_slab)

if __name__ == '__main__':
    main()

