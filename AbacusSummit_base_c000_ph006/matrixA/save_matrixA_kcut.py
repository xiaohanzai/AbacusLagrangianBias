import numpy as np
from nbodykit.lab import *
from nbodykit import setup_logging, style
import pyfftw
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')
sys.path.append('../../')
import os
from run_matrixA import bin_edges_d1, bin_edges_nabla2d1_percentile, bin_edges_G2_percentile
from save_matrixA_nooverlap import load_matrixA_slab
from cosmology_functions import calc_k_grid

interp_method = 'cic'

def calc_A_kcut(boxsize, Nmesh, path, kmax, nbins, Nfiles):
    kx, ky, kz = calc_k_grid(boxsize, Nmesh)
    k2 = kx**2 + ky**2 + kz**2
    ii = k2 > kmax**2
    del kx, ky, kz, k2
    Nmesh2 = Nmesh**2

    # process the matrices in batches
    nbins_ = min(nbins, int(1.5*1024**3/Nmesh**3)) # process this number of bins at a time
    nbatches = int(np.ceil(nbins/nbins_))
    if Nfiles == 1:
        nbatches = 1
    print(nbatches, nbins_)

    outpath = path + '/nooverlap'
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    remove_overlaps=(False, True)

    for i in range(nbatches):
        nbins_ = min((i+1)*nbins_, nbins) - i*nbins_
        A_ = np.zeros((nbins_, Nmesh**3)) # init matrix
        for islab in range(Nfiles):
            with np.load(path + '/matrixA_slab%d_Nmesh%d_%s.npz' % (islab, Nmesh, interp_method)) as tmp:
                A = tmp['A']
                ind_slab = tmp['ind_slab']
            for j,ind in enumerate(ind_slab):
                A_[:, ind*Nmesh2:(ind+1)*Nmesh2] += A[i*nbins_:(i+1)*nbins_, j*Nmesh2:(j+1)*Nmesh2]
        for j in range(nbins_):
            tmp = A_[j].reshape(Nmesh,Nmesh,Nmesh)
            tmp = pyfftw.interfaces.numpy_fft.rfftn(tmp)
            tmp[ii] = 0 # apply k cut
            A_[j] = pyfftw.interfaces.numpy_fft.irfftn(tmp).reshape(-1)

        # make it no overlap
        if Nfiles > 1: # no need to do this for small box
            islab = 0
            while islab < Nfiles:
                A = ind_slab = ind_slabl = ind_slabr = None
                if i == 0: # only need to remove overlap in the first round
                    with np.load(path + '/matrixA_slab%d_Nmesh%d_%s.npz' % (islab, Nmesh, interp_method)) as tmp:
                        ind_slab = tmp['ind_slab']
                    with np.load(path + '/matrixA_slab%d_Nmesh%d_%s.npz' % ((islab-1)%Nfiles, Nmesh, interp_method)) as tmp:
                        ind_slabl = tmp['ind_slab']
                    # left and right
                    imin = 0
                    imax = len(ind_slab)
                    with np.load(path + '/matrixA_slab%d_Nmesh%d_%s.npz' % ((islab+1)%Nfiles, Nmesh, interp_method)) as tmp:
                        ind_slabr = tmp['ind_slab']
                    for l,ind_slab1 in enumerate([ind_slabl,ind_slabr]):
                        remove = remove_overlaps[l]
                        for j in ind_slab1:
                            # find the overlaps
                            ind = np.where(ind_slab == j)[0]
                            if len(ind) > 0:
                                ind = ind[0]
                                if remove:
                                    if l == 0: # left
                                        imin = ind+1
                                    else: # right
                                        if imax > ind:
                                            imax = ind

                    A = np.zeros((nbins, (imax-imin)*Nmesh2))
                    ind_slab = ind_slab[imin:imax]
                else:
                    with np.load(outpath + '/matrixA_slab%d_Nmesh%d_%s_kmax%.2f.npz' % (islab, Nmesh, interp_method, kmax)) as tmp:
                        A = tmp['A']
                        ind_slab = tmp['ind_slab']
                for j,ind in enumerate(ind_slab):
                    A[i*nbins_:(i+1)*nbins_, j*Nmesh2:(j+1)*Nmesh2] = A_[:, ind*Nmesh2:(ind+1)*Nmesh2]
                np.savez_compressed(outpath + '/matrixA_slab%d_Nmesh%d_%s_kmax%.2f' % (islab, Nmesh, interp_method, kmax), A=A, ind_slab=ind_slab)

                islab += 1
                ind_slabl = ind_slab
                ind_slab = ind_slabr
        else:
            np.savez_compressed(outpath+'/matrixA_slab%d_Nmesh%d_%s_kmax%.2f' % (i, Nmesh, interp_method, kmax), A=A_, ind_slab=ind_slab)

def main():
    sim, z, Rf, Nmesh, qname, kmax = sys.argv[1:7]
    # redshift, smoothing scale, mesh size
    z = float(z)
    Rf = float(Rf)
    Nmesh = int(Nmesh)
    kmax = float(kmax)

    boxsize = 2000.
    Nfiles = 34
    if 'small' in sim:
        boxsize = 500.
        Nfiles = 1
        sim = 'small/'+sim

    folder = 'matrixA'
    nbins = len(bin_edges_d1)-1
    if qname in ['nabla2d1', 'G2']:
        folder += '_'+qname
        if qname == 'nabla2d1':
            nbins *= len(bin_edges_nabla2d1_percentile)-1
        else:
            nbins *= len(bin_edges_G2_percentile)-1
    path = '/mnt/store2/xwu/AbacusSummit/%s/z%s_tilde_operators_nbody/Rf%.3g/' % (sim, str(z), Rf) + folder

    calc_A_kcut(boxsize, Nmesh, path, kmax, nbins, Nfiles)

# def main():
#     sim, z, Rf, Nmesh, qname, kmax = sys.argv[1:7]
#     # redshift, smoothing scale, mesh size
#     z = float(z)
#     Rf = float(Rf)
#     Nmesh = int(Nmesh)
#     kmax = float(kmax)
#
#     boxsize = 2000.
#     Nfiles = 34
#     if 'small' in sim:
#         boxsize = 500.
#         Nfiles = 1
#         sim = 'small/'+sim
#
#     folder = 'matrixA'
#     if qname in ['nabla2d1', 'G2']:
#        folder += '_'+qname
#     outpath = '/mnt/store2/xwu/AbacusSummit/%s/z%s_tilde_operators_nbody/Rf%.3g/' % (sim, str(z), Rf) + folder
#
#     kx, ky, kz = calc_k_grid(boxsize, Nmesh)
#     k2 = kx**2 + ky**2 + kz**2
#     ii = k2 > kmax**2
#     del kx, ky, kz, k2
#     # I would only deal with the small boxes for now
#     islab = 0
#     A, ind_slab = load_matrixA_slab(islab, outpath, Nmesh, interp_method=interp_method, direct_load=True)
#     nbins = A.shape[0]
#     for i in range(nbins):
#         A_ = A[i].reshape(Nmesh,Nmesh,Nmesh)
#         A_ = pyfftw.interfaces.numpy_fft.rfftn(A_)
#         A_[ii] = 0
#         A[i] = pyfftw.interfaces.numpy_fft.irfftn(A_).reshape(-1)
#
#     np.savez_compressed(outpath + '/matrixA_slab%d_Nmesh%d_%s_kmax%.2f' % (islab, Nmesh, interp_method, kmax), A=A, ind_slab=ind_slab)

if __name__ == '__main__':
    main()

