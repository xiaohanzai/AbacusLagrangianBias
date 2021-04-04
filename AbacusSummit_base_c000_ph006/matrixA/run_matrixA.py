import numpy as np
from nbodykit.lab import *
from nbodykit import setup_logging, style
import pyfftw
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import warnings
warnings.filterwarnings("ignore")
import sys
import os

use_nabla2d1 = False#True

# some important parameters
boxsize = 2000.
Nfiles = 34
interp_method = 'cic'

# delta1 bins
#bin_edges_d1 = np.concatenate(([-6], np.linspace(-4, 5, 40+1), [6]))
bin_edges_d1 = np.linspace(-4, 5, 40+1)
nbins_d1 = len(bin_edges_d1)-1

# nabla2d1 bins
bin_edges_nabla2d1_percentile = np.array([0, 5, 35, 65, 95, 100])
nbins_nabla2d1 = 1
folder = 'matrixA'
if use_nabla2d1:
    nbins_nabla2d1 = len(bin_edges_nabla2d1_percentile)-1
    folder += '_nabla2d1'

# total number of bins
nbins = nbins_d1*nbins_nabla2d1

def load_field_particles(islab, z):
    '''load in field particles of a slab'''
    cat = CompaSOHaloCatalog(
        '/mnt/store2/bigsims/AbacusSummit/AbacusSummit_base_c000_ph006/halos/z%.3f/halo_info/halo_info_%03d.asdf' %
        (z,islab), fields=[], load_subsamples='A_field_rv')
    xmin, xmax = cat.subsamples['pos'][:,0].min(), cat.subsamples['pos'][:,0].max()
    return xmin, xmax, cat.subsamples['pos']

def load_particles(islab, z, Rf):
    '''load in particles in one slab'''
    # position
    pos = load_field_particles(islab, z)[-1]
    cat = CompaSOHaloCatalog(
        '/mnt/store2/bigsims/AbacusSummit/AbacusSummit_base_c000_ph006/halos/z%.3f/halo_info/halo_info_%03d.asdf' %
        (z,islab), fields=[], load_subsamples='A_halo_rv')
    pos = np.concatenate((pos, cat.subsamples['pos']))
    del cat

    # delta1
    with np.load('/mnt/store2/xwu/AbacusSummit/base_c000_ph006/z%s_tilde_operators_nbody/Rf%.3g/slab%d_sdelta1.npz' % (str(z),Rf,islab)) as tmp:
        sdelta1 = np.concatenate((tmp['field'], tmp['halo']))

    # nabla2d1
    nabla2d1 = None
    if use_nabla2d1:
        with np.load('/mnt/store2/xwu/AbacusSummit/base_c000_ph006/z%s_tilde_operators_nbody/Rf%.3g/slab%d_nabla2d1.npz' % (str(z),Rf,islab)) as tmp:
            nabla2d1 = np.concatenate((tmp['field'], tmp['halo']))

    return pos, sdelta1, nabla2d1

#def calc_ind_inslab(Nmesh, xmin, xmax):
#    # indices in x direction of the grid points that are within xmin xmax
#    x = np.linspace(0, boxsize-boxsize/Nmesh, Nmesh)
#    x[Nmesh//2:] -= boxsize
#    ind_inslab = np.where((x >= xmin) & (x <= xmax))[0]
#    if islab == Nfiles-1:
#        ind_inslab = np.append(ind_inslab, ind_inslab[-1]+1)
#    return ind_inslab

def calc_ind_slab(Nmesh, xmin, xmax):
    # indices in x direction of the grid points that a slab contributes to
    indmin = int(xmin/(boxsize/Nmesh)%Nmesh)
    indmax = int(xmax/(boxsize/Nmesh)%Nmesh)+1
    if indmax < indmin:
        indmax += Nmesh
    ind_slab = np.arange(indmin, indmax+1e-6, 1, dtype=int)%Nmesh
    return ind_slab

def load_and_process_particles(islab, z, Rf, Nmesh, sigma_sdelta1, sigma_nabla2d1):
    '''load in particles, discard unuseful ones, sort according to density'''
    # load in one slab
    pos, sdelta1, nabla2d1 = load_particles(islab, z, Rf)
    # normalize the delta1 values
    sdelta1 /= sigma_sdelta1
    # throw away particles not in delta1 bins
    ii_d1 = (sdelta1 >= bin_edges_d1[0]) & (sdelta1 <= bin_edges_d1[-1])
    sdelta1 = sdelta1[ii_d1]
    pos = pos[ii_d1]

    # find out which particles belong to which delta1 bins
    # first sort delta1 values
    ind = np.argsort(sdelta1)
    sdelta1 = sdelta1[ind]
    pos = pos[ind]
    # indices of particles belonging to each bin
    arr = np.searchsorted(sdelta1, bin_edges_d1)

    # whether to use nabla2d1
    if use_nabla2d1:
        nabla2d1 /= sigma_nabla2d1
        nabla2d1 = nabla2d1[ii_d1][ind]

    # indices in x direction of the grid points that are within xmin xmax
    if islab != Nfiles-1:
        ind_slab = calc_ind_slab(Nmesh, pos[:,0].min(), pos[:,0].max())
    else:
        ii = pos[:,0]>0.
        ind_slab = np.concatenate((calc_ind_slab(Nmesh, pos[ii,0].min(), boxsize/2.)[:-2],
            calc_ind_slab(Nmesh, -boxsize/2.+1e-3, pos[~ii,0].max())))

    return pos, sdelta1, nabla2d1, arr, ind_slab

def calc_A(pos, nabla2d1, arr, ind_slab, Nmesh):
    '''calculate A with nbodykit'''
    ncell = Nmesh**2 * len(ind_slab)
    A = np.zeros((nbins, ncell))

    for m in range(nbins_d1):
        if arr[m] == arr[m+1]: # empty bin
            continue
        if use_nabla2d1:
            nabla2d1_ = nabla2d1[arr[m]:arr[m+1]]
            bin_edges_nabla2d1 = np.percentile(nabla2d1_, bin_edges_nabla2d1_percentile)
        else:
            nabla2d1_ = np.zeros(arr[m+1]-arr[m], dtype=np.float32)
            bin_edges_nabla2d1 = np.linspace(-100, 100, nbins_nabla2d1+1)
        for n in range(nbins_nabla2d1):
            ii = (nabla2d1_ >= bin_edges_nabla2d1[n]) & (nabla2d1_ < bin_edges_nabla2d1[n+1])
            if ii.sum() == 0: # empty bin
                continue
            Am = ArrayCatalog({'Position': pos[arr[m]:arr[m+1]][ii]}).to_mesh(
                Nmesh=Nmesh,
                BoxSize=boxsize,
                resampler=interp_method
                ).compute()
            Am *= ii.sum()/Am.size # need to scale it back
            A[m*nbins_nabla2d1+n] = Am[ind_slab].reshape(-1)

    return A

def main():
    # redshift, smoothing scale, mesh size
    z = float(sys.argv[1])
    Rf = float(sys.argv[2])
    Nmesh = int(sys.argv[3])
    if len(sys.argv) > 4:
        islabs = sys.argv[4:]
    else:
        islabs = np.linspace(0,Nfiles-1,Nfiles,dtype=int)

    # load in the smoothed delta_1 and calculate std
    tmp = np.load('/mnt/store2/xwu/AbacusSummit/AbacusSummit_base_c000_ph006/ic/sdelta1_Rf%.3g.npy' % Rf)
    sigma_sdelta1 = np.std(tmp)
    del tmp
    # load in nabla^2 delta_1 and calculate std
    sigma_nabla2d1 = None
    if use_nabla2d1:
        tmp = np.load('/mnt/store2/xwu/AbacusSummit/AbacusSummit_base_c000_ph006/ic/nabla2d1_Rf%.3g.npy' % Rf)
        sigma_nabla2d1 = np.std(tmp)
        del tmp

    outpath = '/mnt/store2/xwu/AbacusSummit/base_c000_ph006/z%s_tilde_operators_nbody/Rf%.3g/' % (str(z), Rf) + folder
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    for islab in islabs:
        islab = int(islab)
        # load in particles
        pos, sdelta1, nabla2d1, arr, ind_slab = load_and_process_particles(islab, z, Rf, Nmesh, sigma_sdelta1, sigma_nabla2d1)
        # calculate A
        A = calc_A(pos, nabla2d1, arr, ind_slab, Nmesh)
        # save to disk
        np.savez_compressed(outpath + '/matrixA_slab%d_Nmesh%d_%s' % (islab, Nmesh, interp_method), A=A, ind_slab=ind_slab)

if __name__ == '__main__':
    main()

