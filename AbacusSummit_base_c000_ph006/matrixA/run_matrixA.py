import numpy as np
from nbodykit.lab import *
from nbodykit import setup_logging, style
import pyfftw
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import warnings
warnings.filterwarnings("ignore")
import sys
import os

# some important parameters
boxsize = 2000.
Nfiles = 34
N = 1152
interp_method = 'cic'

# delta1 bins
#bin_edges_d1 = np.concatenate(([-6], np.linspace(-4, 5, 40+1), [6]))
bin_edges_d1 = np.linspace(-4, 5, 40+1)
nbins_d1 = len(bin_edges_d1)-1

# bins for the other quantity
bin_edges_nabla2d1_percentile = np.array([0, 5, 35, 65, 95, 100])
bin_edges_G2_percentile = np.array([0, 10, 30, 70, 90, 100])

def load_field_particles(islab, sim, z):
    '''load in field particles of a slab'''
    cat_path = '/mnt/store2/bigsims/AbacusSummit/%s/halos/z%.3f/halo_info/halo_info_%03d.asdf' % (sim,z,islab)
    cat = CompaSOHaloCatalog(cat_path, fields=[], load_subsamples='A_field_rv')
    xmin, xmax = cat.subsamples['pos'][:,0].min(), cat.subsamples['pos'][:,0].max()
    return xmin, xmax, cat.subsamples['pos']

def load_particles(islab, sim, z, Rf, qname):
    '''load in particles in one slab'''
    # position
    cat_path = '/mnt/store2/bigsims/AbacusSummit/%s/halos/z%.3f/halo_info/halo_info_%03d.asdf' % (sim,z,islab)
    cat1 = CompaSOHaloCatalog(cat_path, fields=[], load_subsamples='A_field_rv')
    cat2 = CompaSOHaloCatalog(cat_path, fields=[], load_subsamples='A_halo_rv')
    pos = np.concatenate((cat1.subsamples['pos'], cat2.subsamples['pos']))
    del cat1, cat2

    # delta1
    with np.load('/mnt/store2/xwu/AbacusSummit/%s/z%s_tilde_operators_nbody/Rf%.3g/slab%d_sdelta1.npz' % (sim,str(z),Rf,islab)) as tmp:
        sdelta1 = np.concatenate((tmp['field'], tmp['halo']))

    # nabla2d1 or G2
    q = None
    if qname in ['nabla2d1', 'G2']:
        with np.load('/mnt/store2/xwu/AbacusSummit/%s/z%s_tilde_operators_nbody/Rf%.3g/slab%d_%s.npz' % (sim,str(z),Rf,islab,qname)) as tmp:
            q = np.concatenate((tmp['field'], tmp['halo']))

    return pos, sdelta1, q

def calc_ind_slab(Nmesh, xmin, xmax):
    # indices in x direction of the grid points that a slab contributes to
    indmin = int(xmin/(boxsize/Nmesh)%Nmesh)
    indmax = int(xmax/(boxsize/Nmesh)%Nmesh)+1
    if indmax < indmin:
        indmax += Nmesh
    ind_slab = np.arange(indmin, indmax+1e-6, 1, dtype=int)%Nmesh
    return ind_slab

def load_and_process_particles(islab, sim, z, Rf, Nmesh, sigma_sdelta1, mean_q, sigma_q, qname):
    '''load in particles, discard unuseful ones, sort according to density'''
    # load in one slab
    pos, sdelta1, q = load_particles(islab, sim, z, Rf, qname)
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

    # whether to use nabla2d1 or G2
    if qname in ['nabla2d1', 'G2']:
        q = (q-mean_q)/sigma_q
        q = q[ii_d1][ind]

    # indices in x direction of the grid points that are within xmin xmax
    if islab != Nfiles-1:
        ind_slab = calc_ind_slab(Nmesh, pos[:,0].min(), pos[:,0].max())
    else:
        ii = pos[:,0]>0.
        ind_slab = np.concatenate((calc_ind_slab(Nmesh, pos[ii,0].min(), boxsize/2.)[:-2],
            calc_ind_slab(Nmesh, -boxsize/2.+1e-3, pos[~ii,0].max())))

    return pos, sdelta1, q, arr, ind_slab

def calc_A(pos, q, arr, ind_slab, Nmesh, qname):
    '''calculate A with nbodykit'''
    ncell = Nmesh**2 * len(ind_slab)
    A = np.zeros((nbins, ncell))

    for m in range(nbins_d1):
        if arr[m] == arr[m+1]: # empty bin
            continue
        if qname in ['nabla2d1', 'G2']:
            q_ = q[arr[m]:arr[m+1]]
            bin_edges_q = np.percentile(q_, bin_edges_q_percentile)
        else:
            q_ = np.zeros(arr[m+1]-arr[m], dtype=np.float32)
            bin_edges_q = np.linspace(-100, 100, nbins_q+1)
        for n in range(nbins_q):
            ii = (q_ >= bin_edges_q[n]) & (q_ < bin_edges_q[n+1])
            if ii.sum() == 0: # empty bin
                continue
            Am = ArrayCatalog({'Position': pos[arr[m]:arr[m+1]][ii]}).to_mesh(
                Nmesh=Nmesh,
                BoxSize=boxsize,
                resampler=interp_method
                ).compute()
            Am *= ii.sum()/Am.size # need to scale it back
            A[m*nbins_q+n] = Am[ind_slab].reshape(-1)

    return A

def main():
    sim, z, Rf, Nmesh, qname = sys.argv[1:6]
    # redshift, smoothing scale, mesh size
    z = float(z)
    Rf = float(Rf)
    Nmesh = int(Nmesh)
    if len(sys.argv) > 6:
        islabs = sys.argv[6:]
    else:
        islabs = np.linspace(0,Nfiles-1,Nfiles,dtype=int)

    global nbins_q
    global nbins
    global bin_edges_q_percentile
    nbins_q = 1
    folder = 'matrixA'
    if qname in ['nabla2d1', 'G2']:
        if qname == 'nabla2d1':
            bin_edges_q_percentile = bin_edges_nabla2d1_percentile
        else:
            bin_edges_q_percentile = bin_edges_G2_percentile
        nbins_q = len(bin_edges_q_percentile)-1
        folder += '_'+qname
    # total number of bins
    nbins = nbins_d1*nbins_q

    ic_path = '/mnt/store2/xwu/AbacusSummit/%s/ic_%d/' % (sim, N)
    # load in the smoothed delta_1 and calculate std
    tmp = np.load(ic_path+'/sdelta1_Rf%.3g.npy' % Rf)
    sigma_sdelta1 = np.std(tmp)
    del tmp
    # load in nabla^2 delta_1 or G2 and calculate mean and std
    mean_q = None
    sigma_q = None
    if qname in ['nabla2d1', 'G2']:
        tmp = np.load(ic_path+'/%s_Rf%.3g.npy' % (qname, Rf))
        mean_q = np.mean(tmp)
        sigma_q = np.std(tmp)
        del tmp

    outpath = '/mnt/store2/xwu/AbacusSummit/%s/z%s_tilde_operators_nbody/Rf%.3g/' % (sim, str(z), Rf) + folder
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    for islab in islabs:
        islab = int(islab)
        # load in particles
        pos, sdelta1, q, arr, ind_slab = load_and_process_particles(islab, sim, z, Rf, Nmesh, sigma_sdelta1, mean_q, sigma_q, qname)
        # calculate A
        A = calc_A(pos, q, arr, ind_slab, Nmesh, qname)
        # save to disk
        np.savez_compressed(outpath + '/matrixA_slab%d_Nmesh%d_%s' % (islab, Nmesh, interp_method), A=A, ind_slab=ind_slab)

if __name__ == '__main__':
    main()

