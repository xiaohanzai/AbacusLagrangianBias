import numpy as np
from nbodykit.lab import *
from nbodykit import setup_logging, style
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')
sys.path.append('../../')
from cosmology_functions import *
from save_matrixA_nooverlap import load_matrixA_slab
import os

interp_method = 'cic'

# parameters about the sim
boxsize = 2000.
k0 = 2*np.pi/boxsize
N0 = 6912
N = 1152
Nfiles = 34

def calc_coadded_qs(path, Nmesh, interp_method='cic', remove_overlaps=(False, False),
                   direct_load=True, delta_hs=None, calc_M=True, **kwargs):
    '''
    Load A for each slab and calculate the coadded M and/or b.
    '''
    M = None
    bs = None
    for i in range(Nfiles):
        A, ind_slab = load_matrixA_slab(i, path, Nmesh, interp_method,
                            remove_overlaps=remove_overlaps, direct_load=direct_load, **kwargs)
        if calc_M:
            if i == 0:
                M = np.dot(A, A.T)
            else:
                M += np.dot(A, A.T)
        if delta_hs is not None:
            if i == 0:
                if type(delta_hs) != list:
                    delta_hs = [delta_hs]
                bs = [None]*len(delta_hs)
                for j in range(len(delta_hs)):
                    bs[j] = np.dot(A, delta_hs[j][ind_slab].reshape(-1)+1.)
            else:
                for j in range(len(delta_hs)):
                    bs[j] += np.dot(A, delta_hs[j][ind_slab].reshape(-1)+1.)
    return M, bs

def calc_reduced_M(M):
    # take care of empty bins
    ii_empty = np.diag(M) == 0.
    # calculate the reduced matrix
    D = np.zeros_like(M)
    for i in range(len(M)):
        if ii_empty[i]:
            D[i,i] = 1.
            M[i,i] = 1.
        else:
            D[i,i] = 1/M[i,i]**0.5
    M_ = np.dot( D, np.dot(M, D) )
    return M_, D, ii_empty

def main():
    sim, z, Rf, Nmesh, qname = sys.argv[1:6]
    Nthress = sys.argv[6:]
    z = float(z)
    Rf = float(Rf)
    Nmesh = int(Nmesh)
    for i in range(len(Nthress)):
        Nthress[i] = int(Nthress[i])

    folder = 'matrixA_nabla2d1'
    if qname == 'G2':
        folder = 'matrixA_G2'
    path = '/mnt/store2/xwu/AbacusSummit/%s/z%s_tilde_operators_nbody/Rf%.3g/' % (sim, str(z), Rf) + folder
    outpath = path+'/nooverlap'
        
    # load in snapshot
    cat = CompaSOHaloCatalog('/mnt/store2/bigsims/AbacusSummit/%s/halos/z%.3f/' % (sim, z),
                         fields=['N', 'x_com', 'r100_com'])

    # calculate the delta_h vectors
    delta_hs = [None]*len(Nthress)
    for i in range(len(Nthress)):
        Nthres = Nthress[i]
        ii_h = (cat.halos['N'] > Nthres)
        delta_h = ArrayCatalog({'Position': cat.halos[ii_h]['x_com'], 'Value': cat.halos[ii_h]['N']}).to_mesh(Nmesh=Nmesh, BoxSize=boxsize, resampler=interp_method).compute()
        delta_h = delta_h/np.mean(delta_h)-1.
        delta_hs[i] = delta_h

    nparticles = 0.03*N0**3
    ncell = Nmesh**3

    # load in matrices and calculate solutions
    if qname == 'delta1':
        M, bs = calc_coadded_qs(outpath, Nmesh, interp_method,
                sum_nabla2d1_bins=True, nbins_nabla2d1=5,
                direct_load=True, calc_M=True, delta_hs=delta_hs)
    else:
        M, bs = calc_coadded_qs(outpath, Nmesh, interp_method,
                direct_load=True, calc_M=True, delta_hs=delta_hs)
    M_, D, ii_empty = calc_reduced_M(M)

    f_delta1s = [None]*len(Nthress)
    for i in range(len(Nthress)):
        b = bs[i]
        f_delta1 = np.dot(D, np.dot(np.linalg.inv(M_), b/np.diag(M)**0.5)) * nparticles/ncell
        f_delta1s[i] = f_delta1

    # write to disk
    writepath = '../../'+sim
    if not os.path.exists(writepath):
        os.system('mkdir ' + writepath)
    if qname == 'delta1':
        fname = 'f(delta1)_z%s_Rf%.3g_Nmesh%d.txt' % (str(z), Rf, Nmesh)
    else:
        fname = 'f(delta1,%s)_z%s_Rf%.3g_Nmesh%d.txt' % (qname, str(z), Rf, Nmesh)
    with open(writepath+'/'+fname, 'w') as f:
        f.write('# N>')
        for Nthres in Nthress:
            f.write('%d  ' % Nthres)
        f.write('\n')
        for m in range(len(f_delta1s[0])):
            for j in range(len(Nthress)):
                f.write('%.3g  ' % f_delta1s[j][m])
            f.write('\n')

if __name__ == '__main__':
    main()

