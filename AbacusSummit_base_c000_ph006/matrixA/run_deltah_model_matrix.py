'''
Somehow, the parallel version of this code doesn't work...
'''
import numpy as np
from nbodykit.lab import *
from nbodykit import setup_logging
import matplotlib.pyplot as plt
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import sys
import os
from scipy.interpolate import interp2d, griddata
from run_matrixA import load_and_process_particles, bin_edges_d1, bin_edges_nabla2d1_percentile, bin_edges_G2_percentile
from save_matrixA_nooverlap import load_matrixA_slab

# some important parameters
boxsize = 2000.
Nfiles = 34
N = 1152
N0 = 6912
interp_method = 'cic'

tab = {0: 150, 1: 500, 2: 1000, -1: 3000, 3: 3000} # column of f file -> Nthres

def main():
    sim, z, Rf, Nmesh, qname, Nmesh_, col = sys.argv[1:8]
    # redshift, smoothing scale, mesh size
    z = float(z)
    Rf = float(Rf)
    Nmesh = int(Nmesh)
    Nmesh_ = int(Nmesh_)
    # which columns of the file to work on and what Nthres it corresponds to
    col = int(col)
    Nthres = tab[col]
    try:
        sim2use = sys.argv[8] # use just one f solution; input the name of that sim
    except:
        sim2use = '' # will average over multiple f

    # bins
    nbins_d1 = len(bin_edges_d1)-1
    bin_edges_q_percentile = None
    nbins_q = 1
    folder = 'matrixA'
    if qname in ['nabla2d1', 'G2']:
        if qname == 'nabla2d1':
            bin_edges_q_percentile = bin_edges_nabla2d1_percentile
        else:
            bin_edges_q_percentile = bin_edges_G2_percentile
        nbins_q = len(bin_edges_q_percentile)-1
        folder += '_'+qname

    # f solution
    if qname in ['nabla2d1', 'G2']:
        fname = 'f(delta1,%s)_z%s_Rf%.3g_Nmesh%d' % (qname,str(z),Rf,Nmesh)
    else:
        fname = 'f(delta1)_z%s_Rf%.3g_Nmesh%d' % (str(z),Rf,Nmesh)
    if not sim2use:
        mean_f_delta1 = np.zeros(nbins_d1*nbins_q)
        mean_f_delta1_qp = np.zeros(nbins_d1*nbins_q)
        n = 0
        for i in range(6):
            try:
                f_delta1 = np.loadtxt(
                    '../../AbacusSummit_base_c000_ph00%d/solutions/' % i + fname + '.txt')[:,col]
                mean_f_delta1 += f_delta1
                f_delta1 = np.loadtxt(
                    '../../AbacusSummit_base_c000_ph00%d/solutions/' % i + fname + '_qp.txt')[:,col]
                mean_f_delta1_qp += f_delta1
                n += 1
            except:
                continue
        mean_f_delta1 /= n
        mean_f_delta1_qp /= n
    else:
        if 'small' in sim2use:
            sim2use = 'small/'+sim2use
        mean_f_delta1 = np.loadtxt('../../%s/solutions/' % sim2use + fname + '.txt')[:,col]
        mean_f_delta1_qp = np.loadtxt('../../%s/solutions/' % sim2use + fname + '_qp.txt')[:,col]

    path = '/mnt/store2/xwu/AbacusSummit/%s/z%s_tilde_operators_nbody/Rf%.3g/' % (sim, str(z), Rf) + folder
    if qname in ['nabla2d1', 'G2']:
        outpath = path + '/nooverlap'
    else:
        outpath = path + '_nabla2d1/nooverlap'

    # create and run mesh
    mesh = np.zeros((Nmesh_,Nmesh_,Nmesh_))
    mesh_qp = np.zeros((Nmesh_,Nmesh_,Nmesh_))
    for i in range(Nfiles):
        if qname not in ['nabla2d1', 'G2']:
            A, ind_slab = load_matrixA_slab(i, outpath, Nmesh_, interp_method, direct_load=True,
                    sum_nabla2d1_bins=True, nbins_nabla2d1=len(bin_edges_nabla2d1_percentile)-1)
        else:
            A, ind_slab = load_matrixA_slab(i, outpath, Nmesh_, interp_method, direct_load=True)
        mesh[ind_slab] += np.dot(mean_f_delta1, A).reshape(-1,Nmesh_,Nmesh_)
        mesh_qp[ind_slab] += np.dot(mean_f_delta1_qp, A).reshape(-1,Nmesh_,Nmesh_)
    nparticles = 0.03*N0**3
    mesh = mesh/(nparticles/Nmesh_**3)-1.
    mesh_qp = mesh_qp/(nparticles/Nmesh_**3)-1.

    # save to disk
    outpath = path+'/'+sim2use
    if not os.path.exists(outpath):
        os.system('mkdir -p '+outpath)
    np.save(outpath+'/deltah_model_Nthres%d_Nmesh%d_%d_%s' % (Nthres, Nmesh, Nmesh_, interp_method), mesh)
    np.save(outpath+'/deltah_model_Nthres%d_Nmesh%d_%d_%s_qp' % (Nthres, Nmesh, Nmesh_, interp_method), mesh_qp)

if __name__ == '__main__':
    main()

