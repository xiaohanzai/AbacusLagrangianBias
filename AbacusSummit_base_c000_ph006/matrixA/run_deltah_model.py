'''
Somehow, the parallel version of this code doesn't work...
'''
import numpy as np
#from mpi4py import MPI
from nbodykit.lab import *
from nbodykit import setup_logging
#from nbodykit import CurrentMPIComm
import matplotlib.pyplot as plt
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import sys
import os
from scipy.interpolate import interp2d, griddata
from run_matrixA import load_and_process_particles, bin_edges_d1, bin_edges_nabla2d1_percentile, bin_edges_G2_percentile

# some important parameters
boxsize = 2000.
Nfiles = 34
N = 1152
N0 = 6912
interp_method = 'cic'

tab = {0: 150, 1: 500, 2: 1000, -1: 3000, 3: 3000} # column of f file -> Nthres

def main():
    sim, z, Rf, Nmesh, qname, Nmesh_, col, kmax = sys.argv[1:9]
    # redshift, smoothing scale, mesh size
    z = float(z)
    Rf = float(Rf)
    Nmesh = int(Nmesh)
    Nmesh_ = int(Nmesh_)
    # which columns of the file to work on and what Nthres it corresponds to
    col = int(col)
    Nthres = tab[col]
    kmax = float(kmax)
    if kmax >= 50.:
        kmax = None
    try:
        sim2use = sys.argv[9] # use just one f solution; input the name of that sim
    except:
        sim2use = '' # will average over multiple f

    # bins
    nbins_d1 = len(bin_edges_d1)-1
    global nbins_q
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

    # f solution
    if qname in ['nabla2d1', 'G2']:
        fname = 'f(delta1,%s)_z%s_Rf%.3g_Nmesh%d' % (qname,str(z),Rf,Nmesh)
    else:
        fname = 'f(delta1)_z%s_Rf%.3g_Nmesh%d' % (str(z),Rf,Nmesh)
    if kmax:
        fname += '_kmax%.2f' % kmax
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
        print('using solution averaged over %d sims' % n)
        mean_f_delta1 /= n
        mean_f_delta1_qp /= n
    else:
        if 'small' in sim2use:
            sim2use = 'small/'+sim2use
        tmp = '../../%s/solutions/' % sim2use + fname
        mean_f_delta1 = np.loadtxt(tmp + '.txt')[:,col]
        mean_f_delta1_qp = np.loadtxt(tmp + '_qp.txt')[:,col]
        print('using solution: ' + tmp)

    # calculate sigma_d1 and sigma_q
    ic_path = '/mnt/store2/xwu/AbacusSummit/AbacusSummit_base_c000_ph006/ic_%d/' % N
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

    outpath = '/mnt/store2/xwu/AbacusSummit/%s/z%s_tilde_operators_nbody/Rf%.3g/' % (sim, str(z), Rf) + folder + '/' + sim2use
    if not os.path.exists(outpath):
        os.system('mkdir -p '+outpath)

    # create and run mesh
    mesh = ArrayMesh(np.zeros((Nmesh_,Nmesh_,Nmesh_), dtype=np.float32), BoxSize=boxsize).compute()
    mesh_qp = ArrayMesh(np.zeros((Nmesh_,Nmesh_,Nmesh_), dtype=np.float32), BoxSize=boxsize).compute()
    for i in range(Nfiles):
        pos, sdelta1, q, arr, _ = load_and_process_particles(i, sim, z, Rf, Nmesh, sigma_sdelta1, mean_q, sigma_q, qname)

        #f_delta1_interp = griddata(bins, f_delta1, (sdelta1,q), fill_value=0.)
        #f_delta1_interp_qp = griddata(bins, f_delta1_qp, (sdelta1,nabla2d1), fill_value=0.)

        f_delta1_interp = np.zeros(len(pos))
        f_delta1_interp_qp = np.zeros(len(pos))
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
                f_delta1_interp[arr[m]:arr[m+1]][ii] = mean_f_delta1[m*nbins_q+n]
                f_delta1_interp_qp[arr[m]:arr[m+1]][ii] = mean_f_delta1_qp[m*nbins_q+n]

        # convert to mesh
        fac = len(pos)/(0.03*N0**3)
        mesh += ArrayCatalog({'Position': pos, 'Value': f_delta1_interp}).to_mesh(Nmesh=Nmesh_, resampler=interp_method, BoxSize=boxsize).compute()*fac
        mesh_qp += ArrayCatalog({'Position': pos, 'Value': f_delta1_interp_qp}).to_mesh(Nmesh=Nmesh_, resampler=interp_method, BoxSize=boxsize).compute()*fac

    # calculate mesh and save
    fname = '/deltah_model_Nthres%d_Nmesh%d_%d_%s' % (Nthres, Nmesh, Nmesh_, interp_method)
    if kmax:
        fname += '_kmax%.2f' % kmax
    FieldMesh(mesh).save(outpath+fname+'.bigfile')
    FieldMesh(mesh_qp).save(outpath+fname+'_qp.bigfile')

if __name__ == '__main__':
    main()

