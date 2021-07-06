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
from qpsolvers import solve_qp
from scipy.stats import norm

interp_method = 'cic'

def calc_coadded_qs(path, Nmesh, interp_method='cic', remove_overlaps=(False, False),
                   direct_load=True, delta_hs=None, calc_M=True, **kwargs):
    '''
    Load A for each slab and calculate the coadded M and/or b.
    '''
    M = None
    bs = None
    Nfiles = 34
    if 'small' in path:
        Nfiles = 1
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

def calc_allbins_pdf(qname):
    bin_edges_d1 = np.linspace(-4, 5, 40+1)
    allbins_pdf = norm.cdf(bin_edges_d1[1:]) - norm.cdf(bin_edges_d1[:-1])
    if qname == 'nabla2d1':
        bin_edges_q_percentile = np.array([0, 5, 35, 65, 95, 100])
    elif qname == 'G2':
        bin_edges_q_percentile = np.array([0, 10, 30, 70, 90, 100])
    else:
        return allbins_pdf
    allbins_pdf = allbins_pdf.reshape(-1,1) * (bin_edges_q_percentile[1:] - bin_edges_q_percentile[:-1]).reshape(1,-1)
    allbins_pdf /= allbins_pdf.sum()
    return allbins_pdf.reshape(-1)

def main():
    sim, z, Rf, Nmesh, qname = sys.argv[1:6]
    Nthress = sys.argv[6:]
    z = float(z)
    Rf = float(Rf)
    Nmesh = int(Nmesh)
    for i in range(len(Nthress)):
        Nthress[i] = int(Nthress[i])

    global boxsize
    global N0
    boxsize = 2000.
    N0 = 6912
    if 'small' in sim:
        boxsize = 500.
        N0 = 1728
        sim = 'small/'+sim

    folder = 'matrixA_nabla2d1'
    if qname == 'G2':
        folder = 'matrixA_G2'
    outpath = '/mnt/store2/xwu/AbacusSummit/%s/z%s_tilde_operators_nbody/Rf%.3g/' % (sim, str(z), Rf) + folder
    if 'small' not in sim:
        outpath += '/nooverlap'
        
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

    # no constraint
    f_delta1s = [np.dot(D, np.dot(np.linalg.inv(M_), b/np.diag(M)**0.5)) * nparticles/ncell for b in bs]
    # qp with non negative constraint and \int = 1 constraint
    allbins_pdf = calc_allbins_pdf(qname)
    f_delta1s_qp = [solve_qp(M, -b.reshape(len(M),-1), G=-np.identity(len(M)), h=np.zeros(len(M)),
                A=allbins_pdf, b=np.array([ncell/nparticles]), solver='cvxopt') \
                * nparticles/ncell for b in bs]

    # write to disk
    writepath = '../../'+sim+'/solutions'
    if not os.path.exists(writepath):
        os.system('mkdir -p ' + writepath)
    if qname == 'delta1':
        fname = 'f(delta1)_z%s_Rf%.3g_Nmesh%d.txt' % (str(z), Rf, Nmesh)
        fname_qp = 'f(delta1)_z%s_Rf%.3g_Nmesh%d_qp.txt' % (str(z), Rf, Nmesh)
    else:
        fname = 'f(delta1,%s)_z%s_Rf%.3g_Nmesh%d.txt' % (qname, str(z), Rf, Nmesh)
        fname_qp = 'f(delta1,%s)_z%s_Rf%.3g_Nmesh%d_qp.txt' % (qname, str(z), Rf, Nmesh)
    for fname_, f_delta1s_ in zip([fname, fname_qp], [f_delta1s, f_delta1s_qp]):
        with open(writepath+'/'+fname_, 'w') as f:
            f.write('# N>')
            for Nthres in Nthress:
                f.write('%d  ' % Nthres)
            f.write('\n')
            for m in range(len(f_delta1s_[0])):
                for j in range(len(Nthress)):
                    f.write('%.3g  ' % f_delta1s_[j][m])
                f.write('\n')

if __name__ == '__main__':
    main()

