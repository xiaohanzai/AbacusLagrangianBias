import numpy as np
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import sys
import warnings
warnings.filterwarnings("ignore")
import pyfftw
import os

sim, z, Rf = sys.argv[1:4] # e.g. AbacusSummit_base_c000_ph006
z = float(z)
Rf = float(Rf)

boxsize = 2000.
N0 = 6912
N = 1152
Nfiles = 34
if 'small' in sim:
    boxsize = 500.
    N0 = 1728
    N = 576
    Nfiles = 1
    sim = 'small/'+sim

qs = ['sdelta1', 'G2', 'nabla2d1']
if len(sys.argv) > 4:
    qs = sys.argv[4:]
print('writing out quantities: ', qs)

outpath = '/mnt/store2/xwu/AbacusSummit/%s/z%s_tilde_operators_nbody/' % (sim,str(z))
if not os.path.exists(outpath):
    os.system('mkdir ' + outpath)
outpath = outpath + '/Rf%.3g/' % Rf
if not os.path.exists(outpath):
    os.system('mkdir ' + outpath)

# load in IC grid
ic_path = '/mnt/store2/xwu/AbacusSummit/%s/ic_%d/' % (sim,N)
if 'sdelta1' in qs:
    sdelta1 = np.load(ic_path+'/sdelta1_Rf%.3g.npy' % Rf)
if 'G2' in qs:
    G2 = np.load(ic_path+'/G2_Rf%.3g.npy' % Rf)
if 'nabla2d1' in qs:
    nabla2d1 = np.load(ic_path+'/nabla2d1_Rf%.3g.npy' % Rf)

# load catalogs
for i in range(Nfiles):
    cat_path = '/mnt/store2/bigsims/AbacusSummit/%s/halos/z%.3f/halo_info/halo_info_%03d.asdf' % (sim,z,i)
    # load halo particles
    cat = CompaSOHaloCatalog(cat_path, fields=[], load_subsamples='A_halo_all', unpack_bits='lagr_idx')
    lagr_idx_h = np.int16(np.round(cat.subsamples['lagr_idx']*np.float32(N/N0)))%N
    # load field particles
    cat = CompaSOHaloCatalog(cat_path, fields=[], load_subsamples='A_field_all', unpack_bits='lagr_idx')
    lagr_idx_f = np.int16(np.round(cat.subsamples['lagr_idx']*np.float32(N/N0)))%N
    del cat

    # write to disk
    if 'sdelta1' in qs:
        np.savez_compressed(outpath+'/slab%d_sdelta1' % i, halo=sdelta1[(lagr_idx_h[:,0], lagr_idx_h[:,1], lagr_idx_h[:,2])], field=sdelta1[(lagr_idx_f[:,0], lagr_idx_f[:,1], lagr_idx_f[:,2])])
    if 'G2' in qs:
        np.savez_compressed(outpath+'/slab%d_G2' % i, halo=G2[(lagr_idx_h[:,0], lagr_idx_h[:,1], lagr_idx_h[:,2])], field=G2[(lagr_idx_f[:,0], lagr_idx_f[:,1], lagr_idx_f[:,2])])
    if 'nabla2d1' in qs:
        np.savez_compressed(outpath+'/slab%d_nabla2d1' % i, halo=nabla2d1[(lagr_idx_h[:,0], lagr_idx_h[:,1], lagr_idx_h[:,2])], field=nabla2d1[(lagr_idx_f[:,0], lagr_idx_f[:,1], lagr_idx_f[:,2])])

