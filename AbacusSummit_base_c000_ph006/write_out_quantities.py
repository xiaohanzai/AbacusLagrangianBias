import numpy as np
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import sys
import warnings
warnings.filterwarnings("ignore")
import pyfftw
import os

boxsize = 2000.
k0 = 2*np.pi/boxsize
N0 = 6912
N = 1728
Nfiles = 34

z = float(sys.argv[1])
if len(sys.argv) > 2:
    Rf = float(sys.argv[2])
    if len(sys.argv) > 3:
        qs = sys.argv[3:]
else:
    qs = ['sdelta1', 'G2', 'nabla2d1']
    cellsize = 5. # Mpc/h, cell size in final grid
    Rf = cellsize/(2*np.pi)**0.5 # Gaussian smoothing
print('writing out quantities: ', qs)

outpath = '/mnt/store2/xwu/AbacusSummit/base_c000_ph006/z%s_tilde_operators_nbody/Rf%.3g' % (str(z),Rf)
if not os.path.exists(outpath):
    os.system('mkdir ' + outpath)

# load in IC grid
if 'sdelta1' in qs:
    sdelta1 = np.load('/mnt/store2/xwu/AbacusSummit/AbacusSummit_base_c000_ph006/ic/sdelta1_Rf%.3g.npy' % Rf)
if 'G2' in qs:
    G2 = np.load('/mnt/store2/xwu/AbacusSummit/AbacusSummit_base_c000_ph006/ic/G2_Rf%.3g.npy' % Rf)
if 'nabla2d1' in qs:
    nabla2d1 = np.load('/mnt/store2/xwu/AbacusSummit/AbacusSummit_base_c000_ph006/ic/nabla2d1_Rf%.3g.npy' % Rf)

# load catalogs
for i in range(Nfiles):
    # load halo particles
    cat = CompaSOHaloCatalog(
        '/mnt/store2/bigsims/AbacusSummit/AbacusSummit_base_c000_ph006/halos/z%.3f/halo_info/halo_info_%03d.asdf' % (z, i),
        fields=[], load_subsamples='A_halo_all', unpack_bits='lagr_idx')
    lagr_idx_h = np.int16(np.round(cat.subsamples['lagr_idx']*np.float32(N/N0)))%N
    # load field particles
    cat = CompaSOHaloCatalog(
        '/mnt/store2/bigsims/AbacusSummit/AbacusSummit_base_c000_ph006/halos/z%.3f/halo_info/halo_info_%03d.asdf' % (z, i),
        fields=[], load_subsamples='A_field_all', unpack_bits='lagr_idx')
    lagr_idx_f = np.int16(np.round(cat.subsamples['lagr_idx']*np.float32(N/N0)))%N

    del cat

    # write to disk
    if 'sdelta1' in qs:
        np.savez_compressed('/mnt/store2/xwu/AbacusSummit/base_c000_ph006/z%s_tilde_operators_nbody/Rf%.3g/slab%d_sdelta1' % (str(z),Rf,i), halo=sdelta1[(lagr_idx_h[:,0], lagr_idx_h[:,1], lagr_idx_h[:,2])], field=sdelta1[(lagr_idx_f[:,0], lagr_idx_f[:,1], lagr_idx_f[:,2])])
    if 'G2' in qs:
        np.savez_compressed('/mnt/store2/xwu/AbacusSummit/base_c000_ph006/z%s_tilde_operators_nbody/Rf%.3g/slab%d_G2' % (str(z),Rf,i), halo=G2[(lagr_idx_h[:,0], lagr_idx_h[:,1], lagr_idx_h[:,2])], field=G2[(lagr_idx_f[:,0], lagr_idx_f[:,1], lagr_idx_f[:,2])])
    if 'nabla2d1' in qs:
        np.savez_compressed('/mnt/store2/xwu/AbacusSummit/base_c000_ph006/z%s_tilde_operators_nbody/Rf%.3g/slab%d_nabla2d1' % (str(z),Rf,i), halo=nabla2d1[(lagr_idx_h[:,0], lagr_idx_h[:,1], lagr_idx_h[:,2])], field=nabla2d1[(lagr_idx_f[:,0], lagr_idx_f[:,1], lagr_idx_f[:,2])])

