import numpy as np
#from mpi4py import MPI
from nbodykit.lab import *
from nbodykit import setup_logging
from nbodykit import CurrentMPIComm
import matplotlib.pyplot as plt
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import sys

boxsize = 2000.
k0 = 2*np.pi/boxsize
N0 = 6912
N = 1728
Nfiles = 34

cellsize = 5. # Mpc/h, cell size in final grid
Nmesh = int(boxsize/cellsize)
Rf = cellsize/(2*np.pi)**0.5 # Gaussian smoothing

z = float(sys.argv[1])
q = sys.argv[2]
if len(sys.argv) > 3:
    Rf = float(sys.argv[3])
    if len(sys.argv) > 4:
        Nmesh = int(sys.argv[4])

print('advecting: ', q)
qname = q
if qname == 'sdelta2':
    q = 'sdelta1'

mesh = ArrayMesh(np.zeros((Nmesh,Nmesh,Nmesh), dtype=np.float32), BoxSize=boxsize).compute()
for i in range(Nfiles):
    # particle positions
    cat = CompaSOHaloCatalog(
        '/mnt/store2/bigsims/AbacusSummit/AbacusSummit_base_c000_ph006/halos/z%.3f/halo_info/halo_info_%03d.asdf' % (z, i),
        fields=[], load_subsamples='A_halo_rv')
    pos = cat.subsamples['pos']
    cat = CompaSOHaloCatalog(
        '/mnt/store2/bigsims/AbacusSummit/AbacusSummit_base_c000_ph006/halos/z%.3f/halo_info/halo_info_%03d.asdf' % (z, i),
        fields=[], load_subsamples='A_field_rv')
    pos = np.concatenate((pos, cat.subsamples['pos']))
    del cat

    # load in quantities to advect
    with np.load('/mnt/store2/xwu/AbacusSummit/base_c000_ph006/z%s_tilde_operators_nbody/Rf%.3g/slab%d_%s.npz' % (str(z),Rf,i,q)) as tmp:
        q_ = np.concatenate((tmp['halo'], tmp['field']))
    if qname == 'sdelta2':
        q_ = q_**2

    # convert to mesh
    fac = len(pos)/(0.03*N0**3)
    mesh += ArrayCatalog({'Position': pos, 'Value': q_}).to_mesh(Nmesh=Nmesh, resampler='cic', BoxSize=boxsize).compute()*fac

# calculate mesh and save
FieldMesh(mesh).save('/mnt/store2/xwu/AbacusSummit/base_c000_ph006/z%s_tilde_operators_nbody/Rf%.3g/t%s_Nmesh%d_cic.bigfile' % (str(z), Rf, qname, Nmesh))

