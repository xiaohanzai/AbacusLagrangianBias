import numpy as np
from mpi4py import MPI
from nbodykit.lab import *
from nbodykit import setup_logging
from nbodykit import CurrentMPIComm
import matplotlib.pyplot as plt
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import sys

z = float(sys.argv[1])
Nmesh = int(sys.argv[2])

boxsize = 2000.
k0 = 2*np.pi/boxsize
N0 = 6912
N = 1728
Nfiles = 34

comm = MPI.COMM_WORLD #CurrentMPIComm.get()
rank = comm.rank
size = comm.size

if Nfiles%size < 1e-4:
    Nslabs = int(Nfiles/size)
else:
    Nslabs = int(Nfiles/size)+1

istart = Nslabs*rank
cat = CompaSOHaloCatalog(
    '/mnt/store2/bigsims/AbacusSummit/AbacusSummit_base_c000_ph006/halos/z%.3f/halo_info/halo_info_%03d.asdf' % (z, istart),
    fields=[], load_subsamples='A_rv')
pos = cat.subsamples['pos']
del cat

for i in range(istart+1, min(Nfiles, istart+Nslabs)):
    cat = CompaSOHaloCatalog(
        '/mnt/store2/bigsims/AbacusSummit/AbacusSummit_base_c000_ph006/halos/z%.3f/halo_info/halo_info_%03d.asdf' % (z, i),
        fields=[], load_subsamples='A_rv')
    pos = np.concatenate((pos, cat.subsamples['pos']), axis=0)
    del cat

# convert to mesh
catp = ArrayCatalog({'Position': pos})
print(rank, pos.shape, catp['Position'].shape, catp.size, catp.csize)
mesh = catp.to_mesh(Nmesh=Nmesh, resampler='cic',#compensated=True, resampler='tsc', interlaced=True,
    BoxSize=boxsize)

# calculate mesh and save
mesh.save('/mnt/store2/xwu/AbacusSummit/base_c000_ph006/z%s_tilde_operators_nbody/deltaz_Nmesh%d_cic.bigfile' % (str(z), Nmesh))

