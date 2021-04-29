'''
Somehow, the parallel version of this code doesn't work...
'''
import numpy as np
from mpi4py import MPI
from nbodykit.lab import *
from nbodykit import setup_logging
from nbodykit import CurrentMPIComm
import matplotlib.pyplot as plt
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import sys
from scipy.interpolate import interp2d, griddata

use_nabla2d1 = True

sim, z, Rf, Nmesh, col, Nthres = sys.argv[1:]
# redshift, smoothing scale, mesh size
z = float(z)
Rf = float(Rf)
Nmesh = int(Nmesh)
# which columns of the file to work on and what Nthres it corresponds to
col = int(col)
Nthres = int(Nthres)

# some important parameters
boxsize = 2000.
Nfiles = 34
N = 1152
N0 = 6912
interp_method = 'cic'

# calculate sigma_d1 and sigma_nabla2d1
ic_path = '/mnt/store2/xwu/AbacusSummit/AbacusSummit_base_c000_ph006/ic_%d/' % N
# load in the smoothed delta_1 and calculate std
tmp = np.load(ic_path+'/sdelta1_Rf%.3g.npy' % Rf)
sigma_sdelta1 = np.std(tmp)
del tmp
# load in nabla^2 delta_1 and calculate std
sigma_nabla2d1 = None
if use_nabla2d1:
    tmp = np.load(ic_path+'/nabla2d1_Rf%.3g.npy' % Rf)
    sigma_nabla2d1 = np.std(tmp)
    del tmp

# read in the f(delta1)-delta1 relation
if not use_nabla2d1:
    data = np.loadtxt('f(delta1)_z%s_Rf%.3g_Nmesh400.txt' % (str(z), Rf))
    ii = ~np.isnan(data[:,0])
else:
    data = np.loadtxt('f(delta1,nabla2d1)_z%s_Rf%.3g_Nmesh400.txt' % (str(z), Rf))
    ii = ~np.isnan(data[:,0]) & ~np.isnan(data[:,1])
#ii = data[:,0] < 3.8 # discard the noisy region
f_delta1 = data[ii,col]

comm = MPI.COMM_WORLD #CurrentMPIComm.get()
rank = comm.rank
size = comm.size

if Nfiles%size < 1e-4:
    Nslabs = int(Nfiles/size)
else:
    Nslabs = int(Nfiles/size)+1

istart = Nslabs*rank
mesh = ArrayMesh(np.zeros((Nmesh,Nmesh,Nmesh), dtype=np.float32), BoxSize=boxsize).compute()
for i in range(istart, min(Nfiles, istart+Nslabs)):
    sdelta1_ = np.load('/mnt/store2/xwu/AbacusSummit/%s/z%s_tilde_operators_nbody/Rf%.3g/slab%d_sdelta1.npz' % (sim,str(z),Rf,i))
    nabla2d1_ = np.load('/mnt/store2/xwu/AbacusSummit/%s/z%s_tilde_operators_nbody/Rf%.3g/slab%d_nabla2d1.npz' % (sim,str(z),Rf,i))

    for name in ['halo', 'field']:
        cat = CompaSOHaloCatalog(
            '/mnt/store2/bigsims/AbacusSummit/%s/halos/z%.3f/halo_info/halo_info_%03d.asdf' % (sim, z, i),
            fields=[], load_subsamples='A_%s_rv' % name)
        pos = cat.subsamples['pos']
        del cat

        sdelta1 = sdelta1_[name]/sigma_sdelta1
        if not use_nabla2d1:
            # interpolate to get the predicted halo field
            f_delta1_interp = np.interp(sdelta1, data[ii,0], f_delta1, left=0., right=0.)
        else:
            nabla2d1 = nabla2d1_[name]/sigma_nabla2d1
            f_delta1_interp = griddata(data[ii,:2], f_delta1, (sdelta1,nabla2d1), fill_value=0.)
            #func = interp2d(d1_bins, nabla2d1_bins, f_delta1, fill_value=0.)
            #f_delta1_interp = np.zeros(len(pos))
            #for j in range(len(pos)):
            #    f_delta1_interp[j] = func(sdelta1[j], nabla2d1[j])

        # convert to mesh
        fac = len(pos)/(0.03*N0**3)
        mesh += ArrayCatalog({'Position': pos, 'Value': f_delta1_interp}).to_mesh(Nmesh=Nmesh, resampler='cic', BoxSize=boxsize).compute()*fac

# calculate mesh and save
FieldMesh(mesh).save('/mnt/store2/xwu/AbacusSummit/%s/z%s_tilde_operators_nbody/Rf%.3g/deltah_model_Nthres%d_Nmesh%d_cic.bigfile' % (sim, str(z), Rf, Nthres, Nmesh))

