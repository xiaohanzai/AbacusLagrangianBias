import numpy as np
import pyfftw
from mpi4py import MPI
from nbodykit.lab import *
from nbodykit import setup_logging, style
import sys
import os
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import warnings
warnings.filterwarnings("ignore")

z = 0.5

# quantities to calculate: deltaz, tdelta1, tdelta2, tG2
qs = sys.argv[1:]

savepath = 'z%s_Schmittfull' % str(z)
if not os.path.exists(savepath):
    os.mkdir(savepath)

boxsize = 2000.
k0 = 2*np.pi/boxsize
z_init = 99
Nmesh = 400

OMm = (0.02237+0.12)/0.6736**2
OMl = 1 - OMm
def calc_growthrate(z):
    omegaM_z = OMm*(1+z)**3 / ( OMl + OMm*(1+z)**3 )
    dick_z = 2.5*omegaM_z / ( 1.0/70.0 + omegaM_z*(209-omegaM_z)/140.0 + (omegaM_z)**(4.0/7.0) )
    dick_0 = 2.5*OMm / ( 1.0/70.0 + OMm*(209-OMm)/140.0 + (OMm)**(4.0/7.0) )
    return dick_z / (dick_0 * (1.0+z))

# the IC delta's
N = 1728
Nh = N//2
def load_delta1():
    data = np.fromfile('../AbacusSummit_base_c000_ph006/ic/density%d' % N, dtype=np.float32).reshape(N,N,N)
    delta_1 = data*0.
    x = np.linspace(-Nh, Nh-1, N, dtype=int)
    for i in range(N):
        ind = x[i]%N
        delta_1[ind,Nh:,Nh:] = data[i,:Nh,:Nh]
        delta_1[ind,:Nh,:Nh] = data[i,Nh:,Nh:]
        delta_1[ind,:Nh,Nh:] = data[i,Nh:,:Nh]
        delta_1[ind,Nh:,:Nh] = data[i,:Nh,Nh:]
    del data
    return delta_1

# growth rates
Dinit = calc_growthrate(z_init)
Dz = calc_growthrate(z)
Ddiff = Dz - Dinit
Dratio = Dz/Dinit

# load or calculate position of the particles
if not os.path.exists(savepath+'/pos.npy'):
    # the displacement field; not the correct amplitude yet
    if not os.path.exists('../AbacusSummit_base_c000_ph006/ic/psi.npy'):
        delta_1 = load_delta1()
        deltak_1 = pyfftw.interfaces.numpy_fft.rfftn(delta_1)

        kx = np.arange(0,N,1)
        kx[N//2:] -= N
        kx = kx.reshape(-1,1)*k0
        ky = np.arange(0,N,1)
        ky[N//2:] -= N
        ky = ky.reshape(1,-1)*k0

        k2xy = kx**2 + ky**2
        k2xy[0,0] = 1e-6
        psi1x = deltak_1*0.
        psi1y = deltak_1*0.
        psi1z = deltak_1*0.
        for i in range(Nh+1):
            kz = i*k0
            tmp = deltak_1[:,:,i]/(k2xy + kz**2)*(-1)**0.5
            psi1x[:,:,i] = tmp*kx
            psi1y[:,:,i] = tmp*ky
            psi1z[:,:,i] = tmp*kz

        psi1x = pyfftw.interfaces.numpy_fft.irfftn(psi1x)
        psi1y = pyfftw.interfaces.numpy_fft.irfftn(psi1y)
        psi1z = pyfftw.interfaces.numpy_fft.irfftn(psi1z)

        np.save('/mnt/store2/xwu/AbacusSummit/AbacusSummit_base_c000_ph006/ic/psi.npy', [psi1x, psi1y, psi1z])
        del psi1x, psi1y, psi1z, delta_1, deltak_1

    # displace particles; remember to correct the amplitude of delta_1 (and thus psi)
    pos = np.zeros((N**3,3))
    tmp = np.linspace(0., N-1., N)
    tmp[Nh:] -= N
    tmp *= boxsize/N
    pos[:,0] = (tmp.reshape(-1,1,1) + Ddiff/Dinit*np.load('../AbacusSummit_base_c000_ph006/ic/psi.npy', allow_pickle=True)[0]).ravel()
    pos[:,1] = (tmp.reshape(1,-1,1) + Ddiff/Dinit*np.load('../AbacusSummit_base_c000_ph006/ic/psi.npy', allow_pickle=True)[1]).ravel()
    pos[:,2] = (tmp.reshape(1,1,-1) + Ddiff/Dinit*np.load('../AbacusSummit_base_c000_ph006/ic/psi.npy', allow_pickle=True)[2]).ravel()
    pos[pos > boxsize/2.] -= boxsize
    pos[pos < -boxsize/2.] += boxsize

    np.save(savepath+'/pos.npy', pos)
    del pos

# zeldovich density field
if 'deltaz' in qs:
    print('calculating deltaz...')
    pos = np.load(savepath+'/pos.npy')
    tmp = ArrayCatalog({'Position': pos})
    f_deltaz = tmp.to_mesh(Nmesh=Nmesh, BoxSize=boxsize, resampler='cic')

    f_deltaz.save(savepath+'/deltaz_Nmesh%d_cic.bigfile' % Nmesh)
    del tmp, f_deltaz, pos

# \tilde{\delta}_1
if 'tdelta1' in qs:
    print('calculating tdelta1...')
    delta_1 = load_delta1()
    pos = np.load(savepath+'/pos.npy')
    tmp = ArrayCatalog({'Position': pos, 'Value': np.reshape(delta_1, -1)})
    f_tdelta1 = tmp.to_mesh(Nmesh=Nmesh, BoxSize=boxsize, resampler='cic')

    f_tdelta1.save(savepath+'/tdelta1_Nmesh%d_cic.bigfile' % Nmesh)
    del tmp, f_tdelta1, delta_1, pos

# \tilde{\delta}_2
if 'tdelta2' in qs:
    print('calculating tdelta2...')
    tmp0 = (load_delta1()**2).reshape(-1)
    tmp0 = tmp0-np.mean(tmp0)
    pos = np.load(savepath+'/pos.npy')
    tmp = ArrayCatalog({'Position': pos, 'Value': tmp0})
    f_tdelta2 = tmp.to_mesh(Nmesh=Nmesh, BoxSize=boxsize, resampler='cic')

    f_tdelta2.save(savepath+'/tdelta2_Nmesh%d_cic.bigfile' % Nmesh)
    del tmp0, tmp, f_tdelta2, pos

# \tilde{G2}
if 'tG2' in qs:
    print('calculating tG2...')
    delta_1 = load_delta1()
    deltak_1 = pyfftw.interfaces.numpy_fft.rfftn(delta_1)
    G2tmp = deltak_1*0.

    kx = np.arange(0,N,1)
    kx[N//2:] -= N
    kx = kx.reshape(-1,1)*k0
    ky = np.arange(0,N,1)
    ky[N//2:] -= N
    ky = ky.reshape(1,-1)*k0

    k2xy = kx**2 + ky**2
    k2xy[0,0] = 1e-6
    # add in the tensor elements one by one
    # xx
    #G2tmp = pyfftw.interfaces.numpy_fft.rfftn(delta_1)
    for i in range(Nh+1):
        kz = i*k0
        G2tmp[:,:,i] = deltak_1[:,:,i]*kx**2/(k2xy+kz**2)
    G2 = (pyfftw.interfaces.numpy_fft.irfftn(G2tmp) - 1./3.*delta_1)**2
    # yy
    #G2tmp = pyfftw.interfaces.numpy_fft.rfftn(delta_1)
    for i in range(Nh+1):
        kz = i*k0
        G2tmp[:,:,i] = deltak_1[:,:,i]*ky**2/(k2xy+kz**2)
    G2 += (pyfftw.interfaces.numpy_fft.irfftn(G2tmp) - 1./3.*delta_1)**2
    # zz
    #G2tmp = pyfftw.interfaces.numpy_fft.rfftn(delta_1)
    for i in range(Nh+1):
        kz = i*k0
        G2tmp[:,:,i] = deltak_1[:,:,i]*kz**2/(k2xy+kz**2)
    G2 += (pyfftw.interfaces.numpy_fft.irfftn(G2tmp) - 1./3.*delta_1)**2
    # xy
    #G2tmp = pyfftw.interfaces.numpy_fft.rfftn(delta_1)
    for i in range(Nh+1):
        kz = i*k0
        G2tmp[:,:,i] = deltak_1[:,:,i]*kx*ky/(k2xy+kz**2)
    G2 += 2*pyfftw.interfaces.numpy_fft.irfftn(G2tmp)**2
    # xz
    #G2tmp = pyfftw.interfaces.numpy_fft.rfftn(delta_1)
    for i in range(Nh+1):
        kz = i*k0
        G2tmp[:,:,i] = deltak_1[:,:,i]*kx*kz/(k2xy+kz**2)
    G2 += 2*pyfftw.interfaces.numpy_fft.irfftn(G2tmp)**2
    # yz
    #G2tmp = pyfftw.interfaces.numpy_fft.rfftn(delta_1)
    for i in range(Nh+1):
        kz = i*k0
        G2tmp[:,:,i] = deltak_1[:,:,i]*ky*kz/(k2xy+kz**2)
    G2 += 2*pyfftw.interfaces.numpy_fft.irfftn(G2tmp)**2

    del delta_1, deltak_1, G2tmp

    pos = np.load(savepath+'/pos.npy')
    tmp = ArrayCatalog({'Position': pos, 'Value': np.reshape(G2, -1)})
    f_tG2 = tmp.to_mesh(Nmesh=Nmesh, BoxSize=boxsize, resampler='cic')

    f_tG2.save(savepath+'/tG2_Nmesh%d_cic.bigfile' % Nmesh)
    del tmp, f_tG2

