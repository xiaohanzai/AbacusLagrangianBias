import numpy as np
import sys
import pyfftw
import asdf

boxsize = 2000.
k0 = 2*np.pi/boxsize
N = 1152

sim = sys.argv[1] # e.g. AbacusSummit_base_c000_ph006

if len(sys.argv) > 2:
    Rf = float(sys.argv[2])
    if len(sys.argv) > 3:
        qs = sys.argv[3:]
else:
    cellsize = 5. # Mpc/h, cell size in final grid
    Rf = cellsize/(2*np.pi)**0.5 # Gaussian smoothing
    qs = ['sdelta1', 'G2', 'nabla2d1']
print('calculating: ', qs)

# load in IC grid
Nh = N//2

# delta1
kx = np.arange(0,N,1)
kx[N//2:] -= N
kx = kx.reshape(-1,1)*k0
ky = np.arange(0,N,1)
ky[N//2:] -= N
ky = ky.reshape(1,-1)*k0

k2xy = kx**2 + ky**2
k2xy[0,0] = 1e-6

path = '/mnt/store2/xwu/AbacusSummit/%s/ic_%d/' % (sim,N)
af = asdf.open(path+'/ic_dens_N%d.asdf' % N)
delta1 = af['data']['density']
if Rf < 1e-4:
    sdelta1 = delta1
    delta1 = pyfftw.interfaces.numpy_fft.rfftn(delta1)
else:
    delta1 = pyfftw.interfaces.numpy_fft.rfftn(delta1)
    for i in range(Nh+1):
        kz = i*k0
        k2 = k2xy+kz**2
        delta1[:,:,i] *= np.exp(-k2*Rf**2/2.)
    sdelta1 = pyfftw.interfaces.numpy_fft.irfftn(delta1)
if 'sdelta1' in qs:
    np.save(path+'/sdelta1_Rf%.3g' % Rf, sdelta1)

if 'nabla2d1' in qs:
    tmp = delta1*0.
    for i in range(Nh+1):
        kz = i*k0
        k2 = k2xy+kz**2
        tmp[:,:,i] = -delta1[:,:,i]*k2
    np.save(path+'/nabla2d1_Rf%.3g' % Rf, pyfftw.interfaces.numpy_fft.irfftn(tmp))
    del tmp

if 'G2' in qs:
    # G2
    G2tmp = delta1*0.

    # xx
    for i in range(Nh+1):
        kz = i*k0
        G2tmp[:,:,i] = delta1[:,:,i]*kx**2/(k2xy+kz**2)
    G2 = (pyfftw.interfaces.numpy_fft.irfftn(G2tmp) - 1./3.*sdelta1)**2
    # yy
    for i in range(Nh+1):
        kz = i*k0
        G2tmp[:,:,i] = delta1[:,:,i]*ky**2/(k2xy+kz**2)
    G2 += (pyfftw.interfaces.numpy_fft.irfftn(G2tmp) - 1./3.*sdelta1)**2
    # zz
    for i in range(Nh+1):
        kz = i*k0
        G2tmp[:,:,i] = delta1[:,:,i]*kz**2/(k2xy+kz**2)
    G2 += (pyfftw.interfaces.numpy_fft.irfftn(G2tmp) - 1./3.*sdelta1)**2
    # xy
    for i in range(Nh+1):
        kz = i*k0
        G2tmp[:,:,i] = delta1[:,:,i]*kx*ky/(k2xy+kz**2)
    G2 += 2*pyfftw.interfaces.numpy_fft.irfftn(G2tmp)**2
    # xz
    for i in range(Nh+1):
        kz = i*k0
        G2tmp[:,:,i] = delta1[:,:,i]*kx*kz/(k2xy+kz**2)
    G2 += 2*pyfftw.interfaces.numpy_fft.irfftn(G2tmp)**2
    # yz
    for i in range(Nh+1):
        kz = i*k0
        G2tmp[:,:,i] = delta1[:,:,i]*ky*kz/(k2xy+kz**2)
    G2 += 2*pyfftw.interfaces.numpy_fft.irfftn(G2tmp)**2
    del G2tmp, delta1

    np.save(path+'/G2_Rf%.3g' % Rf, G2)

