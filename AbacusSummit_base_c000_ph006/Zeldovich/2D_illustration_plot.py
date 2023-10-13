import numpy as np
from numpy.fft import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from colossus.lss import peaks
from colossus.lss import bias
from scipy.special import erfc
from scipy.integrate import quad
from scipy.stats import norm
import sys
import os
import warnings
warnings.filterwarnings("ignore")


z = 0.5
z_init = 99
Pk_init = lambda k: k**-1
N = 32
boxsize = N*5.#1000.

#f_fdelta1 = lambda delta1: delta1
def f_fdelta1(delta1):
    params = {'flat': True, 'H0': 67.36, 'Om0': 0.315192, 'Ob0': 0.02237/0.6736**2, 'sigma8': 0.807952, 'ns': 0.9649}
    cosmo = cosmology.setCosmology('myCosmo', params)
    sigma_R = cosmo.sigma(5., z, filt='gaussian')
    M = 1e13
    sigma_M = cosmo.sigma(peaks.lagrangianR(M), z)
    numerator = (peaks.collapseOverdensity() - delta1*sigma_R).clip(1e-6)
    nu1 = numerator / (sigma_M**2 - sigma_R**2)**0.5
    nu = peaks.collapseOverdensity()/sigma_M
    return erfc(nu1/2**0.5)/erfc(nu/2**0.5)


OMm = (0.02237+0.12)/0.6736**2
OMl = 1 - OMm
def calc_growthrate(z):
    omegaM_z = OMm*(1+z)**3 / ( OMl + OMm*(1+z)**3 )
    dick_z = 2.5*omegaM_z / ( 1.0/70.0 + omegaM_z*(209-omegaM_z)/140.0 + (omegaM_z)**(4.0/7.0) )
    dick_0 = 2.5*OMm / ( 1.0/70.0 + OMm*(209-OMm)/140.0 + (OMm)**(4.0/7.0) )
    return dick_z / (dick_0 * (1.0+z))
# growth rates
Dinit = calc_growthrate(z_init)
Dz = calc_growthrate(z)
Ddiff = Dz - Dinit
Dratio = Dz/Dinit


def gen_kgrid():
    dk = 2*np.pi/boxsize
    ks = fftfreq(N)*N*dk
    kx = ks.reshape(-1,1)
    ks = rfftfreq(N)*N*dk
    ky = ks.reshape(1,-1)
    return kx, ky


def gen_xgrid():
    tmp = np.linspace(0, N-1, N)
    # tmp[N//2:] -= N
    tmp *= boxsize/N
    x = tmp.reshape(-1,1)
    y = tmp.reshape(1,-1)
    return x, y


def adj_complex_conj(arr_, set_DC_zero=True):
    '''
    We need to adjust some values of arr to make sure the inverse DFT of arr gives a real array.
    I will only deal with rfft for now, so make sure the shape of arr is correct.
    '''
    arr = arr_.copy()

    ## I am basically copying from 21cmFAST
    dim = arr.shape[0]
    middle = dim//2

    # corners
    for i in [0, middle]:
        for j in [0, middle]:
            arr[i,j] = np.real(arr[i,j])
    # set the DC mode to 0
    if set_DC_zero:
        arr[0,0] = 0

    ind = np.arange(1, middle, 1, dtype=int)
    # do all of i except corners
    # just j corners
    for j in [0, middle]:
        arr[ind,j] = np.conjugate(arr[dim-ind,j])

    return arr


def gen_deltak_Pk():
    '''
    Given a power spectrum, generate a random realization of the source grid.
    L is the box size. N is the number of grid points on a size.
    Pk is the power spectrum function that takes the wavenumber and returns the power.
    Let's use the same Fourier convention as Mesinger & Furlanetto 07.
    '''
    middle = N//2
    volume = boxsize**2

    kx, ky = gen_kgrid()
    kn = (kx**2 + ky**2)**0.5

    # create k space Gaussian random field based on Pk
    # generate random numbers
    a = np.random.randn(N,middle+1)
    b = np.random.randn(N,middle+1)
    rhok = (volume * Pk_init(kn) / 2)**0.5 * (a + b*1j)
    
    return adj_complex_conj(rhok)


def calc_ZA_displacement(deltak_1):
    Nh = N//2
    kx, ky = gen_kgrid()
    k2 = kx**2 + ky**2
    k2[0,0] = 1e-6
    psi1x = deltak_1*0.
    psi1y = deltak_1*0.

    tmp = deltak_1/k2*(-1)**0.5
    psi1x = tmp*kx
    psi1y = tmp*ky

    psi1x = irfftn(psi1x)
    psi1y = irfftn(psi1y)

    return psi1x, psi1y


def calc_pos(psi):
    psi1x, psi1y = psi
    x, y = gen_xgrid()
    pos = np.zeros((N**2,2))
    pos[:,0] = (x + Ddiff/Dinit*psi1x).ravel()
    pos[:,1] = (y + Ddiff/Dinit*psi1y).ravel()
    pos[pos > boxsize] -= boxsize
    pos[pos < 0.] += boxsize
    return pos


def calc_deltah(pos, fs):
    deltah = np.zeros((N,N))
    dx = boxsize/N
    xmin = -boxsize/2.
    for i in range(len(pos)):
        indx = int(np.round((pos[i,0]-xmin)/dx))%N
        indy = int(np.round((pos[i,1]-xmin)/dx))%N
        deltah[indy,indx] += fs[i]
    deltah = deltah/np.mean(deltah)-1.
    deltah[deltah<-1+0.1] = np.nan
    return deltah


def main():
    np.random.seed(8)
    dx = boxsize/N/2.

    # generate deltak_1 and delta1 fields
    deltak_1 = gen_deltak_Pk()
    delta1 = irfftn(deltak_1)
    delta1 /= np.std(delta1) # normalize
    deltak_1 = rfftn(delta1*0.032)

    # f(delta1) values
    fs = f_fdelta1(delta1).ravel()
    print(fs.max())
    fac = f_fdelta1(2)/10.

    # plot the initial density field
    pos = calc_pos((delta1*0., delta1*0.))
    # smooth it so that things are plotted prettier...
    kx, ky = gen_kgrid()
    k2 = kx**2 + ky**2
    W = np.exp(-2*k2*2.**2)
    delta1_ = irfftn(rfftn(delta1)*W)
    delta1_ /= np.std(delta1_)
    # plt.scatter(pos[:,0], pos[:,1], marker='o', s=150, c=delta1_, vmin=-2.5, vmax=2.5, cmap='jet', edgecolors='none', alpha=0.8)
    plt.imshow(delta1_.T, vmin=-2.2, vmax=2.2, cmap='YlOrRd', interpolation='bilinear', extent=(-dx, boxsize-dx, -dx, boxsize-dx), origin='lower')
    cbar = plt.colorbar()
    cbar.set_label(r'$\delta_1$', fontsize=18)
    bbox = cbar.ax.get_position()
    bbox = [bbox.x0-0.03, bbox.y0, bbox.x1-bbox.x0, bbox.y1-bbox.y0]
    cbar.ax.set_position(bbox)
    # for i in range(len(pos)):
    #     plt.plot(pos[i,0], pos[i,1], 'gray', marker='o', alpha=0.6, markersize=fs[i]/fac)
    plt.scatter(pos[:,0], pos[:,1], marker='o', s=fs*30, c='gray', edgecolors='none', alpha=0.7)
    plt.gca().set_aspect(1)
    plt.xlim(-dx, boxsize-dx)
    plt.ylim(-dx, boxsize-dx)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('initial_fields.png')
    plt.show()

    # calculate final positions
    psi = calc_ZA_displacement(deltak_1)
    pos = calc_pos(psi)

    # plot final density field
    plt.scatter(pos[:,0], pos[:,1], marker='o', s=30, c=delta1, vmin=-2.2, vmax=2.2, cmap='YlOrRd', edgecolors='none')
    cbar = plt.colorbar()
    cbar.set_label(r'$\delta_1$', fontsize=18)
    cbar.ax.set_position(bbox)
    plt.gca().set_aspect(1)
    plt.xlim(-dx, boxsize-dx)
    plt.ylim(-dx, boxsize-dx)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('final_density.png')
    plt.show()

    # plot final halo field
    #norm = mpl.colors.Normalize(vmin=0.01, vmax=10)
    plt.scatter(pos[:,0], pos[:,1], marker='o', s=fs*30, c=fs, cmap='plasma', vmin=0.01, vmax=f_fdelta1(2.5), edgecolors='none')
    # for i in range(len(pos)):
    #     plt.plot(pos[i,0], pos[i,1], 'gray', marker='o', alpha=0.7, markersize=fs[i]/fac)
    cbar = plt.colorbar()
    cbar.set_label(r'$f(\delta_1)$', fontsize=18)
    cbar.ax.set_position(bbox)
    plt.gca().set_aspect(1)
    dx = boxsize/N/2.
    plt.xlim(-dx, boxsize-dx)
    plt.ylim(-dx, boxsize-dx)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('final_halo.png')
    plt.show()

    # plot final halo grid (not successful)
    # deltah = calc_deltah(pos, fs)
    # plt.imshow(deltah, vmin=-1, vmax=5, origin='lower')
    # plt.gca().set_aspect(1)
    # cbar = plt.colorbar()
    # cbar.set_label(r'$\delta_h$', fontsize=18)
    # plt.show()


if __name__ == '__main__':
    main()

