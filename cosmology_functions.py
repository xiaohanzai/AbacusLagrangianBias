import numpy as np
import pyfftw

OMm = (0.02237+0.12)/0.6736**2
OMl = 1 - OMm
def calc_growthrate(z):
    omegaM_z = OMm*(1+z)**3 / ( OMl + OMm*(1+z)**3 )
    dick_z = 2.5*omegaM_z / ( 1.0/70.0 + omegaM_z*(209-omegaM_z)/140.0 + (omegaM_z)**(4.0/7.0) )
    dick_0 = 2.5*OMm / ( 1.0/70.0 + OMm*(209-OMm)/140.0 + (OMm)**(4.0/7.0) )
    return dick_z / (dick_0 * (1.0+z))

# shot noise for mass weighted halos
f_Pshot_M = lambda boxsize, Ms: boxsize**3 * np.sum(Ms**2) / np.sum(Ms)**2

# write my own power spectrum function
def calc_k_grid(L, N):
    x = np.arange(0,N,1)
    x[N//2:] -= N
    x = x.reshape(-1,1,1)
    y = np.arange(0,N,1)
    y[N//2:] -= N
    y = y.reshape(1,-1,1)
    z = np.arange(0,N//2+1,1).reshape(1,1,-1)
    dk = 2*np.pi/L
    return x*dk, y*dk, z*dk

def calc_powerspectrum(grid1, L, ind=None, grid2=None, kbins=None, arr=None):
    '''
    grid1 and grid2 should be in k space, obtained by np.fft.rfftn.
    L is the box size, N the mesh size.
    To save time, input the sorted ind directly, as well as arr...
      but make sure all the inputs are consistent with each other.  I'm not doing any checks.
    '''
    N = np.shape(grid1)[0]
    if kbins is None:
        dk = 2*np.pi/L
        kvec = np.arange(1, N//2+1, 1)*dk
        kbins = np.linspace(kvec[0]-0.5*dk, kvec[-1]+0.5*dk, len(kvec)+1)
    else:
        kvec = (kbins[1:] + kbins[:-1])/2.
    Pk = np.zeros(len(kbins)-1, dtype=complex)

    if ind is None or arr is None:
        x, y, z = calc_k_grid(L, N)
        r = ((x**2 + y**2 + z**2)**0.5).ravel()
        if ind is None:
            ind = np.argsort(r)
        r = r[ind]
        if arr is None:
            arr = np.searchsorted(r, kbins)

    if grid2 is None:
        grid2 = grid1
    grid = (grid1*np.conj(grid2)).ravel()[ind]

    for i in range(len(arr)-1):
        Pk[i] = np.mean(grid[arr[i]:arr[i+1]])*L**3/N**6

    return kvec, Pk

def calc_xi(grid1, L, grid2=None, r=None):
    if grid2 is None:
        grid2 = grid1
    Nmesh = np.shape(grid1)[0]

    # grid1 and grid2 should be delta's -- normalized with mean and minus 1
    # I'm setting the threshold to be 1e-4 here but need special care for G2...
    g1 = grid1
    if np.abs(np.mean(grid1)) > 1e-4:
        print('warning: grid not normalized; going to normalize...')
        g1 = grid1/np.mean(grid1)-1.
    g2 = grid2
    if np.abs(np.mean(grid2)) > 1e-4:
        g2 = grid2/np.mean(grid2)-1.

    # FFT
    grid1k = pyfftw.interfaces.numpy_fft.rfftn(g1)
    grid2k = pyfftw.interfaces.numpy_fft.rfftn(g2)
    # 3D power
    Pkgrid = grid1k*np.conj(grid2k) / Nmesh**3
    # 3D corr func with zero-lag being 1
    xigrid = pyfftw.interfaces.numpy_fft.irfftn(Pkgrid)

    sigma = np.mean(g1**2)**0.5 * np.mean(g2**2)**0.5
    print(xigrid[0,0,0]/sigma, sigma)

    # bin in r and measure xi
    if r is None:
        tmp = np.arange(0., Nmesh, 1.)
        tmp[Nmesh//2:] -= Nmesh
        tmp *= L/Nmesh
        r = (tmp.reshape(-1,1,1)**2 + tmp.reshape(1,-1,1)**2 + tmp.reshape(1,1,-1)**2)**0.5
    dr = L/Nmesh
    rbins = np.arange(dr*3**0.5, L, dr)
    rs = (rbins[1:]**4 - rbins[:-1]**4)/(rbins[1:]**3 - rbins[:-1]**3)*0.75
    xi = np.zeros(len(rs))+np.nan
    for i in range(len(rs)):
        ii = (r > rbins[i]) & (r <= rbins[i+1])
    #     rs[i] = r[ii & ii1].mean()
        xi[i] = xigrid[ii].mean()

    return np.append([0], rs), np.append([xigrid[0,0,0]], xi)

