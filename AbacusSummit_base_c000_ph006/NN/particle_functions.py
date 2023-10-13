import numpy as np
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import os


f_ijk2ind = lambda i,j,k,Nmesh: i*Nmesh**2 + j*Nmesh + k

def f_ind2ijk(ind, Nmesh):
    i = int(ind/Nmesh**2)
    j = int((ind - i*Nmesh**2)/Nmesh)
    k = ind - i*Nmesh**2 - j*Nmesh
    return i,j,k


def get_ic_path(sim):
    ic_path = '/mnt/marvin2/'
    if not os.path.exists(ic_path):
        ic_path = '/mnt/store2/'
    ic_path += 'xwu/AbacusSummit/'
    N = 1152
    if 'small' in sim:
        if 'small/' not in sim:
            sim = 'small/'+sim
        N = 576
    return ic_path + '/%s/ic_%d/' % (sim, N)


def load_particle_data(islab, sim, z, Rf, qname):
    _path = '/mnt/marvin2'
    if not os.path.exists(_path):
        _path = '/mnt/store2'

    if 'small' in sim:
        if 'small/' not in sim:
            sim = 'small/'+sim

    if qname == 'pos': # position
        cat_path = _path+'/bigsims/AbacusSummit/%s/halos/z%.3f/halo_info/halo_info_%03d.asdf' % (sim,z,islab)
        cat1 = CompaSOHaloCatalog(cat_path, fields=[], load_subsamples='A_field_rv')
        cat2 = CompaSOHaloCatalog(cat_path, fields=[], load_subsamples='A_halo_rv')
        q = np.concatenate((cat1.subsamples['pos'], cat2.subsamples['pos']))
        del cat1, cat2
    elif qname == 'delta1': # delta1
        with np.load(_path+'/xwu/AbacusSummit/%s/z%s_tilde_operators_nbody/Rf%.3g/slab%d_sdelta1.npz'
                     % (sim,str(z),Rf,islab)) as tmp:
            q = np.concatenate((tmp['field'], tmp['halo']))
    elif qname in ['nabla2d1', 'G2']: # nabla2d1 or G2
        with np.load(_path+'/xwu/AbacusSummit/%s/z%s_tilde_operators_nbody/Rf%.3g/slab%d_%s.npz'
                     % (sim,str(z),Rf,islab,qname)) as tmp:
            q = np.concatenate((tmp['field'], tmp['halo']))
    else:
        print(qname, 'not recognized')
        return
    return q


def load_particle_features(sim, z, islab, Rfs, qs):
    '''
    Rfs is a list of Rf values, and qs is a list of quantities to be used.
      e.g. Rfs = [3, 4], qs = ['dnG', 'dnG'], where d, n, G stand for delta1, nabla2d1, G2.
    '''
    # load in particle position
    pos = load_particle_data(islab, sim, z, None, 'pos')

    # load in features
    n_inputs = 0
    for i in qs:
        n_inputs += len(i)

    features = np.zeros((n_inputs, len(pos)), dtype=np.float32)
    n = 0
    ic_path = get_ic_path(sim)
    for i in range(len(Rfs)):
        Rf = Rfs[i]
        for q in qs[i]:
            if q == 'd':
                features[n] = load_particle_data(islab, sim, z, Rf, 'delta1')/\
                    np.std(np.load(ic_path+'/sdelta1_Rf%.3g.npy' % Rf).astype(np.float64))
            elif q == 'n':
                features[n] = load_particle_data(islab, sim, z, Rf, 'nabla2d1')/\
                    np.std(np.load(ic_path+'/nabla2d1_Rf%.3g.npy' % Rf).astype(np.float64))
            elif q == 'G':
                tmp = np.load(ic_path+'/G2_Rf%.3g.npy' % Rf).astype(np.float64)
                features[n] = (load_particle_data(islab, sim, z, Rf, 'G2') -\
                    np.mean(tmp))/np.std(tmp)
            n += 1

    return pos, features


def decorrelate_features(features, Rfs, qs, eigvecs=None, eigvals=None):
    '''
    Move into a feature space where the delta, nabla, G2's are decorrelated.
    '''
    if eigvecs is None:
        eigvecs = np.zeros((len(features), len(features)))
        eigvals = np.zeros(len(features))
        # determine which ones are G2
        ii = np.zeros(len(eigvals), dtype=bool)
        n = 0
        for i in range(len(Rfs)):
            for q in qs[i]:
                if q == 'G':
                    ii[n] = True
                n += 1
        # calculate covariance matrix for the G2's
        n_G2 = ii.sum()
        if n_G2 > 0:
            covmat = features[ii].astype(np.float64).dot(features[ii].T)/(features.shape[1]-1.)
            # eigvals and eigvecs
            eigvals_, eigvecs_ = np.linalg.eigh(covmat)
            eigvals[:n_G2] = eigvals_
            eigvecs[:,:n_G2][ii] = eigvecs_
        # now the delta's and nabla's
        covmat = features[~ii].astype(np.float64).dot(features[~ii].T)/(features.shape[1]-1.)
        # eigvals and eigvecs
        eigvals_, eigvecs_ = np.linalg.eigh(covmat)
        eigvals[n_G2:] = eigvals_
        eigvecs[:,n_G2:][~ii] = eigvecs_
        # try:
        #     covmat = features.astype(np.float64).dot(features.T)/(features.shape[1]-1.)
        # except: # out of memory
        #     covmat = np.zeros((features.shape[0], features.shape[0]), dtype=np.float64)
        #     for i in range(features.shape[0]):
        #         for j in range(features.shape[0]):
        #             covmat[i,j] = features[i].astype(np.float64).dot(features[j])/(features.shape[1]-1.)
        # eigvals, eigvecs = np.linalg.eigh(covmat)
    features_new = eigvecs.T.dot(features)
    # normalize before return; mean should already be 0
    return eigvals, eigvecs, (features_new/(eigvals**0.5).reshape(-1,1)).astype(np.float32)


def choose_decorrelated_features(eigvals, Rfs, qs):
    '''
    The input features array should be decorrelated and normalized to std=1.
    Choose the first n+1 decorrelated delta and nabla's (largest eigvalues),
      where n is the number of smoothing scales.
    Keep all G2's.
    '''
    # determine which ones are G2
    ii = np.zeros(len(eigvals), dtype=bool)
    n = 0
    for i in range(len(Rfs)):
        for q in qs[i]:
            if q == 'G':
                n += 1
    # keep all G2's, but only keep the first n+1 delta and nabla
    ii[:n] = True
    ii[(n+np.argsort(eigvals[n:]))[-(len(Rfs)+1):]] = True
    return ii

