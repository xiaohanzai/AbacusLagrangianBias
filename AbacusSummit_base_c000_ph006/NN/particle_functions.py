import numpy as np
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import os


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
                    np.std(np.load(ic_path+'/sdelta1_Rf%.3g.npy' % Rf))
            elif q == 'n':
                features[n] = load_particle_data(islab, sim, z, Rf, 'nabla2d1')/\
                    np.std(np.load(ic_path+'/nabla2d1_Rf%.3g.npy' % Rf))
            elif q == 'G':
                features[n] = (load_particle_data(islab, sim, z, Rf, 'G2') -\
                    np.mean(np.load(ic_path+'/G2_Rf%.3g.npy' % Rf)))/np.std(np.load(ic_path+'/G2_Rf%.3g.npy' % Rf))
            n += 1

    return pos, features


