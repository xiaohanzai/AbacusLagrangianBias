import numpy as np
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog


def load_particle_data(islab, sim, z, Rf, qname):
    if qname == 'pos': # position
        cat_path = '/mnt/store2/bigsims/AbacusSummit/%s/halos/z%.3f/halo_info/halo_info_%03d.asdf' % (sim,z,islab)
        cat1 = CompaSOHaloCatalog(cat_path, fields=[], load_subsamples='A_field_rv')
        cat2 = CompaSOHaloCatalog(cat_path, fields=[], load_subsamples='A_halo_rv')
        q = np.concatenate((cat1.subsamples['pos'], cat2.subsamples['pos']))
        del cat1, cat2
    elif qname == 'delta1': # delta1
        with np.load('/mnt/store2/xwu/AbacusSummit/%s/z%s_tilde_operators_nbody/Rf%.3g/slab%d_sdelta1.npz'
                     % (sim,str(z),Rf,islab)) as tmp:
            q = np.concatenate((tmp['field'], tmp['halo']))
    elif qname in ['nabla2d1', 'G2']: # nabla2d1 or G2
        with np.load('/mnt/store2/xwu/AbacusSummit/%s/z%s_tilde_operators_nbody/Rf%.3g/slab%d_%s.npz'
                     % (sim,str(z),Rf,islab,qname)) as tmp:
            q = np.concatenate((tmp['field'], tmp['halo']))
    else:
        print(qname, 'not recognized')
        return
    return q

