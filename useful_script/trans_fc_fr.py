

import jax.numpy as np
import h5py




def read_hdf5(filename):
    f = h5py.File(filename, "r")
    arge = {}
    for key in list(f.keys()):
        val = f[key][()]
        if key == 'num_fourier_coils':
            key = 'number_fourier_coils'
        if isinstance(val, bytes):
            val = str(val, encoding='utf-8')
        arge.update({key: val})
    f.close()
    return arge

filename = 'results/paper/LQA/useful/cs_sn_1.h5'
arge = read_hdf5(filename)
coil_fr = np.array(arge['coil_fr'])
coil_fr = np.transpose(coil_fr, (1,0,2))
arge['coil_fr'] = coil_fr
# coil_arg = np.array(arge['coil_arg'])
# coil_arg = np.transpose(coil_arg, (1,0,2))
# arge['coil_arg'] = coil_arg




with h5py.File('results/paper/LQA/useful/cs_sn_1_t.h5', "w") as f:
    for key in arge:
        f.create_dataset(name=key, data=arge['{}'.format(key)])




