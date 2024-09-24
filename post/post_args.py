


import plotly.graph_objects as go
import jax.numpy as np
import h5py
import sys
sys.path.append('opt_coil')
import read_init

def read_hdf5(filename):
    f = h5py.File(filename, "r")
    arge = {}
    for key in list(f.keys()):
        val = f[key][()]
        if isinstance(val, bytes):
            val = str(val, encoding='utf-8')
        arge.update({key: val})
    f.close()
    return arge


filename = 'results/paper/QA/opt_2.h5'
arge = read_hdf5(filename)


iter = ["iter_method","minimize_method","minimize_tol", 'coil_case','nlopt_algorithm','stop_criteria']
weight = ["weight_bnormal","weight_length","weight_curvature","weight_curvature_max","weight_torsion",
"weight_torsion_max","weight_distance_coil_coil","weight_distance_coil_surface","weight_HTS_strain",
'weight_HTS_force_max','weight_HTS_force_mean','weight_HTS_Icrit']
losskeys = ['loss_Bn_mean','loss_length_mean','loss_length_single','loss_curvature','loss_curva_max','loss_tor_mean','loss_tor_max',
'loss_dcc_min','loss_dcs_min','loss_strain_max','loss_force_mean','loss_force_max','loss_B_coil_max','loss_HTS_Icrit']
target = ['target_length_mean', 'target_curvature_max', 'target_torsion_max', 'target_distance_coil_coil',
'target_distance_coil_surface', 'target_HTS_strain','target_HTS_force_max']
# for key in list(iter):
#     print(arge['{}'.format(key)])
# for key in list(weight):
#     print(arge['{}'.format(key)])
# for key in list(target):
#     print(arge['{}'.format(key)])
for key in list(losskeys):
    print("{} = ".format(key), arge['{}'.format(key)])

# print(arge['coil_I'])


