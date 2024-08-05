

#### losskeys = 
# ['loss_B_max_coil','loss_B_max_surf','loss_Bn_max','loss_Bn_mean',
# 'loss_HTS_Icrit','loss_curva_max','loss_curvature','loss_dcc_min','loss_dcs_min',
# 'loss_length','loss_strain_max','loss_tor_max','loss_tor_mean',
# 'loss_Bn', 'loss_B', 'loss_curva', 'loss_tor']

#### coilkeys = 
# ['coil_I','coil_alpha','coil_arg','coil_binormal','coil_centroid',
# 'coil_der1','coil_der2','coil_der3','coil_dl','coil_fr','coil_normal',
# 'coil_r','coil_tangent','coil_v1','coil_v2']

#### weight = ["weight_bnormal","weight_length","weight_curvature","weight_curvature_max","weight_torsion",
# "weight_torsion_max","weight_distance_coil_coil","weight_distance_coil_surface","weight_HTS_strain"]

#### target = ['target_length_mean', 'target_curvature_max', 'target_torsion_max', 'target_distance_coil_coil',
# 'target_distance_coil_surface', 'target_HTS_strain']

#### initkeys = 
# ["iter_method","number_iteration","minimize_method","minimize_tol",
# "optimizer_coil","optimizer_fr","optimizer_I","step_size_coil","step_size_fr","step_size_I","momentum_mass","axis_resolution","var","eps",
# "number_coils","number_field_periods","stellarator_symmetry","number_independent_coils","number_segments",
# "coil_radius","coil_case","init_coil_option","file_type","init_coil_file",
# "number_fourier_coils","number_control_points","spline_k","optimize_location_nic","optimize_location_ns",
# "length_normal","length_binormal","number_normal","number_binormal",
# "init_fr_case","init_fr_file","number_rotate","number_fourier_rotate",
# "current_independent","current_I","I_optimize",
# "number_theta","number_zeta","surface_case","surface_vmec_file","surface_r_file","surface_nn_file","surface_sg_file",
# "B_extern",
# "material","HTS_width","HTS_signle_width","HTS_signle_thickness","HTS_I_thickness","HTS_sec_area","HTS_temperature","HTS_I_percent","HTS_structural_percent",
# "weight_bnormal","weight_length","weight_curvature","weight_curvature_max","weight_torsion","weight_torsion_max","weight_distance_coil_coil","weight_distance_coil_surface","weight_HTS_strain",
# "plot_coil","plot_loss","plot_poincare","number_points","poincare_r0","poincare_z0","poincare_phi0","number_iter","number_step",
# "save_npy","save_hdf5","save_makegrid","out_hdf5","out_coil_makegrid","save_loss","save_coil_arg","save_fr"]

import plotly.graph_objects as go
import jax.numpy as np
import h5py
import sys
sys.path.append('iteration')
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


filename = 'results/LQA/circle_start/length/l_2t4.2.h5'
arge = read_hdf5(filename)


# iter = ["iter_method","minimize_method","minimize_tol", 'coil_case','nlopt_algorithm','stop_criteria']
# weight = ["weight_bnormal","weight_length","weight_curvature","weight_curvature_max","weight_torsion",
# "weight_torsion_max","weight_distance_coil_coil","weight_distance_coil_surface","weight_HTS_strain",
# 'weight_B_theta','weight_HTS_force']
losskeys = ['loss_Bn_mean','loss_length','loss_curvature','loss_curva_max','loss_tor_mean','loss_tor_max',
'loss_dcc_min','loss_dcs_min','loss_strain_max','loss_force','loss_B_coil_max','loss_HTS_Icrit']
# target = ['target_length_mean', 'target_curvature_max', 'target_torsion_max', 'target_distance_coil_coil',
# 'target_distance_coil_surface', 'target_HTS_strain']
# for key in list(iter):
#     print(arge['{}'.format(key)])
# for key in list(weight):
#     print(arge['{}'.format(key)])
for key in list(losskeys):
    print("{} = ".format(key), arge['{}'.format(key)])




