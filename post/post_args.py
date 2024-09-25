

'optimizer', 'step_size', 'surface_file','loss_weight_normalization',  
all_args=['coil_optimize', 'alpha_optimize', 'I_optimize', 'iter_method', 'minimize_method', 
'minimize_tol', 'nlopt_algorithm', 'stop_criteria', 'inequality_constraint_strain', 
'number_iteration', 
'number_coils', 'number_field_periods', 'stellarator_symmetry', 'number_segments',
'coil_case', 'init_coil_option', 'circle_coil_radius', 'init_coil_file',
'number_fourier_coils', 'number_control_points', 'optimize_location_nic', 'optimize_location_ns',
'length_calculate', 'length_normal', 'length_binormal', 'number_normal', 'number_binormal',
'init_fr_case', 'init_fr_file', 'current_I', 'total_current_I',
'number_theta', 'number_zeta',         
'Bn_extern', 'Bn_extern_file',
'material', 'HTS_signle_width', 'HTS_signle_thickness', 'HTS_temperature', 'HTS_I_percent', 'HTS_structural_percent',

'weight_bnormal', 'weight_length', 'weight_curvature', 'weight_curvature_max', 
'weight_torsion', 'weight_torsion_max', 'weight_distance_coil_coil', 'weight_distance_coil_surface',  
'weight_HTS_strain', 'weight_HTS_force_max', 'weight_HTS_force_mean', 'weight_HTS_Icrit', 
'target_length_mean', 'target_length_single', 'target_curvature_max', 'target_torsion_max', 
'target_distance_coil_coil', 'target_distance_coil_surface', 'target_HTS_strain', 'target_HTS_force_max', 
'save_hdf5', 'out_hdf5', 
'plot_coil', 'plot_loss', 'plot_poincare', 'number_points', 'poincare_number', 
'number_rotate', 'number_fourier_rotate', 'poincare_phi0', 'number_iter', 'number_step',
'save_makegrid', 'out_coil_makegrid',]

'coil_I', 'coil_alpha', 'coil_arg', 'coil_binormal', 'coil_centroid', 
'coil_der1', 'coil_der2', 'coil_der3', 'coil_dl', 'coil_fr', 'coil_normal', 'coil_r', 
'coil_tangent', 'coil_v1', 'coil_v2', 
'loss_B', 'loss_B_coil', 'loss_B_coil_max', 'loss_B_coil_theta', 'loss_B_max_surf', 'loss_Bn', 
'loss_Bn_mean', 'loss_HTS_Icrit', 'loss_HTS_jcrit', 'loss_curva', 'loss_curva_max', 'loss_curvature', 
'loss_dcc_min', 'loss_dcs_min', 'loss_force_max', 'loss_force_mean', 'loss_length_mean', 'loss_length_single', 
'loss_strain', 'loss_strain_max', 'loss_tor', 'loss_tor_max', 'loss_tor_mean', 
'loss_vals', 



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


filename = 'results/paper/QA/opt_1_6.h5'
arge = read_hdf5(filename)



weight = ['weight_bnormal', 'weight_length', 'weight_curvature', 
'weight_curvature_max', 'weight_distance_coil_coil', 'weight_distance_coil_surface', 'weight_HTS_Icrit', 'weight_HTS_force', 
'weight_torsion', 'weight_torsion_max']
losskeys = ['loss_Bn_mean','loss_length_mean','loss_length_single','loss_curvature','loss_curva_max','loss_tor_mean','loss_tor_max',
'loss_dcc_min','loss_dcs_min','loss_strain_max','loss_force_mean','loss_force_max','loss_B_coil_max','loss_HTS_Icrit']
target = ['target_length_mean', 'target_curvature_max', 'target_torsion_max', 'target_distance_coil_coil',
'target_distance_coil_surface', 'target_HTS_strain','target_HTS_force_max']
for key in list(all_args):
    print(arge['{}'.format(key)])
# for key in list(weight):
#     print(arge['{}'.format(key)])
# for key in list(target):
#     print(arge['{}'.format(key)])
# for key in list(losskeys):
#     print("{} = ".format(key), arge['{}'.format(key)])

# print(arge['loss_vals'])


