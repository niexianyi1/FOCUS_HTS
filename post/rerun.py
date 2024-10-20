import json
import h5py
import sys
sys.path.append('opt_coil')
import optimize


init_args=['coil_optimize', 'alpha_optimize', 'I_optimize', 'iter_method', 
'minimize_method', 'minimize_tol', 
'nlopt_algorithm', 'stop_criteria', 'inequality_constraint_strain', 
'number_iteration', 'optimizer', 'step_size',
'number_coils', 'number_field_periods', 'stellarator_symmetry', 'number_segments',
'coil_case', 'init_coil_option', 'circle_coil_radius', 'init_coil_file',
'number_fourier_coils', 'number_control_points', 'optimize_location_nic', 'optimize_location_ns',
'length_calculate', 'length_normal', 'length_binormal', 'number_normal', 'number_binormal',
'init_fr_case', 'init_fr_file', 'number_rotate', 'number_fourier_rotate',
'current_I', 'total_current_I',
'number_theta', 'number_zeta', 'surface_file', 'Bn_background', 'Bn_background_file',
'material', 'HTS_single_width', 'HTS_single_thickness', 'HTS_temperature', 'HTS_I_percent', 'HTS_structural_percent',
'loss_weight_normalization', 'weight_bnormal', 'weight_length', 'weight_curvature', 'weight_curvature_max', 
'weight_torsion', 'weight_torsion_max', 'weight_distance_coil_coil', 'weight_distance_coil_surface',  
'weight_HTS_strain', 'weight_HTS_force_max', 'weight_HTS_force_mean', 'weight_HTS_Icrit', 
'target_length_mean', 'target_length_single', 'target_curvature_max', 'target_torsion_max', 
'target_distance_coil_coil', 'target_distance_coil_surface', 'target_HTS_strain', 'target_HTS_force_max', 
'save_hdf5', 'out_hdf5', 'save_makegrid', 'out_coil_makegrid',
'plot_coil', 'number_points', 'plot_loss',
'plot_poincare', 'poincare_number', 'poincare_phi0', 'number_iter', 'number_step']

def read_hdf5(filename):
    f = h5py.File(filename, "r")
    arge = {}
    for key in list(f.keys()):
        print(key)
        val = f[key][()]
        if isinstance(val, bytes):
            val = str(val, encoding='utf-8')
        arge.update({key: val})
    f.close()
    return arge


filename = 'results/paper/QA/opt_1.h5'
arge = read_hdf5(filename)


args = {}
for item in init_args:
    args['{}'.format(item)] = arge['{}'.format(item)]



with open('initfiles/init_args.json', 'w') as f:
    json.dump(args, f, indent=4)


optimize.main()


