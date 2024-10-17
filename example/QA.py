

import sys
import json
sys.path.append('opt_coil')
import optimize

args = {

### Optimization

    'coil_optimize':            1,          #       int,    
    'alpha_optimize':           0,          #       int,    
    'I_optimize':               0,          #       int,    
## Iteration 
    'iter_method':              'nlopt',    #       str,    'jax', 'min', 'nlopt', #'for-min', 'min-for'
# nlopt
    'nlopt_algorithm':          'LD_MMA',   #       str,     https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
    'stop_criteria':            1e-6,       #       float,                                
    'inequality_constraint_strain':  1,
# min   
    'minimize_method':          'CG',       #       str,     https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    'minimize_tol':             1e-2,       #       float,                                 
# jax
    'number_iteration':         0,          #       int,    
    'optimizer':                'momentum', #       str,    https://jax.readthedocs.io/en/latest/jax.example_libraries.optimizers.html#jax.example_libraries.optimizers.constant
    'step_size':                1e-5,       #       float,  

### Coil 
## Coil parameters
    'number_coils':             16,         # (nc)  int,                   
    'number_field_periods':     2,          # (nfp) int,                   
    'stellarator_symmetry':     1,          # (ss)  int,                      
    'number_segments':          80,         # (ns)  int,    
    
## Initial coil input        
    'coil_case':                'fourier',  #       str,     'spline' or 'fourier' or 'spline_local'
    'init_coil_option':         'circle',   #       str,    'spline' or 'coordinates' or 'fourier' or 'circle'
    'circle_coil_radius':       0.7,        #       float,  
    'init_coil_file':                       #       str,   
            'results/paper/QA/opt_1.h5',       
    
# Fourier
    'number_fourier_coils':        6,       # (nfc) int,   
                     
# Bspline
    'number_control_points':        18,     # (ncp) int,    

# Bspline Local optimization
    'optimize_location_nic':    [0],        #       list,  
    'optimize_location_ns':                 #       list,  
                [[[0, 64]]],               

## Finite-build 
    'length_calculate':         0,          #       int,   
    'length_normal':            [0.05 for i in range(4)],       # (ln)  list,                     
    'length_binormal':          [0.05 for i in range(4)],       # (lb)  list,  
    'number_normal':            2,          # (nn)  int,               
    'number_binormal':          2,          # (nb)  int,              
                     
## Rotation Angle 
    'init_fr_case':             0,          #       int,    
    'init_fr_file':                         #       str,    
            'results_f/circle/s1_fr.npy',               
    'number_rotate':            0,          # (nr)  int,                      
    'number_fourier_rotate':    6,          # (nfr) int,     
   
## Coil Current
    'current_I':                [1e5,1e5,1e5,1e5],      #       list,            
    'total_current_I':          4e5,        #       int,    
        
## Magnetic surface
    'number_theta':             32,         # (nt)  int,    \theta(poloidal)                  
    'number_zeta':              80,         # (nz)  int,    \zeta(toroidal)                      
    'surface_file':                         #       str,    
            'initfiles/QA/plasma_no_well.boundary',

## Background magnetic field (from plasma current) 
    'Bn_background':                 0,          #       int,    
    'Bn_background_file':                        #       str,    
            'initfiles/ncsx_c09r00/c09r00.boundary',

### HTS tape(REBCO) or LTS
    'material':                 'REBCO_LT', #       str,    'REBCO_LT'(<60K) or 'REBCO_HT'(>68K) or 'NbTi' or 'Nb3Sn'    
    'HTS_single_width':         4e-3,       #       float,  
    'HTS_single_thickness':     5e-5,       #       loat,  
    'HTS_temperature':          4.2,        #       float,  
    'HTS_I_percent':            0.6,        #       float,  
    'HTS_structural_percent':   0.03,       #       float,  
    
### Loss function 
    'loss_weight_normalization':     0,
## Weights
    'weight_bnormal':           1,             #        float,  
    'weight_length':            4e-3,          #        float,  
    'weight_curvature':         1e-4,          #        float,  
    'weight_curvature_max':     1e-4,          #        float,  
    'weight_torsion':           2e-4,          #        float,  
    'weight_torsion_max':       1e-5,          #        float,  
    'weight_distance_coil_coil':    4e-3,      #        float,  
    'weight_HTS_strain':            0,         #        float,  
    'weight_HTS_force_max':         1e-9,      #        float,  
    'weight_HTS_force_mean':        1e-9,      #        float,  
    'weight_HTS_Icrit':             0,         #        float,  

## Target values
    'target_length_mean':            0,        #        float,  
    'target_length_single':                    #        list
                            [0, 5, 5, 4.8],
    'target_curvature_max':     4,              #       float,  
    'target_torsion_max':       15,             #       float,  
    'target_distance_coil_coil':    0.1,        #       float,  
    'target_distance_coil_surface': 0.3,        #       float,  
    'target_HTS_strain':            0.008,      #       float,  
    'target_HTS_force_max':             0,      #       float, 

### Save
    'save_hdf5' :               1,          #       int,    
    'out_hdf5':                             #       str,    'h5'
        'results/QA/example.h5',   

    'save_makegrid' :           0,          #       int,    
    'out_coil_makegrid':                    #       str, 
        '',        

### Plot(More functions can be performed in post-processing: post/post_plot.py)
    'plot_coil':                21,         #       int,    0, 1, 11, 2, 21
    'number_points':            500,        #       int,    

    'plot_loss':                0,          #       int,   

    'plot_poincare':            0,          #       int,   
    'poincare_number':          25,         #       int,                                        
    'poincare_phi0':            0,          # (phi0)float, 
    'number_iter':              400,        #       int,  
    'number_step':              1,          #       int,  
}

with open('initfiles/init_args.json', 'w') as f:
    json.dump(args, f, indent=4)

optimize.main()

