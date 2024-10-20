

import sys
import json
sys.path.append('opt_coil')
import optimize

args = {

### Optimization
    'coil_optimize':            1,          #       int,    
    'alpha_optimize':           1,          #       int,    
    'I_optimize':               0,          #       int,    

## Iteration 
    'iter_method':              'nlopt',      #       str,   'jax', 'min', 'nlopt', #'for-min', 'min-for'
# nlopt
    'nlopt_algorithm':          'LD_MMA',   #       str,     https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
    'stop_criteria':             1e-6,      #       float,  
    'inequality_constraint_strain':1,
# min
    'minimize_method':          'CG',       #       str,   https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    'minimize_tol':             1e-2,       #   float,                                  
# jax
    'number_iteration':         0,          #      int,    
    'optimizer':                'momentum',      #    str,    https://jax.readthedocs.io/en/latest/jax.example_libraries.optimizers.html#jax.example_libraries.optimizers.constant
    'step_size':                1e-5,            #       float,                       
    
### Coil 
## Coil parameters
    'number_coils':             50,         # (nc)  int,           
    'number_field_periods':     5,          # (nfp) int,                   
    'stellarator_symmetry':     1,          # (ss)  int,    
    'number_segments':          128,        # (ns)  int,    
    
## Initial coil input       
    'coil_case':                'fourier',  #       str,     'spline' or 'fourier' or 'spline_local'
    'init_coil_option':         'coordinates',  #       str,  'spline' or 'coordinates' or 'fourier' or 'circle'
    'circle_coil_radius':       1.3,        #       float,  
    'init_coil_file':                       #       str,    
            'initfiles/w7x/w7x_coil_5.npy',       


# Fourier
    'number_fourier_coils':     6,          # (nfc) int,   
                     
# Bspline
    'number_control_points':    32,         # (ncp) int,    

# # Bspline Local optimization
    'optimize_location_nic':    [0],        #       list,   
    'optimize_location_ns':                 #       list,   
                                [[[0, 64]]],               

## Finite-build 
    'length_calculate':         0,          #       int,    
    'length_normal':            [0.16 for i in range(5)],       # (ln)  list,  
    'length_binormal':          [0.16 for i in range(5)],       # (lb)  list,              
    'number_normal':            2,          # (nn)  int,            
    'number_binormal':          2,          # (nb)  int,                 
                     
## Rotation Angle 
    'init_fr_case':             0,          #       int,   
    'init_fr_file':                         #       str,    
            'results_f/circle/s1_fr.npy',               
    'number_rotate':            0,          # (nr)  int,    描述线圈绕组组的半旋转数的整数,通常设为0                    
    'number_fourier_rotate':    6,          # (nfr) int,    每个线圈的旋转的傅里叶分量的个数    

## Coil Current
    'current_I':                [1.62e6],   #       list,   
    'total_current_I':          0,          #       int,   

## Magnetic surface
    'number_theta':             40,         # (nt)  int,    \theta(poloidal)                             
    'number_zeta':              128,        # (nz)  int,    \zeta(toroidal)                    
    'surface_file':                         #       str,   
            'initfiles/w7x/plasma.boundary',

## Background magnetic field (from plasma current) 
    'Bn_background':                 0,          #       int,    
    'Bn_background_file':                        #       str,    
            'initfiles/ncsx_c09r00/c09r00.boundary',

### HTS tape(REBCO) or LTS
    'material':                 'REBCO_LT', #       str,    'REBCO_LT'(<60K), 'REBCO_HT'(>68K),'NbTi', 'Nb3Sn'    
    'HTS_single_width':         4e-3,       #       float,  
    'HTS_single_thickness':     5e-5,       #       float, 
    'HTS_temperature':          4.2,        #       float,  
    'HTS_I_percent':            0.6,        #       float,  
    'HTS_structural_percent':   0.03,       #       float, 

### Loss function 
    'loss_weight_normalization':     0,         #       int,    
## Weights
    'weight_bnormal':           1,              #       float,      
    'weight_length':            0.05,           #       float,      
    'weight_curvature':         0.0002,         #       float,      
    'weight_curvature_max':     0.001,          #       float,      
    'weight_torsion':           0.0002,         #       float,      
    'weight_torsion_max':       0.001,          #       float,      
    'weight_distance_coil_coil':    0.001,      #       float,      
    'weight_distance_coil_surface': 0.01,       #       float,      
    'weight_HTS_strain':            0,          #       float,      
    'weight_HTS_force_max':      4e-9,          #       float,      
    'weight_HTS_force_mean':     9e-9,          #       float,        
    'weight_HTS_Icrit':             0,          #       float,       

## Target values
    'target_length_mean':            8.46,      #       float,      
    'target_length_single':                     #       list 
                            [0,1.5,1.5,1.5],
    'target_curvature_max':     2.36,           #       float,   
    'target_torsion_max':       0,              #       float,  
    'target_distance_coil_coil':    0.23,       #       float,   
    'target_distance_coil_surface': 0.4,        #       float,    
    'target_HTS_strain':            0.004,      #       float,      
    'target_HTS_force_max':         6.9e6,      #       float,    

### Save 
    'save_hdf5' :               1,          #       int,    
    'out_hdf5':                             #       str,    'h5'
        'results/w7x/1.h5',   

    'save_makegrid' :           0,          #       int,       
    'out_coil_makegrid':                    #       str,    
        'results/w7x/compare/filameat/makegrid',                

### Plot(More functions can be performed in post-processing: post/post_plot.py)
    'plot_coil':                21,         #       int,    0, 1, 11, 2, 21
    'number_points':            500,        #       int,    

    'plot_loss':                0,          #       int,    

    'plot_poincare':            0,          #       int,   
    'poincare_number':          25,         #       int,                                            
    'poincare_phi0':            0,          #       float,  
    'number_iter':              400,        #       int,  
    'number_step':              1,          #       int,  


}



with open('initfiles/init_args.json', 'w') as f:
    json.dump(args, f, indent=4)


optimize.main()

