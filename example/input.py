
## Summary of required initial parameters,
## Write them in dict{} and save as json file 'initfiles/init_args.json'.
import sys
import json
sys.path.append('opt_coil')
import optimize

args = {

## Abbreviation for some parameter names in the program body.
## When saving an hdf5 file, you cannot have the value: None.

##   Parameter Name (key)      value       Abbreviation    data type     explain

### Optimization
## Optimization parameters
    'coil_optimize':                1,          #       int,    
    # Whether to optimize coil centerline position. 0: no, 1: yes
    'alpha_optimize':               0,          #       int,   
    # Whether to optimize the Rotation Angle of finite-build coil. 0: no, 1: yes
    'I_optimize':                   0,          #       int,    
    # Whether to optimize the coil current. 0: no, 1: yes

## Iteration 
    'iter_method':                  'nlopt',    #       str,    
    # Choose from: 'nlopt', 'min' and 'jax'. Recommend 'nlopt'.

# 'nlopt' :   Set in 'opt_coil/read_init.py/nlopt_op' and 'opt_coil/optimize.py' line 182.
    'nlopt_algorithm':              'LD_MMA',   #       str,    (only set:MMA, CCSAQ and LBFGS), 
    # from https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
    'stop_criteria':                1e-6,       #       float,  
    # set_ftol_rel   
    'inequality_constraint_strain': 0,          #       int,    
    # Whether to turn on the inequality constraint of strain (only set for nlopt).

# 'min' : Set in 'opt_coil/optimize.py' line 174.
    'minimize_method':              'CG',       #       str,    
    # from https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    'minimize_tol':                 1e-6,       #       float,  
    # Tolerance for termination.                                

# 'jax', Set in 'opt_coil/read_init.py/jax_op'.
    'number_iteration':             0,          #       int,    
    # The number of optimizer iterations.
    # If set to 0, does not iterate and obtains the result of the initial parameters.
    'optimizer':                    'momentum', #       str,    (only set:sgd, momentum and adam), 
    # from https://jax.readthedocs.io/en/latest/jax.example_libraries.optimizers.html#jax.example_libraries.optimizers.constant
    'step_size':                    1e-5,       #       float,  
    # Iteration step size for parameters.                  
    

### Coil 
## Coil parameters
    'number_coils':                 16,         # (nc)  int,    
    # The number of all coils.                   
    'number_field_periods':         2,          # (nfp) int,    
    # The periods of magnetic field.                
    'stellarator_symmetry':         1,          # (ss)  int,    
    # If stellarator_symmetry,  1:yes, 0:no.                
    # The 'number_independent_coils' = nc/nfp/(ss+1) for modular coils               
    'number_segments':              80,         # (ns)  int,    
    # Number of segments per coil.
    
## Initial coil input     
    'coil_case':                    'fourier',  #       str,    'spline' or 'fourier' or 'spline_local'
    # Coil representation.
    'init_coil_option':             'circle',   #       str,    'fourier' or 'spline' or 'coordinates' or 'circle'
    # Source of initial coil. 'circle' is generated automatically from the magnetic surface. Others require input files.
    'circle_coil_radius':           1.3,        #       float,  
    # The initial radius of 'circle'.   
    'init_coil_file':                           #       str,    'npy' or 'h5' or 'makegrid'
                    'initfiles/w7x/w7x_coil_5.npy',    
    # Initial coil file name, if not 'circle'.
    # The file name suffix should be 'npy' or 'h5', otherwise regarded as 'makegrid'.              

## Fourier
    'number_fourier_coils':         6,          # (nfc) int,    
    # The fourier modes of the coils.
                     
## Bspline
    'number_control_points':        32,         # (ncp) int,    
    # The control points of the coils.  Contains three start-end repetition points.           

## Bspline Local optimization
    'optimize_location_nic':        [0],        #       list,   
    # The number of coils for local optimization is given by the list, and the first coil starts from 0.  
    'optimize_location_ns':     [[[0, 64]]],    #       list,   
    # Locally optimized coil position, the list gives the specific segment of each coil.            
    # The first order is the number of coils, the second order is that a coil is divided into several segments, 
    # and the third order is the actual position of the starting and ending points of each segment.            
    # The distance between different sections of same coil should > 3, otherwise it is connected together.
    
    # For example, you want to locally optimize the 10~20 segment of first coil and 5~14 and 29~54 of third coil:
    # 'optimize_location_nic':  [0,2]
    # 'optimize_location_ns':   [[[10, 20]],[[5,14],[29,54]]]

## Finite-build 
    'length_calculate':             0,          #       int,    
    # The cross-section size is given by 0：input/ 1: calculate from the Jcrit of the HTS coils.
    'length_normal':                [0.16 for i in range(5)],       # (ln)  list,                   
    'length_binormal':              [0.16 for i in range(5)],       # (lb)  list, 
    # The unit length of the normal and binormal directions(v1, v2) for coils, they can be different.
    # If 'length_calculate'==1, it needs a bigger value than actual.   
    'number_normal':                1,          # (nn)  int,                   
    'number_binormal':              1,          # (nb)  int,    
    # The number of filament of the normal and binormal directions(v1, v2) for coils.
    # If set to 1, there is no finite-build. (Both are 1 or both bigger than 1)
                     
## Rotation Angle 
    'init_fr_case':                 0,          #       int,    
    # Initial fr, 0: generate items to 0, 1: read from file.
    'init_fr_file':                             #       str,    'npy' or 'h5'
                'results_f/circle/s1_fr.npy',               
    # The file that gives the initial value of fr.
    'number_rotate':                0,          # (nr)  int,    
    # The number of half rotations of the finite-build coil, which is normally set to zero                    
    'number_fourier_rotate':        6,          # (nfr) int,   
    # The number of Fourier modes of the rotation angle.

## Coil Current
    'current_I':                    [1.62e6],   #       list,   
    # Initial coil current, list length is 1 or number of modular coils.  
    'total_current_I':              1.62e6*5,          #       int,    
    # If 'I_optimize'==1, there are two ways to set the current.
    # 0: no total current limit; if not 0, it should be the sum of modular coils current.(Recommended)

## Magnetic surface
    'number_theta':                 40,         # (nt)  int,                        
    'number_zeta':                  64,         # (nz)  int,       
    # The number of grid points \theta(poloidal) and \zeta(toroidal) on the magnetic surface.
    # The 'number_zeta' is the number for one period.                  
    'surface_file':                             #       str,   
                'initfiles/w7x/plasma.boundary',
    # Magnetic surface data file.

## Background magnetic field (from plasma current) 
    'Bn_background':                    0,          #       int,    
    # If there is background magnetic field, 0:no, 1:yes.
    'Bn_background_file':                           #       str,    
                'initfiles/ncsx_c09r00/c09r00.boundary',


### HTS tape(REBCO) or LTS
    'material':                     'REBCO_LT', #       str,  'REBCO_LT'(<60K), 'REBCO_HT'(>68K),'NbTi', 'Nb3Sn'    
    'HTS_single_width':             4e-3,       #       float,  
    'HTS_single_thickness':         5e-5,       #       float,  
    # Single width and thickness of HTS tape.
    'HTS_temperature':              4.2,        #       float,  
    # Temperature of HTS in Kelvin. 
    'HTS_I_percent':                0.6,        #       float, 
    # The ratio of the coil current to the Icrit of the HTS. 
    'HTS_structural_percent':       0.03,       #       float,  
    # The proportion of the cross-sectional area of the HTS tape.
    
    
### Loss function 
    'loss_weight_normalization':    0,
    # Whether to normalize the weights, such as 'weight_length_new' = f_length/'weight_length'
## Weights
    'weight_bnormal':               1,          #       float,  
    # The normal field error, is usually set to 1
    'weight_length':                0.001,      #       float,   
    'weight_curvature':             0,          #       float,   
    'weight_curvature_max':         0,          #       float,  
    'weight_torsion':               0,          #       float,   
    'weight_torsion_max':           0,          #       float,  
    'weight_distance_coil_coil':    0,          #       float,   
    'weight_distance_coil_surface': 0,          #       float,   
    'weight_HTS_strain':            0,          #       float,  
    # Here is the average strain.
    # To constrain the maximum strain, inequality constraints(nlopt) are usually used.
    'weight_HTS_force_max':         0,          #       loat,  
    # The maximum value of the electromagnetic force per unit length.
    'weight_HTS_force_mean':        0,          #       float,   
    # The integration of the electromagnetic force along the coil.
    'weight_HTS_Icrit':             0,          #       float,  
    # The critical current of the HTS coils, usually not necessary and set to 0.

## Target values
    'target_length_mean':           0,          #       float,  
    # target length for all coils, 
    # 0: f_length = np.mean(length), not 0: f_length = (length-target)**2
    'target_length_single':                     #       list,   
                            [0,1.5,1.5,1.5],
    # The target length of each coil when the first term is not 0, otherwise not considered.
    'target_curvature_max':         2.36,       #       float,  
    'target_torsion_max':           5,          #       float,  
    'target_distance_coil_coil':    0.25,       #       float,  
    'target_distance_coil_surface': 0.44,       #       float,  
    'target_HTS_strain':            0.0039,     #       float,  
    'target_HTS_force_max':         0,          #       float, 

### Save
    # 0：not save, 1：save. 

    'save_hdf5' :                   0,          #       int,    
    # Save hdf5 file with most parameters.
    'out_hdf5':                                 #       str,   
                'results/test/lossno_test/f2n4_1.h5',       

    'save_makegrid' :               0,          #       int,  
    # Save the coil makegrid file.  
    'out_coil_makegrid':                        #       str,    
                '',       

### Plot(More functions can be performed in post-processing: post/post_plot.py)
    'plot_coil':                    11,         #       int,    0, 1, 11, 2, 21
    # Whether to draw the coil, 0: do not draw, 1: draw the coil points, 2: draw finite-build coils.
    # Draw the coils with a magnetic surface: 11 or 21. 
    'number_points':                500,        #       int,    
    # The number of scattered points when drawing the coil

    'plot_loss':                    0,          #       int,    0：no, 1：yes
    # The value(all steps, not real steps) with the number of iterations.

    'plot_poincare':                0,          #       int,    0：no, 1：yes
    # Whether to draw the poincare plot.
    'poincare_number':              25,         #       int,    
    # The number of circles of the magnetic surface.                                    
    'poincare_phi0':                0,          # (phi0)float,  
    #  The toroidal angle where the poincare plot data saved.
    'number_iter':                  200,        #       int,  
    # Number of toroidal periods in tracing.
    'number_step':                  1,          #       int,  
    # Number of intermediate step for one period.  
}

with open('initfiles/init_args.json', 'w') as f:
    json.dump(args, f, indent=4)

optimize.main()

