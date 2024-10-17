# The input of the strain is individually optimized, 
# with deletions and minor modifications to the input parameters.

import sys
import json
sys.path.append('opt_strain')
import strain_opt

args = {
### Optimization
    'alpha_optimize':           1,          #       int,    
## Iteration 
    'iter_method':              'nlopt',    #       str,    'nlopt'
# nlopt
    'nlopt_algorithm':          'LD_MMA',   #       str,    https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
    'stop_criteria':             1e-6,      #       float,  
                                                 
### Coil 
## Coil parameters
    'number_coils':             16,         # (nc)  int,                    
    'number_field_periods':     2,          # (nfp) int,                     
    'stellarator_symmetry':     1,          # (ss)  int,                    
    'number_segments':          128,        # (ns)  int,    
    
## Initial coil input        
    'coil_case':                'fourier',  #       str,    'spline' or 'fourier'
    'init_coil_option':         'fourier',  #       str,  'spline' or 'coordinates' or 'fourier'
    'init_coil_file':                       #       str,    
            'results/paper/QA/opt_2.h5',       

# Fourier
    'number_fourier_coils':        6,       # (nfc) int,    

# Bspline
    'number_control_points':    24,         # (ncp) int,    

## Finite-build 
    'length_calculate':         0,          #       int,    
    'length_normal':            [0.05 for i in range(4)],       # (ln)  list,              
    'length_binormal':          [0.05 for i in range(4)],       # (lb)  list,                    
    'number_normal':            2,          # (nn)  int,                     
    'number_binormal':          2,          # (nb)  int,         

## Rotation Angle 
    'init_fr_case':             1,          #       int,    
    'init_fr_file':                         #       str,    
            'results/paper/QA/opt_2.h5',               
    'number_rotate':            0,          # (nr)  int,                      
    'number_fourier_rotate':    6,          # (nfr) int,    

### HTS tape(REBCO)
    'material':                 'REBCO_LT', # str,    'REBCO_LT'(<60K) or 'REBCO_HT'(>68K)
    'HTS_single_width':         4e-3,       #       float,  
    'HTS_single_thickness':     1.2e-6,     #       float,  
    
# loss function 
    'target_HTS_strain':        0.004,      #  float,  
    # Mean strain

### Save
    'save_hdf5' :               1,          #       int,    'h5'
    'out_hdf5':                             #       str,   
        'results/paper/QA/opt_2s.h5',   

### Plot
    'number_points':            500,        #       int,                    

# ----------------- The parameter is not required --------------------- #
# ------- However, some data is used for initialization or saving. ------------ #
# ------------------------------------------------------------------------------------- #



## Coil Current
    'current_I':                [1e5],      #       list,   
    'total_current_I':          0,          #       int,    

## Magnetic surface
    'number_theta':             64,         # (nt)  int,    \theta(poloidal)                 
    'number_zeta':              128,        # (nz)  int,    \zeta(toroidal)                       
    'surface_file':                    #       str,   
            'initfiles/QA/plasma_no_well.boundary',

## Background magnetic field (from plasma current) 
    'Bn_background':                 0,          #       int,    
    'Bn_background_file':                        #       str,    
            'initfiles/ncsx_c09r00/c09r00.boundary',
   
### HTS tape(REBCO)
    'HTS_temperature':          4.2,        # (T)   float,  
    'HTS_I_percent':            0.6,        #       float,  
    'HTS_structural_percent':   0.01,       #       float,  
##
    'target_distance_coil_coil':    0.1,        #       float,  
    'target_distance_coil_surface': 0.3,        #       float,            
    'save_makegrid' :           0,          #       int,    

}



with open('initfiles/init_args.json', 'w') as f:
    json.dump(args, f, indent=4)


strain_opt.main()

