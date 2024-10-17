
### Initialization of the program.



import jax.numpy as np
import jax.example_libraries.optimizers as op
import fourier
import spline
import read_file
import coilset
import nlopt
import sys
sys.path.append('HTS')
import section_length

def init(args):
    """
    Initialization of the program.

    Returns:
        args            : dict,     new args with additional information.
        coil_arg_init   : array,    Represents the coil centerline.
        fr_init         : array,    Represents the rotation of the coil.
        surface_data    : tuple,    Magnetic surface data
        Bn_background   : array,    background magnetic field
    """
    args['number_independent_coils'] = int(args['number_coils'] / 
                    args['number_field_periods'] / (args['stellarator_symmetry'] + 1))
    args, surface_data = get_surface_data(args)
    args, coil_arg_init, fr_init = coil_init(args, surface_data)  
    args, I_init = init_I(args)
    args = get_finite_build_length(args, coil_arg_init, fr_init, I_init)
    if args['Bn_background'] != 0:
        args = get_Bn_background(args)
    args = test(args, coil_arg_init)

    return args, coil_arg_init, fr_init, surface_data, I_init


def jax_op(args):
    """ jax optimize """
    if args['coil_optimize'] == 0:
        step_size_coil = 0
    else:
        step_size_coil = args['step_size']  

    if args['alpha_optimize'] == 0:
        step_size_alpha = 0
    else:
        step_size_alpha = args['step_size']  

    if args['I_optimize'] == 0:
        step_size_I = 0
    else:
        step_size_I = args['step_size']  

    if args['optimizer'] == 'sgd':
        return  (op.sgd(step_size_coil), op.sgd(step_size_alpha), op.sgd(step_size_I))
    if args['optimizer'] == 'momentum':
        return  (op.momentum(step_size_coil, 0.9),
                 op.momentum(step_size_alpha, 0.9),
                 op.momentum(step_size_I, 0.9))
    if args['optimizer'] == 'adam':
        return  (op.adam(step_size_coil), op.adam(step_size_alpha), op.adam(step_size_I))


def nlopt_op(args, params):
    ''' nlopt optimize '''
    if args['nlopt_algorithm'] == 'LD_MMA':
        return nlopt.opt(nlopt.LD_MMA, len(params))
    if args['nlopt_algorithm'] == 'LD_CCSAQ':
        return nlopt.opt(nlopt.LD_CCSAQ, len(params))
    if args['nlopt_algorithm'] == 'LD_LBFGS':
        return nlopt.opt(nlopt.LD_LBFGS, len(params))


def get_surface_data(args):          
    args, r, nn, sg = read_file.read_plasma(args)
    surface_data = (r, nn, sg)
    return args, surface_data


def init_I(args):  
    nic = args['number_independent_coils']
    if 'makegrid_I' in args.keys():
        I = args['makegrid_I']  
        
    else:
        current_I = np.array(args['current_I'])
        assert len(current_I) == 1 or len(current_I) == nic
        I = np.zeros(nic)
        I = I.at[:].set(current_I)

    if args['total_current_I'] != 0:
        assert args['total_current_I'] == np.sum(I)
        
    args['I_normalize'] = I[-1]
    I_init = I / I[-1]
    return args, I_init


def coil_init(args, surface_data):

    nic = args['number_independent_coils']
    nc = args['number_coils']
    ns = args['number_segments']

    surface, _, _ = surface_data
    
    ## Rotation Angle
    if args['init_fr_case'] == 0:
        fr_init = np.zeros((nic, 2, args['number_fourier_rotate'])) 
        
    elif args['init_fr_case'] == 1:
        file = args['init_fr_file'].split('.')
        if file[-1] == 'npy':
            fr_init = np.load(args['init_fr_file'])
        elif file[-1] == 'h5':
            arge = read_file.read_hdf5(args['init_fr_file'])
            fr_init = np.array(arge['coil_fr'])

    ## coil centerline
    if args['coil_case'] == 'fourier':
        if args['init_coil_option'] == 'fourier':
            file = args['init_coil_file'].split('.')
            if file[-1] == 'npy':
                fc_init = np.load(args['init_coil_file'])
            elif file[-1] == 'h5':
                arge = read_file.read_hdf5(args['init_coil_file'])
                fc_init = np.array(arge['coil_arg'])

        elif args['init_coil_option'] == 'coordinates':
            file = args['init_coil_file'].split('.')
            if file[-1] == 'npy':
                coil = np.load(args['init_coil_file'])
            elif file[-1] == 'h5':
                arge = read_file.read_hdf5(args['init_coil_file'])
                coil = np.array(arge['coil_centroid'])
            else:
                coil, I = read_file.read_makegrid(args['init_coil_file'], nc, nic)         
                args['makegrid_I'] = I 
            fc_init = fourier.calculate_coil_fourierSeries(coil, args['number_fourier_coils'])

        elif args['init_coil_option'] == 'circle':
            coil = circle_coil(args, surface)
            fc_init = fourier.calculate_coil_fourierSeries(coil, args['number_fourier_coils'])
        
        coil_arg_init = fc_init
        return args, coil_arg_init, fr_init

    elif args['coil_case'] == 'spline':
        if args['init_coil_option'] == 'spline':
            file = args['init_coil_file'].split('.')
            if file[-1] == 'npy':
                c_init = np.load(args['init_coil_file'])
            elif file[-1] == 'h5':
                arge = read_file.read_hdf5(args['init_coil_file'])
                c_init = np.array(arge['coil_arg'])
            assert args['number_control_points'] == c_init.shape[2]
            bc, tj = spline.get_bc_init(ns, args['number_control_points'])

        elif args['init_coil_option'] == 'coordinates':
            file = args['init_coil_file'].split('.')
            if file[-1] == 'npy':
                coil = np.load(args['init_coil_file'])
            elif file[-1] == 'h5':
                arge = read_file.read_hdf5(args['init_coil_file'])
                coil = np.array(arge['coil_centroid'])
            else:
                coil, I = read_file.read_makegrid(args['init_coil_file'], nc, nic)           
                args['makegrid_I'] = I 
            c_init, bc, tj = spline.get_c_init(coil, nic, ns, args['number_control_points'])

        elif args['init_coil_option'] == 'circle':
            coil = circle_coil(args, surface)
            c_init, bc, tj = spline.get_c_init(coil, nic, ns, args['number_control_points'])

        args['bc'] = bc
        args['tj'] = tj
        coil_arg_init = c_init[:, :, :-3]
        return args, coil_arg_init, fr_init

    elif args['coil_case'] == 'spline_local':
        if args['init_coil_option'] == 'spline':
            file = args['init_coil_file'].split('.')
            if file[-1] == 'npy':
                c_init = np.load(args['init_coil_file'])
            elif file[-1] == 'h5':
                arge = read_file.read_hdf5(args['init_coil_file'])
                c_init = np.array(arge['coil_arg'])
            assert args['number_control_points'] == c_init.shape[2]
            bc, tj = spline.get_bc_init(ns, args['number_control_points'])

        elif args['init_coil_option'] == 'coordinates':
            file = args['init_coil_file'].split('.')
            if file[-1] == 'npy':
                coil = np.load(args['init_coil_file'])
            elif file[-1] == 'h5':
                arge = read_file.read_hdf5(args['init_coil_file'])
                coil = np.array(arge['coil_centroid'])
            else:
                coil, I = read_file.read_makegrid(args['init_coil_file'], nc, nic)           
                args['makegrid_I'] = I 
            c_init, bc, tj = spline.get_c_init(coil, nic, ns, args['number_control_points'])
        args['bc'] = bc
        args['tj'] = tj
        coil_arg_init = local_coil(args, c_init)
        args['c_init'] = c_init
        return args, coil_arg_init, fr_init


def circle_coil(args, surface):
    ''' initial coils '''
    nfc = args['number_fourier_coils']
    nz = args['number_zeta']
    r = args['circle_coil_radius']
    nic = args['number_independent_coils']
    ns = args['number_segments']
    nc = args['number_coils']
    axis = np.zeros((nz + 1, 3))
    axis = axis.at[:-1].set(np.mean(surface, axis = 1))
    axis = axis.at[-1].set(axis[0])
    axis = axis[np.newaxis, :, :]
    fa = fourier.calculate_coil_fourierSeries(axis, nfc)
    axis = fourier.calculate_r_centroid(fa, 2*nc)
    axis = np.squeeze(axis)[:-1]

    circlecoil = np.zeros((nic, ns+1, 3))
    zeta = np.linspace(0, 2 * np.pi, nc + 1) + np.pi/(nc+1)
    theta = np.linspace(0, 2 * np.pi, ns + 1)
    for i in range(nic):
        axis_center = axis[2*(i+1)]
        R = (axis_center[0]**2 + axis_center[1]**2)**0.5
        x = (R + r * np.cos(theta)) * np.cos(zeta[i])
        y = (R + r * np.cos(theta)) * np.sin(zeta[i])
        z = r * np.sin(theta) + axis_center[2]
        circlecoil = circlecoil.at[i].set(np.transpose(np.array([x, y, z])))

    return circlecoil


def local_coil(args, c_init):
    nic = args['number_independent_coils']
    a = args['optimize_location_ns']
    loc = args['optimize_location_nic']
    lo_nc = len(loc) 
    assert lo_nc <= nic
    assert lo_nc == len(a)
    coil_arg_init = []
    for i in range(lo_nc):
        lenai = len(a[i])
        coil_args = []
        for j in range(lenai):          
            start = int(a[i][j][0])
            end = int(a[i][j][1])
            coil_arg = [c_init[int(loc[i]), :, start:end]]
            coil_args.append(coil_arg)
        coil_arg_init.append(coil_args)
    return coil_arg_init


def get_finite_build_length(args, coil_arg_init, fr_init, I_init):
    nic = args['number_independent_coils']
    if len(args['length_normal']) != nic:
        ln = np.max(np.array(args['length_normal']))
        args['length_normal'] = [ln for i in range(nic)]
    if len(args['length_binormal']) != nic:
        lb = np.max(np.array(args['length_binormal']))
        args['length_binormal'] = [lb for i in range(nic)]  
    if args['length_calculate'] == 1:
        assert args['number_normal'] > 1 and args['number_binormal'] > 1
        args = section_length.solve_section(args, coil_arg_init, fr_init, I_init)
        print('Complete section calculation.')
    a = np.array(args['length_normal']) * (args['number_normal'] - 1)  
    b = np.array(args['length_binormal']) * (args['number_binormal'] - 1)    
    k = (4*b/(3*a)*np.arctan(a/b) + 4*a/(3*b)*np.arctan(b/a) + b**2/(6*a**2)*np.log(b/a)+
            a**2/(6*b**2)*np.log(a/b) - (a**4-6*(a*b)**2+b**4)/(6*(a*b)**2)*np.log(a/b+b/a) )
    delta = np.exp(-25/6 + k)
    args['delta'] = delta
    return args    


def get_Bn_background(args):
    if args['Bn_background'] == 1:
        Bn = read_file.read_finite_beta_Bn(args)    
        args['Bn_background_surface'] = Bn
    return args


def test(args, coil_arg_init):

    if args['coil_case'] == 'fourier':
        assert args['number_fourier_coils'] == coil_arg_init.shape[2]
    else :
        assert args['number_control_points'] == coil_arg_init.shape[2] + 3

    assert args['number_independent_coils'] == coil_arg_init.shape[0]
    if args['number_normal'] == 1:
        assert  args['number_binormal'] == 1
    elif args['number_normal'] > 1:
        assert  args['number_binormal'] > 1

    nlen = np.max(np.array(args['length_normal'])) * (args['number_normal'] - 1)
    blen = np.max(np.array(args['length_binormal'])) * (args['number_binormal'] - 1)
    max_len = np.sqrt(nlen**2 + blen**2)
    assert args['target_distance_coil_coil'] > max_len or args['target_distance_coil_coil'] == 0
    assert args['target_distance_coil_surface'] > max_len/2 or args['target_distance_coil_surface'] == 0
    
    print('Complete test.')
    return args


