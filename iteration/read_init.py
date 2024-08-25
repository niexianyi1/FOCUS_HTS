
### 获取初始线圈, 磁面
### 包含各种文件类型的读取


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
    根据args的参数给出初始条件, 包括线圈变量, 电流, 磁面等

    Args:
        args : dict,    参数总集

    Returns:
        args : dict,    新的参数总集， 添加了电流项和spline表示的非优化参数
        coil_arg_init : 表示线圈的变量, 类型由coil_case给定
        fr_init :       表示线圈旋转的变量
        surface_data :  磁面数据
        # B_extern :    额外磁场, 一般为背景磁场
    """
    args, surface_data = get_surface_data(args)
    args, coil_arg_init, fr_init = coil_init(args, surface_data)    # coil_arg：线圈参数, Fourier或spline表示
    args, I_init = init_I(args)
    args = get_finite_build_length(args, coil_arg_init, fr_init, I_init)
    if args['Bn_extern'] != 0:
        args = get_Bn_extern(args)
    args = test(args, coil_arg_init)

    return args, coil_arg_init, fr_init, surface_data, I_init


def args_to_op(args, optimizer, step_size):
    """
    根据给定的jax的梯度下降迭代方式, 获取迭代变量

    Args:
        args :      dict,   参数总集
        optimizer : str,    迭代方式
        step_size :        float,  迭代步长
    Returns:
        An (init_fun, update_fun, get_params) triple.
    """

    if optimizer == 'gd':
        return  op.sgd(step_size)
    if optimizer == 'sgd':
        return  op.sgd(step_size)
    if optimizer == 'momentum':
        return  op.momentum(step_size, args['momentum_mass'])
    if optimizer == 'adam':
        return  op.adam(step_size, args['momentum_mass'], args['var'], args['eps'])


def nlopt_op(args, params):
    if args['nlopt_algorithm'] == 'LD_MMA':
        return nlopt.opt(nlopt.LD_MMA, len(params))
    if args['nlopt_algorithm'] == 'LD_CCSAQ':
        return nlopt.opt(nlopt.LD_CCSAQ, len(params))
    if args['nlopt_algorithm'] == 'LD_LBFGS':
        return nlopt.opt(nlopt.LD_LBFGS, len(params))


def get_surface_data(args):           # surface data的获取
    """
    获取磁面参数
    Args:
        args : dict,    参数总集

    Returns:
        surface_data : 磁面参数的合集
            r :   array,  [nz, nt, 3], 磁面坐标
            nn :  array,  [nz, nt, 3], 磁面法向
            sg :  array,  [nz, nt, 3], 磁面面积
    """
    if args['surface_case'] == 0 :
        r = np.load(args['surface_r_file'])
        nn = np.load(args['surface_nn_file'])
        sg = np.load(args['surface_sg_file'])
    elif args['surface_case'] == 1 :
        args, r, nn, sg = read_file.read_plasma(args)
        
    surface_data = (r, nn, sg)
    return args, surface_data


def init_I(args):
    """
    获取初始电流
    Args:
        args : dict,    参数总集

    Returns:
        args : dict,  新的参数总集, 添加电流项

    """    
    nic = args['number_independent_coils']
    if args['coil_file_type'] == 'makegrid':
        I = args['makegrid_I']          
        
    else:
        current_I = args['current_I']
        I = np.zeros(nic)
        if args['current_independent'] == 0:
            I = I.at[:].set(current_I[0])
        elif args['current_independent'] == 1:
            for i in range(nic):
                I = I.at[i].set(current_I[i])
        
        if args['total_current_I'] != 0:
            assert args['total_current_I'] == np.sum(I)
        
    args['I_normalize'] = I[0]
    I_init = I / I[0]
    return args, I_init


def coil_init(args, surface_data):
    """
    获取线圈初始参数, 按照选择的表示方式给出
    Args:
        args : dict,    参数总集

    Returns:
        args :          dict,  新的参数总集, 添加对应表达式的非优化参数
        coil_arg_init : array, 线圈的初始优化参数
            fc_init :   array, [6, nic, nfc], fourier表示的初始参数
            c_init :    array, [nic, ncp-3], spline表示的初始参数, 不包含重复的控制点。
        fr_init :       array, [2, nic, nfr], 有限截面旋转的初始数据
    """    
    
    nic = args['number_independent_coils']
    nc = args['number_coils']
    ns = args['number_segments']
    assert nic * args['number_field_periods'] * (args['stellarator_symmetry'] + 1) == args['number_coils'] 
    
    surface, _, _ = surface_data
    
    ## 有限截面旋转角
    if args['init_fr_case'] == 0:
        fr_init = np.zeros((nic, 2, args['number_fourier_rotate'])) 
        
    elif args['init_fr_case'] == 1:
        file = args['init_fr_file'].split('.')
        if file[-1] == 'npy':
            fr_init = np.load(args['init_fr_file'])
        elif file[-1] == 'h5':
            arge = read_file.read_hdf5(args['init_fr_file'])
            fr_init = np.array(arge['coil_fr'])

    ## 线圈参数
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
            fc_init = fourier.compute_coil_fourierSeries(coil, args['number_fourier_coils'])

        elif args['init_coil_option'] == 'circle':
            coil = circle_coil(args, surface)
            fc_init = fourier.compute_coil_fourierSeries(coil, args['number_fourier_coils'])
        
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
    fa = fourier.compute_coil_fourierSeries(axis, nfc)
    axis = fourier.compute_r_centroid(fa, 2*nc)
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
        for j in range(lenai):          # 需要加判断避免超出边界 或 重复计入控制点
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
        args = section_length.solve_section(args, coil_arg_init, fr_init, I_init)
        print('Complete section calculation.')
    a = np.array(args['length_normal']) * (args['number_normal'] - 1)   # 正方形截面
    b = np.array(args['length_binormal']) * (args['number_binormal'] - 1)    
    k = (4*b/(3*a)*np.arctan(a/b) + 4*a/(3*b)*np.arctan(b/a) + b**2/(6*a**2)*np.log(b/a)+
            a**2/(6*b**2)*np.log(a/b) - (a**4-6*(a*b)**2+b**4)/(6*(a*b)**2)*np.log(a/b+b/a) )
    delta = np.exp(-25/6 + k)
    args['delta'] = delta
    return args    


def get_Bn_extern(args):
    if args['Bn_extern'] == 1:
        Bn = read_file.read_finite_beta_Bn(args)    
        args['Bn_extern_surface'] = Bn
    return args


def test(args, coil_arg_init):
    ## 应用梯度下降优化时, 优化项的步长不能为0
    if args['iter_method'] == 'jax':
        if args['coil_optimize'] == 0:
            args['step_size_coil'] = 0
        else:
            assert args['step_size_coil'] != 0  

        if args['alpha_optimize'] == 0:
            args['step_size_alpha'] = 0
        else:
            assert args['step_size_alpha'] != 0

        if args['I_optimize'] == 0:
            args['step_size_I'] = 0
        else:
            assert args['step_size_I'] != 0  
    
    ## fourier系数和spline系数
    if args['coil_case'] == 'fourier':
        assert args['number_fourier_coils'] == coil_arg_init.shape[2]
    else :
        assert args['number_control_points'] == coil_arg_init.shape[2] + 3

    ## 线圈最小间距应大于截面的长度
    nlen = np.max(np.array(args['length_normal'])) * (args['number_normal'] - 1)
    blen = np.max(np.array(args['length_binormal'])) * (args['number_binormal'] - 1)
    max_len = np.sqrt(nlen**2 + blen**2)
    assert args['target_distance_coil_coil'] > max_len or args['target_distance_coil_coil'] == 0
    assert args['target_distance_coil_surface'] > max_len/2 or args['target_distance_coil_surface'] == 0
    
    print('Complete test.')
    return args


