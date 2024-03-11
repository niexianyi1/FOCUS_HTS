
### 获取初始线圈, 磁面
### 包含各种文件类型的读取


import jax.numpy as np
import jax.example_libraries.optimizers as op
import fourier
import spline
import create_circle
import read_plasma 
import read_file


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
    
    I_init = init_I(args)
    surface_data, args = get_surface_data(args)
    args, coil_arg_init, fr_init = coil_init(args, surface_data)    # coil_arg：线圈参数, Fourier或spline表示
    if args['Bn_extern'] != 0:
        Bn_extern = get_Bn_extern(args)
        args['Bn_extern_surface'] = Bn_extern

    return args, coil_arg_init, fr_init, surface_data, I_init


def args_to_op(args, optimizer, lr):
    """
    根据给定的jax的梯度下降迭代方式, 获取迭代变量

    Args:
        args :      dict,   参数总集
        optimizer : str,    迭代方式
        lr :        float,  迭代步长
    Returns:
        An (init_fun, update_fun, get_params) triple.
    """

    if optimizer == 'gd':
        return  op.sgd(lr)
    if optimizer == 'sgd':
        return  op.sgd(lr)
    if optimizer == 'momentum':
        return  op.momentum(lr, args['momentum_mass'])
    if optimizer == 'adam':
        return  op.adam(lr, args['momentum_mass'], args['var'], args['eps'])


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
        r, nn, sg, args = read_plasma.plasma_surface(args)
        
    surface_data = (r, nn, sg)
    return surface_data, args


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
    assert args['number_independent_coils'] == int(args['number_coils'] / 
                        args['number_field_periods'] / (args['stellarator_symmetry'] + 1))
    
    nic = args['number_independent_coils']
    ns = args['number_segments']
    surface, _, _ = surface_data
    
    ## 有限截面旋转角
    if args['init_fr_case'] == 0:
        fr_init = np.zeros((2, nic, args['number_fourier_rotate'])) 
    elif args['init_fr_case'] == 1:
        fr_init = np.load("{}".format(args['init_fr_file']))

    ## 总变量
    if args['coil_case'] == 'fourier':
        if args['init_coil_option'] == 'fourier':
            fc_init = np.load("{}".format(args['init_coil_file']))

        elif args['init_coil_option'] == 'coil':
            if args['coil_file_type'] == 'npy':
                coil = np.load("{}".format(args['init_coil_file']))
            if args['coil_file_type'] == 'makegrid':
                coil = read_file.read_makegrid(args['init_coil_file'], nic, ns)
            fc_init = fourier.compute_coil_fourierSeries(nic, ns, args['num_fourier_coils'], coil)

        elif args['init_coil_option'] == 'circle':
            coil = create_circle.circle_coil(args, surface)
            fc_init = fourier.compute_coil_fourierSeries(nic, ns, args['num_fourier_coils'], coil)
        
        coil_arg_init = fc_init
        return args, coil_arg_init, fr_init

    elif args['coil_case'] == 'spline':
        if args['init_coil_option'] == 'spline':
            c_init = np.load("{}".format(args['init_coil_file']))
            assert args['number_control_points'] == c_init.shape[2]
            bc, tj = spline.get_bc_init(ns, args['number_control_points'])

        elif args['init_coil_option'] == 'coil':
            if args['coil_file_type'] == 'npy':
                coil = np.load("{}".format(args['init_coil_file']))
            if args['coil_file_type'] == 'makegrid':
                coil = read_file.read_makegrid(args['init_coil_file'], nic, ns)           
            c_init, bc, tj = spline.get_c_init(coil, nic, ns, args['number_control_points'])

        elif args['init_coil_option'] == 'circle':
            coil = create_circle.circle_coil(args, surface)
            c_init, bc, tj = spline.get_c_init(coil, nic, ns, args['number_control_points'])

        args['bc'] = bc
        args['tj'] = tj
        coil_arg_init = c_init[:, :, :-3]
        return args, coil_arg_init, fr_init

    elif args['coil_case'] == 'spline_local':
        if args['init_coil_option'] == 'spline':
            c_init = np.load("{}".format(args['init_coil_file']))
            assert args['number_control_points'] == c_init.shape[2]
            bc, tj = spline.get_bc_init(ns, args['number_control_points'])

        elif args['init_coil_option'] == 'coil':
            if args['coil_file_type'] == 'npy':
                coil = np.load("{}".format(args['init_coil_file']))
            if args['coil_file_type'] == 'makegrid':
                coil = read_file.read_makegrid(args['init_coil_file'], nic, ns)           
            c_init, bc, tj = spline.get_c_init(coil, nic, ns, args['number_control_points'])
        args['bc'] = bc
        args['tj'] = tj
        coil_arg_init = local_coil(args, c_init, ns, nic)
        args['c_init'] = c_init
        return args, coil_arg_init, fr_init


def local_coil(args, c_init, ns, nic):
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
            if end <= ns-3:
                coil_arg = [c_init[int(loc[i]), :, start:end+3]]
            elif end > ns-3:
                coil_arg = [c_init[int(loc[i]), :, start:-3]]
            coil_args.append(coil_arg)
        coil_arg_init.append(coil_args)
    return coil_arg_init


def init_I(args):
    """
    获取初始电流
    Args:
        args : dict,    参数总集

    Returns:
        args : dict,  新的参数总集, 添加电流项

    """    
    nic = args['number_independent_coils']
    current_I = args['current_I']
    I = np.zeros(nic)
    if args['current_independent'] == 0:
        I = I.at[:].set(current_I[0])
    elif args['current_independent'] == 1:
        for i in range(nic):
            I = I.at[i].set(current_I[i])
    if args['I_optimize'] == 0:
        args['learning_rate_I'] = 0
    else:
        assert args['learning_rate_I'] != 0

    return I


def get_Bn_extern(args):
    if args['Bn_extern'] == 1:
        BNC, BNS, MZ, MT, Nfp = read_plasma.read_finite_beta(args['Bn_extern_file'])
        zeta = np.linspace(0,2 * np.pi, args['number_zeta'] + 1)[:-1]
        theta = np.linspace(0, 2 * np.pi, args['number_theta'] + 1)[:-1]
        bn = np.zeros((args['number_zeta'], args['number_theta']))
        for mz in range(-MZ, MZ + 1):
            for mt in range(MT):
                bn += BNS[mz + MZ, mt] * np.sin( mt * theta[np.newaxis, :] - mz * Nfp * zeta[:, np.newaxis] )
    return bn


def test(args):
    nic = args['number_independent_coils']

    assert args['number_coils'] == args['number_field_periods'] * (args['stellarator_symmetry'] + 1) * nic
    
    if len(args['length_normal']) != nic:
        ln = np.max(np.array(args['length_normal']))
        args['length_normal'] = [ln for i in range(nic)]
    if len(args['length_binormal']) != nic:
        lb = np.max(np.array(args['length_binormal']))
        args['length_binormal'] = [lb for i in range(nic)]    











    pass