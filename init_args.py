import json


args = {
    # 优化器参数
    'n': 5,            # int, "--number_iter"
                        # 优化器迭代次数
    'lr': 2.5e-5,       # float, "--learning_rate_c"
                        # 参数c 迭代步长
    'lrfr': 1.0,        # float, "--learning_rate_fr"
                        # 参数fr 迭代步长
    'opt': 'momentum',  # str, "--optimizer"
                        # "Name of optimizer. Either SGD, GD (same), Momentum, or Adam"
    'mom': 0.9,         # float, "--momentum_mass"
                        # Momentum mass parameter.
    'res': 10,          # int, "--axis_resolution"
                        # Resolution of the axis, multiplies NZ.

    # 线圈参数
    'nc': 50,       # int, "--number_coils"
                    # 线圈总数
    'nfp': 5,       # int, "--number_field_periods"
                    # 线圈周期数
    'ns': 64,       # int, "--number_segments"
                    # 每个线圈分段数
    
    'ln': 0.015,    # float, "--length_normal"
                    # 有限截面下每个线圈的法向间隔的长度
    'lb': 0.015,    # float, "--length_binormal"
                    # 有限截面下每个线圈的副法向间隔的长度
    'nnr': 1,       # int, "--number_normal_rotate"
                    # 有限截面下的法向线圈数量
    'nbr': 1,       # int, "--number_binormal_rotate"
                    # 有限截面下的副法向线圈数量
    
    'rc': 2.0,      # float,"--radius_coil"
                    # 线圈半径
    'nr': 0,        # int,  "--number_rotate"
                    # 有限截面下每个线圈的旋转数
    'nfr': 0,       # int, "--number_fourier_rotate"
                    # 每个线圈的旋转的傅里叶分量的个数
   
    # 磁面参数
    'nt': 20,       # int, "--number_theta" 
                    # 磁面上\theta(极向)的网格点数
    'nz': 150,      # int, "--number_zeta"
                    # 磁面上\zeta(环向)的网格点数
    'rs': 0.5,      # float, "--radius_surface"
                    # 磁面的半径
 
    # loss function 权重参数
    'wb': 1,        # float, "--weight_Bnormal"
                    # 法向磁场分量 权重
    'wl': 0.5,        # float, "--weight_length"
                    # 总长度 权重
    'wdcc': 0,      # float, "--weight_distance_coil_coil"
                    # 线圈间距 权重

    
    # 画图参数
    'nps': 500,      # int, "--number_points"
                    # 画线圈时的散点数
    'init': False,  # bool, "--initial"
                    # 是否画初始线圈和优化线圈的对比图
    'log': False,   # bool, "--log10(lossvals)"
                    # 是否画损失函数值的对数图  
    'r0': [6],      # list, 
                    # 画poincare图时的起点径向坐标
    'z0': [0.2],      # list,
                    # 画poincare图时的起点z 向坐标
    'phi0': 0,      # float,
                    # 画poincare图时的环向坐标
    'niter': 200,   # int, 
                    # 画poincare图时的磁力线追踪周期数
    'nstep': 1,     # int, "number_step"
                    # 画poincare图时的每个周期的追踪步数
    'poincare_save': 'None',    # str,
                                # 输出poincare图坐标数据文件名, None不输出

    # 文件
    'init_option': 'init_coil',
    # str, 初始线圈参数的来源, 'init_c' or 'init_coil'

    'init_coil': '/home/nxy/codes/focusadd-spline/initfiles/w7x/circle_coil.npy',       
    # str, makegird 类型, 初始线圈文件名

    'init_c': '/home/nxy/codes/focusadd-spline/results/circle/c_200.npy',
    # str, 初始参数c文件名    

    'surface_r': '/home/nxy/codes/focusadd-spline/initfiles/w7x/highres_r_surf.npy',
    'surface_nn': '/home/nxy/codes/focusadd-spline/initfiles/w7x/highres_nn_surf.npy',
    'surface_sg': '/home/nxy/codes/focusadd-spline/initfiles/w7x/highres_sg_surf.npy',
    # str, 磁面参数文件名

    'axis_file': 'None',        
    # str,  "Name of axis file"

    'out_hdf5': '/home/nxy/codes/focusadd-spline/results/circle/out_hdf5_400',        
    # str, 输出线圈参数文件名

    'out_coil_makegrid': '/home/nxy/codes/focusadd-spline/results/circle/out_coil_makegrid_400',
    # str, makegrid 类型, 输出线圈文件名 
                 
    'out_loss': '/home/nxy/codes/focusadd-spline/results/circle/loss_400.npy',         
    # str, 输出损失函数值（lossvals）文件名

    'out_c': '/home/nxy/codes/focusadd-spline/results/circle/c_400.npy',
    # str, 输出参数c文件名

    'out_fr': '/home/nxy/codes/focusadd-spline/results/circle/fr_400.npy'
    # str, 输出参数fr文件名

}



if __name__ == "__main__":
    with open('/home/nxy/codes/focusadd-spline/initfiles/init_args.json', 'w') as f:
        json.dump(args, f, indent=4)





