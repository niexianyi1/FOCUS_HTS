
## 所需初始参数总览，
## 写为字典格式，存为json文件

import json

# 待添加：电流优化项, 磁面数据生成
args = {

## 参数名简写，在程序正文中由参数全称转换

##   参数类别/参数名             参数值       简写    类型     解释

# 优化器

    #迭代方式
    'number_iteration':         1000,       # (ni)  int,    优化器迭代次数（for循环）,若为0，则不迭代
    'objective_value':          0,          # (obj) float,  （while循环）function目标量，若为0，不考虑此项                                   
                        
    # 优化算法参数
    'optimizer_coil':           'momentum', # (opt) str,    线圈参数迭代方法,  (sgd, gd, momentum, or adam)  
    'optimizer_fr':             'momentum', # (opt) str,    旋转参数迭代方法,  (sgd, gd, momentum, or adam)  
    'learning_rate_coil':       1e-5,       # (lrc) float,  参数c 迭代步长                     
    'learning_rate_fr':         1e-5,       # (lrfr)float,  参数fr 迭代步长             
    'momentum_mass':            0.9,        # (mom) float,  梯度下降的动量参数
    'axis_resolution':          10,         # (res) int,    Resolution of the axis, multiplies NZ                        
    'var':                      0.999,
    'eps':                      1e-8,
    
# 线圈

    # 线圈参数
    'number_coils':             50,         # (nc)  int,    线圈总数                    
    'number_field_periods':     5,          # (nfp) int,    线圈周期数                     
    'stellarator_symmetry':     1,          # (ss)  int,    仿星器对称，1:对称，0:非对称                     
    'number_independent_coils': 5,          # (nic) int,    独立线圈数, (nc=nfp*(ss+1)*nic)                     
    'number_segments':          64,         # (ns)  int,    每个线圈分段数                     
    
    # 线圈生成
    'coil_radius':              1.0,        #       float,  生成初始的线圈半径    

    # 线圈读取       
    'coil_case':                'fourier',  #       str,    线圈表示方法, 'spline' or 'fourier'
    'init_coil_option':         'coil',     #       str,    初始线圈参数的来源, 'spline' or 'coil' or 'fourier'
    'file_type':                'npy',      #       str,    初始线圈文件类型, 'npy' or 'makegrid', 后续可以再加
    'init_coil_file':                       #       str,    初始线圈文件名
            '/home/nxy/codes/focusadd-spline/initfiles/w7x/circle_coil_5.npy',       

    # Fourier表示
    'num_fourier_coils':        6,          # (nfc) int,    表示线圈的fourier分量数
                     
    # Bspline表示
    'number_control_points':    67,         # (ncp) int,    每个线圈控制点数,为输入线圈坐标点数+2，默认有一个坐标点闭合                   
    'spline_k':                 3,          # (k)   int,    Bspline阶数,默认为3阶
                     
    # 有限截面参数
    'length_normal':            0.015,      # (ln)  float,  有限截面下每个线圈的法向间隔的长度                   
    'length_binormal':          0.015,      # (lb)  float,  有限截面下每个线圈的副法向间隔的长度                    
    'number_normal':            1,          # (nn)  int,    有限截面下的法向线圈数量                     
    'number_binormal':          1,          # (nb)  int,    有限截面下的副法向线圈数量                    
                     
    # 旋转角参数
    'number_rotate':            0,          # (nr)  int,    描述线圈绕组组的半旋转数的整数,通常设为0                    
    'number_fourier_rotate':    0,          # (nfr) int,    每个线圈的旋转的傅里叶分量的个数，为0则不考虑旋转                    
   
# 电流

    # 电流初始
    'current_independent':      0,          #       int,    每个独立线圈是否拥有独立电流, 0:否, 1:是
    'current_I':                [1e6],      #       list,   线圈初始电流数值         

    # 电流参数优化



# 磁面
    
    # 磁面参数
    'number_theta':             20,         # (nt)  int,    磁面上\theta(极向)的网格点数                    
    'number_zeta':              150,        # (nz)  int,    磁面上\zeta(环向)的网格点数              
    'surface_case':             0,          #       int,    磁面数据来源, 0:文件读取,1:计算生成

    # 计算磁面           
    'axis_file':                            #       str,    磁轴文件
            'None',    
    'surface_radius':           0.5,        #       float,  磁面的半径                       
    
    # 读取磁面数据
    'surface_r_file':                       #       str,    磁面坐标文件
            '/home/nxy/codes/focusadd-spline/initfiles/w7x/highres_r_surf.npy',
    'surface_nn_file':                      #       str,    磁面法向文件
            '/home/nxy/codes/focusadd-spline/initfiles/w7x/highres_nn_surf.npy',
    'surface_sg_file':                      #       str,    磁面面积文件
            '/home/nxy/codes/focusadd-spline/initfiles/w7x/highres_sg_surf.npy',

# 背景磁场

    'B_extern':                 None,       #       


# loss function 权重

    'weight_bnormal':           1,          # (wb)  float,  法向磁场分量, 一般设为1
    'weight_length':            0,          # (wl)  float,  单根线圈平均长度 
    'weight_curvature':         0,          # (wc)  float,  曲率 
    'weight_curvature_max':     0,          # (wcm) float,  最大曲率
    'weight_torsion':           0,          # (wt)  float,  扭转 
    'weight_torsion_max':       0,          # (wtm) float,  最大扭转
    'weight_distance_coil_coil':    0,      # (wdcc)float,  线圈间距 
    'weight_distance_coil_surface': 0,      # (wdcs)float,  线圈与磁面距离 

# HTS材料

    'width':                    0.004,      # (w)   float,  HTS材料宽度


# 画图

    # 画线圈
    'plot_coil':                1,          #       int,    是否画线圈, 0:不画, 1:画线圈点集, 2:画有限截面
    'number_points':            500,        # (nps) int,    画线圈时的散点数, 建议不少于线圈段数

    # 画迭代曲线
    'plot_loss':                1,          #       int,    是否画迭代曲线, 0：不画, 1：画

    # 画poincare图
    'plot_poincare':            1,          #       int,    是否画poincare图, 0：不画, 1：画
    'poincare_r0':                          # (r0)  list,   画poincare图时的起点径向坐标
            [(6+0.01*i) for i in range(20)],                      
    'poincare_z0':                          # (z0)  list,   画poincare图时的起点z 向坐标 
            [0 for i in range(20)],                                
    'poincare_phi0':            0,          # (phi0)float,  画poincare图时的环向坐标
    'number_iter':              200,        # (niter) int,  画poincare图时的磁力线追踪周期数 
    'number_step':              1,          # (nstep) int,  画poincare图时的每个周期的追踪步数
                                
# 程序输出

    'out_hdf5':                             #       str,    输出线圈参数文件名
        '/home/nxy/codes/focusadd-spline/results/circle/out_hdf5_1000',        
    'out_coil_makegrid':                    #       str,    makegrid 类型, 输出线圈文件名 
        '/home/nxy/codes/focusadd-spline/results/circle/out_coil_makegrid_1000',                
    'save_loss':                             #       str,    输出损失函数值（lossvals）文件名
        '/home/nxy/codes/focusadd-spline/results_f/circle/loss_c1.npy',         
    'save_coil_arg':                               #       str,    输出参数c文件名
        '/home/nxy/codes/focusadd-spline/results_f/circle/fc_c1.npy',
    'save_fr':                               #       str,    输出参数fr文件名
        '/home/nxy/codes/focusadd-spline/results_f/circle/fr_c1.npy'
    
}



if __name__ == "__main__":
    with open('/home/nxy/codes/focusadd-spline/initfiles/init_args.json', 'w') as f:
        json.dump(args, f, indent=4)





