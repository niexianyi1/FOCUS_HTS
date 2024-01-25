
## 所需初始参数总览，
## 写为字典格式，存为json文件
import sys
import json
sys.path.append('/home/nxy/codes/coil_spline_HTS/iteration')
import main
# 待添加：电流优化项, 磁面数据生成
args = {

## 参数名简写，在程序正文中由参数全称转换
## 在保存hdf5文件时，不能有None的值
##   参数类别/参数名             参数值       简写    类型     解释

# 优化器

    #迭代方式
    'iter_method':              'for',     #       str,    优化方式, 'for', 'minimize', 'for-min', 'min-for'
    'number_iteration':         1,        # (ni)  int,    优化器迭代次数（for循环）, 若为0, 则不迭代
    'minimize_method':          'Newton-CG',#       str,    minimize方法, https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    'minimize_tol':             1e-4,       # (mint)float,  （minimize）目标残差, 若为0, 不考虑此项                                   
                        
    # 优化算法参数
    'optimizer_coil':           'momentum', # (opt) str,    线圈参数迭代方法,  (sgd, gd, momentum, or adam)  
    'optimizer_fr':             'momentum', # (opt) str,    旋转参数迭代方法,  (sgd, gd, momentum, or adam)  
    'learning_rate_coil':       1e-6,       # (lrc) float,  参数c 迭代步长                     
    'learning_rate_fr':         1e-5,       # (lrfr)float,  参数fr 迭代步长             
    'momentum_mass':            0.9,        # (mom) float,  梯度下降的动量参数
    'axis_resolution':          10,         # (res) int,    Resolution of the axis, multiplies NZ                        
    'var':                      0.999,
    'eps':                      1e-8,
    
# 线圈

    # 线圈参数
    'number_coils':             16,         # (nc)  int,    线圈总数                    
    'number_field_periods':     2,          # (nfp) int,    线圈周期数                     
    'stellarator_symmetry':     1,          # (ss)  int,    仿星器对称，1:对称，0:非对称                     
    'number_independent_coils': 4,          # (nic) int,    独立线圈数, (nc=nfp*(ss+1)*nic)                     
    'number_segments':          128,         # (ns)  int,    每个线圈分段数                     
    
    # 线圈生成
    'coil_radius':              1.0,        #       float,  生成初始的线圈半径    

    # 线圈读取       
    'coil_case':                'fourier',  #       str,    线圈表示方法, 'spline' or 'fourier'
    'init_coil_option':         'fourier',     #       str,    初始线圈参数的来源, 'spline' or 'coil' or 'fourier'
    'file_type':                'npy',      #       str,    初始线圈文件类型, 'npy' or 'makegrid', 后续可以再加
    'init_coil_file':                       #       str,    初始线圈文件名
            '/home/nxy/codes/coil_spline_HTS/results_f/jpark/j10_fc.npy',       

    # Fourier表示
    'num_fourier_coils':        6,          # (nfc) int,    表示线圈的fourier分量数
                     
    # Bspline表示
    'number_control_points':    67,         # (ncp) int,    每个线圈控制点数,为输入线圈坐标点数+2，默认有一个坐标点闭合                   
    'spline_k':                 3,          # (k)   int,    Bspline阶数,默认为3阶
                     
    # 有限截面参数
    'length_normal':            0.05,       # (ln)  float,  有限截面下每个线圈的法向间隔的长度                   
    'length_binormal':          0.05,       # (lb)  float,  有限截面下每个线圈的副法向间隔的长度                    
    'number_normal':            1,          # (nn)  int,    有限截面下的法向线圈数量                     
    'number_binormal':          1,          # (nb)  int,    有限截面下的副法向线圈数量                    
                     
    # 旋转角参数
    'init_fr_case':             0,          #       int,    初始fr给出方法, 0：自动生成各项为0, 1：读取文件
    'init_fr_file':                         #       str,    给出变量fr的初始值文件
            '/home/nxy/codes/coil_spline_HTS/results_f/circle/s1_fr.npy',
    'number_rotate':            0,          # (nr)  int,    描述线圈绕组组的半旋转数的整数,通常设为0                    
    'number_fourier_rotate':    0,          # (nfr) int,    每个线圈的旋转的傅里叶分量的个数，为0则不考虑旋转                    
   
# 电流

    # 电流初始
    'current_independent':      0,          #       int,    每个独立线圈是否拥有独立电流, 0:否, 1:是
    'current_I':                [1.2990236e6],      #       list,   线圈初始电流数值         

    # 电流参数优化



# 磁面
    
    # 磁面参数
    'number_theta':             32,         # (nt)  int,    磁面上\theta(极向)的网格点数                    
    'number_zeta':              128,        # (nz)  int,    磁面上\zeta(环向)的网格点数              
    'surface_case':             1,          #       int,    磁面数据来源, 0:直接读取文件, 1:计算vmec文件生成
    
    # 计算磁面           
    'surface_vmec_file':                    #       str,    磁面数据的vmec文件
            '/home/nxy/codes/coil_spline_HTS/initfiles/jpark/plasma3.boundary',

    
    # 读取磁面数据
    'surface_r_file':                       #       str,    磁面坐标文件
            '/home/nxy/codes/coil_spline_HTS/initfiles/w7x/highres_r_surf.npy',
    'surface_nn_file':                      #       str,    磁面法向文件
            '/home/nxy/codes/coil_spline_HTS/initfiles/w7x/highres_nn_surf.npy',
    'surface_sg_file':                      #       str,    磁面面积文件
            '/home/nxy/codes/coil_spline_HTS/initfiles/w7x/highres_sg_surf.npy',

# 背景磁场

    'B_extern':                 0,          #       


# loss function 权重

    'weight_bnormal':           1,          # (wb)  float,  法向磁场分量, 一般设为1
    'weight_length':            1e-6,       # (wl)  float,  单根线圈平均长度 
    'weight_curvature':         0.01,       # (wc)  float,  曲率 
    'weight_curvature_max':     0.005,       # (wcm) float,  最大曲率
    'weight_torsion':           0.01,       # (wt)  float,  扭转 
    'weight_torsion_max':       0.002,       # (wtm) float,  最大扭转
    'weight_distance_coil_coil':    1e-5,      # (wdcc)float,  线圈间距 
    'weight_distance_coil_surface': 5e-6,      # (wdcs)float,  线圈与磁面距离 
    'weight_strain':            0.01,

# LTS/HTS材料

    'HTS_material':             'REBCO_Other',    #       str,    HTS材料类型         
    'HTS_width':                0.05,       # (w)   float,  HTS材料宽度 或 LTS材料直径
    'HTS_thick':                0.05,       #       float,  HTS材料总厚度
    'HTS_thickness':            1.2e-6,     #       float,  HTS材料厚度
    'HTS_sec_area':             4.8e-9,     #       float,  HTS材料截面积（厚度*宽度或pi*(w/2)**2）
    'HTS_temperature':          4.2,        # (T)   float,  HTS材料温度, 单位K


# 画图

    # 画图选项
    'plot_coil':                1,          #       int,    是否画线圈, 0:不画, 1:画线圈点集, 2:画有限截面
    'plot_loss':                0,          #       int,    是否画迭代曲线, 0：不画, 1：画
    'plot_poincare':            1,          #       int,    是否画poincare图, 0：不画, 1：画
    
    # 画线圈
    'number_points':            500,        # (nps) int,    画线圈时的散点数, 建议不少于线圈段数

    # 画poincare图
    'poincare_r0':                          # (r0)  list,   画poincare图时的起点径向坐标
            [(1.36+0.01*i) for i in range(17)],                      
    'poincare_z0':                          # (z0)  list,   画poincare图时的起点z 向坐标 
            [0 for i in range(17)],                                
    'poincare_phi0':            0,          # (phi0)float,  画poincare图时的环向坐标
    'number_iter':              400,        # (niter) int,  画poincare图时的磁力线追踪周期数 
    'number_step':              1,          # (nstep) int,  画poincare图时的每个周期的追踪步数
                                
# 程序输出

    # 输出选项,  0：不保存, 1：保存, 
    'save_npy' :                0,          #       int,    保存npy文件, 仅限被优化参数和loss值
    'save_hdf5' :               0,          #       int,    保存hdf5文件, 包含所有参数(大部分)
    'save_makegrid' :           0,          #       int,    保存线圈makegrid文件

    # 输出地址
    'out_hdf5':                             #       str,    hdf5, 输出参数
        '/home/nxy/codes/coil_spline_HTS/results_f/circle/w7x_hdf5.h5',        
    'out_coil_makegrid':                    #       str,    makegrid , 输出线圈
        '/home/nxy/codes/coil_spline_HTS/results_f/circle/w7x_makegrid',                
    'save_loss':                            #       str,    npy, 输出损失函数值(lossvals)
        '/home/nxy/codes/coil_spline_HTS/results_f/circle/w7x_loss.npy',         
    'save_coil_arg':                        #       str,    npy, 输出优化线圈参数(coil_arg)
        '/home/nxy/codes/coil_spline_HTS/results_f/circle/w7x_fc.npy',
    'save_fr':                              #       str,    npy, 输出优化旋转参数(fr)
        '/home/nxy/codes/coil_spline_HTS/results_f/circle/w7x_fr.npy'
    
}




with open('/home/nxy/codes/coil_spline_HTS/initfiles/init_args.json', 'w') as f:
    json.dump(args, f, indent=4)



main.main()

