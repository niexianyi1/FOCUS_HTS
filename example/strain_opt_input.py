######### 单独优化应变的输入, 输入参数有删减和小修改 #########
## 所需初始参数总览，
## 写为字典格式,存为json文件
import sys
import json
sys.path.append('opt_coil')
import strain_opt
# 待添加：电流优化项, 磁面数据生成
args = {

## 参数名简写，在程序正文中由参数全称转换
## 在保存hdf5文件时，不能有None的值
##   参数类别/参数名             参数值       简写    类型     解释

# ------------ 常用参数 ------------ #
# 优化
    # 优化目标
    'alpha_optimize':           1,          #       int,    是否优化有限截面旋转角, 0为否, 1为是

    #迭代方式
    'iter_method':              'nlopt',     #       str,    优化方式, 'min', 'nlopt'
    
    # 优化算法参数:minimize
    'minimize_method':          'BFGS',     #       str,    minimize方法, https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    'minimize_tol':             1e-4,       # (mint)float,  （minimize）目标残差, 若为0, 不考虑此项                                   

    # 优化算法参数:nlopt
    'nlopt_algorithm':          'LD_MMA',     #       str,    nlopt算法, https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
    'stop_criteria':             1e-4,      # (mint)float,  （minimize）目标残差, 若为0, 不考虑此项                                   
                  
# 线圈
    # 线圈参数 # 不变，还是按照所有线圈填写
    'number_coils':             16,         # (nc)  int,    线圈总数                    
    'number_field_periods':     2,          # (nfp) int,    线圈周期数                     
    'stellarator_symmetry':     1,          # (ss)  int,    仿星器对称，1:对称，0:非对称                     
    'number_independent_coils': 4,          # (nic) int,    独立线圈数(半周期线圈数), (nc=nfp*(ss+1)*nic)                     
    'number_segments':          128,         # (ns)  int,    每个线圈分段数   
    
    # 线圈输入方式      
    'coil_case':                'fourier',  #       str,    线圈表示方法, 'spline' or 'fourier'
    'init_coil_option':         'fourier',     #       str,    初始线圈参数的来源, 'spline' or 'coordinates' or 'fourier'
    'coil_file_type':           'hdf5',      #       str,    非'circle'的初始线圈文件类型, 'npy', 'hdf5' or 'makegrid', 后续可以再加
    'init_coil_file':                       #       str,    初始线圈文件名
            'results/LQA/useful/cs_fn_4_t.h5',       

    # Fourier表示
    'number_fourier_coils':        6,          # (nfc) int,    表示线圈的fourier分量数
                     
    # Bspline表示
    'number_control_points':    32,         # (ncp) int,    每个线圈控制点数,为输入线圈坐标点数+2，默认有一个坐标点闭合                   
                     
    # 旋转角参数
    'init_fr_case':             1,          #       int,    初始fr给出方法, 0：自动生成各项为0, 1：读取文件
    'init_fr_file':                         #       str,    给出变量fr的初始值文件, 后缀'h5','npy'
            'results/LQA/useful/cs_fn_4_t.h5',               
    'number_rotate':            0,          # (nr)  int,    描述线圈绕组组的半旋转数的整数,通常设为0                    
    'number_fourier_rotate':    6,          # (nfr) int,    每个线圈的旋转的傅里叶分量的个数 

# LTS/HTS材料(为方便下列参数都以HTS命名)
    'material':                 'REBCO_LT',    # str,    材料类型, REBCO_LT, NbTi, Nb3Sn    
    'HTS_signle_width':         4e-3,       # (sw)  float,  HTS材料单根宽度, 不包括相邻间隙
    'HTS_signle_thickness':     5e-5,       # (st)  float,  HTS材料单层结构厚度, 不包括相邻间隙
    
# loss function 目标和权重
    'target_HTS_strain':            0.004,      #  float,  目标最大应变

# 结果保存
    # 输出选项,  0：不保存, 1：保存, 
    'save_hdf5' :               1,          #       int,    保存hdf5文件, 包含所有参数(大部分)

    # 输出地址
    'out_hdf5':                             #       str,    hdf5, 输出参数
        'results/LQA/non_circle_start/strain_opt/single/fsn4_1.h5',   

                    

# -------------------------------------- 不常变参数 ------------------------------------ #
# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #
    'alpha_optimize':           1,          #       int,    是否优化有限截面旋转角, 0为否, 1为是
    # 优化算法参数:jax,              
    'var':                      0.999,      # adam优化方法的参数
    'eps':                      1e-8,       # adam优化方法的参数


    # Bspline局部优化
    'optimize_location_nic':    [0],      #       list,   局部优化的线圈位置, 由列表给出进行局部优化的线圈是第几个, 第一个线圈从0开始   
    'optimize_location_ns':                 #       list,   局部优化的线圈位置, 由列表给出每个进行局部优化的线圈的具体分段, 
                [[[0, 64]]],               
        # 3阶列表，第一阶表示第几个线圈, 第二阶表示一个线圈中分成几段, 第三阶为每一段的起始与末尾点的位置
        # 此处第三阶的位置是实际的坐标位置, 不是控制点对应的节点区间, 每一项都为int
        # 同线圈不同段间距应大于3, 否则连在一起

    # 有限截面参数
    'length_calculate':         0,          #       int,    0：手动给出截面大小, 1：由超导线圈临界电流给出截面大小
    'length_normal':            [0.001 for i in range(3)],       # (ln)  list,  有限截面下每个线圈的法向间隔的长度                   
    'length_binormal':          [0.05 for i in range(3)],       # (lb)  list,  有限截面下每个线圈的副法向间隔的长度                    
    'number_normal':            2,          # (nn)  int,    有限截面下的法向线圈数量                     
    'number_binormal':          2,          # (nb)  int,    有限截面下的副法向线圈数量  

# 电流

    # 电流初始
    'current_independent':      0,          #       int,    每个独立线圈是否拥有独立电流, 0:否, 1:是
    'current_I':                [1e5],      #       list,   线圈初始电流数值         
    'total_current_I':          0,          #       int,    电流优化时是否保持总电流不变, 
        # 若为0, 则不设置总电流; 若不为0, 则为总电流数值

# 磁面
    
    # 磁面参数
    'surface_case':             1,          #       int,    磁面数据来源, 0:直接读取文件, 1:计算vmec文件生成
    'number_theta':             64,         # (nt)  int,    磁面上\theta(极向)的网格点数                    
    'number_zeta':              128,        # (nz)  int,    磁面上\zeta(环向)的网格点数              
    
    # vmec文件           
    'surface_vmec_file':                    #       str,    磁面数据的vmec文件
            'initfiles/Landreman-Paul_QA/plasma.boundary',

# 背景磁场

    'Bn_extern':                 0,          #       int,    背景磁场设置, 0:无, 1:boundary文件
    'Bn_extern_file':                        #       str,    读取对应的文件
            'initfiles/ncsx_c09r00/c09r00.boundary',
   

    # 读取磁面文件, 若'surface_case': 1,
    'surface_r_file':                       #       str,    磁面坐标文件
            'initfiles/w7x/highres_r_surf.npy',
    'surface_nn_file':                      #       str,    磁面法向文件
            'initfiles/w7x/highres_nn_surf.npy',
    'surface_sg_file':                      #       str,    磁面面积文件
            'initfiles/w7x/highres_sg_surf.npy',

    # LTS/HTS材料
    'HTS_temperature':          4.2,        # (T)   float,  HTS材料温度, 单位K
    'HTS_I_percent':            0.6,        #       float,  线圈电流与超导临界电流的比例,也是临界电流的约束值
    'HTS_structural_percent':   0.01,       #       float,  超导线缆与结构材料的比例
    'HTS_I_thickness':          1.2e-6,     #       float,  HTS材料导电层厚度
    'HTS_sec_area':             4.8e-9,     #       float,  HTS材料截面积（厚度*宽度或pi*(w/2)**2）

    # 权重设置
    'weight_normalization':     0,          #       int,    是否归一化权重 wi*fi=weight

    'weight_bnormal':           0,          # (wb)  float,  法向磁场分量, 一般设为1
    'weight_length':            0,          # (wl)  float,  单根线圈平均长度 
    'weight_curvature':         0,          # (wc)  float,  曲率 
    'weight_curvature_max':     0,          # (wcm) float,  最大曲率
    'weight_torsion':           0,          # (wt)  float,  扭转 
    'weight_torsion_max':       0,          # (wtm) float,  最大扭转
    'weight_distance_coil_coil':    0,      # (wdcc)float,  线圈间距 
    'weight_distance_coil_surface': 0,      # (wdcs)float,  线圈与磁面距离 
    'weight_HTS_force':             0,          # (wf)  float,  线圈受自场力  
    'weight_HTS_Icrit':             0,          #       float,  线圈自场与线圈表面夹角  
    # 目标
    'target_length_mean':            0,         #  float,  目标长度, 若为0则约束其较小
    'target_length_single':                     #  list/float,   每个线圈目标长度,若第一项为0则不约束
                            [5.08, 5.02, 4.91, 4.8],
    'target_curvature_max':     2.36,           #  float,  目标最大曲率
    'target_torsion_max':       5,              #  float,  目标最大扭转
    'target_distance_coil_coil':    0.25,       #  float,  目标最大线圈间距
    'target_distance_coil_surface': 0.44,       #  float,  目标最大线圈与磁面距离
    'target_HTS_force':             0,          #  float, 目标最大受力


    # 画poincare图                        
    'poincare_phi0':            0,          # (phi0)float,  画poincare图时的环向坐标
    'number_iter':              400,        # (niter) int,  画poincare图时的磁力线追踪周期数 
    'number_step':              1,          # (nstep) int,  画poincare图时的每个周期的追踪步数

    # 输出选项,  0：不保存, 1：保存, 
    'save_makegrid' :           0,          #       int,    保存线圈makegrid文件

    # 输出地址    
    'out_coil_makegrid':                    #       str,    makegrid , 输出线圈
        'results/w7x/compare/filameat/makegrid',                
# 简单画图(更多功能可在后处理post/post_plot.py中进行)

    # 画图选项
    'plot_coil':                0,          #       int,    是否画线圈, 0:不画, 1:画线圈点集, 2:画有限截面
    'plot_loss':                0,          #       int,    是否画迭代曲线, 0：不画, 1：画
    'plot_poincare':            0,          #       int,    是否画poincare图, 0：不画, 1：画

    # 画线圈
    'number_points':            500,        # (nps) int,    画线圈时的散点数, 建议不少于线圈段数

    # 画poincare图
    'poincare_number':          25,         # (pn)      int,    画poincare图时的磁面圈数     
}



with open('initfiles/init_args.json', 'w') as f:
    json.dump(args, f, indent=4)


strain_opt.main()

