import jax.numpy as np
from jax import jit, vmap
from jax.config import config
import self_B  
import sys 
sys.path.append('/home/nxy/codes/coil_spline_HTS/HTS')
import material_jcrit
config.update("jax_enable_x64", True)
pi = np.pi

# 应变项调整减数
# 自场需要扫描一下极值点
# 临界电流密度的量级要对上

@jit
def loss(args, coil_output_func, params, surface_data):
    """ 
    Computes the default loss: int (B dot n)^2 dA + weight_length * len(coils) 

    Input: params, a tuple of the fourier series for the coils and a fourier series for the rotation.

    Output: A scalar, which is the loss_val computed by the function. JAX will eventually differentiate
    this in an optimizer.
    """
    def compute_loss_fourier(args, coil_output_func, params, surface_data):
        I, dl, coil, der1, der2, der3, v1, _, _ = coil_output_func(params)
        Bn_mean, _, _ = quadratic_flux(args, I, dl, coil, surface_data)
        length, deltal = average_length(args, coil)
        k_mean, k_max, curva = curvature(der1, der2)
        t_mean, t_max = torsion(der1, der2, der3)
        dcc_min = distance_cc(args, coil)
        dcs_min = distance_cs(args, coil, surface_data)
        strain_max = HTS_strain(args, curva, v1, deltal)
        length = (length - args['target_length'])**2
        k_max = np.max(np.array([k_max , args['target_curvature_max']]))
        t_max = np.max(np.array([t_max , args['target_torsion_max']]))
        dcc_min = np.min(np.array([dcc_min, args['target_distance_coil_coil']]))
        dcs_min = np.min(np.array([dcs_min, args['target_distance_coil_surface']]))
        strain = np.max(np.array([strain_max, args['target_strain']]))
        
        lossvalue = (args['weight_bnormal']  * Bn_mean   + args['weight_length']  * length 
            + args['weight_curvature']  * k_mean    + args['weight_curvature_max']  * k_max 
            + args['weight_torsion']   * t_mean    + args['weight_torsion_max']  * t_max 
            + args['weight_strain']   * strain
            + args['weight_distance_coil_coil'] * dcc_min + args['weight_distance_coil_surface'] * dcs_min )
        return lossvalue
    
    def compute_loss_spline(args, coil_output_func, params, surface_data):
        I, dl, coil, der1, der2, _, v1, _, _ = coil_output_func(params)
        Bn_mean, Bn_max, B_max_surf = quadratic_flux(args, I, dl, coil, surface_data)
        length, deltal = average_length(args, coil)
        k_mean, k_max, curva = curvature(der1, der2)
        dcc_min = distance_cc(args, coil)
        dcs_min = distance_cs(args, coil, surface_data)
        dcc_min = np.exp(-dcc_min)    # 数值约为0.6-0.8，权重设置小一点
        dcs_min = np.exp(-dcc_min)
        strain_max = HTS_strain(args, curva, v1, deltal)
        strain = np.max(np.array([strain_max-0.004, 0]))
        lossvalue = (  args['weight_bnormal']    * Bn_mean   + args['weight_length']         * length 
            + args['weight_curvature']  * k_mean    + args['weight_curvature_max']  * k_max 
            + args['weight_strain']    * strain
            + args['weight_distance_coil_coil'] * dcc_min + args['weight_distance_coil_surface'] * dcs_min )
        return lossvalue

    compute_loss = dict()
    compute_loss['fourier'] = compute_loss_fourier
    compute_loss['spline'] = compute_loss_spline
    compute_loss['spline_local'] = compute_loss_spline
    compute_func = compute_loss[args['coil_case']]
    lossvalue = compute_func(args, coil_output_func, params, surface_data)
    return lossvalue


def compute_signleI(args, coil, I, dl, v1, v2, binormal, curva, deltal, B_extern):
    strain_max = HTS_strain(args, curva, v1, deltal)
    B_coil = self_B.coil_self_B_signle(args, coil, I, dl, v1, v2, binormal, curva) 
    B_coil_max = np.max(np.linalg.norm(B_coil + B_extern, axis=-1), axis = 1)
    # print('B_coil_max =', np.max(np.linalg.norm(B_coil + B_extern, axis=-1)))  
    # print('B_min_coil =', np.min(np.linalg.norm(B_coil + B_extern, axis=-1)))
    # jc,b,t = Jcrit(args, B_coil_max, 0.004)
    # print('jc = ', jc, np.min(jc))
    return B_coil_max, 0, strain_max, B_coil


def quadratic_flux(args, I, dl, coil, surface_data):
    """ 

    Computes the normalized quadratic flux over the whole surface.
        
    Inputs:

    r : Position we want to evaluate at, NZ x NT x 3
    I : Current in ith coil, length NC
    dl : Vector which has coil segment length and direction, NC x NS x NNR x NBR x 3
    l : Positions of center of each coil segment, NC x NS x NNR x NBR x 3
    nn : Normal vector on the surface, NZ x NT x 3
    sg : Area of the surface, 
    
    Returns: 

    A NZ x NT array which computes integral of 1/2(B dot n)^2 dA / integral of B^2 dA. 
    We can eventually sum over this array to get the total integral over the surface. I choose not to
    sum so that we can compute gradients of the surface magnetic normal if we'd like. 

    """
    r_surf, nn, sg = surface_data
    I = I / args['number_normal'] / args['number_binormal'] 
    B = biotSavart(coil, I, dl, r_surf)  # NZ x NT x 3
    Bmax= np.max(np.linalg.norm(B, axis=-1))


    Bn = np.sum(nn * B, axis=-1)
    if args['Bn_extern'] != 0:
        Bn = abs(Bn + args['Bn_extern_surface'])
    else:
        Bn = abs(Bn)
    Bn_mean = 0.5*np.sum((Bn/ np.linalg.norm(B, axis=-1))** 2 * sg)
    # Bn_mean = 0.5*np.sum((Bn)** 2 * sg)
    Bn_max = np.max(abs(Bn))
    return  Bn_mean, Bn_max, Bmax

def biotSavart(coil, I, dl, r_surf):
    """
    Inputs:

    r : Position we want to evaluate at, NZ x NT x 3
    I : Current in ith coil, length NC
    dl : Vector which has coil segment length and direction, NC x NS x NNR x NBR x 3
    l : Positions of center of each coil segment, NC x NS x NNR x NBR x 3

    Returns: 

    A NZ x NT x 3 array which is the magnetic field vector on the surface points 
    """
    mu_0 = 1e-7
    mu_0I = I * mu_0
    mu_0Idl = (mu_0I[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] * dl)  # NC x NNR x NBR x NS x 3
    r_minus_l = (r_surf[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis, :]
        - coil[:, np.newaxis, np.newaxis, :, :, :, :])  # NC x NZ/nfp x NT x NNR x NBR x NS x 3
    top = np.cross(mu_0Idl[:, np.newaxis, np.newaxis, :, :, :, :], r_minus_l)  # NC x NZ x NT x NNR x NBR x NS x 3
    bottom = (np.linalg.norm(r_minus_l, axis=-1) ** 3)  # NC x NZ x NT x NNR x NBR x NS
    B = np.sum(top / bottom[:, :, :, :, :, :, np.newaxis], axis=(0, 3, 4, 5))  # NZ x NT x 3
    return B

def torsion(der1, der2, der3):       # new
    cross12 = np.cross(der1, der2)
    top = (
        cross12[:, :, 0] * der3[:, :, 0]
        + cross12[:, :, 1] * der3[:, :, 1]
        + cross12[:, :, 2] * der3[:, :, 2]
    )
    bottom = np.linalg.norm(cross12, axis=-1) ** 2
    t = abs(top / bottom)     # NC x NS
    t_mean = np.mean(t)
    t_max = np.max(t)
    return t_mean, t_max

def curvature(der1, der2):
    bottom = np.linalg.norm(der1, axis = -1)**3
    top = np.linalg.norm(np.cross(der1, der2), axis = -1)
    k = abs(top / bottom)
    k_mean = np.mean(k)
    k_max = np.max(k)
    curva = np.cross(der1, der2) / (np.linalg.norm(der1, axis = -1)**3)[:,:,np.newaxis]
    return k_mean, k_max, curva

def average_length(args, coil):      #new
    nic = args['number_independent_coils']   
    al = np.zeros((nic, coil.shape[3], 3))   
    r_coil = np.mean(coil, axis = (1,2))   # 有限截面平均
    al = al.at[:, :-1, :].set(r_coil[:nic, 1:, :] - r_coil[:nic, :-1, :])
    al = al.at[:, -1, :].set(r_coil[:nic, 0, :] - r_coil[:nic, -1, :])
    len = np.sum(np.linalg.norm(al, axis=-1)) / nic
    return len, al

def distance_cc(args, coil):  ### 暂未考虑finite-build, 边界处的距离算了2种情况
    nic = args['number_independent_coils']  
    ns = args['number_segments']
    rc = np.mean(coil, axis = (1,2))
    dr = np.zeros((nic+1, ns, ns, 3))
    dr = dr.at[:nic-1].set(rc[:nic-1, :, np.newaxis, :] - rc[1:nic, np.newaxis, :, :])
    dr = dr.at[nic-1].set(rc[nic-1, :, np.newaxis, :] - rc[2*nic-1, np.newaxis, :, :])
    dr = dr.at[nic].set(rc[0, :, np.newaxis, :] - rc[nic, np.newaxis, :, :])
    dr = np.linalg.norm(dr, axis = -1)
    dcc_min = np.min(dr)
    return dcc_min

def distance_cs(args, coil, surface_data):  ### 暂未考虑finite-build
    rc = np.mean(coil, axis = (1,2))
    rs, _, _ = surface_data
    dr = (rc[:args['number_independent_coils'], :, np.newaxis, np.newaxis, :]
                - rs[np.newaxis, np.newaxis, :, :, :])
    dr = np.linalg.norm(dr, axis = -1)
    dcs_min = np.min(dr)
    return dcs_min

##  HTS应变量
def HTS_strain(args, curva, v1, deltal):
    bend = HTS_strain_bend(args, curva, v1)
    tor = HTS_strain_tor(args, deltal, v1)
    strain_max = np.max(bend + tor)
    return strain_max

def HTS_strain_bend(args, curva, v1):
    """弯曲应变,
    Args:
        w, 带材宽度
        v1,有限截面坐标轴
        curva, 线圈曲率

    Returns:
        bend, 弯曲应变

    """
    ### 此处width应取单根线材的宽度，而非总线缆的宽度，后续再更改
    width = args['HTS_signle_width']
    bend = width/2*abs(np.sum(-v1 * curva, axis=-1))
    return bend


def HTS_strain_tor(args, deltal, v1):
    """扭转应变,
    Args:
        w, 带材宽度
        v1,有限截面坐标轴
        deltal, 线圈点间隔

    Returns:
        bend, 弯曲应变

    """
    width = args['HTS_signle_width']
    dv = np.zeros((v1.shape[0], v1.shape[1]))
    dv = dv.at[:, :-1].set(np.sum(v1[:, :-1, :] * v1[:, 1:, :], axis=-1))
    dv = dv.at[:, -1].set(np.sum(v1[:, -1, :] * v1[:, 0, :], axis=-1))
    dtheta = np.arccos(dv)
    deltal = np.linalg.norm(deltal, axis=-1)     
    tor = width**2/12*(dtheta/deltal)**2


    return tor


def Jcrit(args, B_self, strain):

    nic = args['number_independent_coils']
    # assert nic == len(B_self)
    jc = np.zeros((nic))
    for i in range(nic):
        j,b,t = material_jcrit.get_critical_current(args['HTS_temperature'],
                        B_self[i], strain, args['HTS_material'])
        jc = jc.at[i].set(j)

    return jc,b,t


def Icrit_coil(args, jc):
    # if args['HTS_material'] == 'REBCO_Other':
    lw = np.array(args['length_normal']) * args['number_normal'] 
    lt = np.array(args['length_binormal']) * args['number_binormal'] 
    Icc = jc * lw * lt * args['HTS_I_percent'] * args['HTS_structural_percent']
    # else:
    #     Icc = jc * args['HTS_width']**2 * pi # 单位面积cm^2

    return Icc


def loss_save(args, coil_output_func, params, surface_data):
    """ 
    Computes the default loss: int (B dot n)^2 dA + weight_length * len(coils) 

    Input: params, a tuple of the fourier series for the coils and a fourier series for the rotation.

    Output: A scalar, which is the loss_val computed by the function. JAX will eventually differentiate
    this in an optimizer.
    """
    I, dl, coil, der1, der2, der3, v1, v2, binormal = coil_output_func(params)
    Bn_mean, Bn_max, B_max_surf = quadratic_flux(args, I, dl, coil, surface_data)
    length, deltal = average_length(args, coil)
    k_mean, k_max, curva = curvature(der1, der2)
    if args['coil_case']=='fourier':
        t_mean, t_max = torsion(der1, der2, der3)
    else:
        t_mean, t_max = 0, 0
    dcc_min = distance_cc(args, coil)
    dcs_min = distance_cs(args, coil, surface_data)
    strain_max = HTS_strain(args, curva, v1, deltal)
    B_max_coil = self_B.coil_self_B_max(args, coil, I, dl, v1, v2, binormal, curva)
    j_crit,b,t = Jcrit(args, B_max_coil, strain_max)
    I_crit_coil = Icrit_coil(args, j_crit)
    loss_end = {
        'loss_Bn_mean'      :   Bn_mean,
        'loss_Bn_max'       :   Bn_max,
        'loss_B_max_surf'   :   B_max_surf,
        'loss_length'       :   length,
        'loss_curvature'    :   k_mean,
        'loss_curva_max'    :   k_max,
        'loss_dcc_min'      :   dcc_min,
        'loss_dcs_min'      :   dcs_min,
        'loss_tor_mean'     :   t_mean,
        'loss_tor_max'      :   t_max,
        'loss_B_max_coil'   :   B_max_coil,
        'loss_strain_max'   :   strain_max,
        'loss_HTS_Icrit'    :   I_crit_coil
        }

    return loss_end

# def symmetry_B(args, B):
#     B_total = np.zeros((args.nz, args.nt, 3))
#     B_total = B_total.at[:, :, :].add(B)
#     for i in range(args.nfp - 1):        
#         theta = 2 * pi * (i + 1) / args.nfp
#         T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
#         B_total = B_total.at[:, :, :].add(np.dot(B, T))
    
#     return B_total





















