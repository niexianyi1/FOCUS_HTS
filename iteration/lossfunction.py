import jax.numpy as np
from jax import jit, config
import sys 
sys.path.append('/home/nxy/codes/coil_spline_HTS/HTS')
import hts_strain
import B_self
import section_length
import material_jcrit
config.update("jax_enable_x64", True)
pi = np.pi

# 应变项调整减数
# 自场需要扫描一下极值点
# 临界电流密度的量级要对上



@jit
def loss_value(args, coil_output_func, params, surface_data):
    """ 
    Input: params, a tuple of the fourier series for the coils and a fourier series for the rotation.

    Output: A scalar, which is the loss_val computed by the function. JAX will eventually differentiate
    this in an optimizer.
    """
    I, dl, coil, der1, der2, der3, v1, v2, binormal = coil_output_func(params)
    Bn_mean, _, _, _, _ = quadratic_flux(args, I, dl, coil, surface_data)
    length = average_length(args, coil)
    curva = curvature(der1, der2)
    k_mean = curvature_mean(curva)
    k_max = curvature_max(curva)
    tor = torsion(args, der1, der2, der3, coil)
    t_mean = torsion_mean(tor)
    t_max = torsion_max(tor)
    dcc_min = distance_cc(args, coil)
    dcs_min = distance_cs(args, coil, surface_data)
    strain = hts_strain.HTS_strain(args, curva, v2, dl)
    I = I * args['I_normalize']
    B_reg = B_self.loss_B_coil(args, coil, I, dl, v1, v2, binormal, curva, der2)
    Bx, By = B_coil_normal(B_reg, v2)
    j_crit = Jcrit(args, B_reg, strain, Bx, By)
    I_crit = Icrit(args, j_crit)
    force = calculate_force(I, B_reg, dl)

    length = np.max(np.array([length - args['target_length'], 0]))
    k_max = np.max(np.array([k_max-args['target_curvature_max'], 0]))
    t_max = np.max(np.array([t_max-args['target_torsion_max'], 0]))
    dcc_min = np.min(np.array([args['target_distance_coil_coil']-dcc_min, 0]))
    dcs_min = np.min(np.array([args['target_distance_coil_surface']-dcs_min, 0]))
    strain_max = np.max(np.array([np.max(strain)-args['target_strain'], 0]))

    force = np.max(np.array([force-args['target_force'], 0]))
    lossvalue = (args['weight_bnormal']  * Bn_mean   + args['weight_length']  * length 
        + args['weight_curvature']  * k_mean    + args['weight_curvature_max']  * k_max 
        + args['weight_torsion']   * t_mean    + args['weight_torsion_max']  * t_max 
        + args['weight_distance_coil_coil'] * dcc_min + args['weight_distance_coil_surface'] * dcs_min 
        + args['weight_strain']   * strain_max + args['weight_jcrit'] * I_crit
        + args['weight_force']   * force)
    return lossvalue



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
    I = I / args['number_normal'] / args['number_binormal'] * args['I_normalize']

    mu_0 = 1e-7 
    mu_0I = I * mu_0
    mu_0Idl = (mu_0I[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] * dl)  # NC x NNR x NBR x NS x 3
    r_minus_l = (r_surf[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis, :]
        - coil[:, np.newaxis, np.newaxis, :, :, :, :])  # NC x NZ/nfp x NT x NNR x NBR x NS x 3
    top = np.cross(mu_0Idl[:, np.newaxis, np.newaxis, :, :, :, :], r_minus_l)  # NC x NZ x NT x NNR x NBR x NS x 3
    bottom = (np.linalg.norm(r_minus_l, axis=-1) ** 3)  # NC x NZ x NT x NNR x NBR x NS
    B = np.sum(top / bottom[:, :, :, :, :, :, np.newaxis], axis=(0, 3, 4, 5))  # NZ x NT x 3
    
    Bmax= np.max(np.linalg.norm(B, axis=-1))
    bn = np.sum(nn * B, axis=-1)
    if args['Bn_extern'] != 0:
        Bn = abs(bn - args['Bn_extern_surface'])
    else:
        Bn = abs(bn)
    Bn_mean = np.sum(Bn / np.linalg.norm(B, axis=-1) * sg) / np.sum(sg)
    Bn_max = np.max(abs(Bn))

    return  Bn_mean, Bn_max, Bmax, B, Bn


def average_length(args, coil):      #new
    nic = args['number_independent_coils']   
    deltal = np.zeros((nic, coil.shape[3], 3))   
    r_coil = np.mean(coil[:nic], axis = (1,2))   # 有限截面平均
    deltal = deltal.at[:, :-1, :].set(r_coil[:, 1:, :] - r_coil[:, :-1, :])
    deltal = deltal.at[:, -1, :].set(r_coil[:, 0, :] - r_coil[:, -1, :])
    len = np.sum(np.linalg.norm(deltal, axis=-1)) / nic
    return len


def curvature(der1, der2):
    return np.cross(der1, der2) / (np.linalg.norm(der1, axis = -1)**3)[:,:,np.newaxis]
     

def curvature_mean(curva):
    return np.mean(abs(np.linalg.norm(curva, axis = -1)))

def curvature_max(curva):
    return np.max(abs(np.linalg.norm(curva, axis = -1)))


def torsion(args, der1, der2, der3, coil):       # new
    if args['coil_case'] == 'fourier':
        cross12 = np.cross(der1, der2)
        top = ( cross12[:, :, 0] * der3[:, :, 0] + 
                cross12[:, :, 1] * der3[:, :, 1] +
                cross12[:, :, 2] * der3[:, :, 2])
        bottom = np.linalg.norm(cross12, axis=-1) ** 2
        t = abs(top / bottom)     # NC x NS
    else:
        ns = args['number_segments']
        nic = args['number_independent_coils']
        coil = np.mean(coil[:nic], (1, 2))
        dt = 1 / ns ### 此参数为无关变量，都会被约分
        d1, d2, d3 = np.zeros((nic, ns, 3)), np.zeros((nic, ns, 3)), np.zeros((nic, ns, 3))
        d1 = d1.at[:, 1:-1, :].set((coil[:, 2:, :] - coil[:, :-2, :]) / 2 / dt)
        d1 = d1.at[:, 0, :].set((coil[:, 1, :] - coil[:, -1, :]) / 2 / dt)
        d1 = d1.at[:, -1, :].set((coil[:, 0, :] - coil[:, -2, :]) / 2 / dt)

        d2 = d2.at[:, 1:-1, :].set((d1[:, 2:, :] - d1[:, :-2, :]) / 2 / dt)
        d2 = d2.at[:, 0, :].set((d1[:, 1, :] - d1[:, -1, :]) / 2 / dt)
        d2 = d2.at[:, -1, :].set((d1[:, 0, :] - d1[:, -2, :]) / 2 / dt)

        d3 = d3.at[:, 1:-1, :].set((d2[:, 2:, :] - d2[:, :-2, :]) / 2 / dt)
        d3 = d3.at[:, 0, :].set((d2[:, 1, :] - d2[:, -1, :]) / 2 / dt)
        d3 = d3.at[:, -1, :].set((d2[:, 0, :] - d2[:, -2, :]) / 2 / dt)

        cross12 = np.cross(d1, d2)
        top = ( cross12[:, :, 0] * d3[:, :, 0] + 
                cross12[:, :, 1] * d3[:, :, 1] +
                cross12[:, :, 2] * d3[:, :, 2])
        bottom = np.linalg.norm(cross12, axis=-1) ** 2
        t = abs(top / bottom) 
    return t

def torsion_mean(tor):
    return np.mean(tor)

def torsion_max(tor):
    return np.max(tor)


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
    nic = args['number_independent_coils'] 
    rc = np.mean(coil[:nic], axis = (1,2))
    rs, _, _ = surface_data
    dr = (rc[:args['number_independent_coils'], :, np.newaxis, np.newaxis, :]
                - rs[np.newaxis, np.newaxis, :, :, :])
    dr = np.linalg.norm(dr, axis = -1)
    dcs_min = np.min(dr)
    return dcs_min


def B_coil_normal(B_reg, v2):
    B_coil_theta = np.arccos(np.sum(B_reg * v2, axis=-1) /np.linalg.norm(B_reg, axis=-1)-1e-8)                                    
    B_coil_theta = np.max(B_coil_theta)
    return B_coil_theta


def Jcrit(args, B_coil, strain):#, Bx, By
    k2 = 0.2**2
    beta = 0.65
    nic = args['number_independent_coils']
    # assert nic == len(B_self)
    jc0 = np.zeros((nic))
    Bc = np.zeros((nic))
    for i in range(nic):
        j,b,t = material_jcrit.get_critical_current(args['HTS_temperature'],
                        B_coil[i], strain[i], args['HTS_material'])
        jc0 = jc0.at[i].set(j)
        Bc = Bc.at[i].set(b)
    assert B_coil.all() < np.min(Bc)
    # jc = jc0 / (1 + (k2*Bx**2 + By**2)**0.5 / B_self) ** beta
    return jc0

def calculate_force(I, B_reg, dl):  ### 应该除以每个线圈的长度
    nic = B_reg.shape[0]
    dl = np.mean(dl[:nic], axis=(1,2))
    tan = dl / np.linalg.norm(dl, axis=-1)[:, :, np.newaxis]
    force = I[:nic, np.newaxis, np.newaxis] * np.cross(tan, B_reg)
    force = np.linalg.norm(force, axis=-1)
    force = np.max(force) / 1e6
    return force



def Icrit(args, j_crit):
    jc = np.array(j_crit)
    ln, lb = np.array(args['length_normal']), np.array(args['length_binormal'])
    I_crit = (jc * 1e-4 * ln * (args['number_normal'] - 1) * lb * (args['number_binormal'] - 1))
    return I_crit


def loss_save(args, coil_output_func, params, surface_data):
    """ 
    Computes the default loss: int (B dot n)^2 dA + weight_length * len(coils) 

    Input: params, a tuple of the fourier series for the coils and a fourier series for the rotation.

    Output: A scalar, which is the loss_val computed by the function. JAX will eventually differentiate
    this in an optimizer.
    """
    I, dl, coil, der1, der2, der3, v1, v2, binormal = coil_output_func(params)
    Bn_mean, Bn_max, B_max_surf, B, Bn = quadratic_flux(args, I, dl, coil, surface_data)
    length = average_length(args, coil)
    curva = curvature(der1, der2)
    k_mean = curvature_mean(curva)
    k_max = curvature_max(curva)
    tor = torsion(args, der1, der2, der3, coil)
    t_mean = torsion_mean(tor)
    t_max = torsion_max(tor)
    dcc_min = distance_cc(args, coil)
    dcs_min = distance_cs(args, coil, surface_data)
    strain = hts_strain.HTS_strain(args, curva, v2, dl)
    strain_max = np.max(strain)
    I = I * args['I_normalize']
    B_reg = B_self.loss_B_coil(args, coil, I, dl, v1, v2, binormal, curva, der2)
    B_coil_theta = B_coil_normal(B_reg, v2)
    force = calculate_force(I, B_reg, dl)
    B_coil = B_self.coil_self_B_rec(args, coil, I, dl, v1, v2, binormal, curva, der2) 
    # print(np.max(np.linalg.norm(B_coil, axis=-1),))
    B_coil_max = np.max(np.linalg.norm(B_coil, axis=-1), axis = (1,2))
    # print(B_coil_max)
    j_crit = Jcrit(args, B_coil_max, strain)#, Bx, By
    I_crit_coil = Icrit(args, j_crit)
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
        'loss_B_coil_theta' :   B_coil_theta,
        'loss_force'        :   force,
        'loss_Bn'           :   Bn,
        'loss_B'            :   B,
        'loss_curva'        :   curva,
        'loss_tor'          :   tor,
        'loss_B_coil'       :   B_coil,
        'loss_B_coil_max'   :   B_coil_max,
        'loss_strain_max'   :   strain_max,
        'loss_HTS_jcrit'    :   j_crit,
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





















