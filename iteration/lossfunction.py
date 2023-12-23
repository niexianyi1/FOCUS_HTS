import jax.numpy as np
from jax import jit
from jax.config import config
config.update("jax_enable_x64", True)
pi = np.pi

# 用class包括所有的loss function.


@jit
def loss(args, coil_output_func, params, surface_data):
    """ 
    Computes the default loss: int (B dot n)^2 dA + weight_length * len(coils) 

    Input: params, a tuple of the fourier series for the coils and a fourier series for the rotation.

    Output: A scalar, which is the loss_val computed by the function. JAX will eventually differentiate
    this in an optimizer.
    """
    I, dl, coil, der1, der2, der3, v1 = coil_output_func(params)
    Bn_mean, Bn_max = quadratic_flux(I, dl, coil, surface_data)
    length, deltal = average_length(coil)
    k_mean, k_max, curva = curvature(der1, der2)
    t_mean, t_max = torsion(der1, der2, der3)
    dcc_min = distance_cc(coil)
    dcs_min = distance_cs(coil, surface_data)
    bend = HTS_strain_bend(args, curva, v1)
    tor = HTS_strain_tor(args, deltal, v1)

    return (  args['weight_bnormal']    * Bn_mean   + args['weight_length']         * length 
            + args['weight_curvature']  * k_mean    + args['weight_curvature_max']  * k_max 
            + args['weight_torsion']    * t_mean    + args['weight_torsion_max']    * t_max 
            + args['weight_distance_coil_coil'] * dcc_min + args['weight_distance_coil_surface'] * dcs_min )

        
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
    B = biotSavart(coil, I, dl, r_surf)  # NZ x NT x 3

    if args['B_extern'] is not None:
        B_all = B + args['B_extern']
    else:
        B_all = B
    Bn = np.sum(nn * B_all, axis=-1)
    Bn_mean = 0.5*np.sum((Bn/ np.linalg.norm(B, axis=-1))** 2 * sg)
    Bn_max = np.max(abs(Bn))
    return  Bn_mean, Bn_max

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
    mu_0Idl = (mu_0I[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] * dl)  # NC x NS x NNR x NBR x 3
    r_minus_l = (r_surf[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis, :]
        - coil[:, np.newaxis, np.newaxis, :, :, :, :])  # NC x NZ/nfp x NT x NS x NNR x NBR x 3
    top = np.cross(mu_0Idl[:, np.newaxis, np.newaxis, :, :, :, :], r_minus_l)  # NC x NZ x NT x NS x NNR x NBR x 3
    bottom = (np.linalg.norm(r_minus_l, axis=-1) ** 3)  # NC x NZ x NT x NS x NNR x NBR
    B = np.sum(top / bottom[:, :, :, :, :, :, np.newaxis], axis=(0, 3, 4, 5))  # NZ x NT x 3
    return B

# def normalized_error(r, I, dl, l, nn, sg, B_extern = None):
#     B = LossFunction.biotSavart(r, I, dl, l)  # NZ x NT x 3
#     if B_extern is not None:
#         B = B + B_extern

#     B_n = np.abs( np.sum(nn * B, axis=-1) )
#     B_mag = np.linalg.norm(B, axis=-1)
#     A = np.sum(sg)

#     return np.sum( (B_n / B_mag) * sg ) / A

### 新增

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
    al = np.zeros((nic, coil.shape[1], 3))   
    r_coil = np.mean(coil, axis = (2,3))
    al = al.at[:, :-1, :].set(r_coil[:nic, 1:, :] - r_coil[:nic, :-1, :])
    al = al.at[:, -1, :].set(r_coil[:nic, 0, :] - r_coil[:nic, -1, :])
    len = np.sum(np.linalg.norm(al, axis=-1)) / nic
    return len, al

def distance_cc(args, coil):  ### 暂未考虑finite-build
    nic = args['number_independent_coils']  
    rc = np.mean(coil, axis = (2,3))
    dr = rc[:nic, :, :] - rc[1:nic+1, :, :]
    dr = np.linalg.norm(dr, axis = -1)
    dcc_min = np.min(dr)
    return dcc_min

def distance_cs(args, coil, surface_data):  ### 暂未考虑finite-build
    rc = coil[:, :, 0, 0, :]
    rs, _, _ = surface_data
    dr = (rc[:args['number_independent_coils'], :, np.newaxis, np.newaxis, :]
                - rs[np.newaxis, np.newaxis, :, :, :])
    dr = np.linalg.norm(dr, axis = -1)
    dcs_min = np.min(dr)
    return dcs_min

##  HTS应变量
def HTS_strain_bend(args, curva, v1):
    """弯曲应变,
    Args:
        w, 带材宽度
        v1,有限截面坐标轴
        curva, 线圈曲率

    Returns:
        bend, 弯曲应变

    """

    bend = args['width']/2*abs(np.dot(-v1 , curva))
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

    dv = v1[:, :-1, :] * v1[:, 1:, :]
    dv = dv + v1[:, -1, :] * v1[:, 0, :]
    dtheta = np.arccos(dv)
    tor = args['width']**2/12*(dtheta/deltal)**2
    return tor



# def symmetry_B(args, B):
#     B_total = np.zeros((args.nz, args.nt, 3))
#     B_total = B_total.at[:, :, :].add(B)
#     for i in range(args.nfp - 1):        
#         theta = 2 * pi * (i + 1) / args.nfp
#         T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
#         B_total = B_total.at[:, :, :].add(np.dot(B, T))
    
#     return B_total





















