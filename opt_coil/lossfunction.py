## Calculate loss function(also cost function) values.

import jax.numpy as np
from jax import jit, config
import numpy
import sys 
sys.path.append('HTS')
import hts_strain
import B_self
import material_jcrit
config.update("jax_enable_x64", True)
pi = np.pi


@jit
def loss_value(args, coil_output_func, params, surface_data):

    I, dl, coil, der1, der2, der3, v1, v2, binormal = coil_output_func(params)
    curva = curvature(der1, der2)
    I = I * args['I_normalize']
    weight, loss = weight_and_loss(args)

    if args['weight_bnormal'] != 0:
        Bn_mean, _, _, _ = quadratic_flux(args, I, dl, coil, surface_data)
        loss[0] = Bn_mean

    if args['weight_length'] != 0:
        len_mean, len_single = average_length(args, coil)
        if args['target_length_single'][0] != 0:
            length = np.sum((len_single - np.array(args['target_length_single'])) ** 2)
        elif args['target_length_mean'] != 0:
            length = (len_mean - args['target_length_mean']) ** 2
        elif args['target_length_mean'] == 0:
            length = len_mean
        loss[1] = length

    if args['weight_curvature'] !=0 or args['weight_curvature_max'] !=0:
        k_mean, k_max = curvature_mean_max(curva)
        k_max = np.max(np.array([k_max-args['target_curvature_max'], 0]))
        loss[2], loss[3] = k_mean, k_max
    
    if args['weight_torsion'] != 0 or args['weight_torsion_max'] != 0:
        tor = torsion(args, der1, der2, der3, coil)
        t_mean, t_max = torsion_mean_max(tor)
        t_max = np.max(np.array([t_max-args['target_torsion_max'], 0]))
        loss[4], loss[5] = t_mean, t_max

    if args['weight_distance_coil_coil'] != 0:
        dcc_min = distance_cc(args, coil)
        dcc_min = -np.min(np.array([-args['target_distance_coil_coil']+dcc_min, 0]))
        loss[6] = dcc_min

    if args['weight_distance_coil_surface'] != 0:
        dcs_min = distance_cs(args, coil, surface_data)
        dcs_min = -np.min(np.array([-args['target_distance_coil_surface']+dcs_min, 0]))
        loss[7] = dcs_min

    if args['weight_HTS_force_max'] != 0 or args['weight_HTS_force_mean'] != 0:
        B_reg = B_self.coil_B_force(args, coil, I, dl, v1, v2, binormal, curva, der2)
        force_max, force_mean = calculate_force(I, B_reg, dl)
        force_max = np.max(np.array([force_max-args['target_HTS_force_max'], 0]))
        loss[8] = force_max
        loss[9] = force_mean

    if args['weight_HTS_Icrit'] != 0:
        strain = hts_strain.HTS_strain(args, curva, v1, v2, dl)
        strain = np.max(strain, axis=1)
        B_coil = B_self.coil_self_B_rec(args, coil, I, dl, v1, v2, binormal, curva, der2)
        B_coil_theta = calculate_B_theta(B_coil, v1, v2)
        j_crit, _ = Jcrit(args, B_coil, strain, B_coil_theta)
        I_crit = Icrit(args, j_crit)
        loss_I = np.min(I_crit)
        loss[10] = loss_I

    if args['weight_HTS_strain'] != 0:
        strain = hts_strain.HTS_strain(args, curva, v1, v2, dl)
        strain_mean = np.mean(strain)
        loss[11] = strain_mean

    loss = np.array(loss)

    if args['loss_weight_normalization'] == 0:
        lossvalue = np.sum(np.array(weight) * np.array(loss)
)
    if args['loss_weight_normalization'] == 1:
        for i in range(1, 12):
            if loss[i] == 0:
                weight[i] = 0
            else:
                weight[i] = weight[i] / loss[i]
        lossvalue = np.sum(np.array(weight) * np.array(loss))

    return lossvalue


def weight_and_loss(args):
    weight = []
    for key in args:
        k = key.split('_')
        if k[0] == 'weight':
            weight.append(args['{}'.format(key)])
    l = len(weight)
    loss = [0 for i in range(l)]
    return weight, loss



def quadratic_flux(args, I, dl, coil, surface_data):
    """ Calculate the normal error field. """
    r_surf, nn, sg = surface_data
    I = I / args['number_normal'] / args['number_binormal'] 

    mu_0 = 1e-7 
    mu_0I = I * mu_0
    mu_0Idl = (mu_0I[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] * dl) 
    r_minus_l = (r_surf[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis, :]
        - coil[:, np.newaxis, np.newaxis, :, :, :, :])  
    top = np.cross(mu_0Idl[:, np.newaxis, np.newaxis, :, :, :, :], r_minus_l)  
    bottom = (np.linalg.norm(r_minus_l, axis=-1) ** 3)  
    B = np.sum(top / bottom[:, :, :, :, :, :, np.newaxis], axis=(0, 3, 4, 5)) 

    Bmax = np.max(np.linalg.norm(B, axis=-1))
    bn = np.sum(nn * B, axis=-1)
    if args['Bn_background'] != 0:
        Bn = bn + args['Bn_background_surface']
    else:
        Bn = bn

    Bn_mean = np.sum(abs(Bn) / np.linalg.norm(B, axis=-1) * sg) / np.sum(sg)

    # Bn_mean = np.sum((Bn)**2 / np.linalg.norm(B, axis=-1) * sg) / np.sum(sg)

    return  Bn_mean, Bmax, B, Bn


def average_length(args, coil):   
    """ Calculate the length of coils. """  
    nic = args['number_independent_coils']   
    deltal = np.zeros((nic, coil.shape[3], 3))   
    r_coil = np.mean(coil[:nic], axis = (1,2))   
    deltal = deltal.at[:, :-1, :].set(r_coil[:, 1:, :] - r_coil[:, :-1, :])
    deltal = deltal.at[:, -1, :].set(r_coil[:, 0, :] - r_coil[:, -1, :])
    len_mean = np.sum(np.linalg.norm(deltal, axis=-1)) / nic
    len_single = np.sum(np.linalg.norm(deltal, axis=-1), axis=1)
    return len_mean, len_single


def curvature(der1, der2):
    return np.cross(der1, der2) / (np.linalg.norm(der1, axis = -1)**3)[:,:,np.newaxis]
     

def curvature_mean_max(curva):
    k = abs(np.linalg.norm(curva, axis = -1))
    k_mean = np.mean(k)
    k_max = np.max(k)
    return k_mean, k_max


def torsion(args, der1, der2, der3, coil):   
    ''' Calculate torsion of coils. 
    There is no continuous third derivative for Cubic(k=3) B-spline, 
    so it is calculated by microcentral difference method'''    
    if args['coil_case'] == 'fourier':
        cross12 = np.cross(der1, der2)
        top = ( cross12[:, :, 0] * der3[:, :, 0] + 
                cross12[:, :, 1] * der3[:, :, 1] +
                cross12[:, :, 2] * der3[:, :, 2])
        bottom = np.linalg.norm(cross12, axis=-1) ** 2
        t = abs(top / bottom)     
    else:
        ns = args['number_segments']
        nic = args['number_independent_coils']
        coil = np.mean(coil[:nic], (1, 2))
        dt = 1 / ns 
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

def torsion_mean_max(tor):
    t_mean = np.mean(tor)
    t_max = np.max(tor)
    return t_mean, t_max


def distance_cc(args, coil):  
    ''' Distance between adjacent coils. '''
    nic = args['number_independent_coils']  
    ns = args['number_segments']
    rc = np.mean(coil, axis = (1,2))
    if nic == args['number_coils']:
        dr = np.zeros((nic-1, ns, ns, 3))
        dr = dr.at[:nic-1].set(rc[:nic-1, :, np.newaxis, :] - rc[1:nic, np.newaxis, :, :])
    else:
        dr = np.zeros((nic, ns, ns, 3))
        dr = dr.at[:nic-1].set(rc[:nic-1, :, np.newaxis, :] - rc[1:nic, np.newaxis, :, :])
        dr = dr.at[nic-1].set(rc[nic-1, :, np.newaxis, :] - rc[2*nic-1, np.newaxis, :, :])
        dr = dr.at[nic].set(rc[2*nic, :, np.newaxis, :] - rc[nic, np.newaxis, :, :])
    dr = np.linalg.norm(dr, axis = -1)
    dcc_min = np.min(dr)
    return dcc_min

def distance_cs(args, coil, surface_data): 
    ''' Distance between magnetic surface and coils.'''
    nic = args['number_independent_coils'] 
    rc = np.mean(coil[:nic], axis = (1,2))
    rs, _, _ = surface_data
    dr = (rc[:args['number_independent_coils'], :, np.newaxis, np.newaxis, :]
                - rs[np.newaxis, np.newaxis, :, :, :])
    dr = np.linalg.norm(dr, axis = -1)
    dcs_min = np.min(dr)
    return dcs_min


def calculate_B_theta(B_reg, v2):
    eps = numpy.spacing(1)
    B_coil_theta = np.arccos(np.sum(B_reg * v2, axis=-1) / np.linalg.norm(B_reg, axis=-1) - eps)                                    
    B_coil_theta = np.max(B_coil_theta)
    return B_coil_theta


def Jcrit(args, B_coil, strain, B_coil_theta):
    k2 = 0.2**2
    beta = 0.65
    nic = args['number_independent_coils']
    # assert nic == len(B_self)
    jc0 = np.zeros((nic))
    Bc = np.zeros((nic))
    for i in range(nic):
        j,b,t = material_jcrit.get_critical_current(args['HTS_temperature'],
                        B_coil[i], strain[i], args['material'])
        jc0 = jc0.at[i].set(j)
        Bc = Bc.at[i].set(b)
    if B_coil.all() > np.min(Bc):
        print('warning: magnetic field in coils over the critical field of HTS.')

    jc = jc0 / (1 + (k2*(B_coil*np.cos(B_coil_theta))**2 + 
            (B_coil*np.sin(B_coil_theta))**2)**0.5 / B_coil) ** beta
    return jc, Bc

def calculate_force(I, B_reg, dl):  
    nic = B_reg.shape[0]
    dl = np.mean(dl[:nic], axis=(1,2))
    tan = dl / np.linalg.norm(dl, axis=-1)[:, :, np.newaxis]
    force = I[:nic, np.newaxis, np.newaxis] * np.cross(tan, B_reg)
    force_coil = force * np.linalg.norm(dl, axis=-1)[:, :, np.newaxis]
    force_coil = np.sum(np.linalg.norm(np.sum(force_coil , axis=1),axis=-1)) / nic
    force_max = np.max(np.linalg.norm(force, axis=-1))
    return force_max, force_coil

def Icrit(args, j_crit):
    jc = np.array(j_crit)
    ln, lb = np.array(args['length_normal']), np.array(args['length_binormal'])
    I_crit = jc * ln * (args['number_normal'] - 1) * lb * (args['number_binormal'] - 1)
    I_crit = I_crit * args['HTS_structural_percent'] * args['HTS_I_percent']  
    return I_crit


def loss_save(args, coil_output_func, params, surface_data):
    """After the optimization is finished, the parameters are saved through the dictionary"""
    I, dl, coil, der1, der2, der3, v1, v2, binormal = coil_output_func(params)
    I = I * args['I_normalize']
    Bn_mean, B_max_surf, B, Bn = quadratic_flux(args, I, dl, coil, surface_data)
    len_mean, len_single = average_length(args, coil)
    curva = curvature(der1, der2)
    k_mean, k_max = curvature_mean_max(curva)
    tor = torsion(args, der1, der2, der3, coil)
    t_mean, t_max = torsion_mean_max(tor)
    dcc_min = distance_cc(args, coil)
    dcs_min = distance_cs(args, coil, surface_data)
    strain = hts_strain.HTS_strain(args, curva, v1, v2, dl)
    strain_coil = np.max(strain, axis=1)
    strain_max = np.max(strain)
    B_reg = B_self.coil_B_force(args, coil, I, dl, v1, v2, binormal, curva, der2)
    force_max, force_mean = calculate_force(I, B_reg, dl)
    B_coil_theta = calculate_B_theta(B_reg, v2)
    B_coil = B_self.coil_self_B_rec(args, coil, I, dl, v1, v2, binormal, curva, der2) 
    B_coil_max = np.max(np.linalg.norm(B_coil, axis=-1), axis = (1,2))
    j_crit, Bc = Jcrit(args, B_coil_max, strain_coil, B_coil_theta)
    if B_coil_max.all() > np.min(Bc):
        print("Warning : B_coil_max > B_crit")
    I_coil_crit = Icrit(args, j_crit)
    if I_coil_crit.all() < abs(I).all():
        print("Warning : I_coil > I_coil_crit")

    print('**********loss_functions_value**********')
    print('loss_Bn_mean = ', Bn_mean)
    print('loss_length_mean =  ', len_mean, 'm')
    print('loss_length_single = ', len_single, 'm')
    print('loss_curvature =  ', k_mean, '1/m')
    print('loss_curva_max =  ', k_max, '1/m')
    print('loss_tor_mean =  ', t_mean, '1/m')
    print('loss_tor_max =  ', t_max, '1/m')
    print('loss_dcc_min =  ', dcc_min, 'm')
    print('loss_dcs_min =  ', dcs_min, 'm')
    print('loss_strain_max =  ', strain_max)
    print('loss_force_max =  ', force_max, 'N/m')
    print('loss_force_mean =  ', force_mean, 'N')
    print('loss_B_coil_max = ', B_coil_max, 'T')
    print('loss_HTS_Icrit = ', I_coil_crit, 'A')
    print('loss_HTS_jcrit = ', j_crit, 'GA/m2')
    
    loss_end = {
        'loss_Bn_mean'      :   Bn_mean,
        'loss_B_max_surf'   :   B_max_surf,
        'loss_length_mean'  :   len_mean,
        'loss_length_single':   len_single,
        'loss_curvature'    :   k_mean,
        'loss_curva_max'    :   k_max,
        'loss_dcc_min'      :   dcc_min,
        'loss_dcs_min'      :   dcs_min,
        'loss_tor_mean'     :   t_mean,
        'loss_tor_max'      :   t_max,
        'loss_B_coil_theta' :   B_coil_theta,
        'loss_force_max'    :   force_max,
        'loss_force_mean'   :   force_mean,
        'loss_Bn'           :   Bn,
        'loss_B'            :   B,
        'loss_curva'        :   curva,
        'loss_tor'          :   tor,
        'loss_B_coil'       :   B_coil,
        'loss_B_coil_max'   :   B_coil_max,
        'loss_strain'       :   strain,
        'loss_strain_max'   :   strain_max,
        'loss_HTS_jcrit'    :   j_crit,
        'loss_HTS_Icrit'    :   I_coil_crit
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





















