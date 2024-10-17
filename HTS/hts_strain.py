## Calculate the HTS tape strain.

import jax.numpy as np
import numpy

def cn(args, coil_output_func, params):
    '''This function is an inequality constraint for nlopt.'''
    _, dl, _, der1, der2, _, v1, v2, _ = coil_output_func(params)
    curva = np.cross(der1, der2) / (np.linalg.norm(der1, axis = -1)**3)[:,:,np.newaxis]
    strain = HTS_strain(args, curva, v1, v2, dl)
    strain_max = np.max(strain)-args['target_HTS_strain']
    return strain_max 


def HTS_strain(args, curva, v1, v2, dl):
    '''Calculate the HTS tape strain.'''
    width = args['HTS_single_width']
    thickness = args['HTS_single_thickness']
    dl = np.mean(dl[:args['number_independent_coils']], axis=(1,2))
    hard_bend = HTS_strain_hard_bend(width, curva, v1)
    tor = HTS_strain_tor(width, dl, v1)
    easy_bend = HTS_strain_easy_bend(thickness, curva, v2)
    return hard_bend + tor + easy_bend

def HTS_strain_tor(width, dl, v):
    '''d_theta can be calculated by arcsin or arccos, but behave slightly differently in optimization'''
    eps = numpy.spacing(1)
    # cosv = np.zeros((v.shape[0], v.shape[1]))
    # cosv = cosv.at[:, :-1].set(np.sum(v[:, :-1, :] * v[:, 1:, :], axis=-1) ) # 此处分母都为1，省略
    # cosv = cosv.at[:, -1].set(np.sum(v[:, -1, :] * v[:, 0, :], axis=-1) )
    # if np.max(cosv) == 1 and np.min(cosv) > -1+1e-8:
    #     cosv = cosv - eps
    # dtheta = np.arccos(cosv)
    
    sinv = np.zeros((v.shape[0], v.shape[1]))
    sinv = sinv.at[:, :-1].set(np.linalg.norm(np.cross(v[:, :-1, :], v[:, 1:, :]), axis = -1)) # 此处分母都为1，省略
    sinv = sinv.at[:, -1].set(np.linalg.norm(np.cross(v[:, -1, :], v[:, 0, :]), axis = -1))
    if np.min(sinv) == 0 and np.max(sinv) < 1-1e-8:
        sinv = sinv + eps
    dtheta = np.arcsin(sinv)

    dl = np.linalg.norm(dl, axis = -1)
    tor = width**2/12*(dtheta/dl)**2
    return tor

def HTS_strain_hard_bend(width, curva, v):
    bend = width/2*abs(np.sum(abs(v * curva), axis=-1))
    return bend

def HTS_strain_easy_bend(thickness, curva, v):
    bend = thickness/2*abs(np.sum(abs(v * curva), axis=-1))
    return bend






