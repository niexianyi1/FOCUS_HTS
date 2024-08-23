## 计算与HTS相关的线圈应变

import jax.numpy as np
import numpy

def cn(args, coil_output_func, params):
    _, dl, _, der1, der2, _, v1, v2, _ = coil_output_func(params)
    curva = np.cross(der1, der2) / (np.linalg.norm(der1, axis = -1)**3)[:,:,np.newaxis]
    strain = HTS_strain(args, curva, v1, v2, dl)
    strain_max = np.max(strain)-args['target_HTS_strain']
    return strain_max 


def HTS_strain(args, curva, v1, v2, dl):
    width = args['HTS_signle_width']
    thickness = args['HTS_I_thickness']
    dl = np.mean(dl[:args['number_independent_coils']], axis=(1,2))
    bend = HTS_strain_hard_bend(width, curva, v1)
    tor = HTS_strain_tor(width, dl, v1)
    #easy_bend = HTS_strain_easy_bend(thickness, curva, v2)
    return bend + tor

def HTS_strain_hard_bend(width, curva, v):
    """弯曲应变,
    Args:
        w, 带材宽度
        v,有限截面坐标轴
        curva, 线圈曲率

    Returns:
        bend, 弯曲应变

    """
    ### 此处width应取单根线材的宽度，而非总线缆的宽度
    bend = width/2*abs(np.sum(abs(v * curva), axis=-1))
    return bend

def HTS_strain_tor(width, dl, v):
    """扭转应变,
    Args:
        w, 带材宽度
        v,有限截面坐标轴
        dl, 线圈点间隔

    Returns:
        tor, 扭转应变

    """
    eps = numpy.spacing(1)
    cosv = np.zeros((v.shape[0], v.shape[1]))
    cosv = cosv.at[:, :-1].set(np.sum(v[:, :-1, :] * v[:, 1:, :], axis=-1) ) # 此处分母都为1，省略
    cosv = cosv.at[:, -1].set(np.sum(v[:, -1, :] * v[:, 0, :], axis=-1) )
    if np.max(cosv) == 1 and np.min(cosv) > -1+1e-8:
        cosv = cosv - eps
    dtheta = np.arccos(cosv)
    
    # sinv = np.zeros((v.shape[0], v.shape[1]))
    # sinv = sinv.at[:, :-1].set(np.linalg.norm(np.cross(v[:, :-1, :], v[:, 1:, :]), axis = -1)) # 此处分母都为1，省略
    # sinv = sinv.at[:, -1].set(np.linalg.norm(np.cross(v[:, -1, :], v[:, 0, :]), axis = -1))
    # if np.min(sinv) == 0 and np.max(sinv) < 1-1e-8:
    #     sinv = sinv + 1e-8
    # dtheta = np.arcsin(sinv)
    dl = np.linalg.norm(dl, axis = -1)
    tor = width**2/12*(dtheta/dl)**2
    return tor



def HTS_strain_easy_bend(thickness, curva, v):

    bend = thickness/2*abs(np.sum(abs(v * curva), axis=-1))
    return bend






