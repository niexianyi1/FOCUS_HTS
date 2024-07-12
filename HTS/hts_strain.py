## 计算与HTS相关的线圈应变

import jax.numpy as np


def cn(args, coil_output_func, params):
    _, dl, _, der1, der2, _, _, v2, _ = coil_output_func(params)
    curva = np.cross(der1, der2) / (np.linalg.norm(der1, axis = -1)**3)[:,:,np.newaxis]
    strain_max = HTS_strain(args, curva, v2, dl)
    return strain_max - args['target_strain']


def HTS_strain(args, curva, v2, dl):
    width = args['HTS_signle_width']
    dl = np.mean(dl[:args['number_independent_coils']], axis=(1,2))
    bend = HTS_strain_bend(width, curva, v2)
    tor = HTS_strain_tor(width, dl, v2)
    strain_max = np.max(bend + tor, axis=1)
    print(strain_max)
    return strain_max

def HTS_strain_bend(width, curva, v2):
    """弯曲应变,
    Args:
        w, 带材宽度
        v2,有限截面坐标轴
        curva, 线圈曲率

    Returns:
        bend, 弯曲应变

    """
    ### 此处width应取单根线材的宽度，而非总线缆的宽度
    print(curva)
    bend = width/2*abs(np.linalg.norm(-v2 * curva, axis=-1))
    return bend

def HTS_strain_tor(width, dl, v2):
    """扭转应变,
    Args:
        w, 带材宽度
        v2,有限截面坐标轴
        dl, 线圈点间隔

    Returns:
        bend, 弯曲应变

    # """
    cosv = np.zeros((v2.shape[0], v2.shape[1]))
    cosv = cosv.at[:, :-1].set(np.sum(v2[:, :-1, :] * v2[:, 1:, :], axis=-1)) # 此处分母都为1，省略
    cosv = cosv.at[:, -1].set(np.sum(v2[:, -1, :] * v2[:, 0, :], axis=-1))
    dtheta = np.arccos(cosv)
    # sinv = np.zeros((v2.shape[0], v2.shape[1]))
    # sinv = sinv.at[:, :-1].set(np.linalg.norm(np.cross(v2[:, :-1, :], v2[:, 1:, :]), axis = -1)+1e-8) # 此处分母都为1，省略
    # sinv = sinv.at[:, -1].set(np.linalg.norm(np.cross(v2[:, -1, :], v2[:, 0, :]), axis = -1)+1e-8)
    # dtheta = np.arcsin(sinv)
    # dtheta = sinv - 1/6*sinv**3
    dl = np.linalg.norm(dl, axis = -1)
    tor = width**2/12*(dtheta/dl)**2
    return tor