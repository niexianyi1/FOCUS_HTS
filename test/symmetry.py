
import jax.numpy as np
import plotly.graph_objects as go


def stellarator_symmetry_coil(r):
    """计算线圈的仿星器对称"""
    rc = np.zeros((128, 64, 3))
    rc = rc.at[:64, :, :].set(r)
    T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rc = rc.at[64:128, :, :].set(np.dot(r, T))
    return rc

def symmetry_coil(r):
    """计算线圈的周期对称"""
    npc = 128   # 每周期线圈数，number of coils per period
    rc_total = np.zeros((256, 64, 3))
    rc_total = rc_total.at[0:npc, :, :].set(r)
    for i in range(1):        
        theta_t = 2 * np.pi * (i + 1) / 2
        T = np.array([[np.cos(theta_t), -np.sin(theta_t), 0],
                [np.sin(theta_t), np.cos(theta_t), 0], [0, 0, 1]])
        rc_total = rc_total.at[npc*(i+1):npc*(i+2), :, :].set(np.dot(r, T))
    
    return rc_total

def symmetry_sg(sg):
    sgn = np.zeros((256, 64))
    sgn = sgn.at[:64].set(sg)
    sgn = sgn.at[64:128].set(sg)
    sgn = sgn.at[128:].set(sgn[:128])
    return sgn


rs = np.load('initfiles/ellipse/r_surf.npy')
nn = np.load('initfiles/ellipse/nn_surf.npy')
sg = np.load('initfiles/ellipse/sg_surf.npy')

















