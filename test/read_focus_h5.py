

# 'CG_maxiter','CG_wolfe_c1', 'CG_wolfe_c2', 'CG_xtol', 
# 'DF_maxiter', 'DF_tauend', 'DF_tausta', 'DF_xtol', 
# 'Gnorm', 'HN_factor', 'HN_maxiter', 'HN_xtol', 'ISNormWeight', 
# 'Inorm', 'IsNormalize', 'IsQuiet', 'IsSymmetric', 'IsVaryCurrent', 'IsVaryGeometry',
# 'Mnorm', 'NFcoil', 'Ncoils', 'Nfp', 'Nseg', 'Nteta', 'Nzeta',
# 'TN_cr', 'TN_maxiter', 'TN_reorder', 'TN_xtol', 'axis_npoints', 'case_bnormal', 
# 'case_coils', 'case_curv', 'case_init', 'case_length', 'case_optimize',
# 'case_postproc', 'case_straight', 'case_surface', 'case_tors', 
# 'ccsep_alpha', 'ccsep_beta', 'ccsep_skip', 
# 'coilspace', 'curv_alpha', 'curv_k0', 'deriv',
# 'evolution', 'exit_xtol', ,  'overlap','plas_Bn', 
# 'init_current', 'init_radius', 'input_coils', 'input_harm', 'input_surf', 'iout', 
# 'nissin0', 'nissin_alpha', 'nissin_beta', 'nissin_gamma', 'nissin_sigma',  'penfun_nissin', 
# 'pp_maxiter', 'pp_ns', 'pp_phi', 'pp_raxis', 'pp_rmax', 'pp_xtol', 'pp_zaxis', 'pp_zmax', 
# 'save_coils', 'save_filaments', 'save_freq', 'save_harmonics', 
# , 'target_isum', 'target_length', 'target_tflux', 
# 'time_initialize', 'time_optimize', 'time_postproc', 
# 'tors0', 'tors_alpha', 'update_plasma', 'version','knotsurf'
# 'weight_bharm', 'weight_bnorm', 'weight_ccsep', 'weight_cssep', 
# 'weight_curv', 'weight_gnorm', 'weight_inorm', 'weight_isum', 
# 'weight_mnorm', 'weight_sbnorm', 'weight_specw', 'weight_straight',
# 'weight_tflux', 'weight_tors', 'weight_ttlen', 



# 需要的参数：'xsurf', 'ysurf', 'zsurf', 'nn', 'nx', 'ny', 'nz','Bn', 'Bx', 'By', 'Bz',
#           'xt', 'xx', 'yt', 'yy', 'zt', 'zz'
#  'xsurf': 是半周期的参数。
# 'surf_vol' 磁面表面积？

import h5py
import jax.numpy as np
import plotly.graph_objects as go


def read_hdf5(filename):
    f = h5py.File(filename, "r")
    args = {}
    print(f.keys())
    for key in list(f.keys()):
        args.update({key: f[key][:]})
    f.close()
    return args

def symmetry(r):
    nic=128
    """计算线圈的仿星器对称"""
    rc = np.zeros((128*2, 128, 3))
    rc = rc.at[0:nic, :, :].set(r)
    T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rc = rc.at[nic:nic*2, :, :].set(np.dot(r, T))

    npc = 128*2   
    rc_total = np.zeros((128*4, 128, 3))
    rc_total = rc_total.at[:128*2, :, :].set(rc)
    for i in range(2- 1):        
        theta_t = 2 * np.pi * (i + 1) / 2
        T = np.array([[np.cos(theta_t), -np.sin(theta_t), 0],
                [np.sin(theta_t), np.cos(theta_t), 0], [0, 0, 1]])
        rc_total = rc_total.at[npc*(i+1):npc*(i+2), :, :].set(np.dot(rc, T))
        
    return rc_total


ellipse = read_hdf5('/home/nxy/codes/coil_spline_HTS/initfiles/jpark/focus_jpark.h5')


xsurf = ellipse['xsurf']
ysurf = ellipse['ysurf']
zsurf = ellipse['zsurf']
surf = np.array([xsurf,ysurf,zsurf])
surf = np.transpose(surf, (1,2,0))
surf = symmetry(surf)

s= np.reshape(surf, (128*128*4, 3))
# s = np.zeros((128,128,3))
# for i in range(128):
#     s = s.at[i,:,:].set(surf[i*4])
# s = np.reshape(s, (128*128, 3))
np.save()
fig = go.Figure()
fig.add_scatter3d(x=s[:,0],y=s[:,1],z=s[:,2], name='coil', mode='markers', marker_size = 1.5)   
fig.update_layout(scene_aspectmode='data')
fig.show()
