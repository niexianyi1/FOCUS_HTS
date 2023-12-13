

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

def read_hdf5(filename):
    f = h5py.File(filename, "r")
    args = {}
    for key in list(f.keys()):
        args.update({key: f[key][:]})
    f.close()
    return args

ellipse = read_hdf5('/home/nxy/codes/focusadd-spline/results_b/ellipse/focus_ellipse.h5')
print(ellipse['xt'])


