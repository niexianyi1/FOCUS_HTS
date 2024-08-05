
### 运行脚本, python执行即可

import jax.numpy as np
from jax import value_and_grad, jit, config
from scipy.optimize import minimize
import nlopt
import numpy
import json
import time
import read_init
from coilset import CoilSet    
import lossfunction   
import plot
import save
config.update("jax_enable_x64", True)
config.update('jax_disable_jit', True)

with open('initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)

# 获取初始数据
args, coil_arg_init, fr_init, surface_data, I_init = read_init.init(args) 
loss_vals = []
# 两种迭代函数


@jit
def objective_function_nlopt(params, grad):#=None
    global n
    n = n + 1 
    params = list_to_params(params)   
    coil_cal = CoilSet(args)
    coil_output_func = coil_cal.cal_coil           
    loss_val, gradient = value_and_grad(
        lambda params :lossfunction.loss(args, coil_output_func, params, surface_data),
        allow_int = True)(params)    
    if grad.size > 0:
        g = compute_grad(args, gradient)
        grad[:] = list(numpy.array(g))
    loss_val = numpy.float64(loss_val)    
    loss_vals.append(loss_val)
    return  loss_val

@jit
def list_to_params(params):
    nic = args['number_independent_coils']
    fr = np.reshape(params[-2 * nic * args['number_fourier_rotate'] - nic+1:-nic+1],
                        (2, nic, args['number_fourier_rotate']))   
    I = params[-nic+1:]
    def coil_arg_fourier(args, params, nic):
        return np.reshape(params[:6 * nic * args['number_fourier_coils']], 
                            (6, nic, args['number_fourier_coils']) )
    def coil_arg_spline(args, params, nic):
        return np.reshape(params[:nic * 3 * (args['number_control_points']-3)], 
                            (nic, 3, (args['number_control_points']-3)) )
    compute_coil_arg = dict()
    compute_coil_arg['fourier'] = coil_arg_fourier
    compute_coil_arg['spline'] = coil_arg_spline
    compute_func = compute_coil_arg[args['coil_case']]
    coil_arg = compute_func(args, params, nic)
    params = (coil_arg, fr, I)  
    return params

@jit
def compute_grad(args, gradient):
    if args['coil_optimize'] == 0:
        if args['coil_case'] == 'fourier':
            grad0 = np.array([0 for i in range(
                6 * args['number_fourier_coils'] * args['number_independent_coils'])])
        else:
            grad0 = np.array([0 for i in range(
                3 * (args['number_control_points'] - 3) * args['number_independent_coils'])])
    else:
        grad0 = gradient[0]
    if args['alpha_optimize'] == 0:
        grad1 = np.array([0 for i in range(
            2 * args['number_fourier_rotate'] * args['number_independent_coils'])])
    else:
        grad1 = gradient[1]
    if args['I_optimize'] == 0:
        grad2 = np.array([0 for i in range(args['number_independent_coils']-1)])
    else:
        grad2 = gradient[2]
    grad = np.append(np.append(grad0, grad1), grad2)
    return grad

loss_vals = []
n = 0
start = time.time()
args['out_hdf5'] = 'results/w7x/w7x_start/spline3.h5'

opt_init_coil_arg, opt_update_coil_arg, get_params_coil_arg = read_init.args_to_op(
    args, args['optimizer_coil'], args['step_size_coil'])
opt_init_fr, opt_update_fr, get_params_fr = read_init.args_to_op(
    args, args['optimizer_alpha'], args['step_size_alpha'])
opt_init_I, opt_update_I, get_params_I = read_init.args_to_op(
    args, args['optimizer_I'], args['step_size_I'])   
opt_state_coil_arg = opt_init_coil_arg(coil_arg_init)
opt_state_fr = opt_init_fr(fr_init)  
opt_state_I = opt_init_I(I_init[:-1])  
for i in range(args['number_iteration']):
    opt_state_coil_arg, opt_state_fr, opt_state_I, loss_val = objective_function_jax(
        args, opt_state_coil_arg, opt_state_fr, opt_state_I)
    loss_vals.append(loss_val)
    print('iter = ', i, 'value = ', loss_val)
params = (get_params_coil_arg(opt_state_coil_arg), get_params_fr(opt_state_fr),
            get_params_I(opt_state_I))  
end = time.time()
print('time cost = ', end - start, 'n = ', n, )
coil_cal = CoilSet(args)
coil_output_func = coil_cal.cal_coil 
coil_all = coil_cal.end_coil(params)
loss_end = lossfunction.loss_save(args, coil_output_func, params, surface_data)
save.save_file(args, loss_vals, coil_all, loss_end, surface_data)

loss_vals = []
n = 0
start = time.time()
args['coil_case']='fourier'
args, coil_arg_init, fr_init, surface_data, I_init = read_init.init(args) 
args['out_hdf5'] = 'results/w7x/w7x_start/fourier3.h5'
params = np.append(np.append(coil_arg_init, fr_init), I_init[:-1])
opt = nlopt.opt(nlopt.LD_MMA, len(params))
opt.set_min_objective(objective_function_nlopt)
opt.set_ftol_rel(1e-4)
xopt = opt.optimize(params)
print('LD_MMA, 1e-4, loss = ', opt.last_optimum_value())  
params = list_to_params(xopt)   
end = time.time()
print('time cost = ', end - start, 'n = ', n, )
coil_cal = CoilSet(args)
coil_output_func = coil_cal.cal_coil 
coil_all = coil_cal.end_coil(params)
loss_end = lossfunction.loss_save(args, coil_output_func, params, surface_data)
save.save_file(args, loss_vals, coil_all, loss_end, surface_data)



