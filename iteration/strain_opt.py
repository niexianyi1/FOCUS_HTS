
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
import sys
sys.path.append('HTS')
import hts_strain
config.update("jax_enable_x64", True)
config.update('jax_disable_jit', True)
n = 0



def loss_strain(args, coil_output_func, params):
    _, dl, _, der1, der2, _, v1, v2, _ = coil_output_func(params)
    curva = np.cross(der1, der2) / (np.linalg.norm(der1, axis = -1)**3)[:,:,np.newaxis]
    strain = hts_strain.HTS_strain(args, curva, v1, v2, dl)
    strain_max = np.max(np.array([np.max(strain) - args['target_HTS_strain'], 0]))
    return strain_max



def main():
    
    with open('initfiles/init_args.json', 'r') as f:    # 传入地址
        args = json.load(f)
    globals().update(args)
    
    # 获取初始数据
    args, coil_arg_init, fr_init, surface_data, I_init = read_init.init(args)

    loss_vals = []
    # 两种迭代函数

    @jit
    def objective_function_minimize(args, params):
        global n
        n = n + 1 
        params = list_to_params(params)   
        coil_cal = CoilSet(args)
        coil_output_func = coil_cal.cal_coil           
        loss_val, gradient = value_and_grad(
            lambda params :loss_strain(args, coil_output_func, params),
            allow_int = True)(params)    
        g = compute_grad(args, gradient)
        loss_vals.append(loss_val)
        print('iter = ', n, 'value = ', loss_val)
        return loss_val, g

    @jit
    def objective_function_nlopt(params, grad):#=None
        global n
        n = n + 1 
        params = list_to_params(params)   
        coil_cal = CoilSet(args)
        coil_output_func = coil_cal.cal_coil           
        loss_val, gradient = value_and_grad(
            lambda params :loss_strain(args, coil_output_func, params),
            allow_int = True)(params)    
        if grad.size > 0:
            g = compute_grad(args, gradient)
            grad[:] = list(numpy.array(g))
        loss_val = numpy.float64(loss_val)    
        loss_vals.append(loss_val)
        print('iter = ', n, 'value = ', loss_val)
        return loss_val


    @jit
    def list_to_params(params):
        nic = args['number_independent_coils']
        if nic!=1:
            fr = np.reshape(params[-2 * nic * args['number_fourier_rotate'] - nic+1:-nic+1],
                                (nic, 2, args['number_fourier_rotate']))   
            I = params[-nic+1:]
        else:
            fr = np.reshape(params[-2 * nic * args['number_fourier_rotate'] :],
                                (nic, 2, args['number_fourier_rotate']))   
            I = np.array([])

        def coil_arg_fourier(args, params, nic):
            return np.reshape(params[:6 * nic * args['number_fourier_coils']], 
                                (nic, 6, args['number_fourier_coils']) )
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


    # 迭代循环, 得到导数, 带入变量, 得到新变量
    # 分两种情况, 固定迭代次数、给定迭代目标残差
    start = time.time()
    
    if args['iter_method'] == 'min':
        params = np.append(np.append(coil_arg_init, fr_init), I_init[:-1])
        res = minimize(lambda params :objective_function_minimize(args, params), params, jac=True, 
                method = '{}'.format(args['minimize_method']), tol = args['minimize_tol'])
        success, loss_val, params, n = res.success, res.fun, res.x, res.nit
        print(success, 'loss = ', loss_val)  
        params = list_to_params(params)   
   
    elif args['iter_method'] == 'nlopt':
        for i in range(args['number_independent_coils']):
            coil_arg_init, fr_init = coil_arg_init[i], fr_init[i]
            params = np.append(np.append(coil_arg_init, fr_init), [])
            opt = read_init.nlopt_op(args, params)
            opt.set_min_objective(objective_function_nlopt)
            opt.set_ftol_rel(args['stop_criteria'])
            xopt = opt.optimize(params)
            print('loss = ', opt.last_optimum_value())  
            params = list_to_params(xopt)   

    end = time.time()
    print('time cost = ', end - start)
    
    # 得到优化后的线圈参数, 保存到文件中
    coil_cal = CoilSet(args)
    coil_output_func = coil_cal.cal_coil 
    coil_all = coil_cal.end_coil(params)
    loss_end = lossfunction.loss_save(args, coil_output_func, params, surface_data)
    save.save_file(args, loss_vals, coil_all, loss_end, surface_data)
    plot.plot(args, coil_all, loss_end, loss_vals, params, surface_data)
    

