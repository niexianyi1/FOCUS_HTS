
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


def main():
    
    with open('initfiles/init_args.json', 'r') as f:    # 传入地址
        args = json.load(f)
    globals().update(args)
    
    # 获取初始数据
    args, coil_arg_init, fr_init, surface_data, I_init = read_init.init(args)

    # loss计算准备
    # coil_cal = CoilSet(args)
    # args = coil_cal.get_fb_args(params)  
    loss_vals = []
    # 两种迭代函数
    @jit
    def objective_function_jax(args, opt_state_coil_arg, opt_state_fr, opt_state_I):
        """
        迭代过程, 通过value_and_grad得到导数值和loss值, 
        """   
        I = get_params_I(opt_state_I)
        params = (get_params_coil_arg(opt_state_coil_arg), get_params_fr(opt_state_fr), I)
        coil_cal = CoilSet(args)
        coil_output_func = coil_cal.cal_coil   
        loss_val, gradient = value_and_grad(
            lambda params :lossfunction.loss_value(args, coil_output_func, params, surface_data),
            allow_int = True)(params)      
        g_coil_arg, g_fr, g_I = gradient
        opt_state_coil_arg = opt_update_coil_arg(i, g_coil_arg, opt_state_coil_arg)
        opt_state_fr = opt_update_fr(i, g_fr, opt_state_fr)  
        opt_state_I = opt_update_I(i, g_I, opt_state_I)  
        return opt_state_coil_arg, opt_state_fr, opt_state_I, loss_val

    @jit
    def objective_function_minimize(args, params):
        global n
        n = n + 1 
        params = list_to_params(params)   
        coil_cal = CoilSet(args)
        coil_output_func = coil_cal.cal_coil           
        loss_val, gradient = value_and_grad(
            lambda params :lossfunction.loss_value(args, coil_output_func, params, surface_data),
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
            lambda params :lossfunction.loss_value(args, coil_output_func, params, surface_data),
            allow_int = True)(params)    
        if grad.size > 0:
            g = compute_grad(args, gradient)
            grad[:] = list(numpy.array(g))
        loss_val = numpy.float64(loss_val)    
        loss_vals.append(loss_val)
        print('iter = ', n, 'value = ', loss_val)
        return loss_val

    def constrain_nlopt(params, grad):
        params = list_to_params(params)   
        coil_cal = CoilSet(args)
        coil_output_func = coil_cal.cal_coil           
        value, gradient = value_and_grad(
            lambda params :hts_strain.cn(args, coil_output_func, params),
            allow_int = True)(params)    
        if grad.size > 0:
            g = compute_grad(args, gradient)
            grad[:] = list(numpy.array(g))
        value = numpy.float64(value)    
        print('constrain_value = ', value)
        return value


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

    if args['iter_method'] == 'jax':
        # 参数迭代算法        
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
    
    elif args['iter_method'] == 'min':
        params = np.append(np.append(coil_arg_init, fr_init), I_init[:-1])
        res = minimize(lambda params :objective_function_minimize(args, params), params, jac=True, 
                method = '{}'.format(args['minimize_method']), tol = args['minimize_tol'])
        success, loss_val, params, n = res.success, res.fun, res.x, res.nit
        print(success, 'loss = ', loss_val)  
        params = list_to_params(params)   
   
    elif args['iter_method'] == 'nlopt':
        params = np.append(np.append(coil_arg_init, fr_init), I_init[:-1])
        opt = read_init.nlopt_op(args, params)
        opt.set_min_objective(objective_function_nlopt)
        if args['inequality_constraint_strain'] == 1:
            opt.add_inequality_constraint(lambda params, grad:constrain_nlopt(params, grad), 1e-4)
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
    

