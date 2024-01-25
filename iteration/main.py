
### 运行脚本, python执行即可

import jax.numpy as np
from jax import value_and_grad, jit
import jax.example_libraries.optimizers as op
from jax.config import config
from scipy.optimize import minimize
import json
import time
import read_init
from coilset import CoilSet    
import lossfunction   
import plot
import save
config.update("jax_enable_x64", True)
config.update('jax_disable_jit', True)

def main():
    start = time.time()
    with open('/home/nxy/codes/coil_spline_HTS/initfiles/init_args.json', 'r') as f:    # 传入地址
        args = json.load(f)
    globals().update(args)
    
    # 获取初始数据
    args, coil_arg_init, fr_init, surface_data, I_init = read_init.init(args)

    # 参数迭代算法        
    opt_init_coil_arg, opt_update_coil_arg, get_params_coil_arg = read_init.args_to_op(
        args, args['optimizer_coil'], args['learning_rate_coil'])
    opt_init_fr, opt_update_fr, get_params_fr = read_init.args_to_op(
        args, args['optimizer_fr'], args['learning_rate_fr'])
    opt_init_I, opt_update_I, get_params_I = read_init.args_to_op(
        args, args['optimizer_I'], args['learning_rate_I'])   


    # loss计算准备

    coil_cal = CoilSet(args)
    coil_output_func = coil_cal.cal_coil    


    loss_vals = []
    # 两种迭代函数
    @jit
    def update(opt_state_coil_arg, opt_state_fr, opt_state_I):
        """
        迭代过程, 通过value_and_grad得到导数值和loss值, 
        """
        params = (get_params_coil_arg(opt_state_coil_arg), get_params_fr(opt_state_fr),
                    get_params_I(opt_state_I))
        loss_val, gradient = value_and_grad(
            lambda params :lossfunction.loss(args, coil_output_func, params, surface_data),
            allow_int = True)(params)      
        g_coil_arg, g_fr, g_I = gradient
        opt_state_coil_arg = opt_update_coil_arg(i, g_coil_arg, opt_state_coil_arg)
        opt_state_fr = opt_update_fr(i, g_fr, opt_state_fr)  
        opt_state_I = opt_update_I(i, g_I, opt_state_I)  
        return opt_state_coil_arg, opt_state_fr, opt_state_I, loss_val

    @jit
    def f_and_df(params):
        params = scipy_to_params(params)                 
        loss_val, gradient = value_and_grad(
            lambda params :lossfunction.loss(args, coil_output_func, params, surface_data),
            allow_int = True)(params)    
        if args['I_optimize'] == 0:
            g_I = gradient[2]
            g_I = g_I.at[:].set(0)
            g = np.append(np.append(gradient[0], gradient[1]), g_I)
        else:
            g = np.append(np.append(gradient[0], gradient[1]), gradient[2])
        loss_vals.append(loss_val)
        print(loss_val)
        return loss_val, g

    def scipy_to_params(params):
        nic = args['number_independent_coils']
        fr = np.reshape(params[-2 * nic * args['number_fourier_rotate']:-nic],
                            (2, nic, args['number_fourier_rotate']))   
        I = params[-nic:]
        if args['coil_case'] == 'fourier':
            coil_arg = np.reshape(params[:6 * nic * args['num_fourier_coils']], 
                                (6, nic, args['num_fourier_coils']) )
            
        elif args['coil_case'] == 'spline':
            coil_arg = np.reshape(params[:nic * 3 * (args['number_control_points']-3)], 
                                (nic, 3, (args['number_control_points']-3)) )  
        
        params = (coil_arg, fr, I)  
        return params


    # 迭代循环, 得到导数, 带入变量, 得到新变量
    # 分两种情况, 固定迭代次数、给定迭代目标残差


    if args['iter_method'] == 'for':
        opt_state_coil_arg = opt_init_coil_arg(coil_arg_init)
        opt_state_fr = opt_init_fr(fr_init)  
        opt_state_I = opt_init_I(I_init)  
        for i in range(args['number_iteration']):
            print('i = ', i)
            opt_state_coil_arg, opt_state_fr, opt_state_I, loss_val = update(
                opt_state_coil_arg, opt_state_fr, opt_state_I)
            loss_vals.append(loss_val)
            print(loss_val)
        params = (get_params_coil_arg(opt_state_coil_arg), get_params_fr(opt_state_fr),
                    get_params_I(opt_state_I))
    
    elif args['iter_method'] == 'min':

        params = np.append(np.append(coil_arg_init, fr_init), I_init)
        res = minimize(f_and_df, params, jac=True, 
                method = '{}'.format(args['minimize_method']), tol = args['minimize_tol'])
        success, loss_val, params, n = res.success, res.fun, res.x, res.nit
        print(success, 'n = ', n, 'loss = ', loss_val)  
        params = scipy_to_params(params)   
   
    elif args['iter_method'] == 'for-min':
        opt_state_coil_arg = opt_init_coil_arg(coil_arg_init)
        opt_state_fr = opt_init_fr(fr_init)  
        opt_state_I = opt_init_I(I_init)  
        for i in range(args['number_iteration']):
            print('i = ', i)
            opt_state_coil_arg, opt_state_fr, loss_val = update(
                opt_state_coil_arg, opt_state_fr)
            loss_vals.append(loss_val)
            print(loss_val)
        coil_arg = get_params_coil_arg(opt_state_coil_arg)
        fr = get_params_fr(opt_state_fr)
        I = get_params_I(opt_state_I)
        params = np.append(np.append(coil_arg, fr), I)
        res = minimize(f_and_df, params, jac=True, 
                method = '{}'.format(args['minimize_method']), tol = args['minimize_tol'])
        success, loss_val, params, n = res.success, res.fun, res.x, res.nit
        print(success, 'n = ', n, 'loss = ', loss_val)   
        params = scipy_to_params(params)   

    elif args['iter_method'] == 'min-for':
        params = np.append(np.append(coil_arg_init, fr_init), I_init)
        res = minimize(f_and_df, params, jac=True, 
                method = '{}'.format(args['minimize_method']), tol = args['minimize_tol'])
        success, loss_val, params, n = res.success, res.fun, res.x, res.nit
        print(success, 'n = ', n, 'loss = ', loss_val)  
        params = scipy_to_params(params)  
        coil_arg, fr, I = params
        opt_state_coil_arg = opt_init_coil_arg(coil_arg)
        opt_state_fr = opt_init_fr(fr) 
        opt_state_I = opt_init_I(I) 
        for i in range(args['number_iteration']):
            print('i = ', i)
            opt_state_coil_arg, opt_state_fr, opt_state_I, loss_val = update(
                opt_state_coil_arg, opt_state_fr, opt_state_I)
            loss_vals.append(loss_val)
            print(loss_val)
        params = (get_params_coil_arg(opt_state_coil_arg), get_params_fr(opt_state_fr),
                    get_params_I(opt_state_I))
    

    end = time.time()
    print('time cost = ', end - start)

    # 得到优化后的线圈参数, 保存到文件中
    coil_all = coil_cal.end_coil(params)
    loss_end = lossfunction.loss_save(args, coil_output_func, params, surface_data)
    save.save_file(args, loss_vals, coil_all, loss_end, surface_data)
    plot.plot(args, coil_all, loss_vals, params)
    

