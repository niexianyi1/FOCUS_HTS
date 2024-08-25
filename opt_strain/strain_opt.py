

import jax.numpy as np
from jax import value_and_grad, jit, config
from scipy.optimize import minimize
import numpy
import json
import time
from strain_coilset import Strain_CoilSet   
import strain_plot
import sys
sys.path.append('opt_coil')
import read_init
from coilset import CoilSet 
import lossfunction  
import save
sys.path.append('HTS')
import hts_strain



config.update("jax_enable_x64", True)
config.update('jax_disable_jit', True)
n = 0


def loss_strain(args, coil_output_func, fr):
    _, dl, _, der1, der2, _, v1, v2, _ = coil_output_func(fr)
    curva = np.cross(der1, der2) / (np.linalg.norm(der1, axis = -1)**3)[:,:,np.newaxis]
    strain = hts_strain.HTS_strain(args, curva, v1, v2, dl)
    strain_mean = np.mean(strain)
    return strain_mean


def main():
    
    with open('initfiles/init_args.json', 'r') as f:    # 传入地址
        args = json.load(f)
    globals().update(args)
    
    # 获取初始数据
    args, coil_arg_init, fr_init, surface_data, I_init = read_init.init(args)
    nic = args['number_independent_coils']
    loss_vals = []
    # 两种迭代函数

    @jit
    def objective_function_minimize(args, fr):
        global n
        n = n + 1 
        fr = np.reshape(fr, ((1, 2, args['number_fourier_rotate'])))  
        coil_cal = Strain_CoilSet(args)
        coil_output_func = coil_cal.cal_coil           
        loss_val, gradient = value_and_grad(
            lambda fr :loss_strain(args, coil_output_func, fr),
            allow_int = True)(fr)    
        g = gradient
        loss_vals.append(loss_val)
        print('iter = ', n, 'value = ', loss_val)
        return loss_val, g

    @jit
    def objective_function_nlopt(fr, grad):#=None
        global n
        n = n + 1 
        fr = np.reshape(fr, ((1, 2, args['number_fourier_rotate'])))    
        coil_cal = Strain_CoilSet(args)
        coil_output_func = coil_cal.cal_coil           
        loss_val, gradient = value_and_grad(
            lambda fr :loss_strain(args, coil_output_func, fr),
            allow_int = True)(fr)    
        if grad.size > 0:
            g = np.reshape(gradient, (2 * args['number_fourier_rotate']))
            grad[:] = list(numpy.array(g))
        loss_val = numpy.float64(loss_val)    
        loss_vals.append(loss_val)
        print('iter = ', n, 'value = ', loss_val)
        return loss_val

    def constrain_nlopt(fr, grad):
        fr = np.reshape(fr, ((1, 2, args['number_fourier_rotate']))) 
        coil_cal = Strain_CoilSet(args)
        coil_output_func = coil_cal.cal_coil           
        value, gradient = value_and_grad(
            lambda fr :hts_strain.cn(args, coil_output_func, fr),
            allow_int = True)(fr)    
        if grad.size > 0:
            g = np.reshape(gradient, (2 * args['number_fourier_rotate']))
            grad[:] = list(numpy.array(g))
        value = numpy.float64(value)    
        print('constrain_value = ', value)
        return value


    # 迭代循环, 得到导数, 带入变量, 得到新变量
    # 分两种情况, 固定迭代次数、给定迭代目标残差
    fr_total = np.zeros((nic, 2, args['number_fourier_rotate']))
    start = time.time()
    
    if args['iter_method'] == 'min':
        for i in range(nic):
            print('start {}th coil strain_opt'.format(i+1))
            fr = np.reshape(fr_init[i], (2*args['number_fourier_rotate']))
            args['coil_arg_i'] = coil_arg_init[i]
            res = minimize(lambda fr :objective_function_minimize(args, fr), fr, jac=True, 
                    method = args['minimize_method'], tol = args['minimize_tol'])
            success, loss_val, fr, n = res.success, res.fun, res.x, res.nit
            print(success, 'strain_mean = ', loss_val, 'nit = ', n)
            fr = np.reshape(fr, ((1, 2, args['number_fourier_rotate'])))  
            fr_total = fr_total.at[i].set(fr)  

   
    elif args['iter_method'] == 'nlopt':
        for i in range(nic):
            print('start {}th coil strain_opt'.format(i+1))
            fr = np.reshape(fr_init[i], (2*args['number_fourier_rotate']))
            args['coil_arg_i'] = coil_arg_init[i]
            opt = read_init.nlopt_op(args, fr)
            opt.set_min_objective(objective_function_nlopt)
            opt.add_inequality_constraint(constrain_nlopt, 1e-4)
            opt.set_ftol_rel(args['stop_criteria'])
            xopt = opt.optimize(fr)
            print('strain_mean = ', opt.last_optimum_value())  
            fr = np.reshape(xopt, ((2, args['number_fourier_rotate'])))    
            fr_total = fr_total.at[i].set(fr)

    end = time.time()
    print('time cost = ', end - start)
    

    params = (coil_arg_init, fr_total, I_init[:-1])
    # 得到优化后的线圈参数, 保存到文件中
    coil_cal = CoilSet(args)
    coil_output_func = coil_cal.cal_coil 
    coil_all = coil_cal.end_coil(params)
    loss_end = lossfunction.loss_save(args, coil_output_func, params, surface_data)
    save.save_file(args, loss_vals, coil_all, loss_end, surface_data) 

    ### 画对比图
    strain_plot.plot_strain_compare(args['out_hdf5'], args['init_coil_file'])
    

