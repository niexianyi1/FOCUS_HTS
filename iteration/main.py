
### 运行脚本, python执行即可

import jax.numpy as np
from jax import value_and_grad, jit
import jax.example_libraries.optimizers as op
from jax.config import config
import json
import time
import read_init
from coilset import CoilSet    
import lossfunction   
import plot
import save


config.update("jax_enable_x64", True)
config.update('jax_disable_jit', True)

if __name__ == "__main__":

    with open('/home/nxy/codes/focusadd-spline/initfiles/init_args.json', 'r') as f:    # 传入地址
        args = json.load(f)
    globals().update(args)
    
    # 获取初始数据
    args, coil_arg_init, fr_init, surface_data = read_init.init(args)

    # 参数迭代算法
    opt_init_coil_arg, opt_update_coil_arg, get_params_coil_arg = read_init.args_to_op(
        args, args['optimizer_coil'], args['learning_rate_coil'])
    opt_init_fr, opt_update_fr, get_params_fr = read_init.args_to_op(
        args, args['optimizer_fr'], args['learning_rate_fr'])

    opt_state_coil_arg = opt_init_coil_arg(coil_arg_init)
    opt_state_fr = opt_init_fr(fr_init)

    # loss计算准备
    coil_cal = CoilSet(args)
    coil_output_func = coil_cal.cal_coil
    loss_vals = []

    @jit
    def update(opt_state_coil_arg, opt_state_fr, lossfunc):
        """
        迭代过程, 通过value_and_grad得到导数值和loss值, 
        
        """
        params = (get_params_coil_arg(opt_state_coil_arg), get_params_fr(opt_state_fr))
        loss_val, gradient = value_and_grad(
            lambda params :lossfunction.loss(args, coil_output_func, params, surface_data),
            allow_int = True)(params)      
        return gradient, loss_val

    start = time.time()

    # 迭代循环, 得到导数, 带入变量, 得到新变量
    # 分两种情况, 固定迭代次数、给定迭代目标数值
   
    gradient, loss_val = update(opt_state_coil_arg, opt_state_fr, lossfunc)
    print('i = 0' , '\n', loss_val)  

    if args['number_iteration'] != 0:
        for i in range(args['number_iteration']):
            print('i = ', i+1)
            g_coil_arg, g_fr = gradient
            opt_state_coil_arg = opt_update_coil_arg(i, g_coil_arg, opt_state_coil_arg)
            opt_state_fr = opt_update_fr(i, g_fr, opt_state_fr)     
            gradient, loss_val = update(opt_state_coil_arg, opt_state_fr, lossfunc)
            loss_vals.append(loss_val)
            print(loss_val)

    elif args['objective_value'] != 0:
        i=0
        while loss_val > args['objective_value']:
            print('i = ', i)
            i = i+1
            g_coil_arg, g_fr = gradient
            opt_state_coil_arg = opt_update_coil_arg(i, g_coil_arg, opt_state_coil_arg)
            opt_state_fr = opt_update_fr(i, g_fr, opt_state_fr)     
            gradient, loss_val = update(opt_state_coil_arg, opt_state_fr, lossfunc)
            loss_vals.append(loss_val)
            print(loss_val)
    end = time.time()
    print('time cost = ', end - start)

    # 得到优化后的线圈参数, 保存到文件中
    params = (get_params_coil_arg(opt_state_coil_arg), get_params_fr(opt_state_fr))
    coil_all = coil_cal.end_coil(params)

    save.save_file(args, params, loss_vals, coil_all, surface_data)

    plot.plot(args, coil_all['r'], loss_vals, params, coil_all['I_new'])







