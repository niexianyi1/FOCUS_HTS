import jax.numpy as np
from jax import value_and_grad, jit
import jax.example_libraries.optimizers as op
from jax.config import config
import json
import time
from coilset import CoilSet     
from lossfunction import LossFunction    
import fourier
config.update("jax_enable_x64", True)
config.update('jax_disable_jit', True)

if __name__ == "__main__":

    with open('/home/nxy/codes/focusadd-spline/initfiles/init_args.json', 'r') as f:    # 传入地址
        args = json.load(f)
    globals().update(args)
    
    assert args['nic'] == int(args['nc']/args['nfp']/(args['ss']+1))

    if args['init_option'] == 'init_coil':
        fc_init = fourier.get_fourier_init(args['init_coil'], args['nic'], args['ns'], args['nfc'])
    
    I = np.ones(args['nc'])*1e6
    args['I'] = I
    fr_init = np.zeros((2, args['nic'], args['nfr'])) 

    def args_to_op(optimizer_string, lr, mom=0.9, var=0.999, eps=1e-8):
        return {
            "gd": lambda lr, *unused: op.sgd(lr),
            "sgd": lambda lr, *unused: op.sgd(lr),
            "momentum": lambda lr, mom, *unused: op.momentum(lr, mom),
            "adam": lambda lr, mom, var, eps: op.adam(lr, mom, var, eps),
        }["{}".format(optimizer_string)](lr, mom, var, eps)

    def get_surface_data(args):           # surface data的获取

        # if args.axis.lower() == "w7x":
        r = np.load(args['surface_r'])
        nn = np.load(args['surface_nn'])
        sg = np.load(args['surface_sg'])
        # r = r[0:int(args['nz']/args['nfp']), :, :]
        surface_data = (r, nn, sg)
        return surface_data

    @jit
    def update(i, opt_state_fc, opt_state_fr, lossfunc):
        fc = get_params_fc(opt_state_fc)  
        fr = get_params_fr(opt_state_fr)
        params = fc, fr
        loss_val, gradient = value_and_grad(
            lambda params :lossfunc.loss(coil_output_func, params),
            allow_int = True)(params)      
        return gradient, loss_val


    surface_data = get_surface_data(args)
    B_extern = None

    coilset = CoilSet(args)
    coil_output_func = coilset.coilset
    opt_init_fc, opt_update_fc, get_params_fc = args_to_op(
        args['opt'], args['lr'], args['mom'],
    )
    opt_init_fr, opt_update_fr, get_params_fr = args_to_op(
        args['opt'], args['lrfr'], args['mom'],
    )
    opt_state_fc = opt_init_fc(fc_init)
    opt_state_fr = opt_init_fr(fr_init)


    
    start = time.time()

    for j in range(5):
        print('j = ', j)
        args['nfr'] = 2+j      

        args['out_loss'] = '/home/nxy/codes/focusadd-spline/results_f/circle/loss/loss_nfr{}.npy'.format(j+2)
        args['out_fc'] = '/home/nxy/codes/focusadd-spline/results_f/circle/loss/fc_nfr{}.npy'.format(j+2)
        args['out_fr'] = '/home/nxy/codes/focusadd-spline/results_f/circle/loss/fr_nfr{}.npy'.format(j+2)
        loss_vals = []
        lossfunc = LossFunction(args, surface_data, B_extern)
        gradient, loss_val = update(0, opt_state_fc, opt_state_fr, lossfunc)
        print('i = 0')

        for i in range(args['n']-1):
            print('i = ', i+1)
            g_fc, g_fr = gradient
            opt_state_fc = opt_update_fc(i+1, g_fc, opt_state_fc)
            opt_state_fr = opt_update_fr(i+1, g_fr, opt_state_fr)     
            gradient, loss_val = update(i+1, opt_state_fc, opt_state_fr, lossfunc)
            loss_vals.append(loss_val)

        end = time.time()
        print(end - start)
        params = (get_params_fc(opt_state_fc), get_params_fr(opt_state_fr))    

        np.save(args['out_fc'], params[0])        
        np.save(args['out_fr'], params[1])
        np.save(args['out_loss'], loss_vals)

