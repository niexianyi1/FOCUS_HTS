import jax.numpy as np
from jax import value_and_grad, jit
import jax.example_libraries.optimizers as op
from jax.config import config
import json
import time
from surface.readAxis import read_axis
from surface.Surface import Surface
from surface.Axis import Axis
from coilset import CoilSet     #####
from lossfunction import LossFunction    #####
import bspline
import plot

config.update("jax_enable_x64", True)
config.update('jax_disable_jit', True)

if __name__ == "__main__":


    with open('/home/nxy/codes/focusadd-spline/initfiles/init_args.json', 'r') as f:    # 传入地址
        args = json.load(f)
    globals().update(args)
    ncnfp = int(args['nc']/args['nfp'])
    if args['init_option'] == 'init_coil':
        c_init, bc = bspline.get_c_init(args['init_coil'], ncnfp, args['ns'])
    elif args['init_option'] == 'init_c':
        c_init = np.load(args['init_c'])
        bc = bspline.get_bc_init(args['ns'])
    else:
        raise ValueError("init_option must be 'init_coil' or 'init_c'")
    args['bc'] = bc
    fr_init = np.zeros((2, args['nc'], args['nfr'])) 
    
    def args_to_op(optimizer_string, lr, mom=0.9, var=0.999, eps=1e-7):
        return {
            "gd": lambda lr, *unused: op.sgd(lr),
            "sgd": lambda lr, *unused: op.sgd(lr),
            "momentum": lambda lr, mom, *unused: op.momentum(lr, mom),
            "adam": lambda lr, mom, var, eps: op.adam(lr, mom, var, eps),
        }["{}".format(optimizer_string)](lr, mom, var, eps)

    def get_surface_data(args):           # surface data的获取
        # if args.axis.lower() == "w7x":
        assert args['nz'] == 150
        assert args['nt'] == 20
        assert args['nc'] == 50
        # need to assert that axis has right number of points
        r = np.load("/home/nxy/codes/focusadd-spline/initfiles/w7x/w7x_highres_r_surf.npy")
        nn = np.load("/home/nxy/codes/focusadd-spline/initfiles/w7x/w7x_highres_nn_surf.npy")
        sg = np.load("/home/nxy/codes/focusadd-spline/initfiles/w7x/w7x_highres_sg_surf.npy")
        # r = r[0:int(args['nz']/args['nfp']), :, :]

        surface_data = (r, nn, sg)
          
        return surface_data




    @jit
    def update(i, opt_state_c, opt_state_fr, lossfunc, I):
        c = get_params_c(opt_state_c)
        fr = get_params_fr(opt_state_fr)
        params = c, fr
        loss_val, gradient = value_and_grad(
            lambda params :lossfunc.loss(coil_output_func, params, I),
            allow_int = True
        )(params)
        g_c, g_fr = gradient
        return opt_update_c(i, g_c, opt_state_c), opt_update_fr(i, g_fr, opt_state_fr), loss_val



    surface_data = get_surface_data(args)
    B_extern = None
    I = np.ones(args['nc'])


    coilset = CoilSet(args)
    coil_output_func = coilset.coilset
    opt_init_c, opt_update_c, get_params_c = args_to_op(
        args['opt'], args['lr'], args['mom'],
    )
    opt_init_fr, opt_update_fr, get_params_fr = args_to_op(
        args['opt'], args['lrfr'], args['mom'],
    )
    opt_state_c = opt_init_c(c_init)
    opt_state_fr = opt_init_fr(fr_init)

    lossfunc = LossFunction(args, surface_data, B_extern, I)

    loss_vals = []
    start = time.time()

    for i in range(args['n']):
        print('i = ', i)
        opt_state_c, opt_state_fr, loss_val = update(i, opt_state_c, opt_state_fr, lossfunc,I)
        loss_vals.append(loss_val)
        print(loss_val)

    end = time.time()
    print(end - start)
    params = (get_params_c(opt_state_c), get_params_fr(opt_state_fr))


    np.save('/home/nxy/codes/focusadd-spline/results/circle/w7x_circle_coil_params_400.npy', params[0])
    np.save('/home/nxy/codes/focusadd-spline/results/circle/w7x_circle_fr_400.npy', params[1])
    np.save('/home/nxy/codes/focusadd-spline/results/circle/w7x_circle_loss_vals_400.npy', loss_vals)


    coilset.write_hdf5(params, args['out_hdf5'])
    coilset.write_makegrid(params, args['out_coil_makegrid'])

    paint = plot.plot(args, I)
    paint.plot_loss(loss_vals)
    paint.plot_coils(params[0])
    paint.poincare(params[0])

