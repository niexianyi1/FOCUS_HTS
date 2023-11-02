import jax.numpy as np
from jax import value_and_grad, jit, vmap
import jax.example_libraries.optimizers as op
from jax.config import config
import json
import time
from surface.readAxis import read_axis
from surface.Surface import Surface
from surface.Axis import Axis
from coilset import CoilSet     
from lossfunction import LossFunction    
import bspline
import plot
pi = np.pi
config.update("jax_enable_x64", True)
config.update('jax_disable_jit', True)

if __name__ == "__main__":

    with open('/home/nxy/codes/focusadd-spline/initfiles/init_args.json', 'r') as f:    # 传入地址
        args = json.load(f)
    globals().update(args)
    args['ncnfp'] = ncnfp = int(args['nc']/args['nfp'])

    if args['init_option'] == 'init_coil':
        c_init, bc = bspline.get_c_init(args['init_coil'], ncnfp, args['ns'])
    elif args['init_option'] == 'init_c':
        c_init = np.load(args['init_c'])
        if c_init.shape[0] == ncnfp:
            pass
        elif c_init.shape[0] == args['nc']:   ## 默认取前一个周期的线圈数
            c_init = c_init[ncnfp, :, :]
        bc = bspline.get_bc_init(c_init.shape[2])
    else:
        raise ValueError("init_option must be 'init_coil' or 'init_c'")

    args['bc'] = bc
    I = np.ones(args['nc'])*1e6
    args['I'] = I
    fr_init = np.zeros((2, args['nc'], args['nfr'])) 

    def args_to_op(optimizer_string, lr, mom=0.9, var=0.999, eps=1e-7):
        return {
            "gd": lambda lr, *unused: op.sgd(lr),
            "sgd": lambda lr, *unused: op.sgd(lr),
            "momentum": lambda lr, mom, *unused: op.momentum(lr, mom),
            "adam": lambda lr, mom, var, eps: op.adam(lr, mom, var, eps),
        }["{}".format(optimizer_string)](lr, mom, var, eps)

    r_surf = np.load(args['surface_r'])
    nn = np.load(args['surface_nn'])
    sg = np.load(args['surface_sg'])
    surface_data = (r_surf, nn, sg)

    g_fr = np.zeros((2, 50, 0))


    @jit
    def update(i, opt_state_c, opt_state_fr):
        c = get_params_c(opt_state_c)    
        c = c.at[:, :, -3:].set(c[:, :, :3])
        opt_state_c[0][0][0] = c
        dB = np.zeros((10, 3, ns+1))
        bc = bspline.get_bc_init(ns+1)
        r_coil = vmap(lambda c : bspline.splev(bc, c), in_axes=0, out_axes=0)(c)
        der1, wrk1 = vmap(lambda c :bspline.der1_splev(bc, c), in_axes=0, out_axes=0)(c)
        der1 = symmetry(der1)
        dl = der1[:, :, np.newaxis, np.newaxis, :] * (1 / ns)
        r_coil = symmetry(r_coil)[:, :, np.newaxis, np.newaxis, :]
        loss_val = quadratic_flux(nn, sg, r_coil, r_surf, dl)
    
        for a in range(10):
            for b in range(3):
                print(a, b)
                # for k in range(65):
                #     dB = dB.at[a,b,:].set(db(a, b, k, 1e-4, loss_val, c))
                k = np.arange(0, 65, 1)   
                dbb = vmap(lambda k: db(a, b, k, 1e-4, loss_val, c), in_axes=0, out_axes=0)(k)
                dB = dB.at[a,b,:].set(dbb)

        return opt_update_c(i, dB, opt_state_c), opt_update_fr(i, g_fr, opt_state_fr), loss_val

    def symmetry(r):
        rc = np.zeros((nc, ns+1, 3))
        rc = rc.at[:10, :, :].add(r)
        for i in range(nfp - 1):        
            theta = 2 * pi * (i + 1) / nfp
            T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            rc = rc.at[10*(i+1):10*(i+2), :, :].add(np.dot(r, T))
        return rc

    def db(i, j, k, d, Bn, c):
        c_new = c.at[i,j,k].add(d)
        r_coil1 = vmap(lambda c_new : bspline.splev(bc, c_new ), in_axes=0, out_axes=0)(c_new )
        der11, wrk1 = vmap(lambda c_new :bspline.der1_splev(bc, c_new), in_axes=0, out_axes=0)(c_new)
        der11 = symmetry(der11)
        dl1 = der11[:, :, np.newaxis, np.newaxis, :] * (1 / ns)
        r_coil1 = symmetry(r_coil1)[:, :, np.newaxis, np.newaxis, :]
        Bn1 = quadratic_flux(nn, sg, r_coil1, r_surf, dl1) 
        return (Bn1-Bn)/d

    def quadratic_flux(nn, sg, r_coil, r_surf, dl):
        B = biotSavart(r_coil, r_surf, dl)  # NZ x NT x 3
        B_all = B
        return (
            0.5
            * np.sum(np.sum(nn * B_all/np.linalg.norm(B_all, axis=-1)[:, :, np.newaxis], axis=-1) ** 2 * sg)
        )  # NZ x NTf   

    def biotSavart(r_coil, r_surf, dl):
        mu_0 = 1
        I = np.ones(50)
        mu_0I = I * mu_0
        mu_0Idl = (mu_0I[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] * dl)  # NC x NS x NNR x NBR x 3
        r_minus_l = (r_surf[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis, :]
            - r_coil[:, np.newaxis, np.newaxis, :, :, :, :])  # NC x NZ/nfp x NT x NS x NNR x NBR x 3
        top = np.cross(mu_0Idl[:, np.newaxis, np.newaxis, :, :, :, :], r_minus_l)  # NC x NZ x NT x NS x NNR x NBR x 3
        bottom = (np.linalg.norm(r_minus_l, axis=-1) ** 3)  # NC x NZ x NT x NS x NNR x NBR
        B = np.sum(top / bottom[:, :, :, :, :, :, np.newaxis], axis=(0, 3, 4, 5))  # NZ x NT x 3
        return B


    B_extern = None

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

    loss_vals = []
    start = time.time()

    for i in range(args['n']):
        print('i = ', i)        
        opt_state_c, opt_state_fr, loss_val = update(i, opt_state_c, opt_state_fr)
        loss_vals.append(loss_val)
        print(loss_val)
    end = time.time()
    print(end - start)
    params = (get_params_c(opt_state_c), get_params_fr(opt_state_fr))
    c = params[0]    
    fr = params[1] 
    c = c.at[:, :, -3:].set(c[:, :, :3])

    # np.save(args['out_c'], c)          ### c_new有3个重合控制点
    # np.save(args['out_fr'], params[1])
    # np.save(args['out_loss'], loss_vals)

    # coilset.write_hdf5(params)
    # coilset.write_makegrid(params)

    # paint = plot.plot(args)  
    # paint.plot_loss(loss_vals)
    # paint.plot_coils(c)
    # paint.poincare(c)

