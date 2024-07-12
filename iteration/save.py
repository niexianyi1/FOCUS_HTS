

## 运行结束，保存数据到文件
str.encode('utf-8')
import h5py
import jax.numpy as np




def save_file(args, loss_vals, coil_all, loss_end, surface_data):
    if args['save_npy'] != 0:
        save_npy(args, coil_all, loss_vals)
    if args['save_hdf5'] != 0: 
        save_hdf5(args, coil_all, loss_end, surface_data, loss_vals)
    if args['save_makegrid'] != 0:
        save_makegrid(args, coil_all)
    return
    
def save_npy(args, coil_all, loss_vals):
    np.save('{}'.format(args['save_loss']), loss_vals)
    np.save('{}'.format(args['save_coil_arg']), coil_all['coil_arg'])
    np.save('{}'.format(args['save_fr']), coil_all['coil_fr'])
    return



def save_hdf5(args, coil_all, loss_end, surface_data, loss_vals):     # 根据需求写入数据
    """ Write coils in HDF5 output format.
    Input:

    output_file (string): Path to outputfile, string should include .hdf5 format

    """

    with h5py.File(args['out_hdf5'], "w") as f:
        if args['coil_case'] == 'spline' or args['coil_case'] == 'spline_local':
            bc = args['bc']
            t,u,k = bc
            f.create_dataset(name='spline_t', data=t)
            f.create_dataset(name='spline_u', data=u)
            args.pop('bc')
        for key in args:
            f.create_dataset(name=key, data=args['{}'.format(key)])
        for key in coil_all:
            f.create_dataset(name=key, data=coil_all['{}'.format(key)])
        for key in loss_end:
            f.create_dataset(name=key, data=loss_end['{}'.format(key)])
        f.create_dataset(name="surface_data_r", data=surface_data[0])
        f.create_dataset(name="surface_data_nn", data=surface_data[1])
        f.create_dataset(name="surface_data_sg", data=surface_data[2])
        f.create_dataset(name="loss_vals", data=loss_vals)
    return


def save_makegrid(args, coil_all):    # 或者直接输入r, I
    r = coil_all['coil_r']
    I = coil_all['coil_I']
    with open(args['out_coil_makegrid'], "w") as f:
        f.write("periods {}\n".format(0))
        f.write("begin filament\n")
        f.write("FOCUSADD Coils\n")
        for i in range(args['number_independent_coils']):
            for n in range(args['number_normal']):
                for b in range(args['number_binormal']):
                    for s in range(args['number_segments']):
                        f.write(
                            "{} {} {} {}\n".format(
                                r[i, s, n, b, 0],
                                r[i, s, n, b, 1],
                                r[i, s, n, b, 2],
                                I[i],
                            )
                        )
                    f.write(
                        "{} {} {} {} {} {}\n".format(
                            r[i, 0, n, b, 0],
                            r[i, 0, n, b, 1],
                            r[i, 0, n, b, 2],
                            0.0,
                            "{},{},{}".format(i, n, b),
                            "coil/filament1/filament2",
                        )
                    )
    return











