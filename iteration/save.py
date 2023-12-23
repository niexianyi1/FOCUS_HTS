

## 运行结束，保存数据到文件

import tables as tb
import jax.numpy as np




def save_file(args, params, loss_vals, coil_all, surface_data):
    save_npy(args, params, loss_vals)
    save_hdf5(args, params, coil_all, surface_data)
    save_makegrid(args, coil_all)
    return
    
def save_npy(args, params, loss_vals):
    np.save('{}'.format(args['save_loss']), loss_vals)
    np.save('{}'.format(args['save_coil_arg']), params[0])
    np.save('{}'.format(args['save_fr']), params[1])
    return



def save_hdf5(args, params, coil_all, surface_data):     # 根据需求写入数据
    """ Write coils in HDF5 output format.
    Input:

    output_file (string): Path to outputfile, string should include .hdf5 format

    """

    fc, fr = params
    with tb.open_file(args['out_hdf5'], "w") as f:
        for key in args:
            f.create_dataset("{}".format(key), data=args['key'])
        for key in coil_all:
            f.create_dataset("{}".format(key), data=coil_all['key'])
        f.create_dataset("coil_arg", data=params[0])
        f.create_dataset("fr", data=params[1])
        f.create_dataset("surface_data_r", data=surface_data[0])
        f.create_dataset("surface_data_nn", data=surface_data[1])
        f.create_dataset("surface_data_sg", data=surface_data[2])
    return


def save_makegrid(args, coil_all):    # 或者直接输入r, I
    r = coil_all['r']
    I = coil_all['I_new']
    with open(args['out_coil_makegrid'], "w") as f:
        f.write("periods {}\n".format(0))
        f.write("begin filament\n")
        f.write("FOCUSADD Coils\n")
        for i in range(args['nic']):
            for n in range(args['nnr']):
                for b in range(args['nbr']):
                    for s in range(args['ns']):
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











