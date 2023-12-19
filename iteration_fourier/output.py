
import tables as tb
import numpy
from coilset import CoilSet




def write_hdf5(params, args):     # 根据需求写入数据
    """ Write coils in HDF5 output format.
    Input:

    output_file (string): Path to outputfile, string should include .hdf5 format

    """

    fc, fr = params
    with tb.open_file(args['out_hdf5'], "w") as f:
        arguments = numpy.dtype(
            [
                ("nc", int),
                ("ns", int),
                ("ln", float),
                ("lb", float),
                ("nnr", int),
                ("nbr", int),
                ("rc", float),
                ("nr", int),
                ("nfr", int),
            ]
        )
        arr = numpy.array(
            [(self.nc, self.ns, self.ln, self.lb, self.nnr, self.nbr, self.rc, self.nr, self.nfr)], dtype=coildata,
        )
        f.create_table("/", "coildata", coildata)
        f.root.coildata.append(arr)
        f.create_array("/", "coilspline", numpy.asarray(fc))
        f.create_array("/", "coilrotation", numpy.asarray(fr))

    return


def write_makegrid(params, args):    # 或者直接输入r, I
    coils = CoilSet(args)
    I, _, r, _, _, _ = coils.coilset(params)
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
