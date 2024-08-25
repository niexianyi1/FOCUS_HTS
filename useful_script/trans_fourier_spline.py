import jax.numpy as np
import sys
sys.path.append('iteration')
import fourier
import spline
import read_file


## set number
nc = 5      # number of coil
ns = 500    # number of segmnet
nfc = 6     # number of fourier modes
ncp = 24    # number of spline control points

## read file
def read_fc_or_cp(filename):
    file = filename.split('.')
    if file[-1] == 'npy':
        arg = np.load(filename)
    elif file[-1] == 'h5':
        args = read_file.read_hdf5(filename)
        arg = args['coil_arg']
    return arg


def fourier_to_spline(fc):
    if nc == 1:
        fc = fc[:, np.newaxis, :]
    coil = fourier.compute_r_centroid(fc, ns)
    c, bc, tj = spline.get_c_init(coil, nc, ns, ncp)

    return c, bc, tj


def spline_to_fourier(c):
    if nc == 1:
        c = c[np.newaxis, :, :]
    bc, tj = spline.get_bc_init(ns, ncp)
    t, u, k = bc
    coil = spline.splev(t, u, c, tj, ns)
    fc = fourier.compute_coil_fourierSeries(coil, nfc)
    return fc


filename = ''
arg = read_fc_or_cp(filename)

# c, bc, tj = fourier_to_spline(arg)
## or
# fc = spline_to_fourier(arg)






