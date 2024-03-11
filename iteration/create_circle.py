

import plotly.graph_objects as go
import jax.numpy as np
import fourier
pi = np.pi


def circle_coil(args, surface):
    nfc = args['num_fourier_coils']
    nz = args['number_zeta']
    r = args['circle_coil_radius']
    nic = args['number_independent_coils']
    ns = args['number_segments']
    nc = args['number_coils']
    nzs = int(nz/2/args['number_field_periods'])
    # axis = np.zeros((nz + 1, 3))
    # axis = axis.at[:-1, :].set(np.mean(surface, axis = 1))
    # axis = axis.at[-1].set(axis[0])
    # axis = axis[np.newaxis, :, :]
    # fa = fourier.compute_coil_fourierSeries(1, nz, nfc, axis)
    theta = np.linspace(0, 2 * pi, nc + 1) + pi/(nc+1)
    
    # axis_center = fourier.compute_r_centroid(fa, nfc, 1, nc, theta)
    # axis_center = np.squeeze(axis_center)[:-1]
    axis = np.mean(surface, axis = 1)[:nzs]
    
    
    circlecoil = np.zeros((nic, ns+1, 3))
    zeta = theta[:ns]
    theta = np.linspace(0, 2*pi, ns+1)
    for i in range(nic):
        axis_center = axis[i*nzs+int(nzs/2)]
        R = (axis_center[0]**2 + axis_center[1]**2)**0.5
        x = (R + r * np.cos(theta)) * np.cos(zeta[i])
        y = (R + r * np.cos(theta)) * np.sin(zeta[i])
        z = r * np.sin(theta) + axis_center[2]
        circlecoil = circlecoil.at[i].set(np.transpose(np.array([x, y, z])))

    return circlecoil


