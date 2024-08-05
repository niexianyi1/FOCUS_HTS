

import numpy
import h5py
import jax.numpy as np
import plotly.graph_objects as go
import json
import sys
sys.path.append('iteration')
import fourier
with open('initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)


def read_plasma(file, nz, nt):
    def read_plasma_boundary(filename):
        with open(filename) as f:
            _ = f.readline()
            bmn, Nfp, _ = f.readline().split()
            Nfp = int(Nfp)
            bmn = int(bmn)
            N = numpy.zeros((bmn))
            _ = f.readline()
            _ = f.readline()
            for i in range(bmn):
                m, n, rc, _, _, zs = f.readline().split()
                N[i] = n
            NMax = numpy.max(N)
        MZ = int(NMax)
        MT = int((bmn - MZ-1) / (2*MZ+1) + 1)

        R = numpy.zeros((2*MZ+1, MT))
        Z = numpy.zeros((2*MZ+1, MT))

        with open(filename) as f:
            _ = f.readline()
            _ = f.readline()
            _ = f.readline()
            _ = f.readline()
            for i in range(bmn):
                m, n, rc, _, _, zs = f.readline().split()
                n = int(n)
                m = int(m)
                R[n+MZ, m] = float(rc)
                Z[n+MZ, m] = float(zs)
        return R, Z, Nfp, MT, MZ

    def get_xyz(R, Z, zeta, theta, Nfp, MT, MZ):
        r = np.zeros((zeta.shape[0], theta.shape[0]))
        z = np.zeros((zeta.shape[0], theta.shape[0]))
        for mz in range(-MZ, MZ + 1):
            for mt in range(MT):
                r += R[mz + MZ, mt] * np.cos( mt * theta[np.newaxis, :] - mz * Nfp * zeta[:, np.newaxis] )
                z += Z[mz + MZ, mt] * np.sin( mt * theta[np.newaxis, :] - mz * Nfp * zeta[:, np.newaxis] )
        x = r * np.cos(zeta)[:, np.newaxis]
        y = r * np.sin(zeta)[:, np.newaxis]
        return np.concatenate((x[:, :, np.newaxis], y[:, :, np.newaxis], z[:, :, np.newaxis]), axis = -1)

    def compute_drdz(R, Z, zeta, theta, Nfp, MT, MZ):
        x = np.zeros((zeta.shape[0], theta.shape[0]))
        y = np.zeros((zeta.shape[0], theta.shape[0]))
        z = np.zeros((zeta.shape[0], theta.shape[0]))
        for mz in range(-MZ, MZ + 1):
            for mt in range(MT):
                coeff = R[mz + MZ, mt]
                z_coeff = Z[mz + MZ, mt]
                arg = mt * theta[np.newaxis, :] - mz * Nfp * zeta[:, np.newaxis]
                x += coeff * np.cos(arg) * (-np.sin(zeta[:, np.newaxis])) + coeff * (mz * Nfp) * np.cos(zeta[:, np.newaxis]) * np.sin(arg)
                y += coeff * np.cos(arg) * np.cos(zeta[:, np.newaxis]) + coeff * (mz * Nfp) * np.sin(arg) * np.sin(zeta[:, np.newaxis])
                z += z_coeff * np.cos(arg) * -(mz * Nfp)
        return np.concatenate((x[:, :, np.newaxis], y[:, :, np.newaxis], z[:, :, np.newaxis]), axis = -1)

    def compute_drdt(R, Z, zeta, theta, Nfp, MT, MZ):
        x = np.zeros((zeta.shape[0], theta.shape[0]))
        y = np.zeros((zeta.shape[0], theta.shape[0]))
        z = np.zeros((zeta.shape[0], theta.shape[0]))
        for mz in range(-MZ, MZ + 1):
            for mt in range(MT):
                coeff = R[mz + MZ, mt]
                z_coeff = Z[mz + MZ, mt]
                arg = mt * theta[np.newaxis, :] - mz * Nfp * zeta[:, np.newaxis]
                x += coeff * np.sin(arg) * -mt * np.cos(zeta[:, np.newaxis])
                y += coeff * np.sin(arg) * -mt * np.sin(zeta[:, np.newaxis])
                z += z_coeff * np.cos(arg) * mt
        return np.concatenate((x[:, :, np.newaxis], y[:, :, np.newaxis], z[:, :, np.newaxis]), axis = -1)

    def get_plasma_boundary(R, Z, NZ, NT, Nfp, MT, MZ):
        zeta = np.linspace(0,2 * np.pi, NZ + 1)[:-1]
        theta = np.linspace(0, 2 * np.pi, NT + 1)[:-1]
        r = get_xyz(R, Z, zeta, theta, Nfp, MT, MZ)
        drdz = compute_drdz(R, Z, zeta, theta, Nfp, MT, MZ)
        drdt = compute_drdt(R, Z, zeta, theta, Nfp, MT, MZ)
        N = np.cross(drdz, drdt)
        sg = np.linalg.norm(N, axis=-1)
        nn = N / sg[:,:,np.newaxis]
        return r, nn, sg

    R, Z, Nfp, MT, MZ = read_plasma_boundary(file)
    r, nn, sg = get_plasma_boundary(R, Z, nz, nt, Nfp, MT, MZ)
    return Nfp, r, nn, sg 



def circle_coil(args, surface):
    nfc = args['number_fourier_coils']
    nz = args['number_zeta']
    r = args['circle_coil_radius']
    nic = args['number_independent_coils']
    ns = args['number_segments']
    nc = args['number_coils']
    nzs = int(nz/2/args['number_field_periods'])
    axis = np.zeros((nz + 1, 3))
    axis = axis.at[:-1].set(np.mean(surface, axis = 1))
    axis = axis.at[-1].set(axis[0])
    axis = axis[np.newaxis, :, :]
    fa = fourier.compute_coil_fourierSeries(1, nz, nfc, axis)
    axis = fourier.compute_r_centroid(fa, nfc, 1, 2*nc)
    axis = np.squeeze(axis)[:-1]

    circlecoil = np.zeros((nic, ns+1, 3))
    zeta = np.linspace(0, 2 * np.pi, nc + 1) + np.pi/(nc+1)
    theta = np.linspace(0, 2 * np.pi, ns + 1)
    for i in range(nic):
        axis_center = axis[2*(i+1)]
        R = (axis_center[0]**2 + axis_center[1]**2)**0.5
        x = (R + r * np.cos(theta)) * np.cos(zeta[i])
        y = (R + r * np.cos(theta)) * np.sin(zeta[i])
        z = r * np.sin(theta) + axis_center[2]
        circlecoil = circlecoil.at[i].set(np.transpose(np.array([x, y, z])))

    return circlecoil, axis

file = 'initfiles/hsx/plasma.boundary'
nz = 128
nt = 64

Nfp, r, nn, sg = read_plasma(file, nz, nt)
print(r.shape, nn.shape, sg.shape)
print(np.sum(sg))
coil, axis = circle_coil(args, r)
coil = np.reshape(coil, (6*65, 3))
fig = go.Figure()
fig.add_scatter3d(x=axis[:, 0],y=axis[:, 1],z=axis[:, 2], name='axis', mode='markers', marker_size = 1.5)   
fig.add_scatter3d(x=coil[:, 0],y=coil[:, 1],z=coil[:, 2], name='coil', mode='markers', marker_size = 1.5)   
# fig.add_trace(go.Surface(x=r[:,:,0], y=r[:,:,1], z=r[:,:,2]))
fig.update_layout(scene_aspectmode='data')
fig.show()
