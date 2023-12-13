

import json
import plotly.graph_objects as go
import jax.numpy as np
import numpy
pi = np.pi
from jax import vmap
from jax.config import config
import scipy.interpolate as si 
import sys
import time
config.update("jax_enable_x64", True)


with open('/home/nxy/codes/focusadd-spline/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)

ns = 128
nc = 16

MZ = 2
MT = 2

def read_makegrid(filename):      
    r = np.zeros((nc, ns+1, 3))
    with open(filename) as f:
        _ = f.readline()
        _ = f.readline()
        _ = f.readline()
        for i in range(nc):
            for s in range(ns):
                x = f.readline().split()
                r = r.at[i, s, 0].set(float(x[0]))
                r = r.at[i, s, 1].set(float(x[1]))
                r = r.at[i, s, 2].set(float(x[2]))
            _ = f.readline()
    r = r.at[:, -1, :].set(r[:, 0, :])
    return r


def read_w7x(filename):
    R = numpy.zeros((2, 2))
    Z = numpy.zeros((2, 2))
    with open(filename) as f:
        _ = f.readline()
        _, Nfp, _ = f.readline().split()
        Nfp = int(Nfp)
        _ = f.readline()
        _ = f.readline()
        for i in range(4):
            n, m, rc, _, _, zs = f.readline().split()
            n = int(n)
            m = int(m)
            rc = float(rc)
            zs = float(zs)
            R[n, m] = rc
            Z[n, m] = zs
    return R, Z, Nfp

def get_xyz(R, Z, zeta, theta, Nfp):
    r = np.zeros((zeta.shape[0], theta.shape[0]))
    z = np.zeros((zeta.shape[0], theta.shape[0]))
    for mz in range(MZ):
        for mt in range(MT):
            r += R[mz , mt] * np.cos( mt * theta[np.newaxis, :] - mz * Nfp * zeta[:, np.newaxis] )
            z += Z[mz , mt] * np.sin( mt * theta[np.newaxis, :] - mz * Nfp * zeta[:, np.newaxis] )
    x = r * np.cos(zeta)[:, np.newaxis]
    y = r * np.sin(zeta)[:, np.newaxis]
    return np.concatenate((x[:, :, np.newaxis], y[:, :, np.newaxis], z[:, :, np.newaxis]), axis = -1)

def compute_drdz(R, Z, zeta, theta, Nfp):
    x = np.zeros((zeta.shape[0], theta.shape[0]))
    y = np.zeros((zeta.shape[0], theta.shape[0]))
    z = np.zeros((zeta.shape[0], theta.shape[0]))
    for mz in range( MZ ):
        for mt in range(MT):
            coeff = R[mz , mt]
            z_coeff = Z[mz , mt]
            arg = mt * theta[np.newaxis, :] - mz * Nfp * zeta[:, np.newaxis]
            x += coeff * np.cos(arg) * (-np.sin(zeta[:, np.newaxis])) + coeff * (mz * Nfp) * np.cos(zeta[:, np.newaxis]) * np.sin(arg)
            y += coeff * np.cos(arg) * np.cos(zeta[:, np.newaxis]) + coeff * (mz * Nfp) * np.sin(arg) * np.sin(zeta[:, np.newaxis])
            z += z_coeff * np.cos(arg) * -(mz * Nfp)
    return np.concatenate((x[:, :, np.newaxis], y[:, :, np.newaxis], z[:, :, np.newaxis]), axis = -1)

def compute_drdt(R, Z, zeta, theta, Nfp):
    x = np.zeros((zeta.shape[0], theta.shape[0]))
    y = np.zeros((zeta.shape[0], theta.shape[0]))
    z = np.zeros((zeta.shape[0], theta.shape[0]))
    for mz in range(MZ):
        for mt in range(MT):
            coeff = R[mz , mt]
            z_coeff = Z[mz , mt]
            arg = mt * theta[np.newaxis, :] - mz * Nfp * zeta[:, np.newaxis]
            x += coeff * np.sin(arg) * -mt * np.cos(zeta[:, np.newaxis])
            y += coeff * np.sin(arg) * -mt * np.sin(zeta[:, np.newaxis])
            z += z_coeff * np.cos(arg) * mt
    return np.concatenate((x[:, :, np.newaxis], y[:, :, np.newaxis], z[:, :, np.newaxis]), axis = -1)

def get_w7x_data(R, Z, NZ, NT, Nfp):
    zeta = np.linspace(0,2 * np.pi, NZ + 1)[:-1]
    theta = np.linspace(0, 2 * np.pi, NT + 1)[:-1]
    r = get_xyz(R, Z, zeta, theta, Nfp)
    drdz = compute_drdz(R, Z, zeta, theta, Nfp)
    drdt = compute_drdt(R, Z, zeta, theta, Nfp)
    N = np.cross(drdz, drdt)
    sg = np.linalg.norm(N, axis=-1)
    nn = N / sg[:,:,np.newaxis]
    return r, nn, sg


def quadratic_flux(I, dl, r_surf, r_coil, nn, sg):
    def biotSavart(I ,dl, r_surf, r_coil):
        mu_0 = 1e-7
        mu_0I = I * mu_0
        mu_0Idl = (mu_0I[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] * dl)  # NC x NS x NNR x NBR x 3
        r_minus_l = (r_surf[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis, :]
            - r_coil[:, np.newaxis, np.newaxis, :, :, :, :])  # NC x NZ/nfp x NT x NS x NNR x NBR x 3
        top = np.cross(mu_0Idl[:, np.newaxis, np.newaxis, :, :, :, :], r_minus_l)  # NC x NZ x NT x NS x NNR x NBR x 3
        bottom = (np.linalg.norm(r_minus_l, axis=-1) ** 3)  # NC x NZ x NT x NS x NNR x NBR
        B = np.sum(top / bottom[:, :, :, :, :, :, np.newaxis], axis=(0, 3, 4, 5))  # NZ x NT x 3
        return B
    B = biotSavart(I ,dl, r_surf, r_coil)  
      
    print('Bmax = ', np.max(np.linalg.norm(B, axis=-1)))
    Bn = np.sum(B * nn, axis = -1)
    Bn_mean = np.mean(abs(Bn))
    Bnb = np.mean(abs(Bn)/ np.linalg.norm(B, axis=-1))
    Bn_max = np.max(abs(Bn))
    Bns = np.mean(abs(Bn)/ np.linalg.norm(B, axis=-1)*sg)
    return Bn_mean, Bn_max, Bnb, Bns    


def average_length(r_coil):      #new
    al = np.zeros_like(r_coil)
    al = al.at[:, :-1, :].set(r_coil[:, 1:, :] - r_coil[:, :-1, :])
    al = al.at[:, -1, :].set(r_coil[:, 0, :] - r_coil[:, -1, :])
    return np.sum(np.linalg.norm(al, axis=-1)) / (nc)

def curvature(der1, der2):
    bottom = np.linalg.norm(der1, axis = -1)**3
    top = np.linalg.norm(np.cross(der1, der2), axis = -1)
    k = top / bottom
    k_mean = np.mean(k)
    k_max = np.max(k)
    return k_mean, k_max



# rc = read_makegrid('/home/nxy/codes/focusadd-spline/ellipse.coils')
# c, bc = bspline.prep(rc, nc, ns+1, 3)
# der1, wrk1 = vmap(lambda c :bspline.der1_splev(bc, c), in_axes=0, out_axes=0)(c)
# der2 = vmap(lambda wrk1 :bspline.der2_splev(bc, wrk1), in_axes=0, out_axes=0)(wrk1)[:, :-1, :]
# dl = der1[:, :-1, np.newaxis, np.newaxis, :]/ns
# rc = rc[:, :-1, np.newaxis, np.newaxis, :]
# I = np.ones(nc) * 1e6

# al = average_length(rc[:, :-1, :])
# kmean, kmax = curvature(der1[:, :-1, :], der2)
# print(al, kmean, kmax)

R, Z, Nfp = read_w7x("/home/nxy/codes/focusadd-spline/plasma.boundary")
# print(R, Z)
r, nn, sg = get_w7x_data(R, Z, 64*4, 64, Nfp)
np.save('/home/nxy/codes/focusadd-spline/initfiles/ellipse/r_surf.npy', r[:64])
np.save('/home/nxy/codes/focusadd-spline/initfiles/ellipse/nn_surf.npy', nn[:64])
np.save('/home/nxy/codes/focusadd-spline/initfiles/ellipse/sg_surf.npy', sg[:64])
print('finish')
# rc = np.reshape(r, (64*64, 3))
# fig = go.Figure()
# fig.add_scatter3d(x=rc[:, 0],y=rc[:, 1],z=rc[:, 2], name='rc', mode='markers', marker_size = 2)
# fig.update_layout(scene_aspectmode='data')
# fig.show()    



