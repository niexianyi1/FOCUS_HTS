import json
import plotly.graph_objects as go
import bspline
import jax.numpy as np
import plot
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


## finit difference
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

def symmetry(r):
    rc = np.zeros((nc, ns+1, 3))
    rc = rc.at[:10, :, :].add(r)
    for i in range(nfp - 1):        
        theta = 2 * pi * (i + 1) / nfp
        T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        rc = rc.at[10*(i+1):10*(i+2), :, :].add(np.dot(r, T))
    return rc

def average_length(self):      #new
    al = np.zeros_like(self.r_coil)
    al = al.at[:, :-1, :].set(self.r_coil[:, 1:, :] - self.r_coil[:, :-1, :])
    al = al.at[:, -1, :].set(self.r_coil[:, 0, :] - self.r_coil[:, -1, :])
    return np.sum(np.linalg.norm(al, axis=-1)) / (self.nc)



c = np.load("/home/nxy/codes/focusadd-spline/results/circle/c_a0.npy")
nn = np.load('/home/nxy/codes/focusadd-spline/initfiles/w7x/highres_nn_surf.npy')
sg = np.load('/home/nxy/codes/focusadd-spline/initfiles/w7x/highres_sg_surf.npy')
r_surf = np.load('/home/nxy/codes/focusadd-spline/initfiles/w7x/highres_r_surf.npy')
dB = np.zeros((10, 3, ns+1))
bc = bspline.get_bc_init(ns+1)
r_coil = vmap(lambda c : bspline.splev(bc, c), in_axes=0, out_axes=0)(c)
der1, wrk1 = vmap(lambda c :bspline.der1_splev(bc, c), in_axes=0, out_axes=0)(c)
der1 = symmetry(der1)
dl = der1[:, :, np.newaxis, np.newaxis, :] * (1 / ns)
r_coil = symmetry(r_coil)[:, :, np.newaxis, np.newaxis, :]
Bn = quadratic_flux(nn, sg, r_coil, r_surf, dl)

def db(i, j, k, d):
    c_new = c.at[i,j,k].add(d)
    r_coil1 = vmap(lambda c_new : bspline.splev(bc, c_new ), in_axes=0, out_axes=0)(c_new )
    der11, wrk1 = vmap(lambda c_new :bspline.der1_splev(bc, c_new), in_axes=0, out_axes=0)(c_new)
    der11 = symmetry(der11)
    dl1 = der11[:, :, np.newaxis, np.newaxis, :] * (1 / ns)
    r_coil1 = symmetry(r_coil1)[:, :, np.newaxis, np.newaxis, :]
    Bn1 = quadratic_flux(nn, sg, r_coil1, r_surf, dl1)
    return (Bn1-Bn)/d
for a in range(2):
    d = 5*10**(-(a+5))
    for i in range(10):
        for j in range(3):
            print('i=', i, 'j=', j)
            k = np.arange(0, 65, 1)   
            dbb = vmap(lambda k: db(i, j, k, d), in_axes=0, out_axes=0)(k)
            dB = dB.at[i,j,:].set(dbb)
    np.save("/home/nxy/codes/focusadd-spline/dB_{}5.npy".format(a+5), dB)

# print(dB)
# gc = np.load("/home/nxy/codes/focusadd-spline/gc.npy")
# db3 = np.load("/home/nxy/codes/focusadd-spline/dB_3.npy")
# db4 = np.load("/home/nxy/codes/focusadd-spline/dB_4.npy")
# db5 = np.load("/home/nxy/codes/focusadd-spline/dB_5.npy")
# print('gc = ', gc)
# print('db = ', db)
# print('db-gc = ', db-gc)
# print('(db-gc)/db = ', (db-gc)/db)
# print('mean = ', np.mean((db3-gc)/db3), np.mean((db4-gc)/db4), np.mean((db5-gc)/db5))






