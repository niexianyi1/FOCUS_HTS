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

def average_length(r_coil):      #new
    r_coil = r_coil[0]
    al = np.zeros_like(r_coil)
    al = al.at[:-1, :].set(r_coil[1:, :] - r_coil[:-1, :])
    al = al.at[-1, :].set(r_coil[0, :] - r_coil[-1, :])
    len = np.sum(np.linalg.norm(al, axis=-1))
    return len

c = np.load("/home/nxy/codes/focusadd-spline/results/circle/c_a0.npy")

### 磁场项
# nn = np.load('/home/nxy/codes/focusadd-spline/initfiles/w7x/highres_nn_surf.npy')
# sg = np.load('/home/nxy/codes/focusadd-spline/initfiles/w7x/highres_sg_surf.npy')
# r_surf = np.load('/home/nxy/codes/focusadd-spline/initfiles/w7x/highres_r_surf.npy')
# dB = np.zeros((10, 3, ns+1))
# bc = bspline.get_bc_init(ns+1)
# r_coil = vmap(lambda c : bspline.splev(bc, c), in_axes=0, out_axes=0)(c)
# der1, wrk1 = vmap(lambda c :bspline.der1_splev(bc, c), in_axes=0, out_axes=0)(c)
# der1 = symmetry(der1)
# dl = der1[:, :, np.newaxis, np.newaxis, :] * (1 / ns)
# r_coil = symmetry(r_coil)[:, :, np.newaxis, np.newaxis, :]
# Bn = quadratic_flux(nn, sg, r_coil, r_surf, dl)
# d = 1e-4
# def db(i, j, k):
#     c_new = c.at[i,j,k].add(d*c[i,j,k])
#     r_coil1 = vmap(lambda c_new : bspline.splev(bc, c_new ), in_axes=0, out_axes=0)(c_new )
#     der11, wrk1 = vmap(lambda c_new :bspline.der1_splev(bc, c_new), in_axes=0, out_axes=0)(c_new)
#     der11 = symmetry(der11)
#     dl1 = der11[:, :, np.newaxis, np.newaxis, :] * (1 / ns)
#     r_coil1 = symmetry(r_coil1)[:, :, np.newaxis, np.newaxis, :]
#     Bn1 = quadratic_flux(nn, sg, r_coil1, r_surf, dl1) 
#     return (Bn1-Bn)/d


# for i in range(10):
#     for j in range(3):
#         print('i=', i, 'j=', j)
#         for k in range(ns+1):
#             dB = dB.at[i,j,:].set(db(i, j, k))
#         # k = np.arange(0, 65, 1)   
#         # dbb = vmap(lambda k: db(i, j, k, d), in_axes=0, out_axes=0)(k)
#         # dB = dB.at[i,j,:].set(dbb)
# np.save("/home/nxy/codes/focusadd-spline/dB_{}.npy".format(4), dB)
# print(dB)
gc = np.load("/home/nxy/codes/focusadd-spline/results/fd/gbc.npy")
dB = np.load('/home/nxy/codes/focusadd-spline/results/fd/dB_4.npy')
# db3 = np.load("/home/nxy/codes/focusadd-spline/dB_3.npy")
# db4 = np.load("/home/nxy/codes/focusadd-spline/dB_4.npy")
# db5 = np.load("/home/nxy/codes/focusadd-spline/dB_5.npy")
print('gc = ', gc)
print('db = ', dB)
# print('db-gc = ', db-gc)
# print('(db-gc)/db = ', (db-gc)/db)
print('mean = ', np.mean((dB-gc)/gc))


### 长度项
# dl = np.zeros((3, ns+1))
# bc = bspline.get_bc_init(ns+1)
# r_coil = vmap(lambda c : bspline.splev(bc, c), in_axes=0, out_axes=0)(c)
# len = average_length(r_coil)

# def length(i, j, k, d):
#     c_new = c.at[i,j,k].add(d*c[i,j,k])
#     r_coil1 = vmap(lambda c_new : bspline.splev(bc, c_new ), in_axes=0, out_axes=0)(c_new )
#     len1 = average_length(r_coil1)
#     return (len1-len)/d
    
# for a in range(3):
#     d = 10**(-(a+3))
#     i = 0
#     for j in range(3):
#         print('i=', i, 'j=', j)
#         k = np.arange(0, 65, 1)   
#         dll = vmap(lambda k: length(i, j, k, d), in_axes=0, out_axes=0)(k)
#         dl = dl.at[j,:].set(dll)
#     np.save("/home/nxy/codes/focusadd-spline/dl_{}.npy".format(a+5), dl)
# glc = np.load('/home/nxy/codes/focusadd-spline/glc.npy')[0]
# dl3 = np.load('/home/nxy/codes/focusadd-spline/dl_5.npy')
# dl4 = np.load('/home/nxy/codes/focusadd-spline/dl_6.npy')
# dl5 = np.load('/home/nxy/codes/focusadd-spline/dl_7.npy')
# print(glc)
# print(dl4)
# print((dl4-glc)/glc)
# print(np.mean((dl3-glc)/glc), np.mean((dl4-glc)/glc), np.mean((dl5-glc)/glc))

### 从起始开始一步步看，先看第一步splev

# dr = np.zeros((10, 3, ns+1))
# bc = bspline.get_bc_init(ns+1)
# rc = vmap(lambda c : bspline.splev(bc, c), in_axes=0, out_axes=0)(c)
# der1, wrk1 = vmap(lambda c : bspline.der1_splev(bc, c), in_axes=0, out_axes=0)(c)
# # der2, _ = vmap(lambda wrk1 : bspline.der2_splev(bc, wrk1), in_axes=0, out_axes=0)(wrk1)
# def length(i, j, k, d):
#     c_new = c.at[i,j,k].add(d)
#     rc1 = vmap(lambda c_new : bspline.splev(bc, c_new ), in_axes=0, out_axes=0)(c_new )
    
#     return (rc1[i,j,k]-rc[i,j,k])/d

# d = 1e-4
# for i in range(10):
#     for j in range(3):
#         print('i=', i, 'j=', j)
#         k = np.arange(0, 65, 1)   
#         drr = vmap(lambda k: length(i, j, k, d), in_axes=0, out_axes=0)(k)
#         dr = dr.at[i,j,:].set(drr)
# dr = np.transpose(dr, (0, 2, 1))
# np.save("/home/nxy/codes/focusadd-spline/der1_{}.npy".format(4), dr)
# print(dr)
# print(der1)
# print((dr-der1)/der1)
# print(np.mean((dr-der1)/der1))













