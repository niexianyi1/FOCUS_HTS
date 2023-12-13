

import json
import plotly.graph_objects as go
import bspline
import jax.numpy as np
import numpy
import plot
pi = np.pi
from jax import vmap
from jax.config import config
import scipy.interpolate as si 
import sys
import time
import h5py
config.update("jax_enable_x64", True)


with open('/home/nxy/codes/focusadd-spline/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)

def read_hdf5(filename):
    f = h5py.File(filename, "r")
    args = {}
    for key in list(f.keys()):
        args.update({key: f[key][:]})
    f.close()
    return args

ellipse = read_hdf5('/home/nxy/codes/focusadd-spline/focus_ellipse.h5')


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

def symmetry(r):
    rc = np.zeros((4*2, 128, 3))
    rc = rc.at[:4, :, :].set(r)
    T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rc = rc.at[4:4*2, :, :].add(np.dot(r, T))

    rc_total = np.zeros((4*4, 128, 3))
    rc_total = rc_total.at[0:4*2, :, :].add(rc)
    for i in range(2 - 1):        
        theta = 2 * np.pi * (i + 1) / 2
        T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        rc_total = rc_total.at[4*2*(i+1):4*2*(i+2), :, :].add(np.dot(rc, T))
    return rc_total

# 比较磁场分布
# rc = read_makegrid('/home/nxy/codes/focusadd-spline/ellipse.coils')

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

def quadratic_flux(I, dl, r_surf, r_coil, nn, sg):

    B = biotSavart(I ,dl, r_surf, r_coil)  
      
    print('Bmax = ', np.max(np.linalg.norm(B, axis=-1)))
    Bn = np.sum(B * nn, axis = -1)
    Bn_mean = np.mean(abs(Bn))
    Bnb = np.mean(abs(Bn)/ np.linalg.norm(B, axis=-1))
    Bn_max = np.max(abs(Bn))
    Bns = np.mean(abs(Bn)/ np.linalg.norm(B, axis=-1)*sg)
    return Bn_mean, Bn_max, Bnb, Bns    


def curvature(der1, der2):
    bottom = np.linalg.norm(der1, axis = -1)**3
    top = np.linalg.norm(np.cross(der1, der2), axis = -1)
    k = top / bottom
    k_mean = np.mean(k)
    k_max = np.max(k)
    return k_mean, k_max

rc = np.transpose(np.array([ellipse['xx'], ellipse['yy'], ellipse['zz']]), (2,1,0))
print(rc.shape)
# c, bc = bspline.prep(rc, nic, ns, ns+3, 3)
# tj = bspline.tjev(bc)
# t, u, k = bc
c = np.load('/home/nxy/codes/focusadd-spline/coil_c.npy')
t = np.load('/home/nxy/codes/focusadd-spline/coil_t.npy')
u = np.zeros((nic, ns))
for i in range(nic):
    u = u.at[i,:].set(np.arange(0, 1, 1/ns))
bc = [t, u, 3]
tj = bspline.tjev(bc)
coil = der1 = der2 = np.zeros((nic, ns, 3))
for i in range(nic):
    coil = coil.at[i].set(bspline.splev(t[i], u[i], c[i], tj[i], ns))
    d10, wrk1 = bspline.der1_splev(t[i], u[i], c[i], tj[i], ns)
    der1 = der1.at[i].set(d10)
    der2 = der2.at[i].set(bspline.der2_splev(t[i], u[i], wrk1, tj[i], ns))

# print(rc[0])
# print(coil[0])

d1 = np.transpose(np.array([ellipse['xt'], ellipse['yt'], ellipse['zt']]), (2,1,0))
print(np.mean((rc[:,:-1,:]-coil)/rc[:,:-1,:]))
print(np.mean((d1[:,:-1,:]-der1)/d1[:,:-1,:]))

k_mean, k_max = curvature(der1, der2)
print('k_mean = ', k_mean, 'k_max = ', k_max)
I = np.ones(nc) * 1e6
rs = np.array([ellipse['xsurf'], ellipse['ysurf'], ellipse['zsurf']]).transpose(1,2,0)

coil = symmetry(coil)
der1 = symmetry(der1)
dl = der1[:, :, np.newaxis, np.newaxis, :]/ns
coil = coil[:, :, np.newaxis, np.newaxis, :]
B = biotSavart(I ,dl, rs, coil)

rc = symmetry(rc[:, :-1, :])
d1 = symmetry(d1[:, :-1, :])
dl = d1[:, :, np.newaxis, np.newaxis, :]/ns
rc = rc[:, :, np.newaxis, np.newaxis, :]
Bf = biotSavart(I ,dl, rs, rc)
# Bf = np.array([ellipse['Bx'], ellipse['By'], ellipse['Bz']]).transpose(1,2,0)
print(Bf[0])
print(B[0])
print(np.mean((B-Bf)/Bf))


# nn = np.array([ellipse['nx'], ellipse['ny'], ellipse['nz']]).transpose(1,2,0)
# Bnf = ellipse['Bn']
# sg = ellipse['nn']
# Bn = np.sum(B * nn, axis = -1) # 算法正确
# Bnb = np.mean(abs(Bn)/ np.linalg.norm(B, axis=-1))  # 正确
# Bn_max = np.max(abs(Bn))      # 正确
# Bns = np.sum(abs(Bn)/ np.linalg.norm(B, axis=-1)*sg)/np.sum(sg)    # 正确
# Bnormal = np.mean(Bn**2*sg)    # 正确





'''rc = rc.reshape(nc*(ns+1), 3)
# rs = rs.reshape(nz*nt, 3)
fig = go.Figure()
fig.add_scatter3d(x=rc[:, 0],y=rc[:, 1],z=rc[:, 2], name='rc', mode='markers', marker_size = 2)
for i in range(64):
    fig.add_scatter3d(x=rs[i, :, 0],y=rs[i, :, 1],z=rs[i, :, 2], name='rs{}'.format(i), mode='markers', marker_size = 2)
fig.update_layout(scene_aspectmode='data')
fig.show() '''



# # 需要的参数：'xsurf', 'ysurf', 'zsurf', 'nn', 'nx', 'ny', 'nz','Bn', 'Bx', 'By', 'Bz',

# 

# print(ellipse['nn'][0])
