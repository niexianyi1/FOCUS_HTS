

### 对比自动微分的导数值和有限差分的导数值 
import json
import plotly.graph_objects as go
import bspline
import jax.numpy as np
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
    mu_0 = 0.1
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

rc = np.load("/home/nxy/codes/focusadd-spline/initfiles/w7x/circle_coil.npy")
c, bc  = bspline.prep(rc, 10, rc.shape[1], 3)
### 磁场项
nn = np.load('/home/nxy/codes/focusadd-spline/initfiles/w7x/highres_nn_surf.npy')
sg = np.load('/home/nxy/codes/focusadd-spline/initfiles/w7x/highres_sg_surf.npy')
r_surf = np.load('/home/nxy/codes/focusadd-spline/initfiles/w7x/highres_r_surf.npy')
dB = np.zeros((10, 3, c.shape[2]))

d = 1e-4
# def length(c):
#     r_coil = vmap(lambda c : bspline.splev(bc, c ), in_axes=0, out_axes=0)(c)
#     len = average_length(r_coil)
#     return len
def db(c):
    r_coil = vmap(lambda c : bspline.splev(bc, c ), in_axes=0, out_axes=0)(c)
    r_coil = symmetry(r_coil)[:, :-1, np.newaxis, np.newaxis, :]
    der1, _ = vmap(lambda c : bspline.der1_splev(bc, c), in_axes=0, out_axes=0)(c)
    dl = symmetry(der1)[:, :-1, np.newaxis, np.newaxis, :] / ns
    B = quadratic_flux(nn, sg, r_coil, r_surf, dl)
    return B
gbc = np.load("/home/nxy/codes/focusadd-spline/gc.npy")


# len0 = length(c)
b0 = db(c)
for i in range(1):
    for j in range(3):
        print(i,j)
        for k in range(c.shape[2]):
            c_new = c.at[i, j, k].add(c[i, j, k] * 1e-5)
            bnew = db(c_new)
            dB = dB.at[i, j, k].set((bnew-b0)/(c[i, j, k] * 1e-5))
            # len = length(c_new)
            # dB = dB.at[i, j, k].set((len-len0)/(c[i, j, k] * 1e-5))

# print('gc = ', gbc[0, :, 3:-3])
print('db = ', dB[0])

# fig = go.Figure()
# fig.add_scatter(x=np.arange(1, 68, 1), y=dB[0, 0, :], name='gc_fd', line=dict(width=2))
# fig.add_scatter(x=np.arange(1, 68, 1), y=gbc[0, 0, :], name='gc_ad', line=dict(width=2))
# fig.update_xaxes(title_text = "number",title_font = {"size": 30},title_standoff = 15, tickfont=dict(size=30))
# fig.update_yaxes(title_text = "gc",title_font = {"size": 30},title_standoff = 15, tickfont=dict(size=30))
#                 # ,type="log", exponentformat = 'e'
# fig.show()





# print('(db-gc)/db = ', (dB-gbc)/dB)
print('mean = ', np.mean((dB[0, 0, :-1]-gbc[0, 0, :-1])/dB[0, 0, :-1]))













