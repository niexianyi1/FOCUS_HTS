### 单圆形线圈在轴上产生磁场对比


import json
import plotly.graph_objects as go
import bspline 
import jax.numpy as np
import coilpy
pi = np.pi
from jax import vmap
from jax.config import config
import scipy.interpolate as si 
config.update("jax_enable_x64", True)


with open('/home/nxy/codes/coil_spline_HTS/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)



## 单磁面点输入，
def BS(I, dl, r_surf, r_coil):

    mu_0 = 1
    mu_0I = I * mu_0
    mu_0Idl = np.zeros_like(dl)
    for i in range(len(dl)):
        mu_0Idl = mu_0Idl.at[i].set(dl[i] * mu_0I[i] )
    dl = mu_0Idl

    ## r_surf: (3) , r_coil: (nc,ns,nnr,nbr, 3), dl: (nc,ns,nnr,nbr, 3)
    dx = r_surf[0] - r_coil[:,:,:,:,0]
    dy = r_surf[1] - r_coil[:,:,:,:,1]
    dz = r_surf[2] - r_coil[:,:,:,:,2]
    dr = dx * dx + dy * dy + dz * dz
    Bx = (dz * dl[:,:,:,:,1] - dy * dl[:,:,:,:,2]) * np.power(dr, -1.5)
    By = (dx * dl[:,:,:,:,2] - dz * dl[:,:,:,:,0]) * np.power(dr, -1.5) 
    Bz = (dy * dl[:,:,:,:,0] - dx * dl[:,:,:,:,1]) * np.power(dr, -1.5) 
    B = np.array([np.sum(Bx), np.sum(By), np.sum(Bz)]) 

    return B

## 单线圈输入，
def fd(I, r_surf, x,y,z):
    xt = x[1:] - x[:-1]
    yt = y[1:] - y[:-1]
    zt = z[1:] - z[:-1]
    dx = r_surf[0] - (x[:-1] + x[1:]) / 2
    dy = r_surf[1] - (y[:-1] + y[1:]) / 2
    dz = r_surf[2] - (z[:-1] + z[1:]) / 2
    dr = dx * dx + dy * dy + dz * dz
    Bx = (dz * yt - dy * zt) * np.power(dr, -1.5)
    By = (dx * zt - dz * xt) * np.power(dr, -1.5)
    Bz = (dy * xt - dx * yt) * np.power(dr, -1.5)
    B = np.array([np.sum(Bx), np.sum(By), np.sum(Bz)]) *pi/32
    return B

def hh(r_coil, r_surf):
    Rvec = r_surf[:, np.newaxis, :] - r_coil[np.newaxis, :, :]
    assert (Rvec.shape)[-1] == 3
    RR = np.linalg.norm(Rvec, axis=2)
    Riv = Rvec[:, :-1, :]
    Rfv = Rvec[:, 1:, :]
    Ri = RR[:, :-1]
    Rf = RR[:, 1:]
    B = I*1e-7*(np.sum(np.cross(Riv, Rfv) * ((Ri + Rf) / ((Ri * Rf) * (Ri * Rf + np.sum(Riv * Rfv, axis=2))))[:, :, np.newaxis],axis=1,))
    return B


def symmetry(args, r):
    nic = int(args['nc'] / args['nfp'])
    rc_total = np.zeros((args['nc'], args['ns'], 3))
    rc_total = rc_total.at[0:nic, :, :].add(r)
    for i in range(args['nfp'] - 1):        
        theta = 2 * np.pi * (i + 1) / args['nfp']
        T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        rc_total = rc_total.at[nic*(i+1):nic*(i+2), :, :].add(np.dot(r, T))
    
    return rc_total

args['nfp'] = 1
args['nc'] = 1
I = np.ones(1)
args['I'] = I
r_surf = np.load(args['surface_r'])[0,:,:]

##### 轴上磁场解析式
# x = [0 for i in range(20)]
# y = [0 for i in range(20)]
# z = [i for i in range(20)]
# r_surf = np.array([x,y,z]).T
# B0 = np.zeros((20,3))
# for i in range(20):
#     B0 = B0.at[i,2].set(1/2/(1+z[i])**1.5*np.sqrt(1+z[i]**2))

c = np.load('/home/nxy/codes/coil_spline_HTS/results/circle/c_100b.npy')
fr = np.zeros((2, 10, args['nfr'])) 
params = c, fr
bc = bspline.get_bc_init(args['ns']+3)
args['bc'] = bc

theta = np.linspace(0,2*pi,ns+1)
x = np.cos(theta)
y = np.sin(theta)
z = np.zeros(ns+1)
coil = np.array([x,y,z]).T
coil = coil.reshape((1, ns+1, 3)) 

fr = np.zeros((2, 1, args['nfr']))        
c, bc  = bspline.tcku(coil[:,:-1,:], 1, ns, 3)    
args['bc'] = bc
params = c, fr



coil = CoilSet(args)
I, dl, r_coil = coil.coilset(params)
Bss = biotSavart(I, dl, r_surf, r_coil)
Bbs = vmap(lambda r_surf :BS(I, dl, r_surf, r_coil), in_axes=0, out_axes=0)(r_surf)

## coilpy ##
r_coil = np.squeeze(r_coil)[np.newaxis,:,:]
dl = np.squeeze(dl)[np.newaxis,:,:]
x = y = z = np.zeros((1,65))
x = x.at[:,:-1].set(r_coil[:,:, 0])
x = x.at[:,-1].set(x[:,0])
y = y.at[:,:-1].set(r_coil[:,:, 1])
y = y.at[:,-1].set(y[:,0])
z = z.at[:,:-1].set(r_coil[:,:, 2])
z = z.at[:,-1].set(z[:,0])

Bfd = np.zeros((20,3))
for i in range(20):
    b = 0
    for j in range(10):
        b += fd(I, r_surf[i,:], x[j,:], y[j,:], z[j,:])
    Bfd = Bfd.at[i,:].set(b)

r_coil = [x,y,z]
r_coil = np.array(r_coil).transpose([1,2,0])
r_coil = np.reshape(r_coil, (1*65, 3))
Bhh = hh(r_coil, r_surf)


# print('B0 = ', B0)
print('Bss = ', Bss)
print('Bfd = ', Bfd)
print('Bbs = ', Bbs)
print('Bhh = ', Bhh)

# print('Bss - B0/B0 = ', (Bss[2:,:] - B0[:-2, :])/B0[:-2, :])
# print('Bfd - B0/B0 = ', (Bfd[2:,:] - B0[:-2, :])/B0[:-2, :])
# print('Bbs - B0/B0 = ', (Bbs[2:,:] - B0[:-2, :])/B0[:-2, :])
# print('Bhh - B0/B0 = ', (Bhh[2:,:] - B0[:-2, :])/B0[:-2, :])

print('Bss - Bfd/Bfd = ', (Bss - Bfd)/Bfd)
print('Bss - Bbs/Bbs = ', (Bss - Bbs)/Bbs)
print('Bss - Bhh/Bhh = ', (Bss - Bhh)/Bhh)

print('Bfd - Bhh/Bhh = ', (Bfd - Bhh)/Bhh)



