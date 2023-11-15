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


with open('/home/nxy/codes/focusadd-spline/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)


def biotSavart(I, dl, r_surf, r_coil):
    # 先只算一个线圈输入 r_coil = r_coil[0]

    mu_0 = 1
    mu_0I = I * mu_0
    mu_0Idl = mu_0I[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] * dl
    r_minus_l = (r_surf[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, :] - r_coil[:, np.newaxis, :, :, :, :]) 
    print(r_minus_l.shape) 
    top = np.cross(mu_0Idl[:, np.newaxis, :, :, :, :], r_minus_l)  # NC x NZ x NT x NS x NNR x NBR x 3
    bottom = (np.linalg.norm(r_minus_l, axis=-1) ** 3)  # NC x NZ x NT x NS x NNR x NBR
    B = np.sum(top / bottom[:, :, :, :, :, np.newaxis], axis=( 0, 2, 3, 4))  # NZ x NT x 3

    return B

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
    B = (np.sum(np.cross(Riv, Rfv) * ((Ri + Rf) / ((Ri * Rf) * (Ri * Rf + np.sum(Riv * Rfv, axis=2))))[:, :, np.newaxis],axis=1,))
    return B


class CoilSet:
    def __init__(self, args):
        
        self.nc = args['nc']
        self.nfp = args['nfp']
        self.ns = args['ns']
        self.ln = args['ln']
        self.lb = args['lb']
        self.nnr = args['nnr']
        self.nbr = args['nbr']
        self.rc = args['rc']
        self.nr = args['nr']
        self.nfr = args['nfr']
        self.bc = args['bc']      
        self.out_hdf5 = args['out_hdf5']
        self.out_coil_makegrid = args['out_coil_makegrid']
        self.theta = np.linspace(0, 2*pi, self.ns+1)
        self.ncnfp = int(self.nc/self.nfp)
        self.I = args['I']
        return
    

    def coilset(self, params):                           

        c, fr = params                                                                   
        I_new = self.I / (self.nnr * self.nbr)
        r_centroid = CoilSet.compute_r_centroid(self, c)  #r_centroid :[nc, ns+1, 3]
        der1, der2, der3 = CoilSet.compute_der(self, c)
        tangent, normal, binormal = CoilSet.compute_com(self, der1, r_centroid)
        r = CoilSet.compute_r(self, fr, normal, binormal, r_centroid)
        frame = tangent, normal, binormal
        dl = CoilSet.compute_dl(self, params, frame, der1, der2, r_centroid)
        # r = CoilSet.stellarator_symmetry(self, r)
        # r = CoilSet.symmetry(self, r)
        # dl = CoilSet.stellarator_symmetry(self, dl)
        # dl = CoilSet.symmetry(self, dl)
        return I_new, dl, r

    def compute_r_centroid(self, c):         # rc计算的是（nc,ns+1,3）
        rc = vmap(lambda c :bspline.splev(self.bc, c), in_axes=0, out_axes=0)(c)
        return rc

    def compute_der(self, c):                    
        """ Computes  1,2,3 derivatives of the rc """
        der1, wrk1 = vmap(lambda c :bspline.der1_splev(self.bc, c), in_axes=0, out_axes=0)(c)
        der2, wrk2 = vmap(lambda wrk1 :bspline.der2_splev(self.bc, wrk1), in_axes=0, out_axes=0)(wrk1)
        der3 = vmap(lambda wrk2 :bspline.der3_splev(self.bc, wrk2), in_axes=0, out_axes=0)(wrk2)
        return der1, der2, der3

    def compute_com(self, der1, r_centroid):    
        """ Computes T, N, and B """
        tangent = CoilSet.compute_tangent(self, der1)
        normal = -CoilSet.compute_normal(self, r_centroid, tangent)
        binormal = CoilSet.compute_binormal(self, tangent, normal)
        return tangent, normal, binormal

    def compute_com_deriv(self, frame, der1, der2, r_centroid):  
        tangent, normal, _ = frame
        tangent_deriv = CoilSet.compute_tangent_deriv(self, der1, der2)
        normal_deriv = -CoilSet.compute_normal_deriv(self, tangent, tangent_deriv, der1, r_centroid)
        binormal_deriv = CoilSet.compute_binormal_deriv(self, tangent, normal, tangent_deriv, normal_deriv)
        return tangent_deriv, normal_deriv, binormal_deriv

    def compute_tangent(self, der1):          
        """
        Computes the tangent vector of the coils. Uses the equation 
        T = dr/d_theta / |dr / d_theta|
        """
        return der1 / np.linalg.norm(der1, axis=-1)[:, :, np.newaxis]

    def compute_tangent_deriv(self, der1, der2):     
        norm_der1 = np.linalg.norm(der1, axis=-1)
        mag_2 = CoilSet.dot_product_rank3_tensor(der1, der2) / norm_der1 ** 3
        return der2 / norm_der1[:, :, np.newaxis] - der1 * mag_2[:, :, np.newaxis]

    def dot_product_rank3_tensor(a, b):         # dot
        return (a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1] + a[:, :, 2] * b[:, :, 2])

    def compute_coil_mid(self, r_centroid):      # mid_point
        x = r_centroid[:, :-1, 0]
        y = r_centroid[:, :-1, 1]
        z = r_centroid[:, :-1, 2]
        r0 = np.zeros((self.ncnfp, 3))
        for i in range(self.ncnfp):
            r0 = r0.at[i, 0].add(np.sum(x[i]) / self.ns)
            r0 = r0.at[i, 1].add(np.sum(y[i]) / self.ns)
            r0 = r0.at[i, 2].add(np.sum(z[i]) / self.ns)        
        return r0

    def compute_normal(self, r_centroid, tangent):    
        r0 = CoilSet.compute_coil_mid(self, r_centroid)
        delta = r_centroid - r0[:, np.newaxis, :]
        dp = CoilSet.dot_product_rank3_tensor(tangent, delta)
        normal = delta - tangent * dp[:, :, np.newaxis]
        mag = np.linalg.norm(normal, axis=-1)
        return normal / mag[:, :, np.newaxis]

    def compute_normal_deriv(self, tangent, tangent_deriv, der1, r_centroid):          
        r0 = CoilSet.compute_coil_mid(self, r_centroid)
        delta = r_centroid - r0[:, np.newaxis, :]
        dp1 = CoilSet.dot_product_rank3_tensor(tangent, delta)
        dp2 = CoilSet.dot_product_rank3_tensor(tangent, der1)
        dp3 = CoilSet.dot_product_rank3_tensor(tangent_deriv, delta)
        numerator = delta - tangent * dp1[:, :, np.newaxis]
        numerator_norm = np.linalg.norm(numerator, axis=-1)
        numerator_deriv = (
            der1
            - dp1[:, :, np.newaxis] * tangent_deriv
            - tangent * (dp2 + dp3)[:, :, np.newaxis]
        )
        dp4 = CoilSet.dot_product_rank3_tensor(numerator, numerator_deriv)
        return (
            numerator_deriv / numerator_norm[:, :, np.newaxis]
            - (dp4 / numerator_norm ** 3)[:, :, np.newaxis] * numerator
        )

    def compute_binormal(self, tangent, normal):           
        """ Computes the binormal vector of the coils, B = T x N """
        return np.cross(tangent, normal)

    def compute_binormal_deriv(self, tangent, normal, tangent_deriv, normal_deriv):  
        return np.cross(tangent_deriv, normal) + np.cross(tangent, normal_deriv)

    def compute_alpha(self, fr):    # alpha 用fourier
        alpha = np.zeros((self.ncnfp, self.ns + 1))
        alpha += self.theta * self.nr / 2
        Ac = fr[0]
        As = fr[1]
        for m in range(self.nfr):
            arg = self.theta * m
            carg = np.cos(arg)
            sarg = np.sin(arg)
            alpha += (
                Ac[:, np.newaxis, m] * carg[np.newaxis, :]
                + As[:, np.newaxis, m] * sarg[np.newaxis, :]
            )
        return alpha

    def compute_alpha_1(self, fr):    
        alpha_1 = np.zeros((self.ncnfp, self.ns + 1))
        alpha_1 += self.nr / 2
        Ac = fr[0]
        As = fr[1]
        for m in range(self.nfr):
            arg = self.theta * m
            carg = np.cos(arg)
            sarg = np.sin(arg)
            alpha_1 += (
                -m * Ac[:, np.newaxis, m] * sarg[np.newaxis, :]
                + m * As[:, np.newaxis, m] * carg[np.newaxis, :]
            )
        return alpha_1

    def compute_frame(self, fr, N, B):  
        """
        Computes the vectors v1 and v2 for each coil. v1 and v2 are rotated relative to
        the normal and binormal frame by an amount alpha. Alpha is parametrized by a Fourier series.
        """
        alpha = CoilSet.compute_alpha(self, fr)
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)
        v1 = calpha[:, :, np.newaxis] * N - salpha[:, :, np.newaxis] * B
        v2 = salpha[:, :, np.newaxis] * N + calpha[:, :, np.newaxis] * B
        return v1, v2

    def compute_frame_derivative(self, params, frame, der1, der2, r_centroid):    
        _, N, B = frame
        _, fr = params
        alpha = CoilSet.compute_alpha(self, fr)
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)
        alpha1 = CoilSet.compute_alpha_1(self, fr)
        _, dNdt, dBdt = CoilSet.compute_com_deriv(self, frame, der1, der2, r_centroid)
        dv1_dt = (
            calpha[:, :, np.newaxis] * dNdt
            - salpha[:, :, np.newaxis] * dBdt
            - salpha[:, :, np.newaxis] * N * alpha1[:, :, np.newaxis]
            - calpha[:, :, np.newaxis] * B * alpha1[:, :, np.newaxis]
        )
        dv2_dt = (
            salpha[:, :, np.newaxis] * dNdt
            + calpha[:, :, np.newaxis] * dBdt
            + calpha[:, :, np.newaxis] * N * alpha1[:, :, np.newaxis]
            - salpha[:, :, np.newaxis] * B * alpha1[:, :, np.newaxis]
        )
        return dv1_dt, dv2_dt

    def compute_r(self, fr, normal, binormal, r_centroid):      
        """
        Computes the position of the multi-filament coils.

        r is a nc x ns + 1 x nnr x nbr x 3 array which holds the coil endpoints
        dl is a nc x ns x nnr x nbr x 3 array which computes the length of the ns segments
        r_middle is a nc x ns x nnr x nbr x 3 array which computes the midpoint of each of the ns segments

        """

        v1, v2 = CoilSet.compute_frame(self, fr, normal, binormal)
        r = np.zeros((self.ncnfp, self.ns +1, self.nnr, self.nbr, 3))
        r += r_centroid[:, :, np.newaxis, np.newaxis, :]
        for n in range(self.nnr):
            for b in range(self.nbr):
                r = r.at[:, :, n, b, :].add(
                    (n - 0.5 * (self.nnr - 1)) * self.ln * v1 + (b - 0.5 * (self.nbr - 1)) * self.lb * v2
                ) 
        return r[:, :-1, :, :, :]

    def compute_dl(self, params, frame, der1, der2, r_centroid):   
        dl = np.zeros((self.ncnfp, self.ns + 1, self.nnr, self.nbr, 3))
        dl += der1[:, :, np.newaxis, np.newaxis, :]
        dv1_dt, dv2_dt = CoilSet.compute_frame_derivative(self, params, frame, der1, der2, r_centroid)
        for n in range(self.nnr):
            for b in range(self.nbr):
                dl = dl.at[:, :, n, b, :].add(
                    (n - 0.5 * (self.nnr - 1)) * self.ln * dv1_dt + (b - 0.5 * (self.nbr - 1)) * self.lb * dv2_dt
                )

        return dl[:, :-1, :, :, :] * (1 / (self.ns+2))

    def symmetry(self, r):
        rc_total = np.zeros((self.nc, self.ns, self.nnr, self.nbr, 3))
        rc_total = rc_total.at[0:self.ncnfp, :, :, :, :].add(r)
        for i in range(self.nfp - 1):        
            theta = 2 * pi * (i + 1) / self.nfp
            T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            rc_total = rc_total.at[self.ncnfp*(i+1):self.ncnfp*(i+2), :, :, :, :].add(np.dot(r, T))
        
        return rc_total

def symmetry(args, r):
    ncnfp = int(args['nc'] / args['nfp'])
    rc_total = np.zeros((args['nc'], args['ns'], 3))
    rc_total = rc_total.at[0:ncnfp, :, :].add(r)
    for i in range(args['nfp'] - 1):        
        theta = 2 * np.pi * (i + 1) / args['nfp']
        T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        rc_total = rc_total.at[ncnfp*(i+1):ncnfp*(i+2), :, :].add(np.dot(r, T))
    
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

c = np.load('/home/nxy/codes/focusadd-spline/results/circle/c_100b.npy')
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



