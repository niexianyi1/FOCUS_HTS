import json
import sys 
import jax.numpy as np
import plotly.graph_objects as go
sys.path.append('/home/nxy/codes/coil_spline_HTS/HTS')
from material_jcrit import get_critical_current
sys.path.append('/home/nxy/codes/coil_spline_HTS/iteration')
import read_init
import fourier
import spline
sys.path.append('/home/nxy/codes/coil_spline_HTS/test')
from test_coil_cal import CoilSet

with open('/home/nxy/codes/coil_spline_HTS/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)




# t--tfcoil, s--saddlecoil, p--pfcoil
nct, ncs, ncp = 12, 144, 12
nst, nss, nsp = 99, 40, 99
It = np.array([2.281852279e+07 for i in range(12)])


rt = np.load('/home/nxy/codes/coil_spline_HTS/initfiles/aries/coil_3_11/coiltf_12.npy')
# rp = np.load('/home/nxy/codes/coil_spline_HTS/initfiles/aries/coilpf_12.npy')
rs = np.load('/home/nxy/codes/coil_spline_HTS/initfiles/aries/coil_3_11/coilsd_144.npy') 
# Ip = np.load('/home/nxy/codes/coil_spline_HTS/initfiles/aries/coil_Ip_12.npy')
Is = np.load('/home/nxy/codes/coil_spline_HTS/initfiles/aries/coil_3_11/coil_Is_144.npy')

print(np.max(abs(Is)))

def biotSavart(coil, I, dl, r_surf):
    """
    Inputs:

    r : Position we want to evaluate at, NZ x NT x 3
    I : Current in ith coil, length NC
    dl : Vector which has coil segment length and direction, NC x NS x NNR x NBR x 3
    l : Positions of center of each coil segment, NC x NS x NNR x NBR x 3

    Returns: 

    A NZ x NT x 3 array which is the magnetic field vector on the surface points 
    """
    mu_0 = 1e-7
    mu_0I = I * mu_0
    mu_0Idl = (mu_0I[:, np.newaxis, np.newaxis] * dl)  # NC x NNR x NBR x NS x 3
    r_minus_l = (r_surf[np.newaxis, :, :, np.newaxis, :]
        - coil[:, np.newaxis, np.newaxis, :, :])  # NC x NZ/nfp x NT x NNR x NBR x NS x 3
    top = np.cross(mu_0Idl[:, np.newaxis, np.newaxis, :, :], r_minus_l)  # NC x NZ x NT x NNR x NBR x NS x 3
    bottom = (np.linalg.norm(r_minus_l, axis=-1) ** 3)  # NC x NZ x NT x NNR x NBR x NS
    B = np.sum(top / bottom[:, :, :, :, np.newaxis], axis=(0, 3))  # NZ x NT x 3
    return B

nfc = 6
fcs = fourier.compute_coil_fourierSeries(ncs, nss, nfc, rs)
fct = fourier.compute_coil_fourierSeries(nct, nst, nfc, rt)
# fcp = fourier.compute_coil_fourierSeries(ncp, nsp, 6, rp)
rt = rt[:, :-1, :]
rs = rs[:, :-1, :]
# rp = rp[:, :-1, :]
thetas = np.linspace(0, 2 * np.pi, nss + 1)
der1s = fourier.compute_der1(fcs, nfc, ncs, nss, thetas)
dls = der1s[:, :-1, :] * (2*np.pi/nss)
# thetap = np.linspace(0, 2 * np.pi, nsp + 1)
# der1p = fourier.compute_der1(fcp, 6, ncp, nsp, thetap)
# dlp = der1p[:, :-1, :] * (2*np.pi/nsp)
thetat = np.linspace(0, 2 * np.pi, nst + 1)
der1t = fourier.compute_der1(fct, nfc, nct, nst, thetat)
dlt = der1t[:, :-1, :] * (2*np.pi/nst)

# TF coil
# Bothers = biotSavart(rs, Is, dls, rt)
# Botherp = biotSavart(rp, Ip, dlp, rt)

# Bother = Bothers + Botherp

# frt = np.zeros((2, nct, args['number_fourier_rotate'])) 
# coil_cal = CoilSet(args)
# params = (fct, frt, It)
# a = coil_cal.get_fb_args(params, Bother)
# Bself, I_signle, lw, lt, B_coil = a

# PF coil
# Bothers = biotSavart(rs, Is, dls, rp)
# Bothert = biotSavart(rt, It, dlp, rp)
# print(np.max(np.linalg.norm(Bothers, axis=-1), axis=1))
# print(np.max(np.linalg.norm(Bothert, axis=-1), axis=1))
# Bother = Bothers + Bothert

# frp = np.zeros((2, ncp, args['number_fourier_rotate'])) 
# coil_cal = CoilSet(args)
# params = (fcp, frp, Ip)
# a = coil_cal.get_fb_args(params, Bother)
# Bself, I_signle, lw, lt, B_coil = a
# print('B = ', Bself)
# print('I = ', I_signle)
# print('lw, lt = ', lw, lt)

# saddlecoil
frs = np.zeros((2, ncs, args['number_fourier_rotate'])) 
args['length_normal'] = [0.35 for i in range(144)]
args['length_binormal'] = [0.35 for i in range(144)]
args['number_normal'] = 2
args['number_binormal'] = 2
args['number_coils'] = 144
args['number_independent_coils'] = 144
args['number_segments'] = 40

# Botherp = biotSavart(rp, Ip, dlp, rs)
Bothert = biotSavart(rt, It, dlt, rs)
Bother = Bothert  # Botherp +
coil_cal = CoilSet(args)
params = (fcs, frs, Is)
B_coil = coil_cal.get_fb_args(params, Bother)

# print('B = ', Bself)

def plot(args, params, B_coil):
    ns = args['number_segments']
    args['number_points'] = args['number_segments'] 
    coil_cal = CoilSet(args)    
    coil = coil_cal.get_coil(params)
    # args['number_segments'] = ns

    ns = args['number_points']
    nn = args['number_normal']
    nb = args['number_binormal']
    nic = args['number_independent_coils']
    rr = np.zeros((5, nic, ns+1, 3))
    rr = rr.at[0,:,:ns,:].set(coil[:nic, 0, 0, :, :])
    rr = rr.at[1,:,:ns,:].set(coil[:nic, 0, nb-1, :, :])
    rr = rr.at[2,:,:ns,:].set(coil[:nic, nn-1, nb-1, :, :])
    rr = rr.at[3,:,:ns,:].set(coil[:nic, nn-1, 0, :, :])
    rr = rr.at[4,:,:ns,:].set(coil[:nic, 0, 0, :, :])
    rr = rr.at[0,:,-1,:].set(coil[:nic, 0, 0, 0, :])
    rr = rr.at[1,:,-1,:].set(coil[:nic, 0, nb-1, 0, :])
    rr = rr.at[2,:,-1,:].set(coil[:nic, nn-1, nb-1, 0, :])
    rr = rr.at[3,:,-1,:].set(coil[:nic, nn-1, 0, 0, :])
    rr = rr.at[4,:,-1,:].set(coil[:nic, 0, 0, 0, :])
    xx = rr[:,:,:,0]
    yy = rr[:,:,:,1]
    zz = rr[:,:,:,2]
    B = np.zeros((5, nic, ns+1))
    B = B.at[0, :, :-1].set(B_coil)
    B = B.at[0, :, -1].set(B_coil[:, 0])
    for i in range(4):
        B = B.at[i+1].set(B[0])

    cmax = float(np.max(B_coil))
    cmin = float(np.min(B_coil))
    fig = go.Figure()
    for i in range(nic):
        fig.add_trace(go.Surface(x=xx[:,i,:], y=yy[:,i,:], z=zz[:,i,:], 
                surfacecolor = B[:,i,:], cmin = cmin, cmax = cmax))
    fig.update_layout(scene_aspectmode='data')
    # fig.update_coloraxes(cmax = 15, cmin = 5.2, colorbar_title='我是colorbar')

    fig.show() 
    return 



B_coil = np.linalg.norm(B_coil+Bother, axis=-1)
print(np.max(B_coil))
plot(args, params, B_coil)