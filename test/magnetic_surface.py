import plotly.graph_objects as go
import json
import jax.numpy as np
import numpy
pi = np.pi
from jax.config import config
config.update("jax_enable_x64", True)


def plasma_surface(args):
    R, Z, Nfp, MT, MZ = read_plasma_boundary("{}".format(args['surface_vmec_file']))
    r, nn, sg = get_plasma_boundary(R, Z, args['number_zeta'], args['number_theta'], Nfp, MT, MZ)
    args['number_field_periods'] = Nfp
    return r, nn, sg, args

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
            m, n, rc, zs, _, _ = f.readline().split()
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
            m, n, rc, zs, _, _ = f.readline().split()
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


def read_finite_beta(filename):

    with open(filename) as f:
        _ = f.readline()
        bmn, Nfp, nbf = f.readline().split()
        bmn, Nfp, nbf = int(bmn), int(Nfp), int(nbf)
        for i in range(bmn+4):
            _ = f.readline()
        n, m, bnc, bns = f.readline().split()
        n, m = int(n), int(m)
        MZ = n
        MT = int(nbf/(2*MZ+1))
        BNC, BNS = numpy.zeros((2*MZ+1, MT)), numpy.zeros((2*MZ+1, MT))
        BNC[n+MZ, m], BNS[n+MZ, m] = float(bnc), float(bns)
        for i in range(nbf-1):
            n, m, bnc, bns = f.readline().split()
            n, m = int(n), int(m)
            BNC[n+MZ, m], BNS[n+MZ, m] = float(bnc), float(bns)

    return BNC, BNS, MZ, MT, Nfp

with open('/home/nxy/codes/coil_spline_HTS/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
nic = 3
ns = 210
nb = nn = 3
r, nn, sg, args = plasma_surface(args)
coil = np.load('/home/nxy/codes/coil_spline_HTS/initfiles/qas/coils_3.npy')
r_surf = np.zeros((150, args['number_theta']+1, 3))
r_surf = r_surf.at[:,:-1,:].set(r)
r_surf = r_surf.at[:,-1,:].set(r[:,0,:])
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
fig = go.Figure()
for i in range(3):
    fig.add_trace(go.Surface(x=xx[:,i,:], y=yy[:,i,:], z=zz[:,i,:]))
fig.add_trace(go.Surface(x=r_surf[:,:,0], y=r_surf[:,:,1], z=r_surf[:,:,2] ))
fig.update_layout(scene_aspectmode='data')
fig.show() 