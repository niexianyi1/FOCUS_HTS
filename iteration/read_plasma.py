


import jax.numpy as np
import numpy
pi = np.pi
from jax.config import config
config.update("jax_enable_x64", True)




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
            n, m, rc, _, _, zs = f.readline().split()
            N[i] = n
        nmax = numpy.max(N)
    MZ = int(nmax)
    MT = int((bmn - MZ-1) / (2*MZ+1) + 1)

    R = numpy.zeros((2*MZ+1, MT))
    Z = numpy.zeros((2*MZ+1, MT))

    with open(filename) as f:
        _ = f.readline()
        _ = f.readline()
        _ = f.readline()
        _ = f.readline()
        for i in range(bmn):
            n, m, rc, _, _, zs = f.readline().split()
            n = int(n)
            m = int(m)
            rc = float(rc)
            zs = float(zs)
            R[n+8, m] = rc
            Z[n+8, m] = zs
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


def plasma_surface(args):
    R, Z, Nfp, MT, MZ = read_plasma_boundary("{}".format(args['surface_vmec_file']))
    r, nn, sg = get_plasma_boundary(R, Z, args['number_zeta'], args['number_theta'], Nfp, MT, MZ)

    return r, nn, sg

