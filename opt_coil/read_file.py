## Read information from various files.
## 'hdf5', 'makegrid', 'plasma.boundary'

import jax.numpy as np
import h5py
import numpy
pi = np.pi
from jax import config
config.update("jax_enable_x64", True)


def read_hdf5(filename):
    f = h5py.File(filename, "r")
    arge = {}
    for key in list(f.keys()):
        val = f[key][()]
        if isinstance(val, bytes):
            val = str(val, encoding='utf-8')
        arge.update({key: val})
    f.close()
    return arge


def read_makegrid(filename, nc, nic):    
    I = np.zeros((nic))
    with open(filename) as f:
        lines = f.readlines()
    len_lines = len(lines)
    ns = int(((len_lines - 4) / nc) - 1 )
    nfp = int((lines[0].split())[-1])
    coordinates = np.zeros((nic, ns+1, 3))    
    for i in range(nic):
        for s in range(ns):
            x = lines[3+i*2*nfp*(ns+1)+s].split()
            coordinates = coordinates.at[i, s, 0].set(float(x[0]))
            coordinates = coordinates.at[i, s, 1].set(float(x[1]))
            coordinates = coordinates.at[i, s, 2].set(float(x[2]))
        I = I.at[i].set(float(x[3]))
    coordinates = coordinates.at[:, -1, :].set(coordinates[:, 0, :])
    return coordinates, I


def read_plasma(args):
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
                n, m, rc, _, _, zs = f.readline().split()
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

    def calculate_drdz(R, Z, zeta, theta, Nfp, MT, MZ):
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

    def calculate_drdt(R, Z, zeta, theta, Nfp, MT, MZ):
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
        if args['stellarator_symmetry'] == 1:
            zeta = np.linspace(0, np.pi / Nfp, int(NZ/2) + 1)[:-1] + np.pi / int(NZ/2) / Nfp / 2
        else:
            zeta = np.linspace(0, 2 * np.pi / Nfp, NZ + 1)[:-1] + np.pi / NZ / Nfp
        theta = np.linspace(0, 2 * np.pi, NT + 1)[:-1]
        r = get_xyz(R, Z, zeta, theta, Nfp, MT, MZ)
        drdz = calculate_drdz(R, Z, zeta, theta, Nfp, MT, MZ)
        drdt = calculate_drdt(R, Z, zeta, theta, Nfp, MT, MZ)
        N = np.cross(drdz, drdt)
        sg = np.linalg.norm(N, axis=-1)
        nn = N / sg[:,:,np.newaxis]
        return r, nn, sg

    R, Z, Nfp, MT, MZ = read_plasma_boundary("{}".format(args['surface_file']))
    r, nn, sg = get_plasma_boundary(R, Z, args['number_zeta'], args['number_theta'], Nfp, MT, MZ)
    return args, r, nn, sg 



def read_finite_beta_Bn(args):
    filename = args['Bn_background_file']
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
        BNS[n+MZ, m] = float(bns)
        for i in range(nbf-1):
            n, m, _, bns = f.readline().split()
            n, m = int(n), int(m)
            BNS[n+MZ, m] = float(bns)
    zeta = np.linspace(0, 2 * np.pi, args['number_zeta'] + 1)[:-1]
    theta = np.linspace(0, 2 * np.pi, args['number_theta'] + 1)[:-1]
    Bn = np.zeros((args['number_zeta'], args['number_theta']))
    for mz in range(-MZ, MZ + 1):
        for mt in range(MT):
            Bn += BNS[mz + MZ, mt] * np.sin( mt * theta[np.newaxis, :] - mz * Nfp * zeta[:, np.newaxis] )
    return Bn



