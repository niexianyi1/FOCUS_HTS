import jax.numpy as np
from jax import config

config.update("jax_enable_x64", True)

def get_fourier_init(init_coil, file_type, nic, ns, nfc):       
    if file_type == 'npy':
        coil = np.load("{}".format(init_coil))
    if file_type == 'makegrid':
        coil = read_makegrid(init_coil, nic, ns)
    fc = compute_coil_fourierSeries(nic, ns, nfc, coil)
    return fc

def read_makegrid(filename, nic, ns):    
    """读取makegrid文件"""
    r = np.zeros((nic, ns+1, 3))
    with open(filename) as f:
        _ = f.readline()
        _ = f.readline()
        _ = f.readline()
        for i in range(nic):
            for s in range(ns):
                x = f.readline().split()
                r = r.at[i, s, 0].set(float(x[0]))
                r = r.at[i, s, 1].set(float(x[1]))
                r = r.at[i, s, 2].set(float(x[2]))
            _ = f.readline()
    r = r.at[:, -1, :].set(r[:, 0, :])
    return r

def compute_coil_fourierSeries(nic, ns, nfc, coil):
    """ 
    Takes a set of centroid positions and gives the coefficients
    of the coil fourier series in a single array 

    Inputs:
    r_centroid (nparray): vector of length nic x ns + 1 x 3, initial coil centroid

    Returns:
    6 x nic x nfc array with the Fourier Coefficients of the initial coils
    """
    x = coil[:, :-1, 0]  # nic x ns
    y = coil[:, :-1, 1]
    z = coil[:, :-1, 2]
    xc = np.zeros((nic, nfc))
    yc = np.zeros((nic, nfc))
    zc = np.zeros((nic, nfc))
    xs = np.zeros((nic, nfc))
    ys = np.zeros((nic, nfc))
    zs = np.zeros((nic, nfc))
    xc = xc.at[:, 0].set(np.sum(x, axis=1) / ns)
    yc = yc.at[:, 0].set(np.sum(y, axis=1) / ns)
    zc = zc.at[:, 0].set(np.sum(z, axis=1) / ns)
    theta = np.linspace(0, 2 * np.pi, ns + 1)[:-1]
    for m in range(1, nfc):
        xc = xc.at[:, m].set(2.0 * np.sum(x * np.cos(m * theta), axis=1) / ns)
        yc = yc.at[:, m].set(2.0 * np.sum(y * np.cos(m * theta), axis=1) / ns)
        zc = zc.at[:, m].set(2.0 * np.sum(z * np.cos(m * theta), axis=1) / ns)
        xs = xs.at[:, m].set(2.0 * np.sum(x * np.sin(m * theta), axis=1) / ns)
        ys = ys.at[:, m].set(2.0 * np.sum(y * np.sin(m * theta), axis=1) / ns)
        zs = zs.at[:, m].set(2.0 * np.sum(z * np.sin(m * theta), axis=1) / ns)
    fc = np.asarray([xc, yc, zc, xs, ys, zs])  # 6 x nic x nfc
    return fc  

def compute_r_centroid(fc, nfc, nic, ns, theta):
    """ Computes the position of the winding pack centroid using the coil fourier series """
    xc, yc, zc, xs, ys, zs = fc[0], fc[1], fc[2], fc[3], fc[4], fc[5]
    x = np.zeros((nic, ns + 1))
    y = np.zeros((nic, ns + 1))
    z = np.zeros((nic, ns + 1))
    for m in range(nfc):
        arg = m * theta
        carg = np.cos(arg)
        sarg = np.sin(arg)
        x += (xc[:, np.newaxis, m] * carg[np.newaxis, :]
            + xs[:, np.newaxis, m] * sarg[np.newaxis, :])
        y += (yc[:, np.newaxis, m] * carg[np.newaxis, :]
            + ys[:, np.newaxis, m] * sarg[np.newaxis, :])
        z += (zc[:, np.newaxis, m] * carg[np.newaxis, :]
            + zs[:, np.newaxis, m] * sarg[np.newaxis, :])
    rc = np.concatenate(
        (x[:, :, np.newaxis], y[:, :, np.newaxis], z[:, :, np.newaxis]), axis=2)
    return rc

def compute_der1(fc, nfc, nic, ns, theta):
    xc, yc, zc, xs, ys, zs = fc[0], fc[1], fc[2], fc[3], fc[4], fc[5]
    x1 = np.zeros((nic, ns + 1))
    y1 = np.zeros((nic, ns + 1))
    z1 = np.zeros((nic, ns + 1))
    for m in range(nfc):
        arg = m * theta
        carg = np.cos(arg)
        sarg = np.sin(arg)
        x1 += (-m * xc[:, np.newaxis, m] * sarg[np.newaxis, :]
            + m * xs[:, np.newaxis, m] * carg[np.newaxis, :])
        y1 += (-m * yc[:, np.newaxis, m] * sarg[np.newaxis, :]
            + m * ys[:, np.newaxis, m] * carg[np.newaxis, :])
        z1 += (-m * zc[:, np.newaxis, m] * sarg[np.newaxis, :]
            + m * zs[:, np.newaxis, m] * carg[np.newaxis, :])
    der1 = np.concatenate(
        (x1[:, :, np.newaxis], y1[:, :, np.newaxis], z1[:, :, np.newaxis]), axis=2)
    return der1

def compute_der2(fc, nfc, nic, ns, theta):
    xc, yc, zc, xs, ys, zs = fc[0], fc[1], fc[2], fc[3], fc[4], fc[5]
    x2 = np.zeros((nic, ns + 1))
    y2 = np.zeros((nic, ns + 1))
    z2 = np.zeros((nic, ns + 1))
    for m in range(nfc):
        m2 = m ** 2
        arg = m * theta
        carg = np.cos(arg)
        sarg = np.sin(arg)
        x2 += (-m2 * xc[:, np.newaxis, m] * carg[np.newaxis, :]
            - m2 * xs[:, np.newaxis, m] * sarg[np.newaxis, :])
        y2 += (-m2 * yc[:, np.newaxis, m] * carg[np.newaxis, :]
            - m2 * ys[:, np.newaxis, m] * sarg[np.newaxis, :])
        z2 += (-m2 * zc[:, np.newaxis, m] * carg[np.newaxis, :]
            - m2 * zs[:, np.newaxis, m] * sarg[np.newaxis, :])
    der2 = np.concatenate(
        (x2[:, :, np.newaxis], y2[:, :, np.newaxis], z2[:, :, np.newaxis]), axis=2)
    return der2

def compute_der3(fc, nfc, nic, ns, theta):
    xc, yc, zc, xs, ys, zs = fc[0], fc[1], fc[2], fc[3], fc[4], fc[5]
    x3 = np.zeros((nic, ns + 1))
    y3 = np.zeros((nic, ns + 1))
    z3 = np.zeros((nic, ns + 1))
    for m in range(nfc):
        m3 = m ** 3
        arg = m * theta
        carg = np.cos(arg)
        sarg = np.sin(arg)
        x3 += (m3 * xc[:, np.newaxis, m] * sarg[np.newaxis, :]
            - m3 * xs[:, np.newaxis, m] * carg[np.newaxis, :])
        y3 += (m3 * yc[:, np.newaxis, m] * sarg[np.newaxis, :]
            - m3 * ys[:, np.newaxis, m] * carg[np.newaxis, :])
        z3 += (m3 * zc[:, np.newaxis, m] * sarg[np.newaxis, :]
            - m3 * zs[:, np.newaxis, m] * carg[np.newaxis, :])
    der3 = np.concatenate(
        (x3[:, :, np.newaxis], y3[:, :, np.newaxis], z3[:, :, np.newaxis]), axis=2)
    return der3










