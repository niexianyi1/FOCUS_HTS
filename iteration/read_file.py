
import jax.numpy as np
import h5py



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


def read_makegrid(filename, nic, ns):    
    """
    读取初始线圈的makegrid文件
    Args:
        filename : str, 文件地址
        nic : int, 独立线圈数, 
        ns : int, 线圈段数
    Returns:
        r : array, [nic, ns+1, 3], 线圈初始坐标

    """
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


def read_axis(filename):
    """
	Reads the magnetic axis from a file.

	Expects the filename to be in a specified form, which is the same as the default
	axis file given. 

	Parameters: 
		filename (string): A path to the file which has the axis data
		N_zeta_axis (int): The toroidal (zeta) resolution of the magnetic axis in real space
		epsilon: The ellipticity of the axis
		minor_rad: The minor radius of the axis, a
		N_rotate: Number of rotations of the axis
		zeta_off: The offset of the rotation of the surface in the ellipse relative to the zero starting point. 

	Returns: 
		axis (Axis): An axis object for the specified parameters.
	"""
    with open(filename, "r") as file:
        file.readline()
        _, _ = map(int, file.readline().split(" "))
        file.readline()
        xc = np.asarray([float(c) for c in file.readline().split(" ")])
        file.readline()
        xs = np.asarray([float(c) for c in file.readline().split(" ")])
        file.readline()
        yc = np.asarray([float(c) for c in file.readline().split(" ")])
        file.readline()
        ys = np.asarray([float(c) for c in file.readline().split(" ")])
        file.readline()
        zc = np.asarray([float(c) for c in file.readline().split(" ")])
        file.readline()
        zs = np.asarray([float(c) for c in file.readline().split(" ")])
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        epsilon, minor_rad, N_rotate, zeta_off = map(float, file.readline().split(" "))

    return xc, xs, yc, ys, zc, zs, epsilon, minor_rad, N_rotate, zeta_off
