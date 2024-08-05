### 画 poincare图，独立运行
import jax.numpy as np
from jax import vmap
import bspline
import sys
import coilpy
import plotly.graph_objects as go


def poincare(r0, z0, bc, c):

    lenr = len(r0)
    
    rc = vmap(lambda c :bspline.splev(bc, c), in_axes=0, out_axes=0)(c)
    rc = symmetry(rc[:, :-1, :])  
    x = rc[:, :, 0]   
    y = rc[:, :, 1]
    z = rc[:, :, 2]

    I = np.ones(nc)
    name = np.zeros(nc)
    group = np.zeros(nc)
    phi0=0
    coil = coilpy.coils.Coil(x, y, z, I, name, group)
    line = tracing(coil.bfield, r0, z0, phi0, 100, 1, 1)

    line = np.reshape(line, (lenr*(100+1), 2))
    fig = go.Figure()
    fig.add_scatter(x = line[:, 0], y = line[:, 1],  name='rc', mode='markers', marker_size = 1.5)
    fig.update_layout(scene_aspectmode='data')
    fig.show()
    return


def print_progress(
    iteration, total, prefix="Progress", suffix="Complete", decimals=1, bar_length=60
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = "█" * filled_length + "-" * (bar_length - filled_length)

    sys.stdout.write("\r%s |%s| %s%s %s" % (prefix, bar, percents, "%", suffix)),

    if iteration == total:
        sys.stdout.write("\n")
    sys.stdout.flush()
    return


def tracing(bfield, r0, z0, phi0=0.0, niter=100, nfp=1, nstep=1, **kwargs):

    from scipy.integrate import solve_ivp

    # define the integrand in cylindrical coordinates
    def fieldline(phi, rz):
        rpz = np.array([rz[0], phi, rz[1]])
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        xyz = np.array([rpz[0] * cosphi, rpz[0] * sinphi, rpz[2]])
        mag_xyz = np.ravel(bfield(xyz))
        mag_rpz = np.array(
            [
                mag_xyz[0] * cosphi + mag_xyz[1] * sinphi,
                (-mag_xyz[0] * sinphi + mag_xyz[1] * cosphi) / rpz[0],
                mag_xyz[2],
            ]
        )
        return [mag_rpz[0] / mag_rpz[1], mag_rpz[2] / mag_rpz[1]]

    # some settings
    print("Begin field-line tracing: ")
    if kwargs.get("method") is None:
        kwargs.update({"method": "LSODA"})  # using LSODE
    if kwargs.get("rtol") is None:
        kwargs.update({"rtol": 1e-6})  # minimum tolerance
    # begin tracing
    dphi = 2 * np.pi / nfp / nstep
    phi = phi0 + dphi * nstep * np.arange(niter)
    nlines = len(r0)
    lines = []
    for i in range(nlines):  # loop over each field-line
        points = [[r0[i], z0[i]]]
        for j in range(niter):  # loop over each toroidal iteration
            print_progress(i * niter + j + 1, nlines * niter)
            rz = points[j]
            phi_start = phi[j]
            for k in range(nstep):  # loop inside one iteration
                sol = solve_ivp(fieldline, (phi_start, phi_start + dphi), rz, **kwargs)
                rz = sol.y[:, -1]
                phi_start += dphi
            points.append(rz)
        lines.append(np.array(points))
    return np.array(lines)


def symmetry(r):
    rc_total = np.zeros((50, 64, 3))
    rc_total = rc_total.at[0:10, :, :].add(r)
    for i in range(5 - 1):        
        theta = 2 * np.pi * (i + 1) / 5
        T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        rc_total = rc_total.at[10*(i+1):10*(i+2), :, :].add(np.dot(r, T))
    
    return rc_total


# r0 = [ 5.9]
# z0 = [ 0]
bc = bspline.get_bc_init(67)
# c = np.load('results/circle/c_100b.npy')
# poincare(r0, z0, bc, c)