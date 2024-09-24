import jax.numpy as np
from jax.experimental.ode import odeint
from functools import partial
from jax import jit, config
from scipy.integrate import solve_ivp
import sys
import time
config.update("jax_enable_x64", True)


PI = np.pi



def computeB(xyz, dl, r_coil, I):
    mu_0I = I * 1e-7
    mu_0Idl = mu_0I[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] * dl # NC x NNR x NBR x NS x 3
    r_minus_l = xyz[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:] - r_coil # NC x NS x NNR x NBR x 3
    top = np.cross(mu_0Idl,r_minus_l) # NC x x NS x NNR x NBR x 3
    bottom = np.linalg.norm(r_minus_l,axis=-1)**3 # NC x NS x NNR x NBR
    B_xyz = np.sum(top / bottom[:,:,:,:,np.newaxis], axis=(0,1,2,3)) # 3, xyz coordinates
    return B_xyz



def tracing(r_coil, dl, I, r0, z0, phi0, niter, nfp, nstep, **kwargs):
    # define the integrand in cylindrical coordinates
    def fieldline(phi, rz):
        rpz = np.array([rz[0], phi, rz[1]])
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        xyz = np.array([rpz[0] * cosphi, rpz[0] * sinphi, rpz[2]])
        Bxyz = lambda xyz : computeB(xyz, dl, r_coil, I)
        mag_xyz = np.ravel(Bxyz(xyz))
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