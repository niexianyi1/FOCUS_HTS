import jax.numpy as np
from jax.experimental.ode import odeint
from functools import partial
from jax.config import config
import jax
from jax import jit
from scipy.integrate import solve_ivp
import sys
import time
config.update("jax_enable_x64", True)


PI = np.pi



def computeB(xyz, dl, r_coil):
    """
        Inputs:

        r, zeta, z : The coordinates of the point we want the magnetic field at. Cylindrical coordinates.

        Outputs: 

        B_z, B_zeta, B_z : the magnetic field components at the input coordinates created by the currents in the coils. Cylindrical coordinates.
    """
  
    mu_0I = np.ones((r_coil.shape[0]))
    mu_0Idl = mu_0I[:,np.newaxis,np.newaxis] * dl # NC x NS x NNR x NBR x 3
    r_minus_l = xyz[np.newaxis,np.newaxis,:] - r_coil[:,:,:] # NC x NS x NNR x NBR x 3
    top = np.cross(mu_0Idl,r_minus_l) # NC x x NS x NNR x NBR x 3
    bottom = np.linalg.norm(r_minus_l,axis=-1)**3 # NC x NS x NNR x NBR
    B_xyz = np.sum(top / bottom[:,:,np.newaxis], axis=(0,1)) # 3, xyz coordinates
    
    return B_xyz



def tracing(r_coil, dl, r0, z0, phi0, niter, nfp, nstep, **kwargs):
    """Trace magnetic field line in toroidal geometry

    Args:
        bfield (callable): A callable function.
                          The calling signature is `B = bfield(xyz)`, where `xyz`
                          is the position in cartesian coordinates and `B` is the
                          magnetic field at this point (in cartesian coordinates).
        r0 (list): Initial radial coordinates.
        z0 (list): Initial vertical coordinates.
        phi0 (float, optional): The toroidal angle where the poincare plot data saved.
                                Defaults to 0.0.
        niter (int, optional): Number of toroidal periods in tracing. Defaults to 100.
        nfp (int, optional): Number of field periodicity. Defaults to 1.
        nstep (int, optional): Number of intermediate step for one period. Defaults to 1.

    Returns:
        array_like: The stored poincare date, shape is (len(r0), niter+1, 2).
    """
    
    # define the integrand in cylindrical coordinates
    def fieldline(phi, rz):
        rpz = np.array([rz[0], phi, rz[1]])
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        xyz = np.array([rpz[0] * cosphi, rpz[0] * sinphi, rpz[2]])
        mag_xyz = np.ravel(computeB(xyz, dl, r_coil))
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