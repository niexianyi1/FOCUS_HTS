
import h5py
import coilpy
import jax.numpy as np
import plotly.graph_objects as go
import sys
sys.path.append('opt_coil')
from poincare_trace import tracing


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



def plot_poincare(arge):

    r_surf = arge['surface_data_r']
    pn = arge['poincare_number']
    phi0 = arge['poincare_phi0']
    phi = int(phi0/2/np.pi * arge['number_zeta'])
    r_surf = r_surf[phi]
    surf_r = (r_surf[:, 0]**2 + r_surf[:, 1]**2)**0.5
    surf_z = r_surf[:, 2]
    r0 = surf_r[0]
    mid = int(arge['number_theta'] / 2)
    rmid = (surf_r[mid] + r0) / 2
    dr = (r0 - rmid) / (pn-1)
    r0 = [rmid+i*dr for i in range(pn)]
    z0 = [0 for i in range(pn)]
    coil = np.mean(arge['coil_r'], axis = (1,2))
    x = coil[:, :, 0]
    y = coil[:, :, 1]
    z = coil[:, :, 2]
    I = arge['coil_I'] 
    # name = group = np.ones((arge['number_coils']))
    # coil_py = coilpy.coils.Coil(x, y, z, I, name, group)
    # bfield = coil_py.bfield
    # arge['number_step'] = 10
    # line = coilpy.misc.tracing(bfield, r0, z0, arge['poincare_phi0'], 
    #         arge['number_iter'], arge['number_field_periods'], arge['number_step'], )

    coil = arge['coil_r']
    dl = arge['coil_dl']
    line = tracing(coil, dl, I, r0, z0, arge['poincare_phi0'], 
            arge['number_iter'], arge['number_field_periods'], arge['number_step'])

    line = np.reshape(line, (pn*(arge['number_iter']+1), 2))
    
    fig = go.Figure()
    fig.add_scatter(x = line[:, 0], y = line[:, 1],  name='poincare', mode='markers', marker_size = 3)
    fig.add_scatter(x = surf_r, y = surf_z,  name='surface', line = dict(width=2.5))
    fig.update_layout(scene_aspectmode='data')
    fig.show()
    return

def plot_poincare_compare(arge,old_arge):

    r_surf = old_arge['surface_data_r']
    pn = arge['poincare_number']
    phi0 = arge['poincare_phi0']
    phi = int(phi0/2/np.pi * arge['number_zeta'])
    r_surf = r_surf[phi]
    surf_r = (r_surf[:, 0]**2 + r_surf[:, 1]**2)**0.5
    surf_z = r_surf[:, 2]
    r0 = surf_r[0]
    mid = int(arge['number_theta'] / 2)
    rmid = (surf_r[mid] + r0) / 2
    dr = (r0 - rmid) / (pn-1)
    r0 = [rmid+i*dr for i in range(pn)]
    z0 = [0 for i in range(pn)]
    print(r0)
    ### new_coil
    coil = np.mean(arge['coil_r'], axis = (1,2))
    x = coil[:, :, 0]
    y = coil[:, :, 1]
    z = coil[:, :, 2]
    I = arge['coil_I'] 
    name = group = np.ones((arge['number_coils']))
    coil_py = coilpy.coils.Coil(x, y, z, I, name, group)
    bfield = coil_py.bfield
    arge['number_step'] = 10
    line = coilpy.misc.tracing(bfield, r0, z0, arge['poincare_phi0'], 
            arge['number_iter'], arge['number_field_periods'], arge['number_step'], )
    line = np.reshape(line, (pn*(arge['number_iter']+1), 2))
    ### old_coil
    old_coil = np.mean(old_arge['coil_r'], axis = (1,2))
    x = old_coil[:, :, 0]
    y = old_coil[:, :, 1]
    z = old_coil[:, :, 2]
    I = old_arge['coil_I'] 
    name = group = np.ones((arge['number_coils']))
    old_coil_py = coilpy.coils.Coil(x, y, z, I, name, group)
    old_bfield = old_coil_py.bfield
    old_line = coilpy.misc.tracing(old_bfield, r0, z0, arge['poincare_phi0'], 
            arge['number_iter'], arge['number_field_periods'], arge['number_step'], )
    old_line = np.reshape(old_line, (pn*(arge['number_iter']+1), 2))


    
    fig = go.Figure()
    fig.add_scatter(x = line[:, 0], y = line[:, 1],  name='new_poincare', mode='markers', marker_size = 3)
    fig.add_scatter(x = old_line[:, 0], y = old_line[:, 1],  name='old_poincare', mode='markers', marker_size = 3)
    fig.add_scatter(x = surf_r, y = surf_z,  name='surface', line = dict(width=2.5))
    fig.update_layout(scene_aspectmode='data')
    fig.show()

    return

filename = 'results/w7x/w7x.h5'
oldline = 'results/poincare/w7x.npy'
arge = read_hdf5(filename)
arge['poincare_number'] = 20
arge['number_iter'] = 200
plot_poincare(arge)

# compare
# oldfile = 'results/poincare/hsx.npy'
# old_arge = read_hdf5(oldfile)
# plot_poincare_compare(arge,old_arge)


