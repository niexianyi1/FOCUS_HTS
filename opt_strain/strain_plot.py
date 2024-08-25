
import plotly.graph_objects as go
import jax.numpy as np
import h5py
from strain_coilset import Strain_CoilSet
import sys
sys.path.append('opt_coil')
import spline


def read_hdf5(filename):
    f = h5py.File(filename, "r")
    arge = {}
    for key in list(f.keys()):
        val = f[key][()]
        if key == 'num_fourier_coils':
            key = 'number_fourier_coils'
        if isinstance(val, bytes):
            val = str(val, encoding='utf-8')
        arge.update({key: val})
    f.close()
    return arge


def plot_strain_compare(filename, coilfile):
    arge = read_hdf5(filename)
    oldcoil_arge = read_hdf5(coilfile) 
    arge['length_normal'] = [0.001 for i in range(4)]
    oldcoil_arge['length_normal'] = [0.001 for i in range(4)]

    if arge['coil_case'] != 'fourier':
        arge['coil_arg'] = arge['coil_arg'][:, :, :-3]
        bc, tj = spline.get_bc_init(arge['number_points'], arge['number_control_points'])
        arge['bc'] = bc
        arge['tj'] = tj

    if oldcoil_arge['coil_case'] != 'fourier':
        oldcoil_arge['coil_arg'] = oldcoil_arge['coil_arg'][:, :, :-3]
        bc, tj = spline.get_bc_init(arge['number_points'], oldcoil_arge['number_control_points'])
        oldcoil_arge['bc'] = bc
        oldcoil_arge['tj'] = tj

    ns = arge['number_segments']
    arge['number_segments'] = arge['number_points'] 
    coil_cal = Strain_CoilSet(arge)  
    params = (arge['coil_arg'], arge['coil_fr'], arge['coil_I'])
    coil = coil_cal.get_coil(params)
    strain = coil_cal.get_plot_strain(params)

    oldcoil_arge['number_segments'] = arge['number_points'] 
    old_coil_cal = Strain_CoilSet(oldcoil_arge)  
    oldparams = (oldcoil_arge['coil_arg'], oldcoil_arge['coil_fr'], oldcoil_arge['coil_I'])
    oldcoil = old_coil_cal.get_coil(oldparams)
    old_strain = old_coil_cal.get_plot_strain(oldparams)

    arge['number_segments'] = ns
    nic = arge['number_independent_coils'] 
    ns = arge['number_points']
    nn = arge['number_normal']
    nb = arge['number_binormal']

    rr = np.zeros((nic, 5, ns+1, 3))
    rr = rr.at[:,0,:ns,:].set(coil[:nic, 0, 0, :, :])
    rr = rr.at[:,1,:ns,:].set(coil[:nic, 0, nb-1, :, :])
    rr = rr.at[:,2,:ns,:].set(coil[:nic, nn-1, nb-1, :, :])
    rr = rr.at[:,3,:ns,:].set(coil[:nic, nn-1, 0, :, :])
    rr = rr.at[:,4,:ns,:].set(coil[:nic, 0, 0, :, :])
    rr = rr.at[:,0,-1,:].set(coil[:nic, 0, 0, 0, :])
    rr = rr.at[:,1,-1,:].set(coil[:nic, 0, nb-1, 0, :])
    rr = rr.at[:,2,-1,:].set(coil[:nic, nn-1, nb-1, 0, :])
    rr = rr.at[:,3,-1,:].set(coil[:nic, nn-1, 0, 0, :])
    rr = rr.at[:,4,-1,:].set(coil[:nic, 0, 0, 0, :])
    xx = rr[:,:,:,0]
    yy = rr[:,:,:,1]
    zz = rr[:,:,:,2]
    smax, smin = float(np.max(strain)), float(np.min(strain))
    print('max_strain = ', smax, smin)
    s = np.zeros((nic, 5, ns+1))
    for i in range(5):
        s = s.at[:, i, :-1].set(strain)
        s = s.at[:, i, -1].set(strain[:, 0])

    orr = np.zeros((nic, 5, ns+1, 3))
    orr = orr.at[:,0,:ns,:].set(oldcoil[:nic, 0, 0, :, :])
    orr = orr.at[:,1,:ns,:].set(oldcoil[:nic, 0, nb-1, :, :])
    orr = orr.at[:,2,:ns,:].set(oldcoil[:nic, nn-1, nb-1, :, :])
    orr = orr.at[:,3,:ns,:].set(oldcoil[:nic, nn-1, 0, :, :])
    orr = orr.at[:,4,:ns,:].set(oldcoil[:nic, 0, 0, :, :])
    orr = orr.at[:,0,-1,:].set(oldcoil[:nic, 0, 0, 0, :])
    orr = orr.at[:,1,-1,:].set(oldcoil[:nic, 0, nb-1, 0, :])
    orr = orr.at[:,2,-1,:].set(oldcoil[:nic, nn-1, nb-1, 0, :])
    orr = orr.at[:,3,-1,:].set(oldcoil[:nic, nn-1, 0, 0, :])
    orr = orr.at[:,4,-1,:].set(oldcoil[:nic, 0, 0, 0, :])
    oxx = orr[:,:,:,0]
    oyy = orr[:,:,:,1]
    ozz = orr[:,:,:,2]
    osmax, osmin = float(np.max(old_strain)), float(np.min(old_strain))
    print('old_max_strain = ', osmax, osmin)
    os = np.zeros((nic, 5, ns+1))
    for i in range(5):
        os = os.at[:, i, :-1].set(old_strain)
        os = os.at[:, i, -1].set(old_strain[:, 0])

    fig = go.Figure()
    for i in range(nic):
        fig.add_trace(go.Surface(x=oxx[i,:,:], y=oyy[i,:,:], z=ozz[i,:,:], 
            surfacecolor = os[i,:,:], cmax = osmax, cmin = osmin, colorbar_title='old_strain', 
            colorbar = dict(x = 0.2,tickfont = dict(size=20))  ))
    for i in range(nic):
        fig.add_trace(go.Surface(x=xx[i,:,:], y=yy[i,:,:], z=zz[i,:,:], 
            surfacecolor = s[i,:,:], cmax = smax, cmin = smin, colorbar_title='strain', 
            colorbar = dict(x = 0.7,tickfont = dict(size=20)) ,colorscale="Viridis"))

    fig.update_layout(coloraxis_showscale=True)
    fig.update_layout(scene_aspectmode='data',  scene = dict(
        xaxis = dict(#  backgroundcolor="white",  gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white",),
        yaxis = dict(# backgroundcolor="white", gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white"),
        zaxis = dict(# backgroundcolor="white", gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white",),))
    fig.show() 
    return
