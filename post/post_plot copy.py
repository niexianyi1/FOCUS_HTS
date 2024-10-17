


import plotly.graph_objects as go
import jax.numpy as np
import h5py
import coilpy
import post_coilset
import sys
sys.path.append('opt_coil')
import spline
import fourier
import lossfunction
import read_file
sys.path.append('HTS')
import hts_strain




def plot(p):
    if p['plot_coil'] != 0 :
        plot_coil(p)
    if p['plot_loss'] != 0 :
        plot_loss(p)
    if p['plot_poincare'] != 0 :
        plot_poincare(p)
    return


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


def plot_loss(p):
    arge = read_hdf5(p['filename'])   
    lossvals = arge['loss_vals']
    fig = go.Figure()
    fig.add_scatter(x = np.arange(0, len(lossvals), 1), y = lossvals, 
                        name = 'lossvalue', line = dict(width=5))
    fig.update_xaxes(title_text = "opt_coil",title_font = {"size": 25},title_standoff = 12, 
                        tickfont = dict(size=25))
    fig.update_yaxes(title_text = "lossvalue",title_font = {"size": 25},title_standoff = 12, 
                        tickfont = dict(size=25) ,type="log", exponentformat = 'e')
    fig.show()
    return


def plot_poincare(p):
    arge = read_hdf5(p['filename'])   
    r_surf,_ ,_ = arge['surface_data_r']
    pn = p['poincare_number']
    phi0 = p['poincare_phi0']
    phi = int(phi0/2/np.pi * arge['number_zeta'])
    r_surf = r_surf[phi]
    r0 = (r_surf[0,0]**2 + r_surf[0,1]**2)**0.5
    mid = int(arge['number_theta'] / 2)
    rmid = (r_surf[mid,0]**2 + r_surf[mid,1]**2)**0.5
    dr = (r0 - rmid) / (pn-1)
    r0 = [rmid+i*dr for i in range(pn)]
    z0 = [0 for i in range(pn)]

    coil = np.mean(arge['coil_r'], axis = (1,2))
    x = coil[:, :, 0]
    y = coil[:, :, 1]
    z = coil[:, :, 2]
    I = arge['coil_I'] 
    name = group = np.ones((arge['number_coils']))
    coil_py = coilpy.coils.Coil(x, y, z, I, name, group)
    bfield = coil_py.bfield
    line = coilpy.misc.tracing(bfield, r0, z0, phi0, 
            arge['number_iter'], arge['number_field_periods'], arge['number_step'])
    line = np.reshape(line, (pn*(arge['number_iter']+1), 2))

    surf = (r_surf[:,0]**2 + r_surf[:,1]**2)**0.5
    fig = go.Figure()
    fig.add_scatter(x = line[:, 0], y = line[:, 1],  name='poincare', mode='markers', marker_size = 1.5)
    fig.add_scatter(x = surf, y = r_surf[:, 2],  name='surface', line = dict(width=2.5))
    fig.update_layout(scene_aspectmode='data')
    fig.show() 
    return


def plot_coil(p):
    arge = read_hdf5(p['filename'])   
    if arge['coil_case'] != 'fourier':
        arge['coil_arg'] = arge['coil_arg'][:, :, :-3]
        bc, tj = spline.get_bc_init(arge['number_points'], arge['number_control_points'])
        arge['bc'] = bc
        arge['tj'] = tj

    ns = arge['number_segments']
    arge['number_segments'] = p['number_points'] 
    coil_cal = post_coilset.CoilSet(arge)  
    params = (arge['coil_arg'], arge['coil_fr'], arge['coil_I'])
    coil = coil_cal.get_coil(params)

    if p['coil_color'] == 'B':
        _, B_coil = coil_cal.get_plot_args(params)
        argu = np.linalg.norm(B_coil, axis=-1)
        Bmax, Bmin = float(np.max(argu)), float(np.min(argu))
        print('maxB = ', Bmax, 'T', 'minB = ', Bmin, 'T')

    elif p['coil_color'] == 'strain':
        strain = coil_cal.get_plot_strain(params)
        argu = strain
        smax, smin = float(np.max(argu)), float(np.min(argu))
        print('max_strain = ', smax*100, '%', 'min_strain = ', smin*100, '%')

    elif p['coil_color'] == 'force/length':
        force = coil_cal.get_plot_force(params)
        argu = np.linalg.norm(force, axis=-1)
        fmax, fmin = float(np.max(argu)), float(np.min(argu))
        print('max_force = ', fmax, 'min_force = ', fmin)


    nic = arge['number_independent_coils'] 
    ns = arge['number_points']
    nn = arge['number_normal']
    nb = arge['number_binormal']


    

    if p['filament or finite_build'] == 1:
        xx, yy, zz, arg = finite_build(coil, nic, ns, nn, nb, argu)
    
    if p['compare'] == 1:
        oldcoil_arge = read_hdf5(p['compare_file']) 
        if oldcoil_arge['coil_case'] != 'fourier':
            oldcoil_arge['coil_arg'] = oldcoil_arge['coil_arg'][:, :, :-3]
            bc, tj = spline.get_bc_init(arge['number_points'], oldcoil_arge['number_control_points'])
            oldcoil_arge['bc'] = bc
            oldcoil_arge['tj'] = tj

        oldcoil_arge['number_segments'] = p['number_points'] 
        old_coil_cal = post_coilset.CoilSet(oldcoil_arge)  
        oldparams = (oldcoil_arge['coil_arg'], oldcoil_arge['coil_fr'], oldcoil_arge['coil_I'])
        oldcoil = old_coil_cal.get_coil(oldparams)
        _, old_B_coil = old_coil_cal.get_plot_args(oldparams)
        orr = np.zeros((nic, 5, ns+1, 3))

        onn = arge['number_normal']
        onb = arge['number_binormal']

        oB_coil = np.linalg.norm(old_B_coil, axis=-1)
        oBmax, oBmin = float(np.max(oB_coil)), float(np.min(oB_coil))
        print('maxB = ', oBmax, oBmin)

        if p['filament or finite_build'] == 1:
            oxx, oyy, ozz, oarg = finite_build(oldcoil, nic, ns, onn, onb, oargu)


    fig = go.Figure()
    for i in range(nic):
        fig.add_trace(go.Surface(x=oxx[i,:,:], y=oyy[i,:,:], z=ozz[i,:,:], 
            surfacecolor = oargu[i,:,:], cmax = oBmax, cmin = oBmin, colorbar_title='old_B_coil [T]', 
            colorbar = dict(x = 0.2,tickfont = dict(size=20))  ))
    for i in range(nic):
        fig.add_trace(go.Surface(x=xx[i,:,:], y=yy[i,:,:], z=zz[i,:,:], 
            surfacecolor = argu[i,:,:], cmax = Bmax, cmin = Bmin, colorbar_title='B_coil [T]', 
            colorbar = dict(x = 0.7,tickfont = dict(size=20)) ,colorscale="Viridis"))
    # for i in range(nic):
    #     fig.add_trace(go.Surface(x=oxx[i,:,:], y=oyy[i,:,:], z=ozz[i,:,:], 
    #         surfacecolor = old_color[i],cmax = 1, cmin = 0,))
    # for i in range(nic):
    #     fig.add_trace(go.Surface(x=xx[i,:,:], y=yy[i,:,:], z=zz[i,:,:], 
    #         surfacecolor = color[i],cmax = 1, cmin = 0,))
    fig.update_layout(coloraxis_showscale=True)
    fig.update_layout(scene_aspectmode='data',  scene = dict(
        xaxis = dict(#  backgroundcolor="white",  gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white",),
        yaxis = dict(# backgroundcolor="white", gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white"),
        zaxis = dict(# backgroundcolor="white", gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white",),))
    fig.show() 

    pass


def get_argu(p, nic, ns):
    arge = read_hdf5(p['filename'])   
    if arge['coil_case'] != 'fourier':
        arge['coil_arg'] = arge['coil_arg'][:, :, :-3]
        bc, tj = spline.get_bc_init(arge['number_points'], arge['number_control_points'])
        arge['bc'] = bc
        arge['tj'] = tj

    arge['number_segments'] = p['number_points'] 
    coil_cal = post_coilset.CoilSet(arge)  
    params = (arge['coil_arg'], arge['coil_fr'], arge['coil_I'])

    if p['coil_color'] == 'B':
        _, B_coil = coil_cal.get_plot_args(params)
        argu = np.linalg.norm(B_coil, axis=-1)
        max, min = float(np.max(argu)), float(np.min(argu))
        print('maxB = ', max, 'T', 'minB = ', min, 'T')

    elif p['coil_color'] == 'strain':
        strain = coil_cal.get_plot_strain(params)
        argu = strain
        max, min = float(np.max(argu)), float(np.min(argu))
        print('max_strain = ', max*100, '%', 'min_strain = ', min*100, '%')

    elif p['coil_color'] == 'force/length':
        force = coil_cal.get_plot_force(params)
        argu = np.linalg.norm(force, axis=-1)
        max, min = float(np.max(argu)), float(np.min(argu))
        print('max_force = ', max, 'N/m', 'min_force = ', min, 'N/m')

    elif p['coil_color'] == 'contrast':
        color = np.ones((nic, 5, ns+1))
        old_color = np.zeros((nic, 5, ns+1))
        argu = color.at[:,:,0].set(0.99)
        oargu = old_color.at[:,:,0].set(0.01)
        return argu, oargu

    return argu, max, min

def finite_build(coil, nic, ns, nn, nb, argument):
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

    arg = np.zeros((nic, 5, ns+1))
    arg = arg.at[:, :-1, :-1].set(argument)
    arg = arg.at[:, :-1, -1].set(argument[:, :, 0])
    arg = arg.at[:, -1].set(arg[:, 0, :])
    return xx, yy, zz, arg




p = {
'plot_coil':                1, 
'plot_loss':                0,          
'plot_poincare':            0,   
'filename'      :   'results/paper/QA/opt_1.h5',    # only 'h5'

'compare'       :   0,          # 0:no, 1:yes
'compare_file'  :   'results/paper/QA/opt_2.h5',

'filament or finite_build': 1,  # 0:filament, 1:finite_build
'coil_color'      :   'B',                 
# the coils color: 'default', 'B', 'strain', 'force/length', 'contrast'
'number_points'  :   500,

'poincare_number':          25,         #       int,                                        
'poincare_phi0':            0,          # (phi0)float, 
'number_iter':              400,        #       int,  
'number_step':              1,          #       int,  
}

plot(p)




