


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


def plot_loss(lossfile):

    lossvals = np.load('{}'.format(lossfile))
    fig = go.Figure()
    fig.add_scatter(x = np.arange(0, len(lossvals), 1), y = lossvals, 
                        name = 'lossvalue', line = dict(width=5))
    fig.update_xaxes(title_text = "opt_coil",title_font = {"size": 25},title_standoff = 12, 
                        tickfont = dict(size=25))
    fig.update_yaxes(title_text = "lossvalue",title_font = {"size": 25},title_standoff = 12, 
                        tickfont = dict(size=25) ,type="log", exponentformat = 'e')
    fig.show()
    return

def plot_surface_0(filename):
    arge = read_hdf5(filename)
    rs = arge['surface_data_r'][0]
    fig = go.Figure()
    fig.add_scatter3d(x=rs[:, 0],y=rs[:, 1],z=rs[:, 2], name='surface0', mode='markers', marker_size = 1.5)   
    fig.update_layout(scene_aspectmode='data')
    fig.show()
    return

def plot_surface_all(filename):
    arge = read_hdf5(filename)
    rs = arge['surface_data_r']
    rs = np.reshape(rs, (arge['number_zeta']*arge['number_theta'],3))
    fig = go.Figure()
    fig.add_scatter3d(x=rs[:, 0],y=rs[:, 1],z=rs[:, 2], name='surface0', mode='markers', marker_size = 1.5)   
    fig.update_layout(scene_aspectmode='data')
    fig.show()
    return


def plot_coil_0(filename):
    arge = read_hdf5(filename)
    nic = arge['number_independent_coils']  
    nzs = int(arge['number_zeta']/arge['number_field_periods']/(arge['stellarator_symmetry']+1))  
    rs = arge['surface_data_r']
    rs = rs[:nzs]
    r_surf = np.zeros((nzs, arge['number_theta']+1, 3))
    r_surf = r_surf.at[:,:-1,:].set(rs)
    r_surf = r_surf.at[:,-1,:].set(rs[:,0,:])


    if arge['coil_case'] != 'fourier':
        arge['coil_arg'] = arge['coil_arg'][:, :, :-3]
        bc, tj = spline.get_bc_init(arge['number_points'], arge['number_control_points'])
        arge['bc'] = bc
        arge['tj'] = tj

    ns = arge['number_segments']

    arge['number_segments'] = arge['number_points'] 
    params = (arge['coil_arg'], arge['coil_fr'], arge['coil_I'])
    coil_cal = post_coilset.CoilSet(arge)    
    coil = coil_cal.get_coil(params)
    arge['number_segments'] = ns

    ns = arge['number_points']

    coil = np.mean(coil, axis = (1,2))
    coil = coil[0]

    fig = go.Figure()
    fig.add_scatter3d(x=coil[:, 0],y=coil[:, 1],z=coil[:, 2], name='newcoil', mode='markers', marker_size = 3)   
    # fig.add_trace(go.Surface(x=r_surf[:,:,0], y=r_surf[:,:,1], z=r_surf[:,:,2]))
    fig.update_layout(scene_aspectmode='data',  scene = dict(
        xaxis = dict(#  backgroundcolor="white",  gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white",),
        yaxis = dict(# backgroundcolor="white", gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white"),
        zaxis = dict(# backgroundcolor="white", gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white",),))
    fig.show()
    return




def plot_coil(filename):
    arge = read_hdf5(filename)
    nic = arge['number_independent_coils']  
    nzs = int(arge['number_zeta']/arge['number_field_periods']/(arge['stellarator_symmetry']+1))  
    rs = arge['surface_data_r']
    rs = rs[:nzs]
    r_surf = np.zeros((nzs, arge['number_theta']+1, 3))
    r_surf = r_surf.at[:,:-1,:].set(rs)
    r_surf = r_surf.at[:,-1,:].set(rs[:,0,:])
    lossB = arge['loss_B']
    lossB = np.linalg.norm(lossB[:nzs], axis=-1)
    B = np.zeros((nzs, arge['number_theta']+1))
    B = B.at[:,:-1].set(lossB)
    B = B.at[:,-1].set(lossB[:,0])


    if arge['coil_case'] != 'fourier':
        arge['coil_arg'] = arge['coil_arg'][:, :, :-3]
        bc, tj = spline.get_bc_init(arge['number_points'], arge['number_control_points'])
        arge['bc'] = bc
        arge['tj'] = tj

    ns = arge['number_segments']

    arge['number_segments'] = arge['number_points'] 
    params = (arge['coil_arg'], arge['coil_fr'], arge['coil_I'])
    coil_cal = post_coilset.CoilSet(arge)    
    coil = coil_cal.get_coil(params)
    arge['number_segments'] = ns

    ns = arge['number_points']

    coil = np.mean(coil, axis = (1,2))
    coil = np.reshape(coil[:nic], (ns * nic, 3))

    fig = go.Figure()
    fig.add_scatter3d(x=coil[:, 0],y=coil[:, 1],z=coil[:, 2], name='newcoil', mode='markers', marker_size = 1.5)   
    fig.add_trace(go.Surface(x=r_surf[:,:,0], y=r_surf[:,:,1], z=r_surf[:,:,2],
            surfacecolor = B, colorbar_title='B_coil [T]', 
            colorbar = dict(x = 0.8,tickfont = dict(size=20)),colorscale="Viridis" ))
    fig.update_layout(scene_aspectmode='data',  scene = dict(
        xaxis = dict(#  backgroundcolor="white",  gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white",),
        yaxis = dict(# backgroundcolor="white", gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white"),
        zaxis = dict(# backgroundcolor="white", gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white",),))
    fig.show()
    return


def plot_dcs(filename):
    arge = read_hdf5(filename)
    nic = arge['number_independent_coils']  
    nzs = int(arge['number_zeta']/arge['number_field_periods']/(arge['stellarator_symmetry']+1))  
    rs = arge['surface_data_r']
    rs = rs[:nzs]
    r_surf = np.zeros((nzs, arge['number_theta']+1, 3))
    r_surf = r_surf.at[:,:-1,:].set(rs)
    r_surf = r_surf.at[:,-1,:].set(rs[:,0,:])
    lossB = arge['loss_B']
    lossB = np.linalg.norm(lossB[:nzs], axis=-1)
    B = np.zeros((nzs, arge['number_theta']+1))
    B = B.at[:,:-1].set(lossB)
    B = B.at[:,-1].set(lossB[:,0])



    if arge['coil_case'] != 'fourier':
        arge['coil_arg'] = arge['coil_arg'][:, :, :-3]
        bc, tj = spline.get_bc_init(arge['number_points'], arge['number_control_points'])
        arge['bc'] = bc
        arge['tj'] = tj

    ns = arge['number_segments']

    arge['number_segments'] = arge['number_points'] 
    params = (arge['coil_arg'], arge['coil_fr'], arge['coil_I'])
    coil_cal = post_coilset.CoilSet(arge)    
    coil = coil_cal.get_coil(params)
    arge['number_segments'] = ns

    ns = arge['number_points']

    coil = np.mean(coil, axis = (1,2))

    dr = (coil[:nic, :, np.newaxis, np.newaxis, :] - rs[np.newaxis, np.newaxis, :, :, :])
    dcs = np.min(np.linalg.norm(dr, axis = -1), axis=(2, 3))
    print(dcs.shape)
    dcs = np.reshape(dcs, (ns * nic))
    coil = np.reshape(coil[:nic], (ns * nic, 3))

    fig = go.Figure()
    fig.add_scatter3d(x=coil[:, 0],y=coil[:, 1],z=coil[:, 2], name='newcoil', 
                mode='markers', marker = dict(size=1.5, color=dcs))   
    fig.add_trace(go.Surface(x=r_surf[:,:,0], y=r_surf[:,:,1], z=r_surf[:,:,2],
            surfacecolor = B, colorbar_title='B_coil [T]', 
            colorbar = dict(x = 0.8,tickfont = dict(size=20)),colorscale="Viridis" ))
    fig.update_layout(scene_aspectmode='data',  scene = dict(
        xaxis = dict(#  backgroundcolor="white",  gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white",),
        yaxis = dict(# backgroundcolor="white", gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white"),
        zaxis = dict(# backgroundcolor="white", gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white",),))
    fig.show()
    return






def plot_segment(filename):
    arge = read_hdf5(filename)
    coil = arge['coil_centroid']
    nic = arge['number_independent_coils'] 
    ns = int(np.floor(arge['number_segments'] / 10))
    fig = go.Figure()
    for i in range(nic):
        for j in range(ns):
            fig.add_scatter3d(x=coil[i, j*10:(j+1)*10, 0],y=coil[i, j*10:(j+1)*10, 1],
            z=coil[i, j*10:(j+1)*10, 2], name='coil_{}_{}'.format(i,j), mode='markers', marker_size = 1.5)   
        fig.add_scatter3d(x=coil[i, (j+1)*10:, 0],y=coil[i, (j+1)*10:, 1],
            z=coil[i, (j+1)*10:, 2], name='coil_{}_{}'.format(i,j+1), mode='markers', marker_size = 1.5)   
    fig.update_layout(scene_aspectmode='data')
    fig.show()
    return


def plot_alpha(filename):
    arge = read_hdf5(filename)
    alpha = arge['coil_alpha']
    fig = go.Figure()
    for i in range(5):
        fig.add_scatter(x = np.arange(0, 64, 1), y = alpha[i], 
                        name = 'alpha{}'.format(i), line = dict(width=5))
    fig.update_xaxes(title_text = "number_segment",title_font = {"size": 25},title_standoff = 12, 
                        tickfont = dict(size=25))
    fig.update_yaxes(title_text = "alpha",title_font = {"size": 25},title_standoff = 12, 
                        tickfont = dict(size=25) )#,type="log", exponentformat = 'e')
    fig.show()
    return


def plot_coil_compare(filename, coilfile):    # 线圈
    arge = read_hdf5(filename)
    oldcoil_arge = read_hdf5(coilfile) 
    nic = arge['number_independent_coils']  
    nzs = int(arge['number_zeta']/arge['number_field_periods']/(arge['stellarator_symmetry']+1))+1
    rs = arge['surface_data_r']
    rs = rs[:nzs]
    r_surf = np.zeros((nzs, arge['number_theta']+1, 3))
    r_surf = r_surf.at[:,:-1,:].set(rs)
    r_surf = r_surf.at[:,-1,:].set(rs[:,0,:])
    lossB = arge['loss_B']
    lossB = np.linalg.norm(lossB[:nzs], axis=-1)
    B = np.zeros((nzs, arge['number_theta']+1))
    B = B.at[:,:-1].set(lossB)
    B = B.at[:,-1].set(lossB[:,0])

    if arge['coil_case'] != 'fourier':
        arge['coil_arg'] = arge['coil_arg'][:, :, :-3]
        bc, tj = spline.get_bc_init(arge['number_points'], arge['number_control_points'])
        arge['bc'] = bc
        arge['tj'] = tj

    if oldcoil_arge['coil_case'] != 'fourier':
        oldcoil_arge['coil_arg'] = oldcoil_arge['coil_arg'][:, :, :-3]
        bc, tj = spline.get_bc_init(oldcoil_arge['number_points'], oldcoil_arge['number_control_points'])
        oldcoil_arge['bc'] = bc
        oldcoil_arge['tj'] = tj

    # new coil  
    ns = arge['number_segments']
    arge['number_segments'] = arge['number_points']    
    params = (arge['coil_arg'], arge['coil_fr'], arge['coil_I'])
    coil_cal = post_coilset.CoilSet(arge)    
    coil = coil_cal.get_coil(params)
    arge['number_segments'] = ns
    # old coil 
    oldcoil_arge['number_segments'] = arge['number_points'] 
    oldparams = (oldcoil_arge['coil_arg'], oldcoil_arge['coil_fr'], oldcoil_arge['coil_I'])
    coil_cal = post_coilset.CoilSet(oldcoil_arge)    
    oldcoil = coil_cal.get_coil(oldparams)

    coil = np.mean(coil, axis = (1,2))
    coil = np.reshape(coil[:nic], (arge['number_points'] * nic, 3))
    oldcoil = np.mean(oldcoil, axis = (1,2))
    oldcoil = np.reshape(oldcoil[:nic], (arge['number_points'] * nic, 3))

    fig = go.Figure()
    fig.add_scatter3d(x=coil[:, 0],y=coil[:, 1],z=coil[:, 2], name='new_coil', mode='markers', marker_size = 1.5)   
    fig.add_scatter3d(x=oldcoil[:, 0],y=oldcoil[:, 1],z=oldcoil[:, 2], name='old_coil', mode='markers', marker_size = 1.5)   
    fig.add_trace(go.Surface(x=r_surf[:,:,0], y=r_surf[:,:,1], z=r_surf[:,:,2],
            surfacecolor = B, colorbar_title='B_coil [T]', 
            colorbar = dict(x = 0.8,tickfont = dict(size=20)),colorscale="Viridis" ))
    fig.update_layout(scene_aspectmode='data',  scene = dict(
        xaxis = dict(#  backgroundcolor="white",  gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white",),
        yaxis = dict(# backgroundcolor="white", gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white"),
        zaxis = dict(# backgroundcolor="white", gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white",),))
    fig.show()
    # fig.write_image("test.png")
    # fig.write_html("test.html")
    return 



def plot_strain(filename):
    arge = read_hdf5(filename)
    nic = arge['number_independent_coils']
    arge['length_normal'] = [0.001 for i in range(nic)]
    if arge['coil_case'] == 'spline':
        arge['coil_arg'] = arge['coil_arg'][:, :, :-3]
        bc, tj = spline.get_bc_init(arge['number_points'], arge['number_control_points'])
        arge['bc'] = bc
        arge['tj'] = tj

    ns = arge['number_segments']
    arge['number_segments'] = arge['number_points'] 
    coil_cal = post_coilset.CoilSet(arge)  
    params = (arge['coil_arg'], arge['coil_fr'], arge['coil_I'])
    coil = coil_cal.get_coil(params)
    strain = coil_cal.get_plot_strain(params)
    
    arge['number_segments'] = ns
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
    print(np.max(strain, axis=1))
    s = np.zeros((nic, 5, ns+1))
    for i in range(5):
        s = s.at[:, i, :-1].set(strain)
        s = s.at[:, i, -1].set(strain[:, 0])
        
    fig = go.Figure()
    for i in range(nic):
        fig.add_trace(go.Surface(x=xx[i,:,:], y=yy[i,:,:], z=zz[i,:,:], 
            surfacecolor = s[i,:,:], cmax = smax, cmin = smin, colorbar_title='strain', 
            colorbar = dict(x = 0.8,tickfont = dict(size=20)),colorscale="plasma"))
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
    coil_cal = post_coilset.CoilSet(arge)  
    params = (arge['coil_arg'], arge['coil_fr'], arge['coil_I'])
    coil = coil_cal.get_coil(params)
    strain = coil_cal.get_plot_strain(params)

    oldcoil_arge['number_segments'] = arge['number_points'] 
    old_coil_cal = post_coilset.CoilSet(oldcoil_arge)  
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
            colorbar = dict(x = 0.2,tickfont = dict(size=30))  ))
    for i in range(nic):
        fig.add_trace(go.Surface(x=xx[i,:,:], y=yy[i,:,:], z=zz[i,:,:], 
            surfacecolor = s[i,:,:], cmax = smax, cmin = smin, colorbar_title='strain', 
            colorbar = dict(x = 0.7,tickfont = dict(size=30)) ,colorscale="Viridis"))

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


def plot_force(filename):
    arge = read_hdf5(filename)
    nic = arge['number_independent_coils']
    if arge['coil_case'] == 'spline':
        arge['coil_arg'] = arge['coil_arg'][:, :, :-3]
        bc, tj = spline.get_bc_init(arge['number_points'], arge['number_control_points'])
        arge['bc'] = bc
        arge['tj'] = tj

    ns = arge['number_segments']
    arge['number_segments'] = arge['number_points'] 
    coil_cal = post_coilset.CoilSet(arge)  
    params = (arge['coil_arg'], arge['coil_fr'], arge['coil_I'])
    coil = coil_cal.get_coil(params)
    force = coil_cal.get_plot_force(params)
    force = np.linalg.norm(force, axis=-1)
    
    arge['number_segments'] = ns
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
    fmax, fmin = float(np.max(force)), float(np.min(force))
    print('max_force = ', fmax, fmin)
    f = np.zeros((nic, 5, ns+1))
    for i in range(5):
        f = f.at[:, i, :-1].set(force)
        f = f.at[:, i, -1].set(force[:, 0])
        
    fig = go.Figure()
    for i in range(nic):
        fig.add_trace(go.Surface(x=xx[i,:,:], y=yy[i,:,:], z=zz[i,:,:], 
            surfacecolor = f[i,:,:], cmax = fmax, cmin = fmin, colorbar_title='force [N/m]', 
            colorbar = dict(x = 0.8,tickfont = dict(size=20)),colorscale="plasma"))
    fig.update_layout(coloraxis_showscale=True)
    fig.update_layout(scene_aspectmode='data',  scene = dict(
        xaxis = dict(#  backgroundcolor="white",  gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white",),
        yaxis = dict(# backgroundcolor="white", gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white"),
        zaxis = dict(# backgroundcolor="white", gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white",),))
    fig.show() 
    fig.write_image('results/paper/w7x/force_opt_2.pdf')
    return 




def plot_surface(filename):
    arge = read_hdf5(filename)
    nz = arge['number_zeta']
    nzs = int(arge['number_zeta']/arge['number_field_periods']/(arge['stellarator_symmetry']+1)) +1 
    rs = arge['surface_data_r']
    rs = rs[:nzs]
    r_surf = np.zeros((nzs, arge['number_theta']+1, 3))

    r_surf = r_surf.at[:,:-1,:].set(rs)
    r_surf = r_surf.at[:,-1,:].set(rs[:,0,:])
    lossB = arge['loss_B'][:nzs]
    lossB = np.linalg.norm(lossB[:nzs], axis=-1)
    Bs = np.zeros((nzs, arge['number_theta']+1))

    Bs = Bs.at[:,:-1].set(lossB)
    Bs = Bs.at[:,-1].set(lossB[:,0])

    if arge['coil_case'] == 'spline':
        arge['coil_arg'] = arge['coil_arg'][:, :, :-3]
        bc, tj = spline.get_bc_init(arge['number_points'], arge['number_control_points'])
        arge['bc'] = bc
        arge['tj'] = tj

    ns = arge['number_segments']
    arge['number_segments'] = arge['number_points'] 
    coil_cal = post_coilset.CoilSet(arge)  
    params = (arge['coil_arg'], arge['coil_fr'], arge['coil_I'])
    coil = coil_cal.get_coil(params)
    _, B_coil = coil_cal.get_plot_args(params)
    
    arge['number_segments'] = ns

    ns = arge['number_points']
    nn = arge['number_normal']
    nb = arge['number_binormal']
    nic = arge['number_independent_coils']

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
    B_coil = np.linalg.norm(B_coil, axis=-1)
    Bmax, Bmin = float(np.max(B_coil)), float(np.min(B_coil))
    print('maxB = ', Bmax, Bmin)
    B = np.zeros((nic, 5, ns+1))
    B = B.at[:, :-1, :-1].set(B_coil)
    B = B.at[:, :-1, -1].set(B_coil[:, :, 0])
    B = B.at[:, -1].set(B[:, 0, :])

    fig = go.Figure()
    fig.add_trace(go.Surface(x=r_surf[:,:,0], y=r_surf[:,:,1], z=r_surf[:,:,2],
            surfacecolor = Bs, colorbar_title='B_surf [T]', 
            colorbar = dict(x = 0.8,tickfont = dict(size=40)),colorscale="Viridis" ))
    for i in range(nic):
        fig.add_trace(go.Surface(x=xx[i,:,:], y=yy[i,:,:], z=zz[i,:,:], 
            surfacecolor = B[i,:,:], cmax = Bmax, cmin = Bmin, colorbar_title='B_coil [T]', 
            colorbar = dict(x = 0.1,tickfont = dict(size=40)),colorscale="plasma"))
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


def plot_surface_compare(filename, coilfile):
    arge = read_hdf5(filename)
    oldcoil_arge = read_hdf5(coilfile) 
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
    coil_cal = post_coilset.CoilSet(arge)  
    params = (arge['coil_arg'], arge['coil_fr'], arge['coil_I'])
    coil = coil_cal.get_coil(params)
    _, B_coil = coil_cal.get_plot_args(params)

    oldcoil_arge['number_segments'] = arge['number_points'] 
    old_coil_cal = post_coilset.CoilSet(oldcoil_arge)  
    oldparams = (oldcoil_arge['coil_arg'], oldcoil_arge['coil_fr'], oldcoil_arge['coil_I'])
    oldcoil = old_coil_cal.get_coil(oldparams)
    _, old_B_coil = old_coil_cal.get_plot_args(oldparams)

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
    B_coil = np.linalg.norm(B_coil, axis=-1)
    Bmax, Bmin = float(np.max(B_coil)), float(np.min(B_coil))
    print('maxB = ', Bmax, Bmin)
    B = np.zeros((nic, 5, ns+1))
    B = B.at[:, :-1, :-1].set(B_coil)
    B = B.at[:, :-1, -1].set(B_coil[:, :, 0])
    B = B.at[:, -1].set(B[:, 0, :])

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
    oB_coil = np.linalg.norm(old_B_coil, axis=-1)
    oBmax, oBmin = float(np.max(oB_coil)), float(np.min(oB_coil))
    print('maxB = ', oBmax, oBmin)
    oB = np.zeros((nic, 5, ns+1))
    oB = oB.at[:, :-1, :-1].set(oB_coil)
    oB = oB.at[:, :-1, -1].set(oB_coil[:, :, 0])
    oB = oB.at[:, -1].set(oB[:, 0, :])

    color = np.ones((nic, 5, ns+1))
    old_color = np.zeros((nic, 5, ns+1))
    color = color.at[:,:,0].set(0.99)
    old_color = old_color.at[:,:,0].set(0.01)
    fig = go.Figure()
    # for i in range(nic):
    #     fig.add_trace(go.Surface(x=oxx[i,:,:], y=oyy[i,:,:], z=ozz[i,:,:], 
    #         surfacecolor = oB[i,:,:], cmax = oBmax, cmin = oBmin, colorbar_title='old_B_coil [T]', 
    #         colorbar = dict(x = 0.2,tickfont = dict(size=20))  ))
    # for i in range(nic):
    #     fig.add_trace(go.Surface(x=xx[i,:,:], y=yy[i,:,:], z=zz[i,:,:], 
    #         surfacecolor = B[i,:,:], cmax = Bmax, cmin = Bmin, colorbar_title='B_coil [T]', 
    #         colorbar = dict(x = 0.7,tickfont = dict(size=20)) ,colorscale="Viridis"))
    for i in range(nic):
        fig.add_trace(go.Surface(x=oxx[i,:,:], y=oyy[i,:,:], z=ozz[i,:,:], 
            surfacecolor = old_color[i],cmax = 1, cmin = 0,))
    for i in range(nic):
        fig.add_trace(go.Surface(x=xx[i,:,:], y=yy[i,:,:], z=zz[i,:,:], 
            surfacecolor = color[i],cmax = 1, cmin = 0,))
    fig.update_layout(coloraxis_showscale=True)
    fig.update_layout(scene_aspectmode='data',  scene = dict(
        xaxis = dict(#  backgroundcolor="white",  gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white",),
        yaxis = dict(# backgroundcolor="white", gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white"),
        zaxis = dict(# backgroundcolor="white", gridcolor="white",
            title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white",),))
    fig.show() 
    fig.write_html('results/paper/w7x/force_opt_compare.html')
    return


filename = 'results/paper/w7x/force_opt.h5'

# plot_surface_0(filename)
# plot_coil(filename)
# plot_coil_0(filename)
# plot_segment(filename)
# plot_surface(filename)
# plot_alpha(filename)
# plot_dcs(filename)
# plot_strain(filename)
# plot_force(filename)

coilfile = 'results/paper/QA/opt_2.h5'
# plot_coil_compare(filename, coilfile)
# plot_surface_compare(filename, coilfile)
plot_strain_compare(filename, coilfile)

def plot(args, coil_all, loss_end, lossvals, params, surface_data):
    """ Draw as set in input. """

    if args['plot_coil'] != 0 :
        plot_coil(args, params, surface_data, loss_end)
    if args['plot_loss'] != 0 :
        plot_loss(lossvals)
    if args['plot_poincare'] != 0 :
        plot_poincare(args, coil_all, surface_data)

    return


def plot_loss(lossfile):

    lossvals = np.load('{}'.format(lossfile))
    fig = go.Figure()
    fig.add_scatter(x = np.arange(0, len(lossvals), 1), y = lossvals, 
                        name = 'lossvalue', line = dict(width=5))
    fig.update_xaxes(title_text = "opt_coil",title_font = {"size": 25},title_standoff = 12, 
                        tickfont = dict(size=25))
    fig.update_yaxes(title_text = "lossvalue",title_font = {"size": 25},title_standoff = 12, 
                        tickfont = dict(size=25) ,type="log", exponentformat = 'e')
    fig.show()
    return



def plot(p):
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
    _, B_coil = coil_cal.get_plot_args(params)

    nic = arge['number_independent_coils'] 
    ns = arge['number_points']
    nn = arge['number_normal']
    nb = arge['number_binormal']

    B_coil = np.linalg.norm(B_coil, axis=-1)
    Bmax, Bmin = float(np.max(B_coil)), float(np.min(B_coil))
    print('maxB = ', Bmax, Bmin)

    if p['filament or finite_build'] == 1:
        xx, yy, zz, arg = finite_build(coil, nic, ns, nn, nb, B_coil)

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

        orr = orr.at[:,0,:ns,:].set(oldcoil[:nic, 0, 0, :, :])
        orr = orr.at[:,1,:ns,:].set(oldcoil[:nic, 0, onb-1, :, :])
        orr = orr.at[:,2,:ns,:].set(oldcoil[:nic, onn-1, onb-1, :, :])
        orr = orr.at[:,3,:ns,:].set(oldcoil[:nic, onn-1, 0, :, :])
        orr = orr.at[:,4,:ns,:].set(oldcoil[:nic, 0, 0, :, :])
        orr = orr.at[:,0,-1,:].set(oldcoil[:nic, 0, 0, 0, :])
        orr = orr.at[:,1,-1,:].set(oldcoil[:nic, 0, onb-1, 0, :])
        orr = orr.at[:,2,-1,:].set(oldcoil[:nic, onn-1, onb-1, 0, :])
        orr = orr.at[:,3,-1,:].set(oldcoil[:nic, onn-1, 0, 0, :])
        orr = orr.at[:,4,-1,:].set(oldcoil[:nic, 0, 0, 0, :])
        oxx = orr[:,:,:,0]
        oyy = orr[:,:,:,1]
        ozz = orr[:,:,:,2]
        oB_coil = np.linalg.norm(old_B_coil, axis=-1)
        oBmax, oBmin = float(np.max(oB_coil)), float(np.min(oB_coil))
        print('maxB = ', oBmax, oBmin)
        oB = np.zeros((nic, 5, ns+1))
        oB = oB.at[:, :-1, :-1].set(oB_coil)
        oB = oB.at[:, :-1, -1].set(oB_coil[:, :, 0])
        oB = oB.at[:, -1].set(oB[:, 0, :])


    fig = go.Figure()
    for i in range(nic):
        fig.add_trace(go.Surface(x=oxx[i,:,:], y=oyy[i,:,:], z=ozz[i,:,:], 
            surfacecolor = oB[i,:,:], cmax = oBmax, cmin = oBmin, colorbar_title='old_B_coil [T]', 
            colorbar = dict(x = 0.2,tickfont = dict(size=20))  ))
    for i in range(nic):
        fig.add_trace(go.Surface(x=xx[i,:,:], y=yy[i,:,:], z=zz[i,:,:], 
            surfacecolor = B[i,:,:], cmax = Bmax, cmin = Bmin, colorbar_title='B_coil [T]', 
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
### coil:
'filament or finite_build': 1,  # 0:filament, 1:finite_build
'coil_color'      :   'B',      # the coils color: 'default', 'B', 'strain', 'force/length', 'contrast'
'number_points'  :   500,
### poincare:
'poincare_number':          25,         #       int,                                        
'poincare_phi0':            0,          # (phi0)float, 
'number_iter':              400,        #       int,  
'number_step':              1,          #       int,  
}

plot(p)




