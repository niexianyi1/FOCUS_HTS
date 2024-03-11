


import plotly.graph_objects as go
import jax.numpy as np
import h5py
import coilpy
import post_coilset
import sys
sys.path.append('/home/nxy/codes/coil_spline_HTS/iteration')
import spline
import fourier


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


def plot_loss(lossfile):

    lossvals = np.load('{}'.format(lossfile))
    fig = go.Figure()
    fig.add_scatter(x = np.arange(0, len(lossvals), 1), y = lossvals, 
                        name = 'lossvalue', line = dict(width=5))
    fig.update_xaxes(title_text = "iteration",title_font = {"size": 25},title_standoff = 12, 
                        tickfont = dict(size=25))
    fig.update_yaxes(title_text = "lossvalue",title_font = {"size": 25},title_standoff = 12, 
                        tickfont = dict(size=25) ,type="log", exponentformat = 'e')
    fig.show()
    return


def plot_coil(arge):
    nic = arge['number_independent_coils']  
    nzs = int(arge['number_zeta']/arge['number_field_periods']/(arge['stellarator_symmetry']+1))  
    rs = arge['surface_data_r']
    rs = rs[:nzs]
    r_surf = np.zeros((nzs, arge['number_theta']+1, 3))
    r_surf = r_surf.at[:,:-1,:].set(rs)
    r_surf = r_surf.at[:,-1,:].set(rs[:,0,:])

    if arge['coil_case'] == 'spline':
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
    fig.add_trace(go.Surface(x=r_surf[:,:,0], y=r_surf[:,:,1], z=r_surf[:,:,2] ))
    fig.update_layout(scene_aspectmode='data')
    fig.show()


def plot_coil_compare(arge, coilfile):    # 线圈
    nic = arge['number_independent_coils']  
    nzs = int(arge['number_zeta']/arge['number_field_periods']/(arge['stellarator_symmetry']+1))  
    rs = arge['surface_data_r']
    rs = rs[:nzs]
    r_surf = np.zeros((nzs, arge['number_theta']+1, 3))
    r_surf = r_surf.at[:,:-1,:].set(rs)
    r_surf = r_surf.at[:,-1,:].set(rs[:,0,:])

    if arge['coil_case'] == 'spline':
        arge['coil_arg'] = arge['coil_arg'][:, :, :-3]
        bc, tj = spline.get_bc_init(arge['number_points'], arge['number_control_points'])
        arge['bc'] = bc
        arge['tj'] = tj

    ns = arge['number_segments']
    oldcoil = np.load('{}'.format(coilfile))
    fc = fourier.compute_coil_fourierSeries(nic, ns, arge['num_fourier_coils'], oldcoil)   

    arge['number_segments'] = arge['number_points'] 
    params = (arge['coil_arg'], arge['coil_fr'], arge['coil_I'])
    coil_cal = post_coilset.CoilSet(arge)    
    coil = coil_cal.get_coil(params)
    arge['number_segments'] = ns

    ns = arge['number_points']
    theta = np.linspace(0, 2 * np.pi, ns + 1)
    oldcoil = fourier.compute_r_centroid(fc, arge['num_fourier_coils'], nic, ns, theta)
    oldcoil = np.reshape(oldcoil[:nic, :-1, :], (ns * nic, 3))

    coil = np.mean(coil, axis = (1,2))
    coil = np.reshape(coil[:nic], (ns * nic, 3))


    fig = go.Figure()
    fig.add_scatter3d(x=coil[:, 0],y=coil[:, 1],z=coil[:, 2], name='newcoil', mode='markers', marker_size = 1.5)   
    fig.add_scatter3d(x=oldcoil[:, 0],y=oldcoil[:, 1],z=oldcoil[:, 2], name='oldcoil', mode='markers', marker_size = 1.5)   
    fig.add_trace(go.Surface(x=r_surf[:,:,0], y=r_surf[:,:,1], z=r_surf[:,:,2] ))
    fig.update_layout(scene_aspectmode='data')
    fig.show()
    return 



def plot_coil_surface(arge):
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

    rr = np.zeros((5, nic, ns+1, 3))
    rr = rr.at[0,:,:ns,:].set(coil[:nic, 0, 0, :, :])
    rr = rr.at[1,:,:ns,:].set(coil[:nic, 0, nb-1, :, :])
    rr = rr.at[2,:,:ns,:].set(coil[:nic, nn-1, nb-1, :, :])
    rr = rr.at[3,:,:ns,:].set(coil[:nic, nn-1, 0, :, :])
    rr = rr.at[4,:,:ns,:].set(coil[:nic, 0, 0, :, :])
    rr = rr.at[0,:,-1,:].set(coil[:nic, 0, 0, 0, :])
    rr = rr.at[1,:,-1,:].set(coil[:nic, 0, nb-1, 0, :])
    rr = rr.at[2,:,-1,:].set(coil[:nic, nn-1, nb-1, 0, :])
    rr = rr.at[3,:,-1,:].set(coil[:nic, nn-1, 0, 0, :])
    rr = rr.at[4,:,-1,:].set(coil[:nic, 0, 0, 0, :])
    xx = rr[:,:,:,0]
    yy = rr[:,:,:,1]
    zz = rr[:,:,:,2]
    B_coil = np.linalg.norm(B_coil, axis=-1)
    Bmax, Bmin = float(np.max(B_coil)), float(np.min(B_coil))
    print('maxB = ', Bmax, Bmin)
    B = np.zeros((5, nic, ns+1))
    B = B.at[:-1, :, :-1].set(B_coil)
    B = B.at[:-1, :, -1].set(B_coil[:, :, 0])
    B = B.at[-1].set(B[0])


    fig = go.Figure()

    for i in range(nic):
        fig.add_trace(go.Surface(x=xx[:,i,:], y=yy[:,i,:], z=zz[:,i,:], 
            surfacecolor = B[:,i,:], cmax = Bmax, cmin = Bmin, colorbar_title='B_self [T]', 
            colorbar = dict(tickfont = dict(size=20))),
            
            )
    
    fig.update_layout(scene_aspectmode='data',  )
    fig.show() 
    return



filename = '/home/nxy/codes/coil_spline_HTS/results/ncsx/bnl13/hdf5.h5'
arge = read_hdf5(filename)
plot_coil(arge)
# plot_coil_surface(arge)

# coilfile = '/home/nxy/codes/coil_spline_HTS/initfiles/w7x/w7x_coil_5.npy'
# plot_coil_compare(arge, coilfile)

# lossfile = '/home/nxy/codes/coil_spline_HTS/results/ellipse/lossCG3/loss.npy'
# plot_loss(lossfile)



