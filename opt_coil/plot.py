
### Drawing after optimization,
### (More functions can be performed in post-processing: post/post_plot.py)

import jax.numpy as np
import plotly.graph_objects as go
import coilset
import spline
import coilpy
import sys
sys.path.append('HTS')
import B_self



def plot(args, coil_all, loss_end, lossvals, params, surface_data):
    """ Draw as set in input. """

    if args['plot_coil'] != 0 :
        plot_coil(args, params, surface_data, loss_end)
    if args['plot_loss'] != 0 :
        plot_loss(lossvals)
    if args['plot_poincare'] != 0 :
        plot_poincare(args, coil_all, surface_data)

    return

def plot_coil(args, params, surface_data, loss_end):  

    if args['coil_case'] == 'spline':
        bc, tj = spline.get_bc_init(args['number_points'], args['number_control_points'])
        args['bc'] = bc
        args['tj'] = tj

    ns = args['number_segments']
    args['number_segments'] = args['number_points'] 
    coil_cal = coilset.CoilSet(args)    
    coil = coil_cal.get_coil(params)
    I, dl, coil, der1, der2, _, v1, v2, binormal = coil_cal.cal_coil(params)
    I = I * args['I_normalize']
    curva = np.cross(der1, der2) / (np.linalg.norm(der1, axis = -1)**3)[:,:,np.newaxis]
    B_coil = B_self.coil_self_B_rec(args, coil, I, dl, v1, v2, binormal, curva, der2) 
    args['number_segments'] = ns

    ns = args['number_points']
    nn = args['number_normal']
    nb = args['number_binormal']
    nic = args['number_independent_coils']
    nzs = int(args['number_zeta']/args['number_field_periods']/(args['stellarator_symmetry']+1))

    if args['plot_coil'] == 1 :
        coil = np.reshape(coil[:nic], (ns * nic * nn * nb, 3))
        fig = go.Figure()
        fig.add_scatter3d(x=coil[:, 0],y=coil[:, 1],z=coil[:, 2], name='coil', mode='markers', marker_size = 1.5)   
        fig.update_layout(scene_aspectmode='data')


    if args['plot_coil'] == 11 :
        coil = np.reshape(coil[:nic], (ns * nic * nn * nb, 3))
        rs,_,_ =  surface_data
        rs = rs[:nzs]
        r_surf = np.zeros((nzs, args['number_theta']+1, 3))
        r_surf = r_surf.at[:,:-1,:].set(rs)
        r_surf = r_surf.at[:,-1,:].set(rs[:,0,:])
        fig = go.Figure()
        fig.add_trace(go.Surface(x=r_surf[:,:,0], y=r_surf[:,:,1], z=r_surf[:,:,2] ))
        fig.update_layout(scene_aspectmode='data')
        fig.add_scatter3d(x=coil[:, 0],y=coil[:, 1],z=coil[:, 2], name='coil', mode='markers', marker_size = 1.5)   


    if args['plot_coil'] == 2 :

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
        fig = go.Figure()
        for i in range(nic):
            fig.add_trace(go.Surface(x=xx[:,i,:], y=yy[:,i,:], z=zz[:,i,:]))
        fig.update_layout(scene_aspectmode='data')


    if args['plot_coil'] == 21 :
        rs,_,_ =  surface_data
        rs = rs[:nzs]
        r_surf = np.zeros((nzs, args['number_theta']+1, 3))
        r_surf = r_surf.at[:,:-1,:].set(rs)
        r_surf = r_surf.at[:,-1,:].set(rs[:,0,:])
        rr = np.zeros((5, nic, ns+1, 3))
        rr = rr.at[0,:,:ns,:].set(coil[:nic, 0, 0, :, :])
        rr = rr.at[1,:,:ns,:].set(coil[:nic, 0, nb-1, :, :])
        rr = rr.at[3,:,:ns,:].set(coil[:nic, nn-1, nb-1, :, :])
        rr = rr.at[2,:,:ns,:].set(coil[:nic, nn-1, 0, :, :])
        rr = rr.at[4,:,:ns,:].set(coil[:nic, 0, 0, :, :])
        rr = rr.at[0,:,-1,:].set(coil[:nic, 0, 0, 0, :])
        rr = rr.at[1,:,-1,:].set(coil[:nic, 0, nb-1, 0, :])
        rr = rr.at[3,:,-1,:].set(coil[:nic, nn-1, nb-1, 0, :])
        rr = rr.at[2,:,-1,:].set(coil[:nic, nn-1, 0, 0, :])
        rr = rr.at[4,:,-1,:].set(coil[:nic, 0, 0, 0, :])
        xx = rr[:,:,:,0]
        yy = rr[:,:,:,1]
        zz = rr[:,:,:,2]
        fig = go.Figure()

        lossB = loss_end['loss_B']
        lossB = np.linalg.norm(lossB[:nzs], axis=-1)
        Bs = np.zeros((nzs, args['number_theta']+1))
        Bs = Bs.at[:,:-1].set(lossB)
        Bs = Bs.at[:,-1].set(lossB[:,0])

        B_coil = np.linalg.norm(B_coil, axis=-1)
        Bmax, Bmin = float(np.max(B_coil)), float(np.min(B_coil))
        print('maxB = ', Bmax, Bmin)
        B = np.zeros((nic, 5, ns+1))
        B = B.at[:, :-1, :-1].set(B_coil)
        B = B.at[:, :-1, -1].set(B_coil[:, :, 0])
        B = B.at[:, -1].set(B[:, 0, :])


        fig = go.Figure()
        fig.add_trace(go.Surface(x=r_surf[:,:,0], y=r_surf[:,:,1], z=r_surf[:,:,2],
                surfacecolor = Bs, colorbar_title='B_coil [T]', 
                colorbar = dict(x = 0.8,tickfont = dict(size=20)),colorscale="Viridis" ))
        for i in range(nic):
            fig.add_trace(go.Surface(x=xx[:,i,:], y=yy[:,i,:], z=zz[:,i,:], 
                surfacecolor = B[i,:,:], cmax = Bmax, cmin = Bmin, colorbar_title='B_coil [T]', 
                colorbar = dict(x = 0.1,tickfont = dict(size=20)),colorscale="plasma"))
        fig.update_layout(coloraxis_showscale=True)
        fig.update_layout(scene_aspectmode='data',  scene = dict(
            xaxis = dict(title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white"),
            yaxis = dict(title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white"),
            zaxis = dict(title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white")))
    fig.show() 
    return 


def plot_loss(lossvals):
    fig = go.Figure()
    fig.add_scatter(x = np.arange(0, len(lossvals), 1), y = lossvals, 
                        name = 'lossvalue', line = dict(width=5))
    fig.update_xaxes(title_text = "iteration",title_font = {"size": 25},title_standoff = 12, 
                        tickfont = dict(size=25))
    fig.update_yaxes(title_text = "lossvalue",title_font = {"size": 25},title_standoff = 12, 
                        tickfont = dict(size=25) ,type="log", exponentformat = 'e')
    fig.show()
    return


def plot_poincare(args, coil_all, surface_data):
    r_surf,_ ,_ = surface_data
    pn = args['poincare_number']
    phi0 = args['poincare_phi0']
    phi = int(phi0/2/np.pi * args['number_zeta'])
    r_surf = r_surf[phi]
    r0 = (r_surf[0,0]**2 + r_surf[0,1]**2)**0.5
    mid = int(args['number_theta'] / 2)
    rmid = (r_surf[mid,0]**2 + r_surf[mid,1]**2)**0.5
    dr = (r0 - rmid) / (pn-1)
    r0 = [rmid+i*dr for i in range(pn)]
    z0 = [0 for i in range(pn)]

    coil = np.mean(coil_all['coil_r'], axis = (1,2))
    x = coil[:, :, 0]
    y = coil[:, :, 1]
    z = coil[:, :, 2]
    I = coil_all['coil_I'] 
    name = group = np.ones((args['number_coils']))
    coil_py = coilpy.coils.Coil(x, y, z, I, name, group)
    bfield = coil_py.bfield
    line = coilpy.misc.tracing(bfield, r0, z0, args['poincare_phi0'], 
            args['number_iter'], args['number_field_periods'], args['number_step'])
    line = np.reshape(line, (pn*(args['number_iter']+1), 2))
    surf = np.load("{}".format(args['surface_r_file']))[0]
    fig = go.Figure()
    fig.add_scatter(x = line[:, 0], y = line[:, 1],  name='poincare', mode='markers', marker_size = 1.5)
    fig.add_scatter(x = surf[:, 0], y = surf[:, 2],  name='surface', line = dict(width=2.5))
    fig.update_layout(scene_aspectmode='data')
    fig.show() 
    return














    