import json
import sys 
import jax.numpy as np
import plotly.graph_objects as go
sys.path.append('HTS')
from material_jcrit import get_critical_current
sys.path.append('iteration')
import read_init

sys.path.append('test')
from coil_cal import CoilSet
with open('initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)

I = args['current_I'][0]

args, coil_arg_init, fr_init, surface_data, I_init = read_init.init(args)
args['number_segments'] = args['number_points'] 
coil_cal = CoilSet(args)
params = (coil_arg_init, fr_init, I_init)
a = coil_cal.get_fb_args(params,0)
Bself, I_signle, lw, lt, B_coil = a
print('B = ', Bself)
print('I = ', I_signle)
print('lw, lt = ', lw, lt)


B_coil = np.linalg.norm(B_coil, axis=-1)


def plot(args, params):
    ns = args['number_segments']
    # args['number_segments'] = args['number_points'] 
    coil_cal = CoilSet(args)    
    coil = coil_cal.get_coil(params)
    # args['number_segments'] = ns

    ns = args['number_points']
    nn = args['number_normal']
    nb = args['number_binormal']
    nic = args['number_independent_coils']
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
    B = np.zeros((5, nic, ns+1))
    B = B.at[:-1, :, :-1].set(B_coil)
    B = B.at[:-1, :, -1].set(B_coil[:, :, 0])
    B = B.at[-1].set(B[0])



    fig = go.Figure()
    for i in range(nic):
        fig.add_trace(go.Surface(x=xx[:,i,:], y=yy[:,i,:], z=zz[:,i,:], 
                surfacecolor = B[:,i,:], cmax = 10.6, cmin = 5.2, colorbar_title='B_self'))
    fig.update_layout(scene_aspectmode='data')
    # fig.update_coloraxes(cmax = 15, cmin = 5.2, colorbar_title='我是colorbar')

    fig.show() 
    return 

plot(args, params)


