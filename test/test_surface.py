
import jax.numpy as np
import json
import sys 
import plotly.graph_objects as go
# sys.path.append('iteration')
import useful_script.read_plasma as read_plasma



with open('initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)

file = 'initfiles/hsx/plasma.boundary'
nz, nt = 64, 64
R, Z, Nfp, MT, MZ = read_plasma.read_plasma_boundary("{}".format(file))
r_surf, NN, sg = read_plasma.get_plasma_boundary(R, Z, nz, nt, Nfp, MT, MZ)

# fig = go.Figure()
# fig.add_trace(go.Surface(x=r_surf[:,:,0], y=r_surf[:,:,1], z=r_surf[:,:,2] ))
# fig.update_layout(scene_aspectmode='data')
# fig.show()


def circle_coil(args, surface):
    nfc = args['number_fourier_coils']
    nz = args['number_zeta']
    r = args['circle_coil_radius']
    nic = args['number_independent_coils']
    ns = args['number_segments']
    nc = args['number_coils']
    nzs = int(nz/2/args['number_field_periods'])
    # axis = np.zeros((nz + 1, 3))
    # axis = axis.at[:-1, :].set(np.mean(surface, axis = 1))
    # axis = axis.at[-1].set(axis[0])
    # axis = axis[np.newaxis, :, :]
    # fa = fourier.compute_coil_fourierSeries(axis, nfc)
    theta = np.linspace(0, 2 * np.pi, nc + 1) + np.pi/(nc+1)
    
    # axis_center = fourier.compute_r_centroid(fa, nc)
    # axis_center = np.squeeze(axis_center)[:-1]
    axis = np.mean(surface, axis = 1)[:nzs]
    
    
    circlecoil = np.zeros((nic, ns+1, 3))
    zeta = theta[:ns]
    theta = np.linspace(0, 2*np.pi, ns+1)
    for i in range(nic):
        axis_center = axis[i*nzs+int(nzs/2)]
        R = (axis_center[0]**2 + axis_center[1]**2)**0.5
        x = (R + r * np.cos(theta)) * np.cos(zeta[i])
        y = (R + r * np.cos(theta)) * np.sin(zeta[i])
        z = r * np.sin(theta) + axis_center[2]
        circlecoil = circlecoil.at[i].set(np.transpose(np.array([x, y, z])))

    return circlecoil

coil = circle_coil(args, r_surf)
np.save('initfiles/hsx/circle_coils.npy',coil)
coil = np.reshape(coil, (65*6, 3))
fig = go.Figure()
fig.add_scatter3d(x=coil[:, 0],y=coil[:, 1],z=coil[:, 2], name='new_coil', mode='markers', marker_size = 1.5)   
fig.add_trace(go.Surface(x=r_surf[:,:,0], y=r_surf[:,:,1], z=r_surf[:,:,2] ))
fig.update_layout(scene_aspectmode='data')
fig.show()




