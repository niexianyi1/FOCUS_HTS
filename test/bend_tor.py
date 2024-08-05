import json
import h5py
import jax.numpy as np
from jax import jit, vmap
import plotly.graph_objects as go
import sys
sys.path.append('iteration')
import read_init
import fourier
import spline
sys.path.append('post')
import post_coilset
pi = np.pi


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

filename = 'results/hsx/hsx.h5'
args = read_hdf5(filename)

arge['number_segments'] = arge['number_points'] 
params = (args['coil_arg'], args['coil_fr'], args['coil_I'])
coil_cal = post_coilset.CoilSet(args)    
coil = coil_cal.get_coil(params)
arge['number_segments'] = ns
ns = arge['number_points']


nic=args['number_independent_coils']
ns=args['number_segments']
nn=nb=2
rr = np.zeros((nic, ns+1, 5, 3))
rr = rr.at[:,:ns,0,:].set(coil[:nic, :, 0, 0, :])
rr = rr.at[:,:ns,1,:].set(coil[:nic, :, 0, nb-1, :])
rr = rr.at[:,:ns,2,:].set(coil[:nic, :, nn-1, nb-1, :])
rr = rr.at[:,:ns,3,:].set(coil[:nic, :, nn-1, 0, :])
rr = rr.at[:,:ns,4,:].set(coil[:nic, :, 0, 0, :])
rr = rr.at[:,-1,0,:].set(coil[:nic, 0, 0, 0, :])
rr = rr.at[:,-1,1,:].set(coil[:nic, 0, 0, nb-1, :])
rr = rr.at[:,-1,2,:].set(coil[:nic, 0, nn-1, nb-1, :])
rr = rr.at[:,-1,3,:].set(coil[:nic, 0, nn-1, 0, :])
rr = rr.at[:,-1,4,:].set(coil[:nic, 0, 0, 0, :])
rr = np.transpose(rr, [2, 0, 1, 3])     # (5,nic,ns)
xx = rr[:,:,:,0]
yy = rr[:,:,:,1]
zz = rr[:,:,:,2]

strain = np.zeros((5,nic,ns+1))
for i in range(5):
    strain = strain.at[i,:,:-1].set(bend+tor)
    strain = strain.at[i,:,-1].set(strain[i,:,0])
smax,smin = float(np.max((bend+tor)[0])), float(np.min((bend+tor)[0]))
print(smax,smin)
fig = go.Figure()
for i in range(1):
    fig.add_trace(go.Surface(x=xx[:,i,:], y=yy[:,i,:], z=zz[:,i,:],
    surfacecolor=strain[:,i], cmax = smax, cmin = smin,))
fig.update_layout(scene_aspectmode='data')
fig.show() 








