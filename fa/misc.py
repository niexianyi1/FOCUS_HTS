import json
import plotly.graph_objects as go
import bspline 
import jax.numpy as np
import coilpy
pi = np.pi
from jax import vmap
from jax.config import config
import scipy.interpolate as si 
config.update("jax_enable_x64", True)


with open('/home/nxy/codes/focusadd-spline/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)


def symmetry(r):
    rc_total = np.zeros((50, 65, 3))
    rc_total = rc_total.at[0:10, :, :].add(r)
    for i in range(5 - 1):        
        theta = 2 * pi * (i + 1) / 5
        T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        rc_total = rc_total.at[10*(i+1):10*(i+2), :, :].add(np.dot(r, T))
    
    return rc_total

coil = np.load('/home/nxy/codes/focusadd-spline/initfiles/w7x/circle_coil.npy')
rc = symmetry(coil)
rc = rc.reshape(50*65, 3)
print(rc.shape)


fig = go.Figure()
fig.add_scatter3d(x=rc[:, 0],y=rc[:, 1],z=rc[:, 2], name='rc_bspline', mode='markers', marker_size = 1)   
fig.update_layout(scene_aspectmode='data')
fig.show()


