

import json
import plotly.graph_objects as go
import bspline
import jax.numpy as np
import numpy
import plot
pi = np.pi
from jax import vmap
from jax.config import config
import scipy.interpolate as si 
import sys
import time
config.update("jax_enable_x64", True)


with open('/home/nxy/codes/focusadd-spline/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)



rc = np.load('/home/nxy/codes/focusadd-spline/initfiles/w7x/circle_coil_c.npy')
coil = np.zeros((10, 71, 3))
coil = coil.at[:, 3:-3, :].set(rc)
coil = coil.at[:, -3:, :].set(rc[:, 1:4, :])
coil = coil.at[:, :3, :].set(rc[:, -4:-1, :])
np.save('/home/nxy/codes/focusadd-spline/initfiles/w7x/circle_coil_c.npy', coil)







