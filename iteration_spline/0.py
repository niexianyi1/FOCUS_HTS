

import json
import plotly.graph_objects as go
import bspline
import jax.numpy as np
pi = np.pi
from jax import vmap
from jax.config import config
import scipy.interpolate as si 
import sys
import numpy
config.update("jax_enable_x64", True)


with open('/home/nxy/codes/focusadd-spline/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)

sg = np.load('/home/nxy/codes/focusadd-spline/initfiles/w7x/sg_surf.npy')
print(np.sum(sg))


