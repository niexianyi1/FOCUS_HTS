import json
import plotly.graph_objects as go
import bspline
import jax.numpy as np
import plot
pi = np.pi
from jax import vmap
from jax.config import config
import scipy.interpolate as si 
import coilpy
import sys
import time
config.update("jax_enable_x64", True)


with open('/home/nxy/codes/focusadd-spline/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)


## finit difference







