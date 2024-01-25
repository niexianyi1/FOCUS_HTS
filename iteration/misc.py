import json
import jax.numpy as np
import read_init
import jax
from jax import jit, vmap
import fourier
import spline
import scipy.interpolate as si
import numpy
pi = np.pi
import plotly.graph_objects as go
import h5py
import sys 
sys.path.append('/home/nxy/codes/coil_spline_HTS/HTS')
import material_jcrit


with open('/home/nxy/codes/coil_spline_HTS/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)

b = np.array((1,-2,-3,4,5,6))
c = np.array((1,-2,-3))
a = np.array(((1,2,3),(4,5,6,)))
print(np.append(np.append(b, a), c))
