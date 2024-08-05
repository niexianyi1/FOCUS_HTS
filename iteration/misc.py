import json
import jax.numpy as np
import numpy
import read_init
import jax
from jax import jit, vmap
import fourier
import spline
import plotly.graph_objects as go
import h5py
import read_init
import sys 
sys.path.append('HTS')
import material_jcrit

pi = np.pi

with open('initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)

a = [1,1]
b = [2,2]
print((np.array(a)+np.array(b))**2)