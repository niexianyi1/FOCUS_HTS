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

# def read_hdf5(filename):
#     f = h5py.File(filename, "r")
#     arge = {}
#     for key in list(f.keys()):
#         val = f[key][()]
#         if key == 'num_fourier_coils':
#             key = 'number_fourier_coils'
#         if isinstance(val, bytes):
#             val = str(val, encoding='utf-8')
#         arge.update({key: val})
#     f.close()
#     return arge

# filename = 'results/LQA/LQA_no.h5'
# arge = read_hdf5(filename)
# arg = arge['coil_arg']
# fc = np.zeros((6, 4, 6))
# fc = fc.at[:,0].set(arg[:,0])
# fc = fc.at[:,1].set(arg[:,4])
# fc = fc.at[:,2].set(arg[:,8])
# fc = fc.at[:,3].set(arg[:,12])
# np.save('initfiles/Landreman-Paul_QA/fc.npy',fc)


a = numpy.spacing(1)
print(a)