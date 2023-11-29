

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


# 测试bspline

coil = np.load('/home/nxy/codes/focusadd-spline/initfiles/w7x/coil.npy')  
# print(coil.shape)  # (50, 65, 3)
c, bc = bspline.prep(coil, nc, ns+1, 3)
# print(c0.shape)    # (50, 3, 67)
rc = bspline.splev(bc, c[0])
der1, wrk1 = bspline.der1_splev(bc, c[0])
der2 = bspline.der2_splev(bc, wrk1)
print(der2)

# bn需要ns个点，第ns个点的导数与最后一个控制点有关吗
# 仍然无关，基函数为0
# 所以最后一个控制点只影响最后一段线段，且不包括该线段的起点（左开右闭）
# 第一个控制点同样对应于第一段，但是包括起点（左闭右开）

