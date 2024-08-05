

### 对比自动微分的导数值和有限差分的导数值 
import json
import plotly.graph_objects as go
import jax.numpy as np
from jax import value_and_grad, jit
from jax.config import config
import scipy.interpolate as si 
import sys
sys.path.append('iteration')
import fourier
import spline
import read_init
import lossfunction
from coilset import CoilSet 
config.update("jax_enable_x64", True)
pi = np.pi

    
with open('initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)

# 获取初始数据
args, coil_arg_init, fr_init, surface_data, I_init = read_init.init(args)   
params0 = (coil_arg_init, fr_init, I_init)

coil_cal = CoilSet(args)
coil_output_func = coil_cal.cal_coil                  
value0, gradient = value_and_grad(
    lambda params :lossfunction.loss(args, coil_output_func, params, surface_data),
    allow_int = True)(params0) 

d = 1e-4
## finit difference
    # fourier_arg
dloss = np.zeros((6, 5, 6))
for i in range(6):
    for j in range(5):
        print(i,j)
        for k in range(6):
            coil_arg1 = coil_arg_init.at[i, j, k].add(d*coil_arg_init[i, j, k])
            fr1 = fr_init
            params1 = (coil_arg1, fr1, I_init)                
            value1, gradient = value_and_grad(
                lambda params :lossfunction.loss(args, coil_output_func, params, surface_data),
                allow_int = True)(params0) 
            dloss = dloss.at[i,j,k].set((value1-value0)/(d*coil_arg_init[i, j, k]))
                

gbc = np.load("results/grad/loss_f_bn_arg.npy")

print('(db-gc)/db = ', (dloss-gbc)/dloss)
print('mean = ', np.mean((dloss-gbc)/dloss))









