

import jax.numpy as np


pi = np.pi

from jax.config import config

config.update("jax_enable_x64", True)




def read_focus(filename):      # 处理一下
    c = np.zeros((4, 3, 35)) 
    t = np.zeros((4, 39))   
    with open(filename) as f:
        for i in range(4):
            _ = f.readline()
            t0 = f.readline().split()     
            for k in range(39):
                t0[k] = float(t0[k])  
            t0 = np.array(t0)
            t = t.at[i, :].set(t0)
            _ = f.readline()
            for s in range(3):                
                c0 = f.readline().split()
                for j in range(35):
                    c0[j] = float(c0[j])  
                c0 = np.array(c0)
                c = c.at[i, s, :].set(c0)       
    return c, t

c, t = read_focus('/home/nxy/codes/coil_spline_HTS/ellipse.focus')
np.save('/home/nxy/codes/coil_spline_HTS/coil_c.npy', c)
np.save('/home/nxy/codes/coil_spline_HTS/coil_t.npy', t)