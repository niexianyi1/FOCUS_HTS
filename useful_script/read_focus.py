
import jax.numpy as np
pi = np.pi
from jax import config
config.update("jax_enable_x64", True)


# def read_focus(filename):      # 处理一下
#     c = np.zeros((4, 3, 35)) 
#     t = np.zeros((4, 39))   
#     with open(filename) as f:
#         for i in range(4):
#             _ = f.readline()
#             t0 = f.readline().split()     
#             for k in range(39):
#                 t0[k] = float(t0[k])  
#             t0 = np.array(t0)
#             t = t.at[i, :].set(t0)
#             _ = f.readline()
#             for s in range(3):                
#                 c0 = f.readline().split()
#                 for j in range(35):
#                     c0[j] = float(c0[j])  
#                 c0 = np.array(c0)
#                 c = c.at[i, s, :].set(c0)       
#     return c, t

# c, t = read_focus('/home/nxy/codes/coil_spline_HTS/ellipse.focus')
# np.save('/home/nxy/codes/coil_spline_HTS/coil_c.npy', c)
# np.save('/home/nxy/codes/coil_spline_HTS/coil_t.npy', t)



def read_focus(filename):
    with open(filename) as f:
        _ = f.readline()
        _ = f.readline()
        fc = np.zeros((6, 6, 17))
        for i in range(6):
            for j in range(8):
                _ = f.readline()
            for j in range(6):
                line = f.readline().split()
                for k in range(17):
                    line[k] = float(line[k])
                    
                line = np.array(line)
                print(line)
                fc = fc.at[j, i, :].set(line)
    return fc
fc = read_focus('initfiles/hsx/hsx.focus')
fcc = np.zeros((6, 6, 17))
fcc = fcc.at[0].set(fc[0])
fcc = fcc.at[1].set(fc[2])
fcc = fcc.at[2].set(fc[4])
fcc = fcc.at[3].set(fc[1])
fcc = fcc.at[4].set(fc[3])
fcc = fcc.at[5].set(fc[5])
print(fcc)
np.save('initfiles/hsx/fc_HSX.npy', fcc)





# from coilpy import vmec2focus
# vmec2focus(' initfiles/ncsx_c09r00/wout_c09r00_fb.nc', 
#         focus_file=' initfiles/ncsx_c09r00/c09r00.boundary', 
#         bnorm_file=' initfiles/ncsx_c09r00/bnorm.c09r00_fb')