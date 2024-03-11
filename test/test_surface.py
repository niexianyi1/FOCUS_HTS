

import json
import sys 
import plotly.graph_objects as go
sys.path.append('/home/nxy/codes/coil_spline_HTS/iteration')
import read_plasma



# with open('/home/nxy/codes/coil_spline_HTS/initfiles/init_args.json', 'r') as f:    # 传入地址
#     args = json.load(f)

file = '/home/nxy/codes/coil_spline_HTS/initfiles/ncsx_c09r00/c09r00.boundary'
nz, nt = 64, 64
R, Z, Nfp, MT, MZ = read_plasma.read_plasma_boundary("{}".format(file))
r_surf, NN, sg = read_plasma.get_plasma_boundary(R, Z, nz, nt, Nfp, MT, MZ)

fig = go.Figure()
fig.add_trace(go.Surface(x=r_surf[:,:,0], y=r_surf[:,:,1], z=r_surf[:,:,2] ))
fig.update_layout(scene_aspectmode='data')
fig.show()





