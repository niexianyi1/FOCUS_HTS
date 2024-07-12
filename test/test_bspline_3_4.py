import jax.numpy as np
import json
import numpy
import plotly.graph_objects as go
import scipy.interpolate as si
import sys
sys.path.append('/home/nxy/codes/coil_spline_HTS/iteration')
import spline
import fourier

with open('/home/nxy/codes/coil_spline_HTS/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)


nic, ns, ncp, nfc = 1, 64, 67, 6
## xyz坐标
coil = np.load('/home/nxy/codes/coil_spline_HTS/initfiles/w7x/w7x_coil_5.npy')[-1]
fc = fourier.compute_coil_fourierSeries(nic, ns, nfc, coil[np.newaxis, :, :])
coil = numpy.array(np.transpose(coil, (1, 0)))
tck3, u = si.splprep(x=coil, k=3, per=1, s=0)
tck4, u = si.splprep(x=coil, k=4, per=1, s=0)



ns = 1024
u = u = np.linspace(0, (ns-1)/ns ,ns)
xyz3 = si.splev(u, tck3)
xyz4 = si.splev(u, tck4)
xyzf = fourier.compute_r_centroid(fc, nfc, nic, ns)[0]





# fig = go.Figure()
# fig.add_scatter3d(x=xyz3[0][:],y=xyz3[1][:],z=xyz3[2][:], name='三阶spline', mode='markers', marker_size = 1.5)   
# fig.add_scatter3d(x=xyz4[0][:],y=xyz4[1][:],z=xyz4[2][:], name='四阶spline', mode='markers', marker_size = 1.5)   
# fig.update_layout(scene_aspectmode='data')
# fig.show()

fig = go.Figure()
fig.add_scatter3d(x=xyz3[0][:],y=xyz3[1][:],z=xyz3[2][:], name='三阶spline', mode='markers', marker_size = 1.5)   
fig.add_scatter3d(x=xyzf[:, 0],y=xyzf[:, 1],z=xyzf[:, 2], name='fourier', mode='markers', marker_size = 1.5)   
fig.update_layout(scene_aspectmode='data')
fig.show()


























