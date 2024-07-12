
### 比较fourier表达式的挠率 与 Spline表达式用差分算法的挠率

import jax.numpy as np
import numpy
import json
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
coil = np.load('/home/nxy/codes/coil_spline_HTS/initfiles/w7x/w7x_coil_5.npy')[0]
coil = coil[np.newaxis,:,:]

## fourier挠率
fc = fourier.compute_coil_fourierSeries(nic, ns, nfc, coil)
ns = 64
fxyz = fourier.compute_r_centroid(fc, nfc, nic, ns)
der1 = fourier.compute_der1(fc, nfc, nic, ns)
der2 = fourier.compute_der2(fc, nfc, nic, ns)
der3 = fourier.compute_der3(fc, nfc, nic, ns)
cross12 = np.cross(der1, der2)
top = (cross12[:, :, 0] * der3[:, :, 0] + 
        cross12[:, :, 1] * der3[:, :, 1] +
        cross12[:, :, 2] * der3[:, :, 2])
bottom = np.linalg.norm(cross12, axis=-1) ** 2
ftor = abs(top / bottom)[0, :-1]

## spline坐标
c, bc, tj = spline.get_c_init(coil, nic, ns, ncp)
t, u, k = bc
bxyz = spline.splev(t, u, c[0], tj, ns)

## 四阶bspline
coil = numpy.array(np.transpose(coil[0], (1, 0)))
tck4, u = si.splprep(x=coil, k=4, per=1, s=0)
u = u = np.linspace(0, (ns-1)/ns ,ns)
xyz4 = si.splev(u, tck4)
d14 = si.splev(u, tck4, der = 1)
d24 = si.splev(u, tck4, der = 2)
d34 = si.splev(u, tck4, der = 3)
d14, d24, d34 = np.transpose(np.array(d14), (1, 0)), np.transpose(np.array(d24), (1, 0)), np.transpose(np.array(d34), (1, 0))
cross12 = np.cross(d14, d24)
top = (cross12[:, 0] * d34[:, 0] + 
        cross12[:, 1] * d34[:, 1] +
        cross12[:, 2] * d34[:, 2])
bottom = np.linalg.norm(cross12, axis=-1) ** 2
btor4 = abs(top / bottom) 

## 离散点torsion
dt = 1/ns  ### 此参数为无关变量，都会被约分
d1, d2, d3 = np.zeros((ns, 3)), np.zeros((ns, 3)), np.zeros((ns, 3))
d1 = d1.at[1:-1].set((bxyz[2:] - bxyz[:-2]) / 2 / dt)
d1 = d1.at[0].set((bxyz[1] - bxyz[-1]) / 2 / dt)
d1 = d1.at[-1].set((bxyz[0] - bxyz[-2]) / 2 / dt)

d2 = d2.at[1:-1].set((d1[2:] - d1[:-2]) / 2 / dt)
d2 = d2.at[0].set((d1[1] - d1[-1]) / 2 / dt)
d2 = d2.at[-1].set((d1[0] - d1[-2]) / 2 / dt)

d3 = d3.at[1:-1].set((d2[2:] - d2[:-2]) / 2 / dt)
d3 = d3.at[0].set((d2[1] - d2[-1]) / 2 / dt)
d3 = d3.at[-1].set((d2[0] - d2[-2]) / 2 / dt)

cross12 = np.cross(d1, d2)
top = (cross12[:, 0] * d3[:, 0] + 
        cross12[:, 1] * d3[:, 1] +
        cross12[:, 2] * d3[:, 2])
bottom = np.linalg.norm(cross12, axis=-1) ** 2
btor = abs(top / bottom) 

print('fourier_torsion = ', ftor)
print('spline_torsion = ', btor4)
print('(btor - btor4) / btor4 = ', np.mean(abs((btor - btor4) / btor4)))
fig = go.Figure()
fig.add_scatter(x = np.arange(0, ns, 1), y = btor4, 
                    name = '四阶spline_torsion', line = dict(width=5))
fig.add_scatter(x = np.arange(0, ns, 1), y = ftor, 
                    name = 'fourier_torsion', line = dict(width=5))
fig.update_xaxes(title_text = "number",title_font = {"size": 25},title_standoff = 12, 
                    tickfont = dict(size=25))
fig.update_yaxes(title_text = "torsion",title_font = {"size": 25},title_standoff = 12, 
                    tickfont = dict(size=25) ,)#type="log", exponentformat = 'e')
fig.show()

