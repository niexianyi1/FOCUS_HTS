import jax.numpy as np
import plotly.graph_objects as go
import sys
sys.path.append('/home/nxy/codes/coil_spline_HTS/iteration')
import lossfunction
import fourier
pi = np.pi


## rs
zeta = np.linspace(0, 2*pi, 513)
theta = np.linspace(0, 2*pi, 33)
R, r = 4, 1
rs = np.zeros((513, 33, 3))
for i in range(513):
    zetai = zeta[i]
    x = (R + r * np.cos(theta)) * np.cos(zetai)
    y = (R + r * np.cos(theta)) * np.sin(zetai)
    z = r * np.sin(theta)
    rs = rs.at[i].set(np.transpose(np.array([x, y, z])))
    
## rc
zeta = np.arange(0, 2*pi, 2*pi/128)
theta = np.linspace(0, 2*pi, 65)
R, r = 4, 2
rc = np.zeros((128, 65, 3))
for i in range(128):
    zetai = zeta[i]
    x = (R + r * np.cos(theta)) * np.cos(zetai)
    y = (R + r * np.cos(theta)) * np.sin(zetai)
    z = r * np.sin(theta)
    rc = rc.at[i].set(np.transpose(np.array([x, y, z])))

fc = fourier.compute_coil_fourierSeries(128, 64, 6, rc)
der1 = fourier.compute_der1(fc, 6, 128, 64, theta)
der2 = fourier.compute_der2(fc, 6, 128, 64, theta)
der3 = fourier.compute_der3(fc, 6, 128, 64, theta)
dl = der1[:, :-1, np.newaxis, np.newaxis,:]
rcc = rc[:, :-1, np.newaxis, np.newaxis,:]
I = np.array([1e4 for i in range(128)])
B = lossfunction.biotSavart(rcc, I, dl, rs)
B = np.linalg.norm(B, axis=-1)
Bmax = float(np.max(B))
Bmin = float(np.min(B))
print(Bmax, Bmin)


rc = np.reshape(rc, (128*65, 3))

# fig = go.Figure()
# fig.add_trace(go.Surface(x=rs[:,:,0], y=rs[:,:,1], z=rs[:,:,2],
#     surfacecolor = B, cmax = Bmax, cmin = Bmin, colorbar_title='B [T]', 
#             colorbar = dict(tickfont = dict(size=20)) ))
# fig.update_layout(scene_aspectmode='data')
# fig.add_scatter3d(x=rc[:, 0],y=rc[:, 1],z=rc[:, 2], name='surface', line = dict(width=4),marker_size = 1.5)#mode='markers', marker_size = 1.5)    
# fig.show()

ra = np.linspace(3, 5, 41)
B0 = 2.56/ra

y, z = np.array([0 for i in range(41)]), np.array([0 for i in range(41)])
rsn = np.transpose(np.array([ra, y, z]))
rsn = rsn[np.newaxis, :, :]
rc = np.reshape(rc, (128, 65, 3))[:, :-1, np.newaxis, np.newaxis,:]
b1 = lossfunction.biotSavart(rc, I, dl, rsn) 
b1 = np.linalg.norm(b1, axis=-1)
b1 = np.squeeze(b1)

# fig = go.Figure()
# fig.add_scatter(x = np.linspace(3, 5, 41), y = B0, 
#                         name = 'Analytical value', line = dict(width=5))
# fig.add_scatter(x = np.linspace(3, 5, 41), y = b1, 
#                         name = 'b1',  mode='markers', marker_size = 6)
# fig.update_xaxes(title_text = "R [m]",title_font = {"size": 25},title_standoff = 12, 
#                 tickfont = dict(size=25))
# fig.update_yaxes(title_text = "Bt [T]",title_font = {"size": 25},title_standoff = 12, 
#                 tickfont = dict(size=25) )
# fig.show()


def average_length(coil):      #new 
    nic = 128
    al = np.zeros((nic, 64, 3))      # 有限截面平均
    al = al.at[:, :-1, :].set(coil[:nic, 1:, :] - coil[:nic, :-1, :])
    al = al.at[:, -1, :].set(coil[:nic, 0, :] - coil[:nic, -1, :])
    len = np.sum(np.linalg.norm(al, axis=-1)) / 128
    return len


def curvature(der1, der2):
    bottom = np.linalg.norm(der1, axis = -1)**3
    top = np.linalg.norm(np.cross(der1, der2), axis = -1)
    k = abs(top / bottom)
    k_mean = np.mean(k)
    return k_mean


def torsion(der1, der2, der3):       # new
    cross12 = np.cross(der1, der2)
    top = (
        cross12[:, :, 0] * der3[:, :, 0]
        + cross12[:, :, 1] * der3[:, :, 1]
        + cross12[:, :, 2] * der3[:, :, 2]
    )
    bottom = np.linalg.norm(cross12, axis=-1) ** 2
    t = abs(top / bottom)     # NC x NS
    t_mean = np.mean(t)
    return t_mean

rc = np.squeeze(rc)
print(rc.shape)
len = average_length(rc)
k = curvature(der1, der2)
t = torsion(der1, der2, der3)
print(len, k, t)





