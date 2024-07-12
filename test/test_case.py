import jax.numpy as np
import plotly.graph_objects as go
import sys
sys.path.append('/home/nxy/codes/coil_spline_HTS/iteration')
import lossfunction
import fourier
pi = np.pi


nc = 128
I = np.array([1e5 for i in range(nc)])


## bh
def hh(r_coil, r_surf):
    Rvec = r_surf[:, np.newaxis, :] - r_coil[np.newaxis, :, :]
    assert (Rvec.shape)[-1] == 3
    RR = np.linalg.norm(Rvec, axis=2)
    Riv = Rvec[:, :-1, :]
    Rfv = Rvec[:, 1:, :]
    Ri = RR[:, :-1]
    Rf = RR[:, 1:]
    B = I[0]*1e-7*(np.sum(np.cross(Riv, Rfv) * ((Ri + Rf) / ((Ri * Rf) * (Ri * Rf + np.sum(Riv * Rfv, axis=2))))[:, :, np.newaxis],axis=1,))
    return B

## rs
zeta = np.linspace(0, 2*pi, 129)
theta = np.linspace(0, 2*pi, 129)
R, r = 4, 1
rs = np.zeros((129, 129, 3))
for i in range(129):
    x = (R + r * np.cos(theta)) * np.cos(zeta[i])
    y = (R + r * np.cos(theta)) * np.sin(zeta[i])
    z = r * np.sin(theta)
    rs = rs.at[i].set(np.transpose(np.array([x, y, z])))
    
## rc
zeta = np.linspace(0, 2*pi, 1+nc)
theta = np.linspace(0, 2*pi, 65)
R, r = 4, 2
rc = np.zeros((nc, 65, 3))
for i in range(nc):
    x = (R + r * np.cos(theta)) * np.cos(zeta[i])
    y = (R + r * np.cos(theta)) * np.sin(zeta[i])
    z = r * np.sin(theta)
    rc = rc.at[i].set(np.transpose(np.array([x, y, z])))


fc = fourier.compute_coil_fourierSeries(nc, 64, 6, rc)
der1 = fourier.compute_der1(fc, 6, nc, 64, theta)
dl = der1[:, np.newaxis, np.newaxis, :-1,:] * 2*pi/64
rcc = rc[:, np.newaxis, np.newaxis,:-1, :]

B = lossfunction.biotSavart(rcc, I, dl, rs)
B = np.linalg.norm(B, axis=-1)
Bmax = float(np.max(B))
Bmin = float(np.min(B))
print(Bmax, Bmin)


rc = np.reshape(rc, (nc*65, 3))

fig = go.Figure()
fig.add_trace(go.Surface(x=rs[:,:,0], y=rs[:,:,1], z=rs[:,:,2],
    surfacecolor = B, cmax = Bmax, cmin = Bmin, colorbar_title='B [T]', 
            colorbar = dict(tickfont = dict(size=20)) ))
fig.update_layout(scene_aspectmode='data')
fig.add_scatter3d(x=rc[:, 0],y=rc[:, 1],z=rc[:, 2], name='surface', line = dict(width=4),marker_size = 1.5)#mode='markers', marker_size = 1.5)    
fig.show()

ra = np.linspace(3, 5, 41)
B0 = 2.56/ra

y, z = np.array([0 for i in range(41)]), np.array([0 for i in range(41)])
rsn = np.transpose(np.array([ra, y, z]))
rsn = rsn[np.newaxis, :, :]
rc = np.reshape(rc, (nc, 65, 3))[:,  np.newaxis, np.newaxis,:-1,:]
b1 = lossfunction.biotSavart(rc, I, dl, rsn) 
b1 = np.linalg.norm(b1, axis=-1)
b1 = np.squeeze(b1)







fig = go.Figure()
fig.add_scatter(x = np.linspace(3, 5, 41), y = B0, 
                        name = 'Analytical value', line = dict(width=5))
fig.add_scatter(x = np.linspace(3, 5, 41), y = b1, 
                        name = 'b1 * pi /32',  mode='markers', marker_size = 6)
fig.update_xaxes(title_text = "R [m]",title_font = {"size": 25},title_standoff = 12, 
                tickfont = dict(size=25))
fig.update_yaxes(title_text = "Bt [T]",title_font = {"size": 25},title_standoff = 12, 
                tickfont = dict(size=25) )
fig.show()



rc = np.squeeze(rc)
print(rc.shape)






