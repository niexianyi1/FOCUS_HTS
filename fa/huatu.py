import jax.numpy as np
import plotly.express as px
import plotly.graph_objects as go
from jax import vmap
import scipy.interpolate as si 
import plot 
import json
 
def spline():    # 线圈
    # ----- rc_initial -----
    # rc_init = np.load("/home/nxy/codes/FOCUSADD_B/w7x_highres_bc.npy")
    # rc_initial = np.zeros((50, 1000, 3))
    # for i in range(50):
    #     tck, u = si.splprep([rc_init[i, :, 0], rc_init[i, :, 1], rc_init[i, :, 2]], s=0)
    #     u = np.arange(0, 1, 1/1000)
    #     rc_new = np.array(si.splev(u, tck))
    #     rc_new = np.transpose(rc_new, (1, 0))
    #     rc_initial = rc_initial.at[i, :, :].set(rc_new)       
    # rc_initial = np.reshape(rc_initial, (50000, 3))

    # ----- rc_bspline -----
    c = np.load("/home/nxy/codes/focusadd-spline/c.npy")
    N, NC, k = 67, 50, 3   # 参数
    t = np.zeros(N+k+1)
    u = np.zeros(N)  
    t = t.at[N:N+k+1].set(1)
    t = t.at[(k+1):N].set(np.arange(2/(N-1),(N-2)/(N-1),1/(N-1)))
    N = 500
    u = np.arange(0, 1, 1/N)
    rc_bspline = np.zeros((NC, 3, N))
    tck = [[0]*3 for i in range (NC)]
    for i in range(NC):
        tck[i] = [t, c[i], k]
        rc_bspline = rc_bspline.at[i, :, :].set(si.splev(u, tck[i]))  

    rc_bspline = np.transpose(rc_bspline, (0, 2, 1))
    rc_bspline = np.reshape(rc_bspline, (N*NC, 3))

    # ----- plot -----
    fig = go.Figure()
    fig.add_scatter3d(x=rc_bspline[:, 0],y=rc_bspline[:, 1],z=rc_bspline[:, 2], name='rc_bspline', mode='markers', marker_size = 1)
    fig.update_xaxes(title_text = "iteration")
    fig.update_yaxes(title_text = 'lossvalue')
    fig.show()


    return 



def lossvals():
    loss = np.load("/home/nxy/codes/focusadd-spline/results/w7x/highres_loss_100.npy")
    # loss_f = np.load("/home/nxy/codes/FOCUSADD_B/results/w7x_circle_loss_vals_1000f.npy")

    # loss_f = np.log10(loss_f)
    fig = go.Figure()
    fig.add_scatter(x=np.arange(0, 1000, 1), y=loss, name='loss', line=dict(width=5))
    fig.update_xaxes(title_text = "iteration")
    fig.update_yaxes(title_text = 'lossvalue', type="log")

    # fig.add_scatter(x=np.arange(0, 1000, 1), y=loss_f, name='loss_f')
    fig.show()
    return


spline()
# lossvals()

def read_makegrid(filename):
    r = np.zeros((50, 128, 3))
    with open(filename) as f:
        _ = f.readline()
        _ = f.readline()
        _ = f.readline()
        for i in range(50):
            for s in range(128):
                x = f.readline().split()
                r = r.at[i, s, 0].set(float(x[0]))
                r = r.at[i, s, 1].set(float(x[1]))
                r = r.at[i, s, 2].set(float(x[2]))
                # r[i, s, 0] = float(x)
                # r[i, s, 1] = float(y)
                # r[i, s, 2] = float(z)
            _ = f.readline()
    return r


