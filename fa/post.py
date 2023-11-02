import jax.numpy as np
import plotly.express as px
import plotly.graph_objects as go
from jax import vmap
import scipy.interpolate as si 
import bzbt 
import json
import lossfunction
import bspline
import sys


with open('/home/nxy/codes/focusadd-spline/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)



def spline(ai):    # 线圈

    # ----- rc -----
    c = np.load("/home/nxy/codes/focusadd-spline/results/circle/c_a0.npy")  # 实际ns为64   
    N, NC, k = 65, 10, 3   # 参数     
    t = np.linspace(-3/(N-3), N/(N-3), N+4)
    N = 640
    u = np.linspace(0, 1, N)
    rc = np.zeros((10, 3, N))
    tck = [[0]*3 for i in range (10)]
    for i in range(10):
        tck[i] = [t, c[i], k]
        rc = rc.at[i, :, :].set(si.splev(u, tck[i]))  
    rc = np.transpose(rc, (0, 2, 1))
    rc = np.reshape(rc, (N*10, 3))
    # ----- rc_new -----  
    c_new = np.load("/home/nxy/codes/focusadd-spline/results/circle/c_{}.npy".format(ai))  
    rc_new = np.zeros((10, 3, N))
    tck = [[0]*3 for i in range (10)]
    for i in range(10):
        tck[i] = [t, c_new[i], k]
        rc_new = rc_new.at[i, :, :].set(si.splev(u, tck[i]))  
    rc_new = np.transpose(rc_new, (0, 2, 1))
    rc_new = np.reshape(rc_new, (N*10, 3))

    # # ----- 整体 -----
    fig = go.Figure()
    fig.add_scatter3d(x=rc[:, 0],y=rc[:, 1],z=rc[:, 2], name='rc', mode='markers', marker_size = 2)
    fig.add_scatter3d(x=rc_new[:, 0],y=rc_new[:, 1],z=rc_new[:, 2], name='rc_new', mode='markers', marker_size = 2)
    fig.update_layout(scene_aspectmode='data')
    fig.show()    
    # ----- 单线圈加密 -----
    rc = rc[2*640:3*640, :]
    rc_new = rc_new[2*640:3*640, :]
    fig = go.Figure()   
    fig.add_scatter3d(x=rc[:, 0],y=rc[:, 1],z=rc[:, 2], name='rc'+"{}".format(i), mode='markers', marker_size = 5)
    fig.add_scatter3d(x=rc_new[:, 0],y=rc_new[:, 1],z=rc_new[:, 2], name='rc_new'+"{}".format(i), mode='markers', marker_size = 5)
    fig.update_layout(scene_aspectmode='data')
    fig.show()  

    N = 65
    u = np.linspace(0, 1, N)
    rc = np.zeros((NC, 3, N))
    tck = [[0]*3 for i in range (NC)]
    for i in range(NC):
        tck[i] = [t, c[i], k]
        rc = rc.at[i, :, :].set(si.splev(u, tck[i]))  
    rc = np.transpose(rc, (0, 2, 1))

    rc_new = np.zeros((NC, 3, N))
    tck = [[0]*3 for i in range (NC)]
    for i in range(NC):
        tck[i] = [t, c_new[i], k]
        rc_new = rc_new.at[i, :, :].set(si.splev(u, tck[i]))  
    rc_new = np.transpose(rc_new, (0, 2, 1))
    rc_new = rc_new[2]
    rc = rc[2]


    # ----- 单线圈原型 -----
    fig = go.Figure()
    for i in range(9):
        fig.add_scatter3d(x=rc[i*7:(i+1)*7, 0],y=rc[i*7:(i+1)*7, 1],z=rc[i*7:(i+1)*7, 2], name='rc'+"{}".format(i), mode='markers', marker_size = 5)
        fig.add_scatter3d(x=rc_new[i*7:(i+1)*7, 0],y=rc_new[i*7:(i+1)*7, 1],z=rc_new[i*7:(i+1)*7, 2], name='rc_new'+"{}".format(i), mode='markers', marker_size = 5)
    fig.add_scatter3d(x=rc[63:, 0],y=rc[63:, 1],z=rc[63:, 2], name='rc9', mode='markers', marker_size = 5)
    fig.add_scatter3d(x=rc_new[63:, 0],y=rc_new[63:, 1],z=rc_new[63:, 2], name='rc_new9', mode='markers', marker_size = 5)
    fig.update_layout(scene_aspectmode='data')
    fig.show()

    return 

def loss(ai):
    ns = 64
    c = np.load("/home/nxy/codes/focusadd-spline/results/circle/c_{}.npy".format(ai))  # 单周期
    bc = bspline.get_bc_init(ns+1)
    I = np.ones(50)*1e6
    r_coil = vmap(lambda c :bspline.splev(bc, c), in_axes=0, out_axes=0)(c)[:, :-1, np.newaxis, np.newaxis, :]
    r_surf = np.load(args['surface_r'])
    nn = np.load(args['surface_nn'])
    sg = np.load(args['surface_sg'])
    der1, wrk1 = vmap(lambda c :bspline.der1_splev(bc, c), in_axes=0, out_axes=0)(c)
    der2, wrk2 = vmap(lambda wrk1 :bspline.der2_splev(bc, wrk1), in_axes=0, out_axes=0)(wrk1)
    der3 = vmap(lambda wrk2 :bspline.der3_splev(bc, wrk2), in_axes=0, out_axes=0)(wrk2)
    der1 = der1[:, :-1, :]
    der2 = der2[:, :-1, :]
    der3 = der3[:, :-1, :]
    dl = der1[:, :, np.newaxis, np.newaxis, :]/ns

    def quadratic_flux(I, dl, r_surf, r_coil, nn, sg):
        r_coil = symmetry(r_coil)
        dl = symmetry(dl)
        B = biotSavart(I ,dl, r_surf, r_coil)  
        B_all = B        
        print('Bmax = ', np.max(np.linalg.norm(B_all, axis=-1)))
        return ( 0.5 * np.sum( np.sum(nn * B_all/
                    np.linalg.norm(B_all, axis=-1)[:, :, np.newaxis], axis=-1) ** 2 * sg))    

    def biotSavart(I ,dl, r_surf, r_coil):
        mu_0 = 1e-7
        mu_0I = I * mu_0
        mu_0Idl = (mu_0I[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] * dl)  # NC x NS x NNR x NBR x 3
        r_minus_l = (r_surf[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis, :]
            - r_coil[:, np.newaxis, np.newaxis, :, :, :, :])  # NC x NZ/nfp x NT x NS x NNR x NBR x 3
        top = np.cross(mu_0Idl[:, np.newaxis, np.newaxis, :, :, :, :], r_minus_l)  # NC x NZ x NT x NS x NNR x NBR x 3
        bottom = (np.linalg.norm(r_minus_l, axis=-1) ** 3)  # NC x NZ x NT x NS x NNR x NBR
        B = np.sum(top / bottom[:, :, :, :, :, :, np.newaxis], axis=(0, 3, 4, 5))  # NZ x NT x 3
        return B

    def symmetry(r):
        rc_total = np.zeros((50, 64, 1, 1, 3))
        rc_total = rc_total.at[0:10, :, :, :, :].add(r)
        for i in range(5 - 1):        
            theta = 2 * np.pi * (i + 1) / 5
            T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            rc_total = rc_total.at[10*(i+1):10*(i+2), :, :, :, :].add(np.dot(r, T))
        return rc_total

    def average_length(r_coil):      #new
        al = np.zeros_like(r_coil)
        al = al.at[:, :-1, :].set(r_coil[:, 1:, :] - r_coil[:, :-1, :])
        al = al.at[:, -1, :].set(r_coil[:, 0, :] - r_coil[:, -1, :])
        return np.sum(np.linalg.norm(al, axis=-1)) / (10)

    def curvature(der1, der2):
        bottom = np.linalg.norm(der1, axis = -1)**3
        top = np.linalg.norm(np.cross(der1, der2), axis = -1)
        k = top / bottom
        k_mean = np.mean(k)
        k_max = np.max(k)
        return k_mean, k_max
    
    def distance_cc(r_coil):  ### 暂未考虑finite-build
        r_coil = symmetry(r_coil)
        rc = r_coil[:, :, 0, 0, :]
        dr = rc[:10, :, :] - rc[1:11, :, :]
        dr = np.linalg.norm(dr, axis = -1)
        dcc_min = np.min(dr)
        return dcc_min
    
    def distance_cs(r_coil, r_surf):  ### 暂未考虑finite-build
        rc = r_coil[:, :, 0, 0, :]
        rs = r_surf
        dr = rc[:10, :, np.newaxis, np.newaxis, :] - rs[np.newaxis, np.newaxis, :, :, :]
        dr = np.linalg.norm(dr, axis = -1)
        dcs_min = np.min(dr)
        return dcs_min

    Bn = quadratic_flux(I, dl, r_surf, r_coil, nn, sg)
    length = average_length(r_coil)
    k_mean, k_max = curvature(der1, der2)
    dcc = distance_cc(r_coil)
    dcs = distance_cs(r_coil, r_surf)
    print(Bn, length, k_mean, dcc, dcs, k_max)
    print(wb*Bn + wl*length + wc*k_mean + wdcc*dcc + wdcs*dcs + wcm*k_max )
    return Bn, length, k_mean, dcc, dcs, k_max

def lossvals(ai):
    loss = np.load("/home/nxy/codes/focusadd-spline/results/circle/loss_{}.npy".format(ai))
    fig = go.Figure()
    fig.add_scatter(x=np.arange(0, 1000, 1), y=loss, name='loss', line=dict(width=2))
    fig.update_xaxes(title_text = "iteration",title_font = {"size": 30},title_standoff = 15, tickfont=dict(size=30))
    fig.update_yaxes(title_text = "lossvalue",title_font = {"size": 30},title_standoff = 15, tickfont=dict(size=30))
                    # ,type="log", exponentformat = 'e'
    fig.show()
    return





# lossv = np.zeros((6, 6))
# j = 0
# for i in [0,3,5,6,8,9]:
#     ai = 'c{}'.format(i)
#     lossv = lossv.at[j].set(loss(ai))
#     j = j+1
# print(lossv)
ai = 'a1'
lossvals(ai)
spline(ai)
loss(ai)


# name = ['Bn', 'length', 'k_mean', 'dcc', 'dcs', 'k_max']
# fig = go.Figure()
# for i in range(6):
#     fig.add_scatter(x=np.array([3, 4.2, 5, 5.4, 6.2, 6.6]), y=lossv[:, i], name=name[i], line=dict(width=2))
# fig.update_xaxes(title_text = "wc",title_font = {"size": 30},title_standoff = 15, tickfont=dict(size=30))
# fig.update_yaxes(title_text = "lossvalue",title_font = {"size": 30},title_standoff = 15, tickfont=dict(size=30))
#                 # ,type="log", exponentformat = 'e'
# fig.show()



