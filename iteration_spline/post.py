import jax.numpy as np
import plotly.express as px
import plotly.graph_objects as go
from jax import vmap
import scipy.interpolate as si 
import json
import lossfunction
import bspline
import numpy
import coilpy


with open('/home/nxy/codes/focusadd-spline/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)


def get_bc_init(ns, ncp):
    k = 3
    t = np.linspace(-3/(ncp-3), (ncp)/(ncp-3), ncp+4) 
    u = np.linspace(0, (ns-1)/ns ,ns)
    bc = [t, u, k]
    tj = tjev(bc)
    return bc, tj 
def tjev(bc):
    t, u, _ = bc
    t0 = numpy.zeros_like(t)
    u0 = numpy.zeros_like(u)
    tj = numpy.zeros_like(u)          # len(t) = nic
    j = 0
    for i in range(len(t)):
        t0[i] = t[ i]
    for i in range(len(u)):  # len(u[a]) = ns
        u0[i] = u[ i]
        while u0[i]>=t0[j+1] :
            j = j+1
        tj[i] = j
    return tj
def spline(ai):    # 线圈

    # ----- rc_new -----  
    c = np.load("/home/nxy/codes/focusadd-spline/results_b/circle/c_{}.npy".format(ai))

    bc, tj = get_bc_init(nps, ncp)
    t, u, k = bc
    rc_new =  vmap(lambda c :bspline.splev(t, u, c, tj, nps), 
                    in_axes=0, out_axes=0)(c)

    # ----- 单线圈 -----

    fig = go.Figure()   
    # fig.add_scatter3d(x=rc[:, 0],y=rc[:, 1],z=rc[:, 2], name='rc', mode='markers', marker_size = 5)
    fig.add_scatter3d(x=rc_new[1, :, 0],y=rc_new[1, :, 1],z=rc_new[1, :, 2], name='rc_new', mode='markers', marker_size = 5)
    fig.update_layout(scene_aspectmode='data')
    fig.show()  


    rc_new = np.reshape(rc_new, (nps*nic, 3))


    # # ----- 整体 -----
    fig = go.Figure()
    # fig.add_scatter3d(x=rc[:, 0],y=rc[:, 1],z=rc[:, 2], name='rc', mode='markers', marker_size = 2)
    fig.add_scatter3d(x=rc_new[:, 0],y=rc_new[:, 1],z=rc_new[:, 2], name='rc_new', mode='markers', marker_size = 2)
    fig.update_layout(scene_aspectmode='data')
    fig.show()    


    # N, NC, k = 65, 5, 3
    # u = np.linspace(0, 1, N)
    # rc = np.zeros((NC, 3, N))
    # tck = [[0]*3 for i in range (NC)]
    # for i in range(NC):
    #     tck[i] = [t, c[i], k]
    #     rc = rc.at[i, :, :].set(si.splev(u, tck[i]))  
    # rc = np.transpose(rc, (0, 2, 1))

    # rc_new = np.zeros((NC, 3, N))
    # tck = [[0]*3 for i in range (NC)]
    # for i in range(NC):
    #     tck[i] = [t, c_new[i], k]
    #     rc_new = rc_new.at[i, :, :].set(si.splev(u, tck[i]))  
    # rc_new = np.transpose(rc_new, (0, 2, 1))
    # rc_new = rc_new[2]
    # rc = rc[2]


    # # ----- 单线圈原型 -----
    # fig = go.Figure()
    # for i in range(9):
    #     fig.add_scatter3d(x=rc[i*7:(i+1)*7, 0],y=rc[i*7:(i+1)*7, 1],z=rc[i*7:(i+1)*7, 2], name='rc'+"{}".format(i), mode='markers', marker_size = 5)
    #     fig.add_scatter3d(x=rc_new[i*7:(i+1)*7, 0],y=rc_new[i*7:(i+1)*7, 1],z=rc_new[i*7:(i+1)*7, 2], name='rc_new'+"{}".format(i), mode='markers', marker_size = 5)
    # fig.add_scatter3d(x=rc[63:, 0],y=rc[63:, 1],z=rc[63:, 2], name='rc9', mode='markers', marker_size = 5)
    # fig.add_scatter3d(x=rc_new[63:, 0],y=rc_new[63:, 1],z=rc_new[63:, 2], name='rc_new9', mode='markers', marker_size = 5)
    # fig.update_layout(scene_aspectmode='data')
    # fig.show()

    return 

def loss(ai):
    c = np.load("/home/nxy/codes/focusadd-spline/results_b/circle/c_{}.npy".format(ai))  # 单周期
    bc, tj = get_bc_init(ns, ncp)
    t,u,k=bc
    I = np.ones(nc)*1e6
    r_coil = vmap(lambda c :bspline.splev(t,u, c, tj, ns), in_axes=0, out_axes=0)(c)[:, :, np.newaxis, np.newaxis, :]

    r_surf = np.load(args['surface_r'])
    nn = np.load(args['surface_nn'])
    sg = np.load(args['surface_sg'])
    der1, wrk1 = vmap(lambda c :bspline.der1_splev(t,u, c, tj, ns), in_axes=0, out_axes=0)(c)
    der2 = vmap(lambda wrk1 :bspline.der2_splev(t,u, wrk1, tj, ns), in_axes=0, out_axes=0)(wrk1)
    dl = der1[:, :, np.newaxis, np.newaxis, :]/ns

    def quadratic_flux(I, dl, r_surf, r_coil, nn, sg):
        r_coil = symmetry(r_coil)
        dl = symmetry_der(dl)
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
        if ss == 1:
            rc = np.zeros((nic*2, ns, nnr, nbr, 3))
            rc = rc.at[:nic, :, :, :, :].add(r)
            T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            rc = rc.at[nic:nic*2, :, :, :, :].add(np.dot(r, T))
        else:
            rc = r
        rc_total = np.zeros((nc, ns, nnr, nbr, 3))
        rc_total = rc_total.at[:nic*(ss+1), :, :, :, :].add(rc)
        for i in range(nfp - 1):        
            theta = 2 * np.pi * (i + 1) / nfp
            T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            rc_total = rc_total.at[nic*(ss+1)*(i+1):nic*(ss+1)*(i+2), :, :, :, :].add(np.dot(rc, T))
        return rc_total

    def symmetry_der(r):
        if ss == 1:
            rc = np.zeros((nic*2, ns, nnr, nbr, 3))
            rc = rc.at[:nic, :, :, :, :].add(r)
            T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            rc = rc.at[nic:nic*2, :, :, :, :].add(-np.dot(r, T))
        else:
            rc = r
        rc_total = np.zeros((nc, ns, nnr, nbr, 3))
        rc_total = rc_total.at[:nic*(ss+1), :, :, :, :].add(rc)
        for i in range(nfp - 1):        
            theta = 2 * np.pi * (i + 1) / nfp
            T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            rc_total = rc_total.at[nic*(ss+1)*(i+1):nic*(ss+1)*(i+2), :, :, :, :].add(np.dot(rc, T))
        return rc_total

    def average_length(r_coil):      #new
        al = np.zeros_like(r_coil)
        al = al.at[:, :-1, :].set(r_coil[:, 1:, :] - r_coil[:, :-1, :])
        al = al.at[:, -1, :].set(r_coil[:, 0, :] - r_coil[:, -1, :])
        return np.sum(np.linalg.norm(al, axis=-1)) / (nic)

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

    print(Bn)
    print(length)
    print(k_mean)
    print(dcc)
    print(dcs)
    print(k_max)

    print(wb*Bn + wl*length + wc*k_mean + wdcc*dcc + wdcs*dcs + wcm*k_max )
    return Bn, length, k_mean, dcc, dcs, k_max

def lossvals(ai):
    loss = np.load("/home/nxy/codes/focusadd-spline/results_b/circle/loss_{}.npy".format(ai))
    fig = go.Figure()
    fig.add_scatter(x=np.arange(0, len(loss), 1), y=loss, name='loss', line=dict(width=2))
    fig.update_xaxes(title_text = "iteration",title_font = {"size": 30},title_standoff = 15, tickfont=dict(size=30))
    fig.update_yaxes(title_text = "lossvalue",title_font = {"size": 30},title_standoff = 15, tickfont=dict(size=30))
                    # ,type="log", exponentformat = 'e'
    fig.show()
    return

def poincare(ai):
    def symmetry(r):
        rc_total = np.zeros((50, 64, 3))
        rc_total = rc_total.at[0:10, :, :].add(r)
        for i in range(5 - 1):        
            theta = 2 * np.pi * (i + 1) / 5
            T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            rc_total = rc_total.at[10*(i+1):10*(i+2), :, :].add(np.dot(r, T))
        return rc_total
    c = np.load("/home/nxy/codes/focusadd-spline/results/circle/c_{}.npy".format(ai))  
    r0 = [6]
    z0 = [0]
    lenr = len(r0)
    lenz = len(z0)
    assert lenr == lenz
    bc = bspline.get_bc_init(c.shape[2]-2)
    rc = vmap(lambda c :bspline.splev(bc, c), in_axes=0, out_axes=0)(c)
    rc = symmetry(rc[:, :-1, :])  
    rc = np.reshape(rc, (50, 64, 3))
    x = rc[:, :, 0]   
    y = rc[:, :, 1]
    z = rc[:, :, 2]
    nfp = 1
    I = np.ones((nc))
    name = np.zeros(nc)
    group = np.zeros(nc)

    coil = coilpy.coils.Coil(x, y, z, I, name, group)
    line = coilpy.misc.tracing(coil.bfield, r0, z0, phi0, niter, nfp, nstep)

    line = np.reshape(line, (lenr*(niter+1), 2))
    fig = go.Figure()
    fig.add_scatter(x = line[:, 0], y = line[:, 1],  name='poincare', mode='markers', marker_size = 1.5)
    fig.update_layout(scene_aspectmode='data')
    fig.show()

    return


def stellarator_symmetry(r, ns):
    rc = np.zeros((10, ns, 3))
    rc = rc.at[0:5, :, :].add(r)

    T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rc = rc.at[5:10, :, :].add(np.dot(r, T))

    return rc


ai = 'a1'
# lossvals(ai)
# spline(ai)
loss(ai)
# poincare(ai)

