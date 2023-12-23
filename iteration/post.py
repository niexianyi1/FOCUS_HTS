import jax.numpy as np
import plotly.express as px
import plotly.graph_objects as go
from jax import vmap, jit
import scipy.interpolate as si 
import json
import lossfunction
import fourier
import sys
import coilpy
pi = np.pi




with open('/home/nxy/codes/focusadd-spline/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)
I = np.ones(args['nc'])*1e6
args['I'] = I

def cal_coil(ai, args):
    fc = np.load("/home/nxy/codes/focusadd-spline/results_f/circle/fc_{}.npy".format(ai))
    fr = np.load("/home/nxy/codes/focusadd-spline/results_f/circle/fr_{}.npy".format(ai))
    params = fc,fr


    def coilset(params):             # 根据lossfunction的需求再添加新的输出项                   
        fc, fr = params   
        I_new = args['I'] / (args['nnr'] * args['nbr'])
        theta = np.linspace(0, 2 * pi, args['ns'] + 1)
        r_centroid = compute_r_centroid( theta, fc)  # [nc, ns+1, 3]
        der1, der2, der3 = compute_der( theta, fc)   # [nc, ns+1, 3]
        tangent, normal, binormal = compute_com( theta, der1, r_centroid)
        r =  compute_r( theta, fr, normal, binormal, r_centroid)
        frame = tangent, normal, binormal
        dl =  compute_dl( theta, params, frame, der1, der2, r_centroid)
        if args['ss'] == 1 :
            r =  stellarator_symmetry_r( theta, r)
            dl =  stellarator_symmetry_der( theta, dl)
        r =  symmetry( theta, r)
        dl =  symmetry( theta, dl)

        return I_new, dl, r, der1, der2, der3

    def compute_r_centroid( theta, fc):         # rc 是（nc/nfp,ns+1,3）
        rc = fourier.compute_r_centroid(fc, args['nfc'], args['nic'], args['ns'], theta)
        return rc[:, :-1, :]

    def compute_der( theta, fc):                    
        der1 = fourier.compute_der1(fc, args['nfc'], args['nic'], args['ns'], theta)
        der2 = fourier.compute_der2(fc, args['nfc'], args['nic'], args['ns'], theta)
        der3 = fourier.compute_der3(fc, args['nfc'], args['nic'], args['ns'], theta)
        return der1[:, :-1, :], der2[:, :-1, :], der3[:, :-1, :]
        
    def compute_com( theta, der1, r_centroid):    
        """ Computes T, N, and B """
        tangent =  compute_tangent( theta, der1)
        normal = - compute_normal( theta, r_centroid, tangent)
        binormal =  compute_binormal( theta, tangent, normal)
        return tangent, normal, binormal

    def compute_com_deriv( theta, frame, der1, der2, r_centroid):  
        tangent, normal, _ = frame
        tangent_deriv =  compute_tangent_deriv( theta, der1, der2)
        normal_deriv = - compute_normal_deriv( theta, tangent, tangent_deriv, der1, r_centroid)
        binormal_deriv =  compute_binormal_deriv( theta, tangent, normal, tangent_deriv, normal_deriv)
        return tangent_deriv, normal_deriv, binormal_deriv

    def compute_tangent( theta, der1):          
        """
        Computes the tangent vector of the coils. Uses the equation 
        T = dr/d_theta / |dr / d_theta|
        """
        return der1 / np.linalg.norm(der1, axis=-1)[:, :, np.newaxis]

    def compute_tangent_deriv( theta, der1, der2):     
        norm_der1 = np.linalg.norm(der1, axis=-1)
        mag_2 =  dot_product_rank3_tensor(der1, der2) / norm_der1 ** 3
        return der2 / norm_der1[:, :, np.newaxis] - der1 * mag_2[:, :, np.newaxis]

    def dot_product_rank3_tensor(a, b):         # dot
        dotab = (a[:, :, 0] * b[:, :, 0] + 
                a[:, :, 1] * b[:, :, 1] + 
                a[:, :, 2] * b[:, :, 2])
        return dotab

    def compute_coil_mid( theta, r_centroid):      # mid_point   r0=[args['nic'], 3]
        x = r_centroid[:, :-1, 0]
        y = r_centroid[:, :-1, 1]
        z = r_centroid[:, :-1, 2]
        r0 = np.zeros((args['nic'], 3))
        for i in range(args['nic']):
            r0 = r0.at[i, 0].add(np.sum(x[i]) / args['ns'])
            r0 = r0.at[i, 1].add(np.sum(y[i]) / args['ns'])
            r0 = r0.at[i, 2].add(np.sum(z[i]) / args['ns'])        
        return r0

    def compute_normal( theta, r_centroid, tangent):    
        r0 =  compute_coil_mid( theta, r_centroid)
        delta = r_centroid - r0[:, np.newaxis, :]
        dp =  dot_product_rank3_tensor(tangent, delta)
        normal = delta - tangent * dp[:, :, np.newaxis]
        mag = np.linalg.norm(normal, axis=-1)
        return normal / mag[:, :, np.newaxis]

    def compute_normal_deriv( theta, tangent, tangent_deriv, der1, r_centroid):          
        r0 =  compute_coil_mid( theta, r_centroid)
        delta = r_centroid - r0[:, np.newaxis, :]
        dp1 =  dot_product_rank3_tensor(tangent, delta)
        dp2 =  dot_product_rank3_tensor(tangent, der1)
        dp3 =  dot_product_rank3_tensor(tangent_deriv, delta)
        numerator = delta - tangent * dp1[:, :, np.newaxis]
        numerator_norm = np.linalg.norm(numerator, axis=-1)
        numerator_deriv = (
            der1
            - dp1[:, :, np.newaxis] * tangent_deriv
            - tangent * (dp2 + dp3)[:, :, np.newaxis]
        )
        dp4 =  dot_product_rank3_tensor(numerator, numerator_deriv)
        return (
            numerator_deriv / numerator_norm[:, :, np.newaxis]
            - (dp4 / numerator_norm ** 3)[:, :, np.newaxis] * numerator
        )

    def compute_binormal( theta, tangent, normal):           
        """ Computes the binormal vector of the coils, B = T x N """
        return np.cross(tangent, normal)

    def compute_binormal_deriv( theta, tangent, normal, tangent_deriv, normal_deriv):  
        return np.cross(tangent_deriv, normal) + np.cross(tangent, normal_deriv)

    def compute_alpha( theta, fr):    # alpha 用fourier
        alpha = np.zeros((args['nic'], args['ns']+1))
        alpha += theta * args['nr'] / 2
        Ac = fr[0]
        As = fr[1]
        for m in range(args['nfr']):
            arg = theta * m
            carg = np.cos(arg)
            sarg = np.sin(arg)
            alpha += (
                Ac[:, np.newaxis, m] * carg[np.newaxis, :]
                + As[:, np.newaxis, m] * sarg[np.newaxis, :]
            )
        return alpha[:, :-1]

    def compute_alpha_1( theta, fr):    
        alpha_1 = np.zeros((args['nic'], args['ns']+1 ))
        alpha_1 += args['nr'] / 2
        Ac = fr[0]
        As = fr[1]
        for m in range(args['nfr']):
            arg = theta * m
            carg = np.cos(arg)
            sarg = np.sin(arg)
            alpha_1 += (
                -m * Ac[:, np.newaxis, m] * sarg[np.newaxis, :]
                + m * As[:, np.newaxis, m] * carg[np.newaxis, :]
            )
        return alpha_1[:, :-1]

    def compute_frame( theta, fr, N, B):  
        """
        Computes the vectors v1 and v2 for each coil. v1 and v2 are rotated relative to
        the normal and binormal frame by an amount alpha. Alpha is parametrized by a Fourier series.
        """
        alpha =  compute_alpha( theta, fr)
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)
        v1 = calpha[:, :, np.newaxis] * N - salpha[:, :, np.newaxis] * B
        v2 = salpha[:, :, np.newaxis] * N + calpha[:, :, np.newaxis] * B
        return v1, v2

    def compute_frame_derivative( theta, params, frame, der1, der2, r_centroid):    
        _, N, B = frame
        _, fr = params
        alpha =  compute_alpha( theta, fr)
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)
        alpha1 =  compute_alpha_1( theta, fr)
        _, dNdt, dBdt =  compute_com_deriv( theta, frame, der1, der2, r_centroid)
        dv1_dt = (
            calpha[:, :, np.newaxis] * dNdt
            - salpha[:, :, np.newaxis] * dBdt
            - salpha[:, :, np.newaxis] * N * alpha1[:, :, np.newaxis]
            - calpha[:, :, np.newaxis] * B * alpha1[:, :, np.newaxis]
        )
        dv2_dt = (
            salpha[:, :, np.newaxis] * dNdt
            + calpha[:, :, np.newaxis] * dBdt
            + calpha[:, :, np.newaxis] * N * alpha1[:, :, np.newaxis]
            - salpha[:, :, np.newaxis] * B * alpha1[:, :, np.newaxis]
        )
        return dv1_dt, dv2_dt

    def compute_r( theta, fr, normal, binormal, r_centroid):      
        """
        Computes the position of the multi-filament coils.

        r is a nc x ns + 1 x nnr x nbr x 3 array which holds the coil endpoints
        dl is a nc x ns x nnr x nbr x 3 array which computes the length of the ns segments
        r_middle is a nc x ns x nnr x nbr x 3 array which computes the midpoint of each of the ns segments

        """

        v1, v2 =  compute_frame( theta, fr, normal, binormal)
        r = np.zeros((args['nic'], args['ns'], args['nnr'], args['nbr'], 3))
        r += r_centroid[:, :, np.newaxis, np.newaxis, :]
        for n in range(args['nnr']):
            for b in range(args['nbr']):
                r = r.at[:, :, n, b, :].add(
                    (n - 0.5 * (args['nnr'] - 1)) * args['ln'] * v1 + 
                    (b - 0.5 * (args['nbr'] - 1)) * args['lb'] * v2
                ) 
        return r

    def compute_dl( theta, params, frame, der1, der2, r_centroid):   
        dl = np.zeros((args['nic'], args['ns'], args['nnr'], args['nbr'], 3))
        dl += der1[:, :, np.newaxis, np.newaxis, :]
        dv1_dt, dv2_dt =  compute_frame_derivative( theta, params, frame, der1, der2, r_centroid)
        for n in range(args['nnr']):
            for b in range(args['nbr']):
                dl = dl.at[:, :, n, b, :].add(
                    (n - 0.5 * (args['nnr'] - 1)) * args['ln'] * dv1_dt + 
                    (b - 0.5 * (args['nbr'] - 1)) * args['lb'] * dv2_dt
                )

        return dl * (2 * pi / args['ns'])

    def symmetry( theta, r):
        npc = int(args['nc'] / args['nfp'])   # 每周期线圈数，number of coils per period
        rc_total = np.zeros((args['nc'], args['ns'], args['nnr'], args['nbr'], 3))
        rc_total = rc_total.at[0:npc, :, :, :, :].add(r)
        for i in range(args['nfp'] - 1):        
            theta_t = 2 * pi * (i + 1) / args['nfp']
            T = np.array([[np.cos(theta_t), -np.sin(theta_t), 0],
                [np.sin(theta_t), np.cos(theta_t), 0], [0, 0, 1]])
            rc_total = rc_total.at[npc*(i+1):npc*(i+2), :, :, :, :].add(np.dot(r, T))
        
        return rc_total

    def stellarator_symmetry_r( theta, r):
        rc = np.zeros((args['nic']*2, args['ns'], args['nnr'], args['nbr'], 3))
        rc = rc.at[0:args['nic'], :, :, :, :].add(r)
        T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        rc = rc.at[args['nic']:args['nic']*2, :, :, :, :].add(np.dot(r, T))
        return rc

    def stellarator_symmetry_der( theta, r):
        rc = np.zeros((args['nic']*2, args['ns'], args['nnr'], args['nbr'], 3))
        rc = rc.at[0:args['nic'], :, :, :, :].add(r)
        T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        rc = rc.at[args['nic']:args['nic']*2, :, :, :, :].add(-np.dot(r, T))
        return rc


    I_new, dl, r, der1, der2, der3 = coilset( params)

    return I_new, dl, r, der1, der2, der3



def coil(ai):    # 线圈
    N = 500
    # ----- rc -----
    rc = np.load("/home/nxy/codes/focusadd-spline/initfiles/w7x/coil.npy")[:nic, :, :]
    fc = fourier.compute_coil_fourierSeries(nic, rc.shape[1]-1, nfc, rc)
    theta = np.linspace(0, 2 * np.pi, N + 1)
    rc =  fourier.compute_r_centroid(fc, nfc, nic, N, theta)[:, :-1, :]

    # ----- rc_new -----  
    args['ns']=N
    _,_,rc_new,_,_,_ =  cal_coil(ai, args)

    # 有限截面
    if nnr!=1 and nbr!=1:
        rr = np.zeros((nic, N+1, 5, 3))
        rr = rr.at[:,:N,0,:].set(rc_new[:5,:,0,0,:])
        rr = rr.at[:,:N,1,:].set(rc_new[:5,:,0,nbr-1,:])
        rr = rr.at[:,:N,2,:].set(rc_new[:5,:,nnr-1,nbr-1,:])
        rr = rr.at[:,:N,3,:].set(rc_new[:5,:,nnr-1,0,:])
        rr = rr.at[:,:N,4,:].set(rc_new[:5,:,0,0,:])
        rr = rr.at[:,-1,0,:].set(rc_new[:5,0,0,0,:])
        rr = rr.at[:,-1,1,:].set(rc_new[:5,0,0,nbr-1,:])
        rr = rr.at[:,-1,2,:].set(rc_new[:5,0,nnr-1,nbr-1,:])
        rr = rr.at[:,-1,3,:].set(rc_new[:5,0,nnr-1,0,:])
        rr = rr.at[:,-1,4,:].set(rc_new[:5,0,0,0,:])
        rr = np.transpose(rr, [2, 0,1,3])     # (5,nic,N)
        xx = rr[:,:,:,0]
        yy = rr[:,:,:,1]
        zz = rr[:,:,:,2]
        fig = go.Figure()
        for i in range(nic):
            fig.add_trace(go.Surface(x=xx[:,i,:], y=yy[:,i,:], z=zz[:,i,:]))
        fig.update_layout(scene_aspectmode='data')
        fig.show() 

    rc_new = np.reshape(rc_new[:nic], (N*nic*nnr*nbr, 3))
    rc = np.reshape(rc, (N*nic, 3))
    # ----- 单线圈 -----

    # fig = go.Figure()   
    # fig.add_scatter3d(x=rc_new[:N*nnr*nbr, 0],y=rc_new[:N*nnr*nbr, 1],z=rc_new[:N*nnr*nbr, 2], name='rc_new', mode='markers', marker_size = 2)
    # fig.update_layout(scene_aspectmode='data')
    # fig.show() 
    # # ----- 整体 -----
    fig = go.Figure()
    fig.add_scatter3d(x=rc[:, 0],y=rc[:, 1],z=rc[:, 2], name='rc', mode='markers', marker_size = 2)
    fig.add_scatter3d(x=rc_new[:, 0],y=rc_new[:, 1],z=rc_new[:, 2], name='rc_new', mode='markers', marker_size = 2)
    fig.update_layout(scene_aspectmode='data')
    fig.show()    
    



    args['ns']=64
    return 

def loss(ai):
    I, dl, r_coil, der1, der2, der3 = cal_coil(ai, args)
    r_surf = np.load(args['surface_r'])
    nn = np.load(args['surface_nn'])
    sg = np.load(args['surface_sg'])

    def quadratic_flux(I, dl, r_surf, r_coil, nn, sg):

        B = biotSavart(I ,dl, r_surf, r_coil)  
        B_all = B        
        print('Bmax = ', np.max(np.linalg.norm(abs(B_all), axis=-1)))
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

    def average_length(r_coil):   # 取所有线平均吗？   

        r_coil = r_coil[:, :, 0, 0, :]
        al = np.zeros_like(r_coil)
        al = al.at[:, :-1, :].set(r_coil[:, 1:, :] - r_coil[:, :-1, :])
        al = al.at[:, -1, :].set(r_coil[:, 0, :] - r_coil[:, -1, :])
        return np.sum(np.linalg.norm(al, axis=-1)) / (nc)

    def curvature(der1, der2):
        bottom = np.linalg.norm(der1, axis = -1)**3
        top = np.linalg.norm(np.cross(der1, der2), axis = -1)
        k = top / bottom
        k_mean = np.mean(k)
        k_max = np.max(k)
        return k_mean, k_max
    
    def distance_cc(r_coil):  ### 暂未考虑finite-build
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
        t_max = np.max(t)
        return t_mean, t_max

    Bn = quadratic_flux(I, dl, r_surf, r_coil, nn, sg)
    length = average_length(r_coil)
    k_mean, k_max = curvature(der1, der2)
    dcc = distance_cc(r_coil)
    dcs = distance_cs(r_coil, r_surf)
    t_mean, t_max = torsion(der1, der2, der3)
    print(Bn)
    print(length)
    print(k_mean)
    print(dcc)
    print(dcs)
    print(k_max)
    print(t_mean)
    print(t_max)
    print(wb*Bn + wl*length + wc*k_mean + wdcc*dcc + wdcs*dcs + wcm*k_max )
    return 

def lossvals(ai):
    loss = np.load("/home/nxy/codes/focusadd-spline/results_f/circle/loss_{}.npy".format(ai))
    fig = go.Figure()
    fig.add_scatter(x=np.arange(0, len(loss), 1), y=loss, name='loss', line=dict(width=2))
    fig.update_xaxes(title_text = "iteration",title_font = {"size": 30},title_standoff = 15, tickfont=dict(size=30))
    fig.update_yaxes(title_text = "lossvalue",title_font = {"size": 30},title_standoff = 15, tickfont=dict(size=30))
                    # ,type="log", exponentformat = 'e'
    fig.show()
    return

def poincare(ai):
    def symmetry(r):
        npc = int(nc/nfp)
        rc = np.zeros((nic*2, ns, 3))
        rc = rc.at[:nic, :, :].add(r)
        T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        rc = rc.at[nic:nic*2, :, :].add(np.dot(r, T))
        rc_total = np.zeros((nc, ns, 3))
        rc_total = rc_total.at[:npc, :, :].add(rc)
        for i in range(nfp - 1):        
            theta = 2 * np.pi * (i + 1) / nfp
            T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            rc_total = rc_total.at[npc*(i+1):npc*(i+2), :, :].add(np.dot(rc, T))
        return rc_total

      
    fc = np.load("/home/nxy/codes/focusadd-spline/results_f/circle/fc_{}.npy".format(ai))  
    r0 = [5.8]
    z0 = [0.2]
    lenr = len(r0)
    lenz = len(z0)
    assert lenr == lenz
    theta = np.linspace(0, 2 * np.pi, ns + 1)
    rc =  fourier.compute_r_centroid(fc, nfc, nic, ns, theta)[:, :-1, :]
    rc = symmetry(rc)  

    x = rc[:, :, 0]   
    y = rc[:, :, 1]
    z = rc[:, :, 2]
    I = np.ones(nc)
    for i in range(nfp):
        I = I.at[nic*(2*i+1):nic*(2*i+2)].set(-1)
    name = np.zeros(nc)
    group = np.zeros(nc)

    coil = coilpy.coils.Coil(x, y, z, I, name, group)
    line = coilpy.misc.tracing(coil.bfield, r0, z0, phi0, niter, nfp, nstep)

    line = np.reshape(line, (lenr*(niter+1), 2))
    fig = go.Figure()
    fig.add_scatter(x = line[:, 0], y = line[:, 1],  name='poincare', mode='markers', marker_size = 3)
    fig.update_layout(scene_aspectmode='data')
    fig.show()

    return



# for i in range(10):
#     ai = 'lrm{}'.format(i)
#     lossvals(ai)
#     loss(ai)
#     coil(ai)
    


ai = 'a2'
lossvals(ai)
loss(ai)
coil(ai)
poincare(ai)






# fig = go.Figure()
# for i in range(10):
#     lossm = np.load("/home/nxy/codes/focusadd-spline/results_f/circle/loss_lra{}.npy".format(i))
#     fig.add_scatter(x=np.arange(0, len(lossm), 1), y=lossm, name='loss{}'.format(i), line=dict(width=2))
# fig.update_xaxes(title_text = "iteration",title_font = {"size": 30},title_standoff = 15, tickfont=dict(size=30))
# fig.update_yaxes(title_text = "lossvalue",title_font = {"size": 30},title_standoff = 15, tickfont=dict(size=30)
#                 ,type="log", exponentformat = 'e')
# fig.show()

