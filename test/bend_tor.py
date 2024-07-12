import json
import jax.numpy as np
import read_init
from jax import jit, vmap
import fourier
import spline
pi = np.pi
import plotly.graph_objects as go

with open('/home/nxy/codes/coil_spline_HTS/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)

args['theta'] = np.linspace(0, 2 * pi, args['number_segments'] + 1)


 


def cal_coil(args, params):             
    """
    计算线圈数据, 输出给lossfunction

    Args:
        params  :   list, [coil_arg, fr], 优化参数

    Returns:
        I_new   :   array, [nc*nn*nb], 线圈电流, 考虑仿星器对称和有限截面
        dl      :   array, [nc, ns, nn, nb, 3], 计算biot-savart的dl项  
        r       :   array, [nc, ns, nn, nb, 3], 有限截面线圈坐标
        der1, der2, der3 : array, [nc, ns, 3], 中心点线圈各阶导数值
    """
    coil_arg, fr = params   
    I_new = I / (args['number_normal'] * args['number_binormal'])
    coil_centroid = compute_coil_centroid(args, coil_arg)  
    der1, der2, der3 = compute_der(args, coil_arg)   
    tangent, normal, binormal = compute_com(args, der1, coil_centroid)
    v1, v2 = compute_frame(args, fr, normal, binormal)
    r = compute_r(args, fr, normal, binormal, coil_centroid)
    frame = tangent, normal, binormal
    dl = compute_dl(args, params, frame, der1, der2, coil_centroid)
    if args['stellarator_symmetry'] == 1 :
        r = stellarator_symmetry_coil(args, r)
        dl = stellarator_symmetry_coil(args, dl)
        I_new = stellarator_symmetry_I(args, I_new)
    r = symmetry_coil(args, r)
    dl = symmetry_coil(args, dl)
    I_new = symmetry_I(args, I_new)

    return I_new, dl, r, der1, der2, der3, v1

def compute_coil_centroid(args, coil_arg):    
    """
    计算线圈中心位置, 按照所选表达方法计算

    Args:
        coil_arg  :   array, 线圈坐标表达式参数

    Returns:
        coil_centroid   :   array, [nc, ns, 3], 线圈中心位置坐标点

    """     
    if args['coil_case'] == 'fourier':        
        coil_centroid = fourier.compute_r_centroid(coil_arg, args['number_fourier_coils'], args['number_independent_coils'], args['number_segments'], args['theta'])
        coil_centroid = coil_centroid[:, :-1, :]
    if args['coil_case'] == 'spline':
        t, u, k = args['bc']
        coil_centroid = vmap(lambda c :spline.splev(t, u, coil_arg, args['tj'], args['number_segments']), 
                in_axes=0, out_axes=0)(coil_arg)
    
    return coil_centroid

def compute_der(args, coil_arg):  
    """
    计算线圈各阶导数, 按照所选表达方法计算

    Args:
        coil_arg  :   array, 线圈坐标表达式参数

    Returns:
        der1, der2, der3   :   array, [nc, ns, 3], 线圈各阶导数

    """   
    if args['coil_case'] == 'fourier':          
        der1 = fourier.compute_der1(coil_arg, args['number_fourier_coils'], args['number_independent_coils'], args['number_segments'], args['theta'])
        der2 = fourier.compute_der2(coil_arg, args['number_fourier_coils'], args['number_independent_coils'], args['number_segments'], args['theta'])
        der3 = fourier.compute_der3(coil_arg, args['number_fourier_coils'], args['number_independent_coils'], args['number_segments'], args['theta'])
        der1, der2, der3 = der1[:, :-1, :], der2[:, :-1, :], der3[:, :-1, :]
    if args['coil_case'] == 'spline':
        t, u, k = args['bc']
        der1, wrk1 = vmap(lambda coil_arg :spline.splev(t, u, coil_arg, args['tj'], args['number_segments']), 
                in_axes=0, out_axes=0)(coil_arg)
        der2 = vmap(lambda wrk1 :spline.splev(t, u, wrk1, args['tj'], args['number_segments']), 
                in_axes=0, out_axes=0)(wrk1)

    return der1, der2, der3
    
def compute_com(args, der1, coil_centroid):    
    """ 取得centroid坐标框架参数 """
    tangent = compute_tangent(args, der1)
    normal = -compute_normal(args, coil_centroid, tangent)
    binormal = compute_binormal(args, tangent, normal)
    return tangent, normal, binormal

def compute_com_deriv(args, frame, der1, der2, coil_centroid): 
    """ 取得centroid坐标框架参数的导数 """
    tangent, normal, _ = frame
    tangent_deriv = compute_tangent_deriv(args, der1, der2)
    normal_deriv = -compute_normal_deriv(args, tangent, tangent_deriv, der1, coil_centroid)
    binormal_deriv = compute_binormal_deriv(args, tangent, normal, tangent_deriv, normal_deriv)
    return tangent_deriv, normal_deriv, binormal_deriv

def compute_tangent(args, der1):          
    """
    Computes the tangent vector of the coils. Uses the equation 
    T = dr/d_theta / |dr / d_theta|
    """
    tangent = der1 / np.linalg.norm(der1, axis=-1)[:, :, np.newaxis]
    return tangent

def compute_tangent_deriv(args, der1, der2):   
    """
    计算切向量微分
    dT = der2/|der1| - der1 / dot(der1,der2)
    """
    norm_der1 = np.linalg.norm(der1, axis=-1)
    mag_2 = dot_product_rank3_tensor(der1, der2) / norm_der1 ** 3
    tangent_deriv = (der2 / norm_der1[:, :, np.newaxis] - 
                                der1 * mag_2[:, :, np.newaxis])
    
    return tangent_deriv

def dot_product_rank3_tensor(a, b):         # dot
    dotab = (a[:, :, 0] * b[:, :, 0] + 
                a[:, :, 1] * b[:, :, 1] + 
                a[:, :, 2] * b[:, :, 2])
    return dotab

def compute_coil_mid(args, coil_centroid):      # mid_point   r0=[args['number_independent_coils'], 3]
    """得到每个线圈中心点坐标"""
    x = coil_centroid[:, :-1, 0]
    y = coil_centroid[:, :-1, 1]
    z = coil_centroid[:, :-1, 2]
    coil_mid = np.zeros((args['number_independent_coils'], 3))
    for i in range(args['number_independent_coils']):
        coil_mid = coil_mid.at[i, 0].add(np.sum(x[i]) / args['number_segments'])
        coil_mid = coil_mid.at[i, 1].add(np.sum(y[i]) / args['number_segments'])
        coil_mid = coil_mid.at[i, 2].add(np.sum(z[i]) / args['number_segments'])        
    return coil_mid

def compute_normal(args, coil_centroid, tangent):    
    """计算单位法向量"""
    coil_mid = compute_coil_mid(args, coil_centroid)
    delta = coil_centroid - coil_mid[:, np.newaxis, :]
    dp = dot_product_rank3_tensor(tangent, delta)
    normal = delta - tangent * dp[:, :, np.newaxis]
    mag = np.linalg.norm(normal, axis=-1)
    return normal / mag[:, :, np.newaxis]

def compute_normal_deriv(args, tangent, tangent_deriv, der1, coil_centroid):  
    """计算单位法向量微分"""  
    coil_mid = compute_coil_mid(args, coil_centroid)
    delta = coil_centroid - coil_mid[:, np.newaxis, :]
    dp1 = dot_product_rank3_tensor(tangent, delta)
    dp2 = dot_product_rank3_tensor(tangent, der1)
    dp3 = dot_product_rank3_tensor(tangent_deriv, delta)
    numerator = delta - tangent * dp1[:, :, np.newaxis]
    numerator_norm = np.linalg.norm(numerator, axis=-1)
    numerator_deriv = (
        der1
        - dp1[:, :, np.newaxis] * tangent_deriv
        - tangent * (dp2 + dp3)[:, :, np.newaxis]
    )
    dp4 = dot_product_rank3_tensor(numerator, numerator_deriv)
    return (
        numerator_deriv / numerator_norm[:, :, np.newaxis]
        - (dp4 / numerator_norm ** 3)[:, :, np.newaxis] * numerator
    )

def compute_binormal(args, tangent, normal):           
    """ Computes the binormal vector of the coils, B = T x N """
    return np.cross(tangent, normal)

def compute_binormal_deriv(args, tangent, normal, tangent_deriv, normal_deriv):  
    return np.cross(tangent_deriv, normal) + np.cross(tangent, normal_deriv)

def compute_alpha(args, fr):    
    """计算有限截面旋转角"""
    alpha = np.zeros((args['number_independent_coils'], args['number_segments']+1))
    alpha += args['theta'] * args['number_rotate'] / 2
    Ac = fr[0]
    As = fr[1]
    for m in range(args['number_fourier_rotate']):
        arg = args['theta'] * m
        carg = np.cos(arg)
        sarg = np.sin(arg)
        alpha += (
            Ac[:, np.newaxis, m] * carg[np.newaxis, :]
            + As[:, np.newaxis, m] * sarg[np.newaxis, :]
        )
    return alpha[:, :-1]

def compute_alpha_1(args, fr):   
    """计算有限截面旋转角的导数""" 
    alpha_1 = np.zeros((args['number_independent_coils'], args['number_segments']+1 ))
    alpha_1 += args['number_rotate'] / 2
    Ac = fr[0]
    As = fr[1]
    for m in range(args['number_fourier_rotate']):
        arg = args['theta'] * m
        carg = np.cos(arg)
        sarg = np.sin(arg)
        alpha_1 += (
            -m * Ac[:, np.newaxis, m] * sarg[np.newaxis, :]
            + m * As[:, np.newaxis, m] * carg[np.newaxis, :]
        )
    return alpha_1[:, :-1]

def compute_frame(args, fr, N, B):  
    """
    Computes the vectors v1 and v2 for each coil. v1 and v2 are rotated relative to
    the normal and binormal frame by an amount alpha. Alpha is parametrized by a Fourier series.
    """
    alpha = compute_alpha(args, fr)
    calpha = np.cos(alpha)
    salpha = np.sin(alpha)
    v1 = calpha[:, :, np.newaxis] * N - salpha[:, :, np.newaxis] * B
    v2 = salpha[:, :, np.newaxis] * N + calpha[:, :, np.newaxis] * B
    return v1, v2

def compute_frame_derivative(args, params, frame, der1, der2, coil_centroid): 

    _, N, B = frame
    _, fr = params
    alpha = compute_alpha(args, fr)
    calpha = np.cos(alpha)
    salpha = np.sin(alpha)
    alpha1 = compute_alpha_1(args, fr)
    _, dNdt, dBdt = compute_com_deriv(args, frame, der1, der2, coil_centroid)
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

def compute_r(args, fr, normal, binormal, coil_centroid):      
    """
    Computes the position of the multi-filament coils.

    r is a nc x ns + 1 x nn x nb x 3 array which holds the coil endpoints
    dl is a nc x ns x nn x nb x 3 array which computes the length of the ns segments
    r_middle is a nc x ns x nn x nb x 3 array which computes the midpoint of each of the ns segments

    """

    v1, v2 = compute_frame(args, fr, normal, binormal)
    r = np.zeros((args['number_independent_coils'], args['number_segments'], args['number_normal'], args['number_binormal'], 3))
    r += coil_centroid[:, :, np.newaxis, np.newaxis, :]
    for n in range(args['number_normal']):
        for b in range(args['number_binormal']):
            r = r.at[:, :, n, b, :].add(
                (n - 0.5 * (args['number_normal'] - 1)) * args['length_normal'] * v1 + 
                (b - 0.5 * (args['number_binormal'] - 1)) * args['length_binormal'] * v2
            ) 
    return r

def compute_dl(args, params, frame, der1, der2, coil_centroid):   
    dl = np.zeros((args['number_independent_coils'], args['number_segments'], args['number_normal'], args['number_binormal'], 3))
    dl += der1[:, :, np.newaxis, np.newaxis, :]
    dv1_dt, dv2_dt = compute_frame_derivative(args, params, frame, der1, der2, coil_centroid)
    for n in range(args['number_normal']):
        for b in range(args['number_binormal']):
            dl = dl.at[:, :, n, b, :].add(
                (n - 0.5 * (args['number_normal'] - 1)) * args['length_normal'] * dv1_dt + 
                (b - 0.5 * (args['number_binormal'] - 1)) * args['length_binormal'] * dv2_dt
            )

    return dl * (2 * pi / args['number_segments'])


def stellarator_symmetry_coil(args, r):
    """计算线圈的仿星器对称"""
    rc = np.zeros((args['number_independent_coils']*2, args['number_segments'], args['number_normal'], args['number_binormal'], 3))
    rc = rc.at[0:args['number_independent_coils'], :, :, :, :].set(r)
    T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rc = rc.at[args['number_independent_coils']:args['number_independent_coils']*2, :, :, :, :].set(np.dot(r, T))
    return rc

def symmetry_coil(args, r):
    """计算线圈的周期对称"""
    npc = int(args['number_coils'] / args['number_field_periods'])   # 每周期线圈数，number of coils per period
    rc_total = np.zeros((args['number_coils'], args['number_segments'], args['number_normal'], args['number_binormal'], 3))
    rc_total = rc_total.at[0:npc, :, :, :, :].set(r)
    for i in range(args['number_field_periods'] - 1):        
        theta_t = 2 * pi * (i + 1) / args['number_field_periods']
        T = np.array([[np.cos(theta_t), -np.sin(theta_t), 0],
                [np.sin(theta_t), np.cos(theta_t), 0], [0, 0, 1]])
        rc_total = rc_total.at[npc*(i+1):npc*(i+2), :, :, :, :].set(np.dot(r, T))
    
    return rc_total

def stellarator_symmetry_I(args, I):
    """计算电流的仿星器对称"""
    I_new = np.zeros(args['number_independent_coils']*2)
    I_new = I_new.at[:args['number_independent_coils']].set(I)
    for i in range(args['number_independent_coils']):
        I_new = I_new.at[i+args['number_independent_coils']].set(-I[i])

    return I_new

def symmetry_I(args, I):
    """计算电流的周期对称"""
    npc = int(args['number_coils'] / args['number_field_periods'])
    I_new = np.zeros(args['number_coils'])
    for i in range(args['number_field_periods']):
        I_new = I_new.at[npc*i:npc*(i+1)].set(I)
    return I_new

def average_length(args, coil):      #new
    nic = args['number_independent_coils']   
    al = np.zeros((nic, coil.shape[1], 3))   
    r_coil = np.mean(coil, axis = (2,3))
    al = al.at[:, :-1, :].set(r_coil[:nic, 1:, :] - r_coil[:nic, :-1, :])
    al = al.at[:, -1, :].set(r_coil[:nic, 0, :] - r_coil[:nic, -1, :])
    len = np.sum(np.linalg.norm(al, axis=-1)) / nic
    return len, al

def curvature(der1, der2):
    bottom = np.linalg.norm(der1, axis = -1)**3
    top = np.linalg.norm(np.cross(der1, der2), axis = -1)
    k = abs(top / bottom)
    k_mean = np.mean(k)
    k_max = np.max(k)
    curva = np.cross(der1, der2) / (np.linalg.norm(der1, axis = -1)**3)[:,:,np.newaxis]
    return k_mean, k_max, curva

def HTS_strain_bend(args, curva, v1):

    """弯曲应变,
    Args:
        w, 带材宽度
        v1,有限截面坐标轴
        curva, 线圈曲率

    Returns:
        bend, 弯曲应变

    """
    bend = args['width']/2*abs(np.sum(-v1 * curva, axis=-1))

    fig = go.Figure()
    for i in range(5):
        fig.add_scatter(x=np.arange(0, args['number_segments'], 1),y=bend[i, :],
            name='bend{}'.format(i), line = dict(width=2))   
    fig.show()

    return bend


def HTS_strain_tor(args, deltal, v1):
    """扭转应变,
    Args:
        w, 带材宽度
        v1,有限截面坐标轴
        deltal, 线圈点间隔

    Returns:
        bend, 弯曲应变

    """
    dv = np.zeros((v1.shape[0], v1.shape[1]))
    dv = dv.at[:, :-1].set(np.sum(v1[:, :-1, :] * v1[:, 1:, :], axis=-1))
    dv = dv.at[:, -1].set(np.sum(v1[:, -1, :] * v1[:, 0, :], axis=-1))
    dtheta = np.arccos(dv)
    deltal = np.linalg.norm(deltal, axis=-1)
    tor = args['width']**2/12*(dtheta/deltal)**2
    fig = go.Figure()
    for i in range(5):
        fig.add_scatter(x=np.arange(0, args['number_segments'], 1),y=tor[i, :],
            name='tor{}'.format(i), line = dict(width=2))   
    fig.show()
    return tor


args, coil_arg_init, fr_init, surface_data, I = read_init.init(args)
params = (coil_arg_init, fr_init)

I_new, dl, coil, der1, der2, der3, v1 = cal_coil(args, params)

k_mean, k_max, curva = curvature(der1, der2)
length, deltal = average_length(args, coil)
bend = HTS_strain_bend(args, curva, v1)

tor = HTS_strain_tor(args, deltal, v1)
nic=5
ns=64
nn=nb=2
rr = np.zeros((nic, ns+1, 5, 3))
rr = rr.at[:,:ns,0,:].set(coil[:nic, :, 0, 0, :])
rr = rr.at[:,:ns,1,:].set(coil[:nic, :, 0, nb-1, :])
rr = rr.at[:,:ns,2,:].set(coil[:nic, :, nn-1, nb-1, :])
rr = rr.at[:,:ns,3,:].set(coil[:nic, :, nn-1, 0, :])
rr = rr.at[:,:ns,4,:].set(coil[:nic, :, 0, 0, :])
rr = rr.at[:,-1,0,:].set(coil[:nic, 0, 0, 0, :])
rr = rr.at[:,-1,1,:].set(coil[:nic, 0, 0, nb-1, :])
rr = rr.at[:,-1,2,:].set(coil[:nic, 0, nn-1, nb-1, :])
rr = rr.at[:,-1,3,:].set(coil[:nic, 0, nn-1, 0, :])
rr = rr.at[:,-1,4,:].set(coil[:nic, 0, 0, 0, :])
rr = np.transpose(rr, [2, 0, 1, 3])     # (5,nic,ns)
xx = rr[:,:,:,0]
yy = rr[:,:,:,1]
zz = rr[:,:,:,2]
fig = go.Figure()
for i in range(5):
    fig.add_trace(go.Surface(x=xx[:,i,:], y=yy[:,i,:], z=zz[:,i,:]))
fig.update_layout(scene_aspectmode='data')
fig.show() 








