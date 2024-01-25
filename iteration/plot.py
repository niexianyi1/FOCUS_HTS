
### 画图程序，在main.py中调用
### 使用plotly进行画图
### 由plot函数控制画图

import jax.numpy as np
import plotly.graph_objects as go
import coilset
import spline
import poincare_B
# import coilpy




def plot(args, coil_all, lossvals, params, I):
    """
    按照init_args中设置进行画图

    Args:
        args : dict, 参数总集
        coil : array,[nc,ns,nn,nb,3], 优化后线圈坐标
        lossvals : list,[ni], 迭代数据
        params : list,[fc,fr], 优化后参数
        I : array,[nc], 每个线圈电流数据

    Returns:
        plot_coil : 线圈坐标图
        plot_loss : 迭代曲线图
        plot_poincare : 庞加莱图
    """

    if args['plot_coil'] != 0 :
        plot_coil(args, params, I)
    if args['plot_loss'] != 0 :
        plot_loss(lossvals)
    if args['plot_poincare'] != 0 :
        plot_poincare(args, coil_all['coil_r'], coil_all['coil_dl'])

    return

def plot_coil(args, params, I):    # 线圈
    """
    画线圈, 同时按照number_points进行加密。

    Args:
        args : dict, 参数总集
        params : list,[fc,fr], 优化后参数
        
    Returns:
        plot_coil == 1 : 线圈曲线, 可以画有限截面的多根曲线
        plot_coil == 2 : 画出有限截面的表面
    """
    if args['coil_case'] == 'spline':
        bc, tj = spline.get_bc_init(args['number_points'], args['number_control_points'])
        args['bc'] = bc
        args['tj'] = tj

    ns = args['number_segments']
    args['number_segments'] = args['number_points'] 
    coil_cal = coilset.CoilSet(args, I)    
    coil = coil_cal.get_coil(params)
    args['number_segments'] = ns

    ns = args['number_points']
    nn = args['number_normal']
    nb = args['number_binormal']
    nic = args['number_independent_coils']

    if args['plot_coil'] == 1 :
        coil = np.reshape(coil[:nic], (ns * nic * nn * nb, 3))
        fig = go.Figure()
        fig.add_scatter3d(x=coil[:, 0],y=coil[:, 1],z=coil[:, 2], name='coil', mode='markers', marker_size = 1.5)   
        fig.update_layout(scene_aspectmode='data')
        fig.show()

    if args['plot_coil'] == 2 :

        rr = np.zeros((5, nic, ns+1, 3))
        rr = rr.at[0,:,:ns,:].set(coil[:nic, 0, 0, :, :])
        rr = rr.at[1,:,:ns,:].set(coil[:nic, 0, nb-1, :, :])
        rr = rr.at[2,:,:ns,:].set(coil[:nic, nn-1, nb-1, :, :])
        rr = rr.at[3,:,:ns,:].set(coil[:nic, nn-1, 0, :, :])
        rr = rr.at[4,:,:ns,:].set(coil[:nic, 0, 0, :, :])
        rr = rr.at[0,:,-1,:].set(coil[:nic, 0, 0, 0, :])
        rr = rr.at[1,:,-1,:].set(coil[:nic, 0, nb-1, 0, :])
        rr = rr.at[2,:,-1,:].set(coil[:nic, nn-1, nb-1, 0, :])
        rr = rr.at[3,:,-1,:].set(coil[:nic, nn-1, 0, 0, :])
        rr = rr.at[4,:,-1,:].set(coil[:nic, 0, 0, 0, :])
        xx = rr[:,:,:,0]
        yy = rr[:,:,:,1]
        zz = rr[:,:,:,2]
        fig = go.Figure()
        for i in range(nic):
            fig.add_trace(go.Surface(x=xx[:,i,:], y=yy[:,i,:], z=zz[:,i,:]))
        fig.update_layout(scene_aspectmode='data')
        fig.show() 

    return 


def plot_loss(lossvals):
    """
    画迭代曲线
    Args:
        lossvals : list,[ni], 迭代数据
        
    """
    fig = go.Figure()
    fig.add_scatter(x = np.arange(0, len(lossvals), 1), y = lossvals, 
                        name = 'lossvalue', line = dict(width=5))
    fig.update_xaxes(title_text = "iteration",title_font = {"size": 25},title_standoff = 12, 
                        tickfont = dict(size=25))
    fig.update_yaxes(title_text = "lossvalue",title_font = {"size": 25},title_standoff = 12, 
                        tickfont = dict(size=25) ,type="log", exponentformat = 'e')
    fig.show()
    return


def plot_poincare(args, coil, dl):
    """
    画线圈, 同时按照number_points进行加密。

    Args:
        args : dict, 参数总集
        coil : array,[nc,ns,nn,nb,3], 优化后线圈坐标
        I : array,[nc], 每个线圈电流数据
    Returns:
        plot_poincare == 1 : poincare图
        
    """
    lenr = len(args['poincare_r0'])
    lenz = len(args['poincare_z0'])
    assert lenr == lenz

    coil = np.mean(coil, axis = (1,2))
    dl = np.mean(dl, axis = (1,2))

    line = poincare_B.tracing(coil, dl, args['poincare_r0'], args['poincare_z0'], args['poincare_phi0'], 
            args['number_iter'], args['number_field_periods'], args['number_step'])

    line = np.reshape(line, (lenr*(args['number_iter']+1), 2))
    surf = np.load("{}".format(args['surface_r_file']))[0]
    fig = go.Figure()
    fig.add_scatter(x = line[:, 0], y = line[:, 1],  name='poincare', mode='markers', marker_size = 1.5)
    fig.add_scatter(x = surf[:, 0], y = surf[:, 2],  name='surface', line = dict(width=2.5))
    fig.update_layout(scene_aspectmode='data')
    fig.show()

    return














    