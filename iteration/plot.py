
### 画图程序，在main.py中调用
### 使用plotly进行画图
### 由plot函数控制画图

import jax.numpy as np
import plotly.graph_objects as go
import coilset
import coilpy




def plot(args, coil, lossvals, params, I):
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
        plot_coil(args, params)
    if args['plot_loss'] != 0 :
        plot_loss(lossvals)
    if args['plot_poincare'] != 0 :
        plot_poincare(args, coil, I)

    return

def plot_coil(args, params):    # 线圈
    """
    画线圈, 同时按照number_points进行加密。

    Args:
        args : dict, 参数总集
        params : list,[fc,fr], 优化后参数
        
    Returns:
        plot_coil == 1 : 线圈曲线, 可以画有限截面的多根曲线
        plot_coil == 2 : 画出有限截面的表面
    """

    args['number_segments'] = args['number_points']
    cal_coil = coilset.CoilSet(args)    
    _,_,coil,_,_,_ = cal_coil.get_coil(params)

    ns = args['number_segments']
    nn = args['number_normal']
    nb = args['number_binormal']
    nic = args['number_independent_coils']

    if args['plot_coil'] == 1 :
        coil = np.reshape(coil, (ns * nic * nn * nb, 3))
        fig = go.Figure()
        fig.add_scatter3d(x=coil[:, 0],y=coil[:, 1],z=coil[:, 2], name='coil', mode='markers', marker_size = 1.5)   
        fig.update_layout(scene_aspectmode='data')
        fig.show()

    if args['plot_coil'] == 2 :

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


def plot_poincare(args, coil, I):
    """
    画线圈, 同时按照number_points进行加密。

    Args:
        args : dict, 参数总集
        coil : array,[nc,ns,nn,nb,3], 优化后线圈坐标
        I : array,[nc], 每个线圈电流数据
    Returns:
        plot_coil == 1 : 线圈曲线, 可以画有限截面的多根曲线
        plot_coil == 2 : 画出有限截面的表面
    """
    lenr = len(args['poincare_r0'])
    lenz = len(args['poincare_z0'])
    assert lenr == lenz
    nc = args['number_coils'] * args['number_normal'] * args['number_binormal']
    coil = np.transpose(coil, (0, 2, 3, 1, 4))  # [nc,nn,nb,ns,3]
    coil = np.reshape(coil, (nc, args['number_segments'], 3))
    x = coil[:, :, 0]   
    y = coil[:, :, 1]
    z = coil[:, :, 2] 

    name = np.zeros(args.nc)
    group = np.zeros(args.nc)

    coil_py = coilpy.coils.Coil(x, y, z, I, name, group)
    line = coilpy.misc.tracing(coil_py.bfield, args['r0'], args['z0'], args['phi0'], 
            args['number_iter'], args['nfp'], args['number_step'])

    line = np.reshape(line, (lenr*(args['number_iter']+1), 2))
    fig = go.Figure()
    fig.add_scatter(x = line[:, 0], y = line[:, 1],  name='poincare', mode='markers', marker_size = 1.5)
    fig.update_layout(scene_aspectmode='data')
    fig.show()

    return














    