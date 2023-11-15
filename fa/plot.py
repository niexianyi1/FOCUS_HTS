import jax.numpy as np
from jax import vmap, jit
import plotly.express as px
import plotly.graph_objects as go
import scipy.interpolate as si 
import coilpy
import bspline 


class plot:

    def __init__(self, args) -> None:
        self.args = args
        self.nc = args['nc']
        self.nfp = args['nfp']
        self.nps = args['nps']
        self.init = args['init']
        self.init_coil = args['init_coil']
        self.n = args['n']
        self.log = args['log']
        self.ns = args['ns']
        self.r0 = args['r0']
        self.z0 = args['z0']
        self.I = args['I']
        self.phi0 = args['phi0']
        self.niter = args['niter']
        self.nstep = args['nstep']
        self.nz = args['nz']
        self.nt = args['nt']
        self.ncnfp = int(self.nc/self.nfp)

        return

    ### 传入的参数c，不包含重复点，即为ns，主动添加重复后画图；
    ### 或者只在迭代结束后画图时，将参数c转为闭合
    def plot_coils(self, c):    # 线圈
        bc = bspline.get_bc_init(c.shape[2])
        t, u, k = bc
        # ----- rc_bspline -----
        u = np.linspace(0, 1, self.nps)
        rc_bspline = np.zeros((self.nc, 3, self.nps))
        tck = [[0]*3 for i in range (self.nc)]
        for i in range(self.nc):
            tck[i] = [t, c[i], k]
            rc_bspline = rc_bspline.at[i, :, :].set(si.splev(u, tck[i]))  

        rc_bspline = np.transpose(rc_bspline, (0, 2, 1))
        rc_bspline = np.reshape(rc_bspline, (self.nps*self.nc, 3))
        
        # np.save('/home/nxy/codes/FOCUSADD_B/results/bnormal/w7x_highres_rc_1000.npy', rc_bspline)   # 保存数据
        
        # ----- rc_initial -----       # 如果从已优化线圈开始，可以进行对比
        if self.init :
            rc_init = np.load("{}".format(self.init_coil)) # 读取输入线圈/init_coil
            rc_initial = np.zeros((self.nc, self.nps, 3))
            for i in range(self.nc):
                tck, u = si.splprep([rc_init[i, :, 0], rc_init[i, :, 1], rc_init[i, :, 2]], s=0)
                u = np.linspace(0, 1, self.nps)
                rc_new = np.array(si.splev(u, tck))
                rc_new = np.transpose(rc_new, (1, 0))
                rc_initial = rc_initial.at[i, :, :].set(rc_new)       
            rc_initial = np.reshape(rc_initial, (self.nps*self.nc, 3))

        # nc = nc/nfp

        # ----- plot -----
        if self.init :
            fig = go.Figure()
            fig.add_scatter3d(x=rc_initial[:self.nc*self.nps, 0],y=rc_initial[:self.nc*self.nps, 1],z=rc_initial[:self.nc*self.nps, 2], name='rc_initial', mode='markers', marker_size = 1)
            fig.add_scatter3d(x=rc_bspline[:self.nc*self.nps, 0],y=rc_bspline[:self.nc*self.nps, 1],z=rc_bspline[:self.nc*self.nps, 2], name='rc_bspline', mode='markers', marker_size = 1)   
            fig.update_layout(scene_aspectmode='data')
            fig.show()
        else:
            fig = go.Figure()
            fig.add_scatter3d(x=rc_bspline[:self.nc*self.nps, 0],y=rc_bspline[:self.nc*self.nps, 1],z=rc_bspline[:self.nc*self.nps, 2], name='rc_bspline', mode='markers', marker_size = 1)   
            fig.update_layout(scene_aspectmode='data')
            fig.show()

        return 


    def plot_loss(self, lossvals):

        fig = go.Figure()
        fig.add_scatter(x=np.linspace(1, self.n, self.n), y=lossvals, name='lossvalue', line=dict(width=5))
        fig.update_xaxes(title_text = 'iteration', )
        if self.log :
            fig.update_yaxes(title_text = 'lossvalue', type="log")
        else:
            fig.update_yaxes(title_text = 'lossvalue')
        fig.show()
        return


    def poincare(self, c):

        lenr = len(self.r0)
        lenz = len(self.z0)
        assert lenr == lenz
        bc = bspline.get_bc_init(c.shape[2])
        rc = vmap(lambda c :bspline.splev(bc, c), in_axes=0, out_axes=0)(c)
        rc = plot.symmetry(self, rc)  
        x = rc[:, :, 0]   
        y = rc[:, :, 1]
        z = rc[:, :, 2]

        if self.I == None:
            self.I = np.ones((self.nc, self.ns))
        name = np.zeros(self.nc)
        group = np.zeros(self.nc)

        coil = coilpy.coils.Coil(x, y, z, self.I, name, group)
        line = coilpy.misc.tracing(coil.bfield, self.r0, self.z0, self.phi0, self.niter, self.nfp, self.nstep)

        line = np.reshape(line, (lenr*(self.niter+1), 2))
        fig = go.Figure()
        fig.add_scatter(x = line[:, 0], y = line[:, 1],  name='poincare', mode='markers', marker_size = 1.5)
        fig.update_layout(scene_aspectmode='data')
        fig.show()

        return

    # def plot_bzbt(self):
    #     bzbt.bzbt(self.args)
    #     return

    @jit
    def symmetry(self, r):
        rc_total = np.zeros((self.nc, self.ns, 3))
        rc_total = rc_total.at[0:self.ncnfp, :, :].add(r)
        for i in range(self.nfp - 1):        
            theta = 2 * np.pi * (i + 1) / self.nfp
            T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            rc_total = rc_total.at[self.ncnfp*(i+1):self.ncnfp*(i+2), :, :].add(np.dot(r, T))
        
        return rc_total













    