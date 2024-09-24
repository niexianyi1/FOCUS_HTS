import jax.numpy as np
from jax import jit, vmap, config
import sys
sys.path.append('opt_coil')
import fourier
import spline
sys.path.append('HTS')
import hts_strain
config.update("jax_enable_x64", True)
pi = np.pi

class Strain_CoilSet:

    """
	Strain_CoilSet is a class which represents all of the coils surrounding a plasma surface. The coils
	are represented by a spline and a fourier series, one for the coil winding pack centroid and
    one for the rotation of the coils. 
	"""

    def __init__(self, args):
        self.args = args
        if 'coil_arg_i' in args.keys():
            self.coil_arg = args['coil_arg_i'][np.newaxis,:,:]
            self.nic = 1
        else:
            self.coil_arg = args['coil_arg']
            self.nic = args['number_independent_coils'] 
        self.ns = args['number_segments']
        self.nr = args['number_rotate']
        self.nfc = args['number_fourier_coils']
        self.nfr = args['number_fourier_rotate']
        self.theta = np.linspace(0, 2 * pi, self.ns + 1)
        return
 

    @jit
    def cal_coil(self, fr):             
        coil_centroid = Strain_CoilSet.compute_coil_centroid(self)  
        der1, der2, der3, dt = Strain_CoilSet.compute_coil_der(self)   
        tangent, normal, binormal = Strain_CoilSet.compute_com(self, der1, coil_centroid)
        centroid_frame = tangent, normal, binormal
        alpha = Strain_CoilSet.compute_alpha(self, fr)
        v1, v2 = Strain_CoilSet.compute_frame(self, alpha, centroid_frame)
        dl = der1[:, np.newaxis, np.newaxis, :, :] * dt
        a = 0 ## 占位置
        return a, dl, a, der1, der2, a, v1, v2, a


    def compute_coil_centroid(self):    
        """
        计算线圈中心位置, 按照所选表达方法计算

        Args:
            self.coil_arg  :   array, 线圈坐标表达式参数

        Returns:
            coil_centroid   :   array, [nc, ns, 3], 线圈中心位置坐标点

        """     
        def compute_coil_fourier(self):
            coil_centroid = fourier.compute_r_centroid(self.coil_arg, self.ns)
            coil_centroid = coil_centroid[:, :-1, :]
            return coil_centroid

        def compute_coil_spline_all(self):
            t, u, k = self.args['bc']
            coil_arg_c = np.zeros((self.nic, 3, self.args['number_control_points']))
            coil_arg_c = coil_arg_c.at[:, :, :-3].set(self.coil_arg)
            coil_arg_c = coil_arg_c.at[:, :, -3:].set(self.coil_arg[:, :, :3])
            coil_centroid = vmap(lambda coil_arg_c :spline.splev(t, u, coil_arg_c, self.args['tj'], self.ns), 
                        in_axes=0, out_axes=0)(coil_arg_c)
            
            return coil_centroid  

        compute_coil = dict()
        compute_coil['fourier'] = compute_coil_fourier
        compute_coil['spline'] = compute_coil_spline_all
        compute_func = compute_coil[self.args['coil_case']]
        coil_centroid = compute_func(self)
        return coil_centroid


    def compute_coil_der(self):  
        """
        计算线圈各阶导数, 按照所选表达方法计算

        Args:
            self.coil_arg  :   array, 线圈坐标表达式参数

        Returns:
            der1, der2, der3   :   array, [nc, ns, 3], 线圈各阶导数

        """   
        def compute_coil_der_fourier(self):          
            der1 = fourier.compute_der1(self.coil_arg, self.ns)
            der2 = fourier.compute_der2(self.coil_arg, self.ns)
            der3 = fourier.compute_der3(self.coil_arg, self.ns)
            der1, der2, der3 = der1[:, :-1, :], der2[:, :-1, :], der3[:, :-1, :]
            dt = 2 * pi / self.ns
            return der1, der2, der3, dt

        def compute_coil_der_spline_all(self):
            t, u, k = self.args['bc']
            coil_arg_c = np.zeros((self.nic, 3, self.args['number_control_points']))
            coil_arg_c = coil_arg_c.at[:, :, :-3].set(self.coil_arg)
            coil_arg_c = coil_arg_c.at[:, :, -3:].set(self.coil_arg[:, :, :3])
            der1, wrk1 = vmap(lambda coil_arg_c :spline.der1_splev(t, u, coil_arg_c, self.args['tj'], self.ns), 
                    in_axes=0, out_axes=0)(coil_arg_c)
            der2 = vmap(lambda wrk1 :spline.der2_splev(t, u, wrk1, self.args['tj'], self.ns), 
                    in_axes=0, out_axes=0)(wrk1)
            der3 = 0
            dt = 1 / self.ns
            return der1, der2, der3, dt

        compute_coil_der = dict()
        compute_coil_der['fourier'] = compute_coil_der_fourier
        compute_coil_der['spline'] = compute_coil_der_spline_all
        compute_func = compute_coil_der[self.args['coil_case']]
        der1, der2, der3, dt = compute_func(self)
        return der1, der2, der3, dt


    def compute_com(self, der1, coil_centroid):    
        """ 取得centroid坐标框架参数 """
        tangent = Strain_CoilSet.compute_tangent(self, der1)
        normal = -Strain_CoilSet.compute_normal(self, coil_centroid, tangent)
        binormal = Strain_CoilSet.compute_binormal(self, tangent, normal)
        return tangent, normal, binormal


    def compute_tangent(self, der1):          
        """
        Computes the tangent vector of the coils. Uses the equation 
        T = dr/d_theta / |dr / d_theta|
        """
        tangent = der1 / np.linalg.norm(der1, axis=-1)[:, :, np.newaxis]
        return tangent

    def dot_product_rank3_tensor(a, b):         # dot
        dotab = (a[:, :, 0] * b[:, :, 0] + 
                 a[:, :, 1] * b[:, :, 1] + 
                 a[:, :, 2] * b[:, :, 2])
        return dotab


    def compute_coil_mid(self, coil_centroid):      # mid_point   r0=[self.nic, 3]
        """得到每个线圈中心点坐标"""
        x = coil_centroid[:, :-1, 0]
        y = coil_centroid[:, :-1, 1]
        z = coil_centroid[:, :-1, 2]
        coil_mid = np.zeros((self.nic, 3))
        for i in range(self.nic):
            coil_mid = coil_mid.at[i, 0].add(np.sum(x[i]) / self.ns)
            coil_mid = coil_mid.at[i, 1].add(np.sum(y[i]) / self.ns)
            coil_mid = coil_mid.at[i, 2].add(np.sum(z[i]) / self.ns)        
        return coil_mid


    def compute_normal(self, coil_centroid, tangent):    
        """计算单位法向量"""
        coil_mid = Strain_CoilSet.compute_coil_mid(self, coil_centroid)
        delta = coil_centroid - coil_mid[:, np.newaxis, :]
        dp = Strain_CoilSet.dot_product_rank3_tensor(tangent, delta)
        normal = delta - tangent * dp[:, :, np.newaxis]
        mag = np.linalg.norm(normal, axis=-1)
        return normal / mag[:, :, np.newaxis]


    def compute_binormal(self, tangent, normal):           
        """ Computes the binormal vector of the coils, B = T x N """
        return np.cross(tangent, normal)


    def compute_alpha(self, fr):    
        """计算有限截面旋转角"""
        alpha = np.zeros((self.nic, self.ns+1))
        alpha += self.theta * self.nr / 2
        Ac = fr[:, 0]
        As = fr[:, 1]
        for m in range(self.nfr):
            arg = self.theta * m
            carg = np.cos(arg)
            sarg = np.sin(arg)
            alpha += (
                Ac[:, np.newaxis, m] * carg[np.newaxis, :]
                + As[:, np.newaxis, m] * sarg[np.newaxis, :]
            )
        return alpha[:, :-1]


    def compute_frame(self, alpha, frame):  
        """
		Computes the vectors v1 and v2 for each coil. v1 and v2 are rotated relative to
		the normal and binormal frame by an amount alpha. Alpha is parametrized by a Fourier series.
		"""
        _, N, B = frame
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)
        v1 = calpha[:, :, np.newaxis] * N + salpha[:, :, np.newaxis] * B
        v2 = -salpha[:, :, np.newaxis] * N + calpha[:, :, np.newaxis] * B
        return v1, v2


    def compute_r(self, v1, v2, coil_centroid):     ### 每个线圈截面独立计算
        """
        Computes the position of the multi-filament coils.

        r is a nc x ns + 1 x nn x nb x 3 array which holds the coil endpoints
        dl is a nc x ns x nn x nb x 3 array which computes the length of the ns segments
        r_middle is a nc x ns x nn x nb x 3 array which computes the midpoint of each of the ns segments

        """
        nn = self.args['number_normal']
        nb = self.args['number_binormal']
        ln = np.array(self.args['length_normal'])
        lb = np.array(self.args['length_binormal'])
        r = np.zeros((self.nic, nn, nb, self.ns, 3))
        r += coil_centroid[:, np.newaxis, np.newaxis, :, :]
        print(ln)
        for n in range(nn):
            for b in range(nb):
                    r = r.at[:, n, b, :, :].add(
                        (n - 0.5 * (nn - 1)) * ln[:, np.newaxis, np.newaxis] * v1 + 
                        (b - 0.5 * (nb - 1)) * lb[:, np.newaxis, np.newaxis] * v2
                    ) 
        return r


    def compute_cd(self, der1, der2):
        curva = np.cross(der1, der2) / (np.linalg.norm(der1, axis = -1)**3)[:,:,np.newaxis]
        return curva
    def get_coil(self, params):
        coil_arg, fr, I = params   
        self.coil_arg = coil_arg
        coil_centroid = Strain_CoilSet.compute_coil_centroid(self)  
        der1, _, _, _ = Strain_CoilSet.compute_coil_der(self)   
        tangent, normal, binormal = Strain_CoilSet.compute_com(self, der1, coil_centroid)
        centroid_frame = tangent, normal, binormal
        alpha = Strain_CoilSet.compute_alpha(self, fr)
        v1, v2 = Strain_CoilSet.compute_frame(self, alpha, centroid_frame)
        r = Strain_CoilSet.compute_r(self, v1, v2, coil_centroid)
        return r


    def get_plot_strain(self, params):
        coil_arg, fr, I = params  
        self.coil_arg = coil_arg
        coil_centroid = Strain_CoilSet.compute_coil_centroid(self)  
        der1, der2, _, dt = Strain_CoilSet.compute_coil_der(self)   
        tangent, normal, binormal = Strain_CoilSet.compute_com(self, der1, coil_centroid)
        centroid_frame = tangent, normal, binormal
        alpha = Strain_CoilSet.compute_alpha(self, fr)
        v1, v2 = Strain_CoilSet.compute_frame(self, alpha, centroid_frame)
        curva = Strain_CoilSet.compute_cd(self, der1, der2)
        dl = der1[:, np.newaxis, np.newaxis, :, :] * dt
        strain = hts_strain.HTS_strain(self.args, curva, v1, v2, dl)
        return strain