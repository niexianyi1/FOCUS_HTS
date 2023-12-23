import jax.numpy as np
from jax import jit, vmap
import numpy
from jax.config import config
import fourier
import spline
config.update("jax_enable_x64", True)
pi = np.pi

class CoilSet:

    """
	CoilSet is a class which represents all of the coils surrounding a plasma surface. The coils
	are represented by a spline and a fourier series, one for the coil winding pack centroid and
    one for the rotation of the coils. 
	"""

    def __init__(self, args):
        self.args = args
        self.nc = args['number_coils']
        self.nfp = args['number_field_periods']
        self.ss = args['stellarator_symmetry']
        self.nic = args['number_independent_coils']
        self.ns = args['number_segments']
        self.ln = args['length_normal']
        self.lb = args['length_binormal']
        self.nn = args['number_normal']
        self.nb = args['number_binormal']
        self.nr = args['number_rotate']
        self.nfc = args['num_fourier_coils']
        self.nfr = args['number_fourier_rotate']
        self.theta = np.linspace(0, 2 * pi, self.ns + 1)
        return
 

    @jit
    def cal_coil(self, params):             
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
        I_new = self.args['I'] / (self.nn * self.nb)
        coil_centroid = CoilSet.compute_coil_centroid(self, coil_arg)  
        der1, der2, der3 = CoilSet.compute_der(self, coil_arg)   
        tangent, normal, binormal = CoilSet.compute_com(self, der1, coil_centroid)
        v1, v2 = CoilSet.compute_frame(self, fr, normal, binormal)
        r = CoilSet.compute_r(self, fr, normal, binormal, coil_centroid)
        frame = tangent, normal, binormal
        dl = CoilSet.compute_dl(self, params, frame, der1, der2, coil_centroid)
        if self.ss == 1 :
            r = CoilSet.stellarator_symmetry_coil(self, r)
            dl = CoilSet.stellarator_symmetry_coil(self, dl)
            I_new = CoilSet.stellarator_symmetry_I(self, I_new)
        r = CoilSet.symmetry_coil(self, r)
        dl = CoilSet.symmetry_coil(self, dl)
        I_new = CoilSet.symmetry_I(self, I_new)
        I_new = CoilSet.finite_build_I(self, I_new)
        return I_new, dl, r, der1, der2, der3, v1

    def compute_coil_centroid(self, coil_arg):    
        """
        计算线圈中心位置, 按照所选表达方法计算

        Args:
            coil_arg  :   array, 线圈坐标表达式参数

        Returns:
            coil_centroid   :   array, [nc, ns, 3], 线圈中心位置坐标点

        """     
        if self.args['coil_case'] == 'fourier':        
            coil_centroid = fourier.compute_r_centroid(coil_arg, self.nfc, self.nic, self.ns, self.theta)
            coil_centroid = coil_centroid[:, :-1, :]
        if self.args['coil_case'] == 'spline':
            t, u, k = self.args['bc']
            coil_centroid = vmap(lambda c :spline.splev(t, u, coil_arg, self.args['tj'], self.ns), 
                    in_axes=0, out_axes=0)(coil_arg)
        
        return coil_centroid

    def compute_der(self, coil_arg):  
        """
        计算线圈各阶导数, 按照所选表达方法计算

        Args:
            coil_arg  :   array, 线圈坐标表达式参数

        Returns:
            der1, der2, der3   :   array, [nc, ns, 3], 线圈各阶导数

        """   
        if self.args['coil_case'] == 'fourier':          
            der1 = fourier.compute_der1(coil_arg, self.nfc, self.nic, self.ns, self.theta)
            der2 = fourier.compute_der2(coil_arg, self.nfc, self.nic, self.ns, self.theta)
            der3 = fourier.compute_der3(coil_arg, self.nfc, self.nic, self.ns, self.theta)
            der1, der2, der3 = der1[:, :-1, :], der2[:, :-1, :], der3[:, :-1, :]
        if self.args['coil_case'] == 'spline':
            t, u, k = self.args['bc']
            der1, wrk1 = vmap(lambda coil_arg :spline.splev(t, u, coil_arg, self.args['tj'], self.ns), 
                    in_axes=0, out_axes=0)(coil_arg)
            der2 = vmap(lambda wrk1 :spline.splev(t, u, wrk1, self.args['tj'], self.ns), 
                    in_axes=0, out_axes=0)(wrk1)

        return der1, der2, der3
        
    def compute_com(self, der1, coil_centroid):    
        """ 取得centroid坐标框架参数 """
        tangent = CoilSet.compute_tangent(self, der1)
        normal = -CoilSet.compute_normal(self, coil_centroid, tangent)
        binormal = CoilSet.compute_binormal(self, tangent, normal)
        return tangent, normal, binormal

    def compute_com_deriv(self, frame, der1, der2, coil_centroid): 
        """ 取得centroid坐标框架参数的导数 """
        tangent, normal, _ = frame
        tangent_deriv = CoilSet.compute_tangent_deriv(self, der1, der2)
        normal_deriv = -CoilSet.compute_normal_deriv(self, tangent, tangent_deriv, der1, coil_centroid)
        binormal_deriv = CoilSet.compute_binormal_deriv(self, tangent, normal, tangent_deriv, normal_deriv)
        return tangent_deriv, normal_deriv, binormal_deriv

    def compute_tangent(self, der1):          
        """
        Computes the tangent vector of the coils. Uses the equation 
        T = dr/d_theta / |dr / d_theta|
        """
        tangent = der1 / np.linalg.norm(der1, axis=-1)[:, :, np.newaxis]
        return tangent

    def compute_tangent_deriv(self, der1, der2):   
        """
        计算切向量微分
        dT = der2/|der1| - der1 / dot(der1,der2)
        """
        norm_der1 = np.linalg.norm(der1, axis=-1)
        mag_2 = CoilSet.dot_product_rank3_tensor(der1, der2) / norm_der1 ** 3
        tangent_deriv = (der2 / norm_der1[:, :, np.newaxis] - 
                                    der1 * mag_2[:, :, np.newaxis])
        
        return tangent_deriv

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
        coil_mid = CoilSet.compute_coil_mid(self, coil_centroid)
        delta = coil_centroid - coil_mid[:, np.newaxis, :]
        dp = CoilSet.dot_product_rank3_tensor(tangent, delta)
        normal = delta - tangent * dp[:, :, np.newaxis]
        mag = np.linalg.norm(normal, axis=-1)
        return normal / mag[:, :, np.newaxis]

    def compute_normal_deriv(self, tangent, tangent_deriv, der1, coil_centroid):  
        """计算单位法向量微分"""  
        coil_mid = CoilSet.compute_coil_mid(self, coil_centroid)
        delta = coil_centroid - coil_mid[:, np.newaxis, :]
        dp1 = CoilSet.dot_product_rank3_tensor(tangent, delta)
        dp2 = CoilSet.dot_product_rank3_tensor(tangent, der1)
        dp3 = CoilSet.dot_product_rank3_tensor(tangent_deriv, delta)
        numerator = delta - tangent * dp1[:, :, np.newaxis]
        numerator_norm = np.linalg.norm(numerator, axis=-1)
        numerator_deriv = (
            der1
            - dp1[:, :, np.newaxis] * tangent_deriv
            - tangent * (dp2 + dp3)[:, :, np.newaxis]
        )
        dp4 = CoilSet.dot_product_rank3_tensor(numerator, numerator_deriv)
        return (
            numerator_deriv / numerator_norm[:, :, np.newaxis]
            - (dp4 / numerator_norm ** 3)[:, :, np.newaxis] * numerator
        )

    def compute_binormal(self, tangent, normal):           
        """ Computes the binormal vector of the coils, B = T x N """
        return np.cross(tangent, normal)

    def compute_binormal_deriv(self, tangent, normal, tangent_deriv, normal_deriv):  
        return np.cross(tangent_deriv, normal) + np.cross(tangent, normal_deriv)

    def compute_alpha(self, fr):    
        """计算有限截面旋转角"""
        alpha = np.zeros((self.nic, self.ns+1))
        alpha += self.theta * self.nr / 2
        Ac = fr[0]
        As = fr[1]
        for m in range(self.nfr):
            arg = self.theta * m
            carg = np.cos(arg)
            sarg = np.sin(arg)
            alpha += (
                Ac[:, np.newaxis, m] * carg[np.newaxis, :]
                + As[:, np.newaxis, m] * sarg[np.newaxis, :]
            )
        return alpha[:, :-1]

    def compute_alpha_1(self, fr):   
        """计算有限截面旋转角的导数""" 
        alpha_1 = np.zeros((self.nic, self.ns+1 ))
        alpha_1 += self.nr / 2
        Ac = fr[0]
        As = fr[1]
        for m in range(self.nfr):
            arg = self.theta * m
            carg = np.cos(arg)
            sarg = np.sin(arg)
            alpha_1 += (
                -m * Ac[:, np.newaxis, m] * sarg[np.newaxis, :]
                + m * As[:, np.newaxis, m] * carg[np.newaxis, :]
            )
        return alpha_1[:, :-1]

    def compute_frame(self, fr, N, B):  
        """
		Computes the vectors v1 and v2 for each coil. v1 and v2 are rotated relative to
		the normal and binormal frame by an amount alpha. Alpha is parametrized by a Fourier series.
		"""
        alpha = CoilSet.compute_alpha(self, fr)
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)
        v1 = calpha[:, :, np.newaxis] * N - salpha[:, :, np.newaxis] * B
        v2 = salpha[:, :, np.newaxis] * N + calpha[:, :, np.newaxis] * B
        return v1, v2

    def compute_frame_derivative(self, params, frame, der1, der2, coil_centroid): 

        _, N, B = frame
        _, fr = params
        alpha = CoilSet.compute_alpha(self, fr)
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)
        alpha1 = CoilSet.compute_alpha_1(self, fr)
        _, dNdt, dBdt = CoilSet.compute_com_deriv(self, frame, der1, der2, coil_centroid)
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

    def compute_r(self, fr, normal, binormal, coil_centroid):      
        """
        Computes the position of the multi-filament coils.

        r is a nc x ns + 1 x nn x nb x 3 array which holds the coil endpoints
        dl is a nc x ns x nn x nb x 3 array which computes the length of the ns segments
        r_middle is a nc x ns x nn x nb x 3 array which computes the midpoint of each of the ns segments

        """

        v1, v2 = CoilSet.compute_frame(self, fr, normal, binormal)
        r = np.zeros((self.nic, self.ns, self.nn, self.nb, 3))
        r += coil_centroid[:, :, np.newaxis, np.newaxis, :]
        for n in range(self.nn):
            for b in range(self.nb):
                r = r.at[:, :, n, b, :].add(
                    (n - 0.5 * (self.nn - 1)) * self.ln * v1 + 
                    (b - 0.5 * (self.nb - 1)) * self.lb * v2
                ) 
        return r

    def compute_dl(self, params, frame, der1, der2, coil_centroid):   
        dl = np.zeros((self.nic, self.ns, self.nn, self.nb, 3))
        dl += der1[:, :, np.newaxis, np.newaxis, :]
        dv1_dt, dv2_dt = CoilSet.compute_frame_derivative(self, params, frame, der1, der2, coil_centroid)
        for n in range(self.nn):
            for b in range(self.nb):
                dl = dl.at[:, :, n, b, :].add(
                    (n - 0.5 * (self.nn - 1)) * self.ln * dv1_dt + 
                    (b - 0.5 * (self.nb - 1)) * self.lb * dv2_dt
                )

        return dl * (2 * pi / self.ns)


    def stellarator_symmetry_coil(self, r):
        """计算线圈的仿星器对称"""
        rc = np.zeros((self.nic*2, self.ns, self.nn, self.nb, 3))
        rc = rc.at[0:self.nic, :, :, :, :].set(r)
        T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        rc = rc.at[self.nic:self.nic*2, :, :, :, :].set(np.dot(r, T))
        return rc

    def symmetry_coil(self, r):
        """计算线圈的周期对称"""
        npc = int(self.nc / self.nfp)   # 每周期线圈数，number of coils per period
        rc_total = np.zeros((self.nc, self.ns, self.nn, self.nb, 3))
        rc_total = rc_total.at[0:npc, :, :, :, :].set(r)
        for i in range(self.nfp - 1):        
            theta_t = 2 * pi * (i + 1) / self.nfp
            T = np.array([[np.cos(theta_t), -np.sin(theta_t), 0],
                 [np.sin(theta_t), np.cos(theta_t), 0], [0, 0, 1]])
            rc_total = rc_total.at[npc*(i+1):npc*(i+2), :, :, :, :].set(np.dot(r, T))
        
        return rc_total

    def stellarator_symmetry_I(self, I):
        """计算电流的仿星器对称"""
        I_new = np.zeros(self.nic*2)
        I_new = I_new.at[:self.nic].set(I)
        for i in range(self.nic):
            I_new = I_new.at[i+self.nic].set(-I[i])

        return I_new

    def symmetry_I(self, I):
        """计算电流的周期对称"""
        npc = int(self.nc / self.nfp)
        I_new = np.zeros(self.nc)
        for i in range(self.nfp):
            I_new = I_new.at[npc*i:npc*(i+1)].set(I)
        return I_new

    def finite_build_I(self, I):
        """计算电流的有限截面数据"""
        I_new = np.zeros((self.nc * self.nn * self.nb))
        for i in range(self.nc):
            I_new = I_new.at[i*self.nn * self.nb:(i+1)*self.nn * self.nb].set(I[i])
        return I_new

    def get_coil(self, params):
        _,_,coil,_,_,_ = CoilSet.coilset(self, params)
        return coil

    def end_coil(self, params):
        """迭代结束后的输出, 通过字典将相关量整合"""
        coil_arg, fr = params   
        I_new = self.args['I'] / (self.nn * self.nb)
        coil_centroid = CoilSet.compute_coil_centroid(self, coil_arg)  # [nc, ns+1, 3]
        der1, der2, der3 = CoilSet.compute_der(self, coil_arg)   # [nc, ns+1, 3]
        tangent, normal, binormal = CoilSet.compute_com(self, der1, coil_centroid)
        r = CoilSet.compute_r(self, fr, normal, binormal, coil_centroid)
        frame = tangent, normal, binormal
        dl = CoilSet.compute_dl(self, params, frame, der1, der2, coil_centroid)
        if self.ss == 1 :
            r = CoilSet.stellarator_symmetry_coil(self, r)
            dl = CoilSet.stellarator_symmetry_coil(self, dl)
            I_new = CoilSet.stellarator_symmetry_I(self, I_new)
        r = CoilSet.symmetry_coil(self, r)
        dl = CoilSet.symmetry_coil(self, dl)
        I_new = CoilSet.symmetry_I(self, I_new)
        I_new = CoilSet.finite_build_I(self, I_new)
        coil_all = {
            'coil_centroid' :   coil_centroid,
            'der1'          :   der1,
            'der2'          :   der2,
            'der3'          :   der3,
            'tangent'       :   tangent,
            'normal'        :   normal,
            'binormal'      :   binormal,
            'params'        :   params,
            'I_new'         :   I_new,
            'r'             :   r,
            'dl'            :   dl
        }
        return coil_all




