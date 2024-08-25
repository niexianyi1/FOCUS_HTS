import jax.numpy as np
from jax import jit, vmap, config
import numpy
import fourier
import spline
import lossfunction
# import sys
# sys.path.append('HTS')
# import self_B
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
        self.ln = np.array(args['length_normal'])
        self.lb = np.array(args['length_binormal'])
        self.nn = args['number_normal']
        self.nb = args['number_binormal']
        self.nr = args['number_rotate']
        self.nfc = args['number_fourier_coils']
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
        coil_arg, fr, I = params  
        if self.args['total_current_I'] != 0:
            In = self.args['total_current_I'] / self.args['I_normalize'] - np.sum(I)
            I = np.append(I, In)
        else:
            I = np.append(I, 1)
        coil_centroid = CoilSet.compute_coil_centroid(self, coil_arg)  
        der1, der2, der3, dt = CoilSet.compute_coil_der(self, coil_arg)   
        tangent, normal, binormal = CoilSet.compute_com(self, der1, coil_centroid)
        centroid_frame = tangent, normal, binormal
        dTdt, dNdt, dBdt = CoilSet.compute_com_deriv(self, centroid_frame, der1, der2, coil_centroid)
        alpha = CoilSet.compute_alpha(self, fr)
        alpha1 = CoilSet.compute_alpha_der(self, fr)
        v1, v2 = CoilSet.compute_frame(self, alpha, centroid_frame)
        dv1_dt, dv2_dt = CoilSet.compute_frame_der(self, alpha, alpha1, centroid_frame, dNdt, dBdt)
        r = CoilSet.compute_r(self, v1, v2, coil_centroid)
        dl = CoilSet.compute_dl(self, dv1_dt, dv2_dt, der1, dt)
        if self.ss == 1 :
            r = CoilSet.stellarator_symmetry_coil(self, r)
            dl = CoilSet.stellarator_symmetry_coil(self, dl)
            I = CoilSet.stellarator_symmetry_I(self, I)
        r = CoilSet.symmetry_coil(self, r)
        dl = CoilSet.symmetry_coil(self, dl)
        I = CoilSet.symmetry_I(self, I)

        return I, dl, r, der1, der2, der3, v1, v2, binormal


    def local_arg_c(self, coil_arg):
        a = self.args['optimize_location_ns']
        loc = self.args['optimize_location_nic']
        lo_nc = len(loc) 
        coil_arg_c = self.args['c_init']
        for i in range(lo_nc):
            lenai = len(a[i])
            for j in range(lenai):          # 需要加判断避免超出边界 或 重复计入控制点
                start = int(a[i][j][0])
                end = int(a[i][j][1])
                coil_arg_c = coil_arg_c.at[int(loc[i]), :, start:end].set(
                                np.squeeze(np.array(coil_arg[i][j])))
        return coil_arg_c


    def compute_coil_centroid(self, coil_arg):    
        """
        计算线圈中心位置, 按照所选表达方法计算

        Args:
            coil_arg  :   array, 线圈坐标表达式参数

        Returns:
            coil_centroid   :   array, [nc, ns, 3], 线圈中心位置坐标点

        """     
        def compute_coil_fourier(self, coil_arg):
            coil_centroid = fourier.compute_r_centroid(coil_arg, self.ns)
            coil_centroid = coil_centroid[:, :-1, :]
            return coil_centroid

        def compute_coil_spline_all(self, coil_arg):
            t, u, k = self.args['bc']
            coil_arg_c = np.zeros((self.nic, 3, self.args['number_control_points']))
            coil_arg_c = coil_arg_c.at[:, :, :-3].set(coil_arg)
            coil_arg_c = coil_arg_c.at[:, :, -3:].set(coil_arg[:, :, :3])
            coil_centroid = vmap(lambda coil_arg_c :spline.splev(t, u, coil_arg_c, self.args['tj'], self.ns), 
                        in_axes=0, out_axes=0)(coil_arg_c)
            
            return coil_centroid  

        def compute_coil_spline_local(self, coil_arg):
            t, u, k = self.args['bc']
            coil_arg_c = CoilSet.local_arg_c(self, coil_arg)
            coil_centroid = vmap(lambda coil_arg_c :spline.splev(t, u, coil_arg_c, self.args['tj'], self.ns), 
                    in_axes=0, out_axes=0)(coil_arg_c)
            return coil_centroid

        compute_coil = dict()
        compute_coil['fourier'] = compute_coil_fourier
        compute_coil['spline'] = compute_coil_spline_all
        compute_coil['spline_local'] = compute_coil_spline_local
        compute_func = compute_coil[self.args['coil_case']]
        coil_centroid = compute_func(self, coil_arg)
        return coil_centroid


    def compute_coil_der(self, coil_arg):  
        """
        计算线圈各阶导数, 按照所选表达方法计算

        Args:
            coil_arg  :   array, 线圈坐标表达式参数

        Returns:
            der1, der2, der3   :   array, [nc, ns, 3], 线圈各阶导数

        """   
        def compute_coil_der_fourier(self, coil_arg):          
            der1 = fourier.compute_der1(coil_arg, self.ns)
            der2 = fourier.compute_der2(coil_arg, self.ns)
            der3 = fourier.compute_der3(coil_arg, self.ns)
            der1, der2, der3 = der1[:, :-1, :], der2[:, :-1, :], der3[:, :-1, :]
            dt = 2 * pi / self.ns
            return der1, der2, der3, dt

        def compute_coil_der_spline_all(self, coil_arg):
            t, u, k = self.args['bc']
            coil_arg_c = np.zeros((self.nic, 3, self.args['number_control_points']))
            coil_arg_c = coil_arg_c.at[:, :, :-3].set(coil_arg)
            coil_arg_c = coil_arg_c.at[:, :, -3:].set(coil_arg[:, :, :3])
            der1, wrk1 = vmap(lambda coil_arg_c :spline.der1_splev(t, u, coil_arg_c, self.args['tj'], self.ns), 
                    in_axes=0, out_axes=0)(coil_arg_c)
            der2 = vmap(lambda wrk1 :spline.der2_splev(t, u, wrk1, self.args['tj'], self.ns), 
                    in_axes=0, out_axes=0)(wrk1)
            der3 = 0
            dt = 1 / self.ns
            return der1, der2, der3, dt

        def compute_coil_der_spline_local(self, coil_arg):
            t, u, k = self.args['bc']
            coil_arg_c = CoilSet.local_arg_c(self, coil_arg)
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
        compute_coil_der['spline_local'] = compute_coil_der_spline_local
        compute_func = compute_coil_der[self.args['coil_case']]
        der1, der2, der3, dt = compute_func(self, coil_arg)
        return der1, der2, der3, dt


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


    def compute_alpha_der(self, fr):   
        """计算有限截面旋转角的导数""" 
        alpha_1 = np.zeros((self.nic, self.ns+1 ))
        alpha_1 += self.nr / 2
        Ac = fr[:, 0]
        As = fr[:, 1]
        for m in range(self.nfr):
            arg = self.theta * m
            carg = np.cos(arg)
            sarg = np.sin(arg)
            alpha_1 += (
                -m * Ac[:, np.newaxis, m] * sarg[np.newaxis, :]
                + m * As[:, np.newaxis, m] * carg[np.newaxis, :]
            )
        return alpha_1[:, :-1]


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


    def compute_frame_der(self, alpha, alpha1, frame, dNdt, dBdt): 
        _, N, B = frame
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)
        dv1_dt = (
            calpha[:, :, np.newaxis] * dNdt
            + salpha[:, :, np.newaxis] * dBdt
            - salpha[:, :, np.newaxis] * N * alpha1[:, :, np.newaxis]
            + calpha[:, :, np.newaxis] * B * alpha1[:, :, np.newaxis]
        )
        dv2_dt = (
            - salpha[:, :, np.newaxis] * dNdt
            + calpha[:, :, np.newaxis] * dBdt
            - calpha[:, :, np.newaxis] * N * alpha1[:, :, np.newaxis]
            - salpha[:, :, np.newaxis] * B * alpha1[:, :, np.newaxis]
        )
        return dv1_dt, dv2_dt


    def compute_r(self, v1, v2, coil_centroid):     ### 每个线圈截面独立计算
        """
        Computes the position of the multi-filament coils.

        r is a nc x ns + 1 x nn x nb x 3 array which holds the coil endpoints
        dl is a nc x ns x nn x nb x 3 array which computes the length of the ns segments
        r_middle is a nc x ns x nn x nb x 3 array which computes the midpoint of each of the ns segments

        """

        r = np.zeros((self.nic, self.nn, self.nb, self.ns, 3))
        r += coil_centroid[:, np.newaxis, np.newaxis, :, :]

        for n in range(self.nn):
            for b in range(self.nb):
                    r = r.at[:, n, b, :, :].add(
                        (n - 0.5 * (self.nn - 1)) * self.ln[:, np.newaxis, np.newaxis] * v1 + 
                        (b - 0.5 * (self.nb - 1)) * self.lb[:, np.newaxis, np.newaxis] * v2
                    ) 
        return r

    def compute_dl(self, dv1_dt, dv2_dt, der1, dt):   
        dl = np.zeros((self.nic, self.nn, self.nb, self.ns, 3))
        dl += der1[:, np.newaxis, np.newaxis, :, :]
        for n in range(self.nn):
            for b in range(self.nb):
                    dl = dl.at[:, n, b, :, :].add(
                        (n - 0.5 * (self.nn - 1)) * self.ln[:, np.newaxis, np.newaxis] * dv1_dt + 
                        (b - 0.5 * (self.nb - 1)) * self.lb[:, np.newaxis, np.newaxis] * dv2_dt
                    )
        return dl * dt
        
  

    def stellarator_symmetry_coil(self, r):
        """计算线圈的仿星器对称"""
        rc = np.zeros((self.nic*2, self.nn, self.nb, self.ns, 3))
        rc = rc.at[0:self.nic, :, :, :, :].set(r)
        T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        rc = rc.at[self.nic:self.nic*2, :, :, :, :].set(np.dot(r, T))
        return rc

    def symmetry_coil(self, r):
        """计算线圈的周期对称"""
        npc = int(self.nc / self.nfp)   # 每周期线圈数，number of coils per period
        rc_total = np.zeros((self.nc, self.nn, self.nb, self.ns, 3))
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
            # I_new = I_new.at[i+self.nic].set(-I[i])
            I_new = I_new.at[i+self.nic].set(-I[i])
        return I_new

    def symmetry_I(self, I):
        """计算电流的周期对称"""
        npc = int(self.nc / self.nfp)
        I_new = np.zeros(self.nc)
        for i in range(self.nfp):
            I_new = I_new.at[npc*i:npc*(i+1)].set(I)
        return I_new


    def get_coil(self, params):
        _,_,coil,_,_,_,_,_,_ = CoilSet.cal_coil(self, params)
        return coil


    def compute_curva(self, der1, der2):
        curva = np.cross(der1, der2) / (np.linalg.norm(der1, axis = -1)**3)[:,:,np.newaxis]
        return curva


    def get_fb_input(self, params):
        coil_arg, fr, I = params   
        coil_centroid = CoilSet.compute_coil_centroid(self, coil_arg)  
        der1, der2, _, dt = CoilSet.compute_coil_der(self, coil_arg)   
        tangent, normal, binormal = CoilSet.compute_com(self, der1, coil_centroid)
        centroid_frame = tangent, normal, binormal
        alpha = CoilSet.compute_alpha(self, fr)
        v1, v2 = CoilSet.compute_frame(self, alpha, centroid_frame)
        curva = CoilSet.compute_curva(self, der1, der2)
        dl = der1[:, np.newaxis, np.newaxis, :, :] * dt
        coil_centroid = coil_centroid[:, np.newaxis, np.newaxis, :, :]
        B_self_input = (coil_centroid, I, dl, v1, v2, binormal, curva, der2)
        return  B_self_input


    def end_coil(self, params):
        """迭代结束后的输出, 通过字典将相关量整合"""
        coil_arg, fr, I = params   
        if self.args['total_current_I'] != 0:
            In = self.args['total_current_I'] / self.args['I_normalize'] - np.sum(I)
            I = np.append(I, In)
        else:
            I = np.append(I, 1)
        coil_centroid = CoilSet.compute_coil_centroid(self, coil_arg)  
        der1, der2, der3, dt = CoilSet.compute_coil_der(self, coil_arg)   
        tangent, normal, binormal = CoilSet.compute_com(self, der1, coil_centroid)
        centroid_frame = tangent, normal, binormal
        dTdt, dNdt, dBdt = CoilSet.compute_com_deriv(self, centroid_frame, der1, der2, coil_centroid)
        alpha = CoilSet.compute_alpha(self, fr)
        alpha1 = CoilSet.compute_alpha_der(self, fr)
        v1, v2 = CoilSet.compute_frame(self, alpha, centroid_frame)
        dv1_dt, dv2_dt = CoilSet.compute_frame_der(self, alpha, alpha1, centroid_frame, dNdt, dBdt)     
        r = CoilSet.compute_r(self, v1, v2, coil_centroid)
        dl = CoilSet.compute_dl(self, dv1_dt, dv2_dt, der1, dt)
        if self.ss == 1 :
            r = CoilSet.stellarator_symmetry_coil(self, r)
            dl = CoilSet.stellarator_symmetry_coil(self, dl)
            I = CoilSet.stellarator_symmetry_I(self, I)
        r = CoilSet.symmetry_coil(self, r)
        dl = CoilSet.symmetry_coil(self, dl)
        I = CoilSet.symmetry_I(self, I) 
        I = I * self.args['I_normalize']
        
        if self.args['coil_case'] == 'spline':  
            coil_arg_c = np.zeros((self.nic, 3, self.args['number_control_points']))
            coil_arg_c = coil_arg_c.at[:, :, :-3].set(coil_arg)
            coil_arg_c = coil_arg_c.at[:, :, -3:].set(coil_arg[:, :, :3])
            coil_arg = coil_arg_c
                
        elif self.args['coil_case'] == 'spline_local':
            coil_arg_c = CoilSet.local_arg_c(self, coil_arg)
            coil_arg = coil_arg_c

        coil_all = {
            'coil_centroid'     :   coil_centroid,
            'coil_der1'         :   der1,
            'coil_der2'         :   der2,
            'coil_der3'         :   der3,
            'coil_tangent'      :   tangent,
            'coil_normal'       :   normal,
            'coil_binormal'     :   binormal,
            'coil_v1'           :   v1,
            'coil_v2'           :   v2,
            'coil_arg'          :   coil_arg,
            'coil_fr'           :   fr,
            'coil_alpha'        :   alpha,
            'coil_I'            :   I,
            'coil_r'            :   r,
            'coil_dl'           :   dl,

        }
        return coil_all




