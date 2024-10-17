import jax.numpy as np
from jax import jit, vmap, config
import fourier
import spline

config.update("jax_enable_x64", True)
pi = np.pi

class CoilSet:

    """
	CoilSet is a class which represents the modular coils. The coil centerlines are represented by 
    a fourier or a spline series, and the coil section rotation is represented by alpha Angle. 
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
        Calculates coil data and outputs it to lossfunction.py
        Args:
            params  :   list, [coil_arg, fr, I[:-1]]
        """
        coil_arg, fr, I = params  
        if self.args['total_current_I'] != 0:
            In = self.args['total_current_I'] / self.args['I_normalize'] - np.sum(I)
            I = np.append(I, In)
        else:
            I = np.append(I, 1)
        coil_centroid = CoilSet.calculate_coil_centroid(self, coil_arg)  
        der1, der2, der3, dt = CoilSet.calculate_coil_der(self, coil_arg)   
        tangent, normal, binormal = CoilSet.calculate_com(self, der1, coil_centroid)
        centroid_frame = tangent, normal, binormal
        dTdt, dNdt, dBdt = CoilSet.calculate_com_deriv(self, centroid_frame, der1, der2, coil_centroid)
        alpha = CoilSet.calculate_alpha(self, fr)
        alpha1 = CoilSet.calculate_alpha_der(self, fr)
        v1, v2 = CoilSet.calculate_frame(self, alpha, centroid_frame)
        dv1_dt, dv2_dt = CoilSet.calculate_frame_der(self, alpha, alpha1, centroid_frame, dNdt, dBdt)
        r = CoilSet.calculate_r(self, v1, v2, coil_centroid)
        dl = CoilSet.calculate_dl(self, dv1_dt, dv2_dt, der1, dt)
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
            for j in range(lenai):          
                start = int(a[i][j][0])
                end = int(a[i][j][1])
                coil_arg_c = coil_arg_c.at[int(loc[i]), :, start:end].set(
                                np.squeeze(np.array(coil_arg[i][j])))
        return coil_arg_c


    def calculate_coil_centroid(self, coil_arg):    
        """Calculate the coil centerline"""     
        def calculate_coil_fourier(self, coil_arg):
            coil_centroid = fourier.calculate_r_centroid(coil_arg, self.ns)
            coil_centroid = coil_centroid[:, :-1, :]
            return coil_centroid

        def calculate_coil_spline_all(self, coil_arg):
            t, u, k = self.args['bc']
            coil_arg_c = np.zeros((self.nic, 3, self.args['number_control_points']))
            coil_arg_c = coil_arg_c.at[:, :, :-3].set(coil_arg)
            coil_arg_c = coil_arg_c.at[:, :, -3:].set(coil_arg[:, :, :3])
            coil_centroid = vmap(lambda coil_arg_c :spline.splev(t, u, coil_arg_c, self.args['tj'], self.ns), 
                        in_axes=0, out_axes=0)(coil_arg_c)
            
            return coil_centroid  

        def calculate_coil_spline_local(self, coil_arg):
            t, u, k = self.args['bc']
            coil_arg_c = CoilSet.local_arg_c(self, coil_arg)
            coil_centroid = vmap(lambda coil_arg_c :spline.splev(t, u, coil_arg_c, self.args['tj'], self.ns), 
                    in_axes=0, out_axes=0)(coil_arg_c)
            return coil_centroid

        calculate_coil = dict()
        calculate_coil['fourier'] = calculate_coil_fourier
        calculate_coil['spline'] = calculate_coil_spline_all
        calculate_coil['spline_local'] = calculate_coil_spline_local
        calculate_func = calculate_coil[self.args['coil_case']]
        coil_centroid = calculate_func(self, coil_arg)
        return coil_centroid


    def calculate_coil_der(self, coil_arg):  
        """Calculate each derivative(1,2,3) of the coil centerline"""   
        def calculate_coil_der_fourier(self, coil_arg):          
            der1 = fourier.calculate_der1(coil_arg, self.ns)
            der2 = fourier.calculate_der2(coil_arg, self.ns)
            der3 = fourier.calculate_der3(coil_arg, self.ns)
            der1, der2, der3 = der1[:, :-1, :], der2[:, :-1, :], der3[:, :-1, :]
            dt = 2 * pi / self.ns
            return der1, der2, der3, dt

        def calculate_coil_der_spline_all(self, coil_arg):
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

        def calculate_coil_der_spline_local(self, coil_arg):
            t, u, k = self.args['bc']
            coil_arg_c = CoilSet.local_arg_c(self, coil_arg)
            der1, wrk1 = vmap(lambda coil_arg_c :spline.der1_splev(t, u, coil_arg_c, self.args['tj'], self.ns), 
                    in_axes=0, out_axes=0)(coil_arg_c)
            der2 = vmap(lambda wrk1 :spline.der2_splev(t, u, wrk1, self.args['tj'], self.ns), 
                    in_axes=0, out_axes=0)(wrk1)
            der3 = 0
            dt = 1 / self.ns
            return der1, der2, der3, dt

        calculate_coil_der = dict()
        calculate_coil_der['fourier'] = calculate_coil_der_fourier
        calculate_coil_der['spline'] = calculate_coil_der_spline_all
        calculate_coil_der['spline_local'] = calculate_coil_der_spline_local
        calculate_func = calculate_coil_der[self.args['coil_case']]
        der1, der2, der3, dt = calculate_func(self, coil_arg)
        return der1, der2, der3, dt


    def calculate_com(self, der1, coil_centroid):    
        tangent = CoilSet.calculate_tangent(self, der1)
        normal = -CoilSet.calculate_normal(self, coil_centroid, tangent)
        binormal = CoilSet.calculate_binormal(self, tangent, normal)
        return tangent, normal, binormal


    def calculate_com_deriv(self, frame, der1, der2, coil_centroid): 
        tangent, normal, _ = frame
        tangent_deriv = CoilSet.calculate_tangent_deriv(self, der1, der2)
        normal_deriv = -CoilSet.calculate_normal_deriv(self, tangent, tangent_deriv, der1, coil_centroid)
        binormal_deriv = CoilSet.calculate_binormal_deriv(self, tangent, normal, tangent_deriv, normal_deriv)
        return tangent_deriv, normal_deriv, binormal_deriv


    def calculate_tangent(self, der1):          
        """calculates the tangent vector of the coils."""
        tangent = der1 / np.linalg.norm(der1, axis=-1)[:, :, np.newaxis]
        return tangent


    def calculate_tangent_deriv(self, der1, der2):   
        """ dT """
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


    def calculate_coil_mid(self, coil_centroid):      # mid_point   r0=[self.nic, 3]
        """Calculate the center point coordinates of each coil."""
        x = coil_centroid[:, :-1, 0]
        y = coil_centroid[:, :-1, 1]
        z = coil_centroid[:, :-1, 2]
        coil_mid = np.zeros((self.nic, 3))
        for i in range(self.nic):
            coil_mid = coil_mid.at[i, 0].add(np.sum(x[i]) / self.ns)
            coil_mid = coil_mid.at[i, 1].add(np.sum(y[i]) / self.ns)
            coil_mid = coil_mid.at[i, 2].add(np.sum(z[i]) / self.ns)        
        return coil_mid


    def calculate_normal(self, coil_centroid, tangent):    
        """calculates the normal vector of the coils."""
        coil_mid = CoilSet.calculate_coil_mid(self, coil_centroid)
        delta = coil_centroid - coil_mid[:, np.newaxis, :]
        dp = CoilSet.dot_product_rank3_tensor(tangent, delta)
        normal = delta - tangent * dp[:, :, np.newaxis]
        mag = np.linalg.norm(normal, axis=-1)
        return normal / mag[:, :, np.newaxis]


    def calculate_normal_deriv(self, tangent, tangent_deriv, der1, coil_centroid):  
        """ dN """  
        coil_mid = CoilSet.calculate_coil_mid(self, coil_centroid)
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


    def calculate_binormal(self, tangent, normal):           
        """ calculates the binormal vector of the coils. """
        return np.cross(tangent, normal)


    def calculate_binormal_deriv(self, tangent, normal, tangent_deriv, normal_deriv):  
        ''' bB '''
        return np.cross(tangent_deriv, normal) + np.cross(tangent, normal_deriv)


    def calculate_alpha(self, fr):    
        alpha = fourier.calculate_fourier_alpha(fr, self.ns, self.nr)
        return alpha


    def calculate_alpha_der(self, fr):   
        alpha_1 = fourier.calculate_fourier_alpha_der1(fr, self.ns)
        return alpha_1


    def calculate_frame(self, alpha, frame):  
        """calculates the vectors v1 and v2 for each coil. v1 and v2 are rotated relative to
		the normal and binormal frame by an amount alpha. Alpha is parametrized by a Fourier series."""
        _, N, B = frame
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)
        v1 = calpha[:, :, np.newaxis] * N + salpha[:, :, np.newaxis] * B
        v2 = -salpha[:, :, np.newaxis] * N + calpha[:, :, np.newaxis] * B
        return v1, v2


    def calculate_frame_der(self, alpha, alpha1, frame, dNdt, dBdt): 
        ''' d_v1 and d_v2 '''
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


    def calculate_r(self, v1, v2, coil_centroid):     ### 每个线圈截面独立计算
        '''calculates the position of the multi-filament coils.
        r is a nc x ns + 1 x nn x nb x 3 array which holds the coil endpoints.'''
        

        r = np.zeros((self.nic, self.nn, self.nb, self.ns, 3))
        r += coil_centroid[:, np.newaxis, np.newaxis, :, :]

        for n in range(self.nn):
            for b in range(self.nb):
                    r = r.at[:, n, b, :, :].add(
                        (n - 0.5 * (self.nn - 1)) * self.ln[:, np.newaxis, np.newaxis] * v1 + 
                        (b - 0.5 * (self.nb - 1)) * self.lb[:, np.newaxis, np.newaxis] * v2
                    ) 
        return r

    def calculate_dl(self, dv1_dt, dv2_dt, der1, dt):   
        ''' dl is a nc x ns x nn x nb x 3 array which calculates the length of the ns segments.'''
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
        rc = np.zeros((self.nic*2, self.nn, self.nb, self.ns, 3))
        rc = rc.at[0:self.nic, :, :, :, :].set(r)
        T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        rc = rc.at[self.nic:self.nic*2, :, :, :, :].set(np.dot(r, T))
        return rc

    def symmetry_coil(self, r):
        '''Periodic symmetry'''
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
        I_new = np.zeros(self.nic*2)
        I_new = I_new.at[:self.nic].set(I)
        for i in range(self.nic):
            I_new = I_new.at[i+self.nic].set(-I[i])
        return I_new

    def symmetry_I(self, I):
        npc = int(self.nc / self.nfp)
        I_new = np.zeros(self.nc)
        for i in range(self.nfp):
            I_new = I_new.at[npc*i:npc*(i+1)].set(I)
        return I_new


    def get_coil(self, params):
        _,_,coil,_,_,_,_,_,_ = CoilSet.cal_coil(self, params)
        return coil


    def calculate_curva(self, der1, der2):
        curva = np.cross(der1, der2) / (np.linalg.norm(der1, axis = -1)**3)[:,:,np.newaxis]
        return curva


    def get_fb_input(self, params):
        coil_arg, fr, I = params   
        coil_centroid = CoilSet.calculate_coil_centroid(self, coil_arg)  
        der1, der2, _, dt = CoilSet.calculate_coil_der(self, coil_arg)   
        tangent, normal, binormal = CoilSet.calculate_com(self, der1, coil_centroid)
        centroid_frame = tangent, normal, binormal
        alpha = CoilSet.calculate_alpha(self, fr)
        v1, v2 = CoilSet.calculate_frame(self, alpha, centroid_frame)
        curva = CoilSet.calculate_curva(self, der1, der2)
        dl = der1[:, np.newaxis, np.newaxis, :, :] * dt
        coil_centroid = coil_centroid[:, np.newaxis, np.newaxis, :, :]
        B_self_input = (coil_centroid, I, dl, v1, v2, binormal, curva, der2)
        return  B_self_input


    def end_coil(self, params):
        """After the optimization is finished, the parameters are saved through the dictionary"""
        coil_arg, fr, I = params   
        if self.args['total_current_I'] != 0:
            In = self.args['total_current_I'] / self.args['I_normalize'] - np.sum(I)
            I = np.append(I, In)
        else:
            I = np.append(I, 1)
        print('current = ', I * self.args['I_normalize']) 
        coil_centroid = CoilSet.calculate_coil_centroid(self, coil_arg)  
        der1, der2, der3, dt = CoilSet.calculate_coil_der(self, coil_arg)   
        tangent, normal, binormal = CoilSet.calculate_com(self, der1, coil_centroid)
        centroid_frame = tangent, normal, binormal
        dTdt, dNdt, dBdt = CoilSet.calculate_com_deriv(self, centroid_frame, der1, der2, coil_centroid)
        alpha = CoilSet.calculate_alpha(self, fr)
        alpha1 = CoilSet.calculate_alpha_der(self, fr)
        v1, v2 = CoilSet.calculate_frame(self, alpha, centroid_frame)
        dv1_dt, dv2_dt = CoilSet.calculate_frame_der(self, alpha, alpha1, centroid_frame, dNdt, dBdt)     
        r = CoilSet.calculate_r(self, v1, v2, coil_centroid)
        dl = CoilSet.calculate_dl(self, dv1_dt, dv2_dt, der1, dt)
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




