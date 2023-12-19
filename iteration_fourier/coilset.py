import jax.numpy as np
from jax import jit, vmap
import numpy
from jax.config import config
import fourier
config.update("jax_enable_x64", True)
pi = np.pi

class CoilSet:

    """
	CoilSet is a class which represents all of the coils surrounding a plasma surface. The coils
	are represented by a bspline and a fourier series, one for the coil winding pack centroid and
    one for the rotation of the coils. 
	"""

    def __init__(self, args):
        self.args = args
        self.nc = args['nc']
        self.nfp = args['nfp']
        self.nic = args['nic']
        self.ss = args['ss']
        self.ns = args['ns']
        self.ln = args['ln']
        self.lb = args['lb']
        self.nnr = args['nnr']
        self.nbr = args['nbr']
        self.rc = args['rc']
        self.nr = args['nr']
        self.nfc = args['nfc']
        self.nfr = args['nfr']
        self.out_hdf5 = args['out_hdf5']
        self.out_coil_makegrid = args['out_coil_makegrid']
        self.theta = np.linspace(0, 2 * pi, self.ns + 1)
        self.I = args['I']
        return
 

    @jit
    def coilset(self, params):             # 根据lossfunction的需求再添加新的输出项                   
        fc, fr = params   
        I_new = self.I / (self.nnr * self.nbr)

        r_centroid = CoilSet.compute_r_centroid(self, fc)  # [nc, ns+1, 3]
        der1, der2, der3 = CoilSet.compute_der(self, fc)   # [nc, ns+1, 3]
        tangent, normal, binormal = CoilSet.compute_com(self, der1, r_centroid)
        r = CoilSet.compute_r(self, fr, normal, binormal, r_centroid)
        frame = tangent, normal, binormal
        dl = CoilSet.compute_dl(self, params, frame, der1, der2, r_centroid)
        if self.ss == 1 :
            r = CoilSet.stellarator_symmetry_r(self, r)
            dl = CoilSet.stellarator_symmetry_der(self, dl)
        r = CoilSet.symmetry(self, r)
        dl = CoilSet.symmetry(self, dl)

        return I_new, dl, r, der1, der2, der3

    def compute_r_centroid(self, fc):         # rc 是（nc/nfp,ns+1,3）
        rc = fourier.compute_r_centroid(fc, self.nfc, self.nic, self.ns, self.theta)
        return rc[:, :-1, :]

    def compute_der(self, fc):                    
        der1 = fourier.compute_der1(fc, self.nfc, self.nic, self.ns, self.theta)
        der2 = fourier.compute_der2(fc, self.nfc, self.nic, self.ns, self.theta)
        der3 = fourier.compute_der3(fc, self.nfc, self.nic, self.ns, self.theta)
        return der1[:, :-1, :], der2[:, :-1, :], der3[:, :-1, :]
        
    def compute_com(self, der1, r_centroid):    
        """ Computes T, N, and B """
        tangent = CoilSet.compute_tangent(self, der1)
        normal = -CoilSet.compute_normal(self, r_centroid, tangent)
        binormal = CoilSet.compute_binormal(self, tangent, normal)
        return tangent, normal, binormal

    def compute_com_deriv(self, frame, der1, der2, r_centroid):  
        tangent, normal, _ = frame
        tangent_deriv = CoilSet.compute_tangent_deriv(self, der1, der2)
        normal_deriv = -CoilSet.compute_normal_deriv(self, tangent, tangent_deriv, der1, r_centroid)
        binormal_deriv = CoilSet.compute_binormal_deriv(self, tangent, normal, tangent_deriv, normal_deriv)
        return tangent_deriv, normal_deriv, binormal_deriv

    def compute_tangent(self, der1):          
        """
        Computes the tangent vector of the coils. Uses the equation 
        T = dr/d_theta / |dr / d_theta|
        """
        return der1 / np.linalg.norm(der1, axis=-1)[:, :, np.newaxis]

    def compute_tangent_deriv(self, der1, der2):     
        norm_der1 = np.linalg.norm(der1, axis=-1)
        mag_2 = CoilSet.dot_product_rank3_tensor(der1, der2) / norm_der1 ** 3
        return der2 / norm_der1[:, :, np.newaxis] - der1 * mag_2[:, :, np.newaxis]

    def dot_product_rank3_tensor(a, b):         # dot
        dotab = (a[:, :, 0] * b[:, :, 0] + 
                 a[:, :, 1] * b[:, :, 1] + 
                 a[:, :, 2] * b[:, :, 2])
        return dotab

    def compute_coil_mid(self, r_centroid):      # mid_point   r0=[self.nic, 3]
        x = r_centroid[:, :-1, 0]
        y = r_centroid[:, :-1, 1]
        z = r_centroid[:, :-1, 2]
        r0 = np.zeros((self.nic, 3))
        for i in range(self.nic):
            r0 = r0.at[i, 0].add(np.sum(x[i]) / self.ns)
            r0 = r0.at[i, 1].add(np.sum(y[i]) / self.ns)
            r0 = r0.at[i, 2].add(np.sum(z[i]) / self.ns)        
        return r0

    def compute_normal(self, r_centroid, tangent):    
        coil_mid = CoilSet.compute_coil_mid(self, r_centroid)
        delta = r_centroid - coil_mid[:, np.newaxis, :]
        dp = CoilSet.dot_product_rank3_tensor(tangent, delta)
        normal = delta - tangent * dp[:, :, np.newaxis]
        mag = np.linalg.norm(normal, axis=-1)
        return normal / mag[:, :, np.newaxis]

    def compute_normal_deriv(self, tangent, tangent_deriv, der1, r_centroid):          
        coil_mid = CoilSet.compute_coil_mid(self, r_centroid)
        delta = r_centroid - coil_mid[:, np.newaxis, :]
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

    def compute_alpha(self, fr):    # alpha 用fourier
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

    def compute_frame_derivative(self, params, frame, der1, der2, r_centroid):    
        _, N, B = frame
        _, fr = params
        alpha = CoilSet.compute_alpha(self, fr)
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)
        alpha1 = CoilSet.compute_alpha_1(self, fr)
        _, dNdt, dBdt = CoilSet.compute_com_deriv(self, frame, der1, der2, r_centroid)
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

    def compute_r(self, fr, normal, binormal, r_centroid):      
        """
        Computes the position of the multi-filament coils.

        r is a nc x ns + 1 x nnr x nbr x 3 array which holds the coil endpoints
        dl is a nc x ns x nnr x nbr x 3 array which computes the length of the ns segments
        r_middle is a nc x ns x nnr x nbr x 3 array which computes the midpoint of each of the ns segments

        """

        v1, v2 = CoilSet.compute_frame(self, fr, normal, binormal)
        r = np.zeros((self.nic, self.ns, self.nnr, self.nbr, 3))
        r += r_centroid[:, :, np.newaxis, np.newaxis, :]
        for n in range(self.nnr):
            for b in range(self.nbr):
                r = r.at[:, :, n, b, :].add(
                    (n - 0.5 * (self.nnr - 1)) * self.ln * v1 + 
                    (b - 0.5 * (self.nbr - 1)) * self.lb * v2
                ) 
        return r

    def compute_dl(self, params, frame, der1, der2, r_centroid):   
        dl = np.zeros((self.nic, self.ns, self.nnr, self.nbr, 3))
        dl += der1[:, :, np.newaxis, np.newaxis, :]
        dv1_dt, dv2_dt = CoilSet.compute_frame_derivative(self, params, frame, der1, der2, r_centroid)
        for n in range(self.nnr):
            for b in range(self.nbr):
                dl = dl.at[:, :, n, b, :].add(
                    (n - 0.5 * (self.nnr - 1)) * self.ln * dv1_dt + 
                    (b - 0.5 * (self.nbr - 1)) * self.lb * dv2_dt
                )

        return dl * (2 * pi / self.ns)

    def symmetry(self, r):
        npc = int(self.nc / self.nfp)   # 每周期线圈数，number of coils per period
        rc_total = np.zeros((self.nc, self.ns, self.nnr, self.nbr, 3))
        rc_total = rc_total.at[0:npc, :, :, :, :].add(r)
        for i in range(self.nfp - 1):        
            theta_t = 2 * pi * (i + 1) / self.nfp
            T = np.array([[np.cos(theta_t), -np.sin(theta_t), 0],
                 [np.sin(theta_t), np.cos(theta_t), 0], [0, 0, 1]])
            rc_total = rc_total.at[npc*(i+1):npc*(i+2), :, :, :, :].add(np.dot(r, T))
        
        return rc_total

    def stellarator_symmetry_r(self, r):
        rc = np.zeros((self.nic*2, self.ns, self.nnr, self.nbr, 3))
        rc = rc.at[0:self.nic, :, :, :, :].add(r)
        T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        rc = rc.at[self.nic:self.nic*2, :, :, :, :].add(np.dot(r, T))
        return rc

    def stellarator_symmetry_der(self, r):
        rc = np.zeros((self.nic*2, self.ns, self.nnr, self.nbr, 3))
        rc = rc.at[0:self.nic, :, :, :, :].add(r)
        T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        rc = rc.at[self.nic:self.nic*2, :, :, :, :].add(-np.dot(r, T))
        return rc










