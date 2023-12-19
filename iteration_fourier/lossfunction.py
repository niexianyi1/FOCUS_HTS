import jax.numpy as np
from jax import jit
from jax.config import config
config.update("jax_enable_x64", True)
pi = np.pi

# 用class包括所有的loss function.
class LossFunction:

    def __init__(self, args, surface_data, B_extern):  # 不够再加
        r_surf, nn, sg = surface_data
        self.args = args
        self.r_surf = r_surf
        self.nn = nn
        self.sg = sg
        self.nc = args['nc']
        self.nfp = args['nfp']
        self.nic = args['nic'] 
        self.ns = args['ns']
        self.nz = args['nz']
        self.nt = args['nt']
        self.B_extern = B_extern
        self.nznfp = int(self.nz / self.nfp)
        return 
    @jit
    def loss(self, coil_output_func, params):
        """ 
        Computes the default loss: int (B dot n)^2 dA + weight_length * len(coils) 

        Input: params, a tuple of the fourier series for the coils and a fourier series for the rotation.

        Output: A scalar, which is the loss_val computed by the function. JAX will eventually differentiate
        this in an optimizer.
        """
        I, dl, r_coil, der1, der2, der3 = coil_output_func(params)
        self.I = I
        self.dl = dl
        self.r_coil = r_coil
        Bn_mean, Bn_max = LossFunction.quadratic_flux(self)
        length, al = LossFunction.average_length(self)
        k_mean, k_max = LossFunction.curvature(der1, der2)
        t_mean, t_max = LossFunction.torsion(der1, der2, der3)
        dcc_min = LossFunction.distance_cc(self)
        dcs_min = LossFunction.distance_cs(self)
        return (self.args['wb'] * Bn_mean + self.args['wl'] * length 
                + self.args['wc'] * k_mean + self.args['wcm'] * k_max 
                + self.args['wt'] * t_mean + self.args['wtm'] * t_max 
                + self.args['wdcc'] * dcc_min + self.args['wdcs'] * dcs_min )
    
         
    def quadratic_flux(self):
        """ 

		Computes the normalized quadratic flux over the whole surface.
			
		Inputs:

		r : Position we want to evaluate at, NZ x NT x 3
		I : Current in ith coil, length NC
		dl : Vector which has coil segment length and direction, NC x NS x NNR x NBR x 3
		l : Positions of center of each coil segment, NC x NS x NNR x NBR x 3
		nn : Normal vector on the surface, NZ x NT x 3
		sg : Area of the surface, 
		
		Returns: 

		A NZ x NT array which computes integral of 1/2(B dot n)^2 dA / integral of B^2 dA. 
		We can eventually sum over this array to get the total integral over the surface. I choose not to
		sum so that we can compute gradients of the surface magnetic normal if we'd like. 

		"""
        B = LossFunction.biotSavart(self)  # NZ x NT x 3

        # B = LossFunction.symmetry(self, B)

        if self.B_extern is not None:
            B_all = B + self.B_extern
        else:
            B_all = B
        Bn = np.sum(self.nn * B_all, axis=-1)
        Bn_mean = 0.5*np.sum((Bn/ np.linalg.norm(B, axis=-1))**2*self.sg)
        Bn_max = np.max(abs(Bn))
        return  Bn_mean, Bn_max

    def biotSavart(self):
        """
		Inputs:

		r : Position we want to evaluate at, NZ x NT x 3
		I : Current in ith coil, length NC
		dl : Vector which has coil segment length and direction, NC x NS x NNR x NBR x 3
		l : Positions of center of each coil segment, NC x NS x NNR x NBR x 3

		Returns: 

		A NZ x NT x 3 array which is the magnetic field vector on the surface points 
		"""
        mu_0 = 1e-7
        mu_0I = self.I * mu_0
        mu_0Idl = (mu_0I[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] * self.dl)  # NC x NS x NNR x NBR x 3
        r_minus_l = (self.r_surf[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis, :]
            - self.r_coil[:, np.newaxis, np.newaxis, :, :, :, :])  # NC x NZ/nfp x NT x NS x NNR x NBR x 3
        top = np.cross(mu_0Idl[:, np.newaxis, np.newaxis, :, :, :, :], r_minus_l)  # NC x NZ x NT x NS x NNR x NBR x 3
        bottom = (np.linalg.norm(r_minus_l, axis=-1) ** 3)  # NC x NZ x NT x NS x NNR x NBR
        B = np.sum(top / bottom[:, :, :, :, :, :, np.newaxis], axis=(0, 3, 4, 5))  # NZ x NT x 3
        return B
    
    # def normalized_error(r, I, dl, l, nn, sg, B_extern = None):
    #     B = LossFunction.biotSavart(r, I, dl, l)  # NZ x NT x 3
    #     if B_extern is not None:
    #         B = B + B_extern

    #     B_n = np.abs( np.sum(nn * B, axis=-1) )
    #     B_mag = np.linalg.norm(B, axis=-1)
    #     A = np.sum(sg)

    #     return np.sum( (B_n / B_mag) * sg ) / A

### 新增

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

    def curvature(der1, der2):
        bottom = np.linalg.norm(der1, axis = -1)**3
        top = np.linalg.norm(np.cross(der1, der2), axis = -1)
        k = abs(top / bottom)
        k_mean = np.mean(k)
        k_max = np.max(k)
        return k_mean, k_max

    def average_length(self):      #new
        r_coil = self.r_coil[:, :, 0, 0, :]
        al = np.zeros_like(r_coil)
        al = al.at[:, :-1, :].set(r_coil[:, 1:, :] - r_coil[:, :-1, :])
        al = al.at[:, -1, :].set(r_coil[:, 0, :] - r_coil[:, -1, :])
        return np.sum(np.linalg.norm(al, axis=-1)) / (self.nc), al

    def distance_cc(self):  ### 暂未考虑finite-build
        rc = self.r_coil[:, :, 0, 0, :]
        dr = rc[:self.nic, :, :] - rc[1:self.nic+1, :, :]
        dr = np.linalg.norm(dr, axis = -1)
        dcc_min = np.min(dr)
        return dcc_min

    def distance_cs(self):  ### 暂未考虑finite-build
        rc = self.r_coil[:, :, 0, 0, :]
        rs = self.r_surf
        dr = rc[:self.nic, :, np.newaxis, np.newaxis, :] - rs[np.newaxis, np.newaxis, :, :, :]
        dr = np.linalg.norm(dr, axis = -1)
        dcs_min = np.min(dr)
        return dcs_min

##  HTS应变量
    def HTS_strain_bend(self, ):
        """弯曲应变,
        Args:
            w, 带材宽度
            v1,有限截面坐标轴
            curvature, 线圈曲率

        Returns:
            bend, 弯曲应变

        """

        bend = w/2*abs(-v1 * curvature)
        return bend

    def HTS_strain_tor(self, deltal):
        """扭转应变,
        Args:
            w, 带材宽度
            v1,有限截面坐标轴
            deltal, 线圈点间隔

        Returns:
            bend, 弯曲应变

        """

        dv = v1[:, :-1, :] * v1[:, 1:, :]
        dv = dv + v1[:, -1, :] * v1[:, 0, :]
        dtheta = np.arccos(dv)
        tor = w**2/12*(dtheta/deltal)**2
        return tor



    def symmetry_B(self, B):
        B_total = np.zeros((self.nz, self.nt, 3))
        B_total = B_total.at[:, :, :].add(B)
        for i in range(self.nfp - 1):        
            theta = 2 * pi * (i + 1) / self.nfp
            T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            B_total = B_total.at[:, :, :].add(np.dot(B, T))
        
        return B_total





















