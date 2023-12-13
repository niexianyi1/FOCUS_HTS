import jax.numpy as np
from jax import jit, vmap
import tables as tb
import numpy
import h5py
from jax.config import config
import bspline
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
        self.nfr = args['nfr']
        self.bc = args['bc']     
        self.tj = args['tj']
        self.out_hdf5 = args['out_hdf5']
        self.out_coil_makegrid = args['out_coil_makegrid']
        self.theta = np.arange(0, 2*pi, 2*pi/self.ns)
        self.I = args['I']
        return
 

    @jit
    def coilset(self, params):             # 根据lossfunction的需求再添加新的输出项                   
        c, fr = params   
        I_new = self.I / (self.nnr * self.nbr)
        # COMPUTE COIL VARIABLES
        r_centroid = CoilSet.compute_r_centroid(self, c)  # [nc, ns+1, 3]
        der1, der2 = CoilSet.compute_der(self, c)   # [nc, ns+1, 3]
        tangent, normal, binormal = CoilSet.compute_com(self, der1, r_centroid)
        r = CoilSet.compute_r(self, fr, normal, binormal, r_centroid)
        frame = tangent, normal, binormal
        dl = CoilSet.compute_dl(self, params, frame, der1, der2, r_centroid)
        if self.ss == 1 :
            r = CoilSet.stellarator_symmetry(self, r)
            dl = CoilSet.stellarator_symmetry(self, dl)
        r = CoilSet.symmetry(self, r)
        dl = CoilSet.symmetry(self, dl)

        return I_new, dl, r, der1, der2

    def compute_r_centroid(self, c):         # rc 是（nc/nfp,ns+1,3）
        # rc = vmap(lambda c :bspline.splev(self.bc, c, self.tj, self.ns), 
        #             in_axes=0, out_axes=0)(c)
        t, u, k = self.bc
        coil = np.zeros((self.nic, self.ns, 3))
        for i in range(self.nic):
            coil = coil.at[i].set(bspline.splev(t[i], u[i], c[i], self.tj[i], self.ns))
        return coil

    def compute_der(self, c):                    
        """ Computes  1,2,3 derivatives of the rc """
        # der1, wrk1 = vmap(lambda c :bspline.der1_splev(self.bc, c, self.tj, self.ns), 
        #                     in_axes=0, out_axes=0)(c)
        # der2 = vmap(lambda wrk1 :bspline.der2_splev(self.bc, wrk1, self.tj, self.ns),
        #                     in_axes=0, out_axes=0)(wrk1)

        t, u, k = self.bc
        der1 = der2 = np.zeros((self.nic, self.ns, 3))
        for i in range(self.nic):
            d10, wrk1 = bspline.der1_splev(t[i], u[i], c[i], self.tj[i], self.ns)
            der1 = der1.at[i].set(d10)
            der2 = der2.at[i].set(bspline.der2_splev(t[i], u[i], wrk1, self.tj[i], self.ns))
        return der1, der2
        
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
        return (a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1] + a[:, :, 2] * b[:, :, 2])

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
        r0 = CoilSet.compute_coil_mid(self, r_centroid)
        delta = r_centroid - r0[:, np.newaxis, :]
        dp = CoilSet.dot_product_rank3_tensor(tangent, delta)
        normal = delta - tangent * dp[:, :, np.newaxis]
        mag = np.linalg.norm(normal, axis=-1)
        return normal / mag[:, :, np.newaxis]

    def compute_normal_deriv(self, tangent, tangent_deriv, der1, r_centroid):          
        r0 = CoilSet.compute_coil_mid(self, r_centroid)
        delta = r_centroid - r0[:, np.newaxis, :]
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
        alpha = np.zeros((self.nic, self.ns ))
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
        return alpha

    def compute_alpha_1(self, fr):    
        alpha_1 = np.zeros((self.nic, self.ns ))
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
        return alpha_1

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
                    (n - 0.5 * (self.nnr - 1)) * self.ln * v1 + (b - 0.5 * (self.nbr - 1)) * self.lb * v2
                ) 
        return r

    def compute_dl(self, params, frame, der1, der2, r_centroid):   
        dl = np.zeros((self.nic, self.ns, self.nnr, self.nbr, 3))
        dl += der1[:, :, np.newaxis, np.newaxis, :]
        dv1_dt, dv2_dt = CoilSet.compute_frame_derivative(self, params, frame, der1, der2, r_centroid)
        for n in range(self.nnr):
            for b in range(self.nbr):
                dl = dl.at[:, :, n, b, :].add(
                    (n - 0.5 * (self.nnr - 1)) * self.ln * dv1_dt + (b - 0.5 * (self.nbr - 1)) * self.lb * dv2_dt
                )

        return dl * (1 / self.ns)

    def symmetry(self, r):
        nic = self.nic*(self.ss+1)
        rc_total = np.zeros((self.nc, self.ns, self.nnr, self.nbr, 3))
        rc_total = rc_total.at[0:nic, :, :, :, :].add(r)
        for i in range(self.nfp - 1):        
            theta = 2 * pi * (i + 1) / self.nfp
            T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            rc_total = rc_total.at[nic*(i+1):nic*(i+2), :, :, :, :].add(np.dot(r, T))
        
        return rc_total

    def stellarator_symmetry(self, r):
        rc = np.zeros((self.nic*2, self.ns, self.nnr, self.nbr, 3))
        rc = rc.at[0:self.nic, :, :, :, :].add(r)
        T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        rc = rc.at[self.nic:self.nic*2, :, :, :, :].add(np.dot(r, T))
        return rc

    def read_hdf5(self, filename):
        f = h5py.File(filename, "r")
        arge = {}
        for key in list(f.keys()):
            arge.update({key: f[key][:]})
        f.close()
        return arge

    def read_makegrid(self, filename):      # 处理一下
        r = np.zeros((self.nc, self.ns+1, 3))
        with open(filename) as f:
            _ = f.readline()
            _ = f.readline()
            _ = f.readline()
            for i in range(self.nc):
                for s in range(self.ns):
                    x = f.readline().split()
                    r = r.at[i, s, 0].set(float(x[0]))
                    r = r.at[i, s, 1].set(float(x[1]))
                    r = r.at[i, s, 2].set(float(x[2]))
                _ = f.readline()
        r = r.at[:, -1, :].set(r[:, 0, :])
        return r


    def write_hdf5(self, params):     # 根据需求写入数据
        """ Write coils in HDF5 output format.
		Input:

		output_file (string): Path to outputfile, string should include .hdf5 format


		"""

        c, fr = params
        with tb.open_file(self.out_hdf5, "w") as f:
            coildata = numpy.dtype(
                [
                    ("nc", int),
                    ("ns", int),
                    ("ln", float),
                    ("lb", float),
                    ("nnr", int),
                    ("nbr", int),
                    ("rc", float),
                    ("nr", int),
                    ("nfr", int),
                ]
            )
            arr = numpy.array(
                [(self.nc, self.ns, self.ln, self.lb, self.nnr, self.nbr, self.rc, self.nr, self.nfr)], dtype=coildata,
            )
            f.create_table("/", "coildata", coildata)
            f.root.coildata.append(arr)
            f.create_array("/", "coilspline", numpy.asarray(c))
            f.create_array("/", "coilrotation", numpy.asarray(fr))
        return

    def write_makegrid(self, params):    # 或者直接输入r, I
        I, _, r = CoilSet.coilset(self, params)
        with open(self.out_coil_makegrid, "w") as f:
            f.write("periods {}\n".format(0))
            f.write("begin filament\n")
            f.write("FOCUSADD Coils\n")
            for i in range(self.nic):
                for n in range(self.nnr):
                    for b in range(self.nbr):
                        for s in range(self.ns):
                            f.write(
                                "{} {} {} {}\n".format(
                                    r[i, s, n, b, 0],
                                    r[i, s, n, b, 1],
                                    r[i, s, n, b, 2],
                                    I[i],
                                )
                            )
                        f.write(
                            "{} {} {} {} {} {}\n".format(
                                r[i, 0, n, b, 0],
                                r[i, 0, n, b, 1],
                                r[i, 0, n, b, 2],
                                0.0,
                                "{},{},{}".format(i, n, b),
                                "coil/filament1/filament2",
                            )
                        )
        return







