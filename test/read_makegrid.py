import json
import sys 
import jax.numpy as np
import plotly.graph_objects as go
sys.path.append('/home/nxy/codes/coil_spline_HTS/iteration')
import fourier


# def read_makegrid(filename, nic, ns):    
#     """
#     读取初始线圈的makegrid文件
#     Args:
#         filename : str, 文件地址
#         nic : int, 独立线圈数, 
#         ns : int, 线圈段数
#     Returns:
#         r : array, [nic, ns+1, 3], 线圈初始坐标

#     """
#     # r = np.zeros((nic, 250, 3))
#     r2 = []
#     n0=[]
#     n1=0
#     with open(filename) as f:
#         _ = f.readline()
#         _ = f.readline()
#         _ = f.readline()
#         for i in range(nic):
#             r1 = []
#             for s in range(250):
#                 r0 = []
#                 x = f.readline().split()
#                 r0.append(float(x[0]))
#                 r0.append(float(x[1]))
#                 r0.append(float(x[2]))
#                 r1.append(r0)
#                 # r = r.at[i, s, 0].set(float(x[0]))
#                 # r = r.at[i, s, 1].set(float(x[1]))
#                 # r = r.at[i, s, 2].set(float(x[2]))
#                 if float(x[3]) == 0.0:
#                     n0.append(s)
#                     n1=n1+s+1
#                     break
#             r2.append(r1)
#     print(n0)
#     # r = r.at[:, -1, :].set(r[:, 0, :])
#     return r2, n0, n1


# coil, n0, n1 = read_makegrid('/home/nxy/codes/coil_spline_HTS/initfiles/aries/n3are.coils', 18, 184)



# fc = np.zeros((6, 18, 6))
# for i in range(18):
#     rc = np.array(coil[i])[np.newaxis, :, :]
#     f = fourier.compute_coil_fourierSeries(1, n0[i], 6, rc)
#     fc = fc.at[:, i, :].set(np.squeeze(f))
# print(fc)
# theta = np.linspace(0, 2 * np.pi, 211)
# rc = fourier.compute_r_centroid(fc, 6, 18, 210, theta)
# np.save('/home/nxy/codes/coil_spline_HTS/initfiles/qas/coils_18.npy', rc)

def read_makegrid_saddle(filename, nct, ncs, nst, nss):    
    """
    读取初始线圈的makegrid文件
    Args:
        filename : str, 文件地址
        nic : int, 独立线圈数, 
        ns : int, 线圈段数
    Returns:
        r : array, [nic, ns+1, 3], 线圈初始坐标

    """
    rt = np.zeros((nct, nst+1, 3))
    rp = np.zeros((ncp, nsp+1, 3))
    rs = np.zeros((ncs, nss+1, 3))
    Ip = np.zeros((ncp, nsp+1))
    Is = np.zeros((ncs, nss+1))

    with open(filename) as f:
        _ = f.readline()
        _ = f.readline()
        _ = f.readline()
        for i in range(nct):
            for s in range(nst+1):
                x = f.readline().split()
                rt = rt.at[i, s, 0].set(float(x[0]))
                rt = rt.at[i, s, 1].set(float(x[1]))
                rt = rt.at[i, s, 2].set(float(x[2]))
        # for i in range(ncp):
        #     for s in range(nsp+1):
        #         x = f.readline().split()
        #         rp = rp.at[i, s, 0].set(float(x[0]))
        #         rp = rp.at[i, s, 1].set(float(x[1]))
        #         rp = rp.at[i, s, 2].set(float(x[2]))
        #         Ip = Ip.at[i, s].set(float(x[3]))
        for i in range(ncs):
            print(i)
            for j in range(nss+1):
                x = f.readline().split()
                rs = rs.at[i, j, 0].set(float(x[0]))
                rs = rs.at[i, j, 1].set(float(x[1]))
                rs = rs.at[i, j, 2].set(float(x[2]))
                Is = Is.at[i, j].set(float(x[3]))
        x = f.readline().split()
        print(x)
    # Ip = Ip[:,1]
    Is = Is[:,1]
    return rt, rs, Is

nct, ncp, ncs = 12, 12, 144
nst, nsp, nss = 99, 99, 40
It = 2.281852279e+07

rt, rs, Is = read_makegrid_saddle('/home/nxy/codes/coil_spline_HTS/initfiles/aries/coil_3_11/coils.dat', nct, ncs, nst, nss)
np.save('/home/nxy/codes/coil_spline_HTS/initfiles/aries/coil_3_11/coiltf_12.npy', rt)
# np.save('/home/nxy/codes/coil_spline_HTS/initfiles/aries/coilpf_12.npy', rp)
np.save('/home/nxy/codes/coil_spline_HTS/initfiles/aries/coil_3_11/coilsd_144.npy', rs)
# np.save('/home/nxy/codes/coil_spline_HTS/initfiles/aries/coil_Ip_12.npy', Ip)
np.save('/home/nxy/codes/coil_spline_HTS/initfiles/aries/coil_3_11/coil_Is_144.npy', Is)
print(Is)




