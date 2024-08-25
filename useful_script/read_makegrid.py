import json
import sys 
import jax.numpy as np
import plotly.graph_objects as go
sys.path.append('opt_coil')
import fourier


# def read_makegrid(filename, nic, ns):    
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


# coil, n0, n1 = read_makegrid('initfiles/aries/n3are.coils', 18, 184)



# fc = np.zeros((6, 18, 6))
# for i in range(18):
#     rc = np.array(coil[i])[np.newaxis, :, :]
#     f = fourier.compute_coil_fourierSeries(rc, 6)
#     fc = fc.at[:, i, :].set(np.squeeze(f))
# print(fc)
# theta = np.linspace(0, 2 * np.pi, 211)
# rc = fourier.compute_r_centroid(fc, 210)
# np.save('initfiles/qas/coils_18.npy', rc)




# def read_coil(filename, nc, ns):    
#     r = np.zeros((nc, ns+1, 3))
#     I = np.zeros((nc, ns+1))

#     with open(filename) as f:
#         for i in range(nc):
#             for s in range(ns+1):
#                 x = f.readline().split()
#                 r = r.at[i, s, 0].set(float(x[0]))
#                 r = r.at[i, s, 1].set(float(x[1]))
#                 r = r.at[i, s, 2].set(float(x[2]))
#     return r





def read_makegrid_saddle(filename, ncm, nct, nsm, nst):    
    rm = np.zeros((ncm, nsm+1, 3))
    rp1 = np.zeros((ncp1, nsp1+1, 3))
    rp2 = np.zeros((ncp2, nsp2+1, 3))
    rt = np.zeros((nct, nst+1, 3))
    Ip1 = np.zeros((ncp1, nsp1+1))
    Ip2 = np.zeros((ncp2, nsp2+1))
    It = np.zeros((nct, nst+1))

    with open(filename) as f:
        _ = f.readline()
        _ = f.readline()
        _ = f.readline()
        for i in range(ncm):
            print(i)
            for s in range(nsm+1):
                x = f.readline().split()
                rm = rm.at[i, s, 0].set(float(x[0]))
                rm = rm.at[i, s, 1].set(float(x[1]))
                rm = rm.at[i, s, 2].set(float(x[2]))
        for i in range(ncp1):
            print(i)
            for s in range(nsp1+1):
                x = f.readline().split()
                rp1 = rp1.at[i, s, 0].set(float(x[0]))
                rp1 = rp1.at[i, s, 1].set(float(x[1]))
                rp1 = rp1.at[i, s, 2].set(float(x[2]))
                Ip1 = Ip1.at[i, s].set(float(x[3]))
        for i in range(ncp2):
            print(i)
            for s in range(nsp2+1):
                x = f.readline().split()
                rp2 = rp2.at[i, s, 0].set(float(x[0]))
                rp2 = rp2.at[i, s, 1].set(float(x[1]))
                rp2 = rp2.at[i, s, 2].set(float(x[2]))
                Ip2 = Ip2.at[i, s].set(float(x[3]))
        for i in range(nct):
            print(i)
            for j in range(nst+1):
                x = f.readline().split()
                rt = rt.at[i, j, 0].set(float(x[0]))
                rt = rt.at[i, j, 1].set(float(x[1]))
                rt = rt.at[i, j, 2].set(float(x[2]))
                It = It.at[i, j].set(float(x[3]))
        x = f.readline().split()
        print(x)
    Ip1 = Ip1[:,1]
    Ip2 = Ip2[:,1]
    It = It[:,1]
    return rm, rp1, rp2, rt, Ip1, Ip2, It

ncm, ncp1, ncp2, nct = 18, 36, 6, 18
nsm, nsp1, nsp2, nst = 400, 192, 384, 122
Im = [6.52271941985300E+05, 6.51868569367400E+05, 5.37743588647300E+05]

rm, rp1, rp2, rt, Ip1, Ip2, It = read_makegrid_saddle('initfiles/ncsx_c09r00/coilsall.c09r00', ncm, nct, nsm, nst)
# np.save('initfiles/aries/coil_4_1/coiltf_12.npy', rm)
# # np.save('initfiles/aries/coilpf_12.npy', rp1)
# np.save('initfiles/aries/coil_4_1/coilsd_144.npy', rt)
# # np.save('initfiles/aries/coil_Ip_12.npy', Ip)
# np.save('initfiles/aries/coil_4_1/coil_Is_144.npy', It)
# print(It)

rm = np.reshape(rm, (ncm * (nsm+1), 3))
rp1 = np.reshape(rp1, (ncp1 * (nsp1+1), 3))
rp2 = np.reshape(rp2, (ncp2 * (nsp2+1), 3))
rt = np.reshape(rt, (nct * (nst+1), 3))
print(Ip1, Ip2, It)

fig = go.Figure()
fig.add_scatter3d(x=rm[:, 0],y=rm[:, 1],z=rm[:, 2], name='module_coil', mode='markers', marker_size = 1.5)   
fig.add_scatter3d(x=rp1[:, 0],y=rp1[:, 1],z=rp1[:, 2], name='PF1_coil', mode='markers', marker_size = 1.5) 
fig.add_scatter3d(x=rp2[:, 0],y=rp2[:, 1],z=rp2[:, 2], name='PF2_coil', mode='markers', marker_size = 1.5)     
fig.add_scatter3d(x=rt[:, 0],y=rt[:, 1],z=rt[:, 2], name='TF_coil', mode='markers', marker_size = 1.5)   
# fig.add_trace(go.Surface(x=r_surf[:,:,0], y=r_surf[:,:,1], z=r_surf[:,:,2]))
fig.update_layout(scene_aspectmode='data',  scene = dict(
    xaxis = dict(#  backgroundcolor="white",  gridcolor="white",
        title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white",),
    yaxis = dict(# backgroundcolor="white", gridcolor="white",
        title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white"),
    zaxis = dict(# backgroundcolor="white", gridcolor="white",
        title_text = "",showticklabels=False,showbackground=False,zerolinecolor="white",),))
fig.show()
