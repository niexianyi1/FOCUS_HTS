import json
import jax.numpy as np
import numpy
import read_file
import fourier
import spline
import plotly.graph_objects as go
import h5py
import read_init
import sys 
sys.path.append('HTS')
import material_jcrit
import B_self

pi = np.pi

with open('initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)

nfp = 5
coil = np.load('results/test/bs/rcoil.npy')
r_surf = np.load('results/test/bs/rsurf.npy')
dl = np.load('results/test/bs/dl.npy')
I = np.load('results/test/bs/I.npy')
bs = np.load('results/test/bs/bs.npy')

def calb(coil, dl, I):
    mu_0Idl = (1e-7 * I[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] * dl)  # NC x NNR x NBR x NS x 3
    r_minus_l = (r_surf[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis, :]
        - coil[:, np.newaxis, np.newaxis, :, :, :, :])  # NC x NZ/nfp x NT x NNR x NBR x NS x 3
    top = np.cross(mu_0Idl[:, np.newaxis, np.newaxis, :, :, :, :], r_minus_l)  # NC x NZ x NT x NNR x NBR x NS x 3
    bottom = (np.linalg.norm(r_minus_l, axis=-1) ** 3)  # NC x NZ x NT x NNR x NBR x NS
    B = np.sum(top / bottom[:, :, :, :, :, :, np.newaxis], axis=(0, 3, 4, 5))  # NZ x NT x 3
    return B

B1 = calb(coil[:10], dl[:10], I[:10])
B2 = calb(coil[10:20], dl[10:20], I[10:20])
print(B1)
print(B2 )

theta_t = 2 * pi / nfp
T = np.array([[np.cos(theta_t), -np.sin(theta_t), 0],
        [np.sin(theta_t), np.cos(theta_t), 0], [0, 0, 1]])
bt = np.dot(B1[0], T)

print(np.mean(abs(bt-B2[26])))
print(B1[0], bt, B2[26])
# 
# def symmetry_surf(B):
#     B_total = B
#     for i in range(nfp - 1):        
#         theta_t = 2 * pi * (i + 1) / nfp
#         T = np.array([[np.cos(theta_t), -np.sin(theta_t), 0],
#                 [np.sin(theta_t), np.cos(theta_t), 0], [0, 0, 1]])
#         bt = np.dot(B, T)
#         for j in range(130):
#             B_total = B_total.at[j].add(bt[int(j-(i+1)*130/nfp)])
#     return B_total

# B = symmetry_surf(B)
# print(B[0],bs[0])
# print(np.mean(abs(B-bs)))




