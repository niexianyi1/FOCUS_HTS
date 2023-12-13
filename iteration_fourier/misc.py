import jax.numpy as np
import fourier
import json
import plotly.graph_objects as go


with open('/home/nxy/codes/focusadd-spline/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)


fc5 = np.load('/home/nxy/codes/focusadd-spline/results_f/circle/fc_a0.npy')
fc50 = np.load('/home/nxy/codes/focusadd-spline/results_f/circle/fc_a00.npy')
theta = np.linspace(0, 2 * np.pi, ns + 1)
rc5 = fourier.compute_r_centroid(fc5, nfc, 5, ns, theta)[:, :-1, :]
rc50 = fourier.compute_r_centroid(fc50, nfc, 50, ns, theta)[:, :-1, :]

def symmetry_r(r):
    rc = np.zeros((nic*2, ns, 3))
    rc = rc.at[:nic, :, :].add(r)
    T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rc = rc.at[nic:nic*2, :, :].add(np.dot(r, T))
    rc_total = np.zeros((nc, ns, 3))
    rc_total = rc_total.at[:nic*(ss+1), :, :].add(rc)
    for i in range(nfp - 1):        
        theta = 2 * np.pi * (i + 1) / nfp
        T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        rc_total = rc_total.at[nic*(ss+1)*(i+1):nic*(ss+1)*(i+2), :, :].add(np.dot(rc, T))
    return rc_total


def symmetry_der(r):
    rc = np.zeros((nic*2, ns, 3))
    rc = rc.at[:nic, :, :].add(r)
    T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rc = rc.at[nic:nic*2, :, :].add(-np.dot(r, T))
    rc_total = np.zeros((nc, ns, 3))
    rc_total = rc_total.at[:nic*(ss+1), :, :].add(rc)
    for i in range(nfp - 1):        
        theta = 2 * np.pi * (i + 1) / nfp
        T = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        rc_total = rc_total.at[nic*(ss+1)*(i+1):nic*(ss+1)*(i+2), :, :].add(np.dot(rc, T))
    return rc_total


def quadratic_flux(I, dl, r_surf, r_coil, nn, sg):
    B = biotSavart(I ,dl, r_surf, r_coil)  
    B_all = B        
    print('Bmax = ', np.max(np.linalg.norm(abs(B_all), axis=-1)))
    return ( 0.5 * np.sum( np.sum(nn * B_all/
                np.linalg.norm(B_all, axis=-1)[:, :, np.newaxis], axis=-1) ** 2 * sg))    


### 由于线圈次序不同，导致磁场计算不同，公式有问题
def biotSavart(I ,dl, r_surf, r_coil):
    mu_0 = 1e-7
    mu_0I = I * mu_0
    mu_0Idl = (mu_0I[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] * dl)  # NC x NS x NNR x NBR x 3
    r_minus_l = (r_surf[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis, :]
        - r_coil[:, np.newaxis, np.newaxis, :, :, :, :])  # NC x NZ/nfp x NT x NS x NNR x NBR x 3
    top = np.cross(mu_0Idl[:, np.newaxis, np.newaxis, :, :, :, :], r_minus_l)  # NC x NZ x NT x NS x NNR x NBR x 3
    bottom = (np.linalg.norm(r_minus_l, axis=-1) ** 3)  # NC x NZ x NT x NS x NNR x NBR
    B = np.sum(top / bottom[:, :, :, :, :, :, np.newaxis], axis=(0, 3, 4, 5))  # NZ x NT x 3
    return B



I = np.ones(50)*1e6
r_surf = np.load(args['surface_r'])
nn = np.load(args['surface_nn'])
sg = np.load(args['surface_sg'])

# rc5 = symmetry_r(rc5)


der15 = fourier.compute_der1(fc5, nfc, 5, ns, theta)[:, :-1, :]
der15 = symmetry_der(der15)
der150 = fourier.compute_der1(fc50, nfc, 50, ns, theta)[:, :-1, :]
# for i in range(64):
#     print('i=',i,np.mean(abs(der15[5,i,2]+der150[9,-i,2])))
# print(np.mean(abs(der15[4]-der150[4])))

dl5 = der15[:, :, np.newaxis, np.newaxis, :]*2*np.pi/ns
dl50 = der150[:, :, np.newaxis, np.newaxis, :]*2*np.pi/ns

rc5 = symmetry_r(rc5)[:, :, np.newaxis, np.newaxis, :]
rc50 = rc50[:, :, np.newaxis, np.newaxis, :]

b50 = biotSavart(I ,dl50, r_surf, rc50)
b5 = biotSavart(I ,dl5, r_surf, rc5)
print(np.mean(abs(b50-b5)))



bn5 = quadratic_flux(I, dl5, r_surf, rc5, nn, sg)
bn50 = quadratic_flux(I, dl50, r_surf, rc50, nn, sg)
print(bn5, bn50)





