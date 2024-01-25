# 计算B_self, 

import jax.numpy as np
mu_0 = 1e-7

def coil_self_B(args, coil, I, dl, v1, v2, binormal, curva):
    nic = args['number_independent_coils']
    ns = args['number_segments']    
    I = args['number_normal'] * args['number_binormal'] * I
    I_nic = I[:nic, np.newaxis, np.newaxis]

    if args['HTS_material'] == 'NbTi':
        a = args['HTS_width'] / 2
        Bother = B_other(args, I, coil, dl, nic, ns)
        Breg = B_reg_L(I, coil, dl, a, nic, ns)
        Blocal = B_local(I_nic, v1, v2, a, curva, Breg)
        B_self = Bother + Breg + Blocal

    else:
        a = args['HTS_width']   # 正方形截面
        b = args['HTS_thick']
        u, v = 1, 1  # 截面沿坐标轴方向的位置归一化 
        k1 = curva * (binormal*v2)
        k2 = -curva * (binormal*v1)
        if a == b :
            k = 2.5565
            delta = 0.19985
        elif a >= 10*b :
            k = 7/6 + np.log(a/b)
            delta = a / (b*np.exp(1)**3)
        else :
            k = (4*b/(3*a)*np.arctan(a/b) + 4*a/(3*b)*np.arctan(b/a) + b**2/(6*a**2)*np.log(b/a)+
                    a**2/(6*b**2)*np.log(a/b) + (a**4-6*(a*b)**2+b**4)/(6*(a*b)**2)*np.log(a/b+b/a) )
            delta = np.exp(-25/6 + k)

        Bother = B_other(args, I, coil, dl, nic, ns)
        Breg = B_reg_H(I, coil, dl, a, b, nic, ns, delta)
        B0 = B_0(I_nic, u, v, v1, v2, a, b)
        Bk = B_k(I_nic, u, v, v1, v2, a, b, k1, k2)
        Bb = B_b(I_nic, curva, binormal, delta)
        B_self = Bother + Breg + B0 + Bk + Bb

    B_self = np.max(np.linalg.norm(B_self, axis=-1))
    print(B_self)
    return B_self


def B_other(args, I, coil, dl, nic, ns):
    I = I/(args['number_normal'] * args['number_binormal'])
    coil_ev = np.mean(coil[:nic], axis = (1,2))     # 所求的点位
    Bother = np.zeros((nic, ns, 3))    # other: 其他线圈在所求点的场

    for i in range(nic):
        coil_other = np.delete(coil, i, axis=0)     # [nc-1, nn, nb, ns, 3]
        dl_other = np.delete(dl, i, axis=0)         # [nc-1, nn, nb, ns, 3]
        I_other = np.delete(I, i, axis=0)           # [nc-1]
        mu_0Idl_other = (mu_0*I_other[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] *  # [nc-1, nn, nb, ns, 3]
                            dl_other) 
        r_l_other = (coil_ev[np.newaxis, i, :, np.newaxis, np.newaxis, np.newaxis, :]       # [nc-1, (ns), nn, nb, ns, 3]
                - coil_other[:, np.newaxis, :, :, :, :])  
        Bother = Bother.at[i].set(np.sum(                                                 
                (np.cross(mu_0Idl_other[:, np.newaxis, :, :, :, :], r_l_other) /
                    ((np.linalg.norm(r_l_other, axis=-1)) ** 3)[:, :, :, :, :, np.newaxis]), 
                        axis=(0, 2, 3, 4)) )
    return Bother


def B_reg_L(I, coil, dl, a, nic, ns):

    coil_ev = np.mean(coil[:nic], axis = (1,2))     # 所求的点位
    Breg = np.zeros((nic, ns, 3))      # reg: 线圈对自身场的项
    for i in range(nic):
        coil_reg = np.mean(coil[i], axis = (0,1))       # [ns, 3]
        dl_reg = np.mean(dl[i], axis = (0,1))           # [ns, 3]
        mu_0Idl_reg = (mu_0*I[i, np.newaxis, np.newaxis, np.newaxis] * dl_reg[np.newaxis, :, :])  # [1,ns,3]
        r_l_reg = coil_ev[i, :, np.newaxis, :] - coil_reg[np.newaxis, :, :]# [(ns), ns, 3]
        Breg = Breg.at[i].set(np.sum(
                (np.cross(mu_0Idl_reg[:, np.newaxis, :, :], r_l_reg[np.newaxis, :, :, :]) / 
                    ((np.linalg.norm(r_l_reg, axis=-1) ** 2 + a**2/np.exp(1/2)) ** (3/2))[np.newaxis, :, :, np.newaxis]), 
                        axis=(0, 2)) )
    return Breg


def B_reg_H(I, coil, dl, a, b, nic, ns, delta):
    coil_ev = np.mean(coil[:nic], axis = (1,2))     # 所求的点位
    Breg = np.zeros((nic, ns, 3))      # reg: 线圈对自身场的项

    for i in range(nic):
        coil_reg = np.mean(coil[i], axis = (0,1))       # [ns, 3]
        dl_reg = np.mean(dl[i], axis = (0,1))           # [ns, 3]
        mu_0Idl_reg = (mu_0*I[i, np.newaxis, np.newaxis, np.newaxis] * dl_reg[np.newaxis, :, :])  # [1,ns,3]
        r_l_reg = coil_ev[i, :, np.newaxis, :] - coil_reg[np.newaxis, :, :]# [(ns), ns, 3]
        Breg = Breg.at[i].set(np.sum(
                (np.cross(mu_0Idl_reg[:, np.newaxis, :, :], r_l_reg[np.newaxis, :, :, :]) / 
                    ((np.linalg.norm(r_l_reg, axis=-1) ** 2 + delta*a*b) ** (3/2))[np.newaxis, :, :, np.newaxis]), 
                        axis=(0, 2)) )
    return Breg


def B_local(I_nic, v1, v2, a, curva, Breg):
    theta = -np.arctan(Breg*v1/(Breg*v2))
    mu_0I = mu_0 * I_nic
    Blocal = (mu_0I*2/a * (-np.sin(theta)*v1 + np.cos(theta)*v2) + 
                mu_0I/2*curva * (-np.sin(2*theta)/2*v1) + (1+np.cos(2*theta))/2*v2)

    return Blocal


def B_0(I_nic, u, v, v1, v2, a, b):

    def G(x,y):
            if y == 0 or x == 0:
                return 0
            else:
                return(y * np.arctan(x/y) + x/2*np.log(1+(y/x)**2))

    B0 = mu_0 * I_nic/(a*b) * (v2 * G(b*(v+1), a*(u+1)) - v1 * G(a*(u+1), b*(v+1)) + 
                                (-1) * v2 * G(b*(v+1), a*(u-1)) - v1 * G(a*(u-1), b*(v+1)) + 
                                (-1) * v2 * G(b*(v-1), a*(u+1)) - v1 * G(a*(u+1), b*(v-1)) + 
                                    v2 * G(b*(v-1), a*(u-1)) - v1 * G(a*(u-1), b*(v-1)))
    return B0


def B_k(I_nic, u, v, v1, v2, a, b, k1, k2):

    def K(U,V):
        if U == 0 and V == 0:
            return 0
        elif U == 0 or V == 0:
            return ((k2*v2-k1*v1) * (a/b*U**2 + b/a*V**2) * np.log(a/b*U**2 + b/a*V**2))
        else:
            return (-2*U*V * (k1*v2-k2*v1) * np.log(a/b*U**2 + b/a*V**2) + 
                    (k2*v2-k1*v1) * (a/b*U**2 + b/a*V**2) * np.log(a/b*U**2 + b/a*V**2) + 
                    4*a/b*k2*v1*U**2 * np.arctan(b*V/(a*U)) -
                    4*b/a*k1*v2*V**2 * np.arctan(a*U/(b*V)))

    Bk = mu_0 * I_nic/16 * ( K(u+1, v+1) + K(u-1, v-1) +
                                  (-1) * K(u-1, v+1) + (-1) * K(u+1, v-1))
    return Bk


def B_b(I_nic, curva, binormal, delta):

    return  (mu_0 * I_nic/2 * curva * binormal * (4 + 2*np.log(2) + np.log(delta)))

