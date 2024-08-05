# 计算B_self, 
# 更改 1.线圈不用有限截面 

import jax.numpy as np
mu_0 = 1e-7

def coil_self_B(args, coil, I, dl, v1, v2, binormal, curva, der2):
    nic = args['number_independent_coils']
    ns = args['number_segments']    
    nn = args['number_normal']
    nb = args['number_binormal']
    I_nic = I[:nic, np.newaxis, np.newaxis]
    curva = np.linalg.norm(curva, axis=-1)

    # if args['material'] == 'NbTi' or args['material'] == 'Nb3Sn':
    #     a = args['HTS_width'] / 2
    #     Bother = B_other(args, I, coil, dl, nic, ns)
    #     Breg = B_reg_cir(I, coil, dl, a, nic, ns)
    #     Blocal, theta = B_local(I_nic, v1, v2, a, curva, Breg)
    #     B_self = Bother + Breg + Blocal
    #     bmax = np.argmax(np.linalg.norm(B_self, axis=-1), axis=1) 
    #     theta_n = []
    #     for i in range(nic):
    #         theta_n.append(theta[i,bmax[i]])
    #     Blocal = B_local_n(I_nic, v1, v2, a, curva, theta_n)
    #     B_self = Bother + Breg + Blocal

    # else:
    # nn = nb = 3
    u,v = 0.999, 0.999
    a = np.array(args['length_normal']) * (nn - 1)   # 正方形截面
    b = np.array(args['length_binormal']) * (nb - 1)

    k1 = curva * (binormal*v2)
    k2 = -curva * (binormal*v1)
    # if a == b :
    k = np.array([2.5565 for i in range(144)])
    delta = np.array([0.19985 for i in range(144)])
    # elif a >= 10*b :
    #     k = 7/6 + np.log(a/b)
    #     delta = a / (b*np.exp(1)**3)
    # else :
    # k = (4*b/(3*a)*np.arctan(a/b) + 4*a/(3*b)*np.arctan(b/a) + b**2/(6*a**2)*np.log(b/a)+
    #         a**2/(6*b**2)*np.log(a/b) - (a**4-6*(a*b)**2+b**4)/(6*(a*b)**2)*np.log(a/b+b/a) )
    # delta = np.exp(-25/6 + k)
    # u, v = 0.999, 0.999  # 截面沿坐标轴方向的位置归一化     
    

    Bother = B_other(args, I, coil, dl, nic, ns)
    Breg = B_reg_rec(args, I, coil, dl, a, b, delta, curva, binormal, der2)
    B0 = B_0(I_nic, u, v, v1, v2, a, b)
    Bk = B_k(I_nic, u, v, v1, v2, a, b, k1, k2)
    Bb = B_b(I_nic, curva, binormal, delta)
    Bself = Bother + Breg + B0 + Bk + Bb

    # print(np.max(np.linalg.norm(Bother, axis=-1), axis=1))
    # print(np.max(np.linalg.norm(Breg, axis=-1), axis=1))
    # print(np.max(np.linalg.norm(B0, axis=-1), axis=1))
    # print(np.max(np.linalg.norm(Bk, axis=-1), axis=1))
    # print(np.max(np.linalg.norm(Bb, axis=-1), axis=1))

    # B_self = np.max(np.linalg.norm(B_self, axis=-1), axis=1)
    return Bself


def coil_self_B_rec(args, coil, I, dl, v1, v2, binormal, curva, der2):
    nic = args['number_independent_coils']
    ns = args['number_segments']    
    nn = args['number_normal']
    nb = args['number_binormal']
    I_nic = I[:nic, np.newaxis, np.newaxis]
    curva = np.linalg.norm(curva, axis=-1)[:, :, np.newaxis]
    UV = [[-0.999, -0.999], [-0.999, 0.999], [0.999, -0.999], [0.999, 0.999]]
    B_self = np.zeros((nic, 4, ns, 3))  # 只算截面上4边的数值
    a = np.array(args['length_normal']) * (nn - 1)   # 正方形截面
    b = np.array(args['length_binormal']) * (nb - 1)

    k1 = curva * (binormal*v2)
    k2 = -curva * (binormal*v1)
    k = (4*b/(3*a)*np.arctan(a/b) + 4*a/(3*b)*np.arctan(b/a) + b**2/(6*a**2)*np.log(b/a)+
            a**2/(6*b**2)*np.log(a/b) - (a**4-6*(a*b)**2+b**4)/(6*(a*b)**2)*np.log(a/b+b/a) )
    delta = np.exp(-25/6 + k)

    Breg = B_reg_rec(args, I, coil, dl, a, b, delta, curva, binormal, der2)
    Bother = B_other(args, I, coil, dl, nic, ns)
    Bb = B_b(I_nic, curva, binormal, delta)
    for i in range(4):
        u,v = UV[i][0], UV[i][1]
        B0 = B_0(I_nic, u, v, v1, v2, a, b)
        Bk = B_k(I_nic, u, v, v1, v2, a, b, k1, k2)
        Bself =  B0 + Bk + Breg + Bb + Bother[i]
        B_self = B_self.at[:, i, :, :].set(Bself)
    return B_self


def loss_B_coil(args, coil, I, dl, v1, v2, binormal, curva, der2):
    nic = args['number_independent_coils']
    ns = args['number_segments']    
    nn = args['number_normal']
    nb = args['number_binormal']
    a = np.array(args['length_normal']) * (nn - 1)   # 正方形截面
    b = np.array(args['length_binormal']) * (nb - 1)
    delta = args['delta']
    Breg = B_reg_rec(args, I, coil, dl, a, b, delta, curva, binormal, der2)
    
    coil_m = np.mean(coil, (1,2))
    dl_m = np.mean(dl, (1,2))
    coil = np.mean(coil[:nic], axis = (1,2))
    Bother = np.zeros((nic, ns, 3))
    for i in range(nic):
        coil_cal = np.delete(coil_m, i, axis=0) 
        dl_cal = np.delete(dl_m, i, axis=0) 
        I_cal = np.delete(I, i, axis=0) 
        r_r = coil[i, :, np.newaxis, np.newaxis, :] - coil_cal[np.newaxis, :, :, :]
        Idl_cal = I_cal[np.newaxis, :, np.newaxis, np.newaxis] * dl_cal[np.newaxis, :, :, :]
        B = mu_0 * np.sum(np.cross(Idl_cal, r_r) / 
            (np.linalg.norm(r_r, axis=-1)[:, :, :, np.newaxis] )**1.5
            , axis = (1,2)
        )
        Bother = Bother.at[i].set(B)
    return Breg+Bother


def coil_self_B_max(args, coil, I, dl, v1, v2, binormal, curva, der2):
    nic = args['number_independent_coils']
    ns = args['number_segments']    
    # I = args['number_normal'] * args['number_binormal'] * I
    I_nic = I[:nic, np.newaxis, np.newaxis]
    curva = np.linalg.norm(curva, axis=-1)[:, :, np.newaxis]

    # if args['material'] == 'NbTi' or args['material'] == 'Nb3Sn':
    #     a = args['length_normal'] * (args['number_normal'] - 1) / 2
    #     Bother = B_other(args, I, coil, dl, nic, ns)
    #     Breg = B_reg_cir(I, coil, dl, a, nic, ns)
    #     Blocal, theta = B_local(I_nic, v1, v2, a, curva, Breg)
    #     B_self = Bother + Breg + Blocal

    # else:
    a = np.array(args['length_normal']) * (args['number_normal'] - 1)   # 正方形截面
    b = np.array(args['length_binormal']) * (args['number_binormal'] - 1)
    u, v = 0.999, 0.999  # 截面沿坐标轴方向的位置归一化 
    k1 = curva * (binormal*v2)
    k2 = -curva * (binormal*v1)
    # if a == b :
        # k = 2.5565
        # delta = 0.19985
    # elif a >= 10*b :
    #     k = 7/6 + np.log(a/b)
    #     delta = a / (b*np.exp(1)**3)
    # else :
    k = (4*b/(3*a)*np.arctan(a/b) + 4*a/(3*b)*np.arctan(b/a) + b**2/(6*a**2)*np.log(b/a)+
            a**2/(6*b**2)*np.log(a/b) - (a**4-6*(a*b)**2+b**4)/(6*(a*b)**2)*np.log(a/b+b/a) )
    delta = np.exp(-25/6 + k)
    Bother = B_other(args, I, coil, dl, nic, ns)
    Breg = B_reg_rec(I, coil, dl, a, b, nic, ns, delta, curva, binormal, der2)
    B0 = B_0(I_nic, u, v, v1, v2, a, b)
    Bk = B_k(I_nic, u, v, v1, v2, a, b, k1, k2)
    Bb = B_b(I_nic, curva, binormal, delta)
    B_self = Bother + Breg + B0 + Bk + Bb
    B_self = np.max(np.linalg.norm(B_self, axis=-1), axis=1)
    return B_self


def coil_self_B_signle(args, coil, I, dl, v1, v2, binormal, curva, der2): # 方形截面
    nic = args['number_independent_coils']
    ns = args['number_segments']    
    I_nic = I[:nic, np.newaxis, np.newaxis]
    curva = np.linalg.norm(curva, axis=-1)[:, :, np.newaxis]

    a = np.array(args['length_normal']) * (args['number_normal'] - 1)   # 正方形截面
    b = np.array(args['length_binormal']) * (args['number_binormal'] - 1)
    u, v = 0.999, 0.999  # 截面沿坐标轴方向的位置归一化 
    k1 = curva * (binormal*v2)
    k2 = -curva * (binormal*v1)
    k = (4*b/(3*a)*np.arctan(a/b) + 4*a/(3*b)*np.arctan(b/a) + b**2/(6*a**2)*np.log(b/a)+
        a**2/(6*b**2)*np.log(a/b) - (a**4-6*(a*b)**2+b**4)/(6*(a*b)**2)*np.log(a/b+b/a) )
    delta = np.exp(-25/6 + k)

    Bother = B_other(args, I, coil, dl, nic, ns)
    Breg = B_reg_rec(I, coil, dl, a, b, nic, ns, delta, curva, binormal, der2)
    B0 = B_0(I_nic, u, v, v1, v2, a, b)
    Bk = B_k(I_nic, u, v, v1, v2, a, b, k1, k2)
    Bb = B_b(I_nic, curva, binormal, delta)
    B_self = Bother + Breg + B0 + Bk + Bb
    return B_self


def coil_B_section_circle(args, coil, I, dl, v1, v2, curva):  

    curva = np.squeeze(curva)[0]
    curva = np.linalg.norm(curva, axis=-1)
    v1 = np.squeeze(v1)[0]
    v2 = np.squeeze(v2)[0]
    a = 2.0427645683730407/2/np.pi/100

    def B_reg_circle(I, coil, dl, a):
        coil_ev = coil  # 所求的点位
        coil_reg = coil[0]      # [ns, 3]
        dl_reg = dl[0]         # [ns, 3]
        mu_0Idl_reg = (mu_0*I * dl_reg[np.newaxis, :, :])  # [1,ns,3]
        r_l_reg = coil_ev[0, :, np.newaxis, :] - coil_reg[np.newaxis, :, :]# [(ns), ns, 3]
        Breg = (np.cross(mu_0Idl_reg[:, np.newaxis, :, :], r_l_reg[np.newaxis, :, :, :]) / 
                ((np.linalg.norm(r_l_reg, axis=-1) ** 2 + a**2/np.exp(0.5)) ** (3/2))[np.newaxis, :, :, np.newaxis])
        Breg = np.sum(Breg, axis = (0,2))
        return Breg

    def B_local_sec(I, v1, v2, a, curva, theta, r):
        mu_0I = mu_0 * I
        Blocal = (mu_0I*2*r/a * (-np.sin(theta)*v1 + np.cos(theta)*v2) + 
                    mu_0I/2*curva * ((-r**2*np.sin(2*theta)/2*v1) + (3+r**2*(np.cos(2*theta)-2))/2*v2))
        return Blocal


    Breg = B_reg_circle(I, coil, dl, a)[0]

    B = np.zeros((80,80))
    for i in range(80):
        x = 0.999 - 0.0999*i/4
        for j in range(80):
            y = 0.999 - 0.0999*j/4
            if x < 0:    
                theta = np.arctan(y / x) + np.pi
                r = x / np.cos(theta)
            elif x > 0:
                theta = np.arctan(y / x)
                r = x / np.cos(theta)
            elif x == 0:
                if y>=0:
                    theta = np.pi/2
                    r = y
                elif y<0:
                    theta = -np.pi/2
                    r = abs(y)
            Blocal = B_local_sec(I, v1, v2, a, curva, theta, r)
            B_self = Breg + Blocal
            B_self = np.linalg.norm(B_self)
            B = B.at[i,j].set(B_self)
    Breg = B_reg_circle(I, coil, dl, a)

    return B, Breg


def coil_B_section_square(args, coil, I, dl, v1, v2, binormal, curva): 

    # curva = np.squeeze(curva)[0]
    binormal = np.squeeze(binormal)[0]
    v1 = np.squeeze(v1)[0]
    v2 = np.squeeze(v2)[0]
    a = np.array([0.1296]) 
    b = np.array([0.0568])
    k1 = curva * np.dot(binormal,v2)
    k2 = -curva * np.dot(binormal,v1)

    k = (4*b/(3*a)*np.arctan(a/b) + 4*a/(3*b)*np.arctan(b/a) + b**2/(6*a**2)*np.log(b/a)+
            a**2/(6*b**2)*np.log(a/b) - (a**4-6*(a*b)**2+b**4)/(6*(a*b)**2)*np.log(a/b+b/a) )
    delta = np.exp(-25/6 + k)
    k = np.array([k])
    delta = np.array([delta])
    
    def B_reg_s(I, coil, dl, a, b, delta):
        coil_ev = coil  # 所求的点位
        coil_reg = coil[0]      # [ns, 3]
        dl_reg = dl[0]         # [ns, 3]
        mu_0Idl_reg = (mu_0*I * dl_reg[np.newaxis, :, :])  # [1,ns,3]
        r_l_reg = coil_ev[0, :, np.newaxis, :] - coil_reg[np.newaxis, :, :]# [(ns), ns, 3]
        Breg = (np.cross(mu_0Idl_reg[:, np.newaxis, :, :], r_l_reg[np.newaxis, :, :, :]) / 
                ((np.linalg.norm(r_l_reg, axis=-1) ** 2 + delta*a*b) ** (3/2))[np.newaxis, :, :, np.newaxis])
        Breg = np.sum(Breg, axis = (0,2))
        return Breg

    Breg = B_reg_s(I, coil, dl, a, b, delta)[0]
    Bb = B_b(I, curva, binormal, delta)
    B = np.zeros((64,65))
    almost_one = 1 - (1e-6)
    for i in range(64):
        # u = 0.999 - 0.0999*i/4
        u = ((63-i)/63*2-1)*almost_one
        for j in range(65):
            # v = 0.999 - 0.0999*j/4
            v = ((64-j)/64*2-1)*almost_one
            B0 = B_0(I, u, v, v1, v2, a, b)
            Bk = B_k(I, u, v, v1, v2, a, b, k1, k2)
            B_self = Bk + B0 + Breg + Bb
            B_self = np.linalg.norm(B_self)
            B = B.at[i,j].set(B_self)
    Breg = B_reg_s(I, coil, dl, a, b, delta)
    return B, Breg


def B_other(args, I, coil, dl, nic, ns):     # failment近似看一下结果
    nic = args['number_independent_coils']
    ns = args['number_segments'] 
    nn = args['number_normal']
    nb = args['number_binormal']
    coil_m = np.mean(coil, (1,2))
    dl_m = np.mean(dl, (1,2))
    coil_4 = np.zeros((4, nic, ns, 3))
    coil_4 = coil_4.at[0].set(coil[:nic, 0, 0, :, :])
    coil_4 = coil_4.at[1].set(coil[:nic, 0, nb-1, :, :])
    coil_4 = coil_4.at[2].set(coil[:nic, nn-1, 0, :, :])
    coil_4 = coil_4.at[3].set(coil[:nic, nn-1, nb-1, :, :])
    Bother = np.zeros((4, nic, ns, 3))
    for i in range(nic):
        coil_cal = np.delete(coil_m, i, axis=0) 
        dl_cal = np.delete(dl_m, i, axis=0) 
        I_cal = np.delete(I, i, axis=0) 
        r_r = coil_4[:, i, :, np.newaxis, np.newaxis, :] - coil_cal[np.newaxis, np.newaxis, :, :, :]
        Idl_cal = I_cal[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, ] * dl_cal[np.newaxis, np.newaxis, :, :, :]
        B = mu_0 * np.sum(np.cross(Idl_cal, r_r) / 
            (np.linalg.norm(r_r, axis=-1)[:, :, :, :, np.newaxis] )**1.5
            , axis = (2,3)
        )
        Bother = Bother.at[:, i].set(B)

    # coil = np.mean(coil[:nic], axis = (1,2))
    # Bother = np.zeros((nic, ns, 3))
    # for i in range(nic):
    #     coil_cal = np.delete(coil_m, i, axis=0) 
    #     dl_cal = np.delete(dl_m, i, axis=0) 
    #     I_cal = np.delete(I, i, axis=0) 
    #     r_r = coil[i, :, np.newaxis, np.newaxis, :] - coil_cal[np.newaxis, :, :, :]
    #     Idl_cal = I_cal[np.newaxis, :, np.newaxis, np.newaxis] * dl_cal[np.newaxis, :, :, :]
    #     B = mu_0 * np.sum(np.cross(Idl_cal, r_r) / 
    #         (np.linalg.norm(r_r, axis=-1)[:, :, :, np.newaxis] )**1.5
    #         , axis = (1,2)
    #     )
    #     Bother = Bother.at[i].set(B)
    return Bother


def B_reg_cir(I, coil, dl, a, nic, ns):

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


def B_reg_rec(args, I, coil, dl, a, b, delta, curva, binormal, der2):
    nic = args['number_independent_coils']
    ns = args['number_segments']    
    if args['coil_case'] == 'fourier':
        dt = 2 * np.pi / ns
    else:
        dt = 1 / ns
    phi = np.linspace(0, 2*np.pi, ns+1)
    coil = np.mean(coil[:nic], (1,2))
    dl = np.mean(dl[:nic], (1,2))
    Breg = np.zeros((nic, ns, 3))      # reg: 线圈对自身场的项
    for i in range(nic):
        r_r = coil[i, np.newaxis, :, :] - coil[i, :, np.newaxis, :]
        for j in range(ns):
            r_r = r_r.at[j,j,:].set(0)
        B2 = mu_0 * I[i] * np.sum(np.cross(dl[i, :, np.newaxis, :], r_r) / 
            (np.linalg.norm(r_r, axis=-1)[:, :, np.newaxis] + a[i]*b[i]*delta[i])**1.5
            , axis = 1
        )  
        
        Breg = Breg.at[i].set(B2)

        # B1 = mu_0*I[i]*curva[i]*binormal[i]*(np.log(64/(delta[i]*a[i]*b[i])*dl_reg**2)-2)
        # B3 = mu_0*I[i]*np.cross(dl_reg/dt, der2[i]) * np.sum((1-np.cos(phi[np.newaxis, :]-phi[:, np.newaxis]))/
        #     ((2-2*np.cos(phi[np.newaxis, :]-phi[:, np.newaxis]))*np.linalg.norm(dl_reg/dt)**2 + delta[i]*a[i]*b[i]) ** (3/2)
        #                 ,axis=0)
        # Breg = Breg.at[i].add(B1+B2-B3)

    return Breg


def B_0(I_nic, u, v, v1, v2, a, b):
    a = a[:, np.newaxis, np.newaxis]
    b = b[:, np.newaxis, np.newaxis]

    def G(x,y):
        return (y * np.arctan(x/y) + x/2*np.log(1+(y/x)**2))

    B0 = mu_0 * I_nic/(a*b) * ( (v2 * G(b*(v+1), a*(u+1)) - v1 * G(a*(u+1), b*(v+1))) + 
                                (-1) * (v2 * G(b*(v+1), a*(u-1)) - v1 * G(a*(u-1), b*(v+1))) + 
                                (-1) * (v2 * G(b*(v-1), a*(u+1)) - v1 * G(a*(u+1), b*(v-1))) + 
                                (v2 * G(b*(v-1), a*(u-1)) - v1 * G(a*(u-1), b*(v-1))))
    return B0


def B_k(I_nic, u, v, v1, v2, a, b, k1, k2):
    a = a[:, np.newaxis, np.newaxis]
    b = b[:, np.newaxis, np.newaxis]
    def K(U,V):
        return ((-2*U*V * (k1*v2-k2*v1) * np.log(a/b*U**2 + b/a*V**2) + 
                (k2*v2-k1*v1) * (a/b*U**2 + b/a*V**2) * np.log(a/b*U**2 + b/a*V**2) + 
                4*a/b*k2*v1*U**2 * np.arctan(b*V/(a*U)) -
                4*b/a*k1*v2*V**2 * np.arctan(a*U/(b*V))) )

    Bk = mu_0 * I_nic/16 * ( K(u+1, v+1) + K(u-1, v-1) +
                                  (-1) * K(u-1, v+1) + (-1) * K(u+1, v-1))
    return Bk


def B_b(I_nic, curva, binormal, delta):
    return  (mu_0 * I_nic/2 * curva * binormal * (4 + 2*np.log(2) + np.log(delta))[:, np.newaxis, np.newaxis])

