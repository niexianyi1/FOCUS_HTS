import numpy as np


# x 为输入坐标点
m = len(x)
idim = 3
k = 3
k1 = k + 1  # = 4
k2 = k1 + 1 # = 5
s = 0
tol = 1e-3
nest = m + k + 1         # = m+4
nc = ncc = nest * idim   # = (m+4)*3
n = nest
ub = 0
ue = 1
wrk = []
ifp = 1
iz = ifp+nest      # = n + 1
ia = iz+ncc        # = 1 + (m+4)*4
ib = ia+nest*k1    # = 1 + (m+4)*8
ig = ib+nest*k2    # = 1 + (m+4)*13
iq = ig+nest*k2    # = 1 + (m+4)*18
fpint = wrk(ifp)
z,a,b,g,q = wrk(iz), wrk(ia), wrk(ib), wrk(ig), wrk(iq)
w = np.ones(m, float)
xi = []  # 有10项
h = []   # 有7项
iwrk = [] # 类型为自定义的dfitpack_int，没有找到
nrdata = iwrk




def parcur():
    i1 = 0
    i2 = idim
    u = []
    t = []
    u[0] = 0
    for i in range(1, m):             # 计算u，差值平方？
        dist = 0
        for j in range(idim):
            i1 = i1 + 1
            i2 = i2 + 1
            dist = dist + (x[i2] - x[i1])**2
        u[i] = u[i-1] + np.sqrt(dist)
    for i in range(1, m):
        u[i] = u[i] / u[m-1]
    u[m-1] = 1
    j = n

    for i in range(k1):             # 设置t的首末端点， 前四个为0，后四个为1
        t[i] = 0
        t[j] = 1
        j = j - 1

    # return u, t
    pass


def fpbspl(t,n,k,x,l):       ## bspline的基函数B(此处为h)
    hh = []  # 定义为19项
    h[1] = 1
    for j in range(k):
        for i in range(j):
            hh[i] = h[i]
        h[1] = 1
        for i in range(j):
            li = l + i
            lj = li - j
            if t[li] !=t[lj]:
                f = hh[i]/(t[li] - t[lj])
                h[i] = h[i] + f*(t[li] - x)
                h[i+1] = f*[x - t[lj]]
            else:
                h[i+1] = 1
    # return h
    pass


def fpgivs(piv,ww,cos,sin):     ## 这里的返回值包含ww吗
    store = abs(piv)
    if store >= ww:
        dd = store*np.sqrt(1+(ww/piv)**2)
    if store < ww:
        dd = ww*np.sqrt(1+(piv/ww)**2)
    cos = ww/dd
    sin = piv/dd
    ww = dd
    # return ww, cos, sin
    pass



def fprota(cos,sin,a,b):         ## 这里的返回值是a,b吗
    stor1 = a
    stor2 = b
    b = cos*stor2+sin*stor1
    a = cos*stor1-sin*stor2
    # return a, b
    pass




def fppara(t, u):
    nmin = 2 * k1   # = 8
    acc = tol * s   # = 0
    nmax = m + k1   # = m+4
    n = nmax
    assert nmax == nest
    mk1 = m - k1    # = m-4
    # assert mk1 > 0
    k3 = int(k/2)   # = 1
    i = k2          # = 5
    j = k3 + 2      # = 3
    if k3*2!=k:
        for l in range(mk1):     # 令t的中间量[4,m]等于u[2,m-2]
            t[i-1] = u[j-1]
            i = i+1
            j = j+1
## 大循环,                                             60
    for iter in range(m):
        nrint = n - nmin + 1  # = n-7
        nk1 = n - k1          # = n-4
        i = n
        for j in range(k1):      # 确定t的前后四个点 ,  70
            t[j] = 0
            t[i-1] = 1
            i = i-1
        fp = 0
        for j in range(nc):                          # 75
            z[i] = 0
        for i in range(nk1):                         # 80
            for j in range(k1):
                a[i,j] = 0
        l = k1                 # = 4
        jj = 0
        for it in range(m):     ## 中循环               130
            ui = u[it]          # 获取当前数据点u(it)，x(it)
            wi = w[it]
            for j in range(idim):                     # 83
                jj = jj+1
                xi[j] = x[jj]*wi
            while ui >= t[l+1] and l != nk1:   # 定位ui和t 85
                l = l+1
            h = fpbspl(t,n,k,ui,l)       # h就是bspline的基函数B_j,k(u)
            for i in range(k1):                        # 95
                q[it,i] = h[i]
                h[i] = h[i]*wi           # wi是参数c吗
            j = l - k1             # 将观测矩阵的新行旋转成三角形
            for i in range(k1):    ## 小循环             110
                j = j+1
                piv = h[i]
                while piv != 0:                        # 110
                    a[j,1], cos,sin = fpgivs(piv,a(j,1))   # 计算givens变换参数
                    j1 = j
                    for j2 in range(idim):             # 97
                        xi[j2], z[j1] = fprota(cos,sin,xi(j2),z(j1))   # 右手变换是个啥？
                        j1 = j1+n
                    if i != k1-1:                       
                        i2 = 1
                        i3 = i + 1
                        for i1 in range(i3-1,k1):
                            i2 = i2+1
                            h[i1],a[j,i2] = fprota(cos,sin,h(i1),a(j,i2))  # 左手变换？
            for j2 in range(idim):                       # 125
                fp = fp+xi(j2)**2
        fpint[n] = fp0     # fp0是啥在哪？
        fpint[n-1] = fpold
        nrdata[n] = nplus


                       































