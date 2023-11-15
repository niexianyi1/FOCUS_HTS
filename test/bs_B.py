### 计算并验证bspline的基函数

import jax.nupmy as np


ns=65
def B(x, k, i, t):
    if k == 0:
        if t[i] <= x and x < t[i+1]:
            return 1.0  
        else:
            return 0.0
    if t[i+k] == t[i]:
        c1 = 0.0
    else:
        c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
        c2 = 0.0
    else:
        c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
    return c1 + c2

def bspline(x, t, c, k):
    n = len(t) - k - 1
    bb = np.zeros((n, 4))    
    assert (n >= k+1) and (len(c) >= n)
    for i in range(n):
        aa = 0
        for j in range(n):
            b = B(x[i], k, j, t)
            if b != 0:
                bb = bb.at[i,aa].set(b)
                aa = aa+1
        print('---------------------')
    return bb

t = np.linspace(-3/(ns-3), ns/(ns-3), ns+4)
u = np.linspace(0, 1 ,ns)
c = np.ones(ns)
bb = bspline(u, t, c, 3)


def tcku(ns):
    j1 = int(np.ceil(ns/2))
    t = np.linspace(-3/(ns-3), ns/(ns-3), ns+4)
    u = np.linspace(0, 1 ,ns)
    mat = np.array([[1/6, 2/3, 1/6, 0], [-1/2, 0, 1/2, 0], [1/2, -1, 1/2, 0], [-1/6, 1/2, -1/2, 1/6]])  

    X = np.zeros((ns, 4))
    X = X.at[0, :].set(mat[0])
    for i in range(1, j1):
        x1 = (u[i] - t[i+2])*(ns-3)
        X1 = np.array([1, x1, x1*x1, x1*x1*x1])
        # print('i = ', i)
        # print(x1)
        # print(X1)
        # print(np.dot(X1, mat))
        X = X.at[i, :].set(np.dot(X1, mat))

    for i in range(j1, ns-1):
        x1 = (u[i] - t[i+1])*(ns-3)
        X1 = np.array([1, x1, x1*x1, x1*x1*x1])
        # print('i = ', i)
        # print(x1)
        # print(X1)
        # print(np.dot(X1, mat))
        X = X.at[i, :].set(np.dot(X1, mat))
    X = X.at[ns-1, :].set(np.array([0, 1/6, 4/6, 1/6]))
    return X
bx = tcku(ns)
print(bb)
print('----------------')
print(bx)
print('----------------')
print(bb-bx)