import jax.numpy as np
import numpy 
from jax import vmap, config
import scipy.interpolate as si
import jax.scipy.linalg as sl
import plotly.graph_objects as go
config.update("jax_enable_x64", True)
#### 三阶导数不可用

## 在这里需要一个重合点，输入时自带
def get_c_init(init_coil, nc, ns):       
    coil = np.load("{}".format(init_coil))[:, :ns+1, :]
    c, bc  = prep(coil, nc, ns+1, 3)
    return c, bc

## 如果已有参数c,可直接获得bc，ns为参数c的长度
def get_bc_init(ns):
    k = 3
    t = np.linspace(-3/(ns-1), (ns+2)/(ns-1), ns+6) 
    u = np.linspace(0, 1 ,ns)
    bc = [t, u, k]
    return bc


# 考虑闭合线圈，输入线圈需要重合点，coil: [nc/nfp,ns+1,3]
# def tcku(coil, nc, ns, k): 
#     assert coil.shape[1] == ns  
#     j1 = int(np.ceil(ns/2))
#     t = np.linspace(-3/(ns-3), ns/(ns-3), ns+4)
#     u = np.linspace(0, 1 ,ns)
#     mat = np.array([[1/6, 2/3, 1/6, 0], [-1/2, 0, 1/2, 0], [1/2, -1, 1/2, 0], [-1/6, 1/2, -1/2, 1/6]])  

#     c = np.zeros((nc, 3, ns))
#     X = np.zeros((ns, ns))
#     X = X.at[0, 0:4].set(mat[0])
#     for i in range(1, j1):
#         x1 = (u[i] - t[i+2])*(ns-3)
#         X1 = np.array([1, x1, x1*x1, x1*x1*x1])
#         X = X.at[i, i-1:i+3].set(np.dot(X1, mat))

#     for i in range(j1, ns-1):
#         x1 = (u[i] - t[i+1])*(ns-3)
#         X1 = np.array([1, x1, x1*x1, x1*x1*x1])
#         X = X.at[i, i-2:i+2].set(np.dot(X1, mat))
#     X = X.at[ns-1, ns-4:ns].set(np.array([0, 1/6, 4/6, 1/6]))
#     for i in range(nc):
#         x = coil[i, :, 0]   
#         y = coil[i, :, 1]
#         z = coil[i, :, 2]
#         cx = sl.solve(X, x)
#         cy = sl.solve(X, y)
#         cz = sl.solve(X, z)
#         c = c.at[i,:,:].set([cx, cy, cz])
#     bc = [t, u, k]
#     return c, bc


def prep(rc, nc, ns, k):
    x = numpy.array(np.transpose(rc, (0, 2, 1)))
    u = np.linspace(0, 1, ns)
    t = np.linspace(-3/(ns-1), (ns+2)/(ns-1), ns+6) 
    c = np.zeros((nc, 3, ns+2))   
    for i in range(10):
        tck, u = si.splprep(x[i], u=u, k=3, t=t, s=0, per=1)
        c = c.at[i,:,:].set(tck[1])
    bc = [t, u, k]
    return c, bc

def splev(bc, c):            # 矩阵形式计算,与tcku配合 
    t, u, k = bc   
    m = len(u)    # m = 65 = ns
    c = np.array(c)
    xyz = np.zeros((m,3))
    mat = np.array([[1/6, 2/3, 1/6, 0], [-1/2, 0, 1/2, 0], [1/2, -1, 1/2, 0], [-1/6, 1/2, -1/2, 1/6]])  
    for i in range(m-1):
        x1 = (u[i] - t[i+3])*(m-1)
        X1 = np.array([1, x1, x1*x1, x1*x1*x1])
        B1 = np.dot(X1, mat) 
        xyz = xyz.at[i,:].set(np.dot(B1, c[:,i:i+4].T))
    xyz = xyz.at[m-1,:].set(np.dot(np.array([0, 1/6, 2/3, 1/6]), c[:,m-2:m+2].T))
    return xyz

def der1_splev(bc, c):       # 一阶导数  
    t, u, k = bc   
    m = len(u)  
    n = len(c[1])
    wrk1 = np.zeros((3, n-1))    
    c = np.array(c)
    wrk1 = wrk1.at[:, :].set(3*(c[:, 1:]-c[:, :-1])/(3/(m-1)))
    t = np.delete(t, 0)
    der1 = np.zeros((m,3))
    mat = np.array([[1/2, 1/2, 0], [-1, 1, 0], [1/2, -1, 1/2]])
    for i in range(m-1):
        x1 = (u[i] - t[i+2])*(m-1)
        X1 = np.array([1, x1, x1*x1])
        B1 = np.dot(X1, mat) 
        der1 = der1.at[i,:].set(np.dot(B1, wrk1[:,i:i+3].T))
    der1 = der1.at[m-1,:].set(np.dot(np.array([0, 1/2, 1/2]), wrk1[:,m-2:m+1].T))
    return der1, wrk1

def der2_splev(bc, wrk1):       # 一阶导数  
    t, u, k = bc   
    m = len(u)  
    n = len(wrk1[1])
    wrk2 = np.zeros((3, n-1))    
    wrk2 = wrk2.at[:, :].set(2*(wrk1[:, 1:]-wrk1[:, :-1])/(2/(m-1)))
    t = np.delete(t, 0)
    t = np.delete(t, 0)
    der2 = np.zeros((m,3))
    mat = np.array([[1, 0], [-1, 1]])
    for i in range(m-1):
        x1 = (u[i] - t[i+1])*(m-1)
        X1 = np.array([1, x1])
        B1 = np.dot(X1, mat) 
        der2 = der2.at[i,:].set(np.dot(B1, wrk2[:,i:i+2].T))
    der2 = der2.at[m-1,:].set(np.dot(np.array([0, 1]), wrk2[:,m-2:m].T))
    return der2


# def splev(bc, c):            # 矩阵形式计算,与tcku配合 
#     t, u, k = bc   
#     m = len(u)  
#     xyz = np.zeros((m,3))
#     mat = np.array([[1/6, 2/3, 1/6, 0], [-1/2, 0, 1/2, 0], [1/2, -1, 1/2, 0], [-1/6, 1/2, -1/2, 1/6]])  
#     j1 = int(np.ceil(m/2))

#     xyz = xyz.at[0,:].set(np.dot(mat[0], c[:,0:4].T))
#     for i in range(1,j1):
#         x1 = (u[i] - t[i+2])*(m-3)
#         X1 = np.array([1, x1, x1*x1, x1*x1*x1])
#         B1 = np.dot(X1, mat) 
#         xyz = xyz.at[i,:].set(np.dot(B1, c[:,i-1:i+3].T))

#     for i in range(j1, m-1):
#         x1 = (u[i] - t[i+1])*(m-3)
#         X1 = np.array([1, x1, x1*x1, x1*x1*x1])
#         B1 = np.dot(X1, mat) 
#         xyz = xyz.at[i,:].set(np.dot(B1, c[:,i-2:i+2].T))
#     xyz = xyz.at[m-1,:].set(np.dot(np.array([0, 1/6, 2/3, 1/6]), c[:,m-4:m].T))

#     return xyz

# def der1_splev(bc, c):       # 一阶导数  
#     t, u, k = bc   
#     m = len(u)  
#     wrk1 = np.zeros((3, m-1))    
#     wrk1 = wrk1.at[:, :].set(3*(c[:, 1:]-c[:, :-1])/(3/(m-3)))
#     t = np.delete(t, 0)
#     der1 = np.zeros((m,3))
#     mat = np.array([[1/2, 1/2, 0], [-1, 1, 0], [1/2, -1, 1/2]])

#     j1 = int(np.ceil(m/2))
#     der1 = der1.at[0,:].set(np.dot(mat[0], wrk1[:,0:3].T))
#     for i in range(1,j1):
#         x = (u[i] - t[i+2])*(m-3)
#         X = np.array([1, x, x*x])
#         B1 = np.dot(X, mat) 
#         der1 = der1.at[i,:].set(np.dot(B1, wrk1[:,i-1:i+2].T))

#     for i in range(j1, m-1):
#         x = (u[i] - t[i+1])*(m-3)
#         X = np.array([1, x, x*x])
#         B2 = np.dot(X, mat) 
#         der1 = der1.at[i,:].set(np.dot(B2, wrk1[:,i-2:i+1].T))
#     der1 = der1.at[m-1,:].set(np.dot(np.array([0, 1/2, 1/2]), wrk1[:,m-4:m-1].T))

#     return der1, wrk1

# def der2_splev(bc, wrk1):    # 二阶导数 
#     t, u, k = bc 
#     m = len(u)  
#     wrk2 = np.zeros((3, m-2))
#     wrk2 = wrk2.at[:, :].set(2*(wrk1[:, 1:]-wrk1[:, :-1])/(2/(m-3)))
#     t = np.delete(t, 0)
#     t = np.delete(t, 0)
#     der2 = np.zeros((m,3))
#     mat = np.array([[1, 0], [-1, 1]])

#     j1 = int(np.ceil(m/2))
#     der2 = der2.at[0,:].set(np.dot(mat[0], wrk2[:,0:2].T))
#     for i in range(1,j1):
#         x = (u[i] - t[i+2])*(m-3)
#         X = np.array([1, x])
#         B1 = np.dot(X, mat) 
#         der2 = der2.at[i,:].set(np.dot(B1, wrk2[:,i-1:i+1].T))

#     for i in range(j1, m-1):
#         x = (u[i] - t[i+1])*(m-3)
#         X = np.array([1, x])
#         B2 = np.dot(X, mat) 
#         der2 = der2.at[i,:].set(np.dot(B2, wrk2[:,i-2:i].T))
#     der2 = der2.at[m-1,:].set(np.dot(np.array([0, 1]), wrk2[:,m-4:m-2].T))

#     return der2, wrk2


