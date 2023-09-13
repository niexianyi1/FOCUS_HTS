import jax.numpy as np
from jax import jit
import scipy

@jit
def get_c_init(init_coil, nc, ns):       
    coil = np.load("{}".format(init_coil))
    c, bc  = tcku(coil, nc, ns, 3)
    return c, bc 

@jit
def get_bc_init(ns):
    N = ns+3
    k = 3
    t = np.zeros(N+k+1)
    u = np.zeros(N)  
    t = t.at[N:N+k+1].set(1)
    t = t.at[(k+1):N].set(np.arange(2/(N-1),(N-2)/(N-1),1/(N-1)))
    u = np.arange(0,N/(N-1),1/(N-1))
    bc = [t, u, k]
    return bc




@jit
def tcku(coil, nc, ns, k):   
    N = ns+3
    t = np.zeros(N+k+1)
    c = np.zeros((nc, 3, N))
    u = np.zeros(N)    
    X = np.zeros((N, N))
    X = X.at[0, 0].set(1)
    X = X.at[1, 0:4].set([1/8, 37/72, 23/72, 1/24])
    X = X.at[2, 1:4].set([1/9, 5/9, 1/3])
    X = X.at[3, 2:5].set([1/8, 17/24, 1/6])
    X = X.at[N-4, N-5:N-2].set([1/6, 17/24, 1/8])
    X = X.at[N-3, N-4:N-1].set([1/3, 5/9, 1/9])
    X = X.at[N-2, N-4:N].set([1/24, 23/72, 37/72, 1/8])
    X = X.at[N-1, N-1].set(1)
    for i in range(N-8):
        X = X.at[i+4, i+3:i+6].set([1/6, 2/3, 1/6])
    for i in range(nc):
        x0 = coil[i, :, 0]         #重复3个点
        x1 = x0[1:3]
        x0 = np.append(x0, x1)
        y0 = coil[i, :, 1]
        y1 = y0[1:3]
        y0 = np.append(y0, y1)
        z0 = coil[i, :, 2]
        z1 = z0[1:3]
        z0 = np.append(z0, z1)
        cx = scipy.linalg.solve(X, x0)
        cy = scipy.linalg.solve(X, y0)
        cz = scipy.linalg.solve(X, z0)
        c = c.at[i,:,:].set([cx, cy, cz])

    t = t.at[N:N+k+1].set(1)
    t = t.at[(k+1):N].set(np.arange(2/(N-1),(N-2)/(N-1),1/(N-1)))
    u = np.arange(0,N/(N-1),1/(N-1))
    bc = [t, u, k]
    return c, bc

@jit
def splev(bc, c):            # 矩阵形式计算,与tcku配合 
    t, u, k = bc   
    m = len(u)  
    xyz = np.zeros((m,3))
    mat1 = np.array([[1, 0, 0, 0], [-3/2, 3/2, 0, 0], [3/4, -5/4, 1/2, 0], [-1/8, 19/72, -13/72, 1/24]])  #u[0] and u[1]
    mat2 = np.array([[1/9, 5/9, 1/3, 0], [-1/3, -1/6, 1/2, 0], [1/3, -7/12, 1/4, 0], [-1/9, 23/72, -3/8, 1/6]])    #u[2] 
    mat3 = np.array([[1/8, 17/24, 1/6, 0], [-3/8, -1/8, 1/2, 0], [3/8, -7/8, 1/2, 0], [-1/8, 11/24, -1/2, 1/6]])     #u[3]
    mat = np.array([[1/6, 2/3, 1/6, 0], [-1/2, 0, 1/2, 0], [1/2, -1, 1/2, 0], [-1/6, 1/2, -1/2, 1/6]])        #u[4:m-4]
    # mat_4 = np.array([[1/6, 2/3, 1/6, 0], [-1/2, 0, 1/2, 0], [1/2, -1, 1, 0], [-1/6, 1/2, -11/24, 1/8]])
    mat_3 = np.array([[1/6, 17/24, 1/8, 0], [-1/2, 1/8, 3/8, 0], [1/2, -7/8, 3/8, 0], [1/6, 3/8, -23/72, 1/12]])     #u[m-4]
    mat_2 = np.array([[1/3, 5/9, 1/9, 0], [-1/2, 1/6, 1/3, 0], [1/4, -7/12, 1/3, 0], [-1/24, 13/72, -19/72, 1/8]])  #u[m-3] and u[m-2]
    mat_1 = np.array([[0, 0, 0, 1],[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])        #u[m-1]

    xyz = xyz.at[0,:].set(np.dot(mat1[0], c[:,0:4].T))
    x1 = (u[1] - t[3])*(m-1)
    X1 = np.array([1, x1, x1*x1, x1*x1*x1])
    B1 = np.dot(X1, mat1) 
    xyz = xyz.at[1,:].set(np.dot(B1, c[:,0:4].T))
    xyz = xyz.at[2,:].set(np.dot(mat2[0], c[:,1:5].T))
    xyz = xyz.at[3,:].set(np.dot(mat3[0], c[:,2:6].T))
    for i in range(4, m-4):
        xyz = xyz.at[i,:].set(np.dot(mat[0], c[:,i-1:i+3].T))
    xyz = xyz.at[m-4,:].set(np.dot(mat_3[0], c[:,m-5:m-1].T))
    xyz = xyz.at[m-3,:].set(np.dot(mat_2[0], c[:,m-4:m].T))
    x_2 = (u[m-2] - t[m-1])*(m-1)
    X_2 = np.array([1, x_2, x_2*x_2, x_2*x_2*x_2])
    B_2 = np.dot(X_2, mat_2) 
    xyz = xyz.at[m-2,:].set(np.dot(B_2, c[:,m-4:m].T))
    xyz = xyz.at[m-1,:].set(np.dot(mat_1[0], c[:,m-4:m].T))
    return xyz[0:m-2, :]
@jit
def der1_splev(bc, c):       # 一阶导数  
    t, u, k = bc   
    m = len(u)  
    wrk1 = c 
    for i in range(m-1):    #能不能简化
        for j in range(3):    
            wrk1 = wrk1.at[j,i].set(3*(wrk1[j, i+1]-wrk1[j, i])/(t[i+4]-t[i+1]))
    t = np.delete(t, 0)
    der1 = np.zeros((m,3))
    mat1 = np.array([[1, 0, 0], [-1, 1, 0], [1/4, -5/12, 1/6]])  #u[0] and u[1]
    mat2 = np.array([[1/3, 2/3, 0], [-2/3, 2/3, 0], [1/3, -5/6, 1/2]])    #u[2] 
    mat = np.array([[1/2, 1/2, 0], [-1, 1, 0], [1/2, -1, 1/2]])        #u[3:m-3]
    # mat_3 = np.array([[1/2, 1/2, 0], [-1, 1, 0], [1/2, -5/6, 1/3]])
    mat_2 = np.array([[2/3, 1/3, 0], [-2/3, 2/3, 0], [1/6, -5/12, 1/4]])   #u[m-3] and u[m-2]
    mat_1 = np.array([[0, 0, 1],[0, 0, 0], [0, 0, 0]])        #u[m-1]
    
    der1 = der1.at[0,:].set(np.dot(mat1[0], wrk1[:,0:3].T))
    x1 = (u[1] - t[2])*(m-1)
    X1 = np.array([1, x1, x1*x1])
    B1 = np.dot(X1, mat1) 
    der1 = der1.at[1,:].set(np.dot(B1, wrk1[:,0:3].T))
    der1 = der1.at[2,:].set(np.dot(mat2[0], wrk1[:,1:4].T))
    for i in range(3, m-3):
        der1 = der1.at[i,:].set(np.dot(mat[0], wrk1[:,i-1:i+2].T))
    der1 = der1.at[m-3,:].set(np.dot(mat_2[0], wrk1[:,m-4:m-1].T))
    x_2 = (u[m-2] - t[m-2])*(m-1)
    X_2 = np.array([1, x_2, x_2*x_2])
    B_2 = np.dot(X_2, mat_2) 
    der1 = der1.at[m-2,:].set(np.dot(B_2, wrk1[:,m-4:m-1].T))
    der1 = der1.at[m-1,:].set(np.dot(mat_1[0], wrk1[:,m-4:m-1].T))
    der1 = der1.at[0, :].set(der1[m-3, :])
    return der1[0:m-2, :], wrk1
@jit
def der2_splev(bc, wrk1):    # 二阶导数 
    t, u, k = bc 
    m = len(u)  
    wrk2 = wrk1

    for i in range(m-2):    
        for j in range(3):
            wrk2 = wrk2.at[j,i].set(2*(wrk2[j, i+1]-wrk2[j, i])/(t[i+4]-t[i+2]))
    t = np.delete(t, 0)
    t = np.delete(t, 0)
    der2 = np.zeros((m,3))
    mat1 = np.array([[1, 0], [-1/2, 1/2]])  #u[0] and u[1]
    mat = np.array([[1, 0], [-1, 1]])        #u[2:m-2]
    mat_2 = np.array([[1, 0], [-1/2, 1/2]])   #u[m-3] and u[m-2]
    mat_1 = np.array([[0, 1],[0, 0]])        #u[m-1]

    der2 = der2.at[0,:].set(np.dot(mat1[0], wrk2[:,0:2].T))
    x1 = (u[1] - t[1])*(m-1)
    X1 = np.array([1, x1])
    B1 = np.dot(X1, mat1) 
    der2 = der2.at[1,:].set(np.dot(B1, wrk2[:,0:2].T))
    for i in range(2, m-2):
        der2 = der2.at[i,:].set(np.dot(mat[0], wrk2[:,i-1:i+1].T))
    x_2 = (u[m-2] - t[m-3])*(m-1)
    X_2 = np.array([1, x_2])
    B_2 = np.dot(X_2, mat_2) 
    der2 = der2.at[m-2,:].set(np.dot(B_2, wrk2[:,m-4:m-2].T))
    der2 = der2.at[m-1,:].set(np.dot(mat_1[0], wrk2[:,m-4:m-2].T))
    der2 = der2.at[0, :].set(der2[m-3, :])
    return der2[0:m-2, :], wrk2
@jit
def der3_splev(bc, wrk2):    # 三阶导数    
    t, u, k = bc  
    m = len(u)  
    wrk3 = wrk2
    for i in range(m-3):    
        for j in range(3):
            wrk3 = wrk3.at[j,i].set((wrk2[j, i+1]-wrk2[j, i])/(t[i+4]-t[i+3]))
    der3 = np.zeros((m,3))

    for i in range(1, m-2):
        der3 = der3.at[i, :].set(wrk3[:, i-1].T)
    der3 = der3.at[m-2, :].set(der3[m-3, :])
    der3 = der3.at[m-1, :].set(der3[m-3, :])
    der3 = der3.at[0, :].set(der3[m-3, 0])
    return der3[0:m-2, :]


