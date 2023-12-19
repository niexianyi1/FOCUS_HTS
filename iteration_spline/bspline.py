import jax.numpy as np
import numpy 
from jax import config
import scipy.interpolate as si
config.update("jax_enable_x64", True)    # float64
#### 三阶导数不可用
#### 添加ncp项，和ns做区分

## 在这里需要一个重合点，输入时自带
def get_c_init(init_coil, file_type, nic, ns, ncp):       
    """开始运行程序时调用,下接prep和tjev

    Args:
        init_coil, 初始线圈文件(.npy格式)
        file_type, 文件类型
        nic, 独立线圈个数
        ns, 线圈段数
        ncp, 控制点数

    Returns:
        c, B样条函数控制点,优化参数
        bc, B样条函数其他参数,非优化
        tj, 节点序列
    """
    # 按格式读取文件
    if file_type == 'npy':                       
        coil = np.load("{}".format(init_coil))
    if file_type == 'makegrid':
        coil = read_makegrid(init_coil, nic, ns)
    # 引用函数计算
    c, bc  = prep(coil, nic, ns, ncp, 3)    # c是控制点，作为优化变量，其他非优化参数都包含在bc里
    tj = tjev(bc)    # 获取节点序列，见文档
    return c, bc, tj

def read_makegrid(filename, nic, ns):      
    """读取makegrid文件"""
    r = np.zeros((nic, ns+1, 3))
    with open(filename) as f:
        _ = f.readline()
        _ = f.readline()
        _ = f.readline()
        for i in range(nic):
            for s in range(ns):
                x = f.readline().split()
                r = r.at[i, s, 0].set(float(x[0]))
                r = r.at[i, s, 1].set(float(x[1]))
                r = r.at[i, s, 2].set(float(x[2]))
            _ = f.readline()
    r = r.at[:, -1, :].set(r[:, 0, :])
    return r


## 如果已有参数c,可从此处直接获得bc。
def get_bc_init(ns, ncp):
    """初始给定控制点时调用"""
    k = 3
    t = np.linspace(-3/(ncp-3), (ncp)/(ncp-3), ncp+4) 

    t=t.at[3].set(int(0))     # 此处是为了防止1e-17这种情况，手动设为0。
    # 这个.at[].set()的格式是jax支持的格式，等价为numpy的 t[3]=0.

    u = np.linspace(0, (ns-1)/ns ,ns)
    bc = [t, u, k]
    tj = tjev(bc)
    return bc, tj

def tjev(bc):
    """获取节点序列,在后续计算中无需重复"""
    t, u, _ = bc
    t0 = numpy.zeros_like(t)    
    u0 = numpy.zeros_like(u)
    tj = numpy.zeros_like(u)  

    j = 0
    # t 和u长度不同，分开提取，用numpy格式的数组进行比较，避免jax的格式问题。
    for i in range(len(t)):
        t0[i] = t[i]
    for i in range(len(u)):  
        u0[i] = u[i]
        while u0[i]>=t0[j+1] :
            j = j+1
        tj[i] = j
    return tj

def prep(rc, nic, ns, ncp, k):  
    """获取控制点c,
    Args:
        rc, 线圈坐标
        nic, 独立线圈个数
        ns, 线圈段数
        ncp, 控制点数
        k, 阶数
    Returns:
        c, B样条函数控制点,优化参数
        bc, B样条函数其他参数,非优化参数
    """
    rc = numpy.array(np.transpose(rc, (0, 2, 1))) # 格式转换，转为scipy支持的格式
    c = np.zeros((nic, 3, ncp))   
    t = np.linspace(-3/(ncp-3), (ncp)/(ncp-3), ncp+4) 
    t = t.at[3].set(int(0))
    for i in range(nic):
        tck, _ = si.splprep(x = rc[i], k=3, per=1)  # 调用scipy
        c = c.at[i,:,:].set(tck[1])
    u = np.linspace(0, (ns-1)/ns ,ns)
    bc = [t, u, k]
    return c, bc


def splev(t, u, c, tj, ns):   
    """ B样条函数计算
    Args:
        t, B样条函数节点
        u, 输入点，[0,1,1/ns]
        c, 控制点
        tj, 节点序列
        ns, 线圈段数
    Returns:
        xyz, 输出点
    """
    c = np.array(c)
    xyz = np.zeros((ns,3))
    mat = np.array([[1/6, 2/3, 1/6, 0], [-1/2, 0, 1/2, 0], [1/2, -1, 1/2, 0], [-1/6, 1/2, -1/2, 1/6]])  
    for i in range(ns):
        j = int(tj[i])
        x1 = (u[i] - t[j])/(t[j]-t[j-1])
        X1 = np.array([1, x1, x1*x1, x1*x1*x1])
        B1 = np.dot(X1, mat)  
        xyz = xyz.at[i,:].set(np.dot(B1, c[:,j-3:j+1].T))
    return xyz

def der1_splev(t, u, c, tj, ns):       # 一阶导数  
    """ B样条函数一阶导数计算"""
    ncp = len(c[1])
    wrk1 = np.zeros((3, ncp-1))    
    c = np.array(c)
    wrk1 = wrk1.at[:, :].set((c[:, 1:]-c[:, :-1])/(1/(ncp-3)))
    t = np.delete(t, 0)     # 此处删除一个节点是为了能和wrk的序列匹配，
    der1 = np.zeros((ns,3))
    mat = np.array([[1/2, 1/2, 0], [-1, 1, 0], [1/2, -1, 1/2]])
    for i in range(ns):
        j = int(tj[i])-1    # 此处 -1是与上面删除的一个节点匹配
        x1 = (u[i] - t[j])/(t[j]-t[j-1])
        X1 = np.array([1, x1, x1*x1])
        B1 = np.dot(X1, mat) 
        der1 = der1.at[i,:].set(np.dot(B1, wrk1[:,j-2:j+1].T))
    return der1, wrk1

def der2_splev(t, u, wrk1, tj, ns):       # 一阶导数   
    """ B样条函数二阶导数计算"""
    ncp = len(wrk1[1])
    wrk2 = np.zeros((3, ncp-1))    
    wrk2 = wrk2.at[:, :].set((wrk1[:, 1:]-wrk1[:, :-1])/(1/(ncp-3)))
    t = np.delete(t, 0)
    t = np.delete(t, 0)
    der2 = np.zeros((ns,3))
    mat = np.array([[1, 0], [-1, 1]])
    for i in range(ns):
        j = int(tj[i])-2
        x1 = (u[i] - t[j])/(t[j]-t[j-1])
        X1 = np.array([1, x1])
        B1 = np.dot(X1, mat) 
        der2 = der2.at[i,:].set(np.dot(B1, wrk2[:,j-1:j+1].T))
    return der2





'''def splev(bc, c):           
    t, u, k = bc   
    m = len(u)    # m = 65 = ns+1
    c = np.array(c)
    xyz = np.zeros((m,3))
    mat = np.array([[1/6, 2/3, 1/6, 0], [-1/2, 0, 1/2, 0], [1/2, -1, 1/2, 0], [-1/6, 1/2, -1/2, 1/6]])  
    for i in range(m-1):
        x1 = (u[i] - t[i+3])*(m-1)
        X1 = np.array([1, x1, x1*x1, x1*x1*x1])
        B1 = np.dot(X1, mat)  
        xyz = xyz.at[i,:].set(np.dot(B1, c[:,i:i+4].T))
    return xyz

def der1_splev(bc, c):       # 一阶导数  
    t, u, k = bc   
    m = len(u)  
    n = len(c[1])
    wrk1 = np.zeros((3, n-1))    
    c = np.array(c)
    wrk1 = wrk1.at[:, :].set(3*(c[:, 1:]-c[:, :-1])/(3/(m-1)))
    t = np.delete(t, 0)
    der1 = np.zeros((m-1,3))
    mat = np.array([[1/2, 1/2, 0], [-1, 1, 0], [1/2, -1, 1/2]])
    for i in range(m-1):
        x1 = (u[i] - t[i+2])*(m-1)
        X1 = np.array([1, x1, x1*x1])
        B1 = np.dot(X1, mat) 
        der1 = der1.at[i,:].set(np.dot(B1, wrk1[:,i:i+3].T))
    return der1, wrk1

def der2_splev(bc, wrk1):       # 一阶导数  
    t, u, k = bc   
    m = len(u)  
    n = len(wrk1[1])
    wrk2 = np.zeros((3, n-1))    
    wrk2 = wrk2.at[:, :].set(2*(wrk1[:, 1:]-wrk1[:, :-1])/(2/(m-1)))
    t = np.delete(t, 0)
    t = np.delete(t, 0)
    der2 = np.zeros((m-1,3))
    mat = np.array([[1, 0], [-1, 1]])
    for i in range(m-1):
        x1 = (u[i] - t[i+1])*(m-1)
        X1 = np.array([1, x1])
        B1 = np.dot(X1, mat) 
        der2 = der2.at[i,:].set(np.dot(B1, wrk2[:,i:i+2].T))
    return der2

'''

# 考虑闭合线圈，输入线圈需要重合点，coil: [nc/nfp,ns+1,3]
'''def tcku(coil, nc, ns, k): 
    assert coil.shape[1] == ns  
    j1 = int(np.ceil(ns/2))
    t = np.linspace(-3/(ns-3), ns/(ns-3), ns+4)
    u = np.linspace(0, 1 ,ns)
    mat = np.array([[1/6, 2/3, 1/6, 0], [-1/2, 0, 1/2, 0], [1/2, -1, 1/2, 0], [-1/6, 1/2, -1/2, 1/6]])  

    c = np.zeros((nc, 3, ns))
    X = np.zeros((ns, ns))
    X = X.at[0, 0:4].set(mat[0])
    for i in range(1, j1):
        x1 = (u[i] - t[i+2])*(ns-3)
        X1 = np.array([1, x1, x1*x1, x1*x1*x1])
        X = X.at[i, i-1:i+3].set(np.dot(X1, mat))

    for i in range(j1, ns-1):
        x1 = (u[i] - t[i+1])*(ns-3)
        X1 = np.array([1, x1, x1*x1, x1*x1*x1])
        X = X.at[i, i-2:i+2].set(np.dot(X1, mat))
    X = X.at[ns-1, ns-4:ns].set(np.array([0, 1/6, 4/6, 1/6]))
    for i in range(nc):
        x = coil[i, :, 0]   
        y = coil[i, :, 1]
        z = coil[i, :, 2]
        cx = sl.solve(X, x)
        cy = sl.solve(X, y)
        cz = sl.solve(X, z)
        c = c.at[i,:,:].set([cx, cy, cz])
    bc = [t, u, k]
    return c, bc

def splev(bc, c):            # 矩阵形式计算,与tcku配合 
    t, u, k = bc   
    m = len(u)  
    xyz = np.zeros((m,3))
    mat = np.array([[1/6, 2/3, 1/6, 0], [-1/2, 0, 1/2, 0], [1/2, -1, 1/2, 0], [-1/6, 1/2, -1/2, 1/6]])  
    j1 = int(np.ceil(m/2))

    xyz = xyz.at[0,:].set(np.dot(mat[0], c[:,0:4].T))
    for i in range(1,j1):
        x1 = (u[i] - t[i+2])*(m-3)
        X1 = np.array([1, x1, x1*x1, x1*x1*x1])
        B1 = np.dot(X1, mat) 
        xyz = xyz.at[i,:].set(np.dot(B1, c[:,i-1:i+3].T))

    for i in range(j1, m-1):
        x1 = (u[i] - t[i+1])*(m-3)
        X1 = np.array([1, x1, x1*x1, x1*x1*x1])
        B1 = np.dot(X1, mat) 
        xyz = xyz.at[i,:].set(np.dot(B1, c[:,i-2:i+2].T))
    xyz = xyz.at[m-1,:].set(np.dot(np.array([0, 1/6, 2/3, 1/6]), c[:,m-4:m].T))

    return xyz

def der1_splev(bc, c):       # 一阶导数  
    t, u, k = bc   
    m = len(u)  
    wrk1 = np.zeros((3, m-1))    
    wrk1 = wrk1.at[:, :].set(3*(c[:, 1:]-c[:, :-1])/(3/(m-3)))
    t = np.delete(t, 0)
    der1 = np.zeros((m,3))
    mat = np.array([[1/2, 1/2, 0], [-1, 1, 0], [1/2, -1, 1/2]])

    j1 = int(np.ceil(m/2))
    der1 = der1.at[0,:].set(np.dot(mat[0], wrk1[:,0:3].T))
    for i in range(1,j1):
        x = (u[i] - t[i+2])*(m-3)
        X = np.array([1, x, x*x])
        B1 = np.dot(X, mat) 
        der1 = der1.at[i,:].set(np.dot(B1, wrk1[:,i-1:i+2].T))

    for i in range(j1, m-1):
        x = (u[i] - t[i+1])*(m-3)
        X = np.array([1, x, x*x])
        B2 = np.dot(X, mat) 
        der1 = der1.at[i,:].set(np.dot(B2, wrk1[:,i-2:i+1].T))
    der1 = der1.at[m-1,:].set(np.dot(np.array([0, 1/2, 1/2]), wrk1[:,m-4:m-1].T))

    return der1, wrk1

def der2_splev(bc, wrk1):    # 二阶导数 
    t, u, k = bc 
    m = len(u)  
    wrk2 = np.zeros((3, m-2))
    wrk2 = wrk2.at[:, :].set(2*(wrk1[:, 1:]-wrk1[:, :-1])/(2/(m-3)))
    t = np.delete(t, 0)
    t = np.delete(t, 0)
    der2 = np.zeros((m,3))
    mat = np.array([[1, 0], [-1, 1]])

    j1 = int(np.ceil(m/2))
    der2 = der2.at[0,:].set(np.dot(mat[0], wrk2[:,0:2].T))
    for i in range(1,j1):
        x = (u[i] - t[i+2])*(m-3)
        X = np.array([1, x])
        B1 = np.dot(X, mat) 
        der2 = der2.at[i,:].set(np.dot(B1, wrk2[:,i-1:i+1].T))

    for i in range(j1, m-1):
        x = (u[i] - t[i+1])*(m-3)
        X = np.array([1, x])
        B2 = np.dot(X, mat) 
        der2 = der2.at[i,:].set(np.dot(B2, wrk2[:,i-2:i].T))
    der2 = der2.at[m-1,:].set(np.dot(np.array([0, 1]), wrk2[:,m-4:m-2].T))

    return der2
'''

