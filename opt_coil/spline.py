import jax.numpy as np
import numpy 
from jax import config
import scipy.interpolate as si
config.update("jax_enable_x64", True)    # float64
#### 三阶导数不可用
#### 添加ncp项，和ns做区分

## 在这里需要一个重合点，输入时自带
def get_c_init(coil, nic, ns, ncp):       
    """开始运行程序时调用,下接compute_splineseries和tjev

    Args:
        coil, 初始线圈坐标
        nic, 独立线圈个数
        ns, 线圈段数
        ncp, 控制点数

    Returns:
        c, B样条函数控制点,优化参数
        bc, B样条函数其他参数,非优化
        tj, 节点序列
    """

    # 引用函数计算
    c, bc  = compute_splineseries(coil, nic, ns, ncp, 3)    # c是控制点，作为优化变量，其他非优化参数都包含在bc里
    tj = tjev(bc)    # 获取节点序列，见文档
    return c, bc, tj


## 如果已有参数c,可从此处直接获得bc。
def get_bc_init(ns, ncp):
    """
    初始给定控制点时调用
        Args:
        ns, 线圈段数
        ncp, 控制点数

    Returns:
        bc, B样条函数其他参数,非优化
        tj, 节点序列
    """
    k = 3
    t = np.linspace(-3/(ncp-3), (ncp)/(ncp-3), ncp+4) 

    t=t.at[3].set(int(0))     # 此处是为了防止1e-17这种情况，手动设为0。
    # 这个.at[].set()的格式是jax支持的格式，等价为numpy的 t[3]=0.

    u = np.linspace(0, (ns-1)/ns ,ns)
    bc = [t, u, k]
    tj = tjev(bc)
    return bc, tj

def tjev(bc):
    """
    获取节点序列,在后续计算中无需重复。
    
    Args:
        bc, 从bc中获得t,u,进行比较排序

    Returns: 
        tj, 节点序列   
    
    """
    t, u, _ = bc
    if len(u)==len(t)-7:
        tj=np.arange(3,len(u)+2,1)

    else:    
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

def compute_splineseries(rc, nic, ns, ncp, k):  
    """获取控制点c,
    Args:
        rc,     array,  [nic, ns, 3]  线圈坐标
        nic,    int,    独立线圈个数
        ns,     int,    线圈段数
        ncp,    int,    控制点数
        k,      int,    阶数
    Returns:
        c,      array,  [nic, 3, ncp]    B样条函数控制点,优化参数
        bc,     list,   [t,u,k]   B样条函数其他参数,非优化参数
    """
    rc = numpy.array(np.transpose(rc, (0, 2, 1))) # 格式转换，转为scipy支持的格式
    if rc.shape[2] != ncp-2:  # 为了让控制点数与输入点数匹配
        rc_new = np.zeros((nic, 3, ncp-2))  
        u = np.linspace(0, (ncp-4)/(ncp-3) ,ncp-3)
        for i in range(nic):
            tck, _ = si.splprep(x=rc[i], k=3, per=1, s=0)  # 调用scipy
            rc_new = rc_new.at[i, :, :-1].set(si.splev(u, tck))
            rc_new = rc_new.at[i, :, -1].set(rc_new[i, :, 0])
        rc = numpy.array(rc_new)
        
    c = np.zeros((nic, 3, ncp))   
    t = np.linspace(-3/(ncp-3), (ncp)/(ncp-3), ncp+4) 
    t = t.at[3].set(float(0))
    t = t.at[-4].set(float(1))
    u = np.linspace(0, (ns-1)/ns ,ns)
    for i in range(nic):
        tck, _ = si.splprep(x=rc[i], k=3, per=1, s=0)  # 调用scipy
        c = c.at[i,:].set(tck[1])
    
    bc = [t, u, k]
    return c, bc


def splev(t, u, c, tj, ns):   
    """ B样条函数计算
    通过vmap调用, 逐个线圈输入计算, 结束后整体输出
    Args:
        t,  array,  [ncp+4]  B样条函数节点
        u,  array,  [ns]    输入点, 
        c,  array,  [3,ncp] 控制点
        tj, array,  节点序列
        ns, int,    线圈段数
    Returns:
        xyz, array, [ns,3]   输出点
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
    """ 
    B样条函数一阶导数计算
    此处需要输出wrk1, 是二阶spline的控制点, 用于计算二阶导数
    """
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


