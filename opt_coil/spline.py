## B-spline representation
## Only second-order continuous

import jax.numpy as np
import numpy 
from jax import config
import scipy.interpolate as si
config.update("jax_enable_x64", True)    


def get_c_init(coil, nic, ns, ncp):       

    c, bc  = calculate_splineseries(coil, nic, ns, ncp, 3)  
    tj = tjev(bc)    
    return c, bc, tj



def get_bc_init(ns, ncp):
    k = 3
    t = np.linspace(-3/(ncp-3), (ncp)/(ncp-3), ncp+4) 
    t=t.at[3].set(int(0))     
    u = np.linspace(0, (ns-1)/ns ,ns)
    bc = [t, u, k]
    tj = tjev(bc)
    return bc, tj

def tjev(bc):
    '''Obtain the knot sequence.'''
    t, u, _ = bc
    if len(u)==len(t)-7:
        tj=np.arange(3,len(u)+2,1)

    else:    
        t0 = numpy.zeros_like(t)    
        u0 = numpy.zeros_like(u)
        tj = numpy.zeros_like(u)  
        j = 0
     
        for i in range(len(t)):
            t0[i] = t[i]
        for i in range(len(u)):  
            u0[i] = u[i]
            while u0[i]>=t0[j+1] :
                j = j+1
            tj[i] = j
    return tj

def calculate_splineseries(rc, nic, ns, ncp, k):  

    rc = numpy.array(np.transpose(rc, (0, 2, 1))) 
    if rc.shape[2] != ncp-2: 
        rc_new = np.zeros((nic, 3, ncp-2))  
        u = np.linspace(0, (ncp-4)/(ncp-3) ,ncp-3)
        for i in range(nic):
            tck, _ = si.splprep(x=rc[i], k=3, per=1, s=0)  
            rc_new = rc_new.at[i, :, :-1].set(si.splev(u, tck))
            rc_new = rc_new.at[i, :, -1].set(rc_new[i, :, 0])
        rc = numpy.array(rc_new)
        
    c = np.zeros((nic, 3, ncp))   
    t = np.linspace(-3/(ncp-3), (ncp)/(ncp-3), ncp+4) 
    t = t.at[3].set(float(0))
    t = t.at[-4].set(float(1))
    u = np.linspace(0, (ns-1)/ns ,ns)
    for i in range(nic):
        tck, _ = si.splprep(x=rc[i], k=3, per=1, s=0)  
        c = c.at[i,:].set(tck[1])
    
    bc = [t, u, k]
    return c, bc


def splev(t, u, c, tj, ns):   
    ''' spline calculate'''
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

def der1_splev(t, u, c, tj, ns):       
    ncp = len(c[1])
    wrk1 = np.zeros((3, ncp-1))    
    c = np.array(c)
    wrk1 = wrk1.at[:, :].set((c[:, 1:]-c[:, :-1])/(1/(ncp-3)))
    t = np.delete(t, 0)    
    der1 = np.zeros((ns,3))
    mat = np.array([[1/2, 1/2, 0], [-1, 1, 0], [1/2, -1, 1/2]])
    for i in range(ns):
        j = int(tj[i])-1    
        x1 = (u[i] - t[j])/(t[j]-t[j-1])
        X1 = np.array([1, x1, x1*x1])
        B1 = np.dot(X1, mat) 
        der1 = der1.at[i,:].set(np.dot(B1, wrk1[:,j-2:j+1].T))
    return der1, wrk1

def der2_splev(t, u, wrk1, tj, ns):      
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


