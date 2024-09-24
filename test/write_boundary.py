


import jax.numpy as np


NM = np.zeros((61,2))
RBC = np.zeros((61))
ZBS = np.zeros((61))
with open('initfiles/Landreman-Paul_QA/input.nowell', 'r') as f:
    for i in range(42):
        _ = f.readline()
    for i in range(61):
        _,n,m,_,rbc,_,_,_,_,zbs = f.readline().split()
        n = n.split(',')[0]
        m = m.split(')')[0]
        rbc = rbc.split(',')[0]
        NM = NM.at[i,0].set(int(n))
        NM = NM.at[i,1].set(int(m))
        RBC = RBC.at[i].set(float(rbc))
        ZBS = ZBS.at[i].set(float(zbs))
print(NM)


with open('initfiles/Landreman-Paul_QA/plasma_nowell.boundary','w') as f:
    f.write('# Nbmn Nfp  Nbnf\n')
    f.write('61   2   0\n')
    f.write('# plasma boundary parameters\n')
    f.write('# n  m  rbc  rbs  zbc  zbs\n')
    for i in range(61):
        f.write("{}  {}  {}  0  0  {}\n".format(int(NM[i,0]),int(NM[i,1]),RBC[i], ZBS[i]))
