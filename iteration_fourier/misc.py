import jax.numpy as np
import fourier
import json
import plotly.graph_objects as go
import numpy

with open('/home/nxy/codes/focusadd-spline/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)

t = np.linspace(-3/(13-3), (13)/(13-3), 13+4) 
t=t.at[3].set(0)
u = np.linspace(0, (16-1)/16 ,16)
print(t)
t0 = numpy.zeros_like(t)
u0 = numpy.zeros_like(u)
tj = numpy.zeros_like(u)  

j = 0
for i in range(len(t)):
    t0[i] = t[i]
for i in range(len(u)):  # len(u[a]) = ns
    u0[i] = u[i]
    while u0[i]>=t0[j+1] :
        j = j+1
    tj[i] = j
print(tj)






