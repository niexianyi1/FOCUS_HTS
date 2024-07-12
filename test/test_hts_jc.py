import json
import jax.numpy as np
import numpy
from jax import jit, vmap
import plotly.graph_objects as go
import h5py
import sys 
sys.path.append('HTS')
import material_jcrit

pi = np.pi

with open('initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)


jc = []
x = np.linspace(-0.04, 0.04, 20)
print(x)
for i in range(20):
    j, _, _ = material_jcrit.get_critical_current(4.2, 10, x[i], 'REBCO_LT')
    jc.append(j)
print(np.array(jc))

fig = go.Figure()
fig.add_scatter(x = x, y = jc, 
                    name = 'jc', line = dict(width=5))
fig.update_xaxes(title_text = "strain",title_font = {"size": 25},title_standoff = 12, 
                    tickfont = dict(size=25))
fig.update_yaxes(title_text = "lossvalue",title_font = {"size": 25},title_standoff = 12, 
                    tickfont = dict(size=25) ,type="log", exponentformat = 'e')
fig.show()
