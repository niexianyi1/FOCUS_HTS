import json
import jax.numpy as np
import read_init
import jax
from jax import jit, vmap
import fourier
import spline
import self_B
import read_plasma
import plotly.graph_objects as go
import h5py
import read_init
import sys 
sys.path.append('/home/nxy/codes/coil_spline_HTS/HTS')
import material_jcrit

pi = np.pi

with open('/home/nxy/codes/coil_spline_HTS/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)


'-0.88654223  -3.41208993  -7.00782437 -10.35063557 -11.31088441'
'0.76324801   2.8674874    5.68364553   8.02877379 8.26336001'
'-1.84276454  -5.55545782  -6.89911251  -2.16180043 8.82919251'
'-2.00261856  -3.03892395  -2.05141979   1.85489267 8.74974454'
'1.62195057   2.34153865   1.31484959  -1.99879504 -7.39315831'
'-1.92200952  -0.29053128   5.63372348  12.66977925 15.12245189'

'-0.88655879  -3.41209959  -7.0078   -10.35072411 -11.31087354'
'0.76323398   2.86753347   5.68365911   8.02880704 8.26343215'
'-1.84277396  -5.55553472  -6.89907999  -2.16182639 8.82914809'
'-2.00262868  -3.03890148  -2.05141997   1.85488277 8.7497388'
'1.62195127   2.34153549   1.31488157  -1.99880642 -7.39318078'
'-1.92197382  -0.29054062   5.6337304   12.66987503 15.12239608'


a = [1,2,3]
print(np.max(a))

