import jax.numpy as np
import fourier
import json
import plotly.graph_objects as go
import numpy
import save
import read_init

with open('/home/nxy/codes/focusadd-spline/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)

for key in args:
    print(key)



