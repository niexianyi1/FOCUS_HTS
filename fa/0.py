import plotly.graph_objects as go
import numpy as np




nb = [0.11484888, 0.03336977, 0.00294727 ,0.01591175 ,0.0230806 , 0.02656134,0.02827644 ,0.02912775]

fig = go.Figure()
fig.add_scatter(x=np.linspace(5, 12, 8), y=nb, name='diff', mode='markers', marker_size = 10)   
fig.show()
