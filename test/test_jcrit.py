
import json
import jax.numpy as np
import plotly.graph_objects as go
import sys 
sys.path.append('/home/nxy/codes/coil_spline_HTS/HTS')
import material_jcrit

pi = np.pi
with open('initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)


def Jcrit_strain(T, B, strain):
    if T > 60:
        jc,b,t = material_jcrit.get_critical_current(T, B, strain, "REBCO_HT")
    elif T <= 60:
        jc,b,t = material_jcrit.get_critical_current(T, B, strain, "REBCO_LT")
    
    return jc


def Jcrit_theta(Bx, By, B0, jc0):
    k2 = 0.2**2
    beta = 0.65
    jc = jc0 / (1 + (k2*Bx**2 + By**2)**0.5 / B0) ** beta
    return jc


def Jcrit(B,T,strain,theta):
    k2 = 0.15**2
    beta = 0.65
    jcs = Jcrit_strain(T, B, strain)
    jct = jcs / (1 + (k2*(B * np.sin(theta))**2 + (B * np.cos(theta))**2)**0.5 / B) ** beta
    return jct



jt = np.zeros((3, 100))
jc = np.zeros((3, 100))
T = 77
strain = 0

for i in range(3):
    B = 0.1 + 0.5 * i
    jc0 = Jcrit(T, B, strain, pi/2)
    jt0 = Jcrit_theta(B, 0, 0.02, 3e10)
    for j in range(100):
        theta = pi/100*j
        Bx, By = B * np.sin(theta), B * np.cos(theta)
        jc = jc.at[i,j].set(Jcrit(T, B, strain, theta)/jc0)    
        jt = jt.at[i,j].set(Jcrit_theta(Bx, By, 0.02, 3e10)/jt0)   
        # jtx0 = Jcrit_theta(Bx, By, B0, jc)
        # jtx = jtx.at[i, j].set(jtx0)        
        # Bx, By = 0, (j+1)/100
        # jty0 = Jcrit_theta(Bx, By, B0, jc)
        # jty = jty.at[i, j].set(jty0)


color = ['red', 'blue', 'green', 'yellow']
fig = go.Figure()
for i in range(3):
    fig.add_scatter(x = np.arange(0, pi, pi/100), y = jc[i], 
                        name = 'jcrit_{}T'.format(0.5*i+0.1), line = dict(width=5,color=color[i]))
    fig.add_scatter(x = np.arange(0, pi, pi/100), y = jt[i], 
                        name = 'jcrit_{}T_theta'.format(0.5*i+0.1), line = dict(width=5, dash='dashdot',color=color[i]))
    # fig.add_scatter(x = np.arange(0.01, 1.01, 1/100), y = jtx[i+4], 
    #                     name = 'jcrit_theta_X_{}K'.format((i+5)*4), line = dict(width=5,color=color[i]))
    # fig.add_scatter(x = np.arange(0.01, 1.01, 1/100), y = jty[i+4], 
    #                     name = 'jcrit_theta_y_{}K'.format((i+5)*4), line = dict(dash='dashdot',width=5,color=color[i]))
fig.update_xaxes(title_text = "theta", title_font = {"size": 25},title_standoff = 12, 
                    tickfont = dict(size=25))
fig.update_yaxes(title_text = "jcrit",title_font = {"size": 25},title_standoff = 12, 
                    tickfont = dict(size=25), exponentformat = 'e' )#,type="log")
fig.update_layout(legend=dict(x=0.7, y=1, font = dict(size=24)))
fig.show()








