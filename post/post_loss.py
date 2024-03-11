import jax.numpy as np
import plotly.graph_objects as go
import scipy.interpolate as si 
import json
import h5py
import main
pi = np.pi

with open('/home/nxy/codes/coil_spline_HTS/initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
# globals().update(args)



def read_hdf5(filename):
    f = h5py.File(filename, "r")
    print(f.keys())
    all = {}
    for key in f.keys():
        all.update({key: f[key][()]})
    f.close()
    return all

all = read_hdf5('/home/nxy/codes/coil_spline_HTS/results_f/jpark/j10_hdf5.h5')

loss = ['loss_B_max_coil', 'loss_B_max_surf', 'loss_Bn_max', 'loss_Bn_mean', 'loss_HTS_Icrit', 'loss_curva_max', 'loss_curvature', 'loss_dcc_min', 'loss_dcs_min', 'loss_length', 'loss_strain_max', 'loss_strain_mean', 'loss_tor_max', 'loss_tor_mean']


for i in loss:
    print(all['{}'.format(i)])

def plot_loss(lossvals):
    """
    画迭代曲线
    Args:
        lossvals : list,[ni], 迭代数据
        
    """
    fig = go.Figure()
    fig.add_scatter(x = np.arange(0, len(lossvals), 1), y = lossvals, 
                        name = 'lossvalue', line = dict(width=5))
    fig.update_xaxes(title_text = "iteration",title_font = {"size": 25},title_standoff = 12, 
                        tickfont = dict(size=25))
    fig.update_yaxes(title_text = "lossvalue",title_font = {"size": 25},title_standoff = 12, 
                        tickfont = dict(size=25) ,type="log", exponentformat = 'e')
    fig.show()
    return

loss = np.load('/home/nxy/codes/coil_spline_HTS/results_f/jpark/j10_loss.npy')
plot_loss(loss)

for key in args:
    args['{}'.format(key)] = all['{}'.format(key)]

args['plot_coil'], args['plot_loss'], args['plot_poincare'] = 2, 0, 1
args['save_npy'], args['save_hdf5'], args['save_makegrid'] = 0, 0, 0


with open('/home/nxy/codes/coil_spline_HTS/initfiles/init_args.json', 'w') as f:
    json.dump(args, f, indent=4)

main.main()





