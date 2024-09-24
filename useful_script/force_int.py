
import jax.numpy as np
import h5py
import sys 
sys.path.append('opt_coil')
import spline
sys.path.append('post')
import post_coilset

def read_hdf5(filename):
    f = h5py.File(filename, "r")
    arge = {}
    for key in list(f.keys()):
        val = f[key][()]
        if key == 'num_fourier_coils':
            key = 'number_fourier_coils'
        if isinstance(val, bytes):
            val = str(val, encoding='utf-8')
        arge.update({key: val})
    f.close()
    return arge




arge = read_hdf5('results/w7x/w7x_t.h5')
# I = np.array([1.97e6, 1.93e6, 1.83e6, 1.48e6, 1.46e6])

# def stellarator_symmetry_I(arge, I):
#     """计算电流的仿星器对称"""
#     I_new = np.zeros(arge['number_independent_coils']*2)
#     I_new = I_new.at[:arge['number_independent_coils']].set(I)
#     for i in range(arge['number_independent_coils']):
#         # I_new = I_new.at[i+arge['number_independent_coils']].set(-I[i])
#         I_new = I_new.at[i+arge['number_independent_coils']].set(-I[i])
#     return I_new

# def symmetry_I(arge, I):
#     """计算电流的周期对称"""
#     npc = int(arge['number_coils'] / arge['number_field_periods'])
#     I_new = np.zeros(arge['number_coils'])
#     for i in range(arge['number_field_periods']):
#         I_new = I_new.at[npc*i:npc*(i+1)].set(I)
#     return I_new

# I = stellarator_symmetry_I(arge, I)
# I = symmetry_I(arge, I)
# arge['coil_I'] = I
# arge['length_normal'] = [0.144 for i in range(5)]
# arge['length_binormal'] = [0.192 for i in range(5)]

if arge['coil_case'] == 'spline':
    arge['coil_arg'] = arge['coil_arg'][:, :, :-3]
    bc, tj = spline.get_bc_init(arge['number_points'], arge['number_control_points'])
    arge['bc'] = bc
    arge['tj'] = tj
    dt = 1 / arge['number_segments']
elif arge['coil_case'] == 'fourier':
    dt = 2 * np.pi / arge['number_segments']

coil_cal = post_coilset.CoilSet(arge)  
params = (arge['coil_arg'], arge['coil_fr'], arge['coil_I'])
coil = coil_cal.get_coil(params)
force = coil_cal.get_plot_force(params)

force_coil = force * np.linalg.norm((arge['coil_der1'] * dt), axis=-1)[:,:,np.newaxis]

force_all = np.sum(force_coil, axis = 1)
print(force_all/1e6)
force_all = np.linalg.norm(force_all, axis=-1)
print(force_all/1e6)



