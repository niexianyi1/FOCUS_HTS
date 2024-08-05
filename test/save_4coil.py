
import jax.numpy as np
import h5py
import sys
sys.path.append('iteration')
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




def save_4coil(filename):
    arge = read_hdf5(filename)
        
    if arge['coil_case'] == 'spline':
        arge['coil_arg'] = arge['coil_arg'][:, :, :-3]
        bc, tj = spline.get_bc_init(arge['number_points'], arge['number_control_points'])
        arge['bc'] = bc
        arge['tj'] = tj


    coil_cal = post_coilset.CoilSet(arge)  
    params = (arge['coil_arg'], arge['coil_fr'], arge['coil_I'])
    coil = coil_cal.get_coil(params)

    ns = arge['number_segments']
    nn = arge['number_normal']
    nb = arge['number_binormal']
    nic = arge['number_independent_coils']

    rr = np.zeros((nic, 5, ns+1, 3))
    rr = rr.at[:,0,:ns,:].set(coil[:nic, 0, 0, :, :])
    rr = rr.at[:,1,:ns,:].set(coil[:nic, 0, nb-1, :, :])
    rr = rr.at[:,2,:ns,:].set(coil[:nic, nn-1, nb-1, :, :])
    rr = rr.at[:,3,:ns,:].set(coil[:nic, nn-1, 0, :, :])
    rr = rr.at[:,4,:ns,:].set(coil[:nic, 0, 0, :, :])
    rr = rr.at[:,0,-1,:].set(coil[:nic, 0, 0, 0, :])
    rr = rr.at[:,1,-1,:].set(coil[:nic, 0, nb-1, 0, :])
    rr = rr.at[:,2,-1,:].set(coil[:nic, nn-1, nb-1, 0, :])
    rr = rr.at[:,3,-1,:].set(coil[:nic, nn-1, 0, 0, :])
    rr = rr.at[:,4,-1,:].set(coil[:nic, 0, 0, 0, :])

    with open('results/others/strain_0.4_0.45_.txt', "w") as f:
        for i in range(4):
            for s in range(400):
                f.write(
                    "{} {} {} \n".format(
                        rr[0,i, s, 0],
                        rr[0,i, s, 1],
                        rr[0,i, s, 2],
                    )
                )
            f.write(
                "{} {} {}        {} \n".format(
                    rr[0,i, 0, 0],
                    rr[0,i, 0, 1],
                    rr[0,i, 0, 2],
                    "line {}".format(i+1),
                )
            )
    return

filename = 'results/others/strain_0.4_0.45.h5'
save_4coil(filename)


