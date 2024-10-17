## Here we calculate the size of the cross-section when 'length_calculate'==1.

import jax.numpy as np
import material_jcrit
import hts_strain
import B_self
import sys
sys.path.append('opt_coil')
import coilset  


def solve_section(args, coil_arg_init, fr_init, I_init):
    params = (coil_arg_init, fr_init, I_init)
    param = np.array(args['length_normal'])
    newlen = function_section(args, params, param)
    dlen = abs(newlen - param)
    while (dlen > 2e-2).all():
        param = (newlen + param) / 2
        newlen = function_section(args, params, param)
        dlen = abs(newlen - param)
    length = np.array([newlen, param])
    args['length_normal'] = np.max(length, axis=0) * 1.2
    args['length_binormal'] = np.max(length, axis=0) * 1.2
    print(args['length_binormal'])
    return args


def function_section(args, params, param):
    args['length_normal'] = param
    args['length_binormal'] = param
    coil = coilset.CoilSet(args)
    B_self_input = coil.get_fb_input(params)
    new_len = calculate_section(args, B_self_input)
    return new_len


def calculate_section(args, B_self_input):

    coil, I, dl, v1, v2, binormal, curva, der2 = B_self_input
    I = I * args['I_normalize']
    curva = 3 * curva  
    strain = hts_strain.HTS_strain(args, curva, v1, v2, dl)
    strain_max = np.max(strain)
    B_coil = B_self.coil_self_B_rec(args, coil, I, dl, v1, v2, binormal, curva, der2) 
    B_self_max = np.max(np.linalg.norm(B_coil, axis=-1), axis = (1,2))
    jc = Jcrit(args, B_self_max, strain_max)
    n_total = (I / jc / args['HTS_I_percent'] / args['HTS_structural_percent'])
    new_length = np.sqrt(n_total / ((args['number_normal']-1) * (args['number_binormal']-1)))
    return new_length 


def Jcrit(args, B_self, strain):
    nic = args['number_independent_coils']
    # assert nic == len(B_self)
    jc = np.zeros((nic))
    Bc = np.zeros((nic))
    for i in range(nic):
        j,b,t = material_jcrit.get_critical_current(args['HTS_temperature'],
                        B_self[i], strain, args['material'])
        jc = jc.at[i].set(j)
        Bc = Bc.at[i].set(b)
    assert B_self.all() < np.min(Bc)
    return jc







