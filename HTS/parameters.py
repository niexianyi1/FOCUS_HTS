"""
# Parameters
------------
File containing multiple python dictionaries. These contain many of the parameters
used in the functions of the package. Feel free to change them, but note that if some are
deleted and not either replaced within the functions they are used in (highly unrecommended) or specified
at the function call (not recommended without defaults provided) that many high level functions will not work.
Beyond that, they can be varied within their specified limits without issue and redefined at function call
time as key word parameters.
============
Parameter lists included:

- Default global parameters:\
    Specifies various reactor and coil properties to use in high level functions alongside flags for
    different use cases of functions, e.g. whether to use stellarator symmetry or not.

- NbTi parameters:\
    Specifies the intrinsic material properties of the superconducting alloy NbTi used within stellarator
    coils. Source stated in its docstring.

- REBCO parameters:\
    Specifies the intrinsic material properties of rare earth metal high temperature superconducting tapes.
    Source stated within docstring.

- Nb3Sn parameters:\
    Specifies the material properties of the superconducting alloy Nb3Sn. Source within docstring.

"""



NbTi_parameters = {
    "A_0": 1102e6,
    "p": 0.49,
    "q": 0.56,
    "n": 1.83,
    "v": 1.42,
    "c2": -0.0025,
    "c3": -0.0003,
    "c4": -0.0001,
    "em": -0.00002,
    "u": 0,
    "w": 2.2,
    "Bc20max": 14.86,
    "Tc0": 9.04,
}
"""
### NbTi Parameters
------------
Parameters used in the calculation of the critical thermodynamic parameters of the superconducting material NbTi.
- A_0 : overall scaling parameter.
- p, q, n, v : exponent parameters used in the calculation of the critical current density.
- c2, c3, c4, em : parameters used to calculate the strain function of the superconductor.
- u, w : ratio of these used to calculate the exponent of the strain function.
- Bc20max, Tc0 : upper critical values of the superconductor.

These parameters have been taken from the PROCESS systems code and were determined experimentally by
S B L Chislett-McDonald @ Durham University.
"""


REBCO_LT_parameters = {
    "A_0": 62500,
    "p": 0.451,
    "q": 1.44,
    "n": 3.33,
    "s": 5.27,
    "c1":0.00224,
    "c2": -0.0198,
    "c3": 0.0039,
    "c4": 0.00103,
    "em": 0.058,
    "u": 0,
    "w": 2.2,
    "Bc20max": 139,
    "Tc0": 185,
    'gamma':1.422,
    'eta':0.047,
}


REBCO_HT_parameters = {
    "A_0": 6.55e6,
    "p": 0.581,
    "q": 2.86,
    "n": 2.66,
    "s": 1.26,
    'c1':0.00139,
    "c2": -0.0294,
    "c3": 0.0104,
    "c4": 0.0052,
    "em": 0.058,
    "u": 0,
    "w": 2.2,
    "Bc20max": 98.7,
    "Tc0": 90.1,
    'gamma':1.422,
    'eta':0.047,
}
"""
### REBCO Parameters
------------
Parameters used in the calculation of the critical thermodynamic parameters of High Temperature Superconducting (HTS) materials, which make use of a rare earth metal
alloyed with BCO, otherwise known as REBCO. This material is often used in the form of layered superconducting material on a copper tape. This form is assumed in this
analysis.
- A_0 : overall scaling parameter. factor of 1/0.31 is included to account for original parameters being strand parameterisation.
- p, q, n, s : exponent parameters used in the calculation of the critical current density.
- c2, c3, c4, em : parameters used to calculate the strain function of the superconductor.
- u, w : ratio of these used to calculate the exponent of the strain function.
- Bc20max, Tc0 : upper critical values of the superconductor.
These parameters have been taken from the PROCESS systems code and were determined experimentally by
????? S B L Chislett-McDonald @ Durham University. ?????
------------------
补充, 数据来源:<Weak emergence in the angular dependence of the critical current density of the high temperature
superconductor coated conductor REBCO>, Paul Branch et al 2020 Supercond. Sci. Technol. 33 104006
"""

HIJC_REBCO_parameters = {
    "a": 1.4,
    "b": 2.005,
    "A_0": 2.2e8,
    "p": 0.39,
    "q": 0.9,
    "u": 33450,
    "v": -176577,
    "Bc20max": 138,
    "Tc0": 92,
    "hts_tape_width": 0.004,
    "hts_tape_thickness": 1e-6,
}

Nb3Sn_parameters = {
    "C": 83075 / 0.62,
    "p": 0.593,
    "q": 2.156,
    "c1": 50.06,
    "c2": 0,
    "e0a": 0.00312,
    "Bc20max": 32.97,
    "Tc0": 16.06,
}
"""
### Nb3Sn Parameters
------------
Parameters used in the calculation of the critical thermodynamic parameters of the superconducting material Nb3Sn.
- C : overall scaling parameter. factor of 1/0.31 is included to account for original parameters being strand parameterisation.
- p, q : exponent parameters used in the calculation of the critical current density.
- c1, c2, c4, e0a : parameters used to calculate the strain function of the superconductor.
- Bc20max, Tc0 : upper critical values of the superconductor.

These parameters have been taken from the PROCESS systems code and were determined experimentally by
????? S B L Chislett-McDonald @ Durham University. ?????
"""
