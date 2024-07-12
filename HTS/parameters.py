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


default_global_parameters = {
    "starting_width": 0.1,
    "aspect_ratio": 1.22,
    "material": "NbTi",
    "T": 4.2,
    "f_cu": 0.75,
    "f_struct": 0.4,
    "f_he": 0.35,
    "f_safe": 0.8,
    "verbose": False,
    "parallelize": True,
    "use_unique_coils": True,
    "accuracy": (2, 1),
    "nearest_neighbours": 1,
    "stellarator_symmetry": False,
    "find_multiple_coil_sizes": False,
}
"""
## Default global parameters
------------
Parameter list for high level functions within the API. Can be freely changed although note that some
settings have dependencies on other settings. These are noted in the table below.
------------
- starting_width: float, postive real number.\
    Specifies the starting value of coil width which optimisation functions will begin their
    first iteration at. Only for functions using single widths for the coil set.

- aspect_ratio: float, positive real number typically ~1.\
    Specifies the ratio of coil width in the toroidal direction to coil width in the radial
    direction. The same for all coils.
    指定环向线圈宽度与径向线圈宽度的比值。所有线圈都是一样的。

- material: string, name of material to be used as defined in material_functions.py.\
    Specifies the material which the program shall treat the coil as. Different materials have different
    properties and differences in their property functions.

- T: float, positive real number.\
    Defines the temperature at which the coils are being held at. Typically 4.2 (K) for superconducting
    coils, the temperature of liquid helium.

- f_cu: float, in range [0,1].\
    Defines the fraction of the superconducting strands which is made up of copper for quench protection.

- f_struct: float, in range [0,1].\
    Defines the fraction of the overall coil which is made up of structural material and cladding.

- f_he: float, in range [0,1].\
    Defines the fraction of the coil winding pack which is space made available for liquid helium coolant flow.

- f_safe: float, in range [0,1].\
    Defines what fraction of the maximum allowable critical current density will be allowed to flow within the coil.
    Higher values are less safe.

- verbose: bool.\
    Whether or not the function will print its results to the console.
    函数是否将结果打印到控制台。

- parallelize: bool.\
    Whether the C++ based code will compute the magnetic field and other parameters using multi-threaded methods. Does not work
    for every function and may not significantly speed up calculation in others. For instace, the multithreaded field calculators
    operate by partitioning the field to be calculated in to smaller chunks for each thread to operate. In this case, if there is
    only one point or more threads than the number of points being calculated, very little speed up will be acheived.

- use_unique_coils: bool.\
    Whether or not to only use a subset of the full coil set for deciding which points to calculate in
    certain functions, e.g. only a module or half-module.

- accuracy: tuple of ints, both positive and non-zero.\
    Defines the amount of points per GCE which the program will calculate the field at. The first value is the
    number of points in the toroidal direction, while the second is that in the poloidal direction, setting up
    a grid of such points on the radially-inward side of each GCE.

- nearest_neighbours: integer, positive.\
    Defines the number of coils on each side of the coil being analysed that will be considered as full width in the hybrid
    calculation method.
    定义在混合计算方法中被视为全宽的线圈在被分析线圈的每边上的线圈数。
    
- stellarator_symmetry: bool.\
    Determines whether the coil set being analysed makes use of a mirror symmetric module, and thus can cut down on required
    computations in this case.

- find_multiple_coil_sizes: bool.\
    Used to specify whether to use the coil size solver for only a single coil size used on all coils in the set, or individual sizes for each coil type.
"""

VOCUS_parameters = {
    "verbose": True,
    "volume arguments": default_global_parameters,
    "volume weight": 1,
    "grad resolution": 0.01,
    "comm": None,
}

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

colours = {
    0: (0.0, 0.0, 1, 1),
    1: (0.0, 1, 0.0, 1),
    2: (1, 0.0, 0.0, 1),
    3: (0.7, 0.7, 0.0, 1),
    4: (0.7, 0.0, 0.7, 1),
    5: (0.0, 0.7, 0.7, 1),
}


def colour_enum(i):
    """
    Helper function to return a colour from the colours dictionary based on enumeration integers. Can add additional color vec4's at will.
    i must be an unsigned integer.
    """
    return colours[i % len(colours)]
