"""
Nathan Smith
18/10/2021

Python implementation of material functions as Fortran was refusing to compile the original function.
Caller functions/ interfaces are located at the BOTTOM of this file.

General structure when adding materials or using this module:

```python
def Material(parameters):
    #####################
    ### Bunch of code ###
    #####################

    return Answers
materials["material_key"] = Material
```

List of materials:
- NbTi:
    - Key: "NbTi"
    - general superconductor, using an alloy of Niobium and Titanium

- REBCO:
    - Key: "REBCO"
    - Superconductor made using Rare Earth metals and BCO. A very commonly
    used High Temperature superconductor. Comes in the form of tapes on a copper
    substrate.

- Nb3Sn:
    - Key: "Nb3Sn"
    - Another general use superconductor, Using an alloy of Niobium and Tin.
"""

import numpy as np

# Instantiate materials with dummy data
materials = dict()

######## Material Function definitions and parameter inputs ########
def NbTi(temperature, Bmax, strain):
    """
    Function that calculates the critical values of the superconducting material "NbTi".
    Originally authored by S B L Chislett-McDonald @ Durham University
    """
    from parameters import NbTi_parameters as p

    e_I = strain - p["em"]
    strain_func = 1 + p["c2"] * e_I ** 2 + p["c3"] * e_I ** 3 + p["c4"] * e_I ** 4
    T_e = p["Tc0"] * strain_func ** (1 / p["w"])
    t = temperature / T_e
    A_e = p["A_0"] * strain_func ** (p["u"] / p["w"])
    B_crit = p["Bc20max"] * (1 - t ** p["v"]) * strain_func
    b = Bmax / B_crit
    T_crit = T_e
    if b <= 1:
        J_crit = (
            A_e
            * (T_e * (1 - t ** 2)) ** 2
            * B_crit ** (p["n"] - 3)
            * b ** (p["p"] - 1)
            * (1 - b) ** p["q"]
        )
    else:
        J_crit = (
            A_e
            * (T_e * (1 - t ** 2)) ** 2
            * B_crit ** (p["n"] - 3)
            * b ** (p["p"] - 1)
            * (1 - b)
        )
    return J_crit, B_crit, T_crit


materials["NbTi"] = NbTi


def REBCO_Other(temperature, Bmax, strain):
    """
    Function that calculates the critical values of the superconducting material "REBCO".
    Originally authored by S B L Chislett-McDonald @ Durham University
    """
    from parameters import REBCO_Other_parameters as p

    e_I = strain - p["em"]
    strain_func = 1 + p["c2"] * e_I ** 2 + p["c3"] * e_I ** 3 + p["c4"] * e_I ** 4
    T_e = p["Tc0"] * strain_func ** (1 / p["w"])
    t = temperature / T_e
    A_e = p["A_0"] * strain_func ** (p["u"] / p["w"])
    B_crit = p["Bc20max"] * (1 - t) ** p["s"] * strain_func
    b = Bmax / B_crit
    T_crit = T_e
    x = A_e * pow((T_e * (1 - t ** 2)), 2)
    y = pow(B_crit, (p["n"] - 3))
    z = pow(b, (p["p"] - 1)) * pow(1 - b, p["q"])
    J_crit = x * y * z
    return J_crit, B_crit, T_crit


materials["REBCO_Other"] = REBCO_Other
materials["rebco_other"] = REBCO_Other


def REBCO(temperature, Bmax, strain):
    """
    Function that calculates the critical values of the superconducting material "REBCO".
    Originally authored by S B L Chislett-McDonald @ Durham University
    """
    from parameters import REBCO_parameters as p

    e_I = strain - p["em"]
    strain_func = 1 + p["c2"] * e_I ** 2 + p["c3"] * e_I ** 3 + p["c4"] * e_I ** 4
    T_e = p["Tc0"] * strain_func ** (1 / p["w"])
    t = temperature / T_e
    A_e = p["A_0"] * strain_func ** (p["u"] / p["w"])
    B_crit = p["Bc20max"] * (1 - t) ** p["s"] * strain_func
    b = Bmax / B_crit
    T_crit = T_e
    x = A_e * pow((T_e * (1 - t ** 2)), 2)
    y = pow(B_crit, (p["n"] - 3))
    z = pow(b, (p["p"] - 1)) * pow(1 - b, p["q"])
    J_crit = x * y * z
    return J_crit, B_crit, T_crit


materials["REBCO"] = REBCO
materials["rebco"] = REBCO


def HIJC_REBCO(temperature, Bmax, strain):
    """
    uh oh spaghettio
    """
    from parameters import HIJC_REBCO_parameters as p
    B_crit = p["Bc20max"] * (1 - temperature / p["Tc0"]) ** p["a"]
    T_crit = 0.999965 * p["Tc0"]
    A_t = p["A_0"] + p["u"] * temperature ** 2 + p["v"] * temperature
    jcrit = (
        (A_t / Bmax)
        * (B_crit ** p["b"])
        * (Bmax / B_crit) ** p["p"]
        * (1 - Bmax / B_crit) ** p["q"])
    J_crit = (
        jcrit
        * (p["hts_tape_width"] * p["hts_tape_thickness"] * 0.4)
        / (0.004 * 0.000065)   )
    return J_crit, B_crit, T_crit


materials["HIJC_REBCO"] = HIJC_REBCO
materials["hijc_rebco"] = HIJC_REBCO


def Nb3Sn(temperature, Bmax, strain):
    from parameters import Nb3Sn_parameters as p
    epssh = (p["c2"] * p["e0a"]) / (np.sqrt(p["c1"] ** 2 - p["c2"] ** 2))
    S_e = np.sqrt(epssh ** 2 + p["e0a"] ** 2) - np.sqrt(
        (strain - epssh) ** 2 + p["e0a"] ** 2
    )
    S_e = S_e * p["c1"] - p["c2"] * strain
    S_e = 1 + (1 / (1 - p["c1"] * p["e0a"])) * S_e
    T_ce = p["Tc0"] * S_e ** (1 / 3)
    t = temperature / T_ce
    A_e = p["C"] * S_e
    # if p["Tc0"] < 20:
    B_c2Te = p["Bc20max"] * S_e * (1 - t ** 1.52)
    b = Bmax / B_c2Te
    Jc = A_e / B_c2Te
    Jc = Jc * (1 - t ** 1.52) * (1 - t ** 2)
    Jc = Jc * b ** (p["p"] - 1) * (1 - b) ** p["q"]
    return Jc * 1e6, B_c2Te, T_ce


materials["Nb3Sn"] = Nb3Sn

######## Interfacing functions ########

# Caller function to get critical current
def get_critical_current(temperature, Bmax, strain, material):
    """
    ## get_critical_current
    Interface function such that all material functions defined above can be accessed through this. Uses
    dictionary index system via the keys defined in the docstring at the top of the file.
    接口函数，使上面定义的所有材料函数都可以通过它访问。通过文件顶部文档字符串中定义的键使用字典索引系统。
    ------------
    ### Args:
        - temperature: float, the temperature of the helium bath used for coolant
        - Bmax: float, the maximum magnetic field reached at the coil
        - strain: float, the mechanical strain the coil is under
        - Bc20max: float, the upper critical field of the superconductor at 0K
        - Tc0: float, the critical temperature of the superconductor at 0 strain.
        - material: string, the key of material being used. check this is in the given list of materials.
    ------------
    Returns:
        - jcrit, bcrit, tcrit: triplet of floats, defining the critical values of current, magnetic field and temperature
    respectively under the given conditions.
    """

    material_func = materials[material]  # Assigns the function to a callable    将函数分配给可调用对象
    jcrit, bcrit, tcrit = material_func(
        temperature, Bmax, strain
    )  # gives the callable the data   为可调用对象提供数据

    # rescales current to be in MA/m²
    return jcrit, bcrit, tcrit
