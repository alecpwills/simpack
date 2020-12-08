import numpy as np
from os import path
exp_dir = path.dirname(__file__)
amu_to_g = 1.66054e-24
ang_to_cm = 1e-8
R = 0.46151805#kJ/kgK AS DEFINED IN SOURCE; DO NOT CHANGE OR CONVERSIONS WILL BREAK

def Soper2013():
    '''
    Retrieve Soper 2013 (DOI:10.1155/2013/279463) distribution functions for water.

    Returns
    -------
    arr : np.ndarray
        Columns are r_A, O_O, OOError, O_H, OHError, H_H, HHError.
    '''
    arr = np.loadtxt(path.join(exp_dir, 'soper2013_dist_funcs'), comments=['#','@'])
    return arr

def gen_dens(box_length, num_water, nacl=True):
    tot_mass = num_water*18.01528
    if nacl == True:
        tot_mass += 35.453+22.989769
    vol = box_length**3
    
    rho = tot_mass/vol
    
    rho = rho*amu_to_g/ang_to_cm**3
    print(rho, 'g/cm3')
    print(rho/1000*(100**3), 'kg/m^3')
    return rho

def logKam(T, rho_w, method):
    '''
    NaCl solution association constant using Eq. 27 from 10.1021/je300361j.

    Parameters
    ----------
    T : float
        The temperature to use in the calculation.
    rho_w : float
        The density of water in kg/m^3.
    method : int, 1 through 4 inclusive.
        The exact form of the fit equation to use in the calculation.

    Returns
    -------
    lK : float
        The log10 of the association constant for NaCl in solution at given parameters.

    '''
    c = 0
    d = 0
    g = 0
    if method==1:
        a=21.09
        b=0.825e3
        e=-7.52
        f=0
    elif method==2:
        a=-1.046
        b=15.96e3
        e=0
        f=-5.159e3
    elif method==3:
        a=0
        b=15.25e3
        e=-0.36
        f=-4.915e3
    elif method==4:
        a=22.29
        b=0
        e=-7.926
        f=0.281e3
    lK = a+b/T+c/T**2+d/T**3+(e+f/T+g/T**2)*np.log10(rho_w)
    return lK