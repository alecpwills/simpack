import numpy as np
from os import path
exp_dir = path.dirname(__file__)

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
