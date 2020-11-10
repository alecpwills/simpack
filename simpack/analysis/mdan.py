#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:26:50 2020

@author: awills
"""

import numpy as np
from MDAnalysis.analysis.distances import distance_array

def set_water_residues(universe, osel='name O', hsel='name H'):
    #assumes any other atoms have been set to residues already, and given dimensions
    ou = universe.select_atoms(osel)
    hu = universe.select_atoms(hsel)
    startres = ou[0].resid
    startres += 1
    darr = distance_array(ou.positions, hu.positions, box=universe.dimensions)
    for i in range(len(ou)):
        newres = universe.add_Residue(universe.segments[0], resnum = startres, resid = startres, resname='WAT{}'.format(i))
        sA, sB, *_ = np.partition(darr[i], 1)
        iA = np.argmin(abs(darr[i]-sA))
        iB = np.argmin(abs(darr[i]-sB))
        OID = ou[i].id-1
        H1ID = hu[iA].id-1
        H2ID = hu[iB].id-1
        universe.atoms[[OID,H1ID,H2ID]].residues = newres
        startres += 1
