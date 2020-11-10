#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:26:50 2020

@author: awills
"""

from MDAnalysis.transformations.translate import center_in_box
import numpy as np
from collections import Counter
import tess
from tqdm import tqdm
import sys
#%%    
def vorent(container):
    faces = np.array([i.number_of_faces() for i in container])
    fcount = Counter(faces)
    ftotal = sum([fcount[k] for k in fcount.keys()])
    vent = -sum([(fcount[k]/ftotal)*np.log(fcount[k]/ftotal) for k in fcount.keys()])
    return vent

def vorvol(universe, centat, stride=1, nears=None, ent=None, skip=0):
    #nears expects list of 4 objects: selection string, distances by which to select, type 1 and type 2
    cumv = []
    if nears:
        ng1 = [universe.select_atoms(nears[0].format(i, nears[2], nears[2], nears[3])) for i in nears[1]]
        ng2 = [universe.select_atoms(nears[0].format(i, nears[3], nears[2], nears[3])) for i in nears[1]]
        nv1 = {str(i):[] for i in nears[1]}
        nv2 = {str(i):[] for i in nears[1]}
    if ent:
        ved = []
    for its in tqdm(range(len(universe.trajectory[skip::stride])), file=sys.stdout):
        cag = universe.residues[centat].atoms
        # uag = u.atoms
        ts = universe.trajectory[its]
        ts = center_in_box(cag)(ts)
        pos = ts.positions
        vc = tess.Container(pos, limits = universe.dimensions[:3], periodic=True)
        if ent:
            ive = vorent(vc)
            ved.append(ive)
        ivol = np.array([i.volume() for i in vc])
        cumv.append(ivol)
        if nears:
            for inear in range(len(ng1)):
                ids1 = ng1[inear].ids
                ids2 = ng2[inear].ids
                if len(ids1) != 0:
                    ids1 -= 1
                    nw = len(ids1) // 3
                    nwv = ivol[ids1.astype(np.int)].sum()
                    nv1[str(nears[1][inear])].append((nw, nwv))
                else:
                    nw = 0
                    nv1[str(nears[1][inear])].append((nw, 0))
                if len(ids2) != 0 :
                    ids2 -= 1
                    nw = len(ids2) // 3
                    nwv = ivol[ids2.astype(np.int)].sum()
                    nv2[str(nears[1][inear])].append((nw, nwv))
                else:
                    nv2[str(nears[1][inear])].append((0,0))
    cumv = np.array(cumv)
    ret = [cumv]
    if ent:
        ved = np.array(ved)
        ret.append(ved)
    if nears:
        ret.append(nv1)
        ret.append(nv2)
        
    return ret
#%%
# vv = vorvol(u, 0, stride=20, nears=[nearsel, neards, 1, 2], ent=True)
#%%
# tst = '/home/awills/Documents/Research/lammps/colvar/gallicomp/rd/12345/'
# ddirs = sorted([i for i in os.listdir(tst) if os.path.isdir(os.path.join(tst,i))])
# ds = np.array([float(i)/10 for i in ddirs])
# top = os.path.join(tst, ddirs[5], 'system.data')
# dcd = os.path.join(tst, ddirs[5], 'out.dcd')
# u = MD.Universe(top, dcd, atom_style='id resid type charge x y z')
# stride = 1
# #%%
# cumv = []
# near=True
# neards = np.arange(2, 6.1, step=0.2)
# nearsel = '( byres around {} type {} ) and not ( type {} or type {} )'
# centat = 0
# if near==True:
#     ng = [u.select_atoms(nearsel.format(i, 1, 1, 2), updating=True) for i in neards]
#     nvor = {str(i):[] for i in neards}
# #%%
# stride=200
# for its in tqdm(range(len(u.trajectory[::stride])), file=sys.stdout):
#     cag = u.residues[centat].atoms
#     uag = u.atoms
#     ts = u.trajectory[its]
#     ts = center_in_box(cag)(ts)
#     pos = ts.positions
#     vc = tess.Container(pos, limits = u.dimensions[:3], periodic=True)
#     ivol = np.array([i.volume() for i in vc])
#     cumv.append(ivol)
#     if near == True:
#         for ing in range(len(ng)):
#             ids = ng[ing].ids
#             if len(ids) != 0:
#                 ids -= 1
#                 nw = len(ids) // 3
#                 nws = ivol[ids.astype(np.int)].sum()
#                 nvor[str(nears[ing])].append((nw, nws))
#             else:
#                 nw = 0
#                 nvor[str(nears[ing])].append((nw, 0))
# cumv = np.array(cumv)
