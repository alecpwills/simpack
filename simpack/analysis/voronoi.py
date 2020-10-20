#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:26:50 2020

@author: awills
"""

import MDAnalysis as MD
from MDAnalysis.transformations.translate import center_in_box
import numpy as np
import tess
from tqdm import tqdm
import os, sys
#%%
def iVoronoiVolumes(self, near=False):
    lims = self.Universe.atoms.dimensions[:3]
    vollst = []
    if near == True:
        ranges = np.arange(2, 6.1, 0.5)
        vol_dict = {str(i):[] for i in ranges}
    for ts in self.Universe.trajectory:
        print(ts.time)
        ag = self.Universe.residues[0].atoms
        ts = center_in_box(ag)(ts)
        pos = ts.positions
        vor_cont = tess.Container(pos, limits=lims, periodic=True)
        vollst.append(np.array([i.volume() for i in vor_cont]))
        if near == True:
            cinds = []
            for i in ranges:
                rinds = np.where(np.linalg.norm(pos - self.Universe.dimensions[:3]/2, axis=1) < i)[0]
                vol_dict[str(i)].append(np.array([i for i in rinds if i not in cinds]))
                cinds = np.concatenate([cinds, rinds])
    if near == False:
        self.VoronoiVolumes = vollst
    else:
        self.VoronoiVolumes = (vollst, vol_dict)

def iVoronoiEntropy(self, npts=1000):
    lims = self.Universe.atoms.dimensions[:3]
    ents = []
    for ts in self.Universe.trajectory:
        prevind = 0
        print(ts.time)
        pos = ts.positions
        vor_cont = tess.Container(pos, limits=lims, periodic=True)
        faces = np.array([i.number_of_faces() for i in vor_cont])
        fcount = Counter(faces)
        ftotal = sum([fcount[k] for k in fcount.keys()])
        vor_ent = -sum([(fcount[k]/ftotal)*np.log(fcount[k]/ftotal) for k in fcount.keys()])
        ents.append(vor_ent)
    self.VoronoiEntropy =  [ents, np.std(ents)]
#%%
tst = '/home/awills/Documents/Research/lammps/colvar/gallicomp/rd/12345/'
ddirs = sorted([i for i in os.listdir(tst) if os.path.isdir(os.path.join(tst,i))])
ds = np.array([float(i)/10 for i in ddirs])
top = os.path.join(tst, ddirs[5], 'system.data')
dcd = os.path.join(tst, ddirs[5], 'out.dcd')
u = MD.Universe(top, dcd, atom_style='id resid type charge x y z')
stride = 1
#%%
vollst = []
near=True
nears = np.arange(2, 6.1, step=0.2)
nearsel = '( byres around {} type 1 ) and not ( type 2 or type 1 )'
centat = 0
if near==True:
    ng = [u.select_atoms(nearsel.format(i), updating=True) for i in nears]
    nvor = {str(i):[] for i in nears}
#%%
stride=1
for its in tqdm(range(len(u.trajectory[::stride])), file=sys.stdout):
    cag = u.residues[centat].atoms
    uag = u.atoms
    ts = u.trajectory[its]
    ts = center_in_box(cag)(ts)
    pos = ts.positions
    vc = tess.Container(pos, limits = u.dimensions[:3], periodic=True)
    vollst.append(np.array([i.volume() for i in vc]))
    if near == True:
        for ing in range(len(ng)):
            ids = ng[ing].ids
            if len(ids) != 0:
                ids -= 1
                nw = len(ids) // 3
                nws = vollst[0][ids.astype(np.int)].sum()
                
                nvor[str(nears[ing])].append((nw, vollst[0][ids.astype(np.int)].sum()))
            else:
                nw = 0
                nvor[str(nears[ing])].append((nw, 0))
vollst = np.array(vollst)
