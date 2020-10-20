#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:26:50 2020

@author: awills
"""


from simpack.analysis.analysis import pbcwrap
import MDAnalysis as MD
from tqdm import tqdm
import numpy as np
import os, sys
#%%
def coulomb_pairwise(atomgroup, boxlen):
    k = 14.3996 # eV * Ang / |e|^2
    CE = 0
    for i in range(len(atomgroup)):
        for j in range(len(atomgroup)):
            if i >= j:
                continue
            elif i < j:
              rij = atomgroup[i].position - atomgroup[j].position
              rij = pbcwrap(rij, boxlen=boxlen)
              dij = np.linalg.norm(rij)
              CE += k*atomgroup[i].charge*atomgroup[j].charge/dij
    return CE

def coulomb_pairwise_separated(atomgroup1, atomgroup2, boxlen, verbose=False):
    #want to exclude ion+its shell energy
    k = 14.3996 # eV * Ang / |e|^2
    CE = 0
    for i in range(len(atomgroup1)):
        for j in range(len(atomgroup2)):
            #by using two groups, we exclude the ion-shell energy and instead
            #just have interactions between shells            
            #but exclude same atoms
            if atomgroup1[i] == atomgroup2[j]:
                if verbose:
                    print('Same atom between shells, skipping')
                continue
            rij = atomgroup1[i].position - atomgroup2[j].position
            rij = pbcwrap(rij, boxlen=boxlen)
            dij = np.linalg.norm(rij)
            CE += k*atomgroup1[i].charge*atomgroup2[j].charge/dij
    return CE

def find_atom_by_distance(atomgroup, atom1, distance, boxlen):
    rij = atomgroup.positions - atomgroup[atom1].position
    dij = np.linalg.norm(pbcwrap(rij, boxlen), axis=1)
    dij[atom1] = 1e9
    ind = np.argmin(abs(dij-distance))
    a2id = atomgroup[ind].id
    return a2id

def coulombtraj_pairshellwise(root, subdirs, type1=1, type2=2, shellsize=3.0, stride=1,
                     topf = 'system.data', trajf = 'out.dcd', atom_style='id resid type charge x y z',
                     dt=0.5, verbose=False):
    CES = []
    CCES = []
    for idir in subdirs:
        p = os.path.join(root, idir)
        top = os.path.join(p, topf)
        dcd = os.path.join(p, trajf)
        u = MD.Universe(top, dcd, atom_style=atom_style, dt=dt)
        sphsel = 'type {} or type {} or byres around {} ( type {} or type {} )'.format(type1, type2, shellsize, type1, type2)
        sphsel1 = '( type {} or byres around {} type {} ) and not ( type {} )'.format(type1, shellsize, type1, type2)
        sphsel2 = '( type {} or byres around {} type {} ) and not ( type {} )'.format(type2, shellsize, type2, type1)
        sphu = u.select_atoms(sphsel, updating=True)
        sphu1 = u.select_atoms(sphsel1, updating=True)
        sphu2 = u.select_atoms(sphsel2, updating=True)
        CE = 0
        NCE = 0
        CCE = 0
        NCCE1 = 0
        NCCE2 = 0
        for i in tqdm(range(len(u.trajectory[::stride])), file=sys.stdout):
            CE += coulomb_pairwise(sphu, u.dimensions[0])
            NCE += len(sphu)
            CCE += coulomb_pairwise_separated(sphu1, sphu2, u.dimensions[0])
            NCCE1 += len(sphu1)
            NCCE2 += len(sphu2)
            for idt in range(stride):
                try:
                    u.trajectory.next()
                except StopIteration:
                    if verbose:
                        print("End of trajectory.")
        CE /= len(u.trajectory[::stride])
        CCE /= len(u.trajectory[::stride])
        NCE /= len(u.trajectory[::stride])
        NCCE1 /= len(u.trajectory[::stride])
        NCCE2 /= len(u.trajectory[::stride])
        CES.append((CE, NCE))
        CCES.append((CCE,NCCE1,NCCE2))
    
    return (CES, CCES)
    
def coulombtraj_shellwise(root, subdirs, type1=1, type2=2, shellsize=3.0, stride=1,
                     topf = 'system.data', trajf = 'out.dcd', atom_style='id resid type charge x y z',
                     dt=0.5, verbose=False):

    CCES = []
    for idir in subdirs:
        p = os.path.join(root, idir)
        top = os.path.join(p, topf)
        dcd = os.path.join(p, trajf)
        u = MD.Universe(top, dcd, atom_style=atom_style, dt=dt)
        sphsel1 = '( type {} or byres around {} type {} ) and not ( type {} )'.format(type1, shellsize, type1, type2)
        sphsel2 = '( type {} or byres around {} type {} ) and not ( type {} )'.format(type2, shellsize, type2, type1)
        sphu1 = u.select_atoms(sphsel1, updating=True)
        sphu2 = u.select_atoms(sphsel2, updating=True)
        CCE = 0
        NCCE1 = 0
        NCCE2 = 0
        for i in tqdm(range(len(u.trajectory[::stride])), file=sys.stdout):
            CCE += coulomb_pairwise_separated(sphu1, sphu2, u.dimensions[0])
            NCCE1 += len(sphu1)
            NCCE2 += len(sphu2)
            for idt in range(stride):
                try:
                    u.trajectory.next()
                except StopIteration:
                    if verbose:
                        print("End of trajectory.")
        CCE /= len(u.trajectory[::stride])
        NCCE1 /= len(u.trajectory[::stride])
        NCCE2 /= len(u.trajectory[::stride])
        CCES.append((CCE,NCCE1,NCCE2))
    
    return CCES

def coulombtraj_pairwise(root, subdirs, type1=1, type2=2, shellsize=3.0, stride=1,
                     topf = 'system.data', trajf = 'out.dcd', atom_style='id resid type charge x y z',
                     dt=0.5, verbose=False):
    CES = []
    for idir in subdirs:
        p = os.path.join(root, idir)
        top = os.path.join(p, topf)
        dcd = os.path.join(p, trajf)
        u = MD.Universe(top, dcd, atom_style=atom_style, dt=dt)
        sphsel = 'type {} or type {} or byres around {} ( type {} or type {} )'.format(type1, type2, shellsize, type1, type2)
        sphu = u.select_atoms(sphsel, updating=True)
        CE = 0
        NCE = 0
        for i in tqdm(range(len(u.trajectory[::stride])), file=sys.stdout):
            CE += coulomb_pairwise(sphu, u.dimensions[0])
            NCE += len(sphu)
            for idt in range(stride):
                try:
                    u.trajectory.next()
                except StopIteration:
                    if verbose:
                        print("End of trajectory.")
        CE /= len(u.trajectory[::stride])
        NCE /= len(u.trajectory[::stride])
        CES.append((CE, NCE))
    
    return CES

def coulombtraj_wwdpairwise(otype, num1, root, dists, topf='system.data', trajf='out.dcd', atom_style='id resid type charge x y z',
                           shellsize=3.0, dt=0.5, stride=1, verbose=False):
    otopol = os.path.join(root, topf)
    odcd = os.path.join(root, trajf)
    u = MD.Universe(otopol, odcd, atom_style=atom_style, dt=dt)
    og = u.select_atoms('type {}'.format(otype), updating=True)
    osph1 = u.select_atoms('byres around {} bynum {}'.format(shellsize, num1), updating=True)
    oCES = []
    for d in dists:
        oCE = 0
        nCE1 = 0
        nCE2 = 0
        for ifr in tqdm(range(len(u.trajectory[::stride])), file=sys.stdout):
            id2 = find_atom_by_distance(og, 0, d, u.dimensions[0])
            osph2 = u.select_atoms('byres around {} bynum {}'.format(shellsize, id2), updating=True)
            oCE += coulomb_pairwise_separated(osph1, osph2, u.dimensions[0])
            nCE1 += len(osph1)
            nCE2 += len(osph2)
            for idt in range(stride):
                try:
                    u.trajectory.next()
                except StopIteration:
                    if verbose:
                        print("End of trajectory. Restting.")
            u.trajectory.rewind()
        oCE /= len(u.trajectory[::stride])
        nCE1 /= len(u.trajectory[::stride])
        nCE2 /= len(u.trajectory[::stride])
        oCES.append((oCE,nCE1,nCE2))
    return oCES
        
