#TODO: Rewrite basically entire thing; I don't want to deal with pandas memory anymore
from ..classes import Simulation
from ..analysis import pbcwrap
import os, sys, subprocess
import numpy as np
import pandas as pd
import pickle
import MDAnalysis as MD
from tqdm import tqdm
from scipy import fftpack, integrate
from scipy.optimize import leastsq
import tess
from collections import Counter
import MDAnalysis.analysis.rdf as MDrdf


class SiestaSimulation(Simulation):
    ''' Current required arguments:
            path, natoms, box_len, sim_name, timestep
            path: directory in which the simulation is stored
                str, list (str is turned into a list)
            natoms: number of atoms in the simulation
                int
            box_len: length of the box side (assumed cubic box)
                float
            sim_name: the siesta simulation label preceding the file endings
                str
            timestep: the simulation timestep in femtoseconds
                float
        Optional:
            siesta_atom_labels, nstep
            siesta_atom_labels: a dictionary corresponding to the atom label integer in the fdf file
                dict
            nstep: a dictionary of [path:val], where val is the step to stop reading in for a given simulation
                dict
            text_parse: whether or not pandas opens files with full lines as strings and then parses after 
                (i.e., needed for some reason for pbe4/56)
                bool
    '''
    def __init__(self, **kwargs):
        req_keys = ['path', 'natoms', 'box_len', 'sim_name', 'timestep']
        for k in req_keys:
            if k not in kwargs:
                raise KeyError('{} not in required arguments'.format(k))
        self.path = kwargs['path']
        if type(self.path) == str:
            self.path = [self.path]
        self.natoms = kwargs['natoms']
        self.box_len = kwargs['box_len']
        self.sim_name = kwargs['sim_name']
        self.timestep = kwargs['timestep']
        
        if 'siesta_atom_labels' in kwargs:
            self.siesta_atom_labels = kwargs['siesta_atom_labels']
        else:
            self.siesta_atom_labels = None
            
        if 'nstep' in kwargs:
            self.nstep = kwargs['nstep']
        else:
            self.nstep = {path:None for path in self.path}
        
        if 'out_name' in kwargs:
            self.out_name = kwargs['out_name']
            
        if 'text_parse' in kwargs:
            self.text_parse = kwargs['text_parse']
        else:
            self.text_parse = False
            
        if 'mix' in kwargs:
            self.mix = kwargs['mix']
        else:
            self.mix = False
            
        if 'spc' in kwargs:
            self.spc = kwargs['spc']
        else:
            self.spc = False
            
        if 'nacl' in kwargs:
            self.nacl = kwargs['nacl']
        else:
            self.nacl = True
        
        self.RDF = None
            
            
    def itrajectory(self):
        #equivalent to self.ANI, due to naming of SIESTA outputs
        self.itrajectory = self.iANI()

    # BASE CLASS REQUIREMENTS
    #-------------------------------------------------
    # SIESTA SPECIFIC
        
    def iANI(self, fix_pbc=True, file_error=False, save=False, load=False):
        ''' Initializes trajectory of the given simulation. '''
        pkl_path = os.path.join(self.path[0], self.sim_name+'.ANI.pkl')
        if not os.path.exists(pkl_path) and save == True: #saves automatically if not found and save=True
            print('No .ANI.pkl found. Will save in {}'.format(pkl_path))
            load = False
        elif (os.path.exists(pkl_path) and save == True):
            print('Overwriting saved .ANI.pkl.')
        if load == True:
            print('Loading saved .ANI.pkl.')
            with open(pkl_path, 'rb') as f:
                df = pickle.load(f)
                self.ANI = df
                #PBC already corrected in saved file; Do the distance attribute
                if self.nacl == True:
                    nav = df.loc[(df.STEP == 1) & (df.ATOM == 'Na'), ['X', 'Y', 'Z']].values[0]
                    clv = df.loc[(df.STEP == 1) & (df.ATOM == 'Cl'), ['X', 'Y', 'Z']].values[0]
                    dv = nav-clv
                    while (dv[dv<-self.box_len/2]).sum() != 0:
                        dv[dv < -self.box_len/2] = dv[dv<-self.box_len/2] + self.box_len
                    while (dv[dv> self.box_len/2]).sum() != 0:
                        dv[dv > self.box_len/2] = dv[dv > self.box_len/2] - self.box_len
                    self.DV = dv
                    self.D = np.linalg.norm(dv)
                return

        ret_df = pd.DataFrame()
        for path in self.path:
            file_path = os.path.join(path, self.sim_name+'.ANI')
            if not os.path.exists(file_path):
                if file_error == True:
                    raise ValueError("Specified .ANI file does not exist: {}".format(file_path))
                else:
                    print('Specified .ANI file does not exist: {}'.format(file_path))
                    print('Continuing without.')
                    continue
            df = pd.DataFrame()
            #drop first and second row to get columns right, can drop nans later
            drop_list=[0,1]
            if self.nstep[path] == None:
                if self.text_parse:
                    df = pd.read_table(file_path, header=None, skiprows=drop_list)
                    df = df[0].str.split().tolist()
                    df = pd.DataFrame(df)
                    df.loc[:, [1,2,3]] = df.loc[:, [1,2,3]].astype(float)
                else:
                    df = pd.read_table(file_path, header=None, skiprows=drop_list, delim_whitespace=True)
            else:
                #nrows = 290 + (self.natoms+2)*(self.nstep[path]-1))
                nrows = self.natoms + (self.natoms+2)*(self.nstep[path]-1)
                df = pd.read_table(file_path, header=None, skiprows=drop_list, delim_whitespace=True,
                                   nrows=nrows)
            #assign steps, use index mod for step, mod290
            df = df.dropna()
            df.columns = ['ATOM', 'X', 'Y', 'Z']
            if self.text_parse:
                df.loc[:, ['X','Y','Z']] = df.loc[:, ['X', 'Y', 'Z']].astype(float)

            df['STEP'] = df.reset_index().index//self.natoms+1
            df = df.reset_index(drop=True)        
            
            #use this to find ion distance
            if self.nacl == True:
                nav = df.loc[(df.STEP == 1) & (df.ATOM == 'Na'), ['X', 'Y', 'Z']].values[0]
                clv = df.loc[(df.STEP == 1) & (df.ATOM == 'Cl'), ['X', 'Y', 'Z']].values[0]
                dv = nav-clv
                #fix bc; distance in one direction cannot be longer than box length/2, so fold back to opposite side of box
                while (dv[dv<-self.box_len/2]).sum() != 0:
                    dv[dv < -self.box_len/2] = dv[dv<-self.box_len/2] + self.box_len
                while (dv[dv> self.box_len/2]).sum() != 0:
                    dv[dv > self.box_len/2] = dv[dv > self.box_len/2] - self.box_len
            
            #set up the atom_index -> atom name
            df['ATOM_IND'] = df.index%self.natoms + 1
            if fix_pbc == True:
                while (df[['X','Y','Z']] < 0).values.sum() != 0:
                    df.loc[df.X < 0, 'X'] = df.loc[df.X < 0, 'X'] + self.box_len
                    df.loc[df.Y < 0, 'Y'] = df.loc[df.Y < 0, 'Y'] + self.box_len
                    df.loc[df.Z < 0, 'Z'] = df.loc[df.Z < 0, 'Z'] + self.box_len
    
                while (df[['X','Y','Z']] > self.box_len).values.sum() != 0:
                    df.loc[df.X > self.box_len, 'X'] = df.loc[df.X > self.box_len, 'X'] - self.box_len
                    df.loc[df.Y > self.box_len, 'Y'] = df.loc[df.Y > self.box_len, 'Y'] - self.box_len
                    df.loc[df.Z > self.box_len, 'Z'] = df.loc[df.Z > self.box_len, 'Z'] - self.box_len
            
            #concat to base df
            ret_df = pd.concat([ret_df, df])
            ret_df = ret_df.reset_index(drop=True)
            ret_df['STEP'] = ret_df.index//self.natoms+1
            if self.text_parse:
                ret_df.loc[:, ['X','Y','Z']] = ret_df.loc[:, ['X', 'Y', 'Z']].astype(float)

        if save == True:
            with open(pkl_path, 'wb') as f:
                pickle.dump(ret_df, f)

        self.ANI = ret_df
        if self.nacl == True:
            self.DV = dv.astype(np.float)
            self.D = np.linalg.norm(dv)
            
    def iMANI(self, fxyz=True, save=False, load=True):
        #TODO: use chainreader for multiple trajectories in different file
        ppdb = os.path.join(self.path[0], self.sim_name+'.sim.PDB')
        ptraj = os.path.join(self.path[0], self.sim_name+'.traj.trr')
        if load == True:
            assert os.path.exists(ptraj), "Can't load from .pkl -- {} does not exist".format(ptraj)
            MDAU = MD.Universe(ppdb, ptraj, dt=0.5)
            MDAU.trajectory.units['time'] = 'fs'
            self.Universe = MDAU
            return
        traj = os.path.join(self.path[0], self.sim_name+'.ANI')
        MDAU = MD.Universe(traj, format='xyz', dt=self.timestep)
        MDAU.trajectory.units['time'] = 'fs'
        MDAU.dimensions = 3*[self.box_len]+3*[90.0]
        if fxyz:
            print("Assigning forces to trajectory based on .FA file.")
            ftraj = os.path.join(self.path[0], self.sim_name+'.FA')
            tmpU = MD.Universe(ftraj, format='fxyz', dt=self.timestep)
            assert tmpU.trajectory.n_frames == MDAU.trajectory.n_frames, "Lengths of .FA and .ANI file do not match."
            for fts in tqdm(range(tmpU.trajectory.n_frames)):
                MDAU.trajectory[fts].forces = tmpU.trajectory[fts].positions
    
        if save and os.path.exists(ptraj):
            print("Overwriting previous .pkl.")
        if save and not os.path.exists(ptraj):
            print("Saving trajectory to file {}".format(ptraj))
        if save:
            allsave = MDAU.select_atoms("name *")
            allsave.write(ppdb)
            with MD.Writer(ptraj, self.natoms) as W:
                print('Writing structural information in {} and trajectory in {}.'.format(ppdb,ptraj))
                for fts in tqdm(range(MDAU.trajectory.n_frames)):
                    ts = MDAU.trajectory[fts]
                    W.write(ts)
        self.Universe = MDAU
        return
    
    
    def iMFA(self):
        #TODO: use chainreader for multiple trajectories in different file
        #TODO: maybe change implementation -- this requires modifying some package classes as new ones for
        # siesta .FA output, or change siesta.FA output to actually be closer to xyz
        traj = os.path.join(self.path[0], self.sim_name+'.FA')
        self.FUniverse = MD.Universe(traj, format='xyz', dt=self.timestep, units={"time":"fs"})
        self.FUniverse.trajectory.units['time'] = 'fs'
        self.FUniverse.dimensions = 3*[self.box_len]+3*[90.0]


    
    def iMIX(self, save=False, load=True):
        if self.mix == False:
            raise ValueError("self.mix is not valid; simulation is not mixed sampling")
        pkl_path = os.path.join(self.path[0], self.sim_name+'.MIX.pkl')
        if load == True:
            with open(pkl_path, 'rb') as f:
                df, force, ener = pickle.load(f)
                #PBC already corrected in saved file; Do the distance attribute
                nav = df.loc[df.ATOM == 'Na', ['X', 'Y', 'Z']].values[0]
                clv = df.loc[df.ATOM == 'Cl', ['X', 'Y', 'Z']].values[0]
                dv = nav-clv
                while (dv[dv<-self.box_len/2]).sum() != 0:
                    dv[dv < -self.box_len/2] = dv[dv<-self.box_len/2] + self.box_len
                while (dv[dv> self.box_len/2]).sum() != 0:
                    dv[dv > self.box_len/2] = dv[dv > self.box_len/2] - self.box_len
                self.XYZ = df
                self.ANI = self.XYZ
                self.FA = force
                self.MDE = ener
                self.DV = dv
                self.D = np.linalg.norm(dv)
                return
        else:
            #bash prints out in alphanumeric order, so the loop would go through in that order
            steps = sorted([i for i in os.listdir(self.path[0]) if os.path.isdir(os.path.join(self.path[0], i))])
            steps = [int(i) for i in steps]
            #if empty, just a single SPC
            if len(steps) == 0 and self.spc == True:
                steps = [0]
            else:
                raise ValueError("self.spc is False, but there are no step subdirectories in self.path")
            self.steps = steps
            ANI = pd.DataFrame()
            FA = pd.DataFrame()
            MDE = pd.DataFrame()
            for step in steps:
                if self.spc == True:
                    p = os.path.join(self.path[0], self.sim_name+'.out')
                else:
                    p = os.path.join(self.path[0], str(step), self.sim_name+'.out')
                out = pd.read_csv(p, delimiter="\n", header=None, index_col=False)
                out = out[0]
                coor_ind = out[out.str.contains('outcoor')].index
                #every other; the forces are printed twice
                forc_ind = out[out.str.contains('siesta: Atomic forces')].index[::2]
                warn_ind = out[out.str.contains('COOP/COHP')].index
                ener_ind = out[out.str.contains('siesta: Final energy')].index
                coor_inds = []
                forc_inds = []
                ener_inds = []
                #the misprinting will make us go as much further as there are misprints in a given range
                for i in coor_ind:
                    rng = np.arange(i+1, i+1+self.natoms)
                    intsct = np.intersect1d(warn_ind, rng)
                    if len(intsct) != 0:
                        print("Misprint detected, shifting index of range")
                        rng = np.arange(i+1, i+1+self.natoms+len(intsct))
                    coor_inds.append(rng)
                for i in forc_ind:
                    rng = np.arange(i+1, i+1+self.natoms)
                    forc_inds.append(rng)
                for i in ener_ind:
                    rng = np.arange(i+1, i+1+12)
                    ener_inds.append(rng)
                coor_inds = np.concatenate(coor_inds, axis=0)
                forc_inds = np.concatenate(forc_inds, axis=0)
                ener_inds = np.concatenate(ener_inds, axis=0)
                #there is an evident issue in siesta  occasionally 
                #printing COOP/COHP analysis warning in the middle of coordinates
                #['NOTE:', 'Your', 'COOP/COHP', 'analysis', 'might', 'be', 'affected', 'by', 'folding.']
                #up to -9
                ani = out.loc[coor_inds].str.split().tolist()
                misprints = np.intersect1d(coor_inds, warn_ind)
                pops = 0
                for i in misprints:
                    list_ind = np.where(coor_inds == i)[0][0]
                    ani[list_ind-pops] = ani[list_ind-pops][:-9]+ani[list_ind-pops+1]
                    ani.pop(list_ind-pops+1)
                    pops += 1
                fa = out.loc[forc_inds].str.split().tolist()
                ener = out.loc[ener_inds].str.split(':').apply(lambda x: x[-1])
                ener = ener.str.split('=').tolist()
                #for some reason a bunch of "None"s appear in the expansion
                df = pd.DataFrame(ani)
                df.columns = ['X', 'Y', 'Z', 'ATOM_LAB', 'ATOM_IND', 'ATOM']
                df.loc[:, 'STEP'] = step
                df.loc[:, ['X','Y','Z']] = df.loc[:, ['X','Y','Z']].astype(float)
                df.loc[:, ['ATOM_LAB','ATOM_IND']] = df.loc[:, ['ATOM_LAB','ATOM_IND']].astype(int)
                
                fa = pd.DataFrame(fa)
                fa.columns = ['ATOM_IND', 'FX', 'FY', 'FZ']
                fa.loc[:, 'ATOM'] = df.loc[:, 'ATOM']
                fa.loc[:, 'STEP'] = df.loc[:, 'STEP']
                fa.loc[:, ['FX','FY','FZ']] = fa.loc[:, ['FX','FY','FZ']].astype(float)
                
                tmp_ener = pd.DataFrame(ener)
                ener = pd.DataFrame()
                for i in tmp_ener[0].unique():
                    vals = tmp_ener.loc[tmp_ener[0] == i, 1].values.astype(np.float)
                    col = i.strip()
                    ener[col] = vals
                ener.loc[:, 'STEP'] = step
                
                ANI = pd.concat([ANI, df])
                FA = pd.concat([FA, fa])
                MDE = pd.concat([MDE, ener])
            
            ANI = ANI.sort_values('STEP').reset_index(drop=True)
            FA = FA.sort_values('STEP').reset_index(drop=True)
            MDE = MDE.sort_values('STEP').reset_index(drop=True)
            
            nav = ANI.loc[ANI.ATOM == 'Na', ['X', 'Y', 'Z']].values[0]
            clv = ANI.loc[ANI.ATOM == 'Cl', ['X', 'Y', 'Z']].values[0]
            dv = nav-clv
            while (dv[dv<-self.box_len/2]).sum() != 0:
                dv[dv < -self.box_len/2] = dv[dv<-self.box_len/2] + self.box_len
            while (dv[dv> self.box_len/2]).sum() != 0:
                dv[dv > self.box_len/2] = dv[dv > self.box_len/2] - self.box_len

            self.FA = FA
            self.XYZ = ANI
            self.ANI = self.XYZ
            self.DV = dv
            self.MDE = MDE
            self.D = np.linalg.norm(dv)
            
            if save == True:
                with open(pkl_path, 'wb') as f:
                    pickle.dump((ANI, FA, MDE), f)
            return
            
        
        
    def iXYZ(self, fix_pbc=True, file_error=False, save=False, load=False):
        ''' Initializes trajectory of the given simulation. '''
        pkl_path = os.path.join(self.path[0], self.sim_name+'.XYZ.pkl')
        if not os.path.exists(pkl_path) and save == True: #saves automatically if not found and save=True
            print('No .XYZ.pkl found. Will save in {}'.format(pkl_path))
            load = False
        elif (os.path.exists(pkl_path) and save == True):
            print('Overwriting saved .XYZ.pkl.')
        if load == True:
            print('Loading saved .XYZ.pkl.')
            with open(pkl_path, 'rb') as f:
                df = pickle.load(f)
                #PBC already corrected in saved file; Do the distance attribute
                nav = df.loc[(df.STEP == 1) & (df.ATOM == 'Na'), ['X', 'Y', 'Z']].values[0]
                clv = df.loc[(df.STEP == 1) & (df.ATOM == 'Cl'), ['X', 'Y', 'Z']].values[0]
                dv = nav-clv
                while (dv[dv<-self.box_len/2]).sum() != 0:
                    dv[dv < -self.box_len/2] = dv[dv<-self.box_len/2] + self.box_len
                while (dv[dv> self.box_len/2]).sum() != 0:
                    dv[dv > self.box_len/2] = dv[dv > self.box_len/2] - self.box_len
                self.XYZ = df
                self.ANI = self.XYZ
                self.DV = dv
                self.D = np.linalg.norm(dv)
                return

        ret_df = pd.DataFrame()
        for path in self.path:
            file_path = os.path.join(path, self.sim_name+'.xyz')
            if not os.path.exists(file_path):
                if file_error == True:
                    raise ValueError("Specified .xyz file does not exist: {}".format(file_path))
                else:
                    print('Specified .xyz file does not exist: {}'.format(file_path))
                    print('Continuing without.')
                    continue
            df = pd.DataFrame()
            #drop first and second row to get columns right, can drop nans later
            drop_list = []
            if self.nstep[path] == None:
                df = pd.read_table(file_path, header=None, skiprows=drop_list, delim_whitespace=True)
            else:
                pass #should not need to use self.nstep for these
            #assign steps, use index mod for step, mod290
            df = df.dropna()
            df.columns = ['X', 'Y', 'Z', 'ATOM_LABEL']
            df['STEP'] = int(path.split('/')[-1])
            #concat to base df
            ret_df = pd.concat([ret_df, df])
            ret_df = ret_df.reset_index(drop=True)
        
        ret_df['ATOM'] = 'NAN'    
        for k in self.siesta_atom_labels:
            aind = self.siesta_atom_labels[k]
            ret_df.loc[ret_df.ATOM_LABEL == aind, 'ATOM'] = k
        #use this to find ion distance
        nav = ret_df.loc[(ret_df.STEP == 1) & (ret_df.ATOM == 'Na'), ['X', 'Y', 'Z']].values[0]
        clv = ret_df.loc[(ret_df.STEP == 1) & (ret_df.ATOM == 'Cl'), ['X', 'Y', 'Z']].values[0]
        dv = nav-clv
        #fix bc; distance in one direction cannot be longer than t10box length/2, so fold back to opposite side of box
        while (dv[dv<-self.box_len/2]).sum() != 0:
            dv[dv < -self.box_len/2] = dv[dv<-self.box_len/2] + self.box_len
        while (dv[dv> self.box_len/2]).sum() != 0:
            dv[dv > self.box_len/2] = dv[dv > self.box_len/2] - self.box_len
        
        #set up the atom_index -> atom name
        ret_df['ATOM_IND'] = ret_df.index%290 + 1
        if fix_pbc == True:
            while (ret_df[['X','Y','Z']] < 0).values.sum() != 0:
                ret_df.loc[df.X < 0, 'X'] = ret_df.loc[df.X < 0, 'X'] + self.box_len
                ret_df.loc[df.Y < 0, 'Y'] = ret_df.loc[df.Y < 0, 'Y'] + self.box_len
                ret_df.loc[df.Z < 0, 'Z'] = ret_df.loc[df.Z < 0, 'Z'] + self.box_len

            while (ret_df[['X','Y','Z']] > self.box_len).values.sum() != 0:
                ret_df.loc[ret_df.X > self.box_len, 'X'] = ret_df.loc[ret_df.X > self.box_len, 'X'] - self.box_len
                ret_df.loc[ret_df.Y > self.box_len, 'Y'] = ret_df.loc[ret_df.Y > self.box_len, 'Y'] - self.box_len
                ret_df.loc[ret_df.Z > self.box_len, 'Z'] = ret_df.loc[ret_df.Z > self.box_len, 'Z'] - self.box_len
        ret_df = ret_df.sort_values(['STEP', 'ATOM_IND']).reset_index(drop=True)
        if save == True:
            with open(pkl_path, 'wb') as f:
                pickle.dump(ret_df, f)
                
        self.XYZ = ret_df      
        self.ANI = self.XYZ
        self.DV = dv.astype(np.float)
        self.D = np.linalg.norm(dv)
        
    def ION_DISTS(self):
        dists = pd.DataFrame()
        max_step = self.ANI.STEP.max()
        for i in range(max_step):
            sub = self.ANI[self.ANI.STEP == i+1]
            nav = sub[sub.ATOM == 'Na', ['X', 'Y', 'Z']].values
            clv = sub[sub.ATOM == 'Cl', ['X', 'Y', 'Z']].values
            nadists = sub[['X', 'Y', 'Z']].values - nav
            
    def iVEL(self, subcm=True):
        vdf = self.ANI.loc[:, ['X', 'Y', 'Z']].diff(periods=self.natoms).dropna()
        vdf = vdf/self.timestep
        vdf.columns = ['VX','VY', 'VZ']
        newinds = vdf.index
        vdf[['ATOM','STEP']] = self.ANI.loc[newinds, ['ATOM','STEP']]
        vdf = vdf.reset_index(drop=True)
        if subcm == True:
            vcm = vdf.loc[:, ['VX', 'VY', 'VZ']].groupby(np.arange(len(vdf))//self.natoms).mean()
            bigvcm = pd.DataFrame(np.repeat(vcm.values,self.natoms,axis=0))
            bigvcm.columns = vcm.columns
            vdf.loc[:, ['VX','VY','VZ']] = vdf.loc[:,['VX','VY','VZ']]-bigvcm
        self.VEL = vdf
    
    def oTANI(self, overwrite=False):
        tanip = os.path.join(self.path[0], self.sim_name+'.tANI')
        assert not os.path.exists(tanip), ".tANI file already exists in {}. Delete to continue.".format(tanip)
            
        with open(tanip, 'w') as f:
            datarr = self.ANI.loc[:, ['ATOM','X','Y','Z']]
            for s in self.ANI.STEP.unique():
                f.write(str(self.natoms))
                f.write('\n')
                f.write('\n')
                datarr[(s-1)*self.natoms:s*self.natoms].to_csv(f, sep=' ', index=False,
                       header=False) 
                
    def output_coords(self, dir_path, step, form='siesta', overwrite=False, out_add=''):
        if step == -1:
            step = self.ANI.STEP.max()
        step_df = self.ANI.loc[self.ANI.STEP == step].reset_index(drop=True)
        outpath = os.path.join(dir_path, self.sim_name+out_add+'.xyz')
        labelpath = os.path.join(dir_path, self.sim_name+'.label')
        if form == 'siesta':
            # format for siesta coord block = x y z atomic_species_index
            atoms = step_df.ATOM.unique()
            step_df.loc[:, 'INDEX'] = 0
            s=''
            if self.siesta_atom_labels:
                for k in self.siesta_atom_labels.keys():
                    step_df.loc[step_df.ATOM == k, 'INDEX'] = self.siesta_atom_labels[k]
                    s += '{} labeled as {}. '.format(k, self.siesta_atom_labels[k])
            else:
                for i in range(len(atoms)):
                    step_df.loc[step_df.ATOM == atoms[i], 'INDEX'] = i+1
                    s += '{} labeled as {}. '.format(atoms[i], i+1)
            cols = ['X', 'Y', 'Z', 'INDEX']
            write_df = step_df[['X', 'Y', 'Z', 'INDEX']]
            textfile = open(labelpath, 'w')
            textfile.write(s)
            textfile.close()
        elif form == 'cp2k':
            # format for cp2k inp atom coords is ATOM_LETTER x y z
            write_df = step_df[['ATOM', 'X', 'Y', 'Z']]
            cols = ['ATOM', 'X', 'Y', 'Z']
        elif form == 'voro++':
            write_df = step_df[['ATOM_IND', 'X', 'Y', 'Z']]
            cols=['ATOM_IND', 'X', 'Y', 'Z']
        if os.path.exists(outpath) and overwrite == False:
            raise Exception("File {} already exists.".format(outpath))
        elif (os.path.exists(outpath) and overwrite == True) or (not os.path.exists(outpath)):
            write_df.to_csv(outpath, sep=' ', header=False, index=False,
                columns=cols)

    
    def VACFFFT(self, mult_mass = True, norm = True, fft_ax=0):
        ACF = 0.0
        normalize = 0.0
        Trun = self.VEL.STEP.max()*self.timestep
        mass_dct = {'Na':22.989769, 'Cl':35.453, 'H':1.00794, 'O':15.999}
        for i in range(0, self.natoms):
            print('VACFFT: Current atom: {} out of {}'.format(i+1, self.natoms))
            atom_df = self.VEL.loc[i::self.natoms]
            #she multiplies by the mass, which I don't know is necessary, but I'll do it
            vel_arr = atom_df[['VX', 'VY', 'VZ']].values
            if mult_mass==True:
                atom = atom_df.ATOM.unique()[0]
                m = mass_dct[atom]
                vel_arr = vel_arr * np.sqrt(m)
            fft_arr = fftpack.fft(vel_arr, axis=fft_ax)
            #magnitude of each number and square
            fft_arr = np.absolute(fft_arr)**2
            #inverse transform back into direct space, normalize by size of arr
            ifft_arr = fftpack.ifft(fft_arr, axis=fft_ax)
            #ifft_arr = np.absolute(ifft_arr)/len(ifft_arr)
            ifft_arr = ifft_arr/len(ifft_arr)
            #add the dots of each component together for normalization
            normalize += np.einsum('ij,ij->j', vel_arr, vel_arr).sum()/Trun
            if mult_mass==True:
                normalize = normalize*m
            #Autocorrelation is 1D array, sum at each timestep of the velocities
            ACF += ifft_arr.sum(axis=1)
        ACF = ACF/(self.natoms*normalize)
        #Fourier transform autocorrelation back
        VIB = fftpack.fft(ACF, axis=fft_ax)
        VIB = np.absolute(VIB)
        #Generate timestep list corresponding to the data
        steps = self.VEL.STEP.unique()
        c = 29979245800 #cm/s
        fs_to_s = 1e-15 #s/fs
        #want inverse length
        omega = np.linspace(0.0, 1/(self.timestep*fs_to_s*c), max(steps)) #1/fs * fs/s * s/cm
        if norm == True:
            VIB = VIB/VIB[0]
        self.VACFFFT = (omega, VIB)
    
    def iMDE(self, file_error=False, save=True, load=False):
        pkl_path = os.path.join(self.path[0], self.sim_name+'.MDE.pkl')
        if not os.path.exists(pkl_path) and save == True: #saves automatically if not found and true
            print('No .MDE.pkl found. Will save in {}'.format(pkl_path))
            save = True
            load = False
        elif (os.path.exists(pkl_path) and save == True):
            print('Overwriting saved .MDE.pkl.')
        if load == True:
            print('Loading saved .MDE.pkl.')
            with open(pkl_path, 'rb') as f:
                df = pickle.load(f)
                self.MDE = df
                return
        ret_df = pd.DataFrame()
        for path in self.path:
            #first check for accidental appending
            file_path = os.path.join(path, self.sim_name+'.MDE')
            if not os.path.exists(file_path):
                if file_error == True:
                    raise ValueError("Specified .MDE file does not exist: {}".format(file_path))
                else:
                    print('Specified .MDE file does not exist: {}'.format(file_path))
                    print('Continuing without.')
                    continue
            lst = subprocess.check_output('grep -n "T" {}'.format(file_path), shell=True).decode('ascii').split('\n')[:-1]
            lst = [i.split(':')[0] for i in lst]
            if len(lst) > 1:
                print('Appending detected. Will add data to simulation.')
            skip = [int(i)-1 for i in lst]
            if self.nstep[path] == None:
                df = pd.read_table(file_path, header=None, skiprows=skip, delim_whitespace=True)
            else:
                nrows = self.nstep[path]-len(skip)
                df = pd.read_table(file_path, header=None, skiprows=skip, delim_whitespace=True,
                               nrows=nrows)            
            df.columns = ['STEP', 'T', 'E_KS', 'E_TOT', 'VOL', 'P']
            for col in df.columns:
                if col == 'STEP':
                    continue
                else:
                    df[col] = pd.to_numeric(df[col])
            if self.nstep[path] != None:
                df = df.loc[df.STEP <= self.nstep[path]]
            ret_df = pd.concat([ret_df, df])
            ret_df = ret_df.reset_index(drop = True)
            ret_df['STEP'] = ret_df.index+1
        if save == True:
                with open(pkl_path, 'wb') as f:
                    pickle.dump(ret_df, f)
        self.MDE = ret_df
    
    def iOUTEN(self, out_name, col_search_dict, file_error=False, save=True, load=False):
        pkl_path = os.path.join(self.path[0], self.out_name+'.OUTEN.pkl')
        if not os.path.exists(pkl_path) and save == True: #saves automatically if not found and true
            print('No .OUTEN.pkl found. Will save in {}'.format(pkl_path))
            save = True
            load = False
        elif (os.path.exists(pkl_path) and save == True):
            print('Overwriting saved .OUTEN.pkl.')
        if load == True:
            print('Loading saved .OUTEN.pkl.')
            with open(pkl_path, 'rb') as f:
                df = pickle.load(f)
                self.MDE = df
                return
        df_dct = {k:[] for k in col_search_dict}
        df_dct['STEP'] = []
        for path in self.path:
            #first check for accidental appending
            file_path = os.path.join(path, self.out_name+'.out')
            if not os.path.exists(file_path):
                if file_error == True:
                    raise ValueError("Specified .out file does not exist: {}".format(file_path))
                else:
                    print('Specified .out file does not exist: {}'.format(file_path))
                    print('Continuing without.')
                    continue
            
            for k in col_search_dict.keys():
                if k == list(col_search_dict.keys())[0]:
                    df_dct['STEP'].append(int(path.split('/')[-1]))

                search = col_search_dict[k]
                try:
                    ret = subprocess.check_output('grep -n "{}" {}'.format(search, file_path), shell=True).decode('ascii')
                    ret = ret.split('=')[-1].split('\n')[0]
                    df_dct[k].append(float(ret))
                except subprocess.CalledProcessError as e:
                    print(e.output)
                    print('Continuing')
                    df_dct[k].append(np.nan)
                    continue
        self.df_dct = df_dct
        ret_df = pd.DataFrame.from_dict(df_dct).dropna()
            
        if save == True:
            with open(pkl_path, 'wb') as f:
                pickle.dump(ret_df, f)
        self.MDE = ret_df
        
    
    def step_check(self, forces='fa', keep_min=True):
        mde_step = self.MDE.STEP.max()
        ani_step = self.ANI.STEP.max()
        if forces == 'fa':
            force_step = self.FA.STEP.max()
        elif forces == 'fac':
            force_step = self.FAC.STEP.max()
        min_step = min([mde_step, ani_step, force_step])
        print('MAX STEP FOR SIMULATION')
        print('ANI: {}'.format(ani_step))
        print('MDE: {}'.format(mde_step))
        print('FORCE: {}'.format(force_step))
        if keep_min == True:
            print('Keeping minimum number of steps...')
            self.MDE = self.MDE.loc[self.MDE.STEP <= min_step]
            self.ANI = self.ANI.loc[self.ANI.STEP <= min_step]
            self.FORCE = self.FORCE.loc[self.FORCE.STEP <= min_step]
        
    def iFA(self, updated_fa=False, file_error=False, save=False, load=False,
            spc=False):
        pkl_path = os.path.join(self.path[0], self.sim_name+'.FA.pkl')
        if not os.path.exists(pkl_path) and save == True: #saves automatically if not found and true
            print('No .FA.pkl found. Will save in {}'.format(pkl_path))
            save = True
            load = False
        elif (os.path.exists(pkl_path) and save == True):
            print('Overwriting saved .FA.pkl.')
        if load == True:
            print('Loading saved .FA.pkl.')
            with open(pkl_path, 'rb') as f:
                df = pickle.load(f)
                self.FA = df
                self.FORCE = self.FA
                return
        ret_df = pd.DataFrame()
        for path in self.path:
            if updated_fa == False:
                file_path = os.path.join(path, 'PYTHONOUT.FOR')
                if not os.path.exists(file_path):
                    if file_error == True:
                        raise ValueError("Specified .FA file does not exist: {}".format(file_path))
                    else:
                        print('Specified .FA file does not exist: {}'.format(file_path))
                        print('Continuing without.')
                        continue
                cols = ['ATOM_IND', 'FX', 'FY', 'FZ', 'STEP', 'ATOM']
                df = pd.read_table(file_path, header=None, delim_whitespace=True)
                df.columns = cols
            else:
                file_path = os.path.join(path, self.sim_name+'.FA')
                if not os.path.exists(file_path):
                    if file_error == True:
                        raise ValueError("Specified .FA file does not exist: {}".format(file_path))
                    else:
                        print('Specified .FA file does not exist: {}'.format(file_path))
                        print('Continuing without.')
                        continue
                df = pd.DataFrame()
                #drop first row to get columns right, can drop nans later
                drop_list=[0]
                if self.nstep[path] == None:
                    try:
                        if self.text_parse:
                            df = pd.read_table(file_path, header=None, skiprows=drop_list)
                            df = df[0].str.split().tolist()
                            df = pd.DataFrame(df)
                            df.loc[:, [1,2,3]] = df.loc[:, [1,2,3]].astype(float)
                        else:
                            df = pd.read_table(file_path, header=None, skiprows=drop_list, delim_whitespace=True)
                    except ValueError:
                        print('Bad file. Skipping')
                        continue
                else:
                    nrows = 290 + (self.natoms+1)*(self.nstep[path]-1)
                    df = pd.read_table(file_path, header=None, skiprows=drop_list, delim_whitespace=True,
                                   nrows=nrows)
                #assign steps, use index mod for step, mod290
                df = df.dropna()
                df.columns = ['ATOM_IND', 'FX', 'FY', 'FZ']
                if self.text_parse:
                    df.loc[:, ['FX','FY','FZ']] = df.loc[:, ['FX', 'FY', 'FZ']].astype(float)
                if spc == False:
                    df['STEP'] = df.index//self.natoms + 1
                elif spc == True:
                    df['STEP'] = int(self.sim_name.split('_')[-1])
            ret_df = pd.concat([ret_df, df])
            ret_df = ret_df.reset_index(drop=True)
            ret_df['ATOM'] = self.ANI.ATOM[:ret_df.index[-1]+1]
            ret_df['STEP'] = ret_df.index//self.natoms+1
            if self.text_parse:
                ret_df.loc[:, ['FX','FY','FZ']] = ret_df.loc[:, ['FX', 'FY', 'FZ']].astype(float)


            
        if save == True:
            with open(pkl_path, 'wb') as f:
                pickle.dump(ret_df, f)
        self.FA = ret_df
        self.FORCE = self.FA
            
    def iFAC(self, updated_fac=False, file_error=False, save=False, load=False):
        pkl_path = os.path.join(self.path[0], self.sim_name+'.FAC.pkl')
        if not os.path.exists(pkl_path) and save == True: #saves automatically if not found and true
            print('No .FA.pkl found. Will save in {}'.format(pkl_path))
            save = True
            load = False
        elif (os.path.exists(pkl_path) and save == True):
            print('Overwriting saved .FAC.pkl.')
        if load == True:
            print('Loading saved .FAC.pkl.')
            with open(pkl_path, 'rb') as f:
                df = pickle.load(f)
                self.FAC = df
                self.FORCE = self.FAC
                return
        ret_df = pd.DataFrame()
        for path in self.path:
            if updated_fac == False:
                raise ValueError("Cannot retrieve old simulation .FAC files, as it was only printed out for final step.")
            else:
                file_path = os.path.join(path, self.sim_name+'.FAC')
                if not os.path.exists(file_path):
                    if file_error == True:
                        raise ValueError("Specified .ANI file does not exist: {}".format(file_path))
                    else:
                        print('Specified .ANI file does not exist: {}'.format(file_path))
                        print('Continuing without.')
                        continue
                df = pd.DataFrame()
                #drop first row to get columns right, can drop nans later
                drop_list=[0]
                if self.nstep[path] == None:
                    df = pd.read_table(file_path, header=None, skiprows=drop_list, delim_whitespace=True)
                else:
                    nrows = 290 + (self.natoms+1)*(self.nstep[path]-1)
                    df = pd.read_table(file_path, header=None, skiprows=drop_list, delim_whitespace=True,
                                   nrows=nrows)
                #assign steps, use index mod for step, mod290
                df = df.dropna()
                df.columns = ['ATOM_IND', 'FX', 'FY', 'FZ']
                df['STEP'] = df.index//self.natoms+1
            if self.nstep[path] != None:
                df = df.loc[df.index//self.natoms + 1 < self.nstep[path]]
            ret_df = pd.concat([ret_df, df])
            ret_df = ret_df.reset_index(drop=True)
            ret_df['STEP'] = ret_df.index//self.natoms+1
            ret_df['ATOM'] = self.ANI.ATOM[:ret_df.index[-1]+1]
        
        if save == True:
            with open(pkl_path, 'wb') as f:
                pickle.dump(ret_df, f)
        self.FAC = df
        self.FORCE = self.FAC
    
    def iMFPROJ(self, a1=0, a2=1, save=False, load=True):
        pklpath = os.path.join(self.path[0], 'fproj.pkl')
        if load:
            assert os.path.exists(pklpath), "Didn't find force projection pickle file."
            print("Found existing .pkl file. Loading.")
            with open(pklpath, 'rb') as f:
                force_proj = pickle.load(f)
            self.FPROJ = force_proj
            self.FORCE_PROJ = force_proj
            return
        
        force_proj = {'ds':[],
                      'dhs':[],
                      'a1':[],
                      'a2':[],
                      'net':[],
                      'netd':[],
                      'netp':[],
                      'netdp':[]}
        for ts in tqdm(self.Universe.trajectory):
            #center on a1
            x = pbcwrap(ts.positions - ts.positions[a1], self.box_len)
            f1 = ts.forces[a1]
            f2 = ts.forces[a2]
            fn = f1+f2
            fnd = f1-f2
            d12 = x[a1]-x[a2]
            d12h = d12/np.linalg.norm(d12)
            a1fp = np.dot(f1, d12h)
            a2fp = np.dot(f2, d12h)
            afnp = np.dot(fn, d12h)
            afnd = np.dot(fnd, d12h)
            
            force_proj['ds'].append(d12)
            force_proj['dhs'].append(d12h)
            force_proj['a1'].append(a1fp)
            force_proj['a2'].append(a2fp)
            force_proj['net'].append(afnp)
            force_proj['netd'].append(afnd)
            force_proj['netp'].append(a1fp+a2fp)
            force_proj['netdp'].append(a1fp-a2fp)
            
        for k in force_proj.keys():
            force_proj[k] = np.array(force_proj[k])
        
        if save and os.path.exists(pklpath):
            print("Overwriting previously saved projection pkl.")
        if save and not os.path.exists(pklpath):
            print("Saving projections at {}".format(pklpath))
        if save:
            with open(pklpath, 'wb') as f:
                pickle.dump(force_proj, f)

        self.FPROJ = force_proj
        self.FORCE_PROJ = force_proj
        return

    
    def iFORCE_PROJ(self, forces='fafac', overwrite=False):
        pklpath = os.path.join(self.path[0], 'forceproj_{}.pkl'.format(forces))
        if os.path.exists(pklpath) and overwrite == False:
            print("Found existing .pkl file. Loading.")
            with open(pklpath, 'rb') as f:
                force_proj = pickle.load(f)
        else:
            print('No existing .pkl file found.')
            if forces == 'fa':
                forcedf = self.FA
            elif forces == 'fac':
                forcedf = self.FAC
            elif forces == 'fafac':
                fa = self.FA
                fac = self.FAC
                forcedf = fa
                forcedf[['FX','FY','FZ']] = fa[['FX','FY','FZ']] - fac[['FX','FY','FZ']]
            if len(forcedf) == 0:
                print("No forces available.")
                self.FORCE_PROJ = np.nan
                return
            
            force_proj = {}
            forcedf = forcedf[forcedf.ATOM.isin(['Na', 'Cl'])]
            d = self.DV
            #fix pbc; distances cannot exceed box length/2
            dhat = d/self.D
            naf = forcedf.loc[forcedf.ATOM == 'Na', ['FX','FY','FZ']].values
            clf = forcedf.loc[forcedf.ATOM == 'Cl', ['FX','FY','FZ']].values
            netf = naf+clf
            nafproj = np.einsum('ij,j->i', naf, dhat)
            clfproj = np.einsum('ij,j->i', clf, dhat)
            netfproj = np.einsum('ij,j->i', netf, dhat)

            force_proj['Na'] = nafproj
            force_proj['Cl'] = clfproj
            force_proj['Net'] = netfproj
            
            with open(pklpath, 'wb') as f:
                pickle.dump(force_proj, f)
        
        self.FORCE_PROJ = force_proj
        
        
    def ifproj_hist(self, bins=100, atom='Na'):
        forces = self.FORCE_PROJ[atom]
        hist = np.histogram(forces, bins=bins)
        fitfunc = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)
        errfunc = lambda p, x, y: (y-fitfunc(p, x))
        cent_guess = forces.mean()
        std_dev = forces.std()
        init = [1, cent_guess, std_dev]
        x= hist[1]
        x = x[:len(x)-1]
        y=hist[0]
        out = leastsq(errfunc, init, args=(x, y))
        self.fproj_hist = out
       
    def ifmean_time(self, atom='Na'):
        force_proj = self.FORCE_PROJ[atom]
        avg = 1/(len(force_proj)*self.timestep)*integrate.simps(force_proj, dx=self.timestep)
        self.fmean_time = avg
    
    def iPFOR(self, updated_fac=False):
        if updated_fac == True:
            raise ValueError("Updated simulations output .FA, .FAC files. Use those instead.")
        else:
            file_path = os.path.join(self.path, 'PYTHONOUT.FOR')
            cols = ['AINDEX', 'FX', 'FY', 'FZ', 'STEP', 'ATOM']
            df = pd.read_table(file_path, delim_whitespace=True, header=None)
            df.columns = cols
        self.PFOR = df
    
    def iVoronoiEntropy(self, npts=1000, start=1000, near=False):
        lims = (self.box_len, self.box_len, self.box_len)
        dstep = (self.ANI.STEP.max()-start)//npts
        step = np.arange(start, self.ANI.STEP.max(), dstep)
        if near == False:
            ents = []
            for s in step:
                print(s)
                pos = self.ANI.loc[self.ANI.STEP == s, ['X','Y','Z']].values
                vor_cont = tess.Container(pos, limits=lims, periodic=True)
                faces = np.array([i.number_of_faces() for i in vor_cont])
                fcount = Counter(faces)
                ftotal = sum([fcount[k] for k in fcount.keys()])
                vor_ent = -sum([(fcount[k]/ftotal)*np.log(fcount[k]/ftotal) for k in fcount.keys()])
                ents.append(vor_ent)
            self.VoronoiEntropy =  [ents, np.std(ents)]
        if near == True:
            ents = {}
            ranges = np.arange(3, 6.1, 0.5)
            for r in ranges:
                tents = []
                for s in step:
                    print(r, s)
                    pos = self.ANI.loc[(self.ANI.STEP == s), ['X','Y','Z']].values
                    nav = pos[0]
                    clv = pos[1]
                    posis = np.where((np.linalg.norm(pos-nav, axis=1) < r) | (np.linalg.norm(pos-clv, axis=1) < r))
                    pos = pos[posis]
                    vor_cont = tess.Container(pos, limits=lims, periodic=True)
                    faces = np.array([i.number_of_faces() for i in vor_cont])
                    fcount = Counter(faces)
                    ftotal = sum([fcount[k] for k in fcount.keys()])
                    vor_ent = -sum([(fcount[k]/ftotal)*np.log(fcount[k]/ftotal) for k in fcount.keys()])
                    tents.append(vor_ent)
                ents[r] = [tents, np.std(tents)]
            self.VoronoiEntropy = [ents]
                
            
        
    def iVoronoiVolumes(self, drop_dict, ind_dict, npts=100, start=1000, near=False):
        lims = (self.box_len, self.box_len, self.box_len)
        step = int((self.ANI.STEP.max()-start)/npts)
        steps = [start+i*step for i in range(0, npts)]
        vollst = []
        vol_dict = {k:[] for k in drop_dict.keys()}
        if near == True:
            ranges = np.arange(3, 6.1, 0.5)
            vol_dict['near'] = {str(i):[] for i in ranges}
            vol_dict['inds'] = {str(i):[] for i in ranges}
        for s in steps:
            print(s)
            prevind = 0
            pos = self.ANI.loc[self.ANI.STEP == s, ['X','Y','Z']].values
            vor_cont = tess.Container(pos, limits=lims, periodic=True)
            vollst.append([i.volume() for i in vor_cont])
            if near == True: 
                pass
        self.VoronoiVolumes = vollst
        
    def iVoronoi(self, npts=1000, start=1000, save=True, load=True):
        if load == True:
            print('Loading saved .vor.pkl.')
            vorp = os.path.join(self.path, self.name+'.vor.pkl')
            with open(vorp, 'rb') as f:
                self.Voronoi = pickle.load(f)
        lims = self.box_len
        voldct = {}
        step = int((self.ANI.STEP.max()-start)/npts)
        steps = [start+i*step for i in range(0, npts)]
        posarr = self.ANI.loc[self.ANI.STEP.isin(steps), ['X','Y','Z']].values
        self.vorpos = posarr
        for i in range(npts):
            print(i)
            voldct[steps[i]] = tess.Container(posarr[i*self.natoms:(i+1)*self.natoms], limits=lims, periodic=True)
        if save == True:
            vorp = os.path.join(self.path, self.name+'.vor.pkl')
            with open(vorp, 'wb') as f:
                pickle.dump(voldct, f)
        self.Voronoi = voldct
            
    def iRDF(self, sel1, sel2, bins=100, rdfrange=None):
        tanip = os.path.join(self.path[0], self.sim_name+'.tANI')
        tanixyz = os.path.join(self.path[0], self.sim_name+'.tANI.xyz')
        assert os.path.exists(tanip), ".tANI does not exist for MDAnalysis reading."
        if os.path.exists(tanip) and not os.path.exists(tanixyz):
            os.system("ln -s {} {}".format(tanip, tanixyz))
        if self.RDF == None:
            self.RDF = {}
        if rdfrange == None:
            rdfrange = (0.0, self.box_len/2)
        rsel1 = self.Universe.select_atoms("name {}".format(sel1))
        rsel2 = self.Universe.select_atoms("name {}".format(sel2))
        rdf = MDrdf.InterRDF(rsel1, rsel2, nbins=bins, range=rdfrange, verbose=True)
        print('{}-{} running'.format(sel1, sel2))
        rdf.run()
        self.RDF[sel1+sel2] = rdf                
