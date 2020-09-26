"""
GromacsSimulation.py
====================

A class for working with and analyzing GROMACS simulations.
"""

import sys, os, subprocess
from .. import classes
from .. import external
from .. import analysis
import MDAnalysis as MD
from tqdm import tqdm
import pandas as pd
import panedr
import numpy as np


class GromacsSimulation(classes.Simulation):
    def __init__(self, **kwargs):
        """
        
        Parameters
        ----------
        **kwargs : 
            path : str, REQUIRED
                the top directory containing all subdirectories which contain relevant simulation data.
            box_len : float, REQUIRED
                the length of one side of the (presumed cubic) box.
            random_seed : bool, Optional, default : False
                If True, this means that some parameter of the simulation was determined with random seeds.
                NOTE: THIS ASSUMES THE DIRECTORY STRUCTURE IS path/random_seed_value/data...
                Directories with 'py' in the name are ignored in the search.

        Raises
        ------
        KeyError
            If either of the required keywords are not specified, then a KeyError is raised.

        Returns
        -------
        None.
            No value is returned, but implemented __attr__ values are defaulted to empty dicts or specific values.
            
        IMPLEMENTED :
            path, box_len :
                the inputs for the required keys
            gxy, edr, traj, antrav : 
                {}
            walklen : 
                the total length of a single loop over os.walk(self.path)
            random_seed : 
                default : False

        """
        req_keys = ['path', 'box_len']
        for k in req_keys:
            if k not in kwargs:
                raise KeyError('{} not in required arguments.'.format(k))
        self.path = kwargs['path']
        self.box_len = kwargs['box_len']
        
        #subdirectory tree length
        w = os.walk(self.path)
        l = 0
        for path, dirs, files in w:
            l+=1
        self.walklen = l
        
        if 'random_seed' in kwargs:
            self.random_seed = True
            self.seeds = [p for p in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, p)) and 'py' not in p]
        else:
            self.random_seed = False
        
        #implemented:
        self.gxy = {}
        self.edr = {}
        self.traj = {}
        self.antrav = {}
        
        
    def itrajectory(self):
        """
        

        Raises
        ------
        NotImplementedError
            This is not yet implemented. There isn't a way to serialize and save an MDAnalysis.Universe object,
            which is how a lot of future analysis will be done.

        Returns
        -------
        None.

        """
        raise NotImplementedError("GromacsSimulation.iEDR() is not yet implemented.")
        gro = os.path.join(self.path, self.sim_name+'.gro')
        traj = os.path.join(self.path, self.sim_name+'.'+self.traj_end)
        self.Universe = MD.Universe(gro, traj)
    
        
    def iEDR(self, edrfile, egrps=False):
        """
        

        Raises
        ------
        NotImplementedError
            This is not yet implemented. It requires Pandas as written, which I would like to avoid.

        Returns
        -------
        None.

        """
        w = os.walk(self.path)
        print('Initializing edr from file: {} ({})'.format(edrfile, self.path))
        dfs = []
        for path, dirs, files in tqdm(w, total=self.walklen, file=sys.stdout):
            if edrfile in files:
                fpath = os.path.join(path, edrfile)
                df = panedr.edr_to_df(fpath)
                csub = path.split('/')[-1]
                sub = path.split('/')[-2]
                try: #try to turn into float
                    df['SUB'] = float(sub)
                except ValueError:
                    #if sub is not a float, then there are no constraint directories
                    print('No constraint directories.')
                df['CSUB'] = float(csub)
                dfs.append(df)
            else:
                continue

        edrdf = pd.concat(dfs)
        edrarr = edrdf.values
        edrcols = edrdf.columns
        del edrdf
        if egrps:
            self.EGRPS = edrarr
            self.EGRPSC = edrcols
        else:
            self.EDR = edrarr
            self.EDRC = edrcols
                
    def colvar_bin(self, subdirs, outfile, nbin=50, start=10000, verbose=False):
        """
        

        Parameters
        ----------
        subdirs : str
            Subdirectory names for the simulations, if there are any. Current implementation basically assumes
            self.random_seed and the existence of subdirectories in each, not tested elsewise.
        outfile : str
            The name of the file containing the colvar outputs.
            In the case of GROMACS, a pullx file that contains pull distances between restrained atoms.
            As implemented, assumes only two pull atoms so that the columns used are in correct correspondence.
        nbin : int, optional
            The number of bins to use for each subdirectory colvar binning. The default is 50.
        start : int, optional
            The timestep at which to start the filtering of colvar samples. The default is 10000.
            NOTE: Default here is based on a LAMMPS-like simulation regime, where I do not do restarts
            as LAMMPS trajectories are readily continued within a single script. For GROMACS, using checkpoint
            files between equilibration phase and production phase, if only the colvar files for the production runs are
            being used then this should be -1.
        verbose : bool, optional
            Whether or not to print the paths/files found as the walk happens.

        Returns
        -------
        None.
        
        Sets:
            self.HE : list
                The [histogram_values, histogram_edges] for a given seed/sub_dir
            self.CHE : list
                The [c_histogram_values, c_histogram_edges] (cumulative across random_seeds) for a given sub_dir
            self.COLVAR : dict
                The cumulative array containing all pull-coordinate values for each sub_dir across random_seeds

        """
        hists = {sub:[] for sub in subdirs}
        edges = {sub:[] for sub in subdirs}
        cumulative = {sub:[] for sub in subdirs}
        chists = {sub:[] for sub in subdirs}
        cedges = {sub:[] for sub in subdirs}
        w = os.walk(self.path)
        print('Initializing colvar sampling from file: {} ({})'.format(outfile, self.path))
        for path, dirs, files in tqdm(w, total=self.walklen, file=sys.stdout):
            if outfile in files:
                if verbose:
                    print('{} found in {}'.format(outfile, path))
                fpath = os.path.join(path, outfile)
                arr = np.loadtxt(fpath, comments=['#','@'])
                arr = arr[arr[:, 0] > start]
                bins = np.linspace(arr[:, 1].min(), arr[:, 1].max(), num=nbin)
                h, e = np.histogram(arr[:, 1], bins)
                csub = path.split('/')[-1]
                if verbose:
                    print("CSUB: {}".format(csub))
                hists[csub].append(h)
                edges[csub].append(e)
                cumulative[csub].append(arr[:, 1])
                if verbose:
                    print('C[CSUB] LEN: {}'.format(len(cumulative[csub])))
                    print('C[CSUB] ARR.SHAPE: {}'.format(arr[:, 1].shape))
        for sub in subdirs:
            try:
                carr = np.concatenate(cumulative[sub])
                cumulative[sub] = carr
                bins = np.linspace(carr.min(), carr.max(), num=nbin)
                ch, ce = np.histogram(carr, bins)
                chists[sub] = ce
                cedges[sub] = ch
            except ValueError:
                print("No arrays found for {}. Check simulation results.".format(sub))
            
        self.HE = [hists, edges]
        self.CHE = [chists, cedges]
        self.COLVAR = cumulative
    
            
    def run_wham(self, refs=np.zeros(0), dirname='pywham', metafile='MET.WHAM', numbins=200,
                 reffactor = 1.0, kspring = 250, temp = 300, tol = 1e-8, histmin = -1,
                 histmax = -1, output = 'PMF', returnarr = True):
        """
        

        Parameters
        ----------
        refs : list (of floats), optional
            A list of floats for which the subdirectories of the umbrella sampled simulation is referenced to. The default is np.zeros(0).
        dirname : str, optional
            A directory created to store the metadata files and the resulting WHAM calculation. The default is 'pywham'.
        metafile : str, optional
            The name that the WHAM metadata file is called. The default is 'MET.WHAM'.
        numbins : int, optional
            The number of bins each subdirectory value's collective variable is binned into. The default is 200.
        reffactor : float, optional
            A multiplicative factor that converts the string subdirectory name into the reference minima value, if needed. The default is 1.0.
        kspring : float, optional
            The spring constant restraining the atoms (in units of kcal/mol/Angstrom). The default is 250.
        temp : float, optional
            The temperature at which the WHAM calculation should be made. The default is 300.
        tol : float, optional
            The tolerance to which the iterative WHAM procedure should be carried out. The default is 1e-8.
        histmin : float, optional
            The minimum value at which the WHAM calculation sets as the domain of calculation. The default is -1.
            If default, the minimum is set to the minimum value of the reference coordinate in the set of reference windows.
        histmax : float, optional
            The maximum value at which the WHAM calculation sets as the domain of calculation. The default is -1.
            If default, the maximum is set to the maximum value of the reference coodinate in the set of reference windows.
        output : str, optional
            The filename for the output of the WHAM calculation. The default is 'PMF'.
        returnarr : bool, optional
            Whether or not to immediately assign self.PMF as the output of the WHAM calculation. The default is True.

        Returns
        -------
        None.

        """
        dirpath = os.path.join(self.path, dirname)
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)
        cwd = os.getcwd()
        os.chdir(dirpath)
        met = os.path.join(dirpath, metafile)
        subprocess.check_output('touch {}'.format(met), shell=True)
        klist = list(self.COLVAR.keys())
        if histmin == -1:
            histmin = self.COLVAR[klist[0]].min()
        if histmax == -1:
            histmax = self.COLVAR[klist[-1]].max()
        with open(met, 'w') as file:    
            for ik in range(len(klist)):
                if refs.shape[0] != 0:
                    kr = refs[ik]
                    k = klist[ik]
                else:
                    kr = klist[ik]
                    k = klist[ik]
                tf = '{}.wham.tmp'.format(kr)
                kref = str(float(kr)*reffactor)
                zero = np.zeros_like(self.COLVAR[k])
                np.savetxt(tf, np.vstack((zero, self.COLVAR[k])).T)
                file.write(os.path.join(dirpath, tf)+' '+kref+' '+str(kspring)+'\n')
                
        os.chdir(cwd)
        retarr = external.wham_command(dirpath, metafile, histmin, histmax, numbins, temp, tol)
        self.PMF = retarr
    
    def igxy(self, fname, subdirs, subavg=False,
             tdyn=False, refds=np.empty(0), temp=300):
        """
        

        Parameters
        ----------
        fname : str
            The given output filename for a certain radial distribution function between atoms.
        subdirs : list
            The list of subdirectory names by which to create the dictionary of gxy[fname][subdir] RDF arrays.
        subavg : bool, optional
            Whether or not to average subdirectories in the dictionary along the random_seed values. The default is False.
        tdyn : bool, optional
            Whether or not to thermodynamically average the subdirectories across the values of the subdirectories. The default is False.
            If True, then it requires a self.PMF value to use for the energy referencing, and sets the zero to the last value of the self.PMF array.
        refds : list, optional
            A list of umbrella sampling reference windows used to determing the indices to use for thermodynamically averaging. The default is np.empty(0).
        temp : float, optional
            The temperature used (in Kelvin) of the simulation in the beta-factor when thermodynamically averaging. The default is 300.

        Returns
        -------
        None. Sets self.gxy[fname(+.tdyn if tdyn == True)]

        """
        self.gxy[fname] = {sub:[] for sub in subdirs}
        w = os.walk(self.path)
        print('Initializing distributions from file: {} ({})'.format(fname, self.path))
        for path, dirs, files in tqdm(w, total=self.walklen, file=sys.stdout):
            csub = path.split('/')[-1]
            if csub not in subdirs:
                continue
            if fname in files:
                fpath = os.path.join(path, fname)
                with open(fpath) as f: #skip the comments manually so that the next row can be skipped
                    lines = (line for line in f if not line.startswith('#'))
                    arr = np.loadtxt(lines, skiprows=1)
                # arr = np.loadtxt(fpath, comments=['#','@'], skiprows=1)
                self.gxy[fname][csub].append(arr)
        for k in self.gxy[fname].keys():
            self.gxy[fname][k] = (np.mean(self.gxy[fname][k], axis=0), np.std(self.gxy[fname][k], axis=0))
        if subavg:
            means = np.mean([self.gxy[fname][sub][0] for sub in subdirs], axis=0)
            devs = np.std([self.gxy[fname][sub][0] for sub in subdirs], axis=0)
            self.gxy[fname] = (means, devs)
        if tdyn:
            kB = 0.0019872041 #kcal/mol/K
            inds = [np.argmin(abs(self.PMF[:,0]-d)) for d in refds]
            ens = self.PMF[inds, 1] - self.PMF[inds, 1][-1]
            beta = kB*temp
            boltzmann = np.exp(-beta*ens)
            Z = np.sum(boltzmann)
            klist = list(self.gxy[fname].keys())
            num = np.sum([boltzmann[i]*self.gxy[fname][klist[i]][0] for i in range(len(klist))], axis=0)
            numsd = np.sum([boltzmann[i]*self.gxy[fname][klist[i]][1] for i in range(len(klist))], axis=0)
            self.gxy[fname+'.tdyn'] = (num/Z, numsd/Z)
