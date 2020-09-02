import sys, os, subprocess
from .. import classes
from .. import external
from .. import analysis
import MDAnalysis as MD
from tqdm import tqdm
import panedr
import numpy as np
import pickle


class GromacsSimulation(classes.Simulation):
    def __init__(self, **kwargs):
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
        '''initialize the Universe with which we will analyze everything
        gro yields masses, which might be useful'''
        gro = os.path.join(self.path, self.sim_name+'.gro')
        traj = os.path.join(self.path, self.sim_name+'.'+self.traj_end)
        self.Universe = MD.Universe(gro, traj)
    
        
    def iEDR(self):
        ''' generate the dataframe containing the .edr file information associated
        with self.path'''
        edr_path = os.path.join(self.path, self.sim_name+'.edr')
        self.EDR = panedr.edr_to_df(edr_path)
                
    def colvar_bin(self, subdirs, outfile, nbin=50, start=10000):
        hists = {sub:[] for sub in subdirs}
        edges = {sub:[] for sub in subdirs}
        cumulative = {sub:[] for sub in subdirs}
        chists = {sub:[] for sub in subdirs}
        cedges = {sub:[] for sub in subdirs}
        w = os.walk(self.path)
        print('Initializing colvar sampling from file: {} ({})'.format(outfile, self.path))
        for path, dirs, files in tqdm(w, total=self.walklen, file=sys.stdout):
            if outfile in files:
                fpath = os.path.join(path, outfile)
                arr = np.loadtxt(fpath, comments=['#','@'])
                arr = arr[arr[:, 0] > start]
                bins = np.linspace(arr[:, 1].min(), arr[:, 1].max(), num=nbin)
                h, e = np.histogram(arr[:, 1], bins)
                csub = path.split('/')[-1]
                hists[csub].append(h)
                edges[csub].append(e)
                cumulative[csub].append(arr[:, 1])
        for sub in subdirs:
            carr = np.concatenate(cumulative[sub])
            cumulative[sub] = carr
            bins = np.linspace(carr.min(), carr.max(), num=nbin)
            ch, ce = np.histogram(carr, bins)
            chists[sub] = ce
            cedges[sub] = ch
        
        self.HE = [hists, edges]
        self.CHE = [chists, cedges]
        self.COLVAR = cumulative
    
            
    def run_wham(self, refs=np.zeros(0), dirname='pywham', metafile='MET.WHAM', numbins=200,
                 reffactor = 1.0, kspring = 250, temp = 300, tol = 1e-8, histmin = -1,
                 histmax = -1, output = 'PMF', returnarr = True):
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
