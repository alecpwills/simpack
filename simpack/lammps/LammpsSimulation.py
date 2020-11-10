"""
LammpsSimulation.py
===================

A class for working with and analyzing LAMMPS simulations.
"""

import sys, os, subprocess
from .. import classes
from .. import external
from .. import analysis
from tqdm import tqdm
import numpy as np
import pickle
import MDAnalysis as MD

class LammpsSimulation(classes.Simulation):
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
        self.thermo = {}
        self.logthermo = {}
        self.traj = {}
        self.antrav = {}
        
    def itrajectory(self, subdirs, topology, trajout):
        traj = {sub:[] for sub in subdirs}
        w = os.walk(self.path)
        print('Initializing colvar sampling from file: {} ({})'.format(trajout, self.path))
        for path, dirs, files in tqdm(w, total=self.walklen, file=sys.stdout):
            if trajout in files:
                fpath = os.path.join(path, trajout)
                
            
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
        
    def colvar2d_bin(self, subdirsx, subdirsy, outfile, nbinx=50, nbiny=50, start=10000):
        #document. this assumes the y colvar is outer directory, inner directory is x colvar
        histsy = {suby:[] for suby in subdirsy}
        edgesy = {suby:[] for suby in subdirsy}
        cumulativey = {suby:[] for suby in subdirsy}
        chistsy = {suby:[] for suby in subdirsy}
        cedgesy = {suby:[] for suby in subdirsy}
        histsx = {subx:[] for subx in subdirsx}
        edgesx = {subx:[] for subx in subdirsx}
        cumulativex = {subx:[] for subx in subdirsx}
        chistsx = {subx:[] for subx in subdirsx}
        cedgesx = {subx:[] for subx in subdirsx}
        cumulative = {suby:{} for suby in subdirsy}
        w = os.walk(self.path)
        print('Initializing colvar sampling from file: {} ({})'.format(outfile, self.path))
        for path, dirs, files in tqdm(w, total=self.walklen, file=sys.stdout):
            if outfile in files:
                csuby = path.split('/')[-2]
                csubx = path.split('/')[-1]
                if csuby not in subdirsy:
                    continue
                if csubx not in subdirsx:
                    continue
                fpath = os.path.join(path, outfile)
                arr = np.loadtxt(fpath, comments=['#','@'])
                arr = arr[arr[:, 0] > start]
                binsx = np.linspace(arr[:, 1].min(), arr[:, 1].max(), num=nbinx)
                hx, ex = np.histogram(arr[:, 1], binsx)
                binsy = np.linspace(arr[:, 2].min(), arr[:, 2].max(), num=nbiny)
                hy, ey = np.histogram(arr[:, 2], binsy)
                histsx[csubx].append(hx)
                edgesx[csubx].append(ex)
                cumulativex[csubx].append(arr[:, 1])
                histsy[csuby].append(hy)
                edgesy[csuby].append(ey)
                cumulativey[csuby].append(arr[:, 2])
                cumulative[csuby][csubx] = arr
        for suby in subdirsy:
            carry = np.concatenate(cumulativey[suby])
            binsy = np.linspace(carry.min(), carry.max(), num=nbiny)
            chy, cey = np.histogram(carry, binsy)
            chistsy[suby] = cey
            cedgesy[suby] = chy
        for subx in subdirsx:
            carrx = np.concatenate(cumulativex[subx])
            binsx = np.linspace(carrx.min(), carrx.max(), num=nbinx)
            chx, cex = np.histogram(carrx, binsx)
            chistsx[subx] = cex
            cedgesx[subx] = chx
            

        
        self.HEX = [histsx, edgesx]
        self.CHEX = [chistsx, cedgesx]
        self.HEY = [histsy, edgesy]
        self.CHEY = [chistsy, cedgesy]
        self.COLVAR2D = cumulative
    
            
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
        
    def run_wham2d(self, refsx=np.zeros(0), refsy=np.zeros(0), dirname='pywham', metafile='MET.WHAM2D', numbinsx=200,
                   numbinsy=200, reffactorx=1.0, reffactory=1.0, kspringx = 250, kspringy = 250, temp=300, tol=1e-8, histmin=-1,
                   histmax=-1, output='PMF2D', returnarr=True):
        dirpath = os.path.join(self.path, dirname)
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)
        cwd = os.getcwd()
        os.chdir(dirpath)
        met = os.path.join(dirpath, metafile)
        subprocess.check_output('touch {}'.format(met), shell=True)
        klisty = sorted(list(self.COLVAR2D.keys()))
        klistx = sorted(list(self.COLVAR2D[klisty[0]].keys()))
        if histmin == -1:
            histminy = min([self.COLVAR2D[klisty[0]][ikx][:, 2].min() for ikx in klistx])
            histminx = min([self.COLVAR2D[iky][klistx[0]][:, 1].min() for iky in klisty])
        if histmax == -1:
            histmaxy = max([self.COLVAR2D[klisty[-1]][ikx][:, 2].max() for ikx in klistx])
            histmaxx = max([self.COLVAR2D[iky][klistx[-1]][:, 1].max() for iky in klisty])
        with open(met, 'w') as file:    
            for iky in range(len(klisty)):
                for ikx in range(len(klistx)):
                    if refsx.shape[0] != 0:
                        krx = refsx[ikx]
                    if refsy.shape[0] != 0:
                        kry = refsy[iky]
                    else:
                        krx = klistx[ikx]
                        kry = klisty[iky]
                    tf = '{}_{}.wham.tmp'.format(krx, kry)
                    krefx = str(float(krx)*reffactorx)
                    krefy = str(float(kry)*reffactory)
                    np.savetxt(tf, self.COLVAR2D[kry][krx])
                    file.write(os.path.join(dirpath, tf)+' '+krefx+' '+krefy+' '+str(kspringx)+' '+str(kspringy)+' '+'\n')
                
        os.chdir(cwd)
        retarr = external.wham2d_command(dirpath, metafile, histminx, histmaxx, numbinsx,
                                         histminy, histmaxy, numbinsy, temp, tol, output=output)
        self.PMF = retarr
        
    def run_wham2d_to1d(self, refsx=np.zeros(0), refsy=np.zeros(0), dirname='pywham', metafile='MET.WHAM2D1D', numbins=200,
               reffactor=1.0, kspring = 250, temp=300, tol=1e-8, histmin=-1,
               histmax=-1, output='PMF2D1D', returnarr=True, reduce='y'):
        dirpath = os.path.join(self.path, dirname)
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)
        cwd = os.getcwd()
        os.chdir(dirpath)
        met = os.path.join(dirpath, metafile)
        subprocess.check_output('touch {}'.format(met), shell=True)
        klisty = sorted(list(self.COLVAR2D.keys()))
        klistx = sorted(list(self.COLVAR2D[klisty[0]].keys()))
        if histmin == -1:
            if reduce == 'x':
                histmin = min([self.COLVAR2D[klisty[0]][ikx][:, 2].min() for ikx in klistx])
            elif reduce == 'y':
                histmin = min([self.COLVAR2D[iky][klistx[0]][:, 1].min() for iky in klisty])
        if histmax == -1:
            if reduce == 'x':
                histmax = max([self.COLVAR2D[klisty[-1]][ikx][:, 2].max() for ikx in klistx])
            elif reduce == 'y':
                histmax = max([self.COLVAR2D[iky][klistx[-1]][:, 1].max() for iky in klisty])
        if reduce == 'x':
            refs = refsy
            klist = klisty
        elif reduce == 'y':
            refs = refsx
            klist = klistx
        with open(met, 'w') as file:
            for ik in range(len(klist)):
                if refs.shape[0] != 0:
                    kr = refs[ik]
                else:
                    kr = klist[ik]
                tf = '{}_{}.wham.tmp'.format(kr, reduce)
                kref = str(float(kr)*reffactor)
                if reduce == 'x':
                    np.savetxt(tf, np.concatenate([self.COLVAR2D[kr][ikx] for ikx in klistx])[:, [0, 2]])
                elif reduce == 'y':
                    np.savetxt(tf, np.concatenate([self.COLVAR2D[iky][kr] for iky in klisty])[:, [0, 1]])
                file.write(os.path.join(dirpath, tf)+' '+kref+' '+str(kspring)+'\n')
                
        os.chdir(cwd)
        retarr = external.wham_command(dirpath, metafile, histmin, histmax, numbins, temp, tol, output=output)
        self.PMF2D1D = retarr

    
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
        
    def ithermo(self, fname, subdirs, columns, subavg=False, seedavg=False, colvarfile=None,
             tdyn=False, keyname=None, refds=np.empty(0), temp=300, fprojRead=False, top=None, verbose=False,
             **kwargs):
        self.thermoc = columns
        if not keyname:
            dctkey = fname
        else:
            dctkey = keyname
        self.thermo[dctkey] = {sub:[] for sub in subdirs}
        w = os.walk(self.path)
        print('Initializing thermodynamic computes from file: {} ({})'.format(fname, self.path))
        for path, dirs, files in tqdm(w, total=self.walklen, file=sys.stdout):
            csub = path.split('/')[-1]
            seed = path.split('/')[-2]
            if csub not in subdirs:
                continue
            if fname in files:
                fpath = os.path.join(path, fname)
                if fprojRead:
                    print('fprojRead function')
                    ffile = fpath
                    top = os.path.join(path, 'system.data')
                    xfile = os.path.join(path, 'out.dcd')
                    try:
                        dt = kwargs['dt']
                    except KeyError:
                        print('dt not found in kwargs')
                        dt = 0.5
                    try:
                        a1 = kwargs['a1']
                    except KeyError:
                        print('a1 not found in kwargs')
                        a1=0
                    try:
                        a2 = kwargs['a2']
                    except KeyError:
                        print('a2 not found in kwargs')
                        a2=1
                        
                    arr = self.ithermoForceProjU(top, ffile, xfile, dt=dt, a1=a1, a2=a2)
                else:
                    arr = np.loadtxt(fpath, comments=['#','@'])
                if colvarfile:
                    cvp = os.path.join(path, colvarfile)
                    carr = np.loadtxt(cvp, comments=['#', '@'])
                    arr = np.insert(arr, arr.shape[1], np.nan, 1)
                    mask1 = np.in1d(arr[:, 0], carr[:, 0])
                    mask2 = np.in1d(carr[:, 0], arr[:, 0])
                    if verbose:
                        print("{}: shape={}, mask1 shape={}, mask2 shape={}".format(cvp, carr.shape, mask1.shape, mask2.shape))
                        print("mask1: steps from output in colvar = {}".format(mask1.sum()))
                        print("mask2: steps from colvar in output = {}".format(mask2.sum()))
                    try:
                        arr[mask1, -1] = carr[mask2, 1]
                    except ValueError:
                        #sometimes the first step in colvar when output.thermo begins doubles. make sure this is the case to continue
                        if carr[mask2, 0][0] == carr[mask2, 0][1]:
                            arr[mask1, -1] = carr[mask2, 1][1:]
                        else:
                            raise ValueError("Shape mismatch between thermo and colvar masks that isn't just repeat index.")
                self.thermo[dctkey][csub].append((seed, arr))
        if seedavg:
            for k in self.thermo[dctkey].keys():
                self.thermo[dctkey][k] = (np.mean(self.thermo[dctkey][k][1], axis=0), np.std(self.thermo[dctkey][k][1], axis=0))
        if subavg:
            means = np.mean([self.thermo[dctkey][sub][0] for sub in subdirs], axis=0)
            devs = np.std([self.thermo[dctkey][sub][0] for sub in subdirs], axis=0)
            self.thermo[dctkey] = (means, devs)
        if tdyn:
            kB = 0.0019872041 #kcal/mol/K
            inds = [np.argmin(abs(self.PMF[:,0]-d)) for d in refds]
            ens = self.PMF[inds, 1] - self.PMF[inds, 1][-1]
            beta = kB*temp
            boltzmann = np.exp(-beta*ens)
            Z = np.sum(boltzmann)
            klist = list(self.thermo[dctkey].keys())
            num = np.sum([boltzmann[i]*self.thermo[dctkey][klist[i]][0] for i in range(len(klist))], axis=0)
            numsd = np.sum([boltzmann[i]*self.thermo[dctkey][klist[i]][1] for i in range(len(klist))], axis=0)
            self.thermo[dctkey] = (num/Z, numsd/Z)
            
    def ilogthermo(self, fname, subdirs, columns, subavg=False,
             tdyn=False, refds=np.empty(0), temp=300):
        self.logthermoc = columns
        masklen = len(columns)
        self.logthermo[fname] = {sub:[] for sub in subdirs}
        w = os.walk(self.path)
        print('Initializing thermodynamic computes from file: {} ({})'.format(fname, self.path))
        for path, dirs, files in tqdm(w, total=self.walklen, file=sys.stdout):
            csub = path.split('/')[-1]
            if csub not in subdirs:
                continue
            if fname in files:
                fpath = os.path.join(path, fname)
                ls = []
                with open(fpath, 'r') as f:
                    for line in f:
                        if any(c.isalpha() for c in line):
                            continue
                        if len(line.split()) != masklen:
                            continue
                        ls.append([float(l) for l in line.split()])
                arr = np.array(ls)
                self.logthermo[fname][csub].append(arr)
        for k in self.logthermo[fname].keys():
            self.logthermo[fname][k] = (np.mean(self.logthermo[fname][k], axis=0), np.std(self.logthermo[fname][k], axis=0))
        if subavg:
            means = np.mean([self.logthermo[fname][sub][0] for sub in subdirs], axis=0)
            devs = np.std([self.logthermo[fname][sub][0] for sub in subdirs], axis=0)
            self.logthermo[fname] = (means, devs)
        if tdyn:
            kB = 0.0019872041 #kcal/mol/K
            inds = [np.argmin(abs(self.PMF[:,0]-d)) for d in refds]
            ens = self.PMF[inds, 1] - self.PMF[inds, 1][-1]
            beta = kB*temp
            boltzmann = np.exp(-beta*ens)
            Z = np.sum(boltzmann)
            klist = list(self.logthermo[fname].keys())
            num = np.sum([boltzmann[i]*self.logthermo[fname][klist[i]][0] for i in range(len(klist))], axis=0)
            numsd = np.sum([boltzmann[i]*self.logthermo[fname][klist[i]][1] for i in range(len(klist))], axis=0)
            self.logthermo[fname] = (num/Z, numsd/Z)
            
    def travis(self, trajname, keepdct, topol, subdirs, travinp, dirname='travis', dt=10, mintime=10000,
               water_list = ['O','H','H','M','DP'], nwater=96, overwrite = False, verbose=False):
        fends = [keepdct[k][1] for k in keepdct.keys()]
        dirpath = os.path.join(self.path, dirname)
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)
        cwd = os.getcwd()
        os.chdir(dirpath) #NOW IN TRAVIS DIRECTORY
        #Most of these things are hardcoded because of how the travis input is generated
        w = os.walk(self.path)
        print('Analyzing trajectories from files: {}, {}, {} ({})'.format(trajname, topol, travinp, self.path))
        for path, dirs, files in tqdm(w, total=self.walklen, file=sys.stdout):
            travf = os.listdir(dirpath)
            csub = path.split('/')[-1]
            if csub not in subdirs:
                continue
            ksub = path.split('/') [-2]
            check = '{}_{}'.format(ksub, csub) #check to see if this combo has already been analyzed and either abort or continue
            if any(check in f for f in travf) and any(x in f for f in travf for x in fends):
                if overwrite and verbose:
                    print("Found previous travis analysis with combo {}_{}, will overwrite".format(ksub,csub))
                else:
                    if verbose:
                        print("Found previous travis analysis with combo {}_{}, skipping this trajectory".format(ksub,csub))
                    continue
            if trajname in files:
                fpath = os.path.join(path, trajname)
                u = MD.Universe(topol, fpath, atom_style='id resid type charge x y z', dt=dt)
                # This is Na Cl regardless of Na Na simulations -- must be so to combine the two Na into a single molecule for further analysis
                u.add_TopologyAttr('name', ['Na', 'Cl']+water_list*nwater)                
                with MD.Writer('out.xyz', u.atoms.n_atoms) as W:
                    for ts in u.trajectory:
                        if ts.time > mintime:
                            W.write(u)
                #when done, analyze with travis
                subprocess.check_output("travis -p out.xyz -i {}".format(travinp), shell=True)
                genfs = os.listdir(os.getcwd())
                for f in genfs:
                    fp = os.path.join(dirpath, f)
                    fspl = f.split('_')
                    #we want to allow for keeping more than one file from analysis; i.e. cdf can give adf and rdf so don't want to waste them
                    #use information from keepdct: kd[analysis] = [fstart, fend, sels, mvname]
                    for k in keepdct.keys():
                        fstart, fend, sels, mvname = keepdct[k]
                        for sel in sels:
                            selmv = sel.strip('[ ]')
                            if (fstart in fspl[0]) and (fend in fspl[-1]) and (sel in f):
                                if verbose:
                                    print('MATCH: Analysis {} to {}'.format(k, f))
                                if any(x in f for x in subdirs): #if there is any distance in the filename, continue
                                    continue
                                else:
                                    # if ('[' in f) or (']' in f):
                                    #     f = re.sub(r'([\[ \]])', r'\\\1', f) #we have to SINGLE escape the brackets for shell command
                                    mf = mvname if mvname else f
                                    if verbose:
                                        print('renaming file: {} {}_{}_{}_{}'.format(f, ksub, csub, mf, selmv))
                                    mvp = os.path.join(dirpath, '{}_{}_{}_{}{}'.format(ksub,csub,mf,selmv,fend))
                                    os.rename(fp, mvp)
                    if os.path.exists(fp) and not any(x in f for x in subdirs):
                        #remove if the file still remains after being checked above and has no distances in the name
                        os.remove(fp)
        os.chdir(cwd)
    
    def traviscompress(self, fend, dirname='travis', remove = True):
        tpath = os.path.join(self.path, dirname)
        assert os.path.exists(tpath), "Must perform travis analysis first."
        fs = [i for i in os.listdir(tpath) if fend in i]
        for f in fs:
            fpath = os.path.join(tpath, f)
            if '.nparr' in f:
                continue
            arr = np.loadtxt(fpath, delimiter=';')
            with open(os.path.join(tpath, f+'.nparr'), 'wb') as wf:
                arr.tofile(wf)
            if remove:
                subprocess.check_output('rm -f {}'.format(fpath), shell=True)    
    
    def forcecompress(self, ffile, remove = True):
        w = os.walk(self.path)
        for path, dirs, files in tqdm(w, total=self.walklen, file=sys.stdout):
            if ffile in files:
                ff = os.path.join(path, ffile)
                arr = external.outputForceRead(ff).astype(np.float32)
                with open(ff+'.nparr', 'wb') as wf:
                    np.savez_compressed(wf, arr)
                if remove:
                    subprocess.check_output('rm -f {}'.format(ff), shell=True)
                
           
    def iantrav(self, subdirs, fend, sels=[''], dirname='travis'):
        self.antrav[fend] = {sub:{} for sub in subdirs}
        tpath = os.path.join(self.path, dirname)
        assert os.path.exists(tpath), "Must perform travis analysis first."
        files = os.listdir(tpath)
        sfiles = [i for i in files if fend in i]
        refs = list(set([i.split('_')[2] for i in sfiles]))
        obs = list(set([i.split('_')[-1].split('.')[0] for i in sfiles]))
        if 'nparr' in fend:
            RESHAPE = True
            readin = np.fromfile
            kwargs = {}
        else:
            RESHAPE = False
            readin = np.loadtxt
            kwargs = {'delimiter':';'}
        for sub in subdirs:
            for ref in refs:
                self.antrav[fend][sub][ref] = {}
                for ob in obs:
                    for sel in sels:
                        obfiles = [i for i in sfiles if ob in i and ref in i and sub in i and sel in i]
                        #mean currently only for .gp.csv
                        ma = sum([readin(os.path.join(tpath, fi), **kwargs) for fi in obfiles])/len(obfiles)
                        ms = (sum([(readin(os.path.join(tpath, fi), **kwargs)-ma)**2 for fi in obfiles])/(len(obfiles)-1))**(1/2)
                        if RESHAPE:
                            mas = int(len(ma)**(1/2))
                            # mss = int(len(ms)**(1/2))
                            ma = ma.reshape((mas,mas))
                            # ms = ms.reshape((mss,mss))
                            ms = 0 #issue with std right now
                        self.antrav[fend][sub][ref][ob+sel] = [ma, ms]
                        
    def ithermoForceProjU(top, ffile, xfile, dt=0.5, a1=0, a2=1):
        if os.path.exists(ffile):
            uf = np.load(ffile)['arr_0']
        else:
            print("{} not found".format(ffile))
            return None
        if os.path.exists(xfile):
            ux = MD.Universe(top, xfile, atom_style='id resid type charge x y z', dt=dt)
        else:
            print("{} not found".format(xfile))
            return None
        inds = np.arange(uf.shape[0])//int(uf[:, 0].max())
        uf = np.append(uf, inds[:, None], axis=1)
                         
        projlst = []
        for fts in tqdm(range(ux.trajectory.n_frames), file=sys.stdout):
            ux.trajectory[fts].forces = uf[uf[:, -1] == fts, 2:5]
            xts = ux.trajectory[fts]
            x = analysis.pbcwrap(xts.positions - xts.positions[a1], ux.dimensions[0])
            f1 = xts.forces[a1]
            f2 = xts.forces[a2]
            d12 = x[a1] - x[a2]
            d12h = d12/np.linalg.norm(d12)
            a1fp = np.dot(f1, d12h)
            a2fp = np.dot(f2, d12h)
            projlst.append([fts, np.linalg.norm(d12), a1fp, a2fp])
        
        retarr = np.array(projlst)
        return retarr