"""
(simpack.)external.py

Calls to external scripts that rely on outside programs,
through the use of subprocess,
as well as more "exotic" file read-ins not self-contained in the class call
"""

import os, sys, subprocess

def pmf_read(rootdir, pmffile):
    arr = np.loadtxt(os.path.join(rootdir, pmffile), comments=['#', '@'])
    return arr


def wham_command(rootdir, metafile, histmin=2, histmax=6,
             numbins=41, temp=300, tol=1e-8, output='PMF', returnarr=True):
    cwd = os.getcwd()
    os.chdir(rootdir)
    inp = os.path.join(rootdir, metafile)
    out = os.path.join(rootdir, output)
    execstr = "wham {} {} {} {} {} 0 {} {}".format(histmin, histmax, numbins, tol, temp, inp, out)
    print(execstr)
    os.chdir(cwd)
    try:
        cout = subprocess.check_output(execstr, shell=True)
    except subprocess.CalledProcessError as e:
        cout = e.output
        print(cout)
    if returnarr:
        return pmf_read(rootdir, output)

def outputForceRead(file):
    with open(file, 'r') as f:
        lines = (line for line in f if 'ITEM' not in line and len(line.split()) == 5)
        arr = np.genfromtxt(lines)
    return arr