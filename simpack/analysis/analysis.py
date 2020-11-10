import numpy as np
from scipy import integrate, interpolate
from scipy.optimize import leastsq
from scipy.signal import argrelextrema


def weighted_hist(xs, ws, bins=500, countsel=100):
    sums, edges = np.histogram(xs, bins=bins, weights=ws)
    counts, _ = np.histogram(xs, bins=bins)
    inds = np.where(counts > countsel)[0]
    return (edges[inds], sums[inds]/counts[inds])


def kB_units(units):
    if units == 'kcal/mol':
        kb = 0.001985875 #kcal/mol/K
    elif units == 'eV':
        kb = 8.617333262145e-5 #eV/K
    elif units == 'kj/mol':
        kb = 0.008314463 #kj/mol/K
    return kb

# def ka_ke(dists, energies, temps, xsi, ysi, r_u=6.0, units='kcal/mol',
#           freeE = True):
#     #10.1063/1.470707
#     #eqs. 13, 19
#     #importantly, Ke (19) if equilibrium constant b/w states ~rho_ssip/rho_cip
#     kb = kB_units(units)
#     if freeE:
#         #DOI:10.1073/pnas.0610945104
#         #W(r) free energy = w(r) pmf -2 k T ln r        
#         #so if given the free energy (i.e. from WHAM/Meta),
#         #move entropic contribution to get actual PMF
#         pmfys = energies + 2*kb*temps*np.log(dists)
#     else:
#         pmfys = energies
#     gr = np.exp(-pmfys/(kb*temps))
#     gri = interpolate.CubicSpline(x=dists, y=gr)
#     gry = gri(xsi)

#     bs = pmf_barrier(xsi, ysi)
#     r_ts = bs[1][0]
#     r_ts_i = np.where(xsi==r_ts)[0][0]
#     print(r_ts, r_ts_i)
#     ka = 4*np.pi*integrate.simps(y=gry[r_ts_i:]*xsi[r_ts_i:]**2,
#                                  x=xsi[r_ts_i:])
#     kd = 4*np.pi*integrate.simps(y=gry[:r_ts_i]*xsi[:r_ts_i]**2,
#                                  x=xsi[:r_ts_i])
#     ke = ka/kd
#     return ka, ke

def boltzmann_average(x, ens, temps, units='eV'):
    x = np.array(x)
    ens = np.array(ens)
    temps = np.array(temps)
    kb = kB_units(units)
    beta = 1/(kb*temps)
    boltz = np.exp(-beta*ens)
    num = np.sum(x*boltz)
    den = np.sum(boltz)
    return num/den

def boltzmann_average_sd(xarrs, ens, temps, units='eV'):
    xmeans = [i.mean() for i in xarrs]
    xsds = [i.std() for i in xarrs]
    bamean = boltzmann_average(x=xmeans, ens=ens, temps=temps, units=units)
    basd = boltzmann_average(x=xsds, ens=ens, temps=temps, units=units)
    return (xmeans, xsds, bamean, basd)

def pmf2d_int_1d(pmf, axis, temperature, units='kcal/mol'):
    vs = np.unique(pmf[:, axis])
    ps = []
    kb = kB_units(units)
    beta = 1/(kb*temperature)
    for iv in vs:
        iy = pmf[pmf[:, axis] == iv]
        boltz = np.exp(-beta*iy[:, 2])
        nums = np.sum(iy[:, 2]*boltz)
        dens = np.sum(boltz)
        ps.append(nums/dens)
    
    ps = np.array(ps)
    return (vs, ps)


def gauss_hist_fit(binvals, nbins, cgreater = 0):
    counts, edges = np.histogram(binvals, bins=nbins)
    fitfunc = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)
    errfunc = lambda p, x, y: np.sqrt((y-fitfunc(p, x))**2)
    inds = np.where(counts > cgreater)[0]
    counts = counts[inds]
    edges = edges[inds]
    cent_guess = edges[np.where(counts == counts.max())[0]][0]
    std_dev = edges.std()
    init = [counts.max(), cent_guess, std_dev]
    out = leastsq(errfunc, init, args=(edges, counts))
    return out

def pbcwrap(arr, boxlen, dist=True):
    if dist:
        #centering on some atom, so distances can only be half a boxlength away
        bl2 = boxlen/2 #
    else:
        #wrap just all coordinates into box
        bl2 = boxlen
    while arr[arr < -bl2].sum() != 0:
        arr[arr < -bl2] = arr[arr < -bl2] + boxlen
    while arr[arr > bl2].sum() != 0:
        arr[arr > bl2] = arr[arr > bl2] - boxlen
    return arr

# def nneighbors(arr1, arr2, nneigh=2, pbc=True, boxlen=None):
#     inds = []
#     for i in range(len(arr1)):
#         v1 = arr1[i]
#         ds = v1 - arr2
#         if pbc:
#             ds = pbcwrap(ds, boxlen, dist=True)
#         tinds = []
#         for ni in range(nneigh):


def time_average(time, ys, cumulative=True):
    if cumulative == False: #? very different from expected
        dx = time[1]-time[0]
        avg = integrate.simps(ys, dx=dx)/(len(ys)*dx)
    else:
        avg = integrate.cumtrapz(y=ys, x=time)/time[1:]
    return avg

def hist_weighted_average(edges, counts):
    total = counts.sum()
    summed = (edges*counts).sum()
    weighted = summed/total
    return weighted

def cip_ssip_depth(pmfx, pmfy, verbose=True, error=None):
    #Find TS location first; if SSIP/CIP inverted, then using the min as CIP won't work without filtering region
    tbi = argrelextrema(pmfy, np.greater)[0][0]
    tbx = pmfx[tbi]
    tby = pmfy[tbi]
    #Find CIP from beginning to TS
    cipi = np.where(pmfy[:tbi] == pmfy[:tbi].min())[0][0]
    cipx = pmfx[cipi]
    cipy = pmfy[cipi]
    #Find SSIP from TS to end
    ssipi = argrelextrema(pmfy[tbi:], np.less)[0][0]+tbi
    ssipy = pmfy[ssipi]
    ssipx = pmfx[ssipi]
    tce = tby - cipy
    tse = tby - ssipy
    if verbose:
        outstr = """
        TRANSITION BARRIER INFORMATION
        CIP: ({}, {})
        TB: ({}, {})
        SSIP: ({}, {})
        TB-CIP: {}
        TB-SSIP: {}
        """.format(cipx, cipy, tbx, tby, ssipx, ssipy, tce, tse)
        print(outstr)
    return [(cipx, cipy), (tbx, tby), (ssipx, ssipy), tce, tse]

def cip_ssip_depth_error(pmfx, pmfy, pmfye, verbose=True, ssipi_offset=0):
    #Find TS location first; if SSIP/CIP inverted, then using the min as CIP won't work without filtering region
    tbi = argrelextrema(pmfy, np.greater)[0][0]
    tbx = pmfx[tbi]
    tby = pmfy[tbi]
    tbye = abs(pmfye[tbi]-tby)
    #Find CIP from beginning to TS
    cipi = np.where(pmfy[:tbi] == pmfy[:tbi].min())[0][0]
    cipx = pmfx[cipi]
    cipy = pmfy[cipi]
    cipye = abs(pmfye[cipi]-cipy)
    #Find SSIP from TS to end
    #Allow offsetting to move away from potential "hilly" transition region
    ssipi = argrelextrema(pmfy[tbi+ssipi_offset:], np.less)[0][0]+tbi+ssipi_offset
    ssipy = pmfy[ssipi]
    ssipx = pmfx[ssipi]
    ssipye = abs(pmfye[ssipi]-ssipy)
    tce = tby - cipy
    tcee = np.sqrt(tbye**2+cipye**2)
    tse = tby - ssipy
    tsee = np.sqrt(tbye**2+ssipye**2)
    if verbose:
        outstr = """
        TRANSITION BARRIER INFORMATION
        CIP: ({}, {} +- {})
        TB: ({}, {} +- {})
        SSIP: ({}, {} +- {})
        TB-CIP: {} +- {}
        TB-SSIP: {} +- {}
        """.format(cipx, cipy, cipye, tbx, tby, tbye, ssipx, ssipy, ssipye, tce, tcee, tse, tsee)
        print(outstr)
    return [(cipx, cipy), (tbx, tby), (ssipx, ssipy), tce, tse]


def smooth(y, box_pts):
    #https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    y_smooth[:box_pts] = y[:box_pts]
    return y_smooth

def pmf(forces, dists, temps, correct=True, interp='pmf'):
    '''Integrates mean forces over the given distance to generate a PMF.
    forces: a list or array of the mean forces to be integrated (with energy units of eV/ang)
    dists: a list or array of the distances at which the mean forces were calculated
        list, np.array
    temps: a list or real of the temperature(s) used in the entropic correction
        list, np.array, float
    correct: a boolean determining whether or not the entropic force correction is carried out
        bool'''
    kb = 8.6173303e-5 #eV/K
    temps = np.array(temps)
    dists = np.array(dists)
    forces = np.array(forces)
    xs = np.linspace(dists[0], dists[-1], num=500)
    #because positive forces are repulsive, it should be the case that the first entry is ???
    #negative because the constraint force to overcome the large repulsion must cancel the positive
    #which sign to integrate? integrating the negative in dUdr yields wrong sign?
    if forces[0] < 0:
        forces = -forces
    dUdr = -forces
    if interp == 'pmf':
        pmf_ys = integrate.cumtrapz(y=dUdr, x=dists)
        if correct:
            #DOI:10.1073/pnas.0610945104
            #W(r) free energy = w(r) pmf -2 k T ln r        
            pmf_ys -= 2*kb*temps*np.log(dists)
        pmf_int = interpolate.CubicSpline(x=dists[1:], y=pmf_ys)
        pmf_int_ys = pmf_int(xs)
    elif interp == 'force':
        fint = interpolate.CubicSpline(x=dists, y=dUdr)
        pmfint = fint.antiderivative()
        pmf_ys = pmfint(dists)
        pmf_int_ys = pmfint(xs)
        if correct:
            #DOI:10.1073/pnas.0610945104
            #W(r) free energy = w(r) pmf -2 k T ln r        
            pmf_ys -= 2*kb*temps*np.log(dists)
            pmf_int_ys -= 2*kb*temps.mean()*np.log(xs)

    return(xs, pmf_ys, pmf_int_ys)

def pmf_pm(forces, errors, ds, temp, correct=False, interp='force'):
    x, y, yi = pmf(forces, ds, temp, correct, interp)
    _, _, yip = pmf(forces+errors/2, ds, temp, correct, interp)
    _, _, yim = pmf(forces-errors/2, ds, temp, correct, interp)
    return (x,y,yi,yip,yim)
