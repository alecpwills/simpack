import numpy as np
from scipy import integrate, interpolate
from scipy.optimize import leastsq
from scipy.signal import argrelextrema

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
    # eq 9 in my PMF paper:
    # dU/dr = -f(r) + 2kT/r
    dUdr = -(forces - 2*kb*temps/dists) if correct else -forces
    # if correct == True:
        # eq 9 in my PMF paper:
        # dU/dr = -f(r) + 2kT/r
        # dUdr = -(forces - 2*kb*temps/dists)
    if interp == 'pmf':
        pmf_ys = integrate.cumtrapz(y=dUdr, x=dists)
        pmf_int = interpolate.CubicSpline(x=dists[1:], y=pmf_ys)
        pmf_int_ys = pmf_int(xs)
    elif interp == 'force':
        fint = interpolate.CubicSpline(x=dists, y=dUdr)
#        pmf = integrate.cumtrapz(y=fint(xs), x=xs)
#        pmf_ys = pmf
#        pmf_int_ys = pmf
        pmfint = fint.antiderivative()
        pmf_ys = pmfint(dists)
        pmf_int_ys = pmfint(xs)
        
    return(xs, pmf_ys, pmf_int_ys)

def pmf_pm(forces, errors, ds, temp, correct=False, interp='force'):
    x, y, yi = pmf(forces, ds, temp, correct, interp)
    _, _, yip = pmf(forces+errors/2, ds, temp, correct, interp)
    _, _, yim = pmf(forces-errors/2, ds, temp, correct, interp)
    return (x,y,yi,yip,yim)
