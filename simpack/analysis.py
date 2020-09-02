import numpy as np
from scipy import integrate

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

def smooth(y, box_pts):
    #https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    y_smooth[:box_pts] = y[:box_pts]
    return y_smooth
