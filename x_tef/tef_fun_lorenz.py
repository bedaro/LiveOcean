# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:01:15 2018

@author: lorenz, edited by PM
"""

import numpy as np
import scipy.signal as signal

# Rewritten by Ben Roberts to fix some edge cases not handled
# well in the old version by lorenz and PM
def find_extrema(x, comp=5):
    """
    input
    x = Q(S)
    comp = size of the window as an integer number
    """

    indices = []
    minmax = []

    # Find smin and smax locations by examining deltas. The first and
    # last nonzero delta are the min and max salinity
    idx_smin, idx_smax = (x[:-1] - x[1:]).nonzero()[0][[0,-1]]
    idx_smax += 1
    # Use find_peaks to get the sufficiently prominent/wide max and min
    # values
    find_peak_args = {
        'width': comp,
        'prominence': (x.max() - x.min())/50
    }
    maxs, _ = signal.find_peaks(x, **find_peak_args)
    mins, _ = signal.find_peaks(-1*x, **find_peak_args)
    # Assemble indices and minmax, ensuring they interweave. It's possible
    # they won't because find_peaks's width and prominence checks may not
    # work on maxima and minima symmetrically for nonsymmetric signals. If
    # that happens, insert the max/min of the slice between the indices
    # where an extremum is missing
    indices = []
    minmax = []
    prev_is_min = None
    all_peaks_troughs = np.sort(np.concatenate((mins, maxs)))
    # Remove idx_smin and idx_smax if present; those get handled later
    all_peaks_troughs = np.delete(all_peaks_troughs,
            np.isin(all_peaks_troughs, [idx_smin, idx_smax]))
    for idx,prev in zip(all_peaks_troughs,
            [None] + all_peaks_troughs[:-1].tolist()):
        is_max = idx in maxs
        if prev is not None:
            prev_is_min = prev in mins
            if is_max != prev_is_min:
                pick_from = x[prev+1:idx]
                if is_max:
                    indices.append(prev + 1 + pick_from.argmin())
                    minmax.append('min')
                else:
                    indices.append(prev + 1 + pick_from.argmax())
                    minmax.append('max')
        indices.append(idx)
        minmax.append('max' if is_max else 'min')
    if len(indices) > 0:
        # Append smax and figure out if it's a min or max
        last_mm = 'min' if x[idx_smax] < x[indices[-1]] else 'max'
        if last_mm == minmax[-1]:
            pick_from = x[indices[-1]+1:idx_smax]
            if last_mm == 'max':
                indices.append(indices[-1] + 1 + pick_from.argmin())
                minmax.append('min')
            else:
                indices.append(indices[-1] + 1 + pick_from.argmax())
                minmax.append('max')
        minmax.append(last_mm)
        indices.append(idx_smax)
        # Finally, prepend smin and figure out if it's a min or max
        first_mm = 'min' if x[idx_smin] < x[indices[0]] else 'max'
        if first_mm == minmax[0]:
            pick_from = x[idx_smin+1:indices[0]]
            if first_mm == 'max':
                indices.insert(0, idx_smin + 1 + pick_from.argmin())
                minmax.insert(0, 'min')
            else:
                indices.insert(0, idx_smin + 1 + pick_from.argmax())
                minmax.insert(0, 'max')
        minmax.insert(0, first_mm)
        indices.insert(0, idx_smin)
    else:
        # Trivial case
        indices = [idx_smin, idx_smax]
        smin_is_min = x[idx_smin] < x[idx_smax]
        minmax = ['min','max'] if smin_is_min else ['max','min']

    return np.array(indices), np.array(minmax)

def calc_bulk_values(s, Qv, Qs, print_info=False):
    """
    input
    s=salinity array
    Qv=Q(S)
    Qs=Q^s(S)
    min_trans=minimum transport to consider
    """    
    # use the find_extrema algorithm
    ind, minmax = find_extrema(Qv)
    
    # compute dividing salinities
    smin=s[0]
    DS=s[1]-s[0]
    div_sal=[]
    i=0
    while i < len(ind): 
        div_sal.append(smin+DS*ind[i])
        i+=1
        
    #calculate transports etc.
    Q_in_m=[]
    Q_out_m=[]
    s_in_m=[]
    s_out_m=[]
    index=[]
    i=0
    while i < len(ind)-1:
        # compute the transports and sort to in and out
        Q_i=-(Qv[ind[i+1]]-Qv[ind[i]])
        F_i=-(Qs[ind[i+1]]-Qs[ind[i]])
        s_i=np.abs(F_i)/np.abs(Q_i)
        if Q_i<0 and np.abs(Q_i)>1:
            Q_out_m.append(Q_i)
            s_out_m.append(s_i)
        elif Q_i > 0 and np.abs(Q_i)>1:
            Q_in_m.append(Q_i)
            s_in_m.append(s_i)
        else:
            index.append(i)
        i+=1
    div_sal = np.delete(div_sal, index)
        
    return Q_in_m, Q_out_m, s_in_m, s_out_m, div_sal, ind, minmax
