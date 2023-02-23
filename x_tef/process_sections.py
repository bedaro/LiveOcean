"""
Process TEF extractions.

Currently only set up to do salt.

"""

# setup
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pickle

import os
import psutil
from multiprocessing import Pool
import sys
alp = os.path.abspath('../alpha')
if alp not in sys.path:
    sys.path.append(alp)
import Lfun
import zfun

Ldir = Lfun.Lstart()

indir0 = Ldir['LOo'] + 'tef/'
# choose the tef extraction to process
item = Lfun.choose_item(indir0)
indir0 = indir0 + item + '/'
indir = indir0 + 'extractions/'

sect_list_raw = os.listdir(indir)
sect_list_raw.sort()
sect_list = [item for item in sect_list_raw if ('.nc' in item)]
print(20*'=' + ' Extracted Sections ' + 20*'=')
print(*sect_list, sep=", ")
print(61*'=')
# select which sections to process
my_choice = input('-- Input section to process (e.g. sog5, or Return to process all): ')
if len(my_choice)==0:
    # full list
    pass
else: # single item
    if (my_choice + '.nc') in sect_list:
        sect_list = [my_choice + '.nc']
    else:
        print('That section is not available')
        sys.exit()
    
outdir = indir0 + 'processed/'
Lfun.make_dir(outdir)

for tef_file in sect_list:
    print(tef_file)
    fn = indir + tef_file

    # name output file
    out_fn = outdir + tef_file.replace('.nc','.p')
    # get rid of the old version, if it exists
    try:
        os.remove(out_fn)
    except OSError:
        pass

    # load fields
    ds = nc.Dataset(fn)
    q = ds['q'][:]
    s = ds['salt'][:]
    ot = ds['ocean_time'][:]
    zeta = ds['zeta'][:]
    gtagex = ds.gtagex
    ds0 = ds.date_string0
    ds1 = ds.date_string1
    ds.close()

    # TEF sort into salinity bins
    qs = q*s
    NT, NZ, NX = q.shape
    # initialize intermediate results arrays for TEF quantities
    sedges = np.linspace(0, 36, 1001) # original was 1001 used 5001 for Willapa
    sbins = sedges[:-1] + np.diff(sedges)/2
    NS = len(sbins) # number of salinity bins

    g = 9.8
    rho = 1025

    def process_time(tt):

        si = s[tt,:,:].squeeze()
        if isinstance(si, np.ma.MaskedArray):
            sf = si[si.mask==False].data.flatten()
        else:
            sf = si.flatten()

        qi = q[tt,:,:].squeeze()
        if isinstance(qi, np.ma.MaskedArray):
            qf = qi[qi.mask==False].data.flatten()
        else:
            qf = qi.flatten()

        qsi = qs[tt,:,:].squeeze()
        if isinstance(qsi, np.ma.MaskedArray):
            qsf = qsi[si.mask==False].data.flatten()
        else:
            qsf = qsi.flatten()

        # sort into salinity bins
        inds = np.digitize(sf, sedges, right=True)
        indsf = inds.copy().flatten()
        us, ucs = np.unique(indsf, return_counts=True)
        ucs_max = ucs.max()

        counter = 0
        tef_q = np.zeros(NS)
        tef_qs = np.zeros(NS)
        for ii in indsf:
            tef_q[ii-1] += qf[counter]
            tef_qs[ii-1] += qsf[counter]
            counter += 1

        # also keep track of volume transport
        qnet = qf.sum()

        # and tidal energy flux
        zi = zeta[tt,:].squeeze()
        ff = zi.reshape((1,NX)) * qi
        fnet = g * rho * ff.sum()

        return tef_q, tef_qs, qnet, fnet, ucs_max

    tasks = min(len(os.sched_getaffinity(0)), psutil.cpu_count(logical=False))
    with Pool(tasks) as p:
        res = p.map(process_time, range(NT))
        tef_q = np.array([x[0] for x in res])
        assert tef_q.shape == (NT, NS)
        tef_qs = np.array([x[1] for x in res])
        qnet = np.array([x[2] for x in res])
        fnet = np.array([x[3] for x in res])
        ucs_max = np.max([x[4] for x in res])
    if ucs_max > 1:
        print(f'WARNING: salinity bins not small enough? {ucs_max} points in one bin')

    # save results
    tef_dict = dict()
    tef_dict['tef_q'] = tef_q
    tef_dict['tef_qs'] = tef_qs
    tef_dict['sbins'] = sbins
    tef_dict['ot'] = ot
    tef_dict['qnet'] = qnet
    tef_dict['fnet'] = fnet
    tef_dict['ssh'] = np.mean(zeta, axis=1)
    pickle.dump(tef_dict, open(out_fn, 'wb'))


