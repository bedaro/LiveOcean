"""
A tool to explore property distributions in volumes.

"""

# imports
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import netCDF4 as nc
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.abspath('../alpha'))
import Lfun
import zrfun
import zfun

import tef_fun
import flux_fun

from time import time

# get command line arguments
import argparse
parser = argparse.ArgumentParser()
# standard arguments
parser.add_argument('-g', '--gridname', type=str, default='cas6')
parser.add_argument('-t', '--tag', type=str, default='v3')
parser.add_argument('-x', '--ex_name', type=str, default='lo8b')
parser.add_argument('-0', '--date_string0', type=str, default='2019.07.04')
parser.add_argument('-1', '--date_string1', type=str, default='2019.07.04')
parser.add_argument('-testing', type=zfun.boolean_string, default=False)
args = parser.parse_args()
testing = args.testing

# Get Ldir
Ldir = Lfun.Lstart(args.gridname, args.tag)
Ldir['gtagex'] = Ldir['gtag'] + '_' + args.ex_name
# get time limits
ds0 = args.date_string0; ds1 = args.date_string1
Ldir['date_string0'] = ds0; Ldir['date_string1'] = ds1
dt0 = datetime.strptime(ds0, '%Y.%m.%d'); dt1 = datetime.strptime(ds1, '%Y.%m.%d')
ndays = (dt1-dt0).days + 1
print('Working on:')
outname = Ldir['gtagex'] + '_' + ds0 + '_' + ds1
print(outname +'\n')

# get list of history files to process
fn_list = Lfun.get_fn_list('hourly', Ldir, ds0, ds1)
NT = len(fn_list)

# get grid info
fn = fn_list[0]
G = zrfun.get_basic_info(fn, only_G=True)
S = zrfun.get_basic_info(fn, only_S=True)
h = G['h']
DA = G['DX'] * G['DY']
DA3 = DA.reshape((1,G['M'],G['L']))
DXu = (G['DX'][:,1:]+G['DX'][:,:-1])/2
DX3u = DXu.reshape((1,G['M'],G['L']-1))
DYv = (G['DY'][1:,:]+G['DY'][:-1,:])/2
DY3v = DYv.reshape((1,G['M']-1,G['L']))

# set input/output location
indir0 = Ldir['LOo'] + 'tef2/'
voldir = indir0 + 'volumes_' + Ldir['gridname'] + '/'
#
outdir0 = indir0 + outname + '/'
Lfun.make_dir(outdir0)
outdir = outdir0 + 'flux/'
Lfun.make_dir(outdir)

# load DataFrame of volume and associated dicts
v_df = pd.read_pickle(voldir + 'volumes.p')
bathy_dict = pickle.load(open(voldir + 'bathy_dict.p', 'rb'))
ji_dict = pickle.load(open(voldir + 'ji_dict.p', 'rb'))

seg_list = list(v_df.index)



if testing:
    verbose = True
    seg_list = seg_list[-2:]
else:
    verbose = False

N = S['N']
j_dict = {}; i_dict = {}
nji = 0
nji_dict = {}
for seg_name in seg_list:
    jj = []; ii = []
    ji_list_full = ji_dict[seg_name]
    for ji in ji_list_full:
        jj.append(ji[0])
        ii.append(ji[1])
    jjj = np.array(jj)
    iii = np.array(ii)
    j_dict[seg_name] = jjj
    i_dict[seg_name] = iii
    nji += len(jjj) * N
    nji_dict[seg_name] = nji
    
# preallocate an array to hold the salinity and volume
sa = np.nan + np.ones(nji)
va = np.nan + np.ones(nji)

fn = fn_list[0]
    
tt0 = time()
        
print(fn)
    
ds = nc.Dataset(fn)
salt = ds['salt'][0,:,:,:]
zeta = ds['zeta'][0,:,:]
ds.close()
    
# find the volume and volume-mean salinity
ii0 = 0
for seg_name in seg_list:
    
    jjj = j_dict[seg_name]
    iii = i_dict[seg_name]
    z_r, z_w = zrfun.get_z(h[jjj,iii], zeta[jjj,iii], S)
    dz = np.diff(z_w, axis=0)
    DV = dz * DA3[0,jjj,iii]
    
    s = salt[:,jjj,iii]
    s = s.flatten()
    DV = DV.flatten()
    
    nji = nji_dict[seg_name]
    
    ii1 = ii0 + nji
    
    print('%s %d:%d' % (seg_name, ii0, ii1))
    
    sa[ii0:ii1] = s
    va[ii0:ii1] = DV
    
    ii0 = ii1

    print('  ** took %0.1f sec' % (time()-tt0))
    sys.stdout.flush()
    
plt.close('all')
fig = plt.figure(figsize=(14,8))
ax = fig.add_subplot(111)

ax.hist(sa, bins=np.linspace())

plt.show()

    
        
    
