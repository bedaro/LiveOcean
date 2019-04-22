"""
Extract and save extracted fields from a sequence of hycom past days.

"""

# setup
import os
import sys
pth = os.path.abspath('../../alpha')
if pth not in sys.path:
    sys.path.append(pth)
import Lfun
Ldir = Lfun.Lstart()

import pickle
import netCDF4 as nc
from datetime import datetime, timedelta

import hfun
from importlib import reload
reload(hfun)

# optional command line input
def boolean_string(s):
    if s not in ['False', 'True']:
        raise ValueError('Not a valid boolean string')
    return s == 'True' # note use of ==
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-force', '--force_overwrite', nargs='?', default=False, type=boolean_string)
parser.add_argument('-test', '--testing', nargs='?', default=False, type=boolean_string)
args = parser.parse_args()
force_overwrite = args.force_overwrite
testing = args.testing

# initial experiment list
h_list = list(hfun.hy_dict.keys())
h_list.sort()

if testing:
    h_list = h_list[-2:]
    
if testing:
    var_list = 'surf_el'
else:
    var_list = 'surf_el,water_temp,salinity,water_u,water_v'

# specify output directory
out_dir0 = Ldir['data'] + 'hycom2/'
dt_dir = out_dir0 + 'dt_lists/'

f = open(out_dir0 + 'log.txt', 'w+')

# loop over all days in all lists
for hy in h_list:
    print('\n** Working on ' + hy + ' **')
    f.write('\n\n** Working on ' + hy + ' **')
    
    out_dir = out_dir0 + hy + '/'
    Lfun.make_dir(out_dir)
    
    dt_list = pickle.load(open(dt_dir + hy + '.p', 'rb'))
    
    if testing:
        dt_list = dt_list[:2]
        
    for dt in dt_list:
        print(' - ' + datetime.strftime(dt, '%Y.%m.%d'))
        
        out_fn = out_dir + 'h' + datetime.strftime(dt, '%Y.%m.%d') + '.nc'
        
        if os.path.isfile(out_fn):
            if force_overwrite:
                os.remove(out_fn)
                
        if not os.path.isfile(out_fn):
            result = hfun.get_extraction(hy, dt, out_fn, var_list)
            f.write('\n ' + datetime.strftime(dt, '%Y.%m.%d') + ' ' + result)
            
f.close()