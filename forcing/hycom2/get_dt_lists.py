"""
Code to get, or update, the lists of daily datetimes that exist for
the various HYCOM experiments we use.

Run with the -a True flag to get all lists, otherwise the default is
to only update the last one in hfun.hy_dict.

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
parser.add_argument('-a', '--do_all', nargs='?', default=False, type=boolean_string)
args = parser.parse_args()
do_all = args.do_all

# initial experiment list
h_list = list(hfun.hy_dict.keys())
h_list.sort()

# specify output directory
out_dir0 = Ldir['data'] + 'hycom2/'
Lfun.make_dir(out_dir0)
out_dir = out_dir0 + 'dt_lists/'

if do_all == False:
    # just update the last experiment
    h_list = [h_list[-1]]
    Lfun.make_dir(out_dir)
elif do_all == True:
    # get info for all experiments
    Lfun.make_dir(out_dir, clean=True)

def get_dt_list(ds):
    # function to make a list of daily datetime_values that exist in an experiment
    #
    # get the time in datetime format
    t_hycom = ds.variables['time'][:].squeeze()    
    # tu = ds.variables['time'].units
    # print(' time units = ' + tu)
    # should be 'hours since 2000-01-01 00:00:00'
    t_origin = ds.variables['time'].time_origin
    dt00 = datetime.strptime(t_origin, '%Y-%m-%d %H:%M:%S')    
    # check the time reference
    if dt00 != datetime(2000,1,1,0,0,0):
        print('Warning: Unexpected time reference!')
        sys.stdout.flush()            
    dt_list = [] 
    for tt in t_hycom:
        if tt/24 != int(tt/24):
            pass
        else:
            dt_list.append(dt00 + timedelta(days=tt/24))
    return dt_list

# loop over all experiments
for hy in h_list:
    f = open(out_dir + 'dt_info_' + hy + '.txt', 'w+')
    glb = hfun.hy_dict[hy][0]
    exnum = hfun.hy_dict[hy][1]
    print('glb = ' + glb + ', exnum = ' + exnum)
    f.write('glb = ' + glb + ', exnum = ' + exnum + '\n')
    fn = 'http://tds.hycom.org/thredds/dodsC/' + glb + '/expt_' + exnum
    ds = nc.Dataset(fn)
    dt_list = get_dt_list(ds)
    f.write('%s to %s\n' % (datetime.strftime(dt_list[0],'%Y.%m.%d'),
            datetime.strftime(dt_list[-1],'%Y.%m.%d')))
    ndays = (dt_list[-1] - dt_list[0]).days
    nfiles = len(dt_list)
    f.write('missing %d out of %d days' % (ndays-nfiles, ndays))
    f.close()
    ds.close()
    
    # save results
    out_fn = out_dir + hy + '.p'
    pickle.dump(dt_list, open(out_fn, 'wb'))
