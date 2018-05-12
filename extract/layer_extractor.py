#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This creates and a single NetCDF file containing fields from one or more
model layers, for some time range.

"""

from datetime import datetime, timedelta
start_time = datetime.now()
import netCDF4 as nc
import argparse

import os
import sys
pth = os.path.abspath('../alpha')
if pth not in sys.path:
    sys.path.append(pth)
import Lfun
import numpy as np
import zrfun
import zfun

# Command line arguments

# set defaults
gridname = 'cascadia1'
tag = 'base'
ex_name = 'lobio5'
list_type = 'hourly' # hourly, daily, low_passed
dsf = '%Y.%m.%d' # Example of date_string is 2015.09.19
date_string0 = datetime(2013,1,29).strftime(format=dsf)
date_string1 = datetime(2013,1,31).strftime(format=dsf)
#
nlay_str = '-1' # layer number, -1 for top, 0 for bottom

# optional command line arguments, can be input in any order, or omitted
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gridname', nargs='?', type=str, default=gridname)
parser.add_argument('-t', '--tag', nargs='?', type=str, default=tag)
parser.add_argument('-x', '--ex_name', nargs='?', type=str, default=ex_name)
parser.add_argument('-lt', '--list_type', nargs='?', const=list_type, type=str, default=list_type)
parser.add_argument('-0', '--date_string0', nargs='?', type=str, default=date_string0)
parser.add_argument('-1', '--date_string1', nargs='?', type=str, default=date_string1)
#
parser.add_argument('-nlay', '--layer_number', nargs='?', type=str, default=nlay_str)
args = parser.parse_args()

# process the arguments
list_type = args.list_type
nlay_str = args.layer_number
nlay = int(nlay_str)
Ldir = Lfun.Lstart(args.gridname, args.tag)
Ldir['ex_name'] = args.ex_name
Ldir['gtagex'] = Ldir['gtag'] + '_' + Ldir['ex_name']
dt0 = datetime.strptime(args.date_string0, dsf)
dt1 = datetime.strptime(args.date_string1, dsf)

# make sure the output directory exists
outdir = Ldir['LOo'] + 'extract/'
Lfun.make_dir(outdir)
# output file
out_name = 'layer_' + Ldir['gtagex'] + '_' + list_type + '.nc'
out_fn = outdir + out_name
# get rid of the old version, if it exists
try:
    os.remove(out_fn)
except OSError:
    pass

fn_list = []
dt = dt0
while dt <= dt1:
    date_string = dt.strftime(format='%Y.%m.%d')
    Ldir['date_string'] = date_string
    f_string = 'f' + Ldir['date_string']
    in_dir = Ldir['roms'] + 'output/' + Ldir['gtagex'] + '/' + f_string + '/'
    if (list_type == 'low_passed'):
        fn_list.append(in_dir + 'low_passed.nc') 
    elif (list_type == 'daily'):
        fn_list.append(in_dir + 'ocean_his_0001.nc')
    elif list_type == 'hourly':
        if dt == dt0:
            h0 = 1
        else:
            h0 = 2
        for hh in range(h0,26):
            hhhh = ('0000' + str(hh))[-4:]
            fn_list.append(in_dir + 'ocean_his_' + hhhh + '.nc')
    dt = dt + timedelta(days=1)
    
# make some things
fn = fn_list[0]
G = zrfun.get_basic_info(fn, only_G=True)
h = G['h']

ds1 = nc.Dataset(fn)
ds2 = nc.Dataset(out_fn, 'w')

# lists of variables to process
dlist = ['xi_rho', 'eta_rho', 'xi_psi', 'eta_psi', 'ocean_time']
vn_list_2d = [ 'lon_rho', 'lat_rho', 'lon_psi', 'lat_psi', 'mask_rho', 'h']
vn_list_2d_t = ['zeta']
vn_list_3d_t = ['salt', 'temp', 'NO3']
vn_list_2d_uv_t = ['sustr', 'svstr']
vn_list_3d_uv_t = ['u', 'v']

# Create dimensions
for dname, the_dim in ds1.dimensions.items():
    if dname in dlist:
        ds2.createDimension(dname, len(the_dim) if not the_dim.isunlimited() else None)
        
# Create variables and their attributes
# - first time
vn = 'ocean_time'
varin = ds1[vn]
vv = ds2.createVariable(vn, varin.dtype, varin.dimensions)
vv.long_name = varin.long_name
vv.units = varin.units
# - then static 2d fields
for vn in vn_list_2d:
    varin = ds1[vn]
    vv = ds2.createVariable(vn, varin.dtype, varin.dimensions)
    vv.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
    vv[:] = ds1[vn][:]
# - then time-dependent 2d fields
for vn in vn_list_2d_t:
    varin = ds1[vn]
    vv = ds2.createVariable(vn, varin.dtype, varin.dimensions)
    vv.long_name = varin.long_name
    vv.units = varin.units
    vv.time = varin.time
# - then time-dependent 3d fields
for vn in vn_list_3d_t:
    varin = ds1[vn]
    vv = ds2.createVariable(vn, varin.dtype, ('ocean_time', 'eta_rho', 'xi_rho'))
    vv.long_name = varin.long_name
    try:
        vv.units = varin.units
    except AttributeError:
        pass # salt has no units
    vv.time = varin.time
# - then time-dependent 2d fields interpolated from uv grids
for vn in vn_list_2d_uv_t:
    varin = ds1[vn]
    vv = ds2.createVariable(vn, varin.dtype, ('ocean_time', 'eta_rho', 'xi_rho'))
    vv.long_name = varin.long_name
    vv.units = varin.units
    vv.time = varin.time
# - then time-dependent 3d fields interpolated from uv grids
for vn in vn_list_3d_uv_t:
    varin = ds1[vn]
    vv = ds2.createVariable(vn, varin.dtype, ('ocean_time', 'eta_rho', 'xi_rho'))
    vv.long_name = varin.long_name
    vv.units = varin.units
    vv.time = varin.time
    
# Copy data
omat = np.nan * np.ones(h.shape)
omat = np.ma.masked_where(G['mask_rho']==0, omat)

tt = 0
NF = len(fn_list)
for fn in fn_list:
    if np.mod(tt,24)==0:
        print(' working on %d of %d' % (tt, NF))
        sys.stdout.flush()
    ds = nc.Dataset(fn)
    
    ds2['ocean_time'][tt] = ds['ocean_time'][0].squeeze()
    
    for vn in vn_list_2d_t:
        ds2[vn][tt,:,:] = ds[vn][0, :, :].squeeze()

    for vn in vn_list_3d_t:
        ds2[vn][tt,:,:] = ds[vn][0, nlay, :, :].squeeze()
        
    if ('sustr' in vn_list_2d_uv_t) and ('svstr' in vn_list_2d_uv_t):
        sustr0 = ds1['sustr'][0, :, :].squeeze()
        svstr0 = ds1['svstr'][0, :, :].squeeze()
        sustr = omat.copy()
        svstr = omat.copy()
        sustr[:, 1:-1] = (sustr0[:, 1:] + sustr0[:, :-1])/2
        svstr[1:-1, :] = (svstr0[1:, :] + svstr0[:-1, :])/2
        sustr = np.ma.masked_where(np.isnan(sustr), sustr)
        svstr = np.ma.masked_where(np.isnan(svstr), svstr)
        ds2['sustr'][tt,:,:] = sustr
        ds2['svstr'][tt,:,:] = svstr
        
    if ('u' in vn_list_3d_uv_t) and ('v' in vn_list_3d_uv_t):
        u0 = ds1['u'][0, nlay, :, :].squeeze()
        v0 = ds1['v'][0, nlay, :, :].squeeze()
        u = omat.copy()
        v = omat.copy()
        u[:, 1:-1] = (u0[:, 1:] + u0[:, :-1])/2
        v[1:-1, :] = (v0[1:, :] + v0[:-1, :])/2
        u = np.ma.masked_where(np.isnan(u), u)
        v = np.ma.masked_where(np.isnan(v), v)
        ds2['u'][tt,:,:] = u
        ds2['v'][tt,:,:] = v
        
    tt += 1
    ds.close()

ds1.close()
ds2.close()

# finale
import collections
result_dict = collections.OrderedDict()
result_dict['out_fn'] = out_fn
result_dict['date_string0'] = args.date_string0
result_dict['date_string1'] = args.date_string1
time_format = '%Y.%m.%d %H:%M:%S'
result_dict['start_time'] = start_time.strftime(time_format)
end_time = datetime.now()
result_dict['end_time'] = end_time.strftime(time_format)
dt_sec = (end_time - start_time).seconds
result_dict['total_seconds'] = str(dt_sec)
if os.path.isfile(out_fn):
    result_dict['result'] = 'success'
else:
    result_dict['result'] = 'fail'
print('')
for k in result_dict.keys():
    print('%s: %s' % (k, result_dict[k]))
