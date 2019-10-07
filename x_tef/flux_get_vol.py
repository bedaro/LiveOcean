"""
Find the volume of each flux segment
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import netCDF4 as nc

import os; import sys
sys.path.append(os.path.abspath('../alpha'))
import Lfun
Ldir = Lfun.Lstart(gridname='cas6', tag='v3')
import zrfun

sys.path.append(os.path.abspath(Ldir['LO'] + 'plotting'))
import pfun

import tef_fun
from importlib import reload
reload(tef_fun)

import flux_fun
reload(flux_fun)

fng = Ldir['grid'] + 'grid.nc'
G = zrfun.get_basic_info(fng, only_G=True)
h = np.ma.masked_where(G['mask_rho']==False, G['h'])
x = G['lon_rho'].data
y = G['lat_rho'].data
xp = G['lon_psi'].data
yp = G['lat_psi'].data
m = G['mask_rho']

# get the DataFrame of all sections
sect_df = tef_fun.get_sect_df()

testing = True

# segment definitions, assembled by looking at the figure
# created by plot_thalweg_mean.py
segs = flux_fun.segs

if testing == True:
    seg_name_list = ['W2']
else:
     seg_name_list = segs.keys()

plt.close('all')



for seg_name in seg_name_list:
    
    # initialize a DataFrame to hold segment info
    seg_df = pd.DataFrame(columns=['ii0','ii1','jj0','jj1',
        'sdir','side','Lon0','Lon1','Lat0','Lat1',])
    seg = segs[seg_name]
        
    # find volume
    sect_info = {}
    
    if testing:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        ax.pcolormesh(xp, yp, h[1:-1,1:-1], cmap='terrain_r', vmin=-100, vmax = 400)
        pfun.dar(ax)
        pfun.add_coast(ax)
        
    
    for side in list('SNWE'):
        sect_list = seg[side]
        for sect_name in sect_list:
            # get section lat, lon, and other info
            x0, x1, y0, y1, landward = sect_df.loc[sect_name,:]
            # get indices for this section
            ii0, ii1, jj0, jj1, sdir, Lon, Lat, Mask = tef_fun.get_inds(x0, x1, y0, y1, G)
            Lon0 = Lon.min(); Lon1 = Lon.max()
            Lat0 = Lat.min(); Lat1 = Lat.max()
            seg_df.loc[sect_name,'ii0'] = ii0
            seg_df.loc[sect_name,'ii1'] = ii1
            seg_df.loc[sect_name,'jj0'] = jj0
            seg_df.loc[sect_name,'jj1'] = jj1
            seg_df.loc[sect_name,'sdir'] = sdir
            seg_df.loc[sect_name,'side'] = side
            seg_df.loc[sect_name,'Lon0'] = Lon0
            seg_df.loc[sect_name,'Lon1'] = Lon1
            seg_df.loc[sect_name,'Lat0'] = Lat0
            seg_df.loc[sect_name,'Lat1'] = Lat1
            
        
    pad = .1
    ax.axis([seg_df['Lon0'].min()-pad,seg_df['Lon1'].max()+pad,
        seg_df['Lat0'].min()-pad,seg_df['Lat1'].max()+pad])
        
    # initialize a mask
    mm = m.copy().data # boolean array, True over water
    full_ji_list = [] # full list of indices of good rho points inside the volume
    this_ji_list = [] # current list of indices of good rho points inside the volume
    next_ji_list = [] # next list of indices of good rho points inside the volume
        
    for sn in seg_df.index:
        s = seg_df.loc[sn,:]
        # the black dots show rho gridpoints OUTSIDE of the segment volume, while
        # the magenta ones are those just inside
        if s['sdir'] == 'NS' and s['side'] == 'W':
            ax.plot(x[s['jj0']:s['jj1']+1, s['ii0']], y[s['jj0']:s['jj1']+1, s['ii0']], 'ok')
            ax.plot(x[s['jj0']:s['jj1']+1, s['ii1']], y[s['jj0']:s['jj1']+1, s['ii1']], 'om')
            mm[s['jj0']:s['jj1']+1, s['ii0']] = False
        elif s['sdir'] == 'NS' and s['side'] == 'E':
            ax.plot(x[s['jj0']:s['jj1']+1, s['ii0']], y[s['jj0']:s['jj1']+1, s['ii0']], 'om')
            ax.plot(x[s['jj0']:s['jj1']+1, s['ii1']], y[s['jj0']:s['jj1']+1, s['ii1']], 'ok')
            mm[s['jj0']:s['jj1']+1, s['ii1']] = False
        if s['sdir'] == 'EW' and s['side'] == 'S':
            ax.plot(x[s['jj0'], s['ii0']:s['ii1']+1], y[s['jj0'], s['ii0']:s['ii1']+1], 'ok')
            ax.plot(x[s['jj1'], s['ii0']:s['ii1']+1], y[s['jj1'], s['ii0']:s['ii1']+1], 'om')
            mm[s['jj0'], s['ii0']:s['ii1']+1] = False
        if s['sdir'] == 'EW' and s['side'] == 'N':
            ax.plot(x[s['jj0'], s['ii0']:s['ii1']+1], y[s['jj0'], s['ii0']:s['ii1']+1], 'om')
            ax.plot(x[s['jj1'], s['ii0']:s['ii1']+1], y[s['jj1'], s['ii0']:s['ii1']+1], 'ok')
            mm[s['jj1'], s['ii0']:s['ii1']+1] = False


    for sn in seg_df.index:
        print(sn)
        s = seg_df.loc[sn,:]
        if s['sdir'] == 'NS' and s['side'] == 'W':
            ji = (s['jj0'],s['ii1'])
            mm, this_ji_list, full_ji_list, next_ji_list = flux_fun.update_mm(ji, mm,
                    this_ji_list, full_ji_list, next_ji_list)
            ji = (s['jj1'],s['ii1'])
            mm, this_ji_list, full_ji_list, next_ji_list = flux_fun.update_mm(ji, mm,
                    this_ji_list, full_ji_list, next_ji_list)
        elif s['sdir'] == 'NS' and s['side'] == 'E':
            ji = (s['jj0'],s['ii0'])
            mm, this_ji_list, full_ji_list, next_ji_list = flux_fun.update_mm(ji, mm,
                    this_ji_list, full_ji_list, next_ji_list)
            ji = (s['jj1'],s['ii0'])
            mm, this_ji_list, full_ji_list, next_ji_list = flux_fun.update_mm(ji, mm,
                    this_ji_list, full_ji_list, next_ji_list)
        elif s['sdir'] == 'EW' and s['side'] == 'S':
            ji = (s['jj1'],s['ii0'])
            mm, this_ji_list, full_ji_list, next_ji_list = flux_fun.update_mm(ji, mm,
                    this_ji_list, full_ji_list, next_ji_list)
            ji = (s['jj1'],s['ii1'])
            mm, this_ji_list, full_ji_list, next_ji_list = flux_fun.update_mm(ji, mm,
                    this_ji_list, full_ji_list, next_ji_list)
        elif s['sdir'] == 'EW' and s['side'] == 'N':
            ji = (s['jj0'],s['ii0'])
            mm, this_ji_list, full_ji_list, next_ji_list = flux_fun.update_mm(ji, mm,
                    this_ji_list, full_ji_list, next_ji_list)
            ji = (s['jj0'],s['ii1'])
            mm, this_ji_list, full_ji_list, next_ji_list = flux_fun.update_mm(ji, mm,
                    this_ji_list, full_ji_list, next_ji_list)
            

                
        for ji in full_ji_list:
            ax.plot(x[ji],y[ji],'*r')
            
        # elif s['sdir'] == 'NS' and s['side'] == 'E':
        #     ax.plot(x[s['jj0']:s['jj1']+1, s['ii0']], y[s['jj0']:s['jj1']+1, s['ii0']], 'om')
        #     ax.plot(x[s['jj0']:s['jj1']+1, s['ii1']], y[s['jj0']:s['jj1']+1, s['ii1']], 'ok')
        #     mm[s['jj0']:s['jj1']+1, s['ii1']] = False
        # if s['sdir'] == 'EW' and s['side'] == 'S':
        #     ax.plot(x[s['jj0'], s['ii0']:s['ii1']+1], y[s['jj0'], s['ii0']:s['ii1']+1], 'ok')
        #     ax.plot(x[s['jj1'], s['ii0']:s['ii1']+1], y[s['jj1'], s['ii0']:s['ii1']+1], 'om')
        #     mm[s[s['jj0'], s['ii0']:s['ii1']+1] = False
        # if s['sdir'] == 'EW' and s['side'] == 'N':
        #     ax.plot(x[s['jj0'], s['ii0']:s['ii1']+1], y[s['jj0'], s['ii0']:s['ii1']+1], 'om')
        #     ax.plot(x[s['jj1'], s['ii0']:s['ii1']+1], y[s['jj1'], s['ii0']:s['ii1']+1], 'ok')
        #     mm[s['jj1'], s['ii0']:s['ii1']+1] = False
    
        
        plt.show()
        
    