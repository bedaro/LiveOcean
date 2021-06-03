"""
TEF functions.
"""
import pandas as pd
import netCDF4 as nc
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# path to alpha provided by driver
import zfun
import zrfun

def get_sect_df():
    # section definitions
    # * x and y are latitude and longitude and we require sections to be NS or EW so
    # either x0=x1 or y0=y1
    sect_df = pd.DataFrame(columns=['x0', 'x1', 'y0', 'y1'])
    
    # Juan de Fuca
    sect_df.loc['jdf1',:] = [-124.673, -124.673,   48.371,   48.632]
    sect_df.loc['jdf2',:] = [-124.276, -124.276,   48.213,   48.542]
    sect_df.loc['jdf3',:] = [-123.865, -123.865,   48.110,   48.443]
    sect_df.loc['jdf4',:] = [-123.363, -123.363,   48.069,   48.461]

    # Strait of Georgia
    sect_df.loc['sog1',:] = [-123.740, -122.663,   48.857,   48.857]
    sect_df.loc['sog2',:] = [-124.065, -123.073,   49.184,   49.184]
    sect_df.loc['sog3',:] = [-124.223, -124.223,   49.220,   49.946]
    sect_df.loc['sog4',:] = [-125.356, -124.556,   50.002,   50.002]
    sect_df.loc['sog5',:] = [-125.600, -124.400,   50.200,   50.200]
    # San Juans
    sect_df.loc['sji1',:] = [-123.350, -122.65,   48.438,   48.438]
    sect_df.loc['sji2',:] = [-123.449, -122.425,   48.681,   48.681]

    # Channels around the San Juans
    sect_df.loc['dp',:] = [-122.643, -122.643,   48.389,   48.425]

    # Admiralty Inlet
    sect_df.loc['ai1',:] = [-122.762, -122.762,   48.141,   48.227]
    sect_df.loc['ai2',:] = [-122.808, -122.584,   48.083,   48.083]
    sect_df.loc['ai3',:] = [-122.755, -122.537,   48.002,   48.002]
    sect_df.loc['ai4',:] = [-122.537, -122.537,   47.903,   47.979]

    # Whidbey Basin
    sect_df.loc['wb1',:] = [-122.385, -122.286,   47.934,   47.934]
    sect_df.loc['wb2',:] = [-122.504, -122.286,   48.087,   48.087]
    sect_df.loc['wb3',:] = [-122.610, -122.498,   48.173,   48.173]
    sect_df.loc['wb4',:] = [-122.524, -122.524,   48.245,   48.308]

    # Hood Canal
    sect_df.loc['hc1',:] = [-122.670, -122.564,   47.912,   47.912]
    sect_df.loc['hc2',:] = [-122.769, -122.656,   47.795,   47.795]
    sect_df.loc['hc3',:] = [-122.802, -122.802,   47.709,   47.624]
    sect_df.loc['hc4',:] = [-123.013, -122.888,   47.610,   47.610]
    sect_df.loc['hc5',:] = [-123.132, -123.000,   47.484,   47.484]
    sect_df.loc['hc6',:] = [-123.178, -123.086,   47.390,   47.390]
    sect_df.loc['hc7',:] = [-123.079, -123.079,   47.385,   47.331]
    sect_df.loc['hc8',:] = [-122.960, -122.960,   47.358,   47.426]

    # Main Basin
    sect_df.loc['mb1',:] = [-122.544, -122.293,   47.862,   47.862]
    sect_df.loc['mb2',:] = [-122.603, -122.333,   47.732,   47.732]
    sect_df.loc['mb3',:] = [-122.570, -122.379,   47.561,   47.561]
    sect_df.loc['mb4',:] = [-122.544, -122.339,   47.493,   47.493]
    sect_df.loc['mb5',:] = [-122.610, -122.300,   47.349,   47.349]

    # Tacoma Narrows
    sect_df.loc['tn1',:] = [-122.584, -122.537,   47.313,   47.313]
    sect_df.loc['tn2',:] = [-122.564, -122.518,   47.286,   47.286]
    sect_df.loc['tn3',:] = [-122.584, -122.537,   47.259,   47.259]

    # South Sound
    sect_df.loc['ss1',:] = [-122.610, -122.610,   47.151,   47.309]
    sect_df.loc['ss2',:] = [-122.769, -122.769,   47.106,   47.187]
    sect_df.loc['ss3',:] = [-122.888, -122.888,   47.142,   47.313]

    return sect_df
    
def get_inds(x0, x1, y0, y1, G, verbose=False):
    
    # determine the direction of the section
    # and make sure indices are *increasing*
    if (x0==x1) and (y0!=y1):
        sdir = 'NS'
        a = [y0, y1]; a.sort()
        y0 = a[0]; y1 = a[1]
    elif (x0!=x1) and (y0==y1):
        sdir = 'EW'
        a = [x0, x1]; a.sort()
        x0 = a[0]; x1 = a[1]
    else:
        print('Input points do not form a proper section')
        sdir='bad'
        sys.exit()
    
    # we assume a plaid grid, as usual
    if sdir == 'NS':
        lon = G['lon_u'][0,:].squeeze()
        lat = G['lat_u'][:,0].squeeze()
    elif sdir == 'EW':
        lon = G['lon_v'][0,:].squeeze()
        lat = G['lat_v'][:,0].squeeze()
        
    # we get all 4 i's or j's but only 3 are used
    i0, i1, fr = zfun.get_interpolant(np.array([x0]), lon, extrap_nan=True)
    if np.isnan(fr):
        print('Bad x point')
        sys.exit()
    else:
        ii0 = int(i0)
    i0, i1, fr = zfun.get_interpolant(np.array([x1]), lon, extrap_nan=True)
    if np.isnan(fr):
        print('Bad x point')
        sys.exit()
    else:
        ii1 = int(i1)
    j0, j1, fr = zfun.get_interpolant(np.array([y0]), lat, extrap_nan=True)
    if np.isnan(fr):
        print('Bad y0 point')
        sys.exit()
    else:
        jj0 = int(j0)
    j0, j1, fr = zfun.get_interpolant(np.array([y1]), lat, extrap_nan=True)
    if np.isnan(fr):
        print('Bad y1 point')
        sys.exit()
    else:
        jj1 = int(j1)

    # get mask and trim indices
    # Note: the mask in G is True on water points
    if sdir == 'NS':
        mask = G['mask_u'][jj0:jj1+1, ii0]
        # Note: argmax finds the index of the first True in this case
        igood0 = np.argmax(mask)
        igood1 = np.argmax(mask[::-1])
        # check to see if section is "closed"
        if (igood0==0) | (igood1==0):
            print('Warning: not closed one or both ends')
        # keep only to end of water points, to allow for ocean sections
        Mask = mask[igood0:-igood1]
        # and change the indices to match.  These will be the indices
        # of the start and end points.
        jj0 = jj0 + igood0
        jj1 = jj1 - igood1
        if verbose:
            print('  sdir=%2s: jj0=%4d, jj1=%4d, ii0=%4d' % (sdir, jj0, jj1, ii0))
        Lat = lat[jj0:jj1+1]
        Lon = lon[ii0] * np.ones_like(Mask)
    elif sdir == 'EW':
        mask = G['mask_v'][jj0, ii0:ii1+1]
        igood0 = np.argmax(mask)
        igood1 = np.argmax(mask[::-1])
        # check to see if section is "closed"
        if (igood0==0) | (igood1==0):
            print('Warning: not closed one or both ends')
        Mask = mask[igood0:-igood1]
        ii0 = ii0 + igood0
        ii1 = ii1 - igood1
        if verbose:
            print('  sdir=%2s: jj0=%4d, ii0=%4d, ii1=%4d' % (sdir, jj0, ii0, ii1))
        Lon = lon[ii0:ii1+1]
        Lat = lat[jj0] * np.ones_like(Mask)
        
    return ii0, ii1, jj0, jj1, sdir, Lon, Lat, Mask
    
def start_netcdf(fn, out_fn, NT, NX, NZ, Lon, Lat, Ldir):
    try: # get rid of the existing version
        os.remove(out_fn)
    except OSError:
        pass # assume error was because the file did not exist
    ds = nc.Dataset(fn)
    # generating some lists
    vn_list = []
    if False:
        # all 3D variables on the s_rho grid
        for vv in ds.variables:
            vdim = ds.variables[vv].dimensions
            if ( ('ocean_time' in vdim) and ('s_rho' in vdim) ):
                vn_list.append(vv)
    else:
        # override
        vn_list.append('salt')
        # vn_list.append('oxygen')
        # vn_list.append('temp')
        # vn_list.append('NO3')
    # and some dicts of long names and units
    long_name_dict = dict()
    units_dict = dict()
    for vn in vn_list + ['ocean_time']:
        try:
            long_name_dict[vn] = ds.variables[vn].long_name
        except:
            long_name_dict[vn] = ''
        try:
            units_dict[vn] = ds.variables[vn].units
        except:
            units_dict[vn] = ''
    ds.close()
    # add custom dict fields
    long_name_dict['q'] = 'transport'
    units_dict['q'] = 'm3 s-1'
    long_name_dict['vel'] = 'velocity'
    units_dict['vel'] = 'm s-1'
    long_name_dict['lon'] = 'longitude'
    units_dict['lon'] = 'degrees'
    long_name_dict['lat'] = 'latitude'
    units_dict['lat'] = 'degrees'
    long_name_dict['h'] = 'depth'
    units_dict['h'] = 'm'
    long_name_dict['z0'] = 'z on rho-grid with zeta=0'
    units_dict['z0'] = 'm'
    long_name_dict['DA0'] = 'cell area on rho-grid with zeta=0'
    units_dict['DA0'] = 'm2'
    long_name_dict['DA'] = 'cell area on rho-grid'
    units_dict['DA'] = 'm2'

    # initialize netcdf output file
    foo = nc.Dataset(out_fn, 'w')
    foo.createDimension('xi_sect', NX)
    foo.createDimension('s_rho', NZ)
    foo.createDimension('ocean_time', NT)
    foo.createDimension('sdir_str', 2)
    for vv in ['ocean_time']:
        v_var = foo.createVariable(vv, float, ('ocean_time',))
        v_var.long_name = long_name_dict[vv]
        v_var.units = units_dict[vv]
    for vv in vn_list + ['q', 'vel', 'DA']:
        v_var = foo.createVariable(vv, float, ('ocean_time', 's_rho', 'xi_sect'))
        v_var.long_name = long_name_dict[vv]
        v_var.units = units_dict[vv]
    for vv in ['z0', 'DA0']:
        v_var = foo.createVariable(vv, float, ('s_rho', 'xi_sect'))
        v_var.long_name = long_name_dict[vv]
        v_var.units = units_dict[vv]
    for vv in ['lon', 'lat', 'h']:
        v_var = foo.createVariable(vv, float, ('xi_sect'))
        v_var.long_name = long_name_dict[vv]
        v_var.units = units_dict[vv]
    for vv in ['zeta']:
        v_var = foo.createVariable(vv, float, ('ocean_time', 'xi_sect'))
        v_var.long_name = 'Free Surface Height'
        v_var.units = 'm'

    # add static variables
    foo['lon'][:] = Lon
    foo['lat'][:] = Lat

    # add global attributes
    foo.gtagex = Ldir['gtagex']
    foo.date_string0 = Ldir['date_string0']
    foo.date_string1 = Ldir['date_string1']

    foo.close()
    
    return vn_list
    
def add_fields(ds, count, vn_list, G, S, sinfo):
    
    ii0, ii1, jj0, jj1, sdir, NT, NX, NZ, out_fn = sinfo
    
    foo = nc.Dataset(out_fn, 'a')
    
    # get depth and dz, and dd (which is either dx or dy)
    if sdir=='NS':
        h = ds['h'][jj0:jj1+1,ii0:ii1+1].squeeze()
        zeta = ds['zeta'][0,jj0:jj1+1,ii0:ii1+1].squeeze()
        z = zrfun.get_z(h, zeta, S, only_w=True)
        # print(h)
        # print(zeta)
        # print(z)
        dz = np.diff(z, axis=0)
        DZ = dz.mean(axis=2) # fails for a channel one point wide 2019.05.20 (oak)
        dd = G['DY'][jj0:jj1+1,ii0:ii1+1].squeeze()
        DD = dd.mean(axis=1)
        zeta = zeta.mean(axis=1)
        if count==0:
            hh = h.mean(axis=1)
            foo['h'][:] = hh
            
            z0 = zrfun.get_z(hh, 0*hh, S, only_rho=True)
            foo['z0'][:] = z0
            
            zw0 = zrfun.get_z(hh, 0*hh, S, only_w=True)
            DZ0 = np.diff(zw0, axis=0)
            DA0 = DD.reshape((1, NX)) * DZ0
            foo['DA0'][:] = DA0
            
    elif sdir=='EW':
        h = ds['h'][jj0:jj1+1,ii0:ii1+1].squeeze()
        zeta = ds['zeta'][0,jj0:jj1+1,ii0:ii1+1].squeeze()
        z = zrfun.get_z(h, zeta, S, only_w=True)
        dz = np.diff(z, axis=0)
        DZ = dz.mean(axis=1)
        dd = G['DX'][jj0:jj1+1,ii0:ii1+1].squeeze()
        DD = dd.mean(axis=0)
        zeta = zeta.mean(axis=0)
        if count==0:
            hh = h.mean(axis=0)
            foo['h'][:] = hh
            
            z0 = zrfun.get_z(hh, 0*hh, S, only_rho=True)
            foo['z0'][:] = z0
            
            zw0 = zrfun.get_z(hh, 0*hh, S, only_w=True)
            DZ0 = np.diff(zw0, axis=0)
            DA0 = DD.reshape((1, NX)) * DZ0
            foo['DA0'][:] = DA0
            
    # and then create the array of cell areas on the section
    DA = DD.reshape((1, NX)) * DZ
    # then velocity and hence transport
    if sdir=='NS':
        vel = ds['u'][0, :, jj0:jj1+1, ii0].squeeze()
    elif sdir=='EW':
        vel = ds['v'][0, :, jj0, ii0:ii1+1].squeeze()
    q = vel * DA
    
    foo['q'][count, :, :] = q
    foo['vel'][count, :, :] = vel
    foo['DA'][count, :, :] = DA
    foo['zeta'][count, :] = zeta
    foo['ocean_time'][count] = ds['ocean_time'][0]
    
    # save the tracer fields averaged onto this section
    for vn in vn_list:
        if sdir=='NS':
            vvv = (ds[vn][0,:,jj0:jj1+1,ii0].squeeze()
                + ds[vn][0,:,jj0:jj1+1,ii1].squeeze())/2
        elif sdir=='EW':
            vvv = (ds[vn][0,:,jj0,ii0:ii1+1].squeeze()
                + ds[vn][0,:,jj1,ii0:ii1+1].squeeze())/2
        foo[vn][count,:,:] = vvv
        
    foo.close()
