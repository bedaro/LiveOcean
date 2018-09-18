"""
TEF functions.
"""
import pandas as pd
import netCDF4 as nc
import numpy as np
import pickle
import matplotlib.pyplot as plt

# path to alpha provided by driver
import zfun
import zrfun

def get_sect_df():
    # section definitions
    # * x and y are latitude and longitude and we require sections to be NS or EW so
    # either x0=x1 or y0=y1
    # * landward is the sign to multipy the transport by to be landward (1 or -1)
    sect_df = pd.DataFrame(columns=['x0', 'x1', 'y0', 'y1', 'landward'])
    
    # Juan de Fuca
    sect_df.loc['jdf1',:] = [-124.673, -124.673,   48.371,   48.632, 1]
    sect_df.loc['jdf2',:] = [-124.276, -124.276,   48.213,   48.542, 1]
    sect_df.loc['jdf3',:] = [-123.865, -123.865,   48.110,   48.443, 1]
    sect_df.loc['jdf4',:] = [-123.363, -123.363,   48.069,   48.461, 1]

    # Strait of Georgia
    sect_df.loc['sog1',:] = [-123.740, -122.663,   48.857,   48.857, 1]
    sect_df.loc['sog2',:] = [-124.065, -123.073,   49.184,   49.184, 1]
    sect_df.loc['sog3',:] = [-124.223, -124.223,   49.220,   49.946, -1]
    sect_df.loc['sog4',:] = [-125.356, -124.556,   50.002,   50.002, 1]

    # San Juans
    sect_df.loc['sji1',:] = [-123.350, -122.451,   48.438,   48.438, 1]
    sect_df.loc['sji2',:] = [-123.449, -122.425,   48.681,   48.681, 1]

    # Channels around the San Juans
    sect_df.loc['dp',:] = [-122.643, -122.643,   48.389,   48.425, -1]
    sect_df.loc['swin',:] = [-122.531, -122.471,   48.420,   48.420, 1]
    sect_df.loc['haro',:] = [-123.429, -123.099,   48.542,   48.542, 1]

    # Admiralty Inlet
    sect_df.loc['ai1',:] = [-122.762, -122.762,   48.141,   48.227, 1]
    sect_df.loc['ai2',:] = [-122.808, -122.584,   48.083,   48.083, -1]
    sect_df.loc['ai3',:] = [-122.755, -122.537,   48.002,   48.002, -1]
    sect_df.loc['ai4',:] = [-122.537, -122.537,   47.903,   47.979, 1]

    # Whidbey Basin
    sect_df.loc['wb1',:] = [-122.385, -122.286,   47.934,   47.934, 1]
    sect_df.loc['wb2',:] = [-122.504, -122.286,   48.087,   48.087, 1]
    sect_df.loc['wb3',:] = [-122.610, -122.498,   48.173,   48.173, 1]
    sect_df.loc['wb4',:] = [-122.524, -122.524,   48.245,   48.308, 1]

    # Hood Canal
    sect_df.loc['hc1',:] = [-122.670, -122.564,   47.912,   47.912, -1]
    sect_df.loc['hc2',:] = [-122.769, -122.656,   47.795,   47.795, -1]
    sect_df.loc['hc3',:] = [-122.802, -122.802,   47.709,   47.624, -1]
    sect_df.loc['hc4',:] = [-123.013, -122.888,   47.610,   47.610, -1]
    sect_df.loc['hc5',:] = [-123.132, -123.000,   47.484,   47.484, -1]
    sect_df.loc['hc6',:] = [-123.178, -123.086,   47.390,   47.390, -1]
    sect_df.loc['hc7',:] = [-123.079, -123.079,   47.385,   47.331, 1]
    sect_df.loc['hc8',:] = [-122.960, -122.960,   47.358,   47.426, 1]

    # Main Basin
    sect_df.loc['mb1',:] = [-122.544, -122.293,   47.862,   47.862, -1]
    sect_df.loc['mb2',:] = [-122.603, -122.333,   47.732,   47.732, -1]
    sect_df.loc['mb3',:] = [-122.570, -122.379,   47.561,   47.561, -1]
    sect_df.loc['mb4',:] = [-122.544, -122.339,   47.493,   47.493, -1]
    sect_df.loc['mb5',:] = [-122.610, -122.300,   47.349,   47.349, -1]

    # Colvos Passage
    sect_df.loc['clv',:] = [-122.577, -122.485,   47.444,   47.444, 1]

    # Tacoma Narrows
    sect_df.loc['tn1',:] = [-122.584, -122.537,   47.313,   47.313, -1]
    sect_df.loc['tn2',:] = [-122.564, -122.518,   47.286,   47.286, -1]
    sect_df.loc['tn3',:] = [-122.584, -122.537,   47.259,   47.259, -1]

    # South Sound
    sect_df.loc['ss1',:] = [-122.610, -122.610,   47.151,   47.309, -1]
    sect_df.loc['ss2',:] = [-122.769, -122.769,   47.106,   47.187, -1]
    sect_df.loc['ss3',:] = [-122.888, -122.888,   47.142,   47.313, -1]

    # Inlets in South Sound
    sect_df.loc['carr',:] = [-122.769, -122.663,   47.291,   47.291, 1]
    sect_df.loc['case',:] = [-122.868, -122.788,   47.214,   47.214, 1]
    sect_df.loc['budd',:] = [-122.934, -122.894,   47.129,   47.129, -1]
    sect_df.loc['eld',:] = [-122.934, -122.934,   47.133,   47.160, -1]
    sect_df.loc['tott',:] = [-122.973, -122.934,   47.174,   47.174, -1]
    sect_df.loc['oak',:] = [-122.960, -122.960,   47.187,   47.214, -1]

    # Channel in South Sound
    sect_df.loc['pick',:] = [-122.947, -122.907,   47.264,   47.264, 1]
    
    # Coastal sections
    sect_df.loc['willapa_mouth',:] = [-124.051, -124.051,   46.631,   46.748, 1]
    
    return sect_df
    
def get_inds(x0, x1, y0, y1, G):
    
    # determine the direction of the section
    # and make sure indices are increasing
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
        # keep one mask point on each end, just to be sure we have a closed section
        Mask = mask[igood0-1:-igood1+1]
        # and change the indices to match.  These will be the indices
        # of the start and end points.
        jj0 = jj0 + igood0 - 1
        jj1 = jj1 - igood1 + 1
        print('  sdir=%2s: jj0=%4d, jj1=%4d, ii0=%4d' % (sdir, jj0, jj1, ii0))
        Lat = lat[jj0:jj1+1]
        Lon = lon[ii0] * np.ones_like(Mask)
    elif sdir == 'EW':
        mask = G['mask_v'][jj0, ii0:ii1+1]
        igood0 = np.argmax(mask)
        igood1 = np.argmax(mask[::-1])
        Mask = mask[igood0-1:-igood1+1]
        ii0 = ii0 + igood0 - 1
        ii1 = ii1 - igood1 + 1
        print('  sdir=%2s: jj0=%4d, ii0=%4d, ii1=%4d' % (sdir, jj0, ii0, ii1))
        Lon = lon[ii0:ii1+1]
        Lat = lat[jj0] * np.ones_like(Mask)
        
    return ii0, ii1, jj0, jj1, sdir, Lon, Lat, Mask
    
def start_netcdf(fn, out_fn, NT, NX, NZ, Lon, Lat, Ldir):
    # generating some lists
    vn_list = []
    ds = nc.Dataset(fn)
    if True:
        # all 3D variables on the s_rho grid
        for vv in ds.variables:
            vdim = ds.variables[vv].dimensions
            if ( ('ocean_time' in vdim) and ('s_rho' in vdim) ):
                vn_list.append(vv)
    else:
        # override
        vn_list.append('salt')
        vn_list.append('temp')
        # vn_list.append('NO3')
        # vn_list.append('oxygen')
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
    long_name_dict['lon'] = 'longitude'
    units_dict['lon'] = 'degrees'
    long_name_dict['lat'] = 'latitude'
    units_dict['lat'] = 'degrees'

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
    for vv in vn_list + ['q']:
        v_var = foo.createVariable(vv, float, ('ocean_time', 's_rho', 'xi_sect'))
        v_var.long_name = long_name_dict[vv]
        v_var.units = units_dict[vv]
    for vv in ['lon', 'lat']:
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
    
    ii0, ii1, jj0, jj1, sdir, landward, NT, NX, NZ, out_fn = sinfo
    
    foo = nc.Dataset(out_fn, 'a')
    
    # get depth and dz
    if sdir=='NS':
        h = ds['h'][jj0:jj1+1,ii0:ii1+1].squeeze()
        zeta = ds['zeta'][0,jj0:jj1+1,ii0:ii1+1].squeeze()
        z = zrfun.get_z(h, zeta, S, only_w=True)
        dz = np.diff(z, axis=0)
        DZ = dz.mean(axis=2)
        dy = G['DY'][jj0:jj1+1,ii0:ii1+1].squeeze()
        DY = dy.mean(axis=1)
        zeta = zeta.mean(axis=1)
    elif sdir=='EW':
        h = ds['h'][jj0:jj1+1,ii0:ii1+1].squeeze()
        zeta = ds['zeta'][0,jj0:jj1+1,ii0:ii1+1].squeeze()
        z = zrfun.get_z(h, zeta, S, only_w=True)
        dz = np.diff(z, axis=0)
        DZ = dz.mean(axis=1)
        dy = G['DY'][jj0:jj1+1,ii0:ii1+1].squeeze()
        DY = dy.mean(axis=0)
        zeta = zeta.mean(axis=0)
    # and then create the array of cell areas on the section
    DA = DY.reshape((1, NX)) * DZ
    # then velocity and hence transport
    if sdir=='NS':
        vel = ds['u'][0, :, jj0:jj1+1, ii0].squeeze()
    elif sdir=='EW':
        vel = ds['v'][0, :, jj0, ii0:ii1+1].squeeze()
    q = vel * DA * landward
    
    foo['q'][count, :, :] = q
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
    
def tef_integrals(fn):
    # choices
    plot_profiles = True # plot profiles when certain conditions occur
    tidal_average = False # which king of time filtering
    NS_fact = 50 # range to look at for relative maxima (50 means 2% of the salt range)
    Crit_fact = 5 # was 50, Used for dropping small extrema (5 means 20% of max transport)
    nlay_max = 4 # maximum allowable number of layers to process
    Q_crit = 100 # mask transport layers with |Q| smaller than this (m3 s-1)
    
    # load results
    tef_dict = pickle.load(open(fn, 'rb'))
    tef_q = tef_dict['tef_q']
    tef_qs = tef_dict['tef_qs']
    sbins = tef_dict['sbins']
    smax = sbins.max()
    qnet = tef_dict['qnet']
    fnet = tef_dict['fnet']
    ot = tef_dict['ot']
    td = (ot - ot[0])/86400
    NS = len(sbins)
    #print('NS = %d' % (NS))

    # low-pass
    if tidal_average:
        # tidal averaging
        tef_q_lp = zfun.filt_godin_mat(tef_q)
        tef_qs_lp = zfun.filt_godin_mat(tef_qs)
        qnet_lp = zfun.filt_godin(qnet)
        fnet_lp = zfun.filt_godin(fnet)
        pad = 36
    else:
        # nday Hanning window
        nday = 5
        nfilt = nday*24
        tef_q_lp = zfun.filt_hanning_mat(tef_q, n=nfilt)
        tef_qs_lp = zfun.filt_hanning_mat(tef_qs, n=nfilt)
        qnet_lp = zfun.filt_hanning(qnet, n=nfilt)
        fnet_lp = zfun.filt_hanning(fnet, n=nfilt)
        pad = int(np.ceil(nfilt/2))

    # subsample
    tef_q_lp = tef_q_lp[pad:-(pad+1):24, :]
    tef_qs_lp = tef_qs_lp[pad:-(pad+1):24, :]
    td = td[pad:-(pad+1):24]
    qnet_lp = qnet_lp[pad:-(pad+1):24]
    fnet_lp = fnet_lp[pad:-(pad+1):24]

    # # find integrated TEF quantities
    # # alternate method using cumulative sum of the transport
    # # to identify the salinity dividing inflow and outflow
    # # RESULT: this way is not sensitive to the number of
    # # salinity bins.
    # #
    # start by making the low-passed flux arrays sorted
    # from high to low salinity
    rq = np.fliplr(tef_q_lp)
    rqs = np.fliplr(tef_qs_lp)
    sbinsr = sbins[::-1]
    # then form the cumulative sum (the function Q(s))
    qcs = np.cumsum(rq, axis=1)
    nt = len(td)

    # new version to handle more layers
    from scipy.signal import argrelextrema
    Imax = argrelextrema(qcs, np.greater, axis=1, order=int(NS/NS_fact))
    Imin = argrelextrema(qcs, np.less, axis=1, order=int(NS/NS_fact))

    Q = np.zeros((nt, nlay_max))
    QS = np.zeros((nt, nlay_max))

    crit = np.nanmax(np.abs(qcs)) / Crit_fact

    for tt in range(nt):
        # we use these masks because there are multiple values for a given day
        maxmask = Imax[0]==tt
        minmask = Imin[0]==tt
        imax = Imax[1][maxmask]
        imin = Imin[1][minmask]
        # drop extrema indices which are too close to the ends
        if len(imax) > 0:
            mask = np.abs(qcs[tt,imax] - qcs[tt,0]) > crit
            imax = imax[mask]
        if len(imax) > 0:
            mask = np.abs(qcs[tt,imax] - qcs[tt,-1]) > crit
            imax = imax[mask]
        if len(imin) > 0:
            mask = np.abs(qcs[tt,imin] - qcs[tt,0]) > crit
            imin = imin[mask]
        if len(imin) > 0:
            mask = np.abs(qcs[tt,imin] - qcs[tt,-1]) > crit
            imin = imin[mask]
        ivec = np.sort(np.concatenate((np.array([0]), imax, imin, np.array([NS]))))
        nlay = len(ivec)-1
    
        # combine non-alternating layers
        qq = np.zeros(nlay)
        qqs = np.zeros(nlay)
        jj = 0
        for ii in range(nlay):
            qlay = rq[tt, ivec[ii]:ivec[ii+1]].sum()
            qslay = rqs[tt, ivec[ii]:ivec[ii+1]].sum()
            if ii == 0:
                qq[0] = qlay
                qqs[0] = qslay
            else:
                if np.sign(qlay)==np.sign(qq[jj]):
                    qq[jj] += qlay
                    qqs[jj] += qslay
                    nlay -= 1
                else:
                    jj += 1
                    qq[jj] = qlay
                    qqs[jj] = qslay

        if nlay == 1:
            if qq[0] >= 0:
                Q[tt,1] = qq[0]
                QS[tt,1] = qqs[0]
            elif qq[0] < 0:
                Q[tt,2] = qq[0]
                QS[tt,2] = qqs[0]
        elif nlay ==2:
            if qq[0] >= 0:
                Q[tt,1] = qq[0]
                QS[tt,1] = qqs[0]
                Q[tt,2] = qq[1]
                QS[tt,2] = qqs[1]
            elif qq[0] < 0:
                Q[tt,1] = qq[1]
                QS[tt,1] = qqs[1]
                Q[tt,2] = qq[0]
                QS[tt,2] = qqs[0]
        elif nlay ==3:
            if qq[0] >= 0:
                Q[tt,1] = qq[0]
                QS[tt,1] = qqs[0]
                Q[tt,2] = qq[1]
                QS[tt,2] = qqs[1]
                Q[tt,3] = qq[2]
                QS[tt,3] = qqs[2]
            elif qq[0] < 0:
                Q[tt,0] = qq[0]
                QS[tt,0] = qqs[0]
                Q[tt,1] = qq[1]
                QS[tt,1] = qqs[1]
                Q[tt,2] = qq[2]
                QS[tt,2] = qqs[2]
        elif nlay ==4:
            if qq[0] >= 0:
                err_str = 'Backwards 4: td=%5.1f  nlay = %d' % (td[tt],nlay)
                print(err_str)
                Q[tt,:] = np.nan
                QS[tt,:] = np.nan
                # and plot a helpful figure
                if plot_profiles:
                    profile_plot(rq, sbinsr, smax, qcs, ivec, err_str, tt, NS)

            elif qq[0] < 0:
                Q[tt,0] = qq[0]
                QS[tt,0] = qqs[0]
                Q[tt,1] = qq[1]
                QS[tt,1] = qqs[1]
                Q[tt,2] = qq[2]
                QS[tt,2] = qqs[2]
                Q[tt,3] = qq[3]
                QS[tt,3] = qqs[3]
        else:
            print('- Excess layers: td=%5.1f  nlay = %d' % (td[tt],nlay))
            Q[tt,:] = np.nan
            QS[tt,:] = np.nan
        
    # form derived quantities
    Q[np.abs(Q)<Q_crit] = np.nan
    S = QS/Q
    
    return Q, S, QS, qnet_lp, fnet_lp, td
    
def profile_plot(rq, sbinsr, smax, qcs, ivec, err_str, tt, NS):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(121)
    ax.plot(rq[tt,:], sbinsr)
    ax.set_ylim(smax,0)
    ax.grid(True)
    ax.set_ylabel('Salinity')
    ax.set_xlabel('q(s)')
    #
    ax = fig.add_subplot(122)
    ax.plot(qcs[tt,:], sbinsr)
    # print(ivec)
    Ivec = ivec.copy()
    Ivec[Ivec==NS] = NS-1
    ax.plot(qcs[tt,Ivec], sbinsr[Ivec],'*k')
    ax.set_ylim(smax,0)
    ax.grid(True)
    ax.set_xlabel('Q(s)')
    ax.set_title(err_str)
    
def tef_integrals_v2(fn):
    # choices
    tidal_average = False # which kind of time filtering
    nlay_max = 2 # maximum allowable number of layers to process
    
    # load results
    tef_dict = pickle.load(open(fn, 'rb'))
    tef_q = tef_dict['tef_q']
    tef_qs = tef_dict['tef_qs']
    sbins = tef_dict['sbins']
    smax = sbins.max()
    qnet = tef_dict['qnet']
    fnet = tef_dict['fnet']
    ot = tef_dict['ot']
    td = (ot - ot[0])/86400
    NS = len(sbins)

    # low-pass
    if tidal_average:
        # tidal averaging
        tef_q_lp = zfun.filt_godin_mat(tef_q)
        tef_qs_lp = zfun.filt_godin_mat(tef_qs)
        qnet_lp = zfun.filt_godin(qnet)
        fnet_lp = zfun.filt_godin(fnet)
        pad = 36
    else:
        # nday Hanning window
        nday = 5
        nfilt = nday*24
        tef_q_lp = zfun.filt_hanning_mat(tef_q, n=nfilt)
        tef_qs_lp = zfun.filt_hanning_mat(tef_qs, n=nfilt)
        qnet_lp = zfun.filt_hanning(qnet, n=nfilt)
        fnet_lp = zfun.filt_hanning(fnet, n=nfilt)
        pad = int(np.ceil(nfilt/2))

    # subsample
    tef_q_lp = tef_q_lp[pad:-(pad+1):24, :]
    tef_qs_lp = tef_qs_lp[pad:-(pad+1):24, :]
    td = td[pad:-(pad+1):24]
    qnet_lp = qnet_lp[pad:-(pad+1):24]
    fnet_lp = fnet_lp[pad:-(pad+1):24]

    #find integrated TEF quantities
    
    # start by making the low-passed flux arrays sorted
    # from high to low salinity
    rq = np.fliplr(tef_q_lp)
    rqs = np.fliplr(tef_qs_lp)
    sbinsr = sbins[::-1]
    # then form the cumulative sum (the function Q(s))
    Q = np.cumsum(rq, axis=1)
    nt = len(td)

    Qi = np.nan * np.zeros((nt, nlay_max))
    Fi = np.nan * np.zeros((nt, nlay_max))
    Qi_abs = np.nan * np.zeros((nt, nlay_max))
    Fi_abs = np.nan * np.zeros((nt, nlay_max))

    for tt in range(nt):
        
        imax = np.argmax(Q[tt,:])
        imin = np.argmin(Q[tt,:])
                
        # set the dividing salinity by the size of the transport
        Qin = rq[tt, 0:imax].sum()
        Qout = rq[tt, 0:imin].sum()
        if np.abs(Qin) > np.abs(Qout):
            idiv = imax
        else:
            idiv = imin
                
        ivec = np.unique(np.array([0, idiv, NS+1]))
        nlay = len(ivec)-1

        for ii in range(nlay):
            Qi[tt,ii] = rq[tt, ivec[ii]:ivec[ii+1]].sum()
            Qi_abs[tt,ii] = np.abs(rq[tt, ivec[ii]:ivec[ii+1]]).sum()
            Fi[tt,ii] = rqs[tt, ivec[ii]:ivec[ii+1]].sum()
            Fi_abs[tt,ii] = np.abs(rqs[tt, ivec[ii]:ivec[ii+1]]).sum()
        
    # form derived quantities
    Qcrit = np.abs(Qi[:,0]).mean()/5
    Qi[np.abs(Qi)==0] = np.nan
    Si = Fi_abs/Qi_abs
    
    return Qi, Si, Fi, qnet_lp, fnet_lp, td
    
