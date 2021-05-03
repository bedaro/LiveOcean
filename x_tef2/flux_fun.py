"""
Variables and functions used by the "flux" code.
"""
import numpy as np
from datetime import datetime, timedelta
import zfun # path provided by calling code
import pandas as pd
import Lfun
import pickle

from warnings import filterwarnings
filterwarnings('ignore') # skip some warning messages
# associated with lines like QQp[QQ<=0] = np.nan

def get_fluxes(indir, sect_name):
    # Form time series of 2-layer TEF quantities, from the multi-layer bulk values.
    # - include transports for the no-tidal-pumping version
    # - adjust sign convention so that Qs is positive for the saltier layer and call it Qin
    
    bulk = pickle.load(open(indir + 'bulk/' + sect_name + '.p', 'rb'))
    QQ = bulk['QQ']
    SS = bulk['SS']
    SS2 = bulk['SS2']
    ot = bulk['ot']
    dt = []
    for tt in ot:
        dt.append(Lfun.modtime_to_datetime(tt))
        
    # separate positive and negative transports
    QQp = QQ.copy()
    QQp[QQ<=0] = np.nan
    QQm = QQ.copy()
    QQm[QQ>=0] = np.nan
        
    # full transports
    QQm = QQm
    QQp = QQp
    
    # form two-layer versions
    Qp = np.nansum(QQp, axis=1)
    QSp = np.nansum(QQp*SS, axis=1)
    QS2p = np.nansum(QQp*SS2, axis=1)
    Qm = np.nansum(QQm, axis=1)
    QSm = np.nansum(QQm*SS, axis=1)
    QS2m = np.nansum(QQm*SS2, axis=1)
    
    # TEF salinities
    Sp = QSp/Qp
    Sm = QSm/Qm
    
    # TEF variance
    S2p = QS2p/Qp
    S2m = QS2m/Qm
    
    # adjust sign convention so that positive flow is salty
    SP = np.nanmean(Sp)
    SM = np.nanmean(Sm)
    if SP > SM:
        
        # initial positive was inflow
        Sin = Sp
        Sout = Sm
        S2in = S2p
        S2out = S2m
        Qin = Qp
        Qout = Qm
        in_sign = 1
        
        QSin = QSp
        QSout = QSm
        
        QS2in = QS2p
        QS2out = QS2m
        
    elif SM > SP:
        # initial postive was outflow
        Sin = Sm
        Sout = Sp
        S2in = S2m
        S2out = S2p
        Qin = -Qm
        Qout = -Qp
        in_sign = -1
        
        QSin = -QSm
        QSout = -QSp
        
        QS2in = -QS2m
        QS2out = -QS2p
    else:
        print('ambiguous sign!!')
    
    if False:
        print('%s SP=%0.1f SM = %0.1f' % (sect_name, SP, SM))
        import sys
        sys.stdout.flush()
        
    fnet = in_sign * bulk['fnet_lp']/1e6 # net tidal energy flux [MW]
    qabs = bulk['qabs_lp'] # low pass of absolute value of net transport [m3/s]
    
    tef_df = pd.DataFrame(index=dt)
    tef_df['Qin']=Qin
    tef_df['Qout']=Qout
    tef_df['QSin']=QSin
    tef_df['QSout']=QSout
    tef_df['Sin']=Sin
    tef_df['Sout']=Sout

    tef_df['QS2in']=QS2in
    tef_df['QS2out']=QS2out
    tef_df['S2in']=S2in
    tef_df['S2out']=S2out

    tef_df['Ftide'] = fnet
    tef_df['Qtide'] = qabs
    
    return tef_df, in_sign

def get_fluxes_alt(indir, sect_name):
    # Much like get_fluxes() above, but it includes additional fields used by
    # bluk_plot_clean.py, especially related to tidal pumping.
    
    # WARNING: I don't like the "- convert transports to 1000 m3/s" bit.  At this
    # point 2021.04.28 this function is only used by bulk_plot_clean.py.
    
    # Form time series of 2-layer TEF quantities, from the multi-layer bulk values.
    # - convert transports to 1000 m3/s
    # - include transports for the no-tidal-pumping version
    # - adjust sign convention so that Qs is positive for the saltier layer and call it Qin
    
    bulk = pickle.load(open(indir + 'bulk/' + sect_name + '.p', 'rb'))
    QQ = bulk['QQ']
    VV = bulk['VV']
    AA = bulk['AA']
    SS = bulk['SS']
    ot = bulk['ot']
    dt = []
    for tt in ot:
        dt.append(Lfun.modtime_to_datetime(tt))
        
    # separate positive and negative transports
    QQp = QQ.copy()
    QQp[QQ<=0] = np.nan
    QQm = QQ.copy()
    QQm[QQ>=0] = np.nan
    
    VVp = VV.copy()
    VVp[QQ<=0] = np.nan
    VVm = VV.copy()
    VVm[QQ>=0] = np.nan

    AAp = AA.copy()
    AAp[QQ<=0] = np.nan
    AAm = AA.copy()
    AAm[QQ>=0] = np.nan
    
    # full transports
    QQm = QQm/1000
    QQp = QQp/1000
    
    # transports without tidal pumping
    QQp_alt = VVp * AAp /1000
    QQm_alt = VVm * AAm /1000
    
    # form two-layer versions
    Qp = np.nansum(QQp, axis=1)
    QSp = np.nansum(QQp*SS, axis=1)
    Qm = np.nansum(QQm, axis=1)
    QSm = np.nansum(QQm*SS, axis=1)
    Qp_alt = np.nansum(QQp_alt, axis=1)
    Qm_alt = np.nansum(QQm_alt, axis=1)
    
    # TEF salinities
    Sp = QSp/Qp
    Sm = QSm/Qm
    
    # adjust sign convention so that positive flow is salty
    SP = np.nanmean(Sp)
    SM = np.nanmean(Sm)
    if SP > SM:
        
        # initial positive was inflow
        Sin = Sp
        Sout = Sm
        Qin = Qp
        Qout = Qm
        Qin_alt = Qp_alt
        Qout_alt = Qm_alt
        in_sign = 1
        
        QQin = QQp
        QQout = QQm
        QQin_alt = QQp_alt
        QQout_alt = QQm_alt
        
    elif SM > SP:
        # initial postive was outflow
        Sin = Sm
        Sout = Sp
        Qin = -Qm
        Qout = -Qp
        Qin_alt = -Qm_alt
        Qout_alt = -Qp_alt
        in_sign = -1
        
        QQin = -QQm
        QQout = -QQp
        QQin_alt = -QQm_alt
        QQout_alt = -QQp_alt
    else:
        print('ambiguous sign!!')
    
    if False:
        print('%s SP=%0.1f SM = %0.1f' % (sect_name, SP, SM))
        import sys
        sys.stdout.flush()
        
    fnet = in_sign * bulk['fnet_lp']/1e6 # net tidal energy flux [MW]
    qabs = bulk['qabs_lp']/1000 # low pass of absolute value of net transport [1000 m3/s]
    
    tef_df = pd.DataFrame(index=dt)
    tef_df['Qin']=Qin
    tef_df['Qout']=Qout
    tef_df['Qin_alt']=Qin_alt
    tef_df['Qout_alt']=Qout_alt
    tef_df['Sin']=Sin
    tef_df['Sout']=Sout
    tef_df['Ftide'] = fnet
    tef_df['Qtide'] = qabs
    
    return tef_df, in_sign, dt, QQin, QQout, SS, QQin_alt, QQout_alt
    
# desired time ranges, the "seasons"
def get_dtr(year):
    dtr = {}
    dtr['full'] = (datetime(year,1,1,12,0,0), datetime(year,12,31,12,0,0))
    dtr['winter'] = (datetime(year,1,1,12,0,0), datetime(year,3,31,12,0,0)) # JFM
    dtr['spring'] = (datetime(year,4,1,12,0,0), datetime(year,6,30,12,0,0)) # AMJ
    dtr['summer'] = (datetime(year,7,1,12,0,0), datetime(year,9,30,12,0,0)) # JAS
    dtr['fall'] = (datetime(year,10,1,12,0,0), datetime(year,12,31,12,0,0)) # OMD
    return dtr

# Lists of 2-layer segments to use for "initial condition" experiments in the flux_engine.
# The keys should match up with "src" values in flux_engine.py.
def get_seg_list(X,N):
    sl = [X+str(n)+'_s' for n in range(1,N+1)]+[X+str(n)+'_f' for n in range(1,N+1)]
    return sl
    
# make a segment list
ic_seg2_dict = {'IC_HoodCanalInner': ['H'+str(n)+'_s' for n in range(3,9)],
            'IC_HoodCanal': get_seg_list('H',8),
            'IC_SouthSound': get_seg_list('S',4),
            'IC_Whidbey': get_seg_list('W',4),
            'IC_PS':  get_seg_list('A',3) + get_seg_list('M',6)
                    + get_seg_list('T',2) + get_seg_list('S',4)
                    + get_seg_list('H',8) + get_seg_list('W',4),
            'IC_SoG': get_seg_list('G',6),
            'IC_Salish': get_seg_list('A',3) + get_seg_list('M',6)
                    + get_seg_list('T',2) + get_seg_list('S',4)
                    + get_seg_list('H',8) + get_seg_list('W',4)
                    + get_seg_list('G',6),
            }
                            
season_list = list(get_dtr(2017).keys())

# create Series of two-layer volumes
# this is the one place to specify the ratio volume in the "salty" and "fresh" layers
def get_V(v_df):
    V = pd.Series()
    for seg_name in v_df.index:
        V[seg_name+'_s'] = 0.8 * v_df.loc[seg_name,'volume m3']
        V[seg_name+'_f'] = 0.2 * v_df.loc[seg_name,'volume m3']
    return V

# segment definitions, assembled by looking at the figures
# created by plot_thalweg_mean.py
segs = {
        'J1':{'S':[], 'N':[], 'W':['jdf1'], 'E':['jdf2'], 'R':['sanjuan', 'hoko']},
        'J2':{'S':[], 'N':[], 'W':['jdf2'], 'E':['jdf3'], 'R':[]},
        'J3':{'S':[], 'N':[], 'W':['jdf3'], 'E':['jdf4'], 'R':['elwha']},
        'J4':{'S':[], 'N':['sji1'], 'W':['jdf4'], 'E':['ai1','dp'], 'R':['dungeness']},
        
        'G1':{'S':['sji1'], 'N':['sji2'], 'W':[], 'E':[], 'R':['samish']},
        'G2':{'S':['sji2'], 'N':['sog1'], 'W':[], 'E':[], 'R':['nooksack', 'cowichan']},
        'G3':{'S':['sog1'], 'N':['sog2'], 'W':[], 'E':[], 'R':['nanaimo', 'fraser']},
        'G4':{'S':['sog2'], 'N':[], 'W':['sog3'], 'E':[], 'R':['clowhom', 'squamish']},
        'G5':{'S':[], 'N':['sog4'], 'W':[], 'E':['sog3'], 'R':['englishman', 'tsolum', 'oyster']},
        'G6':{'S':['sog4'], 'N':['sog5'], 'W':[], 'E':[], 'R':[]},
        
        'A1':{'S':['ai2'], 'N':[], 'W':['ai1'], 'E':[], 'R':[]},
        'A2':{'S':['ai3'], 'N':['ai2'], 'W':[], 'E':[], 'R':[]},
        'A3':{'S':['hc1'], 'N':['ai3'], 'W':[], 'E':['ai4'], 'R':[]},
        
        'M1':{'S':['mb1'], 'N':['wb1'], 'W':['ai4'], 'E':[], 'R':[]},
        'M2':{'S':['mb2'], 'N':['mb1'], 'W':[], 'E':[], 'R':[]},
        'M3':{'S':['mb3'], 'N':['mb2'], 'W':[], 'E':[], 'R':['green', 'cedar']},
        'M4':{'S':['mb4'], 'N':['mb3'], 'W':[], 'E':[], 'R':[]},
        'M5':{'S':['mb5'], 'N':['mb4'], 'W':[], 'E':[], 'R':[]},
        'M6':{'S':['tn1'], 'N':['mb5'], 'W':[], 'E':[], 'R':['puyallup']},
        
        'T1':{'S':['tn2'], 'N':['tn1'], 'W':[], 'E':[], 'R':[]},
        'T2':{'S':['tn3'], 'N':['tn2'], 'W':[], 'E':[], 'R':[]},
        
        'S1':{'S':[], 'N':['tn3'], 'W':['ss1'], 'E':[], 'R':[]},
        'S2':{'S':[], 'N':[], 'W':['ss2'], 'E':['ss1'], 'R':['nisqually']},
        'S3':{'S':[], 'N':[], 'W':['ss3'], 'E':['ss2'], 'R':[]},
        'S4':{'S':[], 'N':[], 'W':[], 'E':['ss3'], 'R':['deschutes']},
        
        'W1':{'S':['wb1'], 'N':['wb2'], 'W':[], 'E':[], 'R':['snohomish']},
        'W2':{'S':['wb2'], 'N':['wb3'], 'W':[], 'E':[], 'R':['stillaguamish']},
        'W3':{'S':['wb3'], 'N':[], 'W':[], 'E':['wb4'], 'R':[]},
        'W4':{'S':[], 'N':[], 'W':['wb4', 'dp'], 'E':[], 'R':['skagit']},
        
        'H1':{'S':['hc2'], 'N':['hc1'], 'W':[], 'E':[], 'R':[]},
        'H2':{'S':[], 'N':['hc2'], 'W':['hc3'], 'E':[], 'R':[]},
        'H3':{'S':['hc4'], 'N':[], 'W':[], 'E':['hc3'], 'R':['duckabush', 'dosewallips']},
        'H4':{'S':['hc5'], 'N':['hc4'], 'W':[], 'E':[], 'R':['hamma']},
        'H5':{'S':['hc6'], 'N':['hc5'], 'W':[], 'E':[], 'R':[]},
        'H6':{'S':[], 'N':['hc6'], 'W':[], 'E':['hc7'], 'R':['skokomish']},
        'H7':{'S':[], 'N':[], 'W':['hc7'], 'E':['hc8'], 'R':[]},
        'H8':{'S':[], 'N':[], 'W':['hc8'], 'E':[], 'R':[]},
        
        #'##':{'S':[], 'N':[], 'W':[], 'E':[], 'R':[]},
        }
        
# make lists of the various segment sequences (used below)
ssJ = ['J'+str(s) for s in range(1,5)]
ssM = ['M'+str(s) for s in range(1,7)]
ssA = ['A'+str(s) for s in range(1,4)]
ssT = ['T'+str(s) for s in range(1,3)]
ssS = ['S'+str(s) for s in range(1,5)]
ssG = ['G'+str(s) for s in range(1,7)]
ssW = ['W'+str(s) for s in range(1,5)]
ssH = ['H'+str(s) for s in range(1,9)]

# This list is the same as the keys for all the dicts below.
# we make it to have a fixed order for processing things, since
# the order of the keys of a dict may not be fixed.
channel_list = ['Juan de Fuca to Strait of Georgia',
            'Admiralty Inlet to South Sound',
            'Hood Canal',
            'Whidbey Basin']

# also cue up a line for the target salinities from the TEF sections
channel_dict = {'Juan de Fuca to Strait of Georgia':['jdf1','jdf2','jdf3','jdf4',
                'sji1', 'sji2', 'sog1','sog2','sog3','sog4','sog5'],
            'Admiralty Inlet to South Sound': ['ai1', 'ai2', 'ai3','ai4',
                'mb1','mb2','mb3','mb4','mb5',
                'tn1','tn2','tn3',
                'ss1','ss2','ss3'],
            'Hood Canal':['hc1','hc2','hc3','hc4','hc5','hc6','hc7','hc8'],
            'Whidbey Basin':['wb1','wb2','wb3','wb4','dp']}

long_channel_dict = {'Juan de Fuca to Strait of Georgia':['jdf1','jdf2','jdf3','jdf4',
                'sji1', 'sji2', 'sog1','sog2','sog3','sog4','sog5'],
            'Admiralty Inlet to South Sound': ['jdf4', 'ai1', 'ai2', 'ai3','ai4',
                'mb1','mb2','mb3','mb4','mb5',
                'tn1','tn2','tn3',
                'ss1','ss2','ss3'],
            'Hood Canal':['ai3', 'hc1','hc2','hc3','hc4','hc5','hc6','hc7','hc8'],
            'Whidbey Basin':['ai4', 'wb1','wb2','wb3','wb4','dp']}
                
seg_dict = {'Juan de Fuca to Strait of Georgia': ssJ + ssG,
            'Admiralty Inlet to South Sound': ['J4'] + ssA + ssM + ssT + ssS,
            'Hood Canal': ['A3'] + ssH,
            'Whidbey Basin': ['M1'] + ssW}
            
# same as seg_dict, but without the connections to adjoining channels
short_seg_dict = {'Juan de Fuca to Strait of Georgia': ssJ + ssG,
            'Admiralty Inlet to South Sound': ssA + ssM + ssT + ssS,
            'Hood Canal': ssH,
            'Whidbey Basin': ssW}
            
# colors to associate with each channel (the keys in channel_ and seg_dict)
clist = ['blue', 'red', 'olive', 'orange']
c_dict = dict(zip(channel_list, clist))

def make_dist(x,y):
    NS = len(x)
    xs = np.zeros(NS)
    ys = np.zeros(NS)
    xs, ys = zfun.ll2xy(x, y, x[0], y[0])
    dx = np.diff(xs)
    dy = np.diff(ys)
    dd = (dx**2 + dy**2)**.5 # not clear why np.sqrt throws an error
    dist = np.zeros(NS)
    dist[1:] = np.cumsum(dd/1000) # convert m to km
    return dist
        
def update_mm(ji, mm, this_ji_list, full_ji_list, next_ji_list):
    if mm[ji] == True:
        this_ji_list.append(ji)
        full_ji_list.append(ji)
    for ji in this_ji_list:
        mm[ji] = False
    keep_looking = True
    counter = 0
    while len(this_ji_list) > 0:
        #print('iteration ' + str(counter))
        for ji in this_ji_list:
            JI = (ji[0]+1, ji[1]) # North
            if mm[JI] == True:
                next_ji_list.append(JI)
                mm[JI] = False
            else:
                pass
            JI = (ji[0], ji[1]+1) # East
            if mm[JI] == True:
                next_ji_list.append(JI)
                mm[JI] = False
            else:
                pass
            JI = (ji[0]-1, ji[1]) # South
            if mm[JI] == True:
                next_ji_list.append(JI)
                mm[JI] = False
            else:
                pass
            JI = (ji[0], ji[1]-1) # West
            if mm[JI] == True:
                next_ji_list.append(JI)
                mm[JI] = False
            else:
                pass
        for ji in next_ji_list:
            full_ji_list.append(ji)
        this_ji_list = next_ji_list.copy()
        next_ji_list = []
        counter += 1
    return mm, this_ji_list, full_ji_list, next_ji_list