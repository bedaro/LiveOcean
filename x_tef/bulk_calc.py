#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to calculate a TEF time series using Marvin Lorenz' new multi-layer code.

Based on his code, and modified by PM.
"""

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

import numpy as np
import pickle
import matplotlib.pyplot as plt

import tef_fun_lorenz as tfl
from importlib import reload
reload(tfl)

# choose input and organize output
Ldir = Lfun.Lstart()
indir0 = Ldir['LOo'] + 'tef/'
# choose the tef extraction to process
item = Lfun.choose_item(indir0)
indir0 = indir0 + item + '/'
indir = indir0 + 'processed/'
sect_list_raw = os.listdir(indir)
sect_list_raw.sort()
sect_list = [item for item in sect_list_raw if ('.p' in item)]
print(20*'=' + ' Processed Sections ' + 20*'=')
print(*sect_list, sep=", ")
print(61*'=')
# select which sections to process
my_choice = input('-- Input section to process (e.g. sog5, or Return to process all): ')
if len(my_choice)==0:
    # full list
    pass
else: # single item
    if (my_choice + '.p') in sect_list:
        sect_list = [my_choice + '.p']
    else:
        print('That section is not available')
        sys.exit()
outdir = indir0 + 'bulk/'
Lfun.make_dir(outdir)

testing = False
tasks = min(len(os.sched_getaffinity(0)), psutil.cpu_count(logical=False))
for snp in sect_list:
    print('Working on ' + snp)
    out_fn = outdir + snp

    # load the data file
    tef_ex=pickle.load(open(indir + snp, 'rb'))
    # Notes on the data:
    # data.keys() => dict_keys(['tef_q', 'tef_qs', 'sbins', 'ot', 'qnet', 'fnet', 'ssh'])
    # data['tef_q'].shape => (8761, 1000), so packed [hour, salinity bin]
    # sbins are packed low to high
    # ot is time in seconds from 1/1/1970
    sbins = tef_ex['sbins']
    ot = tef_ex['ot']
    tef_q = tef_ex['tef_q']
    tef_qs = tef_ex['tef_qs']
    tef_qc = tef_ex['tef_qc'] if 'tef_qc' in tef_ex else {}
    statevars = list(tef_qc.keys())
    qnet = tef_ex['qnet']
    qabs = np.abs(qnet)
    fnet = tef_ex['fnet']
    ssh = tef_ex['ssh']

    # low-pass
    if True:
        # tidal averaging
        if not testing:
            with Pool(tasks) as p:
                lowpass_in = [tef_q, tef_qs]
                for k in statevars:
                    lowpass_in.append(tef_qc[k])
                res = p.map(zfun.filt_godin_mat, lowpass_in)
                tef_q_lp = res[0]
                tef_qs_lp = res[1]
                tef_qc_lp = {k: res[i+2] for i,k in enumerate(statevars)}
                qnet_lp, qabs_lp, fnet_lp, ssh_lp = p.map(zfun.filt_godin, [
                    qnet, qabs, fnet, ssh])
        else:
            tef_q_lp = zfun.filt_godin_mat(tef_q)
            tef_qs_lp = zfun.filt_godin_mat(tef_qs)
            tef_qc_lp = {k: zfun.filt_godin_mat(tef_qc[k]) for k in statevars}
            qnet_lp = zfun.filt_godin(qnet)
            qabs_lp = zfun.filt_godin(qabs)
            fnet_lp = zfun.filt_godin(fnet)
            ssh_lp = zfun.filt_godin(ssh)
        pad = 36
    else:
        # nday Hanning window
        nday = 120
        nfilt = nday*24
        tef_q_lp = zfun.filt_hanning_mat(tef_q, n=nfilt)
        tef_qs_lp = zfun.filt_hanning_mat(tef_qs, n=nfilt)
        tef_qc_lp = {k: zfun.filt_hanning_mat(tef_qc[k], n=nfilt) for k in statevars}
        qnet_lp = zfun.filt_hanning(qnet, n=nfilt)
        qabs_lp = zfun.filt_hanning(qabs, n=nfilt)
        fnet_lp = zfun.filt_hanning(fnet, n=nfilt)
        ssh_lp = zfun.filt_hanning(ssh, n=nfilt)
        pad = int(np.ceil(nfilt/2))

    # subsample and cut off nans
    tef_q_lp = tef_q_lp[pad:-(pad+1):24, :]
    tef_qs_lp = tef_qs_lp[pad:-(pad+1):24, :]
    for k in statevars:
        tef_qc_lp[k] = tef_qc_lp[k][pad:-(pad+1):24, :]
    ot = ot[pad:-(pad+1):24]
    qnet_lp = qnet_lp[pad:-(pad+1):24]
    qabs_lp = qabs_lp[pad:-(pad+1):24]
    fnet_lp = fnet_lp[pad:-(pad+1):24]
    ssh_lp = ssh_lp[pad:-(pad+1):24]

    # get sizes and make sedges (the edges of sbins)
    DS=sbins[1]-sbins[0]
    sedges = np.concatenate((sbins,np.array([sbins[-1]] + DS))) - DS/2
    NT = len(ot)
    NS = len(sedges)

    # calculate Q(s) and Q_s(s)
    Qv=np.zeros((NT, NS))
    Qs=np.zeros((NT, NS))
    Qc = {k: np.zeros((NT, NS)) for k in statevars}
    # Note that these are organized low s to high s, but still follow
    # the TEF formal definitions from MacCready (2011)
    Qv[:,:-1] = np.fliplr(np.cumsum(np.fliplr(tef_q_lp), axis=1))
    Qs[:,:-1] = np.fliplr(np.cumsum(np.fliplr(tef_qs_lp), axis=1))
    for k in statevars:
        Qc[k][:,:-1] = np.fliplr(np.cumsum(np.fliplr(tef_qc_lp[k]), axis=1))

    #get bulk values
    Qins=[]
    Qouts=[]
    sins=[]
    souts=[]

    if testing:
        plt.close('all')
        dd_list = [154, 260]
        print_info = True
    else:
        dd_list = range(NT)
        print_info = False

    def reduce(list_of_lists):
        """Convert a ragged list of arrays into a 2d numpy array"""
        max_y = np.max([len(l) for l in list_of_lists])
        res = np.zeros((len(list_of_lists), max_y),
                dtype=list_of_lists[0].dtype)
        if np.issubdtype(res.dtype, float):
            # Fill with NaN
            res *= np.nan
        elif np.issubdtype(res.dtype, int):
            # Fill with -1
            res -= 1
        for i,l in enumerate(list_of_lists):
            res[i,:len(l)] = l
        return res

    excs = {}
    def process_time(dd):
        global excs
        try:
            qv = Qv[dd,:]
            qs = Qs[dd,:]
            qc = {k: Qc[k][dd,:] for k in statevars}
    
            if print_info == True:
                print('\n**** dd = %d ***' % (dd))

            Q_in_m, Q_out_m, s_in_m, s_out_m, div_sal, ind, minmax = tfl.calc_bulk_values(sedges,
                qv, qs, print_info=print_info)
            Qc_in_m = {}
            Qc_out_m = {}
            sc_in_m = {}
            sc_out_m = {}
            for k in statevars:
                Qc_in_m[k], Qc_out_m[k], sc_in_m[k], sc_out_m[k], *drop = tfl.calc_bulk_values(sedges, qc[k], qs)

            if print_info == True:
                print(' ind = %s' % (str(ind)))
                print(' minmax = %s' % (str(minmax)))
                print(' div_sal = %s' % (str(div_sal)))
                print(' Q_in_m = %s' % (str(Q_in_m)))
                print(' s_in_m = %s' % (str(s_in_m)))
                print(' Q_out_m = %s' % (str(Q_out_m)))
                print(' s_out_m = %s' % (str(s_out_m)))

                fig = plt.figure(figsize=(12,8))

                ax = fig.add_subplot(121)
                ax.plot(Qv[dd,:], sedges,'.k')
                min_mask = minmax=='min'
                max_mask = minmax=='max'
                print(min_mask)
                print(max_mask)
                ax.plot(Qv[dd,ind[min_mask]], sedges[ind[min_mask]],'*b')
                ax.plot(Qv[dd,ind[max_mask]], sedges[ind[max_mask]],'*r')
                ax.grid(True)
                ax.set_title('Q(s) Time index = %d' % (dd))
                ax.set_ylim(-.1,36.1)
                ax.set_ylabel('Salinity')

                ax = fig.add_subplot(122)
                ax.plot(tef_q_lp[dd,:], sbins)
                ax.grid(True)
                ax.set_title('-dQ/ds')

            # save multi-layer output
            qq = np.concatenate((Q_in_m, Q_out_m))
            ss = np.concatenate((s_in_m, s_out_m))
            ii = np.argsort(ss)
            if len(ii)>0:
                QQ = qq[ii]
                SS = ss[ii]
            else:
                QQ = np.array([])
                SS = np.array([])
            QQC = {}
            SSC = {}
            for k in statevars:
                qqc = np.concatenate((Qc_in_m[k], Qc_out_m[k]))
                ssc = np.concatenate((sc_in_m[k], sc_out_m[k]))
                iic = np.argsort(ssc)
                if len(iic) > 0:
                    QQC[k] = qqc[iic]
                    SSC[k] = ssc[iic]

        except Exception as e:
            # TODO under Python 3.11 can use e.add_note() to annotate the
            # exception, then they can all be re-raised later as an
            # ExceptionGroup
            print(f'Exception {type(e)} with dd = {dd}')
            raise e

        else:
            return QQ, SS, QQC, SSC, ind, minmax, div_sal

    if not testing:
        with Pool(tasks) as p:
            res = p.map(process_time, dd_list)
    else:
        res = [process_time(dd) for dd in dd_list]
    QQ = reduce([x[0] for x in res])
    SS = reduce([x[1] for x in res])
    QQC = {}
    SSC = {}
    for k in statevars:
        if k not in res[0][2]:
            continue
        a = []
        b = []
        for x in res:
            a.append(x[2][k])
            b.append(x[3][k])
        QQC[k] = reduce(a)
        SSC[k] = reduce(b)
    ind = reduce([x[4] for x in res])
    minmax = reduce([x[5] for x in res])
    div_sal = reduce([x[6] for x in res])

    if testing == False:
        # save results
        bulk = dict()
        bulk['QQ'] = QQ
        bulk['SS'] = SS
        bulk['QQC'] = QQC
        bulk['SSC'] = SSC
        bulk['ot'] = ot
        bulk['Qv'] = Qv
        bulk['sedges'] = sedges
        bulk['ind'] = ind
        bulk['minmax'] = minmax
        bulk['qnet_lp'] = qnet_lp
        bulk['qabs_lp'] = qabs_lp
        bulk['fnet_lp'] = fnet_lp
        bulk['ssh_lp'] = ssh_lp
        bulk['div_sal'] = div_sal
        pickle.dump(bulk, open(out_fn, 'wb'))
    else:
        plt.show()
