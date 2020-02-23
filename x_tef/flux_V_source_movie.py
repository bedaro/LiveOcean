"""
Code to plot the volumes of the Salish Sea in a graphically compelling way
that we could use for movies of the flux_engine results.

This version is designed for steady sources like rivers, so not Tres info.

"""

# imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

from matplotlib.cm import get_cmap

import os; import sys
sys.path.append(os.path.abspath('../alpha'))
import Lfun
Ldir = Lfun.Lstart(gridname='cas6', tag='v3')

import tef_fun
import flux_fun

# select the indir
indir = Ldir['LOo'] + 'tef/flux_engine/cas6_v3_lo8b/'

outdir00 = Ldir['LOo'] + 'tef/movies/'
Lfun.make_dir(outdir00)
outdir0 = outdir00 + 'cas6_v3_lo8b/'
Lfun.make_dir(outdir0)

# load the DataFrame of results of flux_engine.py
infile = Lfun.choose_item(indir, tag='S_', exclude_tag='AGE',
    itext='Choose flux engine output file:')
aa = pd.read_pickle(indir + infile)
this_run = infile.replace('.p','')
print(this_run)
outdir = outdir0 + this_run + '/'
Lfun.make_dir(outdir, clean=True)

olist = this_run.split('_')
source = olist[0] + '_' + olist[1]
year_str = olist[2]
season = olist[3]


testing = False

# ==============================================================

# load a Series of the volumes of each segment, created by flux_get_vol.py
v_df = pd.read_pickle(Ldir['LOo'] + 'tef/cas6_v3_lo8b_2017.01.01_2017.12.31/flux/volumes.p')
V = flux_fun.get_V(v_df)

plt.close('all')

cmap = get_cmap('cool') # 'YlOrRd'
# get rgba using cmap(0-255)

def color_scaling(val):
    val_scaled = 1 + np.log10(val*20 + 1e-8)/3
    return val_scaled

day_list = list(aa.index)

if testing == False:
    day_list_short = day_list[:51:5]
    # typically we have saves every 2 days so a step=5 makes 10 days per frame
    # and end=501 makes the last frame at day 1000
else:
    day_list_short = [day_list[0]]


for ii in range(len(day_list_short)):

    nouts = ('0000' + str(ii))[-4:]
    outname = 'plot_' + nouts + '.png'
    outfile = outdir + outname
    print('Plotting ' + outname)
    
    day = day_list_short[ii]

    # PLOTTING

    # vectors of the starting locations of the channels on the plot
    x00_list = [0, 17, 8, 19]
    y00_list = [0, -9.25, -12, -6]

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)

    ch_list = list(flux_fun.short_seg_dict.keys())

    xy = {}

    jj = 0
    for ch in ch_list:
        seg_list = flux_fun.short_seg_dict[ch].copy()
        
        color = flux_fun.clist[jj]

        if ch in ['Hood Canal', 'Whidbey Basin']:
            seg_list.reverse()

        # make vectors of volume
        vs = v_df.loc[seg_list,'volume m3'].to_numpy()
        hs = vs**(1/3)
        hs = hs/1e3

        x00 = x00_list[jj]
        y00 = y00_list[jj]

        dist = np.cumsum(hs)
        dist = np.append(0,dist)
        ii = 0
        for seg in seg_list:
            
            
            c_s = aa.loc[day,seg+'_s']
            c_f = aa.loc[day,seg+'_f']
            
            # let's convert to log10 scaling
            cc_s = color_scaling(c_s)
            cc_f = color_scaling(c_f)
            
            x0 = x00 + dist[ii]
            x1 = x00 + dist[ii+1]
            y0 = y00
            y1 = y00 - hs[ii]
            dy = y0-y1
            
            fr = .2
            # bottom layer
            fh1 = ax.fill([x0,x1,x1,x0],[y0-fr*dy,y0-fr*dy,y1,y1],
                color=cmap(int(cc_s*255)), alpha=.8)
            fh1[0].set_edgecolor('k')
            
            # top layer
            fh2= ax.fill([x0,x1,x1,x0],[y0,y0,y1+(1-fr)*dy,y1+(1-fr)*dy],
                color=cmap(int(cc_f*255)), alpha=.8)
            fh2[0].set_edgecolor('k')
            
            # add a stripe to identify the channel
            ax.plot([x0, x1],[y0, y0],'-',color=color,lw=3)
            
            
            ax.text((x0+x1)/2,y0+.2,seg, horizontalalignment='center', fontsize=10)
            ii += 1
            # save position of center of cell
            xy[seg] = ((x0+x1)/2, (y0+y1)/2)
            

        jj += 1
        
    # add scale
    def add_scalebox(x0, y0, val):
        val_scaled = color_scaling(val)
        #print(val_scaled)
        dx=2; dy=1
        x1=x0+dx; y1=y0-dy
        ax.fill([x0,x1,x1,x0],[y0,y0,y1,y1], color=cmap(int(val_scaled*255)), alpha=.8)
        ax.text((x0+x1)/2, y0+.2, ('%0.3f' % (val)),
            ha='center', size=12)
    #add_scalebox(40, -16, 1)
    add_scalebox(45, -16, .05)
    add_scalebox(50, -16, .005)
    add_scalebox(55, -16, .005)
    ax.text(50, -15, 'Concentration Scale', ha='center', size=13, style='italic')
    
    # add text
    ax.set_title(this_run,
        size=14, style='italic', weight='bold')
    ax.text(-1, -3, 'Pacific\nOcean', ha='right', size=16)
    ax.text(55, -3, 'Johnstone\nStrait', ha='left', size=13)
    ax.text(0, -8, 'Day = %s' % (str(int(day))), size=18, weight='bold')    

    # plot connecting lines
    lw = 3
    al = .3
    ax.plot([xy['J4'][0],xy['A1'][0]], [xy['J4'][1],xy['A1'][1]], '-ok', linewidth=lw, alpha=al)
    ax.plot([xy['H1'][0],xy['A3'][0]], [xy['H1'][1],xy['A3'][1]], '-ok', linewidth=lw, alpha=al)
    ax.plot([xy['M1'][0],xy['W1'][0]], [xy['M1'][1],xy['W1'][1]], '-ok', linewidth=lw, alpha=al)
    ax.plot([xy['W4'][0],xy['J4'][0]], [xy['W4'][1],xy['J4'][1]], '-ok', linewidth=lw-2, alpha=al)

    ax.set_axis_off()
    
    if testing:
        plt.show()
    else:
        fig.savefig(outfile)
        plt.close()

# and make a movie
if testing == False:
    ff_str = ("ffmpeg -r 8 -i " +
    outdir+"plot_%04d.png -vcodec libx264 -pix_fmt yuv420p -crf 25 "
    +outdir+"movie.mp4")
    os.system(ff_str)


