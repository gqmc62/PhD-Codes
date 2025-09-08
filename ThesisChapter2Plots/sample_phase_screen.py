# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:23:10 2024

@author: gqmc62
"""

from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle


from matplotlib.lines import Line2D
import matplotlib as mpl

fontsize = 10
figsize = [4,4]
dpi = 600

mpl.rcParams['axes.titlesize'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['figure.titlesize'] = fontsize
mpl.rcParams['figure.figsize'] = figsize
mpl.rcParams['figure.dpi'] = dpi
mpl.rcParams['savefig.dpi'] = dpi
mpl.rcParams['savefig.bbox'] = 'tight'
# mpl.rcParams['savefig.bbox_inches'] = 'tight'
mpl.rcParams['savefig.format'] = 'pdf'

color_list = np.array(list(mpl.colors.TABLEAU_COLORS.keys()))


D = 1
delta = 0.05/16
r0 = 0.05
L0 = 10

name = 'sample_phase_screen_D{:.0f}_delta{:.0f}cm_r0{:.0f}cm_L0{:.0f}m'.format(D,delta*100,r0*100,L0)

N = int(np.ceil(D/delta))

if os.path.isfile(name + '.pkl'):
    print('read old file')
    
    pkl_file = open(name + '.pkl', 'rb')
    data = pickle.load(pkl_file)
    
    x = data['x']
    scrn = data['scrn']
    
    pkl_file.close()
    
else:
    print('make a new one')
    
    IF = PhaseScreenKolmogorov(N, delta, r0, L0,random_seed=999)
    for i in np.arange(N):
        IF.add_row()
    scrn = IF.scrn
    scrn -= scrn.mean()
    x = (np.arange(N)- N//2) * delta

    data = {'x' : x,
            'scrn' : scrn}
    output = open(name + '.pkl', 'wb')
    pickle.dump(data,output)
    output.close()

plt.figure(figsize=(5,4))
plt.pcolor(x,x,np.angle(np.exp(1j*scrn)))
plt.axis('square')
plt.colorbar()
plt.xlabel('position (m)')
# plt.title('phase (rad)')
plt.savefig(name + '.png')
# plt.savefig(name + '.eps',bbox_inches='tight')
# plt.savefig(name + '.png',bbox_inches='tight')
plt.show()
