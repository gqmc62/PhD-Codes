# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:31:58 2024

@author: gqmc62
"""

import my_phasescreen as mps
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


D = 20
L0 = 2
r0 = 0.1
l0 = 0.02

name = 'PSD_theory_D{:.0f}m_L0{:.0f}m_r0{:.0f}cm_l0{:.0f}cm'.format(D,L0,r0*100,l0*100)

pixel_scale = l0/10
N = D/pixel_scale

if os.path.isfile(name + '.pkl'):
    print('read old file')
    
    pkl_file = open(name + '.pkl', 'rb')
    data = pickle.load(pkl_file)
    
    f = data['f']
    theoretical_PSD = data['theoretical_PSD']
    kolmogorov_PSD = data['kolmogorov_PSD']
    pkl_file.close()
else:
    print('make a new one')
    
    f = (np.arange(N/2-1) + 1)/N/pixel_scale
    theoretical_PSD = mps.psd_function(r0, N, pixel_scale, L0, l0)[int(N/2),int(N/2 + 1) : ]
    kolmogorov_PSD = mps.kolmogorov_psd_function(r0, N, pixel_scale)[int(N/2),int(N/2 + 1) : ]
    
    data = {'f' : f,
            'theoretical_PSD' : theoretical_PSD,
            'kolmogorov_PSD' : kolmogorov_PSD}
    
    output = open(name + '.pkl', 'wb')
    pickle.dump(data,output)
    output.close()

MIN = np.min([theoretical_PSD,kolmogorov_PSD])
MAX = np.max([theoretical_PSD,kolmogorov_PSD])

# plt.figure(figsize=[7,5])

plt.plot(f, theoretical_PSD, label='von Karman',color='r')
plt.plot(f, kolmogorov_PSD, label='Kolmogorov',color='b')

plt.vlines([1/L0],MIN,MAX,label='$1/L_0$',ls='--',color='c')
# plt.vlines([1/r0],MIN,MAX,label='1/r0',ls='--',color='m')
plt.vlines([1/l0],MIN,MAX,label='$1/l_0$',ls='--',color='k')

plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Spatial frequency (m$^{-1}$)')
plt.ylabel('Phase power spectrum density (rad$^2$m$^2$)')
# plt.title('Cross-section of PSD of Phase')
# plt.savefig(name + '.eps',bbox_inches='tight')
# plt.savefig(name + '.png',bbox_inches='tight')
plt.savefig(name)
plt.show()
