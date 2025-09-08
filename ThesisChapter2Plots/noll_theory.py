# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:47:17 2024

@author: gqmc62
"""

import my_phasescreen as mps
import numpy as np
from matplotlib import pyplot as plt
import pickle
import os

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

D = 2
L0 = 20
r0 = 0.05
pixel_scale = 1e-3

name = 'noll_theory_D{:.0f}m_L0{:.0f}m_r0{:.0f}cm_delta{:.0f}mm'.format(D,L0,100*r0,pixel_scale*1000)

n_modes = 20
N = D/pixel_scale

if os.path.isfile(name + '.pkl'):
    print('read old file')
    pkl_file = open(name + '.pkl', 'rb')
    data = pickle.load(pkl_file)
    x = data['x']
    kolmogorov_variance_ZernikeCoeff = data['kolmogorov_variance_ZernikeCoeff']
    theoretical_variance_ZernikeCoeff = data['theoretical_variance_ZernikeCoeff']
    pkl_file.close()
else:
    print('make a new one')
    x = np.arange(n_modes)
    kolmogorov_variance_ZernikeCoeff = mps.noll_variance(n_modes,N,pixel_scale,r0)
    theoretical_variance_ZernikeCoeff = mps.von_Karman_noll_variance(n_modes,N,pixel_scale,r0,L0)
    
    data = {'x' : x,
            'kolmogorov_variance_ZernikeCoeff' : kolmogorov_variance_ZernikeCoeff,
            'theoretical_variance_ZernikeCoeff' : theoretical_variance_ZernikeCoeff}
    
    output = open(name + '.pkl', 'wb')
    pickle.dump(data,output)
    output.close()
    


plt.plot(x, theoretical_variance_ZernikeCoeff, label='von Karman',color='r')
plt.plot(x, kolmogorov_variance_ZernikeCoeff, label='Kolmogorov',color='b')
plt.legend()

plt.yscale('log')
plt.xlabel('Zernike modes')
plt.ylabel('Variances of coefficient (rad$^2$)')
# plt.title('Zernike Mode Decompositions')
plt.savefig(name)
# plt.savefig(name + '.png', bbox_inches='tight')
# plt.savefig(name + '.eps', bbox_inches='tight')
plt.show()
