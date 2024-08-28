# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:22:26 2024

@author: gqmc62
"""

import my_phasescreen as mps
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle

D = 4
L0 = 4
r0 = 0.05
pixel_scale = 1e-3

name = 'structure_function_theory_D{:.0f}m_L0{:.0f}m_r0{:.0f}cm_delta{:.0f}mm'.format(D,L0,r0*100,pixel_scale*1000)

N = D/pixel_scale

if os.path.isfile(name + '.pkl'):
    print('read old file')
    
    pkl_file = open(name + '.pkl', 'rb')
    data = pickle.load(pkl_file)
    
    theoretical_structure_function = data['theoretical_structure_function']
    kolmogorov_structure_function = data['kolmogorov_structure_function']
    x = data['x']
    
    pkl_file.close()
    
else:
    print('make a new one')
    
    theoretical_structure_function = mps.vonKarmanStructureFunction(r0, N, pixel_scale, L0)[1:]
    kolmogorov_structure_function = mps.KolmogorovStructureFunction(r0, N, pixel_scale)[1:]
    x = np.arange(N)[1:]*pixel_scale

    data = {'theoretical_structure_function' : theoretical_structure_function,
            'kolmogorov_structure_function' : kolmogorov_structure_function,
            'x' : x}
    output = open(name + '.pkl', 'wb')
    pickle.dump(data,output)
    output.close()
    
plt.plot(x, theoretical_structure_function, label='von Karman',color='r')
plt.plot(x, kolmogorov_structure_function, label='Kolmogorov',color='b')
plt.vlines([L0],
           0,np.nanmax([theoretical_structure_function,kolmogorov_structure_function]),
           ls='--', label='L0',color='c')
# plt.hlines([np.pi**2],pixel_scale,D,label='$\pi^2$')
# plt.vlines([r0],
#            0,np.nanmax([theoretical_structure_function,kolmogorov_structure_function]),
#            ls='--', label='r0',color='m')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Separation (m)')
plt.ylabel('Variance of Phase Difference (rad$^2$)')
plt.title('Structure Functions')
plt.savefig(name + '.eps',bbox_inches='tight')
plt.savefig(name + '.png',bbox_inches='tight')
plt.show()
