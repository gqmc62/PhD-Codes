# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 17:30:32 2024

@author: gqmc62
"""

import numpy as np
from matplotlib import pyplot as plt

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

wvl = 500e-9
r = 0.05

z_list = np.logspace(2,5)

k = 2.*np.pi/wvl                    # float; rad/m; Wave number.
cn2dz=r**(-5./3.)/0.423/k**2        # float; Turbulence strength.



variance_theory_approx = 0.563 * k**(7./6.) * cn2dz * z_list**(5./6.)
plt.plot(z_list, variance_theory_approx, label='$\sigma^2_R$',
         marker='', ls='-',color='k')
plt.hlines(0.6, xmin=z_list.min(), xmax=z_list.max(), label='Saturation',
            ls='--',color='r')
plt.hlines(0.35, xmin=z_list.min(), xmax=z_list.max(), label='Strong scintillation',
            ls='--',color='b')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('Propagation distance (m)')
plt.ylabel('$\sigma^2_\chi$ (rad$^2$)')
# plt.title('Variance of log amplitude versus propagation distance')
plt.savefig('scintillation_index.pdf',bbox_inches='tight')
plt.show()

