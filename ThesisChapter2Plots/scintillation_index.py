# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 17:30:32 2024

@author: gqmc62
"""

import numpy as np
from matplotlib import pyplot as plt

wvl = 500e-9
r = 0.05

z_list = np.logspace(2,5)

k = 2.*np.pi/wvl                    # float; rad/m; Wave number.
cn2dz=r**(-5./3.)/0.423/k**2        # float; Turbulence strength.



variance_theory_approx = 0.563 * k**(7./6.) * cn2dz * z_list**(5./6.)
plt.plot(z_list, variance_theory_approx, label='Rytov Approximation',
         marker='', ls='-',color='k')
plt.hlines(0.6, xmin=z_list.min(), xmax=z_list.max(), label='Saturation Level at Rytov#=0.6',
            ls='--',color='r')
plt.hlines(0.35, xmin=z_list.min(), xmax=z_list.max(), label='Strong Scintillation at Rytov#=0.35',
            ls='--',color='b')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('Propagation distance (m)')
plt.ylabel('Variance of log amplitude $\sigma^2_\chi$ (rad$^2$)')
plt.title('Variance of log amplitude versus propagation distance')
plt.savefig('scintillation_index.pdf',bbox_inches='tight')
plt.show()

