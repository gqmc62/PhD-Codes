# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:44:09 2022

@author: gqmc62

"""
import pickle
import numpy as np
from matplotlib import pyplot as plt

with open('Test_SHWFS_FoVv3.pkl', 'rb') as file:
    myvar = pickle.load(file)
    lenslet_pixel = myvar['lenslet_pixel']
    # lenslet_pixel = 256
    L0 = myvar['L0']
    l0 = myvar['l0']
    D_lenslet = myvar['D_lenslet']
    r0 = myvar['r0']
    wvl = myvar['wvl']
    pf_lenslet = myvar['pf_lenslet']
    spot_size = myvar['spot_size']
    REPEAT = myvar['REPEAT']
    N = myvar['N']
    f_lenslet = myvar['f_lenslet']
    d = myvar['d']
    wavenumber = myvar['wavenumber']
    dif_SHWFS_slope = myvar['dif_SHWFS_slope']
    g_slope = myvar['g_slope']
    real_SHWFS_slope = myvar['real_SHWFS_slope']

CEN = int(lenslet_pixel/2)
fov = np.arange(CEN) + 1 # fov in wvl/d = half fov in pixel
var = (np.var(dif_SHWFS_slope[0,:,:,:,:],axis=(1,3), ddof=1)).T
# mean = (np.mean(dif_SHWFS_slope,axis=(2,4))[:,0,:]).T
dor = D_lenslet/r0
std_var = 2*var/np.sqrt(2*REPEAT - 2)

for i in np.arange(len(r0)):
    plt.errorbar(fov, var[:,i], yerr=std_var[:,i],
                  marker='.', ls='',
              label='d/r0 = {:.2f}'.format(dor[i]))
# for i in np.arange(len(D_lenslet)):
#     plt.errorbar(fov, mean[:,i], yerr=np.sqrt(var[:,i]),
#                   marker='.', ls='',
#               label='d/r0 = {:.2f}'.format(dor[i]))
for i in np.arange(len(r0)):
    if i == 0:
        plt.plot(fov, 0.5/fov**2*dor[i]**(5./3.), marker='', ls='-', color='k',
                  label='fit')
    else:
        plt.plot(fov, 0.5/fov**2*dor[i]**(5./3.), marker='', ls='-', color='k')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Field-of-View ($\lambda$/d)')
plt.ylabel('Variance (rad$^2$)')
plt.xlim(1,100)
plt.ylim(1e-4,10)
plt.savefig('centroidThomas2006.png')
plt.show()
plt.clf()

# for i in np.arange(len(D_lenslet)):
#     for repeat in np.arange(int(5)):
#         plt.plot(fov, np.abs(dif_SHWFS_slope[i,0,repeat,:,0]), marker='.', alpha=0.5)
#         plt.xscale('log')
#         plt.yscale('log')
#         plt.ylim(0.001,10)
#     plt.title('{:.2f} d/r0'.format(dor[i]))
#     plt.show()
#     plt.clf()