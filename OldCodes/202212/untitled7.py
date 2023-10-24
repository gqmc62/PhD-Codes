# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 18:10:58 2022

@author: gqmc62
"""

import timeit
import numpy as np
import my_WFS as mwfs

mycode1 = '''
import numpy as np
import my_WFS as mwfs
I = np.abs(U)**2
noisy_I = np.random.poisson(
    I/lenslet_pixel**2*SNR**2) / SNR**2*lenslet_pixel**2
(mwfs.get_vec_lenslet(mwfs.get_slope(np.angle(U)) * noisy_I, lenslet_pixel)
        / mwfs.get_val_lenslet(noisy_I, lenslet_pixel))
'''
mycode2 = '''
num_lenslet = int(inputComplexAmp.shape[-1]/lenslet_pixel)
chop = np.asarray(np.split(
    np.asarray(np.split(
        inputComplexAmp, num_lenslet, axis=-1)),
    num_lenslet, axis=-2))
xx, yy = np.meshgrid(np.arange(lenslet_pixel), np.arange(lenslet_pixel))
slope = np.zeros((2, num_lenslet, num_lenslet), dtype=float)
m = outputSpacing/inputSpacing
for i in np.arange(num_lenslet):
    for j in np.arange(num_lenslet):
        I = np.abs(focus(chop[i,j], wvl,
                         inputSpacing, outputSpacing, z, pf))**2*m**2
        noisy_I = np.random.poisson(
            I/lenslet_pixel**2*SNR**2) / SNR**2*lenslet_pixel**2
        noisy_I_total = np.sum(noisy_I.flatten())
        # this slope is deviation angle of spots
        slope[0, i, j] = np.sum((noisy_I*xx).flatten())/noisy_I_total
        slope[1, i, j] = np.sum((noisy_I*yy).flatten())/noisy_I_total
#           recenter the spot          change px to m  get angle
((slope - (lenslet_pixel - 1)/2.) * outputSpacing / z
        * 2. * np.pi / wvl)
    #   convert m to phase gradient (rad/m) @ SHWFS lenslet
'''
mycode3 = '''
a + b
'''

print(timeit.timeit(stmt=mycode1, number=10000))
print(timeit.timeit(stmt=mycode2, number=10000))
print(timeit.timeit(stmt=mycode3, number=10000))