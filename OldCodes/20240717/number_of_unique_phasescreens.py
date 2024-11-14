# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:27:28 2023

@author: gqmc62
"""

# import os
# import soapy
# import numpy as np
# from matplotlib import pyplot as plt
# from astropy.io import fits
# from scipy.optimize import lsq_linear
# from scipy.stats import bootstrap
# from tqdm import tqdm
# from aotools.turbulence.infinitephasescreen import PhaseScreenVonKarman

# # def remove_piston_tiptilt(scrns):
# #     yy, xx = np.meshgrid(np.arange(len(scrns)),np.arange(len(scrns)))
# #     norm = np.sum(xx, axis=(0,1))
# #     temp = scrns - np.mean(scrns, axis=(0,1))
# #     tip = np.sum(temp*yy, axis=(0,1))/norm**2
# #     tilt = np.sum(temp*xx, axis=(0,1))/norm**2
# #     temp -= (tip*yy + tilt*xx)
# #     return temp

# class Clean():
#     def __init__(self, N):
#         x = np.arange(N, dtype=float) - N/2 + 0.5
#         self.yy, self.xx = np.meshgrid(x, x)
#         norm = np.sqrt(np.dot((self.xx).flatten(),(self.xx).flatten()))
#         self.xx /= norm
#         self.yy /= norm
#         del x, norm
        
#     def clean(self, scrn):
#         # plt.imshow(scrn)
#         # plt.colorbar()
#         # plt.title('input')
#         # plt.show()
        
#         clean_scrn = scrn - np.mean(scrn, axis=(0,1))
        
#         # plt.imshow(clean_scrn)
#         # plt.colorbar()
#         # plt.title('piston removed')
#         # plt.show()
        
#         tip = np.dot(clean_scrn.flatten(), (self.xx).flatten())
#         tilt = np.dot(clean_scrn.flatten(), (self.yy).flatten())
#         # tiptilt = (self.xx*tip + self.yy*tilt)
#         # clean_scrn -= tiptilt
#         clean_scrn -= (self.xx*tip + self.yy*tilt)
        
#         # plt.imshow(clean_scrn)
#         # plt.title('tiptilt removed')
#         # plt.colorbar()
#         # plt.show()
        
#         return clean_scrn

# N = 128
# telDiam = 0.4
# pixel_scale = telDiam/N
# r0 = 0.05
# L0 = 10
# TIME = 10000

# phasescreen = PhaseScreenVonKarman(N, pixel_scale, r0, L0)
# myClean = Clean(N)

# scrns = np.zeros((TIME,N,N), dtype=float)

# # for time in tqdm(np.arange(TIME)):
# #     for i in np.arange(N):
# #         phasescreen.add_row()
# #     scrns[time] = myClean.clean(phasescreen.scrn)
# #     # plt.imshow(phasescreen.scrn)
# #     # plt.colorbar()
# #     # plt.title('unclean{:d}'.format(iteration))
# #     # plt.show()
# #     # plt.imshow(myClean.clean(phasescreen.scrn))
# #     # plt.colorbar()
# #     # plt.title('clean{:d}'.format(iteration))
# #     # plt.show()

# # name = 'clean_scrns'
# # hdu = fits.PrimaryHDU(scrns)
# # hdul = fits.HDUList([hdu])
# # hdul.writeto(name + '.fits', overwrite=True)
# # hdul.close()

# name = 'clean_scrns'
# hdulist = fits.open(name + '.fits')
# hdu = hdulist[0]
# scrns = hdu.data
# hdulist.close()

# # coef = np.corrcoef(scrns.reshape((TIME,N*N)))

# # for i in [10,100,1000,10000]:
# #     plt.imshow(coef[:i,:i])
# #     plt.colorbar()
# #     plt.show()

# # offaxisstd = np.zeros((4),dtype=float)

# # for i in np.arange(4):
# #     val = int(10**(i + 1))
# #     count = 0
# #     for j in np.arange(val - 1) + 1:
# #         for k in np.arange(j - 1) + 1:
# #             # print(coef[j,k])
# #             offaxisstd[i] += coef[j, k]**2
# #             count += 1
# #             if j == k:
# #                 print('error')
# #             if k > j:
# #                 print('error')
# #     offaxisstd[i] /= count #((val - 1)**2/2. - 1.)
# #     offaxisstd[i] = np.sqrt(offaxisstd[i])

# # plt.plot([10,100,1000,10000],offaxisstd)
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.show()

# convergence = np.zeros((4), dtype=float)
# ref = np.mean(np.std(scrns,axis=(1,2)))
# for i in np.arange(4):
#     val = int(10**(i + 1))
#     convergence[i] = np.std(np.mean(scrns[:val],axis=0), ddof=1)

# plt.plot([10,100,1000,10000],convergence/ref,marker='o')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('number of unique phase screens')
# plt.ylabel('std(mean(screens)) / mean(std(screens))')
# plt.title('roughness of averaged screens normalized to averaged roughness of screens')
# plt.show()


import numpy as np
from aotools.functions import zernike
from aotools.turbulence.phasescreen import ft_sh_phase_screen
from aotools.turbulence.phasescreen import ft_phase_screen
from aotools.turbulence.infinitephasescreen import PhaseScreenVonKarman
import MyScreen
from tqdm import tqdm
import my_phasescreen as mps


from matplotlib import pyplot as plt
import my_propagation as mprop

import time

###############################################################################
########################################################################

infinite = True
myscreen = False
sh = False
gram_schmidt = True

ovf = 1

PUPIL = 0.4

L0 = 10
r0 = 5e-2

flush = 1
# n_random = 10000

N = 16
delta = PUPIL/N
n_random_list = [10,100,1000,10000]

for n_random in n_random_list:
    ta = time.time()
    n, noll, var, err_of_var = mps.zernike_modes_strength(
        r0, N, delta, L0, 1.5e-10, 32, n_random,
        ovf=ovf, infinite=infinite, myscreen=myscreen, sh=sh,
        gram_schmidt=gram_schmidt, FFT=None, flush=flush)
    
    plt.plot(n, noll, label='Noll variance')
    plt.errorbar(n, var, yerr=err_of_var, label='Measurement', ls='',
                 marker='o')
    # correction see sasiela 2007 book
    # Electromagnetic wave propagation in turbulence : evaluation and application of Mellin transforms
    # 2nd edition
    # chapter 6 page 148-149 fig 6.2 equation 6.8
    e = N*delta/L0
    adjust = (1 - 1.42*e**(1./3.) + 3.70*e**2 - 4.01*e**(7./3.) + 4.21*e**4 - 4.00*e**(13./3.))
    adjustedTipTilt = noll[:2] * adjust
    plt.plot(n[:2],adjustedTipTilt[:2], label='L0 adjust')
    plt.yscale('log')
    plt.title('Comparison of variance of each Zernike modes\n'
              + 'between Nolls theory and measurement, {:d} repeats\n'.format(n_random)
              + 'myscreen, '*myscreen
              + 'infinite flush by {:.2f}m, '.format(flush*N*delta)*infinite + 'sh, '*sh
              + 'oversize by {:d}, '.format(ovf) + 'gram-schmidt,'*gram_schmidt + '\n'
              + '{:d}x{:d} pixel, '.format(N,N)
              + 'pixel scale {:.2f}mm, '.format(delta*1000)
              + 'screen size {:.2f}m, '.format(N*delta)
              + 'L0 {:.0f}m, '.format(L0) + 'r0 {:.1f}cm'.format(r0*100))
    plt.xlabel('Zernike modes')
    plt.ylabel('Variance of each Zernike mode (rad^2)')
    plt.xlim(0,33)
    plt.ylim(noll[0:32].min()*0.8, noll[0:32].max()*1.2)
    plt.grid()
    plt.legend()
    plt.show()
    tb = time.time()
    print(tb-ta)

###############################################################################
########################################################################

infinite = True
myscreen = False
sh = False
gram_schmidt = True

ovf = 1

PUPIL = 0.4

L0 = 10
r0 = 5e-2

flush = 1
# n_random = 10000

N = 128
delta = PUPIL/N
n_random_list = [10,100,1000,10000]

for n_random in n_random_list:
    ta = time.time()
    n, noll, var, err_of_var = mps.zernike_modes_strength(
        r0, N, delta, L0, 1.5e-10, 32, n_random,
        ovf=ovf, infinite=infinite, myscreen=myscreen, sh=sh,
        gram_schmidt=gram_schmidt, FFT=None, flush=flush)
    
    plt.plot(n, noll, label='Noll variance')
    plt.errorbar(n, var, yerr=err_of_var, label='Measurement', ls='',
                 marker='o')
    # correction see sasiela 2007 book
    # Electromagnetic wave propagation in turbulence : evaluation and application of Mellin transforms
    # 2nd edition
    # chapter 6 page 148-149 fig 6.2 equation 6.8
    e = N*delta/L0
    adjust = (1 - 1.42*e**(1./3.) + 3.70*e**2 - 4.01*e**(7./3.) + 4.21*e**4 - 4.00*e**(13./3.))
    adjustedTipTilt = noll[:2] * adjust
    plt.plot(n[:2],adjustedTipTilt[:2], label='L0 adjust')
    plt.yscale('log')
    plt.title('Comparison of variance of each Zernike modes\n'
              + 'between Nolls theory and measurement, {:d} repeats\n'.format(n_random)
              + 'myscreen, '*myscreen
              + 'infinite flush by {:.2f}m, '.format(flush*N*delta)*infinite + 'sh, '*sh
              + 'oversize by {:d}, '.format(ovf) + 'gram-schmidt,'*gram_schmidt + '\n'
              + '{:d}x{:d} pixel, '.format(N,N)
              + 'pixel scale {:.2f}mm, '.format(delta*1000)
              + 'screen size {:.2f}m, '.format(N*delta)
              + 'L0 {:.0f}m, '.format(L0) + 'r0 {:.1f}cm'.format(r0*100))
    plt.xlabel('Zernike modes')
    plt.ylabel('Variance of each Zernike mode (rad^2)')
    plt.xlim(0,33)
    plt.ylim(noll[0:32].min()*0.8, noll[0:32].max()*1.2)
    plt.grid()
    plt.legend()
    plt.show()
    tb = time.time()
    print(tb-ta)


