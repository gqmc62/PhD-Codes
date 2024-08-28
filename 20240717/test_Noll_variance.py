# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 16:20:59 2023

@author: gqmc62
"""

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

# ta = time.time()

# infinite = False
# myscreen = False
# sh = False
# gram_schmidt = True

# ovf = 1

# PUPIL = 0.4
# N = 128
# delta = PUPIL/N
# L0 = 10
# r0 = 5e-2

# flush = 1
# n_random = 10000

# n, noll, var, err_of_var = mps.zernike_modes_strength(
#     r0, N, delta, L0, 1.5e-10, 32, n_random,
#     ovf=ovf, infinite=infinite, myscreen=myscreen, sh=sh,
#     gram_schmidt=gram_schmidt, FFT=None, flush=flush)

# plt.plot(n, noll, label='Noll variance')
# plt.errorbar(n, var, yerr=err_of_var, label='Measurement', ls='',
#              marker='o')
# # correction see sasiela 2007 book
# # Electromagnetic wave propagation in turbulence : evaluation and application of Mellin transforms
# # 2nd edition
# # chapter 6 page 148-149 fig 6.2 equation 6.8
# e = N*delta/L0
# adjust = (1 - 1.42*e**(1./3.) + 3.70*e**2 - 4.01*e**(7./3.) + 4.21*e**4 - 4.00*e**(13./3.))
# adjustedTipTilt = noll[:2] * adjust
# plt.plot(n[:2],adjustedTipTilt[:2], label='L0 adjust')
# plt.yscale('log')
# plt.title('Comparison of variance of each Zernike modes\n'
#           + 'between Nolls theory and measurement, {:d} repeats\n'.format(n_random)
#           + 'myscreen, '*myscreen
#           + 'infinite flush by {:.2f}m, '.format(flush*N*delta)*infinite + 'sh, '*sh
#           + 'oversize by {:d}, '.format(ovf) + 'gram-schmidt,'*gram_schmidt + '\n'
#           + '{:d}x{:d} pixel, '.format(N,N)
#           + 'pixel scale {:.2f}mm, '.format(delta*1000)
#           + 'screen size {:.2f}m, '.format(N*delta)
#           + 'L0 {:.0f}m, '.format(L0) + 'r0 {:.1f}cm'.format(r0*100))
# plt.xlabel('Zernike modes')
# plt.ylabel('Variance of each Zernike mode (rad^2)')
# plt.xlim(0,33)
# plt.ylim(noll[0:32].min()*0.8, noll[0:32].max()*1.2)
# plt.grid()
# plt.legend()
# plt.show()
# tb = time.time()
# print(tb-ta)

###############################################################################
########################################################################

infinite = True
myscreen = False
sh = False
gram_schmidt = False

ovf = 1

PUPIL = 0.4

L0 = 10
r0 = 5e-2

flush = 1
n_random = 10000

N_list = [8,16,32,64,128]

for N in N_list:
    delta = PUPIL/N
    
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

infinitePhaseScreen = PhaseScreenVonKarman(128, 3.125e-3, 5e-2, 10)
tiptiltstd = -(0.4479-0.4480)*(0.4/0.05)**(5./3.)
x = np.arange(128) - 64 + 0.5
yy, xx = np.meshgrid(x,x)
xx /= np.sqrt(np.sum(xx*xx))
yy /= np.sqrt(np.sum(yy*yy))

# %timeit for row in np.arange(128):infinitePhaseScreen.add_row() # 2.94ms +- 124us

# %timeit ft_phase_screen(5e-2, 128, 3.125e-3, 10, 1e-10) # 1.4ms +- 59.4us

# %timeit ft_sh_phase_screen(5e-2, 128, 3.125e-3, 10, 1e-10) # 16.1ms +- 451us

# %timeit mprop.crop2(ft_phase_screen(5e-2, 2*128, 3.125e-3, 10, 1e-10),2) # 8ms +- 117us

# %timeit mprop.crop2(ft_sh_phase_screen(5e-2, 2*128, 3.125e-3, 10, 1e-10),2) # 89ms +- 2.75ms

# this is not exact method but with the same step & type of calculation
# forgot to takout initial tip/tilt
# %timeit phasescreen = mprop.crop2(ft_phase_screen(5e-2, 2*128, 3.125e-3, 10, 1e-10),2);tip=np.sum(xx*phasescreen);tilt=np.sum(yy*phasescreen);phasescreen + xx*(np.random.normal(0,tiptiltstd)-tip) + yy*(np.random.normal(0,tiptiltstd)-tilt)
# 8.77ms +- 389 us