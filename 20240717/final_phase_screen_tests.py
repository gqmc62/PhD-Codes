# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 10:23:20 2023

@author: gqmc62
"""

from aotools.turbulence.infinitephasescreen import PhaseScreenVonKarman
from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov
from aotools.turbulence.phasescreen import ft_phase_screen
from aotools.turbulence.phasescreen import ft_sh_phase_screen
from aotools.functions import zernike
from tqdm import tqdm
import numpy as np
import soapy
from matplotlib import pyplot as plt
from matplotlib import colors
from astropy.io import fits
from scipy.special import gamma, kv

import my_propagation as mprop
import my_phasescreen as mps

import os

###############################################################################
########################################################################

N = 128
D = 0.4
# nxSubaps = 8
# pxlsPerSubap = N//nxSubaps

r0 = 5e-2
L0 = 10
l0 = 1e-10

n_modes = 32 + 1

RANDOM = 10000

# d = D/nxSubaps
pixel_scale = D/N

phase = np.zeros((RANDOM,N,N), dtype=float)

infinite = True
if infinite == True:
    
    vonKarman = True
    
    if vonKarman == True:
        Kolmogorov = False
    else:
        Kolmogorov = True

sh = False

flush_list = [1]
oversize_list = [1]

###############################################################################
########################################################################

if not os.path.isdir('outputTestPhaseScreen'):
    os.makedirs('outputTestPhaseScreen')
# if not os.path.isdir('PhaseScreen'):
#     os.makedirs('PhaseScreen')

for oversize in oversize_list:
    for flush in flush_list:
        name = ('{:d}'.format(RANDOM)
                + 'Phase'
                + ('Infinite' + 'VonKarman'*vonKarman + 'Kolmogorov'*Kolmogorov)*infinite
                + 'ft'*(infinite == False)
                + 'sh'*sh
                + 'x{:d}Oversize'.format(oversize)
                + 'x{:d}Flush'.format(flush)*infinite
                + 'D{:.0f}cm'.format(D*100)
                + 'r0{:.0f}cm'.format(r0*100)
                + 'L0{:.0f}m'.format(L0))
        
        for random in tqdm(np.arange(RANDOM)):
            if infinite == True:
                if vonKarman == True:
                    infinitePhaseScreen = PhaseScreenVonKarman(oversize*N, pixel_scale, r0, L0)
                else:
                    if Kolmogorov == True:
                        infinitePhaseScreen = PhaseScreenKolmogorov(oversize*N, pixel_scale, r0, L0)
                if random == 0:
                    for row in np.arange(flush*oversize*N):
                        infinitePhaseScreen.add_row()
                for row in np.arange(flush*oversize*N):
                    infinitePhaseScreen.add_row()
                phase[random] = mprop.crop2(infinitePhaseScreen.scrn, oversize)
            if infinite == False:
                if sh == True:
                    phase[random] = mprop.crop2(ft_sh_phase_screen(r0, oversize*N, pixel_scale, L0, l0), oversize)
                if sh == False:
                    phase[random] = mprop.crop2(ft_phase_screen(r0, oversize*N, pixel_scale, L0, l0), oversize)
            if random == 0:
                plt.imshow(phase[random])
                plt.colorbar()
                plt.title('{:d}'.format(RANDOM)
                        + 'Phase'
                        + ('Infinite' + 'VonKarman'*vonKarman + 'Kolmogorov'*Kolmogorov)*infinite
                        + 'ft'*(infinite == False)
                        + 'sh'*sh
                        + 'x{:d}Oversize'.format(oversize)
                        + 'x{:d}Flush'.format(flush)*infinite
                        + 'D{:.0f}cm'.format(D*100)
                        + 'r0{:.0f}cm'.format(r0*100)
                        + 'L0{:.0f}m'.format(L0))
                plt.show()
                nb = input('\nProceed?: ')
                if nb != 'y':
                    quit()
        
        hdu = fits.PrimaryHDU(phase)
        hdul = fits.HDUList([hdu])
        hdr = hdul[0].header
        hdr['DIAM'] = D
        hdr['FRIED'] = r0
        hdr['OUTERL'] = L0
        hdr['INNERL'] = l0
        hdul.writeto(name + '.fits', overwrite=True)
        hdul.close()
        
        
        mps.testPhaseScreen(name + '.fits', saveFile=False, saveFigure=False)


###############################################################################
########################################################################



    
