# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:20:54 2023

@author: gqmc62
"""

import numpy as np
import pyfftw
# from aotools.functions.zernike import zernikeArray
from matplotlib import pyplot as plt
from aotools.turbulence.phasescreen import ft_sh_phase_screen
from aotools.opticalpropagation import lensAgainst
import my_propagation as mprop
from aotools.opticalpropagation import angularSpectrum

N = 64
r0 = 0.05
d = 0.15
L0 = 2000
l0 = 1e-12
wvl = 500e-9
zeropad = 16
delta = d/N
f_1step = zeropad * delta**2 * N / wvl

f = 2 * delta**2 * N / wvl

wavenumber = 2*np.pi/wvl

(y1, x1) = np.meshgrid(delta*np.arange(-N/2,N/2),
                      delta*np.arange(-N/2,N/2))

phase = ft_sh_phase_screen(r0, N, delta, L0, l0)
# phase = np.zeros((N,N), dtype=float)

# U2 = mprop.crop2(lensAgainst(mprop.zero_pad2(np.exp(1j*(
#     phase
#     - wavenumber * (x1 + y1) * ((delta/2)/f_1step)
#     )), zeropad), wvl, delta, f_1step), zeropad)
# image = np.abs(U2)**2
# plt.imshow((image/image.max()))
# plt.title('1step')
# plt.colorbar()
# plt.show()

zeropad = 2

U2 = mprop.crop2(angularSpectrum(mprop.zero_pad2(np.exp(1j*(
    phase
    - wavenumber * (x1 + y1) * ((delta/2)/f)
    - wavenumber / 2 / f * (x1**2 + y1**2)
    )), zeropad),
    wvl, delta, delta, f), zeropad)

image = np.abs(U2)**2

ideal = image/image.max()

for zeropad in np.array([1,2,4,8,16],dtype=int):
    U2 = mprop.crop2(angularSpectrum(mprop.zero_pad2(np.exp(1j*(
        phase
        - wavenumber * (x1 + y1) * ((delta/2)/f)
        - wavenumber / 2 / f * (x1**2 + y1**2)
        )), zeropad),
        wvl, delta, delta, f), zeropad)
    
    image = np.abs(U2)**2
    plt.imshow((image/image.max()))
    plt.colorbar()
    plt.title('angular-spectrum, zeropad{:d}'.format(zeropad))
    plt.show()

    # plt.imshow((image/image.max()) - ideal)
    # plt.colorbar()
    # plt.title('residue, zeropad{:d}'.format(zeropad))
    # plt.show()

