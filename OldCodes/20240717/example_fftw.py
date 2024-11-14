# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:17:32 2022

@author: gqmc62
"""

import pyfftw
import time
import numpy as np
from matplotlib import pyplot as plt

N = int(256)

a = np.arange(-int(N/2), int(N/2), dtype=complex)
aa, bb = np.meshgrid(a, a)
x1 = np.sin(aa) + np.cos(bb/4)

A = time.time()
for i in np.arange(100):
    y1 = np.fft.fftshift(np.fft.fft2(x1))
B = time.time()
print('numpy time is {:.3f}ms'.format((B - A)*1000/10000))

# plt.imshow(y1.real)
# plt.colorbar()
# plt.show()

# plt.imshow(y1.imag)
# plt.colorbar()
# plt.show()



x2 = pyfftw.empty_aligned((N, N), dtype=complex)
x2 = np.sin(aa) + np.cos(bb/4)
y2 = pyfftw.empty_aligned((N, N), dtype=complex)
z2 = pyfftw.FFTW(x2, y2)

C = time.time()
for i in np.arange(100):
    z2.execute()
D = time.time()
print('fftw time is {:.3f}ms'.format((D - C)*1000/10000))

# plt.imshow(y2.real)
# plt.colorbar()
# plt.show()

# plt.imshow(y2.imag)
# plt.colorbar()
# plt.show()

