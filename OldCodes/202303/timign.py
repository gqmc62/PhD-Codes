
###############################################################################
##########################################################################
"""
█▀▀█░█░▒█░▀█▀░▀█▀░░▀░░█░░░█░█▀▀▄░▀█▀░░░▒█░▄▀░▄▀▀▄░█▀▀▄░█▀▀▀░█░▄░█▀▀▄░█▀▀░█░░░█
█▄▄█░█░▒█░░█░░░█░░░█▀░▀▄█▄▀░█▄▄█░░█░░░░▒█▀▄░░█░░█░█░▒█░█░▀▄░█▀▄░█▄▄█░█▀▀░▀▄█▄▀
█░░░░░▀▀▀░░▀░░░▀░░▀▀▀░░▀░▀░░▀░░▀░░▀░░░░▒█░▒█░░▀▀░░▀░░▀░▀▀▀▀░▀░▀░▀░░▀░▀▀▀░░▀░▀░
"""
##########################################################################
###############################################################################

import numpy as np
import pickle
from matplotlib import pyplot as plt
import pyfftw
import timeit

###############################################################################

def Ig(U, phase, I, gx, gy):
    
    phase = np.angle(U)
    I = np.abs(U)**2
    
    gx = phase[1 : ,   : ] - phase[ : -1, :   ]
    gy = phase[  : , 1 : ] - phase[ :   , : -1]
    
    gx[gx > np.pi] = gx[gx > np.pi] - (2*np.pi)
    gx[gx < -np.pi] = gx[gx < -np.pi] + (2*np.pi)
    gy[gy > np.pi] = gy[gy > np.pi] - (2*np.pi)
    gy[gy < -np.pi] = gy[gy < -np.pi] + (2*np.pi)

    Ig0 = np.sum(gx * (I[1 : , : ] + I[ : -1, : ]))
    Ig1 = np.sum(gy * (I[ : , 1 : ] + I[ : , : -1]))
    
    return np.asarray([Ig0, Ig1]) * (U.shape[0]/(I.sum()*2))

def SHWFS(U, Q, FF, xwfs, ywfs, Im, Threshold):
    n = U.shape[0]
    a = int(n/2)
    b = int(3*n/2)
    Im = np.abs(np.fft.fftshift(
        FF(np.pad(U*Q, (a, a), 'constant', constant_values=0.))
        )[a : b, a : b])**2
    
    # Im = (Im - np.max([Im[0, : ], Im[-1, : ], Im[ : , 0], Im[ : , -1]]))/Im.max()
    
    # Im = (Im - np.max([Im[0, : ], Im[-1, : ], Im[ : , 0], Im[ : , -1]]))/Im.max() - Threshold
    
    Im = (Im/Im.max() - Threshold) # this is good
    
    Im[Im < 0] = 0.
    Im = Im / Im.max()

    SHWFS0 = np.sum(Im * xwfs)
    SHWFS1 = np.sum(Im * ywfs)
    
    return np.asarray([SHWFS0, SHWFS1]) * (np.pi/Im.sum())


###############################################################################

# set up shwfs propagation here
# also any memory allocation
N = 2**(np.arange(10)+1)
t_Ig = np.zeros_like(N, dtype=float)
t_Ig_error = np.zeros_like(N, dtype=float)
t_SHWFS = np.zeros_like(N, dtype=float)
t_SHWFS_error = np.zeros_like(N, dtype=float)
# for n in N:
for power in (np.arange(10)+1):
    n = 2**power
    Threshold = 0
    U = np.ones((n, n), dtype=complex)
    I = np.ones((n, n), dtype=float)
    phase = np.ones((n, n), dtype=float)
    g0 = np.zeros((n - 1, n), dtype=float)
    g1 = np.zeros((n, n - 1), dtype=float)

    F0 = pyfftw.empty_aligned((2*n, 2*n), dtype=complex)
    F1 = pyfftw.empty_aligned((2*n, 2*n), dtype=complex)
    FF01 = pyfftw.FFTW(F0, F1, axes=(-1,-2), direction='FFTW_FORWARD')
    
    (y1, x1) = np.meshgrid(np.arange(-n/2, n/2),
                           np.arange(-n/2, n/2))
    Q = np.exp(-1j*(x1 + y1)*np.pi/(2*n))
    
    gx = np.zeros((n - 1, n), dtype=float)
    gy = np.zeros((n, n - 1), dtype=float)
    Im = np.zeros((n, n), dtype=float)
    
    (ywfs, xwfs) = np.meshgrid(np.arange(n) + 0.5 - n/2,
                               np.arange(n) + 0.5 - n/2)
    
    # t_Ig[power] = 
    # t_Ig_error[power] = 
    t_SHWFS[power-1] = timeit.timeit(
        'SHWFS(U, Q, FF01, xwfs, ywfs, Im, Threshold)',
        setup="from __main__ import SHWFS, U, Q, FF01, xwfs, ywfs, Im, Threshold", number=10000)/10000
    t_Ig[power-1] = timeit.timeit(
        'Ig(U, phase, I, gx, gy)',
        setup="from __main__ import Ig, U, phase, I, gx, gy", number=10000)/10000
    # t_SHWFS_error[power] = 
    
    # print(n)
    # print(%timeit Ig(U, phase, I, gx, gy))
    # print(%timeit SHWFS(U, Q, FF01, xwfs, ywfs, Im, Threshold))

# N = np.array([2,4,8,16,32,64,128,256,512,1024])
# t_Ig = np.array([31.3,33.6,32.3,51.1,51.4,91.1,261,1310,10700,51700])*1e-6
# t_Ig_error = np.array([2.05,2.74,1.58,2.87,2.41,3.19,9.35,159,758,5090])*1e-6
# t_SHWFS = np.array([69.7,66.9,73.8,83.8,102,228,1290,6760,36800,157000])*1e-6
# t_SHWFS_error = np.array([4.81,2.71,4.52,4.23,2.7,18.3,57.9,730,2860,12000])*1e-6

# plt.errorbar(N, t_Ig, yerr=t_Ig_error, label='Ig')
# plt.errorbar(N, t_SHWFS, yerr=t_SHWFS_error, label='SHWFS')
plt.plot(N, t_Ig, label='Ig')
plt.plot(N, t_SHWFS, label='SHWFS')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('N (pixel)')
plt.ylabel('time ($\mu$s)')
plt.show()

# plt.errorbar(N,t_SHWFS/t_Ig,
#              yerr=np.sqrt((t_Ig_error/t_Ig)**2
#                           + (t_SHWFS_error/t_SHWFS)**2)*(t_SHWFS/t_Ig)
#              , marker='o')
plt.plot(N, t_SHWFS/t_Ig, marker='o')
plt.xscale('log')
plt.xlabel('N (pixel)')
plt.ylabel('SHWFS time / Ig time')
plt.show()
