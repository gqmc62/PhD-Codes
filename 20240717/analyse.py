
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
    I_sum = I.sum()
    
    gx = phase[1 : ,   : ] - phase[ : -1, :   ]
    gy = phase[  : , 1 : ] - phase[ :   , : -1]
    
    gx[gx > np.pi] = gx[gx > np.pi] - 2*np.pi
    gx[gx < -np.pi] = gx[gx < -np.pi] + 2*np.pi
    gy[gy > np.pi] = gy[gy > np.pi] - 2*np.pi
    gy[gy < -np.pi] = gy[gy < -np.pi] + 2*np.pi

    Ig0 = np.sum(gx * (I[1 : , : ] + I[ : -1, : ])/2) / I_sum
    Ig1 = np.sum(gy * (I[ : , 1 : ] + I[ : , : -1])/2) / I_sum
    
    return np.asarray([Ig0, Ig1])*U.shape[0]

def SHWFS(U, Q, FF, xwfs, ywfs, Im, Threshold):
    n = U.shape[0]
    a = int(n/2)
    b = int(3*n/2)
    Im = np.abs(np.fft.fftshift(
        FF(np.pad(U*Q, (a, a), 'constant', constant_values=0.))
        )[a : b, a : b])**2
    
    # Im = (Im - np.max([Im[0, : ], Im[-1, : ], Im[ : , 0], Im[ : , -1]]))/Im.max()
    
    # Im = (Im - np.max([Im[0, : ], Im[-1, : ], Im[ : , 0], Im[ : , -1]]))/Im.max() - Threshold
    
    Im = (Im/Im.max() - Threshold)
    
    Im[Im < 0] = 0.
    Im = Im / Im.max()

    Im_sum = np.sum(Im)
    SHWFS0 = np.sum(Im * xwfs)/Im_sum
    SHWFS1 = np.sum(Im * ywfs)/Im_sum
    
    return np.asarray([SHWFS0, SHWFS1])*np.pi

###############################################################################

with open('Ultimate_input.pkl', 'rb') as file:
    myvar = pickle.load(file)
    TARGET = myvar['TARGET']
    SCREEN_LIMIT = myvar['SCREEN_LIMIT']
    n= myvar['n']
    oversize  = myvar['oversize']
    L0_GL = myvar['L0_GL']
    L0_HA = myvar['L0_HA']
    l0_GL = myvar['l0_GL']
    l0_HA = myvar['l0_HA']
    wvl  = myvar['wvl']
    wavenumber = myvar['wavenumber']
    zenith_angle  = myvar['zenith_angle']
    r0 = myvar['r0']
    dor0   = myvar['dor0']
    Rytov  = myvar['Rytov']
    H_HA = myvar['H_HA']
    II = myvar['II']
    JJ   = myvar['JJ']
    KK = myvar['KK']
    LL  = myvar['LL']
    wavenumber  = myvar['wavenumber']
    N_MAX = myvar['N_MAX']
    d= myvar['d']
    delta = myvar['delta']
    r0_GL = myvar['r0_GL']
    r0_HA  = myvar['r0_HA']
      
    N = myvar['N']
    num_lenslet = myvar['num_lenslet']
    REPEAT = myvar['REPEAT']
    POSSIBLE_SCREEN = myvar['POSSIBLE_SCREEN']
    POSSIBLE_PROP = myvar['POSSIBLE_PROP']
    POSSIBLE = myvar['POSSIBLE']

###############################################################################

# set up shwfs propagation here
# also any memory allocation

U = np.zeros((n, n), dtype=complex)
I = np.zeros((n, n), dtype=float)
phase = np.zeros((n, n), dtype=float)
g0 = np.zeros((n - 1, n), dtype=float)
g1 = np.zeros((n, n - 1), dtype=float)

SHWFS_SLOPE = np.zeros((II, JJ, KK, LL, TARGET, 2), dtype=float)
Ig_SLOPE = np.zeros((II, JJ, KK, LL, TARGET, 2), dtype=float)

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

Threshold = 0

# print('Ig-tilt time',
#       timeit.timeit('[Ig(U, A, B, a, b)]', globals=globals()))
# print('SHWFS-tilt time',
#       timeit.timeit('[SHWFS(U, Q, FF01, xwfs, ywfs, AA)]', globals=globals()))

###############################################################################

# for ii in [2]:
#     for jj in [3]:
#         for kk in [3]:
#             for ll in [3]:

for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                run_name = 'r{:02d}dor{:02d}rytov{:02d}h{:02d}'.format(
                    int(r0[ii]*100), int(dor0[jj]*10), int(Rytov[kk]*100),
                    int(H_HA[ll]/1000))
                for count in np.arange(TARGET):
                    with open(run_name + '/U_tot/U_tot_' + run_name +
                              '_{:04d}.pkl'.format(count), 'rb') as file:
                        U = pickle.load(file)
                    Ig_SLOPE[ii, jj, kk, ll, count , :
                             ] = Ig(U, phase, I, gx, gy)
                    SHWFS_SLOPE[ii, jj, kk, ll, count , :
                                ] = SHWFS(U, Q, FF01, xwfs, ywfs, Im, Threshold)

# plt.plot(Ig_SLOPE[ii, jj, kk, ll, : , :].flatten(),
#           SHWFS_SLOPE[ii, jj, kk, ll, : , :].flatten(),
#           ls='', marker='o', alpha=0.5)
# plt.show()

# plt.plot(Ig_SLOPE[ii, jj, kk, ll, : , :].flatten(),
#           SHWFS_SLOPE[ii, jj, kk, ll, : , :].flatten()
#           - Ig_SLOPE[ii, jj, kk, ll, : , :].flatten(),
#           ls='', marker='.', alpha=0.1)
# plt.show()

myvar = {'Ig_SLOPE' : Ig_SLOPE,
         'SHWFS_SLOPE' : SHWFS_SLOPE}

with open('Ultimate_output.pkl', 'wb') as file:
    pickle.dump(myvar, file)
    file.close()

###############################################################################
##########################################################################
"""
█▀▀█░█░▒█░▀█▀░▀█▀░░▀░░█░░░█░█▀▀▄░▀█▀░░░▒█░▄▀░▄▀▀▄░█▀▀▄░█▀▀▀░█░▄░█▀▀▄░█▀▀░█░░░█
█▄▄█░█░▒█░░█░░░█░░░█▀░▀▄█▄▀░█▄▄█░░█░░░░▒█▀▄░░█░░█░█░▒█░█░▀▄░█▀▄░█▄▄█░█▀▀░▀▄█▄▀
█░░░░░▀▀▀░░▀░░░▀░░▀▀▀░░▀░▀░░▀░░▀░░▀░░░░▒█░▒█░░▀▀░░▀░░▀░▀▀▀▀░▀░▀░▀░░▀░▀▀▀░░▀░▀░
"""
##########################################################################
###############################################################################
