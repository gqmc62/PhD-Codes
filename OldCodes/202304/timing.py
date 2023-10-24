
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
from tqdm import tqdm

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

def SHWFS2(U, Q, FF2, xwfs, ywfs, Im, Threshold):
    n = U.shape[0]
    a = int(n/2)
    b = int(3*n/2)
    Im = np.abs(np.fft.fftshift(
        FF2(np.pad(U*Q, (a, a), 'constant', constant_values=0.))
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

def SHWFS4(U, Q, FF4, xwfs, ywfs, Im, Threshold):
    n = U.shape[0]
    a = int(3*n/2)
    b = int(5*n/2)
    Im = np.abs(np.fft.fftshift(
        FF4(np.pad(U*Q, (a, a), 'constant', constant_values=0.))
        )[a : b, a : b])**2
    
    # Im = (Im - np.max([Im[0, : ], Im[-1, : ], Im[ : , 0], Im[ : , -1]]))/Im.max()
    
    # Im = (Im - np.max([Im[0, : ], Im[-1, : ], Im[ : , 0], Im[ : , -1]]))/Im.max() - Threshold
    
    Im = (Im/Im.max() - Threshold)
    
    Im[Im < 0] = 0.
    Im = Im / Im.max()

    Im_sum = np.sum(Im)
    SHWFS0 = np.sum(Im * xwfs)/Im_sum
    SHWFS1 = np.sum(Im * ywfs)/Im_sum
    
    return np.asarray([SHWFS0, SHWFS1])*np.pi/2

def SHWFS_ANG4(U, Q, QQ, FF2, FFi2, xwfs, ywfs, Im, Threshold):
    n = U.shape[0]
    a = int(n/2)
    b = int(3*n/2)
    
    Im = np.abs(
        np.fft.fftshift(FFi2(
            np.fft.fftshift(QQ * FF2(
                np.pad(U*Q, (a, a), 'constant', constant_values=0.)))
            ))[a : b, a : b])**2
    Im = (Im/Im.max() - Threshold)
    Im[Im < 0] = 0.
    Im = Im / Im.max()

    Im_sum = np.sum(Im)
    SHWFS0 = np.sum(Im * xwfs)/Im_sum
    SHWFS1 = np.sum(Im * ywfs)/Im_sum
    
    return np.asarray([SHWFS0, SHWFS1])*np.pi/2

###############################################################################

# set up shwfs propagation here
# also any memory allocation

REPEAT = 100

N = 2**(np.arange(10)+1)
t_Ig = np.zeros((N.shape[0], REPEAT), dtype=float)
t_SHWFS2 = np.zeros((N.shape[0], REPEAT), dtype=float)
t_SHWFS4 = np.zeros((N.shape[0], REPEAT), dtype=float)
t_SHWFS_ANG4 = np.zeros((N.shape[0], REPEAT), dtype=float)

t_Ig_mean = np.zeros((N.shape[0]), dtype=float)
t_SHWFS2_mean = np.zeros((N.shape[0]), dtype=float)
t_SHWFS4_mean = np.zeros((N.shape[0]), dtype=float)
t_SHWFS_ANG4_mean = np.zeros((N.shape[0]), dtype=float)

t_Ig_error = np.zeros((N.shape[0]), dtype=float)
t_SHWFS2_error = np.zeros((N.shape[0]), dtype=float)
t_SHWFS4_error = np.zeros((N.shape[0]), dtype=float)
t_SHWFS_ANG4_error = np.zeros((N.shape[0]), dtype=float)
# for n in N:



for power in tqdm(np.arange(10)+1):
    n = 2**power
    Threshold = 0
    U = np.ones((n, n), dtype=complex)
    I = np.ones((n, n), dtype=float)
    phase = np.ones((n, n), dtype=float)
    g0 = np.zeros((n - 1, n), dtype=float)
    g1 = np.zeros((n, n - 1), dtype=float)

    F20 = pyfftw.empty_aligned((2*n, 2*n), dtype=complex)
    F21 = pyfftw.empty_aligned((2*n, 2*n), dtype=complex)
    FF201 = pyfftw.FFTW(F20, F21, axes=(-1,-2), direction='FFTW_FORWARD')
    FFi201 = pyfftw.FFTW(F20, F21, axes=(-1,-2), direction='FFTW_BACKWARD')
    
    F40 = pyfftw.empty_aligned((4*n, 4*n), dtype=complex)
    F41 = pyfftw.empty_aligned((4*n, 4*n), dtype=complex)
    FF401 = pyfftw.FFTW(F40, F41, axes=(-1,-2), direction='FFTW_FORWARD')
    
    (y1, x1) = np.meshgrid(np.arange(-n/2, n/2),
                           np.arange(-n/2, n/2))
    Q = np.exp(-1j*(x1 + y1)*np.pi/(2*n))
    del x1, y1
    
    fX,fY = np.meshgrid(np.arange(-n, n),
                        np.arange(-n, n))
    fsq = fX**2 + fY**2
    QQ = np.exp(-1j * np.pi * fsq / n)
    del fX, fY, fsq
        
    gx = np.zeros((n - 1, n), dtype=float)
    gy = np.zeros((n, n - 1), dtype=float)
    Im = np.zeros((n, n), dtype=float)
    
    (ywfs, xwfs) = np.meshgrid(np.arange(n) + 0.5 - n/2,
                               np.arange(n) + 0.5 - n/2)
    
    # t_Ig[power] = 
    # t_Ig_error[power] = 
    for repeat in tqdm(np.arange(REPEAT)):
        t_SHWFS2[power - 1, repeat] = timeit.timeit(
            'SHWFS2(U, Q, FF201, xwfs, ywfs, Im, Threshold)',
            setup="from __main__ import SHWFS2, U, Q, FF201, xwfs, ywfs, Im, Threshold", number=100)/100
        t_SHWFS4[power - 1, repeat] = timeit.timeit(
            'SHWFS4(U, Q, FF401, xwfs, ywfs, Im, Threshold)',
            setup="from __main__ import SHWFS4, U, Q, FF401, xwfs, ywfs, Im, Threshold", number=100)/100
        t_SHWFS_ANG4[power - 1, repeat] = timeit.timeit(
            'SHWFS_ANG4(U, Q, QQ, FF201, FFi201, xwfs, ywfs, Im, Threshold)',
            setup="from __main__ import SHWFS_ANG4, U, Q, QQ, FF201, FFi201, xwfs, ywfs, Im, Threshold", number=100)/100
        t_Ig[power - 1, repeat] = timeit.timeit(
            'Ig(U, phase, I, gx, gy)',
            setup="from __main__ import Ig, U, phase, I, gx, gy", number=100)/100

t_Ig_mean = np.mean(t_Ig, axis=1)
t_SHWFS2_mean = np.mean(t_SHWFS2, axis=1)
t_SHWFS4_mean = np.mean(t_SHWFS4, axis=1)
t_SHWFS_ANG4_mean = np.mean(t_SHWFS_ANG4, axis=1)
t_Ig_error = np.std(t_Ig, ddof=1, axis=1)
t_SHWFS2_error = np.std(t_SHWFS2, ddof=1, axis=1)
t_SHWFS4_error = np.std(t_SHWFS4, ddof=1, axis=1)
t_SHWFS_ANG4_error = np.std(t_SHWFS_ANG4, ddof=1, axis=1)
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
plt.errorbar(N, t_Ig_mean, yerr=t_Ig_error, label='Ig')
plt.errorbar(N, t_SHWFS2_mean, yerr=t_SHWFS2_error, label='SHWFS 2xPad')
plt.errorbar(N, t_SHWFS4_mean, yerr=t_SHWFS4_error, label='SHWFS 4xPad')
plt.errorbar(N, t_SHWFS_ANG4_mean, yerr=t_SHWFS_ANG4_error, label='2xPad AngularSpectrum')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('N (pixel)')
plt.ylabel('time (s)')
plt.savefig('timetrial3.png')
plt.clf()

# plt.errorbar(N,t_SHWFS/t_Ig,
#              yerr=np.sqrt((t_Ig_error/t_Ig)**2
#                           + (t_SHWFS_error/t_SHWFS)**2)*(t_SHWFS/t_Ig)
#              , marker='o')
plt.errorbar(N, t_SHWFS2_mean/t_Ig_mean, yerr=(t_SHWFS2_mean/t_Ig_mean)*np.sqrt((t_SHWFS2_error/t_SHWFS2_mean)**2 + (t_Ig_error/t_Ig_mean)**2), marker='o', label='SHWFS 2xPad')
plt.errorbar(N, t_SHWFS4_mean/t_Ig_mean, yerr=(t_SHWFS4_mean/t_Ig_mean)*np.sqrt((t_SHWFS4_error/t_SHWFS4_mean)**2 + (t_Ig_error/t_Ig_mean)**2), marker='o', label='SHWFS 4xPad')
plt.errorbar(N, t_SHWFS_ANG4_mean/t_Ig_mean, yerr=(t_SHWFS_ANG4_mean/t_Ig_mean)*np.sqrt((t_SHWFS_ANG4_error/t_SHWFS_ANG4_mean)**2 + (t_Ig_error/t_Ig_mean)**2), marker='o', label='2xPad AngularSpectrum')

plt.xscale('log')
plt.legend()
plt.xlabel('N (pixel)')
plt.ylabel('SHWFS time / Ig time')
plt.savefig('timegain3.png')
plt.clf()

myvar = {'t_Ig_mean' : t_Ig_mean,
         't_SHWFS2_mean' : t_SHWFS2_mean,
         't_SHWFS4_mean' : t_SHWFS4_mean,
         't_SHWFS_ANG4_mean' : t_SHWFS_ANG4_mean,
         't_Ig_error' : t_Ig_error,
         't_SHWFS2_error' : t_SHWFS2_error,
         't_SHWFS4_error' : t_SHWFS4_error,
         't_SHWFS_ANG4_error' : t_SHWFS_ANG4_error}

with open('timing_out.pkl', 'wb') as file:
    pickle.dump(myvar, file)
    file.close()