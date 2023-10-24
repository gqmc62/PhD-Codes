# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:31:18 2023

@author: gqmc62
"""


###############################################################################
##########################################################################
"""
█▀▀█░█░▒█░▀█▀░▀█▀░░▀░░█░░░█░█▀▀▄░▀█▀░░░▒█░▄▀░▄▀▀▄░█▀▀▄░█▀▀▀░█░▄░█▀▀▄░█▀▀░█░░░█
█▄▄█░█░▒█░░█░░░█░░░█▀░▀▄█▄▀░█▄▄█░░█░░░░▒█▀▄░░█░░█░█░▒█░█░▀▄░█▀▄░█▄▄█░█▀▀░▀▄█▄▀
█░░░░░▀▀▀░░▀░░░▀░░▀▀▀░░▀░▀░░▀░░▀░░▀░░░░▒█░▒█░░▀▀░░▀░░▀░▀▀▀▀░▀░▀░▀░░▀░▀▀▀░░▀░▀░
"""
##########################################################################
###############################################################################

# imports
import numpy as np
from matplotlib import pyplot as plt
import pickle
import pyfftw
from aotools.turbulence.phasescreen import ft_sh_phase_screen
from tqdm import tqdm
import time
import os

TIME_START = time.time()

# functions
def distribute_coherent_length(r0, Rytov, HA, wvl, zenith_angle):
    if Rytov == 0:
        return r0, np.inf

    k = 2*np.pi/wvl
    c2ndz_tot = r0**(-5./3.)/(0.423*k**2/np.cos(zenith_angle))
    c2ndz_HA = Rytov / (0.5631 * k**(7./6.)
                        / (np.cos(zenith_angle))**(11./6.)
                        * HA**(5./6.))
    if c2ndz_HA >= c2ndz_tot :
        return np.NaN, np.NaN

    c2ndz_GL = c2ndz_tot - c2ndz_HA
    r0_GL = (0.423*k**2/np.cos(zenith_angle)*c2ndz_GL)**(-3./5.)
    r0_HA = (0.423*k**2/np.cos(zenith_angle)*c2ndz_HA)**(-3./5.)

    return r0_GL, r0_HA

def crop2(A, pad_factor=2):
    NN = A.shape[0]
    edge_low = int(NN*((pad_factor - 1)/pad_factor)/2)
    edge_high = int(NN*((pad_factor + 1)/pad_factor)/2)

    return A[edge_low : edge_high, edge_low : edge_high]

def zero_pad2(A, pad_factor=2):
    add_each_side = (pad_factor - 1)/2
    return np.pad(A, (int(A.shape[0]*add_each_side),
                      int(A.shape[1]*add_each_side)),
                  'constant', constant_values=0.)

###############################################################################
# simulations initialization #############################################
###############################################################################

# independent parameters

TARGET = 100
SCREEN_LIMIT = int(2**16)

n = int(64)
oversize = int(2)
L0_GL = float(20)
L0_HA = L0_GL
l0_GL = float(1.5e-10)
l0_HA = l0_GL
wvl = float(500e-9)
zenith_angle = float(0)

###############################################################################

# input conditions

r0 = np.array([11, 8, 5, 2], dtype=float)*1e-2              # II
dor0 = np.array([0.5, 1, 2, 3], dtype=float)                # JJ
Rytov = np.array([0.01, 0.03, 0.1, 0.3], dtype=float)       # KK
H_HA = np.array([5, 10, 15, 20], dtype=float)*1e3           # LL

# r0 = np.array([8, 5, 2], dtype=float)*1e-2              # II
# dor0 = np.array([1, 2, 3], dtype=float)                # JJ
# Rytov = np.array([0.03, 0.1, 0.3], dtype=float)       # KK
# H_HA = np.array([5, 10, 15], dtype=float)*1e3           # LL

# r0 = np.array([11, 8, 5, 2], dtype=float)*1e-2              # II
# dor0 = np.array([0.5, 1, 2, 3, 4], dtype=float)                # JJ
# Rytov = np.array([0.01, 0.03, 0.1, 0.3], dtype=float)       # KK
# H_HA = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50], dtype=float)*1e3           # LL

# r0 = np.array([8, 5, 2], dtype=float)*1e-2              # II
# dor0 = np.array([1, 2, 3], dtype=float)                # JJ
# Rytov = np.array([0.1, 0.3], dtype=float)       # KK
# H_HA = np.array([5, 10, 15], dtype=float)*1e3           # LL

###############################################################################

#4096/((prop_pad/2)**2*POSSIBLE_PROP)

# Common indices ranges and dimension. Easy to keep track of dependencies
II = len(r0)
JJ = len(dor0)
KK = len(Rytov)
LL = len(H_HA)

# dependent simulation parameters

wavenumber = 2*np.pi/wvl
N_MAX = int(SCREEN_LIMIT/oversize / 4.)

d = np.zeros((II, JJ), dtype=float)                         # II, JJ
delta = np.zeros((II, JJ), dtype=float)                     # II, JJ

r0_GL = np.zeros((II, KK, LL), dtype=float)                 # II, KK, LL
r0_HA = np.zeros((II, KK, LL), dtype=float)                 # II, KK, LL

prop_pad = np.zeros((II, JJ, LL), dtype=int)                # II, JJ, LL
N = np.zeros((II, JJ, LL), dtype=int)                       # II, JJ, LL
num_lenslet = np.zeros((II, JJ, LL), dtype=int)             # II, JJ, LL
REPEAT = np.zeros((II, JJ, LL), dtype=int)                  # II, JJ, LL

POSSIBLE_SCREEN = np.ones_like(r0_GL, dtype=bool)           # II, KK, LL
POSSIBLE_PROP = np.ones_like(prop_pad, dtype=bool)          # II, JJ, LL
POSSIBLE_SAMPLE = np.ones((II, JJ, KK, LL), dtype=bool)     # II, JJ, KK, LL
POSSIBLE = np.zeros((II, JJ, KK, LL), dtype=bool)           # II, JJ, KK, LL


###############################################################################

for ii in np.arange(II):
    for jj in np.arange(JJ):
        d[ii, jj] = dor0[jj]*r0[ii]

delta = d/n

# distribute r0
# impossible r0 distribution is ignored
for ii in np.arange(II):
    for kk in np.arange(KK):
        for ll in np.arange(LL):
            r0_GL[ii, kk, ll], r0_HA[ii, kk, ll] = distribute_coherent_length(
                r0[ii], Rytov[kk], H_HA[ll], wvl, zenith_angle)
            if (np.isnan(r0_GL[ii, kk, ll])
                or np.isnan(r0_HA[ii, kk, ll])
                or np.isinf(r0_GL[ii, kk, ll])
                or np.isinf(r0_HA[ii, kk, ll])):
                POSSIBLE_SCREEN[ii, kk, ll] = False
                # the inf r0 is actually generateable but
                # it should have been generate in a one layer scheme instead
                # so NO!!
                
# screens must be well sampled
for ii in range(II):
    for jj in range(JJ):
        for kk in range(KK):
            for ll in range(LL):
                if POSSIBLE_SCREEN[ii, kk, ll]:
                    if ((r0_GL[ii, kk, ll]/delta[ii, jj] <= 16)
                        or (r0_HA[ii, kk, ll]/delta[ii, jj] <= 16)):
                        POSSIBLE_SAMPLE[ii, jj, kk, ll] = False
                              
###############################################################################
# Check possiblity and validity to run

# calculating how big a HA screen is required.
# note this is for free atmosphere propagation

for ii in np.arange(II):
    for jj in np.arange(JJ):
        for ll in np.arange(LL):
            # print(ii,jj,ll)
            waste_px = (wvl*H_HA[ll])/(delta[ii, jj]**2)
            print(int(np.ceil(waste_px)), r0[ii], dor0[jj], Rytov[kk], H_HA[ll])
            available_px1 = np.floor(N_MAX - waste_px)
            optimized_screen = waste_px*2
            # print(available_px1)
            
            # the available size cannot accommodate enough space for 'contamination'
            if available_px1 <= 0:
                REPEAT[ii, jj, ll] = 0
                num_lenslet[ii, jj, ll] = 0
                N[ii, jj, ll] = 0
                # print(ii, jj, ll)
                # print(waste_px)
                # print(REPEAT[ii, jj, ll], num_lenslet[ii, jj, ll])
                # print(num_lenslet[ii, jj, ll])
                # print(N[ii, jj, ll])
            else:
                # the required size is reached before going full max memomory
                if available_px1 >= N_MAX/2:
                    num_lenslet[ii, jj, ll] = int(2**np.ceil(np.log(optimized_screen)/np.log(2))/n/2)
                    REPEAT[ii, jj, ll] = int(np.ceil(float(TARGET) / float(num_lenslet[ii, jj, ll]**2)))
                    N[ii, jj, ll] = num_lenslet[ii, jj, ll] * n
                    # print(ii, jj, ll)
                    # print(waste_px)
                    # print(REPEAT[ii, jj, ll], num_lenslet[ii, jj, ll])
                    # print(N[ii, jj, ll])
                    # print(num_lenslet[ii, jj, ll])
                else:
                    # even going at the max memory, it cannot acheive optimization
                    # however, it still yield some reliable results within a limit area
                    if available_px1 <= N_MAX/2:
                        # print(ii, jj, ll)
                        available_lenslet1 = available_px1//n
                        num_lenslet[ii, jj, ll] = available_lenslet1
                        available_lenslet2 = available_lenslet1**2
                        REPEAT[ii, jj, ll] = int(np.ceil(float(TARGET)/float(available_lenslet2)))
                        N[ii, jj, ll] = N_MAX/2
                        # print(ii, jj, ll)
                        # print(waste_px)
                        # print(REPEAT[ii, jj, ll], num_lenslet[ii, jj, ll])
                        # print(N[ii, jj, ll])
                        # print(num_lenslet[ii, jj, ll])
                    else:
                        print('this is serious error')

POSSIBLE_PROP = (num_lenslet != 0)

for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                POSSIBLE[ii, jj, kk, ll] = (POSSIBLE_PROP[ii, jj, ll]
                                            * POSSIBLE_SCREEN[ii, kk, ll]
                                            * POSSIBLE_SAMPLE[ii, jj, kk, ll])
                # if POSSIBLE[ii, jj, kk, ll] == True:
                #     print(r0[ii], dor0[jj], Rytov[kk], H_HA[ll])



# from aotools.turbulence import infinitephasescreen

# POSSIBLE_delta = np.ones_like(delta)
# POSSIBLE_delta[0, 0] = False
# POSSIBLE_delta[0, 1] = False
# POSSIBLE_delta[1, 0] = False
# POSSIBLE_delta[1, 1] = False
# POSSIBLE_delta[1, 2] = False
# POSSIBLE_delta[2, 0] = False
# POSSIBLE_delta[2, 1] = False
# POSSIBLE_delta[2, 2] = False
# POSSIBLE_delta[2, 3] = False
# POSSIBLE_delta[3, 0] = False
# POSSIBLE_delta[3, 1] = False
# POSSIBLE_delta[3, 2] = False
# POSSIBLE_delta[3, 3] = False

# n = 128
# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 if POSSIBLE_SCREEN[ii, kk, ll]:
#                     print(ii, jj, kk, ll)
#                     plt.pcolor(np.arange(n)*delta[ii, jj], np.arange(n)*delta[ii, jj],
#                                 infinitephasescreen.PhaseScreenKolmogorov(
#                                     n, delta[ii, jj], r0_GL[ii, kk, ll], L0_GL).scrn)
#                     plt.colorbar()
#                     plt.axis('square')
#                     plt.title('GL: r0={:.0f}cm, dor0={:.1f}, Rytov={:.2f}, H_HA={:.0f}'.format(r0[ii]*100, dor0[jj], Rytov[kk], H_HA[ll]/1000))
#                     plt.show()
#                     plt.pcolor(np.arange(n)*delta[ii, jj], np.arange(n)*delta[ii, jj],
#                                 infinitephasescreen.PhaseScreenKolmogorov(
#                                     n, delta[ii, jj], r0_HA[ii, kk, ll], L0_HA).scrn)
#                     plt.colorbar()
#                     plt.axis('square')
#                     plt.title('HA: r0={:.0f}cm, dor0={:.1f}, Rytov={:.2f}, H_HA={:.0f}'.format(r0[ii]*100, dor0[jj], Rytov[kk], H_HA[ll]/1000))
#                     plt.show()
#

import my_propagation as mprop

from mypropagationclass import OneAngularPropAtm
from aotools.turbulence import infinitephasescreen

n = 64 # desired output phase screen
ztest = 50000
op = OneAngularPropAtm(wvl, np.nanmax(delta), n, ztest)
N = op.pad_factor*n

# separation = 2
# s = infinitephasescreen.PhaseScreenKolmogorov(N, np.nanmax(delta), np.nanmin(r0_HA), L0_HA)
# for i in np.arange(separation*N):
#     s.add_row()
# big_screen = s.scrn
# small_screen = mprop.crop2(s.scrn, op.pad_factor)
# big_screen = big_screen - small_screen.mean()
# small_screen = small_screen - small_screen.mean()

scrn = ft_sh_phase_screen(np.nanmin(r0_HA), N, np.nanmax(delta), L0_HA, l0_HA)
big_screen = scrn
small_screen = mprop.crop2(scrn, op.pad_factor)

from aotools.functions import zernikeArray
pupil = zernikeArray(1,n)[0]

# plt.pcolor(np.arange(small_screen.shape[0])*np.nanmax(delta),
#             np.arange(small_screen.shape[0])*np.nanmax(delta),
#             small_screen)
# plt.colorbar()
# plt.axis('square')
# plt.title('HA: r0={:.0f}cm, delta={:.3f}mm, d={:.3f}m'.format(np.nanmin(r0_HA), 1000*np.nanmax(delta), np.nanmax(delta)*n))
# plt.show()
# plt.clf()

from mypropagationclass import ConsecutivePropCircle
# from mypropagationclass import ConsecutivePropSquare
op.xtra_init()
cp_0pad = 2
cp = ConsecutivePropCircle(wvl, np.nanmax(delta), cp_0pad*n, ztest)
# cp = ConsecutivePropCircle(wvl, np.nanmax(delta), small_screen.shape[0], ztest)

# U_out_op = op.prop(np.exp(1j*big_screen))
U_out_op = op.prop(mprop.zero_pad2(np.exp(1j*small_screen*pupil)*pupil, op.pad_factor))
U_out_cp = mprop.crop2(cp.prop(mprop.zero_pad2(np.exp(1j*small_screen*pupil)*pupil, cp_0pad)), cp_0pad)

plt.pcolor(np.arange(small_screen.shape[0])*np.nanmax(delta),
            np.arange(small_screen.shape[0])*np.nanmax(delta),
            np.angle(U_out_cp))
plt.colorbar()
plt.axis('square')
plt.title('CP: HA at GL: r0={:.0f}cm, delta={:.3f}mm, d={:.3f}m'.format(np.nanmin(r0_HA)*100, 1000*np.nanmax(delta), np.nanmax(delta)*n))
plt.show()
plt.clf()

plt.pcolor(np.arange(small_screen.shape[0])*np.nanmax(delta),
            np.arange(small_screen.shape[0])*np.nanmax(delta),
            np.angle(mprop.crop2(U_out_op,op.pad_factor)))
plt.colorbar()
plt.axis('square')
plt.title('OP: HA at GL: r0={:.0f}cm, delta={:.3f}mm, d={:.3f}m'.format(np.nanmin(r0_HA)*100, 1000*np.nanmax(delta), np.nanmax(delta)*n))
plt.show()
plt.clf()

plt.pcolor(np.arange(small_screen.shape[0])*np.nanmax(delta),
            np.arange(small_screen.shape[0])*np.nanmax(delta),
            np.abs(U_out_cp)**2)
plt.colorbar()
plt.axis('square')
plt.title('CP: HA at GL: r0={:.0f}cm, delta={:.3f}mm, d={:.3f}m'.format(np.nanmin(r0_HA)*100, 1000*np.nanmax(delta), np.nanmax(delta)*n))
plt.show()
plt.clf()

plt.pcolor(np.arange(small_screen.shape[0])*np.nanmax(delta),
            np.arange(small_screen.shape[0])*np.nanmax(delta),
            np.abs(mprop.crop2(U_out_op,op.pad_factor))**2)
plt.colorbar()
plt.axis('square')
plt.title('OP: HA at GL: r0={:.0f}cm, delta={:.3f}mm, d={:.3f}m'.format(np.nanmin(r0_HA)*100, 1000*np.nanmax(delta), np.nanmax(delta)*n))
plt.show()
plt.clf()

# from mypropagationclass import ConsecutivePropCircle
from mypropagationclass import ConsecutivePropSquare
op.xtra_init()
cp_0pad = 2
cp = ConsecutivePropSquare(wvl, np.nanmax(delta), cp_0pad*n, ztest)
# cp = ConsecutivePropCircle(wvl, np.nanmax(delta), small_screen.shape[0], ztest)

# U_out_op = op.prop(np.exp(1j*big_screen))
U_out_op = op.prop(mprop.zero_pad2(np.exp(1j*small_screen), op.pad_factor))
U_out_cp = mprop.crop2(cp.prop(mprop.zero_pad2(np.exp(1j*small_screen), cp_0pad)), cp_0pad)

plt.pcolor(np.arange(small_screen.shape[0])*np.nanmax(delta),
            np.arange(small_screen.shape[0])*np.nanmax(delta),
            np.angle(U_out_cp))
plt.colorbar()
plt.axis('square')
plt.title('CP: HA at GL: r0={:.0f}cm, delta={:.3f}mm, d={:.3f}m'.format(np.nanmin(r0_HA)*100, 1000*np.nanmax(delta), np.nanmax(delta)*n))
plt.show()
plt.clf()

plt.pcolor(np.arange(small_screen.shape[0])*np.nanmax(delta),
            np.arange(small_screen.shape[0])*np.nanmax(delta),
            np.angle(mprop.crop2(U_out_op,op.pad_factor)))
plt.colorbar()
plt.axis('square')
plt.title('OP: HA at GL: r0={:.0f}cm, delta={:.3f}mm, d={:.3f}m'.format(np.nanmin(r0_HA)*100, 1000*np.nanmax(delta), np.nanmax(delta)*n))
plt.show()
plt.clf()

plt.pcolor(np.arange(small_screen.shape[0])*np.nanmax(delta),
            np.arange(small_screen.shape[0])*np.nanmax(delta),
            np.abs(U_out_cp)**2)
plt.colorbar()
plt.axis('square')
plt.title('CP: HA at GL: r0={:.0f}cm, delta={:.3f}mm, d={:.3f}m'.format(np.nanmin(r0_HA)*100, 1000*np.nanmax(delta), np.nanmax(delta)*n))
plt.show()
plt.clf()

plt.pcolor(np.arange(small_screen.shape[0])*np.nanmax(delta),
            np.arange(small_screen.shape[0])*np.nanmax(delta),
            np.abs(mprop.crop2(U_out_op,op.pad_factor))**2)
plt.colorbar()
plt.axis('square')
plt.title('OP: HA at GL: r0={:.0f}cm, delta={:.3f}mm, d={:.3f}m'.format(np.nanmin(r0_HA)*100, 1000*np.nanmax(delta), np.nanmax(delta)*n))
plt.show()
plt.clf()

print(cp_0pad*n)
print(op.pad_factor*n)