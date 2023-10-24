# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 12:33:28 2022

@author: gqmc62
"""

#test wfs fov

import numpy as np
from matplotlib import pyplot as plt
import time
import timeit
import random
import pickle
import pyfftw

from aotools.turbulence.phasescreen import ft_sh_phase_screen
from aotools.turbulence.phasescreen import ft_phase_screen
from aotools.opticalpropagation import angularSpectrum
from aotools.opticalpropagation import lensAgainst

import my_propagation as mprop
import my_WFS as mwfs
import my_phasescreen as mps

from r0RytovDistribution import distribute_coherent_length

from aotools import fouriertransform

lenslet_pixel = 64

# Screen physical parameters
# j
# D_lenslet = np.array([32], dtype=float)*1e-2
D_lenslet = np.array([32,16,8,4,2], dtype=float)*1e-2
# D_lenslet = np.array([10,8,6,4,2], dtype=float)*1e-2 # j
# D_lenslet = np.array([32,2], dtype=float)*1e-2

# Atmosphere parameters
L0 = float(20)                  # m, Outer turbulence scale
l0 = float(1e-7)                # m, Inner turbulence scale

# k
# r0 = np.array([10,8,6,4,2], dtype=float)*1e-2 # k
r0 = np.array([8], dtype=float)*1e-2
# r0 = np.array([10], dtype=float)*1e-2
                                # m, Accumulated r0

# Optical parameters
wvl = float(500e-9)         # m, wavelength

# Progation simulation parameters
pf_lenslet = int(2)         # zero padding scale for after pupil prop

# WFS parameters
spot_size = float(2)        # px, d_detector * spot_size = wvl*f_lenslet/D_lenslet
                            # (d_detector/f_lenslet) * spot_size = wvl/D_lenslet
                            # angular_pixel_scale * spot_size = lambda/D
REPEAT = 5
# REPEAT=1


JJ = len(D_lenslet)
KK = len(r0)

N = lenslet_pixel
# j
d = D_lenslet/lenslet_pixel
# j
f_lenslet = d**2*spot_size*lenslet_pixel/wvl

wavenumber = 2*np.pi/wvl

for jj in np.arange(JJ):
    check = True
    check_condition = mprop.prop_lens_param_check(
        lenslet_pixel, d[jj], d[jj], wvl, [f_lenslet[jj]],
        pf_lenslet)
    for temp in check_condition:
        check = check * temp
    if check == False:
        print("One of the propagation condition for lenslet is failing.")
        quit()

phase_unwrap = np.zeros((N, N), dtype=float)
U_pupil = np.ones((N, N), dtype=complex)

NN = N*pf_lenslet
edge_low = int(NN*((pf_lenslet - 1)/pf_lenslet)/2)
edge_high = int(NN*((pf_lenslet + 1)/pf_lenslet)/2)
add_each_side = int((pf_lenslet - 1)/2*N)

F0 = pyfftw.empty_aligned((NN, NN), dtype=complex)
F1 = pyfftw.empty_aligned((NN, NN), dtype=complex)
F2 = pyfftw.empty_aligned((NN, NN), dtype=complex)
F3 = pyfftw.empty_aligned((NN, NN), dtype=complex)
FF01 = pyfftw.FFTW(F0, F1, axes=(-1,-2), direction='FFTW_FORWARD')
FF23 = pyfftw.FFTW(F2, F3, axes=(-1,-2), direction='FFTW_BACKWARD')

convert_g = (lenslet_pixel * wvl / 2. / np.pi / 1e-9)

g_slope = np.zeros((JJ,KK,2), dtype=float)
slope = np.zeros((2), dtype=float)
real_SHWFS_slope = np.zeros((JJ,KK,REPEAT,int(lenslet_pixel/2) - 1,2), dtype=float)
dif_SHWFS_slope = np.zeros((JJ,KK,REPEAT,int(lenslet_pixel/2) - 1,2), dtype=float)

for jj in np.arange(JJ):
    convert_real = (d[jj]**2 / f_lenslet[jj] * lenslet_pixel / 1e-9)
    (x1,y1) = np.meshgrid(d[jj]*np.arange(-N*pf_lenslet/2,N*pf_lenslet/2),
                         d[jj]*np.arange(-N*pf_lenslet/2,N*pf_lenslet/2))
    r1sq = (x1**2 + y1**2) + 1e-10
    df1 = 1. / (N*pf_lenslet*d[jj])
    fX,fY = np.meshgrid(df1*np.arange(-N*pf_lenslet/2,N*pf_lenslet/2),
                       df1*np.arange(-N*pf_lenslet/2,N*pf_lenslet/2))
    fsq = fX**2 + fY**2
    Q2 = np.exp(-1j * np.pi**2 * 2 * f_lenslet[jj]/wavenumber*fsq)
    Q = np.exp(-1j*wavenumber/2./f_lenslet[jj] * (((x1 + y1) * d[jj]) + r1sq))
    for kk in np.arange(KK):
        for repeat in np.arange(REPEAT):
            # phase_unwrap = ft_phase_screen(r0[kk], N, d[jj], L0, l0)
            phase_unwrap = np.zeros((N,N), dtype=float)#ft_phase_screen(r0[kk], N, d[jj], L0, l0)
            U_pupil = np.exp(1j*phase_unwrap)
            g_slope[jj,kk,:] = (np.mean(
                mwfs.get_slope(np.angle(U_pupil)),
                axis=(-1,-2)) * convert_g)
            F0[:,:] = np.fft.fftshift(np.pad(U_pupil, (add_each_side,
                                                        add_each_side),
                                              'constant', constant_values=0.)
                                      * Q)
            F2[:,:] = np.fft.ifftshift(np.fft.fftshift(FF01()) * Q2)
            I_focus = np.abs((np.fft.ifftshift(FF23()))[edge_low : edge_high,
                                                        edge_low : edge_high])**2
            I_focus = I_focus/I_focus.max()
            for half_fov in (np.arange(int(lenslet_pixel/2) - 1) + 1):
                I_fov = I_focus[int(lenslet_pixel/2) - half_fov
                                : int(lenslet_pixel/2) + half_fov,
                                int(lenslet_pixel/2) - half_fov
                                : int(lenslet_pixel/2) + half_fov]
                xx, yy = np.meshgrid(np.arange((half_fov)*2),
                                     np.arange((half_fov)*2))
                cen = half_fov - 0.5
                I_fov_total = np.sum(I_fov)
                slope[0] = np.sum(I_fov*xx)/I_fov_total
                slope[1] = np.sum(I_fov*yy)/I_fov_total
                # if ~np.isclose(slope-cen,0).all():
                #     print('error')
                real_SHWFS_slope[jj,kk,repeat,half_fov - 1,:] = (slope - cen) * convert_real
                dif_SHWFS_slope[jj,kk,repeat,half_fov - 1,:] = g_slope[jj,kk,:] - real_SHWFS_slope[jj,kk,repeat,half_fov - 1,:]
                plt.plot(cen,cen,marker='o')
                plt.imshow(I_fov)
                plt.show()
        print(jj,kk)
        # plt.plot(np.arange(lenslet_pixel/2), 10*np.ones_like(np.arange(lenslet_pixel/2)))
        # plt.xlabel('FoV (wvl/d)')
        # plt.ylabel('Slope difference per lenslet (nm)')
        # plt.title('Diffference between SHWFS and g-tilt\nd/r0={:.2f}'.format(D_lenslet[jj]/r0[kk]))
        # # plt.xscale('log')
        # plt.yscale('log')
        # plt.ylim((1,100))
        # plt.savefig('{:d}{:d}.png'.format(jj,kk))
        # plt.show()
        # plt.clf()

    myvar2 = {'lenslet_pixel' : lenslet_pixel,
              'l0' : l0,
              'L0' : L0,
              'D_lenslet' : D_lenslet,
              'r0' : r0,
              'wvl' : wvl,
              'pf_lenslet' : pf_lenslet,
              'spot_size' : spot_size,
              'REPEAT' : REPEAT,
              'N' : N,
              'f_lenslet' : f_lenslet,
              'd' : d,
              'wavenumber' : wavenumber,
              'dif_SHWFS_slope' : dif_SHWFS_slope,
              'g_slope' : g_slope,
              'real_SHWFS_slope' : real_SHWFS_slope}
    with open('Test_SHWFS_FoVv3.pkl', 'wb') as file:
        pickle.dump(myvar2, file)
        file.close()