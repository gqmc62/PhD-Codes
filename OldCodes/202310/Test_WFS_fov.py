# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 12:33:28 2022

# get one image at wfs focal plane then crop it down to get desired fov

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

from aotools.functions import zernike

lenslet_pixel = 64

# r0 = np.array([1], dtype=float)*1e-2

# Screen physical parameters
# j
D_lenslet = np.array([15], dtype=float)*1e-2
# D_lenslet = np.array([0.5,1,2,3], dtype=float)*r0[0]
# D_lenslet = np.array([10,8,6,4,2], dtype=float)*1e-2 # j
# D_lenslet = np.array([32,2], dtype=float)*1e-2

# Atmosphere parameters
L0 = float(100)                  # m, Outer turbulence scale
l0 = float(1e-7)                # m, Inner turbulence scale

# # k
# # r0 = np.array([10,8,6,4,2], dtype=float)*1e-2 # k
# r0 = np.array([10], dtype=float)*1e-2
# # r0 = np.array([10], dtype=float)*1e-2
#                                 # m, Accumulated r0
r0 = D_lenslet[0] / np.array([0.5,1,2,3], dtype=float)

# Optical parameters
wvl = float(500e-9)         # m, wavelength

# Progation simulation parameters
pf_lenslet = int(2)         # zero padding scale for after pupil prop

# WFS parameters
spot_size = float(2)        # px, d_detector * spot_size = wvl*f_lenslet/D_lenslet
                            # (d_detector/f_lenslet) * spot_size = wvl/D_lenslet
                            # angular_pixel_scale * spot_size = lambda/D

oversize = 1

REPEAT = 50
# REPEAT=1



JJ = len(D_lenslet)
KK = len(r0)

N = lenslet_pixel
piston, tip, tilt = zernike.zernikeArray(3,N)
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

CEN = int(lenslet_pixel/2)
NN = N*pf_lenslet
edge_low = int(NN*((pf_lenslet - 1)/pf_lenslet)/2)
edge_high = int(NN*((pf_lenslet + 1)/pf_lenslet)/2)
add_each_side = int((pf_lenslet - 1)/2*N)

F0 = pyfftw.empty_aligned((NN, NN), dtype=complex)
F1 = pyfftw.empty_aligned((NN, NN), dtype=complex)
FF01 = pyfftw.FFTW(F0, F1, axes=(-1,-2), direction='FFTW_FORWARD')

g_slope = np.zeros((JJ,KK,REPEAT,2), dtype=float)
slope = np.zeros((2),dtype=float)

real_SHWFS_slope = np.zeros((JJ,KK,REPEAT,CEN ,2), dtype=float)
dif_SHWFS_slope = np.zeros((JJ,KK,REPEAT,CEN,2), dtype=float)

# Threshold = np.array([0,0,0,0], dtype=float)
Threshold = np.array([2.5e-4,2.5e-4,0,0], dtype=float) # sh
# Threshold = np.array([2.5e-4,2.5e-4,2.5e-4,2.5e-4], dtype=float) # no sh

for jj in np.arange(JJ):
    convert_real = ((d[jj]/f_lenslet[jj]) # convert pixel drift to angle drift
                    * (d[jj] * lenslet_pixel) # convert angle drift to wavefront error
                    * (2*np.pi/wvl)) # convert wavefront error to phase error
    (x1,y1) = np.meshgrid(d[jj]*np.arange(-N/2,N/2),
                         d[jj]*np.arange(-N/2,N/2))
    # (xx11,yy11) = np.meshgrid(d[jj]*np.arange(-N,N),
    #                      d[jj]*np.arange(-N,N))
    # r1sq = (x1**2 + y1**2) + 1e-10
    Q = np.exp(-1j*wavenumber/2./f_lenslet[jj] * (((x1 + y1) * d[jj])))
    # Q = np.exp(-1j*wavenumber/2./f_lenslet[jj] * (((x1 + y1) * d[jj]) + r1sq))
    # QQ = np.exp(-1j*wavenumber/2./f_lenslet[jj] * (((xx11 + yy11) * d[jj])))

    df = 1/N/d[jj]
    fx = np.arange(1,CEN)*df

    for repeat in np.arange(REPEAT):
        # print(repeat)
        phase_r0min = mprop.crop2(ft_sh_phase_screen(r0.min(), N*oversize, d[jj], L0, l0), oversize)
        # phase_r0min = mprop.crop2(ft_phase_screen(r0.min(), N*oversize, d[jj], L0, l0), oversize)
        # PSD = mps.psd_function(r0.min(), N, d[jj], L0, l0)[CEN + 1 : , CEN]
        # psd = mps.get_PSD(phase_r0min, d[jj])[CEN + 1 : , CEN]
        # plt.plot(fx,PSD,marker='',ls='-',label='Theoretical')
        # plt.plot(fx,psd,marker='x',ls='',label='Measured')
        # plt.xlabel('Spatial Frequency (1/m)')
        # plt.ylabel('PSD$_\phi$ (m$^2$)')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.legend()
        # plt.title('Measured PSD of a generated phase screen')
        # plt.show()
        for kk in np.arange(KK):
            phase_unwrap = phase_r0min * (r0.min()/r0[kk])**(5./6.)
            # phase_unwrap = mprop.crop2(ft_sh_phase_screen(r0[kk], N*oversize, d[jj], L0, l0), oversize)
            # phase_unwrap = mprop.crop2(ft_phase_screen(r0[kk], N*oversize, d[jj], L0, l0), oversize)
            
            # phase_unwrap = np.zeros((N, N), dtype=float)
            # phase_unwrap = wavenumber/f_lenslet[jj] * (((5*x1 + (-8)*y1) * d[jj]))

            U_pupil = np.exp(1j*phase_unwrap)
            
            # unit is rad per lenslet
            g_slope[jj,kk,repeat,1] = np.mean(phase_unwrap[1:,:] - phase_unwrap[:-1,:]) * lenslet_pixel
            g_slope[jj,kk,repeat,0] = np.mean(phase_unwrap[:,1:] - phase_unwrap[:,:-1]) * lenslet_pixel

            F0[:,:] = np.pad(U_pupil * Q,
                              (add_each_side,add_each_side),
                              'constant', constant_values=0.)
            # F0[:,:] = np.pad(U_pupil,
            #                   (add_each_side,add_each_side),
            #                   'constant', constant_values=0.) * QQ
            I_focus = (np.abs(
                np.fft.fftshift(FF01())[edge_low : edge_high,
                                        edge_low : edge_high])**2
                / (spot_size*lenslet_pixel**2)**2)
            # I_focus = (np.abs(
            #     (angularSpectrum(
            #         np.pad(U_pupil * Q,
            #                (add_each_side,add_each_side),
            #                'constant', constant_values=0.),
            #         wvl, d[jj], d[jj], f_lenslet[jj])
            #         )[edge_low : edge_high,
            #           edge_low : edge_high]))**2
            I_focus = I_focus/I_focus.max()
            I_focus = I_focus - Threshold[kk]
            I_focus[np.where(I_focus < 0)] = 0.0
            I_focus = I_focus/I_focus.max()

            # unit of half fov is pixel! = fov in wvl/d
            for half_fov in (np.arange(CEN) + 1):
                xx, yy = np.meshgrid(np.arange(-half_fov, half_fov) + 0.5,
                                     np.arange(-half_fov, half_fov) + 0.5)

                I_fov = I_focus[CEN - half_fov : CEN + half_fov,
                                CEN - half_fov : CEN + half_fov]
                I_fov = I_fov/np.sum(I_fov)

                # unit is rad per lenslet
                real_SHWFS_slope[jj,kk,repeat,half_fov - 1, 0] = (np.sum(I_fov*xx)
                                                                  * convert_real)
                real_SHWFS_slope[jj,kk,repeat,half_fov - 1, 1] = (np.sum(I_fov*yy)
                                                                  * convert_real)

                dif_SHWFS_slope[jj,kk,repeat,half_fov - 1,:] = (g_slope[jj,kk,repeat,:]
                                                                - real_SHWFS_slope[jj,kk,repeat,half_fov - 1,:])

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

