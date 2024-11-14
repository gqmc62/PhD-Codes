# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:39:26 2022

@author: gqmc62
"""

import numpy as np
from matplotlib import pyplot as plt
import time
import timeit
import random
import pickle

from aotools.turbulence.phasescreen import ft_sh_phase_screen
from aotools.turbulence.phasescreen import ft_phase_screen
from aotools.opticalpropagation import angularSpectrum
from aotools.opticalpropagation import lensAgainst

import my_propagation as mprop
import my_WFS as mwfs
import my_phasescreen as mps

from r0RytovDistribution import distribute_coherent_length


# 23) Normalized 1D correlation. #
def norm_correlate1(a, b):
    """
    Generate normalized correlation between 2 1darrays.

    Parameters
    ----------
    a : 1darray
        (N)
        Any type
        Vector a.
    b : 1darray
        (N)
        Any type
        Vector b.

    Returns
    -------
    c : float
        Normalized correlation between a and b.

    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

#    a = a - a.mean()
#    a = a/a.std()

#    b = b - b.mean()
#    b = b/b.std()

    c = np.correlate(a,b)[0]/np.sqrt((a**2).sum()*(b**2).sum())
    return c


#
# Constants & Independent Parameters
#

LOAD_PARAMETERS = False
# LOAD_PARAMETERS = True

if LOAD_PARAMETERS == False:
    # USE NEW PARAMETERS & constants

    # Screen simulation parameters
    N = int(1024)               # px, number of pixel acroos a valid screen
    lenslet_pixel = int(64)     # px, number of pixel across a lenslet

    # Screen physical parameters
    D_lenslet = 10e-2            # m projected lenslet size at telescope pupil

    # Atmosphere parameters
    L0 = float(20)                  # m, Outer turbulence scale
    l0 = float(1e-7)                # m, Inner turbulence scale
    r0 = np.array([10,8,5,2], dtype=float)*1e-2
                                    # m, Accumulated r0

    # Optical parameters
    wvl = float(500e-9)         # m, wavelength
    SNR = float(100)            # unitless, signal-to-noise-ratio

    # Propagation parameters
    z = np.array([0, 20], dtype=float)*1e3   # m, atmospheric layers height
    rytov = np.array([1e-4,1e-2,1e-1,3e-1], dtype=float)
    zenith_angle = 0            # radian, 0 = over head, np.pi/2 = on horizon


    # Progation simulation parameters
    pf = int(2)                 # extra input scale for free atm prop
    pf_lenslet = int(2)         # zero padding scale for after pupil prop

    # WFS parameters
    spot_size = float(2)        # px, d_detector * spot_size = wvl*f_lenslet/D_lenslet
                                # (d_detector/f_lenslet) * spot_size = wvl/D_lenslet
                                # angular_pixel_scale * spot_size = lambda/D

    #SAVE THESE PARAMETERS
    myvar = {'N' : N,
             'lenslet_pixel' : lenslet_pixel,
             'D_lenslet' : D_lenslet,
             'L0' : L0,
             'l0' : l0,
             'r0' : r0,
             'wvl' : wvl,
             'SNR' : SNR,
             'z' : z,
             'rytov' : rytov,
             'zenith_angle' : zenith_angle,
             'pf' : pf,
             'pf_lenslet' : pf_lenslet,
             'spot_size': spot_size}
    with open('file.pkl', 'wb') as file:
        pickle.dump(myvar, file)

if LOAD_PARAMETERS == True:
    #LOAD PARAMETERS
    with open('file.pkl', 'rb') as file:
        myvar = pickle.load(file)
        N = myvar['N']
        lenslet_pixel = myvar['lenslet_pixel']
        D_lenslet = myvar['D_lenslet']
        L0 = myvar['L0']
        l0 = myvar['l0']
        r0 = myvar['r0']
        wvl = myvar['wvl']
        SNR = myvar['SNR']
        z = myvar['z']
        rytov = myvar['rytov']
        zenith_angle = myvar['zenith_angle']
        pf = myvar['pf']
        pf_lenslet = myvar['pf_lenslet']
        spot_size = myvar['spot_size']

#
# Dependent Parameters
#

# Screen parameters
NN = pf*N

# Screen physical sizes
d_in = D_lenslet/lenslet_pixel
d_out = d_in                # pixel size of screen
d_lenslet = d_in
d_detector = d_in

# Propagation parameters
num_lenslet = int(N/lenslet_pixel)
f_lenslet = d_detector*d_lenslet*spot_size*lenslet_pixel/wvl

# Turbulence parameters
r0_GL = np.zeros((r0.shape[0], rytov.shape[0]), dtype=float)
r0_HA = np.zeros((r0.shape[0], rytov.shape[0]), dtype=float)
for i in np.arange(r0.shape[0]):
    for j in np.arange(rytov.shape[0]):
        r0_GL[i,j], r0_HA[i,j] = distribute_coherent_length(
            r0[i], rytov[j], z[1], wvl, zenith_angle)

#
# Checking Parametres #
#

# Checking WFS coverage
wfs_need_space = 2*spot_size*(1 + 2*lenslet_pixel*d_in/r0.min())
if lenslet_pixel >= wfs_need_space:
    print('No problem with wfs coverage')
else:
    print('Please reconsider your wfs parameter choise.')
    temp = 2*spot_size*(1 + 2*lenslet_pixel*d_in/r0.min())
    print('lenslet_pixel should be larger than {:.2f}'.format(
        wfs_need_space))
    quit()

# Propagation between atmospheric layers
check = True
check_condition = mprop.prop_param_check(N, d_in, d_out, wvl, z, pf)
for i in check_condition:
    check = check * i
if check == True:
    print('Good condition for atm layer propagation.')
else:
    print("One of the propagation condition for atm layer is failing.")
    quit()

# Propagation through SHWFS
check = True
check_condition = mprop.prop_lens_param_check(lenslet_pixel, d_lenslet,
                                              d_detector, wvl, [f_lenslet],
                                              pf_lenslet)
for i in check_condition:
    check = check * i
if check == True:
    print('Good condition for lenslet propagation.')
else:
    print("One of the propagation condition for lenslet is failing.")
    quit()
    


#
# Declare & initialize variable
#

phase_HA_large = np.zeros((NN, NN), dtype=float)
phase_HA = np.zeros((N, N), dtype=float)
phase_GL = np.zeros((N, N), dtype=float)
phase = np.zeros((N, N), dtype=float)
I = np.zeros((N, N), dtype=float)

U_pupil = np.ones((N, N), dtype=complex)

rms_slope = np.zeros((r0.shape[0], rytov.shape[0]), dtype=float)

#
# Simulation starts
#

for i in np.arange(r0.shape[0]):
    for j in np.arange(rytov.shape[0]):
        if ~np.isnan(r0_HA[i,j]):       # not invalid strengths

            # make phase screen
            if r0_HA[i,j] != np.inf:    # HA have strength
                phase_HA_large = ft_sh_phase_screen(r0_HA[i,j], NN, d_in,
                                                    L0, l0)
            else:                       # only GL
                phase_HA_large = np.zeros((NN, NN), dtype=float)
            phase_HA = mprop.crop2(phase_HA_large, pf)
            phase_GL = ft_sh_phase_screen(r0[i], N, d_out, L0, l0)

            # propagate
            U_pupil = mprop.crop2(
                angularSpectrum(
                    np.exp(1j*phase_HA_large), wvl, d_in, d_out, z[1]),
                pf) * np.exp(1j*phase_GL)

            # real SHWFS    # phase gradient at lenslet projected to telescope
                            # lenslet size to get phase difference per lenslet
                            # convert phase difference in radian to nm
            realSHWFS_slope = ((mwfs.get_SHWFS_slope(
                U_pupil, wvl, d_lenslet, d_detector, f_lenslet,
                pf_lenslet, lenslet_pixel) * (d_lenslet / d_in))
                * (lenslet_pixel * d_in)) * (wvl / 2 / np.pi) / 1e-9

            # FAST SHWFS    # phase difference per pixel convert to per m get gradient
                            # * lenslet size to get phase difference per lenslet
                            # convert phase difference in radian to nm
            FASTSHWFS_slope = ((mwfs.get_intensity_weighted_slope(
                U_pupil, lenslet_pixel) / d_in) * (lenslet_pixel * d_in)
                * (wvl / 2 / np.pi)) / 1e-9

            # RMS slope
            correlation = norm_correlate1(realSHWFS_slope.flatten(), FASTSHWFS_slope.flatten())
            rms_slope[i,j] = ((FASTSHWFS_slope - realSHWFS_slope)).std()
            # plt.plot(realSHWFS_slope.flatten(), FASTSHWFS_slope.flatten(), marker='.', alpha=0.5, ls='')
            # x = np.linspace(FASTSHWFS_slope.min(), FASTSHWFS_slope.max())
            # plt.plot(x,x, ls='-')
            # plt.title('FAST vs real SHWFS slope per lenslet (nm)\nNormalized Correlation={:.6f}, lenslet={:.0f}cm'.format(correlation,D_lenslet*100))
            # plt.xlabel('FAST')
            # plt.ylabel('REAL')
            # plt.show()

        else:                           # invalid cases
            rms_slope[i,j] = np.NaN

# wavefront error per lenslet



# plt.pcolor(np.log10(rytov),r0*100,rms_slope)
plt.pcolor(np.log10(rytov),r0*100,rms_slope,cmap='rainbow',vmin=0,vmax=wvl/10/1e-9)
plt.colorbar()
plt.xlabel('log10 of log-amplitude variance')
plt.ylabel('r0 case (cm)')
plt.title('Wavefront RMS error per lenslet (nm)')
plt.savefig('realvsfast{:.0f}.png'.format(D_lenslet*100))
plt.show()
