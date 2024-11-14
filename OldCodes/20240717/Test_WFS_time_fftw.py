
import numpy as np
import pyfftw
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

from aotools import fouriertransform



#
# Constants & Independent Parameters
#

LOAD_PARAMETERS = True
# LOAD_PARAMETERS = True

if LOAD_PARAMETERS == False:
    # USE NEW PARAMETERS & constants

    # Screen simulation parameters

    # i
    lenslet_pixel = np.array([4,8,16,32,64,128,256])
    # lenslet_pixel = np.array([16,32,64])
    # Screen physical parameters
    #
    D_lenslet = float(2e-2)

    # Atmosphere parameters
    L0 = float(20)                  # m, Outer turbulence scale
    l0 = float(1e-7)                # m, Inner turbulence scale

    #
    r0 = float(10e-2)
                                    # m, Accumulated r0

    # Optical parameters
    wvl = float(500e-9)         # m, wavelength

    # Progation simulation parameters
    pf_lenslet = int(2)         # zero padding scale for after pupil prop

    # WFS parameters
    spot_size = float(2)        # px, d_detector * spot_size = wvl*f_lenslet/D_lenslet
                                # (d_detector/f_lenslet) * spot_size = wvl/D_lenslet
                                # angular_pixel_scale * spot_size = lambda/D
    REPEAT = 10000#int(100*100)

    #SAVE THESE PARAMETERS
    myvar = {'lenslet_pixel' : lenslet_pixel,
             'D_lenslet' : D_lenslet,
             'L0' : L0,
             'l0' : l0,
             'r0' : r0,
             'wvl' : wvl,
             'pf_lenslet' : pf_lenslet,
             'spot_size': spot_size,
             'REPEAT' : REPEAT}
    with open('FASTTimeInputv3.pkl', 'wb') as file:
        pickle.dump(myvar, file)

    #
    # Dependent Parameters
    #

    # i
    N = lenslet_pixel

    # Screen physical sizes
    # i
    d = D_lenslet/lenslet_pixel
    # Propagation parameters
    # i
    f_lenslet = d**2*spot_size*lenslet_pixel/wvl

    #
    # Checking Parametres #
    #
    
    # Propagation through SHWFS
    for i in np.arange(len(lenslet_pixel)):
        check = True
        check_condition = mprop.prop_lens_param_check(
            lenslet_pixel[i], d[i], d[i], wvl, [f_lenslet[i]],
            pf_lenslet)
        for ii in check_condition:
            check = check * ii
        if check == False:
            print("One of the propagation condition for lenslet is failing.")
            quit()
    
    #
    # Declare & initialize variable
    #

    time_g = np.zeros((lenslet_pixel.shape[0],
                       REPEAT), dtype=float)
    time_REAL = np.zeros((lenslet_pixel.shape[0],
                       REPEAT), dtype=float)
    time_FAST = np.zeros((lenslet_pixel.shape[0],
                       REPEAT), dtype=float)

    #
    # Simulation starts
    #
    
    g_slope = np.zeros((2), dtype=float)
    realSHWFS_slope = np.zeros((2), dtype=float)
    FASTSHWFS_slope = np.zeros((2), dtype=float)

    slope = np.zeros((2), dtype=float)
    wavenumber = 2*np.pi/wvl     #optical wavevector

    

    for i in np.arange(lenslet_pixel.shape[0]):
        phase_unwrap = np.zeros((N[i], N[i]), dtype=float)
        U_pupil = np.ones((N[i], N[i]), dtype=complex)
        xx, yy = np.meshgrid(np.arange(lenslet_pixel[i]), np.arange(lenslet_pixel[i]))
        convert_g = (lenslet_pixel[i] * wvl / 2. / np.pi / 1e-9)
        convert_real = (d[i]**2 / f_lenslet[i] * lenslet_pixel[i] / 1e-9)
        convert_fast = (lenslet_pixel[i] * wvl / 2. / np.pi / 1e-9 )
        
        (x1,y1) = np.meshgrid(d[i]*np.arange(-N[i]*pf_lenslet/2,N[i]*pf_lenslet/2),
                                 d[i]*np.arange(-N[i]*pf_lenslet/2,N[i]*pf_lenslet/2))
        r1sq = (x1**2 + y1**2) + 1e-10
        df1 = 1. / (N[i]*pf_lenslet*d[i])
        fX,fY = np.meshgrid(df1*np.arange(-N[i]*pf_lenslet/2,N[i]*pf_lenslet/2),
                               df1*np.arange(-N[i]*pf_lenslet/2,N[i]*pf_lenslet/2))
        fsq = fX**2 + fY**2
        Q2 = np.exp(-1j * np.pi**2 * 2 * f_lenslet[i]/wavenumber*fsq)
        Q = np.exp(-1j*wavenumber/2./f_lenslet[i] * (((x1 + y1) * d[i]) + r1sq))
        NN = N[i]*pf_lenslet
        edge_low = int(NN*((pf_lenslet - 1)/pf_lenslet)/2)
        edge_high = int(NN*((pf_lenslet + 1)/pf_lenslet)/2)
        add_each_side = int((pf_lenslet - 1)/2*N[i])
        cen = (lenslet_pixel[i] - 1)/2.
        # dd22 = d[i]**2

        F0 = pyfftw.empty_aligned((N[i]*pf_lenslet, N[i]*pf_lenslet), dtype=complex)
        F1 = pyfftw.empty_aligned((N[i]*pf_lenslet, N[i]*pf_lenslet), dtype=complex)
        F2 = pyfftw.empty_aligned((N[i]*pf_lenslet, N[i]*pf_lenslet), dtype=complex)
        F3 = pyfftw.empty_aligned((N[i]*pf_lenslet, N[i]*pf_lenslet), dtype=complex)
        FF01 = pyfftw.FFTW(F0, F1, axes=(-1,-2), direction='FFTW_FORWARD')
        FF23 = pyfftw.FFTW(F2, F3, axes=(-1,-2), direction='FFTW_BACKWARD')
        for repeat in np.arange(REPEAT):
            # make phase screen
            phase_unwrap = ft_sh_phase_screen(r0, N[i], d[i], L0, l0)
            U_pupil = np.exp(1j*phase_unwrap)

            # gradient slope
            A = time.time()
            g_slope = (np.mean(
                mwfs.get_slope(np.angle(U_pupil)),
                axis=(-1,-2)) * convert_g)
            B = time.time()
            time_g[i,repeat] = B - A

            # real SHWFS    # phase gradient at lenslet projected to telescope
                            # lenslet size to get phase difference per lenslet
            C = time.time()
            F0[:,:] = np.fft.fftshift(np.pad(U_pupil, (add_each_side,
                                                       add_each_side),
                                             'constant', constant_values=0.)
                                      * Q)
            F2[:,:] = np.fft.ifftshift(np.fft.fftshift(FF01()) * Q2)
            I_focus = np.abs((np.fft.ifftshift(FF23()))[edge_low : edge_high,
                                                        edge_low : edge_high])**2
            I_focus_total = np.sum(I_focus)
            slope[0] = np.sum(I_focus*xx)/I_focus_total
            slope[1] = np.sum(I_focus*yy)/I_focus_total
            realSHWFS_slope = (slope - cen) * convert_real
            D = time.time()
            time_REAL[i,repeat] = D - C
            # plt.imshow(I_focus*4/(N[i]**2))
            # plt.colorbar()
            # plt.show()

            # FAST SHWFS    # phase difference per pixel convert to per m get gradient
                            # * lenslet size to get phase difference per lenslet
                            # convert phase difference in radian to nm
            E = time.time()
            I_pupil = np.abs(U_pupil[1:-1,1:-1])**2
            FASTSHWFS_slope = (
                np.sum(mwfs.fast_slope(np.angle(U_pupil)) * I_pupil,
                       axis=(-1,-2))
                / np.sum(I_pupil))
            F = time.time()
            time_FAST[i,repeat] = F - E

        print(lenslet_pixel[i])

    myvar2 = {'time_g' : time_g,
              'time_REAL' : time_REAL,
              'time_FAST' : time_FAST}
    with open('FASTTimeOutputv3.pkl', 'wb') as file:
        pickle.dump(myvar2, file)

if LOAD_PARAMETERS == True:
    #LOAD PARAMETERS
    with open('FASTTimeInputv3.pkl', 'rb') as file:
        myvar = pickle.load(file)
        lenslet_pixel = myvar['lenslet_pixel']
        D_lenslet = myvar['D_lenslet']
        L0 = myvar['L0']
        l0 = myvar['l0']
        r0 = myvar['r0']
        wvl = myvar['wvl']
        pf_lenslet = myvar['pf_lenslet']
        spot_size = myvar['spot_size']
        REPEAT = myvar['REPEAT']
    with open('FASTTimeOutputv3.pkl', 'rb') as file:
        myvar2 = pickle.load(file)
        time_g = myvar2['time_g']
        time_REAL = myvar2['time_REAL']
        time_FAST = myvar2['time_FAST']
    N = lenslet_pixel
    d = D_lenslet/lenslet_pixel
    f_lenslet = d**2*spot_size*lenslet_pixel/wvl

time_FAST_mean = np.mean(
    time_FAST, axis=-1)

time_REAL_mean = np.mean(
    time_REAL, axis=-1)

time_FAST_std = np.std(
    time_FAST, axis=-1)

time_REAL_std = np.std(
    time_REAL, axis=-1)

performance = time_FAST_mean/time_REAL_mean*100
per_err = (np.sqrt(
    (time_FAST_std/time_FAST_mean)**2
    + (time_REAL_std/time_REAL_mean)**2)
    * (time_FAST_mean/time_REAL_mean) * 100)
for i in np.arange(len(performance)):
    print('REAL is {:.1f} +- {:.1f} us for n={:d}.'.format(
        time_REAL_mean[i]*1e3, time_REAL_std[i]*1e3, lenslet_pixel[i]))
    print('FAST is {:.1f} +- {:.1f} us for n={:d}.'.format(
        time_FAST_mean[i]*1e3, time_FAST_std[i]*1e3, lenslet_pixel[i]))
    print('FAST is only {:.1f} +- {:.1f} % of REAL for n={:d}.'.format(
        performance[i], per_err[i], lenslet_pixel[i]))

plt.errorbar(lenslet_pixel, performance, yerr=per_err, ls='', marker = 'o')
plt.xlabel('number of pixel across lenslet')
plt.ylabel('Time_FAST / Time_REAL (%)')
plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0,300)
# plt.ylim(20,80)
plt.grid()
plt.savefig('FASToverREALv3.png')
plt.show()
plt.clf()

plt.errorbar(x=time_REAL_mean, y=time_FAST_mean,
             xerr = time_REAL_std, yerr=time_FAST_std,
             marker='o', ls='')
# MIN = np.min(np.array([time_REAL_mean,time_FAST_mean]).flatten())
# MAX = np.max(np.array([time_REAL_mean,time_FAST_mean]).flatten())
# plt.plot(np.linspace(MIN*1e3,MAX*1e3),np.linspace(MIN*1e3,MAX*1e3),ls='-')
plt.xlabel('time REAL (s)')
plt.ylabel('time FAST (s)')
plt.xscale('log')
plt.yscale('log')
# plt.xlim(0.0001,0.1)
# plt.ylim(0.0001,0.01)
plt.grid()
plt.savefig('FASTvsREALv3.png')
plt.show()
plt.clf()

plt.errorbar(lenslet_pixel, time_REAL_mean/time_FAST_mean,
          yerr=(np.sqrt((time_FAST_std/time_FAST_mean)**2
                        + (time_REAL_std/time_REAL_mean)**2)
                * time_REAL_mean/time_FAST_mean), marker='o', ls='')
plt.ylabel('time REAL / time FAST')
plt.xlabel('number of pixel across lenslet')
plt.xscale('log')
# plt.yscale('log')
# plt.ylim(1,4)
# plt.xlim(0,300)
plt.grid()
plt.savefig('REALoverFASTv3.png')
plt.show()
plt.clf()