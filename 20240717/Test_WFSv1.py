
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


#
# Constants & Independent Parameters
#

LOAD_PARAMETERS = True
# LOAD_PARAMETERS = True

if LOAD_PARAMETERS == False:
    # USE NEW PARAMETERS & constants

    # Screen simulation parameters
    num_lenslet = int(1)
    # i
    lenslet_pixel = np.array([16,32,64,128], dtype=int)
    # lenslet_pixel = int(16)     # px, number of pixel across a lenslet

    # Screen physical parameters
    # j
    D_lenslet = np.array([10,8,6,4,2], dtype=float)*1e-2 # j

    # Atmosphere parameters
    L0 = float(20)                  # m, Outer turbulence scale
    l0 = float(1e-7)                # m, Inner turbulence scale

    # k
    r0 = np.array([10,8,6,4,2], dtype=float)*1e-2 # k
                                    # m, Accumulated r0

    # Optical parameters
    wvl = float(500e-9)         # m, wavelength

    # Progation simulation parameters
    pf = int(2)                 # extra input scale for free atm prop
    pf_lenslet = int(2)         # zero padding scale for after pupil prop

    # WFS parameters
    spot_size = float(2)        # px, d_detector * spot_size = wvl*f_lenslet/D_lenslet
                                # (d_detector/f_lenslet) * spot_size = wvl/D_lenslet
                                # angular_pixel_scale * spot_size = lambda/D
    REPEAT = 100

    #SAVE THESE PARAMETERS
    myvar = {'num_lenslet' : num_lenslet,
             'lenslet_pixel' : lenslet_pixel,
             'D_lenslet' : D_lenslet,
             'L0' : L0,
             'l0' : l0,
             'r0' : r0,
             'wvl' : wvl,
             'pf' : pf,
             'pf_lenslet' : pf_lenslet,
             'spot_size': spot_size,
             'REPEAT' : REPEAT}
    with open('file.pkl', 'wb') as file:
        pickle.dump(myvar, file)

    #
    # Dependent Parameters
    #
    
    # i
    N = num_lenslet*lenslet_pixel
    # i
    fov = lenslet_pixel/2 # wvl/D
    
    # Screen physical sizes
    # i j
    d = np.zeros((lenslet_pixel.shape[0],D_lenslet.shape[0]), dtype=float)
    for i in np.arange(lenslet_pixel.shape[0]):
        for j in np.arange(D_lenslet.shape[0]):
            d[i,j] = D_lenslet[j]/lenslet_pixel[i]
    # Propagation parameters
    # i j
    f_lenslet = np.zeros((lenslet_pixel.shape[0],D_lenslet.shape[0]),
                         dtype=float)
    for i in np.arange(lenslet_pixel.shape[0]):
        for j in np.arange(D_lenslet.shape[0]):
            f_lenslet[i,j] = d[i,j]**2*spot_size*lenslet_pixel[i]/wvl
    
    # j k
    seeing = np.zeros((D_lenslet.shape[0],r0.shape[0]), dtype=float)
    for j in np.arange(D_lenslet.shape[0]):
        for k in np.arange(r0.shape[0]):
            seeing[j,k] = D_lenslet[j]/r0[k] # wvl/D
    
    # lenslet_pixel     = i
    # N                 = i
    # D_lenslet         = j
    # r0                = k
    # seeing = j k
    # d = i j
    # f_lenslet = i j
    # fov = i
    
    #
    # Checking Parametres #
    #
    
    # Propagation through SHWFS
    for i in np.arange(len(lenslet_pixel)):
        for j in np.arange(len(D_lenslet)):
            check = True
            check_condition = mprop.prop_lens_param_check(
                lenslet_pixel[i], d[i,j], d[i,j], wvl, [f_lenslet[i,j]],
                pf_lenslet)
            for ii in check_condition:
                check = check * ii
            if check == False:
                print("One of the propagation condition for lenslet is failing.")
                quit()
    
    #
    # Declare & initialize variable
    #
    
    meandifsq_REAL_g = np.zeros((lenslet_pixel.shape[0],
                           D_lenslet.shape[0],
                           r0.shape[0],
                           REPEAT), dtype=float)
    
    time_g = np.zeros((lenslet_pixel.shape[0],
                       D_lenslet.shape[0],
                       r0.shape[0],
                       REPEAT), dtype=float)
    time_REAL = np.zeros((lenslet_pixel.shape[0],
                          D_lenslet.shape[0],
                          r0.shape[0],
                          REPEAT), dtype=float)
    time_FAST = np.zeros((lenslet_pixel.shape[0],
                          D_lenslet.shape[0],
                          r0.shape[0],
                          REPEAT), dtype=float)
    performance = np.zeros((len(lenslet_pixel)),dtype=float)
    per_err = np.zeros((len(lenslet_pixel)),dtype=float)

    fov_to_seeing = np.zeros((lenslet_pixel.shape[0],
                              D_lenslet.shape[0],
                              r0.shape[0]), dtype=float)

    for i in np.arange(lenslet_pixel.shape[0]):
        for j in np.arange(D_lenslet.shape[0]):
            for k in np.arange(r0.shape[0]):
                fov_to_seeing[i,j,k] = fov[i]/seeing[j,k]

    #
    # Simulation starts
    #
    for i in np.arange(lenslet_pixel.shape[0]):
        phase_unwrap = np.zeros((N[i], N[i]), dtype=float)
        U_pupil = np.ones((N[i], N[i]), dtype=complex)
        g_slope = np.zeros((2, N[i], N[i]), dtype=float)
        realSHWFS_slope = np.zeros((2, N[i], N[i]), dtype=float)
        FASTSHWFS_slope = np.zeros((2, N[i], N[i]), dtype=float)
        for j in np.arange(D_lenslet.shape[0]):
            for k in np.arange(r0.shape[0]):
                for repeat in np.arange(REPEAT):
                    # make phase screen
                    phase_unwrap = ft_sh_phase_screen(r0[k], N[i], d[i,j], L0, l0)
                    Nyquist_check = mprop.check_Nyquist(phase_unwrap)
                    # print(d[i,j],D_lenslet[j],r0[k],repeat,Nyquist_check)
                    if Nyquist_check:
                        U_pupil = np.exp(1j*phase_unwrap)
                        # gradient slope
                        A = time.time()
                        g_slope = (mwfs.get_vec_lenslet(
                            mwfs.get_slope(np.angle(U_pupil)), lenslet_pixel[i])
                            * (lenslet_pixel[i] * wvl / 2. / np.pi / 1e-9))
                        B = time.time()
                        time_g[i,j,k,repeat] = B - A
                
                        # real SHWFS    # phase gradient at lenslet projected to telescope
                                        # lenslet size to get phase difference per lenslet
                        C = time.time()
                        realSHWFS_slope = mwfs.get_SHWFS_slope( # get m / lenslet
                            U_pupil, wvl, d[i,j], d[i,j], f_lenslet[i,j],
                            pf_lenslet, lenslet_pixel[i]) / 1e-9 # now nm / lenslet
                        D = time.time()
                        time_REAL[i,j,k,repeat] = D - C
            
                        # FAST SHWFS    # phase difference per pixel convert to per m get gradient
                                        # * lenslet size to get phase difference per lenslet
                                        # convert phase difference in radian to nm
                        E = time.time()
                        FASTSHWFS_slope = (mwfs.get_intensity_weighted_slope(
                            U_pupil, lenslet_pixel[i]) # get phase dif per pixel
                            * (lenslet_pixel[i] * wvl / 2. / np.pi / 1e-9 )) # slope in (nm / lenslet)
                        F = time.time()
                        time_FAST[i,j,k,repeat] = F - E
        
                        # x = np.linspace(FASTSHWFS_slope.min(), FASTSHWFS_slope.max())
        
                        # RMS g slope real
                        meandifsq_REAL_g[i,j,k,repeat] = np.mean((realSHWFS_slope - g_slope)**2) # (nm/lenslet)**2
                        # plt.plot(g_slope.flatten(),
                        #           realSHWFS_slope.flatten() - g_slope.flatten(),
                        #           marker='.', alpha=0.5, ls='')
                        # plt.plot(x, np.zeros_like(x), ls='-')
                        # plt.title('Difference between REAL SHWFS and'
                        #           + ' gradient per lenslet (nm)\nr0='
                        #           + '{:.0f}cm, lenslet={:.0f}cm'.format(
                        #               r0[i]*100, D_lenslet[j]*100))
                        # plt.xlabel('Gradient')
                        # plt.ylabel('REAL SHWFS - gradient')
                        # plt.savefig('real-gradient-{:d}{:d}.png'.format(i,j))
                        # plt.show()
                        # plt.clf()
                    else:
                        time_g[i,j,k,repeat] = np.NaN
                        time_REAL[i,j,k,repeat] = np.NaN
                        time_FAST[i,j,k,repeat] = np.NaN
                        meandifsq_REAL_g[i,j,k,repeat] = np.NaN
                        print(d[i,j],D_lenslet[j],r0[k],repeat,Nyquist_check)

    myvar2 = {'N' : N,
              'fov' : fov,
              'd' : d,
              'f_lenslet' : f_lenslet,
              'seeing' : seeing,
              'meandifsq_REAL_g' : meandifsq_REAL_g,
              'time_g' : time_g,
              'time_REAL' : time_REAL,
              'time_FAST' : time_FAST,
              'fov_to_seeing' : fov_to_seeing}
    with open('output.pkl', 'wb') as file:
        pickle.dump(myvar2, file)

if LOAD_PARAMETERS == True:
    #LOAD PARAMETERS
    with open('file.pkl', 'rb') as file:
        myvar = pickle.load(file)
        num_lenslet = myvar['num_lenslet']
        lenslet_pixel = myvar['lenslet_pixel']
        D_lenslet = myvar['D_lenslet']
        L0 = myvar['L0']
        l0 = myvar['l0']
        r0 = myvar['r0']
        wvl = myvar['wvl']
        pf = myvar['pf']
        pf_lenslet = myvar['pf_lenslet']
        spot_size = myvar['spot_size']
        REPEAT = myvar['REPEAT']
    with open('output.pkl', 'rb') as file:
        myvar2 = pickle.load(file)
        N = myvar2['N']
        fov = myvar2['fov']
        d = myvar2['d']
        f_lenslet = myvar2['f_lenslet']
        seeing = myvar2['seeing']
        meandifsq_REAL_g = myvar2['meandifsq_REAL_g']
        time_g = myvar2['time_g']
        time_REAL = myvar2['time_REAL']
        time_FAST = myvar2['time_FAST']
        fov_to_seeing = myvar2['fov_to_seeing']

    # lenslet_pixel     = i
    # N                 = i
    # D_lenslet         = j
    # r0                = k
    # seeing = j k
    # d = i j
    # f_lenslet = i j
    # fov = i

for i in np.arange(len(fov)):
    # # wavefront error per lenslet
    plt.pcolor(D_lenslet*100,r0*100,np.sqrt(np.nanmean(meandifsq_REAL_g[i], axis=-1)).T,
                cmap='rainbow',vmin=0,vmax=wvl/10/1e-9)
    plt.colorbar()
    plt.xlabel('lenslet size (cm)')
    plt.ylabel('r0 (cm)')
    plt.axis('square')
    plt.title('REAL SHWFS vs gradient\n'
              + 'Wavefront RMS error per lenslet (nm)\n'
              + 'lenslet={:d}px'.format(lenslet_pixel[i]))
    plt.savefig('realvsg{:d}.png'.format(lenslet_pixel[i]))
    plt.show()
    plt.clf()



for i in np.arange(len(fov)):
    plt.plot(fov_to_seeing[i,:,:].flatten(),
                 np.nanmean(meandifsq_REAL_g,axis=-1)[i,:,:].flatten(),
                 marker='.', ls='', alpha=0.5,label='fov={:.0f}wvl/D'.format(fov[i]))
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('fov/seeing')
plt.ylabel('Mean of square of difference (nm/lenslet)^2')
plt.title('REAL SHWFS error vs gradient')
plt.savefig('REALSHWFSerrorVSfov2seeing{:d}.png'.format(num_lenslet**2))
plt.show()
plt.clf()



time_FAST_mean = np.nanmean(
    time_FAST.reshape(
        (lenslet_pixel.shape[0],D_lenslet.shape[0]*r0.shape[0]*REPEAT)
        ), axis=-1)

time_REAL_mean = np.nanmean(
    time_REAL.reshape(
        (lenslet_pixel.shape[0],D_lenslet.shape[0]*r0.shape[0]*REPEAT)
        ), axis=-1)

time_FAST_std = np.nanstd(
    time_FAST.reshape(
        (lenslet_pixel.shape[0],D_lenslet.shape[0]*r0.shape[0]*REPEAT)
        ), axis=-1)

time_REAL_std = np.nanstd(
    time_REAL.reshape(
        (lenslet_pixel.shape[0],D_lenslet.shape[0]*r0.shape[0]*REPEAT)
        ), axis=-1)

performance = time_FAST_mean/time_REAL_mean*100
per_err = (np.sqrt(
    (time_FAST_std/time_FAST_mean)**2
    + (time_REAL_std/time_REAL_mean)**2)
    * (time_FAST_mean/time_REAL_mean) * 100)
for i in np.arange(len(performance)):
    print('REAL is {:.1f} +- {:.1f} ms for n={:d}.'.format(
        time_REAL_mean[i]*1e3, time_REAL_std[i]*1e3, lenslet_pixel[i]))
    print('FAST is {:.1f} +- {:.1f} ms for n={:d}.'.format(
        time_FAST_mean[i]*1e3, time_FAST_std[i]*1e3, lenslet_pixel[i]))
    print('FAST is only {:.1f} +- {:.1f} % of REAL for n={:d}.'.format(
        performance[i], per_err[i], lenslet_pixel[i]))

plt.plot(lenslet_pixel,performance, ls='', marker = 'o')
plt.xlabel('Lenslet diameter in pixels')
plt.ylabel('Time_FAST / Time_REAL %')
plt.show()

plt.plot(1000*time_REAL_mean,1000*time_FAST_mean, marker='o', ls='')
plt.xlabel('time REAL (ms)')
plt.ylabel('time FAST (ms)')
plt.show()

plt.plot(time_REAL_mean/time_FAST_mean, lenslet_pixel, marker='o', ls='')
plt.xlabel('time REAL / time FAST')
plt.ylabel('number of pixel across lenslet')
plt.show()

# fov-->[i],      d/r0 -->  [j,k]      variance-->[i,j,k]

dor0 = np.zeros((len(D_lenslet),len(r0)), dtype=float)
for j in np.arange(len(D_lenslet)):
    for k in np.arange(len(r0)):
        dor0[j,k]=D_lenslet[j]/r0[k]

#          # (nm/lenslet)**2 * (m/nm * rad/m)**2 = (rad/lenslet)**2
variance = np.nanmean(meandifsq_REAL_g,axis=-1) * (1e-9 / wvl * 2 * np.pi)**2
# for i in np.arange(len(D_lenslet)):
#     variance[:,i,:] = variance[:,i,:] * (D_lenslet[i])**2
x=np.logspace(0,2,100)
plt.plot(fov,variance[:,3,1],ls='',marker='+',color='r')
plt.plot(fov,variance[:,4,3],label='d/r0=0.5',ls='',marker='+',color='r')
plt.plot(fov,variance[:,0,0],ls='',marker='*',color='b')
plt.plot(fov,variance[:,1,1],ls='',marker='*',color='b')
plt.plot(fov,variance[:,2,2],ls='',marker='*',color='b')
plt.plot(fov,variance[:,3,3],ls='',marker='*',color='b')
plt.plot(fov,variance[:,4,4],label='d/r0=1',ls='',marker='*',color='b')
plt.plot(fov,variance[:,1,3],ls='',marker='D',color='g')
plt.plot(fov,variance[:,3,4],label='d/r0=2',ls='',marker='D',color='g')
plt.plot(fov,variance[:,2,4],label='d/r0=3',ls='',marker='^',color='c')
plt.plot(x,0.5*x**-2*(0.5)**(5/3),color='r')
plt.plot(x,0.5*x**-2*(1)**(5/3),color='b')
plt.plot(x,0.5*x**-2*(2)**(5/3),color='g')
plt.plot(x,0.5*x**-2*(3)**(5/3),color='c')
plt.xlabel('Sub-aperture FoV (wvl/D)')
plt.ylabel('Variance (rad^2)')
plt.xscale('log')
plt.yscale('log')
plt.xlim((1,100))
plt.ylim((0.0001,10))
plt.legend()
plt.show()
