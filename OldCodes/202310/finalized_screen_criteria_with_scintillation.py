# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:29:29 2023

investigate the following parameter for shwfs

1 number of input pixels
2 fov of the output
3 pixel scale of the output
4 number of output pixels

@author: gqmc62
"""

def distribute_coherent_length(r0, Rytov, HA, wvl, zenith_angle):
    """
    Calculate coherent length in 2-layer atm to get the given r0 and Rytov.

    Parameters
    ----------
    r0 : float
        (m)
        Desired total turbulence strength.
    Rytov : float
        (unitless)
        Desired Rytov number = sigma_chi^2 = variance of log amplitude.
    HA : float
        (m)
        High Altitude height.
    wvl : float
        (m)
        Propagation wavelength.
    zenith_angle : float#
        (radian)

    Returns
    -------
    r0_GL : float
        Required turbulent strength at the Ground Layer.
    r0_HA : float
        Required turbulent strength at the High Altitude.
    np.NaN : np.NaN
        If the given situation is impossible.
    np.inf : if no turbulence
    """
    if Rytov == 0:
        return r0, np.inf
    k = 2*np.pi/wvl
    c2ndz_tot = r0**(-5./3.)/(0.423*k**2/np.cos(zenith_angle))
    c2ndz_HA = Rytov / (0.5631 * k**(7./6.)
                        / (np.cos(zenith_angle))**(11./6.)
                        * HA**(5./6.))
    if c2ndz_HA >= c2ndz_tot :
        # print('WARNING: Some conditions are impossible. Return np.NaN.')
        return np.NaN, np.NaN
    c2ndz_GL = c2ndz_tot - c2ndz_HA
    r0_GL = (0.423*k**2/np.cos(zenith_angle)*c2ndz_GL)**(-3./5.)
    r0_HA = (0.423*k**2/np.cos(zenith_angle)*c2ndz_HA)**(-3./5.)

    return r0_GL, r0_HA

def rebin(arr, new_shape):
    """
    Direct copy from the following source.
    
    https://scipython.com/blog/binning-a-2d-array-in-numpy/

    Parameters
    ----------
    arr : TYPE
        DESCRIPTION.
    new_shape : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

import numpy as np
from matplotlib import pyplot as plt
from aotools.turbulence import infinitephasescreen

import MyWFS as MWFS
from tqdm import tqdm
import mypropagationclass as MPROP

# wvl = 500e-9
# dor0 = float(3)
# r0 = 2e-2
# d = dor0*r0
# N_GL = int(128)
# H_HA = 0.45e3
# delta = d/N_GL
# Rytov = 0.15
# zenith_angle = 0

# wvl = 500e-9
# dor0 = float(3)
# r0 = 2e-2
# d = dor0*r0
# N_GL = int(64)
# H_HA = 0.9e3
# delta = d/N_GL
# Rytov = 0.3
# zenith_angle = 0

wvl = 500e-9
dor0 = float(3)
r0 = 2e-2
d = dor0*r0
N_GL = int(32)
H_HA = 5e3
delta = d/N_GL
Rytov = 0.3
zenith_angle = 0

r0_GL, r0_HA = distribute_coherent_length(r0, Rytov, H_HA, wvl, zenith_angle)
print(r0_GL,r0_HA)

PROP = MPROP.OneAngularPropAtm(wvl,delta,N_GL,H_HA)
N_HA = PROP.NN
print(N_HA)

# L0_GEN_HA = float(20)
# d_GEN_HA = L0_GEN_HA/1
# r0_GEN_HA = d_GEN_HA/dor0
# delta_GEN_HA = d_GEN_HA/N_HA
# scaling_factor_HA = (r0_GEN_HA/delta_GEN_HA*delta/r0)**(5./6.)

L0_GEN_HA = float(20)
d_GEN_HA = L0_GEN_HA/2
r0_GEN_HA = d_GEN_HA/dor0
delta_GEN_HA = d_GEN_HA/N_HA
scaling_factor_HA = (r0_GEN_HA/delta_GEN_HA*delta/r0)**(5./6.)

phasescreen_HA = infinitephasescreen.PhaseScreenVonKarman(N_HA, delta_GEN_HA, r0_GEN_HA, L0_GEN_HA)
for i in np.arange(2*N_HA):
    phasescreen_HA.add_row()
main_phasescreen_HA = phasescreen_HA.scrn - phasescreen_HA.scrn.mean()
plt.imshow(main_phasescreen_HA)
plt.show()

L0_GEN_GL = float(20)
d_GEN_GL = L0_GEN_GL/8
r0_GEN_GL = d_GEN_GL/dor0
delta_GEN_GL = d_GEN_GL/N_GL
scaling_factor_GL = (r0_GEN_GL/delta_GEN_GL*delta/r0)**(5./6.)

phasescreen_GL = infinitephasescreen.PhaseScreenVonKarman(N_GL, delta_GEN_GL, r0_GEN_GL, L0_GEN_GL)
for i in np.arange(2*N_GL):
    phasescreen_GL.add_row()
main_phasescreen_GL = phasescreen_GL.scrn - phasescreen_GL.scrn.mean()
plt.imshow(main_phasescreen_GL)
plt.show()

PROP.xtra_init()

REPEAT = 10
alpha = 0.5 # in wvl/d

screen_sizes = (2**np.arange( 1,int( np.log(N_GL)/np.log(2) + 1) ) )
FOVs = (2**np.arange(1,int(np.log(N_GL)/np.log(2) + 1))) # unit of wvl/d

data = np.zeros((REPEAT,screen_sizes.shape[0],FOVs.shape[0],2), dtype = float)
dif = np.zeros((REPEAT,screen_sizes.shape[0] - 1,FOVs.shape[0] - 1,2), dtype = float)

for repeat in tqdm(np.arange(REPEAT)):
    for i in np.arange(2*N_GL):
        phasescreen_GL.add_row()
    main_phasescreen_GL = phasescreen_GL.scrn - phasescreen_GL.scrn.mean()
    proper_screen_GL = main_phasescreen_GL * scaling_factor_GL

    for i in np.arange(2*N_HA):
        phasescreen_HA.add_row()
    main_phasescreen_HA = phasescreen_HA.scrn - phasescreen_HA.scrn.mean()
    proper_screen_HA = main_phasescreen_HA * scaling_factor_HA

    U_HA_at_GL = (PROP.prop(np.exp(1j*proper_screen_HA))
                         )[int((N_HA - N_GL)/2)
                           : int((N_HA + N_GL)/2),
                           int((N_HA - N_GL)/2)
                           : int((N_HA + N_GL)/2)]
    U = U_HA_at_GL * np.exp(1j*proper_screen_GL)
    # plt.imshow(main_phasescreen_HA)
    # plt.show()
    # plt.imshow(main_phasescreen_GL)
    # plt.show()
    # plt.imshow(np.angle(U_HA_at_GL))
    # plt.show()
    # plt.imshow(np.angle(U))
    # plt.show()

    for size_i in np.arange(len(screen_sizes)):
        screen = rebin(U,(screen_sizes[size_i],screen_sizes[size_i]))
        # plt.imshow(screen)
        # plt.show()
        for fov_i in np.arange(len(FOVs)):
            WFS = MWFS.ConsecutiveAngularPropSHWFS(screen_sizes[size_i], d, alpha, FOVs[fov_i], wvl)
            data[repeat,size_i,fov_i,:] = WFS.measure_no_noise_no_threshold(screen)





most_accurate_data = data[:,-1,-1,:]

rms_fractional_error = np.zeros((screen_sizes.shape[0]-1,FOVs.shape[0]-1),dtype=float)
rms_error = np.zeros((screen_sizes.shape[0]-1,FOVs.shape[0]-1),dtype=float)
rms_error_to_rms_data = np.zeros((screen_sizes.shape[0]-1,FOVs.shape[0]-1),dtype=float)

for repeat in np.arange(REPEAT):
    for size_i in np.arange(len(screen_sizes) - 1):
        for fov_i in np.arange(len(FOVs) - 1):
            dif[repeat,size_i,fov_i,:] = data[repeat,size_i,fov_i,:] - most_accurate_data[repeat,:]

for size_i in np.arange(len(screen_sizes) - 1):
    for fov_i in np.arange(len(FOVs) - 1):
        rms_fractional_error[size_i,fov_i] = np.sqrt(np.mean((dif[:,size_i,fov_i,:]/most_accurate_data)**2))
        rms_error[size_i,fov_i] = np.sqrt(np.mean((dif[:,size_i,fov_i,:])**2))
        rms_error_to_rms_data[size_i,fov_i] = np.sqrt(np.mean((dif[:,size_i,fov_i,:])**2))/np.sqrt(np.mean((most_accurate_data)**2))
        
plt.pcolor(np.log(screen_sizes[:-1])/np.log(2),
           np.log(FOVs[:-1])/np.log(2),
           np.log10(rms_error_to_rms_data).T)
plt.colorbar()
plt.xlabel('screen_sizes in pixels in power of 2')
plt.ylabel('FoV in wvl/d in power of 2')
plt.title('log10 of rms error of shwfs measurement\npixel scale = 0.5wvl/d')
plt.axis('square')
plt.show()

for fov_i in np.arange(len(FOVs)-1):
    plt.plot(screen_sizes[:-1],rms_error_to_rms_data[:,fov_i],label='{:d} wvl/d fov'.format(FOVs[fov_i]),marker='o',ls='--')
    
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of input pixels')
plt.ylabel('rms error to rms data')
plt.legend()
# plt.savefig('good_input_px_number_scintillation.png')
plt.show()

for size_i in np.arange(len(screen_sizes)-1):
    plt.plot(FOVs[:-1],rms_error_to_rms_data[size_i,:],label='{:d} input pixels'.format(screen_sizes[size_i]),marker='o',ls='--')
    
plt.xscale('log')
plt.yscale('log')
plt.xlabel('SHWFS FoV in wvl/d')
plt.ylabel('rms error to rms data')
plt.legend()
# plt.savefig('good_shwfs_fov_scintillation.png')
plt.show()


accuracy_gain = rms_error_to_rms_data[-1,:-1] - rms_error_to_rms_data[-1,1:]

increase_data = screen_sizes[1:-1]**2 - screen_sizes[:-2]**2

plt.plot(screen_sizes[1:-1],accuracy_gain/increase_data,marker='o')
plt.xlabel('pixels across a screen')
plt.ylabel('accuracy gain per pixels')
plt.xscale('log')
plt.yscale('log')
# plt.savefig('phase_screen_size_to_accuracy_gain_in_SHWFS_scintillation.png')
plt.show()

plt.plot(screen_sizes[:-1],rms_error_to_rms_data[:,-1],marker='o')
plt.xlabel('screen size in px')
plt.ylabel('relative error')
plt.show()
