# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:06:45 2022

@author: gqmc62
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:31:03 2022

@author: gqmc62
"""

import numpy as np
from matplotlib import pyplot as plt
from aotools.turbulence.phasescreen import ft_sh_phase_screen
import time
from aotools.opticalpropagation import angularSpectrum
from aotools.functions.zernike import zernikeArray
import my_propagation as mprop
import my_WFS as mwfs
import my_phasescreen as mps



def next_power_of_2(x):
    return int(2**np.ceil(np.log(x)/np.log(2)))



#
# Independent Parameter
#

n_lenslet = int(8)
SHWFS_PV_PX = int(1)
r0_per_lenslet = int(1)
r0 = float(5e-2)
D_lenslet = float(48e-6)
f_lenslet = float(0.6e-3)
wvl = float(500e-9)
z = 5e3
SNR = 100
lenslet_threshold = 0.99
Noll_coef = np.array([0.4479,0.448, # tip/tilt
                      0.023,0.023,0.023, # focus, astimatisms
                      0.0062,0.0062,0.0062,0.0062, # comas , trefoils
                      0.0024,0.0024,0.0024,0.0024,0.0024, # spherical, ...
                      0.0012,0.0012,0.0012,0.0012,0.0012,0.0012], dtype=float)
                        # have up to 20-21 = power of mode 20
                        # next will be 21-22 = power of mode 21
                        # 0.2944*(21**(-np.sqrt(3)/2.)-22**(-np.sqrt(3)/2.))



#
# Dependent Parameter
#

lenslet_pixel = next_power_of_2(2*np.ceil(2*SHWFS_PV_PX
                 + 2*0.98*SHWFS_PV_PX*r0_per_lenslet/2.))

N = int(lenslet_pixel*n_lenslet)
pixel_scale = float(r0/lenslet_pixel)
n_modes = int((n_lenslet/2 + 1)*(n_lenslet/2 + 2)/2 - 1)
d_lenslet = D_lenslet/lenslet_pixel
d_detector = wvl*f_lenslet/SHWFS_PV_PX/D_lenslet

if len(Noll_coef) > n_modes:
    Noll_coef = Noll_coef[0:n_modes]
else:
    Noll_coef = np.append(Noll_coef, 0.2944*(
        (np.arange(n_modes - len(Noll_coef)) + 21)**(-np.sqrt(3)/2.)
        - (np.arange(n_modes - len(Noll_coef)) + 22)**(-np.sqrt(3)/2.)))

# zernike_power = np.array([2,1,-1,-2], dtype=float)*np.sqrt(Noll_coef*(r0_per_lenslet)**(5./3.)*np.pi*(N/2.)**2)
# zernike_power = 2.0*np.sqrt(Noll_coef*(r0_per_lenslet)**(5./3.)*np.pi*(N/2.)**2)

pf = max([next_power_of_2(wvl*z/2./pixel_scale/pixel_scale/N + 1),
          next_power_of_2(wvl*z/pixel_scale/pixel_scale/N)])
# see conditions in  in mprop.prop_lens_param_check.
# derived from reversing requirements of propagation after definition of pupil.
# in this case, the simulation start with a flat pupil. propagate to -5km then to pupil again ...
# so there is a '/2' in there.

pf_lenslet = max([next_power_of_2(wvl*f_lenslet/2./d_lenslet/d_detector/n_lenslet + 1),
                  next_power_of_2(wvl*z/pixel_scale/pixel_scale/N)])

zernike_modes = zernikeArray(n_modes + 1, N)

pupil = zernike_modes[0]
pupil_pad = mprop.zero_pad2(pupil, pf)
lenslet_pupil = (mwfs.get_val_lenslet(pupil, lenslet_pixel)
                 >= (lenslet_threshold))



param_dict = {'n_lenslet':n_lenslet,'SH_WFS_PV_PX':SHWFS_PV_PX,
              'r0_per_lenslet':r0_per_lenslet, 'r0':r0,
              'D_lenslet':D_lenslet, 'f_lenslet':f_lenslet,
              'wvl':wvl, 'z':z, 'SNR':SNR,
              'lenslet_threshold':lenslet_threshold,
              'lenslet_pixel':lenslet_pixel,
              'N':N, 'pixel_scale':pixel_scale, 'n_modes':n_modes,
              'd_lenslet':d_lenslet, 'd_detector':d_detector,
              'pf':pf, 'pf_lenslet':pf_lenslet}
print(param_dict)

########
# Code #
########

# Double check parameters
mprop.prop_lens_param_check(N, pixel_scale, pixel_scale, wvl, [z], pf)
mprop.prop_lens_param_check(n_lenslet, d_lenslet, d_detector, wvl, [f_lenslet], pf_lenslet)

# assume flat input from 'bench source' with telescope size
U_in_big = pupil_pad # sqrt of 1 is 1

# Propagate it to -zkm where high altitude dm is
# U_in_big_HADM = angularSpectrum(U_in_big, wvl, pixel_scale, pixel_scale, -z)
U_in_big_HADM = U_in_big

for i in (np.arange(n_modes) + 1):
    for j in (np.arange(n_modes) + 1):
        
        # Push 'actuator' of HADM one by one
        phase_HADM = zernike_modes[j]
        lenslet_slope_HADM = mwfs.get_vec_lenslet(
            mwfs.get_slope(phase_HADM, pixel_scale, wvl), lenslet_pixel)
        
        U_out_big_HADM = U_in_big_HADM*np.exp(
            1j*mprop.zero_pad2(phase_HADM, pf))
        # U_out_big_HADM = pupil_pad*U_in_big_HADM*np.exp(
        #     1j*mprop.zero_pad2(phase_HADM, pf))
        # U_out_big_HADM = pupil_pad*U_in_big_HADM
        # U_out_big_HADM = U_in_big_HADM
        
        # propagate to the next GLDM
        U_in_big_GLDM = angularSpectrum(U_out_big_HADM, wvl, pixel_scale, pixel_scale, z)
        
        # add existing phase of GLDM
        phase_GLDM = zernike_modes[i]
        lenslet_slope_GLDM = mwfs.get_vec_lenslet(
            mwfs.get_slope(phase_GLDM, pixel_scale, wvl), lenslet_pixel)
        
        U_out_big_GLDM = U_in_big_GLDM*np.exp(
            1j*mprop.zero_pad2(phase_GLDM, pf))
        # U_out_big_GLDM = pupil_pad*U_in_big_GLDM*np.exp(
        #     1j*mprop.zero_pad2(phase_GLDM, pf))
        # U_out_big_GLDM = pupil_pad*U_in_big_GLDM
        # U_out_big_GLDM = U_in_big_GLDM
        
        # Propagate to WFS
        # they are actually the same. but yeah, let's crop it
        U_lenslet = pupil*mprop.crop2(U_out_big_GLDM, pf)
        
        # proper wfs simulation
        I_detector = np.abs(
            mwfs.propagate_lenslet(
                U_lenslet, wvl, d_lenslet, f_lenslet, lenslet_pixel,
                d_detector, pf_lenslet))**2 * (d_detector/d_lenslet)**2
        
        # noisy_I_detector = (np.random.poisson(
        #     I_detector/lenslet_pixel**2*SNR**2)
        #     / SNR**2*lenslet_pixel**2)
        # slope = mwfs.get_SHWFS_slope(noisy_I_detector,
        #                              lenslet_pixel, d_detector,
        #                              f_lenslet) * d_lenslet / pixel_scale
        slope = mwfs.get_SHWFS_slope(I_detector,
                                     lenslet_pixel, d_detector,
                                     f_lenslet) * d_lenslet / pixel_scale
        
        # plots
        
        # magical useful stuff for plot
        ruler = (np.arange(-N/2, N/2) + 0.5)*pixel_scale
        lenslet_edge = np.arange(-n_lenslet/2, n_lenslet/2 + 1)*pixel_scale*lenslet_pixel
        screen_radius = N*pixel_scale/2
        lenslet_center = (np.arange(-n_lenslet/2, n_lenslet/2) + 0.5)*pixel_scale*lenslet_pixel
        
        conv_phase = np.angle(np.exp(1j*(phase_GLDM + phase_HADM)))
        conv_slope = mwfs.get_vec_lenslet(
            mwfs.get_slope(conv_phase, pixel_scale, wvl), lenslet_pixel)
        # final_phase = (phase_GLDM + phase_HADM)
        wfs_phase = pupil*np.angle(U_lenslet)
        # wfs_phase = pupil*(np.unwrap(np.unwrap(np.angle(U_lenslet), axis=-1), axis=-2))
        
        plt.pcolor(ruler, ruler,
                   phase_HADM,
                   shading='auto', snap=True)
        plt.colorbar()
        plt.vlines(lenslet_edge,
                   ymin=-screen_radius,
                   ymax=screen_radius,
                   color='m', linewidth=1)
        plt.hlines(lenslet_edge,
                   xmin=-screen_radius,
                   xmax=screen_radius,
                   color='m', linewidth=1)
        plt.quiver(lenslet_center, lenslet_center,
                   lenslet_slope_HADM[:,:,0] * lenslet_pupil, 
                   lenslet_slope_HADM[:,:,1] * lenslet_pupil,
                   color='r')
        plt.title('HADM phase')
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()
        
        plt.pcolor(ruler, ruler,
                   phase_GLDM,
                   shading='auto', snap=True)
        plt.colorbar()
        plt.vlines(lenslet_edge,
                   ymin=-screen_radius,
                   ymax=screen_radius,
                   color='m', linewidth=1)
        plt.hlines(lenslet_edge,
                   xmin=-screen_radius,
                   xmax=screen_radius,
                   color='m', linewidth=1)
        plt.quiver(lenslet_center, lenslet_center,
                   lenslet_slope_GLDM[:,:,0] * lenslet_pupil,
                   lenslet_slope_GLDM[:,:,1] * lenslet_pupil,
                   color='r')
        plt.title('GLDM phase')
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()
        
        plt.pcolor(ruler, ruler,
                   conv_phase,
                   shading='auto', snap=True)
        plt.colorbar()
        plt.vlines(lenslet_edge,
                   ymin=-screen_radius,
                   ymax=screen_radius,
                   color='m', linewidth=1)
        plt.hlines(lenslet_edge,
                   xmin=-screen_radius,
                   xmax=screen_radius,
                   color='m', linewidth=1)
        plt.quiver(lenslet_center, lenslet_center,
                   conv_slope[:,:,0] * lenslet_pupil, conv_slope[:,:,1] * lenslet_pupil,
                   color='r')
        plt.title('conventional phase')
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()
        
        plt.pcolor(ruler, ruler,
                   wfs_phase,
                   shading='auto', snap=True)
        plt.colorbar()
        plt.vlines(lenslet_edge,
                   ymin=-screen_radius,
                   ymax=screen_radius,
                   color='m', linewidth=1)
        plt.hlines(lenslet_edge,
                   xmin=-screen_radius,
                   xmax=screen_radius,
                   color='m', linewidth=1)
        plt.quiver(lenslet_center, lenslet_center,
                   slope[:,:,0] * lenslet_pupil, slope[:,:,1] * lenslet_pupil,
                   color='r')
        plt.title('wfs phase')
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()
        
