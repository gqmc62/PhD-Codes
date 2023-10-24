# -*- coding: utf-8 -*-
"""
█▀▀█░█░▒█░▀█▀░▀█▀░░▀░░█░░░█░█▀▀▄░▀█▀░░░▒█░▄▀░▄▀▀▄░█▀▀▄░█▀▀▀░█░▄░█▀▀▄░█▀▀░█░░░█
█▄▄█░█░▒█░░█░░░█░░░█▀░▀▄█▄▀░█▄▄█░░█░░░░▒█▀▄░░█░░█░█░▒█░█░▀▄░█▀▄░█▄▄█░█▀▀░▀▄█▄▀
█░░░░░▀▀▀░░▀░░░▀░░▀▀▀░░▀░▀░░▀░░▀░░▀░░░░▒█░▒█░░▀▀░░▀░░▀░▀▀▀▀░▀░▀░▀░░▀░▀▀▀░░▀░▀░
"""

"""Test SHWFS against fov parameter."""

import numpy as np
import pyfftw
# from aotools.functions.zernike import zernikeArray
from matplotlib import pyplot as plt
from aotools.turbulence.phasescreen import ft_sh_phase_screen
from aotools.turbulence.phasescreen import ft_phase_screen
# from tqdm import tqdm

def deming(x,y):
    """
    RETURN best fit parameter based on Deming orthogonal regression with delta=1.

    Parameters
    ----------
    x : x parameter
    y : y parameter

    Returns
    -------
    y = b0 + b1*x
    b0 : intercept
    b1 : slope

    """
    delta = 1
    xb = np.mean(x)
    yb = np.mean(y)
    sxx = np.mean(x**2)
    sxy = np.mean(x*y)
    syy = np.mean(y**2)
    b1 = (syy - delta*sxx + np.sqrt((syy - delta*sxx)**2 + 4*delta*sxy**2))/(2*sxy)
    b0 = yb - b1*xb
    return b0, b1

# # """
# # ░▄█░▒░░▄▀▀░▀█▀▒██▀▒█▀▄░░▒█▀▒█▀▄▒██▀░▄▀▀░█▄░█▒██▀░█▒░
# # ░░█░▀▀▒▄██░▒█▒░█▄▄░█▀▒▒░░█▀░█▀▄░█▄▄▒▄██░█▒▀█░█▄▄▒█▄▄
# # """

# N = int(64)
# oversize = np.array([16], dtype=int)# np.array([1,2,4,8,16], dtype=int) # 1 is default for phase screen generation
# zero_pad = np.array([16], dtype=int)# np.array([2,4,8,16], dtype=int) # 2 is default. # for SHWFS
# wvl = float(500e-9)
# L0 = float(20000)
# l0 = float(1e-12)

# dor0 = np.array([0.5,1,2,3], dtype=float)
# REPEAT2 =  int(1000)

# d = float(15e-2)
# r0 = d/dor0
# delta = d/N

# CEN = int(N/2)

# wavenumber = 2*np.pi/wvl
# # pupil, tip, tilt = zernikeArray(3,N)

# kolmogorov_phase = np.zeros((N, N), dtype=float)
# g_slope_kolmogorov = np.zeros((REPEAT2, len(dor0), 2), dtype=float)
# real_SHWFS_slope_kolmogorov = np.zeros((REPEAT2, len(dor0), CEN, 2), dtype=float)
# dif_SHWFS_slope_kolmogorov = np.zeros((REPEAT2, len(dor0), CEN, 2), dtype=float)

# for nzeropad in np.arange(len(zero_pad)):

# ###############################################################################
# # fov and threshold analysis kolmogorov slope with threshold #############
# ###############################################################################

#     F0 = pyfftw.empty_aligned((zero_pad[nzeropad]*N, zero_pad[nzeropad]*N), dtype=complex)
#     F1 = pyfftw.empty_aligned((zero_pad[nzeropad]*N, zero_pad[nzeropad]*N), dtype=complex)
#     FF01 = pyfftw.FFTW(F0, F1, axes=(-1,-2), direction='FFTW_FORWARD')

#     f_lenslet = delta**2*zero_pad[nzeropad]*N/wvl
#     alpha = 1 / zero_pad[nzeropad] # (unit is wvl / d)
#     N_zero_pad = N * zero_pad[nzeropad]
#     edge_low0 = int(N_zero_pad*((zero_pad[nzeropad] - 1)/zero_pad[nzeropad])/2)
#     edge_high0 = int(N_zero_pad*((zero_pad[nzeropad] + 1)/zero_pad[nzeropad])/2)
#     add_each_side0 = int((zero_pad[nzeropad] - 1)*N/2)

    # (y1, x1) = np.meshgrid(delta*np.arange(-N/2,N/2),
    #                       delta*np.arange(-N/2,N/2))
    # Q = np.exp(-1j*wavenumber/2./f_lenslet * (((x1 + y1) * delta)))
    # convert_real = ((delta/f_lenslet) # convert pixel drift to angle drift
    #                 * (delta * N) # convert angle drift to wavefront error
    #                 * (2*np.pi/wvl)) # convert wavefront error to phase error

#     for noversize in np.arange(len(oversize)):
#         NN = int(oversize[noversize]*N)
#         edge_low = int(NN*((oversize[noversize] - 1)/oversize[noversize])/2)
#         edge_high = int(NN*((oversize[noversize] + 1)/oversize[noversize])/2)
#         for repeat in np.arange(REPEAT2):
#             kolmogorov_phaser0min = ft_sh_phase_screen(r0.min(), oversize[noversize]*N, delta, L0, l0)[edge_low:edge_high,edge_low:edge_high]
#             for num_dor in np.arange(len(dor0)):
#                 kolmogorov_phase = kolmogorov_phaser0min * (r0.min()/r0[num_dor])**(5./6.)
#                 validx = (np.abs(kolmogorov_phase[1:,:] - kolmogorov_phase[:-1,:]) < np.pi).any()
#                 validy = (np.abs(kolmogorov_phase[:,1:] - kolmogorov_phase[:,:-1]) < np.pi).any()
#                 if (validx or validy) is False:
#                     real_SHWFS_slope_kolmogorov[repeat, num_dor, :, 0] = np.nan
#                     real_SHWFS_slope_kolmogorov[repeat, num_dor, :, 1] = np.nan
#                     dif_SHWFS_slope_kolmogorov[repeat, num_dor, :, : ] = np.nan
#                     print('nan')
#                     break

        
#                 U_pupil = np.exp(1j*kolmogorov_phase)
                
#                 # unit is rad per lenslet
#                 g_slope_kolmogorov[repeat,num_dor,0] = np.mean(kolmogorov_phase[1:,:] - kolmogorov_phase[:-1,:]) * N
#                 g_slope_kolmogorov[repeat,num_dor,1] = np.mean(kolmogorov_phase[:,1:] - kolmogorov_phase[:,:-1]) * N
            
#                 F0[:,:] = np.pad(U_pupil * Q,
#                                   (add_each_side0, add_each_side0),
#                                   'constant', constant_values=0.)
#                 I_focus = (np.abs(
#                     np.fft.fftshift(FF01())[edge_low0 : edge_high0,
#                                             edge_low0 : edge_high0])**2
#                     / (2*N**2)**2)
        
#                 I_focus = I_focus/I_focus.max()
#                 # I_focus = I_focus - np.max([I_focus[0,:].max(),
#                 #                             I_focus[-1,:].max(),
#                 #                             I_focus[:,0].max(),
#                 #                             I_focus[:,-1].max()])
#                 # I_focus[np.where(I_focus < 0)] = 0.0
#                 # I_focus = I_focus/np.sum(I_focus)
        
#                 # unit of half fov is pixel! = fov in wvl/d
#                 for half_fov in (np.arange(CEN) + 1):
#                     yy, xx = np.meshgrid(np.arange(-half_fov, half_fov) + 0.5,
#                                           np.arange(-half_fov, half_fov) + 0.5)
#                     # x = np.arange(-half_fov, half_fov) + 0.5
            
#                     I_fov = I_focus[CEN - half_fov : CEN + half_fov,
#                                     CEN - half_fov : CEN + half_fov]
        
#                     # I_fov = I_fov/I_fov.max()
#                     I_fov = I_fov - np.max([I_fov[0,:].max(),
#                                                 I_fov[-1,:].max(),
#                                                 I_fov[:,0].max(),
#                                                 I_fov[:,-1].max()])
#                     # I_fov = I_fov - 1e-4
#                     I_fov[np.where(I_fov < 0)] = 0.0
#                     I_fov = I_fov/np.sum(I_fov)
        
#                     # unit is rad per lenslet
#                     # real_SHWFS_slope_flat[repeat,half_fov - 1, 0] = (np.sum(I_fov*xx)
#                     #                                                  / np.sum(I_fov)
#                     #                                                  * convert_real)
#                     # real_SHWFS_slope_flat[repeat,half_fov - 1, 1] = (np.sum(I_fov*yy)
#                     #                                                 / np.sum(I_fov)
#                     #                                                   * convert_real)
#                     real_SHWFS_slope_kolmogorov[repeat, num_dor, half_fov - 1, 0] = np.sum(I_fov * xx) / np.sum(I_fov) * convert_real
#                     real_SHWFS_slope_kolmogorov[repeat, num_dor, half_fov - 1, 1] = np.sum(I_fov * yy) / np.sum(I_fov) * convert_real
            
#                     dif_SHWFS_slope_kolmogorov[repeat, num_dor, half_fov - 1, : ] = (g_slope_kolmogorov[repeat, num_dor, :]
#                                                                     - real_SHWFS_slope_kolmogorov[repeat, num_dor, half_fov - 1,:])
#                 # if repeat == 0:
#                 #     plt.pcolor(alpha * (np.arange(N) - CEN + 0.5),
#                 #                alpha * (np.arange(N) - CEN + 0.5),
#                 #                I_focus, vmin=0, vmax=1,
#                 #                snap=True)
#                 #     plt.colorbar()
#                 #     plt.xlabel('x (wvl/d)')
#                 #     plt.ylabel('y (wvl/d)')
#                 #     plt.axis('square')
#                 #     plt.vlines(0,ymin=-CEN*alpha,ymax=CEN*alpha)
#                 #     plt.hlines(0,xmin=-CEN*alpha,xmax=CEN*alpha)
#                 #     plt.vlines(-0.5,ymin=-0.5,ymax=0.5)
#                 #     plt.vlines(0.5,ymin=-0.5,ymax=0.5)
#                 #     plt.hlines(-0.5,xmin=-0.5,xmax=0.5)
#                 #     plt.hlines(0.5,xmin=-0.5,xmax=0.5)
#                 #     plt.title('SHWFS image before threshold, FoV={:.0f}wvl/d\n'.format(
#                 #         alpha*N)
#                 #               + 'd/$r_0$={:.1f}, $r_0$={:.1f}cm, d={:.1f}cm'.format(
#                 #                   dor0[num_dor],r0[num_dor]*100,d*100))
#                 #     plt.show()
        
#         # for repeat in np.arange(REPEAT2):
#         #     plt.plot(np.arange(CEN) + 1,
#         #                   np.abs(dif_SHWFS_slope_kolmogorov[repeat,:,0]),marker='.')
#         # plt.legend()
#         # plt.title('Absolute difference between\nSHWFS and g_tilt in flat slope case\nwith threshold for varying centroid shift in px')
#         # plt.xlabel('FoV (wvl/d)')
#         # plt.ylabel('Mean of slope difference (rad)')
#         # plt.xscale('log')
#         # plt.yscale('log')
#         # plt.show()
        
#         CEN = int(N/2)
#         fov = 2 * (np.arange(CEN) + 1) * alpha # fov in wvl/d = half fov in pixel
#         var = (np.nanvar(dif_SHWFS_slope_kolmogorov,axis=(0,3), ddof=1)).T
#         # mean = (np.mean(dif_SHWFS_slope,axis=(2,4))[:,0,:]).T
#         dor = d/r0
#         std_var = 2*var/np.sqrt(2*REPEAT2 - 2)
        
#         for i in np.arange(len(r0)):
#             plt.errorbar(fov, var[:,i], yerr=std_var[:,i],
#                           marker='.', ls='',
#                       label='d/r0 = {:.2f}'.format(dor[i]))
#         # for i in np.arange(len(D_lenslet)):
#         #     plt.errorbar(fov, mean[:,i], yerr=np.sqrt(var[:,i]),
#         #                   marker='.', ls='',
#         #               label='d/r0 = {:.2f}'.format(dor[i]))
#         for i in np.arange(len(r0)):
#             if i == 0:
#                 plt.plot((np.arange(32) + 1), 0.5/(np.arange(32) + 1)**2*dor[i]**(5./3.), marker='', ls='-', color='k',
#                           label='fit')
#             else:
#                 plt.plot((np.arange(32) + 1), 0.5/(np.arange(32) + 1)**2*dor[i]**(5./3.), marker='', ls='-', color='k')
#         plt.legend()
#         plt.xscale('log')
#         plt.yscale('log')
#         plt.xlabel('Field-of-View ($\lambda$/d)')
#         plt.ylabel('Variance (rad$^2$)')
#         plt.xlim(1, 100)
#         plt.ylim(1e-4, 10)
#         plt.title('1-step Fresnel: zeropad={:d}, oversize={:d}'.format(zero_pad[nzeropad],oversize[noversize]))
#         # plt.savefig('centroidThomas2006.png')
#         plt.show()
#         plt.clf()

# # do error between shwfs to g-tilt

# # next do kolmogorov terbulence
# # how about deming fit to the comparison?

# ###############################################################################
# # angular-spectrum #######################################################
# ###############################################################################

# """
# ▒▄▀▄░█▄░█░▄▀▒░█▒█░█▒░▒▄▀▄▒█▀▄░▒░░▄▀▀▒█▀▄▒██▀░▄▀▀░▀█▀▒█▀▄░█▒█░█▄▒▄█
# ░█▀█░█▒▀█░▀▄█░▀▄█▒█▄▄░█▀█░█▀▄░▀▀▒▄██░█▀▒░█▄▄░▀▄▄░▒█▒░█▀▄░▀▄█░█▒▀▒█
# """

# N = int(64)

# # oversize = np.array([1], dtype=int)# np.array([1,2,4,8,16], dtype=int) # 1 is default for phase screen generation
# # zero_pad = np.array([2,4,8,16], dtype=int) # 2 is default. # for SHWFS

# # oversize = np.array([1,2,4,8,16], dtype=int) # 1 is default for phase screen generation
# # zero_pad = np.array([2], dtype=int)# np.array([2,4,8,16], dtype=int) # 2 is default. # for SHWFS

# oversize = np.array([1,16], dtype=int)# np.array([1,2,4,8,16], dtype=int) # 1 is default for phase screen generation
# zero_pad = np.array([2,16], dtype=int)# np.array([2,4,8,16], dtype=int) # 2 is default. # for SHWFS

# wvl = float(500e-9)
# L0 = float(20000)
# l0 = float(1e-12)

# dor0 = np.array([0.5,1,2,3], dtype=float)
# REPEAT2 =  int(50)

# d = float(15e-2)
# r0 = d/dor0
# delta = d/N

# # angular_pixel_scale = 1./2. # unit = wvl/d

# alpha = 1./2. # unit = wvl/d

# CEN = int(N/2)

# wavenumber = 2*np.pi/wvl
# # pupil, tip, tilt = zernikeArray(3,N)

# kolmogorov_phase = np.zeros((N, N), dtype=float)
# g_slope_kolmogorov = np.zeros((2), dtype=float)
# real_SHWFS_slope_kolmogorov = np.zeros((REPEAT2, len(dor0), CEN, 2), dtype=float)

# (y_cog, x_cog) = np.meshgrid(np.arange(-N/2,N/2) + 0.5, np.arange(-N/2,N/2) + 0.5)

# for nzeropad in np.arange(len(zero_pad)):
# ###############################################################################
# # fov and threshold analysis kolmogorov slope with threshold #############
# ###############################################################################

#     F0 = pyfftw.empty_aligned((zero_pad[nzeropad]*N, zero_pad[nzeropad]*N), dtype=complex)
#     F1 = pyfftw.empty_aligned((zero_pad[nzeropad]*N, zero_pad[nzeropad]*N), dtype=complex)
#     FF01 = pyfftw.FFTW(F0, F1, axes=(-1,-2), direction='FFTW_FORWARD')
#     F2 = pyfftw.empty_aligned((zero_pad[nzeropad]*N, zero_pad[nzeropad]*N), dtype=complex)
#     F3 = pyfftw.empty_aligned((zero_pad[nzeropad]*N, zero_pad[nzeropad]*N), dtype=complex)
#     FF23 = pyfftw.FFTW(F2, F3, axes=(-1,-2), direction='FFTW_BACKWARD')

#     f_lenslet = delta**2*N/wvl/alpha
#     N_zero_pad = N * zero_pad[nzeropad]
#     edge_low0 = int(N_zero_pad*((zero_pad[nzeropad] - 1)/zero_pad[nzeropad])/2)
#     edge_high0 = int(N_zero_pad*((zero_pad[nzeropad] + 1)/zero_pad[nzeropad])/2)
#     add_each_side0 = int((zero_pad[nzeropad] - 1)*N/2)

#     (y1, x1) = np.meshgrid(delta*np.arange(-N/2,N/2),
#                           delta*np.arange(-N/2,N/2))
#     r1sq = (x1**2 + y1**2) + 1e-10

#     df1 = 1. / (N_zero_pad*delta)
#     fY,fX = np.meshgrid(df1*np.arange(-N_zero_pad/2,N_zero_pad/2),
#                             df1*np.arange(-N_zero_pad/2,N_zero_pad/2))
#     fsq = fX**2 + fY**2
#     Q = np.exp(-1j*wavenumber/2./f_lenslet * (((x1 + y1) * delta) + r1sq))
#     Q2 = np.exp(-1j * np.pi**2 * 2 * f_lenslet/wavenumber*fsq) # is the zeropad size!!!

#     convert_real = ((delta/f_lenslet) # convert pixel drift to angle drift
#                     * (delta * N) # convert angle drift to wavefront error
#                     * (2*np.pi/wvl)) # convert wavefront error to phase error

#     for noversize in np.arange(len(oversize)):
#         NN = int(oversize[noversize]*N)
#         edge_low = int(NN*((oversize[noversize] - 1)/oversize[noversize])/2)
#         edge_high = int(NN*((oversize[noversize] + 1)/oversize[noversize])/2)
        
#         for repeat in np.arange(REPEAT2):
#             kolmogorov_phaser0min = ft_sh_phase_screen(r0.min(), oversize[noversize]*N, delta, L0, l0)[edge_low:edge_high,edge_low:edge_high]
#             for num_dor in np.arange(len(dor0)):
#                 kolmogorov_phase = kolmogorov_phaser0min * (r0.min()/r0[num_dor])**(5./6.)
#                 g_slope_kolmogorov[0] = np.mean(kolmogorov_phase[1:,:] - kolmogorov_phase[:-1,:])
#                 g_slope_kolmogorov[1] = np.mean(kolmogorov_phase[:,1:] - kolmogorov_phase[:,:-1])
#                 kolmogorov_phase = (kolmogorov_phase - x_cog*g_slope_kolmogorov[0]
#                                     - y_cog*g_slope_kolmogorov[1] - kolmogorov_phase.sum()/N**2)

#                 U_pupil = np.exp(1j*kolmogorov_phase)
#                 # unit is rad per lenslet

#                 F0[:,:] = np.fft.fftshift(np.pad(U_pupil * Q,
#                                   (add_each_side0, add_each_side0),
#                                   'constant', constant_values=0.))
#                 F2[:,:] = np.fft.ifftshift(Q2 * np.fft.fftshift(FF01()))
#                 I_focus = (np.abs(
#                     np.fft.ifftshift(FF23())[edge_low0 : edge_high0,
#                                             edge_low0 : edge_high0])**2
#                     / (2*N**2)**2)

#                 I_focus = I_focus/I_focus.max()

#                 # unit of half fov is pixel! = fov in wvl/d
#                 for half_fov in (np.arange(CEN) + 1):
#                     yy, xx = np.meshgrid(np.arange(-half_fov, half_fov) + 0.5,
#                                           np.arange(-half_fov, half_fov) + 0.5)

#                     I_fov = I_focus[CEN - half_fov : CEN + half_fov,
#                                     CEN - half_fov : CEN + half_fov]

#                     I_fov = I_fov/I_fov.max()
#                     I_fov = I_fov - np.max([I_fov[0,:].max(),
#                                                 I_fov[-1,:].max(),
#                                                 I_fov[:,0].max(),
#                                                 I_fov[:,-1].max()])
#                     I_fov[np.where(I_fov < 0)] = 0.0
#                     I_fov = I_fov/np.max(I_fov)
#                     I_fov = I_fov - 1e-4
#                     I_fov[np.where(I_fov < 0)] = 0.0
#                     I_fov = I_fov/np.max(I_fov)

#                     # unit is rad per lenslet
#                     real_SHWFS_slope_kolmogorov[repeat, num_dor, half_fov - 1, 0] = np.sum(I_fov * xx) / np.sum(I_fov) * convert_real
#                     real_SHWFS_slope_kolmogorov[repeat, num_dor, half_fov - 1, 1] = np.sum(I_fov * yy) / np.sum(I_fov) * convert_real

#         CEN = int(N/2)
#         fov = 2 * (np.arange(CEN) + 1) * alpha # fov in wvl/d = half fov in pixel
#         var = (np.nanvar(real_SHWFS_slope_kolmogorov,axis=(0,3), ddof=1)).T

#         dor = d/r0
#         std_var = 2*var/np.sqrt(2*REPEAT2 - 2)
        
#         for i in np.arange(len(r0)):
#             plt.errorbar(fov, var[:,i], yerr=std_var[:,i],
#                           marker='.', ls='',
#                       label='d/r0 = {:.2f}'.format(dor[i]))
#         for i in np.arange(len(r0)):
#             if i == 0:
#                 plt.plot((np.arange(32) + 1), 0.5/(np.arange(32) + 1)**2*dor[i]**(5./3.), marker='', ls='-', color='k',
#                           label='fit')
#             else:
#                 plt.plot((np.arange(32) + 1), 0.5/(np.arange(32) + 1)**2*dor[i]**(5./3.), marker='', ls='-', color='k')
#         plt.legend()
#         plt.xscale('log')
#         plt.yscale('log')
#         plt.xlabel('Field-of-View ($\lambda$/d)')
#         plt.ylabel('Variance (rad$^2$)')
#         plt.xlim(1, 100)
#         plt.ylim(1e-4, 10)
#         plt.title('angular-spectrum: zeropad={:d}, oversize={:d}'.format(zero_pad[nzeropad],oversize[noversize]))
#         plt.show()
#         plt.clf()


# ###############################################################################
# # angular-spectrum #######################################################
# ###############################################################################

# """
# ░█▀▄░█░░░░█▀▀▄░█▀▀▄░█▀▀▀░█▀▀░░░█▀▀░▄▀▀▄░▄░░░▄░░░▄░░░▄░░▀░░█▀▀▄
# ░█░░░█▀▀█░█▄▄█░█░▒█░█░▀▄░█▀▀░░░█▀░░█░░█░░█▄█░░░░░█▄█░░░█▀░█▄▄█
# ░▀▀▀░▀░░▀░▀░░▀░▀░░▀░▀▀▀▀░▀▀▀░░░▀░░░░▀▀░░░░▀░░░░░░░▀░░░▀▀▀░▀░░▀
# ░▒█▀▀▀█░▒█░▒█░▒█░░▒█░▒█▀▀▀░▒█▀▀▀█░░░█▀▀░▄▀▀▄░█▀▄░█▀▀▄░█░░░░█░░█▀▀░█▀▀▄░█▀▀▀░▀█▀░█░░░
# ░░▀▀▀▄▄░▒█▀▀█░▒█▒█▒█░▒█▀▀░░░▀▀▀▄▄░░░█▀░░█░░█░█░░░█▄▄█░█░░░░█░░█▀▀░█░▒█░█░▀▄░░█░░█▀▀█
# ░▒█▄▄▄█░▒█░▒█░▒▀▄▀▄▀░▒█░░░░▒█▄▄▄█░░░▀░░░░▀▀░░▀▀▀░▀░░▀░▀▀░░░▀▀░▀▀▀░▀░░▀░▀▀▀▀░░▀░░▀░░▀
# """

# REPEAT2 =  int(500)

# # oversize = np.array([16], dtype=int)# np.array([1,2,4,8,16], dtype=int) # 1 is default for phase screen generation
# oversize = np.array([1], dtype=int) # 1 is default for phase screen generation

# N = int(64)
# wvl = float(500e-9)
# L0 = float(20000)
# l0 = float(1e-12)
# # dor0 = np.array([0.5], dtype=float)#dor0 = np.array([3,2,1,0.5], dtype=float)
# dor0 = np.array([3,2,1,0.5], dtype=float)
# d = float(15e-2)
# r0 = d/dor0
# delta = d/N
# CEN = int(N/2)
# wavenumber = 2*np.pi/wvl
# Threshold = 1e-4

# # FoV in wvl/d unit
# FoV = np.arange(32-4+1, dtype=int) + 4 # zeropad >= 64/(np.arange(32)+1) but i can manage only 16 zeropad soooo fov >= 4 wvl/d is suitable
# zero_pad = np.asarray(2**np.ceil(np.log(N/FoV)/np.log(2)), dtype=int)
# f = d / (FoV*wvl/d)
# alpha = delta/f / (wvl/d) #in wvl/d
# # FoV zeropad, f, alpha has the same dimension.

# kolmogorov_phase = np.zeros((N, N), dtype=float)
# g_slope_kolmogorov = np.zeros((REPEAT2, len(dor0), 2), dtype=float)
# real_SHWFS_slope_kolmogorov = np.zeros((REPEAT2, len(dor0), len(FoV), 2), dtype=float)
# dif_SHWFS_slope_kolmogorov = np.zeros((REPEAT2, len(dor0), len(FoV), 2), dtype=float)

# (y1, x1) = np.meshgrid(delta*np.arange(-N/2,N/2),
#                       delta*np.arange(-N/2,N/2)) # for FFT only
# r1sq = (x1**2 + y1**2) + 1e-10                   # for FFT only
                
# (y_cog,x_cog) = np.meshgrid(np.arange(-N/2,N/2) + 0.5, np.arange(-N/2,N/2) + 0.5)

# for noversize in np.arange(len(oversize)):
#     NN = int(oversize[noversize]*N)
#     edge_low = int(NN*((oversize[noversize] - 1)/oversize[noversize])/2)
#     edge_high = int(NN*((oversize[noversize] + 1)/oversize[noversize])/2)

#     for repeat in np.arange(REPEAT2):
#         kolmogorov_phaser0min = ft_sh_phase_screen(r0.min(), oversize[noversize]*N, delta, L0, l0)[edge_low:edge_high,edge_low:edge_high]
#         for num_dor in np.arange(len(dor0)):
#             kolmogorov_phase = kolmogorov_phaser0min * (r0.min()/r0[num_dor])**(5./6.)
#             g_slope_kolmogorov[repeat,num_dor,0] = np.mean(kolmogorov_phase[1:,:] - kolmogorov_phase[:-1,:]) * N
#             g_slope_kolmogorov[repeat,num_dor,1] = np.mean(kolmogorov_phase[:,1:] - kolmogorov_phase[:,:-1]) * N
#             U_pupil = np.exp(1j*kolmogorov_phase)
#             # unit is rad per lenslet
            

#             for nzeropad in np.arange(len(zero_pad)):
#             ###############################################################################
#             # fov and threshold analysis kolmogorov slope with threshold #############
#             ###############################################################################
#                 F0 = pyfftw.empty_aligned((zero_pad[nzeropad]*N, zero_pad[nzeropad]*N), dtype=complex)
#                 F1 = pyfftw.empty_aligned((zero_pad[nzeropad]*N, zero_pad[nzeropad]*N), dtype=complex)
#                 FF01 = pyfftw.FFTW(F0, F1, axes=(-1,-2), direction='FFTW_FORWARD')
#                 F2 = pyfftw.empty_aligned((zero_pad[nzeropad]*N, zero_pad[nzeropad]*N), dtype=complex)
#                 F3 = pyfftw.empty_aligned((zero_pad[nzeropad]*N, zero_pad[nzeropad]*N), dtype=complex)
#                 FF23 = pyfftw.FFTW(F2, F3, axes=(-1,-2), direction='FFTW_BACKWARD')

#                 N_zero_pad = N * zero_pad[nzeropad]
#                 edge_low0 = int(N_zero_pad*((zero_pad[nzeropad] - 1)/zero_pad[nzeropad])/2)
#                 edge_high0 = int(N_zero_pad*((zero_pad[nzeropad] + 1)/zero_pad[nzeropad])/2)
#                 add_each_side0 = int((zero_pad[nzeropad] - 1)*N/2)

#                 df1 = 1. / (N_zero_pad*delta)
#                 fY,fX = np.meshgrid(df1*np.arange(-N_zero_pad/2,N_zero_pad/2),
#                                        df1*np.arange(-N_zero_pad/2,N_zero_pad/2))
#                 fsq = fX**2 + fY**2
#                 Q = np.exp(-1j*wavenumber/2./f[nzeropad] * (((x1 + y1) * delta) + r1sq))
#                 Q2 = np.exp(-1j * np.pi**2 * 2 * f[nzeropad]/wavenumber*fsq) # is the zeropad size!!!
            
#                 convert_real = ((delta/f[nzeropad]) # convert pixel drift to angle drift
#                                 * (delta * N) # convert angle drift to wavefront error
#                                 * (2*np.pi/wvl)) # convert wavefront error to phase error

#                 F0[:,:] = np.fft.fftshift(np.pad(U_pupil * Q,
#                                   (add_each_side0, add_each_side0),
#                                   'constant', constant_values=0.))
#                 F2[:,:] = np.fft.ifftshift(Q2 * np.fft.fftshift(FF01()))
#                 I_focus = (np.abs(
#                     np.fft.ifftshift(FF23())[edge_low0 : edge_high0,
#                                             edge_low0 : edge_high0])**2
#                     / (2*N**2)**2)

#                 I_focus = I_focus/I_focus.max()
#                 I_focus = I_focus - np.max([I_focus[0,:].max(),
#                                             I_focus[-1,:].max(),
#                                             I_focus[:,0].max(),
#                                             I_focus[:,-1].max()])
#                 I_focus[np.where(I_focus < 0)] = 0.0
#                 I_focus = I_focus/I_focus.max()
#                 I_focus = I_focus - Threshold
#                 I_focus[np.where(I_focus < 0)] = 0.0
#                 I_focus = I_focus/I_focus.max()

#                 # unit is rad per lenslet
#                 real_SHWFS_slope_kolmogorov[repeat, num_dor, nzeropad, 0] = np.sum(I_focus * x_cog) / np.sum(I_focus) * convert_real
#                 real_SHWFS_slope_kolmogorov[repeat, num_dor, nzeropad, 1] = np.sum(I_focus * y_cog) / np.sum(I_focus) * convert_real

#                 dif_SHWFS_slope_kolmogorov[repeat, num_dor, nzeropad, : ] = (g_slope_kolmogorov[repeat, num_dor, :]
#                                                                 - real_SHWFS_slope_kolmogorov[repeat, num_dor, nzeropad,:])
#                 if ((repeat == 0) and (num_dor == 0)):
#                     plt.pcolor(x_cog*alpha[nzeropad],
#                                 y_cog*alpha[nzeropad],
#                                 I_focus, vmin=1e-4, vmax=1,
#                                 snap=True)
#                     plt.colorbar()
#                     plt.xlabel('x (wvl/d)')
#                     plt.ylabel('y (wvl/d)')
#                     plt.axis('square')
#                     plt.plot(real_SHWFS_slope_kolmogorov[repeat,num_dor,nzeropad,0]/convert_real*alpha[nzeropad],
#                              real_SHWFS_slope_kolmogorov[repeat,num_dor,nzeropad,1]/convert_real*alpha[nzeropad],
#                              marker='o',color='r')
#                     plt.vlines(0,ymin=-CEN*alpha[nzeropad],ymax=CEN*alpha[nzeropad],color='r')
#                     plt.hlines(0,xmin=-CEN*alpha[nzeropad],xmax=CEN*alpha[nzeropad],color='r')
#                     plt.vlines(-0.5,ymin=-0.5,ymax=0.5,color='r')
#                     plt.vlines(0.5,ymin=-0.5,ymax=0.5,color='r')
#                     plt.hlines(-0.5,xmin=-0.5,xmax=0.5,color='r')
#                     plt.hlines(0.5,xmin=-0.5,xmax=0.5,color='r')
#                     plt.xlim((-5,5))
#                     plt.ylim((-5,5))
#                     plt.title('SHWFS image before threshold, FoV={:.0f}wvl/d\n'.format(
#                         alpha[nzeropad]*N)
#                               + 'd/$r_0$={:.1f}, $r_0$={:.1f}cm, d={:.1f}cm'.format(
#                                   dor0[num_dor],r0[num_dor]*100,d*100))
#                     plt.show()

#     var = (np.nanvar(dif_SHWFS_slope_kolmogorov,axis=(0,3), ddof=1)).T
#     # mean = (np.mean(dif_SHWFS_slope,axis=(2,4))[:,0,:]).T
#     dor = d/r0
#     std_var = 2*var/np.sqrt(2*REPEAT2 - 2)
    
#     for i in np.arange(len(r0)):
#         plt.errorbar(FoV, var[:,i], yerr=std_var[:,i],
#                       marker='.', ls='',
#                   label='d/r0 = {:.2f}'.format(dor[i]))
#     # for i in np.arange(len(D_lenslet)):
#     #     plt.errorbar(fov, mean[:,i], yerr=np.sqrt(var[:,i]),
#     #                   marker='.', ls='',
#     #               label='d/r0 = {:.2f}'.format(dor[i]))
#     for i in np.arange(len(r0)):
#         if i == 0:
#             plt.plot((np.arange(32) + 1), 0.5/(np.arange(32) + 1)**2*dor[i]**(5./3.), marker='', ls='-', color='k',
#                       label='fit')
#         else:
#             plt.plot((np.arange(32) + 1), 0.5/(np.arange(32) + 1)**2*dor[i]**(5./3.), marker='', ls='-', color='k')
#     plt.legend()
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlabel('Field-of-View ($\lambda$/d)')
#     plt.ylabel('Variance (rad$^2$)')
#     plt.xlim(1, 100)
#     plt.ylim(1e-4, 10)
#     plt.title('angular-spectrum: oversize={:d}'.format(oversize[noversize]))
#     # plt.savefig('centroidThomas2006.png')
#     plt.show()
#     plt.clf()
    
#     # var = (np.nanvar(dif_SHWFS_slope_kolmogorov,axis=(0), ddof=1)).T
#     # # mean = (np.mean(dif_SHWFS_slope,axis=(2,4))[:,0,:]).T
#     # dor = d/r0
#     # std_var = 2*var/np.sqrt(2*REPEAT2 - 2)
    
#     # for i in np.arange(len(r0)):
#     #     plt.errorbar(FoV, var[0,:,i], yerr=std_var[0,:,i],
#     #                   marker='.', ls='',
#     #               label='d/r0 = {:.2f}'.format(dor[i]))
#     #     plt.errorbar(FoV, var[1,:,i], yerr=std_var[1,:,i],
#     #                   marker='.', ls='')
#     # # for i in np.arange(len(D_lenslet)):
#     # #     plt.errorbar(fov, mean[:,i], yerr=np.sqrt(var[:,i]),
#     # #                   marker='.', ls='',
#     # #               label='d/r0 = {:.2f}'.format(dor[i]))
#     # for i in np.arange(len(r0)):
#     #     if i == 0:
#     #         plt.plot((np.arange(32) + 1), 0.5/(np.arange(32) + 1)**2*dor[i]**(5./3.), marker='', ls='-', color='k',
#     #                   label='fit')
#     #     else:
#     #         plt.plot((np.arange(32) + 1), 0.5/(np.arange(32) + 1)**2*dor[i]**(5./3.), marker='', ls='-', color='k')
#     # plt.legend()
#     # plt.xscale('log')
#     # plt.yscale('log')
#     # plt.xlabel('Field-of-View ($\lambda$/d)')
#     # plt.ylabel('Variance (rad$^2$)')
#     # plt.xlim(1, 100)
#     # plt.ylim(1e-4, 10)
#     # plt.title('angular-spectrum: oversize={:d}'.format(oversize[noversize]))
#     # # plt.savefig('centroidThomas2006.png')
#     # plt.show()
#     # plt.clf()
    
#     plt.plot(g_slope_kolmogorov.flatten(),
#              real_SHWFS_slope_kolmogorov[:, :, -1,:].flatten(),
#              ls='', marker='.', alpha=0.1)
#     plt.plot(np.linspace(g_slope_kolmogorov.min(),g_slope_kolmogorov.max()),
#              np.linspace(g_slope_kolmogorov.min(),g_slope_kolmogorov.max()))
#     plt.xlabel('G-TILT (rad/lenslet)')
#     plt.ylabel('SHWFS-TILT (rad/lenslet)')
#     plt.show()
    
#     plt.plot(g_slope_kolmogorov.flatten(),
#              real_SHWFS_slope_kolmogorov[:, :, -1,:].flatten()
#              - g_slope_kolmogorov.flatten(),
#              ls='', marker='.', alpha=0.1)
#     plt.plot(np.linspace(g_slope_kolmogorov.min(),g_slope_kolmogorov.max()),
#              np.zeros_like(np.linspace(g_slope_kolmogorov.min(),g_slope_kolmogorov.max())))
#     plt.xlabel('G-TILT (rad/lenslet)')
#     plt.ylabel('SHWFS_TILT - G_TILT (rad/lenslet)')
#     plt.show()
    
# MAX = np.max([g_slope_kolmogorov.max(),real_SHWFS_slope_kolmogorov[:,:,-1,:].max()])
# MIN = np.min([g_slope_kolmogorov.min(),real_SHWFS_slope_kolmogorov[:,:,-1,:].min()])

# b0, b1 = deming(g_slope_kolmogorov.flatten(), real_SHWFS_slope_kolmogorov[:,:,-1,:].flatten())
# error = np.std((b0 + b1*g_slope_kolmogorov.flatten()) - real_SHWFS_slope_kolmogorov[:,:,-1,:].flatten(), ddof=1)

# plt.plot(g_slope_kolmogorov.flatten(), real_SHWFS_slope_kolmogorov[:,:,-1,:].flatten(), ls='', marker='.', alpha=0.1)
# plt.plot(np.linspace(MIN, MAX), np.linspace(MIN, MAX),ls='-',marker='',label='1:1')
# plt.plot(np.linspace(MIN, MAX), b0 + b1*np.linspace(MIN, MAX),ls='-',marker='',label='best fit')
# plt.legend()
# plt.xlabel('G-TILT (rad/lenslet)')
# plt.ylabel('SHWFS-TILT (rad/px)')
# plt.title('Comparison of measurement of Kolmogorov slope\nplot intercept={:.3f}, plot slope={:.3f}'.format(b0,b1))
# plt.show()

# plt.plot(real_SHWFS_slope_kolmogorov[:,:,-1,:].flatten(), ((b0 + b1*g_slope_kolmogorov.flatten()) - real_SHWFS_slope_kolmogorov[:,:,-1,:].flatten())/error, ls='', marker='.', alpha=0.1)
# plt.plot(np.linspace(MIN, MAX,100), np.zeros((100)),ls='-',marker='')
# plt.title('Residue between measurements of Kolmogorov slope')
# plt.xlabel('SHWFS-TILT (rad/lenslet)')
# plt.ylabel('G-TILT - SHWFS-TILT ($\sigma$={:.3f} rad/lenslet)'.format(error))
# plt.show()

###############################################################################
# angular-spectrum #######################################################
###############################################################################

# """
# ░█▀▄░█░░░░█▀▀▄░█▀▀▄░█▀▀▀░█▀▀░░░█▀▀░▄▀▀▄░▄░░░▄░░░▄░░░▄░░▀░░█▀▀▄
# ░█░░░█▀▀█░█▄▄█░█░▒█░█░▀▄░█▀▀░░░█▀░░█░░█░░█▄█░░░░░█▄█░░░█▀░█▄▄█
# ░▀▀▀░▀░░▀░▀░░▀░▀░░▀░▀▀▀▀░▀▀▀░░░▀░░░░▀▀░░░░▀░░░░░░░▀░░░▀▀▀░▀░░▀
# ░▒█▀▀▀█░▒█░▒█░▒█░░▒█░▒█▀▀▀░▒█▀▀▀█░░░█▀▀░▄▀▀▄░█▀▄░█▀▀▄░█░░░░█░░█▀▀░█▀▀▄░█▀▀▀░▀█▀░█░░░
# ░░▀▀▀▄▄░▒█▀▀█░▒█▒█▒█░▒█▀▀░░░▀▀▀▄▄░░░█▀░░█░░█░█░░░█▄▄█░█░░░░█░░█▀▀░█░▒█░█░▀▄░░█░░█▀▀█
# ░▒█▄▄▄█░▒█░▒█░▒▀▄▀▄▀░▒█░░░░▒█▄▄▄█░░░▀░░░░▀▀░░▀▀▀░▀░░▀░▀▀░░░▀▀░▀▀▀░▀░░▀░▀▀▀▀░░▀░░▀░░▀
# """

# REPEAT2 =  int(10)

# # oversize = np.array([16], dtype=int)# np.array([1,2,4,8,16], dtype=int) # 1 is default for phase screen generation
# oversize = np.array([1,16], dtype=int) # 1 is default for phase screen generation

# N = int(64)
# wvl = float(500e-9)
# L0 = float(20000)
# l0 = float(1e-12)
# # dor0 = np.array([0.5], dtype=float)#dor0 = np.array([3,2,1,0.5], dtype=float)
# dor0 = np.array([3,2,1,0.5], dtype=float)
# d = float(15e-2)
# r0 = d/dor0
# delta = d/N
# CEN = int(N/2)
# wavenumber = 2*np.pi/wvl
# Threshold = 1e-4

# # FoV in wvl/d unit
# FoV = np.arange(32-4+1, dtype=int) + 4 # zeropad >= 64/(np.arange(32)+1) but i can manage only 16 zeropad soooo fov >= 4 wvl/d is suitable
# zero_pad = np.asarray(2**np.ceil(np.log(N/FoV)/np.log(2)), dtype=int)
# f = d / (FoV*wvl/d)
# alpha = delta/f / (wvl/d) #in wvl/d
# # FoV zeropad, f, alpha has the same dimension.

# kolmogorov_phase = np.zeros((N, N), dtype=float)
# g_slope_kolmogorov = np.zeros((2), dtype=float)
# recentred_SHWFS_slope_kolmogorov = np.zeros((REPEAT2, len(dor0), len(FoV), 2), dtype=float)

# (y1, x1) = np.meshgrid(delta*np.arange(-N/2,N/2),
#                       delta*np.arange(-N/2,N/2)) # for FFT only
# r1sq = (x1**2 + y1**2) + 1e-10                   # for FFT only
                
# (y_cog, x_cog) = np.meshgrid(np.arange(-N/2,N/2) + 0.5, np.arange(-N/2,N/2) + 0.5)
# # ONE = np.ones((N,N))
# NORM = np.dot(x_cog.flatten(),x_cog.flatten())

# for noversize in np.arange(len(oversize)):
#     NN = int(oversize[noversize]*N)
#     edge_low = int(NN*((oversize[noversize] - 1)/oversize[noversize])/2)
#     edge_high = int(NN*((oversize[noversize] + 1)/oversize[noversize])/2)

#     for repeat in np.arange(REPEAT2):
#         kolmogorov_phaser0min = ft_sh_phase_screen(
#             r0.min(), oversize[noversize]*N, delta, L0, l0
#             )[edge_low:edge_high,edge_low:edge_high]
#         for num_dor in np.arange(len(dor0)):
#             kolmogorov_phase = kolmogorov_phaser0min * (r0.min()/r0[num_dor])**(5./6.)
#             # plt.imshow(kolmogorov_phase);plt.colorbar();plt.show()
#             # z_slope_kolmogorov[0] = np.dot(kolmogorov_phase.flatten(),x_cog.flatten())/NORM
#             # z_slope_kolmogorov[1] = np.dot(kolmogorov_phase.flatten(),y_cog.flatten())/NORM
#             g_slope_kolmogorov[0] = np.mean(kolmogorov_phase[1:,:] - kolmogorov_phase[:-1,:])
#             g_slope_kolmogorov[1] = np.mean(kolmogorov_phase[:,1:] - kolmogorov_phase[:,:-1])
#             kolmogorov_phase = (kolmogorov_phase - x_cog*g_slope_kolmogorov[0]
#                                 - y_cog*g_slope_kolmogorov[1] - kolmogorov_phase.sum()/N**2)
#             # plt.imshow(kolmogorov_phase);plt.colorbar();plt.show()
#             # print(np.dot(kolmogorov_phase.flatten(),x_cog.flatten())/NORM,np.dot(kolmogorov_phase.flatten(),y_cog.flatten())/NORM)
#             U_pupil = np.exp(1j*kolmogorov_phase)

#             for nzeropad in np.arange(len(zero_pad)):
#             ###############################################################################
#             # fov and threshold analysis kolmogorov slope with threshold #############
#             ###############################################################################
#                 F0 = pyfftw.empty_aligned((zero_pad[nzeropad]*N, zero_pad[nzeropad]*N), dtype=complex)
#                 F1 = pyfftw.empty_aligned((zero_pad[nzeropad]*N, zero_pad[nzeropad]*N), dtype=complex)
#                 FF01 = pyfftw.FFTW(F0, F1, axes=(-1,-2), direction='FFTW_FORWARD')
#                 F2 = pyfftw.empty_aligned((zero_pad[nzeropad]*N, zero_pad[nzeropad]*N), dtype=complex)
#                 F3 = pyfftw.empty_aligned((zero_pad[nzeropad]*N, zero_pad[nzeropad]*N), dtype=complex)
#                 FF23 = pyfftw.FFTW(F2, F3, axes=(-1,-2), direction='FFTW_BACKWARD')

#                 N_zero_pad = N * zero_pad[nzeropad]
#                 edge_low0 = int(N_zero_pad*((zero_pad[nzeropad] - 1)/zero_pad[nzeropad])/2)
#                 edge_high0 = int(N_zero_pad*((zero_pad[nzeropad] + 1)/zero_pad[nzeropad])/2)
#                 add_each_side0 = int((zero_pad[nzeropad] - 1)*N/2)

#                 df1 = 1. / (N_zero_pad*delta)
#                 fY,fX = np.meshgrid(df1*np.arange(-N_zero_pad/2,N_zero_pad/2),
#                                         df1*np.arange(-N_zero_pad/2,N_zero_pad/2))
#                 fsq = fX**2 + fY**2
#                 Q = np.exp(-1j*wavenumber/2./f[nzeropad] * (((x1 + y1) * delta) + r1sq))
#                 Q2 = np.exp(-1j * np.pi**2 * 2 * f[nzeropad]/wavenumber*fsq) # is the zeropad size!!!
            
#                 convert_real = ((delta/f[nzeropad]) # convert pixel drift to angle drift
#                                 * (delta * N) # convert angle drift to wavefront error
#                                 * (2*np.pi/wvl)) # convert wavefront error to phase error

#                 F0[:,:] = np.fft.fftshift(np.pad(U_pupil * Q,
#                                   (add_each_side0, add_each_side0),
#                                   'constant', constant_values=0.))
#                 F2[:,:] = np.fft.ifftshift(Q2 * np.fft.fftshift(FF01()))
#                 I_focus = (np.abs(
#                     np.fft.ifftshift(FF23())[edge_low0 : edge_high0,
#                                             edge_low0 : edge_high0])**2
#                     / (2*N**2)**2)

#                 I_focus = I_focus/I_focus.max()
                
#                 I_focus = I_focus - Threshold
#                 I_focus[np.where(I_focus < 0)] = 0.0
#                 I_focus = I_focus/I_focus.max()
                
#                 I_focus = I_focus - np.max([I_focus[0,:].max(),
#                                             I_focus[-1,:].max(),
#                                             I_focus[:,0].max(),
#                                             I_focus[:,-1].max()])
#                 I_focus[np.where(I_focus < 0)] = 0.0
#                 I_focus = I_focus/I_focus.max()

#                 # unit is rad per lenslet
#                 recentred_SHWFS_slope_kolmogorov[repeat, num_dor, nzeropad, 0] = np.sum(I_focus * x_cog) / np.sum(I_focus) * convert_real
#                 recentred_SHWFS_slope_kolmogorov[repeat, num_dor, nzeropad, 1] = np.sum(I_focus * y_cog) / np.sum(I_focus) * convert_real

#                 # if (((repeat == 0) and (num_dor == 0)) or (nzeropad == len(zero_pad) - 1)):
#                 #     plt.pcolor(x_cog*alpha[nzeropad],
#                 #                 y_cog*alpha[nzeropad],
#                 #                 I_focus, vmin=1e-4, vmax=1,
#                 #                 snap=True)
#                 #     plt.colorbar()
#                 #     plt.xlabel('x (wvl/d)')
#                 #     plt.ylabel('y (wvl/d)')
#                 #     plt.axis('square')
#                 #     plt.plot(recentred_SHWFS_slope_kolmogorov[repeat,num_dor,nzeropad,0]/convert_real*alpha[nzeropad],
#                 #              recentred_SHWFS_slope_kolmogorov[repeat,num_dor,nzeropad,1]/convert_real*alpha[nzeropad],
#                 #              marker='o',color='r')
#                 #     plt.vlines(0,ymin=-CEN*alpha[nzeropad],ymax=CEN*alpha[nzeropad],color='r')
#                 #     plt.hlines(0,xmin=-CEN*alpha[nzeropad],xmax=CEN*alpha[nzeropad],color='r')
#                 #     plt.vlines(-0.5,ymin=-0.5,ymax=0.5,color='r')
#                 #     plt.vlines(0.5,ymin=-0.5,ymax=0.5,color='r')
#                 #     plt.hlines(-0.5,xmin=-0.5,xmax=0.5,color='r')
#                 #     plt.hlines(0.5,xmin=-0.5,xmax=0.5,color='r')
#                 #     plt.xlim((-5,5))
#                 #     plt.ylim((-5,5))
#                 #     plt.title('SHWFS image before threshold, FoV={:.0f}wvl/d\n'.format(
#                 #         alpha[nzeropad]*N)
#                 #               + 'd/$r_0$={:.1f}, $r_0$={:.1f}cm, d={:.1f}cm'.format(
#                 #                   dor0[num_dor],r0[num_dor]*100,d*100))
#                 #     plt.show()
#             # print(recentred_SHWFS_slope_kolmogorov[repeat, num_dor, nzeropad, 0],recentred_SHWFS_slope_kolmogorov[repeat, num_dor, nzeropad, 1])
#             # plt.plot(recentred_SHWFS_slope_kolmogorov[repeat, num_dor,:,0], ls='', marker='o')
#             # plt.show()

#     var = (np.nanvar(recentred_SHWFS_slope_kolmogorov,axis=(0,3), ddof=1)).T
#     # mean = (np.mean(dif_SHWFS_slope,axis=(2,4))[:,0,:]).T
#     dor = d/r0
#     std_var = 2*var/np.sqrt(2*REPEAT2 - 2)
    
#     for i in np.arange(len(r0)):
#         plt.errorbar(FoV, var[:,i], yerr=std_var[:,i],
#                       marker='.', ls='',
#                   label='d/r0 = {:.2f}'.format(dor[i]))
#     # for i in np.arange(len(D_lenslet)):
#     #     plt.errorbar(fov, mean[:,i], yerr=np.sqrt(var[:,i]),
#     #                   marker='.', ls='',
#     #               label='d/r0 = {:.2f}'.format(dor[i]))
#     for i in np.arange(len(r0)):
#         if i == 0:
#             plt.plot((np.arange(32) + 1), 0.5/(np.arange(32) + 1)**2*dor[i]**(5./3.), marker='', ls='-', color='k',
#                       label='fit')
#         else:
#             plt.plot((np.arange(32) + 1), 0.5/(np.arange(32) + 1)**2*dor[i]**(5./3.), marker='', ls='-', color='k')
#     plt.legend()
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlabel('Field-of-View ($\lambda$/d)')
#     plt.ylabel('Variance (rad$^2$)')
#     plt.xlim(1, 100)
#     plt.ylim(1e-4, 10)
#     plt.title('angular-spectrum: oversize={:d}'.format(oversize[noversize]))
#     # plt.savefig('centroidThomas2006.png')
#     plt.show()
#     plt.clf()
    
# # for i in np.arange(REPEAT2):
# #     plt.plot(recentred_SHWFS_slope_kolmogorov[i,-1,:,0], ls='', marker='o')

# # plt.xlabel('FoV in (wvl/d)')
# # plt.ylabel('SHWFS slope per lenslet (rad/lenslet)')
# # plt.show()
    
#     # var = (np.nanvar(dif_SHWFS_slope_kolmogorov,axis=(0), ddof=1)).T
#     # # mean = (np.mean(dif_SHWFS_slope,axis=(2,4))[:,0,:]).T
#     # dor = d/r0
#     # std_var = 2*var/np.sqrt(2*REPEAT2 - 2)
    
#     # for i in np.arange(len(r0)):
#     #     plt.errorbar(FoV, var[0,:,i], yerr=std_var[0,:,i],
#     #                   marker='.', ls='',
#     #               label='d/r0 = {:.2f}'.format(dor[i]))
#     #     plt.errorbar(FoV, var[1,:,i], yerr=std_var[1,:,i],
#     #                   marker='.', ls='')
#     # # for i in np.arange(len(D_lenslet)):
#     # #     plt.errorbar(fov, mean[:,i], yerr=np.sqrt(var[:,i]),
#     # #                   marker='.', ls='',
#     # #               label='d/r0 = {:.2f}'.format(dor[i]))
#     # for i in np.arange(len(r0)):
#     #     if i == 0:
#     #         plt.plot((np.arange(32) + 1), 0.5/(np.arange(32) + 1)**2*dor[i]**(5./3.), marker='', ls='-', color='k',
#     #                   label='fit')
#     #     else:
#     #         plt.plot((np.arange(32) + 1), 0.5/(np.arange(32) + 1)**2*dor[i]**(5./3.), marker='', ls='-', color='k')
#     # plt.legend()
#     # plt.xscale('log')
#     # plt.yscale('log')
#     # plt.xlabel('Field-of-View ($\lambda$/d)')
#     # plt.ylabel('Variance (rad$^2$)')
#     # plt.xlim(1, 100)
#     # plt.ylim(1e-4, 10)
#     # plt.title('angular-spectrum: oversize={:d}'.format(oversize[noversize]))
#     # # plt.savefig('centroidThomas2006.png')
#     # plt.show()
#     # plt.clf()
    
# #     plt.plot(g_slope_kolmogorov.flatten(),
# #              real_SHWFS_slope_kolmogorov[:, :, -1,:].flatten(),
# #              ls='', marker='.', alpha=0.1)
# #     plt.plot(np.linspace(g_slope_kolmogorov.min(),g_slope_kolmogorov.max()),
# #              np.linspace(g_slope_kolmogorov.min(),g_slope_kolmogorov.max()))
# #     plt.xlabel('G-TILT (rad/lenslet)')
# #     plt.ylabel('SHWFS-TILT (rad/lenslet)')
# #     plt.show()
    
# #     plt.plot(g_slope_kolmogorov.flatten(),
# #              real_SHWFS_slope_kolmogorov[:, :, -1,:].flatten()
# #              - g_slope_kolmogorov.flatten(),
# #              ls='', marker='.', alpha=0.1)
# #     plt.plot(np.linspace(g_slope_kolmogorov.min(),g_slope_kolmogorov.max()),
# #              np.zeros_like(np.linspace(g_slope_kolmogorov.min(),g_slope_kolmogorov.max())))
# #     plt.xlabel('G-TILT (rad/lenslet)')
# #     plt.ylabel('SHWFS_TILT - G_TILT (rad/lenslet)')
# #     plt.show()
    
# # MAX = np.max([g_slope_kolmogorov.max(),real_SHWFS_slope_kolmogorov[:,:,-1,:].max()])
# # MIN = np.min([g_slope_kolmogorov.min(),real_SHWFS_slope_kolmogorov[:,:,-1,:].min()])

# # b0, b1 = deming(g_slope_kolmogorov.flatten(), real_SHWFS_slope_kolmogorov[:,:,-1,:].flatten())
# # error = np.std((b0 + b1*g_slope_kolmogorov.flatten()) - real_SHWFS_slope_kolmogorov[:,:,-1,:].flatten(), ddof=1)

# # plt.plot(g_slope_kolmogorov.flatten(), real_SHWFS_slope_kolmogorov[:,:,-1,:].flatten(), ls='', marker='.', alpha=0.1)
# # plt.plot(np.linspace(MIN, MAX), np.linspace(MIN, MAX),ls='-',marker='',label='1:1')
# # plt.plot(np.linspace(MIN, MAX), b0 + b1*np.linspace(MIN, MAX),ls='-',marker='',label='best fit')
# # plt.legend()
# # plt.xlabel('G-TILT (rad/lenslet)')
# # plt.ylabel('SHWFS-TILT (rad/px)')
# # plt.title('Comparison of measurement of Kolmogorov slope\nplot intercept={:.3f}, plot slope={:.3f}'.format(b0,b1))
# # plt.show()

# # plt.plot(real_SHWFS_slope_kolmogorov[:,:,-1,:].flatten(), ((b0 + b1*g_slope_kolmogorov.flatten()) - real_SHWFS_slope_kolmogorov[:,:,-1,:].flatten())/error, ls='', marker='.', alpha=0.1)
# # plt.plot(np.linspace(MIN, MAX,100), np.zeros((100)),ls='-',marker='')
# # plt.title('Residue between measurements of Kolmogorov slope')
# # plt.xlabel('SHWFS-TILT (rad/lenslet)')
# # plt.ylabel('G-TILT - SHWFS-TILT ($\sigma$={:.3f} rad/lenslet)'.format(error))
# # plt.show()

###############################################################################
# angular-spectrum #######################################################
###############################################################################

"""
░▒█▀▀▀░░▀░░█▀▀▄░█▀▀░░░█▀▀░█▀▀▄░█▀▄▀█░▄▀▀▄░█░░░▀░░█▀▀▄░█▀▀▀
░▒█▀▀░░░█▀░█░▒█░█▀▀░░░▀▀▄░█▄▄█░█░▀░█░█▄▄█░█░░░█▀░█░▒█░█░▀▄
░▒█░░░░▀▀▀░▀░░▀░▀▀▀░░░▀▀▀░▀░░▀░▀░░▒▀░█░░░░▀▀░▀▀▀░▀░░▀░▀▀▀▀
░▄▀▀▄░█▀▀░░░▄▀▀▄░█░░░░█▀▀▄░█▀▀░█▀▀░░░█▀▀░█▀▄░█▀▀▄░█▀▀░█▀▀░█▀▀▄
░█░░█░█▀░░░░█▄▄█░█▀▀█░█▄▄█░▀▀▄░█▀▀░░░▀▀▄░█░░░█▄▄▀░█▀▀░█▀▀░█░▒█
░░▀▀░░▀░░░░░█░░░░▀░░▀░▀░░▀░▀▀▀░▀▀▀░░░▀▀▀░▀▀▀░▀░▀▀░▀▀▀░▀▀▀░▀░░▀
"""

N = int(64)

# oversize = np.array([1], dtype=int)# np.array([1,2,4,8,16], dtype=int) # 1 is default for phase screen generation
# zero_pad = np.array([2,4,8,16], dtype=int) # 2 is default. # for SHWFS

# oversize = np.array([1,2,4,8,16], dtype=int) # 1 is default for phase screen generation
# zero_pad = np.array([2], dtype=int)# np.array([2,4,8,16], dtype=int) # 2 is default. # for SHWFS

fine = np.array([1], dtype=float)
oversize = np.array([256], dtype=int)# np.array([1,2,4,8,16], dtype=int) # 1 is default for phase screen generation
zero_pad = np.array([2], dtype=int)# np.array([2,4,8,16], dtype=int) # 2 is default. # for SHWFS

wvl = float(500e-9)
L0 = float(20000)
l0 = float(1e-12)

Threshold = 1e-4

dor0 = np.array([0.5,1,2,3], dtype=float)
REPEAT2 =  int(50)

d = float(15e-2)
r0 = d/dor0
delta = d/N

# angular_pixel_scale = 1./2. # unit = wvl/d

alpha = 1./2. # unit = wvl/d

CEN = int(N/2)

wavenumber = 2*np.pi/wvl
# pupil, tip, tilt = zernikeArray(3,N)

kolmogorov_phase = np.zeros((N, N), dtype=float)
g_slope_kolmogorov = np.zeros((2), dtype=float)
real_SHWFS_slope_kolmogorov = np.zeros((REPEAT2, len(dor0), CEN, 2), dtype=float)

(y_cog, x_cog) = np.meshgrid(np.arange(-N/2,N/2) + 0.5, np.arange(-N/2,N/2) + 0.5)

for nzeropad in np.arange(len(zero_pad)):
###############################################################################
# fov and threshold analysis kolmogorov slope with threshold #############
###############################################################################

    F0 = pyfftw.empty_aligned((zero_pad[nzeropad]*N, zero_pad[nzeropad]*N), dtype=complex)
    F1 = pyfftw.empty_aligned((zero_pad[nzeropad]*N, zero_pad[nzeropad]*N), dtype=complex)
    FF01 = pyfftw.FFTW(F0, F1, axes=(-1,-2), direction='FFTW_FORWARD')
    F2 = pyfftw.empty_aligned((zero_pad[nzeropad]*N, zero_pad[nzeropad]*N), dtype=complex)
    F3 = pyfftw.empty_aligned((zero_pad[nzeropad]*N, zero_pad[nzeropad]*N), dtype=complex)
    FF23 = pyfftw.FFTW(F2, F3, axes=(-1,-2), direction='FFTW_BACKWARD')

    f_lenslet = delta**2*N/wvl/alpha
    N_zero_pad = N * zero_pad[nzeropad]
    edge_low0 = int(N_zero_pad*((zero_pad[nzeropad] - 1)/zero_pad[nzeropad])/2)
    edge_high0 = int(N_zero_pad*((zero_pad[nzeropad] + 1)/zero_pad[nzeropad])/2)
    add_each_side0 = int((zero_pad[nzeropad] - 1)*N/2)

    (y1, x1) = np.meshgrid(delta*np.arange(-N/2,N/2),
                          delta*np.arange(-N/2,N/2))
    r1sq = (x1**2 + y1**2) + 1e-10

    df1 = 1. / (N_zero_pad*delta)
    fY,fX = np.meshgrid(df1*np.arange(-N_zero_pad/2,N_zero_pad/2),
                            df1*np.arange(-N_zero_pad/2,N_zero_pad/2))
    fsq = fX**2 + fY**2
    Q = np.exp(-1j*wavenumber/2./f_lenslet * (((x1 + y1) * delta) + r1sq))
    Q2 = np.exp(-1j * np.pi**2 * 2 * f_lenslet/wavenumber*fsq) # is the zeropad size!!!

    convert_real = ((delta/f_lenslet) # convert pixel drift to angle drift
                    * (delta * N) # convert angle drift to wavefront error
                    * (2*np.pi/wvl)) # convert wavefront error to phase error

    for noversize in np.arange(len(oversize)):
        NN = int(oversize[noversize]*N)
        edge_low = int(NN*((oversize[noversize] - 1)/oversize[noversize])/2)
        edge_high = int(NN*((oversize[noversize] + 1)/oversize[noversize])/2)
        
        for repeat in np.arange(REPEAT2):
            kolmogorov_phaser0min = ft_sh_phase_screen(
                r0.min(), oversize[noversize]*N, delta, L0, l0
                )[edge_low:edge_high,edge_low:edge_high]
            for num_dor in np.arange(len(dor0)):
                kolmogorov_phase = kolmogorov_phaser0min * (r0.min()/r0[num_dor])**(5./6.)
                g_slope_kolmogorov[0] = np.mean(kolmogorov_phase[1:,:] - kolmogorov_phase[:-1,:])
                g_slope_kolmogorov[1] = np.mean(kolmogorov_phase[:,1:] - kolmogorov_phase[:,:-1])
                kolmogorov_phase = (kolmogorov_phase - x_cog*g_slope_kolmogorov[0]
                                    - y_cog*g_slope_kolmogorov[1] - kolmogorov_phase.sum()/N**2)

                U_pupil = np.exp(1j*kolmogorov_phase)
                # unit is rad per lenslet

                F0[:,:] = np.fft.fftshift(np.pad(U_pupil * Q,
                                  (add_each_side0, add_each_side0),
                                  'constant', constant_values=0.))
                F2[:,:] = np.fft.ifftshift(Q2 * np.fft.fftshift(FF01()))
                I_focus = (np.abs(
                    np.fft.ifftshift(FF23())[edge_low0 : edge_high0,
                                            edge_low0 : edge_high0])**2
                    / (2*N**2)**2)

                I_focus = I_focus/I_focus.max()

                # unit of half fov is pixel! = fov in wvl/d
                for half_fov in (np.arange(CEN) + 1):
                    yy, xx = np.meshgrid(np.arange(-half_fov, half_fov) + 0.5,
                                          np.arange(-half_fov, half_fov) + 0.5)

                    I_fov = I_focus[CEN - half_fov : CEN + half_fov,
                                    CEN - half_fov : CEN + half_fov]

                    I_fov = I_fov/I_fov.max()

                    I_fov = I_fov - Threshold
                    I_fov[np.where(I_fov < 0)] = 0.0
                    I_fov = I_fov/np.max(I_fov)

                    I_fov = I_fov - np.max([I_fov[0,:].max(),
                                                I_fov[-1,:].max(),
                                                I_fov[:,0].max(),
                                                I_fov[:,-1].max()])
                    I_fov[np.where(I_fov < 0)] = 0.0
                    I_fov = I_fov/np.max(I_fov)

                    # unit is rad per lenslet
                    real_SHWFS_slope_kolmogorov[repeat, num_dor, half_fov - 1, 0] = np.sum(I_fov * xx) / np.sum(I_fov) * convert_real
                    real_SHWFS_slope_kolmogorov[repeat, num_dor, half_fov - 1, 1] = np.sum(I_fov * yy) / np.sum(I_fov) * convert_real

        CEN = int(N/2)
        fov = 2 * (np.arange(CEN) + 1) * alpha # fov in wvl/d = half fov in pixel
        var = (np.nanvar(real_SHWFS_slope_kolmogorov,axis=(0,3), ddof=1)).T

        dor = d/r0
        std_var = 2*var/np.sqrt(2*REPEAT2 - 2)
        
        for i in np.arange(len(r0)):
            plt.errorbar(fov, var[:,i], yerr=std_var[:,i],
                          marker='.', ls='',
                      label='d/r0 = {:.2f}'.format(dor[i]))
        for i in np.arange(len(r0)):
            if i == 0:
                plt.plot((np.arange(32) + 1), 0.5/(np.arange(32) + 1)**2*dor[i]**(5./3.), marker='', ls='-', color='k',
                          label='fit')
            else:
                plt.plot((np.arange(32) + 1), 0.5/(np.arange(32) + 1)**2*dor[i]**(5./3.), marker='', ls='-', color='k')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Field-of-View ($\lambda$/d)')
        plt.ylabel('Variance (rad$^2$)')
        plt.xlim(1, 100)
        plt.ylim(1e-4, 10)
        plt.title('angular-spectrum: zeropad={:d}, oversize={:d}'.format(zero_pad[nzeropad],oversize[noversize]))
        plt.show()
        plt.clf()
        
        
