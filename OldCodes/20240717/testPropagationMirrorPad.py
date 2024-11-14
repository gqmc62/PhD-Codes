# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:44:41 2023

@author: gqmc62
"""

import my_propagation as mprop
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib import colors
import my_phasescreen as mps

# from my_WFS import rebin

N = 1024
wvl = 500e-9
r0HA = 0.05#0.05#0.14
d = 0.05
lenslet_pixel = 128
delta = d/lenslet_pixel
num = 10
k = 2.*np.pi/wvl
cn2dz=r0HA**(-5./3.)/0.423/k**2
ARCSEC2RAD = 2.*np.pi/3600./360.
FoV = wvl/2./delta/ARCSEC2RAD # 2.*wvl/2./delta/ARCSEC2RAD # 

saturate_point = (0.6/(0.563 * k**(7./6.) * cn2dz))**(6./5.) # where rytov approx = 0.6
hard_scintillation_point = (0.35/(0.563 * k**(7./6.) * cn2dz))**(6./5.)

# if too low prop distance, either cannot detect blob size,
# or light could not yet interact with neighboring pixels...
#              detectable fresnel zone   allowed angle
zmin = 4*delta**2/wvl
# expected minimum zone size to be more than 2 pixel...

#       saturate               no more data                freznel zone larger than the screen
zmax = np.min([saturate_point,(N*delta)/(wvl/delta/N),(2.*delta*N)**2/wvl,30000])


# delta = r0HA/16

real_z_list = np.logspace(np.log10(zmin), np.log10(zmax), num=num, base=10)
z_list = np.append(0,real_z_list)


# mprop.testPropLimitedAlpha1Layer(wvl,real_z_list,r0HA,N,delta,FoV,new=True)

try:
    screen_list
    name = 'propagation_sample_1HA_real_d{:.0f}cm_r0{:.0f}cm'.format(d*100,r0HA*100)
    hdulist = fits.open(name + '.fits')
    hdu = hdulist[0]
    real = hdu.data
    hdulist.close()
    name = 'propagation_sample_1HA_imag_d{:.0f}cm_r0{:.0f}cm'.format(d*100,r0HA*100)
    hdulist = fits.open(name + '.fits')
    hdu = hdulist[0]
    imag = hdu.data
    hdulist.close()
    screens_list = real + 1j*imag
    mprop.testPropScintillation(screens_list,wvl,real_z_list,r0HA,N,delta)
    mprop.testPropBlobSize(screens_list,wvl,real_z_list,r0HA,N,delta)
except:
    z_list_prop = np.append(0, real_z_list)
    z_list_prop = np.flip(real_z_list.max() - z_list_prop)
    z_list_prop[0] = 0
    screens_list = mprop.screensForTestProp1Layer(wvl,z_list_prop,r0HA,N,delta)
    mprop.testPropScintillation(screens_list,wvl,real_z_list,r0HA,N,delta)
    mprop.testPropBlobSize(screens_list,wvl,real_z_list,r0HA,N,delta)
    name = 'propagation_sample_1HA_real_d{:.0f}cm_r0{:.0f}cm'.format(d*100,r0HA*100)
    hdu = fits.PrimaryHDU(screens_list.real)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + '.fits', overwrite=True)
    hdul.close()
    name = 'propagation_sample_1HA_imag_d{:.0f}cm_r0{:.0f}cm'.format(d*100,r0HA*100)
    hdu = fits.PrimaryHDU(screens_list.imag)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + '.fits', overwrite=True)
    hdul.close()
    
SHOW = N#lenslet_pixel#
RADPERPIX2NMPERSUBAP = lenslet_pixel*(wvl/1e-9)/(2.*np.pi)
RAD2NM = (wvl/1e-9)/(2.*np.pi)
MAXRYTOVINLENSLET = np.min([(0.563 * k**(7./6.) * cn2dz * ((0.8*SHOW*delta)**2/wvl)**(5./6.)),0.6])


# for i in np.arange(screens_list.shape[0]):
#     plt.imshow(np.angle(screens_list[i]))
#     plt.title('phase'
#               + '\nd={:.0f}cm, r0={:.0f}cm'.format(d*100,r0HA*100)
#               + '\nz={:.3f}km'.format(z_list[i]/1000))
#     plt.colorbar()
#     plt.show()
    
# #     binned = np.angle(rebin(screens_list[i], (N//lenslet_pixel,N//lenslet_pixel)))
# #     plt.imshow(binned)
# #     plt.title('mean phase in a lenslet'
# #               + '\nd={:.0f}cm, r0={:.0f}cm'.format(d*100,r0HA*100)
# #               +'\nz={:.3f}km'.format(z_list[i]/1000))
# #     plt.colorbar()
# #     plt.show()
    
# for i in np.arange(screens_list.shape[0]):
#     intensityx = (np.abs(screens_list[i][:N,:N][1:])**2
#                   + np.abs(screens_list[i][:N,:N][:-1])**2)/2.
#     slopex = np.angle(screens_list[i][:N,:N][1:]
#               / screens_list[i][:N,:N][:-1])#*intensityx/intensityx.mean()
#     # binned = rebin(slopex[ : -lenslet_pixel + 1,
#     #                                 : -lenslet_pixel],
#     #                         ((N - lenslet_pixel)//lenslet_pixel,
#     #                           (N - lenslet_pixel)//lenslet_pixel))
#     plt.imshow(slopex*RADPERPIX2NMPERSUBAP)#,
#                 #vmin=binned.min()*RADPERPIX2NMPERSUBAP,
#                 #vmax=binned.max()*RADPERPIX2NMPERSUBAP)
#     plt.title('slope (nm/sub)'
#               + '\nd={:.0f}cm, r0={:.0f}cm'.format(d*100,r0HA*100)
#               +'\nz={:.3f}km'.format(z_list[i]/1000))
#     plt.colorbar()
#     plt.show()
# #     plt.imshow(binned*RADPERPIX2NMPERSUBAP)
# #     plt.title('mean slope in a lenslet (nm/subap)'
# #               + '\nd={:.0f}cm, r0={:.0f}cm'.format(d*100,r0HA*100)
# #               +'\nz={:.3f}km'.format(z_list[i]/1000))
# #     plt.colorbar()
# #     plt.show()

# # for i in np.arange(screens_list.shape[0]):
# #     intensityx = (np.abs(screens_list[i][:SHOW,:SHOW][1:])**2
# #                   + np.abs(screens_list[i][:SHOW,:SHOW][:-1])**2)/2.
# #     slopex = np.angle(screens_list[i][:SHOW,:SHOW][1:]
# #               / screens_list[i][:SHOW,:SHOW][:-1])#
    
# #     binnedslopex = rebin(slopex[ : -lenslet_pixel + 1,
# #                                     : -lenslet_pixel],
# #                             ((N - lenslet_pixel)//lenslet_pixel,
# #                               (N - lenslet_pixel)//lenslet_pixel))
    
# #     plt.imshow(binnedslopex*RADPERPIX2NMPERSUBAP,
# #                 vmin=binnedslopex.min()*RADPERPIX2NMPERSUBAP,
# #                 vmax=binnedslopex.max()*RADPERPIX2NMPERSUBAP)
# #     plt.title('mean G slope (nm/subap)'
# #               + '\nd={:.0f}cm, r0={:.0f}cm'.format(d*100,r0HA*100)
# #               +'\nz={:.3f}km'.format(z_list[i]/1000))
# #     plt.colorbar()
# #     plt.show()
    
# #     binnedIslopex = rebin((slopex*intensityx)[ : -lenslet_pixel + 1,
# #                                     : -lenslet_pixel],
# #                             ((N - lenslet_pixel)//lenslet_pixel,
# #                              (N - lenslet_pixel)//lenslet_pixel))
# #     binnedIntensityx = rebin(intensityx[ : -lenslet_pixel + 1,
# #                                     : -lenslet_pixel],
# #                             ((N - lenslet_pixel)//lenslet_pixel,
# #                              (N - lenslet_pixel)//lenslet_pixel))
# #     plt.imshow(binnedIslopex/binnedIntensityx*RADPERPIX2NMPERSUBAP,
# #                vmin=binnedslopex.min()*RADPERPIX2NMPERSUBAP,
# #                vmax=binnedslopex.max()*RADPERPIX2NMPERSUBAP)
# #     plt.title('mean IG slope in a lenslet (nm/subap)'
# #               + '\nd={:.0f}cm, r0={:.0f}cm'.format(d*100,r0HA*100)
# #               +'\nz={:.3f}km'.format(z_list[i]/1000))
# #     plt.colorbar()
# #     plt.show()

# RMS_list = np.zeros(screens_list.shape[0] - 1)
# bestTipTiltRemove = (0.134*(d/r0HA)**(5./3.))**0.5
  
# for i in np.arange(screens_list.shape[0] - 1):
#     temp = (screens_list[i + 1][:SHOW,:SHOW]
#                         / screens_list[0][:SHOW,:SHOW])
#     intensity = np.abs(screens_list[i + 1][:SHOW,:SHOW])**2
#     # print(np.std(np.angle(temp))*wvl/2/np.pi/1e-9)
#     mean = temp.mean()
#     mean /= np.abs(mean)
#     # print(np.angle(mean))
#     dif = temp/mean
#     rms = np.std(np.angle(dif))
#     RMS_list[i] = rms
#     # plt.hist(np.angle(dif).flatten(),bins=50)
#     # plt.show()
#     # plt.plot(np.angle(dif.flatten()),np.abs(temp.flatten())**2,ls='',marker=',',alpha=0.5)
#     # plt.show()
#     # wrms = np.std(np.angle(dif).flatten()*intensity.flatten())/intensity.mean()
#     plt.imshow(np.angle(dif)*RAD2NM)
#     plt.colorbar()
#     plt.title('intensity difference (nm) to z=0 at z={:.0f}m'.format(real_z_list[i])
#                 #+ '\nwRMS difference in subaperture={:.0f}nm'.format(wrms*RAD2NM)
#               + '\nRMS difference in subaperture={:.0f}nm'.format(rms*RAD2NM))
#     plt.show()

# # SCALE = RMS_list[real_z_list.shape[0]//2]/np.sqrt(real_z_list[real_z_list.shape[0]//2]*wvl)

# # plt.plot(real_z_list,RMS_list,ls='',marker='o',label='minimum residue')
# # plt.plot(real_z_list,SCALE*np.sqrt(wvl*real_z_list),marker='',ls='--',label='{:.2f}xFresnel Zone'.format(SCALE))
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.xlabel('propagation distance')
# # plt.ylabel('phase rms (rad)')
# # plt.hlines((0.134*(d/r0HA)**(5./3.))**0.5,real_z_list.min(),real_z_list.max(),
# #            label='maximum DM removal from Noll')
# # plt.legend()
# # plt.title('best phase residue (rad) between prop/no prop'
# #           + '\nd={:.0f}cm, r0={:.0f}cm'.format(d*100,r0HA*100))
# # plt.show()

# # for i in np.arange(screens_list.shape[0] - 1):
# #     norm = (np.abs(screens_list[0][:SHOW,:SHOW])**2).mean()
# #     temp = (np.abs(screens_list[i + 1][:SHOW,:SHOW])**2/norm
# #                - np.abs(screens_list[0][:SHOW,:SHOW])**2/norm)
# #     plt.imshow(temp)
# #     plt.colorbar()
# #     plt.title('intensity difference to z=0 at z={:.0f}m'.format(real_z_list[i])
# #               + '\nRMS difference in subaperture={:.3f} norm'.format(temp.std())
# #               + '\nRMS in subaperture limited at = {:.3f} norm'.format(np.sqrt(MAXRYTOVINLENSLET*4))
# #               + '\nat z={:.0f}m'.format((0.8*SHOW*delta)**2/wvl))
# #     plt.show()
    
# # for i in np.arange(screens_list.shape[0] - 1):
# #     slope1 = (screens_list[i + 1][:SHOW,:SHOW][1:]
# #               / screens_list[i + 1][:SHOW,:SHOW][:-1])
# #     slope2 = (screens_list[0][:SHOW,:SHOW][1:]
# #               / screens_list[0][:SHOW,:SHOW][:-1])
    
# #     temp = (slope1/slope2)
# #     mean = temp.mean()#np.product(temp)**(1/temp.size)
# #     dif = temp/mean
    
# #     # plt.hist(np.angle(temp).flatten())
# #     # plt.show()
# #     # plt.hist(np.angle(dif).flatten())
# #     # plt.show()
    
# #     # square_dif = np.angle(dif)**2
# #     rms = np.std(np.angle(dif))#np.mean(square_dif)**0.5
# #     # intensity = (np.abs(screens_list[i + 1][:SHOW,:SHOW][1:])**2
# #     #               + np.abs(screens_list[i + 1][:SHOW,:SHOW][:-1])**2)/2
# #     # wrms = np.mean(square_dif * intensity/intensity.mean())**0.5
# #     plt.imshow((np.angle(1j*temp)%np.pi - np.pi/2)*RADPERPIX2NMPERSUBAP,
# #                 vmax=0.1*RADPERPIX2NMPERSUBAP,vmin=-0.1*RADPERPIX2NMPERSUBAP)
# #                 # norm=colors.CenteredNorm())
# #     plt.colorbar()
# #     plt.title('phase x-slope difference (nm/subap) to z=0 at z={:.0f}m'.format(real_z_list[i])
# #               + '\nRMS difference in subaperture={:.0f}nm/subap'.format(rms*RADPERPIX2NMPERSUBAP)
# #               # + '\nwRMS difference in subaperture={:.0f}nm/subap'.format(wrms*RADPERPIX2NMPERSUBAP)
# #               + '\nmean={:.0f}nm/subap'.format(np.angle(mean)*RADPERPIX2NMPERSUBAP))
# #               # + '\nRMS saturate at {:.3}nm/subap'.format(np.pi/np.sqrt(12)*RADPERPIX2NMPERSUBAP))
# #     plt.show()

# # for i in np.arange(screens_list.shape[0]):
# #     intensity = np.abs(screens_list[i][:SHOW,:SHOW])**2
# #     phase = np.angle(screens_list[i][:SHOW,:SHOW])
# #     plt.imshow(intensity*phase/intensity.mean(),vmin=-np.pi,vmax=np.pi)
# #     plt.colorbar()
# #     plt.title('intensity weighted phase')
# #     plt.show()

# # for i in np.arange(screens_list.shape[0]):
# #     slopex = np.angle(screens_list[i][:SHOW,:SHOW][1:]
# #               / screens_list[i][:SHOW,:SHOW][:-1])
# #     intensityx = (np.abs(screens_list[i][:SHOW,:SHOW][1:])**2
# #                  + np.abs(screens_list[i][:SHOW,:SHOW][:-1])**2)/2.
# #     plt.imshow(intensityx*slopex/intensityx.mean())
# #     plt.colorbar()
# #     plt.title('intensity weighted phase slope x')
# #     plt.show()

# # ARCSEC2RAD = 2.*np.pi/3600./360.
# # h = 5000
# # # Hmin = 30000
# # # H = h*(2**np.ceil(np.log(Hmin/h)/np.log(2)) - 1)
# # H = 35000
# # lenslet_pixel = 16
# # r0 = 0.11
# # d = r0
# # D = d/h*H
# # delta = d/lenslet_pixel
# # wvl = 500e-9
# # turbulenceHeights = np.arange(int(np.ceil(H/h)) + 1)*h
# # FoV = wvl/2./d/ARCSEC2RAD*lenslet_pixel
# # # separation = d/h/ARCSEC2RAD

# # # turbulenceStrengths = 1/2**(np.arange(int(np.ceil(H/h)) + 1)*h/5000)
# # # turbulenceStrengths[-1] = (2 - turbulenceStrengths[:-1].sum()*h/5000)
# # # temp = np.flip(np.sort(turbulenceStrengths))
# # # turbulenceStrengths[-1] = temp[1]
# # # turbulenceStrengths[1] = temp[-1]
# # # np.random.shuffle(turbulenceStrengths[1:-1])
# # # turbulenceStrengths /= turbulenceStrengths.sum()
# # # distributedr0 = (turbulenceStrengths*r0**(-5./3.))**(-3./5.)


for i in np.arange(screens_list.shape[0] - 1):
    a = mps.get_structure_function(np.log(np.abs(screens_list[i + 1])))
    # a, b = mps.get_structure_function_rolling(np.log(np.abs(screens_list[i])))
    plt.plot(np.arange(len(a))*delta,a)#,yerr=b)
    plt.vlines([np.sqrt(wvl*real_z_list[i])],a.min(),a.max())
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
