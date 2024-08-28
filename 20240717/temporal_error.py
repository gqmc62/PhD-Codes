# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:22:41 2023

@author: gqmc62
"""

import os
import soapy
import numpy as np
from matplotlib import pyplot as plt

ARCSEC2RAD = 1./3600./360.*2.*np.pi
RAD2ARCSEC = 1./ARCSEC2RAD

home = ('C:/Users/gqmc62/OneDrive - Durham University/Desktop/'
             + 'soapy-master/soapy-master/conf')
file = 'my.yaml'
os.chdir(home)


sim = soapy.Sim("temporal_error.yaml")

loopTime = sim.config.sim.loopTime

r0 = sim.config.atmos.r0
L0 = sim.config.atmos.L0

telDiam = sim.config.tel.telDiam

nxSubaps = sim.config.wfss[0].nxSubaps
pxlsPerSubap = sim.config.wfss[0].pxlsPerSubap
subapFOV = sim.config.wfss[0].subapFOV
wfs_wavelength = sim.config.wfss[0].wavelength

TT_iMatValue = sim.config.dms[0].iMatValue
DM_iMatValue = sim.config.dms[1].iMatValue
DM_nxActuators = sim.config.dms[1].nxActuators

svdConditioning = sim.config.recon.svdConditioning

sci_FOV = sim.config.scis[0].FOV
sci_wavelength = sim.config.scis[0].wavelength
sci_pxls = sim.config.scis[0].pxls

if nxSubaps + 1 != DM_nxActuators:
    print('Not Fried Geometry. Change either nxSubpars or nxActuators.')
    
subapDiam = telDiam/nxSubaps
wfs_diffraction = wfs_wavelength/2/subapDiam*RAD2ARCSEC
wfs_angular_pxls = subapFOV/pxlsPerSubap
if wfs_angular_pxls > wfs_diffraction:
    print('WFS cannot resolve spot.')
    
sci_diffraction = sci_wavelength/2/telDiam*RAD2ARCSEC
sci_angular_pxls = sci_FOV/sci_pxls
if sci_angular_pxls > sci_diffraction:
    print('Science camera cannot resolve spot.')
    
if (L0 < 8*telDiam).any():
    print('L0 might be too small for telescope to be Kolmogorov-like.')
    
if subapFOV < 5.*wfs_diffraction:
    print('Linearity of WFS is not good.')
    
# nb = input('Proceed?\nType y to proceed: ')
# if nb != 'y':
#     quit()

sim.aoinit()
sim.config.calcParams()
sim.makeIMat()
    
# r0_val = np.array([0.02,0.05,0.08,0.10,0.12,0.15], dtype=float)
# strehl = np.array([26.1,65.9,88.5,90.7,92.2,95.7], dtype=float)/100.
REPEAT = 5
nwind = 10
r0 = 0.5
# sim.config.sim.nIters = 100
sim.config.atmos.r0 = r0
sim.config.sim.loopTime = 0.0025

# L0 = 8 m
# telDiam = 0.4 m
# subapDiam = 0.05 m
# px = 0.003

# loopTime = 0.0025 s

# 3200 = 1 L0/frame
# 160 m/s = 1 telescope aperture/frame
# 20 m/s = 1 subaperture/frame
# 1.25 m/s = 1px/frame

windSpeeds_val = np.logspace(np.log10(1), np.log10(1000) , num=nwind)
strehl_repeat = np.zeros((windSpeeds_val.shape[0], REPEAT))

for n in np.arange(len(sim.config.wfss)):
    sim.config.wfss[n].GSMag = -20
for n in np.arange(len(sim.config.dms)):
    sim.config.dms[n].gain = 1



for n_wind in np.arange(len(windSpeeds_val)):
    sim.config.atmos.windSpeeds = np.ones_like(
        sim.config.atmos.windSpeeds)*windSpeeds_val[n_wind]
    if n_wind < 1:
        sim.config.sim.nIters = 10000
    elif n_wind < 3:
        sim.config.sim.nIters = 1000
    elif n_wind < 6:
        sim.config.sim.nIters = 100
    else:
        sim.config.sim.nIters = 10
    for repeat in np.arange(REPEAT):
        for n in np.arange(len(sim.config.dms)):
            sim.config.dms[n].gain = 1
        sim.config.randomSeed = np.random.randint(0, 999)
        sim.atmos = soapy.atmosphere.atmos(sim.config)
        sim.aoloop()
        # plt.imshow(sim.atmos.scrns[0])
        # plt.show()
        strehl_repeat[n_wind, repeat] = sim.sciCams[0].longExpStrehl # this is the gui reported value
        # strehl_repeat[n_wind, repeat] = sim.sciCams[0].long_exposure_Strehl
        # strehl_repeat[n_wind, repeat] = sim.sciCams[0].instStrehl
        # print(sim.sciCams[0].instStrehl)
        sim.reset_loop()


strehl = np.mean(strehl_repeat, axis=1)
strehl_rms = np.std(strehl_repeat, axis=1, ddof=1)

rms_nm = np.sqrt(-np.log(strehl))/2./np.pi*sci_wavelength
rms_nm_rms = np.abs(-strehl_rms/(2*2.*np.pi/sci_wavelength*strehl*np.sqrt(-np.log(strehl))))


# total_Cn2dz = r0**(-5./3.)/0.423/(2.*np.pi/sci_wavelength)**2
# f_greenwood = (2.31 * sci_wavelength**(-6./5.)
#                 * (total_Cn2dz*windSpeeds_val**(5./3.))**(3./5.))

# temperal_error_nm = np.sqrt(
#     (f_greenwood/(1./sim.config.sim.loopTime))**(5./3.)
#     )/2/np.pi*sci_wavelength

# temperal_constant = (rms_nm/temperal_error_nm)**2
# temperal_constant_std = 2*rms_nm/temperal_error_nm**2*rms_nm_rms

# tau0 = (0.314 * r0 / windSpeeds_val)
tau0 = r0/windSpeeds_val


dmerror = np.sqrt(0.354*(subapDiam/r0)**(5./3.))/2/np.pi*sci_wavelength/1e-9
no_correction = np.sqrt(1.0299*(telDiam/r0)**(5./3.))/2./np.pi*sci_wavelength/1e-9

x = windSpeeds_val
# y = 1*np.sqrt(
#     (f_greenwood/(1./sim.config.sim.loopTime))**(5./3.)
#     )/2/np.pi*sci_wavelength/1e-9
############################################################################
y = 1*np.sqrt(
    (sim.config.sim.loopTime/tau0)**(5./3.)
    )/2/np.pi*sci_wavelength/1e-9

ymin=(rms_nm/1e-9 - dmerror).min()
ymax=(rms_nm/1e-9 - dmerror).max()

plt.hlines(no_correction, 1, 1000, label='Noll no correction', color='m')
plt.errorbar(x, rms_nm/1e-9 - dmerror, yerr=rms_nm_rms/1e-9, marker='o', label='measured',
              capsize=5, ls='')
# plt.errorbar(x, rms_nm/1e-9, yerr=rms_nm_rms/1e-9, marker='o', label='measured',
#              capsize=5, ls='')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('wind speed (m/s)')
plt.ylabel('wavefront error (nm)')
plt.plot(x, y, color='r', label='expected behaviour')
plt.vlines((telDiam/(nxSubaps*pxlsPerSubap))/loopTime, ymin=ymin, ymax=ymax, label='px/frame',color='c')
plt.vlines((telDiam/(nxSubaps))/loopTime, ymin=ymin, ymax=ymax, label='subap/frame',color='b',alpha=0.5,ls='-.')
plt.vlines(r0/loopTime, ymin=ymin, ymax=ymax, label='r0/frame',color='k',alpha=0.5,ls='--')
plt.vlines(L0/loopTime, ymin=ymin, ymax=ymax, label='L0/frame',color='g')
plt.vlines(1e-10/loopTime, ymin=ymin, ymax=ymax, label='l0/frame',color='tab:orange')
plt.xlim(1*0.5,1000*2)
plt.ylim(ymin*0.5,ymax*2)

# plt.xlim((0.04,4))
# plt.ylim((10,100))
plt.legend()
plt.show()


# plt.plot(1/f_greenwood,tau0)
