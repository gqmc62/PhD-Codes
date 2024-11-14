# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:10:59 2023

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


sim = soapy.Sim("fitting_error.yaml")


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
    
nb = input('Proceed?\nType y to proceed: ')
if nb != 'y':
    quit()
    
sim.aoinit()
sim.config.calcParams()
sim.makeIMat()
    
# r0_val = np.array([0.02,0.05,0.08,0.10,0.12,0.15], dtype=float)
# strehl = np.array([26.1,65.9,88.5,90.7,92.2,95.7], dtype=float)/100.
REPEAT = 10
# dor0_min = 1./3.
# dor0_max = 3.
nr0 = 10
# r0_val = np.linspace(subapDiam/dor0_max, subapDiam/dor0_min, num=nr0)
r0_val = np.logspace(np.log10(0.05), np.log10(0.20), num=nr0)
strehl_repeat = np.zeros((r0_val.shape[0], REPEAT))
sim.config.sim.nIters = 10
sim.config.atmos.windSpeeds = np.zeros_like(sim.config.atmos.windSpeeds)
for n in np.arange(len(sim.config.wfss)):
    sim.config.wfss[n].GSMag = -20
for n in np.arange(len(sim.config.dms)):
    sim.config.dms[n].gain = 1

wfs_wavelength/subapDiam
wfs_wavelength/r0_val

for n_r0 in np.arange(len(r0_val)):
    sim.config.atmos.r0 = r0_val[n_r0]
    for repeat in np.arange(REPEAT):
        # for n in np.arange(len(sim.config.dms)):
        #     sim.config.dms[n].gain = 1
        sim.config.randomSeed = np.random.randint(0, 999)
        sim.atmos = soapy.atmosphere.atmos(sim.config)
        sim.aoloop()
        # plt.imshow(sim.atmos.scrns[0])
        # plt.show()
        strehl_repeat[n_r0, repeat] = sim.sciCams[0].instStrehl
        # print(sim.sciCams[0].instStrehl)
        sim.reset_loop()


strehl = np.mean(strehl_repeat, axis=1)
strehl_rms = np.std(strehl_repeat, axis=1, ddof=1)

rms_nm = np.sqrt(-np.log(strehl))/2./np.pi*sci_wavelength
rms_nm_rms = np.abs(-strehl_rms/(2*2.*np.pi/sci_wavelength*strehl*np.sqrt(-np.log(strehl))))

fitting_error_nm = np.sqrt((subapDiam/r0_val)**(5./3.))/2/np.pi*sci_wavelength

dm_fitting = 1/(fitting_error_nm**2/rms_nm**2)
dm_fitting_rms = np.abs(2*rms_nm/fitting_error_nm**2*rms_nm_rms)

dm_fitting_mean = np.mean(dm_fitting)
dm_fitting_std = np.sqrt(np.mean(dm_fitting_rms**2))

# plt.fill_between(r0_val,
#                  np.ones_like(r0_val)*(dm_fitting_mean + dm_fitting_std),
#                  np.ones_like(r0_val)*(dm_fitting_mean - dm_fitting_std),
#                  alpha=0.5, color='r', label='measured range')
# plt.plot(r0_val, np.ones_like(r0_val)*dm_fitting_mean, color='r')
# plt.plot(r0_val, dm_fitting, marker='o', label='measure value')
# plt.xlabel('r0 (m)')
# plt.ylabel('dm fitting constant (K)')
# plt.title('dm fitting constant (K)')
# plt.legend()
# plt.show()

x = r0_val
y = np.sqrt(
    dm_fitting_mean*(subapDiam/r0_val)**(5./3.)
    )/2/np.pi*sci_wavelength/1e-9
y_high = np.sqrt(
    (dm_fitting_mean + dm_fitting_std)*(subapDiam/r0_val)**(5./3.)
    )/2/np.pi*sci_wavelength/1e-9
y_low = np.sqrt(
    (dm_fitting_mean - dm_fitting_std)*(subapDiam/r0_val)**(5./3.)
    )/2/np.pi*sci_wavelength/1e-9

plt.errorbar(x, rms_nm/1e-9, yerr=rms_nm_rms/1e-9, marker='o', label='measured',
             capsize=5, ls='')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('r0 (m)')
plt.ylabel('wavefront error (nm)')
plt.plot(x, y, color='r', label='mean K')
plt.fill_between(x, y_high, y_low,
                  alpha=0.5, color='r', label='+-1 RMS K')
# plt.fill_between(r0_val,
#                  np.sqrt(
#                      (dm_fitting_mean + 2*dm_fitting_std)*(subapDiam/r0_val)**(5./3.)
#                      )/2/np.pi*sci_wavelength,
#                  np.sqrt(
#                      (dm_fitting_mean - 2*dm_fitting_std)*(subapDiam/r0_val)**(5./3.)
#                      )/2/np.pi*sci_wavelength,
#                  alpha=0.25, color='r', label='+-2 RMS')
# plt.vlines(subapDiam,
#            ymax=y_high.max(),
#            ymin=y_low.min(), label='d/r0=1', color='k')
plt.xlim((0.05,0.20))
plt.ylim((10,100))
plt.title('DM fitting constant (K) = {:.3f} +- {:.3f}'.format(dm_fitting_mean, dm_fitting_std))
plt.legend()
plt.show()

print('dm fitting constant = {:.3f} +- {:.3f}'.format(dm_fitting_mean,dm_fitting_std))

