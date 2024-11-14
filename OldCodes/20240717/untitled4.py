# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 10:30:12 2023

@author: gqmc62
"""

import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
import os

def r02cn2(r0,wvl=500e-9,zenith_angle=0.):
    cn2 = r0**(-5./3.)/(0.423*(2.*np.pi/wvl)**2/np.cos(zenith_angle))
    return cn2

def cn22r0(cn2,wvl=500e-9,zenith_angle=0.):
    r0 = (0.423*(2.*np.pi/wvl)**2/np.cos(zenith_angle)*cn2)**(-3./5.)
    return r0

def stdcn22stdr0(cn2,stdcn2,wvl=500e-9,zenith_angle=0.):
    stdr0 = np.abs((0.423*(2.*np.pi/wvl)**2/np.cos(zenith_angle))**(-3./5.)
                   * (-3./5.) * cn2**(-8./5.)) * stdcn2
    return stdr0

def cov2cn2(pinv_refcov,refcn2,cov,cov_noise):
    cn2 = (pinv_refcov @ np.nan_to_num(cov)) * refcn2
    r0 = cn22r0(cn2.sum())
    cn2_noise = np.sqrt((pinv_refcov**2) @ (np.nan_to_num(cov_noise**2))) * refcn2
    r0_noise = stdcn22stdr0(cn2.sum(),cn2_noise.sum())
    return cn2, cn2_noise, r0, r0_noise

HOMEDIR = 'C:/Users/gqmc62/OneDrive - Durham University/Desktop/soapy-master/soapy-master/conf'
os.chdir(HOMEDIR)

REFFILE = 'referenceFunctionsVK_6x6'

name = REFFILE
hdulist = fits.open(name + '.fits')
hdu = hdulist[0]
REF = np.flip(hdu.data, axis=1)
hdulist.close()
REF[REF==0] = np.nan

REFX = REF[::2]
REFY = REF[1::2]

LAYER = REFX.shape[0]
FLAT = REFX.shape[1] * REFX.shape[2]
SCREEN = REFX.shape[1]

AY = np.nan_to_num((REFY.reshape(LAYER, FLAT)).T)
AX = np.nan_to_num((REFX.reshape(LAYER, FLAT)).T)

A = np.zeros((2*FLAT, LAYER), dtype=float)
A[: FLAT] = AX
A[FLAT : 2*FLAT] = AY

PINVA = np.linalg.pinv(A)

COEFFDISTRIBUTION = np.random.rand((LAYER))
COEFFDISTRIBUTION /= COEFFDISTRIBUTION.sum()

REFR0 = 0.05
REFCN2 = r02cn2(REFR0)
MAXR0 = 0.11
MINR0 = 0.02

RANDOMR0 = np.random.rand()*(MAXR0 - MINR0) + MINR0
RANDOMTOTALCN2 = r02cn2(RANDOMR0)
RANDOMCN2 = RANDOMTOTALCN2 * COEFFDISTRIBUTION

RANDOMCOEFF = RANDOMCN2 / REFCN2
RANDOMCOV = A @ RANDOMCOEFF
RANDOMCOV[RANDOMCOV == 0] = np.nan

MAXNOISE = 0.1
MINNOISE = 0.01

NOISE = np.random.rand((2*FLAT)) * (MAXNOISE - MINNOISE) + MINNOISE
NOISE *= np.abs(RANDOMCOV)

NOISYRANDOMCOV = np.random.normal(RANDOMCOV, NOISE)

SOLVEDCN2, SOLVEDCN2NOISE, SOLVEDR0, SOLVEDR0NOISE = cov2cn2(PINVA,REFCN2,NOISYRANDOMCOV,NOISE)

# SOLVEDCOEFF = PINVA @ np.nan_to_num(NOISYRANDOMCOV)
# SOLVEDCOEFFNOISE = np.sqrt((PINVA**2) @ (np.nan_to_num(NOISE**2)))
# SOLVEDCN2 = SOLVEDCOEFF * REFCN2
# SOLVEDCN2NOISE = SOLVEDCOEFFNOISE * REFCN2

# SOLVEDTOTALCN2 = np.sum(SOLVEDCN2)
# SOLVEDTOTALCN2NOISE = np.sum(SOLVEDCN2NOISE)

# SOLVEDR0 = cn22r0(SOLVEDTOTALCN2)
# SOLVEDR0NOISE = stdcn22stdr0(SOLVEDTOTALCN2, SOLVEDTOTALCN2NOISE)


fig, axs = plt.subplots(1,2,sharey=False, figsize=(10,5))

plt.suptitle('Statistics checks on Cn2 decomposition from generated cov map')
axs[0].set_title('Measured Cn2')
axs[0].errorbar(np.arange(LAYER), SOLVEDCN2, yerr=SOLVEDCN2NOISE, ls='', marker='o', capsize=5,label='measured')
axs[0].plot(np.arange(LAYER), RANDOMCN2, ls='-', marker='',label='simulated')
axs[0].set_yscale('log')
axs[0].set_ylabel('Cn2 (m**...)')
axs[0].set_xlabel('Turbulence layer # from ground')
axs[0].legend()
axs[1].errorbar([0],SOLVEDR0,yerr=SOLVEDR0NOISE,marker='o',capsize=5,label='measured')
axs[1].hlines(RANDOMR0,xmin=-1,xmax=1,label='simulated')
axs[1].set_ylim((0.02,0.11))
axs[1].set_title('Total r0 (m)')
axs[1].legend()

plt.show()
