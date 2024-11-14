# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 10:30:12 2023

@author: gqmc62
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from astropy.io import fits
import os
from tqdm import tqdm
from scipy.optimize import lsq_linear

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

def lsq_cov2cn2(refcov,refcn2,cov,cov_noise,n_bootstrap=100):
    RESULT = np.zeros((n_bootstrap,refcov.shape[1]),dtype=float)
    for n in np.arange(n_bootstrap):
        noisy_cov = np.random.normal(np.nan_to_num(cov),np.nan_to_num(cov_noise))
        res = lsq_linear(refcov, noisy_cov, bounds=(0,np.inf))
        RESULT[n] = res.x
    cn2 = np.mean(RESULT,axis=0) * refcn2
    cn2_noise = np.std(RESULT,axis=0,ddof=1) *refcn2
    r0 = cn22r0(cn2.sum())
    r0_noise = stdcn22stdr0(cn2.sum(),cn2_noise.sum())
    return cn2, cn2_noise, r0, r0_noise

def loadTIMBRefCov_wrongcentre(HOMEDIR,REFFILE,wrong):
    os.chdir(HOMEDIR)
    hdulist = fits.open(REFFILE + '.fits')
    hdu = hdulist[0]
    REF = np.flip(hdu.data, axis=1)
    hdulist.close()
    REF[REF==0] = np.nan
    REFX = REF[::2]
    REFY = REF[1::2]
    LAYER = REFX.shape[0]
    FLAT = REFX.shape[1] * REFX.shape[2]
    AY = np.nan_to_num((REFY.reshape(LAYER, FLAT)).T)
    AX = np.nan_to_num((REFX.reshape(LAYER, FLAT)).T)
    for layer in np.arange(LAYER):
        (AX[:,layer])[AX[:,layer] == np.nanmax(AX[:,layer])] *= wrong
        (AY[:,layer])[AY[:,layer] == np.nanmax(AY[:,layer])] *= wrong
    A = np.zeros((2*FLAT, LAYER), dtype=float)
    A[: FLAT] = AX
    A[FLAT : 2*FLAT] = AY
    return A

def loadTIMBRefCov_withcentre(HOMEDIR,REFFILE):
    os.chdir(HOMEDIR)
    hdulist = fits.open(REFFILE + '.fits')
    hdu = hdulist[0]
    REF = np.flip(hdu.data, axis=1)
    hdulist.close()
    REF[REF==0] = np.nan
    REFX = REF[::2]
    REFY = REF[1::2]
    LAYER = REFX.shape[0]
    FLAT = REFX.shape[1] * REFX.shape[2]
    AY = np.nan_to_num((REFY.reshape(LAYER, FLAT)).T)
    AX = np.nan_to_num((REFX.reshape(LAYER, FLAT)).T)
    # for layer in np.arange(LAYER):
    #     (AX[:,layer])[AX[:,layer] == np.nanmax(AX[:,layer])] = 0
    #     (AY[:,layer])[AY[:,layer] == np.nanmax(AY[:,layer])] = 0
    A = np.zeros((2*FLAT, LAYER), dtype=float)
    A[: FLAT] = AX
    A[FLAT : 2*FLAT] = AY
    return A



HOMEDIR = 'C:/Users/gqmc62/OneDrive - Durham University/Desktop/soapy-master/soapy-master/conf'
REFFILE = 'referenceFunctionsVK_6x6'

REFR0 = 0.05
REFTOTALCN2 = r02cn2(REFR0)
MAXR0 = 0.11
MINR0 = 0.02

# MAXNOISE = 0.1 # fraction of cov in each px

RANDOM = 100

CENTRAL_NOISE_LIST = 1. + np.arange(3)
MAXNOISE_LIST = [0.1]

A_withcentre = loadTIMBRefCov_withcentre(HOMEDIR, REFFILE)


LAYER = A_withcentre.shape[1]
FLAT = A_withcentre.shape[0]//2
SCREEN = int(np.sqrt(FLAT))
MID = LAYER - 1

IS_A_CENTRE = np.zeros((2*FLAT,LAYER),dtype=float)
((IS_A_CENTRE[:FLAT,:]).reshape((SCREEN,SCREEN,LAYER)))[:,MID] = 1
((IS_A_CENTRE[FLAT:2*FLAT,:]).reshape((SCREEN,SCREEN,LAYER)))[:,MID] = 1

IS_COV_CENTRE = np.zeros((2*FLAT),dtype=float)
((IS_COV_CENTRE[:FLAT]).reshape((SCREEN,SCREEN)))[:,MID] = 1
((IS_COV_CENTRE[FLAT:2*FLAT]).reshape((SCREEN,SCREEN)))[:,MID] = 1

A_nocentre = (A_withcentre[np.where(IS_A_CENTRE == False)]).reshape((2*SCREEN*(SCREEN - 1),LAYER))

PINVA_nocentre = np.linalg.pinv(A_nocentre)
PINVA_withcentre = np.linalg.pinv(A_withcentre)

II = len(MAXNOISE_LIST)
JJ = len(CENTRAL_NOISE_LIST)

DISTRIBUTION_ERROR_noCentre = np.zeros((II,JJ,RANDOM,LAYER), dtype=float)
DISTRIBUTION_ERROR_withCentre = np.zeros((II,JJ,RANDOM,LAYER), dtype=float)
r0_ERROR_noCentre = np.zeros((II,JJ,RANDOM), dtype=float)
r0_ERROR_withCentre = np.zeros((II,JJ,RANDOM), dtype=float)

for ii in np.arange(II):
    MAXNOISE = MAXNOISE_LIST[ii]
    for jj in np.arange(JJ):
        CENTRAL_NOISE = CENTRAL_NOISE_LIST[jj]
        for random in tqdm(np.arange(RANDOM)):
            A_wrongCentre = loadTIMBRefCov_wrongcentre(HOMEDIR, REFFILE, CENTRAL_NOISE)
            
            np.random.seed(np.random.randint(0,2**31))
            
            MINNOISE = MAXNOISE
            
            COEFFDISTRIBUTION = np.random.rand((LAYER))
            COEFFDISTRIBUTION /= COEFFDISTRIBUTION.sum()
            
            RANDOMR0 = np.random.rand()*(MAXR0 - MINR0) + MINR0
            RANDOMTOTALCN2 = r02cn2(RANDOMR0)
            RANDOMCN2 = RANDOMTOTALCN2 * COEFFDISTRIBUTION
            
            RANDOMCOEFF = RANDOMCN2 / REFTOTALCN2
            RANDOMCOV = A_wrongCentre @ RANDOMCOEFF
            RANDOMCOV[RANDOMCOV == 0] = np.nan
            
            NOISE = np.random.rand((2*FLAT)) * (MAXNOISE - MINNOISE) + MINNOISE
            NOISE *= np.abs(RANDOMCOV)
            
            RANDOMCOV_nocentre = (RANDOMCOV[np.where(IS_COV_CENTRE == False)]).reshape((2*SCREEN*(SCREEN - 1)))
            
            NOISYRANDOMCOV = np.random.normal(RANDOMCOV, NOISE)
            NOISYRANDOMCOV_nocentre = (NOISYRANDOMCOV[np.where(IS_COV_CENTRE == False)]).reshape((2*SCREEN*(SCREEN - 1)))
            
            NOISE_nocentre = (NOISE[np.where(IS_COV_CENTRE == False)]).reshape((2*SCREEN*(SCREEN - 1)))
            
            (SOLVEDCN2_withcentre,
             SOLVEDCN2NOISE_withcentre,
             SOLVEDR0_withcentre,
             SOLVEDR0NOISE_withcentre) = lsq_cov2cn2(A_withcentre,
                                                 REFTOTALCN2,
                                                 NOISYRANDOMCOV,
                                                 NOISE)
            
            (SOLVEDCN2_nocentre,
             SOLVEDCN2NOISE_nocentre,
             SOLVEDR0_nocentre,
             SOLVEDR0NOISE_nocentre) = lsq_cov2cn2(A_nocentre,
                                               REFTOTALCN2,
                                               NOISYRANDOMCOV_nocentre,
                                               NOISE_nocentre)
            
            DISTRIBUTION_ERROR_noCentre[ii,jj,random] = (SOLVEDCN2_nocentre/RANDOMTOTALCN2 - RANDOMCN2/RANDOMTOTALCN2)
            DISTRIBUTION_ERROR_withCentre[ii,jj,random] = (SOLVEDCN2_withcentre/RANDOMTOTALCN2 - RANDOMCN2/RANDOMTOTALCN2)
            r0_ERROR_noCentre[ii,jj,random] = np.abs(SOLVEDR0_nocentre - RANDOMR0)
            r0_ERROR_withCentre[ii,jj,random] = np.abs(SOLVEDR0_withcentre - RANDOMR0)
            
            if random == (RANDOM - 1):
                fig, axs = plt.subplot_mosaic([['top', 'top'],['left', 'right'],],
                              constrained_layout=True,
                              gridspec_kw={'width_ratios': [1,9],'height_ratios': [1,9]},
                              figsize=(7,7))
                plt.suptitle('Statistics checks on Cn2 decomposition from generated cov map\n'
                              + 'Total Cn2={:.2e}m$^-$$^2$$^/$$^3$; Noise={:.0f}%; Centre error=x{:.0f}%\n'.format(
                                  RANDOMTOTALCN2, MAXNOISE*100, (100*CENTRAL_NOISE))
                              + 'Distribution ratio RMS error: NO centre fit = {:.2f}, WITH centre fit = {:.2f}\n'.format(
                                  (DISTRIBUTION_ERROR_noCentre[ii,jj]).std(), (DISTRIBUTION_ERROR_withCentre[ii,jj]).std())
                              + 'r0 RMS error: NO centre fit = {:.2f}cm, WITH centre fit = {:.2f}cm'.format(
                                  (r0_ERROR_noCentre[ii,jj].std())*100,(r0_ERROR_withCentre[ii,jj].std())*100))
                
                axs['right'].set_title('Measured Cn2 distribution')
                axs['right'].errorbar(np.arange(LAYER),
                                SOLVEDCN2_nocentre/RANDOMTOTALCN2,
                                yerr=SOLVEDCN2NOISE_nocentre/RANDOMTOTALCN2,
                                ls='', marker='^', capsize=5,label='measured NO central peak',color='r')
                axs['right'].errorbar(np.arange(LAYER),
                                SOLVEDCN2_withcentre/RANDOMTOTALCN2,
                                yerr=SOLVEDCN2NOISE_withcentre/RANDOMTOTALCN2,
                                ls='', marker='v', capsize=5,label='measured WITH central peak',color='b')
                axs['right'].bar(np.arange(LAYER), RANDOMCN2/RANDOMTOTALCN2,
                            ls='-', label='simulated',alpha=0.5,color='k',width=0.5)
                axs['right'].set_xlabel('Turbulence layer # from ground')
                # axs[1].legend(loc='lower center')
                axs['right'].set_yscale('log')
                axs['right'].set_ylabel('fraction of total simulated Cn2')
                axs['right'].set_ylim(0.01,1)
                
                axs['left'].errorbar([''],SOLVEDR0_nocentre,yerr=SOLVEDR0NOISE_nocentre,
                                marker='^',capsize=5,color='r')#,label='measured')
                axs['left'].errorbar([''],SOLVEDR0_withcentre,yerr=SOLVEDR0NOISE_withcentre,
                                marker='v',capsize=5,color='b')#,label='measured')
                axs['left'].bar([''],RANDOMR0,color='k',alpha=0.5)#,label='simulated')
                axs['left'].set_ylabel('r0 (m)')
                axs['left'].set_ylim((0.02,0.11))
                axs['left'].set_title('Total r0 (m)')
                
                h, l = axs['right'].get_legend_handles_labels()
                axs['top'].legend(h, l, loc='center',ncol=3)
                axs['top'].axis('off')
                
                fig.tight_layout(pad=0.5)
                plt.show()
            
