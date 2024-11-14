# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:52:36 2023

test noise simulation

@author: gqmc62
"""

import numpy as np
from matplotlib import pyplot as plt
from aotools.turbulence import infinitephasescreen
import my_WFSC as MWFS
from tqdm import tqdm
from MyScreen import MyPhaseScreen

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

REPEAT1 = 1000
wvl = 500e-9
N_original = int(64)
L0_original = float(20)
d_original = L0_original/64
delta_original = d_original/N_original
r0 = 2e-2
alpha = 0.5
Threshold = 0

Ig = MWFS.IgTilt(N_original)
SHWFS = MWFS.OneStepSHWFS(N_original)

IgTilt = np.zeros((REPEAT1, 2), dtype=float)
SHWFSTilt = np.zeros((REPEAT1, 2), dtype=float)
cIgTilt = np.zeros((REPEAT1, 2), dtype=float)
cSHWFSTilt = np.zeros((REPEAT1, 2), dtype=float)
varIg = np.zeros((REPEAT1,2),dtype=float)
varSHWFS = np.zeros((REPEAT1,2),dtype=float)



for dor0 in np.array([3,2,1,0.5]):
    # if dor0 == 3:
    #     Threshold = 1e-3 # 1e-3 is good for SNR>10
    # elif dor0 == 2:
    #     Threshold = 0.5e-2 # 0.5e-3 is almost good for SNR>10
    # elif dor0 == 1:
    #     Threshold = 2.5e-2 # 2e-2 for SNR>10
    # elif dor0 == 0.5:
    #     Threshold = 1e-1 # 1 2 3 4 5 not work at all...
    # else:
    #     Threshold = 0
        
    # if dor0 == 3: #       10000         10
    #     Threshold = 6e-2 # 5e-3         6e-2
    # elif dor0 == 2:
    #     Threshold = 4e-2 # 5e-4         4e-2
    # elif dor0 == 1:
    #     Threshold = 1.75e-2 # 5.75e-5   1.75e-2
    # elif dor0 == 0.5:
    #     Threshold = 1e-2 # 3.25e-5      3.25e-5
    # else:
    #     Threshold = 0

    # for SNR in np.array([10,100,1000,10000]):
    
    r0_original = d_original/dor0
    phasescreen = infinitephasescreen.PhaseScreenVonKarman(N_original, delta_original, r0_original, L0_original)
    # Nphot = SNR**2
    d = dor0*r0
    delta = d/N_original
    scaling_factor = (r0_original/delta_original*delta/r0)**(5./6.)
    # if dor0<1:
    #     FWHM = 2*dor0
    # else:
    #     FWHM = 2
    # photon_scale = (N_original/FWHM)**2
    # if dor0<1:
    #     photon_scale = 1
    # else:
    #     photon_scale = 1
    # photon_scale = 1
    for i in np.arange(2*N_original):
        phasescreen.add_row()
    main_phasescreen = phasescreen.scrn - phasescreen.scrn.mean()
    for SNR in np.array([10,100,1000,10000]):
        Nphot = SNR**2
        for repeat1 in tqdm(np.arange(REPEAT1)):
            for i in np.arange(2*N_original):
                phasescreen.add_row()
            main_phasescreen = phasescreen.scrn - phasescreen.scrn.mean()
            proper_screen = main_phasescreen * scaling_factor
            U = np.exp(1j*proper_screen)
            IgTilt[repeat1, : ] = Ig.measure_with_noise(U,Nphot)
            SHWFSTilt[repeat1, : ] = SHWFS.measure_with_noise_no_threshold(U, Nphot, 0)

        varIg = np.var(IgTilt)
        varSHWFS = np.var(SHWFSTilt)
        
        plt.plot(SHWFSTilt.flatten(),IgTilt.flatten(),ls='',marker='.',alpha=0.5)
        limit = np.max(np.abs([SHWFSTilt,IgTilt]))
        plt.plot(np.linspace(-limit,limit),np.linspace(-limit,limit),label='1:1')
        plt.xlabel('SHWFS OPD (rad)')
        plt.ylabel('IG OPD (rad)')
        plt.axis('square')
        plt.legend()
        # plt.title('Nphot={:3.0e}'.format(Nphot))
        plt.show()
        plt.savefig('plot2.png')
        plt.clf()
        
        temp1 = plt.hist(IgTilt.flatten(),alpha=0.5,label='Ig')
        temp2 = plt.hist(SHWFSTilt.flatten(),alpha=0.5,label='shwfs')
        # temp1 = temp2
        thisplotmax = np.max([temp1[1],temp2[1]])
        thisplotmin = np.min([temp1[1],temp2[1]])
        # plt.vlines((0
        #             + 2*N_original/2/Nphot
        #             )
        #             ,ymax=np.max([temp1[0],temp2[0]]),ymin=0,
        #             label='photon noise',
        #             color='m')
        plt.legend()
        plt.xlabel('OPD (rad2)')
        plt.ylabel('occurance')
        # plt.title('Nphot={:3.0e}, d/r0={:.1f}'.format(Nphot, dor0))
        plt.title('d/r0={:.1f}'.format(dor0))
        # plt.xlim(thisplotmin,thisplotmax)
        plt.show()
        plt.savefig('plot4.png')
        plt.clf()
        
        x = np.mean(SHWFSTilt, axis=1).flatten()
        y = np.mean(IgTilt, axis=1).flatten() - np.mean(SHWFSTilt, axis=1).flatten()
        xabsmax = np.abs(x).max()
        yabsmax = np.abs(y).max()
        xyabsmax = np.max([xabsmax,yabsmax])
        a, b = deming(x, y)
        plt.plot(x, y, ls='', marker='.', label='measurements',color='r',alpha=0.5)
        plt.plot(np.linspace(-xyabsmax,xyabsmax), a + b*np.linspace(-xyabsmax,xyabsmax),
                  ls='-',marker='',label='best fit',color='b')
        plt.hlines(0,xmin=-xabsmax,xmax=xabsmax,ls='--',label='zeros',color='k')
        plt.vlines(np.pi*np.arange(-xabsmax//np.pi, xabsmax//np.pi + 1),
                    ymin=-yabsmax,ymax=yabsmax,label='pixel edges',ls='-.',color='m')
        plt.vlines(np.pi*np.arange(-xabsmax//np.pi//2 - 1, xabsmax//np.pi//2 + 2) + np.pi/2,
                    ymin=-yabsmax,ymax=yabsmax,label='pixel centres',ls=':',color='c')
        plt.xlim(-xabsmax,xabsmax)
        plt.ylim(-yabsmax,yabsmax)
        plt.xlabel('SHWFS OPD (rad)')
        plt.ylabel('Ig - SHWFS OPD (rad)')
        plt.legend()
        # plt.title('Nphot={:3.0e}, d/r0={:.1f}'.format(Nphot, dor0))
        plt.title('d/r0={:.1f}'.format(dor0))
        plt.show()
        plt.savefig('plot5.png')
        plt.clf()
        