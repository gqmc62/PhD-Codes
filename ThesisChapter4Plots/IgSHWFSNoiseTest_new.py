# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:52:36 2023

test noise simulation

@author: gqmc62
"""

import numpy as np
from matplotlib import pyplot as plt
# from aotools.turbulence import infinitephasescreen
import my_WFSC as MWFS
from tqdm import tqdm
import pickle
import os
import time

import sys

sample_ii = int(sys.argv[1])
sample_jj = int(sys.argv[2])
sample_kk = int(sys.argv[3])
sample_ll = int(sys.argv[4])

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

Name = '/mt/cfai/pkongkaew/IGPhaseScreen/'#'202302211459/'
os.chdir(Name)
with open('Ultimate_input.pkl', 'rb') as file:
    myvar = pickle.load(file)
    TARGET = myvar['TARGET']
    SCREEN_LIMIT = myvar['SCREEN_LIMIT']
    n = myvar['n']
    oversize  = myvar['oversize']
    L0_GL = myvar['L0_GL']
    L0_HA = myvar['L0_HA']
    l0_GL = myvar['l0_GL']
    l0_HA = myvar['l0_HA']
    wvl  = myvar['wvl']
    wavenumber = myvar['wavenumber']
    zenith_angle  = myvar['zenith_angle']
    r0 = myvar['r0']
    dor0   = myvar['dor0']
    Rytov  = myvar['Rytov']
    H_HA = myvar['H_HA']
    II = myvar['II']
    JJ   = myvar['JJ']
    KK = myvar['KK']
    LL  = myvar['LL']
    wavenumber  = myvar['wavenumber']
    N_MAX = myvar['N_MAX']
    d = myvar['d']
    delta = myvar['delta']
    r0_GL = myvar['r0_GL']
    r0_HA  = myvar['r0_HA']
      
    N = myvar['N']
    num_lenslet = myvar['num_lenslet']
    REPEAT = myvar['REPEAT']
    POSSIBLE_SCREEN = myvar['POSSIBLE_SCREEN']
    POSSIBLE_PROP = myvar['POSSIBLE_PROP']
    POSSIBLE = myvar['POSSIBLE']

os.chdir(Name+'/U_tot')

RON_list = np.array([0,0.000001,0.00001,0.0001,0.001],dtype=float)
NN=len(RON_list)

# Ig = MWFS.IgTilt(n)
SHWFS = MWFS.OneStepSHWFS(n)

N_phot_list = np.array([100,10000,1000000],dtype=float)
MM=len(N_phot_list)

Threshold_list = np.array([0,0.1,0.3,0.5],dtype=float)
OO=len(Threshold_list)

#RANDOM_NOISE_TRIAL = 1024

TARGET = 256
TARGET2 = 256

# IgTilt = np.zeros((II, JJ, KK, LL, MM, TARGET, 2), dtype=float)
SHWFSTilt = np.zeros((MM, NN, OO, TARGET, 2, TARGET2), dtype=float)
NoNoiseSHWFSTilt = np.zeros((TARGET, 2), dtype=float)

# varIg = np.zeros((II, JJ, KK, LL, MM), dtype=float)
# varSHWFS = np.zeros((II, JJ, KK, LL, MM), dtype=float)



U = np.zeros((n, n), dtype=complex)

# if not os.path.isdir('noise'):
#     os.makedirs('noise')

#TIME_START = time.time()



Pii, Pjj, Pkk, Pll = np.where(POSSIBLE==True)
for possible in np.arange(len(Pii)):
    ii = Pii[possible]
    jj = Pjj[possible]
    kk = Pkk[possible]
    ll = Pll[possible]
    #if dor0[jj] == 0.5:
    #    continue

    if sample_ii != ii:
        continue
    if sample_jj != jj:
        continue
    if sample_kk != kk:
        continue

    if sample_ll != ll:
        continue

    run_name = 'r{:02d}dor{:02d}rytov{:02d}h{:02d}'.format(
                        int(r0[ii]*100), int(dor0[jj]*10), int(Rytov[kk]*100),
                        int(H_HA[ll]/1000))
    #print(ii,jj,kk,ll)
    #if jj != 0 :
    #    continue
    for count in np.arange(TARGET):
        try:
            with open('U_tot_' + run_name + '_{:04d}.pkl'.format(count),'rb') as file:
                U = pickle.load(file)

            NoNoiseSHWFSTilt[count, :] = SHWFS.measure_no_noise_no_threshold(U)
            #print(NoNoiseSHWFSTilt[ll,count, :])

            for mm in np.arange(MM):
                Nphot = N_phot_list[mm]
                for nn in np.arange(NN):
                    RON = RON_list[nn]*Nphot
                    for oo in np.arange(OO):
                        Threshold = Threshold_list[oo]
                        #print(ii,jj,kk,ll,mm,nn,oo)
                        
                        
                        #IgTilt[ii, jj, kk, ll, mm, count, : ] = Ig.measure_with_noise(U, Nphot)
                        #for trandom_noise_trial in np.arange(RANDOM_NOISE_TRIAL):

                        for count2 in np.arange(TARGET2):

                            SHWFSTilt[mm,nn, oo,count, : ,count2] = SHWFS.measure_with_noise_with_threshold(U, Nphot, RON, Threshold)
                            #print(SHWFSTilt[ll, mm,nn, oo,count, : ,count2])
        except:
            SHWFSTilt[:,:,:,count,:,:] = np.nan
            NoNoiseSHWFSTilt[count, :] = np.nan
        # plt.plot(SHWFSTilt[ii, jj, kk, ll, mm].flatten(),
        #          IgTilt[ii, jj, kk, ll, mm].flatten(),
        #          ls='',marker=',',alpha=0.1)
        # limit = np.max(np.abs([SHWFSTilt,IgTilt]))
        # plt.plot(np.linspace(-limit,limit),np.linspace(-limit,limit),label='1:1')
        # plt.xlabel('SHWFS OPD (rad)')
        # plt.ylabel('IG OPD (rad)')
        # plt.axis('square')
        # plt.legend()
        # plt.title('Nphot={:3.0e}'.format(Nphot))
        # plt.savefig('noise/'+ run_name + '_SNR{:.0f}_plot2.png'.format(SNR))
        # plt.clf()
        
        # temp1 = plt.hist(IgTilt[ii, jj, kk, ll, mm].flatten(),alpha=0.5,label='Ig')
        # temp2 = plt.hist(SHWFSTilt[ii, jj, kk, ll, mm].flatten(),alpha=0.5,label='shwfs')
        # plt.legend()
        # plt.xlabel('OPD (rad)')
        # plt.ylabel('occurance')
        # plt.title('Nphot={:3.0e}'.format(Nphot))
        # plt.savefig('noise/'+ run_name + '_SNR{:.0f}_plot4.png'.format(SNR))
        # plt.clf()
        
        # x = SHWFSTilt[ii, jj, kk, ll, mm].flatten()
        # y = IgTilt[ii, jj, kk, ll, mm].flatten() - SHWFSTilt[ii, jj, kk, ll, mm].flatten()
        # xabsmax = np.abs(x).max()
        # yabsmax = np.abs(y).max()
        # xyabsmax = np.max([xabsmax,yabsmax])
        # a, b = deming(x, y)
        # plt.plot(x, y, ls='', marker='.', label='measurements',color='r')
        # plt.plot(np.linspace(-xyabsmax,xyabsmax), a + b*np.linspace(-xyabsmax,xyabsmax),
        #         ls='-',marker='',label='best fit',color='b')
        # plt.hlines(0,xmin=-xabsmax,xmax=xabsmax,ls='--',label='zeros',color='k')
        # plt.vlines(np.pi*np.arange(-xabsmax//np.pi, xabsmax//np.pi + 1),
        #             ymin=-yabsmax,ymax=yabsmax,label='pixel edges',ls='-.',color='m')
        # plt.vlines(np.pi*np.arange(-xabsmax//np.pi//2 - 1, xabsmax//np.pi//2 + 2) + np.pi/2,
        #             ymin=-yabsmax,ymax=yabsmax,label='pixel centres',ls=':',color='c')
        # plt.xlim(-xabsmax,xabsmax)
        # plt.ylim(-yabsmax,yabsmax)
        # plt.xlabel('SHWFS OPD (rad)')
        # plt.ylabel('Ig - SHWFS OPD (rad)')
        # plt.legend()
        # plt.title('Nphot={:3.0e}'.format(Nphot))
        # plt.savefig('noise/'+ run_name + '_SNR{:.0f}_plot5.png'.format(SNR))
        # plt.clf()
    
    myvar = {'N_phot_list' : N_phot_list,
            'RON_list' : RON_list,
            'Threshold_list' : Threshold_list,
            #'IgTilt' : IgTilt,
            'SHWFSTilt' : SHWFSTilt,
            'NoNoiseSHWFSTilt' : NoNoiseSHWFSTilt}

    os.chdir('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/Slopes')

    with open('SHWFSNoiseScintillation{:d}{:d}{:d}{:d}.pkl'.format(sample_ii,sample_jj,sample_kk,sample_ll), 'wb') as file:
        pickle.dump(myvar, file)
        file.close()

#TIME_END = time.time()

#np.savetxt('noiselog.txt', [TIME_END-TIME_START])
