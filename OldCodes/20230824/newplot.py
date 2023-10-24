# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:35:25 2023

@author: gqmc62
"""

###############################################################################
##########################################################################
"""
█▀▀█░█░▒█░▀█▀░▀█▀░░▀░░█░░░█░█▀▀▄░▀█▀░░░▒█░▄▀░▄▀▀▄░█▀▀▄░█▀▀▀░█░▄░█▀▀▄░█▀▀░█░░░█
█▄▄█░█░▒█░░█░░░█░░░█▀░▀▄█▄▀░█▄▄█░░█░░░░▒█▀▄░░█░░█░█░▒█░█░▀▄░█▀▄░█▄▄█░█▀▀░▀▄█▄▀
█░░░░░▀▀▀░░▀░░░▀░░▀▀▀░░▀░▀░░▀░░▀░░▀░░░░▒█░▒█░░▀▀░░▀░░▀░▀▀▀▀░▀░▀░▀░░▀░▀▀▀░░▀░▀░
"""
##########################################################################
###############################################################################

import numpy as np
from matplotlib import pyplot as plt
import pickle

###############################################################################

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

with open('Ultimate_input.pkl', 'rb') as file:
    myvar = pickle.load(file)
    TARGET = myvar['TARGET']
    SCREEN_LIMIT = myvar['SCREEN_LIMIT']
    n= myvar['n']
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
    d= myvar['d']
    delta = myvar['delta']
    r0_GL = myvar['r0_GL']
    r0_HA  = myvar['r0_HA']
      
    N = myvar['N']
    num_lenslet = myvar['num_lenslet']
    REPEAT = myvar['REPEAT']
    POSSIBLE_SCREEN = myvar['POSSIBLE_SCREEN']
    POSSIBLE_PROP = myvar['POSSIBLE_PROP']
    POSSIBLE = myvar['POSSIBLE']

with open('NoiselessCircularPupilSlopes.pkl', 'rb') as file:
    myvar = pickle.load(file)
    # PureG = myvar['g_SLOPE']
    PureIG = myvar['Ig_SLOPE']
    PureSHWFS = myvar['SHWFS_SLOPE']

with open('PhotonNoiseNoThresholdCircularPupilSlopes.pkl', 'rb') as file:
    myvar = pickle.load(file)
    SNR_ALL = (myvar['SNR_ALL'])
    RON = myvar['RON']
    NoisyIG = myvar['IgTilt']
    NoisySHWFS = myvar['SHWFSTilt']

with open('fixgtilt.pkl','rb') as file:
    myvar = pickle.load(file)
    PureG = myvar['g_SLOPE']

MM=len(SNR_ALL)

LABEL_r0 = np.zeros((II), dtype=object)
for ii in np.arange(II):
    LABEL_r0[ii] = 'r0={:.0f}cm'.format(r0[ii]*100)

LABEL_DOR0 = np.zeros((JJ), dtype=object)
for jj in np.arange(JJ):
    LABEL_DOR0[jj] = 'd/r0={:.1f}'.format(dor0[jj])
    
LABEL_RYTOV = np.zeros((KK), dtype=object)
for kk in np.arange(KK):
    LABEL_RYTOV[kk] = '$\sigma^2_\chi=${:.2f}'.format(Rytov[kk])
    
LABEL_HHA = np.zeros((LL), dtype=object)
for ll in np.arange(LL):
    LABEL_HHA[ll] = 'HA={:.0f}km'.format(H_HA[ll]/1000)

LABEL_SNR = np.zeros((MM), dtype=object)
for mm in np.arange(MM):
    LABEL_SNR[mm] = 'Nphot={:.0f}'.format(SNR_ALL[mm]**2)

LABEL = np.zeros((II,JJ,KK,LL,MM), dtype=object)

for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                for mm in np.arange(MM):
                    LABEL[ii,jj,kk,ll,mm] = (LABEL_r0[ii] + ', '
                                             + LABEL_DOR0[jj] + ', '
                                             + LABEL_RYTOV[kk] + ', '
                                             + LABEL_HHA[ll] + ', '
                                             + LABEL_SNR[mm])
                    
LABEL_no_m = np.zeros((II,JJ,KK,LL), dtype=object)
for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                for mm in np.arange(MM):
                    LABEL_no_m[ii,jj,kk,ll] = (LABEL_r0[ii] + ', '
                                             + LABEL_DOR0[jj] + ', '
                                             + LABEL_RYTOV[kk] + ', '
                                             + LABEL_HHA[ll])

PureG[PureG==0]=np.nan
PureIG[PureIG==0]=np.nan
PureSHWFS[PureSHWFS==0]=np.nan
NoisyIG[NoisyIG==0]=np.nan
NoisySHWFS[NoisySHWFS==0]=np.nan

ErrorPureG = np.zeros((II,JJ,KK,LL,TARGET,2), dtype=float)
ErrorPureIG = np.zeros((II,JJ,KK,LL,TARGET,2), dtype=float)
# ErrorNoisyIG = np.zeros((II,JJ,KK,LL,MM,TARGET,2), dtype=float)

Pii, Pjj, Pkk, Pll = np.where(POSSIBLE==True)

for possible in np.arange(len(Pii)):
    ii = Pii[possible]
    jj = Pjj[possible]
    kk = Pkk[possible]
    ll = Pll[possible]
    ErrorPureG[ii,jj,kk,ll] = PureG[ii,jj,kk,ll] - PureSHWFS[ii,jj,kk,ll]
    ErrorPureIG[ii,jj,kk,ll] = PureIG[ii,jj,kk,ll] - PureSHWFS[ii,jj,kk,ll]
    # for mm in np.arange(MM):
    #     ErrorNoisyIG[ii,jj,kk,ll,mm] = NoisyIG[ii,jj,kk,ll,mm] - NoisySHWFS[ii,jj,kk,ll,mm]

ErrorPureG[ErrorPureG==0]=np.nan
ErrorPureIG[ErrorPureIG==0]=np.nan
# ErrorNoisyIG[ErrorNoisyIG==0]=np.nan




thisplotmax = np.nanmax([np.nanmax(np.abs([NoisySHWFS,NoisyIG])),
                         np.nanmax(np.abs([PureSHWFS]))])
thisplotmaxy = 0
for mm in np.arange(MM):
    thisplotmaxy = np.nanmax([np.nanmax(np.abs(NoisySHWFS[:,:,:,:,mm,:,:] - PureSHWFS)),
                              np.nanmax(np.abs(NoisyIG[:,:,:,:,mm,:,:] - PureSHWFS)),
                              thisplotmaxy])

thisplotmaxy = thisplotmaxy

for mm in np.arange(MM):
    for possible in np.arange(len(Pii)):
        ii = Pii[possible]
        jj = Pjj[possible]
        kk = Pkk[possible]
        ll = Pll[possible]
        
        # thisplotmaxy = np.nanmax([np.nanmax(np.abs(NoisySHWFS[:,:,:,:,mm,:,:] - PureSHWFS)),
        #                       np.nanmax(np.abs(NoisyIG[:,:,:,:,mm,:,:] - PureSHWFS))])
        plt.plot(PureSHWFS[ii,jj,kk,ll].flatten(),
                  NoisySHWFS[ii,jj,kk,ll,mm].flatten(),
                  ls='',marker='.',alpha=0.1,color='b',markersize=10,
                  label='SHWFS with Photon Noise')
        plt.plot(PureSHWFS[ii,jj,kk,ll].flatten(),
                  NoisyIG[ii,jj,kk,ll,mm].flatten(),
                  ls='',marker='.',alpha=0.1,color='r',markersize=10,
                  label='IG with Photon Noise')
        
        # plt.plot(PureSHWFS[ii,jj,kk,ll].flatten(),
        #           PureIG[ii,jj,kk,ll].flatten(),
        #           ls='',marker='.',alpha=0.25,color='g',label='Noiseless IG')
        plt.xlim(-thisplotmax,thisplotmax)
        plt.ylim(-thisplotmax,thisplotmax)
        plt.legend()
        plt.xlabel('Noiseless SHWFS OPD (rad)')
        plt.ylabel('OPD (rad)')
        plt.title('Slope measurement comparison\n'
                  + LABEL[ii,jj,kk,ll,mm])
        # plt.axis('square')
        plt.show()

        plt.plot(PureSHWFS[ii,jj,kk,ll].flatten(),
                  NoisySHWFS[ii,jj,kk,ll,mm].flatten() - PureSHWFS[ii,jj,kk,ll].flatten(),
                  ls='',marker='.',alpha=0.1,color='b',markersize=10,
                  label='SHWFS with Photon Noise')
        plt.plot(PureSHWFS[ii,jj,kk,ll].flatten(),
                  NoisyIG[ii,jj,kk,ll,mm].flatten() - PureSHWFS[ii,jj,kk,ll].flatten(),
                  ls='',marker='.',alpha=0.1,color='r',markersize=10,
                  label='IG with Photon Noise')
        
        # plt.plot(PureSHWFS[ii,jj,kk,ll].flatten(),
        #           PureIG[ii,jj,kk,ll].flatten() - PureSHWFS[ii,jj,kk,ll].flatten(),
        #           ls='',marker='.',alpha=0.25,color='g',label='Noiseless IG')
        plt.xlim(-thisplotmax,thisplotmax)
        plt.ylim(-thisplotmaxy,thisplotmaxy)
        plt.legend()
        plt.xlabel('Noiseless SHWFS OPD (rad)')
        plt.ylabel('Difference OPD (rad)')
        plt.title('Slope measurement difference comparison\n'
                  + LABEL[ii,jj,kk,ll,mm])
        
        plt.show()



thisplotmax = np.nanmax([np.nanmean((PureIG - PureSHWFS)**2, axis=(-1,-2)),
                         np.nanmean((PureG - PureSHWFS)**2, axis=(-1,-2))])
var = (np.nanmean((PureIG - PureSHWFS)**2, axis=(-1,-2)))
varmin = np.nanmin(var)
varmax = np.nanmax(var)
color_map = plt.cm.get_cmap('inferno')
reversed_color_map = color_map.reversed()
plt.subplots(4,4,sharex=True,sharey=True,squeeze=False)
# plt.subplot(4,4,1)
plt.suptitle('RMS difference of IG and FT OPD (rad/lenslet)\n'
              + 'Noiseless')
for kk in np.arange(KK):
    for ll in np.arange(LL):
        plt.subplot(4,4,LL*(KK - kk - 1) + ll + 1)
        plt.pcolor(r0*100, dor0, np.sqrt(var[:,:,kk,ll]).T,
                    cmap=reversed_color_map,edgecolors='k',linewidths=0.5,
                    vmin=0,vmax=thisplotmax)
        if kk == 0:
            plt.xlabel('r0 (cm)\n' + LABEL_HHA[ll])
        if ll == 0:
            plt.ylabel(LABEL_RYTOV[kk] + '\nd/r0')
        if ll == (LL - 1):
            plt.colorbar()
plt.show()
plt.clf()
var = (np.nanmean((PureG - PureSHWFS)**2, axis=(-1,-2)))
varmin = np.nanmin(var)
varmax = np.nanmax(var)
color_map = plt.cm.get_cmap('inferno')
reversed_color_map = color_map.reversed()
plt.subplots(4,4,sharex=True,sharey=True,squeeze=False)
# plt.subplot(4,4,1)
plt.suptitle('RMS difference of G and FT OPD (rad/lenslet)\n'
              + 'Noiseless')
for kk in np.arange(KK):
    for ll in np.arange(LL):
        plt.subplot(4,4,LL*(KK - kk - 1) + ll + 1)
        plt.pcolor(r0*100, dor0, np.sqrt(var[:,:,kk,ll]).T,
                    cmap=reversed_color_map,edgecolors='k',linewidths=0.5,
                    vmin=0,vmax=thisplotmax)
        if kk == 0:
            plt.xlabel('r0 (cm)\n' + LABEL_HHA[ll])
        if ll == 0:
            plt.ylabel(LABEL_RYTOV[kk] + '\nd/r0')
        if ll == (LL - 1):
            plt.colorbar()
plt.show()
plt.clf()


thisplotmax = np.nanmax([np.nanmean((PureIG - PureSHWFS)**2, axis=(-1,-2))])
var = (np.nanmean((PureIG - PureSHWFS)**2, axis=(-1,-2)))
varmin = np.nanmin(var)
varmax = np.nanmax(var)
color_map = plt.cm.get_cmap('inferno')
reversed_color_map = color_map.reversed()
plt.subplots(4,4,sharex=True,sharey=True,squeeze=False)
# plt.subplot(4,4,1)
plt.suptitle('RMS difference of IG and FT OPD (rad/lenslet)\n'
              + 'Noiseless')
for kk in np.arange(KK):
    for ll in np.arange(LL):
        plt.subplot(4,4,LL*(KK - kk - 1) + ll + 1)
        plt.pcolor(r0*100, dor0, np.sqrt(var[:,:,kk,ll]).T,
                    cmap=reversed_color_map,edgecolors='k',linewidths=0.5,
                    vmin=np.sqrt(varmin),vmax=np.sqrt(varmax))
        if kk == 0:
            plt.xlabel('r0 (cm)\n' + LABEL_HHA[ll])
        if ll == 0:
            plt.ylabel(LABEL_RYTOV[kk] + '\nd/r0')
        if ll == (LL - 1):
            plt.colorbar()
plt.show()
plt.clf()


        
for mm in np.arange(0,MM):
    
    var = np.nanvar(NoisyIG[:,:,:,:,mm,:,:] - PureSHWFS, ddof=1, axis=(-1,-2))
    varmin = np.nanmin(var)
    varmax = np.nanmax(var)
    color_map = plt.cm.get_cmap('inferno')
    reversed_color_map = color_map.reversed()
    plt.subplots(4,4,sharex=True,sharey=True,squeeze=False)
    # plt.subplot(4,4,1)
    plt.suptitle('RMS difference of Noisy IG and SHWFS slopes (rad/lenslet)\n'
                  + 'With Photon Noise, ' + LABEL_SNR[mm])
    for kk in np.arange(KK):
        for ll in np.arange(LL):
            plt.subplot(4,4,LL*(KK - kk - 1) + ll + 1)
            plt.pcolor(r0*100, dor0, np.sqrt(var[:,:,kk,ll]).T,
                        vmin=np.sqrt(varmin),vmax=np.sqrt(varmax),
                        cmap=reversed_color_map,edgecolors='k',linewidths=0.5)
            if kk == 0:
                plt.xlabel('r0 (cm)\n' + LABEL_HHA[ll])
            if ll == 0:
                plt.ylabel(LABEL_RYTOV[kk] + '\nd/r0')
            if ll == (LL - 1):
                plt.colorbar()
    plt.show()
    plt.clf()
    
    
    
    
# SHOW noiseless g ig ft
# show that g tilt is bad

thisplotmax = np.nanmax(np.abs([PureG, PureIG, PureSHWFS]))
thisplotmaxy = np.nanmax(np.abs([PureG - PureSHWFS, PureIG - PureSHWFS]))

thisplotmaxy = thisplotmaxy

for possible in [0,-1]:#np.arange(len(Pii)):
    ii = Pii[possible]
    jj = Pjj[possible]
    kk = Pkk[possible]
    ll = Pll[possible]
    
    # ii=0
    # jj=1
    # kk=0
    # ll=0

    plt.plot(PureSHWFS[ii,jj,kk,ll].flatten(),
              PureG[ii,jj,kk,ll].flatten(),
              ls='',marker='.',alpha=1,color='g',markersize=5,
              label='G-Tilt')
    plt.plot(PureSHWFS[ii,jj,kk,ll].flatten(),
              PureIG[ii,jj,kk,ll].flatten(),
              ls='',marker='.',alpha=1,color='r',markersize=5,
              label='IG-Tilt')
    plt.plot(np.linspace(-thisplotmax,thisplotmax),np.linspace(-thisplotmax,thisplotmax),
             ls='-',marker='',label='FT-Tilt',color='b')
    plt.xlim(-thisplotmax,thisplotmax)
    plt.ylim(-thisplotmax,thisplotmax)
    plt.legend()
    plt.xlabel('Noiseless FT OPD (rad)')
    plt.ylabel('Measured OPD (rad)')
    plt.title('Comparison of measured OPD per lenslet (rad)\n'
              + LABEL_no_m[ii,jj,kk,ll])
    plt.show()

    plt.plot(PureSHWFS[ii,jj,kk,ll].flatten(),
              PureG[ii,jj,kk,ll].flatten() - PureSHWFS[ii,jj,kk,ll].flatten(),
              ls='',marker='.',alpha=1,color='g',markersize=5,
              label='G-FT Tilt')
    plt.plot(PureSHWFS[ii,jj,kk,ll].flatten(),
              PureIG[ii,jj,kk,ll].flatten() - PureSHWFS[ii,jj,kk,ll].flatten(),
              ls='',marker='.',alpha=1,color='r',markersize=5,
              label='IG-FT Tilt')
    plt.hlines(0,xmin=-thisplotmax,xmax=thisplotmax,label='FT - FT = 0',color='b')
    
    plt.xlim(-thisplotmax,thisplotmax)
    plt.ylim(-thisplotmaxy,thisplotmaxy)
    plt.legend()
    plt.xlabel('Noiseless FT OPD (rad)')
    plt.ylabel('Difference between measured OPDs (rad)')
    plt.title('Comparison of error in measured OPD per lenslet (rad)\n'
              + LABEL_no_m[ii,jj,kk,ll])
    
    plt.show()

#now again but without g-tilt

thisplotmax = np.nanmax(np.abs([PureIG, PureSHWFS]))
thisplotmaxy = np.nanmax(np.abs([PureIG - PureSHWFS]))

for possible in [0,-1]:#np.arange(len(Pii)):
    ii = Pii[possible]
    jj = Pjj[possible]
    kk = Pkk[possible]
    ll = Pll[possible]
    
    # ii=0
    # jj=1
    # kk=0
    # ll=0


    plt.plot(PureSHWFS[ii,jj,kk,ll].flatten(),
              PureIG[ii,jj,kk,ll].flatten(),
              ls='',marker='.',alpha=1,color='r',markersize=5,
              label='IG-Tilt')
    plt.plot(np.linspace(-thisplotmax,thisplotmax),np.linspace(-thisplotmax,thisplotmax),
             ls='-',marker='',label='FT-Tilt',color='b')
    plt.xlim(-thisplotmax,thisplotmax)
    plt.ylim(-thisplotmax,thisplotmax)
    plt.legend()
    plt.xlabel('Noiseless FT OPD (rad)')
    plt.ylabel('Measured OPD (rad)')
    plt.title('Comparison of measured OPD per lenslet (rad)\n'
              + LABEL_no_m[ii,jj,kk,ll])
    plt.show()

    plt.plot(PureSHWFS[ii,jj,kk,ll].flatten(),
              PureIG[ii,jj,kk,ll].flatten() - PureSHWFS[ii,jj,kk,ll].flatten(),
              ls='',marker='.',alpha=1,color='r',markersize=5,
              label='IG-FT Tilt')
    plt.hlines(0,xmin=-thisplotmax,xmax=thisplotmax,label='FT - FT = 0',color='b')
    
    plt.xlim(-thisplotmax,thisplotmax)
    plt.ylim(-thisplotmaxy,thisplotmaxy)
    plt.legend()
    plt.xlabel('Noiseless FT OPD (rad)')
    plt.ylabel('Difference between measured OPDs (rad)')
    plt.title('Comparison of error in measured OPD per lenslet (rad)\n'
              + LABEL_no_m[ii,jj,kk,ll])
    
    plt.show()



# show that d/r0 is the strongest
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[0,:,:,:].flatten()**0.5,marker='o',ls='',label='r0=11cm')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[1,:,:,:].flatten()**0.5,marker='o',ls='',label='r0=8cm')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[2,:,:,:].flatten()**0.5,marker='o',ls='',label='r0=5cm')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[3,:,:,:].flatten()**0.5,marker='o',ls='',label='r0=2cm')
plt.legend()

plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,0,:,:].flatten()**0.5,marker='o',ls='',label='d/r0=0.5')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,1,:,:].flatten()**0.5,marker='o',ls='',label='d/r0=1')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,2,:,:].flatten()**0.5,marker='o',ls='',label='d/r0=2')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,3,:,:].flatten()**0.5,marker='o',ls='',label='d/r0=3')
plt.legend()

plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,:,0,:].flatten()**0.5,marker='o',ls='',label='rytov=0.01')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,:,1,:].flatten()**0.5,marker='o',ls='',label='rytov=0.03')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,:,2,:].flatten()**0.5,marker='o',ls='',label='rytov=0.1')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,:,3,:].flatten()**0.5,marker='o',ls='',label='rytov=0.3')
plt.legend()

plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,:,:,0].flatten()**0.5,marker='o',ls='',label='H_HA=5km')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,:,:,1].flatten()**0.5,marker='o',ls='',label='H_HA=10km')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,:,:,2].flatten()**0.5,marker='o',ls='',label='H_HA=15km')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,:,:,3].flatten()**0.5,marker='o',ls='',label='H_HA=20km')
plt.legend()


# show that after d/r0 it is rytov
# plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,0,0,:].flatten()**0.5,marker='^',ls='',label='d/r0=0.5, rytov=0.01')
# plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,0,1,:].flatten()**0.5,marker='^',ls='',label='d/r0=0.5, rytov=0.03')
# plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,0,2,:].flatten()**0.5,marker='^',ls='',label='d/r0=0.5, rytov=0.1')
# plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,0,3,:].flatten()**0.5,marker='^',ls='',label='d/r0=0.5, rytov=0.3')
# # plt.legend()

# plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,1,0,:].flatten()**0.5,marker='o',ls='',label='d/r0=1, rytov=0.01')
# plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,1,1,:].flatten()**0.5,marker='o',ls='',label='d/r0=1, rytov=0.03')
# plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,1,2,:].flatten()**0.5,marker='o',ls='',label='d/r0=1, rytov=0.1')
# plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,1,3,:].flatten()**0.5,marker='o',ls='',label='d/r0=1, rytov=0.3')
# # plt.legend()

# plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,2,0,:].flatten()**0.5,marker='x',ls='',label='d/r0=2, rytov=0.01')
# plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,2,1,:].flatten()**0.5,marker='x',ls='',label='d/r0=2, rytov=0.03')
# plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,2,2,:].flatten()**0.5,marker='x',ls='',label='d/r0=2, rytov=0.1')
# plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,2,3,:].flatten()**0.5,marker='x',ls='',label='d/r0=2, rytov=0.3')
# # plt.legend()

plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[0,3,:,:].flatten()**0.5,marker='v',ls='',label='d/r0=3, r0=11cm')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[1,3,:,:].flatten()**0.5,marker='v',ls='',label='d/r0=3, r0=8cm')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[2,3,:,:].flatten()**0.5,marker='v',ls='',label='d/r0=3, r0=5cm')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[3,3,:,:].flatten()**0.5,marker='v',ls='',label='d/r0=3, r0=2cm')
plt.legend()

plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,3,0,:].flatten()**0.5,marker='v',ls='',label='d/r0=3, rytov=0.01')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,3,1,:].flatten()**0.5,marker='v',ls='',label='d/r0=3, rytov=0.03')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,3,2,:].flatten()**0.5,marker='v',ls='',label='d/r0=3, rytov=0.1')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,3,3,:].flatten()**0.5,marker='v',ls='',label='d/r0=3, rytov=0.3')
plt.legend()

plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,3,:,0].flatten()**0.5,marker='v',ls='',label='d/r0=3, HHA=5km')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,3,:,1].flatten()**0.5,marker='v',ls='',label='d/r0=3, HHA=10km')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,3,:,2].flatten()**0.5,marker='v',ls='',label='d/r0=3, HHA=15km')
plt.plot(np.nanmean(ErrorPureIG**2,axis=(-1,-2))[:,3,:,3].flatten()**0.5,marker='v',ls='',label='d/r0=3, HHA=20km')
plt.legend()


plt.pcolor(np.sqrt(np.nanmean(ErrorPureIG**2,axis=(0,3,-1,-2))).T,edgecolors='k',linewidths=1)
plt.xlabel('d/r0')
plt.ylabel('Rytov')
plt.title('RMS OPD (IG-FT)')
plt.colorbar()

for mm in np.arange(MM):
    X = np.sqrt(np.nanmean((NoisyIG[:,:,:,:,mm]-PureSHWFS)**2,axis=(0,3,-1,-2))).T
    Y = np.sqrt(np.nanmean((NoisySHWFS[:,:,:,:,mm]-PureSHWFS)**2,axis=(0,3,-1,-2))).T
    thismin = np.nanmin([X,Y])
    thismax = np.nanmax([X,Y])
    plt.pcolor(X,edgecolors='k',linewidths=1,vmin=thismin,vmax=thismax)
    plt.xlabel('d/r0')
    plt.ylabel('Rytov')
    plt.title('RMS OPD (Noisy IG - FT) ' + LABEL_SNR[mm])
    plt.colorbar()
    plt.show()
    plt.pcolor(Y,edgecolors='k',linewidths=1,vmin=thismin,vmax=thismax)
    plt.xlabel('d/r0')
    plt.ylabel('Rytov')
    plt.title('RMS OPD (Noisy FT - FT) ' + LABEL_SNR[mm])
    plt.colorbar()
    plt.show()




thisplotmax = np.nanmax(np.abs([PureG, PureIG, PureSHWFS]))
for mm in np.arange(2):
    thisplotmaxy = np.max([np.nanmax(np.abs(NoisyIG[:,:,:,:,mm] - PureSHWFS)),
                           np.nanmax(np.abs(NoisySHWFS[:,:,:,:,mm] - PureSHWFS))])
    for possible in np.arange(len(Pii)):
        ii = Pii[possible]
        jj = Pjj[possible]
        kk = Pkk[possible]
        ll = Pll[possible]
        
        ii = 2
        jj = 1
        kk = 2
        ll = 0
    
        # plt.plot(PureSHWFS[ii,jj,kk,ll].flatten(),
        #           NoisySHWFS[ii,jj,kk,ll,mm].flatten(),
        #           ls='',marker='.',alpha=0.1,color='b',markersize=10,
        #           label='Noisy FT-Tilt')
        # plt.plot(PureSHWFS[ii,jj,kk,ll].flatten(),
        #           NoisyIG[ii,jj,kk,ll,mm].flatten(),
        #           ls='',marker='.',alpha=0.1,color='r',markersize=10,
        #           label='Noisy IG-Tilt')
        # plt.plot(np.linspace(-thisplotmax,thisplotmax),np.linspace(-thisplotmax,thisplotmax),
        #          ls='-',marker='',label='Noiseless FT-Tilt',color='g')
        # plt.xlim(-thisplotmax,thisplotmax)
        # plt.ylim(-thisplotmax,thisplotmax)
        # plt.legend()
        # plt.xlabel('Noiseless FT OPD (rad)')
        # plt.ylabel('Measured OPD (rad)')
        # plt.title('Comparison of measured OPD per lenslet (rad)\n'
        #           + LABEL[ii,jj,kk,ll,mm])
        # plt.show()
    
        plt.plot(PureSHWFS[ii,jj,kk,ll].flatten(),
                  NoisySHWFS[ii,jj,kk,ll,mm].flatten() - PureSHWFS[ii,jj,kk,ll].flatten(),
                  ls='',marker='.',alpha=0.1,color='b',markersize=5,
                  label='Noisy FT - Noiseless FT Tilt')
        plt.plot(PureSHWFS[ii,jj,kk,ll].flatten(),
                  NoisyIG[ii,jj,kk,ll,mm].flatten() - PureSHWFS[ii,jj,kk,ll].flatten(),
                  ls='',marker='.',alpha=0.1,color='r',markersize=5,
                  label='Noisy IG - Noiseless FT Tilt')
        plt.hlines(0,xmin=-thisplotmax,xmax=thisplotmax,label='FT - FT = 0',color='b')
        
        plt.xlim(-thisplotmax,thisplotmax)
        plt.ylim(-thisplotmaxy,thisplotmaxy)
        plt.legend()
        plt.xlabel('Noiseless FT OPD (rad)')
        plt.ylabel('Difference between measured OPDs (rad)')
        plt.title('Comparison of error in measured OPD per lenslet (rad)\n'
                  + LABEL[ii,jj,kk,ll,mm])
        
        plt.show()
        
        plt.hist(NoisySHWFS[ii,jj,kk,ll,mm].flatten() - PureSHWFS[ii,jj,kk,ll].flatten(),
                 label='Noisy FT - Noiseless FT Tilt', color='b')
        plt.hist(NoisySHWFS[ii,jj,kk,ll,mm].flatten() - PureSHWFS[ii,jj,kk,ll].flatten(),
                 label='Noisy FT - Noiseless FT Tilt', color='b')
        
    

for mm in np.arange(MM):
    thisplotmax = np.nanmax([np.nanmean((NoisyIG[:,:,:,:,mm] - PureSHWFS)**2, axis=(-1,-2)),
                             np.nanmean((NoisySHWFS[:,:,:,:,mm] - PureSHWFS)**2, axis=(-1,-2))])
    thisplotmin = np.nanmin([np.nanmean((NoisyIG[:,:,:,:,mm] - PureSHWFS)**2, axis=(-1,-2)),
                             np.nanmean((NoisySHWFS[:,:,:,:,mm] - PureSHWFS)**2, axis=(-1,-2))])
    var = (np.nanmean((NoisyIG[:,:,:,:,mm] - PureSHWFS)**2, axis=(-1,-2)))
    varmin = np.nanmin(var)
    varmax = np.nanmax(var)
    color_map = plt.cm.get_cmap('inferno')
    reversed_color_map = color_map.reversed()
    plt.subplots(4,4,sharex=True,sharey=True,squeeze=False)
    # plt.subplot(4,4,1)
    plt.suptitle('RMS difference of Noisy IG and FT OPD (rad/lenslet)\n'
                  + 'Noiseless ' + LABEL_SNR[mm] )
    for kk in np.arange(KK):
        for ll in np.arange(LL):
            plt.subplot(4,4,LL*(KK - kk - 1) + ll + 1)
            plt.pcolor(r0*100, dor0, np.sqrt(var[:,:,kk,ll]).T,
                        cmap=reversed_color_map,edgecolors='k',linewidths=0.5,
                        vmin=thisplotmin**0.5,vmax=thisplotmax**0.5)
            if kk == 0:
                plt.xlabel('r0 (cm)\n' + LABEL_HHA[ll])
            if ll == 0:
                plt.ylabel(LABEL_RYTOV[kk] + '\nd/r0')
            if ll == (LL - 1):
                plt.colorbar()
    plt.show()
    plt.clf()
    
    var = (np.nanmean((NoisySHWFS[:,:,:,:,mm] - PureSHWFS)**2, axis=(-1,-2)))
    varmin = np.nanmin(var)
    varmax = np.nanmax(var)
    color_map = plt.cm.get_cmap('inferno')
    reversed_color_map = color_map.reversed()
    plt.subplots(4,4,sharex=True,sharey=True,squeeze=False)
    # plt.subplot(4,4,1)
    plt.suptitle('RMS difference of Noisy FT and FT OPD (rad/lenslet)\n'
                  + 'Noiseless ' + LABEL_SNR[mm] )
    for kk in np.arange(KK):
        for ll in np.arange(LL):
            plt.subplot(4,4,LL*(KK - kk - 1) + ll + 1)
            plt.pcolor(r0*100, dor0, np.sqrt(var[:,:,kk,ll]).T,
                        cmap=reversed_color_map,edgecolors='k',linewidths=0.5,
                        vmin=thisplotmin**0.5,vmax=thisplotmax**0.5)
            if kk == 0:
                plt.xlabel('r0 (cm)\n' + LABEL_HHA[ll])
            if ll == 0:
                plt.ylabel(LABEL_RYTOV[kk] + '\nd/r0')
            if ll == (LL - 1):
                plt.colorbar()
    plt.show()
    plt.clf()
    
    X = np.nanmean((NoisySHWFS[:,:,:,:,mm] - PureSHWFS)**2,
                        axis=(-1,-2))
    plt.plot(X[:,0,:,:].flatten()**0.5,
             marker='o',ls='',color='b',label='Noisy FT')
    plt.plot(X[:,1,:,:].flatten()**0.5
             ,marker='o',ls='',color='b')
    plt.plot(X[:,2,:,:].flatten()**0.5,
             marker='o',ls='',color='b')
    plt.plot(X[:,3,:,:].flatten()**0.5,
             marker='o',ls='',color='b')
    # plt.plot(X[:,0,:,:].flatten()**0.5,
    #          marker='o',ls='',label='d/r0=0.5')
    # plt.plot(X[:,1,:,:].flatten()**0.5
    #          ,marker='o',ls='',label='d/r0=1')
    # plt.plot(X[:,2,:,:].flatten()**0.5,
    #          marker='o',ls='',label='d/r0=2')
    # plt.plot(X[:,3,:,:].flatten()**0.5,
    #          marker='o',ls='',label='d/r0=3')
    # plt.legend()
    
    Y = np.nanmean((NoisyIG[:,:,:,:,mm] - PureSHWFS)**2,
                        axis=(-1,-2))
    plt.plot(Y[:,0,:,:].flatten()**0.5,
             marker='o',ls='',color='r',label='Noisy IG')
    plt.plot(Y[:,1,:,:].flatten()**0.5
             ,marker='o',ls='',color='r')
    plt.plot(Y[:,2,:,:].flatten()**0.5,
             marker='o',ls='',color='r')
    plt.plot(Y[:,3,:,:].flatten()**0.5,
             marker='o',ls='',color='r')
    # plt.legend()
    # plt.plot(Y[:,0,:,:].flatten()**0.5,
    #          marker='o',ls='',label='d/r0=0.5')
    # plt.plot(Y[:,1,:,:].flatten()**0.5
    #          ,marker='o',ls='',label='d/r0=1')
    # plt.plot(Y[:,2,:,:].flatten()**0.5,
    #          marker='o',ls='',label='d/r0=2')
    # plt.plot(Y[:,3,:,:].flatten()**0.5,
    #          marker='o',ls='',label='d/r0=3')
    plt.legend()
