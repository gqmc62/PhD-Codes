# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 12:54:56 2022

@author: gqmc62
"""

import numpy as np
from matplotlib import pyplot as plt
from aotools.turbulence.phasescreen import ft_phase_screen
import my_propagation as mprop
from r0RytovDistribution import distribute_coherent_length

r0 = np.array([2,4,6,8,10], dtype=float)*1e-2       # ii
dd = np.array([2,4,6,8,10], dtype=float)*1e-2       # jj
Fresnel = np.array([2,4,6], dtype=float)*1e-2  # kk
Rytov = np.array([0.1,0.2,0.3], dtype=float)      # ll
# fov = np.array([1,2,4,8,16,32,64,128], dtype=int)   # mm # wvl/dd
fov = np.array([32], dtype=int)   # mm # wvl/dd

II = len(r0)
JJ = len(dd)
KK = len(Fresnel)
LL = len(Rytov)
MM = len(fov)

wvl = float(500e-9)
zenith_angle = float(0)
Ntemp = int(16)
Ntrial = int(16)
L0 = float(20)
l0 = float(1e-7)
min_rms_error = float(10e-9)
spot_size = int(2) # pixel_angular_scale*spot_size = wvl/D

HA = (Fresnel/0.8)**2/wvl   # kk
n = fov * spot_size         # mm

rG = np.zeros((II,KK,LL), dtype=float)
rH = np.zeros((II,KK,LL), dtype=float)
N = np.zeros((JJ,KK,MM), dtype=int)
num_lenslet = np.zeros((JJ,KK,MM), dtype=int)
delta = np.zeros((JJ,MM), dtype=float)

for jj in np.arange(JJ):
    for mm in np.arange(MM):
        delta[jj,mm] = dd[jj]/n[mm]    # jj

for ii in np.arange(II):
    for kk in np.arange(KK):
        for ll in np.arange(LL):
            rG[ii,kk,ll] , rH[ii,kk,ll] = distribute_coherent_length(
                r0[ii], Rytov[ll], HA[kk], wvl, zenith_angle)

for jj in np.arange(JJ):
    for kk in np.arange(KK):
        for mm in np.arange(MM):
            N[jj,kk,mm] = mprop.perfect_size_free(
                delta[jj,mm], HA[kk], wvl)

for jj in np.arange(JJ):
    for kk in np.arange(KK):
        for mm in np.arange(MM):
            num_lenslet[jj,kk,mm] = max(
                N[jj,kk,mm]/n[mm], 1)

Nyquist = np.zeros((II,JJ,KK,LL,MM), dtype=bool)
for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                for mm in np.arange(MM):
                    Nyquist[ii,jj,kk,ll,mm] = ((r0[ii]/delta[jj,mm] >= 5)
                                         * (rG[ii,kk,ll]/delta[jj,mm] >= 5)
                                         * (rH[ii,kk,ll]/delta[jj,mm] >= 5))

good_size = np.zeros((JJ,KK,MM), dtype=bool)
for jj in np.arange(JJ):
    for kk in np.arange(KK):
        for mm in np.arange(MM):
            good_size[jj,kk,mm] = (N[jj,kk,mm] <= 2**14)

good_fov = np.zeros((II,JJ,MM), dtype=bool)
for ii in np.arange(II):
    for jj in np.arange(JJ):
        for mm in np.arange(MM):
            good_fov[ii,jj,mm] = (fov[mm]
                                  >= np.sqrt(0.5/(min_rms_error*2*np.pi/wvl)**2
                                             * (dd[jj]/r0[ii])**(5/3)))

good_r = np.zeros((II,KK,LL), dtype=bool)
for ii in np.arange(II):
    for kk in np.arange(KK):
        for ll in np.arange(LL):
            good_r[ii,kk,ll] = ((np.isnan(rG[ii,kk,ll]) == False)
                                * (np.isnan(rH[ii,kk,ll]) == False))

Fresnel_sampled = np.zeros((JJ,KK,MM), dtype=bool)
for jj in np.arange(JJ):
    for kk in np.arange(KK):
        for mm in np.arange(MM):
            Fresnel_sampled[jj,kk,mm] = (Fresnel[kk] >= 2*delta[jj,mm])

valid = np.zeros((II,JJ,KK,LL,MM), dtype=bool)
for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                for mm in np.arange(MM):
                    valid[ii,jj,kk,ll,mm] = (Nyquist[ii,jj,kk,ll,mm]
                                             * good_size[jj,kk,mm]
                                             * good_fov[ii,jj,mm]
                                             * good_r[ii,kk,ll]
                                             * Fresnel_sampled[jj,kk,mm])


III,JJJ,KKK,LLL,MMM = np.where(valid==True)
for i in np.arange(len(III)):
    # print(r0[III[i]],dd[JJJ[i]],Fresnel[KKK[i]],Rytov[LLL[i]],fov[MMM[i]])
    print(r0[III[i]],dd[JJJ[i]],Fresnel[KKK[i]],Rytov[LLL[i]])

print(valid.sum())