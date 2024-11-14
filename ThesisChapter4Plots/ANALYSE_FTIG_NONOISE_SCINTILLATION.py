# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:06:32 2024

@author: gqmc62
"""

import pickle
import numpy as np
from matplotlib import pyplot as plt

def cov(x,y):
    tempx = x - np.nanmean(x)
    tempy = y - np.nanmean(y)
    temp = np.nanmean(tempx*tempy)
    return temp

def corr(x,y):
    temp2 = cov(x,y)/np.nanstd(x)/np.nanstd(y)
    return temp2

with open('Ultimate_input.pkl','rb') as file:
    input_data = pickle.load(file)
keys = input_data.keys()
for item_i in np.arange(len(keys)):
    key_name = list(keys)[item_i]
    data = input_data[key_name]
    globals()[key_name] = data
    
with open('NoiselessCircularPupilSlopes.pkl','rb') as file:
    input_data = pickle.load(file)
keys = input_data.keys()
for item_i in np.arange(len(keys)):
    key_name = list(keys)[item_i]
    data = input_data[key_name]
    globals()[key_name] = data
    
    

# SHWFS_SLOPE noise free II,JJ,KK,LL,REPEAT,2
# g_SLOPE
# Ig_SLOPE
# II,JJ,KK,LL
# r0,dor0,Rytov,H_HA

RMS_IG_FT = np.std(Ig_SLOPE - SHWFS_SLOPE,axis=(-1,-2))
RMS_G_FT = np.std(g_SLOPE - SHWFS_SLOPE,axis=(-1,-2))

RMS_IG_FT[RMS_IG_FT==0] = np.nan
RMS_G_FT[RMS_G_FT==0] = np.nan

# plt.plot((RMS_IG_FT/RMS_G_FT)[POSSIBLE].flatten())
# plt.title('error of ig-tilt to error of g-tilt ratio')
# plt.ylabel('error of ig-tilt to error of g-tilt ratio')
# plt.xlabel('arbitrary index')
# plt.show(

plt.plot(RMS_IG_FT.flatten(),marker='o',ls='-')
plt.xlabel('arbitrary index')
plt.ylabel('error of ig-tilt (rad)')
plt.title('all conditions')
plt.show()

# for ii in np.arange(II):
#     plt.plot(RMS_IG_FT[ii].flatten(),marker='o',ls='-',label='r0={}'.format(r0[ii]))
# plt.legend()
# plt.xlabel('arbitrary index')
# plt.ylabel('error of ig-tilt (rad)')
# plt.title('effect of having varying r0')
# plt.show()


for jj in np.arange(JJ):
    plt.plot(RMS_IG_FT[:,jj].flatten(),marker='o',ls='-',label='dor0={}'.format(dor0[jj]))
plt.legend()
plt.xlabel('arbitrary index')
plt.ylabel('error of ig-tilt (rad)')
plt.title('effect of having varying dor0')
plt.show()

plt.errorbar(dor0,np.nanmean(RMS_IG_FT,axis=(0,2,3)),yerr=np.nanstd(RMS_IG_FT,axis=(0,2,3)),capsize=5,marker='o')
plt.xlabel('dor0')
plt.ylabel('error of ig-tilt (rad)')
plt.title('dor0 is the primary variable')
plt.show()


# for ii in np.arange(4):
#     for kk in np.arange(4):
#         for ll in np.arange(4):
#             plt.plot(RMS_IG_FT[ii,:,kk,ll],ls='-',marker='o')
# plt.show()
            

dor0_mean = np.nanmean(RMS_IG_FT,axis=(0,2,3))

# for kk in np.arange(KK):
#     plt.plot(RMS_IG_FT[:,:,kk].flatten(),marker='o',ls='-',label='Rytov={}'.format(Rytov[kk]))
# plt.legend()
# plt.xlabel('arbitrary index')
# plt.ylabel('error of ig-tilt (rad)')
# plt.title('effect of having varying Rytov')
# plt.show()

# for ll in np.arange(LL):
#     plt.plot(RMS_IG_FT[:,:,:,ll].flatten(),marker='o',ls='-',label='H_HA={}'.format(H_HA[ll]))
# plt.legend()
# plt.xlabel('arbitrary index')
# plt.ylabel('error of ig-tilt (rad)')
# plt.title('effect of having varying H_HA')
# plt.show()



# for jj in np.arange(JJ):
#     for ii in np.arange(II):
#         plt.plot(RMS_IG_FT[ii,jj].flatten(),marker='o',ls='',label='r0={}'.format(r0[ii]))
#     plt.legend()
#     plt.xlabel('arbitrary index')
#     plt.ylabel('error of ig-tilt (rad)')
#     plt.title('effect of having varying r0 with fixed dor0={}'.format(dor0[jj]))
#     plt.show()
    
# for jj in np.arange(JJ):
#     for kk in np.arange(KK):
#         plt.plot(RMS_IG_FT[:,jj,kk].flatten(),marker='o',ls='',label='Rytov={}'.format(Rytov[kk]))
#     plt.legend()
#     plt.xlabel('arbitrary index')
#     plt.ylabel('error of ig-tilt (rad)')
#     plt.title('effect of having varying Rytov with fixed dor0={}'.format(dor0[jj]))
#     plt.show()
    
# for jj in np.arange(JJ):
#     for ll in np.arange(LL):
#         plt.plot(RMS_IG_FT[:,jj,:,ll].flatten(),marker='o',ls='',label='H_HA={}'.format(Rytov[ll]))
#     plt.legend()
#     plt.xlabel('arbitrary index')
#     plt.ylabel('error of ig-tilt (rad)')
#     plt.title('effect of having varying H_HA with fixed dor0={}'.format(dor0[jj]))
#     plt.show()


# X = RMS_IG_FT

# YII = np.zeros_like(RMS_IG_FT)
# for ii in np.arange(II):
#     YII[ii] = r0[ii]

# YJJ = np.zeros_like(RMS_IG_FT)
# for jj in np.arange(JJ):
#     YJJ[:,jj] = dor0[jj]
    
# YKK = np.zeros_like(RMS_IG_FT)
# for kk in np.arange(KK):
#     YKK[:,:,kk] = Rytov[kk]
    
# YLL = np.zeros_like(RMS_IG_FT)
# for ll in np.arange(LL):
#     YLL[:,:,:,ll] = H_HA[ll]
    
# print(corr(X,YII),corr(X,YJJ),corr(X,YKK),corr(X,YLL))



# for jj in np.arange(JJ):
#     X = RMS_IG_FT[:,jj]

#     YII = np.zeros_like(X)
#     for ii in np.arange(II):
#         YII[ii] = r0[ii]

#     YKK = np.zeros_like(X)
#     for kk in np.arange(KK):
#         YKK[:,kk] = Rytov[kk]
        
#     YLL = np.zeros_like(X)
#     for ll in np.arange(LL):
#         YLL[:,:,ll] = H_HA[ll]
    
#     print(corr(X,YII),corr(X,YKK),corr(X,YLL))
    
# plt.plot(np.nanmean(RMS_IG_FT,axis=(0,1,2)),label='avg of dif H_HA',marker='o')
# plt.plot(np.nanmean(RMS_IG_FT,axis=(0,1,3)),label='avg of dif Rytov',marker='o')
# plt.plot(np.nanmean(RMS_IG_FT,axis=(0,2,3)),label='avg of dif dor0',marker='o')
# plt.plot(np.nanmean(RMS_IG_FT,axis=(1,2,3)),label='avg of dif r0',marker='o')
# plt.legend()
# plt.show()

# plt.plot(['H_HA','Rytov','dor0','r0'],[np.nanstd(np.nanmean(RMS_IG_FT,axis=(0,1,2))),
# np.nanstd(np.nanmean(RMS_IG_FT,axis=(0,1,3))),
# np.nanstd(np.nanmean(RMS_IG_FT,axis=(0,2,3))),
# np.nanstd(np.nanmean(RMS_IG_FT,axis=(1,2,3)))],marker='o')
# plt.show()

TEMPMEAN = np.nanmean(RMS_IG_FT,axis=(0,2,3))
TEMP = np.copy(RMS_IG_FT)
for i in np.arange(4):
    TEMP[:,i] = TEMP[:,i] - TEMPMEAN[i]
    
TEMP2 = np.nanmean(TEMP,axis=1)


plt.plot(TEMP2.flatten(),marker='o',ls='')
plt.xlabel('arbitrary index')
plt.ylabel('error of ig-tilt (rad)')
plt.title('all conditions after removing dor0')
plt.show()

# for ii in np.arange(4):
#         for ll in np.arange(4):
#             plt.plot(TEMP2[ii,:,ll],ls='-',marker='o')
# plt.show()
            
for kk in np.arange(KK):
    plt.plot(TEMP2[:,kk].flatten(),marker='o',ls='-',label='Rytov={}'.format(Rytov[kk]))
plt.legend()
plt.xlabel('arbitrary index')
plt.ylabel('error of ig-tilt (rad)')
plt.title('effect of having varying Rytov after removing dor0')
plt.show()


plt.errorbar(Rytov,np.nanmean(TEMP2,axis=(0,2)),yerr=np.nanstd(TEMP2,axis=(0,2)),capsize=5,marker='o')
plt.xlabel('Rytov')
plt.ylabel('error of ig-tilt (rad)')
plt.title('after dor0, Rytov is the next primary variable')
plt.show()

Rytov_mean = np.nanmean(TEMP2,axis=(0,2))

# for ii in np.arange(II):
#     plt.plot(TEMP2[ii].flatten(),marker='o',ls='-',label='r0={}'.format(r0[ii]))
# plt.legend()
# plt.xlabel('arbitrary index')
# plt.ylabel('error of ig-tilt (rad)')
# plt.title('effect of having varying r0 after removing dor0')
# plt.show()


# for ll in np.arange(LL):
#     plt.plot(TEMP2[:,:,ll].flatten(),marker='o',ls='-',label='H_HA={}'.format(H_HA[ll]))
# plt.legend()
# plt.xlabel('arbitrary index')
# plt.ylabel('error of ig-tilt (rad)')
# plt.title('effect of having varying H_HA after removing dor0')
# plt.show()

TEMPMEAN = np.nanmean(TEMP2,axis=(0,2))
TEMP3 = np.copy(TEMP2)
for i in np.arange(4):
    TEMP3[:,i] = TEMP3[:,i] - TEMPMEAN[i]
    
TEMP4 = np.nanmean(TEMP3,axis=1)

plt.plot(TEMP4.flatten(),marker='o',ls='')
plt.xlabel('arbitrary index')
plt.ylabel('error of ig-tilt (rad)')
plt.title('all conditions after removing dor0 and Rytov')
plt.show()

for ii in np.arange(II):
    plt.plot(TEMP4[ii].flatten(),marker='o',ls='-',label='r0={}'.format(r0[ii]))
plt.legend()
plt.xlabel('arbitrary index')
plt.ylabel('error of ig-tilt (rad)')
plt.title('effect of having varying r0 after removing dor0 and rytov')
plt.show()

# for kk in np.arange(KK):
#     plt.plot(TEMP4[:,kk].flatten(),marker='o',ls='-',label='r0={}'.format(r0[ii]))
# plt.legend()
# plt.xlabel('arbitrary index')
# plt.ylabel('error of ig-tilt (rad)')
# plt.title('effect of having varying H_HA after removing dor0 and rytov')
# plt.show()

plt.errorbar(r0,np.nanmean(TEMP4,axis=(1)),yerr=np.nanstd(TEMP4,axis=(1)),capsize=5,marker='o')
plt.xlabel('r0')
plt.ylabel('error of ig-tilt (rad)')
plt.title('after dor0 and Rytov, r0 is the next primary variable')
plt.show()

r0_mean = np.nanmean(TEMP4,axis=(1))

# plt.errorbar(H_HA,np.nanmean(TEMP4,axis=(0)),yerr=np.nanstd(TEMP4,axis=(0)),capsize=5,marker='o')
# plt.xlabel('H_HA')
# plt.ylabel('error of ig-tilt (rad)')
# # plt.title('after dor0, Rytov is the primary variable')
# plt.show()



TEMPMEAN = np.nanmean(TEMP4,axis=(1))
TEMP5 = np.copy(TEMP4)
for i in np.arange(4):
    TEMP5[i] = TEMP5[i] - TEMPMEAN[i]
    
TEMP6 = np.nanmean(TEMP5,axis=0)

plt.plot(TEMP6.flatten(),marker='o',ls='')
plt.xlabel('arbitrary index')
plt.ylabel('error of ig-tilt (rad)')
plt.title('all conditions after removing dor0, Rytov, and r0')
plt.show()






ch2_temp = np.zeros([4],dtype=float)

observed = np.array([],dtype=float)
expected = np.array([],dtype=float)
for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                observed = np.append(observed,RMS_IG_FT[ii,jj,kk,ll])
                expected = np.append(expected,np.nanmean(RMS_IG_FT,axis=(1,2,3))[ii])
                
ch2_temp[0] = np.nansum((observed-expected)**2/expected)

observed = np.array([],dtype=float)
expected = np.array([],dtype=float)
for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                observed = np.append(observed,RMS_IG_FT[ii,jj,kk,ll])
                expected = np.append(expected,np.nanmean(RMS_IG_FT,axis=(0,2,3))[jj])
                
ch2_temp[1] = np.nansum((observed-expected)**2/expected)
            
observed = np.array([],dtype=float)
expected = np.array([],dtype=float)
for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                observed = np.append(observed,RMS_IG_FT[ii,jj,kk,ll])
                expected = np.append(expected,np.nanmean(RMS_IG_FT,axis=(0,1,3))[kk])
                
ch2_temp[2] = np.nansum((observed-expected)**2/expected)


observed = np.array([],dtype=float)
expected = np.array([],dtype=float)
for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                observed = np.append(observed,RMS_IG_FT[ii,jj,kk,ll])
                expected = np.append(expected,np.nanmean(RMS_IG_FT,axis=(0,1,2))[ll])
                
ch2_temp[3] = np.nansum((observed-expected)**2/expected)

ch2_temp[ch2_temp<=0] = np.nan
plt.plot(['r0','dor0','Rytov','H_HA'],ch2_temp,marker='o',ls='')
plt.hlines(0.46,0,3,label='probability=0.5')
plt.ylabel('chi2')
plt.title('chi2 test: 1linear variable dependency,\nch2 = 1 ==> null hypothesis is likely,\nch2 = 0 ==> the model is probably')
plt.legend()
plt.show()


