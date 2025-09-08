# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:06:32 2024

@author: gqmc62
"""

import pickle
import numpy as np
from matplotlib import pyplot as plt

CONVERT = 500/2/np.pi

def cov(x,y):
    tempx = x - np.nanmean(x)
    tempy = y - np.nanmean(y)
    temp = np.nanmean(tempx*tempy)
    return temp

def corr(x,y):
    temp2 = cov(x,y)/np.nanstd(x)/np.nanstd(y)
    return temp2

with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\Ultimate_input.pkl','rb') as file:
    input_data = pickle.load(file)
keys = input_data.keys()
for item_i in np.arange(len(keys)):
    key_name = list(keys)[item_i]
    data = input_data[key_name]
    globals()[key_name] = data
    
with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\Data\NoiselessCircularPupilSlopes-confident.pkl','rb') as file:
# with open(r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\NoiselessCircularPupilSlopes.pkl','rb') as file:
# with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\Data\Ultimate_output2.pkl','rb') as file:

    input_data = pickle.load(file)
keys = input_data.keys()
for item_i in np.arange(len(keys)):
    key_name = list(keys)[item_i]
    data = input_data[key_name]
    globals()[key_name] = data
    
Ig_SLOPE *= CONVERT
g_SLOPE *= CONVERT
SHWFS_SLOPE *= CONVERT

plt.plot(np.nanstd(Ig_SLOPE-SHWFS_SLOPE,axis=(-1,-2)).flatten())
plt.plot(np.nanstd(g_SLOPE-SHWFS_SLOPE,axis=(-1,-2)).flatten())
plt.show()
    
# A2 = np.copy(SHWFS_SLOPE)
# B2 = np.copy(g_SLOPE)
# C2 = np.copy(Ig_SLOPE)

# SHWFS_SLOPE noise free II,JJ,KK,LL,REPEAT,2
# g_SLOPE
# Ig_SLOPE
# II,JJ,KK,LL
# r0,dor0,Rytov,H_HA


RMS_IG_FT = np.nanstd(Ig_SLOPE - SHWFS_SLOPE,axis=(-1,-2))
RMS_G_FT = np.nanstd(g_SLOPE - SHWFS_SLOPE,axis=(-1,-2))


RMS_IG_FT[RMS_IG_FT==0] = np.nan
RMS_G_FT[RMS_G_FT==0] = np.nan
print(np.nanmax(RMS_IG_FT))

# plt.plot((RMS_IG_FT/RMS_G_FT)[POSSIBLE].flatten())
# plt.title('error of ig-tilt to error of g-tilt ratio')
# plt.ylabel('error of ig-tilt to error of g-tilt ratio')
# plt.xlabel('arbitrary index')r6mTb;y9i8'cdh;

# plt.show(

plt.plot(RMS_IG_FT.flatten(),marker='o',ls='-')
plt.xlabel('arbitrary index')
plt.ylabel('error of ig-tilt (nm)')
plt.title('all conditions')
plt.show()

# for ii in np.arange(II):
#     plt.plot(RMS_IG_FT[ii].flatten(),marker='o',ls='-',label='r0={}'.format(r0[ii]))
# plt.legend()
# plt.xlabel('arbitrary index')
# plt.ylabel('error of ig-tilt (nm)')
# plt.title('effect of having varying r0')
# plt.show()


for jj in np.arange(JJ):
    plt.plot(RMS_IG_FT[:,jj].flatten(),marker='o',ls='-',label='dor0={}'.format(dor0[jj]))
plt.legend()
plt.xlabel('arbitrary index')
plt.ylabel('error of ig-tilt (nm)')
plt.title('effect of having varying dor0')
plt.show()

plt.errorbar(dor0,np.nanmean(RMS_IG_FT,axis=(0,2,3)),yerr=np.nanstd(RMS_IG_FT,axis=(0,2,3)),capsize=5,marker='o')
plt.xlabel('dor0')
plt.ylabel('error of ig-tilt (nm)')
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
# plt.ylabel('error of ig-tilt (nm)')
# plt.title('effect of having varying Rytov')
# plt.show()

# for ll in np.arange(LL):
#     plt.plot(RMS_IG_FT[:,:,:,ll].flatten(),marker='o',ls='-',label='H_HA={}'.format(H_HA[ll]))
# plt.legend()
# plt.xlabel('arbitrary index')
# plt.ylabel('error of ig-tilt (nm)')
# plt.title('effect of having varying H_HA')
# plt.show()



# for jj in np.arange(JJ):
#     for ii in np.arange(II):
#         plt.plot(RMS_IG_FT[ii,jj].flatten(),marker='o',ls='',label='r0={}'.format(r0[ii]))
#     plt.legend()
#     plt.xlabel('arbitrary index')
#     plt.ylabel('error of ig-tilt (nm)')
#     plt.title('effect of having varying r0 with fixed dor0={}'.format(dor0[jj]))
#     plt.show()
    
# for jj in np.arange(JJ):
#     for kk in np.arange(KK):
#         plt.plot(RMS_IG_FT[:,jj,kk].flatten(),marker='o',ls='',label='Rytov={}'.format(Rytov[kk]))
#     plt.legend()
#     plt.xlabel('arbitrary index')
#     plt.ylabel('error of ig-tilt (nm)')
#     plt.title('effect of having varying Rytov with fixed dor0={}'.format(dor0[jj]))
#     plt.show()
    
# for jj in np.arange(JJ):
#     for ll in np.arange(LL):
#         plt.plot(RMS_IG_FT[:,jj,:,ll].flatten(),marker='o',ls='',label='H_HA={}'.format(Rytov[ll]))
#     plt.legend()
#     plt.xlabel('arbitrary index')
#     plt.ylabel('error of ig-tilt (nm)')
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
plt.ylabel('error of ig-tilt (nm)')
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
plt.ylabel('error of ig-tilt (nm)')
plt.title('effect of having varying Rytov after removing dor0')
plt.show()


plt.errorbar(Rytov,np.nanmean(TEMP2,axis=(0,2)),yerr=np.nanstd(TEMP2,axis=(0,2)),capsize=5,marker='o')
plt.xlabel('Rytov')
plt.ylabel('error of ig-tilt (nm)')
plt.title('after dor0, Rytov is the next primary variable')
plt.show()

Rytov_mean = np.nanmean(TEMP2,axis=(0,2))

# for ii in np.arange(II):
#     plt.plot(TEMP2[ii].flatten(),marker='o',ls='-',label='r0={}'.format(r0[ii]))
# plt.legend()
# plt.xlabel('arbitrary index')
# plt.ylabel('error of ig-tilt (nm)')
# plt.title('effect of having varying r0 after removing dor0')
# plt.show()


# for ll in np.arange(LL):
#     plt.plot(TEMP2[:,:,ll].flatten(),marker='o',ls='-',label='H_HA={}'.format(H_HA[ll]))
# plt.legend()
# plt.xlabel('arbitrary index')
# plt.ylabel('error of ig-tilt (nm)')
# plt.title('effect of having varying H_HA after removing dor0')
# plt.show()

TEMPMEAN = np.nanmean(TEMP2,axis=(0,2))
TEMP3 = np.copy(TEMP2)
for i in np.arange(4):
    TEMP3[:,i] = TEMP3[:,i] - TEMPMEAN[i]
    
TEMP4 = np.nanmean(TEMP3,axis=1)

plt.plot(TEMP4.flatten(),marker='o',ls='')
plt.xlabel('arbitrary index')
plt.ylabel('error of ig-tilt (nm)')
plt.title('all conditions after removing dor0 and Rytov')
plt.show()

for ii in np.arange(II):
    plt.plot(TEMP4[ii].flatten(),marker='o',ls='-',label='r0={}'.format(r0[ii]))
plt.legend()
plt.xlabel('arbitrary index')
plt.ylabel('error of ig-tilt (nm)')
plt.title('effect of having varying r0 after removing dor0 and rytov')
plt.show()

# for kk in np.arange(KK):
#     plt.plot(TEMP4[:,kk].flatten(),marker='o',ls='-',label='r0={}'.format(r0[ii]))
# plt.legend()
# plt.xlabel('arbitrary index')
# plt.ylabel('error of ig-tilt (nm)')
# plt.title('effect of having varying H_HA after removing dor0 and rytov')
# plt.show()

plt.errorbar(r0,np.nanmean(TEMP4,axis=(1)),yerr=np.nanstd(TEMP4,axis=(1)),capsize=5,marker='o')
plt.xlabel('r0')
plt.ylabel('error of ig-tilt (nm)')
plt.title('after dor0 and Rytov, r0 is the next primary variable')
plt.show()

r0_mean = np.nanmean(TEMP4,axis=(1))

# plt.errorbar(H_HA,np.nanmean(TEMP4,axis=(0)),yerr=np.nanstd(TEMP4,axis=(0)),capsize=5,marker='o')
# plt.xlabel('H_HA')
# plt.ylabel('error of ig-tilt (nm)')
# # plt.title('after dor0, Rytov is the primary variable')
# plt.show()



TEMPMEAN = np.nanmean(TEMP4,axis=(1))
TEMP5 = np.copy(TEMP4)
for i in np.arange(4):
    TEMP5[i] = TEMP5[i] - TEMPMEAN[i]
    
TEMP6 = np.nanmean(TEMP5,axis=0)

plt.plot(TEMP6.flatten(),marker='o',ls='')
plt.xlabel('arbitrary index')
plt.ylabel('error of ig-tilt (nm)')
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

################################################################################

# W = np.zeros_like(RMS_IG_FT)
# W[0] = np.log10(r0[0])
# W[1] = np.log10(r0[1])
# W[2] = np.log10(r0[2])
# W[3] = np.log10(r0[3])

# X = np.zeros_like(RMS_IG_FT)
# X[:,0] = np.log10(dor0[0])
# X[:,1] = np.log10(dor0[1])
# X[:,2] = np.log10(dor0[2])
# X[:,3] = np.log10(dor0[3])

# Y = np.zeros_like(RMS_IG_FT)
# Y[:,:,0] = np.log10(Rytov[0])
# Y[:,:,1] = np.log10(Rytov[1])
# Y[:,:,2] = np.log10(Rytov[2])
# Y[:,:,3] = np.log10(Rytov[3])

# Z = np.zeros_like(RMS_IG_FT)
# Z[:,:,:,0] = np.log10(H_HA[0])
# Z[:,:,:,1] = np.log10(H_HA[1])
# Z[:,:,:,2] = np.log10(H_HA[2])
# Z[:,:,:,3] = np.log10(H_HA[3])

# V = np.log10(RMS_IG_FT)

# plt.errorbar(10**V.flatten(),
#              10**V_calc.flatten(),
#              yerr=10**V_calc_error.flatten()*np.log(10),
#              ls='',marker='.',alpha=1,capsize=5)
# plt.plot([0,6],[0,6])
# plt.xlabel('measured IG-FT rms error (nm)')
# plt.ylabel('fitted IG-FT rms error (nm)')
# plt.show()

# plt.errorbar(10**V.flatten(),
#          10**V_calc.flatten()-10**V.flatten(),
#          yerr=10**V_calc_error.flatten()*np.log(10),
#          ls='',marker='.',alpha=1)
# plt.xlabel('measured IG-FT rms error (nm)')
# plt.ylabel('fitted - measured IG-FT rms error (nm)')
# plt.hlines([0],0,6)
# plt.show()

###############################################################################

W = np.zeros_like(RMS_IG_FT)
W[0] = r0[0]
W[1] = r0[1]
W[2] = r0[2]
W[3] = r0[3]

X = np.zeros_like(RMS_IG_FT)
X[:,0] = dor0[0]
X[:,1] = dor0[1]
X[:,2] = dor0[2]
X[:,3] = dor0[3]

Y = np.zeros_like(RMS_IG_FT)
Y[:,:,0] = Rytov[0]
Y[:,:,1] = Rytov[1]
Y[:,:,2] = Rytov[2]
Y[:,:,3] = Rytov[3]

Z = np.zeros_like(RMS_IG_FT)
Z[:,:,:,0] = H_HA[0]
Z[:,:,:,1] = H_HA[1]
Z[:,:,:,2] = H_HA[2]
Z[:,:,:,3] = H_HA[3]

V = RMS_IG_FT

A = np.column_stack((W[~np.isnan(V)].flatten(),
                     X[~np.isnan(V)].flatten(),
                     Y[~np.isnan(V)].flatten(),
                     Z[~np.isnan(V)].flatten(),
                     np.ones_like(W[~np.isnan(V)].flatten())))
coeffs, residuals, rank, s = np.linalg.lstsq(A,V[~np.isnan(V)].flatten(),rcond=None)
a_w, a_x, a_y, a_z, b = coeffs
a_w_e, a_x_e, a_y_e, a_z_e, b_e = np.sqrt(np.diag(
    residuals[0]/(A.shape[0] - A.shape[1]) 
    * np.linalg.inv(A.T@A)
    ))

with open('fit_error_result.txt','w') as f:
    f.write('a_r0={:.3f}+-{:.3f}nm/cm,\n'.format(a_w/100,a_w_e/100)
          + 'a_dor0={:.3f}+-{:.3f}nm,\n'.format(a_x,a_x_e)
          + 'a_rytov={:.3f}+-{:.3f}nm,\n'.format(a_y,a_y_e)
          + 'a_HHA={:.3f}+-{:.3f}nm/km,\n'.format(a_z*1000,a_z_e*1000)
          + 'b={:.3f}+-{:.3f}nm'.format(b,b_e))


def calc_IG_error(W,X,Y,Z):
    return W*a_w + X*a_x + Y*a_y + Z*a_z + b
def calc_IG_error_error(W,X,Y,Z):
    return W*a_w_e + X*a_x_e + Y*a_y_e + Z*a_z_e + b_e


V_calc = calc_IG_error(W,X,Y,Z)
V_calc_error = calc_IG_error_error(W,X,Y,Z)



plt.errorbar(V.flatten(),
             V_calc.flatten(),
             yerr=V_calc_error.flatten(),
             ls='',marker='.',alpha=0.5,capsize=5)
plt.plot([0,np.ceil(np.nanmax(V))],[0,np.ceil(np.nanmax(V))])
plt.xlabel('measured IG-FT rms error (nm)')
plt.ylabel('fitted IG-FT rms error (nm)')
# plt.savefig(,bbox_inches='tight#')
plt.show()

plt.errorbar(V.flatten(),
         V_calc.flatten()-V.flatten(),
         yerr=V_calc_error.flatten(),
         ls='',marker='.',alpha=0.5,capsize=5)
plt.xlabel('measured IG-FT rms error (nm)')
plt.ylabel('fitted - measured IG-FT rms error (nm)')
plt.hlines([0],0,np.ceil(np.nanmax(V)))
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_all_cross_sections(V, w, x, y, z):
    param_arrays = [w*100, x, y, z/1000]
    param_labels = ['r0 (cm)', 'd/r0', 'Rytov', 'H$_{HA}$ (km)']
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for axis in range(4):
        ax = axs[axis]
        varying_param = param_arrays[axis]
        other_axes = [i for i in range(4) if i != axis]
        other_ranges = [range(V.shape[i]) for i in other_axes]

        for idx_combo in itertools.product(*other_ranges):
            slicer = [slice(None)] * 4
            for a, val in zip(other_axes, idx_combo):
                slicer[a] = val
            v_line = V[tuple(slicer)]
            label = ', '.join(f"{param_labels[a]}={val}" for a, val in zip(other_axes, idx_combo))
            ax.plot(varying_param, v_line, alpha=0.5, label=label,marker='o')

        # ax.set_title(f"cros section : {param_labels[axis]}")
        ax.set_xlabel(f"{param_labels[axis]}")
        ax.set_ylabel("OPD (nm)")
        ax.set_ylim(0,np.ceil(np.nanmax(V)))
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)
        # if axis == 0:
        #     ax.legend(fontsize='x-small', loc='best')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.suptitle('Cross sections of measured RMS OPD difference between IG and FT (nm)')
    plt.show()
    return

plot_all_cross_sections(V, r0, dor0, Rytov, H_HA)
# print(a_w_e/a_w, a_x_e/a_x, a_y_e/a_y, a_z_e/a_z, b_e/b)

def plot_all_cross_sections(V, Ve, w, x, y, z):
    param_arrays = [w*100, x, y, z/1000]
    param_labels = ['r0 (cm)', 'd/r0', 'Rytov', 'H$_{HA}$ (km)']
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for axis in range(4):
        ax = axs[axis]
        varying_param = param_arrays[axis]
        other_axes = [i for i in range(4) if i != axis]
        other_ranges = [range(V.shape[i]) for i in other_axes]

        for idx_combo in itertools.product(*other_ranges):
            slicer = [slice(None)] * 4
            for a, val in zip(other_axes, idx_combo):
                slicer[a] = val
            v_line = V[tuple(slicer)]
            v_line_e = Ve[tuple(slicer)]
            # label = ', '.join(f"{param_labels[a]}={val}" for a, val in zip(other_axes, idx_combo))
            ax.errorbar(varying_param, v_line, yerr=v_line_e, alpha=0.5, capsize=5, marker='o')

        # ax.set_title(f"cross section : {param_labels[axis]}")
        ax.set_xlabel(f"{param_labels[axis]}")
        ax.set_ylabel("OPD (nm)")
        # ax.set_ylim(0,np.ceil(np.nanmax(V)))
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)
        # if axis == 0:
        #     ax.legend(fontsize='x-small', loc='best')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.suptitle('Cross sections of residual fit for RMS OPD difference between IG and FT (nm)')
    plt.show()
    return

plot_all_cross_sections(V-V_calc, V_calc_error, r0, dor0, Rytov, H_HA)
# print(a_w_e/a_w, a_x_e/a_x, a_y_e/a_y, a_z_e/a_z, b_e/b)

def plot_all_cross_sections(V, Ve, w, x, y, z):
    param_arrays = [w*100, x, y, z/1000]
    param_labels = ['r0 (cm)', 'd/r0', 'Rytov', 'H$_{HA}$ (km)']
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for axis in range(4):
        ax = axs[axis]
        varying_param = param_arrays[axis]
        other_axes = [i for i in range(4) if i != axis]
        other_ranges = [range(V.shape[i]) for i in other_axes]

        for idx_combo in itertools.product(*other_ranges):
            slicer = [slice(None)] * 4
            for a, val in zip(other_axes, idx_combo):
                slicer[a] = val
            v_line = V[tuple(slicer)]
            v_line_e = Ve[tuple(slicer)]
            # label = ', '.join(f"{param_labels[a]}={val}" for a, val in zip(other_axes, idx_combo))
            ax.errorbar(varying_param, v_line, yerr=v_line_e, alpha=0.5, capsize=5, marker='o')

        # ax.set_title(f"cross section : {param_labels[axis]}")
        ax.set_xlabel(f"{param_labels[axis]}")
        ax.set_ylabel("OPD (nm)")
        ax.set_ylim(0,np.ceil(np.nanmax(V)))
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)
        # if axis == 0:
        #     ax.legend(fontsize='x-small', loc='best')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.suptitle('Cross sections of best model for RMS OPD difference between IG and FT (nm)')
    plt.show()
    return

plot_all_cross_sections(V_calc, V_calc_error, r0, dor0, Rytov, H_HA)