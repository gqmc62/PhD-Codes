# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:07:18 2025

@author: gqmc62
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\Ultimate_input.pkl','rb') as file:
    input_data = pickle.load(file)
keys = input_data.keys()
for item_i in np.arange(len(keys)):
    key_name = list(keys)[item_i]
    data = input_data[key_name]
    globals()[key_name] = data

with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\NoiseSHWFSSlopes\processedData\new_NoiseVar.pkl', 'rb') as file:
    input_data = pickle.load(file)
file.close()

keys = input_data.keys()
for item_i in np.arange(len(keys)):
    key_name = list(keys)[item_i]
    data = input_data[key_name]
    globals()[key_name] = data
    

II = r0.shape[0]
JJ = dor0.shape[0]
KK = Rytov.shape[0]
LL = H_HA.shape[0]
MM = N_phot_list.shape[0]
NN = RON_error.shape[-1]
OO = THRESHOLD_rms.shape[-1]

W = 16/2
new_atm_error = (0.5*W**-2*(dor0)**(5./3.))**0.5
new_photon_error = (2*W/N_phot_list)**0.5
NEW_RON_list = np.round(NEW_RON_list,9)
final_RON = np.unique(NEW_RON_list)
final_RON_expect = (np.pi**2/3.*(final_RON)**2*(16**2)**2/(2)**2)**0.5

color_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']


# color_list = ['c','m','y','k']

# first_time = [True,True,True,True]
# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 if first_time[jj]:
#                     # plt.plot(N_phot_list,(NoiseRMS[ii,jj,kk,ll,:,0,0]).flatten(),
#                     plt.plot(N_phot_list,(photon_rms[ii,jj,kk,ll,:]).flatten(),
#                              color = color_list[jj],label='dor0={}'.format(dor0[jj]))
#                     # plt.hlines(new_atm_error[jj],
#                     #            N_phot_list.min(),
#                     #            N_phot_list.max(),
#                     #            label='atmos noise, dor0={}'.format(dor0[jj]),
#                     #            ls=':',color=color_list[jj])
#                     first_time[jj] = False
#                 else:
#                     plt.plot(N_phot_list,(photon_rms[ii,jj,kk,ll,:]).flatten(),
#                              color = color_list[jj])
# plt.plot(N_phot_list,new_photon_error,ls='--',label=('photon noise model'))
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Number of photon per lenslet')
# plt.ylabel('RMS-OPD (rad)')
# plt.title('Photon noise')
# plt.legend()
# plt.savefig('photon_noise.png')#/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/photon_noise.png')
# plt.show();plt.clf()

# # N_phot_list[mm], photon_rms[ii,jj,kk,ll,mm]

# color_list = ['c','m','y','k']




first_time = [True,True,True,True]
plt.plot(N_phot_list,new_photon_error,ls='--',label=('photon noise model'))
for jj in np.arange(JJ):
    plt.errorbar(N_phot_list,np.nanmean(photon_rms[:,jj],axis=(0,1,2)),
                 yerr=np.nanstd(photon_rms[:,jj],axis=(0,1,2)),
                 color = color_list[jj],label='dor0={}'.format(dor0[jj]),
                 capsize=5,marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of photon per lenslet')
plt.ylabel('RMS-OPD (rad)')
plt.title('Photon noise')
plt.legend()
plt.savefig('photon_noise.png')#/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/photon_noise.png')
plt.show();plt.clf()

################################################################################

# temp = np.zeros((4,4,4,5,5),dtype=float)
# for i in np.arange(5):
#     # temp[:,:,:,:,i] = (THRESHOLD_rms[:,-1,:,:,:,0,i]**2 - THRESHOLD_rms[:,-1,:,:,:,0,0]**2)**0.5
#     temp[:,:,:,:,i] = THRESHOLD_rms[:,-1,:,:,:,0,i]

# mean = np.nanmean(temp**2,axis=(0,1,2))**0.5
# error = np.nanstd(temp**2,axis=(0,1,2))**0.5

# plt.plot(N_phot_list,photon_error,ls='--',label=('photon noise model'))
# for i in np.arange(5):
#     plt.errorbar(N_phot_list,mean[:,i],yerr=error[:,i],capsize=5,marker='o',label='norm_Threshold={:.0e}'.format(Threshold_list[i]))
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.show()


##################################################################################

# RON_rms = np.zeros((II,JJ,KK,LL,MM,NN),dtype=float)

# NEW_RON_list = np.round(NEW_RON_list,9)

# first_time = [True,True,True,True]
# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 for mm in np.arange(MM):
#                     for nn in np.arange(NN):
#                         if nn == 0:
#                             continue
#                         # RON_rms[ii,jj,kk,ll,mm,nn] = ((NoiseRMS[ii,jj,kk,ll, mm,nn,0]**2
#                         #                - NoiseRMS[ii,jj,kk,ll, mm,0,0]**2)**0.5)
#                         # RON_rms[ii,jj,kk,ll,mm,nn] = NoiseRMS[ii,jj,kk,ll, mm,nn,0]
#                         if first_time[jj]:
#                             # print(NEW_RON_list[mm,nn])
#                             # print(RON_rms[ii,jj,kk,ll,mm,nn])
#                             plt.plot(NEW_RON_list[mm,nn],
#                                      RON_rms[ii,jj,kk,ll,mm,nn].flatten(),
#                                      color = color_list[jj],label='dor0={}'.format(dor0[jj]), marker='.')
#                             first_time[jj] = False
#                             # plt.hlines(new_atm_error[jj],
#                             #            RON_list[1:].min(),
#                             #            RON_list[1:].max(),
#                             #            label='atmos noise, dor0={}'.format(dor0[jj]),
#                             #            ls=':',color=color_list[jj])
#                         else:
#                             plt.plot(NEW_RON_list[mm,nn],
#                                      RON_rms[ii,jj,kk,ll,mm,nn].flatten(),
#                                      color = color_list[jj], marker='.')
                            
# plt.plot(NEW_RON_list[:,1:].flatten(),RON_error[:,1:].flatten(),ls='--',label=('ron noise model'))
# plt.legend()
# plt.ylabel('RMS-OPD (rad)')
# plt.xlabel('Readout e- std per pixel compare to total photon in a lenslet')
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('readout_noise.png')#/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/readout_noise.png')
# plt.show();plt.clf()




final_RON_rms = np.zeros_like(final_RON)
final_RON_rms_count = np.zeros_like(final_RON)
final_RON_rms_error = np.zeros_like(final_RON)

for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                for mm in np.arange(MM):
                    for nn in np.arange(NN):
                        if ~np.isnan(RON_rms[ii,jj,kk,ll,mm,nn]):
                            final_RON_rms[final_RON==NEW_RON_list[mm,nn]] += RON_rms[ii,jj,kk,ll,mm,nn]**2
                            final_RON_rms_count[final_RON==NEW_RON_list[mm,nn]] += 1
final_RON_rms = (final_RON_rms/final_RON_rms_count)**0.5

for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                for mm in np.arange(MM):
                    for nn in np.arange(NN):
                        if ~np.isnan(RON_rms[ii,jj,kk,ll,mm,nn]):
                            final_RON_rms_error[final_RON==NEW_RON_list[mm,nn]] += (RON_rms[ii,jj,kk,ll,mm,nn] - final_RON_rms[final_RON==NEW_RON_list[mm,nn]])**2

final_RON_rms_error = (final_RON_rms_error/final_RON_rms_count)**0.5
# final_RON_rms_error /= final_RON_rms_count**.5




plt.plot(final_RON,final_RON_expect,ls='--',label=('ron model'))
plt.errorbar(final_RON[1:],
             final_RON_rms[1:],
             yerr=final_RON_rms_error[1:],
             ls='-',label=('ron'),capsize=5,marker='o')
plt.legend()
plt.ylabel('RMS-OPD (rad)')
plt.xlabel('ron e-/pxl / total photon in subap')
plt.xscale('log')
plt.yscale('log')
plt.savefig('readout_noise.png')#/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/readout_noise.png')
plt.show();plt.clf()


# Threshold_list = np.array([0,0.0001,0.1,0.3,0.5],dtype=float)
# # THRESHOLD_rms = np.zeros((II,JJ,KK,LL,MM,NN,OO),dtype=float)
# first_time = [True,True,True,True]
# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 for mm in np.arange(MM):
#                     for nn in np.arange(NN):
#                         for oo in np.arange(OO):
#                             if oo == 0:
#                                 continue
#                             # if oo == 1:
#                             #     continue
#                             # print(NoiseRMS[ii,jj,kk,ll, mm,nn,oo].shape)
#                             # print(NoiseRMS[ii,jj,kk,ll, mm,nn,0].shape)
#                             # THRESHOLD_rms[ii,jj,kk,ll,mm,nn,oo] = ((NoiseRMS[ii,jj,kk,ll, mm,nn,oo]**2
#                             #               - NoiseRMS[ii,jj,kk,ll, mm,nn,0]**2)**0.5)
#                             qq = jj
#                             if first_time[qq]:
#                                 plt.plot(Threshold_list[oo],
#                                          THRESHOLD_rms[ii,jj,kk,ll,mm,nn,oo].flatten(),
#                                         color = color_list[qq],label='dor0={}'.format(dor0[qq]), marker='.')
#                                 first_time[qq] = False
#                                 # plt.hlines(atm_error[jj],
#                                 #            Threshold_list[1:].min(),
#                                 #            Threshold_list[1:].max(),
#                                 #            label='atmos noise, dor0={}'.format(dor0[jj]),
#                                 #            ls=':',color=color_list[jj])
#                             else:
#                                 plt.plot(Threshold_list[oo],
#                                          THRESHOLD_rms[ii,jj,kk,ll,mm,nn,oo].flatten(),
#                                         color = color_list[qq], marker='.')
# #plt.plot(RON_list[1:],RON_error[1:],ls='--',label=('ron noise model'))
# plt.legend()
# plt.ylabel('RMS-OPD (rad)')
# plt.xlabel('Threshold level (relative to peak intensity)')#'Readout e- std per pixel compare to total photon in a lenslet')
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('threshold.png')#/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/readout_noise.png')
# plt.show();plt.clf()

# Threshold_list = np.array([0,0.0001,0.1,0.3,0.5],dtype=float)
# plt.errorbar(Threshold_list[1:],
#          np.nanmean(THRESHOLD_rms[...,1:]**2,axis=(0,1,2,3,4,5))**0.5,
#          np.nanstd(THRESHOLD_rms[...,1:]**2,axis=(0,1,2,3,4,5))**0.5,#/(np.sum(np.isnan(THRESHOLD_rms[...,2:]),axis=(0,1,2,3,4,5)))**0.5,
#          marker='o',capsize=5)
# #plt.plot(RON_list[1:],RON_error[1:],ls='--',label=('ron noise model'))
# # plt.legend()
# plt.ylabel('RMS-OPD (rad)')
# plt.xlabel('Threshold level (relative to peak intensity)')#'Readout e- std per pixel compare to total photon in a lenslet')
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('threshold.png')#/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/readout_noise.png')
# plt.show();plt.clf()

Threshold_list = np.array([0,0.0001,0.1,0.3,0.5],dtype=float)
final_threshold_rms = np.zeros((final_RON.shape[0],Threshold_list.shape[0]),dtype=float)
final_threshold_rms_count = np.copy(final_threshold_rms)
final_threshold_rms_error = np.copy(final_threshold_rms)


for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                for mm in np.arange(MM):
                    for nn in np.arange(NN):
                        for oo in np.arange(OO):
                            if ~np.isnan(THRESHOLD_rms[ii,jj,kk,ll,mm,nn,oo]):
                                final_threshold_rms[final_RON==NEW_RON_list[mm,nn],oo] += THRESHOLD_rms[ii,jj,kk,ll,mm,nn,oo]**2
                                final_threshold_rms_count[final_RON==NEW_RON_list[mm,nn],oo] += 1

final_threshold_rms = (final_threshold_rms/final_threshold_rms_count)**0.5

for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                for mm in np.arange(MM):
                    for nn in np.arange(NN):
                        for oo in np.arange(OO):
                            if ~np.isnan(THRESHOLD_rms[ii,jj,kk,ll,mm,nn,oo]):
                                final_threshold_rms_error[final_RON==NEW_RON_list[mm,nn],oo] += (THRESHOLD_rms[ii,jj,kk,ll,mm,nn,oo] - final_threshold_rms[final_RON==NEW_RON_list[mm,nn],oo])**2

final_threshold_rms_error = (final_threshold_rms/final_threshold_rms_count)**0.5

plt.plot(final_RON,final_RON_expect,ls='--',label=('ron model no threshold'))
plt.errorbar(final_RON[1:],
             final_RON_rms[1:],
             yerr=final_RON_rms_error[1:],
             ls='-',label=('norm_threshold=0'),capsize=5,marker='o')
for i in np.arange(Threshold_list[1:].shape[0]):
    # if i%2 != 0:
    #     continue
    if i == 0:
        plt.errorbar(final_RON[:],
                     final_threshold_rms[:,i+1],
                     yerr=final_threshold_rms_error[:,i+1],
                     marker='o',capsize=5,
                     label='norm_threshold={:.0e}'.format(Threshold_list[1:][i]))
    else:
        plt.errorbar(final_RON,
                     final_threshold_rms[:,i+1],
                     yerr=final_threshold_rms_error[:,i+1],
                     marker='o',capsize=5,
                     label='norm_threshold={:.0e}'.format(Threshold_list[1:][i]))
plt.legend()
plt.ylabel('RMS-OPD (rad)')
plt.xlabel('ron e-/pxl / total photon in subap')#'Readout e- std per pixel compare to total photon in a lenslet')
plt.xscale('log')
plt.yscale('log')
plt.savefig('threshold.png')#/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/readout_noise.png')
plt.show();plt.clf()

Another_threshold_old = np.copy(Another_threshold)
Another_threshold = NoiseRMS[:,:,:,:,:,0,:]
Another_threshold[Another_threshold==0] = np.nan #(phot + threshold) - (phot + no threshold) all with no ron

# first_time=[True,True,True,True,True]
# # color_list = ['no','g','b','c','m']
# plt.plot(N_phot_list,new_photon_error,ls='--',label=('photon noise model'),color=color_list[0])
# for jj in np.arange(JJ):

#     final_phot_threshold = np.zeros((MM,OO),dtype=float)
#     final_phot_threshold_count = np.zeros((MM,OO),dtype=float)
#     final_phot_threshold_error = np.zeros((MM,OO),dtype=float)
    
#     for ii in np.arange(II):
#         # for jj in np.arange(JJ):
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 for mm in np.arange(MM):
#                     for oo in np.arange(OO):
#                         if ~np.isnan(Another_threshold[ii,jj,kk,ll,mm,oo]):
#                             final_phot_threshold[mm,oo] += Another_threshold[ii,jj,kk,ll,mm,oo]**2
#                             final_phot_threshold_count[mm,oo] += 1
#     final_phot_threshold = (final_phot_threshold/ final_phot_threshold_count)**0.5
    
#     for ii in np.arange(II):
#         # for jj in np.arange(JJ):
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 for mm in np.arange(MM):
#                     for oo in np.arange(OO):
#                         if ~np.isnan(Another_threshold[ii,jj,kk,ll,mm,oo]):
#                             final_phot_threshold_error[mm,oo] += (Another_threshold[ii,jj,kk,ll,mm,oo] - final_phot_threshold[mm,oo])**2
#     final_phot_threshold_error = (final_phot_threshold_error/final_phot_threshold_count)**0.5
    
    
#     for oo in np.arange(OO):
#         # if oo == 0:
#         #     continue
#         if first_time[oo] == True:
#             plt.errorbar(N_phot_list,
#                          final_phot_threshold[:,oo],
#                          yerr=final_phot_threshold_error[:,oo],
#                          label='threshold={}'.format(Threshold_list[oo]),color=color_list[oo+1],
#                          capsize=5,marker='o')
#             first_time[oo] = False
#         else:
#             plt.errorbar(N_phot_list,
#                          final_phot_threshold[:,oo],
#                          yerr=final_phot_threshold_error[:,oo],
#                          #label='threshold={}'.format(Threshold_list[oo]),
#                          color=color_list[oo+1],
#                          capsize=5,marker='o')
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.title('Effect of threshold on photon noise')
# plt.savefig('threshold-phot.png')
# plt.show();plt.clf()


# color_list = ['no','g','b','c','m']
fig, axs = plt.subplots(2,2,figsize=(7.5,7.5))
MAX = 2
MIN = 1e-3
plt.suptitle('Effect of threshold on photon noise')
# plt.plot(N_phot_list,new_photon_error,ls='--',label=('photon noise model'),color=color_list[0])
for jj in np.arange(JJ):

    final_phot_threshold = np.zeros((MM,OO),dtype=float)
    final_phot_threshold_count = np.zeros((MM,OO),dtype=float)
    final_phot_threshold_error = np.zeros((MM,OO),dtype=float)
    
    for ii in np.arange(II):
        # for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                for mm in np.arange(MM):
                    for oo in np.arange(OO):
                        if ~np.isnan(Another_threshold[ii,jj,kk,ll,mm,oo]):
                            final_phot_threshold[mm,oo] += Another_threshold[ii,jj,kk,ll,mm,oo]**2
                            final_phot_threshold_count[mm,oo] += 1
    final_phot_threshold = (final_phot_threshold/ final_phot_threshold_count)**0.5
    
    for ii in np.arange(II):
        # for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                for mm in np.arange(MM):
                    for oo in np.arange(OO):
                        if ~np.isnan(Another_threshold[ii,jj,kk,ll,mm,oo]):
                            final_phot_threshold_error[mm,oo] += (Another_threshold[ii,jj,kk,ll,mm,oo] - final_phot_threshold[mm,oo])**2
    final_phot_threshold_error = (final_phot_threshold_error/final_phot_threshold_count)**0.5
    
    axs[jj//2,jj%2].plot(N_phot_list,new_photon_error,ls='--',label=('photon noise model'),color=color_list[0])
    for oo in np.arange(OO):
        axs[jj//2,jj%2].errorbar(N_phot_list,
                     final_phot_threshold[:,oo],
                     yerr=final_phot_threshold_error[:,oo],
                     label='threshold={}'.format(Threshold_list[oo]),color=color_list[oo+1],
                     capsize=5,marker='o')
    axs[jj//2,jj%2].set_title('dor0={}'.format(dor0[jj]))
        
    axs[jj//2,jj%2].legend()
    axs[jj//2,jj%2].set_xscale('log')
    axs[jj//2,jj%2].set_yscale('log')
    axs[jj//2,jj%2].set_ylim(MIN,MAX)
    if jj//2 == 1:
        axs[jj//2,jj%2].set_xlabel('N_ph')
    if jj%2 == 0:
        axs[jj//2,jj%2].set_ylabel('RMS-OPD (rad)')

plt.tight_layout
plt.savefig('threshold-phot.png')
plt.show();plt.clf()



