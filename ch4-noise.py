# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:58:36 2024

@author: gqmc62
"""

import pickle
import numpy as np
from matplotlib import pyplot as plt

from scipy import stats

TEST_GAUSS = False
PLOT = True

nbins = 20

def gauss(x, mean, std, area):
    y = area*np.exp(-((x - mean)/std)**2/2.)/(std*np.sqrt(2*np.pi))*(x[1] - x[0])
    return y

with open('/mt/cfai/pkongkaew/IGPhaseScreen/Ultimate_input.pkl','rb') as file:
    input_data = pickle.load(file)
keys = input_data.keys()
for item_i in np.arange(len(keys)):
    key_name = list(keys)[item_i]
    data = input_data[key_name]
    globals()[key_name] = data
del input_data
    
# # with open('NEWRON_PhotonNoiseNoThresholdCircularPupilSlopes.pkl','rb') as file:
# #     input_data = pickle.load(file)
# # keys = input_data.keys()
# # for item_i in np.arange(len(keys)):
# #     key_name = list(keys)[item_i]
# #     data = input_data[key_name]
# #     globals()[key_name] = data

with open('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/Slopes/SHWFSNoiseScintillation0000.pkl', 'rb') as file:
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
NN = RON_list.shape[0]
OO = Threshold_list.shape[0]

# COUNT1 = input_data['NoNoiseSHWFSTilt'].shape[-2]
# COUNT2 = input_data['SHWFSTilt'].shape[-1]
COUNT1 = input_data['Noise'].shape[-3]
COUNT2 = input_data['Noise'].shape[-1]
del input_data    
# NoNoiseSHWFSTilt = np.zeros((II,JJ,KK,LL,
#                               COUNT1,2),dtype=float)

# SHWFSTilt = np.zeros((II,JJ,KK,LL,MM,NN,OO,
#                       COUNT1,2,COUNT2),dtype=float)

Noise = np.zeros((II,JJ,KK,LL,MM,NN,OO,
                  COUNT1,2,COUNT2),dtype=float)

for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                try:
                    with open('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/Slopes/SHWFSNoiseScintillation{:d}{:d}{:d}{:d}.pkl'.format(ii,jj,kk,ll), 'rb') as file:
                        input_data = pickle.load(file)
                    # NoNoiseSHWFSTilt[ii,jj,kk,ll] = input_data['NoNoiseSHWFSTilt']
                    # SHWFSTilt[ii,jj,kk,ll] = input_data['SHWFSTilt']
                    Noise[ii,jj,kk,ll] = input_data['Noise']
                    file.close()
                except:
                    # NoNoiseSHWFSTilt[ii,jj,kk,ll,:] = np.nan
                    # SHWFSTilt[ii,jj,kk,ll,:] = np.nan
                    Noise[ii,jj,kk,ll] = np.nan
del input_data

# II,JJ,KK,LL,MM,NN,OO
# r0,dor0,Rytov,H_HA,N_phot_list,RON_list,Threshold_list


# SHWFSTilt noisy wfs
# NoNoiseSHWFSTilt no noise wfs

# NoNoise = NoNoiseSHWFSTilt
# WithNoise = SHWFSTilt

# NoNoise[NoNoise == 0] = np.nan
# WithNoise[WithNoise == 0] = np.nan

NoiseRMS = np.nanstd(Noise,axis=(-3,-2,-1))
NoiseRMS[NoiseRMS==0] = np.nan
ROI = ((~np.isnan(NoiseRMS[...,0,0,0])))

# Noise = np.zeros_like(WithNoise)
# for mm in np.arange(MM):
#     for nn in np.arange(NN):
#         for oo in np.arange(OO):
#             for item2 in np.arange(COUNT2):
#                 Noise[:,:,:,:,mm,nn,oo,:,:,item2] = (WithNoise[:,:,:,:,mm,nn,oo,:,:,item2]
#                                                       - NoNoiseSHWFSTilt)
# NoiseRMS = np.nanstd(Noise,axis=(-1,-2,-3))
# NoiseRMS[NoiseRMS==0] = np.nan

# NoiseRMS = np.zeros((II,JJ,KK,LL,MM,NN,OO),dtype=float)
# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 for mm in np.arange(MM):
#                     for nn in np.arange(NN):
#                         for oo in np.arange(OO):
#                             temp = []
#                             for item2 in np.arange(COUNT2):
#                                 temp.append((WithNoise[ii,jj,kk,ll,mm,nn,oo,:,:,item2]
#                                   - NoNoiseSHWFSTilt[ii,jj,kk,ll]))
#                             NoiseRMS[ii,jj,kk,ll,mm,nn,oo] = np.nanstd(np.asarray(temp), axis=(-1,-2,-3))

# NoiseRMS[NoiseRMS==0] = np.nan


# ROI = ((~np.isnan(NoiseRMS[...,0,0,0])))


# LABEL_r0 = np.zeros((II), dtype=object)
# for ii in np.arange(II):
#     LABEL_r0[ii] = 'r0={:.0f}cm'.format(r0[ii]*100)

# LABEL_DOR0 = np.zeros((JJ), dtype=object)
# for jj in np.arange(JJ):
#     LABEL_DOR0[jj] = 'd/r0={:.1f}'.format(dor0[jj])
    
# LABEL_RYTOV = np.zeros((KK), dtype=object)
# for kk in np.arange(KK):
#     LABEL_RYTOV[kk] = '$\sigma^2_\chi=${:.2f}'.format(Rytov[kk])
    
# LABEL_HHA = np.zeros((LL), dtype=object)
# for ll in np.arange(LL):
#     LABEL_HHA[ll] = 'HA={:.0f}km'.format(H_HA[ll]/1000)

# LABEL_N_phot = np.zeros((MM), dtype=object)
# for mm in np.arange(MM):
#     LABEL_N_phot[mm] = 'N_phot={:.0f}'.format(N_phot_list[mm])

# LABEL_RON = np.zeros((NN), dtype=object)
# for nn in np.arange(NN):
#     LABEL_RON[nn] = 'RON={:.2f}'.format(RON_list[nn])

# LABEL_Threshold = np.zeros((OO), dtype=object)
# for oo in np.arange(OO):
#     LABEL_Threshold[oo] = 'Threshold={:.2f}'.format(Threshold_list[oo])


# color_map = plt.cm.get_cmap('inferno')
# reversed_color_map = color_map.reversed()
# plt.subplots(4,4,sharex=True,sharey=True,squeeze=False)
# # plt.subplot(4,4,1)
# plt.suptitle('Noisy SHWFS has Gauss Distr? 1 = True, 0 = False'
#               + '\n{:.0f}Nphot, {:.2f}RO, threshold={:.2e}'.format(N_phot_list[mm],RON_list[nn],Threshold_list[oo]))
# for kk in np.arange(KK):
#     for ll in np.arange(LL):
#         plt.subplot(4,4,LL*(KK - kk - 1) + ll + 1)
#         plt.pcolor(r0*100, dor0, ROI[...,kk,ll].T,
#                     #vmin=0,vmax=1,
#                     cmap=reversed_color_map,edgecolors='k',linewidths=0.5)
#         # plt.colorbar()
#         # plt.xlabel('r0 (cm)')
#         # plt.ylabel('d/r0')
#         # plt.title(LABEL_RYTOV[kk] + ', ' + LABEL_HHA[ll])
#         if kk == 0:
#             plt.xlabel('r0 (cm)\n' + LABEL_HHA[ll])
#         if ll == 0:
#             plt.ylabel(LABEL_RYTOV[kk] + '\nd/r0')
#         if ll == (LL - 1):
#             plt.colorbar()
# # plt.subplot(4,4,1)
# # plt.colorbar()
# # plt.xlabel('r0 (cm)')
# # plt.ylabel('d/r0')
# plt.show();plt.clf()
# plt.clf()


# # No Noise distribution
# plt.plot(NoNoise[:,:,:,:,-1,0,0,:,:].flatten(),ls='',marker=',',alpha=0.1)
# plt.title('NoNoiseFTSHWFS')
# plt.xlabel('index')
# plt.ylabel('OPD (rad)')
# plt.show();plt.clf()

# # With Noise distribution
# plt.plot(WithNoise.flatten(),ls='',marker=',',alpha=0.1)
# plt.title('NoisyFTSHWFS')
# plt.xlabel('index')
# plt.ylabel('OPD (rad)')
# plt.show();plt.clf()

# # Noise distribution
# plt.plot(Noise.flatten(),ls='',marker=',',alpha=0.1)
# plt.title('SHWFS Noise')
# plt.xlabel('index')
# plt.ylabel('OPD (rad)')
# plt.show();plt.clf()

# # Noise level
# plt.plot(NoiseRMS.flatten(),ls='',marker=',',alpha=0.5)
# plt.title('SHWFS Noise RMS')
# plt.xlabel('index')
# plt.ylabel('OPD (rad)')
# plt.show();plt.clf()

if TEST_GAUSS == True:

    # GaussNoNoise = np.zeros((II,JJ,KK,LL),dtype=float)
    # GaussNoNoise[:] = np.nan
    # error_count = 0
    # # No Noise distribution
    # good_count = 0
    # for ii in np.arange(II):
    #     for jj in np.arange(JJ):
    #         for kk in np.arange(KK):
    #             for ll in np.arange(LL):
    #                 # print(ii,jj,kk,ll)
    #                 noise = NoNoise[ii,jj,kk,ll].flatten()
    #                 noise_temp = noise[np.isnan(noise) == False].flatten()
    #                 noise = noise_temp[noise_temp!=0]

    #                 if noise.size == 0:
    #                     continue 

    #                 noise -= np.nanmean(noise)
    #                 expected_std = np.std(noise)
                    
    #                 # plt.plot(noise)
    #                 # plt.show();plt.clf()
                    
    #                 if np.isnan(expected_std) or (expected_std == 0):
    #                     continue
                    
    #                 # plt.hist(noise.flatten())
    #                 x = np.linspace(np.nanmin(noise), np.nanmax(noise),num=nbins)
    #                 area = np.sum(np.isnan(noise) == False)
                    
    #                 counts, bins = np.histogram(noise[np.isnan(noise)==False].flatten(),
    #                                             bins=nbins)
                    
    #                 y = gauss(x,0,expected_std,area)
                    
    #                 # fitting_index = np.sum(np.abs(y - counts))/area
                    
    #                 # ks method # prove they are both from the same distribution
    #                 res = stats.kstest(noise,np.random.normal(0,expected_std,area))
    #                 valid0 = (res.pvalue < 0.05)
                    
    #                 # res = stats.goodness_of_fit(stats.norm,noise)
    #                 # valid = (res.pvalue < 0.05)
                    
    #                 # # puttiwat's? method # prove binning statistics if gaussian or not
    #                 bincount = np.nansum(counts!=0)
    #                 percentage = (stats.norm.cdf(1)-stats.norm.cdf(-1))
    #                 expectedtobein = bincount*percentage - 2.*np.sqrt(bincount*percentage)
    #                 valid1 = ((np.nansum(((counts[counts!=0] - y[counts!=0])**2/counts[counts!=0]) < 1) / bincount) > expectedtobein/bincount)
                    
    #                 # show if distribution difference to gaussian is less than 10%
    #                 valid2 = ((np.sum(np.abs((counts - y)*(bins[1]-bins[0])/area)) < 0.1))

    #                 valid = (valid0 or valid1 or valid2)
                    
    #                 GaussNoNoise[ii,jj,kk,ll] = valid
                    
    #                 if (valid == False):
    #                     error_count += 1
    #                     if PLOT == True:
    #                         if error_count <= 10:
    #                             plt.plot(x,y,label='noise equivalent angle model',color='k')
    #                             plt.stairs(counts,bins,alpha=0.5,color='k',fill=True)
    #                             plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
    #                             plt.title('No threshold = standard ; WFS slopes threshold error (rad)'
    #                                     + '\n{:.0f}r0, {:.1f}dor0, {:.2f}Rytov, {:.0f}H_HA'.format(r0[ii]*100,dor0[jj],Rytov[kk],H_HA[ll]/1000)
    #                                     #+ '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
    #                                     + '\nvalid = {:}'.format(valid))
    #                             LIM = np.max(np.abs([x.min(),x.max()]))
    #                             plt.xlim(-LIM,LIM)
    #                             plt.savefig('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/notGaussian/no_noise_error{:d}.png'.format(error_count))
    #                             plt.show();plt.clf()
                        
    #                 if PLOT == True:
    #                     if (valid == True) and (good_count <= 10):
    #                         plt.plot(x,y,label='noise equivalent angle model',color='k')
    #                         plt.stairs(counts,bins,alpha=0.5,color='k',fill=True)
    #                         plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
    #                         plt.title('No threshold = standard ; WFS slopes threshold error (rad)'
    #                                 + '\n{:.0f}r0, {:.1f}dor0, {:.2f}Rytov, {:.0f}H_HA'.format(r0[ii]*100,dor0[jj],Rytov[kk],H_HA[ll]/1000)
    #                                 #+ '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
    #                                 + '\nvalid = {:}'.format(valid))
    #                         LIM = np.max(np.abs([x.min(),x.max()]))
    #                         plt.xlim(-LIM,LIM)
    #                         plt.savefig('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/Gaussian/no_noise_error{:d}{:d}{:d}{:d}.png'.format(ii,jj,kk,ll))
    #                         plt.show();plt.clf()
    #                         good_count += 1
                    
                    

                    

    # with open('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/output.txt','w') as file:
    #     print('No noise fail/valid = {:d}/{:d}'.format(np.nansum(GaussNoNoise==False),np.nansum(np.isnan(GaussNoNoise) == False)),file=file)
    #     print('No noise no threshold fail/valid = {:d}/{:d}'.format(np.nansum(GaussNoNoise[...,0]==False),np.nansum(np.isnan(GaussNoNoise[...,0]) == False)),file=file)
    # with open('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/GaussNoNoise.pkl', 'wb') as file:
    #     pickle.dump(GaussNoNoise,file=file)
    # file.close()


    # error_count = 0

    # GaussWithNoise = np.zeros((II,JJ,KK,LL,MM,NN,OO),dtype=float)
    # GaussWithNoise[:] = np.nan
    # # With Noise distribution
    # good_count = 0
    # for ii in np.arange(II):
    #     for jj in np.arange(JJ):
    #         for kk in np.arange(KK):
    #             for ll in np.arange(LL):
    #                 # print(ii,jj,kk,ll)
    #                 for mm in np.arange(MM):
    #                     for nn in np.arange(NN):
    #                         for oo in np.arange(OO):
    #                             # if mm != 2:
    #                             #     continue
    #                             # if nn != 0 :
    #                             #     continue
    #                             # if oo != 0 :
    #                             #     continue
                                
    #                             noise = WithNoise[ii,jj,kk,ll,mm,nn,oo].flatten()
    #                             noise_temp = noise[np.isnan(noise) == False].flatten()
    #                             noise = noise_temp[noise_temp!=0]

    #                             if noise.size == 0:
    #                                 continue 

    #                             noise -= np.nanmean(noise)
    #                             expected_std = np.std(noise)
                                
    #                             # plt.plot(noise)
    #                             # plt.show();plt.clf()
                                
    #                             if np.isnan(expected_std) or (expected_std == 0):
    #                                 continue
                                
    #                             # plt.hist(noise.flatten())
    #                             x = np.linspace(np.nanmin(noise), np.nanmax(noise),num=nbins)
    #                             area = np.sum(np.isnan(noise) == False)
                                
    #                             counts, bins = np.histogram(noise[np.isnan(noise)==False].flatten(),
    #                                                         bins=nbins)
                                
    #                             y = gauss(x,0,expected_std,area)
                            
    #                             # fitting_index = np.sum(np.abs(y - counts))/area
                                
    #                             # ks method # prove they are both from the same distribution
    #                             res = stats.kstest(noise,np.random.normal(0,expected_std,area))
    #                             valid0 = (res.pvalue < 0.05)
                                
    #                             # res = stats.goodness_of_fit(stats.norm,noise)
    #                             # valid = (res.pvalue < 0.05)
                                
    #                             # # puttiwat's? method # prove binning statistics if gaussian or not
    #                             bincount = np.nansum(counts!=0)
    #                             percentage = (stats.norm.cdf(1)-stats.norm.cdf(-1))
    #                             expectedtobein = bincount*percentage - 2.*np.sqrt(bincount*percentage)
    #                             valid1 = ((np.nansum(((counts[counts!=0] - y[counts!=0])**2/counts[counts!=0]) < 1) / bincount) > expectedtobein/bincount)
                                
    #                             # show if distribution difference to gaussian is less than 10%
    #                             valid2 = ((np.sum(np.abs((counts - y)*(bins[1]-bins[0])/area)) < 0.1))

    #                             valid = (valid0 or valid1 or valid2)

    #                             GaussWithNoise[ii,jj,kk,ll,mm,nn,oo] = valid
                                
    #                             if valid == False:
    #                                 error_count += 1
    #                                 if PLOT == True:
    #                                     if error_count <= 10:
    #                                         plt.plot(x,y,label='noise equivalent angle model',color='k')
    #                                         plt.stairs(counts,bins,alpha=0.5,color='k',fill=True)
    #                                         plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
    #                                         plt.title('No threshold = standard ; WFS slopes threshold error (rad)'
    #                                                 + '\n{:.0f}r0, {:.1f}dor0, {:.2f}Rytov, {:.0f}H_HA'.format(r0[ii]*100,dor0[jj],Rytov[kk],H_HA[ll]/1000)
    #                                                 + '\n{:.0f}Nphot, {:.2f}RO, threshold={:.2e}'.format(N_phot_list[mm],RON_list[nn],Threshold_list[oo])
    #                                                 #+ '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
    #                                                 + '\nvalid = {:}'.format(valid))
    #                                         LIM = np.max(np.abs([x.min(),x.max()]))
    #                                         plt.xlim(-LIM,LIM)
    #                                         plt.savefig('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/notGaussian/with_noise_error{:d}.png'.format(error_count))
    #                                         plt.show();plt.clf()
                                    
    #                             if PLOT == True:
    #                                 if (valid == True) and (good_count <=10):
    #                                     plt.plot(x,y,label='noise equivalent angle model',color='k')
    #                                     plt.stairs(counts,bins,alpha=0.5,color='k',fill=True)
    #                                     plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
    #                                     plt.title('No threshold = standard ; WFS slopes threshold error (rad)'
    #                                                 + '\n{:.0f}r0, {:.1f}dor0, {:.2f}Rytov, {:.0f}H_HA'.format(r0[ii]*100,dor0[jj],Rytov[kk],H_HA[ll]/1000)
    #                                                 + '\n{:.0f}Nphot, {:.2f}RO, threshold={:.2e}'.format(N_phot_list[mm],RON_list[nn],Threshold_list[oo])
    #                                                 #+ '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
    #                                                 + '\nvalid = {:}'.format(valid))
    #                                     LIM = np.max(np.abs([x.min(),x.max()]))
    #                                     plt.xlim(-LIM,LIM)
    #                                     plt.savefig('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/Gaussian/with_noise_error{:d}{:d}{:d}{:d}{:d}{:d}{:d}.png'.format(ii,jj,kk,ll,mm,nn,oo))
    #                                     plt.show();plt.clf()
    #                                     good_count += 1
                                    
    # with open('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/output.txt','a') as file:
    #     print('With noise fail/valid = {:d}/{:d}'.format(np.nansum(GaussWithNoise==False),np.nansum(np.isnan(GaussWithNoise) == False)),file=file)
    #     print('With noise no threshold fail/valid = {:d}/{:d}'.format(np.nansum(GaussWithNoise[...,0]==False),np.nansum(np.isnan(GaussWithNoise[...,0]) == False)),file=file)
    # file.close()
    # with open('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/GaussWithNoise.pkl', 'wb') as file:
    #     pickle.dump(GaussWithNoise,file=file)
    # file.close()

    # Noise = WithNoise
    # # Noise = np.zeros_like(WithNoise)
    # for mm in np.arange(MM):
    #     for nn in np.arange(NN):
    #         for oo in np.arange(OO):
    #             for item2 in np.arange(COUNT2):
    #                 Noise[:,:,:,:,mm,nn,oo,:,:,item2] -= NoNoiseSHWFSTilt          
        
    # del NoNoiseSHWFSTilt
    # for ii in np.arange(4):
    #     with open('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/Noise{:d}.pkl'.format(ii), 'wb') as file:
    #         pickle.dump(Noise[ii],file)
    #     file.close()
    # # del Noise

    # with open('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/Noise0.pkl','rb') as file:
    #     input_data = pickle.load(file)
    # file.close()
    # Noise = np.zeros((4,
    #                  input_data.shape[0],
    #                  input_data.shape[1],
    #                  input_data.shape[2],
    #                  input_data.shape[3],
    #                  input_data.shape[4],
    #                  input_data.shape[5],
    #                  input_data.shape[6],
    #                  input_data.shape[7],
    #                  input_data.shape[8]),dtype=float)
    # del input_data
    # for ii in np.arange(4):
    #     with open('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/Noise{:d}.pkl'.format(ii),'rb') as file:
    #         input_data = pickle.load(file)
    #     Noise[ii] = input_data[0]
    #     file.close()
    #     del input_data

    # NoiseRMS = np.nanstd(Noise,axis=(-3,-2,-1))
    # NoiseRMS[NoiseRMS==0] = np.nan
    # ROI = ((~np.isnan(NoiseRMS[...,0,0,0])))

    # with open('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/NoiseRMS.pkl', 'wb') as file:
    #     pickle.dump(NoiseRMS,file)
    # file.close()

    # LABEL_r0 = np.zeros((II), dtype=object)
    # for ii in np.arange(II):
    #     LABEL_r0[ii] = 'r0={:.0f}cm'.format(r0[ii]*100)

    # LABEL_DOR0 = np.zeros((JJ), dtype=object)
    # for jj in np.arange(JJ):
    #     LABEL_DOR0[jj] = 'd/r0={:.1f}'.format(dor0[jj])
        
    # LABEL_RYTOV = np.zeros((KK), dtype=object)
    # for kk in np.arange(KK):
    #     LABEL_RYTOV[kk] = '$\sigma^2_\chi=${:.2f}'.format(Rytov[kk])
        
    # LABEL_HHA = np.zeros((LL), dtype=object)
    # for ll in np.arange(LL):
    #     LABEL_HHA[ll] = 'HA={:.0f}km'.format(H_HA[ll]/1000)

    # LABEL_N_phot = np.zeros((MM), dtype=object)
    # for mm in np.arange(MM):
    #     LABEL_N_phot[mm] = 'N_phot={:.0f}'.format(N_phot_list[mm])

    # LABEL_RON = np.zeros((NN), dtype=object)
    # for nn in np.arange(NN):
    #     LABEL_RON[nn] = 'RON={:.2f}'.format(RON_list[nn])

    # LABEL_Threshold = np.zeros((OO), dtype=object)
    # for oo in np.arange(OO):
    #     LABEL_Threshold[oo] = 'Threshold={:.2f}'.format(Threshold_list[oo])


    # color_map = plt.cm.get_cmap('inferno')
    # reversed_color_map = color_map.reversed()
    # plt.subplots(4,4,sharex=True,sharey=True,squeeze=False)
    # # plt.subplot(4,4,1)
    # plt.suptitle('Noisy SHWFS has Gauss Distr? 1 = True, 0 = False'
    #               + '\n{:.0f}Nphot, {:.2f}RO, threshold={:.2e}'.format(N_phot_list[mm],RON_list[nn],Threshold_list[oo]))
    # for kk in np.arange(KK):
    #     for ll in np.arange(LL):
    #         plt.subplot(4,4,LL*(KK - kk - 1) + ll + 1)
    #         plt.pcolor(r0*100, dor0, ROI[...,kk,ll].T,
    #                     #vmin=0,vmax=1,
    #                     cmap=reversed_color_map,edgecolors='k',linewidths=0.5)
    #         # plt.colorbar()
    #         # plt.xlabel('r0 (cm)')
    #         # plt.ylabel('d/r0')
    #         # plt.title(LABEL_RYTOV[kk] + ', ' + LABEL_HHA[ll])
    #         if kk == 0:
    #             plt.xlabel('r0 (cm)\n' + LABEL_HHA[ll])
    #         if ll == 0:
    #             plt.ylabel(LABEL_RYTOV[kk] + '\nd/r0')
    #         if ll == (LL - 1):
    #             plt.colorbar()
    # # plt.subplot(4,4,1)
    # # plt.colorbar()
    # # plt.xlabel('r0 (cm)')
    # # plt.ylabel('d/r0')
    # plt.show();plt.clf()
    # plt.clf()
            


# if True:
    GaussNoise = np.zeros((II,JJ,KK,LL,MM,NN,OO),dtype=float)
    GaussNoise[:] = np.nan
    error_count = 0
    good_count = 0

    # Noise distribution
    for ii in np.arange(II):
        for jj in np.arange(JJ):
            for kk in np.arange(KK):
                for ll in np.arange(LL):
                    # print(ii,jj,kk,ll)
                    for mm in np.arange(MM):
                        for nn in np.arange(NN):
                            for oo in np.arange(OO):
                                
                                # if mm!=2:
                                #     continue
                                # if nn!=3:
                                #     continue
                                # if oo!=3:
                                #     continue
                                
                                noise = Noise[ii,jj,kk,ll,mm,nn,oo].flatten()
                                noise_temp = noise[np.isnan(noise) == False].flatten()
                                noise = noise_temp[noise_temp!=0]

                                if noise.size == 0:
                                    continue 

                                noise -= np.nanmean(noise)
                                expected_std = np.std(noise)
                                
                                # plt.plot(noise)
                                # plt.show();plt.clf()
                                
                                if np.isnan(expected_std) or (expected_std == 0):
                                    # print(expected_std)
                                    continue
                                
                                # plt.hist(noise.flatten())
                                x = np.linspace(np.nanmin(noise), np.nanmax(noise),num=nbins)
                                area = np.sum(np.isnan(noise) == False)
                                
                                counts, bins = np.histogram(noise[np.isnan(noise)==False].flatten(),
                                                            bins=nbins)
                                
                                y = gauss(x,0,expected_std,area)
                                
                                # fitting_index = np.sum(np.abs(y - counts))/area
                                
                                # ks method # prove they are both from the same distribution
                                res = stats.kstest(noise,np.random.normal(0,expected_std,area))
                                valid0 = (res.pvalue < 0.05)
                                
                                # res = stats.goodness_of_fit(stats.norm,noise)
                                # valid = (res.pvalue < 0.05)
                                
                                # # puttiwat's? method # prove binning statistics if gaussian or not
                                bincount = np.nansum(counts!=0)
                                percentage = (stats.norm.cdf(1)-stats.norm.cdf(-1))
                                expectedtobein = bincount*percentage - 2.*np.sqrt(bincount*percentage)
                                valid1 = ((np.nansum(((counts[counts!=0] - y[counts!=0])**2/counts[counts!=0]) < 1) / bincount) > expectedtobein/bincount)
                                
                                # show if distribution difference to gaussian is less than 10%
                                valid2 = ((np.sum(np.abs((counts - y)*(bins[1]-bins[0])/area)) < 0.1))
                                
                                valid = (valid0 or valid1 or valid2)

                                GaussNoise[ii,jj,kk,ll,mm,nn,oo] = valid
                                
                                if valid == False:
                                    error_count += 1
                                    if PLOT == True:
                                        if error_count <= 10:
                                            plt.plot(x,y,label='noise equivalent angle model',color='k')
                                            plt.stairs(counts,bins,alpha=0.5,color='k',fill=True)
                                            plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
                                            plt.title('No threshold = standard ; WFS slopes threshold error (rad)'
                                                    + '\n{:.0f}r0, {:.1f}dor0, {:.2f}Rytov, {:.0f}H_HA'.format(r0[ii]*100,dor0[jj],Rytov[kk],H_HA[ll]/1000)
                                                    + '\n{:.0f}Nphot, {:.2f}RO, threshold={:.2e}'.format(N_phot_list[mm],RON_list[nn],Threshold_list[oo])
                                                    # + '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
                                                    + '\nvalid = {:}'.format(valid))
                                            LIM = np.max(np.abs([x.min(),x.max()]))
                                            plt.xlim(-LIM,LIM)
                                            plt.savefig('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/notGaussian/noise_error{:d}.png'.format(error_count))
                                            plt.show();plt.clf()
                                    


                                if PLOT == True:
                                    if (mm == (MM - 1)) and (nn == 0) and (oo == 0):# and (good_count <= 10):
                                        plt.plot(x,y,label='noise equivalent angle model',color='k')
                                        plt.stairs(counts,bins,alpha=0.5,color='k',fill=True)
                                        plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
                                        plt.title('No threshold = standard ; WFS slopes threshold error (rad)'
                                                + '\n{:.0f}r0, {:.1f}dor0, {:.2f}Rytov, {:.0f}H_HA'.format(r0[ii]*100,dor0[jj],Rytov[kk],H_HA[ll]/1000)
                                                + '\n{:.0f}Nphot, {:.2f}RO, threshold={:.2e}'.format(N_phot_list[mm],RON_list[nn],Threshold_list[oo])
                                                # + '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
                                                + '\nvalid = {:}'.format(valid))
                                        LIM = np.max(np.abs([x.min(),x.max()]))
                                        plt.xlim(-LIM,LIM)
                                        plt.savefig('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/Phot/Phot_noise_error{:d}{:d}{:d}{:d}{:d}{:d}{:d}.png'.format(ii,jj,kk,ll,mm,nn,oo))
                                        plt.show();plt.clf()
                                        good_count += 1

                                    if (nn == (NN - 1)) and (oo == 0):# and (good_count <= 10):
                                        plt.plot(x,y,label='noise equivalent angle model',color='k')
                                        plt.stairs(counts,bins,alpha=0.5,color='k',fill=True)
                                        plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
                                        plt.title('No threshold = standard ; WFS slopes threshold error (rad)'
                                                + '\n{:.0f}r0, {:.1f}dor0, {:.2f}Rytov, {:.0f}H_HA'.format(r0[ii]*100,dor0[jj],Rytov[kk],H_HA[ll]/1000)
                                                + '\n{:.0f}Nphot, {:.2f}RO, threshold={:.2e}'.format(N_phot_list[mm],RON_list[nn],Threshold_list[oo])
                                                # + '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
                                                + '\nvalid = {:}'.format(valid))
                                        LIM = np.max(np.abs([x.min(),x.max()]))
                                        plt.xlim(-LIM,LIM)
                                        plt.savefig('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/RON/RON_noise_error{:d}{:d}{:d}{:d}{:d}{:d}{:d}.png'.format(ii,jj,kk,ll,mm,nn,oo))
                                        plt.show();plt.clf()
                                        good_count += 1

                                    if (oo == (OO - 1)):# and (good_count <= 10):
                                        plt.plot(x,y,label='noise equivalent angle model',color='k')
                                        plt.stairs(counts,bins,alpha=0.5,color='k',fill=True)
                                        plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
                                        plt.title('No threshold = standard ; WFS slopes threshold error (rad)'
                                                + '\n{:.0f}r0, {:.1f}dor0, {:.2f}Rytov, {:.0f}H_HA'.format(r0[ii]*100,dor0[jj],Rytov[kk],H_HA[ll]/1000)
                                                + '\n{:.0f}Nphot, {:.2f}RO, threshold={:.2e}'.format(N_phot_list[mm],RON_list[nn],Threshold_list[oo])
                                                # + '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
                                                + '\nvalid = {:}'.format(valid))
                                        LIM = np.max(np.abs([x.min(),x.max()]))
                                        plt.xlim(-LIM,LIM)
                                        plt.savefig('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/Threshold/Threshold_noise_error{:d}{:d}{:d}{:d}{:d}{:d}{:d}.png'.format(ii,jj,kk,ll,mm,nn,oo))
                                        plt.show();plt.clf()
                                        good_count += 1

    with open('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/output.txt','a') as file:
        print('noise fail/valid = {:d}/{:d}'.format(np.nansum(GaussNoise==False),np.nansum(np.isnan(GaussNoise) == False)),file=file)
        print('noise no threshold fail/valid = {:d}/{:d}'.format(np.nansum(GaussNoise[...,0]==False),np.nansum(np.isnan(GaussNoise[...,0]) == False)),file=file)
    file.close()

    with open('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/GaussNoise.pkl', 'wb') as file:
        pickle.dump(GaussNoise,file)
    file.close()




    # # check Noisy WFS no RON, no Threshold         
    # ToStudy = np.asarray(np.where(~GaussWithNoise==True)).T
    # for element in np.arange(ToStudy.shape[0]):
    #     indices = tuple(ToStudy[element])
    #     # GaussWithNoise[indices]
    #     ii = indices[0]
    #     jj = indices[1]
    #     kk = indices[2]
    #     ll = indices[3]
    #     mm = indices[4]
    #     nn = indices[5]
    #     oo = indices[6]
        
    #     # if mm != 1:
    #     #     continue
    #     if nn != 0:
    #         continue
    #     if oo != 0:
    #         continue
        
    #     noise = WithNoise[ii,jj,kk,ll,mm,nn,oo].flatten()
    #     noise_temp = noise[np.isnan(noise) == False].flatten()
    #     noise = noise_temp[noise_temp!=0]
    #     noise -= np.nanmean(noise)
    #     expected_std = np.std(noise)
    #     
    #     # plt.plot(noise)
    #     # plt.show();plt.clf()
    #     
    #     if np.isnan(expected_std) or (expected_std == 0):
    #         continue
    #     
    #     # plt.hist(noise.flatten())
    #     x = np.linspace(np.nanmin(noise), np.nanmax(noise),num=100)
    #     area = np.sum(np.isnan(noise) == False)
    #     
    #     counts, bins = np.histogram(noise[np.isnan(noise)==False].flatten(),
    #                                 bins=100)
    #     
    #     y = gauss(x,0,expected_std,area)
    #     
    #     fitting_index = np.sum(np.abs(y - counts))/area
    #     
    #     # # ks method # prove they are both from the same distribution
    #     # res = stats.kstest(noise,np.random.normal(0,expected_std,area))
    #     # valid = (res.pvalue > 0.01)
    #     
    #     # # # puttiwat's? method # prove binning statistics if gaussian or not
    #     # bincount = np.nansum(counts!=0)
    #     # percentage = (stats.norm.cdf(1)-stats.norm.cdf(-1))
    #     # expectedtobein = bincount*percentage - 2.*np.sqrt(bincount*percentage)
    #     # valid = ((np.nansum(((counts[counts!=0] - y[counts!=0])**2/counts[counts!=0]) < 1) / bincount) > expectedtobein/bincount)
    #     
    #     # show if distribution difference to gaussian is less than 10%
    #     valid = ((np.sum(np.abs((counts - y)*(bins[1]-bins[0])/area)) < 0.1))

    #     GaussNoise[ii,jj,kk,ll,mm,nn,oo] = valid
        
    #     if valid == False:

    #         plt.plot(x,y,label='noise equivalent angle model',color='k')
    #         plt.stairs(counts,bins,alpha=0.5,color='k',fill=True)
    #         plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
    #         plt.title('No threshold = standard ; WFS slopes threshold error (rad)'
    #                   + '\n{:.0f}r0, {:.1f}dor0, {:.2f}Rytov, {:.0f}H_HA'.format(r0[ii]*100,dor0[jj],Rytov[kk],H_HA[ll]/1000)
    #                   + '\n{:.0f}Nphot, {:.2f}RO, threshold={:.2e}'.format(N_phot_list[mm],RON_list[nn],Threshold_list[oo])
    #                   + '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
    #                   + '\nvalid = {:}'.format(valid))
    #         LIM = np.max(np.abs([x.min(),x.max()]))
    #         plt.xlim(-LIM,LIM)
    #         plt.show();plt.clf()
            
    # # check Noisy WFS no Threshold         
    # ToStudy = np.asarray(np.where(~GaussWithNoise==True)).T
    # for element in np.arange(ToStudy.shape[0]):
    #     indices = tuple(ToStudy[element])
    #     # GaussWithNoise[indices]
    #     ii = indices[0]
    #     jj = indices[1]
    #     kk = indices[2]
    #     ll = indices[3]
    #     mm = indices[4]
    #     nn = indices[5]
    #     oo = indices[6]
        
    #     # if mm != 1:
    #     #     continue
    #     # if nn != 0:
    #     #     continue
    #     if oo != 0:
    #         continue
        
    #     noise = WithNoise[ii,jj,kk,ll,mm,nn,oo].flatten()
    #     noise_temp = noise[np.isnan(noise) == False].flatten()
    #     noise = noise_temp[noise_temp!=0]
    #     noise -= np.nanmean(noise)
    #     expected_std = np.std(noise)
    #     
    #     # plt.plot(noise)
    #     # plt.show();plt.clf()
    #     
    #     if np.isnan(expected_std) or (expected_std == 0):
    #         continue
    #     
    #     # plt.hist(noise.flatten())
    #     x = np.linspace(np.nanmin(noise), np.nanmax(noise),num=100)
    #     area = np.sum(np.isnan(noise) == False)
    #     
    #     counts, bins = np.histogram(noise[np.isnan(noise)==False].flatten(),
    #                                 bins=100)
    #     
    #     y = gauss(x,0,expected_std,area)
    #     
    #     fitting_index = np.sum(np.abs(y - counts))/area
    #     
    #     # # ks method # prove they are both from the same distribution
    #     # res = stats.kstest(noise,np.random.normal(0,expected_std,area))
    #     # valid = (res.pvalue > 0.01)
    #     
    #     # # # puttiwat's? method # prove binning statistics if gaussian or not
    #     # bincount = np.nansum(counts!=0)
    #     # percentage = (stats.norm.cdf(1)-stats.norm.cdf(-1))
    #     # expectedtobein = bincount*percentage - 2.*np.sqrt(bincount*percentage)
    #     # valid = ((np.nansum(((counts[counts!=0] - y[counts!=0])**2/counts[counts!=0]) < 1) / bincount) > expectedtobein/bincount)
    #     
    #     # show if distribution difference to gaussian is less than 10%
    #     valid = ((np.sum(np.abs((counts - y)*(bins[1]-bins[0])/area)) < 0.1))
    #     GaussNoise[ii,jj,kk,ll,mm,nn,oo] = valid
        
    #     if valid == False:

    #         plt.plot(x,y,label='noise equivalent angle model',color='k')
    #         plt.stairs(counts,bins,alpha=0.5,color='k',fill=True)
    #         plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
    #         plt.title('No threshold = standard ; WFS slopes threshold error (rad)'
    #                   + '\n{:.0f}r0, {:.1f}dor0, {:.2f}Rytov, {:.0f}H_HA'.format(r0[ii]*100,dor0[jj],Rytov[kk],H_HA[ll]/1000)
    #                   + '\n{:.0f}Nphot, {:.2f}RO, threshold={:.2e}'.format(N_phot_list[mm],RON_list[nn],Threshold_list[oo])
    #                   + '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
    #                   + '\nvalid = {:}'.format(valid))
    #         LIM = np.max(np.abs([x.min(),x.max()]))
    #         plt.xlim(-LIM,LIM)
    #         plt.show();plt.clf()

    # # check Noisy WFS Threshold         
    # ToStudy = np.asarray(np.where(GaussWithNoise1)).T
    # for element in np.arange(ToStudy.shape[0]):
    #     indices = tuple(ToStudy[element])
    #     # GaussWithNoise[indices]
    #     ii = indices[0]
    #     jj = indices[1]
    #     kk = indices[2]
    #     ll = indices[3]
    #     mm = indices[4]
    #     nn = indices[5]
    #     oo = indices[6]
        
    #     # if mm != 1:
    #     #     continue
    #     # if nn != 0:
    #     #     continue
    #     # if oo != 0:
    #     #     continue
        
    #     noise = WithNoise[ii,jj,kk,ll,mm,nn,oo].flatten()
    #     noise_temp = noise[np.isnan(noise) == False].flatten()
    #     noise = noise_temp[noise_temp!=0]
    #     noise -= np.nanmean(noise)
    #     expected_std = np.std(noise)
        
    #     # plt.plot(noise)
    #     # plt.show();plt.clf()
        
    #     if np.isnan(expected_std) or (expected_std == 0):
    #         continue
        
    #     # plt.hist(noise.flatten())
    #     x = np.linspace(np.nanmin(noise), np.nanmax(noise),num=100)
    #     area = np.sum(np.isnan(noise) == False)
        
    #     counts, bins = np.histogram(noise[np.isnan(noise)==False].flatten(),
    #                                 bins=100)
        
    #     y = gauss(x,0,expected_std,area)
        
    #     fitting_index = np.sum(np.abs(y - counts))/area
        
    #     # # ks method # prove they are both from the same distribution
    #     # res = stats.kstest(noise,np.random.normal(0,expected_std,area))
    #     # valid = (res.pvalue > 0.01)
        
    #     # # # puttiwat's? method # prove binning statistics if gaussian or not
    #     # bincount = np.nansum(counts!=0)
    #     # percentage = (stats.norm.cdf(1)-stats.norm.cdf(-1))
    #     # expectedtobein = bincount*percentage - 2.*np.sqrt(bincount*percentage)
    #     # valid = ((np.nansum(((counts[counts!=0] - y[counts!=0])**2/counts[counts!=0]) < 1) / bincount) > expectedtobein/bincount)
        
    #     # show if distribution difference to gaussian is less than 10%
    #     valid = ((np.sum(np.abs((counts - y)*(bins[1]-bins[0])/area)) < 0.1))
        
    #     GaussNoise[ii,jj,kk,ll,mm,nn,oo] = valid
        
    #     if valid == False:

    #         plt.plot(x,y,label='noise equivalent angle model',color='k')
    #         plt.stairs(counts,bins,alpha=0.5,color='k',fill=True)
    #         plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
    #         plt.title('No threshold = standard ; WFS slopes threshold error (rad)'
    #                   + '\n{:.0f}r0, {:.1f}dor0, {:.2f}Rytov, {:.0f}H_HA'.format(r0[ii]*100,dor0[jj],Rytov[kk],H_HA[ll]/1000)
    #                   + '\n{:.0f}Nphot, {:.2f}RO, threshold={:.2e}'.format(N_phot_list[mm],RON_list[nn],Threshold_list[oo])
    #                   + '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
    #                   + '\nvalid = {:}'.format(valid))
    #         LIM = np.max(np.abs([x.min(),x.max()]))
    #         plt.xlim(-LIM,LIM)
    #         plt.show();plt.clf()



    ###############################################################################

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

    LABEL_N_phot = np.zeros((MM), dtype=object)
    for mm in np.arange(MM):
        LABEL_N_phot[mm] = 'N_phot={:.0f}'.format(N_phot_list[mm])

    LABEL_RON = np.zeros((NN), dtype=object)
    for nn in np.arange(NN):
        LABEL_RON[nn] = 'RON={:.2f}'.format(RON_list[nn])

    LABEL_Threshold = np.zeros((OO), dtype=object)
    for oo in np.arange(OO):
        LABEL_Threshold[oo] = 'Threshold={:.2f}'.format(Threshold_list[oo])



    ###############################################################################
    # ░▒█░▒█░█▀▀░█▀▀░█▀▀░█░▒█░█░░░░▒█▀▀█░█░░▄▀▀▄░▀█▀░█▀▀
    # ░▒█░▒█░▀▀▄░█▀▀░█▀░░█░▒█░█░░░░▒█▄▄█░█░░█░░█░░█░░▀▀▄
    # ░░▀▄▄▀░▀▀▀░▀▀▀░▀░░░░▀▀▀░▀▀░░░▒█░░░░▀▀░░▀▀░░░▀░░▀▀▀
    ###############################################################################
    if PLOT == True:
        for mm in np.arange(MM):
            for nn in np.arange(NN):
                for oo in np.arange(OO):
                    # plt.figure()
                    # Figure 1

                    if np.nansum(GaussNoise[...,kk,ll,mm,nn,oo] == False) != 0:

                        color_map = plt.cm.get_cmap('inferno')
                        reversed_color_map = color_map.reversed()
                        plt.subplots(4,4,sharex=True,sharey=True,squeeze=False)
                        # plt.subplot(4,4,1)
                        plt.suptitle('Noise SHWFS has Gauss Distr? 1 = True, 0 = False'
                                    + '\n{:.1e}Nphot, {:.1e}RO, threshold={:.1e}'.format(N_phot_list[mm],RON_list[nn],Threshold_list[oo]))
                        for kk in np.arange(KK):
                            for ll in np.arange(LL):
                                plt.subplot(4,4,LL*(KK - kk - 1) + ll + 1)
                                plt.pcolor(r0*100, dor0, GaussNoise[...,kk,ll,mm,nn,oo].T,
                                        vmin=0,vmax=1,
                                        cmap=reversed_color_map,edgecolors='k',linewidths=0.5)
                                # plt.colorbar()
                                # plt.xlabel('r0 (cm)')
                                # plt.ylabel('d/r0')
                                # plt.title(LABEL_RYTOV[kk] + ', ' + LABEL_HHA[ll])
                                if kk == 0:
                                    plt.xlabel('r0 (cm)\n' + LABEL_HHA[ll])
                                if ll == 0:
                                    plt.ylabel(LABEL_RYTOV[kk] + '\nd/r0')
                                if ll == (LL - 1):
                                    plt.colorbar()
                        # plt.subplot(4,4,1)
                        # plt.colorbar()
                        # plt.xlabel('r0 (cm)')
                        # plt.ylabel('d/r0')
                        plt.savefig('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/Gauss_or_not_{:}{:}{:}.png'.format(mm,nn,oo))
                        plt.show();plt.clf()
                        plt.clf()

# # plt.figure()
# # Figure 1
# g_var = np.nanvar(g_SLOPE - SHWFS_SLOPE, ddof=1, axis=(-1,-2))
# g_var[g_var==0] = np.nan

# color_map = plt.cm.get_cmap('inferno')
# reversed_color_map = color_map.reversed()
# plt.subplots(4,4,sharex=True,sharey=True,squeeze=False)
# # plt.subplot(4,4,1)
# plt.suptitle('RMS difference of Ig and SHWFS slopes (rad/lenslet)')
# for kk in np.arange(KK):
#     for ll in np.arange(LL):
#         plt.subplot(4,4,LL*(KK - kk - 1) + ll + 1)
#         plt.pcolor(r0*100, dor0, np.sqrt(g_var[:,:,kk,ll]).T,
#                    vmin=np.sqrt(np.nanmin(g_var)),vmax=np.sqrt(np.nanmax(g_var)),
#                    cmap=reversed_color_map,edgecolors='k',linewidths=0.5)
#         # plt.colorbar()
#         # plt.xlabel('r0 (cm)')
#         # plt.ylabel('d/r0')
#         # plt.title(LABEL_RYTOV[kk] + ', ' + LABEL_HHA[ll])
#         if kk == 0:
#             plt.xlabel('r0 (cm)\n' + LABEL_HHA[ll])
#         if ll == 0:
#             plt.ylabel(LABEL_RYTOV[kk] + '\nd/r0')
#         if ll == (LL - 1):
#             plt.colorbar()
# # plt.subplot(4,4,1)
# # plt.colorbar()
# # plt.xlabel('r0 (cm)')
# # plt.ylabel('d/r0')
# plt.show();plt.clf()
# plt.clf()



# MAX = np.max(np.abs([np.nanmin(NoiseRMS),np.nanmax(NoiseRMS)]))
# for mm in np.arange(MM):
#     for nn in np.arange(NN):
#         for oo in np.arange(OO):
#             # plt.figure()
#             # Figure 1
#             color_map = plt.cm.get_cmap('inferno')
#             reversed_color_map = color_map.reversed()
#             plt.subplots(4,4,sharex=True,sharey=True,squeeze=False)
#             # plt.subplot(4,4,1)
#             plt.suptitle('Noisy SHWFS has Gauss Distr? 1 = True, 0 = False'
#                          + '\n{:.0f}Nphot, {:.2f}RO, threshold={:.2e}'.format(N_phot_list[mm],RON_list[nn],Threshold_list[oo]))
#             for kk in np.arange(KK):
#                 for ll in np.arange(LL):
#                     plt.subplot(4,4,LL*(KK - kk - 1) + ll + 1)
#                     plt.pcolor(r0*100, dor0, NoiseRMS[...,kk,ll,mm,nn,oo].T,
#                                vmin=-MAX,vmax=MAX,
#                                cmap=reversed_color_map,edgecolors='k',linewidths=0.5)
#                     # plt.colorbar()
#                     # plt.xlabel('r0 (cm)')
#                     # plt.ylabel('d/r0')
#                     # plt.title(LABEL_RYTOV[kk] + ', ' + LABEL_HHA[ll])
#                     if kk == 0:
#                         plt.xlabel('r0 (cm)\n' + LABEL_HHA[ll])
#                     if ll == 0:
#                         plt.ylabel(LABEL_RYTOV[kk] + '\nd/r0')
#                     if ll == (LL - 1):
#                         plt.colorbar()
#             # plt.subplot(4,4,1)
#             # plt.colorbar()
#             # plt.xlabel('r0 (cm)')
#             # plt.ylabel('d/r0')
#             plt.savefig('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/Noise_RMS{:}{:}{:}.png'.format(ll,mm,nn,oo))
#             plt.show();plt.clf()
#             plt.clf()
            
            
# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 plt.plot(N_phot_list,NoiseRMS[ii,jj,kk,ll,:,0,0].flatten())
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('photon_noise.png')
# plt.show();plt.clf()

# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 plt.plot(RON_list,NoiseRMS[ii,jj,kk,ll,-1,:,0].flatten())
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('readout_noise.png')
# plt.show();plt.clf()


# plt.plot(NoiseRMS[:,:,:,:,-1,0,0].flatten())
# plt.xscale('log')
# plt.yscale('log')
# plt.show();plt.clf()

W = 64/2
atm_error = (0.5*W**-2*(dor0)**(5./3.))**0.5
photon_error = (2*W/N_phot_list)**0.5
RON_error = np.zeros((MM,NN),dtype=float)
NEW_RON_list = np.zeros((MM,NN),dtype=float)
for mm in np.arange(MM):
    for nn in np.arange(NN):
        RON_error[mm,nn] = (np.pi**2/3.*(RON_list[nn]/N_phot_list[mm])**2*(64**2)**2/(2)**2)**0.5
        NEW_RON_list[mm,nn] = RON_list[nn]/N_phot_list[mm]


color_list = ['c','m','y','k']

first_time = [True,True,True,True]
for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                if first_time[jj]:
                    plt.plot(N_phot_list,(NoiseRMS[ii,jj,kk,ll,:,0,0]).flatten(),
                             color = color_list[jj],label='dor0={}'.format(dor0[jj]))
                    # plt.hlines(atm_error[jj],
                    #            N_phot_list.min(),
                    #            N_phot_list.max(),
                    #            label='atmos noise, dor0={}'.format(dor0[jj]),
                    #            ls=':',color=color_list[jj])
                    first_time[jj] = False
                else:
                    plt.plot(N_phot_list,(NoiseRMS[ii,jj,kk,ll,:,0,0]).flatten(),
                             color = color_list[jj])
plt.plot(N_phot_list,photon_error,ls='--',label=('photon noise model'))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of photon per lenslet')
plt.ylabel('RMS-OPD (rad)')
plt.title('Photon noise')
plt.legend()
plt.savefig('photon_noise.png')#/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/photon_noise.png')
plt.show();plt.clf()

RON_rms = np.zeros((II,JJ,KK,LL,MM,NN),dtype=float)

first_time = [True,True,True,True]
for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                for mm in np.arange(MM):
                    for nn in np.arange(NN):
                        if nn == 0:
                            continue
                        RON_rms[ii,jj,kk,ll,mm,nn] = ((NoiseRMS[ii,jj,kk,ll, mm,nn,0]**2
                                       - NoiseRMS[ii,jj,kk,ll, mm,0,0]**2)**0.5)
                        # RON_rms[ii,jj,kk,ll,mm,nn] = NoiseRMS[ii,jj,kk,ll, mm,nn,0]
                        if first_time[jj]:
                            # print(NEW_RON_list[mm,nn])
                            # print(RON_rms[ii,jj,kk,ll,mm,nn])
                            plt.plot(NEW_RON_list[mm,nn],
                                     RON_rms[ii,jj,kk,ll,mm,nn].flatten(),
                                     color = color_list[jj],label='dor0={}'.format(dor0[jj]), marker='.')
                            first_time[jj] = False
                            # plt.hlines(atm_error[jj],
                            #            RON_list[1:].min(),
                            #            RON_list[1:].max(),
                            #            label='atmos noise, dor0={}'.format(dor0[jj]),
                            #            ls=':',color=color_list[jj])
                        else:
                            plt.plot(NEW_RON_list[mm,nn],
                                     RON_rms[ii,jj,kk,ll,mm,nn].flatten(),
                                     color = color_list[jj], marker='.')
                            
plt.plot(NEW_RON_list[:,1:].flatten(),RON_error[:,1:].flatten(),ls='--',label=('ron noise model'))
plt.legend()
plt.ylabel('RMS-OPD (rad)')
plt.xlabel('Readout e- std per pixel compare to total photon in a lenslet')
plt.xscale('log')
plt.yscale('log')
plt.savefig('readout_noise.png')#/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/readout_noise.png')
plt.show();plt.clf()

THRESHOLD_rms = np.zeros((II,JJ,KK,LL,MM,NN,OO),dtype=float)
first_time = [True,True,True,True]
for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                for mm in np.arange(MM):
                    for nn in np.arange(NN):
                        for oo in np.arange(OO):
                            if oo == 0:
                                continue
                            # print(NoiseRMS[ii,jj,kk,ll, mm,nn,oo].shape)
                            # print(NoiseRMS[ii,jj,kk,ll, mm,nn,0].shape)
                            THRESHOLD_rms[ii,jj,kk,ll,mm,nn,oo] = ((NoiseRMS[ii,jj,kk,ll, mm,nn,oo]**2
                                          - NoiseRMS[ii,jj,kk,ll, mm,nn,0]**2)**0.5)
                            # THRESHOLD_rms[ii,jj,kk,ll,mm,nn,oo] = NoiseRMS[ii,jj,kk,ll, mm,nn,oo]
                            if first_time[jj]:
                                plt.plot(Threshold_list[oo],
                                         THRESHOLD_rms[ii,jj,kk,ll,mm,nn,oo].flatten(),
                                        color = color_list[jj],label='dor0={}'.format(dor0[jj]), marker='.')
                                first_time[jj] = False
                                # plt.hlines(atm_error[jj],
                                #            Threshold_list[1:].min(),
                                #            Threshold_list[1:].max(),
                                #            label='atmos noise, dor0={}'.format(dor0[jj]),
                                #            ls=':',color=color_list[jj])
                            else:
                                plt.plot(Threshold_list[oo],
                                         THRESHOLD_rms[ii,jj,kk,ll,mm,nn,oo].flatten(),
                                        color = color_list[jj], marker='.')
#plt.plot(RON_list[1:],RON_error[1:],ls='--',label=('ron noise model'))
plt.legend()
plt.ylabel('RMS-OPD (rad)')
plt.xlabel('Threshold level (relative to peak intensity)')#'Readout e- std per pixel compare to total photon in a lenslet')
plt.xscale('log')
plt.yscale('log')
plt.savefig('threshold.png')#/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/readout_noise.png')
plt.show();plt.clf()

with open('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/NoiseVar.pkl', 'wb') as file:
    var = {'NEW_RON_list' : NEW_RON_list,
           'N_phot_list' : N_phot_list,
           'atm_error' : atm_error,
           'photon_error' : photon_error,
           'RON_error' : RON_error,
           'photon_rms' : NoiseRMS[...,0,0],
           'RON_rms' : RON_rms,
           'THRESHOLD_rms' : THRESHOLD_rms}
    pickle.dump(var,file)
file.close()




