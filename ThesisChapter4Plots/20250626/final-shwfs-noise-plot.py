# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 18:01:58 2025

@author: gqmc62
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import pickle
import matplotlib as mpl
from matplotlib.lines import Line2D
from tqdm import tqdm

HOME = r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\20250626\noisyslope'
os.chdir(HOME)

CONVERT = 500/np.pi/2

fontsize = 10
figsize = [4,4]
dpi = 600

n_cog = 17
n_wfs_total = 64

mpl.rcParams['axes.titlesize'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['figure.titlesize'] = fontsize
mpl.rcParams['figure.figsize'] = figsize
mpl.rcParams['figure.dpi'] = dpi
mpl.rcParams['savefig.dpi'] = dpi
mpl.rcParams['savefig.bbox'] = 'tight'
# mpl.rcParams['savefig.bbox_inches'] = 'tight'
mpl.rcParams['savefig.format'] = 'pdf'

color_list = np.array(list(mpl.colors.TABLEAU_COLORS.keys()))
color_list = np.concatenate([color_list,color_list,color_list,color_list,color_list,color_list,color_list,color_list,color_list,color_list,color_list])

# with open('sq-ap-no-ft-bias-SHWFSNoiseNoScintillation00.pkl', 'rb') as file:
#     data = pickle.load(file)
    
# N_phot_list = data['N_phot_list']
# RON_list = data['RON_list']
# Threshold_list = data['Threshold_list']

# error = data['Noise']*CONVERT
# rms = np.nanstd(error,axis=(-1,-2,-3))


# for i in range(5):
#     plt.plot(RON_list,rms[i,:,1],label='Photon:{:.0e}'.format(N_phot_list[i]))
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.xlabel('RMS readout e- per pixel (number)')
# plt.ylabel('RMS OPD error (nm)')
# plt.show()


# for i in range(5):
#     plt.plot(N_phot_list,rms[:,i,1],label='RON:{:.0e}'.format(RON_list[i]))
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.xlabel('Photon e- (number)')
# plt.ylabel('RMS OPD error (nm)')
# plt.show()


# for i in range(5):
#     plt.plot(Threshold_list,rms[i,2,:],
#                 label='Photon:{:.0e}'.format(N_phot_list[i]),
#                 color=color_list[i],ls='-')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Threshold level (normalised)')
# plt.ylabel('RMS OPD error (nm)')
# # legend_elements = [
# #     Line2D([0], [0], color='black', linestyle='-', label='{:.0e}'.format(N_phot_list[0])),
# #     Line2D([0], [0], color='black', linestyle='-.', label='{:.0e}'.format(N_phot_list[2])),
# #     Line2D([0], [0], color='black', linestyle='--', label='{:.0e}'.format(N_phot_list[4])),
# #     Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label='{:.0e}'.format(Threshold_list[0])),
# #     Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label='{:.0e}'.format(Threshold_list[1])),
# #     Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label='{:.0e}'.format(Threshold_list[2])),
# #     Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label='{:.0e}'.format(Threshold_list[3])),
# #     Line2D([0], [0], color=color_list[4], linestyle='-', marker='', label='{:.0e}'.format(Threshold_list[4])),
# #     ]
# # plt.legend(handles=legend_elements, loc='best')
# plt.legend()
# plt.show()

# for i in range(5):
#     plt.plot(Threshold_list,rms[2,i,:],
#                 label='RON:{:.0e}'.format(RON_list[i]),
#                 color=color_list[i],ls='-')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Threshold level (normalised)')
# plt.ylabel('RMS OPD error (nm)')
# # legend_elements = [
# #     Line2D([0], [0], color='black', linestyle='-', label='{:.0e}'.format(N_phot_list[0])),
# #     Line2D([0], [0], color='black', linestyle='-.', label='{:.0e}'.format(N_phot_list[2])),
# #     Line2D([0], [0], color='black', linestyle='--', label='{:.0e}'.format(N_phot_list[4])),
# #     Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label='{:.0e}'.format(Threshold_list[0])),
# #     Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label='{:.0e}'.format(Threshold_list[1])),
# #     Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label='{:.0e}'.format(Threshold_list[2])),
# #     Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label='{:.0e}'.format(Threshold_list[3])),
# #     Line2D([0], [0], color=color_list[4], linestyle='-', marker='', label='{:.0e}'.format(Threshold_list[4])),
# #     ]
# # plt.legend(handles=legend_elements, loc='best')
# plt.legend()
# plt.show()


##################################################################################

with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\Ultimate_input.pkl','rb') as file:
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
##################################################################################

with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\20250626\Slopes\sq-ap-no-ft-bias-SHWFSNoiseNoScintillation00.pkl', 'rb') as file:
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

COUNT1 = input_data['Noise'].shape[-3]
COUNT2 = input_data['Noise'].shape[-1]
del input_data    
NoiseRMS_noscin = np.zeros((II,JJ,MM,NN,OO),dtype=float)

for ii in tqdm(np.arange(II)):
    for jj in np.arange(JJ):
        try:
            with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\20250626\Slopes\sq-ap-no-ft-bias-SHWFSNoiseNoScintillation{:d}{:d}.pkl'.format(ii,jj), 'rb') as file:
                input_data = pickle.load(file)
            NoiseRMS_noscin[ii,jj] = np.nanstd(input_data['Noise'],axis=(-1,-2,-3)) * CONVERT
            file.close()
        except:
            NoiseRMS_noscin[ii,jj] = np.nan
del input_data

NoiseRMS_noscin[NoiseRMS_noscin==0] = np.nan
# ROI = ((~np.isnan(NoiseRMS_noscin[...,0,0,0])))
##################################################################################

with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\20250626\Slopes\sq-ap-no-ft-bias-SHWFSNoiseScintillation0000.pkl', 'rb') as file:
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

COUNT1 = input_data['Noise'].shape[-3]
COUNT2 = input_data['Noise'].shape[-1]
del input_data    
NoiseRMS = np.zeros((II,JJ,KK,LL,MM,NN,OO),dtype=float)

for ii in tqdm(np.arange(II)):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                try:
                    with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\20250626\Slopes\sq-ap-no-ft-bias-SHWFSNoiseScintillation{:d}{:d}{:d}{:d}.pkl'.format(ii,jj,kk,ll), 'rb') as file:
                        input_data = pickle.load(file)
                    NoiseRMS[ii,jj,kk,ll] = np.nanstd(input_data['Noise'],axis=(-1,-2,-3)) * CONVERT
                    file.close()
                except:
                    NoiseRMS[ii,jj,kk,ll] = np.nan
del input_data

NoiseRMS[NoiseRMS==0] = np.nan
ROI = ((~np.isnan(NoiseRMS[...,0,0,0])))


##################################################################################

W = n_wfs_total/2
new_atm_error = (0.5*W**-2*(dor0)**(5./3.))**0.5 * CONVERT
new_photon_error = (2*W/N_phot_list)**0.5 * CONVERT
RON_expect = np.zeros((N_phot_list.shape[0],RON_list.shape[0]),dtype=float)
for i in range(len(N_phot_list)):
    RON_expect[i] = (np.pi**2/3.*(RON_list/N_phot_list[i])**2*(n_cog**2)**2/(2)**2)**0.5 * CONVERT

###############################################################################

PHOT_NOISE = np.nanmean(NoiseRMS[...,0,1]**2,axis=(0,2,3))**0.5
PHOT_NOISE_error = (np.nanstd(NoiseRMS[...,:,0,1]**2,axis=(0,2,3))**0.5
                    )#/ np.sqrt(np.sum(~np.isnan(NoiseRMS[...,0,1]),axis=(0,2,3))))
PHOT_NOISE_noscin = np.nanmean(NoiseRMS_noscin[...,0,1]**2,axis=(0))**0.5
PHOT_NOISE_error_noscin = (np.nanstd(NoiseRMS_noscin[...,0,1]**2,axis=(0))**0.5
                    )#/ np.sqrt(np.sum(~np.isnan(NoiseRMS_noscin[...,0,1]),axis=(0))))


fig = plt.figure(figsize=(5,4))
ax = plt.subplot(111)
for jj in range(len(dor0)):
    # if jj == 1 or jj == 2:
    #     continue
    plt.errorbar(N_phot_list,
                 PHOT_NOISE[jj],
                 yerr=PHOT_NOISE_error[jj],
                 ls='-',capsize=5,marker='x',color=color_list[jj])
    plt.errorbar(N_phot_list,
                 PHOT_NOISE_noscin[jj],
                 yerr=PHOT_NOISE_error_noscin[jj],
                 ls='--',capsize=5,marker='+',color=color_list[jj])
plt.errorbar(N_phot_list,
             new_photon_error,ls=':',color='black')
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', marker='x', label='With scin.'),
    Line2D([0], [0], color='black', linestyle='--', marker='+', label='No scin.'),
    Line2D([0], [0], color='black', linestyle=':', marker='', label='Model'),
    Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label='$d/r_0$={:.1f}'.format(dor0[0])),
    Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label='$d/r_0$={:.1f}'.format(dor0[1])),
    Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label='$d/r_0$={:.1f}'.format(dor0[2])),
    Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label='$d/r_0$={:.1f}'.format(dor0[3])),
    ]
# plt.legend(handles=legend_elements, loc='best')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(handles=legend_elements,loc='center left', bbox_to_anchor=(1, 0.5))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Photo electron in a lenslet')
plt.ylabel('Noise OPD (nm)')
plt.savefig('latest-phot-noise')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform

# --- transforms ---
class SignedSqrtTransform(Transform):
    input_dims = output_dims = 1
    is_separable = True

    def transform_non_affine(self, a):
        a = np.array(a)
        out = np.zeros_like(a, dtype=float)
        pos = a > 0
        neg = a < 0
        out[pos] = np.sqrt(a[pos])
        out[neg] = -np.sqrt(-a[neg])
        return out

    def inverted(self):
        return InvertedSignedSqrtTransform()


class InvertedSignedSqrtTransform(Transform):
    input_dims = output_dims = 1
    is_separable = True

    def transform_non_affine(self, a):
        a = np.array(a)
        out = np.zeros_like(a, dtype=float)
        pos = a > 0
        neg = a < 0
        out[pos] = a[pos]**2
        out[neg] = -a[neg]**2
        return out

    def inverted(self):
        return SignedSqrtTransform()


# --- scale ---
class SignedSqrtScale(ScaleBase):
    name = "signed_sqrt"

    def __init__(self, axis, **kwargs):
        super().__init__(axis)

    def get_transform(self):
        return SignedSqrtTransform()

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(MaxNLocator())
        # custom formatter to show original values
        axis.set_major_formatter(FuncFormatter(
            lambda val, pos: f"{InvertedSignedSqrtTransform().transform([val])[0]:.0f}"
        ))

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return vmin, vmax


# register the scale
mscale.register_scale(SignedSqrtScale)

fig = plt.figure(figsize=(5,4))
ax = plt.subplot(111)
for jj in range(len(dor0)):
    # if jj == 1 or jj == 2:
    #     continue
    # plt.errorbar(N_phot_list,
    #              PHOT_NOISE[jj] - new_photon_error,
    #              yerr=PHOT_NOISE_error[jj],
    #              ls='-',capsize=5,marker='x',color=color_list[jj])
    # plt.errorbar(N_phot_list,
    #              PHOT_NOISE_noscin[jj] - new_photon_error,
    #              yerr=PHOT_NOISE_error_noscin[jj],
    #              ls='--',capsize=5,marker='+',color=color_list[jj])
    plt.errorbar(N_phot_list,
                 -(PHOT_NOISE[jj] - new_photon_error),
                 yerr=PHOT_NOISE_error[jj],
                 ls='-',capsize=5,marker='x',color=color_list[jj])
    plt.errorbar(N_phot_list,
                 -(PHOT_NOISE_noscin[jj] - new_photon_error),
                 yerr=PHOT_NOISE_error_noscin[jj],
                 ls='--',capsize=5,marker='+',color=color_list[jj])
plt.errorbar(N_phot_list,
              new_photon_error - new_photon_error,ls=':',color='black')
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', marker='x', label='With scin.'),
    Line2D([0], [0], color='black', linestyle='--', marker='+', label='No scin.'),
    # Line2D([0], [0], color='black', linestyle=':', marker='', label='Model'),
    Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label='$d/r_0$={:.1f}'.format(dor0[0])),
    Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label='$d/r_0$={:.1f}'.format(dor0[1])),
    Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label='$d/r_0$={:.1f}'.format(dor0[2])),
    Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label='$d/r_0$={:.1f}'.format(dor0[3])),
    ]
# plt.legend(handles=legend_elements, loc='best')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(handles=legend_elements,loc='center left', bbox_to_anchor=(1, 0.5))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Photo electron in a lenslet')
plt.ylabel('Model error (nm)')
plt.grid('True')
plt.savefig('latest-phot-noise-estimate-error')
plt.show()

ls_list = ['solid','dashed','dashdot','dotted',(0,(1,5))]

# fig = plt.figure(figsize=(5,4))
# ax = plt.subplot(111)
# for mm in range(MM):
#     for jj in range(JJ):
#         if jj != 2:
#             continue
#         ax.errorbar(Rytov,
#                      np.nanmean(NoiseRMS[:,jj,:,:,mm,0,1],axis=(1,-1)) - np.nanmean(NoiseRMS_noscin[:,jj,mm,0,1])*np.ones(4),
#                      yerr=(np.nanstd(NoiseRMS[:,jj,:,:,mm,0,1],axis=(1,-1))**2#/np.sum(~np.isnan(NoiseRMS[:,jj,:,:,mm,0,1]),axis=(1,-1))**0.5
#                            + np.nanstd(NoiseRMS_noscin[:,jj,mm,0,1])**2)**0.5,#/(NoiseRMS_noscin[:,jj,mm,0,1].shape[0])**0.5),
#                      color=color_list[mm],capsize=5,marker='o',ls='-')#,ls=ls_list[jj]
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('$\sigma^2_R$')
# plt.ylabel('Noise with - noise without scintillation (nm)')
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# legend_elements = [
#     Line2D([0], [0], color=color_list[0], linestyle='-', marker='o', label='Phot={:.0e}'.format(N_phot_list[0])),
#     Line2D([0], [0], color=color_list[1], linestyle='-', marker='o', label='Phot={:.0e}'.format(N_phot_list[1])),
#     Line2D([0], [0], color=color_list[2], linestyle='-', marker='o', label='Phot={:.0e}'.format(N_phot_list[2])),
#     Line2D([0], [0], color=color_list[3], linestyle='-', marker='o', label='Phot={:.0e}'.format(N_phot_list[3])),
#     Line2D([0], [0], color=color_list[4], linestyle='-', marker='o', label='Phot={:.0e}'.format(N_phot_list[4])),
#     ]
# # plt.legend(handles=legend_elements, loc='best')
# ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('latest-phot-noise-dif')
# plt.show()
    

# plt.figure(figsize=(5,5))
# for mm in range(len(N_phot_list)):
#     for jj in range(len(dor0)):
#         plt.errorbar(RON_list/N_phot_list[mm],
#                      np.nanmean(NoiseRMS[:,jj,...,mm,:,1],axis=(0,1,2)),
#                      yerr=(np.nanstd(NoiseRMS[:,jj,...,mm,:,1],axis=(0,1,2))
#                            / np.sqrt(np.sum(~np.isnan(NoiseRMS[:,jj,...,mm,:,1]),axis=(0,1,2)))),
#                      ls='-',capsize=5,marker='x',color=color_list[mm])
#         plt.errorbar(RON_list/N_phot_list[mm],
#                      np.nanmean(NoiseRMS_noscin[:,jj,mm,:,1],axis=(0)),
#                      yerr=(np.nanstd(NoiseRMS_noscin[:,jj,mm,:,1],axis=(0))
#                            / np.sqrt(np.sum(~np.isnan(NoiseRMS_noscin[:,jj,mm,:,1]),axis=(0)))),
#                      ls='--',capsize=5,marker='+',color=color_list[mm])
# for mm in range(N_phot_list.shape[0]):
#     plt.errorbar(RON_list/N_phot_list[mm],
#                  RON_expect[mm],ls=':',color='black')
# legend_elements = [
#     Line2D([0], [0], color='black', linestyle='-', marker='x', label='With scintillation'),
#     Line2D([0], [0], color='black', linestyle='--', marker='+', label='No scintillation'),
#     Line2D([0], [0], color='black', linestyle=':', marker='', label='Model'),
#     Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label='Photo e- in a lenslet = {:.0e}'.format(N_phot_list[0])),
#     Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label='Photo e- in a lenslet = {:.0e}'.format(N_phot_list[1])),
#     Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label='Photo e- in a lenslet = {:.0e}'.format(N_phot_list[2])),
#     Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label='Photo e- in a lenslet = {:.0e}'.format(N_phot_list[3])),
#     Line2D([0], [0], color=color_list[4], linestyle='-', marker='', label='Photo e- in a lenslet = {:.0e}'.format(N_phot_list[4])),
#     ]
# plt.legend(handles=legend_elements, loc='lower right')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('RON e- / Photo e- in a lenslet')
# plt.ylabel('RMS OPD error (nm)')
# plt.show()

# plt.figure(figsize=(5,5))
# plt.errorbar(Threshold_list,
#              np.nanmean(NoiseRMS[:,3,...,-1,0,:],axis=(0,1,2)),
#              yerr=(np.nanstd(NoiseRMS[:,3,...,-1,0,:],axis=(0,1,2))
#                    / np.sqrt(np.sum(~np.isnan(NoiseRMS[:,3,...,-1,0,:]),axis=(0,1,2)))),
#              ls='-',capsize=5,marker='x',color=color_list[0])
# plt.errorbar(Threshold_list,
#              np.nanmean(NoiseRMS_noscin[:,3,-1,0,:],axis=(0)),
#              yerr=(np.nanstd(NoiseRMS_noscin[:,3,-1,0,:],axis=(0))
#                    / np.sqrt(np.sum(~np.isnan(NoiseRMS_noscin[:,3,-1,0,:]),axis=(0)))),
#              ls='--',capsize=5,marker='+',color=color_list[0])
# plt.errorbar(Threshold_list,
#              np.nanmean(NoiseRMS[:,3,...,2,0,:],axis=(0,1,2)),
#              yerr=(np.nanstd(NoiseRMS[:,3,...,2,0,:],axis=(0,1,2))
#                    / np.sqrt(np.sum(~np.isnan(NoiseRMS[:,3,...,2,0,:]),axis=(0,1,2)))),
#              ls='-',capsize=5,marker='x',color=color_list[1])
# plt.errorbar(Threshold_list,
#              np.nanmean(NoiseRMS_noscin[:,3,2,0,:],axis=(0)),
#              yerr=(np.nanstd(NoiseRMS_noscin[:,3,2,0,:],axis=(0))
#                    / np.sqrt(np.sum(~np.isnan(NoiseRMS_noscin[:,3,2,0,:]),axis=(0)))),
#              ls='--',capsize=5,marker='+',color=color_list[1])
# plt.errorbar(Threshold_list,
#              np.nanmean(NoiseRMS[:,3,...,-1,-1,:],axis=(0,1,2)),
#              yerr=(np.nanstd(NoiseRMS[:,3,...,-1,-1,:],axis=(0,1,2))
#                    / np.sqrt(np.sum(~np.isnan(NoiseRMS[:,3,...,-1,-1,:]),axis=(0,1,2)))),
#              ls='-',capsize=5,marker='x',color=color_list[2])
# plt.errorbar(Threshold_list,
#              np.nanmean(NoiseRMS_noscin[:,3,-1,-1,:],axis=(0)),
#              yerr=(np.nanstd(NoiseRMS_noscin[:,3,-1,-1,:],axis=(0))
#                    / np.sqrt(np.sum(~np.isnan(NoiseRMS_noscin[:,3,-1,-1,:]),axis=(0)))),
#              ls='--',capsize=5,marker='+',color=color_list[2])
# plt.errorbar(Threshold_list,
#              np.nanmean(NoiseRMS[:,3,...,2,-1,:],axis=(0,1,2)),
#              yerr=(np.nanstd(NoiseRMS[:,3,...,2,-1,:],axis=(0,1,2))
#                    / np.sqrt(np.sum(~np.isnan(NoiseRMS[:,3,...,2,-1,:]),axis=(0,1,2)))),
#              ls='-',capsize=5,marker='x',color=color_list[3])
# plt.errorbar(Threshold_list,
#              np.nanmean(NoiseRMS_noscin[:,3,2,-1,:],axis=(0)),
#              yerr=(np.nanstd(NoiseRMS_noscin[:,3,2,-1,:],axis=(0))
#                    / np.sqrt(np.sum(~np.isnan(NoiseRMS_noscin[:,3,2,-1,:]),axis=(0)))),
#              ls='--',capsize=5,marker='+',color=color_list[3])
# legend_elements = [
#     Line2D([0], [0], color='black', linestyle='-', marker='x', label='With scintillation'),
#     Line2D([0], [0], color='black', linestyle='--', marker='+', label='No scintillation'),
#     Line2D([0], [0], color='black', linestyle=':', marker='', label='Model'),
#     Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label='Low noise'),
#     Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label='Photon noise high'),
#     Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label='Readout noise high'),
#     Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label='Both noise high'),
#     ]
# plt.legend(handles=legend_elements, loc='lower right')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Threshold (normalised)')
# plt.ylabel('RMS OPD error (nm)')
# plt.show()

jj=2


RON_RMS = np.zeros((MM,NN),dtype=float)
noscinRON_RMS = np.zeros((MM,NN),dtype=float)
RON_RMS_error = np.zeros((MM,NN),dtype=float)
noscinRON_RMS_error = np.zeros((MM,NN),dtype=float)

for nn in range(NN):
    RON_RMS[:,nn] = np.nanmean(NoiseRMS[:,jj,:,:,:,nn,1]**2 - NoiseRMS[:,jj,:,:,:,0,1]**2,axis=(0,1,2))**0.5
    noscinRON_RMS[:,nn] = np.nanmean(NoiseRMS_noscin[:,jj,:,nn,1]**2 - NoiseRMS_noscin[:,jj,:,0,1]**2,axis=(0))**0.5
    RON_RMS_error[:,nn] = (np.nanstd(NoiseRMS[:,jj,:,:,:,nn,1]**2 - NoiseRMS[:,jj,:,:,:,0,1]**2,axis=(0,1,2))**0.5
                     )#/ np.sum(~np.isnan(NoiseRMS[...,nn,1]),axis=(0,1,2,3)))
    noscinRON_RMS_error[:,nn] = (np.nanstd(NoiseRMS_noscin[:,jj,:,nn,1]**2 - NoiseRMS_noscin[:,jj,:,0,1]**2,axis=(0))**0.5
                     )#/ np.sum(~np.isnan(NoiseRMS_noscin[...,nn,1]),axis=(0,1)))

RON_RMS[RON_RMS==0]=np.nan
noscinRON_RMS[noscinRON_RMS==0]=np.nan

plt.figure(figsize=(5,4))
ax = plt.subplot(111)
for mm in range(MM):
    ax.errorbar(RON_list,
                  RON_RMS[mm],
                  yerr=RON_RMS_error[mm],
                  ls='-',marker='x',color=color_list[mm],capsize=5)
for mm in range(MM):
    ax.errorbar(RON_list,
                  noscinRON_RMS[mm],
                  yerr=noscinRON_RMS_error[mm],
                  ls='--',marker='+',color=color_list[mm],capsize=5)
for mm in range(RON_list.shape[0]):
    ax.plot(RON_list[1:],
                 RON_expect[mm][1:],ls=':')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('RMS readout electron per pixel')
plt.ylabel('Noise OPD (nm)')
# plt.ylim(1e-2,2e2)
# plt.xlim(2e-1,2e1)
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', marker='x', label='With scin.'),
    Line2D([0], [0], color='black', linestyle='--', marker='+', label='No scin.'),
    Line2D([0], [0], color='black', linestyle=':', marker='', label='Model'),
    Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label='Phot={:.0e}'.format(N_phot_list[0])),
    Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label='Phot={:.0e}'.format(N_phot_list[1])),
    Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label='Phot={:.0e}'.format(N_phot_list[2])),
    Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label='Phot={:.0e}'.format(N_phot_list[3])),
    Line2D([0], [0], color=color_list[4], linestyle='-', marker='', label='Phot={:.0e}'.format(N_phot_list[4])),
    ]
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(handles=legend_elements,loc='center left', bbox_to_anchor=(1, 0.5))
# plt.legend(handles=legend_elements, loc='best')
plt.savefig('latest-ron-noise')
plt.show()

plt.figure(figsize=(5,4))
ax = plt.subplot(111)
for mm in range(MM):
    ax.errorbar(RON_list,
                 -(RON_RMS[mm] - RON_expect[mm]),
                 yerr=RON_RMS_error[mm],
                 ls='-',marker='x',color=color_list[mm],capsize=5)
for mm in range(MM):
    ax.errorbar(RON_list,
                 -(noscinRON_RMS[mm] - RON_expect[mm]),
                 yerr=noscinRON_RMS_error[mm],
                 ls='--',marker='+',color=color_list[mm],capsize=5)
for mm in range(RON_list.shape[0]):
    ax.plot(RON_list[1:],
                 -(RON_expect[mm][1:] - RON_expect[mm][1:]),ls=':')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('RMS readout electron per pixel')
plt.ylabel('Model error (nm)')
# plt.ylim(1e-2,2e2)
plt.xlim(2e-1,2e1)
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', marker='x', label='With scin.'),
    Line2D([0], [0], color='black', linestyle='--', marker='+', label='No scin.'),
    # Line2D([0], [0], color='black', linestyle=':', marker='', label='Model'),
    Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label='Phot={:.0e}'.format(N_phot_list[0])),
    Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label='Phot={:.0e}'.format(N_phot_list[1])),
    Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label='Phot={:.0e}'.format(N_phot_list[2])),
    Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label='Phot={:.0e}'.format(N_phot_list[3])),
    Line2D([0], [0], color=color_list[4], linestyle='-', marker='', label='Phot={:.0e}'.format(N_phot_list[4])),
    ]
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(handles=legend_elements,loc='center left', bbox_to_anchor=(1, 0.5))
# plt.legend(handles=legend_elements, loc='best')
plt.savefig('latest-ron-noise-estimate-error')
plt.show()

# jj = 2
# oo = 1

# noise_dif_ron = np.zeros((KK,MM,NN),dtype=float)
# noise_dif_ron_error = np.zeros((KK,MM,NN),dtype=float)
# noise_dif_ron_noscin = np.zeros((MM,NN),dtype=float)
# noise_dif_ron_noscin_error = np.zeros((MM,NN),dtype=float)
# for nn in range(NN):
#     noise_dif_ron[...,nn] = np.nanmean(NoiseRMS[:,jj,:,:,:,nn,oo]**2 - NoiseRMS[:,jj,:,:,:,0,1]**2,axis=(0,2))**0.5
#     noise_dif_ron_noscin[:,nn] = np.nanmean(NoiseRMS_noscin[:,jj,:,nn,oo]**2 - NoiseRMS_noscin[:,jj,:,0,1]**2,axis=(0))**0.5
#     noise_dif_ron_error[...,nn] = (np.nanstd(NoiseRMS[:,jj,:,:,:,nn,oo]**2 - NoiseRMS[:,jj,:,:,:,0,1]**2,axis=(0,2))**0.5
#                                     )#/ np.sum(~np.isnan(NoiseRMS[:,jj,:,:,:,nn,oo]),axis=(0,2)))
#     noise_dif_ron_noscin_error[:,nn] = (np.nanstd(NoiseRMS_noscin[:,jj,:,nn,oo]**2 - NoiseRMS_noscin[:,jj,:,0,1]**2,axis=(0))**0.5
#                                             )#/ np.sum(~np.isnan(NoiseRMS_noscin[:,jj,:,nn,oo]),axis=(0)))
# noise_dif_ron[noise_dif_ron==0] = np.nan
# noise_dif_ron_noscin[noise_dif_ron_noscin==0] = np.nan

# mm = 2
# # for mm in range(MM):
# plt.figure(figsize=(5,4))
# ax = plt.subplot(111)
# for nn in range(NN):
#     ax.errorbar(Rytov,
#                   (noise_dif_ron[:,mm,nn] - noise_dif_ron_noscin[mm,nn]),
#                   yerr=(noise_dif_ron_error[:,mm,nn]*2 + noise_dif_ron_noscin_error[mm,nn]**2)**0.5,
#                   marker='o',capsize=5,color=color_list[nn],ls='-')#,ls=ls_list[mm]
#     ax.errorbar(Rytov,
#                   -(noise_dif_ron[:,mm,nn] - noise_dif_ron_noscin[mm,nn]),
#                   yerr=(noise_dif_ron_error[:,mm,nn]**2 + noise_dif_ron_noscin_error[mm,nn]**2)**0.5,
#                   marker='x',capsize=5,color=color_list[nn],ls='--')#,ls=ls_list[mm]
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('$\sigma^2_R$')
# plt.ylabel('Noise with - noise without scintillation (nm)')
# legend_elements = [
#     Line2D([0], [0], color='black', linestyle='-', marker='o', label='Positive'),
#     Line2D([0], [0], color='black', linestyle='--', marker='x', label='Negative'),
#     Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label='RON={:.0e}'.format(RON_list[0])),
#     Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label='RON={:.0e}'.format(RON_list[1])),
#     Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label='RON={:.0e}'.format(RON_list[2])),
#     Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label='RON={:.0e}'.format(RON_list[3])),
#     Line2D([0], [0], color=color_list[4], linestyle='-', marker='', label='RON={:.0e}'.format(RON_list[4])),
#     ]
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(handles=legend_elements,loc='center left', bbox_to_anchor=(1, 0.5))
# # plt.legend(handles=legend_elements, loc='best')
# plt.savefig('latest-ron-noise-dif')
# plt.show()


###############################################################################
def neg_sqrt(x):
    ret = np.zeros_like(x)
    ret[x>=0] = x[x>=0]**0.5
    ret[x<0] = -(-x[x<0])**0.5
    return ret

#################################################################################
# for oo in range(OO):
#     if oo != 1:
#         plt.plot(NoiseRMS[...,1].flatten(),neg_sqrt((NoiseRMS[...,oo]**2 - NoiseRMS[...,1]**2).flatten()),ls='',marker='.',label='Threshold {:.0e}'.format(Threshold_list[oo]))
# plt.hlines([0],np.nanmin(NoiseRMS[...,1]),np.nanmax(NoiseRMS[...,1]),color='k',ls='--')
# plt.plot(np.linspace(1,100),np.linspace(1,100),ls=':',marker='',color='black')
# plt.plot(np.linspace(1,100),-np.linspace(1,100),ls=':',marker='',color='black')
# plt.xscale('symlog')
# plt.yscale('symlog')
# plt.xlabel('Optimised RMS OPD error (nm)')
# plt.ylabel('Change in RMS OPD error (nm)')
# plt.legend()
# # plt.axis('equal')
# plt.show()




###############################################################################

X = []
Y = []
NoiseDiff = np.zeros_like(NoiseRMS[...,1])
for kk in range(KK):
    for ll in range(LL):
        # NoiseDiff[:,:,kk,ll] = neg_sqrt((NoiseRMS[:,:,kk,ll,...,1]**2 - NoiseRMS_noscin[...,1]**2)/NoiseRMS_noscin[...,1]**2)*100
        # NoiseDiff[:,:,kk,ll] = neg_sqrt((NoiseRMS[:,:,kk,ll,...,1]**2 - NoiseRMS_noscin[...,1]**2))
        # plt.plot(NoiseRMS_noscin.flatten(),NoiseRMS[:,:,kk,ll].flatten(),ls='',marker='.')
        X.append(NoiseRMS_noscin[:,:,:,0,1].flatten())
        Y.append((NoiseRMS[:,:,kk,ll,:,0,1].flatten()-NoiseRMS_noscin[:,:,:,0,1].flatten()))
# plt.show()
X = np.asarray(X)
Y = np.asarray(Y)

X = X.flatten()
Y = Y.flatten()

argsort = np.argsort(X)
Y = Y[argsort]
X = X[argsort]

Y = Y[~np.isnan(X)]
X = X[~np.isnan(X)]

X = X[~np.isnan(Y)]
Y = Y[~np.isnan(Y)]

# Define bins
bins = np.linspace(X.min(), X.max(), 21)  # 20 bins
bin_centers = 0.5 * (bins[1:] + bins[:-1])
from scipy.stats import binned_statistic
# Compute statistics
bin_means, _, _ = binned_statistic(X, Y, statistic="mean", bins=bins)
bin_stds, _, _  = binned_statistic(X, Y, statistic="std", bins=bins)
bin_counts, _, _ = binned_statistic(X, Y, statistic="count", bins=bins)

# Standard error of the mean (SEM)
bin_sems = bin_stds# / np.sqrt(bin_counts)
plt.errorbar(bin_centers, (bin_means)/bin_centers*100, yerr=bin_sems/bin_centers*100, fmt="o",capsize=5,label='Photon')

X = []
Y = []
NoiseDiff = np.zeros_like(NoiseRMS[...,1])
for kk in range(KK):
    for ll in range(LL):
        # NoiseDiff[:,:,kk,ll] = neg_sqrt((NoiseRMS[:,:,kk,ll,...,1]**2 - NoiseRMS_noscin[...,1]**2)/NoiseRMS_noscin[...,1]**2)*100
        # NoiseDiff[:,:,kk,ll] = neg_sqrt((NoiseRMS[:,:,kk,ll,...,1]**2 - NoiseRMS_noscin[...,1]**2))
        # plt.plot(NoiseRMS_noscin.flatten(),NoiseRMS[:,:,kk,ll].flatten(),ls='',marker='.')
        X.append(NoiseRMS_noscin[:,:,:,:,1].flatten())
        Y.append((NoiseRMS[:,:,kk,ll,:,:,1].flatten()-NoiseRMS_noscin[:,:,:,:,1].flatten()))
# plt.show()
X = np.asarray(X)
Y = np.asarray(Y)

X = X.flatten()
Y = Y.flatten()

argsort = np.argsort(X)
Y = Y[argsort]
X = X[argsort]

Y = Y[~np.isnan(X)]
X = X[~np.isnan(X)]

X = X[~np.isnan(Y)]
Y = Y[~np.isnan(Y)]

# Define bins
bins = np.linspace(X.min(), X.max(), 21)  # 20 bins
bin_centers = 0.5 * (bins[1:] + bins[:-1])
from scipy.stats import binned_statistic
# Compute statistics
bin_means, _, _ = binned_statistic(X, Y, statistic="mean", bins=bins)
bin_stds, _, _  = binned_statistic(X, Y, statistic="std", bins=bins)
bin_counts, _, _ = binned_statistic(X, Y, statistic="count", bins=bins)

# Standard error of the mean (SEM)
bin_sems = bin_stds# / np.sqrt(bin_counts)
plt.errorbar(bin_centers, (bin_means)/bin_centers*100, yerr=bin_sems/bin_centers*100, fmt="o",capsize=5,label='Photon + Readout')


X = []
Y = []
NoiseDiff = np.zeros_like(NoiseRMS[...,1])
for kk in range(KK):
    for ll in range(LL):
        # NoiseDiff[:,:,kk,ll] = neg_sqrt((NoiseRMS[:,:,kk,ll,...,1]**2 - NoiseRMS_noscin[...,1]**2)/NoiseRMS_noscin[...,1]**2)*100
        # NoiseDiff[:,:,kk,ll] = neg_sqrt((NoiseRMS[:,:,kk,ll,...,1]**2 - NoiseRMS_noscin[...,1]**2))
        # plt.plot(NoiseRMS_noscin.flatten(),NoiseRMS[:,:,kk,ll].flatten(),ls='',marker='.')
        X.append(NoiseRMS_noscin[...].flatten())
        Y.append((NoiseRMS[:,:,kk,ll,...].flatten()-NoiseRMS_noscin[...].flatten()))
# plt.show()
X = np.asarray(X)
Y = np.asarray(Y)

X = X.flatten()
Y = Y.flatten()

argsort = np.argsort(X)
Y = Y[argsort]
X = X[argsort]

Y = Y[~np.isnan(X)]
X = X[~np.isnan(X)]

X = X[~np.isnan(Y)]
Y = Y[~np.isnan(Y)]

# Define bins
bins = np.linspace(X.min(), X.max(), 21)  # 20 bins
bin_centers = 0.5 * (bins[1:] + bins[:-1])
from scipy.stats import binned_statistic
# Compute statistics
bin_means, _, _ = binned_statistic(X, Y, statistic="mean", bins=bins)
bin_stds, _, _  = binned_statistic(X, Y, statistic="std", bins=bins)
bin_counts, _, _ = binned_statistic(X, Y, statistic="count", bins=bins)

# Standard error of the mean (SEM)
bin_sems = bin_stds# / np.sqrt(bin_counts)

# Plot binned mean with error bars
plt.errorbar(bin_centers, (bin_means)/bin_centers*100, yerr=bin_sems/bin_centers*100, fmt="o",capsize=5,label='Photon + Readout + Threshold')


plt.plot(X,X*0,label='0')
plt.legend()

plt.xlabel('Scintillation free noise OPD (nm)')
plt.ylabel('Difference of noise OPD (%)')
plt.ylim(-15,15)
plt.savefig('latest-all-noise-dif')
plt.show()

# plt.plot(X, Y,ls='',marker='.',alpha=0.05)
# plt.plot(X,X*0)
# plt.xlabel('Model noise OPD (nm)')
# plt.ylabel('Error in noise OPD estimation (%)')
# plt.ylim(-15,15)
# plt.savefig('latest-all-noise-dif')
# plt.show()

# slope = np.sum((X - X.mean()) * (Y - Y.mean())) / np.sum((X - X.mean())**2)
# intercept = Y.mean() -slope*X.mean()

# plt.plot(X,slope*X+intercept)

1 - ((X-Y)**2).sum()/((Y-Y.mean())**2).sum()

NoiseDiffSTD = np.nanstd(NoiseDiff,axis=0)


# plt.hist((X-Y)/X*100,density=True)
# plt.xlabel('Noise estimation error (%)')
# plt.ylabel('Occurrence (normalised)')
# plt.show()

# for mm in range(5):
#     plt.pcolor(dor0.astype(str),Rytov.astype(str),np.nanstd(NoiseDiff[...,mm,0],axis=(0,3)).T)
#     plt.colorbar()
#     plt.xlabel('$d/r_0$')
#     plt.ylabel('$\sigma^2_R$')
#     plt.show()
# for mm in range(5):
#     for nn in range(5):
#         plt.pcolor(dor0.astype(str),Rytov.astype(str),np.nanstd(NoiseDiff[...,mm,nn],axis=(0,3)).T)
#         plt.colorbar()
#         plt.xlabel('$d/r_0$')
#         plt.ylabel('$\sigma^2_R$')
#         plt.title('{},{}'.format(mm,nn))
#         plt.show()

# toplot = np.nanmax(np.nanstd(NoiseDiff,axis=(0,3)),axis=(-1,-2)).T
# toplot[3,0] = np.nan
# plt.pcolor(dor0.astype(str),Rytov.astype(str),toplot)
# plt.colorbar()
# plt.xlabel('$d/r_0$')
# plt.ylabel('$\sigma^2_R$')
# plt.show()


# loc = np.where(np.abs(NoiseDiff)>=5)
# for i in range(len(loc[0][:100])):
#     print(r0[loc[0][i]],
#           dor0[loc[1][i]],
#           Rytov[loc[2][i]],
#           H_HA[loc[3][i]],
#           N_phot_list[loc[4][i]],
#           RON_list[loc[5][i]],
#           NoiseDiff[loc][i])

