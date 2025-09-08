# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 16:27:54 2025

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

CONVERT = 500/2/np.pi

threshold_list = [0,1e-4,1e-3,1e-2,1e-1]

ind_list = [(1,2,1,2),#centre
            (1,2,1,2),#r0low
            (1,2,1,2),#dor0high
            (1,2,1,2),#rytovhigh
            (1,2,1,2),]#hhahigh

from matplotlib.lines import Line2D
import matplotlib as mpl

fontsize = 10
figsize = [4,4]
dpi = 600

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

with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\Ultimate_input.pkl', 'rb') as file:
    myvar = pickle.load(file)
    POSSIBLE = myvar['POSSIBLE']
    POSSIBLE_SCREEN = myvar['POSSIBLE_SCREEN']
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
    
###################################################################################################

# with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\Data\Ultimate_output2.pkl', 'rb') as file:
with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\20250626\NoiselessSquarePupilSlopes-og.pkl','rb') as file:
# with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\20250626\NoiselessSquarePupilSlopes-no-ft-bias.pkl','rb') as file:
    myvar = pickle.load(file)
    g_SLOPE = myvar['g_SLOPE']*CONVERT
    Ig_SLOPE = myvar['Ig_SLOPE']*CONVERT
    SHWFS_SLOPE = myvar['SHWFS_SLOPE']*CONVERT
    g_SLOPE_noscin = myvar['g_SLOPE_noscintillation']*CONVERT
    Ig_SLOPE_noscin = myvar['Ig_SLOPE_noscintillation']*CONVERT
    SHWFS_SLOPE_noscin = myvar['SHWFS_SLOPE_noscintillation']*CONVERT

##############################################################################################################################################

    
###################################################################################################

# with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\Data\Ultimate_output2.pkl', 'rb') as file:
# with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\20250626\NoiselessSquarePupilSlopes-og.pkl','rb') as file:
with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\20250626\NoiselessSquarePupilSlopes-no-ft-bias.pkl','rb') as file:
    myvar = pickle.load(file)
    # g_SLOPE_new = myvar['g_SLOPE']*CONVERT
    # Ig_SLOPE_new = myvar['Ig_SLOPE']*CONVERT
    SHWFS_SLOPE_new = myvar['SHWFS_SLOPE']*CONVERT
    # g_SLOPE_noscin_new = myvar['g_SLOPE_noscintillation']*CONVERT
    # Ig_SLOPE_noscin_new = myvar['Ig_SLOPE_noscintillation']*CONVERT
    SHWFS_SLOPE_noscin_new = myvar['SHWFS_SLOPE_noscintillation']*CONVERT
    
LABEL_r0 = np.zeros((II), dtype=object)
for ii in np.arange(II):
    LABEL_r0[ii] = '$r_0$={:.0f}cm'.format(r0[ii]*100)

LABEL_DOR0 = np.zeros((JJ), dtype=object)
for jj in np.arange(JJ):
    LABEL_DOR0[jj] = '$d/r_0$={:.1f}'.format(dor0[jj])
    
LABEL_RYTOV = np.zeros((KK), dtype=object)
for kk in np.arange(KK):
    LABEL_RYTOV[kk] = '$\sigma^2_R=${:.2f}'.format(Rytov[kk])
    
LABEL_HHA = np.zeros((LL), dtype=object)
for ll in np.arange(LL):
    LABEL_HHA[ll] = '$H$={:.0f}km'.format(H_HA[ll]/1000)
    
    
threshold_string = []
for i_threshold in range(len(threshold_list)):
    threshold_string.append('{:.0e}'.format(threshold_list[i_threshold]))

##############################################################################################################################################
    
g_SLOPE[g_SLOPE==0]=np.nan
Ig_SLOPE[Ig_SLOPE==0]=np.nan
g_SLOPE_noscin[g_SLOPE_noscin==0]=np.nan
Ig_SLOPE_noscin[Ig_SLOPE_noscin==0]=np.nan
SHWFS_SLOPE[SHWFS_SLOPE==0]=np.nan
SHWFS_SLOPE_noscin[SHWFS_SLOPE_noscin==0]=np.nan

SHWFS_SLOPE_new[SHWFS_SLOPE_new==0]=np.nan
SHWFS_SLOPE_noscin_new[SHWFS_SLOPE_noscin_new==0]=np.nan

#########################################################################################################################
# show all g-tilt match ig-tilt in no scintillation

# plt.hist(g_SLOPE_noscin.flatten()-Ig_SLOPE_noscin.flatten(),alpha=0.5)
# plt.xlabel('OPD (G - IG) (nm)')
# plt.ylabel('Occurence')
# plt.show()
##############################################################################

# finding appropriate ft
# calculate rms noscin first

g_rms_noscin_new = np.zeros((II,JJ,len(threshold_list)),dtype=float)
Ig_rms_noscin_new = np.zeros_like(g_rms_noscin_new)
g_rms_noscin = np.zeros((II,JJ,len(threshold_list)),dtype=float)
Ig_rms_noscin = np.zeros_like(g_rms_noscin_new)

# g_dif_noscin_new = np.zeros((II,JJ,len(threshold_list)),dtype=float)
# Ig_dif_noscin_new = np.zeros_like(g_rms_noscin_new)
# g_dif_noscin = np.zeros((II,JJ,len(threshold_list)),dtype=float)
# Ig_dif_noscin = np.zeros_like(g_rms_noscin_new)

# for i_threshold in range(5):
i_threshold = 1
A = np.nanmean(np.nanstd(SHWFS_SLOPE[:,:,:,:,i_threshold],axis=(-1,-2)),axis=(-1,-2,-4))
Ae = np.nanstd(np.nanstd(SHWFS_SLOPE[:,:,:,:,i_threshold],axis=(-1,-2)),axis=(-1,-2,-4))
A0 = np.nanmean(np.nanstd(SHWFS_SLOPE_noscin[:,:,i_threshold],axis=(-1,-2)),axis=(0))
A0e = (np.nanstd(SHWFS_SLOPE_noscin[:,:,i_threshold],axis=(-1,-2))[0])/(2*2048-2)**0.5

B = np.nanmean(np.nanstd(Ig_SLOPE,axis=(-1,-2)),axis=(-1,-2,-4))
Be = np.nanstd(np.nanstd(Ig_SLOPE,axis=(-1,-2)),axis=(-1,-2,-4))
B0 = np.nanmean(np.nanstd(Ig_SLOPE_noscin,axis=(-1,-2)),axis=(0))
B0e = (np.nanstd(Ig_SLOPE_noscin,axis=(-1,-2))[0])/(2*2048-2)**0.5

C = np.nanmean(np.nanstd(g_SLOPE,axis=(-1,-2)),axis=(-1,-2,-4))
Ce = np.nanstd(np.nanstd(g_SLOPE,axis=(-1,-2)),axis=(-1,-2,-4))
C0 = np.nanmean(np.nanstd(g_SLOPE_noscin,axis=(-1,-2)),axis=(0))
C0e = (np.nanstd(g_SLOPE_noscin,axis=(-1,-2))[0])/(2*2048-2)**0.5

# A -= B0
# C -= B0
# B -= B0
# A0 -= B0
# C0 -= B0
# B0 -= B0

# case_label = ['FT','IG','G']

# withscin = np.array([A,B,C])
# withscine = np.array([Ae,Be,Ce])
# noscin = np.array([A0,B0,C0])
# noscine = np.array([A0e,B0e,C0e])

# fig = plt.figure(figsize=(5,4))
# ax = plt.subplot(111)
# for jj in range(4):
#     plt.errorbar(case_label,withscin[:,jj],color=color_list[jj],ls='-',yerr=withscine[:,jj],capsize=5,marker='+')
#     plt.errorbar(case_label,noscin[:,jj],color=color_list[jj],ls='--',yerr=noscine[:,jj],capsize=5,marker='x')

# plt.xlabel('Measurement method')
# plt.ylabel('RMS OPD (nm)')
# # plt.xscale('log')
# # plt.yscale('log')
# legend_elements = [
#     Line2D([0], [0], color='black', linestyle='-', marker='+',label='With scintillation'),
#     Line2D([0], [0], color='black', linestyle='--',marker='x', label='No scintillation'),
#     Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label=LABEL_DOR0[0]),
#     Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label=LABEL_DOR0[1]),
#     Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label=LABEL_DOR0[2]),
#     Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label=LABEL_DOR0[3]),
#     ]
# # plt.legend(handles=legend_elements, loc='best')
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(handles=legend_elements,loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('ft-dor0-benchmark')
# plt.show()

plt.errorbar(dor0,A,yerr=Ae,marker='1',label='FT',markersize=20)
plt.errorbar(dor0,B,yerr=Be,marker='2',label='IG',markersize=20)
plt.errorbar(dor0,C,yerr=Ce,marker='3',label='G',markersize=20)
plt.errorbar(dor0,A0,yerr=A0e,ls='--',label='FT, $\sigma^2_R=0$',markersize=20)
plt.xlabel('$d/r_0$')
plt.ylabel('RMS OPD (nm)')
plt.legend()
plt.savefig('ft-dor0-benchmark')
plt.show()

for i_threshold in range(len(threshold_list))[::-1]:
    g_rms_noscin[:,:,i_threshold] = np.nanstd(g_SLOPE_noscin - SHWFS_SLOPE_noscin[:,:,i_threshold],axis=(-1,-2))
    Ig_rms_noscin[:,:,i_threshold] = np.nanstd(Ig_SLOPE_noscin - SHWFS_SLOPE_noscin[:,:,i_threshold],axis=(-1,-2))
    g_rms_noscin_new[:,:,i_threshold] = np.nanstd(g_SLOPE_noscin - SHWFS_SLOPE_noscin_new[:,:,i_threshold],axis=(-1,-2))
    Ig_rms_noscin_new[:,:,i_threshold] = np.nanstd(Ig_SLOPE_noscin - SHWFS_SLOPE_noscin_new[:,:,i_threshold],axis=(-1,-2))
    # g_dif_noscin[:,:,i_threshold] = np.nanmean(g_SLOPE_noscin - SHWFS_SLOPE_noscin[:,:,i_threshold],axis=(-1,-2))
    # Ig_dif_noscin[:,:,i_threshold] = np.nanmean(Ig_SLOPE_noscin - SHWFS_SLOPE_noscin[:,:,i_threshold],axis=(-1,-2))
    # g_dif_noscin_new[:,:,i_threshold] = np.nanmean(g_SLOPE_noscin - SHWFS_SLOPE_noscin_new[:,:,i_threshold],axis=(-1,-2))
    # Ig_dif_noscin_new[:,:,i_threshold] = np.nanmean(Ig_SLOPE_noscin - SHWFS_SLOPE_noscin_new[:,:,i_threshold],axis=(-1,-2))



###############################################################################
# show accuracy of each method in no scintillation. also at different threshold
fig = plt.figure(figsize=(5,4))
ax = plt.subplot(111)
for jj in range(JJ):
    # if jj == 1 or jj == 2:
    #     continue
    ax.errorbar(threshold_string,
             g_rms_noscin[:,jj].mean(0).flatten(),
             yerr=g_rms_noscin[:,jj].mean(0)/(2*g_SLOPE_noscin.shape[-1]*g_SLOPE_noscin.shape[-2] - 2)**0.5,
             marker='+',ls='-',color=color_list[jj],capsize=5)
    ax.errorbar(threshold_string,
             g_rms_noscin_new[:,jj].mean(0).flatten(),
             yerr=g_rms_noscin_new[:,jj].mean(0)/(2*g_SLOPE_noscin.shape[-1]*g_SLOPE_noscin.shape[-2] - 2)**0.5,
             marker='x',ls='--',color=color_list[jj],capsize=5)
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', marker='+', label='Full'),
    Line2D([0], [0], color='black', linestyle='--', marker='x', label='Partial'),
    Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label=LABEL_DOR0[0]),
    Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label=LABEL_DOR0[1]),
    Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label=LABEL_DOR0[2]),
    Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label=LABEL_DOR0[3]),
    ]
# plt.legend(handles=legend_elements, loc='best')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(handles=legend_elements,loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Threshold')
plt.ylabel('RMS OPD error (nm)')
plt.yscale('log')
plt.savefig('threshold-on-ft-noscin')
plt.show()
###############################################################################
# show the above but swap axis
# for i_threshold in range(len(threshold_list)):
#     plt.plot(dor0,g_rms_noscin.mean(0)[:,i_threshold],
#                   # yerr=g_rms_noscin.mean(0)[:,i_threshold]/(2*2048 - 2)**0.5,capsize=5,
#                   color=color_list[i_threshold],ls='-',marker='')
#     plt.plot(dor0,g_rms_noscin_new.mean(0)[:,i_threshold],
#                   # yerr=g_rms_noscin_new.mean(0)[:,i_threshold]/(2*2048 - 2)**0.5,capsize=5,
#                   color=color_list[i_threshold],ls='--',marker='')
# legend_elements = [
#     Line2D([0], [0], color='black', linestyle='-', marker='', label='Full'),
#     Line2D([0], [0], color='black', linestyle='--', marker='', label='Partial'),
#     Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label='{:.0e}'.format(threshold_list[0])),
#     Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label='{:.0e}'.format(threshold_list[1])),
#     Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label='{:.0e}'.format(threshold_list[2])),
#     Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label='{:.0e}'.format(threshold_list[3])),
#     Line2D([0], [0], color=color_list[4], linestyle='-', marker='', label='{:.0e}'.format(threshold_list[4])),
#     ]
# plt.legend(handles=legend_elements, loc='best')
# plt.yscale('log')
# plt.xlabel('$d/r_0$')
# plt.ylabel('RMS OPD error (nm)')
# # plt.xscale('log')
# plt.savefig('dor0-on-ft-noscin')
# plt.show()
###############################################################################
# show the behaviour of the above error level
# for i_threshold in range(len(threshold_list))[::-1]:
#     plt.plot(g_SLOPE_noscin.flatten(),
#              g_SLOPE_noscin.flatten() - SHWFS_SLOPE_noscin[:,:,i_threshold].flatten(),
#              ls='',marker='.',label=threshold_string[i_threshold])
# plt.legend()
# plt.show()

# for i_threshold in range(len(threshold_list))[::-1]:
#     plt.plot(g_SLOPE_noscin.flatten(),
#              g_SLOPE_noscin.flatten() - SHWFS_SLOPE_noscin_new[:,:,i_threshold].flatten(),
#              ls='',marker='.',label=threshold_string[i_threshold])
# plt.legend()
# plt.show()
##############################################################################
# *****************
# in conclusion, to measure wfs error, we should use the full method but with 1e-4 threshold!!!!
# which means the g_rms_noscin[:,:,1], & Ig_rms_noscin = g_rms_noscin
# g_rms[:,:,:,:,1], Ig_rms[:,:,:,:,1]
# ********************
#########################################################################################################################

g_rms = np.zeros((II,JJ,KK,LL,len(threshold_list)),dtype=float)
Ig_rms = np.zeros_like(g_rms)
g_rms_new = np.zeros((II,JJ,KK,LL,len(threshold_list)),dtype=float)
Ig_rms_new = np.zeros_like(g_rms_new)

for i_threshold in range(len(threshold_list)):
    g_rms[...,i_threshold] = np.nanstd(g_SLOPE - SHWFS_SLOPE[:,:,:,:,i_threshold],axis=(-1,-2))
    Ig_rms[...,i_threshold] = np.nanstd(Ig_SLOPE - SHWFS_SLOPE[:,:,:,:,i_threshold],axis=(-1,-2))
    g_rms_new[:,:,:,:,i_threshold] = np.nanstd(g_SLOPE - SHWFS_SLOPE_new[:,:,:,:,i_threshold],axis=(-1,-2))
    Ig_rms_new[:,:,:,:,i_threshold] = np.nanstd(Ig_SLOPE - SHWFS_SLOPE_new[:,:,:,:,i_threshold],axis=(-1,-2))

g_rms[g_rms==0]=np.nan
Ig_rms[Ig_rms==0]=np.nan
g_rms_new[g_rms_new==0]=np.nan
Ig_rms_new[Ig_rms_new==0]=np.nan

# # # ########################################################################
# # # show that the measurement quality is still best for the 64x64 ft with 1e-4 threshold
# # show that the best quality is g-tilt full as discussed prior.
# fig = plt.figure(figsize=(5,4))
# ax = plt.subplot(111)
# for jj in range(JJ):
#     # plt.errorbar(threshold_string,np.nanmean(Ig_rms[:,jj],axis=(0,1,2)),
#     #               yerr=np.nanmean(Ig_rms[:,jj],axis=(0,1,2))/(np.nansum(~np.isnan(Ig_rms[:,jj]),axis=(0,1,2))*2-2)**0.5,
#     #               color=color_list[jj],ls='-',marker='^',capsize=5)
#     ax.errorbar(threshold_string,np.nanmean(g_rms[:,jj],axis=(0,1,2)),
#                   yerr=np.nanmean(g_rms[:,jj],axis=(0,1,2))/(np.nansum(~np.isnan(g_rms[:,jj]),axis=(0,1,2))*2-2)**0.5,
#                   color=color_list[jj],ls='-',marker='+',capsize=5)
#     # plt.errorbar(threshold_string,np.nanmean(Ig_rms_new[:,jj],axis=(0,1,2)),
#     #               yerr=np.nanmean(Ig_rms_new[:,jj],axis=(0,1,2))/(np.nansum(~np.isnan(Ig_rms_new[:,jj]),axis=(0,1,2))*2-2)**0.5,
#     #               color=color_list[jj],ls='--',marker='^',capsize=5)
#     ax.errorbar(threshold_string,np.nanmean(g_rms_new[:,jj],axis=(0,1,2)),
#                   yerr=np.nanmean(g_rms_new[:,jj],axis=(0,1,2))/(np.nansum(~np.isnan(g_rms_new[:,jj]),axis=(0,1,2))*2-2)**0.5,
#                   color=color_list[jj],ls='--',marker='x',capsize=5)
    
#     # ax.errorbar(threshold_string,
#     #           g_rms_noscin[:,jj].mean(0).flatten(),
#     #           yerr=g_rms_noscin[:,jj].mean(0)/(2*g_SLOPE_noscin.shape[-1]*g_SLOPE_noscin.shape[-2] - 2)**0.5,
#     #           marker='',ls=':',color=color_list[jj],capsize=5)
#     # ax.errorbar(threshold_string,
#     #           g_rms_noscin_new[:,jj].mean(0).flatten(),
#     #           yerr=g_rms_noscin_new[:,jj].mean(0)/(2*g_SLOPE_noscin.shape[-1]*g_SLOPE_noscin.shape[-2] - 2)**0.5,
#     #           marker='',ls=':',color=color_list[jj],capsize=5)
# legend_elements = [
#     # Line2D([0], [0], marker='^', color='black', label='IG'),
#     # Line2D([0], [0], marker='v', color='black', label='G'),
#     Line2D([0], [0], color='black', linestyle='-', label='Full',marker='+'),
#     Line2D([0], [0], color='black', linestyle='--', label='Partial',marker='x'),
#     # Line2D([0], [0], color='black', linestyle=':', label='$\sigma^2_R=0$',marker=''),
#     Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label=LABEL_DOR0[0]),
#     Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label=LABEL_DOR0[1]),
#     Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label=LABEL_DOR0[2]),
#     Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label=LABEL_DOR0[3]),
#     ]
# # plt.legend(handles=legend_elements, loc='best')
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(handles=legend_elements,loc='center left', bbox_to_anchor=(1, 0.5))
# plt.xlabel('Threshold')
# plt.ylabel('RMS OPD error (nm)')
# plt.yscale('log')
# plt.savefig('g-all-new')
# plt.show()


# fig = plt.figure(figsize=(5,4))
# ax = plt.subplot(111)
# for jj in range(JJ):
#     plt.errorbar(threshold_string,np.nanmean(Ig_rms[:,jj],axis=(0,1,2)),
#                   yerr=np.nanmean(Ig_rms[:,jj],axis=(0,1,2))/(np.nansum(~np.isnan(Ig_rms[:,jj]),axis=(0,1,2))*2-2)**0.5,
#                   color=color_list[jj],ls='-',marker='+',capsize=5)
#     # plt.errorbar(threshold_string,np.nanmean(g_rms[:,jj],axis=(0,1,2)),
#     #               yerr=np.nanmean(g_rms[:,jj],axis=(0,1,2))/(np.nansum(~np.isnan(g_rms[:,jj]),axis=(0,1,2))*2-2)**0.5,
#     #               color=color_list[jj],ls='-',marker='v',capsize=5)
#     plt.errorbar(threshold_string,np.nanmean(Ig_rms_new[:,jj],axis=(0,1,2)),
#                   yerr=np.nanmean(Ig_rms_new[:,jj],axis=(0,1,2))/(np.nansum(~np.isnan(Ig_rms_new[:,jj]),axis=(0,1,2))*2-2)**0.5,
#                   color=color_list[jj],ls='--',marker='x',capsize=5)
#     # plt.errorbar(threshold_string,np.nanmean(g_rms_new[:,jj],axis=(0,1,2)),
#     #               yerr=np.nanmean(g_rms_new[:,jj],axis=(0,1,2))/(np.nansum(~np.isnan(g_rms_new[:,jj]),axis=(0,1,2))*2-2)**0.5,
#     #               color=color_list[jj],ls='--',marker='v',capsize=5)
    
#     # ax.errorbar(threshold_string,
#     #           g_rms_noscin[:,jj].mean(0).flatten(),
#     #           yerr=g_rms_noscin[:,jj].mean(0)/(2*g_SLOPE_noscin.shape[-1]*g_SLOPE_noscin.shape[-2] - 2)**0.5,
#     #           marker='',ls=':',color=color_list[jj],capsize=5)
#     # ax.errorbar(threshold_string,
#     #           g_rms_noscin_new[:,jj].mean(0).flatten(),
#     #           yerr=g_rms_noscin_new[:,jj].mean(0)/(2*g_SLOPE_noscin.shape[-1]*g_SLOPE_noscin.shape[-2] - 2)**0.5,
#     #           marker='',ls=':',color=color_list[jj],capsize=5)
# legend_elements = [
#     # Line2D([0], [0], marker='^', color='black', label='IG'),
#     # Line2D([0], [0], marker='v', color='black', label='G'),
#     Line2D([0], [0], color='black', linestyle='-', label='Full',marker='+'),
#     Line2D([0], [0], color='black', linestyle='--', label='Partial',marker='x'),
#     # Line2D([0], [0], color='black', linestyle=':', label='$\sigma^2_R=0$',marker=''),
#     Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label=LABEL_DOR0[0]),
#     Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label=LABEL_DOR0[1]),
#     Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label=LABEL_DOR0[2]),
#     Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label=LABEL_DOR0[3]),
#     ]
# # plt.legend(handles=legend_elements, loc='best')
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(handles=legend_elements,loc='center left', bbox_to_anchor=(1, 0.5))
# plt.xlabel('Threshold')
# plt.ylabel('RMS OPD error (nm)')
# plt.yscale('log')
# plt.savefig('ig-all-new')
# plt.show()

# # ########################################################################
# # show that the measurement quality is still best for the 64x64 ft with 1e-4 threshold
# show that the best quality is g-tilt full as discussed prior.
fig = plt.figure(figsize=(5,4))
ax = plt.subplot(111)
for jj in range(JJ):
    if jj == 1 or jj == 2:
        continue
    plt.errorbar(threshold_string,np.nanmean(Ig_rms[:,jj],axis=(0,1,2)),
                  yerr=np.nanstd(Ig_rms[:,jj],axis=(0,1,2)),#/(np.nansum(~np.isnan(Ig_rms[:,jj]),axis=(0,1,2))*2-2)**0.5,
                  color=color_list[jj],ls='-',marker='+',capsize=5)
    ax.errorbar(threshold_string,np.nanmean(g_rms[:,jj],axis=(0,1,2)),
                  yerr=np.nanstd(g_rms[:,jj],axis=(0,1,2)),#/(np.nansum(~np.isnan(g_rms[:,jj]),axis=(0,1,2))*2-2)**0.5,
                  color=color_list[jj],ls='--',marker='x',capsize=5)
    # plt.errorbar(threshold_string,np.nanmean(Ig_rms_new[:,jj],axis=(0,1,2)),
    #               yerr=np.nanmean(Ig_rms_new[:,jj],axis=(0,1,2))/(np.nansum(~np.isnan(Ig_rms_new[:,jj]),axis=(0,1,2))*2-2)**0.5,
    #               color=color_list[jj],ls='--',marker='^',capsize=5)
    # ax.errorbar(threshold_string,np.nanmean(g_rms_new[:,jj],axis=(0,1,2)),
    #               yerr=np.nanmean(g_rms_new[:,jj],axis=(0,1,2))/(np.nansum(~np.isnan(g_rms_new[:,jj]),axis=(0,1,2))*2-2)**0.5,
    #               color=color_list[jj],ls='--',marker='x',capsize=5)
    
    ax.errorbar(threshold_string,
             g_rms_noscin[:,jj].mean(0).flatten(),
             yerr=g_rms_noscin[:,jj].mean(0)/(2*g_SLOPE_noscin.shape[-1]*g_SLOPE_noscin.shape[-2] - 2)**0.5,
             marker='',ls=':',color=color_list[jj],capsize=5)
    # ax.errorbar(threshold_string,
    #          g_rms_noscin_new[:,jj].mean(0).flatten(),
    #          yerr=g_rms_noscin_new[:,jj].mean(0)/(2*g_SLOPE_noscin_new.shape[-1]*g_SLOPE_noscin_new.shape[-2] - 2)**0.5,
    #          marker='',ls=':',color=color_list[jj],capsize=5)
legend_elements = [
    Line2D([0], [0], marker='+', color='black', ls='-', label='IG'),
    Line2D([0], [0], marker='x', color='black', ls='--', label='G'),
    # Line2D([0], [0], color='black', linestyle='-', label='Full',marker='+'),
    # Line2D([0], [0], color='black', linestyle='--', label='Partial',marker='x'),
    Line2D([0], [0], color='black', linestyle=':', label='$\sigma^2_R=0$',marker=''),
    Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label=LABEL_DOR0[0]),
    # Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label=LABEL_DOR0[1]),
    # Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label=LABEL_DOR0[2]),
    Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label=LABEL_DOR0[3]),
    ]
# plt.legend(handles=legend_elements, loc='best')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(handles=legend_elements,loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Threshold')
plt.ylabel('RMS OPD error (nm)')
plt.yscale('log')
plt.savefig('full-cog-all-new')
plt.show()


fig = plt.figure(figsize=(5,4))
ax = plt.subplot(111)
for jj in range(JJ):
    if jj == 1 or jj == 2:
        continue
    # plt.errorbar(threshold_string,np.nanmean(Ig_rms[:,jj],axis=(0,1,2)),
    #               yerr=np.nanmean(Ig_rms[:,jj],axis=(0,1,2))/(np.nansum(~np.isnan(Ig_rms[:,jj]),axis=(0,1,2))*2-2)**0.5,
    #               color=color_list[jj],ls='-',marker='+',capsize=5)
    # plt.errorbar(threshold_string,np.nanmean(g_rms[:,jj],axis=(0,1,2)),
    #               yerr=np.nanmean(g_rms[:,jj],axis=(0,1,2))/(np.nansum(~np.isnan(g_rms[:,jj]),axis=(0,1,2))*2-2)**0.5,
    #               color=color_list[jj],ls='-',marker='v',capsize=5)
    plt.errorbar(threshold_string,np.nanmean(Ig_rms_new[:,jj],axis=(0,1,2)),
                  yerr=np.nanstd(Ig_rms_new[:,jj],axis=(0,1,2)),#/(np.nansum(~np.isnan(Ig_rms_new[:,jj]),axis=(0,1,2))*2-2)**0.5,
                  color=color_list[jj],ls='-',marker='+',capsize=5)
    plt.errorbar(threshold_string,np.nanmean(g_rms_new[:,jj],axis=(0,1,2)),
                  yerr=np.nanstd(g_rms_new[:,jj],axis=(0,1,2)),#/(np.nansum(~np.isnan(g_rms_new[:,jj]),axis=(0,1,2))*2-2)**0.5,
                  color=color_list[jj],ls='--',marker='x',capsize=5)
    
    # ax.errorbar(threshold_string,
    #          g_rms_noscin[:,jj].mean(0).flatten(),
    #          yerr=g_rms_noscin[:,jj].mean(0)/(2*g_SLOPE_noscin.shape[-1]*g_SLOPE_noscin.shape[-2] - 2)**0.5,
    #          marker='',ls=':',color=color_list[jj],capsize=5)
    ax.errorbar(threshold_string,
             g_rms_noscin_new[:,jj].mean(0).flatten(),
             yerr=g_rms_noscin_new[:,jj].mean(0)/(2*g_SLOPE_noscin.shape[-1]*g_SLOPE_noscin.shape[-2] - 2)**0.5,
             marker='',ls=':',color=color_list[jj],capsize=5)
legend_elements = [
    Line2D([0], [0], marker='+', color='black', ls='-', label='IG'),
    Line2D([0], [0], marker='x', color='black', ls='--', label='G'),
    # Line2D([0], [0], color='black', linestyle='-', label='Full',marker='+'),
    # Line2D([0], [0], color='black', linestyle='--', label='Partial',marker='x'),
    Line2D([0], [0], color='black', linestyle=':', label='$\sigma^2_R=0$',marker=''),
    Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label=LABEL_DOR0[0]),
    # Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label=LABEL_DOR0[1]),
    # Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label=LABEL_DOR0[2]),
    Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label=LABEL_DOR0[3]),
    ]
# plt.legend(handles=legend_elements, loc='best')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(handles=legend_elements,loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Threshold')
plt.ylabel('RMS OPD error (nm)')
plt.yscale('log')
plt.savefig('partial-cog-all-new')
plt.show()

###############################################################################
CENTER_I = 1
CENTER_J = 2
CENTER_K = 1
CENTER_L = 2
CENTER_T = 1
###############################################################################

import matplotlib.colors as colors


LABEL_r0 = np.zeros((II), dtype=object)
for ii in np.arange(II):
    LABEL_r0[ii] = '$r_0$={:.0f}cm'.format(r0[ii]*100)

LABEL_DOR0 = np.zeros((JJ), dtype=object)
for jj in np.arange(JJ):
    LABEL_DOR0[jj] = '$d/r_0$={:.1f}'.format(dor0[jj])
    
LABEL_RYTOV = np.zeros((KK), dtype=object)
for kk in np.arange(KK):
    LABEL_RYTOV[kk] = '$\sigma^2_R=${:.2f}'.format(Rytov[kk])
    
LABEL_HHA = np.zeros((LL), dtype=object)
for ll in np.arange(LL):
    LABEL_HHA[ll] = '$H$={:.0f}km'.format(H_HA[ll]/1000)

POSSIBLE_SCREEN_TEMP = np.zeros((4,4,4,4),dtype=bool)
for ii in range(4):
    for jj in range(4):
        for kk in range(4):
            POSSIBLE_SCREEN_TEMP[ii,:,jj,kk] = POSSIBLE_SCREEN[ii,jj,kk]

color_map = plt.cm.get_cmap('inferno')
reversed_color_map = color_map.reversed()

###############################################################################


avg_g = np.nanmean(g_SLOPE - SHWFS_SLOPE[...,CENTER_T,:,:], axis=(-1,-2))
var_g = np.nanvar(g_SLOPE - SHWFS_SLOPE[...,CENTER_T,:,:], ddof=1, axis=(-1,-2))
avg_g[avg_g==0] = np.nan
var_g[var_g==0] = np.nan

avg_ig = np.nanmean(Ig_SLOPE - SHWFS_SLOPE[...,CENTER_T,:,:], axis=(-1,-2))
var_ig = np.nanvar(Ig_SLOPE - SHWFS_SLOPE[...,CENTER_T,:,:], ddof=1, axis=(-1,-2))
avg_ig[avg_ig==0] = np.nan
var_ig[var_ig==0] = np.nan

# import matplotlib.pyplot as plt
# import numpy as np

# Assume var_g and var_ig are available: shape (II, JJ, KK, LL)

# Shared scale from both tilt types
combined_var = np.concatenate([var_g, var_ig], axis=-1)  # shape (II, JJ, KK, 2*LL)
vmax = np.sqrt(np.nanmax(combined_var))
vmin = 0  # RMS variance is non-negative

#############################################################################


# import matplotlib.pyplot as plt
# import numpy as np

# Assume var_g and var_ig are defined (shape II, JJ, KK, LL)

# Shared scale from both tilt types
combined_var = np.concatenate([var_g, var_ig], axis=-1)  # shape (II, JJ, KK, 2*LL)
vmax = np.sqrt(np.nanmax(combined_var))
vmin = 0  # RMS variance is non-negative

color_map = plt.cm.get_cmap('inferno').reversed()
# all measurments! too much info
# ##########################
# # Plot 1: G-Tilt RMS diff #
# ##########################
# fig1, axes1 = plt.subplots(KK, LL, figsize=(6, 6), sharex=True, sharey=True)
# # fig1.suptitle('RMS difference of G-tilt vs SHWFS (nm/lenslet) - no threshold', fontsize=14)

# for kk in range(KK):
#     for ll in range(LL):
#         ax = axes1[KK-kk-1, ll] if KK > 1 else axes1[ll]  # handle 1D axes array if KK=1
#         data = np.sqrt(var_g[:, :, kk, ll]).T

#         # Optional screen mask overlay
#         if (POSSIBLE_SCREEN_TEMP[:, :, kk, ll].T).sum() != 16:
#             ax.pcolor(r0*100, dor0, (1 - POSSIBLE_SCREEN_TEMP[:, :, kk, ll]).T * 0.5,
#                       vmin=0, vmax=1, cmap='Blues')

#         pc = ax.pcolor(r0*100, dor0, data, cmap=color_map,
#                        vmin=0, vmax=vmax, edgecolors='k', linewidths=0.5)

#         if kk == KK-1:
#             ax.set_title(f'{LABEL_HHA[ll]}')

#         if ll == 0:
#             ax.set_ylabel(LABEL_RYTOV[kk] + '\nd/r$_0$')

#         if kk == 0:
#             ax.set_xlabel('r$_0$ (cm)')

# # Colorbar axis to the right
# cax1 = fig1.add_axes([0.92, 0.15, 0.015, 0.7])  # adjust as needed
# fig1.colorbar(pc, cax=cax1)

# fig1.tight_layout(rect=[0, 0, 0.9, 0.95])
# plt.savefig('G_Tilt_RMS_Comparison.pdf')
# plt.show()


# ###########################
# # Plot 2: IG-Tilt RMS diff #
# ###########################
# fig2, axes2 = plt.subplots(KK, LL, figsize=(6,6), sharex=True, sharey=True)

# for kk in range(KK):
#     for ll in range(LL):
#         ax = axes2[KK-kk-1, ll] if KK > 1 else axes2[ll]
#         data = np.sqrt(var_ig[:, :, kk, ll]).T

#         # Optional screen mask overlay
#         if (POSSIBLE_SCREEN_TEMP[:, :, kk, ll].T).sum() != 16:
#             ax.pcolor(r0*100, dor0, (1 - POSSIBLE_SCREEN_TEMP[:, :, kk, ll]).T * 0.5,
#                       vmin=0, vmax=1, cmap='Blues')

#         pc = ax.pcolor(r0*100, dor0, data, cmap=color_map,
#                        vmin=0, vmax=np.nanmax(np.sqrt(var_ig)), edgecolors='k', linewidths=0.5)

#         if kk == KK-1:
#             ax.set_title(f'{LABEL_HHA[ll]}')

#         if ll == 0:
#             ax.set_ylabel(LABEL_RYTOV[kk] + '\nd/r$_0$')

#         if kk == 0:
#             ax.set_xlabel('r$_0$ (cm)')

# # Colorbar axis to the right
# cax2 = fig2.add_axes([0.92, 0.15, 0.015, 0.7])  # adjust as needed
# fig2.colorbar(pc, cax=cax2)

# fig2.tight_layout(rect=[0, 0, 0.9, 0.95])
# plt.savefig('IG_Tilt_RMS_Comparison-another.pdf')


# plt.show()

# #############################################################################

avg = np.nanmean(g_SLOPE - SHWFS_SLOPE[...,CENTER_T,:,:], axis=(-1,-2))
var = np.nanvar(g_SLOPE - SHWFS_SLOPE[...,CENTER_T,:,:], ddof=1, axis=(-1,-2))
avg[avg==0] = np.nan
var[var==0] = np.nan

varmin = np.nanmin(var)
varmax = np.nanmax(var)
avgmin = np.nanmin(avg)
avgmax = np.nanmax(avg)

min_slope = -np.nanmax([np.abs(SHWFS_SLOPE[...,CENTER_T,:,:]),np.abs(Ig_SLOPE),np.abs(g_SLOPE)])
max_slope = np.nanmax([np.abs(SHWFS_SLOPE[...,CENTER_T,:,:]),np.abs(Ig_SLOPE),np.abs(g_SLOPE)])

rms = np.sqrt(np.nanmean((g_SLOPE - SHWFS_SLOPE[...,CENTER_T,:,:])**2, axis=(-1,-2)))

LABEL_r0 = np.zeros((II), dtype=object)
for ii in np.arange(II):
    LABEL_r0[ii] = '$r_0$={:.0f}cm'.format(r0[ii]*100)

LABEL_DOR0 = np.zeros((JJ), dtype=object)
for jj in np.arange(JJ):
    LABEL_DOR0[jj] = '$d/r_0$={:.1f}'.format(dor0[jj])
    
LABEL_RYTOV = np.zeros((KK), dtype=object)
for kk in np.arange(KK):
    LABEL_RYTOV[kk] = '$\sigma^2_R=${:.2f}'.format(Rytov[kk])
    
LABEL_HHA = np.zeros((LL), dtype=object)
for ll in np.arange(LL):
    LABEL_HHA[ll] = '$H$={:.0f}km'.format(H_HA[ll]/1000)



###############################################################################
##########################################################################
"""
█▀▀█░█░▒█░▀█▀░▀█▀░░▀░░█░░░█░█▀▀▄░▀█▀░░░▒█░▄▀░▄▀▀▄░█▀▀▄░█▀▀▀░█░▄░█▀▀▄░█▀▀░█░░░█
█▄▄█░█░▒█░░█░░░█░░░█▀░▀▄█▄▀░█▄▄█░░█░░░░▒█▀▄░░█░░█░█░▒█░█░▀▄░█▀▄░█▄▄█░█▀▀░▀▄█▄▀
█░░░░░▀▀▀░░▀░░░▀░░▀▀▀░░▀░▀░░▀░░▀░░▀░░░░▒█░▒█░░▀▀░░▀░░▀░▀▀▀▀░▀░▀░▀░░▀░▀▀▀░░▀░▀░
"""
##########################################################################
###############################################################################

Ig_SLOPE[Ig_SLOPE==0]=np.nan
g_SLOPE[g_SLOPE==0]=np.nan
SHWFS_SLOPE[SHWFS_SLOPE==0]=np.nan


###############################################################################

data = np.nanstd(Ig_SLOPE - SHWFS_SLOPE[...,CENTER_T,:,:],axis=(-1,-2))
data[data==0] = np.nan

V = data


import itertools

# def plot_all_cross_sections(V, w, x, y, z):
#     param_arrays = [w*100, x, y, z/1000]
#     param_labels = ['r$_0$ (cm)', 'd/r$_0$', 'Rytov', 'H$_{HA}$ (km)']
    
#     fig, axs = plt.subplots(2, 2, figsize=(6,6))
#     axs = axs.flatten()

#     for axis in range(4):
#         ax = axs[axis]
#         varying_param = param_arrays[axis]
#         other_axes = [i for i in range(4) if i != axis]
#         other_ranges = [range(V.shape[i]) for i in other_axes]

#         for idx_combo in itertools.product(*other_ranges):
#             slicer = [slice(None)] * 4
#             for a, val in zip(other_axes, idx_combo):
#                 slicer[a] = val
#             v_line = V[tuple(slicer)]
#             label = ', '.join(f"{param_labels[a]}={val}" for a, val in zip(other_axes, idx_combo))
#             ax.plot(varying_param, v_line, alpha=0.5, label=label,marker='o')
#         if axis == 1:
#             ax.errorbar(varying_param,Ig_rms_noscin[0,:,CENTER_T],yerr=Ig_rms_noscin[0,:,CENTER_T]/np.sqrt(2*1024-2),
#                         ls=':',color='k',capsize=5)
#         # else:
#         #     ax.hlines(g_rms_noscin,varying_param.min(),varying_param.max(),ls=':',color='k')
#         #     ax.hlines(Ig_rms_noscin,varying_param.min(),varying_param.max(),ls='-',color='k')
#         else:
#             for i in range(4):
#                 ax.errorbar(varying_param,Ig_rms_noscin[:,i,CENTER_T],
#                           yerr=Ig_rms_noscin[:,i,CENTER_T]/np.sqrt(2*1024-2),
#                           ls=':',color='k',capsize=5)
                
#         # ax.set_title(f"cros section : {param_labels[axis]}")
#         ax.set_xlabel(f"{param_labels[axis]}")
#         ax.set_ylabel("OPD (nm)")
#         # ax.set_ylim(0,6)
#         ax.grid(True)
#         ax.tick_params(axis='x', rotation=45)
#         # if axis == 0:
#         #     ax.legend(fontsize='x-small', loc='best')

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.95)
#     # plt.suptitle('Cross sections of measured RMS OPD difference between IG and FT (nm) - no threshold')
#     plt.savefig('cross-section-measure-ig-ft.pdf',bbox_inches='tight')
#     plt.show()
#     return

# # cross sections of all things # too much info
# plot_all_cross_sections(V, r0, dor0, Rytov, H_HA)

#########################################################################################
######################################################################################
#########################################################################################
######################################################################################
#########################################################################################
######################################################################################
#########################################################################################
######################################################################################

# # cross section around the selected values. kinda nice. simple. but lack general story
# CENTER_I = 1
# CENTER_J = 2
# CENTER_K = 1
# CENTER_L = 2
# CENTER_T = 1
# ymin=0.5
# ymax=2.5

# fig, axs = plt.subplots(2,2,layout='constrained')

# axs[0,0].errorbar(r0*100,g_rms_noscin[...,CENTER_T].mean(-1),
#              yerr=g_rms_noscin[...,CENTER_T].mean(-1)/(2*2048 - 2)**0.5,
#              ls='-',marker='o',capsize=5,label='No scintillation')
# axs[0,0].errorbar(r0*100,
#              Ig_rms[:,CENTER_J,CENTER_K,CENTER_L,CENTER_T],
#              yerr=Ig_rms[:,CENTER_J,CENTER_K,CENTER_L,CENTER_T]/(2*2048 - 2)**0.5,
#              ls='-',marker='o',capsize=5,label='With scintillation')
# axs[0,0].set_ylim(ymin,ymax)
# # axs[0,0].set_yscale('log')
# # axs[0,0].set_xscale('log')
# axs[0,0].set_xlabel('$r_0$ (cm)')
# axs[0,0].set_ylabel('RMS OPD error (nm)')

# axs[0,1].errorbar(dor0,g_rms_noscin[...,CENTER_T].mean(0),
#              yerr=g_rms_noscin[...,CENTER_T].mean(0)/(2*2048 - 2)**0.5,
#              ls='-',marker='o',capsize=5,label='No scintillation')
# axs[0,1].errorbar(dor0,
#              Ig_rms[CENTER_I,:,CENTER_K,CENTER_L,CENTER_T],
#              yerr=Ig_rms[CENTER_I,:,CENTER_K,CENTER_L,CENTER_T]/(2*2048 - 2)**0.5,
#              ls='-',marker='o',capsize=5,label='With scintillation')
# axs[0,1].set_ylim(ymin,ymax)
# # axs[0,1].set_yscale('log')
# # axs[0,1].set_xscale('log')
# axs[0,1].set_xlabel('$d/r_0$')
# axs[0,1].set_ylabel('RMS OPD error (nm)')


# axs[1,0].errorbar(Rytov,g_rms_noscin[...,CENTER_T].mean(-1),
#              yerr=g_rms_noscin[...,CENTER_T].mean(-1)/(2*2048 - 2)**0.5,
#              ls='-',marker='o',capsize=5,label='No scintillation')
# axs[1,0].errorbar(Rytov,
#              Ig_rms[CENTER_I,CENTER_J,:,CENTER_L,CENTER_T],
#              yerr=Ig_rms[CENTER_I,CENTER_J,:,CENTER_L,CENTER_T]/(2*2048 - 2)**0.5,
#              ls='-',marker='o',capsize=5,label='With scintillation')
# axs[1,0].set_ylim(ymin,ymax)
# # axs[1,0].set_yscale('log')
# # axs[1,0].set_xscale('log')
# axs[1,0].set_xlabel('$\sigma^2_R$')
# axs[1,0].set_ylabel('RMS OPD error (nm)')

# axs[1,1].errorbar(H_HA/1000,g_rms_noscin[...,CENTER_T].mean(-1),
#              yerr=g_rms_noscin[...,CENTER_T].mean(-1)/(2*2048 - 2)**0.5,
#              ls='-',marker='o',capsize=5,label='No scintillation')
# axs[1,1].errorbar(H_HA/1000,
#              Ig_rms[CENTER_I,CENTER_J,CENTER_K,:,CENTER_T],
#              yerr=Ig_rms[CENTER_I,CENTER_J,CENTER_K,:,CENTER_T]/(2*2048 - 2)**0.5,
#              ls='-',marker='o',capsize=5,label='With scintillation')
# axs[1,1].set_ylim(ymin,ymax)
# # axs[1,1].set_yscale('log')
# # axs[1,1].set_xscale('log')
# axs[1,1].set_xlabel('$H_{HA} (km)$')
# axs[1,1].set_ylabel('RMS OPD error (nm)')

# plt.savefig('ig-tilt-rms-scintillation-all-variable')
# plt.show()

###############################################################################
# dor0_string=['0.5','1','2','3']
Rytov_string=['0.01','0.03','0.1','0.3']
# plt.figure(figsize=(5,4))
# plt.pcolor(dor0_string,Rytov_string,np.nanmean(Ig_rms[...,CENTER_T],axis=(0,3)).T)
# plt.xlabel('$d/r_0$')
# plt.ylabel('$\sigma^2_R$')
# plt.colorbar()
# plt.savefig('ig-tilt-rms-dor0-rytov')
# plt.show()
###############################################################################

# # # average of cross section separate d/r0
# # # very nice but.. i want to show another thing instead
# CENTER_I = 1
# CENTER_J = 2
# CENTER_K = 1
# CENTER_L = 2
# CENTER_T = 1
# ymin=0
# ymax=3

# fig, axs = plt.subplots(2,2,layout='constrained')

# axs[0,1].errorbar(dor0,g_rms_noscin[...,CENTER_T].mean(0),
#              yerr=g_rms_noscin[...,CENTER_T].mean(0)/(2*2048 - 2)**0.5,
#              ls=':',marker='',capsize=5,label='No scintillation',color='k')
# axs[0,1].errorbar(dor0,
#              np.nanmean(Ig_rms[...,CENTER_T],axis=(0,2,3)),
#              yerr=np.nanmean(Ig_rms[...,CENTER_T]**2/(2*2048 - 2),axis=(0,2,3))**0.5,
#              ls='-',marker='',capsize=5,label='With scintillation',color='k')

# for jj in range(JJ):
#     axs[0,0].errorbar(r0*100,g_rms_noscin[:,jj,CENTER_T],
#                  yerr=g_rms_noscin[:,jj,CENTER_T]/(2*2048 - 2)**0.5,
#                  ls=':',marker='o',capsize=5,label='No scintillation',color=color_list[jj])
#     axs[1,0].errorbar(Rytov,g_rms_noscin[:,jj,CENTER_T],
#                  yerr=g_rms_noscin[:,jj,CENTER_T]/(2*2048 - 2)**0.5,
#                  ls=':',marker='o',capsize=5,label='No scintillation',color=color_list[jj])
#     axs[1,1].errorbar(H_HA/1000,g_rms_noscin[:,jj,CENTER_T],
#                  yerr=g_rms_noscin[:,jj,CENTER_T]/(2*2048 - 2)**0.5,
#                  ls=':',marker='o',capsize=5,label='No scintillation',color=color_list[jj])
    
#     axs[0,0].errorbar(r0*100,
#                  np.nanmean(Ig_rms[:,jj,:,:,CENTER_T],axis=(1,2)),
#                  yerr=np.nanmean(Ig_rms[:,jj,:,:,CENTER_T]**2/(2*2048 - 2),axis=(1,2))**0.5,
#                  ls='-',marker='o',capsize=5,label='With scintillation',color=color_list[jj])
#     axs[1,0].errorbar(Rytov,
#                  np.nanmean(Ig_rms[:,jj,:,:,CENTER_T],axis=(0,2)),
#                  yerr=np.nanmean(Ig_rms[:,jj,:,:,CENTER_T]**2/(2*2048 - 2),axis=(0,2))**0.5,
#                  ls='-',marker='o',capsize=5,label='With scintillation',color=color_list[jj])
#     axs[1,1].errorbar(H_HA/1000,
#                  np.nanmean(Ig_rms[:,jj,:,:,CENTER_T],axis=(0,1)),
#                  yerr=np.nanmean(Ig_rms[:,jj,:,:,CENTER_T]**2/(2*2048 - 2),axis=(0,1))**0.5,
#                  ls='-',marker='o',capsize=5,label='With scintillation',color=color_list[jj])

#     axs[0,1].errorbar(dor0[jj],
#                   (g_rms_noscin[:,jj,CENTER_T].mean(0)),
#                   yerr=(g_rms_noscin[:,jj,CENTER_T].mean(0)**2/(2*2048 - 2))**0.5,
#                   ls='',marker='o',capsize=5,label='With scintillation',color=color_list[jj])
#     axs[0,1].errorbar(dor0[jj],
#                  np.nanmean(Ig_rms[:,jj,:,:,CENTER_T]),
#                  yerr=np.nanmean(Ig_rms[:,jj,:,:,CENTER_T]**2/(2*2048 - 2))**0.5,
#                  ls='',marker='o',capsize=5,label='With scintillation',color=color_list[jj])


# axs[0,0].set_ylim(ymin,ymax)
# axs[0,0].set_xlabel('$r_0$ (cm)')
# axs[0,0].set_ylabel('RMS OPD error (nm)')
# axs[0,1].set_ylim(ymin,ymax)
# axs[0,1].set_xlabel('$d/r_0$')
# axs[0,1].set_ylabel('RMS OPD error (nm)')
# axs[1,0].set_ylim(ymin,ymax)
# axs[1,0].set_xlabel('$\sigma^2_R$')
# axs[1,0].set_ylabel('RMS OPD error (nm)')
# axs[1,1].set_ylim(ymin,ymax)
# axs[1,1].set_xlabel('$H_{HA} (km)$')
# axs[1,1].set_ylabel('RMS OPD error (nm)')

# plt.savefig('ig-tilt-rms-scintillation-all-variable')
# plt.show()
# ###############################################################################
# # very very nice looking. easy to tell the story. but color bar is hard to tell values apart.
# toplot = np.swapaxes(Ig_rms[...,1],0,1)
# # vmin=np.nanmin(toplot)
# # vmax=np.nanmax(toplot)
# vmax=2.66
# vmin=0.56
# order = [[0,0],[0,1],[0,2],[1,1],[1,2],[2,2]]
# count = 0
# fig, axes = plt.subplots(3, 3, figsize=(6,6),sharex=False,sharey=False)
# for axis1 in range(len(toplot.shape)):
#     for axis2 in range(len(toplot.shape)):
#         if axis2 <= axis1:
#             continue
#         else:
#             if axis1 == 1:
#                 x = (r0*100).astype(str)
#                 xlabel = '$r_0$ (cm)'
#             if axis1 == 0:
#                 x = (dor0).astype(str)
#                 xlabel = '$d/r_0$'
#             if axis1 == 2:
#                 x = (Rytov).astype(str)
#                 xlabel = '$\sigma^2_R$'
#             if axis1 == 3:
#                 x = (H_HA/1000).astype(str)
#                 xlabel = '$H$ (km)'
#             if axis2 == 1:
#                 y = (r0*100).astype(str)
#                 ylabel = '$r_0$ (cm)'
#             if axis2 == 0:
#                 y = (dor0).astype(str)
#                 ylabel = '$d/r_0$'
#             if axis2 == 2:
#                 y = (Rytov).astype(str)
#                 ylabel = '$\sigma^2_R$'
#             if axis2 == 3:
#                 y = (H_HA/1000).astype(str)
#                 ylabel = '$H$ (km)'
#             # print(xlabel,ylabel)
#             data = np.nanmean(toplot,axis=tuple(k for k in range(toplot.ndim) if k not in (axis1,axis2)))
#             axes[order[count][1],order[count][0]].pcolor(x,y,data.T,vmin=vmin,vmax=vmax)
#             axes[order[count][1],order[count][0]].set_xlabel(xlabel)
#             axes[order[count][1],order[count][0]].set_ylabel(ylabel)
#             # for i in range(4):
#             #     for j in range(4):
#             #         print(x[i],y[j],data[i,j])
#             count += 1
# axes[0, 1].axis("off")
# axes[0, 2].axis("off")
# axes[1, 2].axis("off")
# # plt.axis('square')

# for i in range(3):
#     for j in range(3):
        
#         # Hide inner subplot axes
#         if i < 2:   # hide x-axis except bottom row
#             axes[i, j].set_xticklabels([])
#             axes[i, j].set_xlabel("")
#         if j > 0:   # hide y-axis except first column
#             axes[i, j].set_yticklabels([])
#             axes[i, j].set_ylabel("")
# cmap = plt.cm.viridis

# # Normalize values between vmin and vmax
# norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# # Create a ScalarMappable for the colorbar
# sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# plt.tight_layout()
# # Add colorbar spanning all subplots
# fig.colorbar(sm, ax=axes.ravel().tolist(), orientation="vertical")

# plt.show()

###############################################################################

x = dor0
y = np.nanmean(Ig_rms[:,:,:,:,1],axis=(0,2,3))
yerr = np.nanstd(Ig_rms[:,:,:,:,1],axis=(0,2,3)),#/(np.sum(~np.isnan(Ig_rms[:,:,:,:,1]),axis=(0,2,3)))**0.5

x0 = dor0
y0 = np.nanmean(g_rms_noscin[:,:,1],axis=(0))
yerr0 = np.nanmean(g_rms_noscin[:,:,1],axis=(0))/(2048*2-2)**0.5

x2 = dor0
y2 = np.nanmean(g_rms[:,:,:,:,1],axis=(0,2,3))
yerr2 = np.nanstd(g_rms[:,:,:,:,1],axis=(0,2,3)),#/(np.sum(~np.isnan(g_rms[:,:,:,:,1]),axis=(0,2,3)))**0.5
plt.errorbar(x,y,yerr=yerr,ls='-',marker='+',capsize=5,label='IG')
plt.errorbar(x2,y2,yerr=yerr2,ls='--',marker='x',capsize=5,label='G')
plt.errorbar(x0,y0,yerr=yerr0,ls=':',marker='.',capsize=5,label='$\sigma^2_R=0$')
# plt.xscale('log')
plt.ylim(0.1,200)
plt.yscale('log')
plt.legend()
plt.xlabel('$d/r_0$')
plt.ylabel('RMS OPD error (nm)')
plt.savefig('ig-dor0-analysis')
plt.show()

fig = plt.figure(figsize=(5,4))
ax = plt.subplot(111)
for jj in range(JJ):
    # if jj == 0 or jj == 2:
    #     continue
    color = dor0#(dor0).astype(str)
    x = r0*100
    y = np.nanmean(Ig_rms[:,jj,:,:,1],axis=(1,2))
    yerr = np.nanstd(Ig_rms[:,jj,:,:,1],axis=(1,2)),#/(np.sum(~np.isnan(Ig_rms[:,jj,:,:,1]),axis=(1,2)))**0.5
    ax.errorbar(x,y,yerr=yerr,color=color_list[jj],ls='-',marker='+',capsize=5)#',label='$d/r_0$={:}'.format(((dor0).astype(str))[jj]))
    y2 = np.nanmean(g_rms[:,jj,:,:,1],axis=(1,2))
    yerr2 = np.nanstd(g_rms[:,jj,:,:,1],axis=(1,2)),#/(np.sum(~np.isnan(g_rms[:,jj,:,:,1]),axis=(1,2)))**0.5
    ax.errorbar(x,y2,yerr=yerr2,color=color_list[jj],ls='--',marker='x',capsize=5)#,label='$d/r_0$={:}'.format(((dor0).astype(str))[jj]))
# plt.xscale('log')
plt.yscale('log')
# plt.legend()
legend_elements = [
    # Line2D([0], [0], marker='^', color='black', label='IG'),
    # Line2D([0], [0], marker='v', color='black', label='G'),
    Line2D([0], [0], color='black', linestyle='-', label='IG',marker='+'),
    Line2D([0], [0], color='black', linestyle='--', label='G',marker='x'),
    Line2D([0], [0], color='black', linestyle=':', label='$\sigma^2_R$=0',marker=''),
    Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label=LABEL_DOR0[0]),
    Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label=LABEL_DOR0[1]),
    Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label=LABEL_DOR0[2]),
    Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label=LABEL_DOR0[3]),
    ]
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(handles=legend_elements,loc='center left', bbox_to_anchor=(1, 0.5))
for jj in range(JJ):
    if jj == 0 or jj == 2:
        continue
    plt.hlines([np.nanmean(g_rms_noscin[:,jj,1])],2,11,ls=':',color=color_list[jj])
plt.xlabel('$r_0$ (cm)')
plt.ylabel('RMS OPD error (nm)')
plt.ylim(0.1,200)
plt.savefig('ig-r0-analysis')
plt.show()

fig = plt.figure(figsize=(5,4))
ax = plt.subplot(111)
for jj in range(JJ):
    # if jj == 0 or jj == 2:
    #     continue
    color = dor0#(dor0).astype(str)
    x = Rytov
    y = np.append(Ig_rms_noscin[:,jj,1].mean(0),np.nanmean(Ig_rms[:,jj,:,:,1],axis=(0,2)))
    yerr = np.append(0,
                     np.nanstd(Ig_rms[:,jj,:,:,1],axis=(0,2)))#/(np.sum(~np.isnan(Ig_rms[:,jj,:,:,1]),axis=(0,2)))**0.5)
    plt.errorbar(np.append(0,Rytov_string),y,yerr=yerr,color=color_list[jj],ls='-',marker='+',capsize=5)#,label='$d/r_0$={:}'.format(((dor0).astype(str))[jj]))
    y2 = np.append(g_rms_noscin[:,jj,1].mean(0),np.nanmean(g_rms[:,jj,:,:,1],axis=(0,2)))
    yerr2 = np.append(0,
                      np.nanstd(g_rms[:,jj,:,:,1],axis=(0,2)))#/(np.sum(~np.isnan(Ig_rms[:,jj,:,:,1]),axis=(0,2)))**0.5)
    plt.errorbar(np.append(0,Rytov_string),y2,yerr=yerr2,color=color_list[jj],ls='--',marker='x',capsize=5)#,label='$d/r_0$={:}'.format(((dor0).astype(str))[jj]))
# plt.xscale('log')
plt.yscale('log')
# plt.legend()
legend_elements = [
    # Line2D([0], [0], marker='^', color='black', label='IG'),
    # Line2D([0], [0], marker='v', color='black', label='G'),
    Line2D([0], [0], color='black', linestyle='-', label='IG',marker='+'),
    Line2D([0], [0], color='black', linestyle='--', label='G',marker='x'),
    Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label=LABEL_DOR0[0]),
    Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label=LABEL_DOR0[1]),
    Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label=LABEL_DOR0[2]),
    Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label=LABEL_DOR0[3]),
    ]
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(handles=legend_elements,loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('$\sigma^2_R$')
plt.ylabel('RMS OPD error (nm)')
plt.ylim(0.1,200)
plt.savefig('ig-rytov-analysis')
plt.show()

fig = plt.figure(figsize=(5,4))
ax = plt.subplot(111)
for jj in range(JJ):
    # if jj == 0 or jj == 2:
    #     continue
    color = dor0#(dor0).astype(str)
    x = H_HA/1000
    y = np.nanmean(Ig_rms[:,jj,:,:,1],axis=(0,1))
    yerr = np.nanstd(Ig_rms[:,jj,:,:,1],axis=(0,1))#/(np.sum(~np.isnan(Ig_rms[:,jj,:,:,1]),axis=(0,1)))**0.5
    y2 = np.nanmean(g_rms[:,jj,:,:,1],axis=(0,1))
    yerr2 = np.nanstd(g_rms[:,jj,:,:,1],axis=(0,1))#/(np.sum(~np.isnan(g_rms[:,jj,:,:,1]),axis=(0,1)))**0.5
    plt.errorbar(x,y,yerr=yerr,color=color_list[jj],ls='-',marker='+',capsize=5)#,label='$d/r_0$={:}'.format(((dor0).astype(str))[jj]))
    plt.errorbar(x,y2,yerr=yerr2,color=color_list[jj],ls='--',marker='x',capsize=5)#,label='$d/r_0$={:}'.format(((dor0).astype(str))[jj]))
# plt.xscale('log')
plt.yscale('log')
# plt.legend()
legend_elements = [
    # Line2D([0], [0], marker='^', color='black', label='IG'),
    # Line2D([0], [0], marker='v', color='black', label='G'),
    Line2D([0], [0], color='black', linestyle='-', label='IG',marker='+'),
    Line2D([0], [0], color='black', linestyle='--', label='G',marker='x'),
    Line2D([0], [0], color='black', linestyle=':', label='$\sigma^2_R$=0',marker=''),
    Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label=LABEL_DOR0[0]),
    Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label=LABEL_DOR0[1]),
    Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label=LABEL_DOR0[2]),
    Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label=LABEL_DOR0[3]),
    ]
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(handles=legend_elements,loc='center left', bbox_to_anchor=(1, 0.5))
for jj in range(JJ):
    # if jj == 0 or jj == 2:
    #     continue
    plt.hlines([np.nanmean(g_rms_noscin[:,jj,1])],5,20,ls=':',color=color_list[jj])
plt.xlabel('$H$ (km)')
plt.ylabel('RMS OPD error (nm)')
plt.ylim(0.1,200)
plt.savefig('ig-hha-analysis')
plt.show()


###############################################################################
ii = 3
jj = 3
kk = 3
ll = 1
fig = plt.figure(figsize=(5,4))
ax = plt.subplot(111)
plt.plot(SHWFS_SLOPE[ii,jj,kk,ll,1,...].flatten(),
         (g_SLOPE[ii,jj,kk,ll]-SHWFS_SLOPE[ii,jj,kk,ll,1,...]).flatten(),
         ls='',marker='.',label='G - FT',color=color_list[1],alpha=0.2)
plt.plot(SHWFS_SLOPE[ii,jj,kk,ll,1,...].flatten(),
         (Ig_SLOPE[ii,jj,kk,ll]-SHWFS_SLOPE[ii,jj,kk,ll,1,...]).flatten(),
         ls='',marker='.',label='IG - FT',color=color_list[0],alpha=0.2)
legend_elements = [
    Line2D([0], [0], color=color_list[0], linestyle='', marker='.', label='IG'),
    Line2D([0], [0], color=color_list[1], linestyle='', marker='.', label='G'),
    ]
# plt.legend(handles=legend_elements, loc='best')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(handles=legend_elements,loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('FT OPD (nm)')
plt.ylabel('OPD error (nm)')
plt.savefig('g-ig-all-new.png')
plt.show()



# for ii in range (4):
#     for jj in range(4):
#         for kk in range(4):
#             for ll in range(4):
                
#                 if np.isnan(SHWFS_SLOPE[ii,jj,kk,ll,1,...].sum()):
#                     continue

fig = plt.figure(figsize=(5,4))
ax = plt.subplot(111)
plt.plot(SHWFS_SLOPE[ii,jj,kk,ll,1,...].flatten(),
         (Ig_SLOPE[ii,jj,kk,ll]-SHWFS_SLOPE[ii,jj,kk,ll,1,...]).flatten(),
         ls='',marker='.',label='IG - FT',alpha=0.2)
plt.plot(SHWFS_SLOPE_noscin[ii,jj,1,...].flatten(),
         (Ig_SLOPE_noscin[ii,jj]-SHWFS_SLOPE_noscin[ii,jj,1,...]).flatten(),
         ls='',marker='.',label='$sigma^2_R=0$',alpha=0.2)
legend_elements = [
    Line2D([0], [0], color=color_list[0], linestyle='', marker='.', label='$\sigma^2_R \\neq 0$'),
    Line2D([0], [0], color=color_list[1], linestyle='', marker='.', label='$\sigma^2_R = 0$'),
    ]
# plt.legend(handles=legend_elements, loc='best')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(handles=legend_elements,loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('FT OPD (nm)')
plt.ylabel('OPD error (nm)')
plt.xlim(-1500,1500)
plt.ylim(-15,15)
plt.savefig('ig-all-new.png')
plt.show()








new_possible_screen = np.zeros((II,JJ,KK,LL),dtype=float)
for ii in range(II):
    for kk in range(KK):
        for ll in range(LL):
            new_possible_screen[ii,:,kk,ll] = POSSIBLE_SCREEN[ii,kk,ll]
new_possible_screen[new_possible_screen==True] = np.nan
new_possible = np.array(POSSIBLE,dtype=float)
new_possible[new_possible==True] = np.nan

plt.subplots(4,4,sharex=True,sharey=True,squeeze=False,figsize=(6,6))
for kk in np.arange(KK):
    for ll in np.arange(LL):
        plt.subplot(4,4,LL*(KK - kk - 1) + ll + 1)
        good = np.isnan(new_possible[:,:,kk,ll]) * np.isnan(new_possible_screen[:,:,kk,ll])
        plt.pcolor(r0*100, dor0, good.T*0.5,
                    edgecolors='k',linewidths=0.5,cmap='Greens',
                    vmin=0,vmax=1)
        plt.pcolor(r0*100, dor0, 0.5-(new_possible[:,:,kk,ll]).T*0.5,
                    edgecolors='k',linewidths=0.5,cmap='Reds',
                    vmin=0,vmax=1)
        if kk == 0:
            plt.xlabel('$r_0$ (cm)\n' + '$H$='+'{:.0f}km'.format(H_HA[ll]/1000))
        if ll == 0:
            plt.ylabel(LABEL_RYTOV[kk] + '\n$d/r_0$')
        # if ll == (LL - 1):
            # plt.colorbar()
        plt.pcolor(r0*100, dor0, 0.5-(new_possible_screen[:,:,kk,ll]).T*0.5,
                    edgecolors='k',linewidths=0.5,cmap='Blues',
                    vmin=0,vmax=1)
        
        
plt.savefig('ch4-excluded-condition')
plt.show()
plt.clf()

count = 0
for ii in range(II):
    for jj in range(JJ):
        for kk in range(KK):
            for ll in range(LL):
                good = np.isnan(new_possible[ii,jj,kk,ll]) * np.isnan(new_possible_screen[ii,jj,kk,ll])
                if good:
                    if (count + 1) % 3 == 0:
                        print('{:.0f} & {:.1f} & {:.2f} & {:.0f} \\\\'.format(r0[ii]*100,dor0[jj],Rytov[kk],H_HA[ll]/1000))
                    else:
                        print('{:.0f} & {:.1f} & {:.2f} & {:.0f} &'.format(r0[ii]*100,dor0[jj],Rytov[kk],H_HA[ll]/1000))
                    count += 1

fresnel_to_d = np.zeros((II,JJ,KK,LL),dtype=float)
for ii in range(II):
    for jj in range(JJ):
        for kk in range(KK):
            for ll in range(LL):
                d = delta[ii,jj]*64
                fresnel = np.sqrt(wvl*H_HA[kk])
                fresnel_to_d[ii,jj,kk,ll] = fresnel/d
                
marker_list=['^','>','v','<']
ls_list=['-','--','-.',':']


# for ll in range(LL):
#     for jj in range(JJ):
#         # for ii in range(II):
#         for kk in range(KK):
#             plt.plot(fresnel_to_d[:,jj,kk,ll].flatten(),Ig_rms[:,jj,kk,ll,1].flatten(),color=color_list[jj],ls=ls_list[ll],marker='.')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.show()



