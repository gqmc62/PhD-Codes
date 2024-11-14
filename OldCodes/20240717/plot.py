
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

with open('Ultimate_output2.pkl', 'rb') as file:
    myvar = pickle.load(file)
    g_SLOPE = myvar['g_SLOPE']
    Ig_SLOPE = myvar['Ig_SLOPE']
    SHWFS_SLOPE = myvar['SHWFS_SLOPE']

###############################################################################

avg = np.nanmean(Ig_SLOPE - SHWFS_SLOPE, axis=(-1,-2))
var = np.nanvar(Ig_SLOPE - SHWFS_SLOPE, ddof=1, axis=(-1,-2))
avg[avg==0] = np.nan
var[var==0] = np.nan

varmin = np.nanmin(var)
varmax = np.nanmax(var)
avgmin = np.nanmin(avg)
avgmax = np.nanmax(avg)

min_slope = -np.nanmax([np.abs(SHWFS_SLOPE),np.abs(Ig_SLOPE),np.abs(g_SLOPE)])
max_slope = np.nanmax([np.abs(SHWFS_SLOPE),np.abs(Ig_SLOPE),np.abs(g_SLOPE)])

rms = np.sqrt(np.nanmean((Ig_SLOPE - SHWFS_SLOPE)**2, axis=(-1,-2)))

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



###############################################################################
# ░▒█░▒█░█▀▀░█▀▀░█▀▀░█░▒█░█░░░░▒█▀▀█░█░░▄▀▀▄░▀█▀░█▀▀
# ░▒█░▒█░▀▀▄░█▀▀░█▀░░█░▒█░█░░░░▒█▄▄█░█░░█░░█░░█░░▀▀▄
# ░░▀▄▄▀░▀▀▀░▀▀▀░▀░░░░▀▀▀░▀▀░░░▒█░░░░▀▀░░▀▀░░░▀░░▀▀▀
###############################################################################

# plt.figure()
# Figure 1
color_map = plt.cm.get_cmap('inferno')
reversed_color_map = color_map.reversed()
plt.subplots(4,4,sharex=True,sharey=True,squeeze=False)
# plt.subplot(4,4,1)
plt.suptitle('RMS difference of Ig and SHWFS slopes (rad/lenslet)')
for kk in np.arange(KK):
    for ll in np.arange(LL):
        plt.subplot(4,4,LL*(KK - kk - 1) + ll + 1)
        plt.pcolor(r0*100, dor0, np.sqrt(var[:,:,kk,ll]).T,
                   vmin=np.sqrt(varmin),vmax=np.sqrt(varmax),
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
plt.show()
plt.clf()

###############################################################################
# ░▒█░▒█░█▀▀░█▀▀░█▀▀░█░▒█░█░░░░▒█▀▀█░█░░▄▀▀▄░▀█▀░█▀▀
# ░▒█░▒█░▀▀▄░█▀▀░█▀░░█░▒█░█░░░░▒█▄▄█░█░░█░░█░░█░░▀▀▄
# ░░▀▄▄▀░▀▀▀░▀▀▀░▀░░░░▀▀▀░▀▀░░░▒█░░░░▀▀░░▀▀░░░▀░░▀▀▀
###############################################################################

# for jj in np.arange(JJ):
#     for ll in np.arange(LL):
#         for kk in np.arange(KK):
#             plt.plot(r0*100, var[:,jj,kk,ll],
#                       label=LABEL_RYTOV[kk], marker='o',ls=':')
#         plt.xlabel('r0 (cm)')
#         plt.ylabel('Variance of difference (rad2 per lenslet)')
#         plt.xlim(1, 12)
#         plt.ylim(2e-4,0.002)
#         plt.yscale('log')
#         plt.title(LABEL_DOR0[jj] + ', ' + LABEL_HHA[ll])
#         plt.legend()
#         plt.show()
#         plt.clf()

# for kk in np.arange(KK):
#     for ll in np.arange(LL):
#         for jj in np.arange(JJ):
#             plt.plot(r0*100, var[:,jj,kk,ll],
#                       label=LABEL_DOR0[jj], marker='o',ls=':')
#         plt.xlabel('r0 (cm)')
#         plt.ylabel('Variance of difference (rad2 per lenslet)')
#         plt.xlim(1, 12)
#         plt.ylim(1e-4,1e-2)
#         plt.yscale('log')
#         plt.title(LABEL_RYTOV[kk] + ', ' + LABEL_HHA[ll])
#         plt.legend()
#         plt.show()
#         plt.clf()

# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         plt.subplots(4,4,sharex=True,sharey=True,squeeze=False)
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 count = LL*(KK - kk - 1) + ll + 1
#                 plt.subplot(4,4,count)
#                 plt.axis('square')
#                 plt.xlim(min_slope,max_slope)
#                 plt.ylim(min_slope,max_slope)
#                 if POSSIBLE[ii,jj,kk,ll] == True:
#                     plt.plot(SHWFS_SLOPE[ii,jj,kk,ll].flatten(),
#                               Ig_SLOPE[ii,jj,kk,ll].flatten(),
#                               ls='',marker=',')
#                 if kk == 0:
#                     plt.xlabel('SHWFS\n' + LABEL_HHA[ll])
#                 if ll == 0:
#                     plt.ylabel(LABEL_RYTOV[kk] + '\nIg')
#         plt.suptitle('Ig vs SHWFS slope (rad/lenslet)\n' + LABEL_r0[ii] + ', ' + LABEL_DOR0[jj])
#         plt.show()
#         plt.clf()
    
# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         plt.subplots(4,4,sharex=True,sharey=True,squeeze=False)
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 count = LL*(KK - kk - 1) + ll + 1
#                 plt.subplot(4,4,count)
#                 plt.axis('square')
#                 plt.xlim(min_slope,max_slope)
#                 plt.ylim(min_slope,max_slope)
#                 if POSSIBLE[ii,jj,kk,ll] == True:
#                     plt.plot(g_SLOPE[ii,jj,kk,ll].flatten(),
#                               SHWFS_SLOPE[ii,jj,kk,ll].flatten(),
#                               ls='',marker=',')
#                 if kk == 0:
#                     plt.xlabel('g\n' + LABEL_HHA[ll])
#                 if ll == 0:
#                     plt.ylabel(LABEL_RYTOV[kk] + '\nSHWFS')
#         plt.suptitle('SHWFS vs g slope (rad/lenslet)\n' + LABEL_r0[ii] + ', ' + LABEL_DOR0[jj])
#         plt.show()
#         plt.clf()

# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         plt.subplots(4,4,sharex=True,sharey=True,squeeze=False)
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 count = LL*(KK - kk - 1) + ll + 1
#                 plt.subplot(4,4,count)
#                 plt.axis('square')
#                 plt.xlim(min_slope,max_slope)
#                 plt.ylim(min_slope,max_slope)
#                 if POSSIBLE[ii,jj,kk,ll] == True:
#                     plt.plot(g_SLOPE[ii,jj,kk,ll].flatten(),
#                               Ig_SLOPE[ii,jj,kk,ll].flatten(),
#                               ls='',marker=',')
#                 if kk == 0:
#                     plt.xlabel('g\n' + LABEL_HHA[ll])
#                 if ll == 0:
#                     plt.ylabel(LABEL_RYTOV[kk] + '\nIg')
#         plt.suptitle('Ig vs g slope (rad/lenslet)\n' + LABEL_r0[ii] + ', ' + LABEL_DOR0[jj])
#         plt.show()
#         plt.clf()

# res_max = np.nanmax([np.nanmax(np.abs(SHWFS_SLOPE - g_SLOPE)),
#                       np.nanmax(np.abs(Ig_SLOPE - g_SLOPE)),
#                       np.nanmax(np.abs(Ig_SLOPE - SHWFS_SLOPE))])

# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         plt.subplots(4,4,sharex=True,sharey=True,squeeze=False)
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 count = LL*(KK - kk - 1) + ll + 1
#                 plt.subplot(4,4,count)
#                 plt.axis('square')
#                 plt.xlim(min_slope,max_slope)
#                 plt.ylim(-res_max,res_max)
#                 if POSSIBLE[ii,jj,kk,ll] == True:
#                     plt.plot(g_SLOPE[ii,jj,kk,ll].flatten(),
#                               Ig_SLOPE[ii,jj,kk,ll].flatten()
#                               - g_SLOPE[ii,jj,kk,ll].flatten(),
#                               ls='',marker=',')
#                 if kk == 0:
#                     plt.xlabel('g\n' + LABEL_HHA[ll])
#                 if ll == 0:
#                     plt.ylabel(LABEL_RYTOV[kk] + '\nIg - g')
#         plt.suptitle('Ig - g vs g slope (rad/lenslet)\n' + LABEL_r0[ii] + ', ' + LABEL_DOR0[jj])
#         plt.show()
#         plt.clf()

# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         plt.subplots(4,4,sharex=True,sharey=True,squeeze=False)
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 count = LL*(KK - kk - 1) + ll + 1
#                 plt.subplot(4,4,count)
#                 plt.axis('square')
#                 plt.xlim(min_slope,max_slope)
#                 plt.ylim(-res_max,res_max)
#                 if POSSIBLE[ii,jj,kk,ll] == True:
#                     plt.plot(g_SLOPE[ii,jj,kk,ll].flatten(),
#                               SHWFS_SLOPE[ii,jj,kk,ll].flatten()
#                               - g_SLOPE[ii,jj,kk,ll].flatten(),
#                               ls='',marker=',')
#                 if kk == 0:
#                     plt.xlabel('g\n' + LABEL_HHA[ll])
#                 if ll == 0:
#                     plt.ylabel(LABEL_RYTOV[kk] + '\nSHWFS - g')
#         plt.suptitle('SHWFS - g vs g slope (rad/lenslet)\n' + LABEL_r0[ii] + ', ' + LABEL_DOR0[jj])
#         plt.show()
#         plt.clf()

# THIS_PLOT_MAX = np.nanmax(np.abs(Ig_SLOPE - SHWFS_SLOPE))
# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         plt.subplots(4,4,sharex=True,sharey=True,squeeze=False)
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 count = LL*(KK - kk - 1) + ll + 1
#                 plt.subplot(4,4,count)
#                 plt.axis('square')
#                 plt.xlim(min_slope,max_slope)
#                 plt.ylim(-res_max,res_max)
#                 # plt.ylim(-THIS_PLOT_MAX,THIS_PLOT_MAX)
#                 if POSSIBLE[ii,jj,kk,ll] == True:
#                     plt.plot(SHWFS_SLOPE[ii,jj,kk,ll].flatten(),
#                               Ig_SLOPE[ii,jj,kk,ll].flatten()
#                               - SHWFS_SLOPE[ii,jj,kk,ll].flatten(),
#                               ls='',marker=',')
#                 if kk == 0:
#                     plt.xlabel('SHWFS\n' + LABEL_HHA[ll])
#                 if ll == 0:
#                     plt.ylabel(LABEL_RYTOV[kk] + '\nIg - SHWFS')
#         plt.suptitle('Ig - SHWFS vs g slope (rad/lenslet)\n' + LABEL_r0[ii] + ', ' + LABEL_DOR0[jj])
#         plt.show()
#         plt.clf()

# THIS_PLOT_MAX = np.nanmax(np.abs(Ig_SLOPE - SHWFS_SLOPE))
# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         plt.subplots(4,4,sharex=True,sharey=True,squeeze=False)
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 count = LL*(KK - kk - 1) + ll + 1
#                 plt.subplot(4,4,count)
#                 # plt.axis('square')
#                 plt.xlim(min_slope,max_slope)
#                 # plt.ylim(-res_max,res_max)
#                 plt.ylim(-THIS_PLOT_MAX,THIS_PLOT_MAX)
#                 if POSSIBLE[ii,jj,kk,ll] == True:
#                     plt.plot(SHWFS_SLOPE[ii,jj,kk,ll].flatten(),
#                               Ig_SLOPE[ii,jj,kk,ll].flatten()
#                               - SHWFS_SLOPE[ii,jj,kk,ll].flatten(),
#                               ls='',marker=',')
#                 if kk == 0:
#                     plt.xlabel('SHWFS\n' + LABEL_HHA[ll])
#                 if ll == 0:
#                     plt.ylabel(LABEL_RYTOV[kk] + '\nIg - SHWFS')
#         plt.suptitle('Ig - SHWFS vs g slope (rad/lenslet)\n' + LABEL_r0[ii] + ', ' + LABEL_DOR0[jj])
#         plt.show()
#         plt.clf()

# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 if POSSIBLE[ii, jj, kk, ll] == True:
#                     x = SHWFS_SLOPE[ii,jj,kk,ll,:,:].flatten()
#                     y = Ig_SLOPE[ii,jj,kk,ll,:,:].flatten() - SHWFS_SLOPE[ii,jj,kk,ll,:,:].flatten()
#                     xabsmax = np.abs(x).max()
#                     yabsmax = np.abs(y).max()
#                     xyabsmax = np.max([xabsmax,yabsmax])
#                     a, b = deming(x, y)
#                     plt.plot(x, y, ls='', marker='.', label='measurements',color='r')
#                     plt.plot(np.linspace(-xyabsmax,xyabsmax), a + b*np.linspace(-xyabsmax,xyabsmax),
#                              ls='-',marker='',label='best fit',color='b')
#                     plt.hlines(0,xmin=-xabsmax,xmax=xabsmax,ls='--',label='zeros',color='k')
#                     plt.vlines(np.pi*np.arange(-xabsmax//np.pi, xabsmax//np.pi + 1),
#                                ymin=-yabsmax,ymax=yabsmax,label='pixel edges',ls='-.',color='m')
#                     plt.vlines(np.pi*np.arange(-xabsmax//np.pi//2 - 1, xabsmax//np.pi//2 + 2) + np.pi/2,
#                                ymin=-yabsmax,ymax=yabsmax,label='pixel centres',ls=':',color='c')
#                     plt.xlim(-xabsmax,xabsmax)
#                     plt.ylim(-yabsmax,yabsmax)
#                     plt.xlabel('SHWFS OPD (rad)')
#                     plt.ylabel('Ig - SHWFS OPD (rad)')
#                     plt.legend()
#                     plt.title(LABEL_r0[ii] + ', ' + LABEL_DOR0[jj] + ', ' + LABEL_RYTOV[jj] + ', ' + LABEL_HHA[ll])
#                     plt.show()
                    
# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 if POSSIBLE[ii, jj, kk, ll] == True:
#                     plt.plot(g_SLOPE[ii,jj,kk,ll,:,:].flatten(),
#                               Ig_SLOPE[ii,jj,kk,ll,:,:].flatten()
#                               - g_SLOPE[ii,jj,kk,ll,:,:].flatten(),
#                               ls='', marker=',')
#                     plt.xlabel('g (rad/lenslet)')
#                     plt.ylabel('Ig - g (rad/lenslet)')
#                     plt.xlim(min_slope,max_slope)
#                     plt.ylim(-res_max,res_max)
#                     plt.title(LABEL_r0[ii] + ', ' + LABEL_DOR0[jj] + ', ' + LABEL_RYTOV[jj] + ', ' + LABEL_HHA[ll])
#                     plt.show()
                    

# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 if POSSIBLE[ii, jj, kk, ll] == True:
#                     plt.plot(g_SLOPE[ii,jj,kk,ll,:,:].flatten(),
#                               SHWFS_SLOPE[ii,jj,kk,ll,:,:].flatten()
#                               - g_SLOPE[ii,jj,kk,ll,:,:].flatten(),
#                               ls='', marker=',')
#                     plt.xlabel('g (rad/lenslet)')
#                     plt.ylabel('SHWFS - g (rad/lenslet)')
#                     plt.xlim(min_slope,max_slope)
#                     plt.ylim(-res_max,res_max)
#                     plt.title(LABEL_r0[ii] + ', ' + LABEL_DOR0[jj] + ', ' + LABEL_RYTOV[jj] + ', ' + LABEL_HHA[ll])
#                     plt.show()

# THIS_PLOT_MAX = np.nanmax([np.abs(Ig_SLOPE-SHWFS_SLOPE)])
# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 if POSSIBLE[ii, jj, kk, ll] == True:
#                     plt.plot(Ig_SLOPE[ii,jj,kk,ll,:,:].flatten(),
#                               - Ig_SLOPE[ii,jj,kk,ll,:,:].flatten()
#                               + SHWFS_SLOPE[ii,jj,kk,ll,:,:].flatten(),
#                               ls='', marker=',',alpha=0.5)
#                     plt.xlabel('Ig (rad/lenslet)')
#                     plt.ylabel('SHWFS - Ig (rad/lenslet)')
#                     plt.xlim(min_slope,max_slope)
#                     plt.ylim(-THIS_PLOT_MAX,THIS_PLOT_MAX)
#                     plt.title(LABEL_r0[ii] + ', ' + LABEL_DOR0[jj] + ', ' + LABEL_RYTOV[jj] + ', ' + LABEL_HHA[ll])
#                     plt.show()


# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 if POSSIBLE[ii, jj, kk, ll] == True:
#                     plt.plot(SHWFS_SLOPE[ii,jj,kk,ll,:,:].flatten(),
#                               Ig_SLOPE[ii,jj,kk,ll,:,:].flatten(),
#                               ls='', marker=',')
#                     plt.xlabel('SHWFS (rad/lenslet)')
#                     plt.ylabel('Ig (rad/lenslet)')
#                     plt.title(LABEL_r0[ii] + ', ' + LABEL_DOR0[jj] + ', ' + LABEL_RYTOV[jj] + ', ' + LABEL_HHA[ll])
#                     plt.show()


# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 if POSSIBLE[ii, jj, kk, ll] == True:
#                     plt.plot(g_SLOPE[ii,jj,kk,ll,:,:].flatten(),
#                               Ig_SLOPE[ii,jj,kk,ll,:,:].flatten(),
#                               ls='', marker=',')
#                     plt.xlabel('g (rad/lenslet)')
#                     plt.ylabel('Ig (rad/lenslet)')
#                     plt.xlim()
#                     plt.ylim()
#                     plt.title(LABEL_r0[ii] + ', ' + LABEL_DOR0[jj] + ', ' + LABEL_RYTOV[jj] + ', ' + LABEL_HHA[ll])
#                     plt.show()
                    
# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 if POSSIBLE[ii, jj, kk, ll] == True:
#                     plt.plot(g_SLOPE[ii,jj,kk,ll,:,:].flatten(),
#                               SHWFS_SLOPE[ii,jj,kk,ll,:,:].flatten(),
#                               ls='', marker=',')
#                     plt.xlabel('g (rad/lenslet)')
#                     plt.ylabel('SHWFS (rad/lenslet)')
#                     plt.xlim()
#                     plt.ylim()
#                     plt.title(LABEL_r0[ii] + ', ' + LABEL_DOR0[jj] + ', ' + LABEL_RYTOV[jj] + ', ' + LABEL_HHA[ll])
#                     plt.show()
                    
# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 if POSSIBLE[ii, jj, kk, ll] == True:
#                     plt.plot(SHWFS_SLOPE[ii,jj,kk,ll,:,:].flatten(),
#                               Ig_SLOPE[ii,jj,kk,ll,:,:].flatten(),
#                               ls='', marker=',')
#                     plt.xlabel('SHWFS (rad/lenslet)')
#                     plt.ylabel('Ig (rad/lenslet)')
#                     plt.xlim()
#                     plt.ylim()
#                     plt.title(LABEL_r0[ii] + ', ' + LABEL_DOR0[jj] + ', ' + LABEL_RYTOV[jj] + ', ' + LABEL_HHA[ll])
#                     plt.show()

###############################################################################
##########################################################################
"""
█▀▀█░█░▒█░▀█▀░▀█▀░░▀░░█░░░█░█▀▀▄░▀█▀░░░▒█░▄▀░▄▀▀▄░█▀▀▄░█▀▀▀░█░▄░█▀▀▄░█▀▀░█░░░█
█▄▄█░█░▒█░░█░░░█░░░█▀░▀▄█▄▀░█▄▄█░░█░░░░▒█▀▄░░█░░█░█░▒█░█░▀▄░█▀▄░█▄▄█░█▀▀░▀▄█▄▀
█░░░░░▀▀▀░░▀░░░▀░░▀▀▀░░▀░▀░░▀░░▀░░▀░░░░▒█░▒█░░▀▀░░▀░░▀░▀▀▀▀░▀░▀░▀░░▀░▀▀▀░░▀░▀░
"""
##########################################################################
###############################################################################

# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         for ll in np.arange(LL):
#             print(r0[ii],dor0[jj],Rytov[kk],H_HA[ll],POSSIBLE_SCREEN[ii,kk,ll],POSSIBLE_PROP[ii,jj,ll])