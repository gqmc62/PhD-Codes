
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
    g_SLOPE_noscintillation = myvar['g_SLOPE_noscintillation']*CONVERT
    Ig_SLOPE_noscintillation = myvar['Ig_SLOPE_noscintillation']*CONVERT
    SHWFS_SLOPE_noscintillation = myvar['SHWFS_SLOPE_noscintillation']*CONVERT

##############################################################################################################################################

    
#########################################################################################################################

    
g_SLOPE[g_SLOPE==0]=np.nan
Ig_SLOPE[Ig_SLOPE==0]=np.nan
g_SLOPE_noscintillation[g_SLOPE_noscintillation==0]=np.nan
Ig_SLOPE_noscintillation[Ig_SLOPE_noscintillation==0]=np.nan
SHWFS_SLOPE[SHWFS_SLOPE==0]=np.nan
SHWFS_SLOPE_noscintillation[SHWFS_SLOPE_noscintillation==0]=np.nan


g_rms_noscintillation = np.nanstd(g_SLOPE_noscintillation - SHWFS_SLOPE_noscintillation,axis=(-1,-2))
Ig_rms_noscintillation = np.nanstd(Ig_SLOPE_noscintillation - SHWFS_SLOPE_noscintillation,axis=(-1,-2))

# A1 = np.copy(SHWFS_SLOPE)
# B1 = np.copy(g_SLOPE)
# C1 = np.copy(Ig_SLOPE)
plt.subplots(figsize=(6,4))
g_plot, = plt.plot(SHWFS_SLOPE.flatten(), g_SLOPE.flatten(),
                   label='G-Tilt', ls='', marker='.', alpha=0.01,color='r',markeredgewidth=0)

Ig_plot, = plt.plot(SHWFS_SLOPE.flatten(), Ig_SLOPE.flatten(),
                    label='IG-Tilt', ls='', marker='.', alpha=0.01,color='b',markeredgewidth=0)
# plt.plot([-1500,1500],[-1500,1500],label='FT-Tilt')

# Create custom legend with solid color and more visible marker
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='none', label='g', markerfacecolor=g_plot.get_color(), markersize=6),
    Line2D([0], [0], marker='o', color='none', label='Ig', markerfacecolor=Ig_plot.get_color(), markersize=6)
]

plt.legend(handles=legend_elements, loc='upper left')
plt.xlabel('SHWFS FT OPD (nm)')
plt.ylabel('Approximation of SHWFS OPD (nm)')
plt.grid(True)
plt.title('response g,ig to ft no threshold')
plt.savefig('all-response-wfs.png',bbox_inches='tight',dpi=600)
plt.show()


plt.subplots(figsize=(6,4))
g_plot, = plt.plot(SHWFS_SLOPE.flatten(), (g_SLOPE-SHWFS_SLOPE).flatten(),
                   label='G-Tilt', ls='', marker='.', alpha=0.01,color='r',markeredgewidth=0)

Ig_plot, = plt.plot(SHWFS_SLOPE.flatten(), (Ig_SLOPE-SHWFS_SLOPE).flatten(),
                    label='IG-Tilt', ls='', marker='.', alpha=0.01,color='b',markeredgewidth=0)
# plt.plot([-1500,1500],[-1500,1500],label='FT-Tilt')

# Create custom legend with solid color and more visible marker
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='none', label='g', markerfacecolor=g_plot.get_color(), markersize=6),
    Line2D([0], [0], marker='o', color='none', label='Ig', markerfacecolor=Ig_plot.get_color(), markersize=6)
]

plt.legend(handles=legend_elements, loc='upper left')
plt.xlabel('SHWFS FT OPD (nm)')
plt.ylabel('Error in approximation of SHWFS OPD (nm)')
plt.grid(True)
plt.title('error g,ig to ft no threshold')
plt.savefig('all-residual-wfs.png',bbox_inches='tight',dpi=600)
plt.show()


plt.subplots(figsize=(6,4))
# g_plot, = plt.plot(SHWFS_SLOPE.flatten(), (g_SLOPE-SHWFS_SLOPE).flatten(),
#                    label='G-Tilt', ls='', marker=',', alpha=0.1,color='r')

Ig_plot, = plt.plot(SHWFS_SLOPE.flatten(), (Ig_SLOPE-SHWFS_SLOPE).flatten(),
                    label='IG-Tilt', ls='', marker='.', alpha=0.01,color='b',markeredgewidth=0)
# plt.plot([-1500,1500],[-1500,1500],label='FT-Tilt')

# Create custom legend with solid color and more visible marker
from matplotlib.lines import Line2D

legend_elements = [
    # Line2D([0], [0], marker='o', color='none', label='g', markerfacecolor=g_plot.get_color(), markersize=6),
    Line2D([0], [0], marker='o', color='none', label='Ig', markerfacecolor=Ig_plot.get_color(), markersize=6)
]

plt.legend(handles=legend_elements, loc='upper left')
plt.xlabel('SHWFS FT OPD (nm)')
plt.ylabel('Error in approximation of SHWFS OPD (nm)')
plt.grid(True)
plt.title('error ig to ft no threshold - zoom')
plt.savefig('ig-residual-wfs.png',bbox_inches='tight',dpi=600)
plt.show()

##############################################################################

import matplotlib.colors as colors


LABEL_r0 = np.zeros((II), dtype=object)
for ii in np.arange(II):
    LABEL_r0[ii] = 'r$_0$={:.0f}cm'.format(r0[ii]*100)

LABEL_DOR0 = np.zeros((JJ), dtype=object)
for jj in np.arange(JJ):
    LABEL_DOR0[jj] = 'd/r$_0$={:.1f}'.format(dor0[jj])
    
LABEL_RYTOV = np.zeros((KK), dtype=object)
for kk in np.arange(KK):
    LABEL_RYTOV[kk] = '$\sigma^2_\chi=${:.2f}'.format(Rytov[kk])
    
LABEL_HHA = np.zeros((LL), dtype=object)
for ll in np.arange(LL):
    LABEL_HHA[ll] = 'HA={:.0f}km'.format(H_HA[ll]/1000)

POSSIBLE_SCREEN_TEMP = np.zeros((4,4,4,4),dtype=bool)
for ii in range(4):
    for jj in range(4):
        for kk in range(4):
            POSSIBLE_SCREEN_TEMP[ii,:,jj,kk] = POSSIBLE_SCREEN[ii,jj,kk]

color_map = plt.cm.get_cmap('inferno')
reversed_color_map = color_map.reversed()

###############################################################################


avg_g = np.nanmean(g_SLOPE - SHWFS_SLOPE, axis=(-1,-2))
var_g = np.nanvar(g_SLOPE - SHWFS_SLOPE, ddof=1, axis=(-1,-2))
avg_g[avg_g==0] = np.nan
var_g[var_g==0] = np.nan

avg_ig = np.nanmean(Ig_SLOPE - SHWFS_SLOPE, axis=(-1,-2))
var_ig = np.nanvar(Ig_SLOPE - SHWFS_SLOPE, ddof=1, axis=(-1,-2))
avg_ig[avg_ig==0] = np.nan
var_ig[var_ig==0] = np.nan

# import matplotlib.pyplot as plt
# import numpy as np

# Assume var_g and var_ig are available: shape (II, JJ, KK, LL)

# Shared scale from both tilt types
combined_var = np.concatenate([var_g, var_ig], axis=-1)  # shape (II, JJ, KK, 2*LL)
vmax = np.sqrt(np.nanmax(combined_var))
vmin = 0  # RMS variance is non-negative

# Create subplots: rows = KK, columns = 2*LL
fig, axes = plt.subplots(KK, 2*LL, figsize=(2.5*2*LL, 2.5*KK), sharex=True, sharey=True)
fig.suptitle('RMS difference of g-tilt (left) and IG-tilt (right) vs SHWFS (nm/lenslet) - no threshold', fontsize=14)

# Use reversed colormap
color_map = plt.cm.get_cmap('inferno').reversed()

for kk in range(KK):
    for ll in range(LL):
        for side, var_data, title_prefix in zip(
            [0, 1], [var_g, var_ig], ['G', 'IG']
        ):
            ax = axes[KK-kk-1, ll + side*LL]
            data = np.sqrt(var_data[:, :, kk, ll]).T  # shape (JJ, II)

            # Optional screen mask overlay
            if (POSSIBLE_SCREEN_TEMP[:, :, kk, ll].T).sum() != 16:
                ax.pcolor(r0*100, dor0, (1 - POSSIBLE_SCREEN_TEMP[:, :, kk, ll]).T * 0.5,
                          vmin=0, vmax=1, cmap='Blues')

            pc = ax.pcolor(r0*100, dor0, data, cmap=color_map,
                           vmin=0, vmax=vmax, edgecolors='k', linewidths=0.5)

            if kk == KK-1:
                ax.set_title(f'{title_prefix} - {LABEL_HHA[ll]}')

            if ll == 0 and side == 0:
                ax.set_ylabel(LABEL_RYTOV[kk] + '\nd/r$_0$')

            if kk == 0:
                ax.set_xlabel('r$_0$ (cm)')

# Add a single colorbar
cax = fig.add_axes([1.0, 0.15, 0.015, 0.7])  # [left, bottom, width, height] in figure coords
cbar = fig.colorbar(pc, cax=cax)
cbar.set_label('RMS OPD (nm/lenslet)')
cbar.set_label('RMS OPD (nm/lenslet)')

fig.tight_layout(rect=[0, 0, 0.97, 0.95])
plt.savefig('G_vs_IG_Tilt_RMS_Comparison.pdf')
plt.show()

#############################################################################


# import matplotlib.pyplot as plt
# import numpy as np

# Assume var_g and var_ig are defined (shape II, JJ, KK, LL)

# Shared scale from both tilt types
combined_var = np.concatenate([var_g, var_ig], axis=-1)  # shape (II, JJ, KK, 2*LL)
vmax = np.sqrt(np.nanmax(combined_var))
vmin = 0  # RMS variance is non-negative

color_map = plt.cm.get_cmap('inferno').reversed()

##########################
# Plot 1: G-Tilt RMS diff #
##########################
fig1, axes1 = plt.subplots(KK, LL, figsize=(2.5*LL, 2.5*KK), sharex=True, sharey=True)
fig1.suptitle('RMS difference of G-tilt vs SHWFS (nm/lenslet) - no threshold', fontsize=14)

for kk in range(KK):
    for ll in range(LL):
        ax = axes1[KK-kk-1, ll] if KK > 1 else axes1[ll]  # handle 1D axes array if KK=1
        data = np.sqrt(var_g[:, :, kk, ll]).T

        # Optional screen mask overlay
        if (POSSIBLE_SCREEN_TEMP[:, :, kk, ll].T).sum() != 16:
            ax.pcolor(r0*100, dor0, (1 - POSSIBLE_SCREEN_TEMP[:, :, kk, ll]).T * 0.5,
                      vmin=0, vmax=1, cmap='Blues')

        pc = ax.pcolor(r0*100, dor0, data, cmap=color_map,
                       vmin=0, vmax=vmax, edgecolors='k', linewidths=0.5)

        if kk == KK-1:
            ax.set_title(f'G - {LABEL_HHA[ll]}')

        if ll == 0:
            ax.set_ylabel(LABEL_RYTOV[kk] + '\nd/r$_0$')

        if kk == 0:
            ax.set_xlabel('r$_0$ (cm)')

# Colorbar axis to the right
cax1 = fig1.add_axes([0.92, 0.15, 0.015, 0.7])  # adjust as needed
fig1.colorbar(pc, cax=cax1, label='RMS OPD (nm/lenslet)')

fig1.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.savefig('G_Tilt_RMS_Comparison.pdf')


###########################
# Plot 2: IG-Tilt RMS diff #
###########################
fig2, axes2 = plt.subplots(KK, LL, figsize=(2.5*LL, 2.5*KK), sharex=True, sharey=True)
fig2.suptitle('RMS difference of IG-tilt vs SHWFS (nm/lenslet) - gtilt scale - no threshold', fontsize=14)

for kk in range(KK):
    for ll in range(LL):
        ax = axes2[KK-kk-1, ll] if KK > 1 else axes2[ll]
        data = np.sqrt(var_ig[:, :, kk, ll]).T

        # Optional screen mask overlay
        if (POSSIBLE_SCREEN_TEMP[:, :, kk, ll].T).sum() != 16:
            ax.pcolor(r0*100, dor0, (1 - POSSIBLE_SCREEN_TEMP[:, :, kk, ll]).T * 0.5,
                      vmin=0, vmax=1, cmap='Blues')

        pc = ax.pcolor(r0*100, dor0, data, cmap=color_map,
                       vmin=0, vmax=vmax, edgecolors='k', linewidths=0.5)

        if kk == KK-1:
            ax.set_title(f'IG - {LABEL_HHA[ll]}')

        if ll == 0:
            ax.set_ylabel(LABEL_RYTOV[kk] + '\nd/r$_0$')

        if kk == 0:
            ax.set_xlabel('r$_0$ (cm)')

# Colorbar axis to the right
cax2 = fig2.add_axes([0.92, 0.15, 0.015, 0.7])  # adjust as needed
fig2.colorbar(pc, cax=cax2, label='RMS OPD (nm/lenslet)')

fig2.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.savefig('IG_Tilt_RMS_Comparison.pdf')


plt.show()

###########################
# Plot 2: IG-Tilt RMS diff #
###########################
fig2, axes2 = plt.subplots(KK, LL, figsize=(2.5*LL, 2.5*KK), sharex=True, sharey=True)
fig2.suptitle('RMS difference of IG-tilt vs SHWFS (nm/lenslet) - no threshold', fontsize=14)

for kk in range(KK):
    for ll in range(LL):
        ax = axes2[KK-kk-1, ll] if KK > 1 else axes2[ll]
        data = np.sqrt(var_ig[:, :, kk, ll]).T

        # Optional screen mask overlay
        if (POSSIBLE_SCREEN_TEMP[:, :, kk, ll].T).sum() != 16:
            ax.pcolor(r0*100, dor0, (1 - POSSIBLE_SCREEN_TEMP[:, :, kk, ll]).T * 0.5,
                      vmin=0, vmax=1, cmap='Blues')

        pc = ax.pcolor(r0*100, dor0, data, cmap=color_map,
                       vmin=0, vmax=np.nanmax(np.sqrt(var_ig)), edgecolors='k', linewidths=0.5)

        if kk == KK-1:
            ax.set_title(f'IG - {LABEL_HHA[ll]}')

        if ll == 0:
            ax.set_ylabel(LABEL_RYTOV[kk] + '\nd/r$_0$')

        if kk == 0:
            ax.set_xlabel('r$_0$ (cm)')

# Colorbar axis to the right
cax2 = fig2.add_axes([0.92, 0.15, 0.015, 0.7])  # adjust as needed
fig2.colorbar(pc, cax=cax2, label='RMS OPD (nm/lenslet)')

fig2.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.savefig('IG_Tilt_RMS_Comparison-another.pdf')


plt.show()

#############################################################################

avg = np.nanmean(g_SLOPE - SHWFS_SLOPE, axis=(-1,-2))
var = np.nanvar(g_SLOPE - SHWFS_SLOPE, ddof=1, axis=(-1,-2))
avg[avg==0] = np.nan
var[var==0] = np.nan

varmin = np.nanmin(var)
varmax = np.nanmax(var)
avgmin = np.nanmin(avg)
avgmax = np.nanmax(avg)

min_slope = -np.nanmax([np.abs(SHWFS_SLOPE),np.abs(Ig_SLOPE),np.abs(g_SLOPE)])
max_slope = np.nanmax([np.abs(SHWFS_SLOPE),np.abs(Ig_SLOPE),np.abs(g_SLOPE)])

rms = np.sqrt(np.nanmean((g_SLOPE - SHWFS_SLOPE)**2, axis=(-1,-2)))

LABEL_r0 = np.zeros((II), dtype=object)
for ii in np.arange(II):
    LABEL_r0[ii] = 'r$_0$={:.0f}cm'.format(r0[ii]*100)

LABEL_DOR0 = np.zeros((JJ), dtype=object)
for jj in np.arange(JJ):
    LABEL_DOR0[jj] = 'd/r$_0$={:.1f}'.format(dor0[jj])
    
LABEL_RYTOV = np.zeros((KK), dtype=object)
for kk in np.arange(KK):
    LABEL_RYTOV[kk] = '$\sigma^2_\chi=${:.2f}'.format(Rytov[kk])
    
LABEL_HHA = np.zeros((LL), dtype=object)
for ll in np.arange(LL):
    LABEL_HHA[ll] = 'HA={:.0f}km'.format(H_HA[ll]/1000)



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

data = np.nanstd(Ig_SLOPE - SHWFS_SLOPE,axis=(-1,-2))
data[data==0] = np.nan

V = data


import itertools

def plot_all_cross_sections(V, w, x, y, z):
    param_arrays = [w*100, x, y, z/1000]
    param_labels = ['r$_0$ (cm)', 'd/r$_0$', 'Rytov', 'H$_{HA}$ (km)']
    
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
        if axis == 1:
            ax.errorbar(varying_param,Ig_rms_noscintillation[0],yerr=Ig_rms_noscintillation[0]/np.sqrt(2*1024-2),
                        ls=':',color='k',capsize=5)
        # else:
        #     ax.hlines(g_rms_noscintillation,varying_param.min(),varying_param.max(),ls=':',color='k')
        #     ax.hlines(Ig_rms_noscintillation,varying_param.min(),varying_param.max(),ls='-',color='k')
        else:
            for i in range(4):
                ax.errorbar(varying_param,Ig_rms_noscintillation[:,i],
                          yerr=Ig_rms_noscintillation[:,i]/np.sqrt(2*1024-2),
                          ls=':',color='k',capsize=5)
                
        # ax.set_title(f"cros section : {param_labels[axis]}")
        ax.set_xlabel(f"{param_labels[axis]}")
        ax.set_ylabel("OPD (nm)")
        # ax.set_ylim(0,6)
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)
        # if axis == 0:
        #     ax.legend(fontsize='x-small', loc='best')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.suptitle('Cross sections of measured RMS OPD difference between IG and FT (nm) - no threshold')
    plt.savefig('cross-section-measure-ig-ft.pdf',bbox_inches='tight')
    plt.show()
    return

plot_all_cross_sections(V, r0, dor0, Rytov, H_HA)

#########################################################################################
######################################################################################
#########################################################################################
######################################################################################
#########################################################################################
######################################################################################
#########################################################################################
######################################################################################
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
# with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\20250626\NoiselessSquarePupilSlopes-og.pkl','rb') as file:
with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\20250626\NoiselessSquarePupilSlopes-no-ft-bias.pkl','rb') as file:
    myvar = pickle.load(file)
    g_SLOPE = myvar['g_SLOPE']*CONVERT
    Ig_SLOPE = myvar['Ig_SLOPE']*CONVERT
    SHWFS_SLOPE = myvar['SHWFS_SLOPE']*CONVERT
    g_SLOPE_noscintillation = myvar['g_SLOPE_noscintillation']*CONVERT
    Ig_SLOPE_noscintillation = myvar['Ig_SLOPE_noscintillation']*CONVERT
    SHWFS_SLOPE_noscintillation = myvar['SHWFS_SLOPE_noscintillation']*CONVERT

##############################################################################################################################################

    
#########################################################################################################################

    
g_SLOPE[g_SLOPE==0]=np.nan
Ig_SLOPE[Ig_SLOPE==0]=np.nan
g_SLOPE_noscintillation[g_SLOPE_noscintillation==0]=np.nan
Ig_SLOPE_noscintillation[Ig_SLOPE_noscintillation==0]=np.nan
SHWFS_SLOPE[SHWFS_SLOPE==0]=np.nan
SHWFS_SLOPE_noscintillation[SHWFS_SLOPE_noscintillation==0]=np.nan


g_rms_noscintillation = np.nanstd(g_SLOPE_noscintillation - SHWFS_SLOPE_noscintillation,axis=(-1,-2))
Ig_rms_noscintillation = np.nanstd(Ig_SLOPE_noscintillation - SHWFS_SLOPE_noscintillation,axis=(-1,-2))

# A1 = np.copy(SHWFS_SLOPE)
# B1 = np.copy(g_SLOPE)
# C1 = np.copy(Ig_SLOPE)
plt.subplots(figsize=(6,4))
g_plot, = plt.plot(SHWFS_SLOPE.flatten(), g_SLOPE.flatten(),
                   label='G-Tilt', ls='', marker='.', alpha=0.01,color='r',markeredgewidth=0)

Ig_plot, = plt.plot(SHWFS_SLOPE.flatten(), Ig_SLOPE.flatten(),
                    label='IG-Tilt', ls='', marker='.', alpha=0.01,color='b',markeredgewidth=0)
# plt.plot([-1500,1500],[-1500,1500],label='FT-Tilt')

# Create custom legend with solid color and more visible marker
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='none', label='g', markerfacecolor=g_plot.get_color(), markersize=6),
    Line2D([0], [0], marker='o', color='none', label='Ig', markerfacecolor=Ig_plot.get_color(), markersize=6)
]

plt.legend(handles=legend_elements, loc='upper left')
plt.xlabel('SHWFS FT OPD (nm)')
plt.ylabel('Approximation of SHWFS OPD (nm)')
plt.grid(True)
plt.title('response g,ig to ft no bias')
plt.savefig('all-response-wfs.png',bbox_inches='tight',dpi=600)
plt.show()


plt.subplots(figsize=(6,4))
g_plot, = plt.plot(SHWFS_SLOPE.flatten(), (g_SLOPE-SHWFS_SLOPE).flatten(),
                   label='G-Tilt', ls='', marker='.', alpha=0.01,color='r',markeredgewidth=0)

Ig_plot, = plt.plot(SHWFS_SLOPE.flatten(), (Ig_SLOPE-SHWFS_SLOPE).flatten(),
                    label='IG-Tilt', ls='', marker='.', alpha=0.01,color='b',markeredgewidth=0)
# plt.plot([-1500,1500],[-1500,1500],label='FT-Tilt')

# Create custom legend with solid color and more visible marker
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='none', label='g', markerfacecolor=g_plot.get_color(), markersize=6),
    Line2D([0], [0], marker='o', color='none', label='Ig', markerfacecolor=Ig_plot.get_color(), markersize=6)
]

plt.legend(handles=legend_elements, loc='upper left')
plt.xlabel('SHWFS FT OPD (nm)')
plt.ylabel('Error in approximation of SHWFS OPD (nm)')
plt.grid(True)
plt.title('error g,ig to ft no bias')
plt.savefig('all-residual-wfs.png',bbox_inches='tight',dpi=600)
plt.show()


plt.subplots(figsize=(6,4))
# g_plot, = plt.plot(SHWFS_SLOPE.flatten(), (g_SLOPE-SHWFS_SLOPE).flatten(),
#                    label='G-Tilt', ls='', marker=',', alpha=0.1,color='r')

Ig_plot, = plt.plot(SHWFS_SLOPE.flatten(), (Ig_SLOPE-SHWFS_SLOPE).flatten(),
                    label='IG-Tilt', ls='', marker='.', alpha=0.01,color='b',markeredgewidth=0)
# plt.plot([-1500,1500],[-1500,1500],label='FT-Tilt')

# Create custom legend with solid color and more visible marker
from matplotlib.lines import Line2D

legend_elements = [
    # Line2D([0], [0], marker='o', color='none', label='g', markerfacecolor=g_plot.get_color(), markersize=6),
    Line2D([0], [0], marker='o', color='none', label='Ig', markerfacecolor=Ig_plot.get_color(), markersize=6)
]

plt.legend(handles=legend_elements, loc='upper left')
plt.xlabel('SHWFS FT OPD (nm)')
plt.ylabel('Error in approximation of SHWFS OPD (nm)')
plt.grid(True)
plt.title('error ig to ft no bias - zoom')
plt.savefig('ig-residual-wfs.png',bbox_inches='tight',dpi=600)
plt.show()

##############################################################################

import matplotlib.colors as colors


LABEL_r0 = np.zeros((II), dtype=object)
for ii in np.arange(II):
    LABEL_r0[ii] = 'r$_0$={:.0f}cm'.format(r0[ii]*100)

LABEL_DOR0 = np.zeros((JJ), dtype=object)
for jj in np.arange(JJ):
    LABEL_DOR0[jj] = 'd/r$_0$={:.1f}'.format(dor0[jj])
    
LABEL_RYTOV = np.zeros((KK), dtype=object)
for kk in np.arange(KK):
    LABEL_RYTOV[kk] = '$\sigma^2_\chi=${:.2f}'.format(Rytov[kk])
    
LABEL_HHA = np.zeros((LL), dtype=object)
for ll in np.arange(LL):
    LABEL_HHA[ll] = 'HA={:.0f}km'.format(H_HA[ll]/1000)

POSSIBLE_SCREEN_TEMP = np.zeros((4,4,4,4),dtype=bool)
for ii in range(4):
    for jj in range(4):
        for kk in range(4):
            POSSIBLE_SCREEN_TEMP[ii,:,jj,kk] = POSSIBLE_SCREEN[ii,jj,kk]

color_map = plt.cm.get_cmap('inferno')
reversed_color_map = color_map.reversed()

###############################################################################


avg_g = np.nanmean(g_SLOPE - SHWFS_SLOPE, axis=(-1,-2))
var_g = np.nanvar(g_SLOPE - SHWFS_SLOPE, ddof=1, axis=(-1,-2))
avg_g[avg_g==0] = np.nan
var_g[var_g==0] = np.nan

avg_ig = np.nanmean(Ig_SLOPE - SHWFS_SLOPE, axis=(-1,-2))
var_ig = np.nanvar(Ig_SLOPE - SHWFS_SLOPE, ddof=1, axis=(-1,-2))
avg_ig[avg_ig==0] = np.nan
var_ig[var_ig==0] = np.nan

# import matplotlib.pyplot as plt
# import numpy as np

# Assume var_g and var_ig are available: shape (II, JJ, KK, LL)

# Shared scale from both tilt types
combined_var = np.concatenate([var_g, var_ig], axis=-1)  # shape (II, JJ, KK, 2*LL)
vmax = np.sqrt(np.nanmax(combined_var))
vmin = 0  # RMS variance is non-negative

# Create subplots: rows = KK, columns = 2*LL
fig, axes = plt.subplots(KK, 2*LL, figsize=(2.5*2*LL, 2.5*KK), sharex=True, sharey=True)
fig.suptitle('RMS difference of g-tilt (left) and IG-tilt (right) vs SHWFS (nm/lenslet) - no bias', fontsize=14)

# Use reversed colormap
color_map = plt.cm.get_cmap('inferno').reversed()

for kk in range(KK):
    for ll in range(LL):
        for side, var_data, title_prefix in zip(
            [0, 1], [var_g, var_ig], ['G', 'IG']
        ):
            ax = axes[KK-kk-1, ll + side*LL]
            data = np.sqrt(var_data[:, :, kk, ll]).T  # shape (JJ, II)

            # Optional screen mask overlay
            if (POSSIBLE_SCREEN_TEMP[:, :, kk, ll].T).sum() != 16:
                ax.pcolor(r0*100, dor0, (1 - POSSIBLE_SCREEN_TEMP[:, :, kk, ll]).T * 0.5,
                          vmin=0, vmax=1, cmap='Blues')

            pc = ax.pcolor(r0*100, dor0, data, cmap=color_map,
                           vmin=0, vmax=vmax, edgecolors='k', linewidths=0.5)

            if kk == KK-1:
                ax.set_title(f'{title_prefix} - {LABEL_HHA[ll]}')

            if ll == 0 and side == 0:
                ax.set_ylabel(LABEL_RYTOV[kk] + '\nd/r$_0$')

            if kk == 0:
                ax.set_xlabel('r$_0$ (cm)')

# Add a single colorbar
cax = fig.add_axes([1.0, 0.15, 0.015, 0.7])  # [left, bottom, width, height] in figure coords
cbar = fig.colorbar(pc, cax=cax)
cbar.set_label('RMS OPD (nm/lenslet)')
cbar.set_label('RMS OPD (nm/lenslet)')

fig.tight_layout(rect=[0, 0, 0.97, 0.95])
plt.savefig('G_vs_IG_Tilt_RMS_Comparison.pdf')
plt.show()

#############################################################################


# import matplotlib.pyplot as plt
# import numpy as np

# Assume var_g and var_ig are defined (shape II, JJ, KK, LL)

# Shared scale from both tilt types
combined_var = np.concatenate([var_g, var_ig], axis=-1)  # shape (II, JJ, KK, 2*LL)
vmax = np.sqrt(np.nanmax(combined_var))
vmin = 0  # RMS variance is non-negative

color_map = plt.cm.get_cmap('inferno').reversed()

##########################
# Plot 1: G-Tilt RMS diff #
##########################
fig1, axes1 = plt.subplots(KK, LL, figsize=(2.5*LL, 2.5*KK), sharex=True, sharey=True)
fig1.suptitle('RMS difference of G-tilt vs SHWFS (nm/lenslet) - no bias', fontsize=14)

for kk in range(KK):
    for ll in range(LL):
        ax = axes1[KK-kk-1, ll] if KK > 1 else axes1[ll]  # handle 1D axes array if KK=1
        data = np.sqrt(var_g[:, :, kk, ll]).T

        # Optional screen mask overlay
        if (POSSIBLE_SCREEN_TEMP[:, :, kk, ll].T).sum() != 16:
            ax.pcolor(r0*100, dor0, (1 - POSSIBLE_SCREEN_TEMP[:, :, kk, ll]).T * 0.5,
                      vmin=0, vmax=1, cmap='Blues')

        pc = ax.pcolor(r0*100, dor0, data, cmap=color_map,
                       vmin=0, vmax=vmax, edgecolors='k', linewidths=0.5)

        if kk == KK-1:
            ax.set_title(f'G - {LABEL_HHA[ll]}')

        if ll == 0:
            ax.set_ylabel(LABEL_RYTOV[kk] + '\nd/r$_0$')

        if kk == 0:
            ax.set_xlabel('r$_0$ (cm)')

# Colorbar axis to the right
cax1 = fig1.add_axes([0.92, 0.15, 0.015, 0.7])  # adjust as needed
fig1.colorbar(pc, cax=cax1, label='RMS OPD (nm/lenslet)')

fig1.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.savefig('G_Tilt_RMS_Comparison.pdf')


###########################
# Plot 2: IG-Tilt RMS diff #
###########################
fig2, axes2 = plt.subplots(KK, LL, figsize=(2.5*LL, 2.5*KK), sharex=True, sharey=True)
fig2.suptitle('RMS difference of IG-tilt vs SHWFS (nm/lenslet) - gtilt scale - no bias', fontsize=14)

for kk in range(KK):
    for ll in range(LL):
        ax = axes2[KK-kk-1, ll] if KK > 1 else axes2[ll]
        data = np.sqrt(var_ig[:, :, kk, ll]).T

        # Optional screen mask overlay
        if (POSSIBLE_SCREEN_TEMP[:, :, kk, ll].T).sum() != 16:
            ax.pcolor(r0*100, dor0, (1 - POSSIBLE_SCREEN_TEMP[:, :, kk, ll]).T * 0.5,
                      vmin=0, vmax=1, cmap='Blues')

        pc = ax.pcolor(r0*100, dor0, data, cmap=color_map,
                       vmin=0, vmax=vmax, edgecolors='k', linewidths=0.5)

        if kk == KK-1:
            ax.set_title(f'IG - {LABEL_HHA[ll]}')

        if ll == 0:
            ax.set_ylabel(LABEL_RYTOV[kk] + '\nd/r$_0$')

        if kk == 0:
            ax.set_xlabel('r$_0$ (cm)')

# Colorbar axis to the right
cax2 = fig2.add_axes([0.92, 0.15, 0.015, 0.7])  # adjust as needed
fig2.colorbar(pc, cax=cax2, label='RMS OPD (nm/lenslet)')

fig2.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.savefig('IG_Tilt_RMS_Comparison.pdf')


plt.show()

###########################
# Plot 2: IG-Tilt RMS diff #
###########################
fig2, axes2 = plt.subplots(KK, LL, figsize=(2.5*LL, 2.5*KK), sharex=True, sharey=True)
fig2.suptitle('RMS difference of IG-tilt vs SHWFS (nm/lenslet) - no bias', fontsize=14)

for kk in range(KK):
    for ll in range(LL):
        ax = axes2[KK-kk-1, ll] if KK > 1 else axes2[ll]
        data = np.sqrt(var_ig[:, :, kk, ll]).T

        # Optional screen mask overlay
        if (POSSIBLE_SCREEN_TEMP[:, :, kk, ll].T).sum() != 16:
            ax.pcolor(r0*100, dor0, (1 - POSSIBLE_SCREEN_TEMP[:, :, kk, ll]).T * 0.5,
                      vmin=0, vmax=1, cmap='Blues')

        pc = ax.pcolor(r0*100, dor0, data, cmap=color_map,
                       vmin=0, vmax=np.nanmax(np.sqrt(var_ig)), edgecolors='k', linewidths=0.5)

        if kk == KK-1:
            ax.set_title(f'IG - {LABEL_HHA[ll]}')

        if ll == 0:
            ax.set_ylabel(LABEL_RYTOV[kk] + '\nd/r$_0$')

        if kk == 0:
            ax.set_xlabel('r$_0$ (cm)')

# Colorbar axis to the right
cax2 = fig2.add_axes([0.92, 0.15, 0.015, 0.7])  # adjust as needed
fig2.colorbar(pc, cax=cax2, label='RMS OPD (nm/lenslet)')

fig2.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.savefig('IG_Tilt_RMS_Comparison-another.pdf')


plt.show()

#############################################################################

avg = np.nanmean(g_SLOPE - SHWFS_SLOPE, axis=(-1,-2))
var = np.nanvar(g_SLOPE - SHWFS_SLOPE, ddof=1, axis=(-1,-2))
avg[avg==0] = np.nan
var[var==0] = np.nan

varmin = np.nanmin(var)
varmax = np.nanmax(var)
avgmin = np.nanmin(avg)
avgmax = np.nanmax(avg)

min_slope = -np.nanmax([np.abs(SHWFS_SLOPE),np.abs(Ig_SLOPE),np.abs(g_SLOPE)])
max_slope = np.nanmax([np.abs(SHWFS_SLOPE),np.abs(Ig_SLOPE),np.abs(g_SLOPE)])

rms = np.sqrt(np.nanmean((g_SLOPE - SHWFS_SLOPE)**2, axis=(-1,-2)))

LABEL_r0 = np.zeros((II), dtype=object)
for ii in np.arange(II):
    LABEL_r0[ii] = 'r$_0$={:.0f}cm'.format(r0[ii]*100)

LABEL_DOR0 = np.zeros((JJ), dtype=object)
for jj in np.arange(JJ):
    LABEL_DOR0[jj] = 'd/r$_0$={:.1f}'.format(dor0[jj])
    
LABEL_RYTOV = np.zeros((KK), dtype=object)
for kk in np.arange(KK):
    LABEL_RYTOV[kk] = '$\sigma^2_\chi=${:.2f}'.format(Rytov[kk])
    
LABEL_HHA = np.zeros((LL), dtype=object)
for ll in np.arange(LL):
    LABEL_HHA[ll] = 'HA={:.0f}km'.format(H_HA[ll]/1000)



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

data = np.nanstd(Ig_SLOPE - SHWFS_SLOPE,axis=(-1,-2))
data[data==0] = np.nan

V = data


import itertools

def plot_all_cross_sections(V, w, x, y, z):
    param_arrays = [w*100, x, y, z/1000]
    param_labels = ['r$_0$ (cm)', 'd/r$_0$', 'Rytov', 'H$_{HA}$ (km)']
    
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
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
        if axis == 1:
            ax.errorbar(varying_param,Ig_rms_noscintillation[0],yerr=Ig_rms_noscintillation[0]/np.sqrt(2*1024-2),
                        ls=':',color='k',capsize=5)
        # else:
        #     ax.hlines(g_rms_noscintillation,varying_param.min(),varying_param.max(),ls=':',color='k')
        #     ax.hlines(Ig_rms_noscintillation,varying_param.min(),varying_param.max(),ls='-',color='k')
        else:
            for i in range(4):
                ax.errorbar(varying_param,Ig_rms_noscintillation[:,i],
                          yerr=Ig_rms_noscintillation[:,i]/np.sqrt(2*1024-2),
                          ls=':',color='k',capsize=5)
                
        # ax.set_title(f"cros section : {param_labels[axis]}")
        ax.set_xlabel(f"{param_labels[axis]}")
        ax.set_ylabel("OPD (nm)")
        # ax.set_ylim(0,6)
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)
        # if axis == 0:
        #     ax.legend(fontsize='x-small', loc='best')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    # plt.suptitle('Cross sections of measured RMS OPD difference between IG and FT (nm) - no bias')
    plt.savefig('cross-section-measure-ig-ft.pdf',bbox_inches='tight',dpi=600)
    plt.show()
    return

plot_all_cross_sections(V, r0, dor0, Rytov, H_HA)




