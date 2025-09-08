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

# with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\20250626\no-threshold-noise\NoiseSHWFSSlopes\processedData\2NoiseVar.pkl', 'rb') as file:
# with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\20250626\no-bias-noise\NoiseSHWFSSlopes\processedData\2NoiseVar.pkl', 'rb') as file:
with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\NoiseVar.pkl', 'rb') as file:
    input_data = pickle.load(file)
file.close()


keys = input_data.keys()
for item_i in np.arange(len(keys)):
    key_name = list(keys)[item_i]
    data = input_data[key_name]
    globals()[key_name] = data
    

photon_rms *= CONVERT
RON_rms *= CONVERT
THRESHOLD_rms *= CONVERT
NoiseRMS *= CONVERT
Another_threshold *= CONVERT
Noscinphoton_rms *= CONVERT
NoscinRON_rms *= CONVERT
NoscinTHRESHOLD_rms *= CONVERT
NoscinNoiseRMS *= CONVERT
NoscinAnother_threshold *= CONVERT


II = r0.shape[0]
JJ = dor0.shape[0]
KK = Rytov.shape[0]
LL = H_HA.shape[0]
MM = N_phot_list.shape[0]
NN = RON_error.shape[-1]
OO = THRESHOLD_rms.shape[-1]

CONVERT = 500/2/np.pi # data is preconverted but the code is for nonconverted so i have to undo abit

W = 64/2
n_cog = 17
new_atm_error = (0.5*W**-2*(dor0)**(5./3.))**0.5 * CONVERT
new_photon_error = (2*W/N_phot_list)**0.5 * CONVERT
NEW_RON_list = np.round(NEW_RON_list,9)
final_RON = np.unique(NEW_RON_list)
final_RON_expect = (np.pi**2/3.*(final_RON)**2*(n_cog**2)**2/(2)**2)**0.5 * CONVERT

color_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']


# color_list = ['c','m','y','k']

# first_time = [True,True,True,True]
# for ii in np.arange(II):
#     for jj in np.arange(JJ):
#         for kk in np.arange(KK):
#             for ll in np.arange(LL):
#                 if first_time[jj]:
#                     # plt.plot(N_phot_list,(NoiseRMS[ii,jj,kk,ll,:,0,0]).flatten(),
#                     plt.plot(N_phot_list,(photon_rms[ii,jj,kk,ll,:]).flatten()*CONVERT,
#                               color = color_list[jj],label='dor0={}'.format(dor0[jj]))
#                     # plt.hlines(new_atm_error[jj],
#                     #            N_phot_list.min(),
#                     #            N_phot_list.max(),
#                     #            label='atmos noise, dor0={}'.format(dor0[jj]),
#                     #            ls=':',color=color_list[jj])
#                     first_time[jj] = False
#                 else:
#                     plt.plot(N_phot_list,(photon_rms[ii,jj,kk,ll,:]).flatten()*CONVERT,
#                               color = color_list[jj])
# plt.plot(N_phot_list,new_photon_error*CONVERT,ls='--',label=('photon noise model'))
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Number of photon per lenslet')
# plt.ylabel('RMS-OPD (nm)')
# plt.title('Photon noise')
# plt.legend()
# plt.savefig('photon_noise0.pdf')#/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/photon_noise.png')
# plt.show();plt.clf()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Axis info
axes = ['ii', 'jj', 'kk', 'll']
axis_indices = {'ii': 0, 'jj': 1, 'kk': 2, 'll': 3}
axis_labels = {
    'ii': r'r$_0$ (cm)',
    'jj': r'd/r$_0$',
    'kk': r'$\sigma^2_\chi$',
    'll': r'H$_{HA}$ (km)'
}
axis_values = {
    'ii': r0 * 100,
    'jj': dor0,
    'kk': Rytov,
    'll': H_HA / 1000
}

# Broadcast NoScintillation_photon_rms to match photon_rms shape
no_scint_broadcast = np.broadcast_to(
    Noscinphoton_rms[:, :, None, None, :],
    photon_rms.shape
)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
axs = axs.flatten()

for ax_idx, axis_name in enumerate(axes):
    ax = axs[ax_idx]
    idx = axis_indices[axis_name]
    dim_size = photon_rms.shape[idx]

    shown_color_labels = set()
    ax.set_title(f"Contours on {axis_labels[axis_name]}")
    for i in range(dim_size):
        color = color_list[i % len(color_list)]
        value_label = f"{axis_labels[axis_name]}={axis_values[axis_name][i]:.2f}"

        # Create mask for selecting data
        mask = np.zeros(photon_rms.shape[:4], dtype=bool)
        slicer = [slice(None)] * 4
        slicer[idx] = i
        mask[tuple(slicer)] = True

        # PHOTON_RMS
        selected_data = photon_rms[mask]
        reshaped = selected_data.reshape(-1, photon_rms.shape[4])
        mean_vals = np.nanmean(reshaped, axis=0)
        std_vals = np.nanstd(reshaped, axis=0)

        label = value_label if value_label not in shown_color_labels else None
        if label:
            shown_color_labels.add(label)

        ax.errorbar(N_phot_list, mean_vals, yerr=std_vals,
                    label=label, capsize=3, marker='+',
                    linestyle='-', color=color)

        # NO SCINTILLATION
        selected_ns_data = no_scint_broadcast[mask]
        reshaped_ns = selected_ns_data.reshape(-1, photon_rms.shape[4])
        ns_mean_vals = np.nanmean(reshaped_ns, axis=0)
        ns_std_vals = np.nanstd(reshaped_ns, axis=0)

        # No label to avoid double entry for the same color
        ax.errorbar(N_phot_list, ns_mean_vals, yerr=ns_std_vals,
                    capsize=3, marker='x', linestyle='--', color=color)

    # Theory model
    ax.plot(N_phot_list, new_photon_error, ls=':', color='k')

    # Axes setup
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(50, 2e6)
    ax.set_xlabel('Number of photons per lenslet')
    if ax_idx % 2 == 0:
        ax.set_ylabel('RMS-OPD (nm)')
    ax.grid(True)

    # Add clean legend: styles first, then color-coded entries
    style_legend = [
        Line2D([0], [0], color='gray', linestyle='-', marker='+', label='With scintillation'),
        Line2D([0], [0], color='gray', linestyle='--', marker='x', label='No scintillation'),
        Line2D([0], [0], color='black', linestyle=':', label='Theory model')
    ]

    # Get unique color-encoded value labels
    color_handles, color_labels = ax.get_legend_handles_labels()
    ax.legend(handles=style_legend + color_handles, fontsize='small', loc='best')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("photon_noise_errorbars.pdf")
plt.show()


# # N_phot_list[mm], photon_rms[ii,jj,kk,ll,mm]

# color_list = ['c','m','y','k']




first_time = [True,True,True,True]
plt.plot(N_phot_list,new_photon_error,ls=':',label=('model'))
for jj in np.arange(JJ):
    plt.errorbar(N_phot_list,np.nanmean(photon_rms[:,jj],axis=(0,1,2)),
                 yerr=np.nanstd(photon_rms[:,jj],axis=(0,1,2)),
                 color = color_list[jj],label='d/r$_0$={}'.format(dor0[jj]),
                 capsize=5,marker='+')
    plt.errorbar(N_phot_list,np.nanmean(Noscinphoton_rms[:,jj],axis=(0)),
                 yerr=np.nanstd(photon_rms[:,jj],axis=(0,1,2)),
                 color = color_list[jj],
                 capsize=5,marker='x',ls='--')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of photon per lenslet')
plt.ylabel('RMS-OPD (nm)')
# plt.title('Photon noise')
plt.legend()
plt.savefig('photon_noise.pdf')#/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/photon_noise.png')
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
#                                       RON_rms[ii,jj,kk,ll,mm,nn].flatten(),
#                                       color = color_list[jj],label='dor0={}'.format(dor0[jj]), marker='.')
#                             first_time[jj] = False
#                             # plt.hlines(new_atm_error[jj],
#                             #            RON_list[1:].min(),
#                             #            RON_list[1:].max(),
#                             #            label='atmos noise, dor0={}'.format(dor0[jj]),
#                             #            ls=':',color=color_list[jj])
#                         else:
#                             plt.plot(NEW_RON_list[mm,nn],
#                                       RON_rms[ii,jj,kk,ll,mm,nn].flatten(),
#                                       color = color_list[jj], marker='.')
                            
# # plt.plot(NEW_RON_list[:,1:].flatten(),RON_error[:,1:].flatten(),ls='--',label=('ron noise model'))
# plt.plot(final_RON,final_RON_expect,ls='--',label=('ron model'))
# plt.legend()
# plt.ylabel('RMS-OPD (rad)')
# plt.xlabel('Readout e- std per pixel compare to total photon in a lenslet')
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('readout_noise0.png')#/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/readout_noise.png')
# plt.show();plt.clf()

# def plot_RON_rms_grouped_by(axis='jj'):
#     axes = {'ii': 0, 'jj': 1, 'kk': 2, 'll': 3}
#     group_axis = axes[axis]
#     group_size = [II, JJ, KK, LL][group_axis]
#     label_list = [r0, dor0, Rytov, H_HA][group_axis]  # Replace with correct labels

#     first_time = [True] * group_size

#     for ii in range(II):
#         for jj in range(JJ):
#             for kk in range(KK):
#                 for ll in range(LL):
#                     for mm in range(MM):
#                         for nn in range(1, NN):  # skip nn == 0
#                             indices = [ii, jj, kk, ll, mm]
#                             group_value = indices[group_axis]

#                             if first_time[group_value]:
#                                 plt.plot(NEW_RON_list[mm, nn],
#                                          RON_rms[ii, jj, kk, ll, mm, nn].flatten(),
#                                          color=color_list[group_value],
#                                          label=f'{axis}={label_list[group_value]}',
#                                          marker='.')
#                                 first_time[group_value] = False
#                             else:
#                                 plt.plot(NEW_RON_list[mm, nn],
#                                          RON_rms[ii, jj, kk, ll, mm, nn].flatten(),
#                                          color=color_list[group_value],
#                                          marker='.')

#     # Add RON model overlay
#     plt.plot(final_RON, final_RON_expect, ls='--', label='ron model')
#     plt.legend()
#     plt.ylabel('RMS-OPD (rad)')
#     plt.xlabel('Readout e- std per pixel compare to total photon in a lenslet')
#     plt.xscale('log')
#     plt.yscale('log')

#     # Save figure
#     plt.savefig(f'readout_noise_by_{axis}.png')
#     plt.show()
#     plt.clf()


# plot_RON_rms_grouped_by('ii')  # repeat but group by ii
# plot_RON_rms_grouped_by('jj')  # original
# plot_RON_rms_grouped_by('kk')  # ...
# plot_RON_rms_grouped_by('ll')

# axes = ['ii', 'jj', 'kk', 'll']
# axis_indices = {'ii': 0, 'jj': 1, 'kk': 2, 'll': 3}
# axis_labels = {'ii': r'r$_0$ (cm)', 'jj': r'd/r$_0$', 'kk': r'$\sigma^2_\chi$', 'll': r'H$_{HA}$ (km)'}
# axis_values = {'ii': r0*100, 'jj': dor0, 'kk': Rytov, 'll': H_HA/1000}

# for axis_name in axes:
#     ax_idx = axis_indices[axis_name]
#     dim_size = RON_rms.shape[ax_idx]
#     avg_axes = [i for i in range(4) if i != ax_idx]

#     final_rms = np.zeros((dim_size, final_RON.shape[0]), dtype=float)
#     final_count = np.zeros((dim_size, final_RON.shape[0]), dtype=float)
#     final_error = np.zeros((dim_size, final_RON.shape[0]), dtype=float)

#     for i in range(RON_rms.shape[0]):
#         for j in range(RON_rms.shape[1]):
#             for k in range(RON_rms.shape[2]):
#                 for l in range(RON_rms.shape[3]):
#                     for m in range(RON_rms.shape[4]):
#                         for n in range(RON_rms.shape[5]):
#                             if np.isnan(RON_rms[i, j, k, l, m, n]):
#                                 continue

#                             group_val = [i, j, k, l][ax_idx]
#                             idx = np.where(final_RON == NEW_RON_list[m, n])[0]
#                             if idx.size > 0:
#                                 final_rms[group_val, idx[0]] += RON_rms[i, j, k, l, m, n]**2
#                                 final_count[group_val, idx[0]] += 1

#     final_rms = np.sqrt(final_rms / np.maximum(final_count, 1))

#     for i in range(RON_rms.shape[0]):
#         for j in range(RON_rms.shape[1]):
#             for k in range(RON_rms.shape[2]):
#                 for l in range(RON_rms.shape[3]):
#                     for m in range(RON_rms.shape[4]):
#                         for n in range(RON_rms.shape[5]):
#                             if np.isnan(RON_rms[i, j, k, l, m, n]):
#                                 continue

#                             group_val = [i, j, k, l][ax_idx]
#                             idx = np.where(final_RON == NEW_RON_list[m, n])[0]
#                             if idx.size > 0:
#                                 diff = RON_rms[i, j, k, l, m, n] - final_rms[group_val, idx[0]]
#                                 final_error[group_val, idx[0]] += diff**2

#     final_error = np.sqrt(final_error / np.maximum(final_count, 1))

#     # --- Plotting ---
#     for i in range(dim_size):
#         plt.errorbar(final_RON[1:], final_rms[i, 1:], yerr=final_error[i, 1:],
#                      label=f"{axis_labels[axis_name]}={axis_values[axis_name][i]:.2f}",
#                      capsize=5, marker='o', ls='-', color=color_list[i])

#     plt.plot(final_RON, final_RON_expect, ls='--', label='model', color='k')
#     plt.legend()
#     plt.ylabel('RMS-OPD (rad)')
#     plt.xlabel('RON e-/pxl / total photon in subap')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.title(f"Averaged over 3 axes, grouped by {axis_name}")
#     plt.tight_layout()
#     plt.savefig(f'readout_noise_{axis_name}.png')
#     plt.show(); plt.clf()


# Broadcast NoScintillation_photon_rms to match photon_rms shape
no_scint_broadcast = np.broadcast_to(
    NoscinRON_rms[:, :, None, None, :],
    RON_rms.shape
)

axes = ['ii', 'jj', 'kk', 'll']
axis_indices = {'ii': 0, 'jj': 1, 'kk': 2, 'll': 3}
axis_labels = {'ii': r'r$_0$ (cm)', 'jj': r'd/r$_0$', 'kk': r'$\sigma^2_\chi$', 'll': r'H$_{HA}$ (km)'}
axis_values = {'ii': r0*100, 'jj': dor0, 'kk': Rytov, 'll': H_HA/1000}

fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
axs = axs.flatten()  # Make it easy to index with a single number

for plot_idx, axis_name in enumerate(axes):
    ax = axs[plot_idx]
    ax_idx = axis_indices[axis_name]
    dim_size = RON_rms.shape[ax_idx]

    final_rms = np.zeros((dim_size, final_RON.shape[0]), dtype=float)
    final_count = np.zeros((dim_size, final_RON.shape[0]), dtype=float)
    final_error = np.zeros((dim_size, final_RON.shape[0]), dtype=float)
    
    nsfinal_rms = np.zeros((dim_size, final_RON.shape[0]), dtype=float)
    nsfinal_count = np.zeros((dim_size, final_RON.shape[0]), dtype=float)
    nsfinal_error = np.zeros((dim_size, final_RON.shape[0]), dtype=float)

    for i in range(RON_rms.shape[0]):
        for j in range(RON_rms.shape[1]):
            for k in range(RON_rms.shape[2]):
                for l in range(RON_rms.shape[3]):
                    for m in range(RON_rms.shape[4]):
                        for n in range(RON_rms.shape[5]):
                            if np.isnan(RON_rms[i, j, k, l, m, n]):
                                continue

                            group_val = [i, j, k, l][ax_idx]
                            idx = np.where(final_RON == NEW_RON_list[m, n])[0]
                            if idx.size > 0:
                                final_rms[group_val, idx[0]] += RON_rms[i, j, k, l, m, n]**2
                                final_count[group_val, idx[0]] += 1
                                nsfinal_rms[group_val, idx[0]] += no_scint_broadcast[i, j, k, l, m, n]**2
                                nsfinal_count[group_val, idx[0]] += 1

    final_rms = np.sqrt(final_rms / np.maximum(final_count, 1))
    nsfinal_rms = np.sqrt(nsfinal_rms / np.maximum(nsfinal_count, 1))

    for i in range(RON_rms.shape[0]):
        for j in range(RON_rms.shape[1]):
            for k in range(RON_rms.shape[2]):
                for l in range(RON_rms.shape[3]):
                    for m in range(RON_rms.shape[4]):
                        for n in range(RON_rms.shape[5]):
                            if np.isnan(RON_rms[i, j, k, l, m, n]):
                                continue

                            group_val = [i, j, k, l][ax_idx]
                            idx = np.where(final_RON == NEW_RON_list[m, n])[0]
                            if idx.size > 0:
                                diff = RON_rms[i, j, k, l, m, n] - final_rms[group_val, idx[0]]
                                final_error[group_val, idx[0]] += diff**2
                                nsdiff = no_scint_broadcast[i, j, k, l, m, n] - nsfinal_rms[group_val, idx[0]]
                                nsfinal_error[group_val, idx[0]] += nsdiff**2

    final_error = np.sqrt(final_error / np.maximum(final_count, 1))
    nsfinal_error = np.sqrt(nsfinal_error / np.maximum(nsfinal_count, 1))

    # Plotting to subplot
    for i in range(dim_size):
        ax.errorbar(final_RON[1:], final_rms[i, 1:], yerr=final_error[i, 1:],
                    label=f"{axis_labels[axis_name]}={axis_values[axis_name][i]:.2f}",
                    capsize=3, marker='x', ls='-', color=color_list[i])
        ax.errorbar(final_RON[1:], nsfinal_rms[i, 1:], yerr=nsfinal_error[i, 1:],
                    label=f"noscin-{axis_labels[axis_name]}={axis_values[axis_name][i]:.2f}",
                    capsize=3, marker='+', ls='-', color=color_list[i])

    ax.plot(final_RON, final_RON_expect, ls='--', label='model', color='k')
    # ax.set_title(f"Averaged over 3 axes\nGrouped by {axis_labels[axis_name]}")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('RON e-/pxl / total photon in subap')
    if plot_idx % 2 == 0:
        ax.set_ylabel('RMS-OPD (nm)')
    ax.grid(True)
    ax.legend(fontsize='small')

plt.tight_layout()
plt.savefig('readout_noise_subfigures.pdf')
plt.show()

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




plt.plot(final_RON,final_RON_expect,ls='--',label=('model'))
plt.errorbar(final_RON[1:],
             final_RON_rms[1:],
             yerr=final_RON_rms_error[1:],
             ls='-',label=('ron'),capsize=5,marker='o')
plt.legend()
plt.ylabel('RMS-OPD (nm)')
plt.xlabel('ron e-/pxl / total photon in subap')
plt.xscale('log')
plt.yscale('log')
plt.savefig('readout_noise.pdf')#/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/readout_noise.png')
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

plt.plot(final_RON,final_RON_expect,ls='--',label=('model - no threshold'))
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
plt.ylabel('RMS-OPD (nm)')
plt.xlabel('ron e-/pxl / total photon in subap')#'Readout e- std per pixel compare to total photon in a lenslet')
plt.xscale('log')
plt.yscale('log')
plt.savefig('threshold.pdf')#/mt/cfai/pkongkaew/NoiseSHWFSSlopes/processedData/readout_noise.png')
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
# plt.suptitle('Effect of threshold on photon noise')
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
    
    axs[jj//2,jj%2].plot(N_phot_list,new_photon_error,ls='--',label=('model - no threshold'),color=color_list[0])
    for oo in np.arange(OO):
        axs[jj//2,jj%2].errorbar(N_phot_list,
                     final_phot_threshold[:,oo],
                     yerr=final_phot_threshold_error[:,oo],
                     label='threshold={}'.format(Threshold_list[oo]),color=color_list[oo+1],
                     capsize=5,marker='o')
    axs[jj//2,jj%2].set_title('d/r$_0$={}'.format(dor0[jj]))
        
    axs[jj//2,jj%2].legend()
    axs[jj//2,jj%2].set_xscale('log')
    axs[jj//2,jj%2].set_yscale('log')
    axs[jj//2,jj%2].set_ylim(MIN,MAX)
    if jj//2 == 1:
        axs[jj//2,jj%2].set_xlabel('N_ph')
    if jj%2 == 0:
        axs[jj//2,jj%2].set_ylabel('RMS-OPD (nm)')

plt.tight_layout
plt.savefig('threshold-phot.pdf')
plt.show();plt.clf()



