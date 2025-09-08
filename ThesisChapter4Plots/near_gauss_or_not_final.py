# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 15:54:08 2025

@author: gqmc62
"""

import os
import pickle
from matplotlib import pyplot as plt
import numpy as np

with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\Ultimate_input.pkl','rb') as file:
    input_data = pickle.load(file)
keys = input_data.keys()
for item_i in np.arange(len(keys)):
    key_name = list(keys)[item_i]
    data = input_data[key_name]
    globals()[key_name] = data
del input_data

# r0 = np.array([0.11,0.08,0.05,0.02])
# dor0 = np.array([0.5,1,2,3])
# Rytov = np.array([0.01,0.03,0.1,0.3])
# H_HA = np.array([5,10,15,20])*1e3
# N_phot_list = 10**np.array([2,3,4,5,6])
# RON_list = np.array([0.1,0.3,1,3,10])
# Threshold_list = np.array([0,1e-4,1e-3,1e-1,5e-1])

with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\Slopes\SHWFSNoiseScintillation0000.pkl', 'rb') as file:
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

# print(II*JJ*KK*LL*MM*NN*OO)

# COUNT1 = input_data['NoNoiseSHWFSTilt'].shape[-2]
# COUNT2 = input_data['SHWFSTilt'].shape[-1]
COUNT1 = input_data['Noise'].shape[-3]
COUNT2 = input_data['Noise'].shape[-1]
del input_data

INDICES = np.array([II,JJ,KK,LL,MM,NN,OO])

address = r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\NoiseSHWFSSlopes\processedData'
os.chdir(address)

filename = '21AREA.pkl'

with open(filename,'rb') as file:
    data = pickle.load(file)

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
    LABEL_Threshold[oo] = 'Threshold={:.0e}'.format(Threshold_list[oo])

LABEL_MASTER = [LABEL_r0,LABEL_DOR0,LABEL_RYTOV,LABEL_HHA,LABEL_N_phot,LABEL_RON,LABEL_Threshold]

# for dim in np.arange(len(data.shape)) - len(data.shape):
#     for i in range(INDICES[dim]):
#         toplot = eval('data[' + (len(data.shape) + dim)*':,' + '{:d}]'.format(i)).flatten()
#         # print(toplot)
#         plt.hist(toplot,label=LABEL_MASTER[dim][i],density=True,bins=50,range=(data.min(),data.max()),alpha=0.5)
#     plt.ylim(0,100)
#     plt.legend()
#     plt.show()

# notClose = (data>=0.1)
# for ii in range(II):
#     for jj in range(JJ):
#         for kk in range(KK):
#             for ll in range(LL):
#                 for mm in range(MM):
#                     for nn in range(NN):
#                         for oo in range(OO):
#                             if notClose[ii,jj,kk,ll,mm,nn,oo]:
#                                 label = 'AREA={:.2f}%, '.format(data[ii,jj,kk,ll,mm,nn,oo])
#                                 label += LABEL_MASTER[0][ii] + ', '
#                                 label += LABEL_MASTER[1][jj] + ', '
#                                 label += LABEL_MASTER[2][kk] + ', '
#                                 label += LABEL_MASTER[3][ll] + ', '
#                                 label += LABEL_MASTER[4][mm] + ', '
#                                 label += LABEL_MASTER[5][nn] + ', '
#                                 label += LABEL_MASTER[6][oo]
#                                 print(label)
#                                 # print(ii,jj,kk,ll,mm,nn,oo)
                                
analyse = np.zeros_like(data)
mask = np.ones_like(POSSIBLE,dtype=float)
mask[POSSIBLE==False] = np.nan
for mm in range(MM):
    for nn in range(NN):
        for oo in range(OO):
            analyse[...,mm,nn,oo] = mask*data[...,mm,nn,oo]
            
# for i in range(5):
#     plt.plot(100*np.sort(analyse[...,i].flatten()),ls='-',marker='',label=LABEL_Threshold[i])
#     plt.yscale('log')
# plt.legend()
# plt.ylabel('Area difference between\nthe measured and gaussian distributions\nof the same mean and var (%)')
# plt.xlabel('arbitrary element indices\nsorted by increasing area difference')
# plt.hlines([10],0,3200,color='k')
# plt.show()
            
# for i in range(5):
#     plt.plot(100*np.sort(analyse[...,i,:].flatten()),ls='-',marker='',label=LABEL_RON[i])
#     plt.yscale('log')
# plt.legend()
# plt.ylabel('Area difference between\nthe measured and gaussian distributions\nof the same mean and var (%)')
# plt.xlabel('arbitrary element indices\nsorted by increasing area difference')
# plt.hlines([10],0,3200,color='k')
# plt.show()


# for i in range(5):
#     plt.plot(100*np.sort(analyse[...,i,:,:].flatten()),ls='-',marker='',label=LABEL_N_phot[i])
#     plt.yscale('log')
# plt.legend()
# plt.ylabel('Area difference between\nthe measured and gaussian distributions\nof the same mean and var (%)')
# plt.xlabel('arbitrary element indices\nsorted by increasing area difference')
# plt.hlines([10],0,3200,color='k')
# plt.show()

# for i in range(4):
#     plt.plot(100*np.sort(analyse[...,i,:,:,:].flatten()),ls='-',marker='',label=LABEL_HHA[i])
#     plt.yscale('log')
# plt.legend()
# plt.ylabel('Area difference between\nthe measured and gaussian distributions\nof the same mean and var (%)')
# plt.xlabel('arbitrary element indices\nsorted by increasing area difference')
# plt.hlines([10],0,4625,color='k')
# plt.show()

# for i in range(4):
#     plt.plot(100*np.sort(analyse[...,i,:,:,:,:].flatten()),ls='-',marker='',label=LABEL_RYTOV[i])
#     plt.yscale('log')
# plt.legend()
# plt.ylabel('Area difference between\nthe measured and gaussian distributions\nof the same mean and var (%)')
# plt.xlabel('arbitrary element indices\nsorted by increasing area difference')
# plt.hlines([10],0,4875,color='k')
# plt.show()
            


# for i in range(4):
#     plt.plot(100*np.sort(analyse[...,i,:,:,:,:,:].flatten()),ls='-',marker='',label=LABEL_DOR0[i])
#     plt.yscale('log')
# plt.legend()
# plt.ylabel('Area difference between\nthe measured and gaussian distributions\nof the same mean and var (%)')
# plt.xlabel('arbitrary element indices\nsorted by increasing area difference')
# plt.hlines([10],0,6000,color='k')
# plt.show()

# for i in range(4):
#     plt.plot(100*np.sort(analyse[...,i,:,:,:,:,:,:].flatten()),ls='-',marker='',label=LABEL_r0[i])
#     plt.yscale('log')
# plt.legend()
# plt.ylabel('Area difference between\nthe measured and gaussian distributions\nof the same mean and var (%)')
# plt.xlabel('arbitrary element indices\nsorted by increasing area difference')
# plt.hlines([10],0,6000,color='k')
# plt.show()


#############################################################################


import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_all_nd_cross_sections(data, axis_labels=None, axis_values=None, log_axes=None):
    ndim = data.ndim
    shape = data.shape

    if axis_labels is None:
        axis_labels = [f"Axis {i}" for i in range(ndim)]
    if axis_values is None:
        axis_values = [np.arange(s) for s in shape]
    if log_axes is None:
        log_axes = []

    # Layout: enough rows for all subplots
    ncols = 2
    nrows = (ndim + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axs = axs.flatten()

    for axis_to_plot in range(ndim):
        # fig, axs = plt.subplots(1, 1)
        ax = axs[axis_to_plot]
        other_axes = [i for i in range(ndim) if i != axis_to_plot]
        other_ranges = [range(shape[i]) for i in other_axes]

        for idx_combo in itertools.product(*other_ranges):
            # print(idx_combo)
            slicer = [slice(None)] * ndim
            for a, val in zip(other_axes, idx_combo):
                slicer[a] = val
            y = data[tuple(slicer)]
            x = axis_values[axis_to_plot]
            if axis_to_plot != 5:
                ax.plot(x, y, alpha=0.3)
            else:
                for i in range(5):
                    
                    x = RON_list / N_phot_list[idx_combo[4]]
                    # print(idx_combo)
                    ax.plot(x,y,alpha=0.3)
        # plt.show()

        ax.set_xlabel(axis_labels[axis_to_plot])
        ax.grid(True)

        if axis_to_plot in log_axes:
            ax.set_xscale('log')

    for i in range(ndim, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.suptitle("Shared distribution area (%)", fontsize=16)
    os.chdir(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots')
    plt.savefig('cross-section-gauss-area.png',bbox_inches='tight',dpi=500)
    plt.show()
    return


axis_labels = [
    'r0 (cm)', 'd/r$_0$', '$\sigma^2_\\chi$', 'H$_{HA}$ (km)',
    'N$_{phot}$ (photo e-)', 'N$_{RON}/$N$_{phot}$', 'Normalized threshold level'
]

axis_values = [
    r0 * 100,             # r0 in cm
    dor0,                 # d/r0
    Rytov,                # σ²_χ
    H_HA / 1000,          # H_HA in km
    N_phot_list,          # photons
    RON_list,  # RON / photon
    Threshold_list        # threshold
]

log_axes = [2,4,5,6]  # example: r0, σ²_χ, N_phot, RON/phot in log scale

plot_all_nd_cross_sections(data * 100, axis_labels, axis_values, log_axes)










