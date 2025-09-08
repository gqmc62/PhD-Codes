# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:31:33 2025

@author: gqmc62
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
# import MyWFS as MWFS # square aperture
import my_WFSC as MWFS # circular aperture
from tqdm import tqdm

with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\Ultimate_input.pkl','rb') as file:
    input_data = pickle.load(file)
keys = input_data.keys()
for item_i in np.arange(len(keys)):
    key_name = list(keys)[item_i]
    data = input_data[key_name]
    globals()[key_name] = data
del input_data

with open('D:\MyFiles\CH4\CH4\ALLCh4\AllIGPhase\AllIGPhase\ps_GL\ps_GL_r02dor20rytov01h05_0000.pkl','rb') as file:
    data = pickle.load(file)
    
n = data.shape[0]
    
load_r0 = r0_GL[3,0,0]
load_d = 2*load_r0
load_scale = load_d/n

FT = MWFS.OneStepSHWFS(n)
G = MWFS.WrappedGTilt(n)
IG = MWFS.IgTilt(n)
FOCUS = MWFS.Focus(n)


CONVERT = 500/2/np.pi

###############################################################################

# new_r0_list = r0
# new_dor0_list = dor0

sft = np.zeros((II,JJ,TARGET,2),dtype=float)
sg = np.zeros((II,JJ,TARGET,2),dtype=float)
sig = np.zeros((II,JJ,TARGET,2),dtype=float)

Images = np.zeros((II,JJ,TARGET,n,n),dtype=float)

for repeat in tqdm(range(TARGET)):
    
    with open('D:\MyFiles\CH4\CH4\ALLCh4\AllIGPhase\AllIGPhase\ps_GL\ps_GL_r02dor20rytov01h05_{:04d}.pkl'.format(repeat),'rb') as file:
        data = pickle.load(file)
        
    # U = np.exp(1j*data)
        
    # plt.imshow(np.angle(U))
    # plt.colorbar()
    # plt.show()
    
    for ii in range(II):
        for jj in range(JJ):
            
            
            new_r0 = r0[ii]#0.11
            new_dor0 = dor0[jj]#0.5
            new_d = new_dor0*new_r0
            
            new_data = data*((load_r0/load_d) * (new_d/new_r0))**(5./6.)
                
            
            new_U = np.exp(1j*new_data)
                
            # plt.imshow(np.angle(new_U))
            # plt.colorbar()
            # plt.show()
            
            sft[ii,jj,repeat] = FT.measure_no_noise_no_threshold(new_U)
            sig[ii,jj,repeat] = IG.measure(new_U)
            sg[ii,jj,repeat] = G.measure(new_U)
            Images[ii,jj,repeat] = FOCUS.focus(new_U)
            
for ii in range(II):
    for jj in range(JJ):      
        with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\noScintillationWFSImage\Image_noscintillation_r{:02d}dor{:02d}.pkl'.format(
                int(r0[ii]*100), int(dor0[jj]*10)), 'wb') as file:
            pickle.dump(Images[ii,jj],file)        
            
            
            
with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\Data\noscintillation.pkl','wb') as file:
    pickle.dump({
        'g_SLOPE_noscintillation' : sg,
        'Ig_SLOPE_noscintillation' : sig,
        'SHWFS_SLOPE_noscintillation' : sft,
        }, file)


# plt.plot(sft.flatten()*CONVERT,sig.flatten()*CONVERT,ls='',marker='.',label='ig')
# plt.plot(sft.flatten()*CONVERT,sg.flatten()*CONVERT,ls='',marker='.',label='g')
# plt.plot(sft.flatten()*CONVERT,sft.flatten()*CONVERT,ls='',marker='.',label='ft')
# plt.legend()
# plt.xlabel('OPD (nm/lenslet)')
# plt.ylabel('OPD (nm/lenslet)')
# plt.show()


# plt.plot(sft.flatten()*CONVERT,(sig-sft).flatten()*CONVERT,ls='',marker='.',label='ig')
# plt.plot(sft.flatten()*CONVERT,(sg-sft).flatten()*CONVERT,ls='',marker='.',label='g')
# plt.hlines([0],-1500,1500,label='ft',color='g')
# plt.legend()
# plt.xlabel('OPD (nm/lenslet)')
# plt.ylabel('OPD difference to FT (nm/lenslet)')
# plt.show()

rms_ig = np.std(sig-sft,axis=(-1,-2))
rms_g = np.std(sg-sft,axis=(-1,-2))

color_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
for ii in range(II):
    plt.plot(dor0,rms_ig[ii]*CONVERT,ls='-',marker='+',label='G : r0:{:.0f}cm'.format(r0[ii]*100),color=color_list[ii])
    plt.plot(dor0,rms_g[ii]*CONVERT,ls=':',marker='x',label='IG : r0:{:.0f}cm'.format(r0[ii]*100),color=color_list[ii])
plt.legend()
plt.xlabel('d/r$_0$')
plt.ylabel('RMS OPD (nm/lenslet)')
plt.show()

for jj in range(JJ):
    plt.plot(r0*100,rms_ig[:,jj]*CONVERT,ls='-',marker='+',label='G : d/r$_0$ : {:.1f}'.format(dor0[jj]),color=color_list[jj])
    plt.plot(r0*100,rms_g[:,jj]*CONVERT,ls=':',marker='x',label='IG : d/r$_0$ : {:.1f}'.format(dor0[jj]),color=color_list[jj])
    
plt.legend()
plt.xlabel('r$_0$ (cm)')
plt.ylabel('RMS OPD (nm/lenslet)')
plt.show()


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

for jj in range(JJ):
    plt.plot(r0*100, rms_ig[:, jj]*CONVERT,
             ls='-', marker='+',
             label='d/r$_0$ : {:.1f}'.format(dor0[jj]),
             color=color_list[jj])

    plt.plot(r0*100, rms_g[:, jj]*CONVERT,
             ls=':', marker='x',
             label='_nolegend_',  # suppress duplicate d/r0 label
             color=color_list[jj])

# Custom legend entries for G and IG
custom_lines = [
    Line2D([0], [0], color='black', marker='+', linestyle='-', label='G (marker +, solid)'),
    Line2D([0], [0], color='black', marker='x', linestyle=':', label='IG (marker x, dotted)')
]

# Legend for d/r0 values
color_legend = [
    Line2D([0], [0], color=color_list[jj], marker='o', linestyle='None', label='d/r$_0$ : {:.1f}'.format(dor0[jj]))
    for jj in range(JJ)
]

# Place legends
plt.legend(handles=custom_lines + color_legend,
           loc='center left', bbox_to_anchor=(1, 0.5))

plt.xlabel('r$_0$ (cm)')
plt.ylabel('RMS OPD (nm/lenslet)')
plt.tight_layout()
plt.savefig(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\noscintillation-benchmark.pdf',bbox_inches='tight')
plt.show()

