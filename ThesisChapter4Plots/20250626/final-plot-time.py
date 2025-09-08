# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 18:27:03 2025

@author: gqmc62
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

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


with open(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter4Plots\20250626\time_use_time_all.pkl','rb') as file:
    input_data = pickle.load(file)
    
keys = input_data.keys()
for item_i in np.arange(len(keys)):
    key_name = list(keys)[item_i]
    data = input_data[key_name]
    globals()[key_name] = data
    
MAX = 10**np.ceil(np.log10(np.max([t_FT_mean,t_G_mean,t_IG_mean,t_IPS_mean,t_PROP_mean])))
MIN = 10**np.floor(np.log10(np.min([t_FT_mean,t_G_mean,t_IG_mean,t_IPS_mean,t_PROP_mean])))

plt.clf()
# plt.figure(figsize=(6,4))
plt.errorbar(2**power,t_FT_mean,yerr=t_FT_error,
             label='FT',marker='o',capsize=5)
plt.errorbar(2**power,t_IG_mean,yerr=t_IG_error,
             label='IG',marker='o',capsize=5)
plt.errorbar(2**power,t_G_mean,yerr=t_G_error,
             label='G',marker='o',capsize=5)
# plt.errorbar(2**power,t_correlate_mean,yerr=t_correlate_error,
#               label='correlate',marker='o',capsize=5)
plt.xlim(1,3000)
plt.ylim(MIN,2e-1)
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper left')
plt.ylabel('Mean time to compute (s)')
plt.xlabel('Simulation elements across subaperture ($n$)')
plt.tight_layout()
plt.savefig('time-ft-ig-g.pdf',bbox_inches='tight',dpi=600)
plt.show()

plt.clf()
# plt.figure(figsize=(6,4))
plt.errorbar(2**power,t_IPS_mean,yerr=t_IPS_error,
             label='Phase screen',marker='o',capsize=5)
plt.errorbar(2**power,t_PROP_mean,yerr=t_PROP_error,
             label='Propagation',marker='o',capsize=5)
plt.xscale('log')
plt.yscale('log')
plt.xlim(1,2048)
plt.ylim(MIN,MAX)
plt.legend()
plt.ylabel('Mean time to compute (s)')
plt.xlabel('Simulation elements across simulation ($N$)')
plt.tight_layout()
plt.savefig('time-ips-prop.pdf',bbox_inches='tight',dpi=600)
plt.show()

plt.clf()
# plt.figure(figsize=(6,4))
ratio = (t_FT_mean / t_IG_mean)**-1
ratio_error = ((t_FT_error/t_FT_mean)**2 + (t_IG_error/t_IG_mean)**2)**0.5*(t_FT_mean/t_IG_mean)**-1
plt.errorbar(2**power,ratio,yerr=ratio_error,
             label='IG/FT',marker='o',capsize=5)
ratio = (t_FT_mean / t_G_mean)**-1
ratio_error = ((t_FT_error/t_FT_mean)**2 + (t_G_error/t_G_mean)**2)**0.5*(t_FT_mean/t_G_mean)**-1
plt.errorbar(2**power,ratio,yerr=ratio_error,
             label='G/FT',marker='o',capsize=5)
plt.xscale('log')
# plt.yscale('log')
plt.xlim(1,2048)
plt.ylim(0,1)
plt.legend()
plt.ylabel('Time ratio')
plt.xlabel('Simulation elements across subaperture ($n$)')
plt.tight_layout()
plt.savefig('time-ratio-ft-ig-g.pdf',bbox_inches='tight',dpi=600)
plt.show()

###############################################################################
# calculate table showing what's the expected time to simulate ...
# i forgot to comment this part. this is definitely too hard to understand??
# 
###############################################################################

memory_FT = 1344*(2**power)**2
memory_IG = 512*(2**power)**2 - 256*(2**power)
memory_G = 320*(2**power)**2 - 128*(2**power)

plt.clf()
# plt.figure(figsize=(6,4))
plt.plot(2**power, memory_FT/8, label='FT')
plt.plot(2**power, memory_IG/8, label='IG')
plt.plot(2**power, memory_G/8, label='G')
plt.legend()
plt.xlabel('Simulation elements across subaperture ($n$)')
plt.ylabel('Memory required (Byte)')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1,2048)
# plt.ylim(MIN,MAX)
plt.tight_layout()
plt.savefig('memory-ft-ig-g.pdf',bbox_inches='tight',dpi=600)
plt.show()

print('prop',10**((np.log10(t_PROP_mean[-1]/t_PROP_mean[-2])/np.log10(1024/512))*np.log10(np.array([
    800,4295,1097,6789,261,130,636,1128,965,258,2413])/1024)))
print('IPS',10**((np.log10(t_IPS_mean[-1]/t_IPS_mean[-2])/np.log10(1024/512))*np.log10(np.array([
    1291,5163,2118,8471,428,209,1203,1522,1184,243,972])/1024)))
print('FT-16,FT-64',t_FT_mean[3],t_FT_mean[5])
print('IG-16,IG-64',t_IG_mean[3],t_IG_mean[5])
print('G-16,G-64',t_G_mean[3],t_G_mean[5])
print('Correlate-16,64',t_correlate_mean[3],t_correlate_mean[5])
