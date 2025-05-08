# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:18:58 2025

@author: PuttiwatKongkaew
"""

import numpy as np
import matplotlib.pyplot as plt
import pstats
import pickle

TEST = 11
LOOP = 10000
ORDER = 10
size = 2**(np.arange(ORDER)+1)

# ig_time = np.zeros((ORDER,TEST),dtype=float)
# ft_time = np.zeros((ORDER,TEST),dtype=float)
# ps_time = np.zeros((ORDER,TEST),dtype=float)
# prop_time = np.zeros((ORDER,TEST),dtype=float)

# for order in np.arange(ORDER):

#     for test in np.arange(TEST):
    
#         stats = pstats.Stats('profiling_result{:d}-{:d}.txt'.format(size[order],test))
#         stats.sort_stats('time')
        
#         loc_ig = 0
#         loc_ft = 0
#         loc_ps = 0
#         loc_prop = 0
        
#         for i in np.arange(len(stats.fcn_list)):
#             if 'measure' == stats.fcn_list[i][2]:
#                 loc_ig = i
#             if 'measure_no_noise_no_threshold' == stats.fcn_list[i][2]:
#                 loc_ft = i
#             if 'prop' == stats.fcn_list[i][2]:
#                 loc_prop = i
#             if 'get_new_row' == stats.fcn_list[i][2]:
#                 loc_ps = i
                
#         ig_time[order,test] = stats.stats[stats.fcn_list[loc_ig]][3]/LOOP
#         ft_time[order,test] = stats.stats[stats.fcn_list[loc_ft]][3]/LOOP
#         ps_time[order,test] = stats.stats[stats.fcn_list[loc_ps]][3]/LOOP
#         prop_time[order,test] = stats.stats[stats.fcn_list[loc_prop]][3]/LOOP

# data = {'size' : size,
#         'ig_time' : ig_time,
#         'ft_time' : ft_time,
#         'ps_time' : ps_time,
#         'prop_time' : prop_time}

# with open('combined-data.pkl','wb') as file:
#     pickle.dump(data,file)

with open('combined-data-old.pkl','rb') as file:
    data = pickle.load(file)
    
eom_factor1 = 1#np.sqrt(LOOP)

size = data['size']
ig_time = data['ig_time']
ft_time = data['ft_time']
ps_time = data['ps_time']
prop_time = data['prop_time']

plt.errorbar(size,ig_time.mean(-1),yerr=ig_time.std(-1)*eom_factor1,capsize=5,marker='o',label='ig')
plt.errorbar(size,ft_time.mean(-1),yerr=ft_time.std(-1)*eom_factor1,capsize=5,marker='o',label='ft')
# plt.errorbar(size,ps_time.mean(-1),yerr=ps_time.std(-1),capsize=5,marker='o',label='phase screen')
# plt.errorbar(size,prop_time.mean(-1),yerr=prop_time.std(-1),capsize=5,marker='o',label='propagation')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$n_{in}$ (pixel)')
plt.ylabel('time (s)')
plt.legend()
plt.savefig('wfs-time.png')
plt.show()

plt.errorbar(size,
             (ft_time.mean(-1)/ig_time.mean(-1))**-1,
             yerr=((ft_time.mean(-1)/ig_time.mean(-1))**-2
                   *((ig_time.std(-1)/ig_time.mean(-1))**2
                     + (ft_time.std(-1)/ft_time.mean(-1))**2))**0.5*eom_factor1,
             capsize=5,
             marker='o')
plt.xscale('log')
# plt.yscale('log')
plt.xlabel('$n_{in}$ (pixel)')
plt.ylabel('ig time normalized to ft time')
plt.savefig('wfs-time-gain.png')
plt.show()

# plt.errorbar(size,ig_time.mean(-1),yerr=ig_time.std(-1),capsize=5,marker='o',label='ig')
# plt.errorbar(size,ft_time.mean(-1),yerr=ft_time.std(-1),capsize=5,marker='o',label='ft')
plt.errorbar(size,ps_time.mean(-1),yerr=ps_time.std(-1)*eom_factor1,capsize=5,marker='o',label='phase screen')
plt.errorbar(size,prop_time.mean(-1),yerr=prop_time.std(-1)*eom_factor1,capsize=5,marker='o',label='propagation')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$N$ (pixel)')
plt.ylabel('time (s)')
plt.legend()
plt.savefig('other-process-time.png')
plt.show()
