# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:26:41 2024

@author: gqmc62
"""

import pickle
import numpy as np
from matplotlib import pyplot as plt

# with open('time_use_time_old.pkl','rb') as file:
#     data = pickle.load(file)
# N = 2**(np.arange(10) + 1)    
# t_FT_error = data['t_FT_error']
# t_FT_mean = data['t_FT_mean']
# t_IG_error = data['t_IG_error']
# t_IG_mean = data['t_IG_mean']
# t_IPS_error = data['t_IPS_error']
# t_IPS_mean = data['t_IPS_mean']
# t_PROP_error = data['t_PROP_error']
# t_PROP_mean = data['t_PROP_mean']

# with open('time_use_time.pkl','rb') as file:
#     data = pickle.load(file)
# N = data['N']
# t_FT_error = data['t_FT_error']
# t_FT_mean = data['t_FT_mean']
# t_IG_error = data['t_IG_error']
# t_IG_mean = data['t_IG_mean']
# t_IPS_error = data['t_IPS_error']
# t_IPS_mean = data['t_IPS_mean']
# t_PROP_error = data['t_PROP_error']
# t_PROP_mean = data['t_PROP_mean']

# data = np.loadtxt('percent_timeit_new_4.txt')
# N = data[0,:10]
# t_FT_error = data[4,:10]
# t_FT_mean = data[3,:10]
# t_IG_error = data[2,:10]
# t_IG_mean = data[1,:10]
# t_IPS_error = data[6,:10]
# t_IPS_mean = data[5,:10]
# t_PROP_error = data[8,:10]
# t_PROP_mean = data[7,:10]

# plt.figure(figsize=[7,5])
# plt.errorbar(N,t_FT_mean,yerr=t_FT_error,capsize=5,marker='o',label='FT')
# plt.errorbar(N,t_IG_mean,yerr=t_IG_error,capsize=5,marker='o',label='IG')
# plt.errorbar(N,t_IPS_mean,yerr=t_IPS_error,capsize=5,marker='o',label='IPS')
# plt.errorbar(N,t_PROP_mean,yerr=t_PROP_error,capsize=5,marker='o',label='PROP')
# plt.xscale('log')
# plt.yscale('log')
# # plt.plot(N,N**2/1e6,ls=':',color='k',label='$N^2$ Slope')
# # plt.plot(N,N**2/1e7,ls=':',color='k')
# # plt.ylim(t_IPS_mean.min()*0.5,t_IPS_mean.max()*2)
# plt.xlabel('Number of pixel across a simulation screen ($N$ [pixel])')
# plt.ylabel('Time (s)')
# plt.legend()
# plt.savefig('ch4-time.png',bbox_inches='tight')
# plt.show()

# plt.clf()
# plt.figure(figsize=[7,5])
# plt.errorbar(N,t_FT_mean/t_IG_mean,
#              yerr=((t_FT_mean/t_IG_mean)
#                    *np.sqrt((t_FT_error/t_FT_mean)**2
#                             + (t_IG_error/t_IG_mean)**2)),
#              capsize=5,marker='o')
# # plt.plot(N,np.log(N)/2)
# plt.xscale('log')
# # plt.yscale('log')
# plt.show()

# 0.04 + 0.004 + 0.00015*346626

data = np.loadtxt('cProfileIGTilt.txt')
N = data[0]
IG = data[1]
FT = data[2]
Prop = data[3]
IPS = data[4]

plt.figure(figsize=[7,5])
# plt.plot(N,FT,marker='o',label='FT')
# plt.plot(N,IG,marker='o',label='IG')
plt.plot(N,Prop,marker='o',label='ANG-PROP')
plt.plot(N,IPS,marker='o',label='IPS')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1,2000)
plt.ylim(1e-5,1)
# plt.xlabel('Number of pixel across a turbulence screen ($n_{in}$ [pixel])')
plt.xlabel('Number of pixel across a simulation screen ($N$ [pixel])')
plt.ylabel('Time (s)')
plt.legend()
plt.savefig('ch4-time-other.png',bbox_inches='tight')
plt.show()

plt.figure(figsize=[7,5])
plt.plot(N,FT,marker='o',label='FT')
plt.plot(N,IG,marker='o',label='IG')
# plt.plot(N,Prop,marker='o',label='ANG-PROP')
# plt.plot(N,IPS,marker='o',label='IPS')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1,2000)
plt.ylim(1e-5,1)
plt.xlabel('Number of pixel across a turbulence screen ($n_{in}$ [pixel])')
# plt.xlabel('Number of pixel across a simulation screen ($N$ [pixel])')
plt.ylabel('Time (s)')
plt.legend()
plt.savefig('ch4-time-wfs.png',bbox_inches='tight')
plt.show()

plt.clf()
plt.figure(figsize=[7,5])
plt.plot(N,FT/IG,marker='o')
# plt.plot(N,N/10)
# plt.plot(N,np.log(N)/2)
plt.xscale('log')
# plt.yscale('log')
# plt.xlim(1,600)
# plt.ylim(1,5)
plt.xlabel('Number of pixel across a turbulence screen ($n_{in}$ [pixel])')
plt.ylabel('Time gain (FT/IG)')
plt.savefig('ch4-time-gain.png',bbox_inches='tight')
plt.show()

