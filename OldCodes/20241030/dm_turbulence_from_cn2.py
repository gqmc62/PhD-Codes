# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 11:55:00 2024

@author: gqmc62
"""

import numpy as np
import scipy.interpolate

from matplotlib import pyplot as plt

from matplotlib.scale import FuncScale

# at the beginning cn2dh is the distribution of cn2dh

def r02cn2(r0,wvl=500e-9):# this r0 is turbulence strength in line of sight
    cn2dz = r0**(-5./3.)/(0.423*(2.*np.pi/wvl)**2)
    return cn2dz # this is cn2dz in line of sight

def cn22r0(cn2dz,wvl=500e-9):
    r0 = (0.423*(2.*np.pi/wvl)**2*cn2dz)**(-3./5.)
    return r0

def square(x):
    return x**2
def square_root(y):
    return y**0.5


# cn2dh = np.array([0.65,0.25,0.02,0.01,0.01,0.03,0.02,0.01],dtype=float)
# r0_original = 0.15
# r0_original_ref_wvl = 500e-9
# cn2_original = r02cn2(r0_original,r0_original_ref_wvl)


# h = np.array([0,1,2,4,8,12,16,20],dtype=float)
# dh = h[1:] - h[:-1]
# dh = np.append(dh,1)
# cn2 = np.copy(cn2dh)
# cn2 = cn2dh / dh


# new_h = np.linspace(h.min(),h.max(),int(h.max()) + 1)
# new_cn2 = np.interp(new_h,h,cn2)
# # new_cn2 /= new_cn2.sum()

# # plt.bar(new_h,new_cn2)
# # plt.plot(h,cn2,color='r',marker='o')
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.show()

# # double_check = np.zeros((h.shape[0]),dtype=float)
# # for i in np.arange(h.shape[0]):
# #     for j in np.arange(new_h.shape[0]):
# #         if new_h[j] 
# #             double_check[i] += new_cn2dh[j]

# original_dm_height = np.array([0,5,9,12,20],dtype=float)
# zenith_angle = 60

# dm_height = original_dm_height * np.cos(np.radians(zenith_angle))

# h_bin = np.digitize(new_h,dm_height,right=False) - 1

# dm_cn2dh = np.zeros((dm_height.shape[0]),dtype=float)

# for i in np.arange(h_bin.shape[0]):
#     dm_cn2dh[h_bin[i]] += new_cn2[i]*1

# dm_cn2dh[-1] += ((new_cn2*1).sum() - dm_cn2dh.sum())
# # dm_cn2dh /= dm_cn2dh.sum()

# dm_dh = dm_height[1:] - dm_height[:-1]
# dm_dh = np.append(dm_dh,h.max()-dm_height.max())

# # plt.plot(dm_height,dm_cn2dh,marker='o',color='b')
# # plt.plot(h,cn2dh,marker='o',color='r')
# # plt.show()


# dm_real_cn2dh = dm_cn2dh * cn2_original / np.cos(np.radians(zenith_angle))

# real_cn2dh = cn2dh*cn2_original

# plt.plot(h,real_cn2dh/dh,color='b',marker='o')
# plt.plot(dm_height,dm_real_cn2dh/dm_dh,color='r',marker='o')
# plt.xscale('log')
# plt.yscale('log')



cn2dh_distr = np.array([0.65,0.25,0.02,0.01,0.01,0.03,0.02,0.01],dtype=float)
r0_original = 0.15
r0_original_ref_wvl = 500e-9
sum_cn2dh_original = r02cn2(r0_original,r0_original_ref_wvl)
cn2dh = cn2dh_distr * sum_cn2dh_original

h = np.array([0,1,2,4,8,12,16,20],dtype=float)*1e3
dh = h[1:] - h[:-1]
dh = np.append(dh,dh[-1])

cn2 = cn2dh / dh

new_dh = 100

new_h = np.linspace(h.min(),h.max(),int(h.max()/new_dh) + 1)
new_cn2 = np.zeros_like(new_h)
new_cn2 = scipy.interpolate.pchip_interpolate(h,cn2,new_h)
# interpolater = scipy.interpolate.PchipInterpolator(h,cn2)
# new_cn2 = interpolater(new_h)
# new_cn2 /= new_cn2.sum()

x_min = 10**int(np.floor(np.log10(np.min(np.concatenate([new_cn2,cn2])))))
x_max = 10**int(np.ceil(np.log10(np.max(np.concatenate([new_cn2,cn2])))))

# plt.plot(new_cn2,new_h,color='b',marker='o')
# plt.plot(cn2,h,color='r',marker='o')
# plt.yscale('function',functions=(square_root,square))
# plt.xscale('log')
# plt.ylim(0,20e3)
# plt.grid()
# plt.ylabel('elevation (h) (m)')
# plt.xlabel('$C_n^2$ (m$^{-2/3}$)')
# plt.xlim(x_min,x_max)
# plt.show()

for zenith_angle in np.arange(8)*10:
    
    original_dm_height = np.array([0,5,9,12,20],dtype=float)*1e3
    # zenith_angle = 80
    
    dm_height = original_dm_height * np.cos(np.radians(zenith_angle))
    
    h_bin = np.digitize(new_h,dm_height,right=False) - 1
    
    dm_cn2dh = np.zeros((dm_height.shape[0]),dtype=float)
    
    for i in np.arange(h_bin.shape[0]):
        dm_cn2dh[h_bin[i]] += new_cn2[i]*new_dh
    # print(((new_cn2*new_dh).sum() - dm_cn2dh.sum())/(new_cn2*new_dh).sum())
    dm_cn2dh[-1] += ((new_cn2*new_dh).sum() - dm_cn2dh.sum())
    
    dm_cn2seczdh = dm_cn2dh / np.cos(np.radians(zenith_angle))
    
    # plt.plot(cn2dh,h/1000,color='b',marker='o',label='profile - zenith')
    # plt.plot(dm_cn2seczdh,dm_height/1000,color='r',marker='o',label='dm in los')
    # plt.yscale('function',functions=(square_root,square))
    # plt.xscale('log')
    # plt.ylim(0,20)
    # plt.xlim(1e-15,1e-12)
    # plt.legend()
    # plt.grid()
    # plt.ylabel('elevation (h) (km)')
    # plt.xlabel('$C_n^2dh$sec$\zeta$ (m$^{1/3}$)')
    # plt.show()
    
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    
    ax1.set_ylim(0,20)
    ax2.set_ylim(0,20*np.cos(np.radians(zenith_angle)))
    
    ax2_mark = np.array([0,2,4,8,16])
    new_tick_locations = ax2_mark**0.5
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(new_tick_locations)
    labels = []
    for i in np.arange(ax2_mark.shape[0]):
      labels.append('{:.0f}'.format(ax2_mark[i]))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('distance in los (z) (km)')
    
    x_min = 10**int(np.floor(np.log10(np.min(np.concatenate([cn2dh,dm_cn2seczdh])))))
    x_max = 10**int(np.ceil(np.log10(np.max(np.concatenate([cn2dh,dm_cn2seczdh])))))
    
    
    ax1.plot(cn2dh,h/1000,color='b',marker='o',label='profile - zenith')
    ax1.plot(dm_cn2seczdh,dm_height/1000,color='r',marker='o',label='dm in los')
    ax1.set_yscale('function',functions=(square_root,square))
    ax1.set_xscale('log')
    plt.xlim(x_min,x_max)
    ax1.legend()
    ax1.grid()
    # ax2.grid()
    ax1.set_ylabel('elevation (h) (km)')
    ax1.set_xlabel('$C_n^2dh$sec$\zeta$ (m$^{1/3}$)')
    plt.show()