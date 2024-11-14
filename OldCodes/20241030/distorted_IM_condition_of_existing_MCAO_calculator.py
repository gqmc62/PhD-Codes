# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:06:15 2024

@author: gqmc62
"""


import numpy as np
from matplotlib import pyplot as plt



def r02cn2(r0,wvl=500e-9):# this r0 is turbulence strength in line of sight
    cn2dz = r0**(-5./3.)/(0.423*(2.*np.pi/wvl)**2)
    return cn2dz # this is cn2dz in line of sight

def cn22r0(cn2dz,wvl=500e-9):
    r0 = (0.423*(2.*np.pi/wvl)**2*cn2dz)**(-3./5.)
    return r0

def convert_r0_to_r0500nm(r0,wvl):
    cn2 = r02cn2(r0,wvl)
    r0550nm = cn22r0(cn2,500e-9)
    return r0550nm

def calculate_im_error_all_dm(name,wfs_wvl,dm_height,dm_act,dm_cn2,savefig=False):
    
    k = 2.0*np.pi/wfs_wvl
    
    N_DM = dm_height.shape[0]
    # dm_order = np.arange(N_DM) + 1
    AOA_constant = 0.182*0.423*(2*np.pi)**2
    
    
    rytov = np.zeros((N_DM), dtype=float)
    AS = np.zeros((N_DM), dtype=float)
    
    for n_dm in np.arange(N_DM)[ : : -1]:
        
        if (n_dm == N_DM - 1):
            AS[n_dm] = 0
            rytov[n_dm] = 0
            continue
        
        active_dm_height = dm_height[n_dm : ]
        # active_dm_height_difference = np.abs(active_dm_height[1 : ] - active_dm_height[ : -1])
        active_dm_height_difference = active_dm_height[1 : ] - active_dm_height[ : -1]
        active_dm_cn2 = dm_cn2[n_dm : ]
        active_dm_act = dm_act[n_dm : ]
        
        if (n_dm == N_DM - 1 - 1):
            AOA = (AOA_constant*active_dm_act[1:]**(-1./3.)*active_dm_cn2[1:])**0.5
            AS[n_dm] = np.abs(AOA*active_dm_height_difference[-1]/active_dm_act[0])
            rytov[n_dm] = 0.563 * k**(7./6.) * (active_dm_cn2 * active_dm_height**(5./6.)).sum()
            continue
        
        AOA = (AOA_constant*active_dm_act[1:]**(-1./3.)*active_dm_cn2[1:])**0.5
        shift_distance = 0.
        for j in np.arange(AOA.shape[0])[::-1]:
            shift_distance += AOA[j : ].sum()*np.abs(active_dm_height_difference[j])
        AS[n_dm] = shift_distance/active_dm_act[0]
        rytov[n_dm] = 0.563 * k**(7./6.) * (active_dm_cn2 * active_dm_height**(5./6.)).sum()
        
    # print('For MCAO with the following DM configuration')
    # print('DM order from telescope pupil to WFS from left to right')
    # print('ie. Tel --> DM --> DM --> WFS')
    # print('DM conjugation elevations are')
    # print('Tel: --> {:} --> WFS'.format(dm_height))
    # print('DM actuator spacings are')
    # print(dm_act)
    # print('DMs are correcting turbulence layer with strength in line-of-sight equals')
    # print(dm_cn2)
    # print('WFS are operating at wavelength = {:.0f}nm'.format(wfs_wvl/1e-9))
    # print('DM controls will experience rytov error of')
    # print(rytov)
    # print('DM controls will experience statistical actuator position shift of')
    # print(AS)
    
    DM_NAME = []
    for i in np.arange(N_DM):
        DM_NAME.append('DM{:.0f}'.format(dm_height[i]/1000))
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.suptitle(MCAO_NAME + ' : IM error parameters')
    fig.tight_layout()
    ax1.bar(DM_NAME,rytov)
    ax1.set_ylim(0,0.6)
    # ax1.set_ylabel('Rytov Parameter')
    ax1.set_title('Rytov Parameter')
    # ax1.set_ylabel('Rytov parameter ($\sigma^2_\chi$)')
    # ax1.set_xlabel('DM order from telescope to WFSs from left to right')
    # plt.show()
    
    
    
    ax2.bar(DM_NAME,AS)
    ax2.set_ylim(0,1)
    # ax2.set_ylabel('Actuator Shift (its spacing size)')
    ax2.set_title('Actuator Shift (its spacing size)')
    # ax2.set_ylabel('Actuator shift ($AS$) (of its actuator spacing size)')
    # ax2.set_xlabel('DM order from telescope to WFSs from left to right')
    plt.gcf().set_size_inches(6,6)
    if savefig == True:
        plt.savefig('IM_error' + '_' + MCAO_NAME + '.png')
    plt.show()
    
    return rytov, AS


# order by left to --> right # telescope pupil --> wfs

# MCAO_NAME = 'TEST'
# wfs_wvl = 500e-9
# dm_height = np.array([0, 1.6, 6.6, 10.6, 23], dtype=float)*1e3
# dm_act = np.array([0.12,0.43,1.36,1.06,1.37], dtype=float)
# dm_cn2 = np.array([5e-13, 6e-14, 8.94e-15, 1.35e-14, 8.8e-15], dtype=float)

# MCAO_NAME = 'TEST'
# wfs_wvl = 500e-9
# dm_height = np.array([0, 20], dtype=float)*1e3
# dm_act = np.array([0.05, 0.05], dtype=float)
# dm_cn2 = np.array([1e-11,1e-12], dtype=float)

MCAO_NAME = 'DKIST sample'
wfs_wvl = 500e-9
dm_height = np.array([11, 4, 0], dtype=float)*1e3
dm_act = np.array([18,12,9.3], dtype=float)*1e-2
dm_cn2 = np.array([1.35e-14,8.94e-15,4.49e-13], dtype=float)
dm_cn2 /= dm_cn2.sum()
dm_cn2 *= r02cn2(0.15)#/np.cos(np.radians(80))

# dm_height = dm_height[::-1]
# dm_act = dm_act[::-1]
# dm_cn2 = dm_cn2[::-1]/np.cos(np.radians(70))

rytov, AS = calculate_im_error_all_dm(MCAO_NAME,wfs_wvl,dm_height,dm_act,dm_cn2)

print(rytov)
print(AS)