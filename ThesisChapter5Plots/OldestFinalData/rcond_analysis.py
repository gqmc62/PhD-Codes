# # -*- coding: utf-8 -*-
# """
# Created on Thu Apr 24 10:21:29 2025

# @author: gqmc62
# """

# import os
# from astropy.io import fits
# import numpy as np

# home = r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter5Plots\FinalData\rcond005'

# os.chdir(home)


# TYPELIST = ['A','B','C']

# def my_unwrap(wrapped_phase, period=2*np.pi):
    
#     # numpy unwrap start unwrap at 0 coordinate
#     # but for circular aperture there is no corner!
#     # if we unwrap by each quardrant there will be a corner! noice.
#     # but we may have to stitch them back together nicely
#     # so things should be de-piston to the center
    
#     center = wrapped_phase.shape[-1]//2
#     unwrapped_phase = np.zeros_like(wrapped_phase)
    
#     # ++ quardrant
#     wrapped_quardrant = wrapped_phase[...,center:,center:]
#     unwrapped_quardrant = np.unwrap(np.unwrap(wrapped_quardrant,axis=0),axis=1)
      
#     unwrapped_quardrant -= unwrapped_quardrant[0,0]
#     unwrapped_phase[...,center:,center:] = unwrapped_quardrant
    
#     # +- quardrant
#     wrapped_quardrant = wrapped_phase[...,center:,:center][:,::-1]
#     unwrapped_quardrant = np.unwrap(np.unwrap(wrapped_quardrant,axis=0),axis=1)
      
#     unwrapped_quardrant -= unwrapped_quardrant[0,0]
#     unwrapped_phase[...,center:,:center] = unwrapped_quardrant[:,::-1]
    
#     # -+ quardrant
#     wrapped_quardrant = wrapped_phase[...,:center,center:][::-1,:]
#     unwrapped_quardrant = np.unwrap(np.unwrap(wrapped_quardrant,axis=0),axis=1)
      
#     unwrapped_quardrant -= unwrapped_quardrant[0,0]
#     unwrapped_phase[...,:center,center:] = unwrapped_quardrant[::-1,:]
      
#     # -- quardrant
#     wrapped_quardrant = wrapped_phase[...,:center,:center][::-1,::-1]
#     unwrapped_quardrant = np.unwrap(np.unwrap(wrapped_quardrant,axis=0),axis=1)
      
#     unwrapped_quardrant -= unwrapped_quardrant[0,0]
#     unwrapped_phase[...,:center,:center] = unwrapped_quardrant[::-1,::-1]
    
#     return unwrapped_phase

# conversion = 500/(2.0*np.pi)
# from aotools.functions.pupil import circle
# PUPIL = circle(160,160)

# for TYPE in TYPELIST:
#     os.chdir(home)
#     os.chdir(TYPE)
#     f = open('name_space.txt' ,'r')
#     name_list = (np.asarray(f.readlines()))
#     for name in name_list:
#         hdu = fits.open('perfect_RES_bad_IM_' + TYPE + '-' + name.strip() + '.fits')
#         data = hdu[0].data
#         perfect_RES = data[0] + 1j*data[1]
#         hdu.close()
#         hdu = fits.open('RES_bad_IM_' + TYPE + '-' + name.strip() + '.fits')
#         data = hdu[0].data
#         RES = data[0] + 1j*data[1]
#         hdu.close()
        
#         perfect_RES_phase = my_unwrap(np.angle(perfect_RES))*PUPIL*conversion
#         RES_phase = my_unwrap(np.angle(RES))*PUPIL*conversion
        
#         perfect_WFE = np.std(perfect_RES_phase,axis=(-1,-2))
#         WFE = np.std(RES_phase,axis=(-1,-2))
        
#         hdu = fits.PrimaryHDU(perfect_WFE)
#         hdul = fits.HDUList([hdu])
#         hdul.writeto('perfect_WFE_bad_IM' + TYPE + '-' + name.strip() + '.fits', overwrite=True)
#         hdul.close()
        
#         hdu = fits.PrimaryHDU(WFE)
#         hdul = fits.HDUList([hdu])
#         hdul.writeto('WFE_bad_IM' + TYPE + '-' + name.strip() + '.fits', overwrite=True)
#         hdul.close()
        
##############################################################################
        
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 10:21:29 2025

@author: gqmc62
"""

import os
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt


# HOMELIST = [r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\gain04\rcond016',
#             r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\gain04\rcond005',
#             r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\gain04\rcondAdaptive']
# rcond_list = [0.16,0.05,'adaptive']


# HOMELIST = [r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\gain08\rcond016',
#             r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\gain08\rcond005',
#             r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\gain08\rcondAdaptive']
# rcond_list = [0.16,0.05,'adaptive']

# HOMELIST = [r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\r008x10']
# HOMELIST = [r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\gain04\rcond005']
# HOMELIST = [r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\gain08\rcond005']

# HOMELIST = [r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\Final']

HOMELIST = [r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\FinalSupreme\conf']

rcond_list = [0.05]

TRIAL = 50
LOOP = 5
CASES = 19

r0 = 0.08


TYPELIST = ['A','B','C']
D = 0.4
dsub = D/np.array([10,8,6])


# TYPELIST = ['A2','B2','C2']
# D = np.array([0.4,0.3,0.2])
# dsub = D/np.array([8,6,4])



WFE = np.zeros((len(HOMELIST),len(TYPELIST),CASES,TRIAL,LOOP),dtype=float)
perfect_WFE = np.zeros((len(HOMELIST),len(TYPELIST),CASES,TRIAL,LOOP),dtype=float)
RUNAWAY_POS = np.zeros((len(HOMELIST),len(TYPELIST),CASES,TRIAL))


for i_home in range(len(HOMELIST)):
    home = HOMELIST[i_home]
    NAMELIST = []
    for i_type in range(len(TYPELIST)):
        TYPE = TYPELIST[i_type]
        os.chdir(home)
        os.chdir(TYPE)
        f = open('name_space.txt' ,'r')
        name_list = (np.asarray(f.readlines()))
        NAMELIST.append(name_list)
        # try:
        #     name = name_list[16]
        # except:
        #     name = name_list[0]
        # print(name)
        # i_name = 0
        for i_name in range(len(name_list)):
            name = name_list[i_name]
            hdu = fits.open('perfect_WFE_bad_IM_' + TYPE + '-' + name.strip() + '.fits')
            data = hdu[0].data
            perfect_WFE[i_home,i_type,i_name] = data
            hdu.close()
            hdu = fits.open('WFE_bad_IM_' + TYPE + '-' + name.strip() + '.fits')
            data = hdu[0].data
            WFE[i_home,i_type,i_name] = data
            hdu.close()
            hdu = fits.open('RUNAWAY_POS_bad_IM_' + TYPE + '-' + name.strip() + '.fits')
            data = hdu[0].data
            RUNAWAY_POS[i_home,i_type,i_name] = data[:,0]
            hdu.close()
        
color_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple',
              'tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

RUNAWAY_POS[RUNAWAY_POS==-1] = np.nan

for i_rcond in range(len(rcond_list)):
    for i_dsub in range(len(dsub)):
        for cases in range(CASES):
            for trial in range(TRIAL):
                for loop in range(LOOP):
                    if loop >= RUNAWAY_POS[i_rcond,i_dsub,cases,trial]:
                        WFE[i_rcond,i_dsub,cases,trial,loop] = np.nan

for i_rcond in range(len(rcond_list)):
    for i_dsub in range(len(dsub)):
        plt.plot(np.sum(~np.isnan(RUNAWAY_POS[i_rcond,i_dsub]),axis=-1),
                 color=color_list[i_dsub],
                 label='dsub={:.2f}cm'.format(dsub[i_dsub]*100))
    plt.legend()
    plt.title('rcond={:}'.format(rcond_list[i_rcond]))
    plt.show()

# for i_home in range(len(HOMELIST)):
#     for cases in range(CASES):
#         for i_type in range(len(TYPELIST)):
#             plt.errorbar(np.arange(5),
#                           np.mean(WFE[i_home,i_type,cases],axis=0),
#                           yerr=np.std(WFE[i_home,i_type,cases],axis=0),
#                           marker='o',capsize=5,
#                           color=color_list[i_type],
#                           label='rcond:{:.3f},dsub:{:.1f}cm'.format(rcond_list[i_home],dsub[i_type]*100))
#             plt.plot(np.arange(5),
#                          WFE[i_home,i_type,cases].T,
#                          marker='',
#                          color=color_list[i_type],alpha=0.2)
            
#         # plt.legend()
#         plt.xlabel('loop iteration')
#         plt.ylabel('WFE (nm)')
#         # plt.ylim(5,500)
#         plt.ylim(0,500)
#         plt.title(cases)
#         # plt.yscale('log')
#         # plt.xlim(0,4)
#         plt.show()
            

# # for i_home in range(len(HOMELIST)):
# #     for cases in range(CASES):
#         for i_type in range(len(TYPELIST)):
#             plt.errorbar(np.arange(5),
#                           np.mean(perfect_WFE[i_home,i_type,cases],axis=0),
#                           yerr=np.std(perfect_WFE[i_home,i_type,cases],axis=0),
#                           marker='s',capsize=5,
#                           color=color_list[i_type],ls=':',
#                           label='perfect_rcond:{:.3f},dsub:{:.1f}cm'.format(rcond_list[i_home],dsub[i_type]*100))
#             plt.plot(np.arange(5),
#                          perfect_WFE[i_home,i_type,cases].T,
#                          color=color_list[i_type],alpha=0.2)
#         # plt.legend()
#         plt.xlabel('loop iteration')
#         plt.ylabel('WFE (nm)')
#         # plt.ylim(5,500)
#         plt.ylim(0,500)
#         # plt.yscale('log')
#         # plt.xlim(0,4)
#         plt.title(cases)
#         plt.show()

atm_wfe = (0.134*(0.4/r0)**(5./3.))**0.5*500/2/3.14
ao_wfe = ((0.17+0.354)*((dsub)/r0)**(5./3.))**0.5/2/3.14*500
# ao_wfe = ((0.134)*((dsub)/0.05)**(5./3.))**0.5/2/3.14*500
    
for i_dsub in range(len(dsub)):
    # first compare undistorted of different rcond
    for i_rcond in range(len(rcond_list)):
        # if i_rcond != 1:
        #     continue
        if i_rcond != 0:
            continue
        i_name = 16
        plt.errorbar(np.arange(5),
                      np.nanmean(perfect_WFE[i_rcond,i_dsub,i_name],axis=0),
                      yerr=np.nanstd(perfect_WFE[i_rcond,i_dsub,i_name],axis=0),#/np.sqrt(perfect_WFE.shape[3]),
                  label='no distortion',
                  ls='-',color=color_list[0],
                  capsize=5,marker='o')
        plt.errorbar(np.arange(5),
                      np.nanmean(WFE[i_rcond,i_dsub,i_name],axis=0),
                      yerr=np.nanstd(WFE[i_rcond,i_dsub,i_name],axis=0),#/np.sqrt(WFE.shape[3]),
                      marker='s',capsize=5,ls='-',color=color_list[1],
                  label='with distortion')
        plt.plot(np.arange(5),
                     perfect_WFE[i_rcond,i_dsub,i_name].T,
                 ls='-',color=color_list[0],alpha=0.3)
        plt.plot(np.arange(5),
                     WFE[i_rcond,i_dsub,i_name].T,
                     ls='-',color=color_list[1],alpha=0.3)
        plt.hlines([atm_wfe],0,4,label='turbulence',color=color_list[2])
        plt.hlines([ao_wfe[i_dsub]],0,4,label='ao corrected',color=color_list[3])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        # plt.legend()
        plt.ylim(0,500)
        plt.title('dsub:{:.2f}cm , tub:{} , rcond:{}'.format(dsub[i_dsub]*100,NAMELIST[i_dsub][i_name],rcond_list[i_rcond]))
        plt.show()

###############################################################################
    

for i_name in range(len(name_list)):

    for i_rcond in range(len(rcond_list)):
        # first compare undistorted of different rcond
        for i_dsub in range(len(dsub)):
            # for i_name in range(len(NAMELIST[i_dsub])):
            # i_name = 16
            # for i_rcond in range(len(rcond_list)):
            
            plt.errorbar(np.arange(5),
                          np.nanmean(perfect_WFE[i_rcond,i_dsub,i_name],axis=0),
                          yerr=np.nanstd(perfect_WFE[i_rcond,i_dsub,i_name],axis=0),#/np.sqrt(perfect_WFE.shape[3]),
                      label='no distortion, dsub:{:.2f}m'.format(dsub[i_dsub]),
                      ls=':',color=color_list[i_dsub],
                      capsize=5,marker='o')
            plt.errorbar(np.arange(5),
                          np.nanmean(WFE[i_rcond,i_dsub,i_name],axis=0),
                          yerr=np.nanstd(WFE[i_rcond,i_dsub,i_name],axis=0),#/np.sqrt(WFE.shape[3]),
                          marker='s',capsize=5,ls='-',color=color_list[i_dsub],
                      label='with distortion, dsub:{:.2f}m'.format(dsub[i_dsub]))
            plt.hlines([ao_wfe[i_dsub]],0,4,label='ao corrected, dsub:{:.2f}m'.format(dsub[i_dsub]),color=color_list[i_dsub])
        plt.hlines([atm_wfe],0,4,label='turbulence',color=color_list[i_dsub + 1])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        # plt.legend()
        plt.ylim(0,500)
        plt.title('tub:{},rcond={}'.format(NAMELIST[i_dsub][i_name],rcond_list[i_rcond]))
        plt.show()


# dif = np.zeros_like(WFE)

# for i in range(len(dsub)):

#     dif[:,i] = np.abs(WFE[:,i]**2 - ao_wfe[i]**2)

dif = WFE**2 - perfect_WFE**2


# dif[dif<0] = np.nan
control_error = (np.nanmean(dif,axis=-2))**0.5
control_error_error = (np.nanstd(dif,axis=-2))**0.5
count = np.sum(~np.isnan((dif)**0.5),axis=-2)
control_error_error /= count**0.5



for i in range(3):
    # i_rcond=1
    i_rcond=0
    plt.errorbar(np.arange(CASES),
                 control_error[i_rcond,i,:,-1],
                 control_error_error[i_rcond,i,:,-1],
                 marker='o',ls='-',capsize=5,
                 label='dsub{:.2f}cm'.format(dsub[i]*100))
plt.legend()
# plt.yscale('log')
# plt.ylim(0,500)
plt.show()


# for cases in range(CASES):
#     for i in range(3):
#         plt.errorbar(np.arange(5),
#                      control_error[1,i,cases],
#                      control_error_error[1,i,cases],
#                      marker='o',ls='-',capsize=5,
#                       label='dsub{:.2f}cm'.format(dsub[i]*100))
#                      # label='rcond{:.2f}cm'.format(rcond_list[i]))
#     plt.legend()
#     # plt.yscale('log')
#     plt.ylim(0,250)
#     plt.title('rcond=0.05,gain=0.4,dsub={:.2f}cm\n'.format(dsub[i]*100) + NAMELIST[i][cases])
#     plt.show()


for i_name in range(CASES):
    # i_name = 16
    # i_rcond = 1
    i_rcond = 0
    for i_dsub in range(3):
        plt.plot(np.arange(5),
             dif[i_rcond,i_dsub,i_name].T**0.5,
             ls='-',color=color_list[i_dsub],alpha=0.2)
        plt.errorbar(np.arange(5),
                     np.nanmean(dif[i_rcond,i_dsub,i_name].T**0.5,axis=1),
                     yerr=np.nanstd(dif[i_rcond,i_dsub,i_name].T**0.5,axis=1),#/np.sqrt((~np.isnan(dif[i_rcond,i_dsub,i_name]**0.5)).sum()),
                     marker='o',capsize=5,color=color_list[i_dsub],
                     label='dsub{:.2f}cm'.format(dsub[i_dsub]*100))
    # plt.ylim(0,250)
    plt.legend()
    plt.title(i_name)
    plt.show()        
    
for i_name in range(CASES):
    # i_name = 16
    # i_rcond = 1
    i_rcond = 0
    for i_dsub in range(3):
        # plt.plot(np.arange(5),
        #      dif[i_rcond,i_dsub,i_name].T**0.5,
        #      ls='-',color=color_list[i_dsub],alpha=0.2)
        plt.errorbar(np.arange(5),
                     np.nanmean(dif[i_rcond,i_dsub,i_name].T**0.5,axis=1),
                     yerr=np.nanstd(dif[i_rcond,i_dsub,i_name].T**0.5,axis=1),#/np.sqrt((~np.isnan(dif[i_rcond,i_dsub,i_name]**0.5)).sum()),
                     marker='o',capsize=5,color=color_list[i_dsub],
                     label='dsub{:.2f}cm'.format(dsub[i_dsub]*100))
    # plt.ylim(0,250)
    plt.legend()
    plt.title(i_name)
    plt.show()        
    
for i in range(len(rcond_list)):
    for j in range(len(dsub)):
        if j == 0:
            continue
        plt.errorbar(np.arange(CASES),
                     control_error[i,j,:,-1],
                     yerr=control_error_error[i,j,:,-1],
                     capsize=5,marker='o',ls='',label='dsub:{:.2f}cm'.format(dsub[j]*100))
plt.yscale('log')
plt.legend()
plt.show()
        
        
        
        