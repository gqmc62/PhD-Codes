# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 14:30:16 2025

@author: gqmc62
"""


import os
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt

def calc_spot_size(wvl,r0,d):
    if d < r0:
        return wvl/d
    else:
        return wvl/r0
    
color_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan',
              'tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan',
              'tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

    
def get_rcond_adaptive_threshold_rank(A,plot=False,return_place=False,threshold_rank=0.5):
  _,s,_ = np.linalg.svd(A)
  s /= s.max()
  energy = s**2
  energy /= energy.sum()
  accumulated_energy = np.zeros_like(energy)
  threshold = threshold_rank/np.linalg.matrix_rank(A)

  for i in np.arange(energy.shape[0]):
    accumulated_energy[i] = energy[:i].sum()
  residual = 1 - accumulated_energy
  place = np.where(residual<=threshold)[0][0]
  
  if plot==True:
    # plt.plot(residual,label='residual')
    # plt.hlines(residual[place],0,s.shape[0])
    # plt.vlines(place,0,1)
    # plt.yscale('log')
    # plt.show()

    plt.plot(s)
    plt.hlines(s[place],0,s.shape[0])
    plt.vlines(place,0,1)
    plt.yscale('log')
    plt.show()
  rcond = s[place]
  if return_place == True:
    return rcond, place
  else:
    return rcond
#   return 0.05

def pinv_adaptive_threshold_rank(A,return_rcond=False):
    if return_rcond == False:
        rcond = get_rcond_adaptive_threshold_rank(A)
        pinvA = np.linalg.pinv(A,rcond=rcond)
        return pinvA
    else:
        rcond, place = get_rcond_adaptive_threshold_rank(A,return_place=True)
        pinvA = np.linalg.pinv(A,rcond=rcond)
        return pinvA, rcond, place

ARCSEC2RAD = (2*np.pi)/(3600*360)

HOMELIST = [r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\FinalSupreme\conf\IMCC_bad_IM_A',
            r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\FinalSupreme\conf\IMCC_bad_IM_B',
            r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\FinalSupreme\conf\IMCC_bad_IM_C']

# HOMELIST = [r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\IMCC_bad_IM_A',
#             r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\IMCC_bad_IM_B',
#             r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\IMCC_bad_IM_C']


wvl = 500e-9
r0 = 0.05
tel_diam = 0.4

###############################################################################

# science_fov = 8
# science_pxl = 160
# dsub = tel_diam/np.array([12,8,4])
# pxlPerSubap = np.array([20,20,42])
# fov = np.array([32,21,21])
# sim_size = 160
# TYPELIST = ['C','A','B']

###############################################################################

science_fov = 9
science_pxl = 180
dsub = tel_diam/np.array([10,8,6])

pxlPerSubap = np.array([18,18,22])
fov = np.array([21,17,17])
sim_size = 180

TYPELIST = ['A','B','C']

###############################################################################

science_diffraction = wvl/2/tel_diam/ARCSEC2RAD
science_scale = science_fov/science_pxl
science_seeing = wvl/r0/ARCSEC2RAD


spot_size = np.zeros(3)
for i in range(3):
    spot_size[i] = calc_spot_size(wvl,r0,dsub[i])*206265

diffraction_limited = wvl/2/dsub/ARCSEC2RAD
wfs_pixel_scale = fov/pxlPerSubap

spot_size_pxl = spot_size/wfs_pixel_scale
fov_spot_size = fov/spot_size

LABELLIST = []
for i in range(len(dsub)):
    LABELLIST.append('dsub:{:.0f}cm'.format(dsub[i]*100))

CASES = 19
TRIAL = 50

poke = 1e-9
slope0 = poke/(dsub)/ARCSEC2RAD

print('pxl is {:} of half diffraction limit'.format(wfs_pixel_scale/diffraction_limited))
print('fov can contain {:} FWHM of long exposure spot'.format(fov_spot_size))
print('wfs los will have {:} x more number of pixel of the sim screen'.format(0.4/dsub*pxlPerSubap/sim_size))
print('los pixel size {:}mm'.format(tel_diam/(tel_diam/dsub*pxlPerSubap)*1000))
print('turb pixel size {:}mm'.format(tel_diam/sim_size*1000))
print('spot fwhm size to pxl {:}'.format(spot_size/wfs_pixel_scale))
print('science fov contain {:} fwhm seeing'.format(science_fov/science_seeing))
print('science pxl is {:} diffraction limit'.format(science_scale/science_diffraction))
print('science los is {:} of sim size'.format(science_pxl/sim_size))

IM = []
perfect_IM = []

for i_home in range(len(HOMELIST)):
    os.chdir(HOMELIST[i_home])
    TYPE = TYPELIST[i_home]
    
    f = open('name_space.txt' ,'r')
    name_list = (np.asarray(f.readlines()))
    
    for i_name in range(len(name_list)):
        name = name_list[i_name]
    
        for trial in range(TRIAL):
            hdu = fits.open('IMbad_IM_' + TYPE + '-' + name.strip() + '{:d}'.format(trial) + '.fits')
            data = hdu[0].data*wfs_pixel_scale[i_home]/slope0[i_home]
            hdu.close()
            
            if (trial == 0) and (i_name == 0):
                IM.append(np.zeros((CASES,TRIAL,data.shape[0],data.shape[1]),dtype=float))
            
            IM[i_home][i_name,trial] = data
            
            if (trial == 0) and (i_name == 0):
                perfect_IM.append(np.zeros((CASES,data.shape[0],data.shape[1]),dtype=float))
                
            if trial == 0:
                hdu = fits.open('perfect_IMbad_IM_' + TYPE + '-' + name.strip() + '.fits')
                data = hdu[0].data*wfs_pixel_scale[i_home]/slope0[i_home]
                hdu.close()
                perfect_IM[i_home][i_name] = data
            
            
################################################################################

cases = 16

fig, axes = plt.subplots(1,3,figsize=(3*4,3),sharex=False,sharey=False)
for i in range(3):
    a = axes[i].imshow((perfect_IM[i][cases]))#,vmin=0,vmax=np.max(MAX))
    fig.colorbar(a,ax=axes[i],shrink=0.8)
    axes[i].set_title(LABELLIST[i])
plt.suptitle('perfect IM (1nm/dsub)')
plt.show()

fig, axes = plt.subplots(1,3,figsize=(3*4,3),sharex=False,sharey=False)
for i in range(3):
    a = axes[i].imshow((IM[i][cases].mean(0)))#,vmin=0,vmax=np.max(MAX))
    fig.colorbar(a,ax=axes[i],shrink=0.8)
    axes[i].set_title(LABELLIST[i])
plt.suptitle('IM mean case16 (1nm/dsub)')
plt.show()

fig, axes = plt.subplots(1,3,figsize=(3*4,3),sharex=False,sharey=False)
for i in range(3):
    a = axes[i].imshow((IM[i][cases].std(0)))#,vmin=0,vmax=np.max(MAX))
    fig.colorbar(a,ax=axes[i],shrink=0.8)
    axes[i].set_title(LABELLIST[i])
plt.suptitle('IM std case16 (1nm/dsub)')
plt.show()

fig, axes = plt.subplots(1,3,figsize=(3*4,3),sharex=False,sharey=False)
for i in range(3):
    a = axes[i].imshow((((IM[i][cases]-perfect_IM[i][cases])**2).mean(0)**0.5))#,vmin=0,vmax=np.max(MAX))
    fig.colorbar(a,ax=axes[i],shrink=0.8)
    axes[i].set_title(LABELLIST[i])
plt.suptitle('IM rmse case16 (1nm/dsub)')
plt.show()

for cases in range(19):
    for i in range(3):
        plt.plot((((IM[i][cases])**2).mean(0)**0.5),
                 marker='.',
                 color=color_list[i],alpha=0.2,ls='')
        plt.plot(0,0,marker='.',color=color_list[i],label='dsub:{:.2f}cm'.format(dsub[i]*100))
    plt.legend()
    plt.title('IM abs case16 (1nm/dsub)')
    plt.show()

for cases in range(19):
    for i in range(3):
        plt.plot((((IM[i][cases]-perfect_IM[i][cases])**2).mean(0)**0.5),
                 marker='.',
                 color=color_list[i],alpha=0.2,ls='')
        plt.plot(0,0,marker='.',color=color_list[i],label='dsub:{:.2f}cm'.format(dsub[i]*100))
    plt.legend()
    plt.title('IM rmse case16 (1nm/dsub)')
    plt.show()
    
for cases in range(19):
    for i in range(3):
        plt.plot((((IM[i][cases]-perfect_IM[i][cases])**2).mean(0)**0.5).mean(-1)/perfect_IM[i][cases].max(),marker='o',label='dsub:{:.2f}cm'.format(dsub[i]*100))
    # plt.ylim(0,0.5)
    plt.legend()
    plt.ylabel('RMSE (max slope non distort im)')
    plt.xlabel('actuators')
    plt.suptitle('IM rmse case : {:}'.format(cases))
    plt.show()


for i in range(3):
    plt.plot((((perfect_IM[i][cases])**2)**0.5).T[0]/perfect_IM[i][cases].max(),marker='o',label='dsub:{:.2f}cm'.format(dsub[i]*100))
# plt.ylim(0,0.5)
plt.legend()
plt.ylabel('abs (max slope non distort im)')
plt.xlabel('actuators')
plt.suptitle('IM case : {:}'.format(cases))
plt.show()

# for i in range(3):
#     plt.plot((((IM[i][cases]-perfect_IM[i][cases])**2).mean(0)**0.5).mean(-1)/perfect_IM[i][cases].max(),
#              marker='o',label='dsub:{:.2f}cm'.format(dsub[i]*100))
# plt.legend()
# plt.suptitle('IM rmse case16 (1nm/dsub)')
# plt.show()

for i in range(3):
    for cases in [1,12,16]:
        plt.plot((((IM[i][cases]-perfect_IM[i][cases])**2).mean(0)**0.5).mean(-1),marker='o')#,vmin=0,vmax=np.max(MAX))
    plt.suptitle('IM rmse case16 (1nm/dsub)')
    plt.show()

for i in range(3):
    plt.plot((((IM[i][cases]-perfect_IM[i][cases])**2).mean(0)**0.5).mean(-1)/perfect_IM[i][cases].max(),marker='o',label='dsub:{:.2f}cm'.format(dsub[i]*100))
plt.legend()
plt.suptitle('IM rmse case16 (1nm/dsub)')
plt.show()


###############################################################################

# for i in range(3):
#     plt.plot(np.max(perfect_IM[i],axis=(-1,-2)),label=LABELLIST[i])
# plt.legend()
# plt.ylabel('(1nm/dsub)')
# plt.xlabel('cases)')
# plt.title('non distort IM max value')
# plt.show()

# for i in range(3):
#     plt.plot(np.max(IM[i],axis=(-1,-2,-3)),label=LABELLIST[i])
# plt.legend()
# plt.ylabel('(1nm/dsub)')
# plt.xlabel('cases)')
# plt.title('distort IM max value')
# plt.show()

# # for i in range(3):
# #     plt.plot(np.max(IM[i],axis=(-2,-3)),color=color_list[i],alpha=0.05)
# #     plt.ylim(0,4)
# #     plt.ylabel('(1nm/dsub)')
# #     plt.xlabel('cases)')
# #     plt.title('distort IM max value\n' + LABELLIST[i])
# #     plt.show()


# for i in range(3):
#     plt.plot(np.max(IM[i].std(1),axis=(-1,-2)),label=LABELLIST[i])
# plt.legend()
# plt.xlabel('aberration cases')
# plt.ylabel('(1nm/dsub)')
# plt.title('max std of all IM element')
# plt.show()

###############################################################################


for i in range(3):
    plt.plot(np.max(IM[i].std(1),axis=(-1,-2))/np.max(IM[i],axis=(-1,-2,-3)),
             label=LABELLIST[i])
plt.legend()
plt.ylabel('wfs slope (arcsec)')
plt.xlabel('cases)')
plt.title('max std wrt non distort IM max value')
plt.show()


distortion_IM_level = np.zeros((3,CASES),dtype=float)
for i in range(3):
    distortion_IM_level[i] = np.max(np.std(IM[i],axis=(-3)),axis=(-1,-2)) / np.max(IM[i][0])
    
for i in range(3):
    plt.plot(distortion_IM_level[i]*100,label=LABELLIST[i])
plt.legend()
plt.xlabel('aberration_cases')
plt.ylabel('percentage')
plt.show()



trial = 1
threshold_rank = 0.05


for i_type in range(3):
    
    _,svd_a,_ = np.linalg.svd(IM[i_type][0][0])
    rcond_a, place_a = get_rcond_adaptive_threshold_rank(IM[i_type][0][0],return_place=True,threshold_rank=threshold_rank)
    
    for i_cases in range(8):
        
        cases = [0,1,7,9,12,14,16,17][i_cases]
        
        svd_b = np.zeros((IM[i_type].shape[0],svd_a.shape[0]),dtype=float)
        place = np.zeros(trial)
        rcond = np.zeros(trial)
        for i in range(trial):
            _,svd_b[i],_ = np.linalg.svd(IM[i_type][cases][i])
            rcond[i], place[i] = get_rcond_adaptive_threshold_rank(IM[i_type][cases][i],return_place=True,threshold_rank=threshold_rank)
        
        for i in range(trial):
            plt.plot(place[i],rcond[i],marker='o',color=color_list[i_cases])#,markeredgecolor='k')
            plt.plot(svd_b[i]/svd_b[i].max(),color=color_list[i_cases])
    
    plt.plot(svd_a/svd_a.max(),color='k')
    plt.yscale('log')
    plt.plot(place_a,rcond_a,marker='o',color='k')#,markeredgecolor='k')
plt.hlines([0.05,0.16],0,120,color='k')
plt.xlabel('mode')
plt.ylabel('normalized svd')
plt.show()
    



for i in range(3):
    plt.plot((np.nanmean((IM[i] - perfect_IM[i][0])**2,axis=(-1,-2,-3))**0.5)/IM[i].max()*100,
             label='dsub:{:.2f}cm'.format(dsub[i]*100))
plt.legend()
plt.title('IM distortion %')
plt.xlabel('Varying turbulence condition')
plt.ylabel('Percentage RMSE')
plt.show()


for i in range(3):
    j=16
    plt.imshow(np.nanmean(((IM[i][j] - perfect_IM[i][0])/perfect_IM[i][0].max())**2,axis=(0))**0.5*100)
    plt.title('percentage RMSE (max non-distorted IM slope)\ndsub:{:.2f}cm'.format(dsub[i]))
    plt.colorbar()
    plt.show()

    
        