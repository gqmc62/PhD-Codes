# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 13:07:29 2025

@author: gqmc62
"""

from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import my_phasescreen as mps
from aotools.opticalpropagation import angularSpectrum
from soapy.soapy import interp


def find_centre(corr,nx_subap_interp):
    # M. G. Löfdahl 2010
    try:
        x = np.arange(corr.shape[0],dtype=float)
        x -= x.max()/2.
        x -= 0.5
        x /= nx_subap_interp
        yy,xx = np.meshgrid(x,x)
        
        maxindex = np.array(np.where(corr==np.nanmax(corr)))[:,0]
        

        
        cropped_corr = corr[maxindex[0] - 1 
                            : maxindex[0] + 2,
                            maxindex[1] - 1 
                            : maxindex[1] + 2]
        
        locx = xx[maxindex[0],maxindex[1]]
        locy = yy[maxindex[0],maxindex[1]]
        
        a2 = (cropped_corr[1 + 1,:].mean() - cropped_corr[-1 + 1,:].mean())/2.
        a3 = (cropped_corr[1 + 1,:].mean() - 2.*cropped_corr[0 + 1,:].mean() + cropped_corr[-1 + 1,:].mean())/2.
        a4 = (cropped_corr[:,1 + 1].mean() - cropped_corr[:,-1 + 1].mean())/2.
        a5 = (cropped_corr[:,1 + 1].mean() - 2.*cropped_corr[:,0 + 1].mean() + cropped_corr[:,-1 + 1].mean())/2.
        a6 = (cropped_corr[1 + 1,1 + 1] - cropped_corr[-1 + 1,1 + 1] - cropped_corr[1 + 1,-1 + 1] + cropped_corr[-1 + 1,-1 + 1])/4.
        
        true_locx = ((2.*a2*a5 - a4*a6)/(a6**2 - 4.*a3*a5))
        true_locy = ((2.*a3*a4 - a2*a6)/(a6**2 - 4.*a3*a5))
        
        # plt.pcolor([-1,0,1],[-1,0,1],cropped_corr.T)
        # plt.plot(true_locx,true_locy,marker='o',color='r')
        # plt.axis('square')
        # plt.show()
        
        true_locx /= nx_subap_interp
        true_locy /= nx_subap_interp
        
        true_locx += locx
        true_locy += locy
        
        # plt.pcolor(x,x,corr)
        # plt.plot(true_locx,true_locy,marker='o',color='r')
        # plt.plot((maxindex[0]-64)/16,(maxindex[1]-64)/16,marker='x',color='k')
        # plt.axis('square')
        # plt.show()
        
    
    except:
        return np.nan, np.nan
    
    return true_locx, true_locy



###############################################################################
# check tip/tilt variances in several cases
# without propagation
###############################################################################


dsub = 0.4/4 #0.27
dsub_pxl = 16
nxsub = 4#15
wvl = 500e-9
D = dsub*nxsub
pixel_scale = dsub/dsub_pxl
D_pxl = dsub_pxl*nxsub


r0 = 0.05#0.1
L0 = 20
TRIAL = 100
z = 10000#11200

pad = int(np.ceil(wvl*z/2/pixel_scale**2/2))
needN = int(np.ceil(wvl*z/pixel_scale**2))

saintjaques = 0.162
    

vk_ratio = mps.tiptilt_variance_adjust_to_vK(D,L0)

global_tt = np.zeros((TRIAL,2),dtype=float)
local_tt = np.zeros((TRIAL,2,nxsub**2),dtype=float)
local_tt_no_global = np.zeros((TRIAL,2,nxsub**2),dtype=float)
filter_local_tt_no_global = np.zeros((TRIAL,2,nxsub**2),dtype=float)
ground_subap_tt = np.zeros((TRIAL,2,nxsub**2),dtype=float)

tt_variable = wvl**2*(dsub)**(-1/3)*r0**(-5/3)
global_tt_variable = wvl**2*(D)**(-1/3)*r0**(-5/3)

# tims = 0.349294*(1/3600/360*2*np.pi)**2/tt_variable # 8 nxsub
# tims = 0.48156968391859395*(1/3600/360*2*np.pi)**2/tt_variable
# tims = 0.40838844193319035*(1/3600/360*2*np.pi)**2/tt_variable # 12 nxsub 0.408 
tims = 0.26953547621243934*(1/3600/360*2*np.pi)**2/tt_variable # 4 nxsub


for trial in tqdm(range(TRIAL)):
    
    IPS = PhaseScreenKolmogorov(D_pxl, pixel_scale, r0, L0)
    for row in range(D_pxl):
        IPS.add_row()
    
    global_scrn = IPS.scrn
    
    local_scrns = np.asarray(np.split(np.asarray(np.split(global_scrn,nxsub,axis=0)),nxsub,axis=-1)).reshape(nxsub**2,dsub_pxl,dsub_pxl)
    
    global_tip = (global_scrn[1:] - global_scrn[:-1]).mean()
    global_tilt = (global_scrn[:,1:] - global_scrn[:,:-1]).mean()
    
    global_tt[trial,0] = global_tip/pixel_scale*wvl/2/np.pi
    global_tt[trial,1] = global_tilt/pixel_scale*wvl/2/np.pi
    
    global_x = np.arange(D_pxl)
    global_yy, global_xx = np.meshgrid(global_x,global_x)
    
    local_x = np.arange(dsub_pxl)
    
    global_scrn_no_tt = global_scrn - global_xx*global_tip - global_yy*global_tilt
    global_scrn_no_tt -= global_scrn_no_tt.mean()
    
    
    
    kspace = np.fft.fftshift(np.fft.fft2(np.pad(global_scrn_no_tt,((D_pxl//2,D_pxl//2),(D_pxl//2,D_pxl//2)),mode='symmetric')))
    filter_kspace = np.zeros_like(kspace)
    
    k_scale = 1/D/2 # the correct
    # k_scale = 1/D/2/2 # the one i used in soapy
    k_filter = 1/2/dsub
    n_filter = int(k_filter//k_scale)
    
    filter_kspace[D_pxl - n_filter
                  : D_pxl + n_filter,
                  D_pxl - n_filter
                  : D_pxl + n_filter] = kspace[D_pxl - n_filter
                                                  : D_pxl + n_filter,
                                                  D_pxl - n_filter
                                                  : D_pxl + n_filter]
    filter_global_scrn_no_tt = (np.fft.ifft2(np.fft.ifftshift(
        filter_kspace)).real)[D_pxl//2:3*D_pxl//2,D_pxl//2:3*D_pxl//2]
    
    

    
    local_scrns_no_tt = np.asarray(np.split(np.asarray(np.split(global_scrn_no_tt,nxsub,axis=0)),nxsub,axis=-1)).reshape(nxsub**2,dsub_pxl,dsub_pxl)
    filter_local_scrns_no_tt = np.asarray(np.split(np.asarray(np.split(filter_global_scrn_no_tt,nxsub,axis=0)),nxsub,axis=-1)).reshape(nxsub**2,dsub_pxl,dsub_pxl)
    
    
    
    local_tt[trial,0] = np.mean((local_scrns[:,1:] - local_scrns[:,:-1]),axis=(1,2))/pixel_scale*wvl/2/np.pi # tip angle in rad
    local_tt[trial,1] = np.mean((local_scrns[:,:,1:] - local_scrns[:,:,:-1]),axis=(1,2))/pixel_scale*wvl/2/np.pi
    
    local_tt_no_global[trial,0] = np.mean((local_scrns_no_tt[:,1:] - local_scrns_no_tt[:,:-1]),axis=(1,2))/pixel_scale*wvl/2/np.pi
    local_tt_no_global[trial,1] = np.mean((local_scrns_no_tt[:,:,1:] - local_scrns_no_tt[:,:,:-1]),axis=(1,2))/pixel_scale*wvl/2/np.pi
    
    filter_local_tt_no_global[trial,0] = np.mean((filter_local_scrns_no_tt[:,1:] - filter_local_scrns_no_tt[:,:-1]),axis=(1,2))/pixel_scale*wvl/2/np.pi
    filter_local_tt_no_global[trial,1] = np.mean((filter_local_scrns_no_tt[:,:,1:] - filter_local_scrns_no_tt[:,:,:-1]),axis=(1,2))/pixel_scale*wvl/2/np.pi
    
    
    
    
    Uin = np.pad(np.exp(1j*filter_global_scrn_no_tt),((pad,pad),(pad,pad)))
    
    
    Uout = angularSpectrum(Uin, wvl, pixel_scale, pixel_scale, z)[pad:-pad,pad:-pad]
    Uout /= Uout.mean()
    
    phase_ground = np.angle(Uout)
    
    (phase_ground[1:] - phase_ground[:-1]).mean()
    
    subap_ground = np.asarray(np.split(np.asarray(np.split(phase_ground,nxsub,axis=0)),nxsub,axis=-1)).reshape(nxsub**2,dsub_pxl,dsub_pxl)
    
    raw = subap_ground[:,1:] - subap_ground[:,:-1]
    
    new = np.zeros_like(raw)
    new[np.abs(raw)<=np.pi] = raw[np.abs(raw)<=np.pi]
    new[raw > np.pi] = raw[raw > np.pi] - 2*np.pi
    new[raw < - np.pi] = raw[raw < - np.pi] + 2*np.pi
    
    ground_subap_tt[trial,0] = np.mean(new,axis=(1,2))/pixel_scale*wvl/2/np.pi
    
    raw = subap_ground[:,:,1:] - subap_ground[:,:,:-1]
    
    new = np.zeros_like(raw)
    new[np.abs(raw)<=np.pi] = raw[np.abs(raw)<=np.pi]
    new[raw > np.pi] = raw[raw > np.pi] - 2*np.pi
    new[raw < - np.pi] = raw[raw < - np.pi] + 2*np.pi
    
    ground_subap_tt[trial,1] = np.mean(new,axis=(1,2))/pixel_scale*wvl/2/np.pi
    


plt.hist(global_tt.flatten(),label='ap tt all tt',density=True,alpha=0.5)
plt.hist(local_tt.flatten(),label='subap tt all tt',density=True,alpha=0.5)
plt.hist(local_tt_no_global.flatten(),label='subap tt no global tt',density=True,alpha=0.5)
plt.hist(filter_local_tt_no_global.flatten(),label='filter subap tt no global tt',density=True,alpha=0.5)
plt.hist(ground_subap_tt.flatten(),label='dm after prop - subap',density=True,alpha=0.5)
plt.legend()
# plt.title('coeff angle-of-arrival (rad2), nxsub : {:},\n'.format(nxsub)
#           + 'global={:.2e}, subap all = {:.2e},\n'.format(global_tt.var()/global_tt_variable,local_tt.var()/tt_variable)
#           + 'subap no global = {:.2e}, subap filter no global = {:.2e},\ntim={:.2e}'.format(
#     local_tt_no_global.var()/tt_variable,filter_local_tt_no_global.var()/tt_variable,tims))
plt.show()




###############################################################################
# check if dm actually have the stats that we want
# so it turns out that the dm will have 'edge' of 1 pitch
# the edge is there in the 'inactive' region out of the pupil
###############################################################################


if nxsub == 8:
    i=40
if nxsub == 4:
    i=12
if nxsub == 16:
    i=144
if nxsub == 12:
    i=84
    
if nxsub == 15:
    i=135

actGrid = np.zeros((nxsub + 1, nxsub + 1),dtype=float)
# actGrid[nxsub//2,nxsub//2] = 2*np.pi/100
actGrid[i//(nxsub + 1),i%(nxsub + 1)] = 2*np.pi/100
actGrid = np.pad(actGrid,((1,1),(1,1)))

N = D_pxl + 2*dsub_pxl

dmShape_A = interp.zoom_rbs(
        actGrid, (N, N))


plt.imshow(dmShape_A)
plt.hlines([dsub_pxl, N - dsub_pxl], dsub_pxl, N - dsub_pxl)
plt.vlines([dsub_pxl, N - dsub_pxl], dsub_pxl, N - dsub_pxl)
plt.title('{:}'.format(i))
plt.show()

# nxsub = 16
# D_pxl = 128
# dsub_pxl = D_pxl/nxsub
act_centre = np.zeros(((nxsub + 1)**2,2))


for i in range((nxsub + 1)**2):

    actGrid = np.zeros((nxsub + 1, nxsub + 1),dtype=float)
    # actGrid[nxsub//2,nxsub//2] = 2*np.pi/100
    actGrid[i//(nxsub + 1),i%(nxsub + 1)] = 2*np.pi/100
    actGrid = np.pad(actGrid,((1,1),(1,1)))
    
    N = D_pxl + 2*dsub_pxl
    
    dmShape_B = interp.zoom_rbs(
            actGrid, (N, N))
    
    
    
    # plt.imshow(dmShape_B)
    # plt.hlines(np.arange(dsub_pxl,N,dsub_pxl), dsub_pxl, N - dsub_pxl,color='r')
    # plt.vlines(np.arange(dsub_pxl,N,dsub_pxl), dsub_pxl, N - dsub_pxl,color='r')
    # plt.title('{:}'.format(i))
    # plt.show()
    
    
    
    size_A = dmShape_A.shape[0]
    
    # schmidt 2010 pg 45
    
    temp_corr = np.fft.fftshift(np.fft.ifft2(
        np.fft.fft2(np.pad(
            np.nan_to_num(dmShape_A),
            ((size_A//2,size_A//2),(size_A//2,size_A//2)),mode='constant'))
        * np.conjugate(np.fft.fft2(np.pad(
            np.nan_to_num(dmShape_B),
            ((size_A//2,size_A//2),(size_A//2,size_A//2)),mode='constant')))
        ))#.real[size_A//2:size_A*3//2,size_A//2:size_A*3//2]
    
    mask = np.pad(np.ones((size_A,size_A),dtype=float),
                      ((size_A//2,size_A//2),(size_A//2,size_A//2)),
                      mode='constant')
    
    mask_corr = np.fft.fftshift(np.fft.ifft2(np.abs(
        np.fft.fft2(mask))**2))#.real[size_A//2:size_A*3//2,size_A//2:size_A*3//2]
    
    idx = (mask_corr != 0)
    c = np.zeros_like(temp_corr)
    c[idx] = temp_corr[idx] / mask_corr[idx] * mask[idx]
    corr = c.real[size_A//2:size_A*3//2,size_A//2:size_A*3//2]
    
    corr = temp_corr.real[size_A//2:size_A*3//2,size_A//2:size_A*3//2]
    
    # plt.imshow(corr);plt.colorbar();plt.show()
    
    act_centre[i] = (np.asarray(find_centre(corr, dsub_pxl))*dsub_pxl)

plt.plot(act_centre[:,0]/dsub_pxl,
         act_centre[:,1]/dsub_pxl,
         ls='',marker='o',color='r')
plt.hlines(np.arange(-nxsub/2,nxsub/2+1),-nxsub/2,nxsub/2)
plt.vlines(np.arange(-nxsub/2,nxsub/2+1),-nxsub/2,nxsub/2)
plt.axis('square')
plt.show()


###############################################################################
# test distort dm influence function
###############################################################################
    
actGrid = np.zeros((nxsub + 1, nxsub + 1),dtype=float)
actGrid[nxsub//2,nxsub//2] = 2*np.pi/100
# actGrid[i//(nxsub + 1),i%(nxsub + 1)] = 2*np.pi/100
actGrid = np.pad(actGrid,((1,1),(1,1)))

# actGrid before pad + pad with 1 subap each side
N = D_pxl + 2*dsub_pxl

dmShape = interp.zoom_rbs(
        actGrid, (N, N))

# plt.imshow(dmShape)
# plt.hlines(np.arange(dsub_pxl,N,dsub_pxl), dsub_pxl, N - dsub_pxl,color='r')
# plt.vlines(np.arange(dsub_pxl,N,dsub_pxl), dsub_pxl, N - dsub_pxl,color='r')
# plt.title('{:}'.format(i))
# plt.show()
    



real_influ = dmShape

influ_tt = np.zeros((TRIAL,2),dtype=float)

x = np.arange(-D_pxl/2,D_pxl/2) + 0.5
yy,xx = np.meshgrid(x,x)

mask = ((xx**2+yy**2)<=D_pxl**2/4)

# real_influ[~mask] = np.nan

masked_real_influ = np.copy(real_influ[dsub_pxl:-dsub_pxl,dsub_pxl:-dsub_pxl])
masked_real_influ[~mask] = np.nan

pad = np.max([ int(np.ceil(wvl*z/2/pixel_scale**2/2)), int(np.ceil((wvl*z/pixel_scale**2 - D_pxl)/2))])

temp = (xx**2+yy**2)<=D_pxl**2/4

PUPIL = np.pad(temp,((pad,pad),(pad,pad)))

for trial in tqdm(range(TRIAL)):
    
    # GLDM = np.zeros((D_pxl,D_pxl),dtype=float)
    # GLDM[D_pxl//2-dsub_pxl//2
    #      :D_pxl//2+dsub_pxl//2,
    #      D_pxl//2-dsub_pxl//2
    #      :D_pxl//2+dsub_pxl//2] = 2*np.pi/100
    
    GLDM = np.copy(real_influ)
    
    # real_influ = np.copy(GLDM)
    
    # plt.imshow(GLDM);plt.colorbar();plt.title('GLDM');plt.show()
    
    # z = 5000
    

    if pad > dsub_pxl :
        Uin = np.pad(np.exp(1j*GLDM),((pad-dsub_pxl,pad-dsub_pxl),(pad-dsub_pxl,pad-dsub_pxl))) * PUPIL
    else:
        target = D_pxl + 2*pad
        current = GLDM.shape[0]
        crop = int((current - target)/2)
        Uin = np.exp(1j*GLDM[crop:-crop,crop:-crop]) * PUPIL
    # because there is extra pitch space buffer on each side
    # plt.imshow(np.abs(Uin)**2);plt.show()
    # plt.imshow(np.angle(Uin));plt.show()
    Uout = angularSpectrum(Uin, wvl, pixel_scale, pixel_scale, z)
    
    IPS = PhaseScreenKolmogorov(D_pxl, pixel_scale, r0, L0)
    for row in range(D_pxl):
        IPS.add_row()
    
    global_scrn = IPS.scrn
    
    local_scrns = np.asarray(np.split(np.asarray(np.split(global_scrn,nxsub,axis=0)),nxsub,axis=-1)).reshape(nxsub**2,dsub_pxl,dsub_pxl)
    
    global_tip = (global_scrn[1:] - global_scrn[:-1]).mean()
    global_tilt = (global_scrn[:,1:] - global_scrn[:,:-1]).mean()
    
    global_tt[trial,0] = global_tip/pixel_scale*wvl/2/np.pi
    global_tt[trial,1] = global_tilt/pixel_scale*wvl/2/np.pi
    
    global_x = np.arange(D_pxl)
    global_yy, global_xx = np.meshgrid(global_x,global_x)
    
    local_x = np.arange(dsub_pxl)
    
    global_scrn_no_tt = global_scrn - global_xx*global_tip - global_yy*global_tilt
    global_scrn_no_tt -= global_scrn_no_tt.mean()
    
    kspace = np.fft.fftshift(np.fft.fft2(np.pad(global_scrn_no_tt,((D_pxl//2,D_pxl//2),(D_pxl//2,D_pxl//2)),mode='symmetric')))
    filter_kspace = np.zeros_like(kspace)
    
    k_scale = 1/D/2 # the correct
    # k_scale = 1/D/2/2 # the one i used in soapy
    k_filter = 1/2/dsub
    n_filter = int(k_filter//k_scale)
    
    filter_kspace[D_pxl - n_filter
                  : D_pxl + n_filter,
                  D_pxl - n_filter
                  : D_pxl + n_filter] = kspace[D_pxl - n_filter
                                                  : D_pxl + n_filter,
                                                  D_pxl - n_filter
                                                  : D_pxl + n_filter]
    filter_global_scrn_no_tt = (np.fft.ifft2(np.fft.ifftshift(
        filter_kspace)).real)[D_pxl//2:3*D_pxl//2,D_pxl//2:3*D_pxl//2]
    
    Uin = Uout*np.exp(1j*np.pad(filter_global_scrn_no_tt,((pad,pad),(pad,pad))))
    # plt.imshow(np.angle(Uin));plt.show()
    # plt.imshow(np.abs(Uin)**2);plt.show()
    
    Uout = angularSpectrum(Uin, wvl, pixel_scale, pixel_scale, -z)
    
    # plt.imshow(np.abs(Uout)**2);plt.show()
    # plt.imshow(np.angle(Uout));plt.show()
    
    with_poke = Uout
    
    
    
    GLDM = np.zeros((D_pxl,D_pxl),dtype=float)
    # GLDM[D_pxl//2-dsub_pxl//2
    #      :D_pxl//2+dsub_pxl//2,
    #      D_pxl//2-dsub_pxl//2
    #      :D_pxl//2+dsub_pxl//2] = 2*np.pi/100
    # plt.imshow(GLDM);plt.colorbar();plt.title('GLDM');plt.show()
    
    Uin = np.pad(np.exp(1j*GLDM),((pad,pad),(pad,pad))) * PUPIL
    # plt.imshow(np.abs(Uin)**2);plt.show()
    # plt.imshow(np.angle(Uin));plt.show()
    Uout = angularSpectrum(Uin, wvl, pixel_scale, pixel_scale, z)
    
    
    
    Uin = Uout*np.exp(1j*np.pad(filter_global_scrn_no_tt,((pad,pad),(pad,pad))))
    # plt.imshow(np.angle(Uin));plt.show()
    # plt.imshow(np.abs(Uin)**2);plt.show()
    
    Uout = angularSpectrum(Uin, wvl, pixel_scale, pixel_scale, -z)
    
    # plt.imshow(np.abs(Uout)**2);plt.show()
    # plt.imshow(np.angle(Uout));plt.show()
    
    no_poke = Uout
    
    
    
    influ = np.angle(with_poke/no_poke)[pad:-pad,pad:-pad]
    influ[~mask] = np.nan
    
    # plt.imshow(influ);plt.colorbar();plt.show()
    # plt.imshow(real_influ);plt.colorbar();plt.show()
    
    
    # from scipy.signal import correlate2d
    
    # corr = correlate2d(influ,real_influ)
    
    
    size_A = influ.shape[0]
    
    # schmidt 2010 pg 45
    
    temp_corr = np.fft.fftshift(np.fft.ifft2(
        np.fft.fft2(np.pad(
            np.nan_to_num(influ),
            ((size_A//2,size_A//2),(size_A//2,size_A//2)),mode='constant'))
        * np.conjugate(np.fft.fft2(np.pad(
            np.nan_to_num(-masked_real_influ),
            ((size_A//2,size_A//2),(size_A//2,size_A//2)),mode='constant')))
        ))#.real[size_A//2:size_A*3//2,size_A//2:size_A*3//2]
    
    mask_small = np.pad(np.ones((size_A,size_A),dtype=float),
                      ((size_A//2,size_A//2),(size_A//2,size_A//2)),
                      mode='constant')

    mask_corr = np.fft.fftshift(np.fft.ifft2(np.abs(
        np.fft.fft2(mask_small))**2))#.real[size_A//2:size_A*3//2,size_A//2:size_A*3//2]
    
    idx = (mask_corr != 0)
    c = np.zeros_like(temp_corr)
    c[idx] = temp_corr[idx] / mask_corr[idx] * mask_small[idx]
    corr = c.real[size_A//2:size_A*3//2,size_A//2:size_A*3//2]
    
    corr = -temp_corr.real[size_A//2:size_A*3//2,size_A//2:size_A*3//2]
    
    # plt.imshow(corr);plt.show()
    
    influ_tt[trial] = -np.asarray(find_centre(corr, dsub_pxl))*dsub_pxl
    
    # plt.imshow(corr);plt.plot(D_pxl//2-influ_tt[trial,1],D_pxl//2-influ_tt[trial,0],marker='o',color='r')
    # plt.show()
    
    # plt.imshow(influ,vmin=np.nanmin(real_influ),vmax=np.nanmax(real_influ))
    # plt.plot(D_pxl//2-influ_tt[trial,1],
    #          D_pxl//2-influ_tt[trial,0],
    #          marker='o',color='r')
    # plt.show()
    
    influ_tt[trial] = influ_tt[trial] * pixel_scale/z
    
    
    
    
###############################################################################
# test when use different actuators to measure aperture shift
###############################################################################

if nxsub == 8:
    
    FULL_ACTS = np.array([22,
                             30,31,32,
                             38,39,40,41,42,
                             48,49,50,
                             58],dtype=int)
if nxsub == 12:
    FULL_ACTS = np.array([ 32,
                              43, 44, 45, 46, 47,
                              55, 56, 57, 58, 59, 60, 61,
                              68, 69, 70, 71, 72, 73, 74,
                              80, 81, 82, 83, 84, 85, 86, 87, 88,
                              94, 95, 96, 97, 98, 99,100,
                             107,108,109,110,111,112,113,
                             121,122,123,124,125,
                             136],dtype=int)

if nxsub == 4:
    FULL_ACTS = np.array([12],dtype=int)

soapy_influ_tt = np.zeros((FULL_ACTS.shape[0],TRIAL,2),dtype=float)

for i in np.arange(FULL_ACTS.shape[0]):

    i_act = FULL_ACTS[i]    

    actGrid = np.zeros((nxsub + 1, nxsub + 1),dtype=float)
    # actGrid[nxsub//2,nxsub//2] = 2*np.pi/100
    actGrid[i_act//(nxsub + 1),i_act%(nxsub + 1)] = 2*np.pi/100
    actGrid = np.pad(actGrid,((1,1),(1,1)))
    
    # actGrid before pad + pad with 1 subap each side
    N = D_pxl + 2*dsub_pxl
    
    dmShape = interp.zoom_rbs(
            actGrid, (N, N))
    
    # plt.imshow(dmShape)
    # plt.hlines(np.arange(dsub_pxl,N,dsub_pxl), dsub_pxl, N - dsub_pxl,color='r')
    # plt.vlines(np.arange(dsub_pxl,N,dsub_pxl), dsub_pxl, N - dsub_pxl,color='r')
    # plt.title('{:}'.format(i))
    # plt.show()
        
    
    
    
    real_influ = dmShape
    
    # influ_tt = np.zeros((TRIAL,2),dtype=float)
    
    x = np.arange(-D_pxl/2,D_pxl/2) + 0.5
    yy,xx = np.meshgrid(x,x)
    
    mask = ((xx**2+yy**2)<=D_pxl**2/4)
    
    # real_influ[~mask] = np.nan
    
    masked_real_influ = np.copy(real_influ[dsub_pxl:-dsub_pxl,dsub_pxl:-dsub_pxl])
    masked_real_influ[~mask] = np.nan
    
    pad = np.max([ int(np.ceil(wvl*z/2/pixel_scale**2/2)), int(np.ceil((wvl*z/pixel_scale**2 - D_pxl)/2))])
    
    temp = (xx**2+yy**2)<=D_pxl**2/4
    
    PUPIL = np.pad(temp,((pad,pad),(pad,pad)))
    
    for trial in tqdm(range(TRIAL)):
        
        # GLDM = np.zeros((D_pxl,D_pxl),dtype=float)
        # GLDM[D_pxl//2-dsub_pxl//2
        #      :D_pxl//2+dsub_pxl//2,
        #      D_pxl//2-dsub_pxl//2
        #      :D_pxl//2+dsub_pxl//2] = 2*np.pi/100
        
        GLDM = np.copy(real_influ)
        
        # real_influ = np.copy(GLDM)
        
        # plt.imshow(GLDM);plt.colorbar();plt.title('GLDM');plt.show()
        
        # z = 5000
        
    
        
        Uin = np.pad(np.exp(1j*GLDM),((pad-dsub_pxl,pad-dsub_pxl),(pad-dsub_pxl,pad-dsub_pxl))) * PUPIL
        # because there is extra pitch space buffer on each side
        # plt.imshow(np.abs(Uin)**2);plt.show()
        # plt.imshow(np.angle(Uin));plt.show()
        Uout = angularSpectrum(Uin, wvl, pixel_scale, pixel_scale, z)
        
        IPS = PhaseScreenKolmogorov(D_pxl, pixel_scale, r0, L0)
        for row in range(D_pxl):
            IPS.add_row()
        
        global_scrn = IPS.scrn
        
        local_scrns = np.asarray(np.split(np.asarray(np.split(global_scrn,nxsub,axis=0)),nxsub,axis=-1)).reshape(nxsub**2,dsub_pxl,dsub_pxl)
        
        global_tip = (global_scrn[1:] - global_scrn[:-1]).mean()
        global_tilt = (global_scrn[:,1:] - global_scrn[:,:-1]).mean()
        
        global_tt[trial,0] = global_tip/pixel_scale*wvl/2/np.pi
        global_tt[trial,1] = global_tilt/pixel_scale*wvl/2/np.pi
        
        global_x = np.arange(D_pxl)
        global_yy, global_xx = np.meshgrid(global_x,global_x)
        
        local_x = np.arange(dsub_pxl)
        
        global_scrn_no_tt = global_scrn - global_xx*global_tip - global_yy*global_tilt
        global_scrn_no_tt -= global_scrn_no_tt.mean()
        
        kspace = np.fft.fftshift(np.fft.fft2(np.pad(global_scrn_no_tt,((D_pxl//2,D_pxl//2),(D_pxl//2,D_pxl//2)),mode='symmetric')))
        filter_kspace = np.zeros_like(kspace)
        
        k_scale = 1/D/2 # the correct
        # k_scale = 1/D/2/2 # the one i used in soapy
        k_filter = 1/2/dsub
        n_filter = int(k_filter//k_scale)
        
        filter_kspace[D_pxl - n_filter
                      : D_pxl + n_filter,
                      D_pxl - n_filter
                      : D_pxl + n_filter] = kspace[D_pxl - n_filter
                                                      : D_pxl + n_filter,
                                                      D_pxl - n_filter
                                                      : D_pxl + n_filter]
        filter_global_scrn_no_tt = (np.fft.ifft2(np.fft.ifftshift(
            filter_kspace)).real)[D_pxl//2:3*D_pxl//2,D_pxl//2:3*D_pxl//2]
        
        Uin = Uout*np.exp(1j*np.pad(filter_global_scrn_no_tt,((pad,pad),(pad,pad))))
        # plt.imshow(np.angle(Uin));plt.show()
        # plt.imshow(np.abs(Uin)**2);plt.show()
        
        Uout = angularSpectrum(Uin, wvl, pixel_scale, pixel_scale, -z)
        
        # plt.imshow(np.abs(Uout)**2);plt.show()
        # plt.imshow(np.angle(Uout));plt.show()
        
        with_poke = Uout
        
        
        
        GLDM = np.zeros((D_pxl,D_pxl),dtype=float)
        # GLDM[D_pxl//2-dsub_pxl//2
        #      :D_pxl//2+dsub_pxl//2,
        #      D_pxl//2-dsub_pxl//2
        #      :D_pxl//2+dsub_pxl//2] = 2*np.pi/100
        # plt.imshow(GLDM);plt.colorbar();plt.title('GLDM');plt.show()
        
        Uin = np.pad(np.exp(1j*GLDM),((pad,pad),(pad,pad))) * PUPIL
        # plt.imshow(np.abs(Uin)**2);plt.show()
        # plt.imshow(np.angle(Uin));plt.show()
        Uout = angularSpectrum(Uin, wvl, pixel_scale, pixel_scale, z)
        
        
        
        Uin = Uout*np.exp(1j*np.pad(filter_global_scrn_no_tt,((pad,pad),(pad,pad))))
        # plt.imshow(np.angle(Uin));plt.show()
        # plt.imshow(np.abs(Uin)**2);plt.show()
        
        Uout = angularSpectrum(Uin, wvl, pixel_scale, pixel_scale, -z)
        
        # plt.imshow(np.abs(Uout)**2);plt.show()
        # plt.imshow(np.angle(Uout));plt.show()
        
        no_poke = Uout
        
        
        
        influ = np.angle(with_poke/no_poke)[pad:-pad,pad:-pad]
        influ[~mask] = np.nan
        
        # plt.imshow(influ);plt.colorbar();plt.show()
        # plt.imshow(real_influ);plt.colorbar();plt.show()
        
        
        # from scipy.signal import correlate2d
        
        # corr = correlate2d(influ,real_influ)
        
        
        size_A = influ.shape[0]
        
        # schmidt 2010 pg 45
        
        temp_corr = np.fft.fftshift(np.fft.ifft2(
            np.fft.fft2(np.pad(
                np.nan_to_num(influ),
                ((size_A//2,size_A//2),(size_A//2,size_A//2)),mode='constant'))
            * np.conjugate(np.fft.fft2(np.pad(
                np.nan_to_num(-masked_real_influ),
                ((size_A//2,size_A//2),(size_A//2,size_A//2)),mode='constant')))
            ))#.real[size_A//2:size_A*3//2,size_A//2:size_A*3//2]
        
        mask_small = np.pad(np.ones((size_A,size_A),dtype=float),
                          ((size_A//2,size_A//2),(size_A//2,size_A//2)),
                          mode='constant')
    
        mask_corr = np.fft.fftshift(np.fft.ifft2(np.abs(
            np.fft.fft2(mask_small))**2))#.real[size_A//2:size_A*3//2,size_A//2:size_A*3//2]
        
        idx = (mask_corr != 0)
        c = np.zeros_like(temp_corr)
        c[idx] = temp_corr[idx] / mask_corr[idx] * mask_small[idx]
        corr = c.real[size_A//2:size_A*3//2,size_A//2:size_A*3//2]
        
        corr = -temp_corr.real[size_A//2:size_A*3//2,size_A//2:size_A*3//2]
        
        # plt.imshow(corr);plt.show()
        
        soapy_influ_tt[i,trial] = -np.asarray(find_centre(corr, dsub_pxl))*dsub_pxl
        
        # plt.imshow(corr);plt.plot(D_pxl//2-soapy_influ_tt[i,trial,1],D_pxl//2-soapy_influ_tt[i,trial,0],marker='o',color='r')
        # plt.show()
        
        # plt.imshow(influ,vmin=np.nanmin(real_influ),vmax=np.nanmax(real_influ))
        # plt.plot(D_pxl//2-soapy_influ_tt[i,trial,1],
        #           D_pxl//2-soapy_influ_tt[i,trial,0],
        #           marker='o',color='r')
        # plt.show()
        
        soapy_influ_tt[i,trial] = soapy_influ_tt[i,trial] * pixel_scale/z
    


###############################################################################
    
    
print('saint-jaques coeff : {:.3f}'.format(saintjaques))
print('vk L0 adjust saint-jaques coeff : {:.3f}'.format(vk_ratio*saintjaques))
print('measure global coeff : {:.3f}'.format(global_tt.var()/global_tt_variable))
print('measure local all coeff : {:.3f}'.format(local_tt.var()/tt_variable))
print('simple calc : {:.3f}'.format(saintjaques*(1-nxsub**(-1/3))))
print('butterly coeff : {:.3f}'.format(tims))
print('measure local no global coeff : {:3f}'.format(local_tt_no_global.var()/tt_variable))
print('measure local no global filter coeff : {:.3f}'.format(filter_local_tt_no_global.var()/tt_variable))
print('measure prop filter dm : {:.3f}'.format(ground_subap_tt.var()/tt_variable))
print('centre act influ tiptilt coeff : {:.3f}'.format(influ_tt.var()/tt_variable))
print('all act influ tiptilt coeff : {:.3f}'.format(soapy_influ_tt.var()/tt_variable))




# from soapy.soapy import interp

# actGrid = np.zeros((nxsub + 1, nxsub + 1),dtype=float)
# actGrid[nxsub//2,nxsub//2] = 2*np.pi/100

# dmShape = interp.zoom_rbs(
#         actGrid, (D_pxl,D_pxl))
# plt.imshow(dmShape)



'''
nxsub=12
z=5000
dsub=0.05
r0=0.05
saint-jaques coeff : 0.162
vk L0 adjust saint-jaques coeff : 0.046
measure global coeff : 0.036
measure local all coeff : 0.101
simple calc : 0.091
butterly coeff : 0.096
measure local no global coeff : 0.087498
measure local no global filter coeff : 0.076
measure prop filter dm : 0.067
centre act influ tiptilt coeff : 0.041
all act influ tiptilt coeff : 0.040
'''

'''
nxsub=8
z=5000
dsub=0.05
r0=0.05
saint-jaques coeff : 0.162
vk L0 adjust saint-jaques coeff : 0.058
measure global coeff : 0.048
measure local all coeff : 0.100
simple calc : 0.081
butterly coeff : 0.082
measure local no global coeff : 0.080136
measure local no global filter coeff : 0.065
measure prop filter dm : 0.058
centre act influ tiptilt coeff : 0.031
all act influ tiptilt coeff : 0.029
'''

'''
nxsub=4
z=5000
dsub=0.05
r0=0.05
saint-jaques coeff : 0.162
vk L0 adjust saint-jaques coeff : 0.078
measure global coeff : 0.066
measure local all coeff : 0.093
simple calc : 0.060
butterly coeff : 0.063
measure local no global coeff : 0.058771
measure local no global filter coeff : 0.042
measure prop filter dm : 0.036
centre act influ tiptilt coeff : 0.005
all act influ tiptilt coeff : 0.006
'''

'''
nxsub=4
z=10000
dsub=0.05
r0=0.05
saint-jaques coeff : 0.162
vk L0 adjust saint-jaques coeff : 0.078
measure global coeff : 0.065
measure local all coeff : 0.099
simple calc : 0.060
butterly coeff : 0.063
measure local no global coeff : 0.061332
measure local no global filter coeff : 0.042
measure prop filter dm : 0.030
centre act influ tiptilt coeff : 0.025
all act influ tiptilt coeff : 0.020
'''

'''
nxsub=8
z=10000
dsub=0.05
r0=0.05
saint-jaques coeff : 0.162
vk L0 adjust saint-jaques coeff : 0.058
measure global coeff : 0.053
measure local all coeff : 0.103
simple calc : 0.081
butterly coeff : 0.082
measure local no global coeff : 0.080013
measure local no global filter coeff : 0.066
measure prop filter dm : 0.045
centre act influ tiptilt coeff : 0.043
all act influ tiptilt coeff : 0.039
'''



'''
nxsub=8
z=10000
dsub=0.05
r0=0.10
saint-jaques coeff : 0.162
vk L0 adjust saint-jaques coeff : 0.058
measure global coeff : 0.048
measure local all coeff : 0.102
simple calc : 0.081
butterly coeff : 0.201
measure local no global coeff : 0.079222
measure local no global filter coeff : 0.066
measure prop filter dm : 0.055
centre act influ tiptilt coeff : 0.022
all act influ tiptilt coeff : 0.027
'''


'''
dkist dm11km
dsub = 0.27
dsub_pxl = 16
nxsub = 15
wvl = 500e-9
D = dsub*nxsub
pixel_scale = dsub/dsub_pxl
D_pxl = dsub_pxl*nxsub


r0 = 0.1
L0 = 20
TRIAL = 100
z = 11200

saint-jaques coeff : 0.162
vk L0 adjust saint-jaques coeff : 0.036
measure global coeff : 0.030
measure local all coeff : 0.106
simple calc : 0.096
butterly coeff : 
measure local no global coeff : 0.094320
measure local no global filter coeff : 0.083
measure prop filter dm : 0.083
centre act influ tiptilt coeff : 0.045
all act influ tiptilt coeff : 
'''


