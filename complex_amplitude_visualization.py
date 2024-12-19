# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:37:21 2024

@author: gqmc62
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov
from tqdm import tqdm
from aotools.opticalpropagation import angularSpectrum



def crop(A,ratio):
    NN = A.shape[0]
    N = NN//ratio
    B = A[(NN-N)//2:(NN+N)//2,
          (NN-N)//2:(NN+N)//2]
    return B

def prep_plot(U,norm_phase=False):
    intensity = np.abs(U)**2
    intensity /= intensity.mean()
    if norm_phase == True:
        phase = np.angle(U/U.mean())
    else:
        phase = np.angle(U/U.mean())
    return phase, intensity

def hsv_map(phase,intensity,imin=0,imax=1,color_offset=0):
    h = (phase/2./np.pi + color_offset)
    h -= np.floor(h)
    s = (intensity - imin)/(imax - imin)
    v = 1
    if s > 1:
        s = 1
    if s < 0:
        s = 0
    return (h,s,v,)

def anymap_new(phase,intensity,imin=0,imax=1,color_offset=0,cmap=plt.colormaps["viridis"]):
    phase_factor = (phase/2./np.pi + color_offset)
    phase_factor -= np.floor(phase_factor)
    intensity_factor = (intensity - imin)/(imax - imin)
    if intensity_factor > 1:
        intensity_factor = 1
    if intensity_factor < 0:
        intensity_factor = 0
        
    og_tuple = cmap(phase_factor)
    og_hsv = colors.rgb_to_hsv((og_tuple[0],og_tuple[1],og_tuple[2],))
    
    h = og_hsv[0]
    s = og_hsv[1]
    v = og_hsv[2]
    
    new_h = h
    
    if intensity_factor <= 0.5:
        new_s = 1
        new_v = intensity_factor*2
    
    if intensity_factor > 0.5:
        new_v = 1
        new_s = 2 - 2*intensity_factor
    
    new_v = new_v*v
    new_s = new_s*s
    
    new_v = np.log(1+new_v)/np.log(2)
    new_s = np.log(1+new_s)/np.log(2)
    
    new_hsv = (new_h,new_s,new_v,)
    
    return new_hsv

def anymap_og(phase,intensity,imin=0,imax=1,color_offset=0,cmap=plt.colormaps["viridis"]):
    phase_factor = (phase/2./np.pi + color_offset)
    phase_factor -= np.floor(phase_factor)
    intensity_factor = (intensity - imin)/(imax - imin)
    if intensity_factor > 1:
        intensity_factor = 1
    if intensity_factor < 0:
        intensity_factor = 0
        
    og_tuple = cmap(phase_factor)
    og_hsv = colors.rgb_to_hsv((og_tuple[0],og_tuple[1],og_tuple[2],))
    
    h = og_hsv[0]
    s = og_hsv[1]
    v = og_hsv[2]
    
    new_h = h
    
    # if intensity_factor <= 0.5:
    #     new_s = 1
    #     new_v = intensity_factor*2
    
    # if intensity_factor > 0.5:
    #     new_v = 1
    #     new_s = 2 - 2*intensity_factor
    
    new_s = 1
    new_v = intensity_factor

    new_v = new_v*v
    new_s = new_s*s
    
    new_v = np.log(1+new_v)/np.log(2)
    # new_s = np.log(1+new_s)/np.log(2)
    
    new_hsv = (new_h,new_s,new_v,)
    
    return new_hsv

def anymap_newer(phase,intensity,range_logi=3,color_offset=0,cmap=plt.colormaps["viridis"]):
    phase_factor = (phase/2./np.pi + color_offset)
    phase_factor -= np.floor(phase_factor)
    intensity_factor = np.log(intensity)/(2*range_logi)
    intensity_factor = intensity_factor + 0.5
    if intensity_factor > 1:
        intensity_factor = 1
    if intensity_factor < 0:
        intensity_factor = 0
        
    og_tuple = cmap(phase_factor)
    og_hsv = colors.rgb_to_hsv((og_tuple[0],og_tuple[1],og_tuple[2],))
    
    h = og_hsv[0]
    s = og_hsv[1]
    v = og_hsv[2]
    
    new_h = h
    
    # if intensity_factor <= 0.5:
    #     new_s = 1
    #     new_v = intensity_factor*2
    
    # if intensity_factor > 0.5:
    #     new_v = 1
    #     new_s = 2 - 2*intensity_factor
    
    new_s = 1
    new_v = intensity_factor

    new_v = new_v*v
    new_s = new_s*s
    
    # new_v = np.log(1+new_v)/np.log(2)
    # new_s = np.log(1+new_s)/np.log(2)
    
    new_hsv = (new_h,new_s,new_v,)
    
    return new_hsv

def anymap(phase,intensity,range_logi=3,color_offset=0,cmap=plt.colormaps["viridis"]):
    phase_factor = (phase/2./np.pi + color_offset)
    phase_factor -= np.floor(phase_factor)
    intensity_factor = np.log(intensity)/(2*range_logi)
    intensity_factor = intensity_factor + 0.5
    if intensity_factor > 1:
        intensity_factor = 1
    if intensity_factor < 0:
        intensity_factor = 0
        
    og_tuple = cmap(phase_factor)
    og_hsv = colors.rgb_to_hsv((og_tuple[0],og_tuple[1],og_tuple[2],))
    
    h = og_hsv[0]
    s = og_hsv[1]
    v = og_hsv[2]
    
    new_h = h
    
    if intensity_factor <= 0.5:
        new_s = 1
        new_v = intensity_factor*2
    
    if intensity_factor > 0.5:
        new_v = 1
        new_s = 2 - 2*intensity_factor
    
    # new_s = 1
    # new_v = intensity_factor

    new_v = new_v*v
    new_s = new_s*s
    
    # new_v = np.log(1+new_v)/np.log(2)
    # new_s = np.log(1+new_s)/np.log(2)
    
    new_hsv = (new_h,new_s,new_v,)
    
    return new_hsv

def detect_branch_point(B):
    A = B/np.abs(B)
    sx = np.angle(A[1:,:]/A[:-1,:])
    sy = np.angle(A[:,1:]/A[:,:-1])
    # POS = []
    # for i in np.arange(A.shape[0] - 2):
    #     for j in np.arange(A.shape[1] - 2):
    #         test = sx[i, j] + sy[i + 1, j] - sx[i + 1, j + 1] - sy[i , j + 1]
    #         print(sx[i, j] , sy[i + 1, j] , sx[i + 1, j + 1] , sy[i , j + 1],test)
    #         CON = (np.isclose(np.abs(test),0,atol=0.1) == False)
    #         if CON:
    #             POS.append([i,j])
    test = sx[:,:-1]-sy[:-1,:]-sx[:,1:]+sy[1:,:]
    # print(test)
    # print((np.isclose(test,0,atol=0.1)==False).sum())
    # POS = np.asarray(POS)
    POS = np.where(np.isclose(test,0,atol=0.1)==False)
    return POS
    
# N = 256
# wvl = 500e-9
# D = 5e-3
# delta = D/N
# k = 2*np.pi/wvl
# f = 25e-3
# scale = 16
# delta_f = wvl/scale/D


# NN = N*scale

# x = np.arange(-NN/2,NN/2) + 0.5

# yy,xx = np.meshgrid(x,x)
# rr = np.sqrt(xx**2+yy**2)

# U_pupil = np.ones((NN,NN),dtype=complex)
# U_pupil[rr>N//2] = 0

# plt.imshow(np.abs(crop(U_pupil,scale))**2)
# plt.colorbar()
# plt.title('pupil plane intensity')
# plt.show()

# TT = np.exp(-1j*k*(xx+yy)*delta*delta_f/2.)

# U_focal = np.exp(1j*k*f)/1j/wvl/f*np.exp(1j*(rr*delta_f)**2*k/2.)*np.fft.fftshift(np.fft.fft2(U_pupil*TT))*TT

# plt.imshow(np.abs(crop(U_focal,scale))**2)
# plt.colorbar()
# plt.title('focal plane intensity')
# plt.show()

# # plt.imshow(np.unwrap(np.unwrap(np.angle(crop(U_focal,scale)),axis=-1),axis=-2))
# plt.imshow(np.angle(crop(U_focal,scale)))
# plt.colorbar()
# plt.title('focal plane phase')
# plt.show()


# N = 64
# D = 5e-3
# f = 2
# z = f#25e-3
# wvl = 500e-9
# k = 2.*np.pi/wvl
# delta = D/N

# pad = wvl*z/2./delta/delta

# NN = 2**int(np.ceil(np.log(np.max([N + 2*pad,
#             wvl*z/delta**2]))/np.log(2)))

# if NN > 4096:
#     quit

# delta_f = 1./NN/delta

# scale = NN//N

# x = np.arange(-NN/2,NN/2) + 0.5

# yy,xx = np.meshgrid(x,x)
# rr = np.sqrt(xx**2+yy**2)

# U_pupil = np.ones((NN,NN),dtype=complex)
# U_pupil[rr>N//2] = 0

# plt.imshow(np.abs(crop(U_pupil,scale))**2)
# plt.colorbar()
# plt.title('pupil plane intensity')
# plt.show()

# U_lens = np.exp(-1j*k/2./f*(rr*delta)**2)
# # plt.imshow(np.unwrap(np.unwrap(np.angle(U_lens),axis=0),axis=1))
# # plt.colorbar()
# # plt.title('lens')
# # plt.show()

# U_TT = np.exp(-1j*k*(xx+yy)*delta*delta/z/2.)
# # plt.imshow(np.unwrap(np.unwrap(np.angle(U_TT),axis=0),axis=1))
# # plt.colorbar()
# # plt.title('TT')
# # plt.show()

# U_optical_transfer = np.exp(1j*k*z)*np.exp(-1j*np.pi*wvl*z*(rr*delta_f)**2)
# U_focus = np.fft.ifft2(np.fft.fftshift(U_optical_transfer)*np.fft.fft2((U_pupil*U_TT*U_lens)))*U_TT

# plt.imshow(np.abs(crop(U_focus,scale))**2)
# plt.colorbar()
# plt.title('focal plane intensity')
# plt.show()

# # plt.imshow(np.unwrap(np.unwrap(np.angle(crop(U_focus,scale)),axis=0),axis=1))
# # plt.colorbar()
# # plt.title('focal plane phase')
# # plt.show()

# plt.imshow(np.angle(crop(U_focus,scale)),cmap='twilight',vmin=-np.pi,vmax=np.pi)
# plt.colorbar()
# plt.title('focal plane phase')
# plt.show()


# phase,intensity = prep_plot(crop(U_focus,scale))

# plt.imshow(phase,cmap='twilight',vmin=-np.pi,vmax=np.pi)
# plt.colorbar()
# plt.title('focal plane complex amplitude')
# plt.show()

# plt.imshow(intensity,cmap='binary',vmin=0,vmax=1)
# plt.colorbar()
# plt.title('focal plane complex amplitude')
# plt.show()



# xx_crop = crop(xx,scale)
# yy_crop = crop(yy,scale)

# for ii in np.arange(N):
#     for jj in np.arange(N):
#         plt.plot(xx_crop[ii,jj],yy_crop[ii,jj],color=colors.hsv_to_rgb(hsv_map(phase[ii,jj],intensity[ii,jj])),ls='',marker='s')
# plt.axis('square')
# plt.show()

cmap = plt.colormaps["hsv"]


N = 64
D = 0.2
z = 20e3
wvl = 500e-9
k = 2.*np.pi/wvl
delta = D/N

MAX_INTENSITY = 3
CMAP_LW = 2
range_logi = 3

pad = wvl*z/2./delta/delta

NN = 2**int(np.ceil(np.log(np.max([
    N + 2*pad,
    wvl*z/delta**2
    ]))/np.log(2)))

if NN > 4096:
    quit

delta_f = 1./NN/delta

scale = NN//N

x = np.arange(-NN/2,NN/2) + 0.5

yy,xx = np.meshgrid(x,x)
rr = np.sqrt(xx**2+yy**2)


IPS = PhaseScreenKolmogorov(NN,delta,0.05,4)
if scale < 10:    
    for i in tqdm(np.arange(NN)):
        IPS.add_row()
turb = IPS.scrn
turb -= turb.mean()

# plt.imshow(crop(turb,scale))
# plt.colorbar()
# plt.show()

# plt.imshow(crop(turb,scale)%(2*np.pi)-np.pi,cmap='twilight')
# plt.colorbar()
# plt.title('turb')
# plt.show()

U_in = np.exp(1j*turb)


xx_crop = crop(xx,scale)
yy_crop = crop(yy,scale)

CMAPI = np.logspace(-range_logi,range_logi,64,base=np.e)
CMAPPHI = np.linspace(-np.pi,np.pi,64)



phase,intensity = prep_plot(crop(U_in,scale))

# plt.pcolor(delta*xx_crop,delta*yy_crop,phase,cmap='viridis',vmin=-np.pi,vmax=np.pi)
# plt.colorbar()
# plt.axis('square')
# # plt.plot(xx_crop[BRANCH_POINT] + 0.5*delta, yy_crop[BRANCH_POINT] + 0.5*delta, ls='', marker='o', color='r')
# plt.title('phase')
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
# plt.show()

# plt.pcolor(delta*xx_crop,delta*yy_crop,np.log(intensity),cmap='binary_r',vmin=-range_logi,vmax=range_logi)
# plt.colorbar()
# plt.axis('square')
# plt.title('log(intensity) (mean=0)')
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
# plt.show()

# BRANCH_POINT = detect_branch_point(crop(U_out,scale))
plt.clf()
fig = plt.figure(figsize=(5,5),dpi=500)
ax1 = fig.add_axes([0.1,0.5,0.9,0.5])
for ii in np.arange(N):
    for jj in np.arange(N):
        ax1.plot(delta*xx_crop[ii,jj],delta*yy_crop[ii,jj],
                 color=colors.hsv_to_rgb(anymap(phase[ii,jj],intensity[ii,jj],cmap=cmap,range_logi=range_logi)),ls='',marker='s',markersize=5)
# plt.plot(xx_crop[BRANCH_POINT] + 0.5*delta, yy_crop[BRANCH_POINT] + 0.5*delta, ls='', marker='o', color='k')
ax1.axis('square')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_title('Complex Amplitude')

ax2 = fig.add_axes([0.88,0.5,0.1,0.5])
# ax2.invert_xaxis()
for i in np.arange(CMAPPHI.shape[0]):
    for j in np.arange(CMAPI.shape[0]):
        ax2.plot(np.log(CMAPI[j]),CMAPPHI[i],ls='',marker='s',markersize=5,
                 color=colors.hsv_to_rgb(anymap(CMAPPHI[i],CMAPI[j],cmap=cmap,range_logi=range_logi)))
ax2.set_ylabel('phase')
ax2.yaxis.set_label_position("right")
ax2.set_xlabel('log(intensity)\n(mean=0)')
ax2.set_xlim(range_logi,range_logi)
ax2.set_xticks([-3,0,3])
ax2.set_xticklabels(['-3','0','3'])
# ax2.vlines([-3,0,3],-np.pi,np.pi,color='k',linewidth=CMAP_LW)
# ax2.hlines([0,np.pi/2,-np.pi/2,np.pi/4,-np.pi/4,3*np.pi/4,-3*np.pi/4],0,3,color='k')
ax2.hlines(0,-range_logi,range_logi,color=colors.hsv_to_rgb((0.5,1,1,)),linewidth=CMAP_LW)
ax2.hlines([np.pi,-np.pi],-range_logi,range_logi,color=colors.hsv_to_rgb((1,1,1,)),linewidth=CMAP_LW)
ax2.hlines(np.pi/2,-range_logi,range_logi,color=colors.hsv_to_rgb((0.75,1,1,)),linewidth=CMAP_LW)
ax2.hlines(-np.pi/2,-range_logi,range_logi,color=colors.hsv_to_rgb((0.25,1,1,)),linewidth=CMAP_LW)
ax2.hlines([np.pi/4,-np.pi/4,3*np.pi/4,-3*np.pi/4],-range_logi,range_logi,color='k',ls=':',linewidth=CMAP_LW)


# plt.axis('equal')
plt.title('color scale')
plt.show()
    
# plt.pcolor(delta*xx_crop,delta*yy_crop,phase,vmin=-np.pi,vmax=np.pi,cmap='viridis')
# plt.colorbar()
# # plt.plot(delta*xx_crop[BRANCH_POINT] + 0.5*delta**2, delta*yy_crop[BRANCH_POINT] + 0.5*delta**2, ls='', marker='o', color='r')
# plt.axis('square')
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
# CS = plt.contour(delta*xx_crop,delta*yy_crop,np.log(intensity),colors='k',
#                  vmin=-3,vmax=3)
# plt.clabel(CS,colors='k')
# plt.title('color map = phase'
#           + '\ncontour : log(intensity) (mean=0)')
# plt.show()


numz=20
for i_z in np.arange(numz):
    temp_z = z/numz*(i_z + 1)
    
    U_out = angularSpectrum(U_in, wvl, delta, delta, temp_z)
    
    # intensity distribution
    # log normal
    # for variance log amplitude = 0.6
    # I<=2 ==> 70%
    # I<=3 ==> 80%
    # I<=5 ==> 90%
    # I<=8 ==> 95%
    
    # plt.pcolor(xx_crop,yy_crop,np.abs(crop(U_out,scale))**2,vmin=0,vmax=3)
    # plt.colorbar()
    # plt.axis('square')
    # plt.title('intensity')
    # plt.show()
    
    # plt.pcolor(xx_crop,yy_crop,np.angle(crop(U_out,scale)),vmin=-np.pi,vmax=np.pi)
    # plt.colorbar()
    # plt.title('phase')
    # plt.axis('square')
    # plt.show()
    
    phase,intensity = prep_plot(crop(U_out,scale))
    
    BRANCH_POINT = detect_branch_point(crop(U_out,scale))
    
    # plt.pcolor(delta*xx_crop,delta*yy_crop,phase,cmap='viridis',vmin=-np.pi,vmax=np.pi)
    # plt.colorbar()
    # plt.axis('square')
    # plt.plot(delta*xx_crop[BRANCH_POINT] + delta*0.5*delta, delta*yy_crop[BRANCH_POINT] + delta*0.5*delta, ls='', marker='o', color='r')
    # plt.title('phase')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    # plt.show()
    
    # plt.pcolor(delta*xx_crop,delta*yy_crop,np.log(intensity),cmap='binary_r',vmin=-range_logi,vmax=range_logi)
    # plt.plot(delta*xx_crop[BRANCH_POINT] + 0.5*delta**2, delta*yy_crop[BRANCH_POINT] + 0.5*delta**2, ls='', marker='o', color='r')
    # plt.axis('square')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    # plt.colorbar()
    # plt.title('log(intensity) (mean=0)')
    # plt.show()
    
    # BRANCH_POINT = detect_branch_point(crop(U_out,scale))
    plt.clf()
    fig = plt.figure(figsize=(5,5),dpi=500)
    ax1 = fig.add_axes([0.1,0.5,0.9,0.5])
    for ii in np.arange(N):
        for jj in np.arange(N):
            ax1.plot(delta*xx_crop[ii,jj],delta*yy_crop[ii,jj],
                     color=colors.hsv_to_rgb(anymap(phase[ii,jj],intensity[ii,jj],cmap=cmap,range_logi=range_logi)),ls='',marker='s',markersize=5)
    plt.plot(delta*xx_crop[BRANCH_POINT] + 0.5*delta**2, delta*yy_crop[BRANCH_POINT] + 0.5*delta**2, ls='', marker='o', color='grey')
    ax1.axis('square')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Complex Amplitude')
    
    ax2 = fig.add_axes([0.88,0.5,0.1,0.5])
    # ax2.invert_xaxis()
    for i in np.arange(CMAPPHI.shape[0]):
        for j in np.arange(CMAPI.shape[0]):
            ax2.plot(np.log(CMAPI[j]),CMAPPHI[i],ls='',marker='s',markersize=5,
                     color=colors.hsv_to_rgb(anymap(CMAPPHI[i],CMAPI[j],cmap=cmap,range_logi=range_logi)))
    ax2.set_ylabel('phase')
    ax2.yaxis.set_label_position("right")
    ax2.set_xlabel('log(intensity)\n(mean=0)')
    ax2.set_xlim(range_logi,range_logi)
    ax2.set_xticks([-3,0,3])
    ax2.set_xticklabels(['-3','0','3'])
    # ax2.vlines([-3,0,3],-np.pi,np.pi,color='k',linewidth=CMAP_LW)
    # ax2.hlines([0,np.pi/2,-np.pi/2,np.pi/4,-np.pi/4,3*np.pi/4,-3*np.pi/4],0,3,color='k')
    ax2.hlines(0,-range_logi,range_logi,color=colors.hsv_to_rgb((0.5,1,1,)),linewidth=CMAP_LW)
    ax2.hlines([np.pi,-np.pi],-range_logi,range_logi,color=colors.hsv_to_rgb((1,1,1,)),linewidth=CMAP_LW)
    ax2.hlines(np.pi/2,-range_logi,range_logi,color=colors.hsv_to_rgb((0.75,1,1,)),linewidth=CMAP_LW)
    ax2.hlines(-np.pi/2,-range_logi,range_logi,color=colors.hsv_to_rgb((0.25,1,1,)),linewidth=CMAP_LW)
    ax2.hlines([np.pi/4,-np.pi/4,3*np.pi/4,-3*np.pi/4],-range_logi,range_logi,color='k',ls=':',linewidth=CMAP_LW)
    # plt.axis('equal')
    plt.title('color scale')
    plt.show()
    
    
# for i_z in np.arange(numz):
#     temp_z = z/numz*(i_z + 1)
    
#     U_out = angularSpectrum(U_in, wvl, delta, delta, temp_z)
    
#     # intensity distribution
#     # log normal
#     # for variance log amplitude = 0.6
#     # I<=2 ==> 70%
#     # I<=3 ==> 80%
#     # I<=5 ==> 90%
#     # I<=8 ==> 95%
    
#     # plt.pcolor(xx_crop,yy_crop,np.abs(crop(U_out,scale))**2,vmin=0,vmax=3)
#     # plt.colorbar()
#     # plt.axis('square')
#     # plt.title('intensity')
#     # plt.show()
    
#     # plt.pcolor(xx_crop,yy_crop,np.angle(crop(U_out,scale)),vmin=-np.pi,vmax=np.pi)
#     # plt.colorbar()
#     # plt.title('phase')
#     # plt.axis('square')
#     # plt.show()
    
#     phase,intensity = prep_plot(crop(U_out,scale))
    
    # # BRANCH_POINT = detect_branch_point(crop(U_out,scale))
    # plt.pcolor(delta*xx_crop,delta*yy_crop,phase,vmin=-np.pi,vmax=np.pi,cmap='viridis')
    # plt.colorbar()
    # plt.plot(delta*xx_crop[BRANCH_POINT] + 0.5*delta**2, delta*yy_crop[BRANCH_POINT] + 0.5*delta**2, ls='', marker='o', color='r')
    # plt.axis('square')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    # CS = plt.contour(delta*xx_crop,delta*yy_crop,np.log(intensity),colors='k',
    #                  vmin=-3,vmax=3)
    # plt.clabel(CS,colors='k')
    # plt.title('color map = phase'
    #           + '\ncontour : log(intensity) (mean=0)')
    # plt.show()
