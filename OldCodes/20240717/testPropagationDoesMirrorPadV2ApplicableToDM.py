# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:10:37 2024

@author: gqmc62
"""


from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov
from aotools.opticalpropagation import angularSpectrum
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import pickle
import my_phasescreen as mps

"""

SF, SFerror = mps.get_structure_function_rolling(phase)
PSD, PSDerror = 

"""

D = 0.4
delta = 0.05/16
r0 = 0.10
L0 = 8
z = 10e3
wvl = 500e-9

N = int(np.ceil(D/delta))

#to show that the mirror pad works fine
x = (np.arange(N)- N//2) * delta
atm_pad = int(np.ceil(2*wvl*z/r0/delta))
# atm pad is good with wvl*z/r0/delta 2*wvl*z/r0/delta is already overkill
prop_pad = int(np.ceil(wvl*z/2/delta**2))
# print(2*atm_pad,prop_pad)

IF3 = PhaseScreenKolmogorov(N, delta, r0, L0)
for i in np.arange(N):
    IF3.add_row()
scrn = IF3.scrn
scrn -= scrn.mean()

yy,xx = np.meshgrid(x,x)
rr = xx**2+yy**2
PUPIL = (rr<=D**2/4)
# plt.imshow(PUPIL)
# plt.colorbar()
# plt.title('pupil')
# plt.show()

plt.pcolor(x,x,np.angle(np.exp(1j*scrn))*PUPIL)
plt.colorbar()
plt.axis('square')
plt.xlabel('position (m)')
plt.title('phase before propagation')
plt.show()

plt.pcolor(x,x,np.abs(np.exp(1j*scrn))**2*PUPIL)
plt.colorbar()
plt.axis('square')
plt.xlabel('position (m)')
plt.title('phase before propagation')
plt.show()

###############################################################################
###############################################################################

#perfect math

Uin = np.exp(1j*scrn)*PUPIL
prop_pad_Uin = np.pad(Uin,((prop_pad//2 + atm_pad, prop_pad//2 + atm_pad),
                           (prop_pad//2 + atm_pad, prop_pad//2 + atm_pad)),
                      'constant',constant_values=(0,0))
prop_pad_Uout = angularSpectrum(prop_pad_Uin, wvl, delta, delta, z)[prop_pad//2:-(prop_pad//2),
                                                                    prop_pad//2:-(prop_pad//2)]

intensity_prop_pad = np.abs(prop_pad_Uout)**2
phase_prop_pad = np.angle(prop_pad_Uout)

x = (np.arange(N + 2*atm_pad) - N//2 - atm_pad) * delta
plt.pcolor(x,x,phase_prop_pad)
plt.colorbar()
plt.xlabel('position (m)')
plt.axis('square')
plt.title('phase_prop_pad (rad)')
# plt.savefig('sample_prop_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
plt.show()

plt.pcolor(x,x,intensity_prop_pad,vmax=2,vmin=0)
plt.colorbar()
plt.axis('square')
plt.xlabel('position (m)')
plt.title('normalized intensity_prop_pad')
# plt.savefig('sample_prop_intensity_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
plt.show()

###############################################################################

PSDUprop = np.zeros((N + 2*atm_pad, N + 2*atm_pad),dtype=float)
for i in np.arange(len(prop_pad_Uout)):
    PSDUprop[i] = mps.get_PSD(prop_pad_Uout[i],delta)
PSDUprop[:] += PSDUprop[:,::-1]
PSDUprop /= 2.

PSDphaseprop = np.zeros((N + 2*atm_pad, N + 2*atm_pad),dtype=float)
for i in np.arange(len(phase_prop_pad)):
    PSDphaseprop[i] = mps.get_PSD(phase_prop_pad[i],delta)
PSDphaseprop[:] += PSDphaseprop[:,::-1]
PSDphaseprop /= 2.

PSDintensityprop = np.zeros((N + 2*atm_pad, N + 2*atm_pad),dtype=float)
for i in np.arange(len(intensity_prop_pad)):
    PSDintensityprop[i] = mps.get_PSD(intensity_prop_pad[i],delta)
PSDintensityprop[:] += PSDintensityprop[:,::-1]
PSDintensityprop /= 2.

SFphaseprop = np.zeros((N + 2*atm_pad, N + 2*atm_pad), dtype=float)
# for i in np.arange(N):
#     SFphaseprop[i] = mps.get_structure_function(np.roll(phase_prop_pad,(i),axis=(0)))
# SFphaseprop = (phase_prop_pad[:,0] - phase_prop_pad[:,:])**2
for i in np.arange(N + 2*atm_pad):
    SFphaseprop[i] = np.angle((prop_pad_Uout[i,i] / np.roll(prop_pad_Uout[i],(-i),axis=(0))))**2
    



SFintensityprop = np.zeros((N + 2*atm_pad, N + 2*atm_pad), dtype=float)
# for i in np.arange(N):
#     SFintensityprop[i] = mps.get_structure_function(np.roll(intensity_prop_pad,(i),axis=(0)))
# SFintensityprop = (intensity_prop_pad[:,0] - intensity_prop_pad[:,:])**2
for i in np.arange(N + 2*atm_pad):
    SFintensityprop[i] = (intensity_prop_pad[i,i] - np.roll(intensity_prop_pad[i],(-i),axis=(0)))**2
    


###############################################################################
###############################################################################

# for atm_dif_ratio in np.arange(5):
# for extra_atm_pad in np.arange(4):

# atm_pad = int(np.ceil(atm_dif_ratio*wvl*z/r0/delta))

###############################################################################

# ##reduce infinite phase screen size & reduce propagation size = not working
# # PSD intensity not match at high frequency
# TAG = 'prop screen size = N + 2*2ATM_PAD'
# atm_pad_Uin = np.pad(Uin,((2*atm_pad,2*atm_pad),
#                           (2*atm_pad,2*atm_pad)),
#                      'constant',constant_values=(0,0))
# atm_pad_Uout = angularSpectrum(atm_pad_Uin, wvl, delta, delta, z)[atm_pad:-atm_pad,
#                                                                   atm_pad:-atm_pad]

# intensity_atm_pad = np.abs(atm_pad_Uout)**2
# phase_atm_pad = np.angle(atm_pad_Uout)
# x = (np.arange(N + 2* atm_pad) - N//2 - atm_pad) * delta
# plt.pcolor(x,x,phase_atm_pad)
# plt.colorbar()
# plt.xlabel('position (m)')
# plt.axis('square')
# plt.title('phase_atm_pad (rad)' + '\n' + TAG)
# # plt.savefig('sample_prop_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
# plt.show()

# plt.pcolor(x,x,intensity_atm_pad,vmax=2,vmin=0)
# plt.colorbar()
# plt.axis('square')
# plt.xlabel('position (m)')
# plt.title('normalized intensity_atm_pad'  + '\n' + TAG)
# # plt.savefig('sample_prop_intensity_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
# plt.show()

# PSDUatm = np.zeros((N + 2* atm_pad, N + 2* atm_pad),dtype=float)
# for i in np.arange(len(atm_pad_Uout)):
#     PSDUatm[i] = mps.get_PSD(atm_pad_Uout[i],delta)
# PSDUatm[:] += PSDUatm[:,::-1]
# PSDUatm /= 2.

# f=(np.arange((N + 2* atm_pad)//2-1) + 1)/(N + 2* atm_pad)/delta
# A = np.mean(PSDUatm[:,(N + 2* atm_pad)//2+1:],axis=0)
# Ae = np.std(PSDUatm[:,(N + 2* atm_pad)//2+1:],axis=0)/np.sqrt(N + 2* atm_pad)
# B = np.mean(PSDUprop[:,(N + 2* atm_pad)//2+1:],axis=0)
# Be = np.std(PSDUprop[:,(N + 2* atm_pad)//2+1:],axis=0)/np.sqrt(N + 2* atm_pad)

# plt.plot(f,A,color='r',label='atm pad')
# plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
# plt.plot(f,B,color='b',label='prop pad')
# plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
# plt.legend()
# plt.ylim(B.min()*0.9,B.max()*1.1)
# plt.title('psd U' + '\n' + TAG)
# plt.xlabel('f (1/m)')
# plt.ylabel('PSD_U (normI m$^2$)')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()



# PSDphaseatm = np.zeros((N + 2* atm_pad, N + 2* atm_pad),dtype=float)
# for i in np.arange(len(phase_atm_pad)):
#     PSDphaseatm[i] = mps.get_PSD(phase_atm_pad[i],delta)
# PSDphaseatm[:] += PSDphaseatm[:,::-1]
# PSDphaseatm /= 2.

# f=(np.arange((N + 2* atm_pad)//2-1) + 1)/(N + 2* atm_pad)/delta
# A = np.mean(PSDphaseatm[:,(N + 2* atm_pad)//2+1:],axis=0)
# Ae = np.std(PSDphaseatm[:,(N + 2* atm_pad)//2+1:],axis=0)/np.sqrt(N + 2* atm_pad)
# B = np.mean(PSDphaseprop[:,(N + 2* atm_pad)//2+1:],axis=0)
# Be = np.std(PSDphaseprop[:,(N + 2* atm_pad)//2+1:],axis=0)/np.sqrt(N + 2* atm_pad)

# plt.plot(f,A,color='r',label='atm pad')
# plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
# plt.plot(f,B,color='b',label='prop pad')
# plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
# plt.legend()
# plt.ylim(B.min()*0.9,B.max()*1.1)
# plt.title('psd phase' + '\n' + TAG)
# plt.xlabel('f (1/m)')
# plt.ylabel('PSD_$\phi$ (rad$^2$ m$^2$)')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()



# PSDintensityatm = np.zeros((N + 2* atm_pad, N + 2* atm_pad),dtype=float)
# for i in np.arange(len(intensity_atm_pad)):
#     PSDintensityatm[i] = mps.get_PSD(intensity_atm_pad[i],delta)
# PSDintensityatm[:] += PSDintensityatm[:,::-1]
# PSDintensityatm /= 2.

# f = (np.arange((N + 2* atm_pad)//2-1) + 1)/(N + 2* atm_pad)/delta
# A = np.mean(PSDintensityatm[:,(N + 2* atm_pad)//2+1:],axis=0)
# Ae = np.std(PSDintensityatm[:,(N + 2* atm_pad)//2+1:],axis=0)/np.sqrt(N + 2* atm_pad)
# B = np.mean(PSDintensityprop[:,(N + 2* atm_pad)//2+1:],axis=0)
# Be = np.std(PSDintensityprop[:,(N + 2* atm_pad)//2+1:],axis=0)/np.sqrt(N + 2* atm_pad)

# plt.plot(f,A,color='r',label='atm pad')
# plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
# plt.plot(f,B,color='b',label='prop pad')
# plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
# plt.vlines([1/np.sqrt(wvl*z)],B.min(),B.max(),label='1/$\sqrt{\lambda z}$',color='k')
# plt.legend()
# plt.ylim(B.min()*0.9,B.max()*1.1)
# plt.title('psd intensity' + '\n' + TAG)
# plt.xlabel('f (1/m)')
# plt.ylabel('PSD_I (normI m$^2$)')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()



# SFphaseatm = np.zeros((N + 2* atm_pad, N + 2* atm_pad), dtype=float)
# # for i in np.arange(N):
# #     SFphaseatm[i] = mps.get_structure_function(np.roll(phase_atm_pad,(i),axis=(0)))
# # SFphaseatm = (phase_atm_pad[:,0] - phase_atm_pad[:,:])**2
# for i in np.arange(N + 2* atm_pad):
#     SFphaseatm[i] = np.angle((atm_pad_Uout[i,i] / np.roll(atm_pad_Uout[i],(-i),axis=(0))))**2


# x = np.arange((N + 2* atm_pad)//2)*delta
# A = np.mean(SFphaseatm[:,:(N + 2* atm_pad)//2],axis=0)
# Ae = np.std(SFphaseatm[:,:(N + 2* atm_pad)//2],axis=0)/np.sqrt(N + 2* atm_pad)
# B = np.mean(SFphaseprop[:,:(N + 2* atm_pad)//2],axis=0)
# Be = np.std(SFphaseprop[:,:(N + 2* atm_pad)//2],axis=0)/np.sqrt(N + 2* atm_pad)

# plt.plot(x,A,color='r',label='atm pad')
# plt.fill_between(x,A-Ae,A+Ae,alpha=0.2,color='r')
# plt.plot(x,B,color='b',label='prop pad')
# plt.fill_between(x,B-Be,B+Be,alpha=0.2,color='b')
# plt.legend()
# plt.ylim(B[1:].min()*0.9,B.max()*1.1)
# plt.title('structure function $\phi$' + '\n' + TAG)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('separation (m)')
# plt.ylabel('$\sigma^2_\phi$ (rad$^2$)')
# plt.show()



# SFintensityatm = np.zeros(((N + 2* atm_pad), (N + 2* atm_pad)), dtype=float)
# # for i in np.arange(N):
# #     SFintensityatm[i] = mps.get_structure_function(np.roll(intensity_atm_pad,(i),axis=(0)))
# # SFintensityatm = (intensity_atm_pad[:,0] - intensity_atm_pad[:,:])**2
# for i in np.arange((N + 2* atm_pad)):
#     SFintensityatm[i] = (intensity_atm_pad[i,i] - np.roll(intensity_atm_pad[i],(-i),axis=(0)))**2

# x = np.arange((N + 2* atm_pad)//2)*delta
# A = np.mean(SFintensityatm[:,:(N + 2* atm_pad)//2],axis=0)
# Ae = np.std(SFintensityatm[:,:(N + 2* atm_pad)//2],axis=0)/np.sqrt(N + 2* atm_pad)
# B = np.mean(SFintensityprop[:,:(N + 2* atm_pad)//2],axis=0)
# Be = np.std(SFintensityprop[:,:(N + 2* atm_pad)//2],axis=0)/np.sqrt(N + 2* atm_pad)

# plt.plot(x,A,color='r',label='atm pad')
# plt.fill_between(x,A-Ae,A+Ae,alpha=0.2,color='r')
# plt.plot(x,B,color='b',label='prop pad')
# plt.fill_between(x,B-Be,B+Be,alpha=0.2,color='b')
# plt.vlines([np.sqrt(wvl*z)],
#             B.min(),B.max(),label='$\sqrt{\lambda z}$',color='k')
# plt.legend()
# plt.title('structure function I' + '\n' + TAG)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('separation (m)')
# plt.ylabel('$\sigma^2_I$ (normI$^2$)')
# plt.show()

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################


# reduce infinite phase screen size # pad to 2x(ATM) # symmetric pad
# everything matching!!!
TAG = 'prop screen size N+2*ATM_PAD + mirror_pad to --> 2x(N+2*ATM_PAD)'
atm_pad_scrn = np.pad(Uin,((atm_pad,atm_pad),
                           (atm_pad,atm_pad)),
                      'constant',constant_values=(0,0))
atm_pad_Uin = np.pad(atm_pad_scrn,
                     ((atm_pad_scrn.shape[0]//2,atm_pad_scrn.shape[0]//2),
                      (atm_pad_scrn.shape[0]//2,atm_pad_scrn.shape[0]//2)),'symmetric')#'symmetric',reflect_type='even'))
atm_pad_Uout = angularSpectrum(atm_pad_Uin, wvl, delta, delta, z)[(N+2*atm_pad)//2:-(N+2*atm_pad)//2,
                                                                  (N+2*atm_pad)//2:-(N+2*atm_pad)//2]

intensity_atm_pad = np.abs(atm_pad_Uout)**2
phase_atm_pad = np.angle(atm_pad_Uout)

x = (np.arange((N + 2* atm_pad))- (N + 2* atm_pad)//2) * delta
plt.pcolor(x,x,phase_atm_pad)
plt.colorbar()
plt.xlabel('position (m)')
plt.axis('square')
plt.title('phase_atm_pad (rad)' + '\n' + TAG)
# plt.savefig('sample_prop_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
plt.show()

plt.pcolor(x,x,intensity_atm_pad,vmax=2,vmin=0)
plt.colorbar()
plt.axis('square')
plt.xlabel('position (m)')
plt.title('normalized intensity_atm_pad' + '\n' + TAG)
# plt.savefig('sample_prop_intensity_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
plt.show()

PSDUatm = np.zeros(((N + 2* atm_pad),(N + 2* atm_pad)),dtype=float)
for i in np.arange(len(atm_pad_Uout)):
    PSDUatm[i] = mps.get_PSD(atm_pad_Uout[i],delta)
PSDUatm[:] += PSDUatm[:,::-1]
PSDUatm /= 2.

f=(np.arange((N + 2* atm_pad)//2-1) + 1)/(N + 2* atm_pad)/delta
A = np.mean(PSDUatm[:,(N + 2* atm_pad)//2+1:],axis=0)
Ae = np.std(PSDUatm[:,(N + 2* atm_pad)//2+1:],axis=0)/np.sqrt((N + 2* atm_pad))
B = np.mean(PSDUprop[:,(N + 2* atm_pad)//2+1:],axis=0)
Be = np.std(PSDUprop[:,(N + 2* atm_pad)//2+1:],axis=0)/np.sqrt((N + 2* atm_pad))

plt.plot(f,A,color='r',label='atm pad')
plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
plt.plot(f,B,color='b',label='prop pad')
plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
plt.legend()
plt.ylim(B.min()*0.9,B.max()*1.1)
plt.title('psd U' + '\n' + TAG)
plt.xlabel('f (1/m)')
plt.ylabel('PSD_U (normI m$^2$)')
plt.xscale('log')
plt.yscale('log')
plt.show()



PSDphaseatm = np.zeros(((N + 2* atm_pad),(N + 2* atm_pad)),dtype=float)
for i in np.arange(len(phase_atm_pad)):
    PSDphaseatm[i] = mps.get_PSD(phase_atm_pad[i],delta)
PSDphaseatm[:] += PSDphaseatm[:,::-1]
PSDphaseatm /= 2.

f=(np.arange((N + 2* atm_pad)//2-1) + 1)/(N + 2* atm_pad)/delta
A = np.mean(PSDphaseatm[:,(N + 2* atm_pad)//2+1:],axis=0)
Ae = np.std(PSDphaseatm[:,(N + 2* atm_pad)//2+1:],axis=0)/np.sqrt((N + 2* atm_pad))
B = np.mean(PSDphaseprop[:,(N + 2* atm_pad)//2+1:],axis=0)
Be = np.std(PSDphaseprop[:,(N + 2* atm_pad)//2+1:],axis=0)/np.sqrt((N + 2* atm_pad))

plt.plot(f,A,color='r',label='atm pad')
plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
plt.plot(f,B,color='b',label='prop pad')
plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
plt.legend()
plt.ylim(B.min()*0.9,B.max()*1.1)
plt.title('psd phase' + '\n' + TAG)
plt.xlabel('f (1/m)')
plt.ylabel('PSD_$\phi$ (rad$^2$ m$^2$)')
plt.xscale('log')
plt.yscale('log')
plt.show()



PSDintensityatm = np.zeros(((N + 2* atm_pad),(N + 2* atm_pad)),dtype=float)
for i in np.arange(len(intensity_atm_pad)):
    PSDintensityatm[i] = mps.get_PSD(intensity_atm_pad[i],delta)
PSDintensityatm[:] += PSDintensityatm[:,::-1]
PSDintensityatm /= 2.

f = (np.arange((N + 2* atm_pad)//2-1) + 1)/(N + 2* atm_pad)/delta
A = np.mean(PSDintensityatm[:,(N + 2* atm_pad)//2+1:],axis=0)
Ae = np.std(PSDintensityatm[:,(N + 2* atm_pad)//2+1:],axis=0)/np.sqrt((N + 2* atm_pad))
B = np.mean(PSDintensityprop[:,(N + 2* atm_pad)//2+1:],axis=0)
Be = np.std(PSDintensityprop[:,(N + 2* atm_pad)//2+1:],axis=0)/np.sqrt((N + 2* atm_pad))

plt.plot(f,A,color='r',label='atm pad')
plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
plt.plot(f,B,color='b',label='prop pad')
plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
plt.vlines([1/np.sqrt(wvl*z)],B.min(),B.max(),label='1/$\sqrt{\lambda z}$',color='k')
plt.legend()
plt.ylim(B.min()*0.9,B.max()*1.1)
plt.title('psd intensity' + '\n' + TAG)
plt.xlabel('f (1/m)')
plt.ylabel('PSD_I (normI m$^2$)')
plt.xscale('log')
plt.yscale('log')
plt.show()



SFphaseatm = np.zeros(((N + 2* atm_pad),(N + 2* atm_pad)), dtype=float)
# for i in np.arange(N):
#     SFphaseatm[i] = mps.get_structure_function(np.roll(phase_atm_pad,(i),axis=(0)))
# SFphaseatm = (phase_atm_pad[:,0] - phase_atm_pad[:,:])**2
for i in np.arange((N + 2* atm_pad)):
    SFphaseatm[i] = np.angle((atm_pad_Uout[i,i] / np.roll(atm_pad_Uout[i],(-i),axis=(0))))**2


x = np.arange((N + 2* atm_pad)//2)*delta
A = np.mean(SFphaseatm[:,:(N + 2* atm_pad)//2],axis=0)
Ae = np.std(SFphaseatm[:,:(N + 2* atm_pad)//2],axis=0)/np.sqrt((N + 2* atm_pad))
B = np.mean(SFphaseprop[:,:(N + 2* atm_pad)//2],axis=0)
Be = np.std(SFphaseprop[:,:(N + 2* atm_pad)//2],axis=0)/np.sqrt((N + 2* atm_pad))

plt.plot(x,A,color='r',label='atm pad')
plt.fill_between(x,A-Ae,A+Ae,alpha=0.2,color='r')
plt.plot(x,B,color='b',label='prop pad')
plt.fill_between(x,B-Be,B+Be,alpha=0.2,color='b')
plt.legend()
plt.ylim(B[1:].min()*0.9,B.max()*1.1)
plt.title('structure function $\phi$' + '\n' + TAG)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('separation (m)')
plt.ylabel('$\sigma^2_\phi$ (rad$^2$)')
plt.show()



SFintensityatm = np.zeros(((N + 2* atm_pad),(N + 2* atm_pad)), dtype=float)
# for i in np.arange(N):
#     SFintensityatm[i] = mps.get_structure_function(np.roll(intensity_atm_pad,(i),axis=(0)))
# SFintensityatm = (intensity_atm_pad[:,0] - intensity_atm_pad[:,:])**2
for i in np.arange((N + 2* atm_pad)):
    SFintensityatm[i] = (intensity_atm_pad[i,i] - np.roll(intensity_atm_pad[i],(-i),axis=(0)))**2

x = np.arange((N + 2* atm_pad)//2)*delta
A = np.mean(SFintensityatm[:,:(N + 2* atm_pad)//2],axis=0)
Ae = np.std(SFintensityatm[:,:(N + 2* atm_pad)//2],axis=0)/np.sqrt((N + 2* atm_pad))
B = np.mean(SFintensityprop[:,:(N + 2* atm_pad)//2],axis=0)
Be = np.std(SFintensityprop[:,:(N + 2* atm_pad)//2],axis=0)/np.sqrt((N + 2* atm_pad))

plt.plot(x,A,color='r',label='atm pad')
plt.fill_between(x,A-Ae,A+Ae,alpha=0.2,color='r')
plt.plot(x,B,color='b',label='prop pad')
plt.fill_between(x,B-Be,B+Be,alpha=0.2,color='b')
plt.vlines([np.sqrt(wvl*z)],
            B.min(),B.max(),label='$\sqrt{\lambda z}$',color='k')
plt.legend()
plt.title('structure function I' + '\n' + TAG)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('separation (m)')
plt.ylabel('$\sigma^2_I$ (normI$^2$)')
plt.show()

