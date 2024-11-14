# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:31:48 2024

@author: gqmc62
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov
from aotools.opticalpropagation import angularSpectrum


N = 1024
n = 64
delta = 0.05/8
wvl = 500e-9
wvl2 = 1500e-9
L0 = 10
r0 = 0.05
z = 5e3

PERIOD = 16

# U_uniform = np.ones((n,n),dtype=float)
# for t in np.arange(PERIOD):
#     temp = U_uniform*np.exp(1j*t/PERIOD*2.*np.pi)
#     plt.quiver(np.arange(n),np.arange(n),temp.real,temp.imag,scale=30)
#     plt.axis('square')
#     plt.show()

IPS = PhaseScreenKolmogorov(N, delta, r0, L0)
for i in np.arange(N):
    IPS.add_row()
    
phase_0_unwrapped = IPS.scrn[(N-n)//2:(N+n)//2,(N-n)//2:(N+n)//2]
phase_0_unwrapped -= phase_0_unwrapped.mean()
# plt.imshow(phase_0_unwrapped)
# plt.colorbar()
# plt.show()

phase_0_unwrapped15 = IPS.scrn[(N-n)//2:(N+n)//2,(N-n)//2:(N+n)//2]*wvl/wvl2
phase_0_unwrapped15 -= phase_0_unwrapped15.mean()
# plt.imshow(phase_0_unwrapped15)
# plt.colorbar()
# plt.show()

phase_0_full = IPS.scrn
phase_0_full -= phase_0_full.mean()

phase_0_full15 = IPS.scrn*wvl/wvl2
phase_0_full15 -= phase_0_full15.mean()

U_0_full = np.exp(1j*phase_0_full)
U_0 = np.exp(1j*phase_0_unwrapped)

U_0_full15 = np.exp(1j*phase_0_full15)
U_015 = np.exp(1j*phase_0_unwrapped15)

phase_0_wrapped = np.angle(U_0)
phase_0_wrapped15 = np.angle(U_015)

plt.imshow(phase_0_unwrapped*wvl/1e-9/2/np.pi)
plt.colorbar()
plt.show()

plt.imshow(phase_0_wrapped)
plt.colorbar()
plt.show()

plt.imshow(phase_0_wrapped15)
plt.colorbar()
plt.show()

x = np.arange(n)
xx,yy = np.meshgrid(x,x)
# for t in np.arange(PERIOD):
#     plt.pcolor(xx,yy,phase_0_wrapped.T,alpha=0.75)
#     plt.colorbar()
#     temp = U_0*np.exp(1j*t/PERIOD*2.*np.pi)
#     plt.quiver(xx,yy,temp.real,temp.imag,color='k')
#     # plt.contour(phase_0_wrapped,levels=[0])
#     plt.axis('square')
#     plt.show()

# for t in np.arange(PERIOD):
#     temp = U_015*np.exp(1j*t/PERIOD*2.*np.pi)
#     plt.quiver(np.arange(n),np.arange(n),temp.real,temp.imag)
#     # plt.contour(phase_0_wrapped,levels=[0])
#     plt.axis('square')
#     plt.show()

z0 = np.copy(z)
# angular prop doesn't include the oscillation with wavelength kinda thing... just modulate through 'time'
for t in np.arange(1):
    dz = t/PERIOD*wvl
    z_tot = z0 + dz

    pad = wvl*z_tot/2/delta**2 + wvl/r0*z_tot*4
    cmap = plt.colormaps['viridis']#["twilight"]
    min_size = pad*2 + n
    if N < min_size:
        print('larger phase screen please')
        rec_min_size = 2**int(np.ceil(np.log(min_size)/np.log(2)))
        print('suggesting N = {:} pixels'.format(rec_min_size))
    else:
        U_0_pad = np.pad(U_0_full,((N//2,N//2),(N//2,N//2)),mode='symmetric')
        U_z_raw = angularSpectrum(U_0_pad, wvl, delta, delta, z_tot)
        accurate_size = int(np.floor((N - 2*pad)/2)*2)
        edge_a = int((2*N - accurate_size)//2)
        edge_b = int((2*N + accurate_size)//2)
        U_z_accurate = U_z_raw[edge_a:edge_b,edge_a:edge_b]
        edge_a_2 = int((accurate_size - n)//2)
        edge_b_2 = int((accurate_size + n)//2)
        U_z = U_z_accurate[edge_a_2:edge_b_2,edge_a_2:edge_b_2]
        
        phase_z = np.angle(U_z)
        plt.pcolor(xx,yy,phase_z,cmap=cmap)
        plt.colorbar()
        plt.axis('square')
        plt.show()
        intensity_z = np.abs(U_z)**2
        intensity_z /= intensity_z.mean()
        plt.imshow(intensity_z)
        plt.colorbar()
        plt.show()
        
        scaled_phase = (np.pi+phase_z.flatten())/2/np.pi
        scaled_intensity = np.digitize(intensity_z,np.arange(0,8)).flatten()
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for i in np.arange(xx.flatten().size):
            ax1.plot(xx.flatten()[i],
                     yy.flatten()[i],
                     color=cmap(scaled_phase[i]),
                     markersize=scaled_intensity[i],
                     ls='',marker='o')
        plt.axis('square')
        plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(-np.pi,np.pi),cmap=cmap),ax=ax1)
        plt.show()
        
    
    for t in np.arange(PERIOD):
        temp = U_z*np.exp(1j*t/PERIOD*2.*np.pi)
        plt.imshow(np.angle(temp),cmap='viridis')
        # plt.quiver(np.arange(n),np.arange(n),temp.real,temp.imag)
        # plt.contour(phase_z,levels=[0])
        # plt.axis('square')
        plt.show()
    
    # MAX = np.max(intensity_z**0.5)
    # for t in np.arange(PERIOD):
    #     temp = (U_z_accurate*np.exp(1j*t/PERIOD*2.*np.pi)).real
    #     plt.imshow(temp,vmin=-MAX,vmax=MAX,cmap='RdBu')
    #     plt.colorbar()
    #     # plt.contour(phase_z,levels=[0])
    #     # plt.axis('square')
    #     plt.show()
    
    
# pad2 = wvl2*z/2/delta**2

# min_size2 = pad2*2 + n
# if N < min_size2:
#     print('larger phase screen please')
#     rec_min_size2 = 2**int(np.ceil(np.log(min_size2)/np.log(2)))
#     print('suggesting N = {:} pixels'.format(rec_min_size2))
# else:
#     U_0_pad2 = np.pad(U_0_full15,((N//2,N//2),(N//2,N//2)),mode='symmetric')
#     U_z_raw2 = angularSpectrum(U_0_pad2, wvl2, delta, delta, z)
#     accurate_size2 = int(np.floor((N - 2*pad2)/2)*2)
#     edge_a2 = int((2*N - accurate_size2)//2)
#     edge_b2 = int((2*N + accurate_size2)//2)
#     U_z_accurate2 = U_z_raw2[edge_a2:edge_b2,edge_a2:edge_b2]
#     edge_a_22 = int((accurate_size2 - n)//2)
#     edge_b_22 = int((accurate_size2 + n)//2)
#     U_z2 = U_z_accurate2[edge_a_22:edge_b_22,edge_a_22:edge_b_22]
    
#     phase_z2 = np.angle(U_z2)
#     plt.imshow(phase_z2)
#     plt.colorbar()
#     plt.show()
#     intensity_z2 = np.abs(U_z2)**2
#     intensity_z2 /= intensity_z2.mean()
#     plt.imshow(intensity_z2)
#     plt.colorbar()
#     plt.show()