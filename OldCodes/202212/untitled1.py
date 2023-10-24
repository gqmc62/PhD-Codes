# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:31:03 2022

@author: gqmc62
"""

import numpy as np
from matplotlib import pyplot as plt
from aotools.turbulence.phasescreen import ft_sh_phase_screen
import time
from aotools.opticalpropagation import angularSpectrum
from aotools.functions.zernike import zernikeArray
import my_propagation as mprop
import my_WFS as mwfs
import my_phasescreen as mps

N = 1024
pf = 2
r0 = 0.05
z_list = [1e3,2e3,5e3,10e3,25e3,50e3,100e3]
delta = 5e-3
L0 = 20.
l0 = 1.5e-10
wvl = 500e-9
lenslet_pixel=16

NN = N*pf

# Propagation between atmospheric layers
check = True
check_condition = mprop.prop_param_check(N, delta, delta, wvl, z_list, pf,
                                        scin_blob=False)#True)#True

phase_in_big = np.zeros((NN,NN),dtype=float)
I_in_big = np.ones((NN,NN),dtype=float)
#x = np.arange(-NN/2,NN/2)
#xx,yy = np.meshgrid(x,x)
#I_in_big = I_in_big*(xx**2 + yy**2 < N**2/4)
U_in_big = np.zeros((NN,NN),dtype=complex)

phase_in_big = ft_sh_phase_screen(r0, NN, delta, L0, l0, seed=99)
U_in_big = np.sqrt(I_in_big)*np.exp(1j*phase_in_big)

phase_in = mprop.crop2(phase_in_big, pf)
phase_in_wrap = np.angle(mprop.crop2(U_in_big, pf))
I_in = mprop.crop2(I_in_big, pf)

# PSD = mps.psd_function(r0, N, delta, L0, l0)
# PSD_in = mps.pick_even_index(mps.get_PSD(mps.mirror_pad2(phase_in), delta))

mprop.prop_test(N,delta,wvl,delta/2.,[1000,2000,5000,10000,25000,50000,100000,200000],pf)
mprop.prop_test2(N,delta,wvl,delta/2.,[1000,2000,5000,10000,25000,50000,100000,200000],pf)
for z in z_list:
    
    U_out_big = angularSpectrum(U_in_big, wvl, delta, delta, z)
    
    phase_out_wrap = np.angle(mprop.crop2(U_out_big, pf))
    I_out = np.abs(mprop.crop2(U_out_big, pf))**2
    phase_out = np.unwrap(np.unwrap(phase_out_wrap, axis=-2), axis=-1)
    
    plt.imshow(phase_in[:1024,:1024])
    plt.title('in')
    plt.show()
    plt.imshow(phase_out[:1024,:1024])
    plt.title('out')
    plt.show()
    # plt.imshow(phase_in[:128,:128])
    # plt.title('in')
    # plt.show()
    # plt.imshow(phase_out[:128,:128])
    # plt.title('out')
    # plt.show()
    # plt.imshow(phase_in[:16,:16])
    # plt.title('in')
    # plt.show()
    # plt.imshow(phase_out[:16,:16])
    # plt.title('out')
    # plt.show()
    
    plt.imshow(I_in[:1024,:1024])
    plt.title('in')
    plt.show()
    plt.imshow(I_out[:1024,:1024])
    plt.title('out')
    plt.show()
    
    # phase_dif=phase_out-phase_in
    
    # PSD_out = mps.pick_even_index(mps.get_PSD(mps.mirror_pad2(phase_out), delta))
    # # PSD_dif = mps.pick_even_index(mps.get_PSD(mps.mirror_pad2(phase_dif), delta))
    
    # plt.plot((np.arange(int(N/2-1))+1)/(N*delta),
    #           PSD[int(N/2),int(N/2)+1:],
    #           ls='-', marker='',
    #           color='r', label='theoretical PSD')
    # plt.plot((np.arange(int(N/2-1))+1)/(N*delta),
    #           PSD_in[int(N/2),int(N/2)+1:],
    #           ls='', marker='.', alpha=0.25,
    #           color='g', label='PSD altitude')
    # plt.plot((np.arange(int(N/2)-1)+1)/(N*delta),
    #           PSD_out[int(N/2),int(N/2)+1:],
    #           ls='', marker='.', alpha=0.25,
    #           color='b', label='PSD ground')
    # # plt.plot((np.arange(int(N/2-1))+1)/(N*delta),
    # #           PSD_dif[int(N/2),int(N/2)+1:],
    # #           ls='', marker='<', alpha=0.25,
    # #           color='c', label='PSD of phase difference')
    # plt.vlines(1./(2.*lenslet_pixel*delta),1e-10,1e2,
    #             label='lenslet detectable',color='m')
    # plt.vlines(1./(2.*np.sqrt(wvl*z)),1e-10,1e2,
    #             label='1st Fresnel zone size',color='k')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Spatial frequency (1/m)')
    # plt.ylabel('PSD (m^2)')
    # plt.legend()
    # plt.title('Comparison of phase screen power spectrum density\n'
    #           + 'before and after propagation\n'
    #           + 'r={:.0f}cm, dz={:.0f}km, '.format(r0*100,z/1000)
    #           + 'accurate screen pixel={:d}px\n'.format(N)
    #           + 'pixel={:.0f}mm, L0={:.0f}m'.format(delta*1000,L0))
    # plt.show()
    
    # plt.plot((np.arange(int(N/2-1))+1)/(N*delta),
    #           PSD_out[int(N/2),int(N/2)+1:] / PSD_in[int(N/2),int(N/2)+1:],
    #           ls='', marker='.', alpha=0.25,
    #           color='b', label='PSD ground / PSD altitude')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Spatial frequency (1/m)')
    # plt.ylabel('PSD (m^2)')
    # plt.hlines(1,1e-1,1e2,label='equal',color='r')
    # plt.vlines(1./(2.*lenslet_pixel*delta),1e-3,1e3,
    #             label='lenslet detectable',color='m')
    # plt.vlines(1./(2.*np.sqrt(wvl*z)),1e-3,1e3,
    #             label='frequency of 1st Fresnel zone size',color='k')
    # plt.legend()
    # plt.title('Comparison of phase screen power spectrum density\n'
    #           + 'before - after propagation\n'
    #           + 'r={:.0f}cm, dz={:.0f}km, '.format(r0*100,z/1000)
    #           + 'accurate screen pixel={:d}px\n'.format(N)
    #           + 'pixel={:.0f}mm, L0={:.0f}m'.format(delta*1000,L0))
    # plt.show()
