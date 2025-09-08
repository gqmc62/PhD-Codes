# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 10:54:50 2025

@author: gqmc62
"""
##################################################################################
# ░▒█▀▀█░█░▒█░▀█▀░▀█▀░░▀░░█░░░█░█▀▀▄░▀█▀░░░▒█░▄▀░▄▀▀▄░█▀▀▄░█▀▀▀░█░▄░█▀▀▄░█▀▀░█░░░█
# ░▒█▄▄█░█░▒█░░█░░░█░░░█▀░▀▄█▄▀░█▄▄█░░█░░░░▒█▀▄░░█░░█░█░▒█░█░▀▄░█▀▄░█▄▄█░█▀▀░▀▄█▄▀
# ░▒█░░░░░▀▀▀░░▀░░░▀░░▀▀▀░░▀░▀░░▀░░▀░░▀░░░░▒█░▒█░░▀▀░░▀░░▀░▀▀▀▀░▀░▀░▀░░▀░▀▀▀░░▀░▀░
##################################################################################

###############################################################################
# import
###############################################################################

import numpy as np
from matplotlib import pyplot as plt
from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov
from aotools.opticalpropagation import angularSpectrum

###############################################################################
# declare variable
###############################################################################

D1 = 4 # pupil size at ground (m)
delta1 = 1e-2 # pixel size at ground (m)
wvl = 500e-9 # wavelength (m)
r0 = 5e-2 # turbulence fried parameter (m)
L0 = 20 # turbulence outer scale (m)
coverage_factor = 1 # 1==>90%, 2==>97%, 4==>99% # see c pg173 in schmidt 2010 numerical simulation of optical propagation
z_list = np.logspace(3,5,num=5)#np.array([1e3,1e4,1e5],dtype=float) # list of propagation distance to test (m)

###############################################################################
# common parameters
###############################################################################

zr = np.pi*D1**2/wvl # rayleigh range

###############################################################################
# at ground
###############################################################################

n1 = int(np.ceil(D1/delta1)) # number of pixel in pupil
if n1 % 2 == 1: # i want only even number. ... or else my simulation usualy die. lol
    n1 += 1
    
PS = PhaseScreenKolmogorov(n1, delta1, r0, L0) # phase screen

plt.pcolor(np.arange(n1)*delta1,
           np.arange(n1)*delta1,
           np.angle(np.exp(1j*PS.scrn)))
plt.axis('square')
plt.title('phase turbulence at groundd (rad)'
          + '\nTotal intensity {:.3f}'.format((
              np.sum(np.abs(np.exp(1j*PS.scrn))**2)*delta1**2)))
plt.xlabel('m')
plt.ylabel('m')
plt.colorbar()
plt.show()


for z in z_list:
    
    
    
    # the beam diameter to show is the max of either
    wz = np.max([D1*np.sqrt(1+(z/zr)**2), # gaussian beam expansion (m))
                  D1 + coverage_factor*wvl*z/r0]) # turbulence expansion (m)
    # wz = D1
    
    # the mathematically correct region either the
    D2 = np.max([D1 + wvl*z/delta1, # mathematically expansion of angular propagation (m)
                 wz]) # or the beam size that expand (m) mostly gonna be the top one though
    
    # the pixel scale at altitude is the plot we want to show while maintaining the same number of pixel with n1
    # lol n2=n1 calc delta2
    delta2 = wz/D1*delta1 # m
    n2 = int(np.ceil(wz/delta2)) # pixel
    if n2 % 2 == 1: # make even
        n2 += 1
    
    # number of pixel in a screen to be propagated is the largest one of either
    N = int(np.ceil(np.max([D1/delta1, # number of input pixel
                            D2/delta2, # number of accurate output pixel
                            wz/delta2, # number of desired output pixel
                            wvl*z/delta1/delta2, # anti aliasing
                            D1/2/delta1+D2/2/delta2+wvl*z/2/delta1/delta2]))) # no wrapping
    
    if N % 2 == 1: # make it even
        N += 1
        
    Uin = np.pad(np.exp(1j*PS.scrn),((N-n1)//2,(N-n1)//2)) # complex amplitude at ground to be propagate
    Uout = angularSpectrum(Uin, wvl, delta1, delta2, z) # complex amplitude at altitude subject to cropping# the accurate region is the D2
    
    
    plt.pcolor(np.arange(n2)*delta2,
               np.arange(n2)*delta2,
               np.angle(Uout[(N-n2)//2:-(N-n2)//2,
                             (N-n2)//2:-(N-n2)//2]))
    plt.axis('square')
    plt.colorbar()
    plt.title('phase turbulence at groundd (rad)'
              + '\nTotal intensity {:.3f}'.format((
                  np.sum(np.abs(Uout[(N-n2)//2:-(N-n2)//2,
                                     (N-n2)//2:-(N-n2)//2])**2)*delta2**2)))
    plt.xlabel('m')
    plt.ylabel('m')
    plt.show()
    
    plt.pcolor(np.arange(n2)*delta2,
               np.arange(n2)*delta2,
               np.abs(Uout[(N-n2)//2:-(N-n2)//2,
                           (N-n2)//2:-(N-n2)//2])**2)
    plt.axis('square')
    plt.colorbar()
    plt.title('intensity turbulence at groundd (rad)'
              + '\nTotal intensity {:.3f}'.format((
                  np.sum(np.abs(Uout[(N-n2)//2:-(N-n2)//2,
                                     (N-n2)//2:-(N-n2)//2])**2)*delta2**2)))
    plt.xlabel('m')
    plt.ylabel('m')
    plt.show()
    
    
    