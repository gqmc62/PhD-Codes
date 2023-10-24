# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:35:52 2023

@author: gqmc62
"""

import os
import soapy
import numpy as np
from matplotlib import pyplot as plt

ARCSEC2RAD = 1./3600./360.*2.*np.pi
RAD2ARCSEC = 1./ARCSEC2RAD

home = ('C:/Users/gqmc62/OneDrive - Durham University/Desktop/'
             + 'soapy-master/soapy-master/conf')
os.chdir(home)

file = 'slope_covariance2.yaml'
sim = soapy.Sim(file)

sim.aoinit()
sim.config.calcParams()
sim.makeIMat()

for n in np.arange(len(sim.config.dms)):
    sim.config.dms[n].gain = 0
    
sim.aoloop()


N = sim.wfss[0].n_subaps
if sim.wfss[1].n_subaps != N:
    print('error')

C = np.zeros((2,N,N),dtype=float) # xy, wfs1subap, wfs2subap
A1 = np.zeros((2,N,N),dtype=float) # autocovariance1
A2 = np.zeros((2,N,N),dtype=float) # autocovariance2

Sx1 = sim.allSlopes[ : , 0*N : (0+1)*N]
Sy1 = sim.allSlopes[ : , 1*N : (1+1)*N]
Sx2 = sim.allSlopes[ : , 2*N : (2+1)*N]
Sy2 = sim.allSlopes[ : , 3*N : (3+1)*N]

for ii in np.arange(N):
    for jj in np.arange(N):
        C[0,ii,jj] = np.mean(Sx1[:,ii]*Sx2[:,jj],axis=0)
        C[1,ii,jj] = np.mean(Sy1[:,ii]*Sy2[:,jj],axis=0)
        A1[0,ii,jj] = np.mean(Sx1[:,ii]*Sx1[:,jj],axis=0)
        A1[1,ii,jj] = np.mean(Sy1[:,ii]*Sy1[:,jj],axis=0)
        A2[0,ii,jj] = np.mean(Sx2[:,ii]*Sx2[:,jj],axis=0)
        A2[1,ii,jj] = np.mean(Sy2[:,ii]*Sy2[:,jj],axis=0)

A = (A1 + A2)/2

# plt.imshow(C[0])
# plt.title('slope x cross covariance')
# plt.show()

# plt.imshow(C[1])
# plt.title('slope y cross covariance')
# plt.show()

# plt.imshow(AC1[0])
# plt.title('slope x wfs1 auto covariance')
# plt.show()

# plt.imshow(AC1[1])
# plt.title('slope y wfs1 auto covariance')
# plt.show()    

# plt.imshow(AC2[0])
# plt.title('slope x wfs2 auto covariance')
# plt.show()

# plt.imshow(AC2[1])
# plt.title('slope y wfs2 auto covariance')
# plt.show()

# plt.imshow((AC1[0]+AC2[0])/2)
# plt.title('slope x auto covariance')
# plt.show()

# plt.imshow((AC1[1]+AC2[1])/2)
# plt.title('slope y auto covariance')
# plt.show()    

plt.imshow(np.abs(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(C[0]))/np.fft.fftshift(np.fft.fft2(A[0])))))
plt.title('Cn2 x slope')
plt.show()

plt.imshow(np.abs(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(C[1]))/np.fft.fftshift(np.fft.fft2(A[1])))))
plt.title('Cn2 y slope')
plt.show()