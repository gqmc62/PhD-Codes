# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 10:27:55 2023

@author: gqmc62
"""

import os
import soapy
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from scipy.optimize import lsq_linear
from scipy.stats import bootstrap
from tqdm import tqdm

def remove_piston_tiptilt(scrns):
    yy, xx = np.meshgrid(np.arange(len(scrns)),np.arange(len(scrns)))
    norm = np.sum(xx, axis=(0,1))
    temp = scrns - np.mean(scrns, axis=(0,1))
    tip = np.sum(temp*yy, axis=(0,1))/norm**2
    tilt = np.sum(temp*xx, axis=(0,1))/norm**2
    temp -= (tip*yy + tilt*xx)
    return temp

class Clean():
    def __init__(self, N):
        self.yy, self.xx = np.meshgrid(np.arange(N, dtype=float), np.arange(N, dtype=float))
        norm = np.sum(self.xx, axis=(0,1))
        self.xx /= norm**2
        self.yy /= norm**2
        del norm
        
    def clean(self, scrn):
        clean_scrn = scrn - np.mean(scrn, axis=(0,1))
        tip = np.dot(clean_scrn.flatten(), (self.xx).flatten())
        tilt = np.dot(clean_scrn.flatten(), (self.yy).flatten())
        clean_scrn -= (self.xx*tip + self.yy*tilt)
        return clean_scrn

# def diagonal_stack(A):

#     N = A.shape[0]
#     stack_A = np.zeros(2*N - 1, dtype=type(A[0,0]))
#     error_stack_A = np.zeros(2*N - 1, dtype=type(A[0,0]))

#     for i in np.arange(N):
#         stack_A[N - 1 - i : N - 1] = stack_A[N - 1 - i : N - 1] + A[: i, i]
#         stack_A[N - 1 : 2*N - 1 - i] = stack_A[N - 1 : 2*N - 1 - i] + A[i :, i]
#     for i in np.arange(N - 1) + 1: # excluding an odd edge
#         stack_A[i] = stack_A[i]/(i + 1)
#         stack_A[2*N - i - 1] = stack_A[2*N - i - 1] / (i)
#     for i in np.arange(N):
#         error_stack_A[N - 1 - i
#                       : N - 1] = (error_stack_A[N - 1 - i : N - 1]
#                                   + (A[: i, i]
#                                      - stack_A[N - 1 - i : N - 1])**2)
#         error_stack_A[N - 1
#                       : 2*N - 1 - i] = (error_stack_A[N - 1 : 2*N - 1 - i]
#                                         + (A[i :, i]
#                                            - stack_A[N - 1 : 2*N - 1 - i])**2)
#     for i in np.arange(N - 2) + 2: 
#         error_stack_A[i] = error_stack_A[i]/(i)
#         error_stack_A[2*N - i - 1] = error_stack_A[2*N - i - 1]/(i)
#     error_stack_A = np.sqrt(error_stack_A)
#     error_stack_A[0] = np.NaN # the n=1 get np.NaN
#     error_stack_A[-1] = np.NaN
#     return stack_A[1 : -1], error_stack_A[1 : -1]

# def zero_pad2(a):
#     return np.pad(a, (int(a.shape[0]/2), int(a.shape[1]/2)),
#                   'constant', constant_values=0)

# def mirror_pad2(a):
#     return np.pad(a,(int(a.shape[0]/2),int(a.shape[1]/2)),'symmetric')

# def pick_even_index(AA):
#     return AA[tuple(np.meshgrid(np.arange(0,AA.shape[0],2),
#                               np.arange(0,AA.shape[1],2)))]

# def calc_cn2(c,a):
#     cc = mirror_pad2(c)
#     aa = mirror_pad2(a)
#     fc = pick_even_index(f_tran2(cc))
#     fa = pick_even_index(f_tran2(aa))
#     temp = fc/(fa+1e-12)
#     return pick_even_index(np.fft.ifftshift(np.fft.ifft2(mirror_pad2(temp))))

# def f_tran2(a):
#     return np.fft.fftshift(np.fft.fft2(a))

def rearrange1(A, position):
    """
    rearrange soapy reported wfs value from 1d with skips into 2d

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    position : TYPE
        DESCRIPTION.

    Returns
    -------
    a : TYPE
        DESCRIPTION.

    """
    N = position.shape[0]
    size = np.max(position) - np.min(position) + 1
    DIM = position.shape[1]
    a = np.zeros((size,)*DIM, dtype=float)
    for n in np.arange(N):
        a[tuple(position[n])] = A[n]
    return a

# def overlay1(A,eA,position):
#     a = np.zeros((2*position.max() + 1, 2*position.max() + 1), dtype=float)
#     ae = np.zeros((2*position.max() + 1, 2*position.max() + 1), dtype=float)
#     count = np.zeros((2*position.max() + 1, 2*position.max() + 1), dtype=float)
#     N = len(A)
#     mid = position.max()
#     for ii in np.arange(N):
#         for jj in np.arange(N):
#             xcoord = mid + (position[jj, 0] - position[ii, 0])
#             ycoord = mid + (position[jj, 1] - position[ii, 1])
#             a[xcoord, ycoord] += A[ii, jj]/eA[ii,jj]**2
#             ae[xcoord, ycoord] += 1./eA[ii, jj]**2
#             count[xcoord, ycoord] += 1
#     a /= ae**2
#     ae = 1./np.sqrt(ae)
#     return a, ae

def overlay1(A,position):
    a = np.zeros((2*position.max() + 1, 2*position.max() + 1), dtype=float)
    ae = np.zeros((2*position.max() + 1, 2*position.max() + 1), dtype=float)
    count = np.zeros((2*position.max() + 1, 2*position.max() + 1), dtype=float)
    N = len(A)
    mid = position.max()
    for ii in np.arange(N):
        for jj in np.arange(N):
            xcoord = mid + (position[jj, 0] - position[ii, 0])
            ycoord = mid + (position[jj, 1] - position[ii, 1])
            a[xcoord, ycoord] += A[ii, jj]
            count[xcoord, ycoord] += 1
    a /= count
    for ii in np.arange(N):
        for jj in np.arange(N):
            xcoord = mid + (position[jj, 0] - position[ii, 0])
            ycoord = mid + (position[jj, 1] - position[ii, 1])
            ae[xcoord, ycoord] += (A[ii, jj] - a[xcoord, ycoord])**2
    ae /= count
    ae = np.sqrt(ae)
    return a, ae

# def overlay1(A,position):
#     a = np.zeros((2*position.max() + 1, 2*position.max() + 1), dtype=float)
#     count = np.zeros((2*position.max() + 1, 2*position.max() + 1), dtype=float)
#     N = len(A)
#     mid = position.max()
#     for ii in np.arange(N):
#         for jj in np.arange(N):
#             xcoord = mid + (position[jj, 0] - position[ii, 0])
#             ycoord = mid + (position[jj, 1] - position[ii, 1])
#             a[xcoord, ycoord] += A[ii, jj]
#             count[xcoord, ycoord] += 1
#     a /= count
#     return a

# def my_cov(A, B):
#     N = A.shape[0]
#     C = np.zeros((N, N), dtype=float)
#     eC = np.zeros((N, N), dtype=float)
#     for ii in np.arange(N):
#         for jj in np.arange(N):
#             temp = (A[ii] - A[ii].mean())*(B[jj] - B[jj].mean())
#             C[ii, jj] = np.mean(temp)
#             res = bootstrap((temp,),np.std)
#             eC[ii, jj] = res.standard_error
#             # eC[ii, jj] = np.std(temp,ddof=1)
#     return C, eC

def my_cov(A, B):
    N = A.shape[0]
    C = np.zeros((N, N), dtype=float)
    for ii in np.arange(N):
        for jj in np.arange(N):
            temp = (A[ii] - A[ii].mean())*(B[jj] - B[jj].mean())
            C[ii, jj] = np.mean(temp)
    return C

def r02cn2(r0,wvl=500e-9,zenith_angle=0.):
    cn2 = r0**(-5./3.)/(0.423*(2.*np.pi/wvl)**2/np.cos(zenith_angle))
    return cn2

def cn22r0(cn2,wvl=500e-9,zenith_angle=0.):
    r0 = (0.423*(2.*np.pi/wvl)**2/np.cos(zenith_angle)*cn2)**(-3./5.)
    return r0

#
# set up test
#

ARCSEC2RAD = 1./3600./360.*2.*np.pi
RAD2ARCSEC = 1./ARCSEC2RAD

home = ('C:/Users/gqmc62/OneDrive - Durham University/Desktop/'
             + 'soapy-master/soapy-master/conf')
os.chdir(home)

file = 'slope_covariance_most_correct.yaml'
sim = soapy.Sim(file)

sim.aoinit()
sim.config.calcParams()
sim.makeIMat()

# d=(sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)
# print('subapDiam={:.6f}m\n'.format(d))

# plt.plot(sim.wfss[0].pupil_subap_coords[:,0],
#          sim.wfss[0].pupil_subap_coords[:,1],
#          ls='',marker='o')
# plt.axis('square')
# plt.title('pupil')
# plt.show()

N = sim.wfss[0].n_subaps

# if sim.wfss[1].n_subaps != N:
#     print('error')

# if sim.config.wfss[0].GSPosition[0] - sim.config.wfss[1].GSPosition[0] == 0:
#     print('wfs should be pointing differently on x direction')

# if sim.config.wfss[0].GSPosition[1] - sim.config.wfss[1].GSPosition[1] != 0:
#     print('wfs should be pointing similarly on y direction')

# nb = input('Please configure reference slope covariance\nProceed?\nType y to proceed: ')
# if nb != 'y':
#     quit()



# scrnHeights_list = np.array([10000], dtype=float)
# scrnHeights_list = np.array([0,5,10,15,20,25,30], dtype=float)*1e3
# scrnHeights_list = np.array([0], dtype=float)

scrnHeights_list = (np.arange(sim.config.wfss[0].nxSubaps - 2, dtype=float)
                    * (sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)
                    / ((sim.config.wfss[0].GSPosition[0]
                        - sim.config.wfss[1].GSPosition[0])*ARCSEC2RAD))
LAYER = scrnHeights_list.shape[0]

position = np.asarray(sim.wfss[0].pupil_subap_coords/sim.wfss[0].nx_subap_pixels,dtype=int) - 1

mid = sim.config.wfss[0].nxSubaps - 1 - 1 - 1
# mid = position.max()

grid = np.arange(mid+1,dtype=float)
grid -= len(grid)/2
grid += 0.5

#
# measure Cross covariance / auto covariance of turbulence at different height
#

C = np.zeros((LAYER,2,N,N),dtype=float) # xy, wfs1subap, wfs2subap

for layer in np.arange(0):#LAYER):
    print('at layer {:d}'.format(layer))
    scrnHeights = scrnHeights_list[layer]
    # sim.config.atmos.windDirs = [windDirs]
    sim.config.atmos.scrnHeights = [scrnHeights]
    sim.aoinit()
    sim.config.calcParams()
    sim.makeIMat()
    
    # sim.config.dms[0].gain = 0.
    # sim.config.dms[1].gain = 0.
    
    sim.aoloop()
    
    Sx1 = sim.allSlopes[ : , 0*N : (0+1)*N]
    Sy1 = sim.allSlopes[ : , 1*N : (1+1)*N]
    Sx2 = sim.allSlopes[ : , 2*N : (2+1)*N]
    Sy2 = sim.allSlopes[ : , 3*N : (3+1)*N]
    
    TIME = Sx1.shape[0]
    for time in np.arange(TIME):
        Sx1[time] -= np.mean(Sx1[time])
        Sy1[time] -= np.mean(Sy1[time])
        Sx2[time] -= np.mean(Sx2[time])
        Sy2[time] -= np.mean(Sy2[time])

for scrns in [Sx1,Sx2,Sy1,Sy2]:
    convergence = np.zeros((4), dtype=float)
    ref = np.mean(np.std(scrns,axis=(1)))
    for i in np.arange(4):
        val = int(10**(i + 1))
        convergence[i] = np.std(np.mean(scrns[:val],axis=0), ddof=1)
    
    plt.plot([10,100,1000,10000],convergence/ref,marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('number of unique slope screens')
    plt.ylabel('std(mean(screens)) / mean(std(screens))')
    plt.title('roughness of averaged screens normalized to averaged roughness of screens')
    plt.show()


