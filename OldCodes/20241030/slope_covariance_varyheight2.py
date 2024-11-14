# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:35:52 2023

@author: gqmc62
"""

import os
import soapy
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits

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

def overlay1(A,eA,position):
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
            ae[xcoord, ycoord] += eA[ii, jj]**2
            count[xcoord, ycoord] += 1
    a /= count
    ae /= count
    ae = np.sqrt(ae)
    return a, ae

def my_cov(A, B):
    N = A.shape[0]
    n = np.sqrt(A.shape[1])
    C = np.zeros((N, N), dtype=float)
    eC = np.zeros((N, N), dtype=float)
    for ii in np.arange(N):
        for jj in np.arange(N):
            temp = (A[ii] - A[ii].mean())*(B[jj] - B[jj].mean())
            C[ii, jj] = np.mean(temp)
            eC[ii, jj] = np.std(temp, ddof=1)/n
    return C, eC

#
# set up test
#

ARCSEC2RAD = 1./3600./360.*2.*np.pi
RAD2ARCSEC = 1./ARCSEC2RAD

home = ('C:/Users/gqmc62/OneDrive - Durham University/Desktop/'
             + 'soapy-master/soapy-master/conf')
os.chdir(home)

file = 'slope_covariance.yaml'
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
                    * (sim.config.tel.telDiam/(sim.config.wfss[0].nxSubaps))
                    / ((sim.config.wfss[0].GSPosition[0]
                        - sim.config.wfss[1].GSPosition[0])*ARCSEC2RAD))
LAYER = scrnHeights_list.shape[0]

position = np.asarray(sim.wfss[0].pupil_subap_coords/sim.wfss[0].nx_subap_pixels,dtype=int)

mid = (position.max() - position.min())

grid = np.arange(mid + 1,dtype=float)
grid -= len(grid)/2
grid += 0.5

#
# measure Cross covariance / auto covariance of turbulence at different height
#

C = np.zeros((LAYER,2,N,N),dtype=float) # xy, wfs1subap, wfs2subap
# A1 = np.zeros((LAYER,2,N,N),dtype=float) # autocovariance1
# A2 = np.zeros((LAYER,2,N,N),dtype=float) # autocovariance2
# A = np.zeros((LAYER,2,N,N),dtype=float) # autocovariance2

eC = np.zeros((LAYER,2,N,N),dtype=float) # xy, wfs1subap, wfs2subap
# eA1 = np.zeros((LAYER,2,N,N),dtype=float) # autocovariance1
# eA2 = np.zeros((LAYER,2,N,N),dtype=float) # autocovariance2
# eA = np.zeros((LAYER,2,N,N),dtype=float) # autocovariance2

for layer in np.arange(LAYER):
    scrnHeights = scrnHeights_list[layer]
    # sim.config.atmos.windDirs = [windDirs]
    sim.config.atmos.scrnHeights = [scrnHeights]
    sim.aoinit()
    sim.config.calcParams()
    sim.makeIMat()
    
    for n in np.arange(len(sim.config.dms)):
        sim.config.dms[n].gain = 0
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
    
    # # sanity check
    # for time in np.arange(np.min([TIME,10])):
    #     slopex1 = rearrange1(Sx1[time],position).T
    #     slopex2 = rearrange1(Sx2[time],position).T
    #     slopey1 = rearrange1(Sy1[time],position).T
    #     slopey2 = rearrange1(Sy2[time],position).T
    #     plt.quiver(grid,grid,slopex1,slopey1,color='r',label='wfs1')
    #     plt.quiver(grid,grid,slopex2,slopey2,color='b',label='wfs2')
    #     plt.axis('square')
    #     plt.xlabel('first index')
    #     plt.ylabel('second index')
    #     plt.title('{:d} {:d}'.format(layer,time))
    #     plt.legend()
    #     plt.show()
    
    C[layer,0], eC[layer,0] = my_cov(Sx1.T,Sx2.T)
    C[layer,1], eC[layer,1] = my_cov(Sy1.T,Sy2.T)
    # A1[layer,0], eA1[layer,0] = my_cov(Sx1.T,Sx1.T)
    # A1[layer,1], eA1[layer,1] = my_cov(Sy1.T,Sy1.T)
    # A2[layer,0], eA2[layer,0] = my_cov(Sx2.T,Sx2.T)
    # A2[layer,1], eA2[layer,1] = my_cov(Sy2.T,Sy2.T)

    # A[layer] = (A1[layer] + A2[layer])/2.
    # eA[layer] = eA1[layer] + eA2[layer]

# name = 'Aheight{:d}'.format(sim.config.sim.nIters)
# hdu = fits.PrimaryHDU(A)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()
name = 'Cheight{:d}'.format(sim.config.sim.nIters)
hdu = fits.PrimaryHDU(C)
hdul = fits.HDUList([hdu])
hdul.writeto(name + '.fits', overwrite=True)
hdul.close()
# name = 'eAheight{:d}'.format(sim.config.sim.nIters)
# hdu = fits.PrimaryHDU(eA)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()
name = 'eCheight{:d}'.format(sim.config.sim.nIters)
hdu = fits.PrimaryHDU(eC)
hdul = fits.HDUList([hdu])
hdul.writeto(name + '.fits', overwrite=True)
hdul.close()




# # name = 'Aheight{:d}'.format(sim.config.sim.nIters)
# # hdulist = fits.open(name + '.fits')
# # hdu = hdulist[0]
# # A = hdu.data
# # hdulist.close()
# name = 'Cheight{:d}'.format(sim.config.sim.nIters)
# hdulist = fits.open(name + '.fits')
# hdu = hdulist[0]
# C = hdu.data
# hdulist.close()
# # name = 'eAheight{:d}'.format(sim.config.sim.nIters)
# # hdulist = fits.open(name + '.fits')
# # hdu = hdulist[0]
# # eA = hdu.data
# # hdulist.close()
# name = 'eCheight{:d}'.format(sim.config.sim.nIters)
# hdulist = fits.open(name + '.fits')
# hdu = hdulist[0]
# eC = hdu.data
# hdulist.close()



# name = 'slopeCovariance'
# hdulist = fits.open(name + '.fits')
# hdu = hdulist[0]
# test = np.flip(np.swapaxes(hdu.data,1,2),axis=1)
# hdulist.close()
# test[test==0] = np.nan

name = 'referenceFunctions3'
hdulist = fits.open(name + '.fits')
hdu = hdulist[0]
ref = np.flip(hdu.data, axis=1)
hdulist.close()
ref[ref==0] = np.nan

refx = ref[::2]
refy = ref[1::2]

# refx = test[::2]
# refy = test[1::2]


# C_tot = (C[:,0,:,:] + C[:,1,:,:])/2.
# eC_tot = np.sqrt((eC[:,0,:,:]**2 + eC[:,1,:,:])/2.)


# for layer in np.arange(LAYER):
#     # print(layer,np.where(Cscreen[layer,0]==np.nanmax(Cscreen[layer,0])))
#     plt.pcolor(np.arange(C.shape[2]),np.arange(C.shape[3]),C[layer,0].T)
#     plt.axis('square')
#     plt.colorbar()
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('Slope cross-covariance (T) {:d}'.format(layer))
#     plt.show()
# for layer in np.arange(LAYER):
#     # print(layer,np.where(Cscreen[layer,1]==np.nanmax(Cscreen[layer,1])))
#     plt.pcolor(np.arange(C.shape[2]),np.arange(C.shape[3]),C[layer,1].T)
#     plt.axis('square')
#     plt.colorbar()
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('Slope cross-covariance (L) {:d}'.format(layer))
#     plt.show()
# # for layer in np.arange(LAYER):
# #     # print(layer,np.where(Ascreen[layer,0]==np.nanmax(Ascreen[layer,0])))
# #     plt.pcolor(np.arange(C.shape[2]),np.arange(C.shape[3]),A[layer,0].T)
# #     plt.axis('square')
# #     plt.colorbar()
# #     plt.xlabel('x = first index')
# #     plt.ylabel('y = second index')
# #     plt.title('Ax{:d}'.format(layer))
# #     plt.show()
# # for layer in np.arange(LAYER):
# #     # print(layer,np.where(Ascreen[layer,1]==np.nanmax(Ascreen[layer,1])))
# #     plt.pcolor(np.arange(C.shape[2]),np.arange(C.shape[3]),A[layer,1].T)
# #     plt.axis('square')
# #     plt.colorbar()
# #     plt.xlabel('x = first index')
# #     plt.ylabel('y = second index')
# #     plt.title('Ay{:d}'.format(layer))
# #     plt.show()



#
# generate covariance map on wfs pupil
#

biggrid = np.arange(2*mid + 1, dtype=float)
biggrid -= len(biggrid)/2
biggrid += 0.5    
    
# Ascreen = np.zeros((LAYER,2,2*mid + 1,2*mid + 1), dtype=float)
# Acount = np.zeros((LAYER,2,2*mid + 1,2*mid + 1), dtype=float)
Cscreen = np.zeros((LAYER,2,2*mid + 1,2*mid + 1), dtype=float)
Ccount = np.zeros((LAYER,2,2*mid + 1,2*mid + 1), dtype=float)
# Aescreen = np.zeros((LAYER,2,2*mid + 1,2*mid + 1), dtype=float)
# Aecount = np.zeros((LAYER,2,2*mid + 1,2*mid + 1), dtype=float)
Cescreen = np.zeros((LAYER,2,2*mid + 1,2*mid + 1), dtype=float)
Cecount = np.zeros((LAYER,2,2*mid + 1,2*mid + 1), dtype=float)
for layer in np.arange(LAYER):
    # Ascreen[layer, 0], Aescreen[layer, 0] = overlay1(A[layer, 0], eA[layer, 0], position)
    Cscreen[layer, 0], Cescreen[layer, 0] = overlay1(C[layer, 0], eC[layer, 0], position)
    # Ascreen[layer, 1], Aescreen[layer, 1] = overlay1(A[layer, 1], eA[layer, 1], position)
    Cscreen[layer, 1], Cescreen[layer, 1] = overlay1(C[layer, 1], eC[layer, 1], position)
    
# for layer in np.arange(LAYER):
#     # print(layer,np.where(Cscreen[layer,0]==np.nanmax(Cscreen[layer,0])))
#     plt.pcolor(biggrid,biggrid,Cscreen[layer,0].T)
#     plt.axis('square')
#     plt.colorbar()
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('Slope cross-covariance (T) {:d}'.format(layer))
#     plt.show()
# for layer in np.arange(LAYER):
#     # print(layer,np.where(Cscreen[layer,1]==np.nanmax(Cscreen[layer,1])))
#     plt.pcolor(biggrid,biggrid,Cscreen[layer,1].T)
#     plt.axis('square')
#     plt.colorbar()
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('Slope cross-covariance (L) {:d}'.format(layer))
#     plt.show()
    
# for layer in np.arange(LAYER):
#     # print(layer,np.where(Ascreen[layer,0]==np.nanmax(Ascreen[layer,0])))
#     plt.pcolor(biggrid,biggrid,Ascreen[layer,0].T)
#     plt.axis('square')
#     plt.colorbar()
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('Ax{:d}'.format(layer))
#     plt.show()
# for layer in np.arange(LAYER):
#     # print(layer,np.where(Ascreen[layer,1]==np.nanmax(Ascreen[layer,1])))
#     plt.pcolor(biggrid,biggrid,Ascreen[layer,1].T)
#     plt.axis('square')
#     plt.colorbar()
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('Ay{:d}'.format(layer))
#     plt.show()
    
# for i in np.arange(test.shape[0]):
#     plt.pcolor(biggrid,biggrid,test[i].T)
#     plt.axis('square')
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('slope covariance {:d}'.format(i))
#     plt.colorbar()
#     plt.show()   

# for i in np.arange(refx.shape[0]):
#     plt.pcolor(biggrid,biggrid,refx[i].T)
#     plt.axis('square')
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('x reference function {:d}'.format(i))
#     plt.colorbar()
#     plt.show()
# for i in np.arange(refy.shape[0]):
#     plt.pcolor(biggrid,biggrid,refy[i].T)
#     plt.axis('square')
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('y reference function {:d}'.format(i))
#     plt.colorbar()
#     plt.show()


#
# cross section of covariance map
# only x is used
# y cross section is not so helpful as it is not in the direction where wfs point differently
# 

xCrossCx = Cscreen[:,0,:,mid]
xCrossCy = Cscreen[:,1,:,mid]
# xCrossAx = Ascreen[:,0,:,mid]
# xCrossAy = Ascreen[:,1,:,mid]

exCrossCx = Cescreen[:,0,:,mid]
exCrossCy = Cescreen[:,1,:,mid]
# exCrossAx = Aescreen[:,0,:,mid]
# exCrossAy = Aescreen[:,1,:,mid]

xCrossRefCx = refx[:,:,mid]
xCrossRefCy = refy[:,:,mid]

color_list = ['r','g','b','c','m','y','k','tab:purple']

for layer in np.arange(LAYER):    
    plt.errorbar(biggrid,xCrossCx[layer],yerr=exCrossCx[layer],
                 label='{:.0f}km'.format(scrnHeights_list[layer]/1000),
                 marker='o',capsize=5,ls=':',color=color_list[layer])
    plt.plot(biggrid,xCrossRefCx[layer],ls='-',color=color_list[layer])
plt.title('Slope cross-covariance (T), {:d} frames'.format(sim.config.sim.nIters))
plt.legend(loc='upper left')
plt.show()
for layer in np.arange(LAYER):    
    plt.errorbar(biggrid,xCrossCy[layer],yerr=exCrossCy[layer],
                 label='{:.0f}km'.format(scrnHeights_list[layer]/1000),
                 marker='o',capsize=5,ls=':',color=color_list[layer])
    plt.plot(biggrid,xCrossRefCy[layer],ls='-',color=color_list[layer])
plt.title('Slope cross-covariance (L), {:d} frames'.format(sim.config.sim.nIters))
plt.legend(loc='upper left')
plt.show()
# for layer in np.arange(LAYER):    
#     plt.errorbar(biggrid,xCrossAx[layer],yerr=exCrossAx[layer],
#                  label='{:.0f}km'.format(scrnHeights_list[layer]/1000),
#                  marker='o',capsize=5)
# plt.title('x crosssection of Ax, {:d} frames'.format(sim.config.sim.nIters))
# plt.legend()
# plt.show()
# for layer in np.arange(LAYER):    
#     plt.errorbar(biggrid,xCrossAy[layer],yerr=exCrossAy[layer],
#                  label='{:.0f}km'.format(scrnHeights_list[layer]/1000),
#                  marker='o',capsize=5)
# plt.title('x crosssection of Ay, {:d} frames'.format(sim.config.sim.nIters))
# plt.legend()
# plt.show()

# multiplierx = (np.max(xCrossCx,axis=1)/np.max(xCrossRefCx,axis=1))**-1
# multipliery = (np.max(xCrossCy,axis=1)/np.max(xCrossRefCy,axis=1))**-1
multiplierx=1
multipliery=1
for layer in np.arange(LAYER):    
    plt.errorbar(biggrid,multiplierx[layer]*xCrossCx[layer],yerr=exCrossCx[layer],
                 label='{:.0f}km'.format(scrnHeights_list[layer]/1000),
                 marker='o',capsize=5,ls=':',color=color_list[layer])
    plt.plot(biggrid,xCrossRefCx[layer],ls='-',color=color_list[layer])
plt.title('Slope cross-covariance (T), {:d} frames'.format(sim.config.sim.nIters))
plt.legend(loc='upper left')
plt.show()
for layer in np.arange(LAYER):    
    plt.errorbar(biggrid,multipliery[layer]*xCrossCy[layer],yerr=exCrossCy[layer],
                 label='{:.0f}km'.format(scrnHeights_list[layer]/1000),
                 marker='o',capsize=5,ls=':',color=color_list[layer])
    plt.plot(biggrid,xCrossRefCy[layer],ls='-',color=color_list[layer])
plt.title('Slope cross-covariance (L), {:d} frames'.format(sim.config.sim.nIters))
plt.legend(loc='upper left')
plt.show()

