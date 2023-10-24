# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:18:38 2023

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
            a[xcoord, ycoord] += A[ii, jj]/eA[ii,jj]**2
            ae[xcoord, ycoord] += 1./eA[ii, jj]**2
            count[xcoord, ycoord] += 1
    a /= ae
    ae = 1./np.sqrt(ae)
    return a, ae

# def overlay1(A,position):
#     a = np.zeros((2*position.max() + 1, 2*position.max() + 1), dtype=float)
#     ae = np.zeros((2*position.max() + 1, 2*position.max() + 1), dtype=float)
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
#     for ii in np.arange(N):
#         for jj in np.arange(N):
#             xcoord = mid + (position[jj, 0] - position[ii, 0])
#             ycoord = mid + (position[jj, 1] - position[ii, 1])
#             ae[xcoord, ycoord] += (A[ii, jj] - a[xcoord, ycoord])**2
#     ae /= count
#     ae = np.sqrt(ae)
#     return a, ae

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

def my_cov(A, B):
    N = A.shape[0]
    C = np.zeros((N, N), dtype=float)
    eC = np.zeros((N, N), dtype=float)
    for ii in np.arange(N):
        for jj in np.arange(N):
            temp = (A[ii] - A[ii].mean())*(B[jj] - B[jj].mean())
            C[ii, jj] = np.mean(temp)
            res = bootstrap((temp,),np.std)
            eC[ii, jj] = res.standard_error
            # eC[ii, jj] = np.std(temp,ddof=1)
    return C, eC

# def my_cov(A, B):
#     N = A.shape[0]
#     C = np.zeros((N, N), dtype=float)
#     for ii in np.arange(N):
#         for jj in np.arange(N):
#             temp = (A[ii] - A[ii].mean())*(B[jj] - B[jj].mean())
#             C[ii, jj] = np.mean(temp)
#     return C

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
eC = np.zeros((LAYER,2,N,N),dtype=float) # xy, wfs1subap, wfs2subap

for layer in np.arange(LAYER):
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

name = 'LatestC'
hdu = fits.PrimaryHDU(C)
hdul = fits.HDUList([hdu])
hdul.writeto(name + '.fits', overwrite=True)
hdul.close()

name = 'LatesteC'
hdu = fits.PrimaryHDU(eC)
hdul = fits.HDUList([hdu])
hdul.writeto(name + '.fits', overwrite=True)
hdul.close()

# name = 'LatestC'
# hdulist = fits.open(name + '.fits')
# hdu = hdulist[0]
# C = hdu.data
# hdulist.close()

# name = 'LatesteC'
# hdulist = fits.open(name + '.fits')
# hdu = hdulist[0]
# eC = hdu.data
# hdulist.close()

name = 'referenceFunctionsVK_6x6'
hdulist = fits.open(name + '.fits')
hdu = hdulist[0]
ref = np.flip(hdu.data, axis=1)
hdulist.close()
ref[ref==0] = np.nan

refx = ref[::2]
refy = ref[1::2]

# for layer in np.arange(LAYER):
#     # print(layer,np.where(Cscreen[layer,0]==np.nanmax(Cscreen[layer,0])))
#     plt.pcolor(np.arange(C.shape[2]),np.arange(C.shape[3]),C[layer,0].T)
#     plt.axis('square')
#     plt.colorbar()
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('Slope cross-covariance (L) {:d}'.format(layer))
#     plt.show()
# for layer in np.arange(LAYER):
#     # print(layer,np.where(Cscreen[layer,1]==np.nanmax(Cscreen[layer,1])))
#     plt.pcolor(np.arange(C.shape[2]),np.arange(C.shape[3]),C[layer,1].T)
#     plt.axis('square')
#     plt.colorbar()
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('Slope cross-covariance (T) {:d}'.format(layer))
#     plt.show()

#
# generate covariance map on wfs pupil
#

biggrid = np.arange(2*mid + 1, dtype=float)
biggrid -= len(biggrid)/2
biggrid += 0.5
    
Cscreen = np.zeros((LAYER,2,2*mid + 1,2*mid + 1), dtype=float)
Cescreen = np.zeros((LAYER,2,2*mid + 1,2*mid + 1), dtype=float)
for layer in tqdm(np.arange(LAYER)):
    Cscreen[layer, 0], Cescreen[layer, 0] = overlay1(C[layer, 0], eC[layer, 0],position)
    Cscreen[layer, 1], Cescreen[layer, 1] = overlay1(C[layer, 1], eC[layer, 1],position)
    
# # for layer in np.arange(LAYER):
# #     plt.pcolor(biggrid,biggrid,Cscreen[layer,0].T)
# #     plt.axis('square')
# #     plt.colorbar()
# #     plt.xlabel('x = first index')
# #     plt.ylabel('y = second index')
# #     plt.title('Cx{:d}'.format(layer))
# #     plt.show()
# # for layer in np.arange(LAYER):
# #     plt.pcolor(biggrid,biggrid,Cscreen[layer,1].T)
# #     plt.axis('square')
# #     plt.colorbar()
# #     plt.xlabel('x = first index')
# #     plt.ylabel('y = second index')
# #     plt.title('Cy{:d}'.format(layer))
# #     plt.show()
# # for i in np.arange(test.shape[0]):
# #     plt.pcolor(biggrid,biggrid,test[i].T)
# #     plt.axis('square')
# #     plt.xlabel('x = first index')
# #     plt.ylabel('y = second index')
# #     plt.title('slope covariance {:d}'.format(i))
# #     plt.colorbar()
# #     plt.show()   

MAX = np.nanmax([np.nanmax(refx),np.nanmax(refy)])
MIN = np.nanmin([np.nanmin(refx),np.nanmin(refy)])
MAX2 = np.max([np.abs(MAX),np.abs(MIN)])/10.
MIN2 = -MAX2


# for layer in np.arange(LAYER):
#     # print(layer,np.where(Cscreen[layer,0]==np.nanmax(Cscreen[layer,0])))
#     plt.pcolor(biggrid,biggrid,Cscreen[layer,0].T,vmin=MIN,vmax=MAX)
#     plt.axis('square')
#     plt.colorbar()
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('Slope cross-covariance (L) {:d}'.format(layer))
#     plt.show()
# for i in np.arange(refx.shape[0]):
#     plt.pcolor(biggrid,biggrid,refx[i].T,vmin=MIN,vmax=MAX)
#     plt.axis('square')
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('x reference function {:d}'.format(i))
#     plt.colorbar()
#     plt.show()
# for i in np.arange(refx.shape[0]):
#     plt.pcolor(biggrid,biggrid,(Cscreen[i,0] - refx[i]).T,vmin=MIN2,vmax=MAX2,cmap='RdBu')
#     plt.axis('square')
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('L residual slope covariance {:d}'.format(i))
#     plt.colorbar()
#     plt.show()
# for i in np.arange(refx.shape[0]):
#     plt.pcolor(biggrid,biggrid,((Cscreen[i,0] - refx[i])/Cescreen[i,0]).T,vmin=-3,vmax=3,cmap='RdBu')
#     plt.axis('square')
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('residual x slope covariance {:d} in sd unit'.format(i))
#     plt.colorbar()
#     plt.show()


# for layer in np.arange(LAYER):
#     # print(layer,np.where(Cscreen[layer,1]==np.nanmax(Cscreen[layer,1])))
#     plt.pcolor(biggrid,biggrid,Cscreen[layer,1].T,vmin=MIN,vmax=MAX)
#     plt.axis('square')
#     plt.colorbar()
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('Slope cross-covariance (T) {:d}'.format(layer))
#     plt.show()
# for i in np.arange(refy.shape[0]):
#     plt.pcolor(biggrid,biggrid,refy[i].T,vmin=MIN,vmax=MAX)
#     plt.axis('square')
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('y reference function {:d}'.format(i))
#     plt.colorbar()
#     plt.show()
# for i in np.arange(refy.shape[0]):
#     plt.pcolor(biggrid,biggrid,(Cscreen[i,1] - refy[i]).T,vmin=MIN2,vmax=MAX2,cmap='RdBu')
#     plt.axis('square')
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('T residual slope covariance {:d}'.format(i))
#     plt.colorbar()
#     plt.show()
# for i in np.arange(refy.shape[0]):
#     plt.pcolor(biggrid,biggrid,((Cscreen[i,1] - refy[i])/Cescreen[i,1]).T,vmin=-3,vmax=3,cmap='RdBu')
#     plt.axis('square')
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('residual y slope covariance {:d} in sd unit'.format(i))
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

yCrossCy = np.zeros_like(xCrossCy)
eyCrossCy = np.zeros_like(xCrossCy)
yCrossRefCy = np.zeros_like(xCrossCy)
for layer in np.arange(LAYER):
    yCrossCy[layer] = Cscreen[layer,1,mid+layer,:]
    eyCrossCy[layer] = Cescreen[layer,1,mid+layer,:]
    yCrossRefCy[layer] = refy[layer,mid+layer,:]
yCrossCx = np.zeros_like(xCrossCy)
eyCrossCx = np.zeros_like(xCrossCy)
yCrossRefCx = np.zeros_like(xCrossCy)
for layer in np.arange(LAYER):
    yCrossCx[layer] = Cscreen[layer,0,mid+layer,:]
    eyCrossCx[layer] = Cescreen[layer,0,mid+layer,:]
    yCrossRefCx[layer] = refx[layer,mid+layer,:]


color_list = ['r','g','b','c','m','y','k','tab:purple']

for layer in np.arange(LAYER):
    if layer == 0:
        plt.errorbar(biggrid,xCrossCx[layer],yerr=exCrossCx[layer],
                      label='measured',
                      marker='o',capsize=5,ls='',color=color_list[layer])
        plt.plot(biggrid,xCrossRefCx[layer],ls='--',color=color_list[layer],label='ref')
    else:
        plt.errorbar(biggrid,xCrossCx[layer],yerr=exCrossCx[layer],
                      marker='o',capsize=5,ls='',color=color_list[layer])
        plt.plot(biggrid,xCrossRefCx[layer],ls='--',color=color_list[layer])
plt.title('Slope cross-covariance (L), {:d} frames'.format(sim.config.sim.nIters))
plt.legend(loc='upper left')
plt.show()
# for layer in np.arange(LAYER//2)*2:
#     plt.errorbar(biggrid,(xCrossCx[layer]-xCrossRefCx[layer]),yerr=exCrossCx[layer],
#                   marker='o',ls='-',color=color_list[layer],capsize=5)
# plt.hlines(0,biggrid.min(),biggrid.max(),ls='--')
# plt.title('Residual of Slope cross-covariance (L), {:d} frames'.format(sim.config.sim.nIters))
# plt.show()

# for layer in np.arange(LAYER):    
#     plt.errorbar(biggrid,yCrossCx[layer],yerr=eyCrossCx[layer],
#                   label='layer{:.0f}'.format(layer),
#                   marker='o',capsize=5,ls='',color=color_list[layer],alpha=0.5)
#     plt.plot(biggrid,yCrossRefCx[layer],ls='--',color=color_list[layer])
# # plt.title('Slope cross-covariance (L), {:d} frames'.format(sim.config.sim.nIters))
# plt.legend(loc='upper left')
# plt.show()
# for layer in np.arange(LAYER):    
#     plt.errorbar(biggrid,(yCrossCx[layer]-yCrossRefCx[layer]),yerr=eyCrossCx[layer],
#                   label='layer{:.0f}'.format(layer),
#                   marker='o',ls='-',color=color_list[layer],alpha=0.5,capsize=5)
# plt.hlines(0,biggrid.min(),biggrid.max(),ls='--')
# # plt.title('Residual of Slope cross-covariance (L), {:d} frames'.format(sim.config.sim.nIters))
# plt.legend(loc='upper left')
# plt.show()



for layer in np.arange(LAYER):    
    if layer == 0:
        plt.errorbar(biggrid,xCrossCy[layer],yerr=exCrossCy[layer],
                      label='measured',
                      marker='o',capsize=5,ls='',color=color_list[layer])
        plt.plot(biggrid,xCrossRefCy[layer],ls='--',color=color_list[layer],label='ref')
    else:
        plt.errorbar(biggrid,xCrossCy[layer],yerr=exCrossCy[layer],
                      marker='o',capsize=5,ls='',color=color_list[layer])
        plt.plot(biggrid,xCrossRefCy[layer],ls='--',color=color_list[layer])
plt.title('Slope cross-covariance (T), {:d} frames'.format(sim.config.sim.nIters))
plt.legend(loc='upper left')
plt.show()
# for layer in np.arange(LAYER):    
#     plt.errorbar(biggrid,(xCrossCy[layer]-xCrossRefCy[layer]),yerr=exCrossCy[layer],
#                   marker='o',ls='-',color=color_list[layer],capsize=5)
# plt.title('Residual of Slope cross-covariance (T), {:d} frames'.format(sim.config.sim.nIters))
# plt.hlines(0,biggrid.min(),biggrid.max(),ls='--')
# plt.legend(loc='upper left')
# plt.show()

# for layer in np.arange(LAYER):    
#     plt.errorbar(biggrid,yCrossCy[layer],yerr=eyCrossCy[layer],
#                   label='layer{:.0f}'.format(layer),
#                   marker='o',capsize=5,ls='',color=color_list[layer],alpha=0.5)
#     plt.plot(biggrid,yCrossRefCy[layer],ls='--',color=color_list[layer])
# # plt.title('Slope cross-covariance (T), {:d} frames'.format(sim.config.sim.nIters))
# plt.legend(loc='upper left')
# plt.show()
# for layer in np.arange(LAYER):    
#     plt.errorbar(biggrid,(yCrossCy[layer]-yCrossRefCy[layer]),yerr=eyCrossCy[layer],
#                   label='layer{:.0f}'.format(layer),
#                   marker='o',ls='-',color=color_list[layer],alpha=0.5,capsize=5)
# plt.hlines(0,biggrid.min(),biggrid.max(),ls='--')
# # plt.title('Residual of Slope cross-covariance (T), {:d} frames'.format(sim.config.sim.nIters))
# plt.legend(loc='upper left')
# plt.show()





# # next file?

# color_list = ['r','g','b','c','m','y','k','tab:purple']
# # NORMALIZED = np.max([np.nanmax(np.abs(Cscreen)),np.nanmax(np.abs(refx)),np.nanmax(np.abs(refy))])
# Cscreen = Cscreen#/NORMALIZED
# refx = refx#/NORMALIZED
# refy = refy#/NORMALIZED

# REFR0 = sim.atmos.r0
# REFCN2 = r02cn2(REFR0)

# LAYER = refy.shape[0]
# SCREEN = (refy.shape[1],refy.shape[2])
# FLAT = refy.shape[1]*refy.shape[2]

# xres = np.zeros((LAYER,LAYER))

# MAX = np.nanmax(Cscreen)
# MIN = np.nanmin(Cscreen)
# MAX2 = np.max([MAX,np.abs(MIN)])
# MIN2 = -MAX2

# # A = np.nan_to_num((refy.reshape(LAYER, FLAT)).T)
# # for layer in np.arange(LAYER):
# #     b = np.nan_to_num(Cscreen[layer,1].reshape(FLAT))
# #     res = lsq_linear(A, b, bounds=(0,np.inf))
# #     # BRecon = np.matmul(A,res.x)
# #     xres[layer] = res.x * REFCN2
# #     # plt.imshow(b.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# #     # plt.title('Measured')
# #     # plt.colorbar()
# #     # plt.show()
# #     # plt.imshow(BRecon.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# #     # plt.title('Reconstructed')
# #     # plt.colorbar()
# #     # plt.show()
# # #     plt.imshow(res.fun.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# # #     plt.title('Residual {:f}'.format(((res.fun[res.fun**2>0])**2).sum()**0.5))
# # #     plt.colorbar()
# # #     plt.show()
# # # for layer in np.arange(LAYER):
# #     x0 = np.zeros((LAYER))
# #     x0[layer] = 1 * REFCN2
# #     if layer == 0:
# #         plt.plot(xres[layer],ls='--',label='measured',color=color_list[layer],marker='v')
# #         plt.plot(x0,ls='-',color=color_list[layer],marker='^',label='input')
# #     else:
# #         plt.plot(xres[layer],ls='--',color=color_list[layer],marker='v')
# #         plt.plot(x0,ls='-',color=color_list[layer],marker='^')
# # plt.title('measured cn2')
# # plt.legend()
# # plt.show()

# # A = np.nan_to_num((refx.reshape(LAYER, FLAT)).T)
# # for layer in np.arange(LAYER):
# #     b = np.nan_to_num(Cscreen[layer,0].reshape(FLAT))
# #     res = lsq_linear(A, b, bounds=(0,np.inf))
# #     # BRecon = np.matmul(A,res.x)
# #     xres[layer] = res.x * REFCN2
# #     # plt.imshow(b.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# #     # plt.title('Measured')
# #     # plt.colorbar()
# #     # plt.show()
# #     # plt.imshow(BRecon.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# #     # plt.title('Reconstructed')
# #     # plt.colorbar()
# #     # plt.show()
# # #     plt.imshow(res.fun.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# # #     plt.title('Residual {:f}'.format(((res.fun[res.fun**2>0])**2).sum()**0.5))
# # #     plt.colorbar()
# # #     plt.show()
# # # for layer in np.arange(LAYER):
# #     x0 = np.zeros((LAYER))
# #     x0[layer] = 1 * REFCN2
# #     if layer == 0:
# #         plt.plot(xres[layer],ls='--',label='measured',color=color_list[layer],marker='v')
# #         plt.plot(x0,ls='-',color=color_list[layer],marker='^',label='input')
# #     else:
# #         plt.plot(xres[layer],ls='--',color=color_list[layer],marker='v')
# #         plt.plot(x0,ls='-',color=color_list[layer],marker='^')
# # plt.title('measured cn2')
# # plt.legend()
# # plt.show()

# Ax = np.nan_to_num((refx.reshape(LAYER, FLAT)).T)
# Ay = np.nan_to_num((refy.reshape(LAYER, FLAT)).T)
# for layer in np.arange(LAYER):
#     bx = np.nan_to_num(Cscreen[layer,0].reshape(FLAT))
#     by = np.nan_to_num(Cscreen[layer,1].reshape(FLAT))
#     resx = lsq_linear(Ax, bx, bounds=(0,np.inf))
#     resy = lsq_linear(Ay, by, bounds=(0,np.inf))
#     # BRecon = np.matmul(A,res.x)
#     xres[layer] = (resx.x+resy.x)/2. * REFCN2
#     # plt.imshow(b.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
#     # plt.title('Measured')
#     # plt.colorbar()
#     # plt.show()
#     # plt.imshow(BRecon.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
#     # plt.title('Reconstructed')
#     # plt.colorbar()
#     # plt.show()
# #     plt.imshow(res.fun.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# #     plt.title('Residual {:f}'.format(((res.fun[res.fun**2>0])**2).sum()**0.5))
# #     plt.colorbar()
# #     plt.show()
# # for layer in np.arange(LAYER):
#     x0 = np.zeros((LAYER))
#     x0[layer] = 1 * REFCN2
#     if layer == 0:
#         plt.plot(xres[layer],ls='--',label='measured',color=color_list[layer],marker='v')
#         plt.plot(x0,ls='-',color=color_list[layer],marker='^',label='input')
#     else:
#         plt.plot(xres[layer],ls='--',color=color_list[layer],marker='v')
#         plt.plot(x0,ls='-',color=color_list[layer],marker='^')
# plt.title('measured cn2 both LT fit whole screen')
# plt.legend()
# plt.show()

# Ax = np.nan_to_num((refx.reshape(LAYER, FLAT)).T)
# Ay = np.nan_to_num((refy.reshape(LAYER, FLAT)).T)
# for layer in np.arange(LAYER):
#     bx = np.nan_to_num(Cscreen[layer,0].reshape(FLAT))
#     by = np.nan_to_num(Cscreen[layer,1].reshape(FLAT))
#     resx = lsq_linear(Ax, bx, bounds=(0,np.inf))
#     resy = lsq_linear(Ay, by, bounds=(0,np.inf))
#     # BRecon = np.matmul(A,res.x)
#     xres[layer] = (resx.x+resy.x)/2. * REFCN2
#     xres[layer] = cn22r0(xres[layer])
#     # plt.imshow(b.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
#     # plt.title('Measured')
#     # plt.colorbar()
#     # plt.show()
#     # plt.imshow(BRecon.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
#     # plt.title('Reconstructed')
#     # plt.colorbar()
#     # plt.show()
# #     plt.imshow(res.fun.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# #     plt.title('Residual {:f}'.format(((res.fun[res.fun**2>0])**2).sum()**0.5))
# #     plt.colorbar()
# #     plt.show()
# # for layer in np.arange(LAYER):
#     x0 = np.zeros((LAYER))
#     x0[layer] = 1 * REFCN2
#     x0 = cn22r0(x0)
#     if layer == 0:
#         plt.plot(xres[layer],label='measured',color=color_list[layer],marker='v',ls='')
#         plt.plot(x0,color=color_list[layer],marker='^',label='input',ls='')
#     else:
#         plt.plot(xres[layer],color=color_list[layer],marker='v',ls='')
#         plt.plot(x0,color=color_list[layer],marker='^',ls='')
# plt.ylim(0,0.1)
# plt.title('measured cn2 both LT fit whole screen')
# plt.legend()
# plt.show()


# #crosssection
# # A = np.nan_to_num((refy[:,:,mid]).T)
# # for layer in np.arange(LAYER):
# #     b = np.nan_to_num(Cscreen[layer,1,:,mid])
# #     res = lsq_linear(A, b, bounds=(0,np.inf))
# #     # BRecon = np.matmul(A,res.x)
# #     xres[layer] = res.x * REFCN2
# #     # plt.imshow(b.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# #     # plt.title('Measured')
# #     # plt.colorbar()
# #     # plt.show()
# #     # plt.imshow(BRecon.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# #     # plt.title('Reconstructed')
# #     # plt.colorbar()
# #     # plt.show()
# # #     plt.imshow(res.fun.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# # #     plt.title('Residual {:f}'.format(((res.fun[res.fun**2>0])**2).sum()**0.5))
# # #     plt.colorbar()
# # #     plt.show()
# # # for layer in np.arange(LAYER):
# #     x0 = np.zeros((LAYER))
# #     x0[layer] = 1 * REFCN2
# #     if layer == 0:
# #         plt.plot(xres[layer],ls='--',label='measured',color=color_list[layer],marker='v')
# #         plt.plot(x0,ls='-',color=color_list[layer],marker='^',label='input')
# #     else:
# #         plt.plot(xres[layer],ls='--',color=color_list[layer],marker='v')
# #         plt.plot(x0,ls='-',color=color_list[layer],marker='^')
# # plt.title('measured cn2')
# # plt.legend()
# # plt.show()

# # A = np.nan_to_num((refx[:,:,mid]).T)
# # for layer in np.arange(LAYER):
# #     b = np.nan_to_num(Cscreen[layer,0,:,mid])
# #     res = lsq_linear(A, b, bounds=(0,np.inf))
# #     # BRecon = np.matmul(A,res.x)
# #     xres[layer] = res.x * REFCN2
# #     # plt.imshow(b.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# #     # plt.title('Measured')
# #     # plt.colorbar()
# #     # plt.show()
# #     # plt.imshow(BRecon.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# #     # plt.title('Reconstructed')
# #     # plt.colorbar()
# #     # plt.show()
# # #     plt.imshow(res.fun.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# # #     plt.title('Residual {:f}'.format(((res.fun[res.fun**2>0])**2).sum()**0.5))
# # #     plt.colorbar()
# # #     plt.show()
# # # for layer in np.arange(LAYER):
# #     x0 = np.zeros((LAYER))
# #     x0[layer] = 1 * REFCN2
# #     if layer == 0:
# #         plt.plot(xres[layer],ls='--',label='measured',color=color_list[layer],marker='v')
# #         plt.plot(x0,ls='-',color=color_list[layer],marker='^',label='input')
# #     else:
# #         plt.plot(xres[layer],ls='--',color=color_list[layer],marker='v')
# #         plt.plot(x0,ls='-',color=color_list[layer],marker='^')
# # plt.title('measured cn2')
# # plt.legend()
# # plt.show()

# Ax = np.nan_to_num((refx[:,:,mid]).T)
# Ay = np.nan_to_num((refy[:,:,mid]).T)
# for layer in np.arange(LAYER):
#     bx = np.nan_to_num(Cscreen[layer,0,:,mid])
#     by = np.nan_to_num(Cscreen[layer,1,:,mid])
#     resx = lsq_linear(Ax, bx, bounds=(0,np.inf))
#     resy = lsq_linear(Ay, by, bounds=(0,np.inf))
#     # BRecon = np.matmul(A,res.x)
#     xres[layer] = (resx.x+resy.x)/2. * REFCN2
#     # plt.imshow(b.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
#     # plt.title('Measured')
#     # plt.colorbar()
#     # plt.show()
#     # plt.imshow(BRecon.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
#     # plt.title('Reconstructed')
#     # plt.colorbar()
#     # plt.show()
# #     plt.imshow(res.fun.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# #     plt.title('Residual {:f}'.format(((res.fun[res.fun**2>0])**2).sum()**0.5))
# #     plt.colorbar()
# #     plt.show()
# # for layer in np.arange(LAYER):
#     x0 = np.zeros((LAYER))
#     x0[layer] = 1 * REFCN2
#     if layer == 0:
#         plt.plot(xres[layer],ls='--',label='measured',color=color_list[layer],marker='v')
#         plt.plot(x0,ls='-',color=color_list[layer],marker='^',label='input')
#     else:
#         plt.plot(xres[layer],ls='--',color=color_list[layer],marker='v')
#         plt.plot(x0,ls='-',color=color_list[layer],marker='^')
# plt.title('measured cn2 both LT cross section fit')
# plt.legend()
# plt.show()

# Ax = np.nan_to_num((refx[:,:,mid]).T)
# Ay = np.nan_to_num((refy[:,:,mid]).T)
# for layer in np.arange(LAYER):
#     bx = np.nan_to_num(Cscreen[layer,0,:,mid])
#     by = np.nan_to_num(Cscreen[layer,1,:,mid])
#     resx = lsq_linear(Ax, bx, bounds=(0,np.inf))
#     resy = lsq_linear(Ay, by, bounds=(0,np.inf))
#     # BRecon = np.matmul(A,res.x)
#     xres[layer] = (resx.x+resy.x)/2. * REFCN2
#     xres[layer] = cn22r0(xres[layer])
#     # plt.imshow(b.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
#     # plt.title('Measured')
#     # plt.colorbar()
#     # plt.show()
#     # plt.imshow(BRecon.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
#     # plt.title('Reconstructed')
#     # plt.colorbar()
#     # plt.show()
# #     plt.imshow(res.fun.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# #     plt.title('Residual {:f}'.format(((res.fun[res.fun**2>0])**2).sum()**0.5))
# #     plt.colorbar()
# #     plt.show()
# # for layer in np.arange(LAYER):
#     x0 = np.zeros((LAYER))
#     x0[layer] = 1 * REFCN2
#     x0 = cn22r0(x0)
#     if layer == 0:
#         plt.plot(xres[layer],label='measured',color=color_list[layer],marker='v',ls='')
#         plt.plot(x0,color=color_list[layer],marker='^',label='input',ls='')
#     else:
#         plt.plot(xres[layer],color=color_list[layer],marker='v',ls='')
#         plt.plot(x0,color=color_list[layer],marker='^',ls='')
# plt.title('measured cn2 both LT cross section fit')
# plt.ylim(0,0.1)
# plt.legend()
# plt.show()


# # with reference function
# NTEST = 10
# A = np.nan_to_num((refy.reshape(LAYER, FLAT)).T)
# for ntest in np.arange(NTEST):
#     random_coeff = np.random.rand(6)*2
#     test_func = np.matmul((refy.reshape(LAYER,FLAT)).T,random_coeff)
#     b = np.nan_to_num((test_func).reshape(FLAT))
#     res = lsq_linear(A, b, bounds=(0,np.inf))
#     # BRecon = np.matmul(A,res.x)
#     # plt.imshow(b.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
#     # plt.title('Measured')
#     # plt.colorbar()
#     # plt.show()
#     # plt.imshow(BRecon.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
#     # plt.title('Reconstructed')
#     # plt.colorbar()
#     # plt.show()
#     # plt.imshow(res.fun.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
#     # plt.title('Residual {:f}'.format((res.fun[res.fun>0]).std()))
#     # plt.colorbar()
#     # plt.show()
#     plt.plot(res.x,label='output',marker='o')
#     plt.plot(random_coeff,label='input',marker='x')
#     plt.title('Test LSQ recon')
#     plt.legend()
#     plt.show()
    
    
    
# # with simulation??? ==> next file
# NTEST = 10
# Ay = np.nan_to_num((refy.reshape(LAYER, FLAT)).T)
# Ax = np.nan_to_num((refx.reshape(LAYER, FLAT)).T)
# for ntest in np.arange(NTEST):
#     random_coeff = np.random.rand(6)*2
#     test_funcy = np.matmul((Cscreen[:,1,:,:].reshape(LAYER,FLAT)).T,random_coeff)
#     by = np.nan_to_num((test_funcy).reshape(FLAT))
#     test_funcx = np.matmul((Cscreen[:,0,:,:].reshape(LAYER,FLAT)).T,random_coeff)
#     bx = np.nan_to_num((test_funcx).reshape(FLAT))
#     resx = lsq_linear(Ax, bx, bounds=(0,np.inf))
#     resy = lsq_linear(Ay, by, bounds=(0,np.inf))
#     # BRecon = np.matmul(A,res.x)
#     # plt.imshow(b.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
#     # plt.title('Measured')
#     # plt.colorbar()
#     # plt.show()
#     # plt.imshow(BRecon.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
#     # plt.title('Reconstructed')
#     # plt.colorbar()
#     # plt.show()
#     # plt.imshow(res.fun.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
#     # plt.title('Residual {:f}'.format((res.fun[res.fun>0]).std()))
#     # plt.colorbar()
#     # plt.show()
#     plt.plot((resx.x+resy.x)/2.,label='output',marker='o')
#     plt.plot(random_coeff,label='input',marker='x')
#     plt.title('Residual {:f}'.format(((res.fun[res.fun**2>0])**2).sum()**0.5))
#     plt.legend()
#     plt.show()


