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

N = sim.wfss[0].n_subaps

if sim.wfss[1].n_subaps != N:
    print('error')

if sim.config.wfss[0].GSPosition[0] - sim.config.wfss[1].GSPosition[0] == 0:
    print('wfs should be pointing differently on x direction')

if sim.config.wfss[0].GSPosition[1] - sim.config.wfss[1].GSPosition[1] != 0:
    print('wfs should be pointing similarly on y direction')


# !!! this is actually changed to wind Dir but im too lazy to change ...
# scrnHeights_list = (np.arange(sim.config.wfss[0].nxSubaps, dtype=float)
#                     * (sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)
#                     / ((sim.config.wfss[0].GSPosition[0]
#                         - sim.config.wfss[1].GSPosition[0])*ARCSEC2RAD))
scrnHeights_list = np.arange(8) * 45
LAYER = scrnHeights_list.shape[0]



#
# measure Cross covariance / auto covariance of turbulence at different height
#

C = np.zeros((LAYER,2,N,N),dtype=float) # xy, wfs1subap, wfs2subap
A1 = np.zeros((LAYER,2,N,N),dtype=float) # autocovariance1
A2 = np.zeros((LAYER,2,N,N),dtype=float) # autocovariance2
A = np.zeros((LAYER,2,N,N),dtype=float) # autocovariance2

eC = np.zeros((LAYER,2,N,N),dtype=float) # xy, wfs1subap, wfs2subap
eA1 = np.zeros((LAYER,2,N,N),dtype=float) # autocovariance1
eA2 = np.zeros((LAYER,2,N,N),dtype=float) # autocovariance2
eA = np.zeros((LAYER,2,N,N),dtype=float) # autocovariance2

position = np.asarray(sim.wfss[0].pupil_subap_coords/sim.wfss[0].nx_subap_pixels,dtype=int)

mid = position.max()

grid = np.arange(mid+1,dtype=float)
grid -= len(grid)/2
grid += 0.5

for layer in np.arange(LAYER):
    scrnHeights = scrnHeights_list[layer]
    # sim.config.atmos.windDirs = [windDirs]
    # sim.config.atmos.scrnHeights = [scrnHeights]
    sim.config.atmos.windDirs = [scrnHeights] # !!! im too lazy to change variable name
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
    A1[layer,0], eA1[layer,0] = my_cov(Sx1.T,Sx1.T)
    A1[layer,1], eA1[layer,1] = my_cov(Sy1.T,Sy1.T)
    A2[layer,0], eA2[layer,0] = my_cov(Sx2.T,Sx2.T)
    A2[layer,1], eA2[layer,1] = my_cov(Sy2.T,Sy2.T)

    A[layer] = (A1[layer] + A2[layer])/2.
    eA[layer] = eA1[layer] + eA2[layer]


name = 'AwindDir{:d}'.format(sim.config.sim.nIters)
hdu = fits.PrimaryHDU(A)
hdul = fits.HDUList([hdu])
hdul.writeto(name + '.fits', overwrite=True)
hdul.close()
name = 'CwindDir{:d}'.format(sim.config.sim.nIters)
hdu = fits.PrimaryHDU(C)
hdul = fits.HDUList([hdu])
hdul.writeto(name + '.fits', overwrite=True)
hdul.close()

name = 'eAwindDir{:d}'.format(sim.config.sim.nIters)
hdu = fits.PrimaryHDU(eA)
hdul = fits.HDUList([hdu])
hdul.writeto(name + '.fits', overwrite=True)
hdul.close()
name = 'eCwindDir{:d}'.format(sim.config.sim.nIters)
hdu = fits.PrimaryHDU(eC)
hdul = fits.HDUList([hdu])
hdul.writeto(name + '.fits', overwrite=True)
hdul.close()

# name = 'A'
# hdulist = fits.open(name + '.fits')
# hdu = hdulist[0]
# A = hdu.data
# hdulist.close()
# name = 'C'
# hdulist = fits.open(name + '.fits')
# hdu = hdulist[0]
# C = hdu.data
# hdulist.close()

# for layer in np.arange(LAYER):
#     # print(layer,np.where(Cscreen[layer,0]==np.nanmax(Cscreen[layer,0])))
#     plt.pcolor(np.arange(C.shape[2]),np.arange(C.shape[3]),C[layer,0].T)
#     plt.axis('square')
#     plt.colorbar()
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('Cx{:d}'.format(layer))
#     plt.show()
# for layer in np.arange(LAYER):
#     # print(layer,np.where(Cscreen[layer,1]==np.nanmax(Cscreen[layer,1])))
#     plt.pcolor(np.arange(C.shape[2]),np.arange(C.shape[3]),C[layer,1].T)
#     plt.axis('square')
#     plt.colorbar()
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('Cy{:d}'.format(layer))
#     plt.show()
# for layer in np.arange(LAYER):
#     # print(layer,np.where(Ascreen[layer,0]==np.nanmax(Ascreen[layer,0])))
#     plt.pcolor(np.arange(C.shape[2]),np.arange(C.shape[3]),A[layer,0].T)
#     plt.axis('square')
#     plt.colorbar()
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('Ax{:d}'.format(layer))
#     plt.show()
# for layer in np.arange(LAYER):
#     # print(layer,np.where(Ascreen[layer,1]==np.nanmax(Ascreen[layer,1])))
#     plt.pcolor(np.arange(C.shape[2]),np.arange(C.shape[3]),A[layer,1].T)
#     plt.axis('square')
#     plt.colorbar()
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('Ay{:d}'.format(layer))
#     plt.show()



#
# generate covariance map on wfs pupil
#

biggrid = np.arange(2*mid+1,dtype=float)
biggrid -= len(biggrid)/2
biggrid += 0.5    
    
Ascreen = np.zeros((LAYER,2,2*mid + 1,2*mid + 1), dtype=float)
Acount = np.zeros((LAYER,2,2*mid + 1,2*mid + 1), dtype=float)
Cscreen = np.zeros((LAYER,2,2*mid + 1,2*mid + 1), dtype=float)
Ccount = np.zeros((LAYER,2,2*mid + 1,2*mid + 1), dtype=float)
Aescreen = np.zeros((LAYER,2,2*mid + 1,2*mid + 1), dtype=float)
Aecount = np.zeros((LAYER,2,2*mid + 1,2*mid + 1), dtype=float)
Cescreen = np.zeros((LAYER,2,2*mid + 1,2*mid + 1), dtype=float)
Cecount = np.zeros((LAYER,2,2*mid + 1,2*mid + 1), dtype=float)
for layer in np.arange(LAYER):
    Ascreen[layer, 0], Aescreen[layer, 0] = overlay1(A[layer, 0], eA[layer, 0], position)
    Cscreen[layer, 0], Cescreen[layer, 0] = overlay1(C[layer, 0], eC[layer, 0], position)
    Ascreen[layer, 1], Aescreen[layer, 1] = overlay1(A[layer, 1], eA[layer, 1], position)
    Cscreen[layer, 1], Cescreen[layer, 1] = overlay1(C[layer, 1], eC[layer, 1], position)
    
# for layer in np.arange(LAYER):
#     # print(layer,np.where(Cscreen[layer,0]==np.nanmax(Cscreen[layer,0])))
#     plt.pcolor(biggrid,biggrid,Cscreen[layer,0].T)
#     plt.axis('square')
#     plt.colorbar()
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('Cx{:d}'.format(layer))
#     plt.show()
# for layer in np.arange(LAYER):
#     # print(layer,np.where(Cscreen[layer,1]==np.nanmax(Cscreen[layer,1])))
#     plt.pcolor(biggrid,biggrid,Cscreen[layer,1].T)
#     plt.axis('square')
#     plt.colorbar()
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('Cy{:d}'.format(layer))
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



#
# cross section of covariance map
# only x is used
# y cross section is not so helpful as it is not in the direction where wfs point differently
# 

xCrossCx = Cscreen[:,0,:,mid]
xCrossCy = Cscreen[:,1,:,mid]
xCrossAx = Ascreen[:,0,:,mid]
xCrossAy = Ascreen[:,1,:,mid]

exCrossCx = Cescreen[:,0,:,mid]
exCrossCy = Cescreen[:,1,:,mid]
exCrossAx = Aescreen[:,0,:,mid]
exCrossAy = Aescreen[:,1,:,mid]

for layer in np.arange(LAYER):    
    plt.errorbar(biggrid,xCrossCx[layer],yerr=exCrossCx[layer],
                 label='windDir{:.0f}deg'.format(scrnHeights_list[layer]),
                 marker='o',capsize=5)
plt.title('x crosssection of Cx, {:d} frames'.format(sim.config.sim.nIters))
plt.legend()
plt.show()
for layer in np.arange(LAYER):    
    plt.errorbar(biggrid,xCrossCy[layer],yerr=exCrossCy[layer],
                 label='windDir{:.0f}deg'.format(scrnHeights_list[layer]),
                 marker='o',capsize=5)
plt.title('x crosssection of Cy, {:d} frames'.format(sim.config.sim.nIters))
plt.legend()
plt.show()
for layer in np.arange(LAYER):    
    plt.errorbar(biggrid,xCrossAx[layer],yerr=exCrossAx[layer],
                 label='windDir{:.0f}deg'.format(scrnHeights_list[layer]),
                 marker='o',capsize=5)
plt.title('x crosssection of Ax, {:d} frames'.format(sim.config.sim.nIters))
plt.legend()
plt.show()
for layer in np.arange(LAYER):    
    plt.errorbar(biggrid,xCrossAy[layer],yerr=exCrossAy[layer],
                 label='windDir{:.0f}deg'.format(scrnHeights_list[layer]),
                 marker='o',capsize=5)
plt.title('x crosssection of Ay, {:d} frames'.format(sim.config.sim.nIters))
plt.legend()
plt.show()




