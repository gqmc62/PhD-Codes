# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:22:09 2023

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


ARCSEC2RAD = 1./3600./360.*2.*np.pi
RAD2ARCSEC = 1./ARCSEC2RAD



home = ('C:/Users/gqmc62/OneDrive - Durham University/Desktop/'
             + 'soapy-master/soapy-master/conf')
file = 'SLODAR1L.yaml'
TESTCONF = False

os.chdir(home)
sim = soapy.Sim(file)

if TESTCONF == True:
    
    loopTime = sim.config.sim.loopTime
    
    r0 = sim.config.atmos.r0
    L0 = sim.config.atmos.L0
    
    telDiam = sim.config.tel.telDiam
    
    TT_iMatValue = sim.config.dms[0].iMatValue
    DM_iMatValue = sim.config.dms[1].iMatValue
    DM_nxActuators = sim.config.dms[1].nxActuators
    
    svdConditioning = sim.config.recon.svdConditioning
    
    windSpeeds = sim.config.atmos.windSpeeds
    turbStrength = sim.config.atmos.scrnStrengths
    turbStrength = turbStrength/turbStrength.sum()
    windSpeed53 = (windSpeeds**(5./3.)*turbStrength).sum()**(3./5.)
    tau0 = 0.314*r0/windSpeed53
    
    for n in np.arange(sim.config.atmos.scrnNo):
        if L0[n] < 8*telDiam:
            print('layer{:d}, L0 might be too small for telescope to be Kolmogorov-like.'.format(n))
    
    for ndm in np.arange(len(sim.config.dms)):
        for nwfs in np.arange(len(sim.config.wfss)):
            if sim.config.dms[ndm].type == 'FastPiezo':
                if sim.config.wfss[nwfs].nxSubaps + 1 != sim.config.dms[ndm].nxActuators:
                    print('dm{:d} and wfs{:d} are Not Fried Geometry.\nChange either nxSubpars or nxActuators.'.format(ndm,nwfs))
                    
    for n in np.arange(len(sim.config.wfss)):
        nxSubaps = sim.config.wfss[n].nxSubaps
        pxlsPerSubap = sim.config.wfss[n].pxlsPerSubap
        subapFOV = sim.config.wfss[n].subapFOV
        wfs_wavelength = sim.config.wfss[n].wavelength
        wfsTime = sim.config.wfss[n].exposureTime
        r0wfs = r0*(wfs_wavelength/500e-9)**(6./5.)
        tau0wfs = tau0*(wfs_wavelength/500e-9)**(6./5.)
        if tau0wfs < wfsTime:
            print('exposure time of wfs{:d} might be too short'.format(n))
            print('try {:f}'.format(tau0wfs))
        subapDiam = telDiam/nxSubaps
        wfs_diffraction = wfs_wavelength/2/subapDiam*RAD2ARCSEC
        wfs_angular_pxls = subapFOV/pxlsPerSubap
        if wfs_angular_pxls > wfs_diffraction:
            print('WFS{:d} cannot resolve spot.'.format(n))
        if subapFOV < 5.*wfs_diffraction:
            print('Linearity of WFS{:d} is not good.'.format(n))
        if subapDiam > r0wfs:
            print('r0 is too small for wfs{:d} lenslet'.format(n))
        for natm in np.arange(sim.config.atmos.scrnNo):
            if windSpeeds[natm]*wfsTime < telDiam/nxSubaps/pxlsPerSubap:
                print('shifing of atm{:d} is too slow to accurately simulate with wfs{:d}'.format(natm,n))
    
    for natm in np.arange(sim.config.atmos.scrnNo):
        if windSpeeds[natm]*loopTime < telDiam/nxSubaps/pxlsPerSubap:
            print('shifing of atm{:d} is too slow to accurately simulate'.format(natm))
                
    for n in np.arange(len(sim.config.scis)):
        sci_FOV = sim.config.scis[n].FOV
        sci_wavelength = sim.config.scis[n].wavelength
        sci_pxls = sim.config.scis[n].pxls
        sci_diffraction = sci_wavelength/2/telDiam*RAD2ARCSEC
        sci_angular_pxls = sci_FOV/sci_pxls
        if sci_angular_pxls > sci_diffraction:
            print('Science camera {:d} cannot resolve spot.'.format(n))
        for ndm in np.arange(len(sim.config.dms)):
            if sim.config.dms[ndm].type == 'FastPiezo':
                if (telDiam / (sim.config.dms[ndm].nxActuators - 1)) > (r0*(sci_wavelength/500e-9)**(6./5.)):
                    print('dm{:d} cannot correct for r0 of sci{:d}'.format(ndm,n))
    
    sim.aoinit()
    sim.config.calcParams()
    sim.makeIMat()
    
    plt.imshow(sim.recon.interaction_matrix)
    plt.colorbar()
    plt.title('IM')
    plt.show()
    
    if ((np.isclose(np.max(sim.recon.interaction_matrix[ : 2]), 1, atol=1e-1) == False)
        or (np.isclose(np.max(sim.recon.interaction_matrix[2 : ]), 1, atol=1e-1) == False)):
        print(np.max(sim.recon.interaction_matrix[ : 2]))
        print(np.max(sim.recon.interaction_matrix[2 : ]))
        print('suggest tip tilt dm to have imat value of {:.3f}'.format(TT_iMatValue/np.max(sim.recon.interaction_matrix[:2])))
        print('suggest high order dm to have imat value of {:.3f}'.format(DM_iMatValue/np.max(sim.recon.interaction_matrix[2:])))
    
    _, svd, _ = np.linalg.svd(sim.recon.interaction_matrix)
    
    drop = np.where((svd[ : -1] - svd[1 : ]) == np.max(svd[2:-1] - svd[2 + 1 : ]))[0][0]
    if np.isclose((9.*svd[drop] + 1.*svd[drop + 1])/10./svd.max(), svdConditioning, atol=1e-1) == False:
        print('suggest svd at {:.3f}'.format((9.*svd[drop] + 1.*svd[drop + 2])/10./svd.max()))
    
    svd_drop = svd[drop]/svd.max()
    svd_max = 2*svd[drop]/svd.max()
    svd_min = svd[drop + 1]/svd.max()/2.
    # svd_chosen = (3.*svd[drop] + svd[drop + 1])/4./svd.max()
    
    plt.plot(svd/svd.max(),marker='o')
    # plt.hlines(svd[drop - 1]/svd.max(),0,len(svd),label='{:.3f}'.format(svd[drop - 1]/svd.max()), color='c')
    plt.hlines(svd[drop]/svd.max(),0,len(svd),label='{:.3f}'.format(svd[drop]/svd.max()), color='m')
    plt.hlines(svd[drop + 1]/svd.max(),0,len(svd),label='{:.3f}'.format(svd[drop + 1]/svd.max()), color='y')
    plt.hlines(svdConditioning,0,len(svd),label='current svd')
    # plt.hlines(svd[drop + 2]/svd.max(),0,len(svd),label='{:.3f}'.format(svd[drop + 2]/svd.max()), color='k')
    plt.legend()
    plt.title('svd(interaction matrix)')
    plt.ylabel('normalized svd')
    plt.show()
    
    plt.imshow(sim.recon.control_matrix)
    plt.colorbar()
    plt.title('CM')
    plt.show()
    
    for n in np.arange(len(sim.config.dms)):
        sim.config.dms[n].gain = 1
    sim.aoloop()
    
    
    for nwfs in np.arange(len(sim.wfss)):
        plt.imshow(sim.wfss[nwfs].wfsDetectorPlane)
        plt.colorbar()
        plt.title('wfs{:d}'.format(nwfs))
        plt.show()
    MAX = 0.
    MIN = np.nan
    for nwfs in np.arange(len(sim.wfss)):
        MAX = np.nanmax([MAX,np.nanmax(sim.wfss[nwfs].los.phase_screens)])
        MIN = np.nanmin([MIN,np.nanmin(sim.wfss[nwfs].los.phase_screens)])
    for nwfs in np.arange(len(sim.wfss)):    
        plt.imshow(np.sum(sim.wfss[nwfs].los.phase_screens, axis=0),vmin=MIN,vmax=MAX)
        plt.colorbar()
        plt.title('wfs{:d}'.format(nwfs))
        plt.show()
    plt.imshow(np.sum(sim.combinedCorrection,axis=0),vmin=MIN,vmax=MAX)
    plt.colorbar()
    plt.show()
    plt.imshow(sim.sciCams[0].residual*sim.mask,vmin=MIN,vmax=MAX)
    plt.colorbar()
    plt.show()
    plt.imshow(sim.sciCams[0].detector)
    plt.colorbar()
    plt.show()
    
    nb = input('Please configure reference slope covariance\nProceed?\nType y to proceed: ')
    if nb != 'y':
        quit()

sim.aoinit()

N = sim.wfss[0].n_subaps
scrnHeights_list = (np.arange(sim.config.wfss[0].nxSubaps, dtype=float)
                    * (sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)
                    / ((sim.config.wfss[0].GSPosition[0]
                        - sim.config.wfss[1].GSPosition[0])*ARCSEC2RAD))
LAYER = scrnHeights_list.shape[0]
position = np.asarray(sim.wfss[0].pupil_subap_coords/sim.wfss[0].nx_subap_pixels,dtype=int)
mid = sim.config.wfss[0].nxSubaps - 1
# mid = position.max()
grid = np.arange(mid + 1,dtype=float)
grid -= len(grid)/2
grid += 0.5

if sim.wfss[1].n_subaps != N:
    print('error')

if sim.config.wfss[0].GSPosition[0] - sim.config.wfss[1].GSPosition[0] == 0:
    print('wfs should be pointing differently on x direction')

if sim.config.wfss[0].GSPosition[1] - sim.config.wfss[1].GSPosition[1] != 0:
    print('wfs should be pointing similarly on y direction')

name = 'referenceFunctions'
hdulist = fits.open(name + '.fits')
hdu = hdulist[0]
ref = np.flip(hdu.data, axis=1)
hdulist.close()
ref[ref==0] = np.nan

refx = ref[::2]
refy = ref[1::2]

#
# measure Cross covariance / auto covariance of turbulence at different height
#

C = np.zeros((2,N,N),dtype=float) # xy, wfs1subap, wfs2subap
eC = np.zeros((2,N,N),dtype=float) # xy, wfs1subap, wfs2subap

random_layer = np.random.randint(LAYER)
scrnHeights = scrnHeights_list[random_layer]
r0min = sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps
r0max = 0.15
random_r0 = np.random.random()*(r0max - r0min) + r0min

sim.config.atmos.scrnHeights = [scrnHeights]
sim.config.atmos.r0 = random_r0
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
#     plt.title('{:d} {:d}'.format(random_layer,time))
#     plt.legend()
#     plt.show()

Cx, eCx = my_cov(Sx1.T,Sx2.T)
Cy, eCy = my_cov(Sy1.T,Sy2.T)
SCx, eSCx = overlay1(Cx, eCx, position)
SCy, eSCy = overlay1(Cy, eCy, position)

signalx = np.zeros((LAYER), dtype=float)
signaly = np.zeros((LAYER), dtype=float)

for layer in np.arange(LAYER):
    signalx[layer] = (np.nansum(SCx*refx[layer])/np.nansum(refx[layer]*refx[layer]))
    signaly[layer] = (np.nansum(SCy*refy[layer])/np.nansum(refy[layer]*refy[layer]))

YMAX = np.max([signalx,signaly])
YMIN = np.min([signalx,signaly])
plt.plot(np.arange(LAYER),signalx,label='L')
plt.plot(np.arange(LAYER),signaly,label='T')
plt.vlines(random_layer,YMIN,YMAX,label='Random Layer')
plt.legend()
plt.show()

