# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:41:18 2023

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

ARCSEC2RAD = 1./3600./360.*2.*np.pi
RAD2ARCSEC = 1./ARCSEC2RAD

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
#     if eA != None:
#         a = np.zeros((2*position.max() + 1, 2*position.max() + 1), dtype=float)
#         ae = np.zeros((2*position.max() + 1, 2*position.max() + 1), dtype=float)
#         count = np.zeros((2*position.max() + 1, 2*position.max() + 1), dtype=float)
#         N = len(A)
#         mid = position.max()
#         for ii in np.arange(N):
#             for jj in np.arange(N):
#                 xcoord = mid + (position[jj, 0] - position[ii, 0])
#                 ycoord = mid + (position[jj, 1] - position[ii, 1])
#                 a[xcoord, ycoord] += A[ii, jj]
#                 ae[xcoord, ycoord] += eA[ii, jj]**2
#                 count[xcoord, ycoord] += 1
#         a /= count
#         ae /= count
#         ae = np.sqrt(ae)
#         return a, ae
#     else:
#         a = np.zeros((2*position.max() + 1, 2*position.max() + 1), dtype=float)
#         count = np.zeros((2*position.max() + 1, 2*position.max() + 1), dtype=float)
#         N = len(A)
#         mid = position.max()
#         for ii in np.arange(N):
#             for jj in np.arange(N):
#                 xcoord = mid + (position[jj, 0] - position[ii, 0])
#                 ycoord = mid + (position[jj, 1] - position[ii, 1])
#                 a[xcoord, ycoord] += A[ii, jj]
#                 count[xcoord, ycoord] += 1
#         a /= count
#         return a

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

def stdcn22stdr0(cn2,stdcn2,wvl=500e-9,zenith_angle=0.):
    stdr0 = np.abs((0.423*(2.*np.pi/wvl)**2/np.cos(zenith_angle))**(-3./5.)
                   * (-3./5.) * cn2**(-8./5.)) * stdcn2
    return stdr0


home = ('C:/Users/gqmc62/OneDrive - Durham University/Desktop/'
             + 'soapy-master/soapy-master')
# file = 'SLODAR.yaml'
reffile = 'conf/referenceFunctionsVK_6x6'
file_list = ['conf/SLODAR_CloseLoop_6x6_11cm.yaml',
             'conf/SLODAR_CloseLoop_6x6_08cm.yaml',
             'conf/SLODAR_CloseLoop_6x6_05cm.yaml',
             'conf/SLODAR_CloseLoop_6x6_02cm.yaml',
             'conf/SLODAR_OpenLoop_6x6_11cm.yaml',
             'conf/SLODAR_OpenLoop_6x6_08cm.yaml',
             'conf/SLODAR_OpenLoop_6x6_05cm.yaml',
             'conf/SLODAR_OpenLoop_6x6_02cm.yaml']

# CHECK_AO = False
# CHECK_SLODAR = False

# if CHECK_AO == True:
    
#     os.chdir(home)
#     sim = soapy.Sim(file)

#     loopTime = sim.config.sim.loopTime

#     r0 = sim.config.atmos.r0
#     L0 = sim.config.atmos.L0

#     telDiam = sim.config.tel.telDiam

#     TT_iMatValue = sim.config.dms[0].iMatValue
#     DM_iMatValue = sim.config.dms[1].iMatValue
#     DM_nxActuators = sim.config.dms[1].nxActuators

#     svdConditioning = sim.config.recon.svdConditioning

#     windSpeeds = sim.config.atmos.windSpeeds
#     turbStrength = sim.config.atmos.scrnStrengths
#     turbStrength = turbStrength/turbStrength.sum()
#     windSpeed53 = (windSpeeds**(5./3.)*turbStrength).sum()**(3./5.)
#     tau0 = 0.314*r0/windSpeed53
    
#     for n in np.arange(sim.config.atmos.scrnNo):
#         if L0[n] < 8*telDiam:
#             print('layer{:d}, L0 might be too small for telescope to be Kolmogorov-like.'.format(n))
    
#     for ndm in np.arange(len(sim.config.dms)):
#         for nwfs in np.arange(len(sim.config.wfss)):
#             if sim.config.dms[ndm].type == 'FastPiezo':
#                 if sim.config.wfss[nwfs].nxSubaps + 1 != sim.config.dms[ndm].nxActuators:
#                     print('dm{:d} and wfs{:d} are Not Fried Geometry.\nChange either nxSubpars or nxActuators.'.format(ndm,nwfs))
                    
#     for n in np.arange(len(sim.config.wfss)):
#         nxSubaps = sim.config.wfss[n].nxSubaps
#         pxlsPerSubap = sim.config.wfss[n].pxlsPerSubap
#         subapFOV = sim.config.wfss[n].subapFOV
#         wfs_wavelength = sim.config.wfss[n].wavelength
#         wfsTime = sim.config.wfss[n].exposureTime
#         r0wfs = r0*(wfs_wavelength/500e-9)**(6./5.)
#         tau0wfs = tau0*(wfs_wavelength/500e-9)**(6./5.)
#         if tau0wfs < wfsTime:
#             print('exposure time of wfs{:d} might be too short'.format(n))
#         subapDiam = telDiam/nxSubaps
#         wfs_diffraction = wfs_wavelength/2/subapDiam*RAD2ARCSEC
#         wfs_angular_pxls = subapFOV/pxlsPerSubap
#         if wfs_angular_pxls > wfs_diffraction:
#             print('WFS{:d} cannot resolve spot.'.format(n))
#         if subapFOV < 5.*wfs_diffraction:
#             print('Linearity of WFS{:d} is not good.'.format(n))
#         if subapDiam > r0wfs:
#             print('r0 is too small for wfs{:d} lenslet'.format(n))
#         for natm in np.arange(sim.config.atmos.scrnNo):
#             if windSpeeds[natm]*wfsTime < telDiam/nxSubaps/pxlsPerSubap:
#                 print('shifing of atm{:d} is too slow to accurately simulate with wfs{:d}'.format(natm,n))
#         recommend_looptime = subapDiam/pxlsPerSubap/sim.config.atmos.windSpeeds.min()
#         if loopTime < recommend_looptime:
#             print('recommend looptime = {:.f}'.format(recommend_looptime))
    
#     for natm in np.arange(sim.config.atmos.scrnNo):
#         if windSpeeds[natm]*loopTime < telDiam/nxSubaps/pxlsPerSubap:
#             print('shifing of atm{:d} is too slow to accurately simulate'.format(natm))
    
#     print('recommend loopTime = {:.6f}'.format(recommend_looptime))
                
#     for n in np.arange(len(sim.config.scis)):
#         sci_FOV = sim.config.scis[n].FOV
#         sci_wavelength = sim.config.scis[n].wavelength
#         sci_pxls = sim.config.scis[n].pxls
#         sci_diffraction = sci_wavelength/2/telDiam*RAD2ARCSEC
#         sci_angular_pxls = sci_FOV/sci_pxls
#         if sci_angular_pxls > sci_diffraction:
#             print('Science camera {:d} cannot resolve spot.'.format(n))
#         for ndm in np.arange(len(sim.config.dms)):
#             if sim.config.dms[ndm].type == 'FastPiezo':
#                 if (telDiam / (sim.config.dms[ndm].nxActuators - 1)) > (r0*(sci_wavelength/500e-9)**(6./5.)):
#                     print('dm{:d} cannot correct for r0 of sci{:d}'.format(ndm,n))
    
#     sim.config.sim.loopTime = tau0
#     sim.config.sim.verbosity = 2
#     # if sim.config.sim.nIters != 10:
#     #     print('Please set nIters = 10')
#     #     quit()
#     sim.aoinit()
#     sim.config.calcParams()
#     sim.makeIMat()
    
#     plt.imshow(sim.recon.interaction_matrix)
#     plt.colorbar()
#     plt.title('IM')
#     plt.show()
    
#     if ((np.isclose(np.max(sim.recon.interaction_matrix[ : 2]), 1, atol=1e-1) == False)
#         or (np.isclose(np.max(sim.recon.interaction_matrix[2 : ]), 1, atol=1e-1) == False)):
#         print('suggest tip tilt dm to have imat value of {:.3f}'.format(TT_iMatValue/np.max(sim.recon.interaction_matrix[:2])))
#         print('suggest high order dm to have imat value of {:.3f}'.format(DM_iMatValue/np.max(sim.recon.interaction_matrix[2:])))
    
#     _, svd, _ = np.linalg.svd(sim.recon.interaction_matrix)
    
#     drop = np.where((svd[ : -1] - svd[1 : ]) == np.max(svd[2:-1] - svd[2 + 1 : ]))[0][0]
#     if np.isclose((9.*svd[drop] + 1.*svd[drop + 1])/10./svd.max(), svdConditioning, atol=1e-1) == False:
#         print('suggest svd at {:.3f}'.format((9.*svd[drop] + 1.*svd[drop + 2])/10./svd.max()))
    
#     svd_drop = svd[drop]/svd.max()
#     svd_max = 2*svd[drop]/svd.max()
#     svd_min = svd[drop + 1]/svd.max()/2.
    
#     plt.plot(svd/svd.max(),marker='o')
#     plt.hlines(svd[drop]/svd.max(),0,len(svd),label='{:.3f}'.format(svd[drop]/svd.max()), color='m')
#     plt.hlines(svd[drop + 1]/svd.max(),0,len(svd),label='{:.3f}'.format(svd[drop + 1]/svd.max()), color='y')
#     plt.hlines(svdConditioning,0,len(svd),label='current svd')
#     plt.legend()
#     plt.title('svd(interaction matrix)')
#     plt.ylabel('normalized svd')
#     plt.show()
    
#     plt.imshow(sim.recon.control_matrix)
#     plt.colorbar()
#     plt.title('CM')
#     plt.show()
    
#     for n in np.arange(len(sim.config.dms)):
#         sim.config.dms[n].gain = 1
#     sim.aoloop()
    
#     for nwfs in np.arange(len(sim.wfss)):
#         plt.imshow(sim.wfss[nwfs].wfsDetectorPlane)
#         plt.colorbar()
#         plt.title('wfs{:d}'.format(nwfs))
#         plt.show()
#     MAX = 0.
#     MIN = np.nan
#     for nwfs in np.arange(len(sim.wfss)):
#         MAX = np.nanmax([MAX,np.nanmax(sim.wfss[nwfs].los.phase_screens)])
#         MIN = np.nanmin([MIN,np.nanmin(sim.wfss[nwfs].los.phase_screens)])
#     for nwfs in np.arange(len(sim.wfss)):    
#         plt.imshow(np.sum(sim.wfss[nwfs].los.phase_screens, axis=0),vmin=MIN,vmax=MAX)
#         plt.colorbar()
#         plt.title('wfs{:d}'.format(nwfs))
#         plt.show()
#     plt.imshow(np.sum(sim.combinedCorrection,axis=0),vmin=MIN,vmax=MAX)
#     plt.colorbar()
#     plt.title('DM')
#     plt.show()
#     plt.imshow(sim.sciCams[0].residual*sim.mask,vmin=MIN,vmax=MAX)
#     plt.colorbar()
#     plt.title('Residual WF')
#     plt.show()
#     plt.imshow(sim.sciCams[0].detector)
#     plt.colorbar()
#     plt.title('On-Axis PSF')
#     plt.show()

#     nb = input('Proceed?: ')
#     if nb != 'y':
#         quit()

# if CHECK_SLODAR == True:
#     os.chdir(home)
#     sim = soapy.Sim(file)

#     loopTime = sim.config.sim.loopTime

#     r0 = sim.config.atmos.r0
#     L0 = sim.config.atmos.L0

#     telDiam = sim.config.tel.telDiam

#     TT_iMatValue = sim.config.dms[0].iMatValue
#     DM_iMatValue = sim.config.dms[1].iMatValue
#     DM_nxActuators = sim.config.dms[1].nxActuators

#     svdConditioning = sim.config.recon.svdConditioning

#     windSpeeds = sim.config.atmos.windSpeeds
#     turbStrength = sim.config.atmos.scrnStrengths
#     turbStrength = turbStrength/turbStrength.sum()
#     windSpeed53 = (windSpeeds**(5./3.)*turbStrength).sum()**(3./5.)
#     tau0 = 0.314*r0/windSpeed53

#     sim.aoinit()

#     N = sim.wfss[0].n_subaps

#     if sim.wfss[1].n_subaps != N:
#         print('error')

#     if sim.config.wfss[0].GSPosition[0] - sim.config.wfss[1].GSPosition[0] == 0:
#         print('wfs should be pointing differently on x direction')

#     if sim.config.wfss[0].GSPosition[1] - sim.config.wfss[1].GSPosition[1] != 0:
#         print('wfs should be pointing similarly on y direction')
        
#     d = (sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)
#     print('subapDiam={:.6f}m\n'.format(d))

#     plt.plot(sim.wfss[0].pupil_subap_coords[:,0],
#               sim.wfss[0].pupil_subap_coords[:,1],
#               ls='',marker='o')
#     plt.axis('square')
#     plt.title('pupil')
#     plt.show()
    
#     recommend_looptime = d/sim.wfss[0].nx_subap_pixels/sim.config.atmos.windSpeeds.min()
#     recommend_nIters = telDiam*10/sim.config.atmos.windSpeeds.min()/recommend_looptime
#     if ((0.5*recommend_looptime > loopTime ) or (loopTime > 2*recommend_looptime)):
#         print('recommend loopTime = {:f}'.format(recommend_looptime))
#     if ((0.5*recommend_nIters > sim.config.sim.nIters) or (sim.config.sim.nIters > 2*recommend_nIters)):
#         print('recommend nIters = {:f}'.format(recommend_nIters))
#     nb = input('Proceed?: ')
#     if nb != 'y':
#         quit()

for file in file_list:
    
    os.chdir(home)
    sim = soapy.Sim(file)
    
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
    
    name = reffile
    hdulist = fits.open(name + '.fits')
    hdu = hdulist[0]
    ref = np.flip(hdu.data, axis=1)
    hdulist.close()
    
    refx = ref[::2]
    refy = ref[1::2]
    
    LAYER = refy.shape[0]
    SCREEN = (refy.shape[1],refy.shape[2])
    FLAT = refy.shape[1]*refy.shape[2]
    
    REFX = refx.reshape(LAYER,FLAT).T
    REFY = refy.reshape(LAYER,FLAT).T
    
    REPEAT = 10
    RANDOM = 1
    MAXR0 = r0
    MINR0 = r0
    REFR0 = sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps
    REFCN2 = r02cn2(REFR0)
    
    scrnHeights_list = (np.arange(LAYER, dtype=float)
                        * (sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)
                        / ((sim.config.wfss[0].GSPosition[0]
                            - sim.config.wfss[1].GSPosition[0])*ARCSEC2RAD))
    sim.config.atmos.scrnHeights = scrnHeights_list
    
    # list_random_r0 = np.random.random(RANDOM)*(MAXR0 - MINR0) + MINR0
    list_random_r0 = np.linspace(MINR0,MAXR0,RANDOM)
    list_random_scrnStrengths = np.random.random((RANDOM,LAYER))
    
    for random in np.arange(RANDOM):
        list_random_scrnStrengths[random] /= np.sum(list_random_scrnStrengths[random])
    # np.sum(list_random_scrnStrengths,axis=1)
    
    color_list = ['r','g','b','c','m','y','k','tab:purple']
    
    sim.aoinit()
    # sim.config.calcParams()
    sim.makeIMat(forceNew=True)
    
    N = sim.wfss[0].n_subaps
    position = np.asarray(sim.wfss[0].pupil_subap_coords/sim.wfss[0].nx_subap_pixels,dtype=int) - 1
    mid = sim.config.wfss[0].nxSubaps - 1 - 1 - 1
    grid = np.arange(mid + 1,dtype=float)
    grid -= len(grid)/2
    grid += 0.5
    biggrid = np.arange(2*mid + 1, dtype=float)
    biggrid -= len(biggrid)/2
    biggrid += 0.5
    
    Cx = np.zeros((N,N),dtype=float) # xy, wfs1subap, wfs2subap
    eCx = np.zeros((N,N),dtype=float) # xy, wfs1subap, wfs2subap
    Cy = np.zeros((N,N),dtype=float) # xy, wfs1subap, wfs2subap
    eCy = np.zeros((N,N),dtype=float) # xy, wfs1subap, wfs2subap
    Cxscreen = np.zeros((2*mid + 1,2*mid + 1), dtype=float)
    Cxescreen = np.zeros((2*mid + 1,2*mid + 1), dtype=float)
    Cyscreen = np.zeros((2*mid + 1,2*mid + 1), dtype=float)
    Cyescreen = np.zeros((2*mid + 1,2*mid + 1), dtype=float)
    
    cn2x = np.zeros((RANDOM,LAYER),dtype=float)
    cn2y = np.zeros((RANDOM,LAYER),dtype=float)
    cn2x_all = np.zeros((RANDOM,REPEAT,LAYER),dtype=float)
    cn2y_all = np.zeros((RANDOM,REPEAT,LAYER),dtype=float)
    ecn2x = np.zeros((RANDOM,LAYER),dtype=float)
    ecn2y = np.zeros((RANDOM,LAYER),dtype=float)
    
    r0x = np.zeros((RANDOM),dtype=float)
    r0y = np.zeros((RANDOM),dtype=float)
    std_r0x = np.zeros((RANDOM),dtype=float)
    std_r0y = np.zeros((RANDOM),dtype=float)
    XscrnStrengths = np.zeros((RANDOM,LAYER),dtype=float)
    YscrnStrengths = np.zeros((RANDOM,LAYER),dtype=float)
    
    for random in tqdm(np.arange(RANDOM)):
        for repeat in tqdm(np.arange(REPEAT)):
            print('at random {:d}x{:d}'.format(random,repeat))
            scrnStrengths = list_random_scrnStrengths[random]
            sim.config.atmos.r0 = list_random_r0[random]
            sim.config.atmos.scrnStrengths = scrnStrengths
            
            sim.config.randomSeed = np.random.randint(0, 999)
            sim.atmos = soapy.atmosphere.atmos(sim.config)
            sim.aoinit()
            # sim.config.calcParams()
            sim.makeIMat(forceNew=True)
            sim.allDmCommands = np.zeros((sim.config.sim.nIters, sim.config.sim.totalActs))
            sim.aoloop()
            
            openSlope = sim.allSlopes[1:] + np.matmul(sim.allDmCommands[:-1], sim.recon.interaction_matrix)
            
            Sx1 = openSlope[ : , 0*N : (0+1)*N]
            Sy1 = openSlope[ : , 1*N : (1+1)*N]
            Sx2 = openSlope[ : , 2*N : (2+1)*N]
            Sy2 = openSlope[ : , 3*N : (3+1)*N]
            
            TIME = Sx1.shape[0]
            for time in np.arange(TIME):
                Sx1[time] -= np.mean(Sx1[time])
                Sy1[time] -= np.mean(Sy1[time])
                Sx2[time] -= np.mean(Sx2[time])
                Sy2[time] -= np.mean(Sy2[time])
            
            Cx = my_cov(Sx1.T,Sx2.T)
            Cy = my_cov(Sy1.T,Sy2.T)
        
            Cxscreen, Cxescreen = overlay1(Cx, position)
            Cyscreen, Cyescreen = overlay1(Cy, position)
        
            soln_cn2x = lsq_linear(REFX, np.nan_to_num((Cxscreen).reshape(FLAT)), bounds=(0,np.inf))
            soln_cn2y = lsq_linear(REFY, np.nan_to_num((Cyscreen).reshape(FLAT)), bounds=(0,np.inf))
            
            cn2x_all[random, repeat] = soln_cn2x.x * REFCN2
            cn2y_all[random, repeat] = soln_cn2y.x * REFCN2
            
            sim.reset_loop()
    
    cn2x = np.mean(cn2x_all, axis=1)
    cn2y = np.mean(cn2y_all, axis=1)
    ecn2x = np.std(cn2x_all, axis=1, ddof=1)
    ecn2y = np.std(cn2y_all, axis=1, ddof=1)
    
    for random in tqdm(np.arange(RANDOM)):
        print('at random {:d}'.format(random))
        # total_cn2x = np.sum(cn2x[random])
        # etotal_cn2x = np.sum(ecn2x[random])
        # r0x[random] = cn22r0(total_cn2x)
        # std_r0x[random] = stdcn22stdr0(total_cn2x,etotal_cn2x)
        # total_cn2y = np.sum(cn2y[random])
        # r0y[random] = cn22r0(total_cn2y)
        # etotal_cn2y = np.sum(ecn2y[random])
        # std_r0y[random] = stdcn22stdr0(total_cn2y,etotal_cn2y)
        
        cn2 = (cn2x[random]/ecn2x[random]**2 + cn2y[random]/ecn2y[random]**2)/(1/ecn2x[random]**2+1/ecn2y[random]**2)
        ecn2 = np.sqrt(1/(1/ecn2x[random]**2+1/ecn2y[random]**2))
        total_cn2 = np.sum(cn2)
        etotal_cn2 = np.sum(ecn2)
        r0 = cn22r0(total_cn2)
        er0 = stdcn22stdr0(total_cn2, etotal_cn2)
        
        
        XscrnStrengths[random] = cn2x[random]# / total_cn2x
        YscrnStrengths[random] = cn2y[random]# / total_cn2y
        scrnStrengths = cn2# / total_cn2
            
        plt.plot(list_random_scrnStrengths[random]*r02cn2(list_random_r0[random]),label='Simulating',ls='-')
        # plt.plot(np.arange(LAYER),
        #              XscrnStrengths[random],
        #              label='MeasuredX',ls='--',marker='^')
        # plt.plot(np.arange(LAYER),
        #              YscrnStrengths[random],
        #              label='MeasuredY',ls='--',marker='v')
        plt.errorbar(np.arange(LAYER),
                      cn2,yerr=ecn2,capsize=5,
                      label='Measured',ls='--',marker='^')
        plt.legend()
        plt.ylabel('cn2')
        plt.xlabel('SLODAR layer')
        # plt.title('Distribution measurement, r0 = {:.3f}m,\n'.format(list_random_r0[random])
        #           + 'measuredX = {:.3f}+-{:.3f}m,\n'.format(r0x[random],std_r0x[random])
        #           + 'measuredY = {:.3f}+-{:.3f}m'.format(r0y[random],std_r0y[random]))
        plt.title('Distribution measurement, r0 = {:.3f}m,\n'.format(list_random_r0[random])
                  + 'measured = {:.3f}m\n'.format(r0))
        plt.yscale('log')
        plt.show()

