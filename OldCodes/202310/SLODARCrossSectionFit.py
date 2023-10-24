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
    a /= ae**2
    ae = 1./np.sqrt(ae)
    return a, ae

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
    return C, eC

# def my_cov(A, B):
#     N = A.shape[0]
#     #n = np.sqrt(A.shape[1])
#     C = np.zeros((N, N), dtype=float)
#     #eC = np.zeros((N, N), dtype=float)
#     for ii in np.arange(N):
#         for jj in np.arange(N):
#             temp = (A[ii] - A[ii].mean())*(B[jj] - B[jj].mean())
#             C[ii, jj] = np.mean(temp)
#             #eC[ii, jj] = np.std(temp, ddof=1)/n
    return C#, eC

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
             + 'soapy-master/soapy-master/conf')
file = 'SLODAR.yaml'
reffile = 'referenceFunctions'

CHECK_AO = False
CHECK_SLODAR = False

if CHECK_AO == True:
    
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
        recommend_looptime = subapDiam/pxlsPerSubap/sim.config.atmos.windSpeeds.min()
        if loopTime < recommend_looptime:
            print('recommend looptime = {:.f}'.format(recommend_looptime))
    
    for natm in np.arange(sim.config.atmos.scrnNo):
        if windSpeeds[natm]*loopTime < telDiam/nxSubaps/pxlsPerSubap:
            print('shifing of atm{:d} is too slow to accurately simulate'.format(natm))
    
    print('recommend loopTime = {:.6f}'.format(recommend_looptime))
                
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
    
    sim.config.sim.loopTime = tau0
    sim.config.sim.verbosity = 2
    # if sim.config.sim.nIters != 10:
    #     print('Please set nIters = 10')
    #     quit()
    sim.aoinit()
    sim.config.calcParams()
    sim.makeIMat()
    
    plt.imshow(sim.recon.interaction_matrix)
    plt.colorbar()
    plt.title('IM')
    plt.show()
    
    if ((np.isclose(np.max(sim.recon.interaction_matrix[ : 2]), 1, atol=1e-1) == False)
        or (np.isclose(np.max(sim.recon.interaction_matrix[2 : ]), 1, atol=1e-1) == False)):
        print('suggest tip tilt dm to have imat value of {:.3f}'.format(TT_iMatValue/np.max(sim.recon.interaction_matrix[:2])))
        print('suggest high order dm to have imat value of {:.3f}'.format(DM_iMatValue/np.max(sim.recon.interaction_matrix[2:])))
    
    _, svd, _ = np.linalg.svd(sim.recon.interaction_matrix)
    
    drop = np.where((svd[ : -1] - svd[1 : ]) == np.max(svd[2:-1] - svd[2 + 1 : ]))[0][0]
    if np.isclose((9.*svd[drop] + 1.*svd[drop + 1])/10./svd.max(), svdConditioning, atol=1e-1) == False:
        print('suggest svd at {:.3f}'.format((9.*svd[drop] + 1.*svd[drop + 2])/10./svd.max()))
    
    svd_drop = svd[drop]/svd.max()
    svd_max = 2*svd[drop]/svd.max()
    svd_min = svd[drop + 1]/svd.max()/2.
    
    plt.plot(svd/svd.max(),marker='o')
    plt.hlines(svd[drop]/svd.max(),0,len(svd),label='{:.3f}'.format(svd[drop]/svd.max()), color='m')
    plt.hlines(svd[drop + 1]/svd.max(),0,len(svd),label='{:.3f}'.format(svd[drop + 1]/svd.max()), color='y')
    plt.hlines(svdConditioning,0,len(svd),label='current svd')
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
    plt.title('DM')
    plt.show()
    plt.imshow(sim.sciCams[0].residual*sim.mask,vmin=MIN,vmax=MAX)
    plt.colorbar()
    plt.title('Residual WF')
    plt.show()
    plt.imshow(sim.sciCams[0].detector)
    plt.colorbar()
    plt.title('On-Axis PSF')
    plt.show()

    nb = input('Proceed?: ')
    if nb != 'y':
        quit()

if CHECK_SLODAR == True:
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

    sim.aoinit()

    N = sim.wfss[0].n_subaps

    if sim.wfss[1].n_subaps != N:
        print('error')

    if sim.config.wfss[0].GSPosition[0] - sim.config.wfss[1].GSPosition[0] == 0:
        print('wfs should be pointing differently on x direction')

    if sim.config.wfss[0].GSPosition[1] - sim.config.wfss[1].GSPosition[1] != 0:
        print('wfs should be pointing similarly on y direction')
        
    d = (sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)
    print('subapDiam={:.6f}m\n'.format(d))

    plt.plot(sim.wfss[0].pupil_subap_coords[:,0],
              sim.wfss[0].pupil_subap_coords[:,1],
              ls='',marker='o')
    plt.axis('square')
    plt.title('pupil')
    plt.show()
    
    recommend_looptime = d/sim.wfss[0].nx_subap_pixels/sim.config.atmos.windSpeeds.min()
    recommend_nIters = telDiam*10/sim.config.atmos.windSpeeds.min()/recommend_looptime
    if ((0.5*recommend_looptime > loopTime ) or (loopTime > 2*recommend_looptime)):
        print('recommend loopTime = {:f}'.format(recommend_looptime))
    if ((0.5*recommend_nIters > sim.config.sim.nIters) or (sim.config.sim.nIters > 2*recommend_nIters)):
        print('recommend nIters = {:f}'.format(recommend_nIters))
    nb = input('Proceed?: ')
    if nb != 'y':
        quit()

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

REFX = (refx[:,:, (SCREEN[0] - 1)//2]).T
REFY = (refy[:,:, (SCREEN[0] - 1)//2]).T

RANDOM = 5
REPEAT = 5
MAXR0 = 0.11
MINR0 = 0.02
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
sim.config.calcParams()
sim.makeIMat()
sim.config.dms[0].gain = 0.
sim.config.dms[1].gain = 0.

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
# eCx = np.zeros((N,N),dtype=float) # xy, wfs1subap, wfs2subap
Cy = np.zeros((N,N),dtype=float) # xy, wfs1subap, wfs2subap
# eCy = np.zeros((N,N),dtype=float) # xy, wfs1subap, wfs2subap
Cxscreen = np.zeros((2*mid + 1,2*mid + 1), dtype=float)
# Cxescreen = np.zeros((2*mid + 1,2*mid + 1), dtype=float)
# Cyscreen = np.zeros((2*mid + 1,2*mid + 1), dtype=float)
Cyescreen = np.zeros((2*mid + 1,2*mid + 1), dtype=float)

cn2x = np.zeros((RANDOM,REPEAT,LAYER),dtype=float)
cn2y = np.zeros((RANDOM,REPEAT,LAYER),dtype=float)

mean_cn2x = np.zeros((RANDOM,LAYER),dtype=float)
mean_cn2y = np.zeros((RANDOM,LAYER),dtype=float)
std_cn2x = np.zeros((RANDOM,LAYER),dtype=float)
std_cn2y = np.zeros((RANDOM,LAYER),dtype=float)
r0x = np.zeros((RANDOM),dtype=float)
r0y = np.zeros((RANDOM),dtype=float)
std_r0x = np.zeros((RANDOM),dtype=float)
std_r0y = np.zeros((RANDOM),dtype=float)
XscrnStrengths = np.zeros((RANDOM,LAYER),dtype=float)
YscrnStrengths = np.zeros((RANDOM,LAYER),dtype=float)
errorXscrnStrengths = np.zeros((RANDOM,LAYER),dtype=float)
errorYscrnStrengths = np.zeros((RANDOM,LAYER),dtype=float)

for random in tqdm(np.arange(RANDOM)):
    print('at random {:d}'.format(random))
    scrnStrengths = list_random_scrnStrengths[random]
    sim.config.atmos.r0 = list_random_r0[random]
    sim.config.atmos.scrnStrengths = scrnStrengths
    for repeat in tqdm(np.arange(REPEAT)):
        print('at repeat {:d}'.format(repeat))
        
        sim.config.randomSeed = np.random.randint(0, 999)
        sim.atmos = soapy.atmosphere.atmos(sim.config)
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
        
        Cx, eCx = my_cov(Sx1.T,Sx2.T)
        Cy, eCy = my_cov(Sy1.T,Sy2.T)
        # Cx = my_cov(Sx1.T,Sx2.T)
        # Cy = my_cov(Sy1.T,Sy2.T)
        Cxscreen, Cxescreen = overlay1(Cx, eCx, position)
        Cyscreen, Cyescreen = overlay1(Cy, eCy, position)
        # Cxscreen = overlay1(Cx, None, position)
        # Cyscreen = overlay1(Cy, None, position)
        CrossX = Cxscreen[:,mid]
        CrossY = Cyscreen[:,mid]
        # MAX = np.nanmax([Cxscreen,Cyscreen])
        # MIN = np.nanmin([Cxscreen,Cyscreen])
        # MAX2 = np.max([MAX,np.abs(MIN)])
        # MIN2 = -MAX2
        # plt.imshow(Cxscreen,vmin=MIN,vmax=MAX)
        # plt.colorbar()
        # plt.title('Slope covariance map (T)')
        # plt.show()
        # plt.imshow(Cyscreen,vmin=MIN,vmax=MAX)
        # plt.colorbar()
        # plt.title('Slope covariance map (L)')
        # plt.show()
    
        soln_cn2x = lsq_linear(REFX, CrossX)
        soln_cn2y = lsq_linear(REFY, CrossY)
        
        cn2x[random,repeat] = soln_cn2x.x * REFCN2
        cn2y[random,repeat] = soln_cn2y.x * REFCN2
        
        sim.reset_loop()
        
    mean_cn2x[random] = np.mean(cn2x[random],axis=0)
    std_cn2x[random] = np.std(cn2x[random],axis=0,ddof=1)
    total_cn2x = np.sum(mean_cn2x[random])
    std_total_cn2x = np.sqrt(np.sum((std_cn2x[random])**2))
    r0x[random] = cn22r0(total_cn2x)
    std_r0x[random] = stdcn22stdr0(total_cn2x,std_total_cn2x)
    
    mean_cn2y[random] = np.mean(cn2y[random],axis=0)
    std_cn2y[random] = np.std(cn2y[random],axis=0,ddof=1)
    total_cn2y = np.sum(mean_cn2y[random])
    std_total_cn2y = np.sqrt(np.sum((std_cn2y[random])**2))
    r0y[random] = cn22r0(total_cn2y)
    std_r0y[random] = stdcn22stdr0(total_cn2y,std_total_cn2y)
    
    XscrnStrengths[random] = mean_cn2x[random] / total_cn2x
    YscrnStrengths[random] = mean_cn2y[random] / total_cn2y
    errorXscrnStrengths[random] = np.sqrt(((std_cn2x[random]/mean_cn2x[random])**2
                                            + (std_total_cn2x/total_cn2x)**2)
                                          * XscrnStrengths[random]**2)
    errorYscrnStrengths[random] = np.sqrt(((std_cn2y[random]/mean_cn2y[random])**2
                                            + (std_total_cn2y/total_cn2y)**2)
                                          * YscrnStrengths[random]**2)
    # errorXscrnStrengths[random] = std_cn2x[random] / total_cn2x
    # errorYscrnStrengths[random] = std_cn2y[random] / total_cn2y
        
    plt.plot(list_random_scrnStrengths[random],label='Simulating',ls='-')
    plt.errorbar(np.arange(LAYER),
                 XscrnStrengths[random],
                 yerr=errorXscrnStrengths[random],
                 label='MeasuredX',capsize=5,ls='--',marker='^')
    plt.errorbar(np.arange(LAYER),
                 YscrnStrengths[random],
                 yerr=errorYscrnStrengths[random],
                 label='MeasuredY',capsize=5,ls='--',marker='v')
    plt.legend()
    plt.ylabel('normalized cn2')
    plt.xlabel('SLODAR layer')
    plt.title('Distribution measurement, r0 = {:.3f}m,\n'.format(list_random_r0[random])
              + 'measuredX = {:.3f}+-{:.3f}m,\n'.format(r0x[random],std_r0x[random])
              + 'measuredY = {:.3f}+-{:.3f}m'.format(r0y[random],std_r0y[random]))
    plt.show()
