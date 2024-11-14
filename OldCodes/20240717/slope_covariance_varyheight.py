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
#             ae[xcoord, ycoord] += eA[ii, jj]**2
#             count[xcoord, ycoord] += 1
#     a /= count
#     ae /= count
#     ae = np.sqrt(ae)
#     return a, ae

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
             + 'soapy-master/soapy-master/conf')
file = 'SLODAR1L.yaml'
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
RANDOM = 5

REFX = (refx[:,:, (SCREEN[0] - 1)//2]).T
REFY = (refy[:,:, (SCREEN[0] - 1)//2]).T

scrnHeights_list = (np.arange(LAYER, dtype=float)
                    * (sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)
                    / ((sim.config.wfss[0].GSPosition[0]
                        - sim.config.wfss[1].GSPosition[0])*ARCSEC2RAD))
sim.config.atmos.scrnHeights = scrnHeights_list

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

Cx = np.zeros((LAYER,N,N),dtype=float) # xy, wfs1subap, wfs2subap
eCx = np.zeros((LAYER,N,N),dtype=float) # xy, wfs1subap, wfs2subap
Cy = np.zeros((LAYER,N,N),dtype=float) # xy, wfs1subap, wfs2subap
eCy = np.zeros((LAYER,N,N),dtype=float) # xy, wfs1subap, wfs2subap



# Cx_all = np.zeros((LAYER,RANDOM,N,N),dtype=float) # xy, wfs1subap, wfs2subap
# Cy_all = np.zeros((LAYER,RANDOM,N,N),dtype=float) # xy, wfs1subap, wfs2subap
# for layer in tqdm(np.arange(LAYER)):
#     print(layer,'\n')
#     for random in tqdm(np.arange(RANDOM)):
#         print(random, '\n')
#         scrnHeights = scrnHeights_list[layer]
#         sim.config.atmos.scrnHeights = [scrnHeights]
#         sim.config.randomSeed = np.random.randint(0, 999)
#         sim.atmos = soapy.atmosphere.atmos(sim.config)
#         sim.aoinit()
#         sim.config.calcParams()
#         sim.makeIMat()
#         sim.config.dms[0].gain = 0.
#         sim.config.dms[1].gain = 0.
#         sim.aoloop()
        
#         Sx1 = sim.allSlopes[ : , 0*N : (0+1)*N]
#         Sy1 = sim.allSlopes[ : , 1*N : (1+1)*N]
#         Sx2 = sim.allSlopes[ : , 2*N : (2+1)*N]
#         Sy2 = sim.allSlopes[ : , 3*N : (3+1)*N]
#         TIME = Sx1.shape[0]
#         for time in np.arange(TIME):
#             Sx1[time] -= np.mean(Sx1[time])
#             Sy1[time] -= np.mean(Sy1[time])
#             Sx2[time] -= np.mean(Sx2[time])
#             Sy2[time] -= np.mean(Sy2[time])
        
#         if random == 0:
#             for time in np.arange(np.min([TIME,5])):
#                 plt.quiver(position[:,0],position[:,1],Sx1[time],Sy1[time],color='r')
#                 plt.quiver(position[:,0],position[:,1],Sx2[time],Sy2[time],color='b')
#                 plt.axis('square')
#                 plt.title(layer)
#                 plt.show()
                
#         Cx_all[layer,random] = my_cov(Sx1.T,Sx2.T)
#         Cy_all[layer,random] = my_cov(Sy1.T,Sy2.T)
        
#         sim.reset_loop()

# Cx = np.mean(Cx_all,axis=1)
# eCx = np.std(Cx_all,axis=1,ddof=1)
# Cy = np.mean(Cy_all,axis=1)
# eCy = np.std(Cy_all,axis=1,ddof=1)

# name = 'Cxheight'
# hdu = fits.PrimaryHDU(Cx)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()
# name = 'exCheight'
# hdu = fits.PrimaryHDU(eCx)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()
# name = 'Cyheight'
# hdu = fits.PrimaryHDU(Cy)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()
# name = 'eCyheight'
# hdu = fits.PrimaryHDU(eCy)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()

name = 'Cxheight'
hdulist = fits.open(name + '.fits')
hdu = hdulist[0]
Cx = hdu.data
hdulist.close()
name = 'eCxheight'
hdulist = fits.open(name + '.fits')
hdu = hdulist[0]
eCx = hdu.data
hdulist.close()
name = 'Cyheight'
hdulist = fits.open(name + '.fits')
hdu = hdulist[0]
Cy = hdu.data
hdulist.close()
name = 'eCyheight'
hdulist = fits.open(name + '.fits')
hdu = hdulist[0]
eCy = hdu.data
hdulist.close()

name = reffile
hdulist = fits.open(name + '.fits')
hdu = hdulist[0]
ref = np.flip(hdu.data, axis=1)
hdulist.close()
ref[ref==0] = np.nan

refx = ref[::2]
refy = ref[1::2]

Cxscreen = np.zeros((LAYER,2*mid + 1,2*mid + 1), dtype=float)
Cxescreen = np.zeros((LAYER,2*mid + 1,2*mid + 1), dtype=float)
Cyscreen = np.zeros((LAYER,2*mid + 1,2*mid + 1), dtype=float)
Cyescreen = np.zeros((LAYER,2*mid + 1,2*mid + 1), dtype=float)
CrossX = np.zeros((LAYER,2*mid + 1), dtype=float)
CrossY = np.zeros((LAYER,2*mid + 1), dtype=float)

# for layer in np.arange(LAYER):
#     plt.imshow(Cx[layer].T)
#     plt.title('x covariance matrix {:d}'.format(layer))
#     plt.colorbar()
#     plt.show()
#     plt.imshow(Cy[layer].T)
#     plt.title('y covariance matrix {:d}'.format(layer))
#     plt.colorbar()
#     plt.show()
    

for layer in np.arange(LAYER):
    Cxscreen[layer], Cxescreen[layer] = overlay1(Cx[layer], eCx[layer], position)
    Cyscreen[layer], Cyescreen[layer] = overlay1(Cy[layer], eCy[layer], position)
    
    # plt.pcolor(biggrid,biggrid,Cxscreen[layer].T)
    # plt.title('x covariance map {:d}'.format(layer))
    # plt.colorbar()
    # plt.axis('square')
    # plt.show()
    # plt.pcolor(biggrid,biggrid,Cyscreen[layer].T)
    # plt.title('y covariance map {:d}'.format(layer))
    # plt.colorbar()
    # plt.axis('square')
    # plt.show()
    
CrossX = Cxscreen[:,:,mid]
CrossY = Cyscreen[:,:,mid]
CrossXe = Cxescreen[:,:,mid]
CrossYe = Cyescreen[:,:,mid]
CrossRefX = refx[:,:,mid]
CrossRefY = refy[:,:,mid]

for layer in np.arange(LAYER):
    if layer == 0:
        plt.errorbar(biggrid,CrossX[layer],yerr=CrossXe[layer],color=color_list[layer],capsize=5,marker='^',ls='--',label='simulation')
        plt.plot(biggrid,CrossRefX[layer],color=color_list[layer],marker='o',ls='-',label='theoretical')
    else:
        plt.errorbar(biggrid,CrossX[layer],yerr=CrossXe[layer],color=color_list[layer],capsize=5,marker='^',ls='--')
        plt.plot(biggrid,CrossRefX[layer],color=color_list[layer],marker='o',ls='-')
plt.legend(loc='upper left')
plt.title('Cross Section of X Slope Covariance Map')
plt.show()

for layer in np.arange(LAYER):
    if layer == 0:
        plt.errorbar(biggrid,CrossY[layer],yerr=CrossYe[layer],color=color_list[layer],capsize=5,marker='^',ls='--',label='simulation')
        plt.plot(biggrid,CrossRefY[layer],color=color_list[layer],marker='o',ls='-',label='theoretical')
    else:
        plt.errorbar(biggrid,CrossY[layer],yerr=CrossYe[layer],color=color_list[layer],capsize=5,marker='^',ls='--')
        plt.plot(biggrid,CrossRefY[layer],color=color_list[layer],marker='o',ls='-')
plt.legend(loc='upper left')
plt.title('Cross Section of Y Slope Covariance Map')
plt.show()



# MAX = np.nanmax([np.nanmax(refx),np.nanmax(refy)])
# MIN = np.nanmin([np.nanmin(refx),np.nanmin(refy)])
# MAX2 = np.max([np.abs(MAX),np.abs(MIN)])/10.
# MIN2 = -MAX2


# # for layer in np.arange(LAYER):
# #     # print(layer,np.where(Cscreen[layer,0]==np.nanmax(Cscreen[layer,0])))
# #     plt.pcolor(biggrid,biggrid,Cscreen[layer,0].T,vmin=MIN,vmax=MAX)
# #     plt.axis('square')
# #     plt.colorbar()
# #     plt.xlabel('x = first index')
# #     plt.ylabel('y = second index')
# #     plt.title('Slope cross-covariance (L) {:d}'.format(layer))
# #     plt.show()
# # for i in np.arange(refx.shape[0]):
# #     plt.pcolor(biggrid,biggrid,refx[i].T,vmin=MIN,vmax=MAX)
# #     plt.axis('square')
# #     plt.xlabel('x = first index')
# #     plt.ylabel('y = second index')
# #     plt.title('x reference function {:d}'.format(i))
# #     plt.colorbar()
# #     plt.show()
# for i in np.arange(refx.shape[0]//3)*3:
#     plt.pcolor(biggrid,biggrid,(Cscreen[i,0] - refx[i]).T,vmin=MIN2,vmax=MAX2,cmap='RdBu')
#     plt.axis('square')
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('L residual slope covariance {:d}'.format(i))
#     plt.colorbar()
#     plt.show()
# # for i in np.arange(refx.shape[0]):
# #     plt.pcolor(biggrid,biggrid,((Cscreen[i,0] - refx[i])/Cescreen[i,0]).T,vmin=-3,vmax=3,cmap='RdBu')
# #     plt.axis('square')
# #     plt.xlabel('x = first index')
# #     plt.ylabel('y = second index')
# #     plt.title('residual x slope covariance {:d} in sd unit'.format(i))
# #     plt.colorbar()
# #     plt.show()


# # for layer in np.arange(LAYER):
# #     # print(layer,np.where(Cscreen[layer,1]==np.nanmax(Cscreen[layer,1])))
# #     plt.pcolor(biggrid,biggrid,Cscreen[layer,1].T,vmin=MIN,vmax=MAX)
# #     plt.axis('square')
# #     plt.colorbar()
# #     plt.xlabel('x = first index')
# #     plt.ylabel('y = second index')
# #     plt.title('Slope cross-covariance (T) {:d}'.format(layer))
# #     plt.show()
# # for i in np.arange(refy.shape[0]):
# #     plt.pcolor(biggrid,biggrid,refy[i].T,vmin=MIN,vmax=MAX)
# #     plt.axis('square')
# #     plt.xlabel('x = first index')
# #     plt.ylabel('y = second index')
# #     plt.title('y reference function {:d}'.format(i))
# #     plt.colorbar()
# #     plt.show()
# for i in np.arange(refy.shape[0]//3)*3:
#     plt.pcolor(biggrid,biggrid,(Cscreen[i,1] - refy[i]).T,vmin=MIN2,vmax=MAX2,cmap='RdBu')
#     plt.axis('square')
#     plt.xlabel('x = first index')
#     plt.ylabel('y = second index')
#     plt.title('T residual slope covariance {:d}'.format(i))
#     plt.colorbar()
#     plt.show()
# # for i in np.arange(refy.shape[0]):
# #     plt.pcolor(biggrid,biggrid,((Cscreen[i,1] - refy[i])/Cescreen[i,1]).T,vmin=-3,vmax=3,cmap='RdBu')
# #     plt.axis('square')
# #     plt.xlabel('x = first index')
# #     plt.ylabel('y = second index')
# #     plt.title('residual y slope covariance {:d} in sd unit'.format(i))
# #     plt.colorbar()
# #     plt.show()




    



# #
# # cross section of covariance map
# # only x is used
# # y cross section is not so helpful as it is not in the direction where wfs point differently
# # 

# xCrossCx = Cscreen[:,0,:,mid]
# xCrossCy = Cscreen[:,1,:,mid]
# # xCrossAx = Ascreen[:,0,:,mid]
# # xCrossAy = Ascreen[:,1,:,mid]

# exCrossCx = Cescreen[:,0,:,mid]
# exCrossCy = Cescreen[:,1,:,mid]
# # exCrossAx = Aescreen[:,0,:,mid]
# # exCrossAy = Aescreen[:,1,:,mid]

# xCrossRefCx = refx[:,:,mid]
# xCrossRefCy = refy[:,:,mid]

# yCrossCy = np.zeros_like(xCrossCy)
# eyCrossCy = np.zeros_like(xCrossCy)
# yCrossRefCy = np.zeros_like(xCrossCy)
# for layer in np.arange(LAYER):
#     yCrossCy[layer] = Cscreen[layer,1,mid+layer,:]
#     eyCrossCy[layer] = Cescreen[layer,1,mid+layer,:]
#     yCrossRefCy[layer] = refy[layer,mid+layer,:]
# yCrossCx = np.zeros_like(xCrossCy)
# eyCrossCx = np.zeros_like(xCrossCy)
# yCrossRefCx = np.zeros_like(xCrossCy)
# for layer in np.arange(LAYER):
#     yCrossCx[layer] = Cscreen[layer,0,mid+layer,:]
#     eyCrossCx[layer] = Cescreen[layer,0,mid+layer,:]
#     yCrossRefCx[layer] = refx[layer,mid+layer,:]


# color_list = ['r','g','b','c','m','y','k','tab:purple']

# for layer in np.arange(LAYER//2)*2:
#     if layer == 0:
#         plt.errorbar(biggrid,xCrossCx[layer],yerr=exCrossCx[layer],
#                       label='measured',
#                       marker='o',capsize=5,ls='',color=color_list[layer])
#         plt.plot(biggrid,xCrossRefCx[layer],ls='--',color=color_list[layer],label='ref')
#     else:
#         plt.errorbar(biggrid,xCrossCx[layer],yerr=exCrossCx[layer],
#                       marker='o',capsize=5,ls='',color=color_list[layer])
#         plt.plot(biggrid,xCrossRefCx[layer],ls='--',color=color_list[layer])
# plt.title('Slope cross-covariance (L), {:d} frames'.format(sim.config.sim.nIters))
# plt.legend(loc='upper left')
# plt.show()
# for layer in np.arange(LAYER//2)*2:
#     plt.errorbar(biggrid,(xCrossCx[layer]-xCrossRefCx[layer]),yerr=exCrossCx[layer],
#                   marker='o',ls='-',color=color_list[layer],capsize=5)
# plt.hlines(0,biggrid.min(),biggrid.max(),ls='--')
# plt.title('Residual of Slope cross-covariance (L), {:d} frames'.format(sim.config.sim.nIters))
# plt.show()

# # for layer in np.arange(LAYER):    
# #     plt.errorbar(biggrid,yCrossCx[layer],yerr=eyCrossCx[layer],
# #                   label='layer{:.0f}'.format(layer),
# #                   marker='o',capsize=5,ls='',color=color_list[layer],alpha=0.5)
# #     plt.plot(biggrid,yCrossRefCx[layer],ls='--',color=color_list[layer])
# # # plt.title('Slope cross-covariance (L), {:d} frames'.format(sim.config.sim.nIters))
# # plt.legend(loc='upper left')
# # plt.show()
# # for layer in np.arange(LAYER):    
# #     plt.errorbar(biggrid,(yCrossCx[layer]-yCrossRefCx[layer]),yerr=eyCrossCx[layer],
# #                   label='layer{:.0f}'.format(layer),
# #                   marker='o',ls='-',color=color_list[layer],alpha=0.5,capsize=5)
# # plt.hlines(0,biggrid.min(),biggrid.max(),ls='--')
# # # plt.title('Residual of Slope cross-covariance (L), {:d} frames'.format(sim.config.sim.nIters))
# # plt.legend(loc='upper left')
# # plt.show()



# for layer in np.arange(LAYER//2)*2:    
#     if layer == 0:
#         plt.errorbar(biggrid,xCrossCy[layer],yerr=exCrossCy[layer],
#                       label='measured',
#                       marker='o',capsize=5,ls='',color=color_list[layer])
#         plt.plot(biggrid,xCrossRefCy[layer],ls='--',color=color_list[layer],label='ref')
#     else:
#         plt.errorbar(biggrid,xCrossCy[layer],yerr=exCrossCy[layer],
#                       marker='o',capsize=5,ls='',color=color_list[layer])
#         plt.plot(biggrid,xCrossRefCy[layer],ls='--',color=color_list[layer])
# plt.title('Slope cross-covariance (T), {:d} frames'.format(sim.config.sim.nIters))
# plt.legend(loc='upper left')
# plt.show()
# for layer in np.arange(LAYER//2)*2:    
#     plt.errorbar(biggrid,(xCrossCy[layer]-xCrossRefCy[layer]),yerr=exCrossCy[layer],
#                   marker='o',ls='-',color=color_list[layer],capsize=5)
# plt.title('Residual of Slope cross-covariance (T), {:d} frames'.format(sim.config.sim.nIters))
# plt.hlines(0,biggrid.min(),biggrid.max(),ls='--')
# plt.legend(loc='upper left')
# plt.show()

# # for layer in np.arange(LAYER):    
# #     plt.errorbar(biggrid,yCrossCy[layer],yerr=eyCrossCy[layer],
# #                   label='layer{:.0f}'.format(layer),
# #                   marker='o',capsize=5,ls='',color=color_list[layer],alpha=0.5)
# #     plt.plot(biggrid,yCrossRefCy[layer],ls='--',color=color_list[layer])
# # # plt.title('Slope cross-covariance (T), {:d} frames'.format(sim.config.sim.nIters))
# # plt.legend(loc='upper left')
# # plt.show()
# # for layer in np.arange(LAYER):    
# #     plt.errorbar(biggrid,(yCrossCy[layer]-yCrossRefCy[layer]),yerr=eyCrossCy[layer],
# #                   label='layer{:.0f}'.format(layer),
# #                   marker='o',ls='-',color=color_list[layer],alpha=0.5,capsize=5)
# # plt.hlines(0,biggrid.min(),biggrid.max(),ls='--')
# # # plt.title('Residual of Slope cross-covariance (T), {:d} frames'.format(sim.config.sim.nIters))
# # plt.legend(loc='upper left')
# # plt.show()





# # # next file?

# # color_list = ['r','g','b','c','m','y','k','tab:purple']
# # # NORMALIZED = np.max([np.nanmax(np.abs(Cscreen)),np.nanmax(np.abs(refx)),np.nanmax(np.abs(refy))])
# # Cscreen = Cscreen#/NORMALIZED
# # refx = refx#/NORMALIZED
# # refy = refy#/NORMALIZED

# # REFR0 = sim.atmos.r0
# # REFCN2 = r02cn2(REFR0)

# # LAYER = refy.shape[0]
# # SCREEN = (refy.shape[1],refy.shape[2])
# # FLAT = refy.shape[1]*refy.shape[2]

# # xres = np.zeros((LAYER,LAYER))

# # MAX = np.nanmax(Cscreen)
# # MIN = np.nanmin(Cscreen)
# # MAX2 = np.max([MAX,np.abs(MIN)])
# # MIN2 = -MAX2

# # A = np.nan_to_num((refy.reshape(LAYER, FLAT)).T)
# # for layer in np.arange(LAYER):
# #     b = np.nan_to_num(Cscreen[layer,1].reshape(FLAT))
# #     res = lsq_linear(A, b, bounds=(0.01,100))
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

# # # A = np.nan_to_num((refy.reshape(LAYER, FLAT)).T)
# # # for layer in np.arange(LAYER):
# # #     b = np.nan_to_num(Cscreen[layer,1].reshape(FLAT))
# # #     res = lsq_linear(A, b, bounds=(0.01,100))
# # #     xres[layer] = cn22r0(res.x * REFCN2)
# # #     x0 = np.zeros((LAYER))
# # #     x0[layer] = 1 * REFCN2
# # #     x0 = cn22r0(x0)
# # #     plt.plot(xres[layer],ls='--',label='{:d}'.format(layer),color=color_list[layer],marker='v')
# # #     plt.plot(x0,ls='-',color=color_list[layer],marker='^')
# # # plt.title('measured r0')
# # # plt.legend()
# # # # plt.yscale('log')
# # # plt.show()

# # # test fitting with ref functyion
# # for layer in np.arange(LAYER):
# #     x0 = np.zeros((LAYER))
# #     x0[layer] = 1 * REFCN2
# #     plt.plot(xres[layer] - x0,ls='-',label='{:d}'.format(layer),color=color_list[layer],marker='o')
# #     # plt.plot(x0,ls='-',color=color_list[layer],marker='x')
# # plt.title('cn2 measurement error, refcn2={:.3e}'.format(REFCN2))
# # plt.legend()
# # plt.show()


# # # with reference function
# # NTEST = 10
# # A = np.nan_to_num((refy.reshape(LAYER, FLAT)).T)
# # for ntest in np.arange(NTEST):
# #     random_coeff = np.random.rand(8)*2
# #     test_func = np.matmul((refy.reshape(LAYER,FLAT)).T,random_coeff)
# #     b = np.nan_to_num((test_func).reshape(FLAT))
# #     res = lsq_linear(A, b, bounds=(0.01,100))
# #     # BRecon = np.matmul(A,res.x)
# #     # plt.imshow(b.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# #     # plt.title('Measured')
# #     # plt.colorbar()
# #     # plt.show()
# #     # plt.imshow(BRecon.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# #     # plt.title('Reconstructed')
# #     # plt.colorbar()
# #     # plt.show()
# #     # plt.imshow(res.fun.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# #     # plt.title('Residual {:f}'.format((res.fun[res.fun>0]).std()))
# #     # plt.colorbar()
# #     # plt.show()
# #     plt.plot(res.x,label='output',marker='o')
# #     plt.plot(random_coeff,label='input',marker='x')
# #     plt.title('Test LSQ recon')
# #     plt.legend()
# #     plt.show()
    
    
    
# # # with simulation??? ==> next file
# # NTEST = 10
# # A = np.nan_to_num((refy.reshape(LAYER, FLAT)).T)
# # for ntest in np.arange(NTEST):
# #     random_coeff = np.random.rand(8)*2
# #     test_func = np.matmul((Cscreen[:,1,:,:].reshape(LAYER,FLAT)).T,random_coeff)
# #     b = np.nan_to_num((test_func).reshape(FLAT))
# #     res = lsq_linear(A, b, bounds=(0.01,100))
# #     # BRecon = np.matmul(A,res.x)
# #     # plt.imshow(b.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# #     # plt.title('Measured')
# #     # plt.colorbar()
# #     # plt.show()
# #     # plt.imshow(BRecon.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# #     # plt.title('Reconstructed')
# #     # plt.colorbar()
# #     # plt.show()
# #     # plt.imshow(res.fun.reshape(SCREEN),cmap='RdBu',vmax=MAX2,vmin=MIN2)
# #     # plt.title('Residual {:f}'.format((res.fun[res.fun>0]).std()))
# #     # plt.colorbar()
# #     # plt.show()
# #     plt.plot(res.x,label='output',marker='o')
# #     plt.plot(random_coeff,label='input',marker='x')
# #     plt.title('Residual {:f}'.format(((res.fun[res.fun**2>0])**2).sum()**0.5))
# #     plt.legend()
# #     plt.show()
