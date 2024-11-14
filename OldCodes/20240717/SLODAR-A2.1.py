# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:05:37 2023

@author: gqmc62
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:30:29 2023

@author: gqmc62
"""

###############################################################################
########################################################################

import os
import soapy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from astropy.io import fits
from scipy.optimize import lsq_linear
from scipy.stats import bootstrap
from tqdm import tqdm
import time
# import my_WFS as mwfs
import my_phasescreen as mps
from aotools.functions import zernike
import my_propagation as mprop
import copy
import wfsnoiselib

ARCSEC2RAD = 1./3600./360.*2.*np.pi
RAD2ARCSEC = 1./ARCSEC2RAD

def lsq_cov2cn2(refcov,refcn2,cov,cov_noise,n_bootstrap=100):
    RESULT = np.zeros((n_bootstrap,refcov.shape[1]),dtype=float)
    for n in np.arange(n_bootstrap):
        noisy_cov = np.random.normal(cov,cov_noise)
        res = lsq_linear(refcov, noisy_cov, bounds=(0,np.inf))
        RESULT[n] = res.x
    cn2 = np.nanmean(RESULT,axis=0) * refcn2
    cn2_noise = np.nanstd(RESULT,axis=0,ddof=1) *refcn2
    r0 = cn22r0(cn2.sum())
    r0_noise = stdcn22stdr0(cn2.sum(),cn2_noise.sum())
    return cn2, cn2_noise, r0, r0_noise

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

def overlay1(A, eA, position):
    a = np.zeros((2*position.max() + 1, 2*position.max() + 1), dtype=float)
    ea = np.zeros((2*position.max() + 1, 2*position.max() + 1), dtype=float)
    count = np.zeros((2*position.max() + 1, 2*position.max() + 1), dtype=float)
    N = len(A)
    mid = position.max()
    for ii in np.arange(N):
        for jj in np.arange(N):
            xcoord = mid + (position[jj, 0] - position[ii, 0])
            ycoord = mid + (position[jj, 1] - position[ii, 1])
            a[xcoord, ycoord] += A[ii, jj]/eA[ii,jj]**2
            ea[xcoord, ycoord] += 1/eA[ii,jj]**2
            count[xcoord, ycoord] += 1
        # plt.imshow(a)
        # plt.show()
        # a*=0
    a /= ea
    ea = 1/np.sqrt(ea)
    a[count == 0] = np.nan
    ea[count == 0] = np.nan
    return a, ea

def my_cov(A, B):
    N = A.shape[0]
    TIME = A.shape[1]
    C = np.zeros((N, N), dtype=float)
    eC = np.zeros((N, N), dtype=float)
    A = (A.T - np.mean(A,axis=1)).T
    B = (B.T - np.mean(B,axis=1)).T
    for ii in np.arange(N):
        for jj in np.arange(N):
            temp = A[ii]*B[jj]
            C[ii,jj] = np.mean(temp)
            eC[ii,jj] = np.std(temp)
    eC /= np.sqrt(TIME)
    return C, eC

def r02cn2(r0,wvl=500e-9,zenith_angle=0.):
    cn2 = r0**(-5./3.)/(0.423*(2.*np.pi/wvl)**2/np.cos(zenith_angle))
    return cn2

def cn22r0(cn2,wvl=500e-9,zenith_angle=0.):
    r0 = (0.423*(2.*np.pi/wvl)**2/np.cos(zenith_angle)*cn2)**(-3./5.)
    return r0

def loadTIMBRefCov_withcentre(HOMEDIR,REFFILE):
    os.chdir(HOMEDIR)
    hdulist = fits.open(REFFILE + '.fits')
    hdu = hdulist[0]
    REF = np.flip(hdu.data, axis=1)
    hdulist.close()
    REF[REF==0] = np.nan
    REFX = REF[::2]
    REFY = REF[1::2]
    LAYER = REFX.shape[0]
    FLAT = REFX.shape[1] * REFX.shape[2]
    AY = np.nan_to_num((REFY.reshape(LAYER, FLAT)).T)
    AX = np.nan_to_num((REFX.reshape(LAYER, FLAT)).T)
    # for layer in np.arange(LAYER):
    #     (AX[:,layer])[AX[:,layer] == np.nanmax(AX[:,layer])] = 0
    #     (AY[:,layer])[AY[:,layer] == np.nanmax(AY[:,layer])] = 0
    A = np.zeros((2*FLAT, LAYER), dtype=float)
    A[: FLAT] = AX
    A[FLAT : 2*FLAT] = AY
    return A

def stdcn22stdr0(cn2,stdcn2,wvl=500e-9,zenith_angle=0.):
    stdr0 = np.abs((0.423*(2.*np.pi/wvl)**2/np.cos(zenith_angle))**(-3./5.)
                   * (-3./5.) * cn2**(-8./5.)) * stdcn2
    return stdr0

def cov2cn2(pinv_refcov,refcn2,cov,cov_noise):
    cn2 = (pinv_refcov @ np.nan_to_num(cov)) * refcn2
    r0 = cn22r0(cn2.sum())
    cn2_noise = np.sqrt((pinv_refcov**2) @ (np.nan_to_num(cov_noise**2))) * refcn2
    r0_noise = stdcn22stdr0(cn2.sum(),cn2_noise.sum())
    return cn2, cn2_noise, r0, r0_noise


def make_cov_map(soapy_slope):
    WFS_open = soapy_slope
    
    
    cov_x_open = np.zeros((N, N), dtype=float)
    cov_y_open = np.zeros((N, N), dtype=float)
    
    ecov_x_open = np.zeros((N, N), dtype=float)
    ecov_y_open = np.zeros((N, N), dtype=float)
    
    map_cov_x_open = np.zeros((2*mid + 1, 2*mid + 1), dtype=float)
    map_cov_y_open = np.zeros((2*mid + 1, 2*mid + 1), dtype=float)
    
    emap_cov_x_open = np.zeros((2*mid + 1, 2*mid + 1), dtype=float)
    emap_cov_y_open = np.zeros((2*mid + 1, 2*mid + 1), dtype=float)
    
    Sx1_open = WFS_open[ : , (0)*N : (1)*N]
    Sy1_open = WFS_open[ : , (1)*N : (2)*N]
    Sx2_open = WFS_open[ : , (2)*N : (3)*N]
    Sy2_open = WFS_open[ : , (3)*N : (4)*N]
    
    Sx1_open = (Sx1_open.T - np.mean(Sx1_open, axis=1)).T
    Sy1_open = (Sy1_open.T - np.mean(Sy1_open, axis=1)).T
    Sx2_open = (Sx2_open.T - np.mean(Sx2_open, axis=1)).T
    Sy2_open = (Sy2_open.T - np.mean(Sy2_open, axis=1)).T
    
    cov_x_open, ecov_x_open = my_cov(Sx1_open.T, Sx2_open.T)
    cov_y_open, ecov_y_open = my_cov(Sy1_open.T, Sy2_open.T)
    
    map_cov_x_open, emap_cov_x_open = overlay1(cov_x_open, ecov_x_open, position)
    map_cov_y_open, emap_cov_y_open = overlay1(cov_y_open, ecov_y_open, position)
    
    return map_cov_x_open, emap_cov_x_open, map_cov_y_open, emap_cov_y_open

###############################################################################
########################################################################

GSMag_list = np.array([np.nan,3,5], dtype=float)
readout_list = np.array([0,4,8], dtype=float)


GOOD_SNR = np.zeros((GSMag_list.shape[0],readout_list.shape[0]), dtype=bool)

REFR0 = 0.05
REFTOTALCN2 = r02cn2(REFR0)
MAXR0 = 0.11
MINR0 = 0.02

home = ('C:/Users/gqmc62/OneDrive - Durham University/Documents/PhD/Codes/conf')
os.chdir(home)

LOAD = False

# nb = input('LOAD = {:} ... Proceed?: '.format(LOAD))
# if nb != 'y':
#     quit()

file = 'SLODAR6L_withNoise.yaml'
sim = soapy.Sim(file)
RANDOMSEED = np.random.randint(0, 2**31 - 1)
sim.config.atmos.randomSeed = RANDOMSEED
sim.config.sim.nIters = 1000
LAYER = sim.config.atmos.scrnNo
TIME = sim.config.sim.nIters

WFSPX2NM = (sim.config.wfss[0].subapFOV/sim.config.wfss[0].pxlsPerSubap) \
    *ARCSEC2RAD*(sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)/1e-9
WFSPX2OPDRAD = (sim.config.wfss[0].subapFOV/sim.config.wfss[0].pxlsPerSubap) \
    * ARCSEC2RAD*(sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps) \
        * (2.*np.pi/sim.config.wfss[0].wavelength)

SAVENAME = ('SLODARA2'
            + '_{:d}Frames'.format(TIME)
            + '_L0{:.0f}m'.format(sim.config.atmos.L0[0])
            + '_D{:.2f}m'.format(sim.config.tel.telDiam)
            + '_d{:.0f}cm'.format(100*sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps))

###############################################################################
########################################################################
r0_list = np.array([0.02,0.05,0.08,0.11], dtype=float)
measurer0_withcentre = np.zeros_like(r0_list)
measurer0_nocentre = np.zeros_like(r0_list)
measurer0noise_withcentre = np.zeros_like(r0_list)
measurer0noise_nocentre = np.zeros_like(r0_list)




REFR0 = 0.05
# if LOAD == False:    
RANDOMDISTR = np.random.rand((LAYER))
sim.config.atmos.scrnStrengths = RANDOMDISTR
sim.config.atmos.scrnStrengths /= sim.config.atmos.scrnStrengths.sum()
sim.config.atmos.r0 = REFR0
scrnHeights_list = (np.arange(sim.config.wfss[0].nxSubaps - 2, dtype=float)
                    * (sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)
                    / ((sim.config.wfss[0].GSPosition[0]
                        - sim.config.wfss[1].GSPosition[0])*ARCSEC2RAD))
sim.config.atmos.scrnHeights = scrnHeights_list

sim.aoinit()
sim.config.calcParams()
sim.makeIMat()
sim.allDmCommands = np.zeros((sim.config.sim.nIters, sim.config.sim.totalActs))

N = sim.wfss[0].n_subaps

position = np.asarray(sim.wfss[0].pupil_subap_coords/sim.wfss[0].nx_subap_pixels,dtype=int) - 1

mid = sim.config.wfss[0].nxSubaps - 1 - 1 - 1

pixel_scale = sim.config.tel.telDiam / sim.config.sim.pupilSize

grid = np.arange(mid + 1,dtype=float)
grid -= len(grid)/2
grid += 0.5

ALPHA1 = 0.5
ALPHA2 = 0.5
saveFigure = True



D = sim.config.tel.telDiam
L0 = sim.config.atmos.L0[0]
l0 = 1e-10
r0 = sim.config.atmos.r0
n_modes = 32 + 1
NPXLSCRN = (sim.config.sim.pupilSize)
ovf = sim.config.sim.simOversize
initial_flush = int(np.ceil(L0/D/ovf))

if LOAD == False:

    ##############################################################################
    #######################################################################
    
    ALLSLOPES = np.zeros((r0_list.shape[0],GSMag_list.shape[0],readout_list.shape[0],TIME,4*N), dtype=float)
    
    
    for i in np.arange(len(sim.dms)):
        sim.dms[i].gain = 0.
        sim.recon.dms[i].gain = 0.
        sim.dms[i].dmConfig.gain = 0.
    
    for layer in np.arange(LAYER):
        for row in np.arange(sim.atmos.infinite_phase_screens[layer].scrn.shape[0]*initial_flush):
            sim.atmos.infinite_phase_screens[layer].add_row()
    
    for iFrame in np.arange(TIME):
        t = time.time()
        
        REFSCREENS = sim.atmos.moveScrns()
        
        sim.scrns = REFSCREENS
        
        sim.Tatmos += time.time() - t
    
        # Run Loop...
        ########################################
    
        # Get dmCommands from reconstructor
        t_recon = time.time()
        if sim.config.sim.nDM:
            sim.dmCommands[:] = sim.recon.reconstruct(sim.slopes)
        sim.Trecon += (time.time() - t_recon)
    
        # Delay the dmCommands if loopDelay is configured
        sim.dmCommands = sim.buffer.delay(sim.dmCommands, sim.config.sim.loopDelay)
    
        # Get dmShape from closed loop DMs
        sim.closed_correction = sim.runDM(
                sim.dmCommands, closed=True)
    
        # for open loop only!!!
        for r0_index in np.arange(r0_list.shape[0]):
            r0 = r0_list[r0_index]
            R0CONVERT = (r0/REFR0)**(-5./6.)
            sim.scrns = REFSCREENS * R0CONVERT
            for GSMag_index in np.arange(GSMag_list.shape[0]):
                GSMag = GSMag_list[GSMag_index]
                for iwfs in np.arange(len(sim.wfss)):
                    sim.wfss[iwfs].config.GSMag = GSMag
                    if np.isnan(GSMag):
                        sim.wfss[iwfs].config.GSMag = np.nanmin(GSMag_list)
                        sim.wfss[iwfs].config.photonNoise = False
                    else:
                        sim.wfss[iwfs].config.photonNoise = True
                if np.isnan(GSMag):
                    GSMag = np.nanmin(GSMag_list)
                avg_photons = soapy.wfs.shackhartmann.photons_per_mag(
                    GSMag, sim.wfss[0].mask, sim.wfss[0].phase_scale,
                    sim.wfss[0].config.exposureTime, sim.wfss[0].config.photometric_zp)/sim.wfss[0].n_subaps
                for readout_index in np.arange(readout_list.shape[0]):
                    readout = readout_list[readout_index]
                    TRUE_READOUT = readout*avg_photons/sim.wfss[0].nx_subap_pixels**2
                    for iwfs in np.arange(len(sim.wfss)):
                        sim.wfss[iwfs].config.eReadNoise = TRUE_READOUT
                    
                    ALLSLOPES[r0_index, GSMag_index, readout_index, iFrame] = sim.runWfs(
                        dmShape=sim.closed_correction,loopIter=sim.iters)
                    
                    if r0 == REFR0:
                        # Run WFS, with closed loop DM shape applied
                        sim.slopes = ALLSLOPES[r0_index, GSMag_index, readout_index, iFrame]
                        
    
        
    
        # Get DM shape for open loop DMs
        sim.open_correction = sim.runDM(sim.dmCommands,
                                          closed=False)
    
        # Pass whole combined DM shapes to science target
        sim.combinedCorrection = sim.open_correction + sim.closed_correction
    
        sim.runSciCams(sim.combinedCorrection)
    
        # Save Data
        i = sim.iters % sim.config.sim.nIters # If sim is run continuously in loop, overwrite oldest data in buffer
        sim.storeData(i)
    
        sim.printOutput(sim.iters, strehl=True)
    
        sim.addToGuiQueue()
    
        sim.iters += 1
    
    sim.saveData()
    sim.finishUp()
    
    
    name = SAVENAME + '_ALLSLOPES'
    hdu = fits.PrimaryHDU(ALLSLOPES)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + '.fits', overwrite=True)
    hdul.close()
    
    ALLCOVMAPX = np.zeros((r0_list.shape[0],
                           GSMag_list.shape[0],
                           readout_list.shape[0],
                           2*position.max() + 1,
                           2*position.max() + 1), dtype=float)
    ALLECOVMAPX = np.zeros((r0_list.shape[0],
                            GSMag_list.shape[0],
                            readout_list.shape[0],
                            2*position.max() + 1,
                            2*position.max() + 1), dtype=float)
    ALLCOVMAPY = np.zeros((r0_list.shape[0],
                           GSMag_list.shape[0],
                           readout_list.shape[0],
                           2*position.max() + 1,
                           2*position.max() + 1), dtype=float)
    ALLECOVMAPY = np.zeros((r0_list.shape[0],
                            GSMag_list.shape[0],
                            readout_list.shape[0],
                            2*position.max() + 1,
                            2*position.max() + 1), dtype=float)
    
    for i in np.arange(r0_list.shape[0]):
        for j in np.arange(GSMag_list.shape[0]):
            for k in np.arange(readout_list.shape[0]):
                ALLCOVMAPX[i,j,k], ALLECOVMAPX[i,j,k], ALLCOVMAPY[i,j,k], ALLECOVMAPY[i,j,k] = make_cov_map(ALLSLOPES[i,j,k])
    
    ###############################################################################
    ########################################################################
    
    np.savetxt(SAVENAME + 'RANDOMDISTR.txt', RANDOMDISTR)
    np.savetxt(SAVENAME + 'r0_list.txt', r0_list)
    np.savetxt(SAVENAME + 'GSMag_list.txt', GSMag_list)
    np.savetxt(SAVENAME + 'readout_list.txt', readout_list)
        
    name = SAVENAME + '_map_cov_x_open'
    hdu = fits.PrimaryHDU(ALLCOVMAPX)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + '.fits', overwrite=True)
    hdul.close()
    
    name = SAVENAME + '_map_cov_y_open'
    hdu = fits.PrimaryHDU(ALLCOVMAPY)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + '.fits', overwrite=True)
    hdul.close()
    
    name = SAVENAME + '_emap_cov_x_open'
    hdu = fits.PrimaryHDU(ALLECOVMAPX)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + '.fits', overwrite=True)
    hdul.close()
    
    name = SAVENAME + '_emap_cov_y_open'
    hdu = fits.PrimaryHDU(ALLECOVMAPY)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + '.fits', overwrite=True)
    hdul.close()

    ##############################################################################
    #######################################################################

else:
    name = SAVENAME + '_ALLSLOPES'
    hdulist = fits.open(name + '.fits')
    hdu = hdulist[0]
    ALLSLOPES = hdu.data
    hdulist.close()
    
    name = SAVENAME + '_map_cov_x_open'
    hdulist = fits.open(name + '.fits')
    hdu = hdulist[0]
    ALLCOVMAPX = hdu.data
    hdulist.close()
    
    name = SAVENAME + '_map_cov_y_open'
    hdulist = fits.open(name + '.fits')
    hdu = hdulist[0]
    ALLCOVMAPY = hdu.data
    hdulist.close()
    
    name = SAVENAME + '_emap_cov_x_open'
    hdulist = fits.open(name + '.fits')
    hdu = hdulist[0]
    ALLECOVMAPX = hdu.data
    hdulist.close()
    
    name = SAVENAME + '_emap_cov_y_open'
    hdulist = fits.open(name + '.fits')
    hdu = hdulist[0]
    ALLECOVMAPY = hdu.data
    hdulist.close()
    
    RANDOMDISTR = np.loadtxt(SAVENAME + 'RANDOMDISTR.txt')
    r0_list = np.loadtxt(SAVENAME + 'r0_list.txt')
    GSMag_list = np.loadtxt(SAVENAME + 'GSMag_list.txt')
    readout_list = np.loadtxt(SAVENAME + 'readout_list.txt')

###############################################################################
########################################################################

HOMEDIR = home
REFFILE = 'referenceFunctionsVK_6x6'

A_withcentre = loadTIMBRefCov_withcentre(HOMEDIR, REFFILE)
A_withcentre = np.nan_to_num(A_withcentre)
# A_withcentre[A_withcentre == 0] = np.nan


LAYER = A_withcentre.shape[1]
FLAT = A_withcentre.shape[0]//2
SCREEN = int(np.sqrt(FLAT))
MID = LAYER - 1

IS_A_CENTRE = np.zeros((2*FLAT,LAYER),dtype=float)
((IS_A_CENTRE[:FLAT,:]).reshape((SCREEN,SCREEN,LAYER)))[:,MID] = 1
((IS_A_CENTRE[FLAT:2*FLAT,:]).reshape((SCREEN,SCREEN,LAYER)))[:,MID] = 1

IS_COV_CENTRE = np.zeros((2*FLAT),dtype=float)
((IS_COV_CENTRE[:FLAT]).reshape((SCREEN,SCREEN)))[:,MID] = 1
((IS_COV_CENTRE[FLAT:2*FLAT]).reshape((SCREEN,SCREEN)))[:,MID] = 1

A_nocentre = (A_withcentre[np.where(IS_A_CENTRE == False)]).reshape((2*SCREEN*(SCREEN - 1),LAYER))


for i in np.arange(r0_list.shape[0]):
    for j in np.arange(GSMag_list.shape[0]):
        for k in np.arange(readout_list.shape[0]):
            
            # i = 1
            # j = -1
            # # k = 2
            
            RANDOMR0 = r0_list[i]
            
            COEFFDISTRIBUTION = RANDOMDISTR
            RANDOMTOTALCN2 = r02cn2(RANDOMR0)
            RANDOMCN2 = RANDOMTOTALCN2 * COEFFDISTRIBUTION
            
            RANDOMCOEFF = RANDOMCN2 / REFTOTALCN2
            
            map_cov_x_open = ALLCOVMAPX[i,j,k]
            map_cov_y_open = ALLCOVMAPY[i,j,k]
            emap_cov_x_open = ALLECOVMAPX[i,j,k]
            emap_cov_y_open = ALLECOVMAPY[i,j,k]
            
            # NOISYRANDOMCOV = A_withcentre @ RANDOMCOEFF #np.concatenate([(map_cov_x_open).flatten(),(map_cov_y_open).flatten()])
            NOISYRANDOMCOV = np.concatenate([(map_cov_x_open).flatten(),(map_cov_y_open).flatten()])
            NOISYRANDOMCOV = np.nan_to_num(NOISYRANDOMCOV)
            # NOISYRANDOMCOV[NOISYRANDOMCOV == 0] = np.nan
            
            NOISYRANDOMCOV_nocentre = (NOISYRANDOMCOV[np.where(IS_COV_CENTRE == False)]).reshape((2*SCREEN*(SCREEN - 1)))
            
            NOISE = np.concatenate([(emap_cov_x_open).flatten(),(emap_cov_y_open).flatten()])
            NOISE = np.nan_to_num(NOISE)
            # NOISE[NOISE == 0] = np.nan
            
            NOISE_nocentre = (NOISE[np.where(IS_COV_CENTRE == False)]).reshape((2*SCREEN*(SCREEN - 1)))
            
            (SOLVEDCN2_withcentre,
              SOLVEDCN2NOISE_withcentre,
              SOLVEDR0_withcentre,
              SOLVEDR0NOISE_withcentre) = lsq_cov2cn2(A_withcentre,
                                                  REFTOTALCN2,
                                                  NOISYRANDOMCOV,
                                                  NOISE)
            (SOLVEDCN2_nocentre,
              SOLVEDCN2NOISE_nocentre,
              SOLVEDR0_nocentre,
              SOLVEDR0NOISE_nocentre) = lsq_cov2cn2(A_nocentre,
                                                REFTOTALCN2,
                                                NOISYRANDOMCOV_nocentre,
                                                NOISE_nocentre)
            
            DISTRIBUTION_ERROR_noCentre = (((SOLVEDCN2_nocentre/RANDOMTOTALCN2 - RANDOMCN2/RANDOMTOTALCN2)**2).sum())**0.5
            DISTRIBUTION_ERROR_withCentre = (((SOLVEDCN2_withcentre/RANDOMTOTALCN2 - RANDOMCN2/RANDOMTOTALCN2)**2).sum())**0.5
            r0_ERROR_noCentre = np.abs(SOLVEDR0_nocentre - RANDOMR0)
            r0_ERROR_withCentre = np.abs(SOLVEDR0_withcentre - RANDOMR0)
            
            
            fig, axs = plt.subplot_mosaic([['top', 'top'],['left', 'right'],],
                          constrained_layout=True,
                          gridspec_kw={'width_ratios': [1,9],'height_ratios': [1,9]},
                          figsize=(7,7))
            plt.suptitle('r0={:.0f}cm, magnitude={:.0f}, readout ={:.2f}e-/lenslet/photon'.format(r0_list[i]*100,GSMag_list[j],readout_list[k])
                         #+ '\nStatistics checks on Cn2 decomposition from generated cov map\n'
                          #+ 'Total Cn2={:.2e}m$^-$$^2$$^/$$^3$\n'.format(RANDOMTOTALCN2)
                          + '\nRMS error of distribution ratio : NO centre fit = {:.2f}, WITH centre fit = {:.2f}\n'.format(
                              (DISTRIBUTION_ERROR_noCentre), (DISTRIBUTION_ERROR_withCentre))
                          + 'r0 error: NO centre fit = {:.2f}cm, WITH centre fit = {:.2f}cm'.format(
                              (r0_ERROR_noCentre)*100,(r0_ERROR_withCentre)*100))
            
            axs['right'].set_title('Measured Cn2 distribution')
            axs['right'].errorbar(np.arange(LAYER),
                            SOLVEDCN2_nocentre/RANDOMTOTALCN2,
                            yerr=SOLVEDCN2NOISE_nocentre/RANDOMTOTALCN2,
                            ls='', marker='^', capsize=5,label='measured NO central peak',color='r')
            axs['right'].errorbar(np.arange(LAYER),
                            SOLVEDCN2_withcentre/RANDOMTOTALCN2,
                            yerr=SOLVEDCN2NOISE_withcentre/RANDOMTOTALCN2,
                            ls='', marker='v', capsize=5,label='measured WITH central peak',color='b')
            axs['right'].bar(np.arange(LAYER), RANDOMCN2/RANDOMTOTALCN2,
                        ls='-', label='simulated',alpha=0.5,color='k',width=0.5)
            axs['right'].set_xlabel('Turbulence layer # from ground')
            # axs[1].legend(loc='lower center')
            axs['right'].set_yscale('log')
            axs['right'].set_ylabel('fraction of total simulated Cn2')
            axs['right'].set_ylim(0.01,1)
            
            axs['left'].errorbar([''],SOLVEDR0_nocentre,yerr=SOLVEDR0NOISE_nocentre,
                            marker='^',capsize=5,color='r')#,label='measured')
            axs['left'].errorbar([''],SOLVEDR0_withcentre,yerr=SOLVEDR0NOISE_withcentre,
                            marker='v',capsize=5,color='b')#,label='measured')
            axs['left'].bar([''],RANDOMR0,color='k',alpha=0.5)#,label='simulated')
            axs['left'].set_ylabel('r0 (m)')
            axs['left'].set_ylim((0.02,0.11))
            axs['left'].set_title('Total r0 (m)')
            
            h, l = axs['right'].get_legend_handles_labels()
            axs['top'].legend(h, l, loc='center',ncol=3)
            axs['top'].axis('off')
            
            fig.tight_layout(pad=0.5)
            
            if saveFigure == True:
                plt.savefig(SAVENAME
                        + '_turbulenceProfile'
                        + '.png')
            plt.show()
                
# plt.errorbar(r0_list,measurer0_withcentre,yerr=measurer0noise_withcentre,label='with centre',ls='',marker='o',capsize=5)
# plt.errorbar(r0_list,measurer0_nocentre,yerr=measurer0noise_nocentre,label='no centre',ls='',marker='o',capsize=5)
# plt.plot(r0_list,r0_list,ls='-',label='simulated',marker='x')
# plt.legend()
# plt.xlabel('simulated r0 (m)')
# plt.ylabel('measured r0 (m)')
# plt.title('SLODAR-A1')
# plt.axis('square')
# plt.savefig('SLODAR-A1-measure-vs-true')
# plt.show()

