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

###############################################################################
########################################################################

ARCSEC2RAD = 1./3600./360.*2.*np.pi
RAD2ARCSEC = 1./ARCSEC2RAD

REFR0 = 0.05
REFTOTALCN2 = r02cn2(REFR0)
MAXR0 = 0.05
MINR0 = 0.05

# home = ('C:/Users/gqmc62/OneDrive - Durham University/Desktop/'
#               + 'soapy-master/soapy-master/conf')
home = 'C:/Users/gqmc62/OneDrive - Durham University/Documents/PhD/Codes/conf'
os.chdir(home)

LOAD = True

file = 'SLODAR6L_PseudoOpen.yaml'
sim = soapy.Sim(file)
LAYER = sim.config.atmos.scrnNo
TIME = sim.config.sim.nIters




###############################################################################
########################################################################

frameRateList = np.array([1/8.,1/4.,1/2.,1.,2.,4.,8.], dtype=float)
r0List = np.array([0.05], dtype=float)

all_measured_r0_O = np.zeros((r0List.shape[0], frameRateList.shape[0]), dtype=float)
all_measured_r0_O_nocentre = np.zeros((r0List.shape[0], frameRateList.shape[0]), dtype=float)
all_measured_r0_PO = np.zeros((r0List.shape[0], frameRateList.shape[0]), dtype=float)
all_measured_r0_PO_nocentre = np.zeros((r0List.shape[0], frameRateList.shape[0]), dtype=float)

all_measured_r0_noise_O = np.zeros((r0List.shape[0], frameRateList.shape[0]), dtype=float)
all_measured_r0_noise_O_nocentre = np.zeros((r0List.shape[0], frameRateList.shape[0]), dtype=float)
all_measured_r0_noise_PO = np.zeros((r0List.shape[0], frameRateList.shape[0]), dtype=float)
all_measured_r0_noise_PO_nocentre = np.zeros((r0List.shape[0], frameRateList.shape[0]), dtype=float)

scrnHeights_list = (np.arange(sim.config.wfss[0].nxSubaps - 2, dtype=float)
                    * (sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)
                    / ((sim.config.wfss[0].GSPosition[0]
                        - sim.config.wfss[1].GSPosition[0])*ARCSEC2RAD))

for ir0 in np.arange(r0List.shape[0]):
    r0 = r0List[ir0]
    for iframeRate in np.arange(frameRateList.shape[0]):
        frameRate = frameRateList[iframeRate]
        
        # RANDOMR0 = np.random.rand()*(MAXR0 - MINR0) + MINR0
        RANDOMDISTR = np.random.rand((LAYER))
        # sim.config.atmos.scrnStrengths = np.array([0.01,0.01,0.95,0.01,0.01,0.01], dtype=float)#np.random.random((LAYER))
        sim.config.atmos.scrnStrengths = RANDOMDISTR
        sim.config.atmos.scrnStrengths /= sim.config.atmos.scrnStrengths.sum()
        # sim.config.atmos.r0 = RANDOMR0
        sim.config.atmos.r0 = r0

        sim.config.atmos.scrnHeights = scrnHeights_list
        
        vBar53 = (sim.config.atmos.windSpeeds[:sim.config.atmos.scrnNo]**(5./3.) * sim.config.atmos.scrnStrengths[:sim.config.atmos.scrnNo]).sum() ** (3./5.)
        tau0 = 0.314 * sim.config.atmos.r0 / vBar53
    
        sim.config.sim.loopTime = tau0*frameRate
        
        sim.aoinit()

        sim.config.calcParams()
        sim.makeIMat()
        sim.allDmCommands = np.zeros((sim.config.sim.nIters, sim.config.sim.totalActs))

        N = sim.wfss[0].n_subaps
        
        position = np.asarray(sim.wfss[0].pupil_subap_coords/sim.wfss[0].nx_subap_pixels,dtype=int) - 1
        
        mid = sim.config.wfss[0].nxSubaps - 1 - 1 - 1
        
        pixel_scale = sim.config.tel.telDiam / (sim.config.wfss[0].nxSubaps * sim.config.wfss[0].pxlsPerSubap)
        
        grid = np.arange(mid + 1,dtype=float)
        grid -= len(grid)/2
        grid += 0.5
        
        ALPHA1 = 0.5
        ALPHA2 = 0.5
        saveFigure = True

        SAVENAME = ('PSEUDOOPENSLODAR'
                    + '_{:d}Frames'.format(TIME)
                    + '_L0{:.0f}m'.format(sim.config.atmos.L0[0])
                    + '_D{:.0f}cm'.format(sim.config.tel.telDiam*100)
                    + '_d{:.0f}cm'.format(100*sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)
                    + '_r0{:.0f}cm'.format(100*sim.config.atmos.r0)
                    + '_framerate{:.0f}ms'.format(1000*sim.config.sim.loopTime)
                    + '_tau0{:.0f}ms'.format(1000*sim.atmos.tau0))
        
        # print(SAVENAME)
        
        WFSPX2NM = (sim.config.wfss[0].subapFOV/sim.config.wfss[0].pxlsPerSubap)*ARCSEC2RAD*(sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)/1e-9
        
        
        # SAVENAME = 'test'
        
        frames_to_close = 10
        
        D = sim.config.tel.telDiam
        L0 = sim.config.atmos.L0[0]
        l0 = 1e-10
        r0 = sim.config.atmos.r0
        n_modes = 32 + 1
        NPXLSCRN = (sim.config.wfss[0].nxSubaps * sim.config.wfss[0].pxlsPerSubap)
        ovf = sim.config.sim.simOversize
        initial_flush = int(np.ceil(L0/D/ovf))
        L0_flush = int(np.ceil(L0/pixel_scale))
        D_flush = int(np.ceil(D/pixel_scale))
        Zernike = zernike.zernikeArray(n_modes, NPXLSCRN)
        GramSchmidt = mps.gram_schmidt_process(Zernike)
        PUPIL = Zernike[0]
        FlatZernike = Zernike.reshape((n_modes, NPXLSCRN*NPXLSCRN))
        FlatGramSchmidt = GramSchmidt.reshape((n_modes, NPXLSCRN*NPXLSCRN))
        GramSchmidtCoeff2ZernikeCoeff = mps.get_GramSchmidtCoeff2ZernikeCoeff(FlatZernike, FlatGramSchmidt)
        
        if LOAD == False:
        
            ##############################################################################
            #######################################################################
            
            WFS_open_all = np.zeros((TIME,4*N), dtype=float)
            WFS_pseudoOpen_all = np.zeros((TIME,4*N), dtype=float)
            WFS_pseudoOpen = np.zeros((TIME//frames_to_close,4*N), dtype=float)
            WFS_open = np.zeros((TIME//frames_to_close,4*N), dtype=float)
            Flat_DM = np.zeros((sim.dmCommands.shape),dtype=float)
            
            PSD = np.zeros((TIME//frames_to_close, NPXLSCRN, NPXLSCRN), dtype=float)
            ZernikeCoeff = np.zeros((TIME//frames_to_close, n_modes), dtype=float)
            phaseDifSquareX = np.zeros((TIME//frames_to_close, NPXLSCRN, NPXLSCRN), dtype=float)
            phaseDifSquareY = np.zeros((TIME//frames_to_close, NPXLSCRN, NPXLSCRN), dtype=float)    
            
            for i in np.arange(len(sim.dms)):
                sim.dms[i].gain = 1.
                sim.recon.dms[i].gain = 1.
                sim.dms[i].dmConfig.gain = 1.
            
            for layer in np.arange(LAYER):
                for row in np.arange(sim.atmos.infinite_phase_screens[layer].scrn.shape[0]*initial_flush):
                    sim.atmos.infinite_phase_screens[layer].add_row()
            
            for iFrame in np.arange(TIME):
                t = time.time()
                if ((iFrame % frames_to_close) == 0):
                    for layer in np.arange(LAYER):
                        for row in np.arange(D_flush):
                            sim.atmos.infinite_phase_screens[layer].add_row()
                    sim.dmCommands[:] = 0
                sim.scrns = sim.atmos.moveScrns()
                # x = np.arange(sim.atmos.scrns[0].shape[0])/128*8*500/10
                # yy, xx = np.meshgrid(x,x)
                # sim.scrns = (yy)*np.ones_like(sim.atmos.scrns)
                
                # if (iFrame % frames_to_close) == 0:
                    
                #     tempa = (sim.config.sim.simSize - sim.config.sim.pupilSize)//2
                #     tempb = (sim.config.sim.simSize + sim.config.sim.pupilSize)//2
                #     phase = (sim.scrns[0])[tempa:tempb,tempa:tempb] * (2. * np.pi / (sim.wfss[0].wavelength/1e-9))
                #     phaseDifSquareX[iFrame//frames_to_close] = mps.get_phaseDifSquareX(phase)
                #     phaseDifSquareY[iFrame//frames_to_close] = mps.get_phaseDifSquareY(phase)
                #     PSD[iFrame//frames_to_close] = mps.get_PSD(phase, pixel_scale)
                #     ZernikeCoeff[iFrame//frames_to_close] = mps.get_ZernikeCoeff((phase*PUPIL).flatten(), FlatGramSchmidt, GramSchmidtCoeff2ZernikeCoeff)
                #     # plt.imshow(phase)
                #     # plt.title('iFrame = {:d}'.format(iFrame))
                #     # plt.colorbar()
                #     # plt.show()
                
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
                # Run WFS, with closed loop DM shape applied
                WFS_open_all[iFrame] = sim.runWfs(dmShape=sim.open_correction,
                                          loopIter=sim.iters)
                sim.slopes = sim.runWfs(dmShape=sim.closed_correction,
                                          loopIter=sim.iters)
            
                # Get DM shape for open loop DMs
                sim.open_correction = sim.runDM(sim.dmCommands,
                                                  closed=False)
            
                # Pass whole combined DM shapes to science target
                sim.combinedCorrection = sim.open_correction + sim.closed_correction
                
                # plt.imshow(sim.closed_correction[0])
                # plt.show()
            
                sim.runSciCams(sim.closed_correction)
            
                # Save Data
                i = sim.iters % sim.config.sim.nIters # If sim is run continuously in loop, overwrite oldest data in buffer
                sim.storeData(i)
            
                sim.printOutput(sim.iters, strehl=True)
            
                sim.addToGuiQueue()
            
                sim.iters += 1
            
            sim.saveData()
            sim.finishUp()
            
            WFS_pseudoOpen_all = sim.allSlopes[:-1] + sim.allDmCommands[1:] @ sim.recon.interaction_matrix
            
            # ri = cn22r0(r02cn2(r0) * sim.config.atmos.scrnStrengths[0])
            
            # structure_function_x = np.mean(phaseDifSquareX, axis=(0,2))
            # error_structure_function_x = np.std(phaseDifSquareX, axis=(0,2), ddof=1)/np.sqrt(TIME)
            # structure_function_y = np.mean(phaseDifSquareY, axis=(0,1))
            # error_structure_function_y = np.std(phaseDifSquareY, axis=(0,1), ddof=1)/np.sqrt(TIME)
            # theoretical_structure_function = mps.vonKarmanStructureFunction(ri, NPXLSCRN, pixel_scale, L0)
            # radius = np.arange(NPXLSCRN)*pixel_scale
            # plt.plot(radius, theoretical_structure_function,
            #           label='von Karman Structure Function', ls='-', marker='')
            # plt.errorbar(radius, structure_function_x,
            #               yerr=error_structure_function_x,
            #               ls='',marker='o', capsize=5,label='measured X-axis',
            #               alpha=ALPHA1)
            # plt.errorbar(radius, structure_function_y,
            #               yerr=error_structure_function_y,
            #               ls='',marker='o', capsize=5,label='measured Y-axis',
            #               alpha=ALPHA1)
            # plt.errorbar(radius, np.abs(structure_function_x - theoretical_structure_function),
            #               yerr=error_structure_function_x,
            #               ls='',marker='o', capsize=5,label='abs difference X-axis',
            #               alpha=ALPHA1)
            # plt.errorbar(radius, np.abs(structure_function_y - theoretical_structure_function),
            #               yerr=error_structure_function_y,
            #               ls='',marker='o', capsize=5,label='abs difference Y-axis',
            #               alpha=ALPHA1)
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.legend()
            # plt.xlabel('separation (m)')
            # plt.ylabel('structure function (radian^2)')
            # plt.xlim(radius.min(),radius.max()*2)
            # plt.ylim(np.nanmin(theoretical_structure_function)/10,
            #           np.nanmax(theoretical_structure_function)*10)
            # plt.title(SAVENAME + '\n' + '{:d}'.format(TIME)
            #         + 'structure_function'
            #         + 'D{:.0f}cm'.format(D*100)
            #         + 'r0{:.0f}cm'.format(ri*100)
            #         + 'L0{:.0f}m'.format(L0))
            # if saveFigure == True:
            #     plt.savefig(SAVENAME
            #             + '_structure_function'
            #             + '.png')
            # plt.show()
            
            # mean_PSD_x = (np.mean(PSD, axis=0).T)[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
            # error_mean_PSD_x = ((np.std(PSD, axis=0, ddof=1)/np.sqrt(TIME)).T)[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
            # mean_PSD_y = (np.mean(PSD, axis=0))[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
            # error_mean_PSD_y = ((np.std(PSD, axis=0, ddof=1)/np.sqrt(TIME)))[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
            # theoretical_PSD = mps.psd_function(ri, NPXLSCRN, pixel_scale, L0, l0)[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
            # f = (np.arange(NPXLSCRN/2-1) + 1)/NPXLSCRN/pixel_scale
            # plt.plot(f, theoretical_PSD,
            #           label='von Karman PSD', ls='-', marker='')
            # plt.errorbar(f, mean_PSD_x,
            #               yerr=error_mean_PSD_x,ls='',marker='o', capsize=5,
            #               label='measured X-axis',alpha=ALPHA2)
            # plt.errorbar(f, mean_PSD_y,
            #               yerr=error_mean_PSD_y,ls='',marker='o', capsize=5,
            #               label='measured Y-axis',alpha=ALPHA2)
            # plt.errorbar(f, np.abs(mean_PSD_x - theoretical_PSD),
            #               yerr=error_mean_PSD_x,ls='',marker='o', capsize=5,
            #               label='abs difference X-axis',alpha=ALPHA2)
            # plt.errorbar(f, np.abs(mean_PSD_y - theoretical_PSD),
            #               yerr=error_mean_PSD_y,ls='',marker='o', capsize=5,
            #               label='abs difference Y-axis',alpha=ALPHA2)
            # plt.ylim(np.nanmin(theoretical_PSD)/10,np.nanmax(theoretical_PSD)*10)
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.xlabel('spatial frequency (1/m)')
            # plt.ylabel('phase structure function (??radian??)')
            # plt.title(SAVENAME + '\n' + '{:d}'.format(TIME)
            #         + 'PSD'
            #         + 'D{:.0f}cm'.format(D*100)
            #         + 'r0{:.0f}cm'.format(ri*100)
            #         + 'L0{:.0f}m'.format(L0))
            # plt.legend()
            # if saveFigure == True:
            #     plt.savefig(SAVENAME
            #             + '_PSD'
            #             + '.png')
            # plt.show()
            
            # variance_ZernikeCoeff = np.var(ZernikeCoeff, ddof=1, axis=0)[1:]
            # error_variance_ZernikeCoeff = 2*variance_ZernikeCoeff/np.sqrt(2*NPXLSCRN - 2)
            # n_modes = variance_ZernikeCoeff.shape[0]
            # modes = np.arange(n_modes) + 1
            # theoretical_variance_ZernikeCoeff = mps.noll_variance(n_modes,NPXLSCRN,pixel_scale,ri)
            # e = NPXLSCRN*pixel_scale/L0
            # adjust = (1 - 1.42*e**(1./3.) + 3.70*e**2 - 4.01*e**(7./3.) + 4.21*e**4 - 4.00*e**(13./3.))
            # L0TipTiltVariance = theoretical_variance_ZernikeCoeff[:2] * adjust
            
            # plt.plot(modes, theoretical_variance_ZernikeCoeff, label='Kolmogorov',marker='',ls='-')
            # plt.errorbar(modes, variance_ZernikeCoeff, yerr=error_variance_ZernikeCoeff,
            #               label='measured', marker='o', ls='', capsize=5)
            # plt.plot(modes[:2], L0TipTiltVariance, label='L0 limited tip/tilt variance')
            # theoretical_variance_ZernikeCoeff[:2] = theoretical_variance_ZernikeCoeff[:2] * adjust
            # plt.errorbar(modes, np.abs(variance_ZernikeCoeff - theoretical_variance_ZernikeCoeff),
            #               yerr=error_variance_ZernikeCoeff,
            #               label='abs difference', marker='o', ls='', capsize=5)
            # plt.yscale('log')
            # plt.xlabel('zernike modes')
            # plt.ylabel('variance of zernike coefficient')
            # plt.legend()
            # plt.ylim(np.nanmin(theoretical_variance_ZernikeCoeff)/10, np.nanmax(theoretical_variance_ZernikeCoeff)*10)
            # plt.title(SAVENAME + '\n' + '{:d}'.format(TIME)
            #         + 'ZernikeVariances'
            #         + 'D{:.0f}cm'.format(D*100)
            #         + 'r0{:.0f}cm'.format(ri*100)
            #         + 'L0{:.0f}m'.format(L0))
            # if saveFigure == True:
            #     plt.savefig(SAVENAME
            #             + '_ZernikeCoeff'
            #             + '.png')
            # plt.show()
            
            WFS_pseudoOpen = WFS_pseudoOpen_all[frames_to_close - 1::frames_to_close]
            WFS_open = WFS_open_all[frames_to_close - 1:-1:frames_to_close]
            
            # VMAX = np.max([np.nanmax(WFS_pseudoOpen),np.nanmax(WFS_open)])
            # VMIN = np.min([np.nanmin(WFS_pseudoOpen),np.nanmin(WFS_open)])
            
            # plt.imshow(WFS_pseudoOpen,vmax=VMAX,vmin=VMIN)
            # plt.title('PseudoOpen')
            # plt.colorbar()
            # plt.show()
            
            # plt.imshow(WFS_open,vmax=VMAX,vmin=VMIN)
            # plt.title('open')
            # plt.colorbar()
            # plt.show()
            
            # WFS_pseudoOpen *= 2
            # WFS_open *= 2
            
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
            
            
            
            
            cov_x_pseudoOpen = np.zeros((N, N), dtype=float)
            cov_y_pseudoOpen = np.zeros((N, N), dtype=float)
            
            ecov_x_pseudoOpen = np.zeros((N, N), dtype=float)
            ecov_y_pseudoOpen = np.zeros((N, N), dtype=float)
            
            map_cov_x_pseudoOpen = np.zeros((2*mid + 1, 2*mid + 1), dtype=float)
            map_cov_y_pseudoOpen = np.zeros((2*mid + 1, 2*mid + 1), dtype=float)
            
            emap_cov_x_pseudoOpen = np.zeros((2*mid + 1, 2*mid + 1), dtype=float)
            emap_cov_y_pseudoOpen = np.zeros((2*mid + 1, 2*mid + 1), dtype=float)
            
            Sx1_pseudoOpen = WFS_pseudoOpen[ : , (0)*N : (1)*N]
            Sy1_pseudoOpen = WFS_pseudoOpen[ : , (1)*N : (2)*N]
            Sx2_pseudoOpen = WFS_pseudoOpen[ : , (2)*N : (3)*N]
            Sy2_pseudoOpen = WFS_pseudoOpen[ : , (3)*N : (4)*N]
            
            Sx1_pseudoOpen = (Sx1_pseudoOpen.T - np.mean(Sx1_pseudoOpen, axis=1)).T
            Sy1_pseudoOpen = (Sy1_pseudoOpen.T - np.mean(Sy1_pseudoOpen, axis=1)).T
            Sx2_pseudoOpen = (Sx2_pseudoOpen.T - np.mean(Sx2_pseudoOpen, axis=1)).T
            Sy2_pseudoOpen = (Sy2_pseudoOpen.T - np.mean(Sy2_pseudoOpen, axis=1)).T
            
            cov_x_pseudoOpen, ecov_x_pseudoOpen = my_cov(Sx1_pseudoOpen.T, Sx2_pseudoOpen.T)
            cov_y_pseudoOpen, ecov_y_pseudoOpen = my_cov(Sy1_pseudoOpen.T, Sy2_pseudoOpen.T)
            
            map_cov_x_pseudoOpen, emap_cov_x_pseudoOpen = overlay1(cov_x_pseudoOpen, ecov_x_pseudoOpen, position)
            map_cov_y_pseudoOpen, emap_cov_y_pseudoOpen = overlay1(cov_y_pseudoOpen, ecov_y_pseudoOpen, position)
            
            ###############################################################################
            ########################################################################
            name = SAVENAME + '_WFS_open'
            hdu = fits.PrimaryHDU(WFS_open)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
            
            name = SAVENAME + '_WFS_pseudoOpen'
            hdu = fits.PrimaryHDU(WFS_pseudoOpen)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
            
            name = SAVENAME + '_interaction_matrix'
            hdu = fits.PrimaryHDU(sim.recon.interaction_matrix)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
            
            np.savetxt(SAVENAME + 'RANDOMDISTR.txt', RANDOMDISTR)
            # np.savetxt(SAVENAME + 'RANDOMR0.txt', [r0])
                
            name = SAVENAME + '_map_cov_x_open'
            hdu = fits.PrimaryHDU(map_cov_x_open)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
            
            name = SAVENAME + '_map_cov_y_open'
            hdu = fits.PrimaryHDU(map_cov_y_open)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
            
            name = SAVENAME + '_emap_cov_x_open'
            hdu = fits.PrimaryHDU(emap_cov_x_open)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
            
            name = SAVENAME + '_emap_cov_y_open'
            hdu = fits.PrimaryHDU(emap_cov_y_open)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
        
            name = SAVENAME + '_map_cov_x_pseudoOpen'
            hdu = fits.PrimaryHDU(map_cov_x_pseudoOpen)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
            
            name = SAVENAME + '_map_cov_y_pseudoOpen'
            hdu = fits.PrimaryHDU(map_cov_y_pseudoOpen)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
            
            name = SAVENAME + '_emap_cov_x_pseudoOpen'
            hdu = fits.PrimaryHDU(emap_cov_x_pseudoOpen)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
            
            name = SAVENAME + '_emap_cov_y_pseudoOpen'
            hdu = fits.PrimaryHDU(emap_cov_y_pseudoOpen)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
        
        
            ##############################################################################
            #######################################################################
        
        else:
            name = SAVENAME + '_WFS_open'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            WFS_open = hdu.data
            hdulist.close()
            
            name = SAVENAME + '_WFS_pseudoOpen'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            WFS_pseudoOpen = hdu.data
            hdulist.close()
            
            name = SAVENAME + '_interaction_matrix'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            IM = hdu.data
            hdulist.close()
            
            name = SAVENAME + '_map_cov_x_open'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            map_cov_x_open = hdu.data
            hdulist.close()
            
            name = SAVENAME + '_map_cov_y_open'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            map_cov_y_open = hdu.data
            hdulist.close()
            
            name = SAVENAME + '_emap_cov_x_open'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            emap_cov_x_open = hdu.data
            hdulist.close()
            
            name = SAVENAME + '_emap_cov_y_open'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            emap_cov_y_open = hdu.data
            hdulist.close()
            
            
            name = SAVENAME + '_map_cov_x_pseudoOpen'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            map_cov_x_pseudoOpen = hdu.data
            hdulist.close()
            
            name = SAVENAME + '_map_cov_y_pseudoOpen'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            map_cov_y_pseudoOpen = hdu.data
            hdulist.close()
            
            name = SAVENAME + '_emap_cov_x_pseudoOpen'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            emap_cov_x_pseudoOpen = hdu.data
            hdulist.close()
            
            name = SAVENAME + '_emap_cov_y_pseudoOpen'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            emap_cov_y_pseudoOpen = hdu.data
            hdulist.close()
            
            RANDOMDISTR = np.loadtxt(SAVENAME + 'RANDOMDISTR.txt')
            # RANDOMR0 = float(np.loadtxt(SAVENAME + 'RANDOMR0.txt'))
        
        ###############################################################################
        ########################################################################
        
        VMAX = np.nanmax([WFS_open,WFS_pseudoOpen])*WFSPX2NM
        VMIN = np.nanmin([WFS_open,WFS_pseudoOpen])*WFSPX2NM
        RMAX = np.max(np.abs([VMAX,VMIN]))
        RMSDIF = np.sqrt(np.mean((WFS_pseudoOpen - WFS_open)**2))
        # plt.imshow(WFS_open[:100]*WFSPX2NM,cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('sample open loop WFS OPD (nm)')
        # plt.colorbar()
        # plt.show()
        # plt.imshow(WFS_pseudoOpen[:100]*WFSPX2NM,cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('sample pseudo open loop WFS OPD (nm)')
        # plt.colorbar()
        # plt.show()
        # plt.imshow((WFS_pseudoOpen[:100] - WFS_open[:100])*WFSPX2NM,cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('sample difference between open and pseudo open loop WFS OPD (nm)\nOPD RMS={:.3f}nm'.format(RMSDIF*WFSPX2NM))
        # plt.colorbar()
        # plt.show()
        
        # HOMEDIR = 'C:/Users/gqmc62/OneDrive - Durham University/Desktop/soapy-master/soapy-master/conf'
        HOMEDIR = 'C:/Users/gqmc62/OneDrive - Durham University/Documents/PhD/Codes/conf'
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
        
        COEFFDISTRIBUTION = RANDOMDISTR
        RANDOMTOTALCN2 = r02cn2(r0)
        RANDOMCN2 = RANDOMTOTALCN2 * COEFFDISTRIBUTION
        
        RANDOMCOEFF = RANDOMCN2 / REFTOTALCN2
        
        
        
        
        NOISYRANDOMCOV_O = np.concatenate([(map_cov_x_open).flatten(),(map_cov_y_open).flatten()])
        NOISYRANDOMCOV_O = np.nan_to_num(NOISYRANDOMCOV_O)
        
        NOISYRANDOMCOV_nocentre_O = (NOISYRANDOMCOV_O[np.where(IS_COV_CENTRE == False)]).reshape((2*SCREEN*(SCREEN - 1)))
        
        NOISE_O = np.concatenate([(emap_cov_x_open).flatten(),(emap_cov_y_open).flatten()])
        NOISE_O = np.nan_to_num(NOISE_O)
        
        NOISE_nocentre_O = (NOISE_O[np.where(IS_COV_CENTRE == False)]).reshape((2*SCREEN*(SCREEN - 1)))
        
        (SOLVEDCN2_withcentre_O,
          SOLVEDCN2NOISE_withcentre_O,
          SOLVEDR0_withcentre_O,
          SOLVEDR0NOISE_withcentre_O) = lsq_cov2cn2(A_withcentre,
                                              REFTOTALCN2,
                                              NOISYRANDOMCOV_O,
                                              NOISE_O)
        (SOLVEDCN2_nocentre_O,
          SOLVEDCN2NOISE_nocentre_O,
          SOLVEDR0_nocentre_O,
          SOLVEDR0NOISE_nocentre_O) = lsq_cov2cn2(A_nocentre,
                                            REFTOTALCN2,
                                            NOISYRANDOMCOV_nocentre_O,
                                            NOISE_nocentre_O)
                                                
        
        DISTRIBUTION_ERROR_noCentre_O = (((SOLVEDCN2_nocentre_O/RANDOMTOTALCN2 - RANDOMCN2/RANDOMTOTALCN2)**2).sum())**0.5
        DISTRIBUTION_ERROR_withCentre_O = (((SOLVEDCN2_withcentre_O/RANDOMTOTALCN2 - RANDOMCN2/RANDOMTOTALCN2)**2).sum())**0.5
        r0_ERROR_noCentre_O = np.abs(SOLVEDR0_nocentre_O - r0)
        r0_ERROR_withCentre_O = np.abs(SOLVEDR0_withcentre_O - r0)
        
        
        NOISYRANDOMCOV_PO = np.concatenate([(map_cov_x_pseudoOpen).flatten(),(map_cov_y_pseudoOpen).flatten()])
        NOISYRANDOMCOV_PO = np.nan_to_num(NOISYRANDOMCOV_PO)
        
        NOISYRANDOMCOV_nocentre_PO = (NOISYRANDOMCOV_PO[np.where(IS_COV_CENTRE == False)]).reshape((2*SCREEN*(SCREEN - 1)))
        
        NOISE_PO = np.concatenate([(emap_cov_x_pseudoOpen).flatten(),(emap_cov_y_pseudoOpen).flatten()])
        NOISE_PO = np.nan_to_num(NOISE_PO)
        
        NOISE_nocentre_PO = (NOISE_PO[np.where(IS_COV_CENTRE == False)]).reshape((2*SCREEN*(SCREEN - 1)))
        
        (SOLVEDCN2_withcentre_PO,
          SOLVEDCN2NOISE_withcentre_PO,
          SOLVEDR0_withcentre_PO,
          SOLVEDR0NOISE_withcentre_PO) = lsq_cov2cn2(A_withcentre,
                                              REFTOTALCN2,
                                              NOISYRANDOMCOV_PO,
                                              NOISE_PO)
        (SOLVEDCN2_nocentre_PO,
          SOLVEDCN2NOISE_nocentre_PO,
          SOLVEDR0_nocentre_PO,
          SOLVEDR0NOISE_nocentre_PO) = lsq_cov2cn2(A_nocentre,
                                            REFTOTALCN2,
                                            NOISYRANDOMCOV_nocentre_PO,
                                            NOISE_nocentre_PO)
                                                
        
        DISTRIBUTION_ERROR_noCentre_PO = (((SOLVEDCN2_nocentre_PO/RANDOMTOTALCN2 - RANDOMCN2/RANDOMTOTALCN2)**2).sum())**0.5
        DISTRIBUTION_ERROR_withCentre_PO = (((SOLVEDCN2_withcentre_PO/RANDOMTOTALCN2 - RANDOMCN2/RANDOMTOTALCN2)**2).sum())**0.5
        r0_ERROR_noCentre_PO = np.abs(SOLVEDR0_nocentre_PO - r0)
        r0_ERROR_withCentre_PO = np.abs(SOLVEDR0_withcentre_PO - r0)
        
        all_measured_r0_O[ir0, iframeRate] = SOLVEDR0_withcentre_O
        all_measured_r0_O_nocentre[ir0, iframeRate] = SOLVEDR0_nocentre_O
        all_measured_r0_PO[ir0, iframeRate] = SOLVEDR0_withcentre_PO
        all_measured_r0_PO_nocentre[ir0, iframeRate] = SOLVEDR0_nocentre_PO
        
        all_measured_r0_noise_O[ir0, iframeRate] = SOLVEDR0NOISE_withcentre_O
        all_measured_r0_noise_O_nocentre[ir0, iframeRate] = SOLVEDR0NOISE_nocentre_O
        all_measured_r0_noise_PO[ir0, iframeRate] = SOLVEDR0NOISE_withcentre_PO
        all_measured_r0_noise_PO_nocentre[ir0, iframeRate] = SOLVEDR0NOISE_nocentre_PO
        
        
        fig, axs = plt.subplot_mosaic([['top', 'top'],['left', 'right'],],
                      constrained_layout=True,
                      gridspec_kw={'width_ratios': [1,9],'height_ratios': [1,9]},
                      figsize=(7,7))
        plt.suptitle(SAVENAME + '\nWFS slope RMS difference = {:.0f} nm'.format(RMSDIF*WFSPX2NM))
        axs['right'].set_title('Measured Cn2 distribution')
        axs['right'].errorbar(np.arange(LAYER),
                        SOLVEDCN2_nocentre_O/RANDOMTOTALCN2,
                        yerr=SOLVEDCN2NOISE_nocentre_O/RANDOMTOTALCN2,
                        ls='', marker='^', capsize=5,label='measured NO central peak open',color='r')
        axs['right'].errorbar(np.arange(LAYER),
                        SOLVEDCN2_nocentre_PO/RANDOMTOTALCN2,
                        yerr=SOLVEDCN2NOISE_nocentre_PO/RANDOMTOTALCN2,
                        ls='', marker='^', capsize=5,label='measured NO central peak pseudo open',color='g')
        axs['right'].errorbar(np.arange(LAYER),
                        SOLVEDCN2_withcentre_O/RANDOMTOTALCN2,
                        yerr=SOLVEDCN2NOISE_withcentre_O/RANDOMTOTALCN2,
                        ls='', marker='v', capsize=5,label='measured WITH central peak open',color='b')
        axs['right'].errorbar(np.arange(LAYER),
                        SOLVEDCN2_withcentre_PO/RANDOMTOTALCN2,
                        yerr=SOLVEDCN2NOISE_withcentre_PO/RANDOMTOTALCN2,
                        ls='', marker='v', capsize=5,label='measured WITH central peak pseudo open',color='k')
        axs['right'].bar(np.arange(LAYER), RANDOMCN2/RANDOMTOTALCN2,
                    ls='-', label='simulated',alpha=0.5,color='k',width=0.5)
        axs['right'].set_xlabel('Turbulence layer # from ground')
        # axs[1].legend(loc='lower center')
        axs['right'].set_yscale('log')
        axs['right'].set_ylabel('fraction of total simulated Cn2')
        axs['right'].set_ylim(0.01,1)
        
        axs['left'].errorbar([''],SOLVEDR0_nocentre_O,yerr=SOLVEDR0NOISE_nocentre_O,
                        marker='^',capsize=5,color='r')#,label='measured')
        axs['left'].errorbar([''],SOLVEDR0_nocentre_PO,yerr=SOLVEDR0NOISE_nocentre_PO,
                        marker='^',capsize=5,color='g')#,label='measured')
        axs['left'].errorbar([''],SOLVEDR0_withcentre_O,yerr=SOLVEDR0NOISE_withcentre_O,
                        marker='v',capsize=5,color='b')#,label='measured')
        axs['left'].errorbar([''],SOLVEDR0_withcentre_PO,yerr=SOLVEDR0NOISE_withcentre_PO,
                        marker='v',capsize=5,color='k')#,label='measured')
        axs['left'].bar([''],r0,color='k',alpha=0.5)#,label='simulated')
        axs['left'].set_ylabel('r0 (m)')
        axs['left'].set_ylim((0.02,0.11))
        axs['left'].set_title('Total r0 (m)')
        
        h, l = axs['right'].get_legend_handles_labels()
        axs['top'].legend(h, l, loc='center',ncol=3)
        axs['top'].axis('off')
        
        fig.tight_layout(pad=0.5)
        plt.show()
        
        
        # expected = A_withcentre @ RANDOMCOEFF
        # measured = NOISYRANDOMCOV_O
        # background = measured - expected
        
        # VMIN = np.nanmin(NOISYRANDOMCOV_O)
        # VMAX = np.nanmax(NOISYRANDOMCOV_O)
        # RMAX= np.nanmax(np.abs([VMIN,VMAX]))
        
        # plt.imshow(expected.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('expected open loop')
        # plt.colorbar()
        # if saveFigure == True:
        #     plt.savefig(SAVENAME
        #             + '_expectedCOVMapOpen'
        #             + '.png')
        # plt.show()
        
        # plt.imshow(measured.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('measured open loop')
        # plt.colorbar()
        # if saveFigure == True:
        #     plt.savefig(SAVENAME
        #             + '_measuredCOVMapOpen'
        #             + '.png')
        # plt.show()
        
        # plt.imshow(background.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('difference open loop')
        # plt.colorbar()
        # if saveFigure == True:
        #     plt.savefig(SAVENAME
        #             + '_COVMapErrorOpen'
        #             + '.png')
        # plt.show()
        
        # expected = A_withcentre @ RANDOMCOEFF
        # measured = NOISYRANDOMCOV_PO
        # background = measured - expected
        
        # VMIN = np.nanmin(NOISYRANDOMCOV_O)
        # VMAX = np.nanmax(NOISYRANDOMCOV_O)
        # RMAX= np.nanmax(np.abs([VMIN,VMAX]))
        
        # plt.imshow(expected.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('expected pseudo open loop')
        # plt.colorbar()
        # if saveFigure == True:
        #     plt.savefig(SAVENAME
        #             + '_expectedCOVMapPseudoOpen'
        #             + '.png')
        # plt.show()
        
        # plt.imshow(measured.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('measured pseudo open loop')
        # plt.colorbar()
        # if saveFigure == True:
        #     plt.savefig(SAVENAME
        #             + '_measuredCOVMapPseudoOpen'
        #             + '.png')
        # plt.show()
        
        # plt.imshow(background.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('difference pseudo open loop')
        # plt.colorbar()
        # if saveFigure == True:
        #     plt.savefig(SAVENAME
        #             + '_COVMapErrorPseudoOpen'
        #             + '.png')
        # plt.show()
        
        
        # expected = NOISYRANDOMCOV_O
        # measured = NOISYRANDOMCOV_PO
        # background = measured - expected
        
        # VMIN = np.nanmin(NOISYRANDOMCOV_O)
        # VMAX = np.nanmax(NOISYRANDOMCOV_O)
        # RMAX= np.nanmax(np.abs([VMIN,VMAX]))
        
        # plt.imshow(expected.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('measured open loop')
        # plt.colorbar()
        # if saveFigure == True:
        #     plt.savefig(SAVENAME
        #             + '_COVMapOpen'
        #             + '.png')
        # plt.show()
        
        # plt.imshow(measured.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('measured pseudo open loop')
        # plt.colorbar()
        # if saveFigure == True:
        #     plt.savefig(SAVENAME
        #             + '_COVMapPseudoOpen'
        #             + '.png')
        # plt.show()
        
        # plt.imshow(background.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('difference between open and pseudo open loop')
        # plt.colorbar()
        # if saveFigure == True:
        #     plt.savefig(SAVENAME
        #             + '_COVMapErrorOpenPseudoOpen'
        #             + '.png')
        # plt.show()
        

for ir0 in np.arange(r0List.shape[0]):
    plt.errorbar(frameRateList, all_measured_r0_O[ir0], yerr=all_measured_r0_noise_O[ir0],
                 label='open', ls='', capsize=5, marker='o')
    plt.errorbar(frameRateList, all_measured_r0_PO[ir0], yerr=all_measured_r0_noise_PO[ir0],
                 label='pseudo open', ls='', capsize=5, marker='o')
    plt.errorbar(frameRateList, all_measured_r0_O_nocentre[ir0], yerr=all_measured_r0_noise_O_nocentre[ir0],
                 label='open no centre', ls='', capsize=5, marker='o')
    plt.errorbar(frameRateList, all_measured_r0_PO_nocentre[ir0], yerr=all_measured_r0_noise_PO_nocentre[ir0],
                 label='pseudo open no centre', ls='', capsize=5, marker='o')
    plt.hlines([r0List[ir0]], frameRateList.min(), frameRateList.max(), label='simulated')
    plt.legend()
    plt.xlabel('frame rate (tau0)')
    plt.ylabel('r0 (m)')
    plt.xlim(0.1,10)
    plt.ylim(0,.10)
    plt.xscale('log')
    plt.title('measured r0')
    plt.savefig('1r0{:d}r0{:.0}cmSLODARB1.png'.format(TIME,r0List[ir0]))
    plt.show()
    
# for iframeRate in np.arange(frameRateList.shape[0]):
#     plt.errorbar(r0List, all_measured_r0_O[:,iframeRate], yerr=all_measured_r0_noise_O[:,iframeRate],
#                  label='open', ls='', capsize=5, marker='o')
#     plt.errorbar(r0List, all_measured_r0_PO[:,iframeRate], yerr=all_measured_r0_noise_PO[:,iframeRate],
#                  label='pseudo open', ls='', capsize=5, marker='o')
#     plt.errorbar(r0List, all_measured_r0_O_nocentre[:,iframeRate], yerr=all_measured_r0_noise_O_nocentre[:,iframeRate],
#                  label='open no centre', ls='', capsize=5, marker='o')
#     plt.errorbar(r0List, all_measured_r0_PO_nocentre[:,iframeRate], yerr=all_measured_r0_noise_PO_nocentre[:,iframeRate],
#                  label='pseudo open no centre', ls='', capsize=5, marker='o')
#     plt.plot(r0List, r0List, label='simulated')
#     plt.legend()
#     plt.xlabel('simulated r0 (m)')
#     plt.ylabel('measured r0 (m)')
#     plt.xlim(0,.15)
#     plt.ylim(0,.15)
#     plt.title('frame rate = {:.2f} tau0'.format(frameRateList[iframeRate]))
#     plt.savefig('1r0{:d}frameRate{:.2}tau0SLODARB1.png'.format(TIME,frameRateList[iframeRate]))
#     plt.show()

np.savetxt('1r0{:d}r0O'.format(TIME), all_measured_r0_O)
np.savetxt('1r0{:d}r0NoiseO'.format(TIME), all_measured_r0_noise_O)
np.savetxt('1r0{:d}r0ONoCen'.format(TIME), all_measured_r0_O_nocentre)
np.savetxt('1r0{:d}r0NoiseONoCen'.format(TIME), all_measured_r0_noise_O_nocentre)
np.savetxt('1r0{:d}r0PO'.format(TIME), all_measured_r0_PO)
np.savetxt('1r0{:d}r0NoisePO'.format(TIME), all_measured_r0_noise_PO)
np.savetxt('1r0{:d}r0PONoCen'.format(TIME), all_measured_r0_PO_nocentre)
np.savetxt('1r0{:d}r0NoisePONoCen'.format(TIME), all_measured_r0_noise_PO_nocentre)

###############################################################################

frameRateList = np.array([1.], dtype=float)
r0List = np.array([0.02,0.05,0.08,0.11], dtype=float)

all_measured_r0_O = np.zeros((r0List.shape[0], frameRateList.shape[0]), dtype=float)
all_measured_r0_O_nocentre = np.zeros((r0List.shape[0], frameRateList.shape[0]), dtype=float)
all_measured_r0_PO = np.zeros((r0List.shape[0], frameRateList.shape[0]), dtype=float)
all_measured_r0_PO_nocentre = np.zeros((r0List.shape[0], frameRateList.shape[0]), dtype=float)

all_measured_r0_noise_O = np.zeros((r0List.shape[0], frameRateList.shape[0]), dtype=float)
all_measured_r0_noise_O_nocentre = np.zeros((r0List.shape[0], frameRateList.shape[0]), dtype=float)
all_measured_r0_noise_PO = np.zeros((r0List.shape[0], frameRateList.shape[0]), dtype=float)
all_measured_r0_noise_PO_nocentre = np.zeros((r0List.shape[0], frameRateList.shape[0]), dtype=float)

scrnHeights_list = (np.arange(sim.config.wfss[0].nxSubaps - 2, dtype=float)
                    * (sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)
                    / ((sim.config.wfss[0].GSPosition[0]
                        - sim.config.wfss[1].GSPosition[0])*ARCSEC2RAD))

for ir0 in np.arange(r0List.shape[0]):
    r0 = r0List[ir0]
    for iframeRate in np.arange(frameRateList.shape[0]):
        frameRate = frameRateList[iframeRate]
        
        # RANDOMR0 = np.random.rand()*(MAXR0 - MINR0) + MINR0
        RANDOMDISTR = np.random.rand((LAYER))
        # sim.config.atmos.scrnStrengths = np.array([0.01,0.01,0.95,0.01,0.01,0.01], dtype=float)#np.random.random((LAYER))
        sim.config.atmos.scrnStrengths = RANDOMDISTR
        sim.config.atmos.scrnStrengths /= sim.config.atmos.scrnStrengths.sum()
        # sim.config.atmos.r0 = RANDOMR0
        sim.config.atmos.r0 = r0

        sim.config.atmos.scrnHeights = scrnHeights_list
        sim.config.sim.loopTime = sim.atmos.tau0*frameRate
        sim.aoinit()

        sim.config.calcParams()
        sim.makeIMat()
        sim.allDmCommands = np.zeros((sim.config.sim.nIters, sim.config.sim.totalActs))

        N = sim.wfss[0].n_subaps
        
        position = np.asarray(sim.wfss[0].pupil_subap_coords/sim.wfss[0].nx_subap_pixels,dtype=int) - 1
        
        mid = sim.config.wfss[0].nxSubaps - 1 - 1 - 1
        
        pixel_scale = sim.config.tel.telDiam / (sim.config.wfss[0].nxSubaps * sim.config.wfss[0].pxlsPerSubap)
        
        grid = np.arange(mid + 1,dtype=float)
        grid -= len(grid)/2
        grid += 0.5
        
        ALPHA1 = 0.5
        ALPHA2 = 0.5
        saveFigure = True
        
        

        SAVENAME = ('PSEUDOOPENSLODAR'
                    + '_{:d}Frames'.format(TIME)
                    + '_L0{:.0f}m'.format(sim.config.atmos.L0[0])
                    + '_D{:.0f}cm'.format(sim.config.tel.telDiam*100)
                    + '_d{:.0f}cm'.format(100*sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)
                    + '_r0{:.0f}cm'.format(100*sim.config.atmos.r0)
                    + '_framerate{:.0f}ms'.format(1000*sim.config.sim.loopTime)
                    + '_tau0{:.0f}ms'.format(1000*sim.atmos.tau0))
        
        # print(SAVENAME)
        
        WFSPX2NM = (sim.config.wfss[0].subapFOV/sim.config.wfss[0].pxlsPerSubap)*ARCSEC2RAD*(sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)/1e-9
        
        
        # SAVENAME = 'test'
        
        frames_to_close = 10
        
        D = sim.config.tel.telDiam
        L0 = sim.config.atmos.L0[0]
        l0 = 1e-10
        r0 = sim.config.atmos.r0
        n_modes = 32 + 1
        NPXLSCRN = (sim.config.wfss[0].nxSubaps * sim.config.wfss[0].pxlsPerSubap)
        ovf = sim.config.sim.simOversize
        initial_flush = int(np.ceil(L0/D/ovf))
        L0_flush = int(np.ceil(L0/pixel_scale))
        D_flush = int(np.ceil(D/pixel_scale))
        Zernike = zernike.zernikeArray(n_modes, NPXLSCRN)
        GramSchmidt = mps.gram_schmidt_process(Zernike)
        PUPIL = Zernike[0]
        FlatZernike = Zernike.reshape((n_modes, NPXLSCRN*NPXLSCRN))
        FlatGramSchmidt = GramSchmidt.reshape((n_modes, NPXLSCRN*NPXLSCRN))
        GramSchmidtCoeff2ZernikeCoeff = mps.get_GramSchmidtCoeff2ZernikeCoeff(FlatZernike, FlatGramSchmidt)
        
        if LOAD == False:
        
            ##############################################################################
            #######################################################################
            
            WFS_open_all = np.zeros((TIME,4*N), dtype=float)
            WFS_pseudoOpen_all = np.zeros((TIME,4*N), dtype=float)
            WFS_pseudoOpen = np.zeros((TIME//frames_to_close,4*N), dtype=float)
            WFS_open = np.zeros((TIME//frames_to_close,4*N), dtype=float)
            Flat_DM = np.zeros((sim.dmCommands.shape),dtype=float)
            
            PSD = np.zeros((TIME//frames_to_close, NPXLSCRN, NPXLSCRN), dtype=float)
            ZernikeCoeff = np.zeros((TIME//frames_to_close, n_modes), dtype=float)
            phaseDifSquareX = np.zeros((TIME//frames_to_close, NPXLSCRN, NPXLSCRN), dtype=float)
            phaseDifSquareY = np.zeros((TIME//frames_to_close, NPXLSCRN, NPXLSCRN), dtype=float)    
            
            for i in np.arange(len(sim.dms)):
                sim.dms[i].gain = 1.
                sim.recon.dms[i].gain = 1.
                sim.dms[i].dmConfig.gain = 1.
            
            for layer in np.arange(LAYER):
                for row in np.arange(sim.atmos.infinite_phase_screens[layer].scrn.shape[0]*initial_flush):
                    sim.atmos.infinite_phase_screens[layer].add_row()
            
            for iFrame in np.arange(TIME):
                t = time.time()
                if ((iFrame % frames_to_close) == 0):
                    for layer in np.arange(LAYER):
                        for row in np.arange(D_flush):
                            sim.atmos.infinite_phase_screens[layer].add_row()
                    sim.dmCommands[:] = 0
                sim.scrns = sim.atmos.moveScrns()
                # x = np.arange(sim.atmos.scrns[0].shape[0])/128*8*500/10
                # yy, xx = np.meshgrid(x,x)
                # sim.scrns = (yy)*np.ones_like(sim.atmos.scrns)
                
                # if (iFrame % frames_to_close) == 0:
                    
                #     tempa = (sim.config.sim.simSize - sim.config.sim.pupilSize)//2
                #     tempb = (sim.config.sim.simSize + sim.config.sim.pupilSize)//2
                #     phase = (sim.scrns[0])[tempa:tempb,tempa:tempb] * (2. * np.pi / (sim.wfss[0].wavelength/1e-9))
                #     phaseDifSquareX[iFrame//frames_to_close] = mps.get_phaseDifSquareX(phase)
                #     phaseDifSquareY[iFrame//frames_to_close] = mps.get_phaseDifSquareY(phase)
                #     PSD[iFrame//frames_to_close] = mps.get_PSD(phase, pixel_scale)
                #     ZernikeCoeff[iFrame//frames_to_close] = mps.get_ZernikeCoeff((phase*PUPIL).flatten(), FlatGramSchmidt, GramSchmidtCoeff2ZernikeCoeff)
                #     # plt.imshow(phase)
                #     # plt.title('iFrame = {:d}'.format(iFrame))
                #     # plt.colorbar()
                #     # plt.show()
                
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
                # Run WFS, with closed loop DM shape applied
                WFS_open_all[iFrame] = sim.runWfs(dmShape=sim.open_correction,
                                          loopIter=sim.iters)
                sim.slopes = sim.runWfs(dmShape=sim.closed_correction,
                                          loopIter=sim.iters)
            
                # Get DM shape for open loop DMs
                sim.open_correction = sim.runDM(sim.dmCommands,
                                                  closed=False)
            
                # Pass whole combined DM shapes to science target
                sim.combinedCorrection = sim.open_correction + sim.closed_correction
                
                # plt.imshow(sim.closed_correction[0])
                # plt.show()
            
                sim.runSciCams(sim.closed_correction)
            
                # Save Data
                i = sim.iters % sim.config.sim.nIters # If sim is run continuously in loop, overwrite oldest data in buffer
                sim.storeData(i)
            
                sim.printOutput(sim.iters, strehl=True)
            
                sim.addToGuiQueue()
            
                sim.iters += 1
            
            sim.saveData()
            sim.finishUp()
            
            WFS_pseudoOpen_all = sim.allSlopes[:-1] + sim.allDmCommands[1:] @ sim.recon.interaction_matrix
            
            # ri = cn22r0(r02cn2(r0) * sim.config.atmos.scrnStrengths[0])
            
            # structure_function_x = np.mean(phaseDifSquareX, axis=(0,2))
            # error_structure_function_x = np.std(phaseDifSquareX, axis=(0,2), ddof=1)/np.sqrt(TIME)
            # structure_function_y = np.mean(phaseDifSquareY, axis=(0,1))
            # error_structure_function_y = np.std(phaseDifSquareY, axis=(0,1), ddof=1)/np.sqrt(TIME)
            # theoretical_structure_function = mps.vonKarmanStructureFunction(ri, NPXLSCRN, pixel_scale, L0)
            # radius = np.arange(NPXLSCRN)*pixel_scale
            # plt.plot(radius, theoretical_structure_function,
            #           label='von Karman Structure Function', ls='-', marker='')
            # plt.errorbar(radius, structure_function_x,
            #               yerr=error_structure_function_x,
            #               ls='',marker='o', capsize=5,label='measured X-axis',
            #               alpha=ALPHA1)
            # plt.errorbar(radius, structure_function_y,
            #               yerr=error_structure_function_y,
            #               ls='',marker='o', capsize=5,label='measured Y-axis',
            #               alpha=ALPHA1)
            # plt.errorbar(radius, np.abs(structure_function_x - theoretical_structure_function),
            #               yerr=error_structure_function_x,
            #               ls='',marker='o', capsize=5,label='abs difference X-axis',
            #               alpha=ALPHA1)
            # plt.errorbar(radius, np.abs(structure_function_y - theoretical_structure_function),
            #               yerr=error_structure_function_y,
            #               ls='',marker='o', capsize=5,label='abs difference Y-axis',
            #               alpha=ALPHA1)
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.legend()
            # plt.xlabel('separation (m)')
            # plt.ylabel('structure function (radian^2)')
            # plt.xlim(radius.min(),radius.max()*2)
            # plt.ylim(np.nanmin(theoretical_structure_function)/10,
            #           np.nanmax(theoretical_structure_function)*10)
            # plt.title(SAVENAME + '\n' + '{:d}'.format(TIME)
            #         + 'structure_function'
            #         + 'D{:.0f}cm'.format(D*100)
            #         + 'r0{:.0f}cm'.format(ri*100)
            #         + 'L0{:.0f}m'.format(L0))
            # if saveFigure == True:
            #     plt.savefig(SAVENAME
            #             + '_structure_function'
            #             + '.png')
            # plt.show()
            
            # mean_PSD_x = (np.mean(PSD, axis=0).T)[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
            # error_mean_PSD_x = ((np.std(PSD, axis=0, ddof=1)/np.sqrt(TIME)).T)[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
            # mean_PSD_y = (np.mean(PSD, axis=0))[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
            # error_mean_PSD_y = ((np.std(PSD, axis=0, ddof=1)/np.sqrt(TIME)))[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
            # theoretical_PSD = mps.psd_function(ri, NPXLSCRN, pixel_scale, L0, l0)[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
            # f = (np.arange(NPXLSCRN/2-1) + 1)/NPXLSCRN/pixel_scale
            # plt.plot(f, theoretical_PSD,
            #           label='von Karman PSD', ls='-', marker='')
            # plt.errorbar(f, mean_PSD_x,
            #               yerr=error_mean_PSD_x,ls='',marker='o', capsize=5,
            #               label='measured X-axis',alpha=ALPHA2)
            # plt.errorbar(f, mean_PSD_y,
            #               yerr=error_mean_PSD_y,ls='',marker='o', capsize=5,
            #               label='measured Y-axis',alpha=ALPHA2)
            # plt.errorbar(f, np.abs(mean_PSD_x - theoretical_PSD),
            #               yerr=error_mean_PSD_x,ls='',marker='o', capsize=5,
            #               label='abs difference X-axis',alpha=ALPHA2)
            # plt.errorbar(f, np.abs(mean_PSD_y - theoretical_PSD),
            #               yerr=error_mean_PSD_y,ls='',marker='o', capsize=5,
            #               label='abs difference Y-axis',alpha=ALPHA2)
            # plt.ylim(np.nanmin(theoretical_PSD)/10,np.nanmax(theoretical_PSD)*10)
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.xlabel('spatial frequency (1/m)')
            # plt.ylabel('phase structure function (??radian??)')
            # plt.title(SAVENAME + '\n' + '{:d}'.format(TIME)
            #         + 'PSD'
            #         + 'D{:.0f}cm'.format(D*100)
            #         + 'r0{:.0f}cm'.format(ri*100)
            #         + 'L0{:.0f}m'.format(L0))
            # plt.legend()
            # if saveFigure == True:
            #     plt.savefig(SAVENAME
            #             + '_PSD'
            #             + '.png')
            # plt.show()
            
            # variance_ZernikeCoeff = np.var(ZernikeCoeff, ddof=1, axis=0)[1:]
            # error_variance_ZernikeCoeff = 2*variance_ZernikeCoeff/np.sqrt(2*NPXLSCRN - 2)
            # n_modes = variance_ZernikeCoeff.shape[0]
            # modes = np.arange(n_modes) + 1
            # theoretical_variance_ZernikeCoeff = mps.noll_variance(n_modes,NPXLSCRN,pixel_scale,ri)
            # e = NPXLSCRN*pixel_scale/L0
            # adjust = (1 - 1.42*e**(1./3.) + 3.70*e**2 - 4.01*e**(7./3.) + 4.21*e**4 - 4.00*e**(13./3.))
            # L0TipTiltVariance = theoretical_variance_ZernikeCoeff[:2] * adjust
            
            # plt.plot(modes, theoretical_variance_ZernikeCoeff, label='Kolmogorov',marker='',ls='-')
            # plt.errorbar(modes, variance_ZernikeCoeff, yerr=error_variance_ZernikeCoeff,
            #               label='measured', marker='o', ls='', capsize=5)
            # plt.plot(modes[:2], L0TipTiltVariance, label='L0 limited tip/tilt variance')
            # theoretical_variance_ZernikeCoeff[:2] = theoretical_variance_ZernikeCoeff[:2] * adjust
            # plt.errorbar(modes, np.abs(variance_ZernikeCoeff - theoretical_variance_ZernikeCoeff),
            #               yerr=error_variance_ZernikeCoeff,
            #               label='abs difference', marker='o', ls='', capsize=5)
            # plt.yscale('log')
            # plt.xlabel('zernike modes')
            # plt.ylabel('variance of zernike coefficient')
            # plt.legend()
            # plt.ylim(np.nanmin(theoretical_variance_ZernikeCoeff)/10, np.nanmax(theoretical_variance_ZernikeCoeff)*10)
            # plt.title(SAVENAME + '\n' + '{:d}'.format(TIME)
            #         + 'ZernikeVariances'
            #         + 'D{:.0f}cm'.format(D*100)
            #         + 'r0{:.0f}cm'.format(ri*100)
            #         + 'L0{:.0f}m'.format(L0))
            # if saveFigure == True:
            #     plt.savefig(SAVENAME
            #             + '_ZernikeCoeff'
            #             + '.png')
            # plt.show()
            
            WFS_pseudoOpen = WFS_pseudoOpen_all[frames_to_close - 1::frames_to_close]
            WFS_open = WFS_open_all[frames_to_close - 1:-1:frames_to_close]
            
            VMAX = np.max([np.nanmax(WFS_pseudoOpen),np.nanmax(WFS_open)])
            VMIN = np.min([np.nanmin(WFS_pseudoOpen),np.nanmin(WFS_open)])
            
            # plt.imshow(WFS_pseudoOpen,vmax=VMAX,vmin=VMIN)
            # plt.title('PseudoOpen')
            # plt.colorbar()
            # plt.show()
            
            # plt.imshow(WFS_open,vmax=VMAX,vmin=VMIN)
            # plt.title('open')
            # plt.colorbar()
            # plt.show()
            
            # WFS_pseudoOpen *= 2
            # WFS_open *= 2
            
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
            
            
            
            
            cov_x_pseudoOpen = np.zeros((N, N), dtype=float)
            cov_y_pseudoOpen = np.zeros((N, N), dtype=float)
            
            ecov_x_pseudoOpen = np.zeros((N, N), dtype=float)
            ecov_y_pseudoOpen = np.zeros((N, N), dtype=float)
            
            map_cov_x_pseudoOpen = np.zeros((2*mid + 1, 2*mid + 1), dtype=float)
            map_cov_y_pseudoOpen = np.zeros((2*mid + 1, 2*mid + 1), dtype=float)
            
            emap_cov_x_pseudoOpen = np.zeros((2*mid + 1, 2*mid + 1), dtype=float)
            emap_cov_y_pseudoOpen = np.zeros((2*mid + 1, 2*mid + 1), dtype=float)
            
            Sx1_pseudoOpen = WFS_pseudoOpen[ : , (0)*N : (1)*N]
            Sy1_pseudoOpen = WFS_pseudoOpen[ : , (1)*N : (2)*N]
            Sx2_pseudoOpen = WFS_pseudoOpen[ : , (2)*N : (3)*N]
            Sy2_pseudoOpen = WFS_pseudoOpen[ : , (3)*N : (4)*N]
            
            Sx1_pseudoOpen = (Sx1_pseudoOpen.T - np.mean(Sx1_pseudoOpen, axis=1)).T
            Sy1_pseudoOpen = (Sy1_pseudoOpen.T - np.mean(Sy1_pseudoOpen, axis=1)).T
            Sx2_pseudoOpen = (Sx2_pseudoOpen.T - np.mean(Sx2_pseudoOpen, axis=1)).T
            Sy2_pseudoOpen = (Sy2_pseudoOpen.T - np.mean(Sy2_pseudoOpen, axis=1)).T
            
            cov_x_pseudoOpen, ecov_x_pseudoOpen = my_cov(Sx1_pseudoOpen.T, Sx2_pseudoOpen.T)
            cov_y_pseudoOpen, ecov_y_pseudoOpen = my_cov(Sy1_pseudoOpen.T, Sy2_pseudoOpen.T)
            
            map_cov_x_pseudoOpen, emap_cov_x_pseudoOpen = overlay1(cov_x_pseudoOpen, ecov_x_pseudoOpen, position)
            map_cov_y_pseudoOpen, emap_cov_y_pseudoOpen = overlay1(cov_y_pseudoOpen, ecov_y_pseudoOpen, position)
            
            ###############################################################################
            ########################################################################
            name = SAVENAME + '_WFS_open'
            hdu = fits.PrimaryHDU(WFS_open)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
            
            name = SAVENAME + '_WFS_pseudoOpen'
            hdu = fits.PrimaryHDU(WFS_pseudoOpen)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
            
            name = SAVENAME + '_interaction_matrix'
            hdu = fits.PrimaryHDU(sim.recon.interaction_matrix)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
            
            np.savetxt(SAVENAME + 'RANDOMDISTR.txt', RANDOMDISTR)
            # np.savetxt(SAVENAME + 'RANDOMR0.txt', [r0])
                
            name = SAVENAME + '_map_cov_x_open'
            hdu = fits.PrimaryHDU(map_cov_x_open)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
            
            name = SAVENAME + '_map_cov_y_open'
            hdu = fits.PrimaryHDU(map_cov_y_open)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
            
            name = SAVENAME + '_emap_cov_x_open'
            hdu = fits.PrimaryHDU(emap_cov_x_open)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
            
            name = SAVENAME + '_emap_cov_y_open'
            hdu = fits.PrimaryHDU(emap_cov_y_open)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
        
            name = SAVENAME + '_map_cov_x_pseudoOpen'
            hdu = fits.PrimaryHDU(map_cov_x_pseudoOpen)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
            
            name = SAVENAME + '_map_cov_y_pseudoOpen'
            hdu = fits.PrimaryHDU(map_cov_y_pseudoOpen)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
            
            name = SAVENAME + '_emap_cov_x_pseudoOpen'
            hdu = fits.PrimaryHDU(emap_cov_x_pseudoOpen)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
            
            name = SAVENAME + '_emap_cov_y_pseudoOpen'
            hdu = fits.PrimaryHDU(emap_cov_y_pseudoOpen)
            hdul = fits.HDUList([hdu])
            hdul.writeto(name + '.fits', overwrite=True)
            hdul.close()
        
        
            ##############################################################################
            #######################################################################
        
        else:
            name = SAVENAME + '_WFS_open'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            WFS_open = hdu.data
            hdulist.close()
            
            name = SAVENAME + '_WFS_pseudoOpen'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            WFS_pseudoOpen = hdu.data
            hdulist.close()
            
            name = SAVENAME + '_interaction_matrix'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            IM = hdu.data
            hdulist.close()
            
            name = SAVENAME + '_map_cov_x_open'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            map_cov_x_open = hdu.data
            hdulist.close()
            
            name = SAVENAME + '_map_cov_y_open'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            map_cov_y_open = hdu.data
            hdulist.close()
            
            name = SAVENAME + '_emap_cov_x_open'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            emap_cov_x_open = hdu.data
            hdulist.close()
            
            name = SAVENAME + '_emap_cov_y_open'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            emap_cov_y_open = hdu.data
            hdulist.close()
            
            
            name = SAVENAME + '_map_cov_x_pseudoOpen'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            map_cov_x_pseudoOpen = hdu.data
            hdulist.close()
            
            name = SAVENAME + '_map_cov_y_pseudoOpen'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            map_cov_y_pseudoOpen = hdu.data
            hdulist.close()
            
            name = SAVENAME + '_emap_cov_x_pseudoOpen'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            emap_cov_x_pseudoOpen = hdu.data
            hdulist.close()
            
            name = SAVENAME + '_emap_cov_y_pseudoOpen'
            hdulist = fits.open(name + '.fits')
            hdu = hdulist[0]
            emap_cov_y_pseudoOpen = hdu.data
            hdulist.close()
            
            RANDOMDISTR = np.loadtxt(SAVENAME + 'RANDOMDISTR.txt')
            # RANDOMR0 = float(np.loadtxt(SAVENAME + 'RANDOMR0.txt'))
        
        ###############################################################################
        ########################################################################
        
        VMAX = np.nanmax([WFS_open,WFS_pseudoOpen])*WFSPX2NM
        VMIN = np.nanmin([WFS_open,WFS_pseudoOpen])*WFSPX2NM
        RMAX = np.max(np.abs([VMAX,VMIN]))
        RMSDIF = np.sqrt(np.mean((WFS_pseudoOpen - WFS_open)**2))
        # plt.imshow(WFS_open[:100]*WFSPX2NM,cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('sample open loop WFS OPD (nm)')
        # plt.colorbar()
        # plt.show()
        # plt.imshow(WFS_pseudoOpen[:100]*WFSPX2NM,cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('sample pseudo open loop WFS OPD (nm)')
        # plt.colorbar()
        # plt.show()
        # plt.imshow((WFS_pseudoOpen[:100] - WFS_open[:100])*WFSPX2NM,cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('sample difference between open and pseudo open loop WFS OPD (nm)\nOPD RMS={:.3f}nm'.format(RMSDIF*WFSPX2NM))
        # plt.colorbar()
        # plt.show()
        
        # HOMEDIR = 'C:/Users/gqmc62/OneDrive - Durham University/Desktop/soapy-master/soapy-master/conf'
        HOMEDIR = 'C:/Users/gqmc62/OneDrive - Durham University/Documents/PhD/Codes/conf'
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
        
        COEFFDISTRIBUTION = RANDOMDISTR
        RANDOMTOTALCN2 = r02cn2(r0)
        RANDOMCN2 = RANDOMTOTALCN2 * COEFFDISTRIBUTION
        
        RANDOMCOEFF = RANDOMCN2 / REFTOTALCN2
        
        
        
        
        NOISYRANDOMCOV_O = np.concatenate([(map_cov_x_open).flatten(),(map_cov_y_open).flatten()])
        NOISYRANDOMCOV_O = np.nan_to_num(NOISYRANDOMCOV_O)
        
        NOISYRANDOMCOV_nocentre_O = (NOISYRANDOMCOV_O[np.where(IS_COV_CENTRE == False)]).reshape((2*SCREEN*(SCREEN - 1)))
        
        NOISE_O = np.concatenate([(emap_cov_x_open).flatten(),(emap_cov_y_open).flatten()])
        NOISE_O = np.nan_to_num(NOISE_O)
        
        NOISE_nocentre_O = (NOISE_O[np.where(IS_COV_CENTRE == False)]).reshape((2*SCREEN*(SCREEN - 1)))
        
        (SOLVEDCN2_withcentre_O,
          SOLVEDCN2NOISE_withcentre_O,
          SOLVEDR0_withcentre_O,
          SOLVEDR0NOISE_withcentre_O) = lsq_cov2cn2(A_withcentre,
                                              REFTOTALCN2,
                                              NOISYRANDOMCOV_O,
                                              NOISE_O)
        (SOLVEDCN2_nocentre_O,
          SOLVEDCN2NOISE_nocentre_O,
          SOLVEDR0_nocentre_O,
          SOLVEDR0NOISE_nocentre_O) = lsq_cov2cn2(A_nocentre,
                                            REFTOTALCN2,
                                            NOISYRANDOMCOV_nocentre_O,
                                            NOISE_nocentre_O)
                                                
        
        DISTRIBUTION_ERROR_noCentre_O = (((SOLVEDCN2_nocentre_O/RANDOMTOTALCN2 - RANDOMCN2/RANDOMTOTALCN2)**2).sum())**0.5
        DISTRIBUTION_ERROR_withCentre_O = (((SOLVEDCN2_withcentre_O/RANDOMTOTALCN2 - RANDOMCN2/RANDOMTOTALCN2)**2).sum())**0.5
        r0_ERROR_noCentre_O = np.abs(SOLVEDR0_nocentre_O - r0)
        r0_ERROR_withCentre_O = np.abs(SOLVEDR0_withcentre_O - r0)
        
        
        NOISYRANDOMCOV_PO = np.concatenate([(map_cov_x_pseudoOpen).flatten(),(map_cov_y_pseudoOpen).flatten()])
        NOISYRANDOMCOV_PO = np.nan_to_num(NOISYRANDOMCOV_PO)
        
        NOISYRANDOMCOV_nocentre_PO = (NOISYRANDOMCOV_PO[np.where(IS_COV_CENTRE == False)]).reshape((2*SCREEN*(SCREEN - 1)))
        
        NOISE_PO = np.concatenate([(emap_cov_x_pseudoOpen).flatten(),(emap_cov_y_pseudoOpen).flatten()])
        NOISE_PO = np.nan_to_num(NOISE_PO)
        
        NOISE_nocentre_PO = (NOISE_PO[np.where(IS_COV_CENTRE == False)]).reshape((2*SCREEN*(SCREEN - 1)))
        
        (SOLVEDCN2_withcentre_PO,
          SOLVEDCN2NOISE_withcentre_PO,
          SOLVEDR0_withcentre_PO,
          SOLVEDR0NOISE_withcentre_PO) = lsq_cov2cn2(A_withcentre,
                                              REFTOTALCN2,
                                              NOISYRANDOMCOV_PO,
                                              NOISE_PO)
        (SOLVEDCN2_nocentre_PO,
          SOLVEDCN2NOISE_nocentre_PO,
          SOLVEDR0_nocentre_PO,
          SOLVEDR0NOISE_nocentre_PO) = lsq_cov2cn2(A_nocentre,
                                            REFTOTALCN2,
                                            NOISYRANDOMCOV_nocentre_PO,
                                            NOISE_nocentre_PO)
                                                
        
        DISTRIBUTION_ERROR_noCentre_PO = (((SOLVEDCN2_nocentre_PO/RANDOMTOTALCN2 - RANDOMCN2/RANDOMTOTALCN2)**2).sum())**0.5
        DISTRIBUTION_ERROR_withCentre_PO = (((SOLVEDCN2_withcentre_PO/RANDOMTOTALCN2 - RANDOMCN2/RANDOMTOTALCN2)**2).sum())**0.5
        r0_ERROR_noCentre_PO = np.abs(SOLVEDR0_nocentre_PO - r0)
        r0_ERROR_withCentre_PO = np.abs(SOLVEDR0_withcentre_PO - r0)
        
        all_measured_r0_O[ir0, iframeRate] = SOLVEDR0_withcentre_O
        all_measured_r0_O_nocentre[ir0, iframeRate] = SOLVEDR0_nocentre_O
        all_measured_r0_PO[ir0, iframeRate] = SOLVEDR0_withcentre_PO
        all_measured_r0_PO_nocentre[ir0, iframeRate] = SOLVEDR0_nocentre_PO
        
        all_measured_r0_noise_O[ir0, iframeRate] = SOLVEDR0NOISE_withcentre_O
        all_measured_r0_noise_O_nocentre[ir0, iframeRate] = SOLVEDR0NOISE_nocentre_O
        all_measured_r0_noise_PO[ir0, iframeRate] = SOLVEDR0NOISE_withcentre_PO
        all_measured_r0_noise_PO_nocentre[ir0, iframeRate] = SOLVEDR0NOISE_nocentre_PO
        
        
        fig, axs = plt.subplot_mosaic([['top', 'top'],['left', 'right'],],
                      constrained_layout=True,
                      gridspec_kw={'width_ratios': [1,9],'height_ratios': [1,9]},
                      figsize=(7,7))
        plt.suptitle(SAVENAME + '\nWFS slope RMS difference = {:.0f} nm'.format(RMSDIF*WFSPX2NM))
        axs['right'].set_title('Measured Cn2 distribution')
        axs['right'].errorbar(np.arange(LAYER),
                        SOLVEDCN2_nocentre_O/RANDOMTOTALCN2,
                        yerr=SOLVEDCN2NOISE_nocentre_O/RANDOMTOTALCN2,
                        ls='', marker='^', capsize=5,label='measured NO central peak open',color='r')
        axs['right'].errorbar(np.arange(LAYER),
                        SOLVEDCN2_nocentre_PO/RANDOMTOTALCN2,
                        yerr=SOLVEDCN2NOISE_nocentre_PO/RANDOMTOTALCN2,
                        ls='', marker='^', capsize=5,label='measured NO central peak pseudo open',color='g')
        axs['right'].errorbar(np.arange(LAYER),
                        SOLVEDCN2_withcentre_O/RANDOMTOTALCN2,
                        yerr=SOLVEDCN2NOISE_withcentre_O/RANDOMTOTALCN2,
                        ls='', marker='v', capsize=5,label='measured WITH central peak open',color='b')
        axs['right'].errorbar(np.arange(LAYER),
                        SOLVEDCN2_withcentre_PO/RANDOMTOTALCN2,
                        yerr=SOLVEDCN2NOISE_withcentre_PO/RANDOMTOTALCN2,
                        ls='', marker='v', capsize=5,label='measured WITH central peak pseudo open',color='k')
        axs['right'].bar(np.arange(LAYER), RANDOMCN2/RANDOMTOTALCN2,
                    ls='-', label='simulated',alpha=0.5,color='k',width=0.5)
        axs['right'].set_xlabel('Turbulence layer # from ground')
        # axs[1].legend(loc='lower center')
        axs['right'].set_yscale('log')
        axs['right'].set_ylabel('fraction of total simulated Cn2')
        axs['right'].set_ylim(0.01,1)
        
        axs['left'].errorbar([''],SOLVEDR0_nocentre_O,yerr=SOLVEDR0NOISE_nocentre_O,
                        marker='^',capsize=5,color='r')#,label='measured')
        axs['left'].errorbar([''],SOLVEDR0_nocentre_PO,yerr=SOLVEDR0NOISE_nocentre_PO,
                        marker='^',capsize=5,color='g')#,label='measured')
        axs['left'].errorbar([''],SOLVEDR0_withcentre_O,yerr=SOLVEDR0NOISE_withcentre_O,
                        marker='v',capsize=5,color='b')#,label='measured')
        axs['left'].errorbar([''],SOLVEDR0_withcentre_PO,yerr=SOLVEDR0NOISE_withcentre_PO,
                        marker='v',capsize=5,color='k')#,label='measured')
        axs['left'].bar([''],r0,color='k',alpha=0.5)#,label='simulated')
        axs['left'].set_ylabel('r0 (m)')
        axs['left'].set_ylim((0.02,0.11))
        axs['left'].set_title('Total r0 (m)')
        
        h, l = axs['right'].get_legend_handles_labels()
        axs['top'].legend(h, l, loc='center',ncol=3)
        axs['top'].axis('off')
        
        fig.tight_layout(pad=0.5)
        plt.show()
        
        
        # expected = A_withcentre @ RANDOMCOEFF
        # measured = NOISYRANDOMCOV_O
        # background = measured - expected
        
        # VMIN = np.nanmin(NOISYRANDOMCOV_O)
        # VMAX = np.nanmax(NOISYRANDOMCOV_O)
        # RMAX= np.nanmax(np.abs([VMIN,VMAX]))
        
        # plt.imshow(expected.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('expected open loop')
        # plt.colorbar()
        # if saveFigure == True:
        #     plt.savefig(SAVENAME
        #             + '_expectedCOVMapOpen'
        #             + '.png')
        # plt.show()
        
        # plt.imshow(measured.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('measured open loop')
        # plt.colorbar()
        # if saveFigure == True:
        #     plt.savefig(SAVENAME
        #             + '_measuredCOVMapOpen'
        #             + '.png')
        # plt.show()
        
        # plt.imshow(background.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('difference open loop')
        # plt.colorbar()
        # if saveFigure == True:
        #     plt.savefig(SAVENAME
        #             + '_COVMapErrorOpen'
        #             + '.png')
        # plt.show()
        
        # expected = A_withcentre @ RANDOMCOEFF
        # measured = NOISYRANDOMCOV_PO
        # background = measured - expected
        
        # VMIN = np.nanmin(NOISYRANDOMCOV_O)
        # VMAX = np.nanmax(NOISYRANDOMCOV_O)
        # RMAX= np.nanmax(np.abs([VMIN,VMAX]))
        
        # plt.imshow(expected.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('expected pseudo open loop')
        # plt.colorbar()
        # if saveFigure == True:
        #     plt.savefig(SAVENAME
        #             + '_expectedCOVMapPseudoOpen'
        #             + '.png')
        # plt.show()
        
        # plt.imshow(measured.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('measured pseudo open loop')
        # plt.colorbar()
        # if saveFigure == True:
        #     plt.savefig(SAVENAME
        #             + '_measuredCOVMapPseudoOpen'
        #             + '.png')
        # plt.show()
        
        # plt.imshow(background.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('difference pseudo open loop')
        # plt.colorbar()
        # if saveFigure == True:
        #     plt.savefig(SAVENAME
        #             + '_COVMapErrorPseudoOpen'
        #             + '.png')
        # plt.show()
        
        
        # expected = NOISYRANDOMCOV_O
        # measured = NOISYRANDOMCOV_PO
        # background = measured - expected
        
        # VMIN = np.nanmin(NOISYRANDOMCOV_O)
        # VMAX = np.nanmax(NOISYRANDOMCOV_O)
        # RMAX= np.nanmax(np.abs([VMIN,VMAX]))
        
        # plt.imshow(expected.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('measured open loop')
        # plt.colorbar()
        # if saveFigure == True:
        #     plt.savefig(SAVENAME
        #             + '_COVMapOpen'
        #             + '.png')
        # plt.show()
        
        # plt.imshow(measured.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('measured pseudo open loop')
        # plt.colorbar()
        # if saveFigure == True:
        #     plt.savefig(SAVENAME
        #             + '_COVMapPseudoOpen'
        #             + '.png')
        # plt.show()
        
        # plt.imshow(background.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
        # plt.title('difference between open and pseudo open loop')
        # plt.colorbar()
        # if saveFigure == True:
        #     plt.savefig(SAVENAME
        #             + '_COVMapErrorOpenPseudoOpen'
        #             + '.png')
        # plt.show()
        

# for ir0 in np.arange(r0List.shape[0]):
#     plt.errorbar(frameRateList, all_measured_r0_O[ir0], yerr=all_measured_r0_noise_O[ir0],
#                  label='open', ls='', capsize=5, marker='o')
#     plt.errorbar(frameRateList, all_measured_r0_PO[ir0], yerr=all_measured_r0_noise_PO[ir0],
#                  label='pseudo open', ls='', capsize=5, marker='o')
#     plt.errorbar(frameRateList, all_measured_r0_O_nocentre[ir0], yerr=all_measured_r0_noise_O_nocentre[ir0],
#                  label='open no centre', ls='', capsize=5, marker='o')
#     plt.errorbar(frameRateList, all_measured_r0_PO_nocentre[ir0], yerr=all_measured_r0_noise_PO_nocentre[ir0],
#                  label='pseudo open no centre', ls='', capsize=5, marker='o')
#     plt.hlines([r0List[ir0]], frameRateList.min(), frameRateList.max(), label='simulated')
#     plt.legend()
#     plt.xlabel('frame rate (tau0)')
#     plt.ylabel('r0 (m)')
#     plt.xlim(0.1,10)
#     plt.ylim(0,.15)
#     plt.xscale('log')
#     plt.title('measured r0')
#     plt.savefig('1fr{:d}r0{:.0}cmSLODARB1.png'.format(TIME,r0List[ir0]))
#     plt.show()
    
for iframeRate in np.arange(frameRateList.shape[0]):
    plt.errorbar(r0List, all_measured_r0_O[:,iframeRate], yerr=all_measured_r0_noise_O[:,iframeRate],
                 label='open', ls='', capsize=5, marker='o')
    plt.errorbar(r0List, all_measured_r0_PO[:,iframeRate], yerr=all_measured_r0_noise_PO[:,iframeRate],
                 label='pseudo open', ls='', capsize=5, marker='o')
    plt.errorbar(r0List, all_measured_r0_O_nocentre[:,iframeRate], yerr=all_measured_r0_noise_O_nocentre[:,iframeRate],
                 label='open no centre', ls='', capsize=5, marker='o')
    plt.errorbar(r0List, all_measured_r0_PO_nocentre[:,iframeRate], yerr=all_measured_r0_noise_PO_nocentre[:,iframeRate],
                 label='pseudo open no centre', ls='', capsize=5, marker='o')
    plt.plot(r0List, r0List, label='simulated')
    plt.legend()
    plt.xlabel('simulated r0 (m)')
    plt.ylabel('measured r0 (m)')
    plt.xlim(0,.15)
    plt.ylim(0,.15)
    plt.title('frame rate = {:.2f} tau0'.format(frameRateList[iframeRate]))
    plt.axis('square')
    plt.savefig('1fr{:d}frameRate{:.2}tau0SLODARB1.png'.format(TIME,frameRateList[iframeRate]))
    plt.show()

np.savetxt('1fr{:d}r0O'.format(TIME), all_measured_r0_O)
np.savetxt('1fr{:d}r0NoiseO'.format(TIME), all_measured_r0_noise_O)
np.savetxt('1fr{:d}r0ONoCen'.format(TIME), all_measured_r0_O_nocentre)
np.savetxt('1fr{:d}r0NoiseONoCen'.format(TIME), all_measured_r0_noise_O_nocentre)
np.savetxt('1fr{:d}r0PO'.format(TIME), all_measured_r0_PO)
np.savetxt('1fr{:d}r0NoisePO'.format(TIME), all_measured_r0_noise_PO)
np.savetxt('1fr{:d}r0PONoCen'.format(TIME), all_measured_r0_PO_nocentre)
np.savetxt('1fr{:d}r0NoisePONoCen'.format(TIME), all_measured_r0_noise_PO_nocentre)


for iframeRate in np.arange(frameRateList.shape[0]):
    plt.errorbar(x=all_measured_r0_O[:,iframeRate],
                 y=all_measured_r0_PO[:,iframeRate],
                 xerr=all_measured_r0_noise_O[:,iframeRate],
                 yerr=all_measured_r0_noise_PO[:,iframeRate],
                 label='with centre', ls='', capsize=5, marker='o')
    plt.errorbar(x=all_measured_r0_O_nocentre[:,iframeRate],
                 y=all_measured_r0_PO_nocentre[:,iframeRate],
                 xerr=all_measured_r0_noise_O_nocentre[:,iframeRate],
                 yerr=all_measured_r0_noise_PO_nocentre[:,iframeRate],
                 label='no centre', ls='', capsize=5, marker='o')
    plt.plot(r0List, r0List, label='simulated',ls='',marker='x')
    plt.legend()
    plt.xlabel('Open loop r0 (m)')
    plt.ylabel('Pseudo-open loop r0 (m)')
    plt.xlim(0,.15)
    plt.ylim(0,.15)
    plt.axis('square')
    plt.title('frame rate = {:.2f} tau0'.format(frameRateList[iframeRate]))
    plt.savefig('1fr{:d}frameRate{:.2}tau0OvsPOSLODARB1.png'.format(TIME,frameRateList[iframeRate]))
    plt.show()
    
    