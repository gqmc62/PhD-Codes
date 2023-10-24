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

def lsq_cov2cn2_with_iterative_bg(refcov,refcn2,cov,cov_noise,n_bootstrap=100):
    RESULT = np.zeros((n_bootstrap,refcov.shape[1]),dtype=float)
    PUPIL = np.ones(cov.shape)
    PUPIL[refcov[:,0] == 0] = 0
    FLAT = cov.shape[0]//2
    PUPILX = PUPIL[:FLAT]
    SCREEN0 = int(np.ceil(np.sqrt(FLAT)))
    SCREEN1 = FLAT//SCREEN0
    bg = np.zeros(cov.shape[0])
    error_change = 1e6
    old_error = 1e6
    count = 0
    for n in np.arange(n_bootstrap):
        noisy_cov = np.random.normal(cov,cov_noise)
        while ((error_change > 1e-3) and (count < 100)):
            cov_no_bg = noisy_cov - bg
            res = lsq_linear(refcov, cov_no_bg, bounds=(0,np.inf))
            solution = (refcov @ res.x) + bg
            error = noisy_cov - solution
            bg += np.concatenate([(get_bg_pistontiptilt(error[:FLAT].reshape(SCREEN0,SCREEN1),
                                                        PUPILX.reshape(SCREEN0,SCREEN1))).flatten(),
                                          (get_bg_pistontiptilt(error[FLAT:].reshape(SCREEN0,SCREEN1),
                                                                PUPILX.reshape(SCREEN0,SCREEN1))).flatten()])
            new_error = np.sqrt(np.sum(error**2))
            error_change = np.abs(new_error - old_error)/old_error
            old_error = new_error                      
            count += 1
            if count == 100:
                print('takes too long')
                return None
        RESULT[n] = res.x
        
    # VMIN = np.nanmin(cov[:FLAT])
    # VMAX = np.nanmax(cov[:FLAT])
    # plt.imshow(cov[:FLAT].reshape((SCREEN0,SCREEN1)),vmin=VMIN,vmax=VMAX)
    # plt.title('input')
    # plt.colorbar()
    # plt.show()
    # plt.imshow(((refcov @ np.mean(RESULT,axis=0))[:FLAT]).reshape((SCREEN0,SCREEN1)),vmin=VMIN,vmax=VMAX)
    # plt.title('solution')
    # plt.colorbar()
    # plt.show()
    # plt.imshow(cov[:FLAT].reshape((SCREEN0,SCREEN1)) -
    #            ((refcov @ np.mean(RESULT,axis=0))[:FLAT]).reshape((SCREEN0,SCREEN1)),vmin=VMIN,vmax=VMAX)
    # plt.title('residual')
    # plt.colorbar()
    # plt.show()
    
    cn2 = np.nanmean(RESULT,axis=0) * refcn2
    cn2_noise = np.nanstd(RESULT,axis=0,ddof=1) *refcn2
    r0 = cn22r0(cn2.sum())
    r0_noise = stdcn22stdr0(cn2.sum(),cn2_noise.sum())
    
    return cn2, cn2_noise, r0, r0_noise

def lsq_cov2cn2_with_lsq_bg(refcov,refcn2,cov,cov_noise,n_bootstrap=100):
    
    RESULT = np.zeros((n_bootstrap,refcov.shape[1] + 6),dtype=float)
    PUPIL = np.ones(cov.shape)
    PUPIL[refcov[:,0] == 0] = 0
    FLAT = cov.shape[0]//2
    SCREEN0 = int(np.ceil(np.sqrt(FLAT)))
    SCREEN1 = FLAT//SCREEN0
    PUPILX = (PUPIL[:FLAT]).reshape(SCREEN0,SCREEN1)
    bg_modes = np.zeros((refcov.shape[0],6), dtype=float)
    yy, xx = np.meshgrid(np.arange(SCREEN1,dtype=float) - SCREEN1/2 + 0.5,
                         np.arange(SCREEN0,dtype=float) - SCREEN0/2 + 0.5)
    yy *= PUPILX
    xx *= PUPILX
    normy = (yy**2).sum()
    normx = (yy**2).sum()
    yy /= normy**0.5
    xx /= normx**0.5
    
    bg_modes[:FLAT,0] = PUPILX.flatten()
    bg_modes[FLAT:,1] = PUPILX.flatten()
    bg_modes[:FLAT,2] = yy.flatten()
    bg_modes[FLAT:,3] = yy.flatten()
    bg_modes[:FLAT,4] = xx.flatten()
    bg_modes[FLAT:,5] = xx.flatten()
    
    refcov_bg = np.concatenate([refcov,bg_modes],axis=1)
    
    for n in np.arange(n_bootstrap):
        noisy_cov = np.random.normal(cov,cov_noise)
        res = lsq_linear(refcov_bg, noisy_cov, bounds=(0,np.inf))        
        RESULT[n] = res.x
        
    # VMIN = np.nanmin(cov[:FLAT])
    # VMAX = np.nanmax(cov[:FLAT])
    # plt.imshow(cov[:FLAT].reshape((SCREEN0,SCREEN1)),vmin=VMIN,vmax=VMAX,cmap='RdBu_r')
    # plt.title('input')
    # plt.colorbar()
    # plt.show()
    # plt.imshow(((refcov_bg @ np.mean(RESULT,axis=0))[:FLAT]).reshape((SCREEN0,SCREEN1)),
    #            vmin=VMIN,vmax=VMAX,cmap='RdBu_r')
    # plt.title('solution')
    # plt.colorbar()
    # plt.show()
    # plt.imshow(cov[:FLAT].reshape((SCREEN0,SCREEN1))
    #             - ((refcov_bg @ np.mean(RESULT,axis=0))[:FLAT]).reshape((SCREEN0,SCREEN1)),
    #             vmin=VMIN,vmax=VMAX,cmap='RdBu_r')
    # plt.title('residual')
    # plt.colorbar()
    # plt.show()
    # plt.imshow((bg_modes @ np.nanmean(RESULT[:,-6:],axis=0))[:FLAT].reshape(SCREEN0,SCREEN1),
    #            vmin=VMIN,vmax=VMAX,cmap='RdBu_r')
    # plt.title('Background')
    # plt.colorbar()
    # plt.show()
    # plt.imshow((refcov @ np.nanmean(RESULT[:,:-6],axis=0))[:FLAT].reshape(SCREEN0,SCREEN1)
    #            ,vmin=VMIN,vmax=VMAX,cmap='RdBu_r')
    # plt.title('No Background')
    # plt.colorbar()
    # plt.show()
    cn2 = (np.nanmean(RESULT,axis=0) * refcn2)[:refcov.shape[1]]
    cn2_noise = (np.nanstd(RESULT,axis=0,ddof=1) *refcn2)[:refcov.shape[1]]
    r0 = (cn22r0(cn2.sum()))
    r0_noise = (stdcn22stdr0(cn2.sum(),cn2_noise.sum()))
    
    return cn2, cn2_noise, r0, r0_noise

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

def get_bg_pistontiptilt(scrn,PUPIL):
    scrn = scrn*PUPIL
    yy, xx = np.meshgrid(np.arange(scrn.shape[1],dtype=float) - scrn.shape[1]/2 + 0.5,
                         np.arange(scrn.shape[0],dtype=float) - scrn.shape[0]/2 + 0.5)
    yy *= PUPIL
    xx *= PUPIL
    normy = (yy**2).sum()
    normx = (yy**2).sum()
    yy /= normy**0.5
    xx /= normx**0.5
            
    expected_BG = np.zeros(scrn.shape,dtype=float)
    expected_piston = (scrn*PUPIL).sum()/PUPIL.sum()*PUPIL
    
    
    expected_tip = np.sum((scrn - expected_piston)*yy)*yy*PUPIL
    
    expected_tilt = np.sum((scrn - expected_piston)*xx)*xx*PUPIL
    
    
    expected_BG = (expected_piston + expected_tip + expected_tilt)
    
    return expected_BG

def gauss(x, mean, std, area):
    y = area*np.exp(-((x - mean)/std)**2/2.)/(std*np.sqrt(2*np.pi))*(x[1] - x[0])
    return y

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
MAXR0 = 0.11
MINR0 = 0.02

home = ('C:/Users/gqmc62/OneDrive - Durham University/Documents/PhD/Codes/conf')
os.chdir(home)

LOAD = True

file = 'SLODAR6L.yaml'
sim = soapy.Sim(file)
LAYER = sim.config.atmos.scrnNo
TIME = sim.config.sim.nIters

SAVENAME = ('OPENSLODAR'
            + '_{:d}Frames'.format(TIME)
            + '_L0{:.0f}m'.format(sim.config.atmos.L0[0])
            + '_D{:.2f}m'.format(sim.config.tel.telDiam)
            + '_d{:.0f}cm'.format(100*sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps))


###############################################################################
########################################################################

RANDOMR0 = np.random.rand()*(MAXR0 - MINR0) + MINR0
RANDOMDISTR = np.random.rand((LAYER))
# sim.config.atmos.scrnStrengths = np.array([0.01,0.01,0.95,0.01,0.01,0.01], dtype=float)#np.random.random((LAYER))
sim.config.atmos.scrnStrengths = RANDOMDISTR
sim.config.atmos.scrnStrengths /= sim.config.atmos.scrnStrengths.sum()
sim.config.atmos.r0 = RANDOMR0
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
Zernike = zernike.zernikeArray(n_modes, NPXLSCRN)
GramSchmidt = mps.gram_schmidt_process(Zernike)
PUPIL = Zernike[0]
FlatZernike = Zernike.reshape((n_modes, NPXLSCRN*NPXLSCRN))
FlatGramSchmidt = GramSchmidt.reshape((n_modes, NPXLSCRN*NPXLSCRN))
GramSchmidtCoeff2ZernikeCoeff = mps.get_GramSchmidtCoeff2ZernikeCoeff(FlatZernike, FlatGramSchmidt)

if LOAD == False:

    ##############################################################################
    #######################################################################
    
    
    WFS_open = np.zeros((TIME,4*N), dtype=float)
    
    PSD = np.zeros((TIME, NPXLSCRN, NPXLSCRN), dtype=float)
    ZernikeCoeff = np.zeros((TIME, n_modes), dtype=float)
    phaseDifSquareX = np.zeros((TIME, NPXLSCRN, NPXLSCRN), dtype=float)
    phaseDifSquareY = np.zeros((TIME, NPXLSCRN, NPXLSCRN), dtype=float)
    
    
    
    # sim.aoinit()
    # sim.config.calcParams()
    # sim.makeIMat()
    # sim.atmos = soapy.atmosphere.atmos(sim.config)
    # sim.allDmCommands = np.zeros((sim.config.sim.nIters, sim.config.sim.totalActs))
    
    
    
    for i in np.arange(len(sim.dms)):
        sim.dms[i].gain = 0.
        sim.recon.dms[i].gain = 0.
        sim.dms[i].dmConfig.gain = 0.
    
    for layer in np.arange(LAYER):
        for row in np.arange(sim.atmos.infinite_phase_screens[layer].scrn.shape[0]*initial_flush):
            sim.atmos.infinite_phase_screens[layer].add_row()
    
    for iFrame in np.arange(TIME):
        t = time.time()
        sim.scrns = sim.atmos.moveScrns()
        
        tempa = (sim.config.sim.simSize - sim.config.sim.pupilSize)//2
        tempb = (sim.config.sim.simSize + sim.config.sim.pupilSize)//2
        phase = (sim.scrns[0])[tempa:tempb,tempa:tempb] * (2. * np.pi / (sim.wfss[0].wavelength/1e-9))
        phaseDifSquareX[iFrame] = mps.get_phaseDifSquareX(phase)
        phaseDifSquareY[iFrame] = mps.get_phaseDifSquareY(phase)
        PSD[iFrame] = mps.get_PSD(phase, pixel_scale)
        ZernikeCoeff[iFrame] = mps.get_ZernikeCoeff((phase*PUPIL).flatten(), FlatGramSchmidt, GramSchmidtCoeff2ZernikeCoeff)
        
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
        sim.slopes = sim.runWfs(dmShape=sim.closed_correction,
                                  loopIter=sim.iters)
    
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
    
    WFS_open = sim.allSlopes
    ri = cn22r0(r02cn2(RANDOMR0) * sim.config.atmos.scrnStrengths[0])
    
    structure_function_x = np.mean(phaseDifSquareX, axis=(0,2))
    error_structure_function_x = np.std(phaseDifSquareX, axis=(0,2), ddof=1)/np.sqrt(TIME)
    structure_function_y = np.mean(phaseDifSquareY, axis=(0,1))
    error_structure_function_y = np.std(phaseDifSquareY, axis=(0,1), ddof=1)/np.sqrt(TIME)
    theoretical_structure_function = mps.vonKarmanStructureFunction(ri, NPXLSCRN, pixel_scale, L0)
    radius = np.arange(NPXLSCRN)*pixel_scale
    plt.plot(radius, theoretical_structure_function,
              label='von Karman Structure Function', ls='-', marker='')
    plt.errorbar(radius, structure_function_x,
                  yerr=error_structure_function_x,
                  ls='',marker='o', capsize=5,label='measured X-axis',
                  alpha=ALPHA1)
    plt.errorbar(radius, structure_function_y,
                  yerr=error_structure_function_y,
                  ls='',marker='o', capsize=5,label='measured Y-axis',
                  alpha=ALPHA1)
    plt.errorbar(radius, np.abs(structure_function_x - theoretical_structure_function),
                  yerr=error_structure_function_x,
                  ls='',marker='o', capsize=5,label='abs difference X-axis',
                  alpha=ALPHA1)
    plt.errorbar(radius, np.abs(structure_function_y - theoretical_structure_function),
                  yerr=error_structure_function_y,
                  ls='',marker='o', capsize=5,label='abs difference Y-axis',
                  alpha=ALPHA1)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('separation (m)')
    plt.ylabel('structure function (radian^2)')
    plt.xlim(radius.min(),radius.max()*2)
    plt.ylim(np.nanmin(theoretical_structure_function)/10,
              np.nanmax(theoretical_structure_function)*10)
    plt.title(SAVENAME + '\n' + '{:d}'.format(TIME)
            + 'structure_function'
            + 'D{:.0f}cm'.format(D*100)
            + 'r0{:.0f}cm'.format(ri*100)
            + 'L0{:.0f}m'.format(L0))
    if saveFigure == True:
        plt.savefig(SAVENAME
                + '_structure_function'
                + '.png')
    plt.show()
    
    mean_PSD_x = (np.mean(PSD, axis=0).T)[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
    error_mean_PSD_x = ((np.std(PSD, axis=0, ddof=1)/np.sqrt(TIME)).T)[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
    mean_PSD_y = (np.mean(PSD, axis=0))[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
    error_mean_PSD_y = ((np.std(PSD, axis=0, ddof=1)/np.sqrt(TIME)))[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
    theoretical_PSD = mps.psd_function(ri, NPXLSCRN, pixel_scale, L0, l0)[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
    f = (np.arange(NPXLSCRN/2-1) + 1)/NPXLSCRN/pixel_scale
    plt.plot(f, theoretical_PSD,
              label='von Karman PSD', ls='-', marker='')
    plt.errorbar(f, mean_PSD_x,
                  yerr=error_mean_PSD_x,ls='',marker='o', capsize=5,
                  label='measured X-axis',alpha=ALPHA2)
    plt.errorbar(f, mean_PSD_y,
                  yerr=error_mean_PSD_y,ls='',marker='o', capsize=5,
                  label='measured Y-axis',alpha=ALPHA2)
    plt.errorbar(f, np.abs(mean_PSD_x - theoretical_PSD),
                  yerr=error_mean_PSD_x,ls='',marker='o', capsize=5,
                  label='abs difference X-axis',alpha=ALPHA2)
    plt.errorbar(f, np.abs(mean_PSD_y - theoretical_PSD),
                  yerr=error_mean_PSD_y,ls='',marker='o', capsize=5,
                  label='abs difference Y-axis',alpha=ALPHA2)
    plt.ylim(np.nanmin(theoretical_PSD)/10,np.nanmax(theoretical_PSD)*10)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('spatial frequency (1/m)')
    plt.ylabel('phase structure function (??radian??)')
    plt.title(SAVENAME + '\n' + '{:d}'.format(TIME)
            + 'PSD'
            + 'D{:.0f}cm'.format(D*100)
            + 'r0{:.0f}cm'.format(ri*100)
            + 'L0{:.0f}m'.format(L0))
    plt.legend()
    if saveFigure == True:
        plt.savefig(SAVENAME
                + '_PSD'
                + '.png')
    plt.show()
    
    variance_ZernikeCoeff = np.var(ZernikeCoeff, ddof=1, axis=0)[1:]
    error_variance_ZernikeCoeff = 2*variance_ZernikeCoeff/np.sqrt(2*NPXLSCRN - 2)
    n_modes = variance_ZernikeCoeff.shape[0]
    modes = np.arange(n_modes) + 1
    theoretical_variance_ZernikeCoeff = mps.noll_variance(n_modes,NPXLSCRN,pixel_scale,ri)
    e = NPXLSCRN*pixel_scale/L0
    adjust = (1 - 1.42*e**(1./3.) + 3.70*e**2 - 4.01*e**(7./3.) + 4.21*e**4 - 4.00*e**(13./3.))
    L0TipTiltVariance = theoretical_variance_ZernikeCoeff[:2] * adjust
    
    plt.plot(modes, theoretical_variance_ZernikeCoeff, label='Kolmogorov',marker='',ls='-')
    plt.errorbar(modes, variance_ZernikeCoeff, yerr=error_variance_ZernikeCoeff,
                  label='measured', marker='o', ls='', capsize=5)
    plt.plot(modes[:2], L0TipTiltVariance, label='L0 limited tip/tilt variance')
    theoretical_variance_ZernikeCoeff[:2] = theoretical_variance_ZernikeCoeff[:2] * adjust
    plt.errorbar(modes, np.abs(variance_ZernikeCoeff - theoretical_variance_ZernikeCoeff),
                  yerr=error_variance_ZernikeCoeff,
                  label='abs difference', marker='o', ls='', capsize=5)
    plt.yscale('log')
    plt.xlabel('zernike modes')
    plt.ylabel('variance of zernike coefficient')
    plt.legend()
    plt.ylim(np.nanmin(theoretical_variance_ZernikeCoeff)/10, np.nanmax(theoretical_variance_ZernikeCoeff)*10)
    plt.title(SAVENAME + '\n' + '{:d}'.format(TIME)
            + 'ZernikeVariances'
            + 'D{:.0f}cm'.format(D*100)
            + 'r0{:.0f}cm'.format(ri*100)
            + 'L0{:.0f}m'.format(L0))
    if saveFigure == True:
        plt.savefig(SAVENAME
                + '_ZernikeCoeff'
                + '.png')
    plt.show()
    
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
    
    ###############################################################################
    ########################################################################
    name = SAVENAME + '_WFS_open'
    hdu = fits.PrimaryHDU(WFS_open)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + '.fits', overwrite=True)
    hdul.close()
    
    np.savetxt(SAVENAME + 'RANDOMDISTR.txt', RANDOMDISTR)
    np.savetxt(SAVENAME + 'RANDOMR0.txt', [RANDOMR0])
        
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

    ##############################################################################
    #######################################################################

else:
    name = SAVENAME + '_WFS_open'
    hdulist = fits.open(name + '.fits')
    hdu = hdulist[0]
    WFS_open = hdu.data
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
    
    RANDOMDISTR = np.loadtxt(SAVENAME + 'RANDOMDISTR.txt')
    RANDOMR0 = float(np.loadtxt(SAVENAME + 'RANDOMR0.txt'))

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

COEFFDISTRIBUTION = RANDOMDISTR
RANDOMTOTALCN2 = r02cn2(RANDOMR0)
RANDOMCN2 = RANDOMTOTALCN2 * COEFFDISTRIBUTION

RANDOMCOEFF = RANDOMCN2 / REFTOTALCN2

# NOISYRANDOMCOV = A_withcentre @ RANDOMCOEFF #np.concatenate([(map_cov_x_open).flatten(),(map_cov_y_open).flatten()])
NOISYRANDOMCOV = np.concatenate([(map_cov_x_open).flatten(),(map_cov_y_open).flatten()])
NOISYRANDOMCOV = np.nan_to_num(NOISYRANDOMCOV)
# NOISYRANDOMCOV[NOISYRANDOMCOV == 0] = np.nan

NOISYRANDOMCOV_nocentre = (NOISYRANDOMCOV[np.where(IS_COV_CENTRE == False)]).reshape((2*SCREEN*(SCREEN - 1)))

NOISE = np.concatenate([(emap_cov_x_open).flatten(),(emap_cov_y_open).flatten()])
NOISE = np.nan_to_num(NOISE)
# NOISE[NOISE == 0] = np.nan

NOISE_nocentre = (NOISE[np.where(IS_COV_CENTRE == False)]).reshape((2*SCREEN*(SCREEN - 1)))

# (SOLVEDCN2_withcentre,
#   SOLVEDCN2NOISE_withcentre,
#   SOLVEDR0_withcentre,
#   SOLVEDR0NOISE_withcentre) = lsq_cov2cn2_with_bg(A_withcentre,
#                                       REFTOTALCN2,
#                                       NOISYRANDOMCOV,
#                                       NOISE)
# (SOLVEDCN2_nocentre,
#   SOLVEDCN2NOISE_nocentre,
#   SOLVEDR0_nocentre,
#   SOLVEDR0NOISE_nocentre) = lsq_cov2cn2_with_bg(A_nocentre,
#                                     REFTOTALCN2,
#                                     NOISYRANDOMCOV_nocentre,
#                                     NOISE_nocentre)

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
                                        
# (SOLVEDCN2_withcentre,
#   SOLVEDCN2NOISE_withcentre,
#   SOLVEDR0_withcentre,
#   SOLVEDR0NOISE_withcentre) = lsq_cov2cn2_with_lsq_bg(A_withcentre,
#                                       REFTOTALCN2,
#                                       NOISYRANDOMCOV,
#                                       NOISE)
# (SOLVEDCN2_nocentre,
#   SOLVEDCN2NOISE_nocentre,
#   SOLVEDR0_nocentre,
#   SOLVEDR0NOISE_nocentre) = lsq_cov2cn2_with_lsq_bg(A_nocentre,
#                                     REFTOTALCN2,
#                                     NOISYRANDOMCOV_nocentre,
#                                     NOISE_nocentre)

                                    
# (SOLVEDCN2_withcentre,
#   SOLVEDCN2NOISE_withcentre,
#   SOLVEDR0_withcentre,
#   SOLVEDR0NOISE_withcentre) = cov2cn2(np.linalg.pinv(A_withcentre),
#                                       REFTOTALCN2,
#                                       NOISYRANDOMCOV,
#                                       NOISE)
# (SOLVEDCN2_nocentre,
#   SOLVEDCN2NOISE_nocentre,
#   SOLVEDR0_nocentre,
#   SOLVEDR0NOISE_nocentre) = cov2cn2(np.linalg.pinv(A_nocentre),
#                                     REFTOTALCN2,
#                                     NOISYRANDOMCOV_nocentre,
#                                     NOISE_nocentre)

DISTRIBUTION_ERROR_noCentre = (((SOLVEDCN2_nocentre/RANDOMTOTALCN2 - RANDOMCN2/RANDOMTOTALCN2)**2).sum())**0.5
DISTRIBUTION_ERROR_withCentre = (((SOLVEDCN2_withcentre/RANDOMTOTALCN2 - RANDOMCN2/RANDOMTOTALCN2)**2).sum())**0.5
r0_ERROR_noCentre = np.abs(SOLVEDR0_nocentre - RANDOMR0)
r0_ERROR_withCentre = np.abs(SOLVEDR0_withcentre - RANDOMR0)


fig, axs = plt.subplot_mosaic([['top', 'top'],['left', 'right'],],
              constrained_layout=True,
              gridspec_kw={'width_ratios': [1,9],'height_ratios': [1,9]},
              figsize=(7,7))
plt.suptitle('Statistics checks on Cn2 decomposition from generated cov map\n'
              + 'Total Cn2={:.2e}m$^-$$^2$$^/$$^3$\n'.format(RANDOMTOTALCN2)
              + 'RMS error of distribution ratio : NO centre fit = {:.2f}, WITH centre fit = {:.2f}\n'.format(
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


expected = A_withcentre @ RANDOMCOEFF
measured = NOISYRANDOMCOV
background = measured - expected

VMIN = np.nanmin(NOISYRANDOMCOV)
VMAX = np.nanmax(NOISYRANDOMCOV)
RMAX= np.nanmax(np.abs([VMIN,VMAX]))

plt.imshow(expected.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
plt.title('expected')
plt.colorbar()
if saveFigure == True:
    plt.savefig(SAVENAME
            + '_expectedCOVMap'
            + '.png')
plt.show()

plt.imshow(measured.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
plt.title('measured')
plt.colorbar()
if saveFigure == True:
    plt.savefig(SAVENAME
            + '_measuredCOVMap'
            + '.png')
plt.show()

plt.imshow(background.reshape(22,11),cmap='RdBu_r',norm=colors.CenteredNorm(vcenter=0,halfrange=RMAX))
plt.title('difference (measure - expected)')
plt.colorbar()
if saveFigure == True:
    plt.savefig(SAVENAME
            + '_COVMapError'
            + '.png')
plt.show()

