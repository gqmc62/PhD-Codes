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
import my_WFS as mwfs
import my_phasescreen as mps
from aotools.functions import zernike
import my_propagation as mprop

def remove_piston_tiptilt(scrns):
    yy, xx = np.meshgrid(np.arange(len(scrns)),np.arange(len(scrns)))
    norm = np.sum(xx, axis=(0,1))
    temp = scrns - np.mean(scrns, axis=(0,1))
    tip = np.sum(temp*yy, axis=(0,1))/norm**2
    tilt = np.sum(temp*xx, axis=(0,1))/norm**2
    temp -= (tip*yy + tilt*xx)
    return temp

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

###############################################################################
########################################################################

ARCSEC2RAD = 1./3600./360.*2.*np.pi
RAD2ARCSEC = 1./ARCSEC2RAD

home = ('C:/Users/gqmc62/OneDrive - Durham University/Desktop/'
              + 'soapy-master/soapy-master/conf')
os.chdir(home)

file = 'slope_covariance_most_correct.yaml'
sim = soapy.Sim(file)

sim.config.randomSeed = 0

sim.aoinit()
sim.config.calcParams()
sim.makeIMat()

N = sim.wfss[0].n_subaps

scrnHeights_list = (np.arange(sim.config.wfss[0].nxSubaps - 2, dtype=float)
                    * (sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)
                    / ((sim.config.wfss[0].GSPosition[0]
                        - sim.config.wfss[1].GSPosition[0])*ARCSEC2RAD))
# LAYER = scrnHeights_list.shape[0]
LAYER = 1
TIME = sim.config.sim.nIters
RANDOM_SEED_LIST = np.random.randint(0, 2**31 - 1, (LAYER, TIME))

position = np.asarray(sim.wfss[0].pupil_subap_coords/sim.wfss[0].nx_subap_pixels,dtype=int) - 1

mid = sim.config.wfss[0].nxSubaps - 1 - 1 - 1

pixel_scale = sim.config.tel.telDiam / (sim.config.wfss[0].nxSubaps * sim.config.wfss[0].pxlsPerSubap)

grid = np.arange(mid + 1,dtype=float)
grid -= len(grid)/2
grid += 0.5



ALPHA1 = 0.5
ALPHA2 = 0.5
saveFigure = True


###############################################################################
########################################################################

WFS_open = np.zeros((LAYER,TIME,4*N), dtype=float)


D = sim.config.tel.telDiam
L0 = sim.config.atmos.L0[0]
l0 = 1e-10
r0 = sim.config.atmos.r0
n_modes = 32 + 1
NPXLSCRN = (sim.config.wfss[0].nxSubaps * sim.config.wfss[0].pxlsPerSubap)
ovf = sim.config.sim.simOversize
initial_flush = 4
LATER_FLUSH = 1
later_flush = int(np.ceil(LATER_FLUSH*L0/pixel_scale))
Zernike = zernike.zernikeArray(n_modes, NPXLSCRN)
GramSchmidt = mps.gram_schmidt_process(Zernike)
PUPIL = Zernike[0]
FlatZernike = Zernike.reshape((n_modes, NPXLSCRN*NPXLSCRN))
FlatGramSchmidt = GramSchmidt.reshape((n_modes, NPXLSCRN*NPXLSCRN))
GramSchmidtCoeff2ZernikeCoeff = mps.get_GramSchmidtCoeff2ZernikeCoeff(FlatZernike, FlatGramSchmidt)
SAVENAME = 'PLZWRK'+'_{:d}Frames_x{:d}InitialFlush_x{:d}LaterFlush_x{:d}Oversize'.format(TIME,initial_flush, LATER_FLUSH, ovf)

for layer in tqdm(np.arange(LAYER)):
    PSD = np.zeros((TIME, NPXLSCRN, NPXLSCRN), dtype=float)
    ZernikeCoeff = np.zeros((TIME, n_modes), dtype=float)
    phaseDifSquareX = np.zeros((TIME, NPXLSCRN, NPXLSCRN), dtype=float)
    phaseDifSquareY = np.zeros((TIME, NPXLSCRN, NPXLSCRN), dtype=float)
    
    print('at layer {:d}'.format(layer))
    scrnHeights = scrnHeights_list[layer]
    sim.config.atmos.scrnHeights = [scrnHeights]
    sim.aoinit()
    sim.config.calcParams()
    sim.makeIMat()
    # sim.config.randomSeed = SEEDS[layer]
    sim.atmos = soapy.atmosphere.atmos(sim.config)
    sim.allDmCommands = np.zeros((sim.config.sim.nIters, sim.config.sim.totalActs))
    
    for i in np.arange(len(sim.dms)):
        sim.dms[i].gain = 0.
        sim.recon.dms[i].gain = 0.
        sim.dms[i].dmConfig.gain = 0.
    sim.atmos.infinite_phase_screens[0].random_seed = RANDOM_SEED_LIST[layer,0]
    sim.atmos.infinite_phase_screens[0].make_initial_screen()
    for row in np.arange(sim.atmos.infinite_phase_screens[0].scrn.shape[0]*initial_flush):
        sim.atmos.infinite_phase_screens[0].add_row()
    
    # sim.aoloop()
    for iFrame in np.arange(TIME):
        t = time.time()
        
        # sim.scrns = sim.atmos.moveScrns()
        # sim.atmos.infinite_phase_screens[0].random_seed = RANDOM_SEED_LIST[layer,iFrame]
        # sim.atmos.infinite_phase_screens[0].make_initial_screen()
        for row in np.arange(later_flush):
            sim.atmos.infinite_phase_screens[0].add_row()
        
        if layer == 0:
            phase = mprop.crop2(sim.atmos.infinite_phase_screens[0].scrn, ovf)
            phaseDifSquareX[iFrame] = mps.get_phaseDifSquareX(phase)
            phaseDifSquareY[iFrame] = mps.get_phaseDifSquareY(phase)
            PSD[iFrame] = mps.get_PSD(phase, pixel_scale)
            ZernikeCoeff[iFrame] = mps.get_ZernikeCoeff((phase*PUPIL).flatten(), FlatGramSchmidt, GramSchmidtCoeff2ZernikeCoeff)
        
        sim.scrns = sim.atmos.infinite_phase_screens[0].scrn / 2. / np.pi * 500
        
        sim.Tatmos += time.time()-t

        if iFrame == 0:
            plt.imshow(phase)
            plt.colorbar()
            plt.show()

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

        # plt.pcolor(sim.scrns[0],cmap='RdBu_r')
        # plt.colorbar()
        # # plt.title('phase screen {:d}'.format(iFrame))
        # # plt.show()
        # plt.quiver((position[:,0] + 1.5)*sim.config.wfss[0].pxlsPerSubap,
        #            (position[:,1] + 1.5)*sim.config.wfss[0].pxlsPerSubap,
        #            sim.slopes[:N], sim.slopes[N:2*N])
        # plt.axis('square')
        # plt.title('WFS slopes')
        # plt.show()
        
        # plt.pcolor(sim.scrns[0],cmap='RdBu_r')
        # plt.colorbar()
        # plt.quiver((position[:,0] + 1.5)*sim.config.wfss[0].pxlsPerSubap,
        #            (position[:,1] + 1.5)*sim.config.wfss[0].pxlsPerSubap,
        #            sim.slopes[:N], sim.slopes[N:2*N],color='r')
        # plt.axis('square')
        # plt.title('WFS slopes')
        # plt.show()
        
    sim.saveData()
    sim.finishUp()
    
    WFS_open[layer] = sim.allSlopes
    
    if layer == 0:
    
        structure_function_x = np.mean(phaseDifSquareX, axis=(0,2))
        error_structure_function_x = np.std(phaseDifSquareX, axis=(0,2), ddof=1)/np.sqrt(TIME)
        structure_function_y = np.mean(phaseDifSquareY, axis=(0,1))
        error_structure_function_y = np.std(phaseDifSquareY, axis=(0,1), ddof=1)/np.sqrt(TIME)
        theoretical_structure_function = mps.vonKarmanStructureFunction(r0, NPXLSCRN, pixel_scale, L0)
        radius = np.arange(NPXLSCRN)*pixel_scale
        plt.plot(radius, theoretical_structure_function,
                 label='von Karman Structure Function', ls='-', marker='')
        plt.errorbar(radius, structure_function_x,
                     yerr=error_structure_function_x,
                     ls='',marker='o', capsize=5,label='measured T-axis',
                     alpha=ALPHA1)
        plt.errorbar(radius, structure_function_y,
                     yerr=error_structure_function_y,
                     ls='',marker='o', capsize=5,label='measured L-axis',
                     alpha=ALPHA1)
        plt.errorbar(radius, np.abs(structure_function_x - theoretical_structure_function),
                     yerr=error_structure_function_x,
                     ls='',marker='o', capsize=5,label='abs difference T-axis',
                     alpha=ALPHA1)
        plt.errorbar(radius, np.abs(structure_function_y - theoretical_structure_function),
                     yerr=error_structure_function_y,
                     ls='',marker='o', capsize=5,label='abs difference L-axis',
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
                + 'r0{:.0f}cm'.format(r0*100)
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
        theoretical_PSD = mps.psd_function(r0, NPXLSCRN, pixel_scale, L0, l0)[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
        f = (np.arange(NPXLSCRN/2-1) + 1)/NPXLSCRN/pixel_scale
        plt.plot(f, theoretical_PSD,
                 label='von Karman PSD', ls='-', marker='')
        plt.errorbar(f, mean_PSD_x,
                     yerr=error_mean_PSD_x,ls='',marker='o', capsize=5,
                     label='measured T-axis',alpha=ALPHA2)
        plt.errorbar(f, mean_PSD_y,
                     yerr=error_mean_PSD_y,ls='',marker='o', capsize=5,
                     label='measured L-axis',alpha=ALPHA2)
        plt.errorbar(f, np.abs(mean_PSD_x - theoretical_PSD),
                     yerr=error_mean_PSD_x,ls='',marker='o', capsize=5,
                     label='abs difference T-axis',alpha=ALPHA2)
        plt.errorbar(f, np.abs(mean_PSD_y - theoretical_PSD),
                     yerr=error_mean_PSD_y,ls='',marker='o', capsize=5,
                     label='abs difference L-axis',alpha=ALPHA2)
        plt.ylim(np.nanmin(theoretical_PSD)/10,np.nanmax(theoretical_PSD)*10)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('spatial frequency (1/m)')
        plt.ylabel('phase structure function (??radian??)')
        plt.title(SAVENAME + '\n' + '{:d}'.format(TIME)
                + 'PSD'
                + 'D{:.0f}cm'.format(D*100)
                + 'r0{:.0f}cm'.format(r0*100)
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
        theoretical_variance_ZernikeCoeff = mps.noll_variance(n_modes,NPXLSCRN,pixel_scale,r0)
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
                + 'r0{:.0f}cm'.format(r0*100)
                + 'L0{:.0f}m'.format(L0))
        if saveFigure == True:
            plt.savefig(SAVENAME
                    + '_ZernikeCoeff'
                    + '.png')
        plt.show()
    

# WFS_close = np.zeros((LAYER,TIME,4*N), dtype=float)
# WFS_pseudo_open = np.zeros((LAYER,TIME,4*N), dtype=float)
# DM_close = np.zeros((LAYER,sim.config.sim.nIters, sim.config.sim.totalActs), dtype=float)

# for layer in np.arange(LAYER):
#     print('at layer {:d}'.format(layer))
#     scrnHeights = scrnHeights_list[layer]
#     sim.config.atmos.scrnHeights = [scrnHeights]
#     sim.aoinit()
#     sim.makeIMat(forceNew=True)
#     # sim.config.randomSeed = SEEDS[layer]
#     sim.atmos = soapy.atmosphere.atmos(sim.config)
#     sim.allDmCommands = np.zeros((sim.config.sim.nIters, sim.config.sim.totalActs))
    
#     for i in np.arange(len(sim.dms)):
#         sim.dms[i].gain = 1.
#         sim.recon.dms[i].gain = 1.
#         sim.dms[i].dmConfig.gain = 1.
    
#     # sim.aoloop()
#     for iFrame in np.arange(TIME):
#         t = time.time()
        
#         # sim.scrns = sim.atmos.moveScrns()
#         sim.atmos.infinite_phase_screens[0].random_seed = RANDOM_SEED_LIST[layer,iFrame]
#         sim.atmos.infinite_phase_screens[0].make_initial_screen()
#         for row in np.arange(sim.atmos.infinite_phase_screens[0].scrn.shape[0]*16):
#             sim.atmos.infinite_phase_screens[0].add_row()
#         sim.scrns = sim.atmos.infinite_phase_screens[0].scrn / 2. / np.pi * 500
#         # plt.imshow(sim.scrns)
#         # plt.title('{:d}'.format(iFrame))
#         # plt.show()
        
#         sim.Tatmos += time.time()-t

#         # Run Loop...
#         ########################################

#         # Get dmCommands from reconstructor
#         t_recon = time.time()
#         if sim.config.sim.nDM:
#             sim.dmCommands[:] = sim.recon.reconstruct(sim.slopes)
#         sim.Trecon += (time.time() - t_recon)

#         # Delay the dmCommands if loopDelay is configured
#         sim.dmCommands = sim.buffer.delay(sim.dmCommands, sim.config.sim.loopDelay)

#         # Get dmShape from closed loop DMs
#         sim.closed_correction = sim.runDM(
#                 sim.dmCommands, closed=True)

#         # Run WFS, with closed loop DM shape applied
#         sim.slopes = sim.runWfs(dmShape=sim.closed_correction,
#                                   loopIter=sim.iters)

#         # Get DM shape for open loop DMs
#         sim.open_correction = sim.runDM(sim.dmCommands,
#                                           closed=False)

#         # Pass whole combined DM shapes to science target
#         sim.combinedCorrection = sim.open_correction + sim.closed_correction

#         sim.runSciCams(sim.combinedCorrection)

#         # Save Data
#         i = sim.iters % sim.config.sim.nIters # If sim is run continuously in loop, overwrite oldest data in buffer
#         sim.storeData(i)

#         sim.printOutput(sim.iters, strehl=True)

#         sim.addToGuiQueue()

#         sim.iters += 1
    
#     WFS_close[layer] = sim.allSlopes
#     DM_close[layer] = sim.allDmCommands
#     WFS_pseudo_open[layer] = sim.allSlopes + np.matmul(sim.allDmCommands, sim.recon.interaction_matrix)


###############################################################################
########################################################################

name = SAVENAME + '_WFS_open'
hdu = fits.PrimaryHDU(WFS_open)
hdul = fits.HDUList([hdu])
hdul.writeto(name + '.fits', overwrite=True)
hdul.close()


# name = 'WFS_close'
# hdu = fits.PrimaryHDU(WFS_close)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()

# name = 'WFS_pseudo_open'
# hdu = fits.PrimaryHDU(WFS_pseudo_open)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()

# name = 'DM_close'
# hdu = fits.PrimaryHDU(DM_close)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()

# name = 'CM'
# hdu = fits.PrimaryHDU(sim.recon.control_matrix)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()

# name = 'IM'
# hdu = fits.PrimaryHDU(sim.recon.interaction_matrix)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()

###############################################################################
########################################################################

name = SAVENAME + '_WFS_open'
hdulist = fits.open(name + '.fits')
hdu = hdulist[0]
WFS_open = hdu.data
hdulist.close()

# name = 'WFS_close'
# hdulist = fits.open(name + '.fits')
# hdu = hdulist[0]
# WFS_close = hdu.data
# hdulist.close()

# name = 'WFS_pseudo_open'
# hdulist = fits.open(name + '.fits')
# hdu = hdulist[0]
# WFS_pseudo_open = hdu.data
# hdulist.close()

# name = 'DM_close'
# hdulist = fits.open(name + '.fits')
# hdu = hdulist[0]
# DM_close = hdu.data
# hdulist.close()

# name = 'CM'
# hdulist = fits.open(name + '.fits')
# hdu = hdulist[0]
# CM = hdu.data
# hdulist.close()

# name = 'IM'
# hdulist = fits.open(name + '.fits')
# hdu = hdulist[0]
# IM = hdu.data
# hdulist.close()

###############################################################################
########################################################################

# MAX = np.nanmax(WFS_open)
# MIN = np.nanmin(WFS_open)

# plt.imshow(WFS_open[2,:100],vmin=MIN,vmax=MAX)
# plt.colorbar()
# plt.xlabel('slopes')
# plt.ylabel('unique screens')
# plt.title('WFS open slope')
# plt.show()

# plt.imshow(WFS_close[2,:100],vmin=MIN,vmax=MAX)
# plt.colorbar()
# plt.xlabel('slopes')
# plt.ylabel('unique screens')
# plt.title('WFS close slope')
# plt.show()

# plt.imshow((np.matmul(DM_close[2], sim.recon.interaction_matrix))[:100],vmin=MIN,vmax=MAX)
# plt.colorbar()
# plt.xlabel('slopes')
# plt.ylabel('unique screens')
# plt.title('DM close slope')
# plt.show()

# plt.imshow(WFS_close[2,:100] + (np.matmul(DM_close[2], sim.recon.interaction_matrix))[:100],vmin=MIN,vmax=MAX)
# plt.colorbar()
# plt.xlabel('slopes')
# plt.ylabel('unique screens')
# plt.title('pseudo-open slope')
# plt.show()

# plt.imshow(WFS_close[2,:100] - (np.matmul(DM_close[2], sim.recon.interaction_matrix))[:100],vmin=MIN,vmax=MAX)
# plt.colorbar()
# plt.xlabel('slopes')
# plt.ylabel('unique screens')
# plt.title('?pseudo-open? slope')
# plt.show()

# plt.imshow(WFS_close[2,1:101] + (np.matmul(DM_close[2], sim.recon.interaction_matrix))[:100],vmin=MIN,vmax=MAX)
# plt.colorbar()
# plt.xlabel('slopes')
# plt.ylabel('unique screens')
# plt.title('?pseudo-open? slope')
# plt.show()

# plt.imshow(WFS_close[2,1:101] - (np.matmul(DM_close[2], sim.recon.interaction_matrix))[:100],vmin=MIN,vmax=MAX)
# plt.colorbar()
# plt.xlabel('slopes')
# plt.ylabel('unique screens')
# plt.title('?pseudo-open? slope')
# plt.show()

# plt.imshow(WFS_close[2,:100] + (np.matmul(DM_close[2], sim.recon.interaction_matrix))[1:101],vmin=MIN,vmax=MAX)
# plt.colorbar()
# plt.xlabel('slopes')
# plt.ylabel('unique screens')
# plt.title('?pseudo-open? slope')
# plt.show()

# plt.imshow(WFS_close[2,:100] - (np.matmul(DM_close[2], sim.recon.interaction_matrix))[1:101],vmin=MIN,vmax=MAX)
# plt.colorbar()
# plt.xlabel('slopes')
# plt.ylabel('unique screens')
# plt.title('?pseudo-open? slope')
# plt.show()

cov_x_open = np.zeros((LAYER, N, N), dtype=float)
cov_y_open = np.zeros((LAYER, N, N), dtype=float)
# cov_x_pseudo_open = np.zeros((LAYER, N, N), dtype=float)
# cov_y_pseudo_open = np.zeros((LAYER, N, N), dtype=float)

ecov_x_open = np.zeros((LAYER, N, N), dtype=float)
ecov_y_open = np.zeros((LAYER, N, N), dtype=float)
# ecov_x_pseudo_open = np.zeros((LAYER, N, N), dtype=float)
# ecov_y_pseudo_open = np.zeros((LAYER, N, N), dtype=float)

map_cov_x_open = np.zeros((LAYER, 2*mid + 1, 2*mid + 1), dtype=float)
map_cov_y_open = np.zeros((LAYER, 2*mid + 1, 2*mid + 1), dtype=float)
# map_cov_x_pseudo_open = np.zeros((LAYER, 2*mid + 1, 2*mid + 1), dtype=float)
# map_cov_y_pseudo_open = np.zeros((LAYER, 2*mid + 1, 2*mid + 1), dtype=float)

emap_cov_x_open = np.zeros((LAYER, 2*mid + 1, 2*mid + 1), dtype=float)
emap_cov_y_open = np.zeros((LAYER, 2*mid + 1, 2*mid + 1), dtype=float)
# emap_cov_x_pseudo_open = np.zeros((LAYER, 2*mid + 1, 2*mid + 1), dtype=float)
# emap_cov_y_pseudo_open = np.zeros((LAYER, 2*mid + 1, 2*mid + 1), dtype=float)

for layer in np.arange(LAYER):
    # layer= 2
    Sx1_open = WFS_open[layer, : , (0)*N : (1)*N]
    Sy1_open = WFS_open[layer, : , (1)*N : (2)*N]
    Sx2_open = WFS_open[layer, : , (2)*N : (3)*N]
    Sy2_open = WFS_open[layer, : , (3)*N : (4)*N]
    
    # Sx1_pseudo_open = WFS_pseudo_open[layer, : , (0)*N : (1)*N]
    # Sy1_pseudo_open = WFS_pseudo_open[layer, : , (1)*N : (2)*N]
    # Sx2_pseudo_open = WFS_pseudo_open[layer, : , (2)*N : (3)*N]
    # Sy2_pseudo_open = WFS_pseudo_open[layer, : , (3)*N : (4)*N]
    
    # Sx1_open = np.subtract(Sx1_open, np.mean(Sx1_open, axis=1))
    # Sy1_open = np.subtract(Sy1_open, np.mean(Sy1_open, axis=1))
    # Sx2_open = np.subtract(Sx2_open, np.mean(Sx2_open, axis=1))
    # Sy2_open = np.subtract(Sy2_open, np.mean(Sy2_open, axis=1))
    
    # Sx1_pseudo_open = np.subtract(Sx1_pseudo_open, np.mean(Sx1_pseudo_open, axis=1))
    # Sy1_pseudo_open = np.subtract(Sy1_pseudo_open, np.mean(Sy1_pseudo_open, axis=1))
    # Sx2_pseudo_open = np.subtract(Sx2_pseudo_open, np.mean(Sx2_pseudo_open, axis=1))
    # Sy2_pseudo_open = np.subtract(Sy2_pseudo_open, np.mean(Sy2_pseudo_open, axis=1))
    
    Sx1_open = (Sx1_open.T - np.mean(Sx1_open, axis=1)).T
    Sy1_open = (Sy1_open.T - np.mean(Sy1_open, axis=1)).T
    Sx2_open = (Sx2_open.T - np.mean(Sx2_open, axis=1)).T
    Sy2_open = (Sy2_open.T - np.mean(Sy2_open, axis=1)).T
    
    # Sx1_pseudo_open = (Sx1_pseudo_open.T - np.mean(Sx1_pseudo_open, axis=1)).T
    # Sy1_pseudo_open = (Sy1_pseudo_open.T - np.mean(Sy1_pseudo_open, axis=1)).T
    # Sx2_pseudo_open = (Sx2_pseudo_open.T - np.mean(Sx2_pseudo_open, axis=1)).T
    # Sy2_pseudo_open = (Sy2_pseudo_open.T - np.mean(Sy2_pseudo_open, axis=1)).T
    
    # # check that tiptilt of each time frame is taken out
    # for time in np.arange(10):
    #     plt.imshow(rearrange1(Sx1_open[time],position),cmap='RdBu')
    #     plt.colorbar()
    #     plt.show()
    
    # # check that for open loop, there is 'correlation' between wfss
    # for time in np.arange(10):
    #     plt.quiver(rearrange1(Sx1_open[time],position),
    #                rearrange1(Sy1_open[time],position),
    #                color='r',label='1')
    #     plt.quiver(rearrange1(Sx2_open[time],position),
    #                rearrange1(Sy2_open[time],position),
    #                color='b',label='2')
    #     plt.axis('square')
    #     plt.legend(loc='upper left')
    #     plt.title('slopes comparison between wfs1 vs wfs2')
    #     plt.show()
    
    # # check that for open loop, there is 'correlation' between wfss
    # for time in np.arange(10):
    #     plt.quiver(rearrange1(Sx1_open[time],position),
    #                rearrange1(Sy1_open[time],position),
    #                color='r',label='open')
    #     plt.quiver(rearrange1(Sx1_pseudo_open[time],position),
    #                rearrange1(Sy1_pseudo_open[time],position),
    #                color='b',label='pseudo-open')
    #     plt.axis('square')
    #     plt.legend(loc='upper left')
    #     plt.title('slopes comparison between open vs pseudo-open')
    #     plt.show()
    
    # COV = np.zeros((N,N),dtype=float)
    
    # for lenslet1 in np.arange(N):
    #     for lenslet2 in np.arange(N):
    #         # lenslet1 = 4
    #         # lenslet2 = 16
    #         A = (Sx1_open[:,lenslet1] - np.mean(Sx1_open[:,lenslet1], axis=0))
    #         B = (Sx2_open[:,lenslet2] - np.mean(Sx2_open[:,lenslet2], axis=0))
    #         cov = A*B
    #         COV[lenslet1,lenslet2] = np.mean(cov)
    
    #         # show A and B
    #         plt.plot(A[:100].flatten(),
    #                   ls='-', marker='.',
    #                   label='WFS1')
    #         plt.plot(B[:100].flatten(),
    #                   ls='-', marker='.',
    #                   label='WFS2')
    #         plt.legend(loc='upper left')
    #         plt.title('covariance between wfs1 and wfs2 {:.3f}'.format(COV[lenslet1,lenslet2]))
    #         plt.show()
    
    #         # this show A*B
    #         plt.plot(cov[:100],
    #                   ls='', marker='o')
    #         plt.grid()
    #         plt.title('mean of this is cov {:.3f}'.format(COV[lenslet1,lenslet2]))
    #         plt.show()
    
    #         # this show population of 'A*B'
    #         # we want to measure the mean.
    #         # error of the mean of the population is not the distribution of the population.
    #         plt.hist(cov, bins=100)
    #         plt.vlines([np.mean(cov),
    #                     np.mean(cov) + np.std(cov, ddof=1)/np.sqrt(TIME),
    #                     np.mean(cov) - np.std(cov, ddof=1)/np.sqrt(TIME)],
    #                     ymin=0, ymax=1000, color='r')
    #         plt.title('mean = {:.3f} +- {:.3f}'.format(np.mean(cov),np.std(cov,ddof=1)/np.sqrt(TIME)))
    #         plt.show()
    
    #         # this show distribution of the measured mean
    #         # the error of measurement of the mean is the standard error of the mean
    #         minicov = np.zeros((TIME),dtype=float)
    #         for repeat in np.arange(TIME):
    #             minicov[repeat] = np.mean(cov[np.random.randint(0,TIME,TIME)])
                
    #         plt.hist(minicov, bins=100)
    #         x = np.linspace(minicov.min(),minicov.max())
    #         plt.plot(x, gauss(x, minicov.mean(), minicov.std(), TIME)/2)
    #         plt.vlines([minicov.mean(),
    #                     minicov.mean() + minicov.std(),
    #                     minicov.mean() - minicov.std()],
    #                     ymin=0, ymax=500, color='r')
    #         plt.title('{:.3f} +- {:.3f}'.format(np.mean(minicov),np.std(minicov, ddof=1)))
    #         plt.show()
    
    
    # cov_x_open[layer] = my_cov(Sx1_open.T, Sx2_open.T)
    # cov_y_open[layer] = my_cov(Sy1_open.T, Sy2_open.T)
    # cov_x_pseudo_open[layer] = my_cov(Sx1_pseudo_open.T, Sx2_pseudo_open.T)
    # cov_y_pseudo_open[layer] = my_cov(Sy1_pseudo_open.T, Sy2_pseudo_open.T)
    
    # map_cov_x_open[layer], emap_cov_x_open[layer] = overlay1(cov_x_open[layer], position)
    # map_cov_y_open[layer], emap_cov_y_open[layer] = overlay1(cov_y_open[layer], position)
    # map_cov_x_pseudo_open[layer], emap_cov_x_pseudo_open[layer] = overlay1(cov_x_pseudo_open[layer], position)
    # map_cov_y_pseudo_open[layer], emap_cov_y_pseudo_open[layer] = overlay1(cov_y_pseudo_open[layer], position)
    
    cov_x_open[layer], ecov_x_open[layer] = my_cov(Sx1_open.T, Sx2_open.T)
    cov_y_open[layer], ecov_y_open[layer] = my_cov(Sy1_open.T, Sy2_open.T)
    # cov_x_pseudo_open[layer], ecov_x_pseudo_open[layer] = my_cov(Sx1_pseudo_open.T, Sx2_pseudo_open.T)
    # cov_y_pseudo_open[layer], ecov_y_pseudo_open[layer] = my_cov(Sy1_pseudo_open.T, Sy2_pseudo_open.T)
    
    map_cov_x_open[layer], emap_cov_x_open[layer] = overlay1(cov_x_open[layer], ecov_x_open[layer], position)
    map_cov_y_open[layer], emap_cov_y_open[layer] = overlay1(cov_y_open[layer], ecov_y_open[layer], position)
    # map_cov_x_pseudo_open[layer], emap_cov_x_pseudo_open[layer] = overlay1(cov_x_pseudo_open[layer], ecov_x_pseudo_open[layer], position)
    # map_cov_y_pseudo_open[layer], emap_cov_y_pseudo_open[layer] = overlay1(cov_y_pseudo_open[layer], ecov_y_pseudo_open[layer], position)
    
###############################################################################
########################################################################

    
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

# name = 'map_cov_x_pseudo_open'
# hdu = fits.PrimaryHDU(map_cov_x_pseudo_open)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()

# name = 'map_cov_y_pseudo_open'
# hdu = fits.PrimaryHDU(map_cov_y_pseudo_open)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()



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

# name = 'emap_cov_x_pseudo_open'
# hdu = fits.PrimaryHDU(emap_cov_x_pseudo_open)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()

# name = 'emap_cov_y_pseudo_open'
# hdu = fits.PrimaryHDU(emap_cov_y_pseudo_open)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()

###############################################################################
########################################################################

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

# name = 'map_cov_x_pseudo_open'
# hdulist = fits.open(name + '.fits')
# hdu = hdulist[0]
# map_cov_x_pseudo_open = hdu.data
# hdulist.close()

# name = 'map_cov_y_pseudo_open'
# hdulist = fits.open(name + '.fits')
# hdu = hdulist[0]
# map_cov_y_pseudo_open = hdu.data
# hdulist.close()



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

# name = 'emap_cov_x_pseudo_open'
# hdulist = fits.open(name + '.fits')
# hdu = hdulist[0]
# emap_cov_x_pseudo_open = hdu.data
# hdulist.close()

# name = 'emap_cov_y_pseudo_open'
# hdulist = fits.open(name + '.fits')
# hdu = hdulist[0]
# emap_cov_y_pseudo_open = hdu.data
# hdulist.close()


name = 'referenceFunctionsVK_6x6'
hdulist = fits.open(name + '.fits')
hdu = hdulist[0]
ref = np.flip(hdu.data, axis=1)
hdulist.close()
ref[ref==0] = np.nan

refx = ref[::2]
refy = ref[1::2]

xaxis = np.arange(map_cov_x_open.shape[1]) - map_cov_x_open.shape[1]/2 + 0.5

###############################################################################
########################################################################

color_list = ['r','g','b','c','m','y','k','tab:purple']

# plt.errorbar(x=xaxis,
#               y=map_cov_x_open[2,:,mid],
#               yerr=emap_cov_x_open[2,:,mid],
#               label='x=T', capsize=5, marker='o', ls='')
# plt.errorbar(x=xaxis,
#               y=map_cov_y_open[2,:,mid],
#               yerr=emap_cov_y_open[2,:,mid],
#               label='y=L', capsize=5, marker='o', ls='')
# plt.plot(xaxis,refx[2,:,mid],label='refx')
# plt.plot(xaxis,refy[2,:,mid],label='refy')
# plt.legend()
# plt.show()

for layer in np.arange(LAYER):
    plt.errorbar(x=xaxis,
                  y=map_cov_x_open[layer,:,mid],
                  yerr=emap_cov_x_open[layer,:,mid],
                  label='open T', capsize=5, marker='o', ls='', color=color_list[layer])
    # plt.errorbar(x=xaxis,
    #               y=map_cov_x_pseudo_open[layer,:,mid],
    #               yerr=emap_cov_x_pseudo_open[layer,:,mid],
    #               label='pseudo-open T', capsize=5, marker='o', ls='', color='b')
    plt.plot(xaxis,refx[layer,:,mid],label='reference T', color='k')
    if layer == 0:
        plt.legend()
plt.title(SAVENAME + '\n' + 'cross section of slope covariance T')
plt.savefig(SAVENAME + '_T.png')
plt.show()

for layer in np.arange(LAYER):
    plt.errorbar(x=xaxis,
                  y=map_cov_x_open[layer,:,mid] - refx[layer,:,mid],
                  yerr=emap_cov_x_open[layer,:,mid],
                  capsize=5, marker='o', label='open T', color=color_list[layer])
    # plt.errorbar(x=xaxis,
    #              y=map_cov_x_pseudo_open[layer,:,mid] - refx[layer,:,mid],
    #              yerr=emap_cov_x_pseudo_open[layer,:,mid],
    #              capsize=5, marker='o', label='pseudo-open T', color='b')
    if layer == 0:
        plt.legend()
        plt.hlines(0,xmin=xaxis.min(),xmax=xaxis.max(), label='reference T', color='k')
plt.title(SAVENAME + '\n' + 'difference between measured and reference T')
plt.savefig(SAVENAME + '_Terror.png')
plt.show()

for layer in np.arange(LAYER):
    plt.errorbar(x=xaxis,
                  y=map_cov_y_open[layer,:,mid],
                  yerr=emap_cov_y_open[layer,:,mid],
                  label='open L', capsize=5, marker='o', ls='', color=color_list[layer])
    # plt.errorbar(x=xaxis,
    #               y=map_cov_y_pseudo_open[layer,:,mid],
    #               yerr=emap_cov_y_pseudo_open[layer,:,mid],
    #               label='pseudo-open L', capsize=5, marker='o', ls='', color='b')
    plt.plot(xaxis,refy[layer,:,mid],label='reference L', color=color_list[layer])
    if layer == 0:
        plt.legend()
plt.title(SAVENAME + '\n' + 'cross section of slope covariance L')
plt.savefig(SAVENAME + '_L.png')
plt.show()

for layer in np.arange(LAYER):
    plt.errorbar(x=xaxis,
                  y=map_cov_y_open[layer,:,mid] - refy[layer,:,mid],
                  yerr=emap_cov_y_open[layer,:,mid],
                  capsize=5, marker='o', label='open L', color=color_list[layer])
    # plt.errorbar(x=xaxis,
    #              y=map_cov_y_pseudo_open[layer,:,mid] - refy[layer,:,mid],
    #              yerr=emap_cov_y_pseudo_open[layer,:,mid],
    #              capsize=5, marker='o', label='pseudo-open L', color='b')
    if layer == 0:
        plt.hlines(0,xmin=xaxis.min(),xmax=xaxis.max(), label='reference', color='k')
        plt.legend()
plt.title(SAVENAME + '\n' + 'difference between measured and reference L')
plt.savefig(SAVENAME + '_L_error.png')
plt.show()




# MAXX = np.nanmax(refx)/10
# MINX = np.nanmin(refx)/10
# MAXY = np.nanmax(refy)/10
# MINY = np.nanmin(refy)/10

# # MAXX = np.min(np.abs([np.nanmax(refx),np.nanmin(refx)]))
# # MINX = -MAXX
# # MAXY = np.min(np.abs([np.nanmax(refy),np.nanmin(refy)]))
# # MINY = -MAXY

# xaxis = np.arange(map_cov_x_open.shape[1]) - map_cov_x_open.shape[1]/2 + 0.5
# # norm=colors.CenteredNorm(vcenter=0,halfrange=MAXX,clip=False)
# # norm=colors.TwoSlopeNorm(vmin=MINX,vcenter=0,vmax=MAXX)

# for layer in np.arange(LAYER):
#     # plt.pcolormesh(xaxis,xaxis,
#     #            map_cov_x_open[layer],
#     #            cmap='RdBu_r',
#     #            norm=colors.TwoSlopeNorm(vmin=MINX,vcenter=0,vmax=MAXX))
#     # plt.colorbar()
#     # plt.axis('square')
#     # plt.title('measured slope covariance map; open T')
#     # plt.show()
#     # plt.pcolormesh(xaxis,xaxis,
#     #            refx[layer],
#     #            cmap='RdBu_r',
#     #            norm=colors.TwoSlopeNorm(vmin=MINX,vcenter=0,vmax=MAXX))
#     # plt.colorbar()
#     # plt.axis('square')
#     # plt.title('reference slope covariance map; open T')
#     # plt.show()
#     plt.pcolormesh(xaxis,xaxis,
#                map_cov_x_open[layer] - refx[layer],
#                cmap='RdBu_r',
#                norm=colors.TwoSlopeNorm(vmin=MINX,vcenter=0,vmax=MAXX))
#     plt.colorbar()
#     plt.axis('square')
#     plt.title('difference between reference and measured\n'
#               + 'slope covariance map; open T')
#     plt.show()
#     plt.pcolormesh(xaxis,xaxis,
#                map_cov_x_open[layer] - refx[layer],
#                cmap='RdBu_r',
#                norm=colors.TwoSlopeNorm(vmin=MINX,vcenter=0,vmax=MAXX))
#     plt.colorbar()
#     plt.axis('square')
#     plt.title('difference between reference and measured\n'
#               + 'slope covariance map; pseudo open T')
#     plt.show()
    
#     # plt.pcolormesh(xaxis,xaxis,
#     #            map_cov_y_open[layer],
#     #            cmap='RdBu_r',
#     #            norm=colors.TwoSlopeNorm(vmin=MINY,vcenter=0,vmax=MAXY))
#     # plt.colorbar()
#     # plt.axis('square')
#     # plt.title('measured slope covariance map; open L')
#     # plt.show()
#     # plt.pcolormesh(xaxis,xaxis,
#     #            refy[layer],
#     #            cmap='RdBu_r',
#     #            norm=colors.TwoSlopeNorm(vmin=MINY,vcenter=0,vmax=MAXY))
#     # plt.colorbar(extend='max')
#     # plt.axis('square')
#     # plt.title('reference slope covariance map; open L')
#     # plt.show()
#     plt.pcolormesh(xaxis,xaxis,
#                map_cov_y_open[layer] - refy[layer],
#                cmap='RdBu_r',
#                norm=colors.TwoSlopeNorm(vmin=MINY,vcenter=0,vmax=MAXY))
#     plt.colorbar()
#     plt.axis('square')
#     plt.title('difference between reference and measured\n'
#               + 'slope covariance map; open L')
#     plt.show()
#     plt.pcolormesh(xaxis,xaxis,
#                map_cov_y_open[layer] - refy[layer],
#                cmap='RdBu_r',
#                norm=colors.TwoSlopeNorm(vmin=MINY,vcenter=0,vmax=MAXY))
#     plt.colorbar()
#     plt.axis('square')
#     plt.title('difference between reference and measured\n'
#               + 'slope covariance map; pseudo open L')
#     plt.show()


## DISTRIBUTION = np.random.randint(LAYER)
## DISTRIBUTION /= DISTRIBUTION.sum()
## R0 = (np.random.rand()*(11-2) + 2)/100
## TOTALCN2 = r02cn2(R0)
## CN2 = TOTALCN2*DISTRIBUTION
## COVCOEFF = CN2/r02cn2(r0)
## COVX = np.matmul(COVCOEFF,crossrefx)
## COVY = np.matmul(COVCOEFF,crossrefy)