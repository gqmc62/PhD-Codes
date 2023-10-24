# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:30:29 2023

@author: gqmc62

THIS IS THE MOST CORRECT WAY OF DOING SLODAR!!!

This is photon noise analysis
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

def rebin(arr, new_shape):
    """
    Direct copy from the following source.
    
    https://scipython.com/blog/binning-a-2d-array-in-numpy/

    Parameters
    ----------
    arr : TYPE
        DESCRIPTION.
    new_shape : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

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

def expected_noAO_jitter(sim):
    """
    get jitter OPD rms in radian.
    
    This noise supposedly does not exist when close ao loop is running

    Parameters
    ----------
    sim : soapy.Sim

    Returns
    -------
    jitter_phase : float
        radian OPD RMS.

    """
    wvl = sim.config.wfss[0].wavelength
    diam = sim.config.tel.telDiam / sim.config.wfss[0].nxSubaps
    r0 = sim.config.atmos.r0
    jitter_angle = np.sqrt((0.182 * wvl**2 * diam**(-1./3.) * r0**(-5./3.))) # angle of arrival not phase
    jitter_phase = jitter_angle * diam * (2.*np.pi/wvl) # now it is OPD in rad
    return jitter_phase

def expected_dor0_error(sim):
    """
    get expected wfs error due to atmospheric turbulence in rms OPD in radian
    
    always exist. this is due to higher aberration modes exists within a lenslet

    Parameters
    ----------
    sim : soapy.Sim

    Returns
    -------
    dor0_error : float
        radian OPD RMS.

    """
    if test_wfs_well_sampled(sim) == False:
        print('WFS spot is not well sampled. The estimation is not true.')
        return np.nan
    else:
        K = 0.5
    FOV = sim.config.wfss[0].subapFOV*ARCSEC2RAD
    wvl = sim.config.wfss[0].wavelength
    diam = sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps
    r0 = sim.config.atmos.r0
    W = FOV / (wvl/diam)
    dor0_error = np.sqrt(K * W**-2 * (diam/r0)**(5./3.))
    return dor0_error

def test_wfs_well_sampled(sim):
    """
    test if wfs spot is well_sampled

    Parameters
    ----------
    sim : soapy.Sim

    Returns
    -------
    well_sample : bool
        True = well-sample
        False = NO.

    """
    wvl = sim.config.wfss[0].wavelength
    diam = sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps
    pixel_angular_scale = sim.config.wfss[0].subapFOV/sim.config.wfss[0].pxlsPerSubap * ARCSEC2RAD
    N_samp = (wvl/diam) / pixel_angular_scale
    well_sampled = (N_samp > 1.5)
    return well_sampled

def expected_photon_noise(sim):
    """
    calculate expected error of WFS slope OPD in rad caused by photon
    
    347835312.5450372 ~ -10 mag
    sun has -26.7mag in area=(31.6*60")**2
    my FoV = 16" , imagine it be a point source so divide by another 16**2
    
    soapy.wfs.shackhartmann.photons_per_mag(
        -26.74, sim.wfss[0].mask, sim.wfss[0].phase_scale,
        sim.config.wfss[0].exposureTime, sim.config.wfss[0].photometric_zp)/(31.6*60)**2*16**2/16**2

    Parameters
    ----------
    sim : soapy.Sim
        DESCRIPTION.

    Returns
    -------
    photon_noise : float
        radian OPD RMS.

    """
    N_ph = soapy.wfs.shackhartmann.photons_per_mag(
        sim.config.wfss[0].GSMag, sim.wfss[0].mask, sim.wfss[0].phase_scale,
        sim.config.wfss[0].exposureTime, sim.config.wfss[0].photometric_zp)/sim.wfss[0].n_subaps
    if N_ph < 1:
        print('There is not a single photon in WFS!!!')
        return np.nan
    wvl = sim.config.wfss[0].wavelength
    r0 = sim.config.atmos.r0
    diam = sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps
    if r0 > diam :
        spot_FWHM = wvl/diam
    else:
        spot_FWHM = wvl/r0
    pixel_angular_scale = (sim.config.wfss[0].subapFOV/sim.config.wfss[0].pxlsPerSubap * ARCSEC2RAD)
    N_T = spot_FWHM / pixel_angular_scale
    N_samp = (wvl/diam) / pixel_angular_scale
    photon_noise = np.sqrt(np.pi**2/(2.*np.log(2)) * (1./N_ph) * (N_T/N_samp)**2)
    return photon_noise

def expected_readout_noise(sim):
    """
    calculate expected error of WFS slope OPD in rad caused by readout noise

    Parameters
    ----------
    sim : soapy.Sim

    Returns
    -------
    readout_noise : float
        radian OPD RMS.

    """
    N_r = sim.config.wfss[0].eReadNoise
    N_ph = soapy.wfs.shackhartmann.photons_per_mag(
        sim.config.wfss[0].GSMag, sim.wfss[0].mask, sim.wfss[0].phase_scale,
        sim.config.wfss[0].exposureTime, sim.config.wfss[0].photometric_zp)
    N_s = sim.config.wfss[0].pxlsPerSubap**2
    wvl = sim.config.wfss[0].wavelength
    diam = sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps
    pixel_angular_scale = (sim.config.wfss[0].subapFOV/sim.config.wfss[0].pxlsPerSubap * ARCSEC2RAD)
    N_samp = (wvl/diam)/pixel_angular_scale
    readout_noise = np.sqrt((np.pi**2/3.) * (N_r**2/N_ph**2) * (N_s**4/N_samp**2))
    return readout_noise
                            
def expected_all_wfs_spread_open_loop(sim):
    total_noise = np.sqrt(expected_noAO_jitter(sim)**2
                          + expected_dor0_error(sim)**2
                          + expected_photon_noise(sim)**2
                          + expected_readout_noise(sim)**2)
    return total_noise

def expected_all_wfs_noise_close_loop(sim):
    total_noise = np.sqrt(expected_dor0_error(sim)**2
                          + expected_photon_noise(sim)**2
                          + expected_readout_noise(sim)**2)
    return total_noise


###############################################################################
########################################################################

ARCSEC2RAD = 1./3600./360.*2.*np.pi
RAD2ARCSEC = 1./ARCSEC2RAD

home = ('C:/Users/gqmc62/OneDrive - Durham University/Documents/PhD/Codes/conf')
os.chdir(home)

file = 'slope_covariance_most_correct3.yaml'
sim = soapy.Sim(file)

sim.aoinit()
sim.config.calcParams()
sim.makeIMat()
# sim.config.calcParams()
N = sim.wfss[0].n_subaps

scrnHeights_list = (np.arange(sim.config.wfss[0].nxSubaps - 2, dtype=float)
                    * (sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)
                    / ((sim.config.wfss[0].GSPosition[0]
                        - sim.config.wfss[1].GSPosition[0])*ARCSEC2RAD))
LAYER = scrnHeights_list.shape[0]
# LAYER = 1
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
NPXLSCRN = sim.config.sim.pupilSize
ovf = sim.config.sim.simOversize
initial_flush = int(np.ceil(L0/D/ovf))
Zernike = zernike.zernikeArray(n_modes, NPXLSCRN)
GramSchmidt = mps.gram_schmidt_process(Zernike)
PUPIL = Zernike[0]
FlatZernike = Zernike.reshape((n_modes, NPXLSCRN*NPXLSCRN))
FlatGramSchmidt = GramSchmidt.reshape((n_modes, NPXLSCRN*NPXLSCRN))
GramSchmidtCoeff2ZernikeCoeff = mps.get_GramSchmidtCoeff2ZernikeCoeff(FlatZernike, FlatGramSchmidt)

magnitude_list = np.array([-10,5,7], dtype=float)
#  7 = 1.24 photon = 2.3 rad
# 6   3.1             1.5
#   5  7.8          0.95


WFSPX2NM = (sim.config.wfss[0].subapFOV/sim.config.wfss[0].pxlsPerSubap)*ARCSEC2RAD*(sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)/1e-9
WFSPX2RAD = WFSPX2NM*1e-9 * (2.*np.pi/sim.config.wfss[0].wavelength)



for magnitude_index in np.arange(magnitude_list.shape[0]):
    magnitude = magnitude_list[magnitude_index]
    sim.config.wfss[0].GSMag = magnitude
    sim.config.wfss[1].GSMag = magnitude
    photon_noise = expected_photon_noise(sim)
    avg_photons = soapy.wfs.shackhartmann.photons_per_mag(
        sim.config.wfss[0].GSMag, sim.wfss[0].mask, sim.wfss[0].phase_scale,
        sim.config.wfss[0].exposureTime, sim.config.wfss[0].photometric_zp)/sim.wfss[0].n_subaps
    # print(photon_noise/WFSPX2RAD)
    
    SAVENAME = 'PLZWRKNoise'+'_{:d}Frames_x{:.2f}Oversize_avg{:.0f}photonsperlenslet'.format(TIME, ovf, avg_photons)
    
    sim.config.calcParams()
    
    # for layer in tqdm(np.arange(LAYER)):
    
    #     # if layer == 0:
    #     #     PSD = np.zeros((TIME, NPXLSCRN, NPXLSCRN), dtype=float)
    #     #     ZernikeCoeff = np.zeros((TIME, n_modes), dtype=float)
    #     #     phaseDifSquareX = np.zeros((TIME, NPXLSCRN, NPXLSCRN), dtype=float)
    #     #     phaseDifSquareY = np.zeros((TIME, NPXLSCRN, NPXLSCRN), dtype=float)
        
    #     print('at layer {:d}'.format(layer))
    #     scrnHeights = scrnHeights_list[layer]
    #     sim.config.atmos.scrnHeights = [scrnHeights]
    #     sim.aoinit()
    #     sim.config.calcParams()
    #     sim.makeIMat()
    #     sim.atmos = soapy.atmosphere.atmos(sim.config)
    #     sim.allDmCommands = np.zeros((sim.config.sim.nIters, sim.config.sim.totalActs))
        
    #     for i in np.arange(len(sim.dms)):
    #         sim.dms[i].gain = 0.
    #         sim.recon.dms[i].gain = 0.
    #         sim.dms[i].dmConfig.gain = 0.
    #     sim.atmos.infinite_phase_screens[0].random_seed = RANDOM_SEED_LIST[layer,0]
    #     sim.atmos.infinite_phase_screens[0].make_initial_screen()
    #     for row in np.arange(sim.atmos.infinite_phase_screens[0].scrn.shape[0]*initial_flush):
    #         sim.atmos.infinite_phase_screens[0].add_row()
        
    #     # sim.aoloop()
    #     for iFrame in np.arange(TIME):
    #         t = time.time()
    #         sim.scrns = sim.atmos.moveScrns()
            
    #         # if layer == 0:
    #         #     tempa = (sim.config.sim.simSize - sim.config.sim.pupilSize)//2
    #         #     tempb = (sim.config.sim.simSize + sim.config.sim.pupilSize)//2
    #         #     phase = (sim.scrns[0])[tempa:tempb,tempa:tempb] * (2. * np.pi / (sim.wfss[0].wavelength/1e-9))
    #         #     phaseDifSquareX[iFrame] = mps.get_phaseDifSquareX(phase)
    #         #     phaseDifSquareY[iFrame] = mps.get_phaseDifSquareY(phase)
    #         #     PSD[iFrame] = mps.get_PSD(phase, pixel_scale)
    #         #     ZernikeCoeff[iFrame] = mps.get_ZernikeCoeff((phase*PUPIL).flatten(), FlatGramSchmidt, GramSchmidtCoeff2ZernikeCoeff)
            
    #         sim.Tatmos += time.time() - t
    
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
    
    #         if ((layer == 0) and (iFrame == 0)):
    #             IMAGE = sim.wfss[0].wfsDetectorPlane
    #             LENSLET = rebin(IMAGE,(sim.wfss[0].nx_subaps,sim.wfss[0].nx_subaps))*(sim.wfss[0].nx_subap_pixels**2)
    #             plt.hist(LENSLET[LENSLET!=0].flatten())
    #             plt.title('avg photons per lenslet = {:.0f}'.format(LENSLET[LENSLET!=0].mean()))
    #             plt.show()
                
                
    
    #     sim.saveData()
    #     sim.finishUp()
        
    #     WFS_open[layer] = sim.allSlopes
        
    #     if layer == 0:
    #         plt.title(SAVENAME)
    #         plt.imshow(sim.wfss[0].detector)
    #         plt.colorbar()
    #         plt.savefig(SAVENAME + '_WFSDetector{:.0f}TotalPhotons'.format(avg_photons) + '.png')
    #         plt.show()
        
    #     # if layer == 0:
        
    #     #     structure_function_x = np.mean(phaseDifSquareX, axis=(0,2))
    #     #     error_structure_function_x = np.std(phaseDifSquareX, axis=(0,2), ddof=1)/np.sqrt(TIME)
    #     #     structure_function_y = np.mean(phaseDifSquareY, axis=(0,1))
    #     #     error_structure_function_y = np.std(phaseDifSquareY, axis=(0,1), ddof=1)/np.sqrt(TIME)
    #     #     theoretical_structure_function = mps.vonKarmanStructureFunction(r0, NPXLSCRN, pixel_scale, L0)
    #     #     radius = np.arange(NPXLSCRN)*pixel_scale
    #     #     plt.plot(radius, theoretical_structure_function,
    #     #               label='von Karman Structure Function', ls='-', marker='')
    #     #     plt.errorbar(radius, structure_function_x,
    #     #                   yerr=error_structure_function_x,
    #     #                   ls='',marker='o', capsize=5,label='measured X-axis',
    #     #                   alpha=ALPHA1)
    #     #     plt.errorbar(radius, structure_function_y,
    #     #                   yerr=error_structure_function_y,
    #     #                   ls='',marker='o', capsize=5,label='measured Y-axis',
    #     #                   alpha=ALPHA1)
    #     #     plt.errorbar(radius, np.abs(structure_function_x - theoretical_structure_function),
    #     #                   yerr=error_structure_function_x,
    #     #                   ls='',marker='o', capsize=5,label='abs difference X-axis',
    #     #                   alpha=ALPHA1)
    #     #     plt.errorbar(radius, np.abs(structure_function_y - theoretical_structure_function),
    #     #                   yerr=error_structure_function_y,
    #     #                   ls='',marker='o', capsize=5,label='abs difference Y-axis',
    #     #                   alpha=ALPHA1)
    #     #     plt.xscale('log')
    #     #     plt.yscale('log')
    #     #     plt.legend()
    #     #     plt.xlabel('separation (m)')
    #     #     plt.ylabel('structure function (radian^2)')
    #     #     plt.xlim(radius.min(),radius.max()*2)
    #     #     plt.ylim(np.nanmin(theoretical_structure_function)/10,
    #     #               np.nanmax(theoretical_structure_function)*10)
    #     #     plt.title(SAVENAME + '\n' + '{:d}'.format(TIME)
    #     #             + 'structure_function'
    #     #             + 'D{:.0f}cm'.format(D*100)
    #     #             + 'r0{:.0f}cm'.format(r0*100)
    #     #             + 'L0{:.0f}m'.format(L0))
    #     #     if saveFigure == True:
    #     #         plt.savefig(SAVENAME
    #     #                 + '_structure_function'
    #     #                 + '.png')
    #     #     plt.show()
            
    #     #     mean_PSD_x = (np.mean(PSD, axis=0).T)[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
    #     #     error_mean_PSD_x = ((np.std(PSD, axis=0, ddof=1)/np.sqrt(TIME)).T)[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
    #     #     mean_PSD_y = (np.mean(PSD, axis=0))[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
    #     #     error_mean_PSD_y = ((np.std(PSD, axis=0, ddof=1)/np.sqrt(TIME)))[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
    #     #     theoretical_PSD = mps.psd_function(r0, NPXLSCRN, pixel_scale, L0, l0)[int(NPXLSCRN/2),int(NPXLSCRN/2 + 1) : ]
    #     #     f = (np.arange(NPXLSCRN/2-1) + 1)/NPXLSCRN/pixel_scale
    #     #     plt.plot(f, theoretical_PSD,
    #     #               label='von Karman PSD', ls='-', marker='')
    #     #     plt.errorbar(f, mean_PSD_x,
    #     #                   yerr=error_mean_PSD_x,ls='',marker='o', capsize=5,
    #     #                   label='measured X-axis',alpha=ALPHA2)
    #     #     plt.errorbar(f, mean_PSD_y,
    #     #                   yerr=error_mean_PSD_y,ls='',marker='o', capsize=5,
    #     #                   label='measured Y-axis',alpha=ALPHA2)
    #     #     plt.errorbar(f, np.abs(mean_PSD_x - theoretical_PSD),
    #     #                   yerr=error_mean_PSD_x,ls='',marker='o', capsize=5,
    #     #                   label='abs difference X-axis',alpha=ALPHA2)
    #     #     plt.errorbar(f, np.abs(mean_PSD_y - theoretical_PSD),
    #     #                   yerr=error_mean_PSD_y,ls='',marker='o', capsize=5,
    #     #                   label='abs difference Y-axis',alpha=ALPHA2)
    #     #     plt.ylim(np.nanmin(theoretical_PSD)/10,np.nanmax(theoretical_PSD)*10)
    #     #     plt.xscale('log')
    #     #     plt.yscale('log')
    #     #     plt.xlabel('spatial frequency (1/m)')
    #     #     plt.ylabel('phase structure function (??radian??)')
    #     #     plt.title(SAVENAME + '\n' + '{:d}'.format(TIME)
    #     #             + 'PSD'
    #     #             + 'D{:.0f}cm'.format(D*100)
    #     #             + 'r0{:.0f}cm'.format(r0*100)
    #     #             + 'L0{:.0f}m'.format(L0))
    #     #     plt.legend()
    #     #     if saveFigure == True:
    #     #         plt.savefig(SAVENAME
    #     #                 + '_PSD'
    #     #                 + '.png')
    #     #     plt.show()
            
    #     #     variance_ZernikeCoeff = np.var(ZernikeCoeff, ddof=1, axis=0)[1:]
    #     #     error_variance_ZernikeCoeff = 2*variance_ZernikeCoeff/np.sqrt(2*NPXLSCRN - 2)
    #     #     n_modes = variance_ZernikeCoeff.shape[0]
    #     #     modes = np.arange(n_modes) + 1
    #     #     theoretical_variance_ZernikeCoeff = mps.noll_variance(n_modes,NPXLSCRN,pixel_scale,r0)
    #     #     e = NPXLSCRN*pixel_scale/L0
    #     #     adjust = (1 - 1.42*e**(1./3.) + 3.70*e**2 - 4.01*e**(7./3.) + 4.21*e**4 - 4.00*e**(13./3.))
    #     #     L0TipTiltVariance = theoretical_variance_ZernikeCoeff[:2] * adjust
            
    #     #     plt.plot(modes, theoretical_variance_ZernikeCoeff, label='Kolmogorov',marker='',ls='-')
    #     #     plt.errorbar(modes, variance_ZernikeCoeff, yerr=error_variance_ZernikeCoeff,
    #     #                   label='measured', marker='o', ls='', capsize=5)
    #     #     plt.plot(modes[:2], L0TipTiltVariance, label='L0 limited tip/tilt variance')
    #     #     theoretical_variance_ZernikeCoeff[:2] = theoretical_variance_ZernikeCoeff[:2] * adjust
    #     #     plt.errorbar(modes, np.abs(variance_ZernikeCoeff - theoretical_variance_ZernikeCoeff),
    #     #                   yerr=error_variance_ZernikeCoeff,
    #     #                   label='abs difference', marker='o', ls='', capsize=5)
    #     #     plt.yscale('log')
    #     #     plt.xlabel('zernike modes')
    #     #     plt.ylabel('variance of zernike coefficient')
    #     #     plt.legend()
    #     #     plt.ylim(np.nanmin(theoretical_variance_ZernikeCoeff)/10, np.nanmax(theoretical_variance_ZernikeCoeff)*10)
    #     #     plt.title(SAVENAME + '\n' + '{:d}'.format(TIME)
    #     #             + 'ZernikeVariances'
    #     #             + 'D{:.0f}cm'.format(D*100)
    #     #             + 'r0{:.0f}cm'.format(r0*100)
    #     #             + 'L0{:.0f}m'.format(L0))
    #     #     if saveFigure == True:
    #     #         plt.savefig(SAVENAME
    #     #                 + '_ZernikeCoeff'
    #     #                 + '.png')
    #     #     plt.show()
    
    # ##############################################################################
    # #######################################################################
    
    # name = SAVENAME + '_WFS_open'
    # hdu = fits.PrimaryHDU(WFS_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto(name + '.fits', overwrite=True)
    # hdul.close()
    
    # ###############################################################################
    # ########################################################################
    
    name = SAVENAME + '_WFS_open'
    hdulist = fits.open(name + '.fits')
    hdu = hdulist[0]
    WFS_open = hdu.data
    hdulist.close()
    
    ###############################################################################
    ########################################################################
    
    cov_x_open = np.zeros((LAYER, N, N), dtype=float)
    cov_y_open = np.zeros((LAYER, N, N), dtype=float)
    
    ecov_x_open = np.zeros((LAYER, N, N), dtype=float)
    ecov_y_open = np.zeros((LAYER, N, N), dtype=float)
    
    map_cov_x_open = np.zeros((LAYER, 2*mid + 1, 2*mid + 1), dtype=float)
    map_cov_y_open = np.zeros((LAYER, 2*mid + 1, 2*mid + 1), dtype=float)
    
    emap_cov_x_open = np.zeros((LAYER, 2*mid + 1, 2*mid + 1), dtype=float)
    emap_cov_y_open = np.zeros((LAYER, 2*mid + 1, 2*mid + 1), dtype=float)
    
    acov_x_open = np.zeros((LAYER, N, N), dtype=float)
    acov_y_open = np.zeros((LAYER, N, N), dtype=float)
    
    eacov_x_open = np.zeros((LAYER, N, N), dtype=float)
    eacov_y_open = np.zeros((LAYER, N, N), dtype=float)
    
    map_acov_x_open = np.zeros((LAYER, 2*mid + 1, 2*mid + 1), dtype=float)
    map_acov_y_open = np.zeros((LAYER, 2*mid + 1, 2*mid + 1), dtype=float)
    
    emap_acov_x_open = np.zeros((LAYER, 2*mid + 1, 2*mid + 1), dtype=float)
    emap_acov_y_open = np.zeros((LAYER, 2*mid + 1, 2*mid + 1), dtype=float)
    
    for layer in np.arange(LAYER):
        # layer= 2
        Sx1_open = WFS_open[layer, : , (0)*N : (1)*N]
        Sy1_open = WFS_open[layer, : , (1)*N : (2)*N]
        Sx2_open = WFS_open[layer, : , (2)*N : (3)*N]
        Sy2_open = WFS_open[layer, : , (3)*N : (4)*N]
        
        Sx1_open = (Sx1_open.T - np.mean(Sx1_open, axis=1)).T
        Sy1_open = (Sy1_open.T - np.mean(Sy1_open, axis=1)).T
        Sx2_open = (Sx2_open.T - np.mean(Sx2_open, axis=1)).T
        Sy2_open = (Sy2_open.T - np.mean(Sy2_open, axis=1)).T
        
        cov_x_open[layer], ecov_x_open[layer] = my_cov(Sx1_open.T, Sx2_open.T)
        cov_y_open[layer], ecov_y_open[layer] = my_cov(Sy1_open.T, Sy2_open.T)
        
        map_cov_x_open[layer], emap_cov_x_open[layer] = overlay1(cov_x_open[layer], ecov_x_open[layer], position)
        map_cov_y_open[layer], emap_cov_y_open[layer] = overlay1(cov_y_open[layer], ecov_y_open[layer], position)
        
        acov_x_open[layer], eacov_x_open[layer] = my_cov(Sx1_open.T, Sx1_open.T)
        acov_y_open[layer], eacov_y_open[layer] = my_cov(Sy1_open.T, Sy1_open.T)
        
        map_acov_x_open[layer], emap_acov_x_open[layer] = overlay1(acov_x_open[layer], eacov_x_open[layer], position)
        map_acov_y_open[layer], emap_acov_y_open[layer] = overlay1(acov_y_open[layer], eacov_y_open[layer], position)
        
    plt.hist(Sx1_open.flatten()*WFSPX2RAD,bins=50)
    plt.title('std = {:.3f}rad, expected = {:.3f}rad'.format(Sx1_open.std()*WFSPX2RAD,expected_noAO_jitter(sim)))
    plt.show()
    
    ###############################################################################
    ########################################################################
    
        
    # name = SAVENAME + '_map_cov_x_open'
    # hdu = fits.PrimaryHDU(map_cov_x_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto(name + '.fits', overwrite=True)
    # hdul.close()
    
    # name = SAVENAME + '_map_cov_y_open'
    # hdu = fits.PrimaryHDU(map_cov_y_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto(name + '.fits', overwrite=True)
    # hdul.close()
    
    # name = SAVENAME + '_emap_cov_x_open'
    # hdu = fits.PrimaryHDU(emap_cov_x_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto(name + '.fits', overwrite=True)
    # hdul.close()
    
    # name = SAVENAME + '_emap_cov_y_open'
    # hdu = fits.PrimaryHDU(emap_cov_y_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto(name + '.fits', overwrite=True)
    # hdul.close()
    
    # name = SAVENAME + '_map_acov_x_open'
    # hdu = fits.PrimaryHDU(map_acov_x_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto(name + '.fits', overwrite=True)
    # hdul.close()
    
    # name = SAVENAME + '_map_acov_y_open'
    # hdu = fits.PrimaryHDU(map_acov_y_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto(name + '.fits', overwrite=True)
    # hdul.close()
    
    # name = SAVENAME + '_emap_acov_x_open'
    # hdu = fits.PrimaryHDU(emap_acov_x_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto(name + '.fits', overwrite=True)
    # hdul.close()
    
    # name = SAVENAME + '_emap_acov_y_open'
    # hdu = fits.PrimaryHDU(emap_acov_y_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto(name + '.fits', overwrite=True)
    # hdul.close()
    
    # ###############################################################################
    # ########################################################################
    
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
    
    name = SAVENAME + '_map_acov_x_open'
    hdulist = fits.open(name + '.fits')
    hdu = hdulist[0]
    map_acov_x_open = hdu.data
    hdulist.close()
    
    name = SAVENAME + '_map_acov_y_open'
    hdulist = fits.open(name + '.fits')
    hdu = hdulist[0]
    map_acov_y_open = hdu.data
    hdulist.close()
    
    name = SAVENAME + '_emap_acov_x_open'
    hdulist = fits.open(name + '.fits')
    hdu = hdulist[0]
    emap_acov_x_open = hdu.data
    hdulist.close()
    
    name = SAVENAME + '_emap_acov_y_open'
    hdulist = fits.open(name + '.fits')
    hdu = hdulist[0]
    emap_acov_y_open = hdu.data
    hdulist.close()
    
    name = 'referenceFunctionsVK_6x6'
    hdulist = fits.open(name + '.fits')
    hdu = hdulist[0]
    ref = np.flip(hdu.data, axis=1)
    hdulist.close()
    ref[ref==0] = np.nan
    
    refx = ref[::2]
    refy = ref[1::2]
    
    xaxis = np.arange(map_cov_x_open.shape[1]) - map_cov_x_open.shape[1]/2 + 0.5
    color_list = ['r','g','b','c','m','y','k','tab:purple']
    ###############################################################################
    ########################################################################
    
    
    fig, axs = plt.subplots(2,3,sharex=True,sharey=True,squeeze=True,figsize=(9,6))
    for layer in np.arange(LAYER):
        row = layer // 3
        col = layer % 3
        axs[row,col].errorbar(x=xaxis,
                      y=map_cov_x_open[layer,:,mid],
                      yerr=emap_cov_x_open[layer,:,mid],
                      label='open T', capsize=5, marker='o', ls='', color=color_list[layer])
        axs[row,col].plot(xaxis,refx[layer,:,mid],label='reference T', color=color_list[layer])
        axs[row,col].set_title('Layer {:d}'.format(layer))
        if (col == 0):
            axs[row,col].set_ylabel('covariance')
        if (row == 1):
            axs[row,col].set_xlabel('lenslet position')
    plt.suptitle(SAVENAME + '\n' + 'cross section of slope covariance T')
    plt.savefig(SAVENAME + '_T.png')
    plt.show()
    
    fig, axs = plt.subplots(2,3,sharex=True,sharey=True,squeeze=True,figsize=(9,6))
    for layer in np.arange(LAYER):
        row = layer // 3
        col = layer % 3
        axs[row,col].errorbar(x=xaxis,
                      y=map_cov_y_open[layer,:,mid],
                      yerr=emap_cov_y_open[layer,:,mid],
                      label='open L', capsize=5, marker='o', ls='', color=color_list[layer])
        axs[row,col].plot(xaxis,refy[layer,:,mid],label='reference L', color=color_list[layer])
        axs[row,col].set_title('Layer {:d}'.format(layer))
        if (col == 0):
            axs[row,col].set_ylabel('covariance')
        if (row == 1):
            axs[row,col].set_xlabel('lenslet position')
    plt.suptitle(SAVENAME + '\n' + 'cross section of slope covariance L')
    plt.savefig(SAVENAME + '_L.png')
    plt.show()
    
    # fig, axs = plt.subplots(2,3,sharex=True,sharey=True,squeeze=True,figsize=(9,6))
    # for layer in np.arange(LAYER):
    #     row = layer // 3
    #     col = layer % 3
    #     axs[row,col].errorbar(x=xaxis,
    #                   y=np.abs(map_cov_x_open[layer,:,mid]),
    #                   yerr=emap_cov_x_open[layer,:,mid],
    #                   label='open T', capsize=5, marker='o', ls='', color=color_list[layer])
    #     axs[row,col].errorbar(x=xaxis,
    #                   y=np.abs(map_cov_x_open[layer,:,mid] - refx[layer,:,mid]),
    #                   yerr=emap_cov_x_open[layer,:,mid],
    #                   label='open T', capsize=5, marker='^', ls=':', color=color_list[layer])
    #     axs[row,col].plot(xaxis,np.abs(refx[layer,:,mid]),label='reference T', color=color_list[layer])
    #     axs[row,col].set_title('Layer {:d}'.format(layer))
    #     axs[row,col].set_yscale('log')
    #     if (col == 0):
    #         axs[row,col].set_ylabel('covariance')
    #     if (row == 1):
    #         axs[row,col].set_xlabel('lenslet position')
    # plt.suptitle(SAVENAME + '\n' + 'cross section of slope covariance T')
    # plt.legend()
    # plt.savefig(SAVENAME + '_T_error.png')
    # plt.show()
    
    # fig, axs = plt.subplots(2,3,sharex=True,sharey=True,squeeze=True,figsize=(9,6))
    # for layer in np.arange(LAYER):
    #     row = layer // 3
    #     col = layer % 3
    #     axs[row,col].errorbar(x=xaxis,
    #                   y=np.abs(map_cov_y_open[layer,:,mid]),
    #                   yerr=emap_cov_y_open[layer,:,mid],
    #                   label='open L', capsize=5, marker='o', ls='', color=color_list[layer])
    #     axs[row,col].errorbar(x=xaxis,
    #                   y=np.abs(map_cov_y_open[layer,:,mid] - refy[layer,:,mid]),
    #                   yerr=emap_cov_y_open[layer,:,mid],
    #                   label='open L', capsize=5, marker='^', ls=':', color=color_list[layer])
    #     axs[row,col].plot(xaxis,np.abs(refy[layer,:,mid]),label='reference L', color=color_list[layer])
    #     axs[row,col].set_title('Layer {:d}'.format(layer))
    #     axs[row,col].set_yscale('log')
    #     if (col == 0):
    #         axs[row,col].set_ylabel('covariance')
    #     if (row == 1):
    #         axs[row,col].set_xlabel('lenslet position')
    # plt.suptitle(SAVENAME + '\n' + 'cross section of slope covariance L')
    # plt.legend()
    # plt.savefig(SAVENAME + '_L_error.png')
    # plt.show()
    
    fig, axs = plt.subplots(2,3,sharex=True,sharey=True,squeeze=True,figsize=(9,6))
    for layer in np.arange(LAYER):
        row = layer // 3
        col = layer % 3
        axs[row,col].errorbar(x=xaxis,
                      y=map_acov_x_open[layer,:,mid],
                      yerr=emap_acov_x_open[layer,:,mid],
                      label='open T', capsize=5, marker='o', ls='', color=color_list[layer])
        axs[row,col].plot(xaxis,refx[0,:,mid],label='reference T', color=color_list[layer])
        axs[row,col].set_title('Layer {:d}'.format(layer))
        if (col == 0):
            axs[row,col].set_ylabel('covariance')
        if (row == 1):
            axs[row,col].set_xlabel('lenslet position')
    plt.suptitle(SAVENAME + '\n' + 'cross section of slope auto-covariance T')
    plt.savefig(SAVENAME + '_aT.png')
    plt.show()
    
    fig, axs = plt.subplots(2,3,sharex=True,sharey=True,squeeze=True,figsize=(9,6))
    for layer in np.arange(LAYER):
        row = layer // 3
        col = layer % 3
        axs[row,col].errorbar(x=xaxis,
                      y=map_acov_y_open[layer,:,mid],
                      yerr=emap_acov_y_open[layer,:,mid],
                      label='open L', capsize=5, marker='o', ls='', color=color_list[layer])
        axs[row,col].plot(xaxis,refy[0,:,mid],label='reference L', color=color_list[layer])
        axs[row,col].set_title('Layer {:d}'.format(layer))
        if (col == 0):
            axs[row,col].set_ylabel('covariance')
        if (row == 1):
            axs[row,col].set_xlabel('lenslet position')
    plt.suptitle(SAVENAME + '\n' + 'cross section of slope auto-covariance L')
    plt.savefig(SAVENAME + '_aL.png')
    plt.show()
    
    
    
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
    
    def lsq_cov2cn2(refcov,refcn2,cov,cov_noise,n_bootstrap=100):
        RESULT = np.zeros((n_bootstrap,refcov.shape[1]),dtype=float)
        for n in np.arange(n_bootstrap):
            noisy_cov = np.random.normal(np.nan_to_num(cov),np.nan_to_num(cov_noise))
            res = lsq_linear(refcov, noisy_cov, bounds=(0,np.inf))
            RESULT[n] = res.x
        cn2 = np.mean(RESULT,axis=0) * refcn2
        cn2_noise = np.std(RESULT,axis=0,ddof=1) *refcn2
        r0 = cn22r0(cn2.sum())
        r0_noise = stdcn22stdr0(cn2.sum(),cn2_noise.sum())
        return cn2, cn2_noise, r0, r0_noise
    
    HOMEDIR = home
    REFFILE = 'referenceFunctionsVK_6x6'
    
    A_withcentre = loadTIMBRefCov_withcentre(HOMEDIR, REFFILE)
    
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
    
    REFR0 = 0.05
    REFTOTALCN2 = r02cn2(REFR0)
    RANDOMTOTALCN2 = REFTOTALCN2
    RANDOMR0 = REFR0
    
    for layer in np.arange(LAYER):
        RANDOMCOEFF = np.zeros((LAYER), dtype=float)
        RANDOMCOEFF[layer] = 1
        RANDOMCN2 = RANDOMCOEFF * RANDOMTOTALCN2
        
        NOISYRANDOMCOV = np.nan_to_num(
            np.concatenate(
                [(map_cov_x_open[layer]).flatten(),
                 (map_cov_y_open[layer]).flatten()]))
        NOISE = np.nan_to_num(
            np.concatenate(
                [(emap_cov_x_open[layer]).flatten(),
                 (emap_cov_y_open[layer]).flatten()]))
        
        
        NOISYRANDOMCOV_nocentre = (NOISYRANDOMCOV[np.where(IS_COV_CENTRE == False)]).reshape((2*SCREEN*(SCREEN - 1)))
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
                                           
        if (((SOLVEDCN2_nocentre < 0).sum() > 0) or ((SOLVEDCN2_withcentre < 0).sum() > 0)):
            print('wrong')                                   
    
        DISTRIBUTION_ERROR_noCentre = (SOLVEDCN2_nocentre/RANDOMTOTALCN2 - RANDOMCN2/RANDOMTOTALCN2)
        DISTRIBUTION_ERROR_withCentre = (SOLVEDCN2_withcentre/RANDOMTOTALCN2 - RANDOMCN2/RANDOMTOTALCN2)
        r0_ERROR_noCentre = np.abs(SOLVEDR0_nocentre - RANDOMR0)
        r0_ERROR_withCentre = np.abs(SOLVEDR0_withcentre - RANDOMR0)
                                           
        fig, axs = plt.subplot_mosaic([['top', 'top'],['left', 'right'],],
                      constrained_layout=True,
                      gridspec_kw={'width_ratios': [1,9],'height_ratios': [1,9]},
                      figsize=(7,7))
        plt.suptitle('Total Cn2={:.2e}m$^-$$^2$$^/$$^3$\n'.format(RANDOMTOTALCN2)
                      + 'Distribution ratio RMS error: NO centre fit = {:.2f}, WITH centre fit = {:.2f}\n'.format((DISTRIBUTION_ERROR_noCentre).std(), (DISTRIBUTION_ERROR_withCentre).std())
                      + 'r0 abs error: NO centre fit = {:.2f}cm, WITH centre fit = {:.2f}cm'.format((r0_ERROR_noCentre)*100,(r0_ERROR_withCentre)*100))
        
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
        axs['right'].set_ylim(0.01*0.9,1*1.1)
        
        axs['left'].errorbar([''],SOLVEDR0_nocentre,yerr=SOLVEDR0NOISE_nocentre,
                        marker='^',capsize=5,color='r')#,label='measured')
        axs['left'].errorbar([''],SOLVEDR0_withcentre,yerr=SOLVEDR0NOISE_withcentre,
                        marker='v',capsize=5,color='b')#,label='measured')
        axs['left'].bar([''],RANDOMR0,color='k',alpha=0.5)#,label='simulated')
        axs['left'].set_ylabel('r0 (m)')
        axs['left'].set_ylim((0.02*0.9,0.11*1.1))
        axs['left'].set_title('Total r0 (m)')
        
        h, l = axs['right'].get_legend_handles_labels()
        axs['top'].legend(h, l, loc='center',ncol=3)
        axs['top'].axis('off')
        
        fig.tight_layout(pad=0.5)
        plt.show()
    
                             
