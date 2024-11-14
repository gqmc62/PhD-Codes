# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:38:50 2023

study photon noise

show that
0 show detector
1 we get the photon we want
X 1.5 we get the photon noise we want
      ---> not doing it because it also depends on the sampling and distribution of photons within the lenslet
2 we get the WFS measurement noise that we want from Thomas 2006


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
from scipy import stats

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

def expected_photon_noise_approx(sim):
    """
    calculate expected error of WFS slope OPD in rad caused by photon.
    
    This is Thomas 2006 eqn 13. It suppose to be an approximation
    and not as much as accurate as equation 11.
    
    However, this predict better photon related noise in SHWFS.

    Parameters
    ----------
    sim : TYPE
        DESCRIPTION.

    Returns
    -------
    noise : TYPE
        DESCRIPTION.

    """
    N_ph = soapy.wfs.shackhartmann.photons_per_mag(
        sim.config.wfss[0].GSMag, sim.wfss[0].mask, sim.wfss[0].phase_scale,
        sim.config.wfss[0].exposureTime, sim.config.wfss[0].photometric_zp)/sim.wfss[0].n_subaps
    wvl = sim.config.wfss[0].wavelength
    diam = sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps
    noise = np.sqrt(2.*sim.config.wfss[0].subapFOV*ARCSEC2RAD/(wvl/diam)/N_ph)
    return noise

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
        sim.config.wfss[0].exposureTime, sim.config.wfss[0].photometric_zp)/sim.wfss[0].n_subaps
    N_s = sim.config.wfss[0].pxlsPerSubap
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

file = 'allWFSNoise.yaml'
sim = soapy.Sim(file)

sim.aoinit()
# sim.config.calcParams()
# sim.makeIMat()
# sim.config.calcParams()
N = sim.wfss[0].n_subaps
sim.config.sim.nIters = 1000
TIME = sim.config.sim.nIters

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

sim.config.atmos.r0 = 0.05
SAVENAME = 'ALLWFSNOISE_{:d}Frames_allThreshold'.format(TIME)


D = sim.config.tel.telDiam
L0 = sim.config.atmos.L0[0]
l0 = 1e-10
r0 = sim.config.atmos.r0

NPXLSCRN = sim.config.sim.pupilSize
ovf = sim.config.sim.simOversize
initial_flush = int(np.ceil(L0/D/ovf))

readout_list = np.array([0.25,0.5,1,2,4,8], dtype=float)
GSMag_list = np.array([-3,-1,1,3,5,7], dtype=float)
threshold_list = np.array([0,1e-4,1e-3,1e-2,1e-1,0.3], dtype=float)

WFSPX2NM = (sim.config.wfss[0].subapFOV/sim.config.wfss[0].pxlsPerSubap)*ARCSEC2RAD*(sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)/1e-9
WFSPX2RAD = WFSPX2NM*1e-9 * (2.*np.pi/sim.config.wfss[0].wavelength)
MAXSLOPE = (sim.wfss[0].config.pxlsPerSubap/2. - 1.) * WFSPX2RAD



# WFS_open = np.zeros((GSMag_list.shape[0],readout_list.shape[0],threshold_list.shape[0],TIME, 2*N), dtype=float)
# WFS_noNoise = np.zeros((GSMag_list.shape[0],readout_list.shape[0],threshold_list.shape[0],TIME, 2*N), dtype=float)
# # WFS_noNoisenoThreshold = np.zeros((GSMag_list.shape[0],readout_list.shape[0],threshold_list.shape[0],TIME, 2*N), dtype=float)
# WFS_RO = np.zeros((GSMag_list.shape[0],readout_list.shape[0],threshold_list.shape[0],TIME, 2*N), dtype=float)
# WFS_PN = np.zeros((GSMag_list.shape[0],readout_list.shape[0],threshold_list.shape[0],TIME, 2*N), dtype=float)

# measured_photon_in_lenslet = np.zeros((readout_list.shape[0],TIME,sim.config.wfss[0].nxSubaps**2))
# measured_readout = np.zeros((GSMag_list.shape[0],readout_list.shape[0],threshold_list.shape[0]),dtype=float)
# measured_photon = np.zeros((GSMag_list.shape[0],readout_list.shape[0],threshold_list.shape[0]),dtype=float)
# measured_WFS_error = np.zeros((GSMag_list.shape[0],readout_list.shape[0],threshold_list.shape[0]),dtype=float)
# measured_PN_error = np.zeros((GSMag_list.shape[0],readout_list.shape[0],threshold_list.shape[0]),dtype=float)
# measured_RO_error = np.zeros((GSMag_list.shape[0],readout_list.shape[0],threshold_list.shape[0]),dtype=float)
# expected_WFS_error = np.zeros((GSMag_list.shape[0],readout_list.shape[0],threshold_list.shape[0]),dtype=float)

# RANDOM_SEED_list = np.random.randint(0, 2**31 - 1, size=TIME)

# BADSLOPE = -(sim.wfss[0].staticData.min() + sim.wfss[0].config.pxlsPerSubap/2.) * WFSPX2RAD



        
# sim.config.calcParams()

# sim.aoinit()
# sim.config.calcParams()
# sim.makeIMat()
# sim.atmos = soapy.atmosphere.atmos(sim.config)
# sim.allDmCommands = np.zeros((sim.config.sim.nIters, sim.config.sim.totalActs))

# for i in np.arange(len(sim.dms)):
#     sim.dms[i].gain = 0.
#     sim.recon.dms[i].gain = 0.
#     sim.dms[i].dmConfig.gain = 0.


# for iFrame in np.arange(TIME):
#     t = time.time()
#     sim.atmos.infinite_phase_screens[0].random_seed = RANDOM_SEED_list[iFrame]
#     sim.atmos.infinite_phase_screens[0].make_initial_screen()
#     for row in np.arange(sim.atmos.infinite_phase_screens[0].scrn.shape[0]*initial_flush):
#         sim.atmos.infinite_phase_screens[0].add_row()
#     sim.scrns = sim.atmos.moveScrns()
    
#     # if ((iFrame == 0) or (iFrame == (TIME - 1))):
#     #     plt.imshow(sim.scrns[0])
#     #     plt.show()
    
#     sim.Tatmos += time.time() - t

#     # Run Loop...
#     ########################################

#     # Get dmCommands from reconstructor
#     t_recon = time.time()
#     if sim.config.sim.nDM:
#         sim.dmCommands[:] = sim.recon.reconstruct(sim.slopes)
#     sim.Trecon += (time.time() - t_recon)

#     # Delay the dmCommands if loopDelay is configured
#     sim.dmCommands = sim.buffer.delay(sim.dmCommands, sim.config.sim.loopDelay)

#     # Get dmShape from closed loop DMs
#     sim.closed_correction = sim.runDM(
#             sim.dmCommands, closed=True)

#     # Run WFS, with closed loop DM shape applied
    
#     for GSMag_index in np.arange(GSMag_list.shape[0]):
#         for readout_index in np.arange(readout_list.shape[0]):
#             for threshold_index in np.arange(threshold_list.shape[0]):
#                 threshold = threshold_list[threshold_index]
#                 sim.config.wfss[0].centThreshold = threshold
#                 GSMag = GSMag_list[GSMag_index]
#                 sim.config.wfss[0].GSMag = GSMag
#                 readout = readout_list[readout_index]
#                 avg_photons = soapy.wfs.shackhartmann.photons_per_mag(
#                     sim.config.wfss[0].GSMag, sim.wfss[0].mask, sim.wfss[0].phase_scale,
#                     sim.config.wfss[0].exposureTime, sim.config.wfss[0].photometric_zp)/sim.wfss[0].n_subaps
#                 TRUE_READOUT = readout*avg_photons/sim.wfss[0].nx_subap_pixels**2
#                 sim.wfss[0].config.eReadNoise = TRUE_READOUT                  
        
#                 # sim.wfss[0].config.photonNoise = False
#                 # sim.wfss[0].config.eReadNoise = 0
#                 # sim.wfss[0].config.centThreshold = 0
#                 # WFS_noNoisenoThreshold[GSMag_index,readout_index,threshold_index,iFrame] = sim.runWfs(dmShape=sim.closed_correction,
#                 #                           loopIter=sim.iters) * WFSPX2RAD
                
#                 sim.wfss[0].config.photonNoise = False
#                 sim.wfss[0].config.eReadNoise = 0
#                 sim.wfss[0].config.centThreshold = threshold
#                 WFS_noNoise[GSMag_index,readout_index,threshold_index,iFrame] = sim.runWfs(dmShape=sim.closed_correction,
#                                           loopIter=sim.iters) * WFSPX2RAD
                
                
            
#                 sim.wfss[0].config.photonNoise = False
#                 sim.wfss[0].config.eReadNoise = TRUE_READOUT
#                 sim.wfss[0].config.centThreshold = threshold
#                 WFS_RO[GSMag_index,readout_index,threshold_index,iFrame] = sim.runWfs(dmShape=sim.closed_correction,
#                                           loopIter=sim.iters) * WFSPX2RAD
                
                
                
            
#                 sim.wfss[0].config.photonNoise = True
#                 sim.wfss[0].config.eReadNoise = 0.
#                 sim.wfss[0].config.centThreshold = threshold
#                 WFS_PN[GSMag_index,readout_index,threshold_index,iFrame] = sim.runWfs(dmShape=sim.closed_correction,
#                                           loopIter=sim.iters) * WFSPX2RAD
                
                
#                 sim.wfss[0].config.photonNoise = True
#                 sim.wfss[0].config.eReadNoise = TRUE_READOUT
#                 sim.wfss[0].config.centThreshold = threshold
#                 sim.slopes = sim.runWfs(dmShape=sim.closed_correction,
#                                           loopIter=sim.iters)
#                 WFS_open[GSMag_index,readout_index,threshold_index,iFrame] = sim.slopes * WFSPX2RAD
    
#     # Get DM shape for open loop DMs
#     sim.open_correction = sim.runDM(sim.dmCommands,
#                                       closed=False)

#     # Pass whole combined DM shapes to science target
#     sim.combinedCorrection = sim.open_correction + sim.closed_correction

#     sim.runSciCams(sim.combinedCorrection)

#     # Save Data
#     i = sim.iters % sim.config.sim.nIters # If sim is run continuously in loop, overwrite oldest data in buffer
#     sim.storeData(i)

#     sim.printOutput(sim.iters, strehl=True)

#     sim.addToGuiQueue()

#     sim.iters += 1
    
#     # IMAGE = sim.wfss[0].wfsDetectorPlane
#     # LENSLET = rebin(IMAGE,(sim.wfss[0].nx_subaps,sim.wfss[0].nx_subaps))*(sim.wfss[0].nx_subap_pixels**2)
#     # # plt.imshow(LENSLET)
#     # # plt.colorbar()
#     # # plt.show()
#     # NOLIGHT = np.where(LENSLET.flatten()==0)
    
#     # WFS_noNoise[GSMag_index,readout_index,iFrame][NOLIGHT] = np.nan
#     # WFS_RO[GSMag_index,readout_index,iFrame][NOLIGHT] = np.nan
#     # WFS_PN[GSMag_index,readout_index,iFrame][NOLIGHT] = np.nan
#     # WFS_open[GSMag_index,readout_index,iFrame][NOLIGHT] = np.nan

# sim.saveData()
# sim.finishUp()
    
# WFS_noNoise[WFS_noNoise <= -MAXSLOPE] = np.nan
# # WFS_noNoisenoThreshold[WFS_noNoise <= -MAXSLOPE] = np.nan
# WFS_open[WFS_open <= -MAXSLOPE] = np.nan
# WFS_RO[WFS_RO <= -MAXSLOPE] = np.nan
# WFS_PN[WFS_PN <= -MAXSLOPE] = np.nan

# WFS_noNoise[WFS_noNoise >= MAXSLOPE] = np.nan
# # WFS_noNoisenoThreshold[WFS_noNoise >= MAXSLOPE] = np.nan
# WFS_open[WFS_open >= MAXSLOPE] = np.nan
# WFS_RO[WFS_RO >= MAXSLOPE] = np.nan
# WFS_PN[WFS_PN >= MAXSLOPE] = np.nan


# name = SAVENAME + '_NoNoise'
# hdu = fits.PrimaryHDU(WFS_noNoise)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()

# # name = SAVENAME + '_NoNoiseNoThreshold'
# # hdu = fits.PrimaryHDU(WFS_noNoisenoThreshold)
# # hdul = fits.HDUList([hdu])
# # hdul.writeto(name + '.fits', overwrite=True)
# # hdul.close()

# name = SAVENAME + '_PhotonNoise'
# hdu = fits.PrimaryHDU(WFS_PN)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()

# name = SAVENAME + '_ReadoutNoise'
# hdu = fits.PrimaryHDU(WFS_RO)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()

# name = SAVENAME + '_AllNoise'
# hdu = fits.PrimaryHDU(WFS_open)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()

 ###############################################################################
 ########################################################################


name = SAVENAME + '_NoNoise2'
hdulist = fits.open(name + '.fits')
hdu = hdulist[0]
WFS_noNoise = hdu.data
hdulist.close()

# name = SAVENAME + '_NoNoiseNoThreshold'
# hdulist = fits.open(name + '.fits')
# hdu = hdulist[0]
# WFS_noNoisenoThreshold = hdu.data
# hdulist.close()

name = SAVENAME + '_PhotonNoise2'
hdulist = fits.open(name + '.fits')
hdu = hdulist[0]
WFS_PN = hdu.data
hdulist.close()

name = SAVENAME + '_ReadoutNoise2'
hdulist = fits.open(name + '.fits')
hdu = hdulist[0]
WFS_RO = hdu.data
hdulist.close()

name = SAVENAME + '_AllNoise2'
hdulist = fits.open(name + '.fits')
hdu = hdulist[0]
WFS_open = hdu.data
hdulist.close()

    ###############################################################################
    ########################################################################



measured_WFS_error = np.sqrt(np.nanmean((WFS_open - WFS_noNoise)**2, axis=(-1,-2)))
measured_PN_error = np.sqrt(np.nanmean((WFS_PN - WFS_noNoise)**2, axis=(-1,-2)))
measured_RO_error = np.sqrt(np.nanmean((WFS_RO - WFS_noNoise)**2, axis=(-1,-2)))
expected_WFS_error = np.sqrt(measured_PN_error**2 + measured_RO_error**2)

plt.plot(expected_WFS_error[:-2,:-2].flatten(),
          measured_WFS_error[:-2,:-2].flatten(),
          ls='',marker='x', label='measured')
plt.plot(expected_WFS_error[:-2,:-2].flatten(),
          expected_WFS_error[:-2,:-2].flatten(),
          ls='',marker='o', label='expected')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('expected_WFS_error (rad)')
plt.ylabel('measured_WFS_error (rad)')
plt.xlim(expected_WFS_error.min(),expected_WFS_error.max())
plt.ylim(expected_WFS_error.min(),expected_WFS_error.max())
plt.title(SAVENAME)
plt.legend()
plt.show()


        


# #####################################
# for k in np.arange(WFS_noNoise.shape[2]):
#     for i in np.arange(WFS_noNoise.shape[0]):
#         for j in np.arange(WFS_noNoise.shape[1]):
            
#             error = WFS_noNoise[i,j,k] - WFS_noNoise[i,j,0]
            
#             plt.plot(WFS_noNoise[i,j,0].flatten(),error.flatten(),ls='',marker=',',alpha=0.5)
#             plt.xlim(-sim.config.wfss[0].pxlsPerSubap * WFSPX2RAD / 2.,sim.config.wfss[0].pxlsPerSubap * WFSPX2RAD / 2.)
#             plt.ylim(-sim.config.wfss[0].pxlsPerSubap * WFSPX2RAD / 2.,sim.config.wfss[0].pxlsPerSubap * WFSPX2RAD / 2.)
#             plt.axis('equal')
#             plt.xlabel('no noise no threshold measurement (rad)')
#             plt.ylabel('wfs error (noise - no noise no threshold) (rad)')
#             plt.title(SAVENAME
#                       + '\nWFS slopes threshold error (rad) {:.0f}mag, {:.2f}RO, threshold={:.2e}'.format(GSMag_list[i],readout_list[j],threshold_list[k]))
#             plt.show()
            

# for k in np.arange(1):#WFS_noNoise.shape[2]):
#     for i in np.arange(WFS_noNoise.shape[0]):
#         for j in np.arange(WFS_noNoise.shape[1]):
            
#             error = WFS_PN[i,j,k] - WFS_noNoise[i,j,0]
            
#             plt.plot(WFS_noNoise[i,j,0].flatten(),error.flatten(),ls='',marker=',',alpha=0.5)
#             plt.xlim(-sim.config.wfss[0].pxlsPerSubap * WFSPX2RAD / 2.,sim.config.wfss[0].pxlsPerSubap * WFSPX2RAD / 2.)
#             plt.ylim(-sim.config.wfss[0].pxlsPerSubap * WFSPX2RAD / 2.,sim.config.wfss[0].pxlsPerSubap * WFSPX2RAD / 2.)
#             plt.axis('equal')
#             plt.xlabel('no noise no threshold measurement (rad)')
#             plt.ylabel('wfs error (noise - no noise no threshold) (rad)')
#             plt.title(SAVENAME
#                       + '\nWFS slopes photon error (rad) {:.0f}mag, {:.2f}RO, threshold={:.2e}'.format(GSMag_list[i],readout_list[j],threshold_list[k]))
#             plt.show()
            

            
# for k in np.arange(1):#WFS_noNoise.shape[2]):
#     for j in np.arange(WFS_noNoise.shape[1]):
#         for i in np.arange(WFS_noNoise.shape[0]):
#             error = WFS_RO[i,j,k] - WFS_noNoise[i,j,0]
            
#             plt.plot(WFS_noNoise[i,j,0].flatten(),error.flatten(),ls='',marker=',',alpha=0.5)
#             plt.xlim(-sim.config.wfss[0].pxlsPerSubap * WFSPX2RAD / 2.,sim.config.wfss[0].pxlsPerSubap * WFSPX2RAD / 2.)
#             plt.ylim(-sim.config.wfss[0].pxlsPerSubap * WFSPX2RAD / 2.,sim.config.wfss[0].pxlsPerSubap * WFSPX2RAD / 2.)
#             plt.axis('equal')
#             plt.xlabel('no noise no threshold measurement (rad)')
#             plt.ylabel('wfs error (noise - no noise no threshold) (rad)')
#             plt.title(SAVENAME
#                       + '\nWFS slopes readout error (rad) {:.0f}mag, {:.2f}RO, threshold={:.2e}'.format(GSMag_list[i],readout_list[j],threshold_list[k]))
#             plt.show()

# for k in np.arange(1):#WFS_noNoise.shape[2]):
#     for j in np.arange(WFS_noNoise.shape[1]):
#         for i in np.arange(WFS_noNoise.shape[0]):
#             error = WFS_open[i,j,k] - WFS_noNoise[i,j,0]
            
#             plt.plot(WFS_noNoise[i,j,0].flatten(),error.flatten(),ls='',marker=',',alpha=0.5)
#             plt.xlim(-sim.config.wfss[0].pxlsPerSubap * WFSPX2RAD / 2.,sim.config.wfss[0].pxlsPerSubap * WFSPX2RAD / 2.)
#             plt.ylim(-sim.config.wfss[0].pxlsPerSubap * WFSPX2RAD / 2.,sim.config.wfss[0].pxlsPerSubap * WFSPX2RAD / 2.)
#             plt.axis('equal')
#             plt.xlabel('no noise measurement (rad)')
#             plt.ylabel('wfs error (noise - no noise) (rad)')
#             plt.title(SAVENAME
#                       + '\nWFS slopes all error (rad) {:.0f}mag, {:.2f}RO, threshold={:.2e}'.format(GSMag_list[i],readout_list[j],threshold_list[k]))
#             plt.show()
        
# ##########################################################################

pvalue_list_RO = np.zeros(WFS_noNoise.shape[:3])
fitting_index_list_RO = np.zeros(WFS_noNoise.shape[:3])
pvalue_list_PN = np.zeros(WFS_noNoise.shape[:3])
fitting_index_list_PN = np.zeros(WFS_noNoise.shape[:3])
pvalue_list_all = np.zeros(WFS_noNoise.shape[:3])
fitting_index_list_all = np.zeros(WFS_noNoise.shape[:3])


for k in np.arange(WFS_noNoise.shape[2] - 1) + 1:
    i=0
    j=0
    
    NP = soapy.wfs.shackhartmann.photons_per_mag(
        GSMag_list[i], sim.wfss[0].mask, sim.wfss[0].phase_scale,
        sim.config.wfss[0].exposureTime, sim.config.wfss[0].photometric_zp)/sim.wfss[0].n_subaps
    SNR = NP/np.sqrt(NP)
    
    if SNR >= 5:#10:
    
        error = WFS_noNoise[i,j,k] - WFS_noNoise[i,j,0]
        expected_std = error.std()
        
        x = np.linspace(np.nanmin(error), np.nanmax(error),num=100)
        area = np.sum(np.isnan(error) == False)
        
        counts, bins = np.histogram(error[np.isnan(error)==False].flatten(),bins=100)
        
        y = gauss(x,0,expected_std,area)
        
        fitting_index = np.sum(np.abs(y - counts))/area
        
        temp = error[np.isnan(error) == False].flatten()
        res = stats.kstest(temp,np.random.normal(0,expected_std,area))
        
        # pvalue_list_PN[i,j,k] = res.pvalue
        # fitting_index_list_PN[i,j,k] = fitting_index
        
        bincount = np.nansum(counts!=0)
        percentage = (stats.norm.cdf(1)-stats.norm.cdf(-1))
        expectedtobein = bincount*percentage - 2.*np.sqrt(bincount*percentage)
        valid = ((np.nansum(((counts[counts!=0] - y[counts!=0])**2/counts[counts!=0]) < 1) / bincount) > expectedtobein/bincount)
        
        # if j == 0:
        plt.plot(x,y,label='noise equivalent angle model',color='k')
        plt.stairs(counts,bins,alpha=0.5,color='k',fill=True)
        plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
        plt.title(SAVENAME
                  + '\nNo threshold = standard ; WFS slopes threshold error (rad)'
                  + '\n{:.0f}mag, {:.2f}RO, threshold={:.2e}, SNR = {:.0f}'.format(GSMag_list[i],readout_list[j],threshold_list[k],SNR)
                  + '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
                  + '\nvalid = {:}'.format(valid))
        plt.show()

for k in np.arange(1)+5:#WFS_noNoise.shape[2]):
    for i in np.arange(WFS_noNoise.shape[0]):
        for j in np.arange(WFS_noNoise.shape[1]):
            
            NP = soapy.wfs.shackhartmann.photons_per_mag(
                GSMag_list[i], sim.wfss[0].mask, sim.wfss[0].phase_scale,
                sim.config.wfss[0].exposureTime, sim.config.wfss[0].photometric_zp)/sim.wfss[0].n_subaps
            SNR = NP/np.sqrt(NP)
            
            if SNR >= 5:#10:
            
                error = WFS_PN[i,j,k] - WFS_noNoise[i,j,0]
                expected_std = np.nanstd(error)
                
                x = np.linspace(np.nanmin(error), np.nanmax(error),num=100)
                area = np.sum(np.isnan(error) == False)
                
                counts, bins = np.histogram(error[np.isnan(error)==False].flatten(),bins=100)
                
                y = gauss(x,0,expected_std,area)
                
                fitting_index = np.sum(np.abs(y - counts))/area
                
                temp = error[np.isnan(error) == False].flatten()
                res = stats.kstest(temp,np.random.normal(0,expected_std,area))
                
                pvalue_list_PN[i,j,k] = res.pvalue
                fitting_index_list_PN[i,j,k] = fitting_index
                
                bincount = np.nansum(counts!=0)
                percentage = (stats.norm.cdf(1)-stats.norm.cdf(-1))
                expectedtobein = bincount*percentage - 2.*np.sqrt(bincount*percentage)
                valid = ((np.nansum(((counts[counts!=0] - y[counts!=0])**2/counts[counts!=0]) < 1) / bincount) > expectedtobein/bincount)
                
                # if j == 0:
                plt.plot(x,y,label='noise equivalent angle model',color='r')
                plt.stairs(counts,bins,alpha=0.5,color='r',fill=True)
                plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
                plt.title(SAVENAME
                          + '\nNo threshold = Standard ; WFS slopes photon error (rad)'
                          + '\n{:.0f}mag, {:.2f}RO, threshold={:.2e}, SNR = {:.0f}'.format(GSMag_list[i],readout_list[j],threshold_list[k],SNR)
                          + '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
                          + '\nvalid = {:}'.format(valid))
                plt.show()
                
for k in np.arange(1)+5:#WFS_noNoise.shape[2]):
    for j in np.arange(WFS_noNoise.shape[1]):
        for i in np.arange(WFS_noNoise.shape[0]):
            NP = soapy.wfs.shackhartmann.photons_per_mag(
                GSMag_list[i], sim.wfss[0].mask, sim.wfss[0].phase_scale,
                sim.config.wfss[0].exposureTime, sim.config.wfss[0].photometric_zp)/sim.wfss[0].n_subaps
            NRO = readout_list[j]*NP/sim.wfss[0].nx_subap_pixels**2
            SNR = NP/np.sqrt(NRO**2*sim.wfss[0].nx_subap_pixels**2)
            
            if SNR >= 5:#10:
            
                error = WFS_RO[i,j,k] - WFS_noNoise[i,j,0]
                expected_std = np.nanstd(error)
                
                x = np.linspace(np.nanmin(error), np.nanmax(error),num=100)
                area = np.sum(np.isnan(error) == False)
                
                counts, bins = np.histogram(error[np.isnan(error)==False].flatten(),bins=100)
                
                y = gauss(x,0,expected_std,area)
                
                fitting_index = np.sum(np.abs(y - counts))/area
                
                temp = error[np.isnan(error) == False].flatten()
                res = stats.kstest(temp,np.random.normal(0,expected_std,area))
                
                error[np.isnan(error) == False], 
                
                bincount = np.nansum(counts!=0)
                percentage = (stats.norm.cdf(1)-stats.norm.cdf(-1))
                expectedtobein = bincount*percentage - 2.*np.sqrt(bincount*percentage)
                valid = ((np.nansum(((counts[counts!=0] - y[counts!=0])**2/counts[counts!=0]) < 1) / bincount) > expectedtobein/bincount)
                
                pvalue_list_RO[i,j,k] = res.pvalue
                fitting_index_list_RO[i,j,k] = fitting_index
                
                
                
                # if i == 0:
                plt.stairs(counts,bins,alpha=0.5,color='b',fill=True)
                plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
                plt.plot(x,y,label='noise equivalent angle model',color='b')
                plt.title(SAVENAME
                          + '\nNo threshold = standard ; WFS slopes readout error (rad)'
                          + '\n{:.0f}mag, {:.2f}RO, threshold={:.2e}, SNR = {:.0f}'.format(GSMag_list[i],readout_list[j],threshold_list[k],SNR)
                          + '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
                          + '\nvalid = {:}'.format(valid))
                plt.show()
        
for k in np.arange(1)+5:#WFS_noNoise.shape[2]):
    for i in np.arange(WFS_noNoise.shape[0]):
        for j in np.arange(WFS_noNoise.shape[1]):
            
            NP = soapy.wfs.shackhartmann.photons_per_mag(
                GSMag_list[i], sim.wfss[0].mask, sim.wfss[0].phase_scale,
                sim.config.wfss[0].exposureTime, sim.config.wfss[0].photometric_zp)/sim.wfss[0].n_subaps
            NRO = readout_list[j]*NP/sim.wfss[0].nx_subap_pixels**2
            SNR = NP/np.sqrt(NP + NRO**2*sim.wfss[0].nx_subap_pixels**2)
            
            if SNR >= 5:#10:
            
                error = WFS_open[i,j,k] - WFS_noNoise[i,j,0]
                expected_std = np.nanstd(error)
                
                x = np.linspace(np.nanmin(error), np.nanmax(error),num=100)
                area = np.sum(np.isnan(error) == False)
                
                counts, bins = np.histogram(error[np.isnan(error)==False].flatten(),bins=100)
                
                y = gauss(x,0,expected_std,area)
                
                fitting_index = np.sum(np.abs(y - counts))/area
                
                temp = error[np.isnan(error) == False].flatten()
                res = stats.kstest(temp,np.random.normal(0,expected_std,area))
                bincount = np.nansum(counts!=0)
                percentage = (stats.norm.cdf(1)-stats.norm.cdf(-1))
                expectedtobein = bincount*percentage - 2.*np.sqrt(bincount*percentage)
                valid = ((np.nansum(((counts[counts!=0] - y[counts!=0])**2/counts[counts!=0]) < 1) / bincount) > expectedtobein/bincount)
                pvalue_list_all[i,j,k] = res.pvalue
                fitting_index_list_all[i,j,k] = fitting_index
                
                
                
                plt.stairs(counts,bins,alpha=0.5,color='g',fill=True)
                plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
                plt.plot(x,y,label='noise equivalent angle model',color='g')
                plt.title(SAVENAME
                          + '\nNo threshold = standard ; WFS slopes all error (rad)'
                          + '\n{:.0f}mag, {:.2f}RO, threshold={:.2e}, SNR = {:.0f}'.format(GSMag_list[i],readout_list[j],threshold_list[k],SNR)
                          + '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
                          + '\nvalid = {:}'.format(valid))
                plt.show()
        
        
for k in np.arange(1)+5:#WFS_noNoise.shape[2]):
    for i in np.arange(WFS_noNoise.shape[0]):
        for j in np.arange(WFS_noNoise.shape[1]):
            
            NP = soapy.wfs.shackhartmann.photons_per_mag(
                GSMag_list[i], sim.wfss[0].mask, sim.wfss[0].phase_scale,
                sim.config.wfss[0].exposureTime, sim.config.wfss[0].photometric_zp)/sim.wfss[0].n_subaps
            SNR = NP/np.sqrt(NP)
            
            if SNR >= 5:#10:
            
                error = WFS_PN[i,j,k] - WFS_noNoise[i,j,k]
                expected_std = np.nanstd(error)
                
                x = np.linspace(np.nanmin(error), np.nanmax(error),num=100)
                area = np.sum(np.isnan(error) == False)
                
                counts, bins = np.histogram(error[np.isnan(error)==False].flatten(),bins=100)
                
                y = gauss(x,0,expected_std,area)
                
                fitting_index = np.sum(np.abs(y - counts))/area
                
                temp = error[np.isnan(error) == False].flatten()
                res = stats.kstest(temp,np.random.normal(0,expected_std,area))
                
                pvalue_list_PN[i,j,k] = res.pvalue
                fitting_index_list_PN[i,j,k] = fitting_index
                
                bincount = np.nansum(counts!=0)
                percentage = (stats.norm.cdf(1)-stats.norm.cdf(-1))
                expectedtobein = bincount*percentage - 2.*np.sqrt(bincount*percentage)
                valid = ((np.nansum(((counts[counts!=0] - y[counts!=0])**2/counts[counts!=0]) < 1) / bincount) > expectedtobein/bincount)
                
                # if j == 0:
                plt.plot(x,y,label='noise equivalent angle model',color='r')
                plt.stairs(counts,bins,alpha=0.5,color='r',fill=True)
                plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
                plt.title(SAVENAME
                          + '\nWith same threshold = Standard ; WFS slopes photon error (rad)'
                          + '\n{:.0f}mag, {:.2f}RO, threshold={:.2e}, SNR = {:.0f}'.format(GSMag_list[i],readout_list[j],threshold_list[k],SNR)
                          + '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
                          + '\nvalid = {:}'.format(valid))
                plt.show()
                
for k in np.arange(1)+5:#WFS_noNoise.shape[2]):
    for j in np.arange(WFS_noNoise.shape[1]):
        for i in np.arange(WFS_noNoise.shape[0]):
            NP = soapy.wfs.shackhartmann.photons_per_mag(
                GSMag_list[i], sim.wfss[0].mask, sim.wfss[0].phase_scale,
                sim.config.wfss[0].exposureTime, sim.config.wfss[0].photometric_zp)/sim.wfss[0].n_subaps
            NRO = readout_list[j]*NP/sim.wfss[0].nx_subap_pixels**2
            SNR = NP/np.sqrt(NRO**2*sim.wfss[0].nx_subap_pixels**2)
            
            if SNR >= 5:#10:
            
                error = WFS_RO[i,j,k] - WFS_noNoise[i,j,k]
                expected_std = np.nanstd(error)
                
                x = np.linspace(np.nanmin(error), np.nanmax(error),num=100)
                area = np.sum(np.isnan(error) == False)
                
                counts, bins = np.histogram(error[np.isnan(error)==False].flatten(),bins=100)
                
                y = gauss(x,0,expected_std,area)
                
                fitting_index = np.sum(np.abs(y - counts))/area
                
                temp = error[np.isnan(error) == False].flatten()
                res = stats.kstest(temp,np.random.normal(0,expected_std,area))
                
                error[np.isnan(error) == False], 
                
                bincount = np.nansum(counts!=0)
                percentage = (stats.norm.cdf(1)-stats.norm.cdf(-1))
                expectedtobein = bincount*percentage - 2.*np.sqrt(bincount*percentage)
                valid = ((np.nansum(((counts[counts!=0] - y[counts!=0])**2/counts[counts!=0]) < 1) / bincount) > expectedtobein/bincount)
                
                pvalue_list_RO[i,j,k] = res.pvalue
                fitting_index_list_RO[i,j,k] = fitting_index
                
                
                
                # if i == 0:
                plt.stairs(counts,bins,alpha=0.5,color='b',fill=True)
                plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
                plt.plot(x,y,label='noise equivalent angle model',color='b')
                plt.title(SAVENAME
                          + '\nWith same threshold = standard ; WFS slopes readout error (rad)'
                          + '\n{:.0f}mag, {:.2f}RO, threshold={:.2e}, SNR = {:.0f}'.format(GSMag_list[i],readout_list[j],threshold_list[k],SNR)
                          + '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
                          + '\nvalid = {:}'.format(valid))
                plt.show()
        
for k in np.arange(1)+5:#WFS_noNoise.shape[2]):
    for i in np.arange(WFS_noNoise.shape[0]):
        for j in np.arange(WFS_noNoise.shape[1]):
            
            NP = soapy.wfs.shackhartmann.photons_per_mag(
                GSMag_list[i], sim.wfss[0].mask, sim.wfss[0].phase_scale,
                sim.config.wfss[0].exposureTime, sim.config.wfss[0].photometric_zp)/sim.wfss[0].n_subaps
            NRO = readout_list[j]*NP/sim.wfss[0].nx_subap_pixels**2
            SNR = NP/np.sqrt(NP + NRO**2*sim.wfss[0].nx_subap_pixels**2)
            
            if SNR >= 5:#10:
            
                error = WFS_open[i,j,k] - WFS_noNoise[i,j,k]
                expected_std = np.nanstd(error)
                
                x = np.linspace(np.nanmin(error), np.nanmax(error),num=100)
                area = np.sum(np.isnan(error) == False)
                
                counts, bins = np.histogram(error[np.isnan(error)==False].flatten(),bins=100)
                
                y = gauss(x,0,expected_std,area)
                
                fitting_index = np.sum(np.abs(y - counts))/area
                
                temp = error[np.isnan(error) == False].flatten()
                res = stats.kstest(temp,np.random.normal(0,expected_std,area))
                bincount = np.nansum(counts!=0)
                percentage = (stats.norm.cdf(1)-stats.norm.cdf(-1))
                expectedtobein = bincount*percentage - 2.*np.sqrt(bincount*percentage)
                valid = ((np.nansum(((counts[counts!=0] - y[counts!=0])**2/counts[counts!=0]) < 1) / bincount) > expectedtobein/bincount)
                pvalue_list_all[i,j,k] = res.pvalue
                fitting_index_list_all[i,j,k] = fitting_index
                
                
                
                plt.stairs(counts,bins,alpha=0.5,color='g',fill=True)
                plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
                plt.plot(x,y,label='noise equivalent angle model',color='g')
                plt.title(SAVENAME
                          + '\nWith same threshold = standard ; WFS slopes all error (rad)'
                          + '\n{:.0f}mag, {:.2f}RO, threshold={:.2e}, SNR = {:.0f}'.format(GSMag_list[i],readout_list[j],threshold_list[k],SNR)
                          + '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
                          + '\nvalid = {:}'.format(valid))
                plt.show()
        
                
        
