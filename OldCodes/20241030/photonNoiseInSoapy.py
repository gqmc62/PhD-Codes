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
                          # + expected_photon_noise_approx(sim)**2
                          + expected_photon_noise(sim)**2
                          + expected_readout_noise(sim)**2)
    return total_noise

def expected_all_wfs_noise_close_loop(sim):
    total_noise = np.sqrt(expected_dor0_error(sim)**2
                          # + expected_photon_noise_approx(sim)**2
                          + expected_photon_noise(sim)**2
                          + expected_readout_noise(sim)**2)
    return total_noise


###############################################################################
########################################################################

ARCSEC2RAD = 1./3600./360.*2.*np.pi
RAD2ARCSEC = 1./ARCSEC2RAD

home = ('C:/Users/gqmc62/OneDrive - Durham University/Documents/PhD/Codes/conf')
os.chdir(home)

file = 'photonNoise.yaml'
sim = soapy.Sim(file)

# sim.config.atmos.r0 = 0.3

sim.aoinit()
sim.config.calcParams()
sim.makeIMat()
# sim.config.calcParams()
N = sim.wfss[0].n_subaps

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



D = sim.config.tel.telDiam
L0 = sim.config.atmos.L0[0]
l0 = 1e-10
r0 = sim.config.atmos.r0

NPXLSCRN = sim.config.sim.pupilSize
ovf = sim.config.sim.simOversize
initial_flush = int(np.ceil(L0/D/ovf))

magnitude_list = np.array([   -3,  -1,  1, 3,5,7], dtype=float)
#                          10000,2000,300,50,8,1


WFS_open = np.zeros((TIME, 2*N), dtype=float)
nonoiseslopes = np.zeros((TIME, 2*N), dtype=float)

WFSPX2NM = (sim.config.wfss[0].subapFOV/sim.config.wfss[0].pxlsPerSubap)*ARCSEC2RAD*(sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)/1e-9
WFSPX2RAD = WFSPX2NM*1e-9 * (2.*np.pi/sim.config.wfss[0].wavelength)

measured_photon_in_lenslet = np.zeros((magnitude_list.shape[0],TIME,sim.config.wfss[0].nxSubaps**2))
measured_photons = np.zeros((magnitude_list.shape[0]),dtype=float)
expected_photons = np.zeros((magnitude_list.shape[0]),dtype=float)
measured_WFS_error = np.zeros((magnitude_list.shape[0]),dtype=float)
expected_WFS_error = np.zeros((magnitude_list.shape[0]),dtype=float)

RANDOM_SEED = np.random.randint(0, 2**31 - 1)

for magnitude_index in np.arange(magnitude_list.shape[0]):
    magnitude = magnitude_list[magnitude_index]
    sim.config.wfss[0].GSMag = magnitude
    photon_noise = expected_photon_noise(sim)
    avg_photons = soapy.wfs.shackhartmann.photons_per_mag(
        sim.config.wfss[0].GSMag, sim.wfss[0].mask, sim.wfss[0].phase_scale,
        sim.config.wfss[0].exposureTime, sim.config.wfss[0].photometric_zp)/sim.wfss[0].n_subaps
    # print(photon_noise/WFSPX2RAD)
    
    SAVENAME = 'photonNoise'+'_{:d}Frames_avg{:.0f}photonsInLenslet'.format(TIME, avg_photons)
    
    sim.config.calcParams()
    
    sim.aoinit()
    sim.config.calcParams()
    sim.makeIMat()
    sim.atmos = soapy.atmosphere.atmos(sim.config)
    sim.allDmCommands = np.zeros((sim.config.sim.nIters, sim.config.sim.totalActs))
    
    for i in np.arange(len(sim.dms)):
        sim.dms[i].gain = 0.
        sim.recon.dms[i].gain = 0.
        sim.dms[i].dmConfig.gain = 0.
    sim.atmos.infinite_phase_screens[0].random_seed = RANDOM_SEED
    sim.atmos.infinite_phase_screens[0].make_initial_screen()
    for row in np.arange(sim.atmos.infinite_phase_screens[0].scrn.shape[0]*initial_flush):
        sim.atmos.infinite_phase_screens[0].add_row()
    
    for iFrame in np.arange(TIME):
        t = time.time()
        sim.scrns = sim.atmos.moveScrns()
        
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
        
        IMAGE = sim.wfss[0].wfsDetectorPlane
        LENSLET = rebin(IMAGE,(sim.wfss[0].nx_subaps,sim.wfss[0].nx_subaps))*(sim.wfss[0].nx_subap_pixels**2)
        measured_photon_in_lenslet[magnitude_index,iFrame] = LENSLET.flatten()
        # plt.imshow(LENSLET)
        # plt.colorbar()
        # plt.show()
        NOLIGHT = np.where(LENSLET.flatten()==0)
        
        original_config_eReadNoise = sim.config.wfss[0].eReadNoise
        original_config_mag = sim.config.wfss[0].GSMag
        sim.config.photonNoise = False
        sim.config.wfss[0].GSMag = -10
        sim.config.eReadNoise = 0
        nonoiseslopes[iFrame] = sim.runWfs(dmShape=sim.closed_correction,
                                  loopIter=sim.iters) * WFSPX2RAD
        nonoiseslopes[iFrame][NOLIGHT] = np.nan
        sim.config.wfss[0].photonNoise = True
        sim.config.wfss[0].GSMag = original_config_mag
        sim.config.wfss[0].eReadNoise = original_config_eReadNoise
        
        sim.slopes = sim.runWfs(dmShape=sim.closed_correction,
                                  loopIter=sim.iters)
        WFS_open[iFrame] = sim.slopes * WFSPX2RAD
        WFS_open[iFrame][NOLIGHT] = np.nan
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
        
        
        if iFrame == 0:
            plt.title(SAVENAME)
            plt.imshow(sim.wfss[0].detector)
            plt.colorbar()
            plt.savefig(SAVENAME + '_WFSDetector{:.0f}TotalPhotons'.format(avg_photons) + '.png')
            plt.show()

    sim.saveData()
    sim.finishUp()
    
    avgLENSLET = np.nanmean(measured_photon_in_lenslet[magnitude_index],
                          axis=0).reshape(sim.config.wfss[0].nxSubaps,
                                          sim.config.wfss[0].nxSubaps)
    avgLENSLET[avgLENSLET < (avgLENSLET.max()*0.1)] = np.nan
    
    VALID = (np.isnan(avgLENSLET.flatten()) == False)
    valid_photon_lenslet = measured_photon_in_lenslet[magnitude_index,:,VALID]
    measured_photons[magnitude_index] = np.nanmean(valid_photon_lenslet)
    expected_photons[magnitude_index] = avg_photons

    
    
    # WFS_open[ : , (0)*N : (1)*N] \
    #     = (WFS_open[ : , (0)*N : (1)*N].T
    #        - np.nanmean(WFS_open[ : , (0)*N : (1)*N], axis=1)).T
    # WFS_open[ : , (1)*N : (2)*N] \
    #     = (WFS_open[ : , (1)*N : (2)*N].T
    #        - np.nanmean(WFS_open[ : , (1)*N : (2)*N], axis=1)).T
    # plt.plot(nonoiseslopes.flatten(),WFS_open.flatten(),ls='',marker='.')
    # plt.axis('square')
    # plt.show()
    BADSLOPE = -(sim.wfss[0].staticData.min() + sim.wfss[0].config.pxlsPerSubap/2.)
    GOOD1 = (nonoiseslopes >= BADSLOPE) 
    # bad slope comes from when soapy wfs report nan as 0 which is mapped to this value
    # see soapy.wfs.shackhartmann.ShackHartmann.calculateSlopes
    # 
    GOOD2 = (WFS_open >= BADSLOPE)
    GOOD = (GOOD1 * GOOD2)
    plt.plot(nonoiseslopes[GOOD].flatten(),WFS_open[GOOD].flatten(),ls='',marker='.')
    # plt.plot(nonoiseslopes.flatten(),WFS_open.flatten(),ls='',marker='.')
    plt.xlabel('no noise WFS slopes OPD (rad)')
    plt.ylabel('noisy WFS slopes OPD (rad)')
    plt.axis('equal')
    plt.show()
    measured_WFS_error[magnitude_index] = np.mean((WFS_open[GOOD] - nonoiseslopes[GOOD])**2)**0.5
    # measured_WFS_error[magnitude_index] = np.nanmean((WFS_open - nonoiseslopes)**2)**0.5
    expected_WFS_error[magnitude_index] = expected_all_wfs_noise_close_loop(sim)


###############################################################################
########################################################################

plt.plot(expected_photons,measured_photons,ls='',marker='o')
plt.plot(expected_photons,expected_photons,ls='-',marker='')
plt.title('simulated vs expected number of photons in a lenslet')
plt.xlabel('expected number of photons')
plt.ylabel('simulated number of photons')
# plt.axis('square')
plt.xscale('log')
plt.yscale('log')
plt.show()

plt.plot(expected_WFS_error,measured_WFS_error,ls='',marker='o')
plt.plot(expected_WFS_error,expected_WFS_error,ls='-',marker='')
plt.title('expected WFS error (rad)')
plt.xlabel('expected WFS error (rad)')
plt.ylabel('simulated WFS error (rad)')
plt.xscale('log')
plt.yscale('log')
plt.show()

