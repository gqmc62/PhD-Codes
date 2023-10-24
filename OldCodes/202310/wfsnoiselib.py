# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 09:16:25 2023

@author: gqmc62

SH-WFS noise library
"""

import numpy as np
import soapy
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import colors

ARCSEC2RAD = 1./3600./360.*2.*np.pi
RAD2ARCSEC = 1./ARCSEC2RAD

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

def rebin1D(arr, new_shape):
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
    shape = (new_shape, arr.size // new_shape)
    return arr.reshape(shape).mean(-1)

def gauss(x, mean, std, area):
    y = area*np.exp(-((x - mean)/std)**2/2.)/(std*np.sqrt(2*np.pi))*(x[1] - x[0])
    return y

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
        sim.wfss[0].config.exposureTime, sim.wfss[0].config.photometric_zp)/(31.6*60)**2*16**2/16**2

    Parameters
    ----------
    sim : soapy.Sim
        DESCRIPTION.

    Returns
    -------
    photon_noise : float
        radian OPD RMS.

    """
    if sim.wfss[0].config.photonNoise == False:
        return 0
    else:
        N_ph = soapy.wfs.shackhartmann.photons_per_mag(
            sim.wfss[0].config.GSMag, sim.wfss[0].mask, sim.wfss[0].phase_scale,
            sim.wfss[0].config.exposureTime, sim.wfss[0].config.photometric_zp)/sim.wfss[0].n_subaps
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
    if sim.wfss[0].config.photonNoise == False:
        return 0
    else:
        N_ph = soapy.wfs.shackhartmann.photons_per_mag(
            sim.wfss[0].config.GSMag, sim.wfss[0].mask, sim.wfss[0].phase_scale,
            sim.wfss[0].config.exposureTime, sim.wfss[0].config.photometric_zp)/sim.wfss[0].n_subaps
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
    if sim.wfss[0].config.eReadNoise == 0:
        return 0
    else:
        threshold = sim.wfss[0].config.centThreshold
        if threshold == 0:
            N_s = sim.config.wfss[0].pxlsPerSubap
            N_r = sim.wfss[0].config.eReadNoise
            N_ph = soapy.wfs.shackhartmann.photons_per_mag(
                sim.wfss[0].config.GSMag, sim.wfss[0].mask, sim.wfss[0].phase_scale,
                sim.wfss[0].config.exposureTime, sim.wfss[0].config.photometric_zp)/sim.wfss[0].n_subaps
            wvl = sim.config.wfss[0].wavelength
            diam = sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps
            pixel_angular_scale = (sim.config.wfss[0].subapFOV/sim.config.wfss[0].pxlsPerSubap * ARCSEC2RAD)
            N_samp = (wvl/diam)/pixel_angular_scale
            readout_noise = np.sqrt((np.pi**2/3.) * (N_r**2/N_ph**2) * (N_s**4/N_samp**2))
        else:
            N_r = sim.wfss[0].config.eReadNoise
            N_ph = soapy.wfs.shackhartmann.photons_per_mag(
                sim.wfss[0].config.GSMag, sim.wfss[0].mask, sim.wfss[0].phase_scale,
                sim.wfss[0].config.exposureTime, sim.wfss[0].config.photometric_zp)/sim.wfss[0].n_subaps
            wvl = sim.config.wfss[0].wavelength
            r0 = sim.config.atmos.r0
            diam = sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps
            if r0 > diam :
                spot_FWHM = wvl/diam
            else:
                spot_FWHM = wvl/r0
            pixel_angular_scale = (sim.config.wfss[0].subapFOV/sim.config.wfss[0].pxlsPerSubap * ARCSEC2RAD)
            sigmaspot = spot_FWHM/(2.*np.sqrt(2.*np.log(2)))/pixel_angular_scale
            coordinate = np.linspace(0, sim.config.wfss[0].pxlsPerSubap, num=64) - sim.config.wfss[0].pxlsPerSubap/2.
            profile = gauss(coordinate,0.,sigmaspot,1)
            profile /= profile.max()
            N_s = (np.sum(profile >= threshold)/2. * sim.config.wfss[0].pxlsPerSubap/coordinate.size)**2*np.pi
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

def calc_SNR(sim):
    NP = np.floor(soapy.wfs.shackhartmann.photons_per_mag(
        sim.wfss[0].config.GSMag, sim.wfss[0].mask, sim.wfss[0].phase_scale,
        sim.wfss[0].config.exposureTime, sim.wfss[0].config.photometric_zp)/sim.wfss[0].n_subaps)
    NRO = sim.wfss[0].config.eReadNoise
    SNR = NP/np.sqrt(NRO**2*sim.wfss[0].nx_subap_pixels**2 + NP)
    return SNR

def calc_SNR_v2(N_photon_in_lenslet,std_readout_in_pixel,n_pixel_across_lenslet,bg=0,gain=1):
    SNR = N_photon_in_lenslet/np.sqrt((std_readout_in_pixel**2/gain**2 + bg**2)*n_pixel_across_lenslet**2 + N_photon_in_lenslet)
    return SNR

def calc_SNR_v21(N_photon_in_lenslet,var_readout_in_lenslet):
    SNR = N_photon_in_lenslet/np.sqrt(var_readout_in_lenslet + N_photon_in_lenslet)
    return SNR

# need another SNR version for thresholding stuff...

def calc_SNR_v3(sim):
    """
    no bg. no gain. consider thresholding level.
    
    truncated noise calculated from
    https://en.wikipedia.org/wiki/Truncated_normal_distribution
    
    example
    
    sim.wfss[0].config.centThreshold = 0.3
    sim.wfss[0].config.GSMag = 5
    readout = 8
    avg_photons = soapy.wfs.shackhartmann.photons_per_mag(
        sim.wfss[0].config.GSMag, sim.wfss[0].mask, sim.wfss[0].phase_scale,
        sim.wfss[0].config.exposureTime, sim.wfss[0].config.photometric_zp)/sim.wfss[0].n_subaps
    TRUEREADOUT = readout * avg_photons / sim.wfss[0].nx_subap_pixels**2
    sim.wfss[0].config.eReadNoise = TRUEREADOUT
    print('Magnitude={:.0f}, readout={:.2f}e-/photon/pixel, threshold={:} of Max'.format(sim.wfss[0].config.GSMag,readout,sim.wfss[0].config.centThreshold))
    print('no threshold version {:.2f}'.format(calc_SNR(sim)))
    print('with threshold version {:.2f}'.format(calc_SNR_v3(sim)))
    
    Parameters
    ----------
    sim : TYPE
        DESCRIPTION.

    Returns
    -------
    SNR : TYPE
        DESCRIPTION.

    """
    threshold = sim.wfss[0].config.centThreshold
    if ((threshold is None) or (np.isnan(threshold) == True)):
        NP = np.floor(soapy.wfs.shackhartmann.photons_per_mag(
            sim.wfss[0].config.GSMag, sim.wfss[0].mask, sim.wfss[0].phase_scale,
            sim.wfss[0].config.exposureTime, sim.wfss[0].config.photometric_zp)/sim.wfss[0].n_subaps)
        signal = NP
        noise = np.sqrt(NP + sim.wfss[0].config.eReadNoise**2*sim.config.wfss[0].pxlsPerSubap**2)
        SNR = signal/noise
        return SNR
    else:
        wvl = sim.config.wfss[0].wavelength
        r0 = sim.config.atmos.r0
        diam = sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps
        if r0 > diam :
            spot_FWHM = wvl/diam
        else:
            spot_FWHM = wvl/r0
        pixel_angular_scale = (sim.config.wfss[0].subapFOV/sim.config.wfss[0].pxlsPerSubap * ARCSEC2RAD)
        sigmaspot = spot_FWHM/(2.*np.sqrt(2.*np.log(2)))/pixel_angular_scale
        coordinate = np.linspace(0, sim.config.wfss[0].pxlsPerSubap, num=64) - sim.config.wfss[0].pxlsPerSubap/2.
        profile = gauss(coordinate,0.,sigmaspot,1)
        profile /= profile.max()
        NP = np.floor(soapy.wfs.shackhartmann.photons_per_mag(
            sim.wfss[0].config.GSMag, sim.wfss[0].mask, sim.wfss[0].phase_scale,
            sim.wfss[0].config.exposureTime, sim.wfss[0].config.photometric_zp)/sim.wfss[0].n_subaps)
        PROFILE = rebin1D(profile,sim.config.wfss[0].pxlsPerSubap)
        PROFILE /= PROFILE.sum()
        signal = np.sum(profile[profile >= threshold]) / profile.sum() * NP
    
        readout_noise = np.sqrt(sim.wfss[0].config.eReadNoise**2*sim.config.wfss[0].pxlsPerSubap**2)
    
        truncated_photon_noise = np.sqrt(signal)
    

        alpha = threshold*NP/readout_noise
        Z = 1 - stats.norm.cdf(alpha)
        factor = np.sqrt(1 - ( -alpha*stats.norm.pdf(alpha))/Z - ((stats.norm.pdf(alpha))/Z)**2)
        
        truncated_readout_noise = readout_noise * np.nan_to_num(factor)
    
        noise = np.sqrt(truncated_readout_noise**2 + truncated_photon_noise**2)
        SNR = signal/noise
    return SNR


inputreadout = np.array([0,4,8])
mag = 5
photon = np.floor(soapy.wfs.shackhartmann.photons_per_mag(
    mag, sim.wfss[0].mask, sim.wfss[0].phase_scale,
    sim.wfss[0].config.exposureTime, sim.wfss[0].config.photometric_zp)/sim.wfss[0].n_subaps)
readoutuse = inputreadout*photon/sim.wfss[0].nx_subap_pixels**2
snr = photon/np.sqrt(photon + readoutuse**2*sim.wfss[0].nx_subap_pixels**2)
print(snr)

photon = np.logspace(1,4)
readout = np.logspace(1,4)
yy,xx = np.meshgrid(photon,readout)
plt.pcolor(np.log10(xx),np.log10(yy),yy/np.sqrt(xx+yy),norm=colors.CenteredNorm(vcenter=5,halfrange=5),cmap='RdBu_r')
plt.colorbar()
plt.ylabel('log Photon (photo e-)')
plt.xlabel('log VarReadout in lenslet ((RO_e-_std_in_pixel)**2 * (total pixel))')
plt.plot(np.log10(((10**np.linspace(1,4))**2-25*(10**np.linspace(1,4)))/25),np.linspace(1,4),ls='-',color='r')
plt.xlim(1,4)
plt.ylim(1,4)
plt.show()