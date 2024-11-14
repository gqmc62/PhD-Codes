# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:35:52 2023

@author: gqmc62
"""

import os
import soapy
import numpy as np
from matplotlib import pyplot as plt

def get_PSD1_axis0(phase, delta):
    """
    Calculate PSD of a phase screen.

    Parameters
    ----------
    phase : numpy.ndarray
            (N,N)
            Phase screen to be measured.
    delta : float
            Pixel scale.

    Returns
    -------
    PSD : numpy.ndarray
          (N,N)
          Power Spectral Density.

    Example
    -------
    delta = 5e-3
    r0 = 0.1
    N = 1024
    L0 = 20
    l0 = 1e-10
    f = (np.arange(N/2-1)+1)/N/delta
    phase = ft_phase_screen(r0, N, delta, L0, l0)
    PSD_measure = get_PSD(phase,delta)[int(N/2),int(N/2)+1:]
    PSD_theory = psd_function(r0, N, delta, L0, l0)[int(N/2),int(N/2)+1:]

    plt.plot(f, PSD_theory, label='Theory')
    plt.plot(f, PSD_measure, label='Measured',
             marker='.',ls='')
    plt.title('Comparison between theoretical and measured PSD')
    plt.xlabel('Spatial Frequency (1/m)')
    plt.ylabel('PSD')
    plt.xlim(f[0]/2,f[-1]*2)
    plt.ylim(1e-13,1e4)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    """
    N = phase.shape[0]
    del_f = 1./(N*2.*delta)
    phase = phase - np.mean(phase, axis=0)
    cn = my_ft1_axis0(mirror_pad1_axis0(phase), 1./N/2.)
    PSD = (np.abs(cn)**2/del_f**2)*2.   # * 2. is from the fact that
                                        # the input phase is not complex
                                        # phase are complex when generated
                                        # from cn.
    PSD[int(N/2), : ] = 0.         # PSD is 0 at the centre.
    return pick_even_index1_axis0(PSD)

def my_ft1_axis0(g, delta):
    """
    Complete 2D Fourier transform.

    Parameters
    ----------
    g : numpy.ndarray
        (N,N)
        Function.
    delta : float
        Pixel scale.

    Returns
    -------
    G : numpy.ndarray
        (N,N)
        Transformed function.

    """
    # G = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(g))) * delta**2
    return np.fft.ifftshift(np.fft.fft(g, axis=0)) * delta**2

def pick_even_index1_axis0(AA):
    """
    Pick only pair of even indices, applicable to 2d only.

    Parameters
    ----------
    AA : numpy.ndarray
         (N,N)
         Parent array to be wittled.

    Returns
    -------
    A : numpy.2darray
        (N/2,N/2)
        Daugther array.

    """

    return AA[np.arange(0,AA.shape[0],2), : ]

def mirror_pad1_axis0(A):
    r"""
    Use AA = np.pad(A,(int(A.shape[0]/2),int(A.shape[1]/2)),'symmetric').

    Pad A with its mirrored version.

    If  /\ is the parent, then /\/\ is the daughter.
        /\                     \/\/
                               \/\/
                               /\/\
    The parent will be at the center of the daughter,
    surrounded by a half of the parrent's mirrored self,
    doubling in size.

    Parameters
    ----------
    A : numpy.ndarray
        (N,N)
        Parent array.

    Returns
    -------
    AA : numpy.ndarray
         (2N,2N)
         Daughter array.

    Examples
    --------
    x = np.arange(10)
    xx, yy = np.meshgrid(x, x)
    p = xx + yy
    plt.imshow(p)
    plt.title('Parent')
    plt.show()
    pd = mirror_pad2(p)
    plt.imshow(pd)
    plt.title('Daughter')
    plt.show()

    """

    return np.pad(A,((int(A.shape[0]/2),int(A.shape[0]/2)),(0,0)),'symmetric')

ARCSEC2RAD = 1./3600./360.*2.*np.pi
RAD2ARCSEC = 1./ARCSEC2RAD

home = ('C:/Users/gqmc62/OneDrive - Durham University/Desktop/'
             + 'soapy-master/soapy-master/conf')
file = 'PSD3.yaml'
# file = 'PSD2.yaml'
# file = 'PSD1.yaml'
os.chdir(home)


sim = soapy.Sim(file)

loopTime = sim.config.sim.loopTime

r0 = sim.config.atmos.r0
L0 = sim.config.atmos.L0

telDiam = sim.config.tel.telDiam

nxSubaps = sim.config.wfss[0].nxSubaps
pxlsPerSubap = sim.config.wfss[0].pxlsPerSubap
subapFOV = sim.config.wfss[0].subapFOV
wfs_wavelength = sim.config.wfss[0].wavelength

TT_iMatValue = sim.config.dms[0].iMatValue
DM_iMatValue = sim.config.dms[1].iMatValue
DM_nxActuators = sim.config.dms[1].nxActuators

svdConditioning = sim.config.recon.svdConditioning

sci_FOV = sim.config.scis[0].FOV
sci_wavelength = sim.config.scis[0].wavelength
sci_pxls = sim.config.scis[0].pxls

if nxSubaps + 1 != DM_nxActuators:
    print('Not Fried Geometry. Change either nxSubpars or nxActuators.')
    
subapDiam = telDiam/nxSubaps
wfs_diffraction = wfs_wavelength/2/subapDiam*RAD2ARCSEC
wfs_angular_pxls = subapFOV/pxlsPerSubap
if wfs_angular_pxls > wfs_diffraction:
    print('WFS cannot resolve spot.')
    
sci_diffraction = sci_wavelength/2/telDiam*RAD2ARCSEC
sci_angular_pxls = sci_FOV/sci_pxls
if sci_angular_pxls > sci_diffraction:
    print('Science camera cannot resolve spot.')
    
if (L0 < 8*telDiam).any():
    print('L0 might be too small for telescope to be Kolmogorov-like.')
    
if subapFOV < 5.*wfs_diffraction:
    print('Linearity of WFS is not good.')
    

    
nb = input('Proceed?\nType y to proceed: ')
if nb != 'y':
    quit()

for n in np.arange(len(sim.config.dms)):
    sim.config.dms[n].gain = 1

sim.aoinit()
sim.config.calcParams()
sim.makeIMat()

# plt.imshow(sim.recon.interaction_matrix)
# plt.colorbar()
# plt.title('IM')
# plt.show()

# if ((np.isclose(np.max(sim.recon.interaction_matrix[:2]), 1, atol=1e-3) == False)
#     or (np.isclose(np.max(sim.recon.interaction_matrix[2:]), 1, atol=1e-3) == False)):
#     print(np.max(sim.recon.interaction_matrix[:2]))
#     print(np.max(sim.recon.interaction_matrix[2:]))
#     print('suggest tip tilt dm to have imat value of {:.3f}'.format(TT_iMatValue/np.max(sim.recon.interaction_matrix[:2])))
#     print('suggest high order dm to have imat value of {:.3f}'.format(DM_iMatValue/np.max(sim.recon.interaction_matrix[2:])))

# _,svd,_ = np.linalg.svd(sim.recon.interaction_matrix)
# plt.plot(svd/svd.max(),marker='o')
# plt.hlines(svdConditioning,0,len(svd))
# plt.show()

# drop = np.where((svd[:-1] - svd[1:]) == np.max(svd[2:-1] - svd[2+1:]))[0][0]
# if np.isclose((3.*svd[drop] + svd[drop + 1])/4./svd.max(), svdConditioning, atol=1e-3) == False:
#     print('suggest svd at {:.3f}'.format((3.*svd[drop] + svd[drop + 1])/4./svd.max()))

# plt.imshow(sim.recon.control_matrix)
# plt.colorbar()
# plt.title('CM')
# plt.show()


REPEAT = 5

TIME = sim.config.sim.nIters
time = np.arange(TIME)*loopTime
dfrequency = 1./TIME/loopTime
frequency = np.arange(-TIME/2,TIME/2) * dfrequency
total_Cn2dz = r0**(-5./3.)/0.423/(2.*np.pi/sci_wavelength)**2

f_greenwood = (2.31 * sci_wavelength**(-6./5.)
                * (sim.atmos.config.scrnStrengths/sim.atmos.config.scrnStrengths.sum()*total_Cn2dz*sim.atmos.windSpeeds**(5./3.)).sum()**(3./5.))
tau0 = 0.314 * r0 / (sim.atmos.config.scrnStrengths/sim.atmos.config.scrnStrengths.sum()*sim.atmos.windSpeeds**(5./3.)).sum()**(3./5.)

PSD_C = np.zeros((TIME,REPEAT), dtype=float)
PSD_UC = np.zeros((TIME,REPEAT), dtype=float)


for repeat in np.arange(REPEAT):
    sim.reset_loop()
    sim.config.randomSeed = np.random.randint(0, 999)
    sim.atmos = soapy.atmosphere.atmos(sim.config)
    sim.aoloop()
    
    corrected = sim.allSlopes
    uncorrected = sim.allDmCommands # + np.matmul(sim.allSlopes,sim.recon.control_matrix)
    
    psd_corrected = np.mean(get_PSD1_axis0(corrected,loopTime), axis=1)
    psd_uncorrected = np.mean(get_PSD1_axis0(uncorrected,loopTime), axis=1)
    PSD_C[:,repeat] = psd_corrected
    PSD_UC[:,repeat] = psd_uncorrected
    
    # psd_corrected = np.mean(np.abs(np.fft.fftshift(np.fft.fft(corrected,axis=0)))**2,axis=1)/dfrequency**2
    # psd_uncorrected = np.mean(np.abs(np.fft.fftshift(np.fft.fft(uncorrected,axis=0)))**2,axis=1)/dfrequency**2
    # PSD_C[:,repeat] = psd_corrected
    # PSD_UC[:,repeat] = psd_uncorrected
    
    # plt.vlines(f_greenwood,psd_uncorrected.max(),psd_uncorrected.min(),label='greenwood frequency',color='r')
    # plt.plot(frequency[int(TIME/2):],psd_corrected[int(TIME/2):],label='corrected',marker='.')
    # plt.plot(frequency[int(TIME/2):],psd_uncorrected[int(TIME/2):],label='uncorrected',marker='.')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('tip tilt frequency (Hz)')
    # plt.ylabel('PSD')
    # plt.legend()
    # plt.show()
    
    
mean_PSD_UC = np.nanmean(PSD_UC,axis=1)
mean_PSD_C = np.nanmean(PSD_C,axis=1)
rms_PSD_UC = np.nanstd(PSD_UC,axis=1,ddof=1)
rms_PSD_C = np.nanstd(PSD_C,axis=1,ddof=1)



# plt.vlines(f_greenwood,mean_PSD_UC.max(),mean_PSD_UC.min(),label='greenwood frequency',color='r')

# plt.errorbar(frequency[int(TIME/2):],
#              mean_PSD_C[int(TIME/2):],
#              yerr=rms_PSD_C[int(TIME/2):],
#              label='corrected',marker='.',
#              ls='',capsize=2,alpha=0.5)
# plt.errorbar(frequency[int(TIME/2):],
#              mean_PSD_UC[int(TIME/2):],
#              yerr=rms_PSD_UC[int(TIME/2):],
#              label='uncorrected',marker='.',
#              ls='',capsize=2,alpha=0.5)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('tip tilt frequency (Hz)')
# plt.ylabel('PSD')
# plt.legend()
# plt.show()
    
    
    
    
plt.vlines(f_greenwood,ymax=np.nanmax(mean_PSD_UC),ymin=np.nanmin(mean_PSD_UC),label='greenwood frequency',color='k')
plt.vlines(1/(telDiam/nxSubaps/sim.config.atmos.windSpeeds),ymax=np.nanmax(mean_PSD_UC),ymin=np.nanmin(mean_PSD_UC),label='subap frequencies',color='g')
plt.vlines(1/(2.*telDiam/pxlsPerSubap/nxSubaps/sim.config.atmos.windSpeeds),ymax=np.nanmax(mean_PSD_UC),ymin=np.nanmin(mean_PSD_UC),label='pixel Nyquist frequencies',color='c')
plt.plot(frequency[int(TIME/2) + 1:],
         mean_PSD_C[int(TIME/2) + 1:],color='r')
plt.fill_between(frequency[int(TIME/2) + 1:],
             mean_PSD_C[int(TIME/2) + 1:] + rms_PSD_C[int(TIME/2) + 1:],
             mean_PSD_C[int(TIME/2) + 1:] - rms_PSD_C[int(TIME/2) + 1:],
             label='corrected',
             ls='',alpha=0.5,color='r')

plt.plot(frequency[int(TIME/2) + 1:],
         mean_PSD_UC[int(TIME/2) + 1:],color='b')
plt.fill_between(frequency[int(TIME/2) + 1:],
             mean_PSD_UC[int(TIME/2) + 1:] + rms_PSD_UC[int(TIME/2) + 1:],
             mean_PSD_UC[int(TIME/2) + 1:] - rms_PSD_UC[int(TIME/2) + 1:],
             label='uncorrected',color='b',
             ls='',alpha=0.5)

plt.xlim(frequency[int(TIME/2) + 1], frequency[-1])

plt.xscale('log')
plt.yscale('log')
plt.xlabel('tip tilt frequency (Hz)')
plt.ylabel('PSD')
plt.legend()
plt.show()
    