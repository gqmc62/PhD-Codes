# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 12:09:45 2023

@author: gqmc62
"""

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

import numpy as np
from matplotlib import pyplot as plt
from aotools.turbulence import infinitephasescreen
import MyWFS as mwfs
from tqdm import tqdm
from astropy.io import fits
import my_phasescreen as mps
from aotools.functions import zernike

N = 256
L0 = 20
dor0 = np.array([0.5,1,3], dtype=float)
r0_original = L0/dor0.max()/2
wvl = 500e-9
REPEAT = 10000
r0 = np.array([2], dtype=float)*1e-2
data = np.zeros((len(dor0),len(r0), REPEAT, int(np.log(N)/np.log(2)), 2),dtype=float)
dif = np.zeros((len(dor0),len(r0), REPEAT,int(np.log(N)/np.log(2) - 1),2), dtype=float)

dataIG = np.zeros_like(data)
difIG = np.zeros_like(dif)

RAD2NM = wvl/(2.*np.pi)/1e-9

# inputFileName = 'noscaling.fits'
# hdulist = fits.open(inputFileName)
# hdu = hdulist[0]
# phase = hdu.data
# D = hdu.header['DIAM']
# r0 = hdu.header['FRIED']
# L0 = hdu.header['OUTERL']
# l0 = hdu.header['INNERL']
# hdulist.close()



x = np.arange(N, dtype=float)/(N/4)
x -= x.mean()
yy, xx = np.meshgrid(x,x)
n_modes = 3
Zernike = zernike.zernikeArray(n_modes, N)
GramSchmidt = mps.gram_schmidt_process(Zernike)
PUPIL = Zernike[0]
FlatZernike = Zernike.reshape((n_modes, N*N))
FlatGramSchmidt = GramSchmidt.reshape((n_modes, N*N))
GramSchmidtCoeff2ZernikeCoeff = mps.get_GramSchmidtCoeff2ZernikeCoeff(FlatZernike, FlatGramSchmidt)



for ntest in np.arange(len(dor0)):
    
    d_original = dor0[ntest]*r0_original
    pixel_scale_original = d_original/N
    
    TTstd = np.sqrt(mps.noll_variance(1,N,pixel_scale_original,r0_original)[0])
    
    phasescreen = infinitephasescreen.PhaseScreenKolmogorov(N, pixel_scale_original, r0_original, L0)
    for i in np.arange(N):
        phasescreen.add_row()

    for repeat in tqdm(np.arange(REPEAT)):
        for i in np.arange(N):
            phasescreen.add_row()
        ZernikeCoeff = mps.get_ZernikeCoeff((phasescreen.scrn*PUPIL).flatten(), FlatGramSchmidt, GramSchmidtCoeff2ZernikeCoeff)
        original_main_phasescreen = phasescreen.scrn \
            - ZernikeCoeff[0] - ZernikeCoeff[1]*yy - ZernikeCoeff[2]*xx \
                + np.random.normal(0,TTstd)*yy + np.random.normal(0,TTstd)*xx
        for ir0 in np.arange(len(r0)):
            pixel_scale = dor0[ntest]*r0[ir0]/N
            factor = (r0_original/pixel_scale_original
               * pixel_scale/r0[ir0])**(5./6.)
            d = dor0[ntest]*r0[ir0]
            main_phasescreen = original_main_phasescreen * (factor)
            for power in (np.arange(np.log(N)/np.log(2)) + 1):
                n_binned_screen = 2**power
                binned_phase_screen = rebin(main_phasescreen, (int(n_binned_screen),int(n_binned_screen)))
                WFS = mwfs.AngularPropSHWFS(binned_phase_screen.shape[0], d, 0.5, 128, wvl)
                IG = mwfs.IgTilt(binned_phase_screen.shape[0])
                data[ntest,ir0, repeat, int(power - 1), : ] = WFS.measure_no_noise_no_threshold(np.exp(1j*binned_phase_screen))
                # WFS = mwfs.AngularPropSHWFS(binned_phase_screen.shape[0], d, 0.5, 32, wvl)
                # WFS.measure_no_noise_no_threshold(np.exp(1j*binned_phase_screen))
                dataIG[ntest,ir0, repeat, int(power - 1), : ] = IG.measure(np.exp(1j*binned_phase_screen))
    
for ntest in np.arange(len(dor0)):
    for i in np.arange(REPEAT):
        for j in np.arange(2):
            dif[ntest,ir0,i,:,j] = data[ntest,ir0,i,:-1,j] - data[ntest,ir0,i,-1,j]
            difIG[ntest,ir0,i,:,j] = dataIG[ntest,ir0,i,:-1,j] - data[ntest,ir0,i,-1,j]
            
color_list=['r','g','b','k']

for ntest in np.arange(len(dor0)):
    for ir0 in np.arange(len(r0)):
        plt.plot(2**((np.arange(np.log(N)/np.log(2) - 1) + 1)),
                 np.std(dif[ntest,ir0]*RAD2NM,axis=(0,-1)),
                 # yerr=np.sqrt(2*np.var(dif[ntest,ir0],axis=(0,-1))**2/(dif.shape[2]*dif.shape[4] - 1)),
                 marker='',ls='-',color=color_list[ntest],
                 label='FT, d/r0={:.1f}'.format(dor0[ntest]))
        plt.plot(2**((np.arange(np.log(N)/np.log(2) - 1) + 1)),
                  np.std(difIG[ntest,ir0]*RAD2NM,axis=(0,-1)),
                  # yerr=np.sqrt(2*np.var(dif[ntest,ir0],axis=(0,-1))**2/(dif.shape[2]*dif.shape[4] - 1)),
                  marker='',ls=':',color=color_list[ntest],
                  label='IG, d/r0={:.1f}'.format(dor0[ntest]))
        plt.hlines(5,xmin=1,xmax=256)
plt.title('convergence of slopes to 256x256 phase pixel FT vs number of pixels in phase screen\n'
          +'compared to FT {:d} px'.format(N) + '\nsame output angular scale and output fov')
plt.xlabel('number of pixels in a lenslet input')
plt.ylabel('RMS of slopes difference (nm)')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1,N)
plt.legend()
plt.show()
