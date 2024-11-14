# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:42:48 2023

maiking kolmogorov statistics

@author: gqmc62
"""
# L0/D <= 6.25
# 
import numpy as np
from matplotlib import pyplot as plt
from aotools.turbulence import infinitephasescreen
import MyWFS as mwfs
from tqdm import tqdm
from astropy.io import fits
import my_phasescreen as mps
from aotools.functions import zernike

N = 256

wvl = 500e-9
REPEAT = 10000
r0 = 11*1e-2

original_main_phasescreen_list = np.zeros((REPEAT,N,N), dtype=float)

pixel_scale = 1

d_original = N*pixel_scale

L0 = 2**12*pixel_scale

phasescreen = infinitephasescreen.PhaseScreenKolmogorov(N, pixel_scale, r0, L0)
# for i in np.arange(N):
#     phasescreen.add_row()
# main_phasescreen = phasescreen.scrn - phasescreen.scrn.mean()
# plt.imshow(main_phasescreen)
# plt.colorbar()
# plt.show()

# main_U = np.exp(1j*main_phasescreen)

TTstd = np.sqrt(mps.noll_variance(1,N,pixel_scale,r0))

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

for repeat in tqdm(np.arange(REPEAT)):
    for i in np.arange(N):
        phasescreen.add_row()
    ZernikeCoeff = mps.get_ZernikeCoeff((phasescreen.scrn*PUPIL).flatten(), FlatGramSchmidt, GramSchmidtCoeff2ZernikeCoeff)
    original_main_phasescreen_list[repeat] = phasescreen.scrn \
        - ZernikeCoeff[0] - ZernikeCoeff[1]*yy - ZernikeCoeff[2]*xx \
            + np.random.normal(0,TTstd)*yy + np.random.normal(0,TTstd)*xx
    # plt.imshow(original_main_phasescreen_list[repeat])
    # plt.colorbar()
    # plt.show()


name = 'noscaling2.fits'
hdu = fits.PrimaryHDU(original_main_phasescreen_list)
hdul = fits.HDUList([hdu])
hdr = hdul[0].header
hdu.header['DIAM'] = N * pixel_scale
hdu.header['FRIED'] = r0
hdu.header['OUTERL'] = L0
hdu.header['INNERL'] = 1e-10
hdul.writeto(name, overwrite=True)
hdul.close()

mps.testPhaseScreenKolmogorov(name)


name = 'scaledr02.fits'

new_r0 = 0.05
new_d = new_r0
new_pixel_scale = new_d/N

factor = (r0/pixel_scale
   * new_pixel_scale/new_r0)**(5./6.)
new_L0 = L0/pixel_scale*new_pixel_scale
hdu = fits.PrimaryHDU(original_main_phasescreen_list * factor)
hdul = fits.HDUList([hdu])
hdr = hdul[0].header
hdu.header['DIAM'] = N * new_pixel_scale
hdu.header['FRIED'] = new_r0
hdu.header['OUTERL'] = new_L0
hdu.header['INNERL'] = 1e-10
hdul.writeto(name, overwrite=True)
hdul.close()

mps.testPhaseScreenKolmogorov(name)

