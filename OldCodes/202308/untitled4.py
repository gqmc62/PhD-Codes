# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:30:29 2023

@author: gqmc62
"""
import os
import soapy
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from scipy.optimize import lsq_linear
from scipy.stats import bootstrap
from tqdm import tqdm

ARCSEC2RAD = 1./3600./360.*2.*np.pi
RAD2ARCSEC = 1./ARCSEC2RAD

home = ('C:/Users/gqmc62/OneDrive - Durham University/Desktop/'
             + 'soapy-master/soapy-master/conf')
os.chdir(home)

file = 'slope_covariance_most_correct.yaml'
sim = soapy.Sim(file)

sim.aoinit()
sim.config.calcParams()
sim.makeIMat()

N = sim.wfss[0].n_subaps

scrnHeights_list = (np.arange(sim.config.wfss[0].nxSubaps - 2, dtype=float)
                    * (sim.config.tel.telDiam/sim.config.wfss[0].nxSubaps)
                    / ((sim.config.wfss[0].GSPosition[0]
                        - sim.config.wfss[1].GSPosition[0])*ARCSEC2RAD))
LAYER = scrnHeights_list.shape[0]

SEEDS = np.random.randint(0, 999, LAYER)

position = np.asarray(sim.wfss[0].pupil_subap_coords/sim.wfss[0].nx_subap_pixels,dtype=int) - 1

mid = sim.config.wfss[0].nxSubaps - 1 - 1 - 1
TIME = sim.config.sim.nIters

grid = np.arange(mid + 1,dtype=float)
grid -= len(grid)/2
grid += 0.5

WFS_open = np.zeros((LAYER,TIME,4*N), dtype=float)

# for layer in np.arange(LAYER):
#     print('at layer {:d}'.format(layer))
#     scrnHeights = scrnHeights_list[layer]
#     sim.config.atmos.scrnHeights = [scrnHeights]
#     sim.aoinit()
#     sim.config.calcParams()
#     sim.makeIMat()
#     sim.config.randomSeed = SEEDS[layer]
#     sim.atmos = soapy.atmosphere.atmos(sim.config)
    
#     for i in np.arange(len(sim.dms)):
#         sim.dms[i].gain = 0.
#     sim.aoloop()
    
#     WFS_open[layer] = sim.allSlopes

# name = 'WFS_open'
# hdu = fits.PrimaryHDU(WFS_open)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()

WFS_close = np.zeros((LAYER,TIME,4*N), dtype=float)
DM_close = np.zeros((LAYER,sim.config.sim.nIters, sim.config.sim.totalActs), dtype=float)

for layer in np.arange(LAYER):
    print('at layer {:d}'.format(layer))
    scrnHeights = scrnHeights_list[layer]
    sim.config.atmos.scrnHeights = [scrnHeights]
    sim.aoinit()
    sim.makeIMat(forceNew=True)
    sim.config.randomSeed = SEEDS[layer]
    sim.atmos = soapy.atmosphere.atmos(sim.config)
    sim.allDmCommands = np.zeros((sim.config.sim.nIters, sim.config.sim.totalActs))
    
    for i in np.arange(len(sim.dms)):
        sim.dms[i].gain = 1.
    sim.aoloop()
    
    WFS_close[layer] = sim.allSlopes
    DM_close[layer] = sim.allDmCommands


name = 'WFS_close'
hdu = fits.PrimaryHDU(WFS_close)
hdul = fits.HDUList([hdu])
hdul.writeto(name + '.fits', overwrite=True)
hdul.close()

name = 'DM_close'
hdu = fits.PrimaryHDU(DM_close)
hdul = fits.HDUList([hdu])
hdul.writeto(name + '.fits', overwrite=True)
hdul.close()

name = 'CM'
hdu = fits.PrimaryHDU(sim.recon.control_matrix)
hdul = fits.HDUList([hdu])
hdul.writeto(name + '.fits', overwrite=True)
hdul.close()
