# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:29:11 2023

@author: gqmc62
"""

import os

from matplotlib import pyplot as plt
import numpy as np
from aotools.turbulence import infinitephasescreen,phasescreen
import random
import time
from numpy import fft
os.chdir("C:\\Users\gqmc62\OneDrive - Durham University\Desktop\soapy-master\soapy-master")
# print(os.getcwd())
import soapy

def jitter(wvl,D,r0):
    """
    return in arcsec

    Parameters
    ----------
    wvl : TYPE
        DESCRIPTION. m
    D : TYPE
        DESCRIPTION. m
    r0 : TYPE
        DESCRIPTION. m

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.sqrt(0.182*wvl**2*D**(-1/3.)*r0**(-5/3.))*206265

def diffract(wvl,D):
    """
    return in arcsec

    Parameters
    ----------
    wvl : TYPE
        DESCRIPTION.
    D : TYPE
        DESCRIPTION.
    r0 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    return wvl/2/D*206265

def atm(wvl,r0):
    """
    this is spread of instantaneous image of atm
    in arcsecond

    Parameters
    ----------
    wvl : TYPE
        DESCRIPTION.
    r0 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return wvl/2/r0*206265
    

sim = soapy.Sim("conf\sh_8x8.yaml")
sim.aoinit()
sim.makeIMat()
sim.aoloop()
# sim.reset_loop()

# n_sci = len(sim.config.scis)
# n_wfs = len(sim.config.wfss)
# n_dm = len(sim.config.dms)

# wvl = (sim.config.scis[0])['wavelength']
# for i in np.arange(n_wfs):
#     if wvl != (sim.config.wfss[i])['wavelength']:
#         print('different wavelengths')
# for i in np.arange(n_scis):
#     if wvl != (sim.config.scis[i])['wavelength']:
#         print('different wavelengths')

# n_px_screen = sim.config.sim['pupilSize']
# if n_px_screen != sim.config.

# D = (sim.config.tel)['telDiam']
# nlenslet = (sim.config.wfss[0])['nxSubaps']
# d = D / nlenslet

plt.imshow(sim.allSlopes+np.matmul(sim.allDmCommands,sim.recon.interaction_matrix))
plt.show()


