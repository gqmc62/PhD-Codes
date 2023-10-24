# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:26:27 2023

@author: gqmc62
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

latest propagation criteria check

"""

import numpy as np
import mypropagationclass as mpc
from matplotlib import pyplot as plt
from matplotlib import colors
from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov

def next_2_power(n):
    N = int(2**(np.ceil(np.log(n)/np.log(2))))
    return N

MAX_N = 2048


r0 = 0.05           # fried parameter
L0 = 5              # outer scale
wvl = 0.5e-6        # wavelength
D = 1.6             # telescope pupil
d = 0.05            # wfs subaperture
subapPxl = 16
subapPlateScale = wvl/2./d
subapFoV = subapPxl*subapPlateScale
delta = d/subapPxl

HA = 10e3

DHA = D + subapFoV*HA

