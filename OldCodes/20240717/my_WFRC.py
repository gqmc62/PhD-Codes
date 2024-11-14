# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 13:35:21 2022

@author: gqmc62
"""

import numpy as np
from aotools.functions import zernike
import my_WFS as mwfs
from matplotlib import pyplot as plt
import my_phasescreen as mps

def make_IM(f, lenslet_pixel):
    N = f.shape[-1]
    n_modes = f.shape[0] - 1
    IM = np.zeros((int(2 * (N/lenslet_pixel)**2), n_modes), dtype=float)
    for i in np.arange(n_modes):
        IM[ : , i] = (mwfs.get_vec_lenslet(mwfs.get_slope(
            f[i + 1])*f[0], lenslet_pixel)).flatten()
    return IM