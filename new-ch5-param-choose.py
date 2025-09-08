# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 11:55:45 2025

@author: gqmc62
"""

import numpy as np
from aotools.turbulence import cn2_to_seeing, r0_to_cn2

RAD2ARCSEC = 360/2/np.pi*3600

D = 0.4
nxsub = 8
d = D/nxsub

r0 = 0.023
wvl = 500e-9

cn2 = r0_to_cn2(r0,wvl)
seeing = cn2_to_seeing(cn2,wvl)

###
### wfs
###

wfs_dif_limit = wvl/2/d*RAD2ARCSEC

wfs_fov = 4*seeing
pxl_per_subap = wfs_fov/wfs_dif_limit
wfs_pupil_pxl = nxsub*pxl_per_subap

###
### sci
###

sci_dif_limit = wvl/2/D*RAD2ARCSEC
sci_fov = 4*seeing
sci_pupil_pxl = sci_fov/sci_dif_limit


print('suggest wfs fov = {:}"'.format(wfs_fov))
print('suggest pxl/subap = {:}'.format(pxl_per_subap))
print('suggest pxl/pupil = {:}'.format(wfs_pupil_pxl))

