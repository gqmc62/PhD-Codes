# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:35:09 2023

@author: gqmc62
"""

import numpy as np
from matplotlib import pyplot as plt
import soapy

def powerOf2(val):
    VAL = int(2**(np.ceil(np.log(val)/np.log(2))))
    return VAL

ARCSEC2RAD = 1./3600./360.*2.*np.pi
RAD2ARCSEC = 1./ARCSEC2RAD




r0 = 0.05
nxSubaps = 8
pxlsPerSubap = 16
wavelength = 500e-9


telDiam = r0*nxSubaps
wholeScrnSize = nxSubaps*pxlsPerSubap
pupilSize = wholeScrnSize
subapFOV = wavelength/2/(telDiam/nxSubaps)*RAD2ARCSEC*pxlsPerSubap

DM_TT_iMatValue = wavelength / 1e-9 / wholeScrnSize / 2 #??
nxActuators = nxSubaps + 1
DM_FastPiezo_iMatValue = wavelength / 1e-9


sim = soapy.Sim(
    'C:/Users/gqmc62/OneDrive - Durham University/'
    + 'Desktop/soapy-master/soapy-master/conf/sh_8x8.yaml')
sim.aoinit()
sim.makeIMat()

IM = sim.recon.interaction_matrix
_ , svd, _ = np.linalg.svd(IM)
plt.plot(svd/svd.max())
plt.show()

IM = sim.recon.interaction_matrix
plt.imshow(IM)
plt.colorbar()
plt.show()

CC = sim.recon.control_matrix
plt.imshow(CC)
plt.colorbar()
plt.show()


# drop = (0.60) # y-axis value where IM plot drop
# svdConditioning = drop/svdIM.max()


FOV = wavelength/r0*RAD2ARCSEC*2*2
pxls = FOV/(wavelength/telDiam/2*RAD2ARCSEC)

IM = sim.recon.interaction_matrix
CC = sim.recon.control_matrix
plt.imshow(np.matmul(IM,CC))
plt.colorbar()
plt.show()

"""
sim
pupilSize

tel
wholeScrnSize
telDiam

wfs
nxSubaps
pxlsPerSubap
subapFOV
wavelength

DM
DM_TT_iMatValue
nxActuators
DM_FastPiezo_iMatValue

recon
svdConditioning

science
FOV
wavelength
pxls
"""





