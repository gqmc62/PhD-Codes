# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:29:46 2023

@author: gqmc62
"""

import os
import soapy
import numpy as np
from matplotlib import pyplot as plt

ARCSEC2RAD = 1./3600./360.*2.*np.pi
RAD2ARCSEC = 1./ARCSEC2RAD



home = ('C:/Users/gqmc62/OneDrive - Durham University/Desktop/'
             + 'soapy-master/soapy-master/conf')
file = 'lily.yaml'

os.chdir(home)
sim = soapy.Sim(file)



loopTime = sim.config.sim.loopTime

r0 = sim.config.atmos.r0
L0 = sim.config.atmos.L0

telDiam = sim.config.tel.telDiam

TT_iMatValue = sim.config.dms[0].iMatValue
DM_iMatValue = sim.config.dms[1].iMatValue
DM_nxActuators = sim.config.dms[1].nxActuators

svdConditioning = sim.config.recon.svdConditioning

windSpeeds = sim.config.atmos.windSpeeds
turbStrength = sim.config.atmos.scrnStrengths
turbStrength = turbStrength/turbStrength.sum()
windSpeed53 = (windSpeeds**(5./3.)*turbStrength).sum()**(3./5.)
tau0 = 0.314*r0/windSpeed53

for n in np.arange(sim.config.atmos.scrnNo):
    if L0[n] < 8*telDiam:
        print('layer{:d}, L0 might be too small for telescope to be Kolmogorov-like.'.format(n))

for ndm in np.arange(len(sim.config.dms)):
    for nwfs in np.arange(len(sim.config.wfss)):
        if sim.config.dms[ndm].type == 'FastPiezo':
            if sim.config.wfss[nwfs].nxSubaps + 1 != sim.config.dms[ndm].nxActuators:
                print('dm{:d} and wfs{:d} are Not Fried Geometry.\nChange either nxSubpars or nxActuators.'.format(ndm,nwfs))
                
for n in np.arange(len(sim.config.wfss)):
    nxSubaps = sim.config.wfss[n].nxSubaps
    pxlsPerSubap = sim.config.wfss[n].pxlsPerSubap
    subapFOV = sim.config.wfss[n].subapFOV
    wfs_wavelength = sim.config.wfss[n].wavelength
    wfsTime = sim.config.wfss[n].exposureTime
    r0wfs = r0*(wfs_wavelength/500e-9)**(6./5.)
    tau0wfs = tau0*(wfs_wavelength/500e-9)**(6./5.)
    if tau0wfs < wfsTime:
        print('exposure time of wfs{:d} might be too short'.format(n))
    subapDiam = telDiam/nxSubaps
    wfs_diffraction = wfs_wavelength/2/subapDiam*RAD2ARCSEC
    wfs_angular_pxls = subapFOV/pxlsPerSubap
    if wfs_angular_pxls > wfs_diffraction:
        print('WFS{:d} cannot resolve spot.'.format(n))
    if subapFOV < 5.*wfs_diffraction:
        print('Linearity of WFS{:d} is not good.'.format(n))
    if subapDiam > r0wfs:
        print('r0 is too small for wfs{:d} lenslet'.format(n))
    for natm in np.arange(sim.config.atmos.scrnNo):
        if windSpeeds[natm]*wfsTime < telDiam/nxSubaps/pxlsPerSubap:
            print('shifing of atm{:d} is too slow to accurately simulate with wfs{:d}'.format(natm,n))

for natm in np.arange(sim.config.atmos.scrnNo):
    if windSpeeds[natm]*loopTime < telDiam/nxSubaps/pxlsPerSubap:
        print('shifing of atm{:d} is too slow to accurately simulate'.format(natm))
            
for n in np.arange(len(sim.config.scis)):
    sci_FOV = sim.config.scis[n].FOV
    sci_wavelength = sim.config.scis[n].wavelength
    sci_pxls = sim.config.scis[n].pxls
    sci_diffraction = sci_wavelength/2/telDiam*RAD2ARCSEC
    sci_angular_pxls = sci_FOV/sci_pxls
    if sci_angular_pxls > sci_diffraction:
        print('Science camera {:d} cannot resolve spot.'.format(n))
    for ndm in np.arange(len(sim.config.dms)):
        if sim.config.dms[ndm].type == 'FastPiezo':
            if (telDiam / (sim.config.dms[ndm].nxActuators - 1)) > (r0*(sci_wavelength/500e-9)**(6./5.)):
                print('dm{:d} cannot correct for r0 of sci{:d}'.format(ndm,n))

sim.aoinit()
sim.config.calcParams()
sim.makeIMat()



plt.imshow(sim.recon.interaction_matrix)
plt.colorbar()
plt.title('IM')
plt.show()

if ((np.isclose(np.max(sim.recon.interaction_matrix[ : 2]), 1, atol=1e-1) == False)
    or (np.isclose(np.max(sim.recon.interaction_matrix[2 : ]), 1, atol=1e-1) == False)):
    print(np.max(sim.recon.interaction_matrix[ : 2]))
    print(np.max(sim.recon.interaction_matrix[2 : ]))
    print('suggest tip tilt dm to have imat value of {:.3f}'.format(TT_iMatValue/np.max(sim.recon.interaction_matrix[:2])))
    print('suggest high order dm to have imat value of {:.3f}'.format(DM_iMatValue/np.max(sim.recon.interaction_matrix[2:])))

_, svd, _ = np.linalg.svd(sim.recon.interaction_matrix)

drop = np.where((svd[ : -1] - svd[1 : ]) == np.max(svd[2:-1] - svd[2 + 1 : ]))[0][0]
if np.isclose((9.*svd[drop] + 1.*svd[drop + 1])/10./svd.max(), svdConditioning, atol=1e-1) == False:
    print('suggest svd at {:.3f}'.format((9.*svd[drop] + 1.*svd[drop + 2])/10./svd.max()))

svd_drop = svd[drop]/svd.max()
svd_max = 2*svd[drop]/svd.max()
svd_min = svd[drop + 1]/svd.max()/2.
# svd_chosen = (3.*svd[drop] + svd[drop + 1])/4./svd.max()

plt.plot(svd/svd.max(),marker='o')
# plt.hlines(svd[drop - 1]/svd.max(),0,len(svd),label='{:.3f}'.format(svd[drop - 1]/svd.max()), color='c')
plt.hlines(svd[drop]/svd.max(),0,len(svd),label='{:.3f}'.format(svd[drop]/svd.max()), color='m')
plt.hlines(svd[drop + 1]/svd.max(),0,len(svd),label='{:.3f}'.format(svd[drop + 1]/svd.max()), color='y')
# plt.hlines(svd[drop + 2]/svd.max(),0,len(svd),label='{:.3f}'.format(svd[drop + 2]/svd.max()), color='k')
plt.legend()
plt.title('svd(interaction matrix)')
plt.ylabel('normalized svd')
plt.show()

plt.imshow(sim.recon.control_matrix)
plt.colorbar()
plt.title('CM')
plt.show()



for n in np.arange(len(sim.config.dms)):
    sim.config.dms[n].gain = 1
sim.aoloop()



plt.imshow(sim.wfss[0].wfsDetectorPlane)
plt.colorbar()
plt.show()
plt.imshow(np.sum(sim.scrns, axis=0)*sim.mask)
plt.colorbar()
plt.show()
plt.imshow((sim.dms[0].dm_screen + sim.dms[1].dm_screen)*sim.mask)
plt.colorbar()
plt.show()
plt.imshow(sim.sciCams[0].residual*sim.mask)
plt.colorbar()
plt.show()
plt.imshow(sim.sciCams[0].detector)
plt.colorbar()
plt.show()