# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:19:06 2023

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
file = 'SVD.yaml'
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
    
for n in np.arange(len(sim.config.dms)):
    sim.config.dms[n].gain = 1

sim.aoinit()
sim.config.calcParams()
sim.makeIMat()

plt.imshow(sim.recon.interaction_matrix)
plt.colorbar()
plt.title('IM')
plt.show()

if ((np.isclose(np.max(sim.recon.interaction_matrix[:2]), 1, atol=1e-3) == False)
    or (np.isclose(np.max(sim.recon.interaction_matrix[2:]), 1, atol=1e-3) == False)):
    print(np.max(sim.recon.interaction_matrix[:2]))
    print(np.max(sim.recon.interaction_matrix[2:]))
    print('suggest tip tilt dm to have imat value of {:.3f}'.format(TT_iMatValue/np.max(sim.recon.interaction_matrix[:2])))
    print('suggest high order dm to have imat value of {:.3f}'.format(DM_iMatValue/np.max(sim.recon.interaction_matrix[2:])))

_,svd,_ = np.linalg.svd(sim.recon.interaction_matrix)

drop = np.where((svd[:-1] - svd[1:]) == np.max(svd[2:-1] - svd[2+1:]))[0][0]
if np.isclose((3.*svd[drop] + svd[drop + 1])/4./svd.max(), svdConditioning, atol=1e-3) == False:
    print('suggest svd at {:.3f}'.format((8.*svd[drop] + svd[drop + 2])/10./svd.max()))

svd_drop = svd[drop]/svd.max()
svd_max = 2*svd[drop]/svd.max()
svd_min = svd[drop + 1]/svd.max()/2.
# svd_chosen = (3.*svd[drop] + svd[drop + 1])/4./svd.max()

plt.plot(svd/svd.max(),marker='o')
plt.hlines(svd[drop - 1]/svd.max(),0,len(svd),label='{:.3f}'.format(svd[drop - 1]/svd.max()), color='c')
plt.hlines(svd[drop]/svd.max(),0,len(svd),label='{:.3f}'.format(svd[drop]/svd.max()), color='m')
plt.hlines(svd[drop + 1]/svd.max(),0,len(svd),label='{:.3f}'.format(svd[drop + 1]/svd.max()), color='y')
plt.hlines(svd[drop + 2]/svd.max(),0,len(svd),label='{:.3f}'.format(svd[drop + 2]/svd.max()), color='k')
plt.legend()
plt.title('svd(interaction matrix)')
plt.ylabel('normalized svd')
plt.show()

plt.imshow(sim.recon.control_matrix)
plt.colorbar()
plt.title('CM')
plt.show()

nb = input('Proceed?\nType y to proceed: ')
if nb != 'y':
    quit()

REPEAT = 100
N_svd_sampled = 10
svds = np.linspace(svd_min,svd_max,num=N_svd_sampled)

TIME = sim.config.sim.nIters
time = np.arange(TIME)*loopTime

list_randomSeed = np.random.randint(0, 999, size=REPEAT)

strehl = np.zeros((N_svd_sampled, REPEAT, TIME), dtype=float)

for nsvd in np.arange(N_svd_sampled):
    sim.config.recon.svdConditioning = svds[nsvd]
    sim.config.calcParams()
    sim.makeIMat()
    for n in np.arange(len(sim.config.dms)):
        sim.config.dms[n].gain = 1.
    for repeat in np.arange(REPEAT):
        sim.config.randomSeed = list_randomSeed[repeat]
        sim.reset_loop()
        sim.atmos = soapy.atmosphere.atmos(sim.config)
        sim.aoloop()
        strehl[nsvd, repeat, : ] = sim.instStrehl

STREHL_mean = np.mean(strehl, axis=1)
STREHL_std = np.std(strehl, axis=1, ddof=1)
COLOR = ['k','tab:brown','r','tab:orange','y',
         'g','b','tab:purple','tab:gray','tab:olive']
# for i in np.arange(N_svd_sampled):
#     plt.errorbar(time, STREHL_mean[i, : ],yerr=STREHL_std[i], label='{:.3f}'.format(svds[i]),capsize=2,marker='o',
#              color=COLOR[i])
#     # plt.fill_between(time,
#     #                  STREHL_mean[i, :] + STREHL_std[i, :],
#     #                  STREHL_mean[i, :] - STREHL_std[i, :])

# plt.legend()
# plt.ylabel('Instant Strehl ratio')
# plt.xlabel('time (s)')
# # plt.xscale('log')
# # plt.yscale('log')
# plt.title('AO performance of the same atmospheric condition with different svds')
# plt.show()

for i in np.arange(N_svd_sampled):
    plt.plot(time, STREHL_mean[i, : ], label='{:.3f}'.format(svds[i]),marker='o',
              color=COLOR[i])
    # plt.fill_between(time,
    #                  STREHL_mean[i, :] + STREHL_std[i, :],
    #                  STREHL_mean[i, :] - STREHL_std[i, :])

plt.legend()
plt.ylabel('Instant Strehl ratio')
plt.xlabel('time (s)')
# plt.xscale('log')
# plt.yscale('log')
plt.title('AO performance of the same atmospheric condition with different svds')
plt.show()