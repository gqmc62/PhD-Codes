# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:58:34 2024

@author: gqmc62
"""


from matplotlib import pyplot as plt
from soapy import soapy
import os
import numpy as np
from tqdm import tqdm
import yaml
import time

from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

def rearrange1(A, position):
    """
    rearrange soapy reported wfs value from 1d with skips into 2d

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    position : TYPE
        DESCRIPTION.

    Returns
    -------
    a : TYPE
        DESCRIPTION.

    """
    N = position.shape[0]
    size = np.max(position) - np.min(position) + 1
    DIM = position.shape[1]
    a = np.zeros((size,)*DIM, dtype=float)
    for n in np.arange(N):
        a[tuple(position[n])] = A[n]
    return a

def get_rcond_advantage_gain(A,plot=False,return_place=False):
  _,s,_ = np.linalg.svd(A)
  s /= s.max()
  energy = s**2
  energy /= energy.sum()
  accumulated_energy = np.zeros_like(energy)
  for i in np.arange(energy.shape[0]):
    accumulated_energy[i] = energy[:i].sum()
  residual = 1 - accumulated_energy
  
  metric = (residual[1:] - energy[:-1])[2:]
  if (metric > 0).all():
    place = energy.shape[0] - 1
  else:
    place = np.where(metric <= 0)[0][0] + 3
    if place > energy.shape[0] - 1:
      place = energy.shape[0] - 1
  rcond = s[place]

  if plot==True:
    # plt.plot(residual[1:],label='residual')
    # plt.plot(energy[:-1],label='energy')
    # plt.legend()
    # plt.yscale('log')
    # plt.vlines(place,0,1)
    # plt.show()

    plt.plot(s)
    plt.hlines(rcond,0,s.shape[0])
    plt.vlines(place,0,1)
    plt.yscale('log')
    plt.show()
  if return_place == True:
    return rcond, place
  else:
    return rcond

def get_rcond_energy_threshold(A,threshold,plot=False,return_place=False):
  _,s,_ = np.linalg.svd(A)
  s /= s.max()
  energy = s**2
  energy /= energy.sum()
  accumulated_energy = np.zeros_like(energy)

  for i in np.arange(energy.shape[0]):
    accumulated_energy[i] = energy[:i].sum()
  residual = 1 - accumulated_energy
  place = np.where(residual<=threshold)[0][0]
  
  if plot==True:
    # plt.plot(residual,label='residual')
    # plt.hlines(residual[place],0,s.shape[0])
    # plt.vlines(place,0,1)
    # plt.yscale('log')
    # plt.show()

    plt.plot(s)
    plt.hlines(s[place],0,s.shape[0])
    plt.vlines(place,0,1)
    plt.yscale('log')
    plt.show()
  rcond = s[place]
  if return_place == True:
    return rcond, place
  else:
    return rcond

def angle_of_arrival(wvl,D,r0):
    alpha = np.sqrt(0.182 * wvl**2 * D**(-1./3.) * r0**(-5./3.))
    return alpha

def scrn_size_at_alt(wvl, D, r0, z):
    size = 4*wvl*z/r0 + D
    return size

def rytov_number(wvl, r0, z):
    # calculate rytov number from rytov approximation
    k = 2.*np.pi/wvl                    # float; rad/m; Wave number.
    cn2dz=r0**(-5./3.)/0.423/k**2        # float; Turbulence strength.
    variance_theory_approx = 0.563 * k**(7./6.) * cn2dz * z**(5./6.)
    return variance_theory_approx

# ###############################################################################
# ###############################################################################
# ###############################################################################


# home = r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf'
# saveto = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\GLDM-IM-through-turb'

# os.chdir(home)

# TSTAMP = time.strftime('%Y-%m-%d-%H-%M-%S')

# mode_list = np.arange(91 - 1 - 2) + 2 + 2#np.arange(91 - 1 - 2) + 2 + 2
# MODE = mode_list.shape[0]
# aber_r0 = 0.05
# aber_L0 = 8
# aberrationHeight = 5000
# TRIAL = 100

# original_name = 'bad_IM'
# original_sim = soapy.Sim(original_name + '.yaml')
# original_config = original_sim.config.configDict


# data = {'home' : home,
#         'saveto' : saveto,
#         'TSTAMP' : TSTAMP,
#         'mode_list' : mode_list,
#         'MODE' : MODE,
#         'aber_r0' : aber_r0,
#         'aber_L0' : aber_L0,
#         'aberrationHeight' : aberrationHeight,
#         'TRIAL' : TRIAL,
#         'original_name' : original_name}

# output = open(original_name + TSTAMP + '.pkl', 'wb')
# pickle.dump(data,output)
# output.close()

# os.chdir(saveto)
# output = open(original_name + TSTAMP + '.pkl', 'wb')
# pickle.dump(data,output)
# output.close()

# os.chdir(home)

# if not os.path.exists(original_name):
#     os.makedirs(original_name)
# home = home + '/' + original_name
# os.chdir(original_name)



# new_config = dict(original_config)
# new_config['simName'] = original_config['simName']
# ndm = new_config['nDM']
# new_idm = 0
# for idm in np.arange(ndm):
#     if new_config['DM'][idm]['type'] == 'Aberration':
#         new_config['DM'][idm]['subtype'] = 'Zernike'
#         new_config['DM'][idm]['calibrate'] = True
#         new_config['DM'][idm]['r0'] = float(aber_r0)
#         new_config['DM'][idm]['L0'] = float(aber_L0)
#         new_config['DM'][idm]['nollMode'] = int(mode_list[-1])




# new_name = original_config['simName']
# with open(new_name + '.yaml','w') as file:
#     yaml.dump(new_config, file)
# name = new_name + TSTAMP


# ###############################################################################
# ###############################################################################
# ###############################################################################


# SIM = soapy.Sim(new_name + '.yaml')
# SIM.aoinit()

# for idm in np.arange(SIM.config.sim.nDM):
#     if SIM.dms[idm].config.type == 'Aberration' :
#         SIM.dms[idm].config.calibrate = False
        
# SIM.makeIMat(forceNew=True)

# for idm in np.arange(SIM.config.sim.nDM):
#     if SIM.dms[idm].config.type == 'Aberration' :
#         SIM.dms[idm].config.calibrate = True

# perfect_IM = SIM.recon.interaction_matrix

# os.chdir(saveto)
# plt.imshow(perfect_IM)
# plt.title('no aberration IM')
# plt.colorbar()
# # plt.title('perfect IM')
# plt.savefig(name + 'perfect_IM' + '.png', bbox_inches='tight')
# plt.show()

# position = np.array(SIM.wfss[0].pupil_subap_coords/16,dtype=int)
# plt.quiver(rearrange1(perfect_IM[22,:44]
#                       + perfect_IM[26,:44]
#                       + perfect_IM[58,:44]
#                       + perfect_IM[62,:44], position),
#             rearrange1(perfect_IM[22,44:]
#                       + perfect_IM[26,44:]
#                       + perfect_IM[58,44:]
#                       + perfect_IM[62,44:], position),
#             scale=5, scale_units='inches')
# plt.axis('square')
# plt.title('perfect_DM_poke')
# plt.savefig(name + 'perfect_IM' + 'pokes' + '.png', bbox_inches='tight')
# plt.show()

# hdu = fits.PrimaryHDU(perfect_IM)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + 'standard' + '.fits', overwrite=True)
# hdul.close()
# os.chdir(home)




# ###############################################################################
# ###############################################################################
# ###############################################################################







# try:
#     del SIM.dms[2].aberrationStrength, SIM.dms[2].aberration
# except:
#     pass

# one_mode_IM = np.zeros((MODE,perfect_IM.shape[0],perfect_IM.shape[1]),dtype=float)

# for i in tqdm(np.arange(MODE)):
    
#     SIM.dms[2].aberrationStrength = np.zeros((MODE),dtype=float)
#     SIM.dms[2].aberrationStrength[i] = 1e-6/500e-9*2*np.pi 
#     # SIM.dms[2].aberrationStrength[i] = soapy.DM.get_noll_variance(
#     #     SIM.dms[2].telescope_diameter,SIM.dms[2].config.r0,SIM.dms[2].config.L0,mode_list[i])**0.5
#     # SIM.dms[2].makeDMFrame(0)
    
#     SIM.makeIMat(forceNew=True)
    
#     one_mode_IM[i] = SIM.recon.interaction_matrix
#     try:
#         del SIM.dms[2].aberrationStrength, SIM.dms[2].aberration
#     except:
#         pass
    
# previous_spot = os.getcwd()

# os.chdir(saveto)
# for mode in np.arange(MODE):
#     plt.imshow(one_mode_IM[mode])
#     plt.title('one_mode_IM' + '{:d}Z'.format(mode_list[mode]))
#     # plt.title('one_mode_IM')
#     plt.colorbar()
#     plt.savefig(name + 'one_mode_IM_' + '{:d}Z'.format(mode_list[mode]) + '.png', bbox_inches='tight')
#     plt.show()
    
#     plt.imshow(one_mode_IM[mode]-perfect_IM)
#     plt.title('dif_one_mode_IM' + '{:d}Z'.format(mode_list[mode]))
#     # plt.title('one_mode_IM')
#     plt.colorbar()
#     plt.savefig(name + 'dif_one_mode_IM_' + '{:d}Z'.format(mode_list[mode]) + '.png', bbox_inches='tight')
#     plt.show()
    
#     plt.quiver(rearrange1(one_mode_IM[mode][22,:44]
#                           + one_mode_IM[mode][26,:44]
#                           + one_mode_IM[mode][58,:44]
#                           + one_mode_IM[mode][62,:44], position),
#                 rearrange1(one_mode_IM[mode][22,44:]
#                           + one_mode_IM[mode][26,44:]
#                           + one_mode_IM[mode][58,44:]
#                           + one_mode_IM[mode][62,44:], position),
#                 scale=1e-2, scale_units='inches')
#     plt.axis('square')
#     plt.title('aberrated_DM_poke' + '{:d}Z'.format(mode_list[mode]))
#     plt.savefig(name + 'one_mode_IM_pokes' + '{:d}Z'.format(mode_list[mode]) + '.png', bbox_inches='tight')
#     plt.show()
    
    
     
#     plt.quiver(rearrange1(one_mode_IM[mode][22,:44]
#                           + one_mode_IM[mode][26,:44]
#                           + one_mode_IM[mode][58,:44]
#                           + one_mode_IM[mode][62,:44]
#                           - (perfect_IM[22,:44]
#                               + perfect_IM[26,:44]
#                               + perfect_IM[58,:44]
#                               + perfect_IM[62,:44]), position),
#                 rearrange1(one_mode_IM[mode][22,44:]
#                           + one_mode_IM[mode][26,44:]
#                           + one_mode_IM[mode][58,44:]
#                           + one_mode_IM[mode][62,44:]
#                           - (perfect_IM[22,44:]
#                               + perfect_IM[26,44:]
#                               + perfect_IM[58,44:]
#                               + perfect_IM[62,44:]), position),
#                 scale=1e-2, scale_units='inches')
#     plt.axis('square')
#     plt.title('difference_aberrated_DM_poke' + '{:d}Z'.format(mode_list[mode]))
#     plt.savefig(name + 'dif_one_mode_IM_pokes' + '{:d}Z'.format(mode_list[mode]) + '.png', bbox_inches='tight')
#     plt.show()
    




# hdu = fits.PrimaryHDU(one_mode_IM)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + 'eachModes' + '.fits', overwrite=True)
# hdul.close()
# os.chdir(previous_spot)



# ###############################################################################
# ###############################################################################
# ###############################################################################






# MASTER_all_modes_IM = np.zeros((TRIAL,
#                                     perfect_IM.shape[0],
#                                     perfect_IM.shape[1]),dtype=float)
# MASTER_strengths = np.zeros((TRIAL,MODE),dtype=float)

# for idm in np.arange(SIM.config.sim.nDM):
#     if SIM.dms[idm].config.type == 'Aberration' :
#         SIM.dms[idm].config.calibrate = True

# for trial in tqdm(np.arange(TRIAL)):

#     SIM.makeIMat(forceNew=True)
#     MASTER_all_modes_IM[trial] = SIM.recon.interaction_matrix
    
#     # phase = np.zeros((SIM.recon.n_dms, SIM.recon.scrn_size, SIM.recon.scrn_size))
#     # for i in np.arange(MODE):
#     #     idm = ndm + i - 1
#     #     phase[idm] = SIM.dms[idm].makeDMFrame(0)
#     # plt.imshow(phase.sum(0))
#     # plt.title('random phase')
#     # plt.colorbar()
#     # plt.show()
    
#     # strengths = np.zeros((MODE),dtype=float)
    
#     MASTER_strengths[trial] = SIM.dms[2].aberrationStrength

#     del SIM.dms[2].aberrationStrength, SIM.dms[2].aberration
#     # print( SIM.dms[idm].aberrationStrength, SIM.dms[idm].aberration)
    



# # previous_spot = os.getcwd()
# os.chdir(saveto)

# MIN = MASTER_all_modes_IM.min()
# MAX = MASTER_all_modes_IM.max()

# name = new_name + TSTAMP

# TITLE = 'standard_plus_stack_bg' + 'measured' + 'height={:.0f}km,r0={:.0f}cm,L0={:.0f}m'.format(
#         aberrationHeight/1000,aber_r0*100,aber_L0)

# plt.imshow(MASTER_all_modes_IM[0])#,vmin=MIN,vmax=MAX)
# plt.title(TITLE)
# plt.colorbar()
# # plt.savefig(name + 'standard_plus_stack_bg'  + 'measured' + '.png', bbox_inches='tight')
# plt.show()


# TITLE = 'standard_plus_stack_bg' + 'dif' + 'height={:.0f}km,r0={:.0f}cm,L0={:.0f}m'.format(
#         aberrationHeight/1000,aber_r0*100,aber_L0)


# # name = new_name + TSTAMP



# hdu = fits.PrimaryHDU(MASTER_all_modes_IM)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + 'measured' + '.fits', overwrite=True)
# hdul.close()

# hdu = fits.PrimaryHDU(MASTER_strengths)
# hdul = fits.HDUList([hdu])
# hdul.writeto(name + 'randomStrength' + '.fits', overwrite=True)
# hdul.close()











# # ###############################################################################
# # # fixed number of modes (choose the best gain on perfect case)
# # # worst performance # lots of time dm act is entangled more in noise
# # ###############################################################################

# # title = 'fixed number of modes from perfect case (gain method)'

# # plot = True

# # perfect_rcond,place = get_rcond_advantage_gain(perfect_IM,plot=False,return_place=True)
# # perfect_CC = np.linalg.pinv(perfect_IM,rcond=perfect_rcond)
# # if plot == True:
# #     _,s,_ = np.linalg.svd(perfect_IM)
# #     plt.plot(s/s.max())
# #     plt.hlines(perfect_rcond,0,s.shape[0])
# #     plt.yscale('log')
# #     plt.title(title)
# #     plt.show()
# #     plt.imshow(perfect_CC)
# #     plt.colorbar()
# #     plt.title('perfect cc ' + title)
# #     plt.show()
# # MAX = perfect_CC.max()
# # MIN = perfect_CC.min()

# # MASTER_all_modes_CC = np.zeros((MASTER_all_modes_IM.shape[0],
# #                                 MASTER_all_modes_IM.shape[2],
# #                                 MASTER_all_modes_IM.shape[1]),dtype=float)
# # for i in np.arange(MASTER_all_modes_IM.shape[0]):
# #     if i <10:
# #         plot = True
# #     else:
# #         plot = False
# #     _,s,_ = np.linalg.svd(MASTER_all_modes_IM[i])
    
# #     MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],rcond=s[place]/s.max())
# #     if plot:
# #         plt.plot(s/s.max())
# #         plt.hlines(s[place]/s.max(),0,s.shape[0])
# #         plt.yscale('log')
# #         plt.title(title)
# #         plt.show()
# #         # plt.imshow(MASTER_all_modes_IM[i])
# #         # plt.title(title)
# #         # plt.colorbar()
# #         # plt.show()
# #         plt.imshow(MASTER_all_modes_CC[i],vmax=MAX,vmin=MIN)
# #         plt.colorbar()
# #         plt.title(title)
# #         plt.show()
# # plt.imshow(MASTER_all_modes_CC.mean(0))
# # plt.colorbar()
# # plt.title('mean cc '+title)
# # plt.show()
# # plt.imshow(MASTER_all_modes_CC.std(0))
# # plt.colorbar()
# # plt.title('std cc '+title)
# # plt.show()


# ###############################################################################
# # same svd as perfect case
# # could be noisy
# ###############################################################################

# title = 'same svd as the perfect case'

# plot = False

# perfect_rcond,place = get_rcond_advantage_gain(perfect_IM,plot=False,return_place=True)
# _,s,_=np.linalg.svd(perfect_IM)
# s /= s.max()
# perfect_rcond = (s[place] + s[place + 1])/2.
# perfect_CC = np.linalg.pinv(perfect_IM,rcond=perfect_rcond)

# MAX = perfect_CC.max()
# MIN = perfect_CC.min()

# if plot == True:
#     _,s,_ = np.linalg.svd(perfect_IM)
#     plt.plot(s/s.max())
#     plt.hlines(perfect_rcond,0,s.shape[0])
#     plt.yscale('log')
#     plt.title(title)
#     plt.show()
#     plt.imshow(perfect_CC,vmax=MAX,vmin=MIN)
#     plt.colorbar()
#     plt.title('perfect cc ' + title)
#     plt.show()


# MASTER_all_modes_CC = np.zeros((MASTER_all_modes_IM.shape[0],
#                                 MASTER_all_modes_IM.shape[2],
#                                 MASTER_all_modes_IM.shape[1]),dtype=float)
# for i in np.arange(MASTER_all_modes_IM.shape[0]):
#     if i <10:
#         plot = False
#     else:
#         plot = False
    
    
#     MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],rcond=perfect_rcond)
#     if plot:
#         _,s,_ = np.linalg.svd(MASTER_all_modes_IM[i])
#         plt.plot(s/s.max())
#         plt.hlines(perfect_rcond,0,s.shape[0])
#         plt.yscale('log')
#         plt.title(title)
#         plt.show()
#         # plt.imshow(MASTER_all_modes_IM[i])
#         # plt.title(title)
#         # plt.colorbar()
#         # plt.show()
#         plt.imshow(MASTER_all_modes_CC[i],vmax=MAX,vmin=MIN)
#         plt.colorbar()
#         plt.title(title)
#         plt.show()
# plt.imshow(MASTER_all_modes_CC.mean(0),vmax=MAX,vmin=MIN)
# plt.colorbar()
# plt.title('mean cc '+title)
# plt.show()
# plt.imshow(MASTER_all_modes_CC.std(0),vmax=MAX,vmin=MIN)
# plt.colorbar()
# plt.title('std cc '+title)
# plt.show()



# # ###############################################################################
# # # include only modes with power larger than 1/rank
# # # could be noisy
# # ###############################################################################

# # title = 'modes power > 1/rank'
# # threshold = ((s**2).sum()/np.linalg.matrix_rank(perfect_IM))**0.5/s.sum()

# # plot = False

# # perfect_CC = np.linalg.pinv(perfect_IM,rcond=threshold)
# # if plot == True:
# #     _,s,_ = np.linalg.svd(perfect_IM)
# #     plt.plot(s/s.max())
# #     plt.hlines(1/np.linalg.matrix_rank(perfect_IM)**0.5,0,s.shape[0])
# #     plt.yscale('log')
# #     plt.title(title)
# #     plt.show()
# #     plt.imshow(perfect_CC)
# #     plt.colorbar()
# #     plt.title('perfect cc ' + title)
# #     plt.show()
# # MAX = perfect_CC.max()
# # MIN = perfect_CC.min()

# # MASTER_all_modes_CC = np.zeros((MASTER_all_modes_IM.shape[0],
# #                                 MASTER_all_modes_IM.shape[2],
# #                                 MASTER_all_modes_IM.shape[1]),dtype=float)
# # for i in np.arange(MASTER_all_modes_IM.shape[0]):
# #     if i <10:
# #         plot = False
# #     else:
# #         plot = False
# #     MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],rcond=threshold)
# #     if plot:
# #         _,s,_ = np.linalg.svd(MASTER_all_modes_IM[i])
# #         plt.plot(s/s.max())
# #         plt.hlines(1/np.linalg.matrix_rank(perfect_IM)**0.5,0,s.shape[0])
# #         plt.yscale('log')
# #         plt.title(title)
# #         plt.show()
# #         # plt.imshow(MASTER_all_modes_IM[i])
# #         # plt.title(title)
# #         # plt.colorbar()
# #         # plt.show()
# #         plt.imshow(MASTER_all_modes_CC[i],vmax=MAX,vmin=MIN)
# #         plt.colorbar()
# #         plt.title(title)
# #         plt.show()
# # plt.imshow(MASTER_all_modes_CC.mean(0))
# # plt.colorbar()
# # plt.title('mean cc '+title)
# # plt.show()
# # plt.imshow(MASTER_all_modes_CC.std(0))
# # plt.colorbar()
# # plt.title('std cc '+title)
# # plt.show()

# ###############################################################################
# # include until residue is less than the average energy per svd element
# # could be noisy
# ###############################################################################

# title = 'include until residue is less than half 1/rank'

# threshold = (1/s.shape[0]/2.)
# # threshold = (1/np.min(s.shape))**2
# # threshold = (1/perfect_IM.shape[1])**2

# plot = False

# perfect_rcond,place = get_rcond_energy_threshold(perfect_IM,threshold=threshold,plot=False,return_place=True)
# perfect_CC = np.linalg.pinv(perfect_IM,rcond=perfect_rcond)
# if plot == True:
#     _,s,_ = np.linalg.svd(perfect_IM)
#     plt.plot(s/s.max())
#     plt.hlines(perfect_rcond,0,s.shape[0])
#     plt.yscale('log')
#     plt.title(title)
#     plt.show()
#     plt.imshow(perfect_CC)
#     plt.title('perfect cc ' + title)
#     plt.colorbar()
#     plt.show()
# # MAX = perfect_CC.max()
# # MIN = perfect_CC.min()


# MASTER_all_modes_CC = np.zeros((MASTER_all_modes_IM.shape[0],
#                                 MASTER_all_modes_IM.shape[2],
#                                 MASTER_all_modes_IM.shape[1]),dtype=float)
# for i in np.arange(MASTER_all_modes_IM.shape[0]):
#     # plot = True
#     if i <10:
#         plot = False
#     else:
#         plot = False
#     perfect_rcond = get_rcond_energy_threshold(
#         MASTER_all_modes_IM[i],
#         threshold=threshold,plot=False)
#     MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],
#                                             rcond=perfect_rcond)
#     if plot:
#         _,s,_ = np.linalg.svd(MASTER_all_modes_IM[i])
#         plt.plot(s/s.max())
#         plt.hlines(perfect_rcond,0,s.shape[0])
#         plt.yscale('log')
#         plt.title(title)
#         plt.show()
#         # plt.imshow(MASTER_all_modes_IM[i].T)
#         # plt.title(title)
#         # plt.colorbar()
#         # plt.show()
#         plt.imshow(MASTER_all_modes_CC[i],vmax=MAX,vmin=MIN)
#         plt.title(title)
#         plt.colorbar()
#         plt.show()
    
# plt.imshow(MASTER_all_modes_CC.mean(0),vmax=MAX,vmin=MIN)
# plt.colorbar()
# plt.title('mean cc ' + title)
# plt.show()
# plt.imshow(MASTER_all_modes_CC.std(0),vmax=MAX,vmin=MIN)
# plt.title('std cc ' + title)
# plt.colorbar()
# plt.show()

hdul = fits.open(r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\SPIE-data\2024-05-21-19-20-32\bad_IM-r05cm-HA6kmall_IM.fits')
MASTER_all_modes_IM = hdul[0].data
hdul.close()
hdul = fits.open(r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\SPIE-data\2024-05-21-19-20-32\bad_IM-r04cm-HA1kmperfect_IM.fits')
perfect_IM = hdul[0].data
hdul.close()



###############################################################################
# include until residue is less than 1%
# this method sounds arbitrary but still making sense.
# 1-99.73% is too strict
# 95.44 shows good promise. i think this is the best method
# 99.0 could be noisy
###############################################################################

title = 'include 99% of all energy'

threshold = 1-99./100.#1-95.44/100#(1/s.shape[0]) # 68.26,95.44,99.73,99.99
# threshold = (1/np.min(s.shape))**2
# threshold = (1/perfect_IM.shape[1])**2

plot = False

perfect_rcond,place = get_rcond_energy_threshold(perfect_IM,threshold=threshold,plot=False,return_place=True)
perfect_CC = np.linalg.pinv(perfect_IM,rcond=perfect_rcond)
if plot == True:
    _,s,_ = np.linalg.svd(perfect_IM)
    plt.plot(s/s.max())
    plt.hlines(perfect_rcond,0,s.shape[0])
    plt.yscale('log')
    plt.title(title)
    plt.show()
    plt.imshow(perfect_CC)
    plt.title('perfect cc ' + title)
    plt.colorbar()
    plt.show()
MAX = perfect_CC.max()
MIN = perfect_CC.min()


MASTER_all_modes_CC = np.zeros((MASTER_all_modes_IM.shape[0],
                                MASTER_all_modes_IM.shape[2],
                                MASTER_all_modes_IM.shape[1]),dtype=float)
for i in np.arange(MASTER_all_modes_IM.shape[0]):
    # plot = True
    if i <10:
        plot = False
    else:
        plot = False
    perfect_rcond = get_rcond_energy_threshold(
        MASTER_all_modes_IM[i],
        threshold=threshold,plot=False)
    MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],
                                            rcond=perfect_rcond)
    if plot:
        _,s,_ = np.linalg.svd(MASTER_all_modes_IM[i])
        plt.plot(s/s.max())
        plt.hlines(perfect_rcond,0,s.shape[0])
        plt.yscale('log')
        plt.title(title)
        plt.show()
        # plt.imshow(MASTER_all_modes_IM[i].T)
        # plt.title(title)
        # plt.colorbar()
        # plt.show()
        plt.imshow(MASTER_all_modes_CC[i],vmax=MAX,vmin=MIN)
        plt.title(title)
        plt.colorbar()
        plt.show()
    
plt.imshow(MASTER_all_modes_CC.mean(0),vmax=MAX,vmin=MIN)
plt.colorbar()
plt.title('mean cc ' + title)
plt.show()
plt.imshow(MASTER_all_modes_CC.std(0),vmax=MAX,vmin=MIN)
plt.title('std cc ' + title)
plt.colorbar()
plt.show()


###############################################################################
# all
###############################################################################

# title = 'fixed number of modes from perfect case (gain method)'

# plot = True

# perfect_rcond,place = get_rcond_advantage_gain(perfect_IM,plot=False,return_place=True)
# perfect_CC = np.linalg.pinv(perfect_IM,rcond=perfect_rcond)
# if plot == True:
#     _,s,_ = np.linalg.svd(perfect_IM)
#     plt.plot(s/s.max())
#     plt.hlines(perfect_rcond,0,s.shape[0])
#     plt.yscale('log')
#     plt.title(title)
#     plt.show()
#     plt.imshow(perfect_CC)
#     plt.colorbar()
#     plt.title('perfect cc ' + title)
#     plt.show()

title = 'same svd as the perfect case'

plot = True

perfect_rcond,place = get_rcond_advantage_gain(perfect_IM,plot=False,return_place=True)
_,s,_=np.linalg.svd(perfect_IM)
s /= s.max()
perfect_rcond = (s[place] + s[place + 1])/2.
perfect_CC = np.linalg.pinv(perfect_IM,rcond=perfect_rcond)
if plot == True:
    _,s,_ = np.linalg.svd(perfect_IM)
    plt.plot(s/s.max())
    plt.hlines(perfect_rcond,0,s.shape[0])
    plt.yscale('log')
    plt.title(title)
    plt.show()
    plt.imshow(perfect_CC)
    plt.colorbar()
    plt.title('perfect cc ' + title)
    plt.show()


title = 'modes power > half 1/rank'

plot = True
_,s,_ = np.linalg.svd(perfect_IM)
s/=s.max()
threshold = ((s**2).sum()/2/np.linalg.matrix_rank(perfect_IM))**0.5
perfect_CC = np.linalg.pinv(perfect_IM,rcond=threshold)
if plot == True:
    
    plt.plot(s/s.max())
    
    plt.hlines(threshold,0,s.shape[0])
    plt.yscale('log')
    plt.title(title)
    plt.show()
    plt.imshow(perfect_CC)
    plt.colorbar()
    plt.title('perfect cc ' + title)
    plt.show()

title = 'include until residue is less than half 1/rank'

threshold = (1/s.shape[0]/2.)

plot = True

perfect_rcond,place = get_rcond_energy_threshold(perfect_IM,threshold=threshold,plot=False,return_place=True)
perfect_CC = np.linalg.pinv(perfect_IM,rcond=perfect_rcond)
if plot == True:
    _,s,_ = np.linalg.svd(perfect_IM)
    plt.plot(s/s.max())
    plt.hlines(perfect_rcond,0,s.shape[0])
    plt.yscale('log')
    plt.title(title)
    plt.show()
    plt.imshow(perfect_CC)
    plt.title('perfect cc ' + title)
    plt.colorbar()
    plt.show()

title = 'include 99% of all energy'

threshold = 1-99./100.

plot = True

perfect_rcond,place = get_rcond_energy_threshold(perfect_IM,threshold=threshold,plot=False,return_place=True)
perfect_CC = np.linalg.pinv(perfect_IM,rcond=perfect_rcond)
if plot == True:
    _,s,_ = np.linalg.svd(perfect_IM)
    plt.plot(s/s.max())
    plt.hlines(perfect_rcond,0,s.shape[0])
    plt.yscale('log')
    plt.title(title)
    plt.show()
    plt.imshow(perfect_CC)
    plt.title('perfect cc ' + title)
    plt.colorbar()
    plt.show()
    


MASTER_all_modes_CC = np.zeros((MASTER_all_modes_IM.shape[0],
                                MASTER_all_modes_IM.shape[2],
                                MASTER_all_modes_IM.shape[1]),dtype=float)


for i in np.arange(MASTER_all_modes_IM.shape[0]):
    if i < 10:
        plot = True
    else:
        plot = False
        
    # title = 'fixed number of modes from perfect case (gain method)'
    # perfect_rcond,place = get_rcond_advantage_gain(perfect_IM,plot=False,return_place=True)
    # _,s,_ = np.linalg.svd(MASTER_all_modes_IM[i])
    # MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],rcond=s[place]/s.max())
    # if plot:
    #     plt.plot(s/s.max())
    #     plt.hlines(s[place]/s.max(),0,s.shape[0])
    #     plt.yscale('log')
    #     plt.title(title)
    #     plt.show()
    #     # plt.imshow(MASTER_all_modes_IM[i])
    #     # plt.title(title)
    #     # plt.colorbar()
    #     # plt.show()
    #     plt.imshow(MASTER_all_modes_CC[i],vmax=MAX,vmin=MIN)
    #     plt.colorbar()
    #     plt.title(title)
    #     plt.show()
    
    
    title = 'same svd as the perfect case'
    perfect_rcond,place = get_rcond_advantage_gain(perfect_IM,plot=False,return_place=True)
    MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],rcond=perfect_rcond)
    if plot:
        # _,s,_ = np.linalg.svd(MASTER_all_modes_IM[i])
        # plt.plot(s/s.max())
        # plt.hlines(perfect_rcond,0,s.shape[0])
        # plt.yscale('log')
        # plt.title(title)
        # plt.show()
        # plt.imshow(MASTER_all_modes_IM[i])
        # plt.title(title)
        # plt.colorbar()
        # plt.show()
        plt.imshow(MASTER_all_modes_CC[i],vmax=MAX,vmin=MIN)
        plt.colorbar()
        plt.title(title)
        plt.show()
        
    _,s,_ = np.linalg.svd(MASTER_all_modes_IM[i])
    s/=s.max()
    title = 'modes power > half 1/rank'
    threshold = ((s**2).sum()/2/np.linalg.matrix_rank(perfect_IM))**0.5
    MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],rcond=threshold)
    if plot:
        # _,s,_ = np.linalg.svd(MASTER_all_modes_IM[i])
        # plt.plot(s/s.max())
        # plt.hlines(threshold,0,s.shape[0])
        # plt.yscale('log')
        # plt.title(title)
        # plt.show()
        # plt.imshow(MASTER_all_modes_IM[i])
        # plt.title(title)
        # plt.colorbar()
        # plt.show()
        plt.imshow(MASTER_all_modes_CC[i],vmax=MAX,vmin=MIN)
        plt.colorbar()
        plt.title(title)
        plt.show()
    
    title = 'include until residue is less than half 1/rank'
    threshold = (1./np.linalg.matrix_rank(perfect_IM)/2.)
    perfect_rcond = get_rcond_energy_threshold(
        MASTER_all_modes_IM[i],
        threshold=threshold,plot=False)
    MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],
                                            rcond=perfect_rcond)
    if plot:
        # _,s,_ = np.linalg.svd(MASTER_all_modes_IM[i])
        # plt.plot(s/s.max())
        # plt.hlines(perfect_rcond,0,s.shape[0])
        # plt.yscale('log')
        # plt.title(title)
        # plt.show()
        # plt.imshow(MASTER_all_modes_IM[i].T)
        # plt.title(title)
        # plt.colorbar()
        # plt.show()
        plt.imshow(MASTER_all_modes_CC[i],vmax=MAX,vmin=MIN)
        plt.title(title)
        plt.colorbar()
        plt.show()

    title = 'include 99% of all energy'
    threshold = 1-99./100.
    perfect_rcond = get_rcond_energy_threshold(
        MASTER_all_modes_IM[i],
        threshold=threshold,plot=False)
    MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],
                                            rcond=perfect_rcond)
    if plot:
        # _,s,_ = np.linalg.svd(MASTER_all_modes_IM[i])
        # plt.plot(s/s.max())
        # plt.hlines(perfect_rcond,0,s.shape[0])
        # plt.yscale('log')
        # plt.title(title)
        # plt.show()
        # plt.imshow(MASTER_all_modes_IM[i].T)
        # plt.title(title)
        # plt.colorbar()
        # plt.show()
        plt.imshow(MASTER_all_modes_CC[i],vmax=MAX,vmin=MIN)
        plt.title(title)
        plt.colorbar()
        plt.show()
        
        
###############################################################################
# compare scd choise
###############################################################################

_,s,_ = np.linalg.svd(perfect_IM)
plt.plot(s/s.max())

title = 'same svd as the perfect case'
perfect_rcond,place = get_rcond_advantage_gain(perfect_IM,plot=False,return_place=True)
plt.hlines(perfect_rcond,0,s.shape[0],label=title,color='c')
plt.vlines(np.where(s/s.max()<=perfect_rcond)[0][0],1e-6,1,color='c')

title = 'modes power > half 1/rank'
threshold = ((s**2).sum()/2/np.linalg.matrix_rank(perfect_IM))**0.5/s.max()
plt.hlines(threshold,0,s.shape[0],label=title,color='m')
plt.vlines(np.where(s<=((s**2).sum()/2/np.linalg.matrix_rank(perfect_IM))**0.5)[0][0],1e-6,1,color='m')

title = 'include until residue is less than half 1/rank'
threshold = (1./np.linalg.matrix_rank(perfect_IM)/2.)
perfect_rcond = get_rcond_energy_threshold(
    perfect_IM,
    threshold=threshold,plot=False)
plt.hlines(perfect_rcond,0,s.shape[0],label=title,color='y')
plt.vlines(np.where(s/s.max()<=perfect_rcond)[0][0],1e-6,1,color='y')

title = 'include 99% of all energy'
threshold = 1-99./100.
perfect_rcond = get_rcond_energy_threshold(
    perfect_IM,
    threshold=threshold,plot=False)
plt.hlines(perfect_rcond,0,s.shape[0],label=title,color='k')
plt.vlines(np.where(s/s.max()<=perfect_rcond)[0][0],1e-6,1,color='k')
# plt.yscale('log')
plt.ylim(1e-6,1)
plt.legend()
plt.show()

for i in np.arange(MASTER_all_modes_IM.shape[0]):
    
    _,s,_ = np.linalg.svd(MASTER_all_modes_IM[i])
    plt.plot(s/s.max())
    
    title = 'same svd as the perfect case'
    perfect_rcond,place = get_rcond_advantage_gain(perfect_IM,plot=False,return_place=True)
    plt.hlines(perfect_rcond,0,s.shape[0],label=title,color='c')
    plt.vlines(np.where(s/s.max()<=perfect_rcond)[0][0],1e-6,1,color='c')
       
    title = 'modes power > half 1/rank'
    threshold = ((s**2).sum()/2/np.linalg.matrix_rank(perfect_IM))**0.5/s.max()
    plt.hlines(threshold,0,s.shape[0],label=title,color='m')
    plt.vlines(np.where(s/s.max()<=threshold)[0][0],1e-6,1,color='m')
    
    title = 'include until residue is less than half 1/rank'
    threshold = (1./np.linalg.matrix_rank(perfect_IM)/2.)
    perfect_rcond = get_rcond_energy_threshold(
        MASTER_all_modes_IM[i],
        threshold=threshold,plot=False)
    plt.hlines(perfect_rcond,0,s.shape[0],label=title,color='y')
    plt.vlines(np.where(s/s.max()<=perfect_rcond)[0][0],1e-6,1,color='y')

    title = 'include 99% of all energy'
    threshold = 1-99./100.
    perfect_rcond = get_rcond_energy_threshold(
        MASTER_all_modes_IM[i],
        threshold=threshold,plot=False)
    plt.hlines(perfect_rcond,0,s.shape[0],label=title,color='k')
    plt.vlines(np.where(s/s.max()<=perfect_rcond)[0][0],1e-6,1,color='k')
    # plt.yscale('log')
    plt.ylim(1e-6,1)
    plt.legend()
    plt.show()
        

###############################################################################




# perfect_rcond,place = get_rcond_advantage_gain(perfect_IM,plot=False,return_place=True)
# _,s,_=np.linalg.svd(perfect_IM)
# s /= s.max()
# perfect_rcond = (s[place] + s[place + 1])/2.
# perfect_CC = np.linalg.pinv(perfect_IM,rcond=perfect_rcond)
# MASTER_all_modes_CC = np.zeros((MASTER_all_modes_IM.shape[0],
#                                 MASTER_all_modes_IM.shape[2],
#                                 MASTER_all_modes_IM.shape[1]),dtype=float)
# for i in np.arange(MASTER_all_modes_IM.shape[0]):
#     MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],rcond=perfect_rcond)

# IM = MASTER_all_modes_IM
# CC = MASTER_all_modes_CC

# plt.imshow(perfect_IM)
# plt.colorbar()
# plt.title('no aberration IM')
# plt.show()
# plt.imshow(perfect_CC)
# plt.colorbar()
# plt.title('no aberration CC')
# plt.show()
# plt.imshow(IM.mean(0))
# plt.colorbar()
# plt.title('average of actual IM')
# plt.show()
# plt.imshow(CC.mean(0))
# plt.colorbar()
# plt.title('average of actual CC')
# plt.show()
# plt.imshow(IM.std(0))
# plt.colorbar()
# plt.title('std of actual IM')
# plt.show()
# plt.imshow(CC.std(0))
# plt.colorbar()
# plt.title('std of actual CC')
# plt.show()
# plt.imshow((IM-perfect_IM).mean(0))
# plt.colorbar()
# plt.title('mean difference between no aberration and actual IM')
# plt.show()
# plt.imshow((CC-perfect_CC).mean(0))
# plt.colorbar()
# plt.title('mean difference between no aberration and actual CC')
# plt.show()
# plt.imshow((IM-perfect_IM).std(0))
# plt.colorbar()
# plt.title('std difference between no aberration and actual IM')
# plt.show()
# plt.imshow((CC-perfect_CC).std(0))
# plt.colorbar()
# plt.title('std difference between no aberration and actual CC')
# plt.show()


###############################################################################


# os.chdir(home)

# position = np.array(SIM.wfss[0].pupil_subap_coords/16,dtype=int)



# ACTSLOPEx = np.zeros((MASTER_all_modes_IM.shape[0],MASTER_all_modes_IM.shape[1],position.max()+1,position.max()+1),dtype=float)
# ACTSLOPEy = np.zeros((MASTER_all_modes_IM.shape[0],MASTER_all_modes_IM.shape[1],position.max()+1,position.max()+1),dtype=float)

# for i in np.arange(ACTSLOPEx.shape[0]):
#     for j in np.arange(ACTSLOPEx.shape[1]):
#         ACTSLOPEx[i,j] = rearrange1(MASTER_all_modes_IM[i,j,:44],position)
#         ACTSLOPEy[i,j] = rearrange1(MASTER_all_modes_IM[i,j,44:],position)
        
# # MIN = ACTSLOPEx.std(0).min()
# # MAX = ACTSLOPEx.std(0).max()
# # for i in np.arange(MASTER_all_modes_IM.shape[1]):
# #     plt.imshow(ACTSLOPEx[:,i].std(0),vmin=MIN,vmax=MAX)
# #     plt.colorbar()
# #     plt.show()
    
# # MIN = ACTSLOPEy.std(0).min()
# # MAX = ACTSLOPEy.std(0).max()
# # for i in np.arange(MASTER_all_modes_IM.shape[1]):
# #     plt.imshow(ACTSLOPEy[:,i].std(0),vmin=MIN,vmax=MAX)
# #     plt.colorbar()
# #     plt.show()
# # trial = 0
# # i=42
# # #for i in np.arange(ACTSLOPEx.shape[1]):
# # for trial in np.arange(ACTSLOPEx.shape[0]):
# #     plt.quiver(np.arange(8),np.arange(8),
# #                ACTSLOPEx[trial,i],# - ACTSLOPEx[:,i].mean(0),
# #                ACTSLOPEy[trial,i])# - ACTSLOPEy[:,i].mean(0))
# #     plt.axis('square')
# #     plt.show()
    
# perfect_ACTSLOPEx = np.zeros((perfect_IM.shape[0],position.max()+1,position.max()+1),dtype=float)
# perfect_ACTSLOPEy = np.zeros((perfect_IM.shape[0],position.max()+1,position.max()+1),dtype=float)

# for j in np.arange(perfect_ACTSLOPEx.shape[0]):
#     perfect_ACTSLOPEx[j] = rearrange1(perfect_IM[j,:44],position)
#     perfect_ACTSLOPEy[j] = rearrange1(perfect_IM[j,44:],position)
        
# MIN = perfect_ACTSLOPEx.min()
# MAX = perfect_ACTSLOPEx.max()

# i=42
# # for i in np.arange(MASTER_all_modes_IM.shape[1]):
# plt.imshow(perfect_ACTSLOPEx[i],vmin=MIN,vmax=MAX)
# plt.colorbar()
# plt.show()

# trial = 0
# i=42
# # for i in np.arange(MASTER_all_modes_IM.shape[1]):
# plt.imshow(ACTSLOPEx[trial,i],vmin=MIN,vmax=MAX)
# plt.colorbar()
# plt.show()
    
# # MIN = perfect_ACTSLOPEx.min()
# # MAX = perfect_ACTSLOPEx.max()

# # i=42
# # # for i in np.arange(MASTER_all_modes_IM.shape[1]):
# # plt.imshow(perfect_ACTSLOPEy[i],vmin=MIN,vmax=MAX)
# # plt.colorbar()
# # plt.show()

# # i=42
# # #for i in np.arange(ACTSLOPEx.shape[1]):
# # # for trial in np.arange(ACTSLOPEx.shape[0]):
# # plt.quiver(np.arange(8),np.arange(8),
# #            perfect_ACTSLOPEx[i],# - ACTSLOPEx[:,i].mean(0),
# #            perfect_ACTSLOPEy[i])# - ACTSLOPEy[:,i].mean(0))
# # plt.axis('square')
# # plt.show()