# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:59:55 2024

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

home = r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf'
saveto = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\GLDM-IM-through-turb'

os.chdir(home)

TSTAMP = time.strftime('%Y-%m-%d-%H-%M-%S')

mode_list = np.arange(91 - 1 - 2) + 2 + 2#np.arange(91 - 1 - 2) + 2 + 2
MODE = mode_list.shape[0]
aber_r0 = 0.05
aber_L0 = 8
aberrationHeight = 5000
TRIAL = 50

original_name = 'sh_8x8_DM_through_aber_IM_generator'
original_sim = soapy.Sim(original_name + '.yaml')
original_config = original_sim.config.configDict


data = {'home' : home,
        'saveto' : saveto,
        'TSTAMP' : TSTAMP,
        'mode_list' : mode_list,
        'MODE' : MODE,
        'aber_r0' : aber_r0,
        'aber_L0' : aber_L0,
        'aberrationHeight' : aberrationHeight,
        'TRIAL' : TRIAL,
        'original_name' : original_name}

output = open(original_name + TSTAMP + '.pkl', 'wb')
pickle.dump(data,output)
output.close()

os.chdir(saveto)
output = open(original_name + TSTAMP + '.pkl', 'wb')
pickle.dump(data,output)
output.close()

os.chdir(home)

if not os.path.exists(original_name):
    os.makedirs(original_name)
os.chdir(original_name)



new_config = dict(original_config)
new_config['simName'] = original_config['simName']
ndm = new_config['nDM']
new_idm = 0
for idm in np.arange(ndm):
    if new_config['DM'][idm]['type'] == 'Aberration':
        new_config['DM'][idm]['subtype'] = 'Zernike'
        new_config['DM'][idm]['calibrate'] = True
        new_config['DM'][idm]['r0'] = float(aber_r0)
        new_config['DM'][idm]['L0'] = float(aber_L0)
        new_config['DM'][idm]['nollMode'] = int(mode_list[-1])




new_name = original_config['simName']
with open(new_name + '.yaml','w') as file:
    yaml.dump(new_config, file)
name = new_name + TSTAMP

SIM = soapy.Sim(new_name + '.yaml')
SIM.aoinit()

for idm in np.arange(SIM.config.sim.nDM):
    if SIM.dms[idm].config.type == 'Aberration' :
        SIM.dms[idm].config.calibrate = False
SIM.makeIMat(forceNew=True)
perfect_IM = SIM.recon.interaction_matrix

os.chdir(saveto)
plt.imshow(perfect_IM)
plt.title('perfect IM')
plt.savefig(name + 'perfect IM' + '.png', bbox_inches='tight')
plt.show()

hdu = fits.PrimaryHDU(perfect_IM)
hdul = fits.HDUList([hdu])
hdul.writeto(name + 'standard' + '.fits', overwrite=True)
hdul.close()
os.chdir(home)

try:
    del SIM.dms[idm].aberrationStrength, SIM.dms[idm].aberration
except:
    pass

one_mode_IM = np.zeros((MODE,perfect_IM.shape[1]),dtype=float)

for i in tqdm(np.arange(MODE)):
    phase = np.zeros((SIM.recon.n_dms, SIM.recon.scrn_size, SIM.recon.scrn_size))
    SIM.dms[2].aberrationStrength = np.zeros((MODE),dtype=float)
    SIM.dms[2].aberrationStrength[i] = 1
    phase[2] = SIM.dms[idm].makeDMFrame(0)
    # plt.imshow(phase.sum(0))
    # plt.title('nollMode={:d}'.format(mode_list[i]))
    # plt.show()

    one_mode_IM[i] = -SIM.wfss[0].frame(
            None,phase,iMatFrame=True)
    
    del SIM.dms[idm].aberrationStrength, SIM.dms[idm].aberration
    
previous_spot = os.getcwd()
os.chdir(saveto)
plt.imshow(one_mode_IM)
plt.title('one_mode_IM')
plt.savefig(name + 'one_mode_IM' + '.png', bbox_inches='tight')
plt.show()

# for i in np.arange(MODE):
#     plt.plot(one_mode_IM[i],label='{:d}'.format(mode_list[i]))
# plt.legend()
# plt.title('one_mode_IM')
# plt.savefig(name + 'one_mode_IM_stack' + '.png', bbox_inches='tight')
# plt.show()

hdu = fits.PrimaryHDU(one_mode_IM)
hdul = fits.HDUList([hdu])
hdul.writeto(name + 'eachModes' + '.fits', overwrite=True)
hdul.close()
os.chdir(previous_spot)


MASTER_reconstructed_IM = np.zeros((TRIAL,
                                    perfect_IM.shape[0],
                                    perfect_IM.shape[1]),dtype=float)
MASTER_all_modes_IM = np.zeros((TRIAL,
                                    perfect_IM.shape[0],
                                    perfect_IM.shape[1]),dtype=float)
MASTER_strengths = np.zeros((TRIAL,MODE),dtype=float)

for idm in np.arange(SIM.config.sim.nDM):
    if SIM.dms[idm].config.type == 'Aberration' :
        SIM.dms[idm].config.calibrate = True

for trial in tqdm(np.arange(TRIAL)):

    SIM.makeIMat(forceNew=True)
    MASTER_all_modes_IM[trial] = SIM.recon.interaction_matrix
    
    # phase = np.zeros((SIM.recon.n_dms, SIM.recon.scrn_size, SIM.recon.scrn_size))
    # for i in np.arange(MODE):
    #     idm = ndm + i - 1
    #     phase[idm] = SIM.dms[idm].makeDMFrame(0)
    # plt.imshow(phase.sum(0))
    # plt.title('random phase')
    # plt.colorbar()
    # plt.show()
    
    # strengths = np.zeros((MODE),dtype=float)
    
    MASTER_strengths[trial] = SIM.dms[idm].aberrationStrength

    del SIM.dms[idm].aberrationStrength, SIM.dms[idm].aberration
    # print( SIM.dms[idm].aberrationStrength, SIM.dms[idm].aberration)
    
    MASTER_reconstructed_IM[trial] = perfect_IM + (MASTER_strengths[trial] @ one_mode_IM)
    




# previous_spot = os.getcwd()
os.chdir(saveto)

MIN = MASTER_all_modes_IM.min()
MAX = MASTER_all_modes_IM.max()

name = new_name + TSTAMP

TITLE = 'standard_plus_stack_bg' + 'reconstructed' + 'height={:.0f}km,r0={:.0f}cm,L0={:.0f}m'.format(
        aberrationHeight/1000,aber_r0*100,aber_L0)

plt.imshow(MASTER_reconstructed_IM[0])#,vmin=MIN,vmax=MAX)
plt.title(TITLE)
plt.colorbar()
# plt.savefig(name + 'standard_plus_stack_bg'  + 'reconstructed' + '.png', bbox_inches='tight')
plt.show()

TITLE = 'standard_plus_stack_bg' + 'measured' + 'height={:.0f}km,r0={:.0f}cm,L0={:.0f}m'.format(
        aberrationHeight/1000,aber_r0*100,aber_L0)

plt.imshow(MASTER_all_modes_IM[0])#,vmin=MIN,vmax=MAX)
plt.title(TITLE)
plt.colorbar()
# plt.savefig(name + 'standard_plus_stack_bg'  + 'measured' + '.png', bbox_inches='tight')
plt.show()

# TITLE = 'standard_plus_stack_bg' + 'reconstructed' + 'height={:.0f}km,r0={:.0f}cm,L0={:.0f}m'.format(
#         aberrationHeight/1000,aber_r0*100,aber_L0)

plt.imshow(MASTER_reconstructed_IM[0] - MASTER_all_modes_IM[0])#,vmin=MIN,vmax=MAX)
# plt.title(TITLE)
plt.colorbar()
# plt.savefig(name + 'standard_plus_stack_bg'  + 'reconstructed' + '.png', bbox_inches='tight')
plt.show()

TITLE = 'standard_plus_stack_bg' + 'dif' + 'height={:.0f}km,r0={:.0f}cm,L0={:.0f}m'.format(
        aberrationHeight/1000,aber_r0*100,aber_L0)


plt.imshow((MASTER_reconstructed_IM - MASTER_all_modes_IM).std(0))#,vmin=MIN,vmax=MAX)
plt.title(TITLE)
plt.colorbar()
plt.savefig(name + 'standard_plus_stack_bg'  + 'rms_dif' + '.png', bbox_inches='tight')
plt.show()

# name = new_name + TSTAMP



hdu = fits.PrimaryHDU(MASTER_all_modes_IM)
hdul = fits.HDUList([hdu])
hdul.writeto(name + 'measured' + '.fits', overwrite=True)
hdul.close()

hdu = fits.PrimaryHDU(MASTER_strengths)
hdul = fits.HDUList([hdu])
hdul.writeto(name + 'randomStrength' + '.fits', overwrite=True)
hdul.close()

hdu = fits.PrimaryHDU(MASTER_reconstructed_IM)
hdul = fits.HDUList([hdu])
hdul.writeto(name + 'reconstructed' + '.fits', overwrite=True)
hdul.close()


os.chdir(home)

DIF = MASTER_reconstructed_IM - MASTER_all_modes_IM
position = np.array(SIM.wfss[0].pupil_subap_coords/16,dtype=int)

MIN=DIF.min()
MAX=DIF.max()

DIFx = DIF[:,:,:44]
DIFy = DIF[:,:,44:]

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

ACTSLOPEx = np.zeros((DIF.shape[0],DIF.shape[1],position.max()+1,position.max()+1),dtype=float)
ACTSLOPEy = np.zeros((DIF.shape[0],DIF.shape[1],position.max()+1,position.max()+1),dtype=float)

for i in np.arange(ACTSLOPEx.shape[0]):
    for j in np.arange(ACTSLOPEx.shape[1]):
        ACTSLOPEx[i,j] = rearrange1(DIFx[i,j],position)
        ACTSLOPEy[i,j] = rearrange1(DIFy[i,j],position)
        
# MIN = ACTSLOPEx.std(0).min()
# MAX = ACTSLOPEx.std(0).max()
# for i in np.arange(DIF.shape[1]):
#     plt.imshow(ACTSLOPEx[:,i].std(0),vmin=MIN,vmax=MAX)
#     plt.colorbar()
#     plt.show()
    
# MIN = ACTSLOPEy.std(0).min()
# MAX = ACTSLOPEy.std(0).max()
# for i in np.arange(DIF.shape[1]):
#     plt.imshow(ACTSLOPEy[:,i].std(0),vmin=MIN,vmax=MAX)
#     plt.colorbar()
#     plt.show()
     

# U,s,V=np.linalg.svd(MASTER_all_modes_IM[0])
# S = np.zeros((U.shape[1],V.shape[0]),dtype=float)
# S[:s.shape[0],:s.shape[0]]=np.diag(s)
# J = np.zeros((s.shape[0]),dtype=float)
# for i in np.arange(s.shape[0]):
#     J[i]=((s[:i]**2).sum()/(s**2).sum(0))










###############################################################################
# fixed number of modes (choose the best gain on perfect case)
###############################################################################

title = 'fixed number of modes from perfect case (gain method)'

plot = True

perfect_rcond,place = get_rcond_advantage_gain(perfect_IM,plot=plot,return_place=True)
perfect_CC = np.linalg.pinv(perfect_IM,rcond=perfect_rcond)
if plot == True:
    plt.imshow(perfect_CC)
    plt.colorbar()
    plt.title('perfect cc ' + title)
    plt.show()
MAX = perfect_CC.max()
MIN = perfect_CC.min()

MASTER_all_modes_CC = np.zeros((MASTER_all_modes_IM.shape[0],
                                MASTER_all_modes_IM.shape[2],
                                MASTER_all_modes_IM.shape[1]),dtype=float)
for i in np.arange(MASTER_all_modes_IM.shape[0]):
    _,s,_ = np.linalg.svd(MASTER_all_modes_IM[i])
    MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],rcond=(s[place] + s[place])/2./s.max())
    if i < 10:
        plt.imshow(MASTER_all_modes_IM[i])
        plt.title(title)
        plt.colorbar()
        plt.show()
        plt.imshow(MASTER_all_modes_CC[i],vmax=MAX,vmin=MIN)
        plt.colorbar()
        plt.title(title)
        plt.show()
plt.imshow(MASTER_all_modes_CC.mean(0))
plt.colorbar()
plt.title('mean cc '+title)
plt.show()
plt.imshow(MASTER_all_modes_CC.std(0))
plt.colorbar()
plt.title('std cc '+title)
plt.show()


###############################################################################
# same svd as perfect case
###############################################################################

# title = 'same svd as gain method of the perfect case'

# plot = True

# perfect_rcond,place = get_rcond_advantage_gain(perfect_IM,plot=plot,return_place=True)
# _,s,_=np.linalg.svd(perfect_IM)
# s /= s.max()
# perfect_rcond = (s[place] + s[place + 1])/2.
# perfect_CC = np.linalg.pinv(perfect_IM,rcond=perfect_rcond)
# if plot == True:
#     plt.imshow(perfect_CC)
#     plt.colorbar()
#     plt.title('perfect cc ' + title)
#     plt.show()
# MAX = perfect_CC.max()
# MIN = perfect_CC.min()

# MASTER_all_modes_CC = np.zeros((MASTER_all_modes_IM.shape[0],
#                                 MASTER_all_modes_IM.shape[2],
#                                 MASTER_all_modes_IM.shape[1]),dtype=float)
# for i in np.arange(MASTER_all_modes_IM.shape[0]):
#     _,s,_ = np.linalg.svd(MASTER_all_modes_IM[i])
#     MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],rcond=perfect_rcond)
#     if i < 10:
#         plt.imshow(MASTER_all_modes_IM[i])
#         plt.title(title)
#         plt.colorbar()
#         plt.show()
#         plt.imshow(MASTER_all_modes_CC[i],vmax=MAX,vmin=MIN)
#         plt.colorbar()
#         plt.title(title)
#         plt.show()
# plt.imshow(MASTER_all_modes_CC.mean(0))
# plt.colorbar()
# plt.title('mean cc '+title)
# plt.show()
# plt.imshow(MASTER_all_modes_CC.std(0))
# plt.colorbar()
# plt.title('std cc '+title)
# plt.show()

# ###############################################################################
# # until energy gain is less than the residual
# ###############################################################################

# title = 'until energy gain is less than the residual'

# plot = False

# perfect_rcond,place = get_rcond_advantage_gain(perfect_IM,plot=plot,return_place=True)
# perfect_CC = np.linalg.pinv(perfect_IM,rcond=perfect_rcond)
# # plt.imshow(perfect_CC)
# # plt.colorbar()
# # plt.title('perfect cc ' + title)
# # plt.show()
# MAX = perfect_CC.max()
# MIN = perfect_CC.min()

# MASTER_all_modes_CC = np.zeros((MASTER_all_modes_IM.shape[0],
#                                 MASTER_all_modes_IM.shape[2],
#                                 MASTER_all_modes_IM.shape[1]),dtype=float)
# for i in np.arange(MASTER_all_modes_IM.shape[0]):
#     plot = False
#     _,s,_ = np.linalg.svd(MASTER_all_modes_IM[i])
#     # MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],rcond=(s[place]+s[place+1])/2./s.max())
#     # MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],rcond=s[place]/s.max())
#     # if i <10:
#     #     plot = True
#     # else:
#     #     plot = False
#     MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],rcond=get_rcond_advantage_gain(
#         MASTER_all_modes_IM[i],plot=plot))
#     # if i < 10:
#     #     plt.imshow(MASTER_all_modes_IM[i])
#     #     plt.title(title)
#     #     plt.colorbar()
#     #     plt.show()
#     #     plt.imshow(MASTER_all_modes_CC[i],vmax=MAX,vmin=MIN)
#     #     plt.title(title)
#     #     plt.colorbar()
#     #     plt.show()
# plt.imshow(MASTER_all_modes_CC.mean(0))
# plt.colorbar()
# plt.title('mean cc ' + title)
# plt.show()
# plt.imshow(MASTER_all_modes_CC.std(0))
# plt.title('std cc ' + title)
# plt.colorbar()
# plt.show()


# ###############################################################################
# # include until residue is less than the average energy per 'pixel'
# ###############################################################################

# title = 'include until residue is less than the average energy per dm_act*wfs_subp'

# plot = False

# threshold = (1/perfect_IM.size)
# # threshold = (1/np.min(s.shape))**2
# # threshold = (1/np.max(s.shape))**2

# perfect_rcond,place = get_rcond_energy_threshold(perfect_IM,threshold=threshold,plot=plot,return_place=True)
# perfect_CC = np.linalg.pinv(perfect_IM,rcond=perfect_rcond)
# # plt.imshow(perfect_CC)
# # plt.title('perfect cc ' + title)
# # plt.colorbar()
# # plt.show()
# MAX = perfect_CC.max()
# MIN = perfect_CC.min()

# MASTER_all_modes_CC = np.zeros((MASTER_all_modes_IM.shape[0],
#                                 MASTER_all_modes_IM.shape[2],
#                                 MASTER_all_modes_IM.shape[1]),dtype=float)
# for i in np.arange(MASTER_all_modes_IM.shape[0]):
#     plot = False
#     # if i <10:
#     #     plot = True
#     # else:
#     #     plot = False
#     MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],
#                                             rcond=get_rcond_energy_threshold(MASTER_all_modes_IM[i],
#                                                                              threshold=threshold,
#                                                                              plot=plot))
#     # if i < 10:
#     #     plt.imshow(MASTER_all_modes_IM[i])
#     #     plt.title(title)
#     #     plt.colorbar()
#     #     plt.show()
#     #     plt.imshow(MASTER_all_modes_CC[i],vmax=MAX,vmin=MIN)
#     #     plt.title(title)
#     #     plt.colorbar()
#     #     plt.show()
    
# plt.imshow(MASTER_all_modes_CC.mean(0))
# plt.colorbar()
# plt.title('mean cc ' + title)
# plt.show()
# plt.imshow(MASTER_all_modes_CC.std(0))
# plt.title('std cc ' + title)
# plt.colorbar()
# plt.show()

# ###############################################################################
# # include until residue is less than the average energy per dm_act
# ###############################################################################

# title = 'include until residue is less than dm_act'

# plot = False

# # threshold = (1/s.size)
# threshold = (1/perfect_IM.shape[0])
# # threshold = (1/np.max(s.shape))**2

# perfect_rcond,place = get_rcond_energy_threshold(perfect_IM,threshold=threshold,plot=plot,return_place=True)
# perfect_CC = np.linalg.pinv(perfect_IM,rcond=perfect_rcond)
# # plt.imshow(perfect_CC)
# # plt.title('perfect cc ' + title)
# # plt.colorbar()
# # plt.show()
# MAX = perfect_CC.max()
# MIN = perfect_CC.min()

# MASTER_all_modes_CC = np.zeros((MASTER_all_modes_IM.shape[0],
#                                 MASTER_all_modes_IM.shape[2],
#                                 MASTER_all_modes_IM.shape[1]),dtype=float)
# for i in np.arange(MASTER_all_modes_IM.shape[0]):
#     plot = False
#     # if i <10:
#     #     plot = True
#     # else:
#     #     plot = False
#     MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],
#                                             rcond=get_rcond_energy_threshold(MASTER_all_modes_IM[i],
#                                                                              threshold=threshold,
#                                                                              plot=plot))
#     # if i < 10:
#     #     plt.imshow(MASTER_all_modes_IM[i])
#     #     plt.title(title)
#     #     plt.colorbar()
#     #     plt.show()
#     #     plt.imshow(MASTER_all_modes_CC[i],vmax=MAX,vmin=MIN)
#     #     plt.title(title)
#     #     plt.colorbar()
#     #     plt.show()
    
# plt.imshow(MASTER_all_modes_CC.mean(0))
# plt.colorbar()
# plt.title('mean cc ' + title)
# plt.show()
# plt.imshow(MASTER_all_modes_CC.std(0))
# plt.title('std cc ' + title)
# plt.colorbar()
# plt.show()

# ###############################################################################
# # include until residue is less than the average energy per dm_act**2
# ###############################################################################

# title = 'include until residue is less than dm_act**2'

# plot = False

# # threshold = (1/s.size)
# threshold = (1/perfect_IM.shape[0])**2
# # threshold = (1/np.max(s.shape))**2

# perfect_rcond,place = get_rcond_energy_threshold(perfect_IM,threshold=threshold,plot=plot,return_place=True)
# perfect_CC = np.linalg.pinv(perfect_IM,rcond=perfect_rcond)
# # plt.imshow(perfect_CC)
# # plt.title('perfect cc ' + title)
# # plt.colorbar()
# # plt.show()
# MAX = perfect_CC.max()
# MIN = perfect_CC.min()

# MASTER_all_modes_CC = np.zeros((MASTER_all_modes_IM.shape[0],
#                                 MASTER_all_modes_IM.shape[2],
#                                 MASTER_all_modes_IM.shape[1]),dtype=float)
# for i in np.arange(MASTER_all_modes_IM.shape[0]):
#     plot = False
#     # if i <10:
#     #     plot = True
#     # else:
#     #     plot = False
#     MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],
#                                             rcond=get_rcond_energy_threshold(MASTER_all_modes_IM[i],
#                                                                              threshold=threshold,
#                                                                              plot=plot))
#     # if i < 10:
#     #     plt.imshow(MASTER_all_modes_IM[i])
#     #     plt.title(title)
#     #     plt.colorbar()
#     #     plt.show()
#     #     plt.imshow(MASTER_all_modes_CC[i],vmax=MAX,vmin=MIN)
#     #     plt.title(title)
#     #     plt.colorbar()
#     #     plt.show()
    
# plt.imshow(MASTER_all_modes_CC.mean(0))
# plt.colorbar()
# plt.title('mean cc ' + title)
# plt.show()
# plt.imshow(MASTER_all_modes_CC.std(0))
# plt.title('std cc ' + title)
# plt.colorbar()
# plt.show()



# ###############################################################################
# # include until residue is less than the average energy per wfs_subap
# ###############################################################################

# title = 'include until residue is less than wfs_subap'

# plot = False

# # threshold = (1/s.size)
# # threshold = (1/np.min(s.shape))**2
# threshold = (1/perfect_IM.shape[1])

# perfect_rcond,place = get_rcond_energy_threshold(perfect_IM,threshold=threshold,plot=plot,return_place=True)
# perfect_CC = np.linalg.pinv(perfect_IM,rcond=perfect_rcond)
# # plt.imshow(perfect_CC)
# # plt.title('perfect cc ' + title)
# # plt.colorbar()
# # plt.show()
# MAX = perfect_CC.max()
# MIN = perfect_CC.min()


# MASTER_all_modes_CC = np.zeros((MASTER_all_modes_IM.shape[0],
#                                 MASTER_all_modes_IM.shape[2],
#                                 MASTER_all_modes_IM.shape[1]),dtype=float)
# for i in np.arange(MASTER_all_modes_IM.shape[0]):
#     plot = False
#     # if i <10:
#     #     plot = True
#     # else:
#     #     plot = False
#     MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],
#                                             rcond=get_rcond_energy_threshold(
#                                                 MASTER_all_modes_IM[i],
#                                                 threshold=threshold,
#                                                 plot=plot))
#     # if i < 10:
#     #     plt.imshow(MASTER_all_modes_IM[i])
#     #     plt.title(title)
#     #     plt.colorbar()
#     #     plt.show()
#     #     plt.imshow(MASTER_all_modes_CC[i],vmax=MAX,vmin=MIN)
#     #     plt.title(title)
#     #     plt.colorbar()
#     #     plt.show()
    
# plt.imshow(MASTER_all_modes_CC.mean(0))
# plt.colorbar()
# plt.title('mean cc ' + title)
# plt.show()
# plt.imshow(MASTER_all_modes_CC.std(0))
# plt.title('std cc ' + title)
# plt.colorbar()
# plt.show()


# ###############################################################################
# # include until residue is less than the average energy per wfs_subap**2
# ###############################################################################

# title = 'include until residue is less than wfs_subap**2'

# # threshold = (1/s.size)
# # threshold = (1/np.min(s.shape))**2
# threshold = (1/perfect_IM.shape[1])**2

# plot = False

# perfect_rcond,place = get_rcond_energy_threshold(perfect_IM,threshold=threshold,plot=plot,return_place=True)
# perfect_CC = np.linalg.pinv(perfect_IM,rcond=perfect_rcond)
# # plt.imshow(perfect_CC)
# # plt.title('perfect cc ' + title)
# # plt.colorbar()
# # plt.show()
# MAX = perfect_CC.max()
# MIN = perfect_CC.min()


# MASTER_all_modes_CC = np.zeros((MASTER_all_modes_IM.shape[0],
#                                 MASTER_all_modes_IM.shape[2],
#                                 MASTER_all_modes_IM.shape[1]),dtype=float)
# for i in np.arange(MASTER_all_modes_IM.shape[0]):
#     plot = False
#     # if i <10:
#     #     plot = True
#     # else:
#     #     plot = False
#     MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],
#                                             rcond=get_rcond_energy_threshold(
#                                                 MASTER_all_modes_IM[i],
#                                                 threshold=threshold,plot=plot))
#     # if i < 10:
#     #     plt.imshow(MASTER_all_modes_IM[i])
#     #     plt.title(title)
#     #     plt.colorbar()
#     #     plt.show()
#     #     plt.imshow(MASTER_all_modes_CC[i],vmax=MAX,vmin=MIN)
#     #     plt.title(title)
#     #     plt.colorbar()
#     #     plt.show()
    
# plt.imshow(MASTER_all_modes_CC.mean(0))
# plt.colorbar()
# plt.title('mean cc ' + title)
# plt.show()
# plt.imshow(MASTER_all_modes_CC.std(0))
# plt.title('std cc ' + title)
# plt.colorbar()
# plt.show()


###############################################################################
# include until residue is less than the average energy per svd element
###############################################################################

title = 'include until residue is less than svd element'

threshold = (1/s.size)
# threshold = (1/np.min(s.shape))**2
# threshold = (1/perfect_IM.shape[1])**2

plot = True

perfect_rcond,place = get_rcond_energy_threshold(perfect_IM,threshold=threshold,plot=plot,return_place=True)
perfect_CC = np.linalg.pinv(perfect_IM,rcond=perfect_rcond)
if plot == True:
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
    # plot = False
    if i <10:
        plot = True
    else:
        plot = False
    MASTER_all_modes_CC[i] = np.linalg.pinv(MASTER_all_modes_IM[i],
                                            rcond=get_rcond_energy_threshold(
                                                MASTER_all_modes_IM[i],
                                                threshold=threshold,plot=plot))
    if i < 10:
        plt.imshow(MASTER_all_modes_IM[i])
        plt.title(title)
        plt.colorbar()
        plt.show()
        plt.imshow(MASTER_all_modes_CC[i],vmax=MAX,vmin=MIN)
        plt.title(title)
        plt.colorbar()
        plt.show()
    
plt.imshow(MASTER_all_modes_CC.mean(0))
plt.colorbar()
plt.title('mean cc ' + title)
plt.show()
plt.imshow(MASTER_all_modes_CC.std(0))
plt.title('std cc ' + title)
plt.colorbar()
plt.show()


###############################################################################

IM = MASTER_all_modes_IM
CC = MASTER_all_modes_CC

plt.imshow(perfect_IM)
plt.colorbar()
plt.title('no aberration IM')
plt.show()
plt.imshow(perfect_CC)
plt.colorbar()
plt.title('no aberration CC')
plt.show()
plt.imshow(IM.mean(0))
plt.colorbar()
plt.title('average of actual IM')
plt.show()
plt.imshow(CC.mean(0))
plt.colorbar()
plt.title('average of actual CC')
plt.show()
plt.imshow(IM.std(0))
plt.colorbar()
plt.title('std of actual IM')
plt.show()
plt.imshow(CC.std(0))
plt.colorbar()
plt.title('std of actual CC')
plt.show()
plt.imshow((IM-perfect_IM).mean(0))
plt.colorbar()
plt.title('mean difference between no aberration and actual IM')
plt.show()
plt.imshow((CC-perfect_CC).mean(0))
plt.colorbar()
plt.title('mean difference between no aberration and actual CC')
plt.show()
plt.imshow((IM-perfect_IM).std(0))
plt.colorbar()
plt.title('std difference between no aberration and actual IM')
plt.show()
plt.imshow((CC-perfect_CC).std(0))
plt.colorbar()
plt.title('std difference between no aberration and actual CC')
plt.show()




