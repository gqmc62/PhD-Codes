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




home = r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf'
saveto = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\GLDM-IM-through-turb'

os.chdir(home)

TSTAMP = time.strftime('%Y-%m-%d-%H-%M-%S')

mode_list = np.arange(6 - 1) + 2
MODE = mode_list.shape[0]
aber_r0 = 0.1
aber_L0 = 4
aberrationHeight = 5000
TRIAL = 5

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

if not os.path.exists(original_name + TSTAMP):
    os.makedirs(original_name + TSTAMP)
os.chdir(original_name + TSTAMP)





for i in np.arange(MODE):
    for j in np.arange(MODE):
        
        new_config = dict(original_config)
        new_config['simName'] = original_config['simName'] + 'two_zernike_bg_IM' + '{:d}'.format(i) + TSTAMP
        ndm = new_config['nDM']
        new_idm = 0
        new_dm_config = {}
        for idm in np.arange(ndm):
            if new_config['DM'][idm]['type'] == 'Aberration':
                new_config['DM'][idm]['subtype'] = 'OneZernike'
                new_config['DM'][idm]['calibrate'] = True
                new_config['DM'][idm]['r0'] = float(aber_r0)
                new_config['DM'][idm]['L0'] = float(aber_L0)
                new_config['DM'][idm]['nollMode'] = int(mode_list[i])
                new_dm_config[new_idm] = new_config['DM'][idm]
                new_idm += 1
        new_config['nDM'] = new_idm*2
        
        for idm in np.arange(new_idm):
            new_dm_config[int(idm + new_idm)] = dict(new_dm_config[idm])
            new_dm_config[int(idm + new_idm)]['nollMode'] = int(mode_list[j])
            
        new_config['DM'] = new_dm_config
        new_name = original_name + 'two_zernike_bg_IM'  + '{:d}'.format(i)  + '{:d}'.format(j) + TSTAMP
        with open(new_name + '.yaml','w') as file:
            yaml.dump(new_config, file)
        
        SIM = soapy.Sim(new_name + '.yaml')
        SIM.aoinit()
        
        
        
        

        phase = np.zeros((SIM.recon.n_dms, SIM.recon.scrn_size, SIM.recon.scrn_size))
        
        for idm in np.arange(SIM.config.sim.nDM):
            SIM.dms[idm].aberrationStrength = soapy.DM.get_noll_variance(
                SIM.config.dms[idm].diameter,
                SIM.config.dms[idm].r0,
                SIM.config.dms[idm].L0,SIM.config.dms[idm].nollMode)
            phase[idm] = SIM.dms[idm].makeDMFrame(0)
        
        
        twoZ = np.zeros((SIM.recon.sim_config.totalWfsData),dtype=float)
        
        AB = np.copy(phase)
        
        # plt.imshow(AB.sum(0))
        # plt.title('A&B')
        # plt.colorbar()
        # plt.show()
        
        for wfs_n, wfs in SIM.recon.wfss.items():
            twoZ = -SIM.wfss[0].frame(
                    None,AB,iMatFrame=True)
        
        
        
        AZ = np.zeros((SIM.recon.sim_config.totalWfsData),dtype=float)
        
        A = np.copy(phase)
        A[-1] = 0
        
        # plt.imshow(A.sum(0))
        # plt.title('A')
        # plt.colorbar()
        # plt.show()
        
        for wfs_n, wfs in SIM.recon.wfss.items():
            AZ = -SIM.wfss[0].frame(
                    None,A,iMatFrame=True)
        
        
            
        BZ = np.zeros((SIM.recon.sim_config.totalWfsData),dtype=float)
        
        
        
        B = np.copy(phase)
        B[-2] = 0
        
        # plt.imshow(B.sum(0))
        # plt.title('B')
        # plt.colorbar()
        # plt.show()
        
        for wfs_n, wfs in SIM.recon.wfss.items():
            BZ = -SIM.wfss[0].frame(
                    None,B,iMatFrame=True)
            
            
        del SIM.dms[0].aberrationStrength, SIM.dms[0].aberration
        del SIM.dms[1].aberrationStrength, SIM.dms[1].aberration
        
        os.chdir(saveto)
        
        name = new_name + TSTAMP
        TITLE = 'two_zernike_bg_IM' + 'noll_mode={:d}&{:d},height={:.0f}km,r0={:.0f}cm,L0={:.0f}m'.format(
                mode_list[i],mode_list[j],aberrationHeight/1000,aber_r0*100,aber_L0)
        
        plt.plot(twoZ,label='A$&$B')
        # plt.plot(AZ,label='A')
        # plt.plot(BZ,label='B')
        plt.plot(AZ+BZ,label='A+B')
        plt.title(TITLE)
        plt.legend()
        plt.savefig(name + 'add_two_IM' + '.png', bbox_inches='tight')
        plt.show()
        
        
        
        # plt.imshow(A.sum(0))
        # plt.title('A')
        # plt.colorbar()
        # plt.show()
        # plt.imshow(B.sum(0))
        # plt.title('B')
        # plt.colorbar()
        # plt.show()
        # plt.imshow((A+B).sum(0))
        # plt.title('A+B')
        # plt.colorbar()
        # plt.show()
        # plt.imshow(AB.sum(0))
        # plt.title('A&B')
        # plt.colorbar()
        # plt.show()
        
        hdu = fits.PrimaryHDU(twoZ)
        hdul = fits.HDUList([hdu])
        hdul.writeto(name + 'bothModes' + '.fits', overwrite=True)
        hdul.close()
        
        hdu = fits.PrimaryHDU(AZ)
        hdul = fits.HDUList([hdu])
        hdul.writeto(name + 'firstMode' + '.fits', overwrite=True)
        hdul.close()
        
        hdu = fits.PrimaryHDU(BZ)
        hdul = fits.HDUList([hdu])
        hdul.writeto(name + 'secondMode' + '.fits', overwrite=True)
        hdul.close()
        
        os.chdir(home)
        



