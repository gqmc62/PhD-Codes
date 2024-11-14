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





new_config = dict(original_config)
new_config['simName'] = original_config['simName'] + 'perfect_IM' + TSTAMP
ndm = new_config['nDM']
for idm in np.arange(ndm):
    if new_config['DM'][idm]['type'] == 'Aberration':
        new_config['DM'][idm]['calibrate'] = False
        new_config['DM'][idm]['nollMode'] = int(1)
        new_config['DM'][idm]['r0'] = float(aber_r0)
        new_config['DM'][idm]['L0'] = float(aber_L0)
new_name = original_name + 'perfect_IM'
with open(new_name + '.yaml','w') as file:
    yaml.dump(new_config, file)

SIM = soapy.Sim(new_name + '.yaml')
SIM.aoinit()
SIM.makeIMat(forceNew=True)

TITLE = 'perfect_IM'

IM_perfect = SIM.recon.interaction_matrix
MAX = IM_perfect.max()
MIN = IM_perfect.min()

os.chdir(saveto)

plt.imshow(IM_perfect,vmin=MIN,vmax=MAX)
plt.colorbar()
plt.title('perfect_IM')
plt.savefig(new_name + TSTAMP + '.png', bbox_inches='tight')
plt.show()

name = new_name + TSTAMP
hdu = fits.PrimaryHDU(IM_perfect)
hdul = fits.HDUList([hdu])

hdul.writeto(name + '.fits', overwrite=True)
hdul.close()

os.chdir(home)



