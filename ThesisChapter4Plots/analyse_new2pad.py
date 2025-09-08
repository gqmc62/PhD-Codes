
###############################################################################
##########################################################################
"""
█▀▀█░█░▒█░▀█▀░▀█▀░░▀░░█░░░█░█▀▀▄░▀█▀░░░▒█░▄▀░▄▀▀▄░█▀▀▄░█▀▀▀░█░▄░█▀▀▄░█▀▀░█░░░█
█▄▄█░█░▒█░░█░░░█░░░█▀░▀▄█▄▀░█▄▄█░░█░░░░▒█▀▄░░█░░█░█░▒█░█░▀▄░█▀▄░█▄▄█░█▀▀░▀▄█▄▀
█░░░░░▀▀▀░░▀░░░▀░░▀▀▀░░▀░▀░░▀░░▀░░▀░░░░▒█░▒█░░▀▀░░▀░░▀░▀▀▀▀░▀░▀░▀░░▀░▀▀▀░░▀░▀░
"""
##########################################################################
###############################################################################

import numpy as np
import pickle
from matplotlib import pyplot as plt
import pyfftw
import timeit
from tqdm import tqdm
import os

###############################################################################


###############################################################################

import my_WFSC as MWFS

Name = '202302211459/'

os.chdir(Name)

with open('Ultimate_input.pkl', 'rb') as file:
    myvar = pickle.load(file)
    TARGET = myvar['TARGET']
    SCREEN_LIMIT = myvar['SCREEN_LIMIT']
    n = myvar['n']
    oversize  = myvar['oversize']
    L0_GL = myvar['L0_GL']
    L0_HA = myvar['L0_HA']
    l0_GL = myvar['l0_GL']
    l0_HA = myvar['l0_HA']
    wvl  = myvar['wvl']
    wavenumber = myvar['wavenumber']
    zenith_angle  = myvar['zenith_angle']
    r0 = myvar['r0']
    dor0   = myvar['dor0']
    Rytov  = myvar['Rytov']
    H_HA = myvar['H_HA']
    II = myvar['II']
    JJ   = myvar['JJ']
    KK = myvar['KK']
    LL  = myvar['LL']
    wavenumber  = myvar['wavenumber']
    N_MAX = myvar['N_MAX']
    d= myvar['d']
    delta = myvar['delta']
    r0_GL = myvar['r0_GL']
    r0_HA  = myvar['r0_HA']
      
    N = myvar['N']
    num_lenslet = myvar['num_lenslet']
    REPEAT = myvar['REPEAT']
    POSSIBLE_SCREEN = myvar['POSSIBLE_SCREEN']
    POSSIBLE_PROP = myvar['POSSIBLE_PROP']
    POSSIBLE = myvar['POSSIBLE']

###############################################################################

# set up shwfs propagation here
# also any memory allocation

U = np.zeros((n, n), dtype=complex)

SHWFS_SLOPE = np.zeros((II, JJ, KK, LL, TARGET, 2), dtype=float)
Ig_SLOPE = np.zeros((II, JJ, KK, LL, TARGET, 2), dtype=float)
g_SLOPE = np.zeros((II, JJ, KK, LL, TARGET, 2), dtype=float)

###############################################################################

GWFS = MWFS.WrappedGTilt(n)
IGWFS = MWFS.IgTilt(n)
SHWFS = MWFS.OneStepSHWFS(n)

Pii, Pjj, Pkk, Pll = np.where(POSSIBLE==True)
for possible in np.arange(len(Pii)):
    ii = Pii[possible]
    jj = Pjj[possible]
    kk = Pkk[possible]
    ll = Pll[possible]
    print(ii,jj,kk,ll)
    
    run_name = 'r{:02d}dor{:02d}rytov{:02d}h{:02d}'.format(
        int(r0[ii]*100), int(dor0[jj]*10), int(Rytov[kk]*100),
        int(H_HA[ll]/1000))
    for count in tqdm(np.arange(TARGET)):
        with open(run_name + '/U_tot/U_tot_' + run_name +
                '_{:04d}.pkl'.format(count), 'rb') as file:
            U = pickle.load(file)
        g_SLOPE[ii, jj, kk, ll, count , :
                ] = GWFS.measure(U)
        Ig_SLOPE[ii, jj, kk, ll, count , :
                ] = IGWFS.measure(U)
        SHWFS_SLOPE[ii, jj, kk, ll, count , :
                    ] = SHWFS.measure_no_noise_no_threshold(U)

myvar = {'g_SLOPE' : g_SLOPE,
         'Ig_SLOPE' : Ig_SLOPE,
         'SHWFS_SLOPE' : SHWFS_SLOPE}

with open('NoiselessCircularPupilSlopes.pkl', 'wb') as file:
    pickle.dump(myvar, file)
    file.close()

###############################################################################
##########################################################################
"""
█▀▀█░█░▒█░▀█▀░▀█▀░░▀░░█░░░█░█▀▀▄░▀█▀░░░▒█░▄▀░▄▀▀▄░█▀▀▄░█▀▀▀░█░▄░█▀▀▄░█▀▀░█░░░█
█▄▄█░█░▒█░░█░░░█░░░█▀░▀▄█▄▀░█▄▄█░░█░░░░▒█▀▄░░█░░█░█░▒█░█░▀▄░█▀▄░█▄▄█░█▀▀░▀▄█▄▀
█░░░░░▀▀▀░░▀░░░▀░░▀▀▀░░▀░▀░░▀░░▀░░▀░░░░▒█░▒█░░▀▀░░▀░░▀░▀▀▀▀░▀░▀░▀░░▀░▀▀▀░░▀░▀░
"""
##########################################################################
###############################################################################
