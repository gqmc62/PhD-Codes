# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:52:36 2023

test noise simulation

@author: gqmc62
"""

import numpy as np
from matplotlib import pyplot as plt
# from aotools.turbulence import infinitephasescreen
import my_WFSC as MWFS
from tqdm import tqdm
import pickle
import os
import time

import sys

sample_ii = int(sys.argv[1])
sample_jj = int(sys.argv[2])
sample_kk = int(sys.argv[3])
sample_ll = int(sys.argv[4])

Name = '/mt/cfai/pkongkaew/IGPhaseScreen/'#'202302211459/'
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
    d = myvar['d']
    delta = myvar['delta']
    r0_GL = myvar['r0_GL']
    r0_HA  = myvar['r0_HA']
      
    N = myvar['N']
    num_lenslet = myvar['num_lenslet']
    REPEAT = myvar['REPEAT']
    POSSIBLE_SCREEN = myvar['POSSIBLE_SCREEN']
    POSSIBLE_PROP = myvar['POSSIBLE_PROP']
    POSSIBLE = myvar['POSSIBLE']

SHWFS = MWFS.Focus(n)

U = np.zeros((n, n), dtype=complex)

Pii, Pjj, Pkk, Pll = np.where(POSSIBLE==True)

Image = np.zeros((TARGET,n,n),dtype=float)

for possible in np.arange(len(Pii)):
    ii = Pii[possible]
    jj = Pjj[possible]
    kk = Pkk[possible]
    ll = Pll[possible]

    if sample_ii != ii:
        continue
    if sample_jj != jj:
        continue
    if sample_kk != kk:
        continue

    if sample_ll != ll:
        continue

    run_name = 'r{:02d}dor{:02d}rytov{:02d}h{:02d}'.format(
                        int(r0[ii]*100), int(dor0[jj]*10), int(Rytov[kk]*100),
                        int(H_HA[ll]/1000))

    for count in np.arange(TARGET):
        with open('/mt/cfai/pkongkaew/IGPhaseScreen/U_tot/U_tot_' + run_name + '_{:04d}.pkl'.format(count),'rb') as file:
            U = pickle.load(file)

        Image[count] = SHWFS.focus(U)

    with open('/mt/cfai/pkongkaew/IGPhaseScreen/Image/Image_' + run_name + '.pkl', 'wb') as file:
        pickle.dump(Image, file)
        file.close()
