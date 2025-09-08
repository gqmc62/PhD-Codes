# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:52:36 2023

test noise simulation

@author: gqmc62
"""

# ###############################################################################
# ##########################################################################
# """
# █▀▀█░█░▒█░▀█▀░▀█▀░░▀░░█░░░█░█▀▀▄░▀█▀░░░▒█░▄▀░▄▀▀▄░█▀▀▄░█▀▀▀░█░▄░█▀▀▄░█▀▀░█░░░█
# █▄▄█░█░▒█░░█░░░█░░░█▀░▀▄█▄▀░█▄▄█░░█░░░░▒█▀▄░░█░░█░█░▒█░█░▀▄░█▀▄░█▄▄█░█▀▀░▀▄█▄▀
# █░░░░░▀▀▀░░▀░░░▀░░▀▀▀░░▀░▀░░▀░░▀░░▀░░░░▒█░▒█░░▀▀░░▀░░▀░▀▀▀▀░▀░▀░▀░░▀░▀▀▀░░▀░▀░
# """
# ##########################################################################
# ###############################################################################

import numpy as np
import pickle
from matplotlib import pyplot as plt
import pyfftw
import timeit
from tqdm import tqdm
import os

n_cog = 16
threshold = 1e-2

# ###############################################################################


# ###############################################################################

circularAperture_list = [False]

for circularAperture in circularAperture_list:

    if circularAperture == True:
        import my_WFSC as MWFS
    else:
        import MyWFS as MWFS



    # Name = '202302211459/'

    # os.chdir(Name)

    with open('/mt/cfai/pkongkaew/c4data/Ultimate_input.pkl', 'rb') as file:
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

    Pii, Pjj, Pkk, Pll = np.where(POSSIBLE==True)
    for possible in np.arange(len(Pii)):
        ii = Pii[possible]
        jj = Pjj[possible]
        kk = Pkk[possible]
        ll = Pll[possible]
        # print(ii,jj,kk,ll)
        

    # ###############################################################################
    # ##########################################################################
    # """
    # █▀▀█░█░▒█░▀█▀░▀█▀░░▀░░█░░░█░█▀▀▄░▀█▀░░░▒█░▄▀░▄▀▀▄░█▀▀▄░█▀▀▀░█░▄░█▀▀▄░█▀▀░█░░░█
    # █▄▄█░█░▒█░░█░░░█░░░█▀░▀▄█▄▀░█▄▄█░░█░░░░▒█▀▄░░█░░█░█░▒█░█░▀▄░█▀▄░█▄▄█░█▀▀░▀▄█▄▀
    # █░░░░░▀▀▀░░▀░░░▀░░▀▀▀░░▀░▀░░▀░░▀░░▀░░░░▒█░▒█░░▀▀░░▀░░▀░▀▀▀▀░▀░▀░▀░░▀░▀▀▀░░▀░▀░
    # """
    # ##########################################################################
    # ###############################################################################

    from concurrent.futures import ProcessPoolExecutor
    import pickle
    import numpy as np
    import os

    TARGET = 1024

    if circularAperture == False:

        def compute_slopes_single(args):
            ii, jj, kk, ll, run_name, TARGET, n = args[:7]
            SHWFS_class = args[7:][0]
            SHWFS = SHWFS_class(n)

            Image = np.zeros((TARGET,n,n),dtype=float)

            for count in np.arange(TARGET):
                try:
                    with open('/mt/cfai/pkongkaew/U_tot/U_tot_' + run_name + '_{:04d}.pkl'.format(count),'rb') as file:
                        U = pickle.load(file)

                    Image[count] = SHWFS.focus(U)
                except:
                    Image[count,:] = np.nan


            with open('/mt/cfai/pkongkaew/IGPhaseScreen/Image/sq-ap-Image_' + run_name + '.pkl', 'wb') as file:
                pickle.dump(Image, file)
                file.close()
            return

        # Create a task list for all valid (ii, jj, kk, ll) combinations
        task_list = []
        for ii, jj, kk, ll in zip(*np.where(POSSIBLE == True)):
            run_name = 'r{:02d}dor{:02d}rytov{:02d}h{:02d}'.format(
                int(r0[ii]*100), int(dor0[jj]*10), int(Rytov[kk]*100), int(H_HA[ll]/1000))
            task_list.append((ii, jj, kk, ll, run_name, TARGET, n, MWFS.Focus))


        from tqdm import tqdm

        with ProcessPoolExecutor() as executor:
            for result in tqdm(executor.map(compute_slopes_single, task_list), total=len(task_list)):
                _ = result
                
        ###########################################################################################

        def compute_slopes_single_noscintillation(args):
            ii, jj, TARGET, n, run_name = args[:5]
            load_r0, load_d, new_r0, new_d = args[5:9]
            SHWFS_class = args[9:][0]
            SHWFS = SHWFS_class(n)

            Image = np.zeros((TARGET,n,n),dtype=float)

            for count in range(TARGET):
                filepath = f'/mt/cfai/pkongkaew/U_tot/ps_GL_r02dor20rytov01h05_{count:04d}.pkl'
                try:
                    with open(filepath, 'rb') as file:
                        load_phase = pickle.load(file)

                    phase = load_phase*((load_r0/load_d) * (new_d/new_r0))**(5./6.)
                    U = np.exp(1j*phase)

                    Image[count] = SHWFS.focus(U)

                except:
                    Image[count,:] = np.nan

            with open('/mt/cfai/pkongkaew/IGPhaseScreen/Image/sq-ap-noscintillation-Image_' + run_name + '.pkl', 'wb') as file:
                pickle.dump(Image, file)
                file.close()

            return

        # Create a task list for all valid (ii, jj, kk, ll) combinations

        load_r0 = r0_GL[3,0,0]
        load_d = 2*load_r0

        task_list_noscintillation = []
        for ii in range(II):
            for jj in range(JJ):
                new_r0 = r0[ii]#0.11
                new_dor0 = dor0[jj]#0.5
                new_d = new_dor0*new_r0
                run_name = 'r{:02d}dor{:02d}'.format(int(r0[ii]*100), int(dor0[jj]*10))
                task_list_noscintillation.append((ii, jj, TARGET, n, run_name,
                                                load_r0, load_d, new_r0, new_d,
                                                MWFS.Focus))


        from tqdm import tqdm


        with ProcessPoolExecutor() as executor:
            for result in tqdm(executor.map(compute_slopes_single_noscintillation, task_list_noscintillation), total=len(task_list_noscintillation)):
                _ = result

    else:
        def compute_slopes_single(args):
            ii, jj, kk, ll, run_name, TARGET, n = args[:7]
            SHWFS_class = args[7:][0]
            SHWFS = SHWFS_class(n)

            Image = np.zeros((TARGET,n,n),dtype=float)

            for count in np.arange(TARGET):
                try:
                    with open('/mt/cfai/pkongkaew/U_tot/U_tot_' + run_name + '_{:04d}.pkl'.format(count),'rb') as file:
                        U = pickle.load(file)

                    Image[count] = SHWFS.focus(U)

                except:
                    Image[count,:] = np.nan

            with open('/mt/cfai/pkongkaew/IGPhaseScreen/Image/circ-ap-Image_' + run_name + '.pkl', 'wb') as file:
                pickle.dump(Image, file)
                file.close()
            return

        # Create a task list for all valid (ii, jj, kk, ll) combinations
        task_list = []
        for ii, jj, kk, ll in zip(*np.where(POSSIBLE == True)):
            run_name = 'r{:02d}dor{:02d}rytov{:02d}h{:02d}'.format(
                int(r0[ii]*100), int(dor0[jj]*10), int(Rytov[kk]*100), int(H_HA[ll]/1000))
            task_list.append((ii, jj, kk, ll, run_name, TARGET, n, MWFS.Focus))


        from tqdm import tqdm

        with ProcessPoolExecutor() as executor:
            for result in tqdm(executor.map(compute_slopes_single, task_list), total=len(task_list)):
                _ = result
                
        ###########################################################################################

        def compute_slopes_single_noscintillation(args):
            ii, jj, TARGET, n, run_name = args[:5]
            load_r0, load_d, new_r0, new_d = args[5:9]
            SHWFS_class = args[9:][0]
            SHWFS = SHWFS_class(n)

            Image = np.zeros((TARGET,n,n),dtype=float)

            for count in range(TARGET):
                filepath = f'/mt/cfai/pkongkaew/U_tot/ps_GL_r02dor20rytov01h05_{count:04d}.pkl'
                try:
                    with open(filepath, 'rb') as file:
                        load_phase = pickle.load(file)

                    phase = load_phase*((load_r0/load_d) * (new_d/new_r0))**(5./6.)
                    U = np.exp(1j*phase)

                    Image[count] = SHWFS.focus(U)
                except:
                    Image[count,:] = np.nan

            with open('/mt/cfai/pkongkaew/IGPhaseScreen/Image/circ-ap-noscintillation-Image_' + run_name + '.pkl', 'wb') as file:
                pickle.dump(Image, file)
                file.close()

            return

        # Create a task list for all valid (ii, jj, kk, ll) combinations

        load_r0 = r0_GL[3,0,0]
        load_d = 2*load_r0

        task_list_noscintillation = []
        for ii in range(II):
            for jj in range(JJ):
                new_r0 = r0[ii]#0.11
                new_dor0 = dor0[jj]#0.5
                new_d = new_dor0*new_r0
                run_name = 'r{:02d}dor{:02d}'.format(int(r0[ii]*100), int(dor0[jj]*10))
                task_list_noscintillation.append((ii, jj, TARGET, n, run_name,
                                                load_r0, load_d, new_r0, new_d,
                                                MWFS.Focus))


        from tqdm import tqdm


        with ProcessPoolExecutor() as executor:
            for result in tqdm(executor.map(compute_slopes_single_noscintillation, task_list_noscintillation), total=len(task_list_noscintillation)):
                _ = result
                

    ###########################################################################################






###############################################################################################################################



    
