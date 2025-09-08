# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:06:17 2024

@author: gqmc62
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:52:36 2023

test noise simulation

@author: gqmc62
"""

circularAperture_list = [False]
standard_threshold = 1e-2

for circularAperture in circularAperture_list:

    import numpy as np
    from matplotlib import pyplot as plt
    # from aotools.turbulence import infinitephasescreen
    if circularAperture == True:
        import my_WFSC as MWFS
        quit
    else:
        import MyWFS as MWFS

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

    with open('/mt/cfai/pkongkaew/IGPhaseScreen/Ultimate_input.pkl', 'rb') as file:
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



    RON_list = np.array([0.1,0.3,1,3,10],dtype=float)
    NN = len(RON_list)

    # Ig = MWFS.IgTilt(n)
    SHWFS = MWFS.SHWFS_from_image_adaptive_ROI(16,n)

    N_phot_list = np.array([100,1000,10000,100000,1000000],dtype=float)
    MM=len(N_phot_list)

    Threshold_list = np.array([0,0.0001,0.1,0.3,0.5],dtype=float)
    OO=len(Threshold_list)

    #RANDOM_NOISE_TRIAL = 1024

    TARGET = 400
    TARGET2 = TARGET
    RANDOM_INDEX = np.arange(1024)
    np.random.shuffle(RANDOM_INDEX)

    SHWFSTilt = np.zeros((2), dtype=float)
    NoNoiseSHWFSTilt = np.zeros((2), dtype=float)
    Noise = np.zeros((MM, NN, OO, TARGET, 2, TARGET2), dtype=float)

    U = np.zeros((n, n), dtype=complex)

    # if not os.path.isdir('noise'):
    #     os.makedirs('noise')

    #TIME_START = time.time()



    Pii, Pjj, Pkk, Pll = np.where(POSSIBLE==True)
    for possible in np.arange(len(Pii)):
        ii = Pii[possible]
        jj = Pjj[possible]
        kk = Pkk[possible]
        ll = Pll[possible]
        #if dor0[jj] == 0.5:
        #    continue

        if sample_ii != ii:
            continue
        if sample_jj != jj:
            continue
        if sample_kk != kk:
            continue

        if sample_ll != ll:
            continue

        run_name = 'r{:02d}dor{:02d}rytov{:02d}h{:02d}'.format(
                            int(r0[ii]*100), int(dor0[jj]*10), int(Rytov[kk]*100), int(H_HA[ll]/1000))


        #print(ii,jj,kk,ll)
        #if jj != 0 :
        #    continue
        # print('/mt/cfai/pkongkaew/IGPhaseScreen/Image/Image_' + run_name + '.pkl')
        # print(os.path.exists('/mt/cfai/pkongkaew/IGPhaseScreen/Image/Image_' + run_name + '.pkl'))
        if os.path.exists('/mt/cfai/pkongkaew/IGPhaseScreen/Image/sq-ap-Image_' + run_name + '.pkl'):
            with open('/mt/cfai/pkongkaew/IGPhaseScreen/Image/sq-ap-Image_' + run_name + '.pkl','rb') as file:
                Image = pickle.load(file)
            for count in np.arange(TARGET):
                if (np.isnan(Image[RANDOM_INDEX[count]])).sum() == 0:
                    NoNoiseSHWFSTilt[:] = SHWFS.measure_no_noise_with_threshold(Image[RANDOM_INDEX[count]],standard_threshold)
                    # if count == 0:
                    #     print(NoNoiseSHWFSTilt[ll,count, :])

                    for mm in np.arange(MM):
                        Nphot = N_phot_list[mm]
                        for nn in np.arange(NN):
                            RON = RON_list[nn]
                            for oo in np.arange(OO):
                                Threshold = Threshold_list[oo]
                                #print(ii,jj,kk,ll,mm,nn,oo)


                                #IgTilt[ii, jj, kk, ll, mm, count, : ] = Ig.measure_with_noise(U, Nphot)
                                #for trandom_noise_trial in np.arange(RANDOM_NOISE_TRIAL):

                                for count2 in np.arange(TARGET2):

                                    SHWFSTilt[:] = SHWFS.measure_with_noise_with_threshold(Image[RANDOM_INDEX[count]], Nphot, RON, Threshold)
                                    Noise[mm,nn,oo,count,:,count2] = SHWFSTilt - NoNoiseSHWFSTilt
                else:
                    Noise[:,:,:,count, :,:] = np.nan
        else:
            Noise[:] = np.nan
                
        myvar = {'N_phot_list' : N_phot_list,
                'RON_list' : RON_list,
                'Threshold_list' : Threshold_list,
                #'IgTilt' : IgTilt,
                # 'SHWFSTilt' : SHWFSTilt,
                # 'NoNoiseSHWFSTilt' : NoNoiseSHWFSTilt,
                'Noise' : Noise,
                'RANDOM_INDEX' : RANDOM_INDEX}

        os.chdir('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/Slopes')

        with open('sq-ap-no-ft-bias-SHWFSNoiseScintillation{:d}{:d}{:d}{:d}.pkl'.format(sample_ii,sample_jj,sample_kk,sample_ll), 'wb') as file:
            pickle.dump(myvar, file)
            file.close()

###################################################################################################################


circularAperture_list = [False]
standard_threshold = 0.0

for circularAperture in circularAperture_list:

    import numpy as np
    from matplotlib import pyplot as plt
    # from aotools.turbulence import infinitephasescreen
    if circularAperture == True:
        import my_WFSC as MWFS
        quit
    else:
        import MyWFS as MWFS

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

    with open('/mt/cfai/pkongkaew/IGPhaseScreen/Ultimate_input.pkl', 'rb') as file:
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



    RON_list = np.array([0.1,0.3,1,3,10],dtype=float)
    NN = len(RON_list)

    # Ig = MWFS.IgTilt(n)
    SHWFS = MWFS.SHWFS_from_image_adaptive_ROI(16,n)

    N_phot_list = np.array([100,1000,10000,100000,1000000],dtype=float)
    MM=len(N_phot_list)

    Threshold_list = np.array([0,0.0001,0.1,0.3,0.5],dtype=float)
    OO=len(Threshold_list)

    #RANDOM_NOISE_TRIAL = 1024

    TARGET = 400
    TARGET2 = TARGET
    RANDOM_INDEX = np.arange(1024)
    np.random.shuffle(RANDOM_INDEX)

    SHWFSTilt = np.zeros((2), dtype=float)
    NoNoiseSHWFSTilt = np.zeros((2), dtype=float)
    Noise = np.zeros((MM, NN, OO, TARGET, 2, TARGET2), dtype=float)


    U = np.zeros((n, n), dtype=complex)

    # if not os.path.isdir('noise'):
    #     os.makedirs('noise')

    #TIME_START = time.time()



    Pii, Pjj, Pkk, Pll = np.where(POSSIBLE==True)
    for possible in np.arange(len(Pii)):
        ii = Pii[possible]
        jj = Pjj[possible]
        kk = Pkk[possible]
        ll = Pll[possible]
        #if dor0[jj] == 0.5:
        #    continue

        if sample_ii != ii:
            continue
        if sample_jj != jj:
            continue
        if sample_kk != kk:
            continue

        if sample_ll != ll:
            continue

        run_name = 'r{:02d}dor{:02d}rytov{:02d}h{:02d}'.format(
                            int(r0[ii]*100), int(dor0[jj]*10), int(Rytov[kk]*100), int(H_HA[ll]/1000))


        #print(ii,jj,kk,ll)
        #if jj != 0 :
        #    continue
        # print('/mt/cfai/pkongkaew/IGPhaseScreen/Image/Image_' + run_name + '.pkl')
        # print(os.path.exists('/mt/cfai/pkongkaew/IGPhaseScreen/Image/Image_' + run_name + '.pkl'))
        if os.path.exists('/mt/cfai/pkongkaew/IGPhaseScreen/Image/sq-ap-Image_' + run_name + '.pkl'):
            with open('/mt/cfai/pkongkaew/IGPhaseScreen/Image/sq-ap-Image_' + run_name + '.pkl','rb') as file:
                Image = pickle.load(file)
            for count in np.arange(TARGET):
                if (np.isnan(Image[RANDOM_INDEX[count]])).sum() == 0:
                    NoNoiseSHWFSTilt[:] = SHWFS.measure_no_noise_with_threshold(Image[RANDOM_INDEX[count]],standard_threshold)
                    # if count == 0:
                    #     print(NoNoiseSHWFSTilt[ll,count, :])

                    for mm in np.arange(MM):
                        Nphot = N_phot_list[mm]
                        for nn in np.arange(NN):
                            RON = RON_list[nn]
                            for oo in np.arange(OO):
                                Threshold = Threshold_list[oo]
                                #print(ii,jj,kk,ll,mm,nn,oo)


                                #IgTilt[ii, jj, kk, ll, mm, count, : ] = Ig.measure_with_noise(U, Nphot)
                                #for trandom_noise_trial in np.arange(RANDOM_NOISE_TRIAL):

                                for count2 in np.arange(TARGET2):

                                    SHWFSTilt[:] = SHWFS.measure_with_noise_with_threshold(Image[RANDOM_INDEX[count]], Nphot, RON, Threshold)
                                    Noise[mm,nn,oo,count,:,count2] = SHWFSTilt - NoNoiseSHWFSTilt
                else:
                    Noise[:,:,:,count,:,:] = np.nan
        else:
            Noise[:] = np.nan
        
        myvar = {'N_phot_list' : N_phot_list,
                'RON_list' : RON_list,
                'Threshold_list' : Threshold_list,
                #'IgTilt' : IgTilt,
                # 'SHWFSTilt' : SHWFSTilt,
                # 'NoNoiseSHWFSTilt' : NoNoiseSHWFSTilt,
                'Noise' : Noise,
                'RANDOM_INDEX' : RANDOM_INDEX}

        os.chdir('/mt/cfai/pkongkaew/NoiseSHWFSSlopes/Slopes')

        with open('sq-ap-og-SHWFSNoiseScintillation{:d}{:d}{:d}{:d}.pkl'.format(sample_ii,sample_jj,sample_kk,sample_ll), 'wb') as file:
            pickle.dump(myvar, file)
            file.close()
