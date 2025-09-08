
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

n_cog = 17
threshold_list = [0,1e-4,1e-3,1e-2,1e-1]

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

    def compute_slopes_single(args):
        ii, jj, kk, ll, run_name, TARGET, n, threshold_list = args[:8]
        GWFS_class, IGWFS_class, SHWFS_class = args[8:]

        g_slope = np.zeros((TARGET, 2))
        ig_slope = np.zeros((TARGET, 2))
        shwfs_slope = np.zeros((len(threshold_list), TARGET, 2))

        GWFS = GWFS_class(n)
        IGWFS = IGWFS_class(n)
        SHWFS = SHWFS_class(n)

        for count in range(TARGET):
            filepath = f'/mt/cfai/pkongkaew/U_tot/U_tot/U_tot_{run_name}_{count:04d}.pkl'
            try:
                with open(filepath, 'rb') as file:
                    U = pickle.load(file)
                g_slope[count] = GWFS.measure(U)
                ig_slope[count] = IGWFS.measure(U)
                for i_threshold in range(len(threshold_list)):
                    threshold = threshold_list[i_threshold]
                    shwfs_slope[i_threshold,count] = SHWFS.measure_no_noise_with_threshold(U,threshold)

            except:
                g_slope[count] = np.nan
                ig_slope[count] = np.nan
                shwfs_slope[:count] = np.nan

            
        return (ii, jj, kk, ll, g_slope, ig_slope, shwfs_slope)

    # Create a task list for all valid (ii, jj, kk, ll) combinations
    task_list = []
    for ii, jj, kk, ll in zip(*np.where(POSSIBLE == True)):
        run_name = 'r{:02d}dor{:02d}rytov{:02d}h{:02d}'.format(
            int(r0[ii]*100), int(dor0[jj]*10), int(Rytov[kk]*100), int(H_HA[ll]/1000))
        task_list.append((ii, jj, kk, ll, run_name, TARGET, n, threshold_list, MWFS.WrappedGTilt, MWFS.IgTilt, MWFS.OneStepSHWFS))


    from tqdm import tqdm

    # Preallocate results
    g_SLOPE = np.zeros((II, JJ, KK, LL, TARGET, 2), dtype=float)
    Ig_SLOPE = np.zeros_like(g_SLOPE)
    SHWFS_SLOPE = np.zeros((II, JJ, KK, LL, len(threshold_list), TARGET, 2), dtype=float)

    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(compute_slopes_single, task_list), total=len(task_list)):
            ii, jj, kk, ll, g, ig, sh = result
            g_SLOPE[ii, jj, kk, ll] = g
            Ig_SLOPE[ii, jj, kk, ll] = ig
            SHWFS_SLOPE[ii, jj, kk, ll] = sh

    ###########################################################################################

    def compute_slopes_single_noscintillation(args):
        ii, jj, TARGET, n, threshold_list = args[:5]
        load_r0, load_d, new_r0, new_d = args[5:9]
        GWFS_class, IGWFS_class, SHWFS_class = args[9:]

        g_slope = np.zeros((TARGET, 2))
        ig_slope = np.zeros((TARGET, 2))
        shwfs_slope = np.zeros((len(threshold_list),TARGET, 2))

        GWFS = GWFS_class(n)
        IGWFS = IGWFS_class(n)
        SHWFS = SHWFS_class(n)

        for count in range(TARGET):
            filepath = f'/mt/cfai/pkongkaew/U_tot/ps_GL/ps_GL_r02dor20rytov01h05_{count:04d}.pkl'
            try:
                with open(filepath, 'rb') as file:
                    load_phase = pickle.load(file)

                phase = load_phase*((load_r0/load_d) * (new_d/new_r0))**(5./6.)
                U = np.exp(1j*phase)

                g_slope[count] = GWFS.measure(U)
                ig_slope[count] = IGWFS.measure(U)
                for i_threshold in range(len(threshold_list)):
                    threshold = threshold_list[i_threshold]
                    shwfs_slope[i_threshold,count] = SHWFS.measure_no_noise_with_threshold(U,threshold)

            except:
                g_slope[count] = np.nan
                ig_slope[count] = np.nan
                shwfs_slope[:,count] = np.nan

        return (ii, jj, g_slope, ig_slope, shwfs_slope)

    # Create a task list for all valid (ii, jj, kk, ll) combinations

    load_r0 = r0_GL[3,0,0]
    load_d = 2*load_r0

    task_list_noscintillation = []
    for ii in range(II):
        for jj in range(JJ):
            new_r0 = r0[ii]#0.11
            new_dor0 = dor0[jj]#0.5
            new_d = new_dor0*new_r0
            task_list_noscintillation.append((ii, jj, TARGET, n, threshold_list,
                                              load_r0, load_d, new_r0, new_d,
                                              MWFS.WrappedGTilt, MWFS.IgTilt, MWFS.OneStepSHWFS))


    from tqdm import tqdm

    # Preallocate results
    g_SLOPE_noscintillation = np.zeros((II, JJ, TARGET, 2), dtype=float)
    Ig_SLOPE_noscintillation = np.zeros_like(g_SLOPE_noscintillation)
    SHWFS_SLOPE_noscintillation = np.zeros((II, JJ, len(threshold_list), TARGET, 2), dtype=float)

    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(compute_slopes_single_noscintillation, task_list_noscintillation), total=len(task_list_noscintillation)):
            ii, jj, g, ig, sh = result
            g_SLOPE_noscintillation[ii, jj] = g
            Ig_SLOPE_noscintillation[ii, jj] = ig
            SHWFS_SLOPE_noscintillation[ii, jj] = sh

    ###########################################################################################

    if circularAperture == True:

        with open('/mt/cfai/pkongkaew/c4data/NoiselessCircularPupilSlopes-og.pkl', 'wb') as file:
            pickle.dump({
                'threshold_list' : threshold_list,
                'n_cog' : n_cog,
                'g_SLOPE': g_SLOPE,
                'Ig_SLOPE': Ig_SLOPE,
                'SHWFS_SLOPE': SHWFS_SLOPE,
                'g_SLOPE_noscintillation': g_SLOPE_noscintillation,
                'Ig_SLOPE_noscintillation': Ig_SLOPE_noscintillation,
                'SHWFS_SLOPE_noscintillation': SHWFS_SLOPE_noscintillation
            }, file)

    else:
        with open('/mt/cfai/pkongkaew/c4data/NoiselessSquarePupilSlopes-og.pkl', 'wb') as file:
            pickle.dump({
                'threshold_list' : threshold_list,
                'n_cog' : n_cog,
                'g_SLOPE': g_SLOPE,
                'Ig_SLOPE': Ig_SLOPE,
                'SHWFS_SLOPE': SHWFS_SLOPE,
                'g_SLOPE_noscintillation': g_SLOPE_noscintillation,
                'Ig_SLOPE_noscintillation': Ig_SLOPE_noscintillation,
                'SHWFS_SLOPE_noscintillation': SHWFS_SLOPE_noscintillation
            }, file)




###########################################################################################################
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

    def compute_slopes_single(args):
        ii, jj, kk, ll, run_name, TARGET, n, n_cog, threshold_list = args[:9]
        GWFS_class, IGWFS_class, SHWFS_class = args[9:]

        g_slope = np.zeros((TARGET, 2))
        ig_slope = np.zeros((TARGET, 2))
        shwfs_slope = np.zeros((len(threshold_list),TARGET, 2))

        GWFS = GWFS_class(n)
        IGWFS = IGWFS_class(n)
        SHWFS = SHWFS_class(n_cog,n)

        for count in range(TARGET):
            filepath = f'/mt/cfai/pkongkaew/U_tot/U_tot/U_tot_{run_name}_{count:04d}.pkl'
            try:
                with open(filepath, 'rb') as file:
                    U = pickle.load(file)

                g_slope[count] = GWFS.measure(U)
                ig_slope[count] = IGWFS.measure(U)
                for i_threshold in range(len(threshold_list)):
                    threshold = threshold_list[i_threshold]
                    shwfs_slope[i_threshold,count] = SHWFS.measure_no_noise_with_threshold(U,threshold)

            except:
                g_slope[count] = np.nan
                ig_slope[count] = np.nan
                shwfs_slope[:,count] = np.nan

        return (ii, jj, kk, ll, g_slope, ig_slope, shwfs_slope)

    # Create a task list for all valid (ii, jj, kk, ll) combinations
    task_list = []
    for ii, jj, kk, ll in zip(*np.where(POSSIBLE == True)):
        run_name = 'r{:02d}dor{:02d}rytov{:02d}h{:02d}'.format(
            int(r0[ii]*100), int(dor0[jj]*10), int(Rytov[kk]*100), int(H_HA[ll]/1000))
        task_list.append((ii, jj, kk, ll, run_name, TARGET, n, n_cog, threshold_list, MWFS.WrappedGTilt, MWFS.IgTilt, MWFS.SHWFS_from_U_adaptive_ROI))


    from tqdm import tqdm

    # Preallocate results
    g_SLOPE = np.zeros((II, JJ, KK, LL, TARGET, 2), dtype=float)
    Ig_SLOPE = np.zeros_like(g_SLOPE)
    SHWFS_SLOPE = np.zeros((II,JJ,KK,LL,len(threshold_list),TARGET,2),dtype=float)

    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(compute_slopes_single, task_list), total=len(task_list)):
            ii, jj, kk, ll, g, ig, sh = result
            g_SLOPE[ii, jj, kk, ll] = g
            Ig_SLOPE[ii, jj, kk, ll] = ig
            SHWFS_SLOPE[ii, jj, kk, ll] = sh

    ###########################################################################################

    def compute_slopes_single_noscintillation(args):
        ii, jj, TARGET, n, n_cog, threshold_list = args[:6]
        load_r0, load_d, new_r0, new_d = args[6:10]
        GWFS_class, IGWFS_class, SHWFS_class = args[10:]

        g_slope = np.zeros((TARGET, 2))
        ig_slope = np.zeros((TARGET, 2))
        shwfs_slope = np.zeros((len(threshold_list),TARGET, 2))

        GWFS = GWFS_class(n)
        IGWFS = IGWFS_class(n)
        SHWFS = SHWFS_class(n_cog,n)

        for count in range(TARGET):
            filepath = f'/mt/cfai/pkongkaew/U_tot/ps_GL/ps_GL_r02dor20rytov01h05_{count:04d}.pkl'
            try:
                with open(filepath, 'rb') as file:
                    load_phase = pickle.load(file)

                phase = load_phase*((load_r0/load_d) * (new_d/new_r0))**(5./6.)
                U = np.exp(1j*phase)

                g_slope[count] = GWFS.measure(U)
                ig_slope[count] = IGWFS.measure(U)
                for i_threshold in range(len(threshold_list)):
                    threshold = threshold_list[i_threshold]
                    shwfs_slope[i_threshold,count] = SHWFS.measure_no_noise_with_threshold(U,threshold)

            except:
                g_slope[count] = np.nan
                ig_slope[count] = np.nan
                shwfs_slope[:,count] = np.nan

        return (ii, jj, g_slope, ig_slope, shwfs_slope)

    # Create a task list for all valid (ii, jj, kk, ll) combinations

    load_r0 = r0_GL[3,0,0]
    load_d = 2*load_r0

    task_list_noscintillation = []
    for ii in range(II):
        for jj in range(JJ):
            new_r0 = r0[ii]#0.11
            new_dor0 = dor0[jj]#0.5
            new_d = new_dor0*new_r0
            task_list_noscintillation.append((ii, jj, TARGET, n, n_cog, threshold_list,
                                              load_r0, load_d, new_r0, new_d,
                                              MWFS.WrappedGTilt, MWFS.IgTilt, MWFS.SHWFS_from_U_adaptive_ROI))


    from tqdm import tqdm

    # Preallocate results
    g_SLOPE_noscintillation = np.zeros((II, JJ, TARGET, 2), dtype=float)
    Ig_SLOPE_noscintillation = np.zeros_like(g_SLOPE_noscintillation)
    SHWFS_SLOPE_noscintillation = np.zeros((II,JJ,len(threshold_list),TARGET,2),dtype=float)

    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(compute_slopes_single_noscintillation, task_list_noscintillation), total=len(task_list_noscintillation)):
            ii, jj, g, ig, sh = result
            g_SLOPE_noscintillation[ii, jj] = g
            Ig_SLOPE_noscintillation[ii, jj] = ig
            SHWFS_SLOPE_noscintillation[ii, jj] = sh

    ###########################################################################################

    if circularAperture == True:

        with open('/mt/cfai/pkongkaew/c4data/NoiselessCircularPupilSlopes-no-ft-bias.pkl', 'wb') as file:
            pickle.dump({
                'threshold_list' : threshold_list,
                'n_cog' : n_cog,
                'g_SLOPE': g_SLOPE,
                'Ig_SLOPE': Ig_SLOPE,
                'SHWFS_SLOPE': SHWFS_SLOPE,
                'g_SLOPE_noscintillation': g_SLOPE_noscintillation,
                'Ig_SLOPE_noscintillation': Ig_SLOPE_noscintillation,
                'SHWFS_SLOPE_noscintillation': SHWFS_SLOPE_noscintillation
            }, file)

    else:
        with open('/mt/cfai/pkongkaew/c4data/NoiselessSquarePupilSlopes-no-ft-bias.pkl', 'wb') as file:
            pickle.dump({
                'threshold_list' : threshold_list,
                'n_cog' : n_cog,
                'g_SLOPE': g_SLOPE,
                'Ig_SLOPE': Ig_SLOPE,
                'SHWFS_SLOPE': SHWFS_SLOPE,
                'g_SLOPE_noscintillation': g_SLOPE_noscintillation,
                'Ig_SLOPE_noscintillation': Ig_SLOPE_noscintillation,
                'SHWFS_SLOPE_noscintillation': SHWFS_SLOPE_noscintillation
            }, file)