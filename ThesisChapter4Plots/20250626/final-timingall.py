
###############################################################################
##########################################################################
"""
█▀▀█░█░▒█░▀█▀░▀█▀░░▀░░█░░░█░█▀▀▄░▀█▀░░░▒█░▄▀░▄▀▀▄░█▀▀▄░█▀▀▀░█░▄░█▀▀▄░█▀▀░█░░░█
█▄▄█░█░▒█░░█░░░█░░░█▀░▀▄█▄▀░█▄▄█░░█░░░░▒█▀▄░░█░░█░█░▒█░█░▀▄░█▀▄░█▄▄█░█▀▀░▀▄█▄▀
█░░░░░▀▀▀░░▀░░░▀░░▀▀▀░░▀░▀░░▀░░▀░░▀░░░░▒█░▒█░░▀▀░░▀░░▀░▀▀▀▀░▀░▀░▀░░▀░▀▀▀░░▀░▀░
"""
##########################################################################
###############################################################################

circularAperture = False

import numpy as np
import pickle
from matplotlib import pyplot as plt
import pyfftw
import time
from tqdm import tqdm
if circularAperture == True:
    from my_WFSC import FastestFTSHWFS
    from my_WFSC import IgTilt
else:
    from MyWFS import FastestFTSHWFS
    from MyWFS import IgTilt
from mypropagationclass import BlindOneAngularPropAtm
from aotools.turbulence.infinitephasescreen import PhaseScreenVonKarman

###############################################################################



###############################################################################

# set up shwfs propagation here
# also any memory allocation

REPEAT = 100
N_power = 10

t_IG = np.zeros((N_power,REPEAT), dtype=float)
t_FT = np.zeros((N_power,REPEAT), dtype=float)
t_IPS = np.zeros((N_power,REPEAT), dtype=float)
t_PROP = np.zeros((N_power,REPEAT), dtype=float)

t_IG_mean = 0
t_FT_mean = 0
t_IPS_mean = 0
t_PROP_mean = 0

t_IG_error = 0
t_FT_error = 0
t_IPS_error = 0
t_PROP_error = 0

power = np.arange(N_power) + 1

N_list = 2**power

for i_n in range(len(N_list)):
    n = N_list[i_n]
    U = np.ones((n, n), dtype=complex)
    IG = IgTilt(n)
    FT = FastestFTSHWFS(n)
    IPS = PhaseScreenVonKarman(n,0.1,0.1,20)
    U = np.copy(IPS.scrn)
    PROP = BlindOneAngularPropAtm(500e-9,0.1,n,0.1**2/500e-9/2.)

    for repeat in tqdm(np.arange(REPEAT)):
        A = time.perf_counter()
        IG.measure(U)
        B = time.perf_counter()
        t_IG[i_n,repeat] = B - A
        
        A = time.perf_counter()
        FT.measure_no_noise_no_threshold(U)
        B = time.perf_counter()
        t_FT[i_n,repeat] = B - A
        
        A = time.perf_counter()
        for i in np.arange(n):
            IPS.add_row()
        B = time.perf_counter()
        t_IPS[i_n,repeat] = B - A
        
        A = time.perf_counter()
        PROP.prop(U)
        B = time.perf_counter()
        t_PROP[i_n,repeat] = B - A

t_IG_mean = np.mean(t_IG, axis=-1)
t_FT_mean = np.mean(t_FT, axis=-1)
t_IPS_mean = np.mean(t_IPS, axis=-1)
t_PROP_mean = np.mean(t_PROP, axis=-1)
t_IG_error = np.std(t_IG, ddof=1, axis=-1)
t_FT_error = np.std(t_FT, ddof=1, axis=-1)
t_IPS_error = np.std(t_IPS, ddof=1, axis=-1)
t_PROP_error = np.std(t_PROP, ddof=1, axis=-1)

myvar = {'power' : power,
        't_IG_mean' : t_IG_mean,
        't_FT_mean' : t_FT_mean,
        't_IPS_mean' : t_IPS_mean,
        't_PROP_mean' : t_PROP_mean,
        't_IG_error' : t_IG_error,
        't_IPS_error' : t_IPS_error,
        't_PROP_error' : t_PROP_error,
        't_FT_error' : t_FT_error}

with open('time_use_time_all.pkl', 'wb') as file:
    pickle.dump(myvar, file)
    file.close()