
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
import time
from tqdm import tqdm
from my_WFSC import OneStepSHWFS
from my_WFSC import IgTilt
from mypropagationclass import BlindOneAngularPropAtm
from aotools.turbulence.infinitephasescreen import PhaseScreenVonKarman

###############################################################################



###############################################################################

# set up shwfs propagation here
# also any memory allocation

REPEAT = 1

t_IG = np.zeros((REPEAT), dtype=float)
t_FT = np.zeros((REPEAT), dtype=float)
t_IPS = np.zeros((REPEAT), dtype=float)
t_PROP = np.zeros((REPEAT), dtype=float)

t_IG_mean = 0
t_FT_mean = 0
t_IPS_mean = 0
t_PROP_mean = 0

t_IG_error = 0
t_FT_error = 0
t_IPS_error = 0
t_PROP_error = 0

power = 7

n = 2**power
U = np.ones((n, n), dtype=complex)
IG = IgTilt(n)
FT = OneStepSHWFS(n)
IPS = PhaseScreenVonKarman(n,0.1,0.1,20)
U = np.copy(IPS.scrn)
PROP = BlindOneAngularPropAtm(500e-9,0.1,n,0.1**2/500e-9/2.)

for repeat in np.arange(REPEAT):
    A = time.perf_counter()
    IG.measure(U)
    B = time.perf_counter()
    t_IG[repeat] = B - A
    
    A = time.perf_counter()
    FT.measure_no_noise_no_threshold(U)
    B = time.perf_counter()
    t_FT[repeat] = B - A
    
    A = time.perf_counter()
    for i in np.arange(n):
        IPS.add_row()
    B = time.perf_counter()
    t_IPS[repeat] = B - A
    
    A = time.perf_counter()
    PROP.prop(U)
    B = time.perf_counter()
    t_PROP[repeat] = B - A

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

with open('time_use_time2.pkl', 'wb') as file:
    pickle.dump(myvar, file)
    file.close()