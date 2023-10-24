
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
from my_WFSC import OneStepSHWFS
from my_WFSC import IgTilt
from mypropagationclass import BlindOneAngularPropAtm
from aotools.turbulence.infinitephasescreen import PhaseScreenVonKarman

###############################################################################



###############################################################################

# set up shwfs propagation here
# also any memory allocation

REPEAT = 2
REPEAT2 = 2

element = 10
N = 2**(np.arange(element)+1)
t_IG = np.zeros((N.shape[0], REPEAT), dtype=float)
t_FT = np.zeros((N.shape[0], REPEAT), dtype=float)
t_IPS = np.zeros((N.shape[0], REPEAT), dtype=float)
t_PROP = np.zeros((N.shape[0], REPEAT), dtype=float)

t_IG_mean = np.zeros((N.shape[0]), dtype=float)
t_FT_mean = np.zeros((N.shape[0]), dtype=float)
t_IPS_mean = np.zeros((N.shape[0]), dtype=float)
t_PROP_mean = np.zeros((N.shape[0]), dtype=float)

t_IG_error = np.zeros((N.shape[0]), dtype=float)
t_FT_error = np.zeros((N.shape[0]), dtype=float)
t_IPS_error = np.zeros((N.shape[0]), dtype=float)
t_PROP_error = np.zeros((N.shape[0]), dtype=float)

for power in tqdm(np.arange(element)+1):
    n = 2**power
    U = np.ones((n, n), dtype=complex)
    IG = IgTilt(n)
    FT = OneStepSHWFS(n)
    IPS = PhaseScreenVonKarman(n,0.1,0.1,20)
    PROP = BlindOneAngularPropAtm(500e-9,0.1,n,0.1**2/500e-9/2.)
    
    # for repeat in tqdm(np.arange(REPEAT)):
    #     t_IG[power - 1, repeat] = timeit.timeit(
    #         'IG.measure_with_noise(U,1000)',
    #         setup="from __main__ import U, IG", number=REPEAT2)/REPEAT2
    #     t_FT[power - 1, repeat] = timeit.timeit(
    #         'FT.measure_photon_noise(U,100)',
    #         setup="from __main__ import U, FT", number=REPEAT2)/REPEAT2
    #     t_IPS[power - 1, repeat] = timeit.timeit(
    #         'for i in np.arange(2*n): IPS.add_row()',
    #         setup="from __main__ import IPS, n, np", number=REPEAT2)/REPEAT2
    #     t_PROP[power - 1, repeat] = timeit.timeit(
    #         'PROP.prop(U)',
    #         setup="from __main__ import PROP, U", number=REPEAT2)/REPEAT2
    for repeat in tqdm(np.arange(REPEAT)):
        t_IG[power - 1, repeat] = timeit.timeit(
            'IG.measure(U)',
            setup="from __main__ import U, IG", number=REPEAT2)/REPEAT2
        t_FT[power - 1, repeat] = timeit.timeit(
            'FT.measure_no_noise_no_threshold(U)',
            setup="from __main__ import U, FT", number=REPEAT2)/REPEAT2
        t_IPS[power - 1, repeat] = timeit.timeit(
            'for i in np.arange(n): IPS.add_row()',
            setup="from __main__ import IPS, n, np", number=REPEAT2)/REPEAT2
        t_PROP[power - 1, repeat] = timeit.timeit(
            'PROP.prop(U)',
            setup="from __main__ import PROP, U", number=REPEAT2)/REPEAT2
    

t_IG_mean = np.mean(t_IG, axis=1)
t_FT_mean = np.mean(t_FT, axis=1)
t_IPS_mean = np.mean(t_IPS, axis=1)
t_PROP_mean = np.mean(t_PROP, axis=1)
t_IG_error = np.std(t_IG, ddof=1, axis=1)
t_FT_error = np.std(t_FT, ddof=1, axis=1)
t_IPS_error = np.std(t_IPS, ddof=1, axis=1)
t_PROP_error = np.std(t_PROP, ddof=1, axis=1)

t_IG_mean = np.array([108,123,111,122,177,281,630,2190,7920,36700])*1e-6
t_IG_error = np.array([0.209,11.1,0.293,1.68,22.5,33.5,1.75,266,8.22,2700])*1e-6
t_FT_mean = np.array([171,163,171,185,242,448,1390,5360,30600,187000])*1e-6
t_FT_error = np.array([21.5,21,21.3,23.8,29.9,13.3,1.68,40.7,2270,17300])*1e-6

plt.errorbar(N, t_IG_mean, yerr=t_IG_error, label='IG-Tilt', marker='o')
plt.errorbar(N, t_FT_mean, yerr=t_FT_error, label='FT-Tilt', marker='o')
# plt.errorbar(N, t_IPS_mean, yerr=t_IPS_error, label='Generate phase screen', marker='o')
# plt.errorbar(N, t_PROP_mean, yerr=t_PROP_error, label='Propagate phase screen', marker='o')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('Number of pixels across a screen')
plt.ylabel('time (s)')
# plt.savefig('timetrial4.png')
plt.show()
plt.clf()

# plt.errorbar(N,t_SHWFS/t_Ig,
#              yerr=np.sqrt((t_Ig_error/t_Ig)**2
#                           + (t_SHWFS_error/t_SHWFS)**2)*(t_SHWFS/t_Ig)
#              , marker='o')
plt.errorbar(N, t_FT_mean/t_IG_mean, yerr=(t_FT_mean/t_IG_mean)*np.sqrt((t_FT_error/t_FT_mean)**2 + (t_IG_error/t_IG_mean)**2), marker='o', label='FT/IG')

plt.xscale('log')
plt.legend()
plt.xlabel('Number of pixels across a screen')
plt.ylabel('FT time / IG time')
# plt.savefig('timegain4.png')
plt.show()
plt.clf()

# myvar = {'t_IG_mean' : t_IG_mean,
#          't_FT_mean' : t_FT_mean,
#          't_IPS_mean' : t_IPS_mean,
#          't_PROP_mean' : t_PROP_mean,
#          't_IG_error' : t_IG_error,
#          't_IPS_error' : t_IPS_error,
#          't_PROP_error' : t_PROP_error,
#          't_FT_error' : t_FT_error}

# with open('timing_out.pkl', 'wb') as file:
#     pickle.dump(myvar, file)
#     file.close()