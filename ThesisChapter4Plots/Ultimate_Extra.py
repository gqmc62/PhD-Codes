
###############################################################################
##########################################################################
"""
█▀▀█░█░▒█░▀█▀░▀█▀░░▀░░█░░░█░█▀▀▄░▀█▀░░░▒█░▄▀░▄▀▀▄░█▀▀▄░█▀▀▀░█░▄░█▀▀▄░█▀▀░█░░░█
█▄▄█░█░▒█░░█░░░█░░░█▀░▀▄█▄▀░█▄▄█░░█░░░░▒█▀▄░░█░░█░█░▒█░█░▀▄░█▀▄░█▄▄█░█▀▀░▀▄█▄▀
█░░░░░▀▀▀░░▀░░░▀░░▀▀▀░░▀░▀░░▀░░▀░░▀░░░░▒█░▒█░░▀▀░░▀░░▀░▀▀▀▀░▀░▀░▀░░▀░▀▀▀░░▀░▀░
"""
##########################################################################
###############################################################################

# imports
import numpy as np
from matplotlib import pyplot as plt
import pickle
import pyfftw
from aotools.turbulence.phasescreen import ft_sh_phase_screen
from tqdm import tqdm
import time
import os

# TIME_START = time.time()

# functions
def distribute_coherent_length(r0, Rytov, HA, wvl, zenith_angle):
    if Rytov == 0:
        return r0, np.inf

    k = 2*np.pi/wvl
    c2ndz_tot = r0**(-5./3.)/(0.423*k**2/np.cos(zenith_angle))
    c2ndz_HA = Rytov / (0.5631 * k**(7./6.)
                        / (np.cos(zenith_angle))**(11./6.)
                        * HA**(5./6.))
    if c2ndz_HA >= c2ndz_tot :
        return np.NaN, np.NaN

    c2ndz_GL = c2ndz_tot - c2ndz_HA
    r0_GL = (0.423*k**2/np.cos(zenith_angle)*c2ndz_GL)**(-3./5.)
    r0_HA = (0.423*k**2/np.cos(zenith_angle)*c2ndz_HA)**(-3./5.)

    return r0_GL, r0_HA

def crop2(A, pad_factor=2):
    NN = A.shape[0]
    edge_low = int(NN*((pad_factor - 1)/pad_factor)/2)
    edge_high = int(NN*((pad_factor + 1)/pad_factor)/2)

    return A[edge_low : edge_high, edge_low : edge_high]

def zero_pad2(A, pad_factor=2):
    add_each_side = (pad_factor - 1)/2
    return np.pad(A, (int(A.shape[0]*add_each_side),
                      int(A.shape[1]*add_each_side)),
                  'constant', constant_values=0.)

###############################################################################
# simulations initialization #############################################
###############################################################################

# independent parameters

TARGET = 64*64
SCREEN_LIMIT = int(2**16)

n = int(64)
oversize = int(2)
L0_GL = float(20)
L0_HA = L0_GL
l0_GL = float(1.5e-10)
l0_HA = l0_GL
wvl = float(500e-9)
zenith_angle = float(0)

###############################################################################

# input conditions

r0 = np.array([11, 8, 5, 2], dtype=float)*1e-2              # II
dor0 = np.array([0.5, 1, 2, 3],dtype=float)#np.array([0.5, 1, 2, 3], dtype=float)                # JJ
Rytov = np.array([0.01, 0.03, 0.1, 0.3], dtype=float)       # KK
H_HA = np.array([5, 10, 15, 20], dtype=float)*1e3           # LL

###############################################################################

# Common indices ranges and dimension. Easy to keep track of dependencies
II = len(r0)
JJ = len(dor0)
KK = len(Rytov)
LL = len(H_HA)

# dependent simulation parameters

wavenumber = 2*np.pi/wvl
N_MAX = int(SCREEN_LIMIT/oversize / 4.)

d = np.zeros((II, JJ), dtype=float)                         # II, JJ
delta = np.zeros((II, JJ), dtype=float)                     # II, JJ

r0_GL = np.zeros((II, KK, LL), dtype=float)                 # II, KK, LL
r0_HA = np.zeros((II, KK, LL), dtype=float)                 # II, KK, LL

N = np.zeros((II, JJ, LL), dtype=int)                       # II, JJ, LL
num_lenslet = np.zeros((II, JJ, LL), dtype=int)             # II, JJ, LL
REPEAT = np.zeros((II, JJ, LL), dtype=int)                  # II, JJ, LL

POSSIBLE_SCREEN = np.ones_like(r0_GL, dtype=bool)           # II, KK, LL
POSSIBLE_PROP = np.ones((II, JJ, LL), dtype=bool)          # II, JJ, LL
POSSIBLE_SAMPLE = np.ones((II, JJ, KK, LL), dtype=bool)     # II, JJ, KK, LL

POSSIBLE = np.zeros((II, JJ, KK, LL), dtype=bool)           # II, JJ, KK, LL


###############################################################################

for ii in np.arange(II):
    for jj in np.arange(JJ):
        d[ii, jj] = dor0[jj]*r0[ii]

delta = d/n

# distribute r0
# impossible r0 distribution is ignored
for ii in np.arange(II):
    for kk in np.arange(KK):
        for ll in np.arange(LL):
            r0_GL[ii, kk, ll], r0_HA[ii, kk, ll] = distribute_coherent_length(
                r0[ii], Rytov[kk], H_HA[ll], wvl, zenith_angle)
            if (np.isnan(r0_GL[ii, kk, ll])
                or np.isnan(r0_HA[ii, kk, ll])
                or np.isinf(r0_GL[ii, kk, ll])
                or np.isinf(r0_HA[ii, kk, ll])):
                POSSIBLE_SCREEN[ii, kk, ll] = False
                # the inf r0 is actually generateable but
                # it should have been generate in a one layer scheme instead
                # so NO!!
                
# screens must be well sampled
for ii in range(II):
    for jj in range(JJ):
        for kk in range(KK):
            for ll in range(LL):
                if POSSIBLE_SCREEN[ii, jj, ll]:
                    if ((r0_GL[ii, kk, ll]/delta[ii, jj] <= 16)
                        or (r0_HA[ii, kk, ll]/delta[ii, jj] <= 16)):
                        POSSIBLE_SAMPLE[ii, jj, kk, ll] = False
                              
###############################################################################
# Check possiblity and validity to run

# calculating how big a HA screen is required.
# note this is for free atmosphere propagation

for ii in np.arange(II):
    for jj in np.arange(JJ):
        for ll in np.arange(LL):
            # print(ii,jj,ll)
            waste_px = (wvl*H_HA[ll])/(delta[ii, jj]**2)
            available_px1 = np.floor(N_MAX - waste_px)
            # print(available_px1)
            
            # the available size cannot accommodate enough space
            # for 'contamination'
            if available_px1 <= 0:
                REPEAT[ii, jj, ll] = 0
                num_lenslet[ii, jj, ll] = 0
                N[ii, jj, ll] = 0
                # print(ii, jj, ll)
                # print(waste_px)
                # print(REPEAT[ii, jj, ll], num_lenslet[ii, jj, ll])
                # print(num_lenslet[ii, jj, ll])
                # print(N[ii, jj, ll])
            else:
                # the required size is reached before going full max memomory
                if available_px1 >= N_MAX/2:
                    num_lenslet[ii, jj, ll] = int(2**np.ceil(
                        np.log(waste_px*2)/np.log(2))
                                                  /n/2)
                    REPEAT[ii, jj, ll] = int(np.ceil(
                        float(TARGET) / float(num_lenslet[ii, jj, ll]**2)))
                    N[ii, jj, ll] = num_lenslet[ii, jj, ll] * n
                    # print(ii, jj, ll)
                    # print(waste_px)
                    # print(REPEAT[ii, jj, ll], num_lenslet[ii, jj, ll])
                    # print(N[ii, jj, ll])
                    # print(num_lenslet[ii, jj, ll])
                else:
                    # even going at the max memory,
                    # it cannot acheive optimization
                    # however, it still yield some reliable results
                    # within a limited area
                    if available_px1 <= N_MAX/2:
                        # print(ii, jj, ll)
                        num_lenslet[ii, jj, ll] = available_px1//n
                        REPEAT[ii, jj, ll] = int(np.ceil(
                            float(TARGET)/float(num_lenslet[ii, jj, ll]**2)))
                        N[ii, jj, ll] = N_MAX/2
                        # print(ii, jj, ll)
                        # print(waste_px)
                        # print(REPEAT[ii, jj, ll], num_lenslet[ii, jj, ll])
                        # print(N[ii, jj, ll])
                        # print(num_lenslet[ii, jj, ll])
                    else:
                        print('this is serious error')

POSSIBLE_PROP = (num_lenslet != 0)

for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                POSSIBLE[ii, jj, kk, ll] = (POSSIBLE_PROP[ii, jj, ll]
                                            * POSSIBLE_SCREEN[ii, kk, ll]
                                            * POSSIBLE_SAMPLE[ii, jj, kk, ll])
                
del waste_px, available_px1

###############################################################################
# saving all input parameters before crazy simulation actually runs

myvar = {'TARGET' : TARGET,
         'SCREEN_LIMIT' : SCREEN_LIMIT,
         'n' : n,
         'oversize' : oversize,
         'L0_GL' : L0_GL,
         'L0_HA' : L0_HA,
         'l0_GL' : l0_GL,
         'l0_HA' : l0_HA,
         'wvl' : wvl,
         'wavenumber' : wavenumber,
         'zenith_angle' : zenith_angle,
         
         'r0' : r0,
         'dor0' : dor0,
         'Rytov' : Rytov,
         'H_HA' : H_HA,
         
         
         'II' : II,
         'JJ' : JJ,
         'KK' : KK,
         'LL' : LL,
         
         'wavenumber' : wavenumber,
         'N_MAX' : N_MAX,
         
         'd' : d,
         'delta' : delta,
         'r0_GL' : r0_GL,
         'r0_HA' : r0_HA,
         
         'N' : N,
         'num_lenslet' : num_lenslet,
         'REPEAT' : REPEAT,
         'POSSIBLE_SCREEN' : POSSIBLE_SCREEN,
         'POSSIBLE_PROP' : POSSIBLE_PROP,
         'POSSIBLE' : POSSIBLE}

# BWD = os.getcwd()
# RUN_NAME = time.strftime('%Y%m%d%H%M')
# if not os.path.isdir(RUN_NAME):
#     os.makedirs(RUN_NAME)
# os.chdir(RUN_NAME)

# with open('Ultimate_input_'
#           + time.strftime('%Y%m%d%H%M')
#           + '.pkl', 'wb') as file:

os.chdir('/mt/cfai/pkongkaew/IGPhaseScreen')

with open('Ultimate_input_extra2.pkl', 'wb') as file:
    pickle.dump(myvar, file)
    file.close()

###############################################################################
# actual simulation starts ###############################################
###############################################################################

for ii in np.arange(II):
    for jj in np.arange(JJ):
        for ll in np.arange(LL):
            print(r0[ii], dor0[jj], H_HA[ll], N[ii,jj,ll], num_lenslet[ii,jj,ll], REPEAT[ii,jj,ll])
            
EXPECTED_RUN_TIME = float(0)
time_N = np.zeros_like(N)
time_N = [11.8, 13.205, 14.894, 15.9375, 17.514, 14.880] # in minutes
EXPECTED_RUN_TIME = ((N == 128).sum()*time_N[0]
                     + (N == 256).sum()*time_N[1]
                     + (N == 512).sum()*time_N[2]
                     + (N == 1024).sum()*time_N[3]
                     + (N == 2048).sum()*time_N[4]
                     + time_N[5] * ((N == 4096)*REPEAT).sum())
print('EXPECTED_RUN_TIME (hours) = ', EXPECTED_RUN_TIME/60)
print('Should be done by', np.datetime64('now') + np.timedelta64(int(EXPECTED_RUN_TIME),'m'))
del EXPECTED_RUN_TIME, time_N

r0_GL_min = np.nanmin(r0_GL)
delta_max = np.nanmax(delta)

N_GL = int(np.sqrt(TARGET) * n)
N_GEN_GL = N_GL * oversize

if N_GEN_GL > SCREEN_LIMIT:
    print('we have to redo the code.')

ps_GL = np.zeros((n, n), dtype=float)
U_total_at_GL = np.zeros((n, n), dtype=complex)

ps_GL_all = np.zeros((TARGET, n, n), dtype=float)

ps_GL_temp = np.zeros((N_GEN_GL, N_GEN_GL), dtype=float)
F0_GEN_GL = pyfftw.empty_aligned((N_GEN_GL, N_GEN_GL), dtype=complex)
F1_GEN_GL = pyfftw.empty_aligned((N_GEN_GL, N_GEN_GL), dtype=complex)
FF01_GEN_GL = pyfftw.FFTW(F0_GEN_GL, F1_GEN_GL, axes=(-1, -2),
                        direction='FFTW_BACKWARD')

ps_GL_temp = crop2(ft_sh_phase_screen(
    r0_GL_min, N_GEN_GL, delta_max, L0_GL, l0_GL, FFT=FF01_GEN_GL), oversize)
ps_GL_all = np.asarray(np.split(
    np.asarray(np.split(
        ps_GL_temp,
        int(np.sqrt(TARGET)), axis=-1)),
    int(np.sqrt(TARGET)), axis=-2)).reshape(TARGET, n, n)


del ps_GL_temp, F0_GEN_GL, F1_GEN_GL, FF01_GEN_GL

for screen_size_power in tqdm(np.arange(int(np.log(N.max())/np.log(2)) - 5) + 6):
    screen_size = int(2**screen_size_power)
    print(screen_size)
    xxx, yyy, zzz = np.where(N == screen_size)
    
    N_GEN_HA = 2*oversize*screen_size
    N_PROP = 2*screen_size
    
    U_HA_at_GL_original = np.zeros((screen_size, screen_size), dtype=complex)
    ps_HA_original = np.zeros((screen_size, screen_size), dtype=float)

    F0_GEN_HA = pyfftw.empty_aligned(               # GEN,GEN
        (N_GEN_HA, N_GEN_HA), dtype=complex)
    F1_GEN_HA = pyfftw.empty_aligned(               # GEN,GEN
        (N_GEN_HA, N_GEN_HA), dtype=complex)
    FF01_GEN_HA = pyfftw.FFTW(
        F0_GEN_HA, F1_GEN_HA, axes=(-1, -2), direction='FFTW_BACKWARD')

    F0_PROP = pyfftw.empty_aligned(                 # PROP,PROP
        (N_PROP, N_PROP), dtype=complex)
    F1_PROP = pyfftw.empty_aligned(                 # PROP,PROP
        (N_PROP, N_PROP), dtype=complex)
    FF01_PROP = pyfftw.FFTW(
        F0_PROP, F1_PROP, axes=(-1, -2), direction='FFTW_FORWARD')
    FF10_PROP = pyfftw.FFTW(
        F1_PROP, F0_PROP, axes=(-1, -2), direction='FFTW_BACKWARD')
    
    ps_HA_repeat = np.zeros(                        # PROP,PROP
        (N_PROP, N_PROP), dtype=float)
    Q2_PROP = np.zeros(                             # PROP,PROP
        (N_PROP, N_PROP), dtype=complex)
    
    for index in tqdm(np.arange(len(xxx))):
        ii = xxx[index]
        jj = yyy[index]
        ll = zzz[index]

        # if it is not possible to MAKE screens,
        # then skip
        if (POSSIBLE[ii, jj, : , ll] == False).all():
            print('Skip: ii,jj,ll=', ii, jj, ll)
            break

        df_PROP = 1. / (N_PROP*delta[ii, jj])
        r0_HA_min = np.nanmin(r0_HA[ii, :, ll])

        (fX_PROP, fY_PROP) = np.meshgrid(               # PROP,PROP
            np.arange(-N_PROP/2, N_PROP/2) * df_PROP,
            np.arange(-N_PROP/2, N_PROP/2) * df_PROP)
        fsq_PROP = fX_PROP**2 + fY_PROP**2
        
        Q2_PROP = np.exp(-1j*(np.pi**2*2*H_HA[ll]/wavenumber) * fsq_PROP)
        
        del fX_PROP, fY_PROP, fsq_PROP
        
        print('r{:02d}dor{:02d}h{:02d}'.format(
                    int(r0[ii]*100), int(dor0[jj]*10),
                    int(H_HA[ll]/1000)))
                
        for repeat in tqdm(np.arange(REPEAT[ii, jj, ll])):
            
            U_HA_at_GL_all = np.zeros((num_lenslet[ii, jj, ll], n, n),
                                      dtype=complex)
            ps_HA_all = np.zeros((num_lenslet[ii, jj, ll], n, n),
                                      dtype=float)
            
            ps_HA_repeat = crop2(
                ft_sh_phase_screen(
                    r0_HA_min, N_GEN_HA, delta_max,
                    L0_HA, l0_HA, FFT=FF01_GEN_HA),
                oversize)

            ###################################################################
            # end of simulation overhead #################################
            # start of calculations ######################################
            ###################################################################

            # only consider possible condition of kk for a given ii jj ll
            for kk in np.where(POSSIBLE[ii, jj, : , ll] == True)[0]:
                
                run_name = 'r{:02d}dor{:02d}rytov{:02d}h{:02d}'.format(
                    int(r0[ii]*100), int(dor0[jj]*10), int(Rytov[kk]*100),
                    int(H_HA[ll]/1000))

                # if not os.path.isdir(run_name):
                #     os.makedirs(run_name)
                # if not os.path.isdir(run_name + '/ps_GL'):
                #     os.makedirs(run_name + '/ps_GL')
                # if not os.path.isdir(run_name + '/ps_HA'):
                #     os.makedirs(run_name + '/ps_HA')
                # if not os.path.isdir(run_name + '/U_tot'):
                #     os.makedirs(run_name + '/U_tot')
                
                ###############################################################
                
                # HA to ground
                # scale phase screen
                ps_HA_original = (ps_HA_repeat
                            * (r0_HA_min/delta_max
                            * delta[ii, jj]/r0_HA[ii, kk, ll])**(5./6.))
                ps_HA_all = np.asarray(np.split(
                    np.asarray(np.split(
                        ps_HA_original[ : num_lenslet[ii, jj, ll]*n,
                                       : num_lenslet[ii, jj, ll]*n],
                        num_lenslet[ii, jj, ll], axis=-1)),
                    num_lenslet[ii, jj, ll], axis=-2)).reshape(
                        num_lenslet[ii, jj, ll]**2, n, n)

                # now propagate
                F0_PROP[:] = np.fft.fftshift(
                    np.exp(1j*ps_HA_original),
                    axes=(-1, -2))
                F1_PROP[:] = np.fft.ifftshift(
                    Q2_PROP * np.fft.fftshift(
                        FF01_PROP(),
                        axes=(-1, -2)),
                    axes=(-1, -2))
                U_HA_at_GL_original = crop2(
                    np.fft.ifftshift(
                        FF10_PROP(),
                        axes=(-1, -2)),
                    2)
                
                U_HA_at_GL_all = np.asarray(np.split(
                    np.asarray(np.split(
                        U_HA_at_GL_original[int((screen_size - num_lenslet[ii, jj, ll]*n)/2) : int((screen_size + num_lenslet[ii, jj, ll]*n)/2),
                                            int((screen_size - num_lenslet[ii, jj, ll]*n)/2) : int((screen_size + num_lenslet[ii, jj, ll]*n)/2)],
                        num_lenslet[ii, jj, ll], axis=-1)),
                    num_lenslet[ii, jj, ll], axis=-2)).reshape(
                        num_lenslet[ii, jj, ll]**2, n, n)

                ###############################################################
                # combine with ground conditions
                
                for lenslet in np.arange(num_lenslet[ii, jj, ll]**2):
                    # scale phase screen
                    count = repeat * num_lenslet[ii, jj, ll]**2 + lenslet
                    # print('number of lenslet, repeat, REPEAT =')
                    # print(num_lenslet[ii, jj, ll], repeat, REPEAT[ii, jj, ll])
                    # print('ii,jj,kk,ll,lenslet,count=')
                    # print(ii,jj,kk,ll,lenslet,count)
                    if count < TARGET:
                        ps_GL = (ps_GL_all[count, : , : ]
                                * (r0_GL_min / delta_max
                                    * delta[ii, jj]
                                    / r0_GL[ii, kk, ll])**(5./6.))
                        
                        # combine
                        U_total_at_GL = (U_HA_at_GL_all[lenslet]
                                         * np.exp(1j*ps_GL))

                        #######################################################
                        # save output ####################################
                        #######################################################

                        with open('/mt/cfai/pkongkaew/IGPhaseScreen/ps_GL/ps_GL_' + run_name
                                + '_{:04d}.pkl'.format(
                                    count), 'wb') as file:
                            pickle.dump(ps_GL, file)
                            file.close()
                            
                        with open('/mt/cfai/pkongkaew/IGPhaseScreen/ps_HA/ps_HA_' + run_name
                                + '_{:04d}.pkl'.format(
                                    count), 'wb') as file:
                            pickle.dump(ps_HA_all[lenslet], file)
                            file.close()
                            
                        with open('/mt/cfai/pkongkaew/IGPhaseScreen/U_tot/U_tot_' + run_name 
                                + '_{:04d}.pkl'.format(
                                    count), 'wb') as file:
                            pickle.dump(U_total_at_GL, file)
                            file.close()

                        #######################################################
                        
                        # if (count < 4):

                        #     if not os.path.isdir('phase'):
                        #         os.makedirs('phase')
                        #     if not os.path.isdir('intensity'):
                        #         os.makedirs('intensity')
                        #     if not os.path.isdir('phase_GL'):
                        #         os.makedirs('phase_GL')
                        #     if not os.path.isdir('phase_HA'):
                        #         os.makedirs('phase_HA')

                        #     ###################################################
                            
                        #     plt.imshow(np.angle(U_total_at_GL))
                        #     plt.colorbar()
                        #     plt.title('r0={:.0f}cm, d/r0={:.1f}, '.format(
                        #         r0[ii]*100, dor0[jj])
                        #             + 'Rytov={:.2f}, H_HA={:.0f}km'.format(
                        #                 Rytov[kk], H_HA[ll]/1000))
                        #     plt.savefig('phase/phase_' + run_name
                        #                 + '_{:04d}.png'.format(
                        #                     count))
                        #     plt.clf()
                            
                        #     plt.imshow(np.abs(U_total_at_GL)**2,
                        #                vmin=0, vmax=2)
                        #     plt.colorbar()
                        #     plt.title('r0={:.0f}cm, d/r0={:.1f}, '.format(
                        #         r0[ii]*100, dor0[jj])
                        #             + 'Rytov={:.2f}, H_HA={:.0f}km'.format(
                        #                 Rytov[kk], H_HA[ll]/1000))
                        #     plt.savefig('intensity/intensity_' + run_name
                        #                 + '_{:04d}.png'.format(
                        #                     count))
                        #     plt.clf()
                            
                        #     plt.imshow(ps_GL)
                        #     plt.colorbar()
                        #     plt.title('r0={:.0f}cm, d/r0={:.1f}, '.format(
                        #         r0[ii]*100, dor0[jj])
                        #             + 'Rytov={:.2f}, H_HA={:.0f}km'.format(
                        #                 Rytov[kk], H_HA[ll]/1000))
                        #     plt.savefig('phase_GL/phase_GL_' + run_name
                        #                 + '_{:04d}.png'.format(
                        #                     count))
                        #     plt.clf()
                            
                        #     plt.imshow(ps_HA_all[lenslet])
                        #     plt.colorbar()
                        #     plt.title('r0={:.0f}cm, d/r0={:.1f}, '.format(
                        #         r0[ii]*100, dor0[jj])
                        #             + 'Rytov={:.2f}, H_HA={:.0f}km'.format(
                        #                 Rytov[kk], H_HA[ll]/1000))
                        #     plt.savefig('phase_HA/phase_HA_' + run_name
                        #                 + '_{:04d}.png'.format(
                        #                     count))
                        #     plt.clf()
                    else:
                        break
                    
            ###################################################################
            
            del U_HA_at_GL_all, ps_HA_all
                        
    ###########################################################################
    
    del F0_GEN_HA, F1_GEN_HA, FF01_GEN_HA, Q2_PROP
    del F0_PROP, F1_PROP, FF01_PROP, FF10_PROP
    del U_HA_at_GL_original, ps_HA_repeat, ps_HA_original
    
###############################################################################

del ps_GL, U_total_at_GL, ps_GL_all

###############################################################################

# TIME_END = time.time()
# np.savetxt('totaltime.txt', [TIME_END - TIME_START])

###############################################################################
##########################################################################
"""
█▀▀█░█░▒█░▀█▀░▀█▀░░▀░░█░░░█░█▀▀▄░▀█▀░░░▒█░▄▀░▄▀▀▄░█▀▀▄░█▀▀▀░█░▄░█▀▀▄░█▀▀░█░░░█
█▄▄█░█░▒█░░█░░░█░░░█▀░▀▄█▄▀░█▄▄█░░█░░░░▒█▀▄░░█░░█░█░▒█░█░▀▄░█▀▄░█▄▄█░█▀▀░▀▄█▄▀
█░░░░░▀▀▀░░▀░░░▀░░▀▀▀░░▀░▀░░▀░░▀░░▀░░░░▒█░▒█░░▀▀░░▀░░▀░▀▀▀▀░▀░▀░▀░░▀░▀▀▀░░▀░▀░
"""
##########################################################################
###############################################################################
