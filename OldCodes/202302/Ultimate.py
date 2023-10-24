
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

TIME_START = time.time()

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

REPEAT = int(100)
SCREEN_LIMIT = int(2**16)

num_lenslet = int(1)
n = int(64)
oversize = int(16)
L0_GL = float(2000)
L0_HA = L0_GL
l0_GL = float(1e-12)
l0_HA = l0_GL
wvl = float(500e-9)
zenith_angle = float(0)

###############################################################################

# input conditions

r0 = np.array([11, 8, 5, 2], dtype=float)*1e-2              # II
dor0 = np.array([0.5, 1, 2, 3], dtype=float)                # JJ
Rytov = np.array([0.01, 0.03, 0.1, 0.3], dtype=float)       # KK
H_HA = np.array([5, 10, 15, 20], dtype=float)*1e3           # LL

# r0 = np.array([11, 2], dtype=float)*1e-2                    # II
# dor0 = np.array([0.5, 3], dtype=float)                      # JJ
# Rytov = np.array([0.1, 0.3], dtype=float)                   # KK
# H_HA = np.array([5, 20], dtype=float)*1e3                   # LL

# r0 = np.array([5], dtype=float)*1e-2                        # II
# dor0 = np.array([3], dtype=float)                           # JJ
# Rytov = np.array([0.1], dtype=float)                        # KK
# H_HA = np.array([5], dtype=float)*1e3                       # LL

# r0 = np.array([2], dtype=float)*1e-2                        # II
# dor0 = np.array([3], dtype=float)                           # JJ
# Rytov = np.array([0.3], dtype=float)                        # KK
# H_HA = np.array([20], dtype=float)*1e3                      # LL

# r0 = np.array([11, 8], dtype=float)*1e-2                    # II
# dor0 = np.array([0.5, 1], dtype=float)                      # JJ
# Rytov = np.array([0, 0.01], dtype=float)                    # KK
# H_HA = np.array([5, 10], dtype=float)*1e3                   # LL

###############################################################################

# Common indices ranges and dimension. Easy to keep track of dependencies
II = len(r0)
JJ = len(dor0)
KK = len(Rytov)
LL = len(H_HA)

# dependent simulation parameters
N = n*num_lenslet
wavenumber = 2*np.pi/wvl
MAX_prop_pad = int(SCREEN_LIMIT/N/oversize/8)
print('max prop pad = {:d}'.format(MAX_prop_pad))

d = np.zeros((II, JJ), dtype=float)                         # II, JJ
delta = np.zeros((II, JJ), dtype=float)                     # II, JJ
r0_GL = np.zeros((II, KK, LL), dtype=float)                 # II, KK, LL
r0_HA = np.zeros((II, KK, LL), dtype=float)                 # II, KK, LL
prop_pad = np.zeros((II, JJ, LL), dtype=int)                # II, JJ, LL
POSSIBLE_SCREEN = np.ones_like(r0_GL, dtype=bool)           # II, KK, LL
POSSIBLE_SAMPLE = np.ones((II, JJ, KK, LL), dtype=bool)     # II, JJ, KK, LL
POSSIBLE_PROP = np.ones_like(prop_pad, dtype=bool)          # II, JJ, LL
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
                
###############################################################################
# Check possiblity and validity to run
                
# screens must be well sampled
for ii in range(II):
    for jj in range(JJ):
        for kk in range(KK):
            for ll in range(LL):
                if POSSIBLE_SCREEN[ii, jj, ll]:
                    if ((r0_GL[ii, kk, ll]/delta[ii, jj] <= 16)
                        or (r0_HA[ii, kk, ll]/delta[ii, jj] <= 16)):
                        POSSIBLE_SAMPLE[ii, jj, kk, ll] = False
                              
# calculating how big a HA screen is required.
# note this is for free atmosphere propagation
temp1 = float(0)
temp2 = float(0)
temp3 = float(0)
temp11 = int(0)
temp22 = int(0)
temp33 = int(0)

for ii in np.arange(II):
    for jj in np.arange(JJ):
        for ll in np.arange(LL):
            temp1 = 1 + (wvl*H_HA[ll])/(2*delta[ii, jj]**2*N*oversize)
            temp2 = (wvl*H_HA[ll])/(delta[ii, jj]**2*N*oversize)
            temp3 = 1 + (2*wvl*H_HA[ll])/(2*delta[ii, jj]**2*N*oversize)
            temp11 = int(2**np.ceil(np.log(temp1)/np.log(2)))
            temp22 = int(2**np.ceil(np.log(temp2)/np.log(2)))
            temp33 = int(2**np.ceil(np.log(temp3)/np.log(2)))
            prop_pad[ii, jj, ll] = np.max([temp11, temp22, temp33])
            
del temp1, temp2, temp3, temp11, temp22, temp33

# if the required screen size is too large for memory, just ignore it.
# put it into a masking array
POSSIBLE_PROP[np.where(prop_pad >= MAX_prop_pad)] = False

# possible cases to look at

for ii in np.arange(II):
    for jj in np.arange(JJ):
        for kk in np.arange(KK):
            for ll in np.arange(LL):
                POSSIBLE[ii, jj, kk, ll] = (POSSIBLE_PROP[ii, jj, ll]
                                            * POSSIBLE_SCREEN[ii, kk, ll]
                                            * POSSIBLE_SAMPLE[ii, jj, kk, ll])

###############################################################################
# saving all input parameters before crazy simulation actually runs

myvar = {'REPEAT' : REPEAT,
         'SCREEN_LIMIT' : SCREEN_LIMIT,
         'num_lenslet' : num_lenslet,
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
         'N' : N,
         'wavenumber' : wavenumber,
         'd' : d,
         'delta' : delta,
         'r0_GL' : r0_GL,
         'r0_HA' : r0_HA,
         'prop_pad' : prop_pad,
         'POSSIBLE_SCREEN' : POSSIBLE_SCREEN,
         'POSSIBLE_SAMPLE' : POSSIBLE_SAMPLE,
         'POSSIBLE_PROP' : POSSIBLE_PROP,
         'POSSIBLE' : POSSIBLE}

# with open('Ultimate_input_'
#           + time.strftime('%Y%m%d%H%m')
#           + '.pkl', 'wb') as file:
with open('Ultimate_input.pkl', 'wb') as file:
    pickle.dump(myvar, file)
    file.close()

###############################################################################
# actual simulation starts ###############################################
###############################################################################

ESTIMATED_RUN_TIME = float(0)
tempa = np.array([4,8,16,     32,     64], dtype=int)
tempb = np.array([3,7,27,2*60+10,8*60+40], dtype=float)*16*16 / 10.

for i in np.arange(len(tempa)):
    print('There are {:d} of {:d} pad.'.format(
        ((prop_pad == tempa[i])).sum(), tempa[i]))
    print('Usually takes {:f}s for each iteration of {:d} pad.'.format(
        tempb[i], tempa[i]))
    ESTIMATED_RUN_TIME = (ESTIMATED_RUN_TIME
                          + tempb[i]*REPEAT*(prop_pad == tempa[i]).sum())
del tempa, tempb
ESTIMATED_RUN_TIME = np.ceil(ESTIMATED_RUN_TIME / 3600.)
print('Estimated run time is {:.0f} hours. ^2?.'.format(
    int(ESTIMATED_RUN_TIME)))

r0_GL_min = np.nanmin(r0_GL)
delta_max = np.nanmax(delta)

N_GEN_GL = oversize*N

ps_GL = np.zeros((N, N), dtype=float)
ps_GL_all = np.zeros((REPEAT, N, N), dtype=float)
U_HA_at_GL = np.zeros((N, N), dtype=complex)
U_total_at_GL = np.zeros((N, N), dtype=complex)

F0_GEN_GL = pyfftw.empty_aligned((N_GEN_GL, N_GEN_GL), dtype=complex)
F1_GEN_GL = pyfftw.empty_aligned((N_GEN_GL, N_GEN_GL), dtype=complex)
FF01_GEN_GL = pyfftw.FFTW(F0_GEN_GL, F1_GEN_GL, axes=(-1, -2),
                        direction='FFTW_BACKWARD')

for repeat in tqdm(np.arange(REPEAT)):
    ps_GL_all[repeat, : , : ] = crop2(
        ft_sh_phase_screen(
            r0_GL_min, N_GEN_GL, delta_max, L0_GL, l0_GL, FFT=FF01_GEN_GL),
        oversize)

#for pad_power in np.arange([128]):
for pad_power in (np.arange(int(np.log(MAX_prop_pad)/np.log(2))) + 1):
    print(2**pad_power)
    xxx, yyy, zzz = np.where(prop_pad == 2**pad_power)
    this_PROP_PAD = 2**pad_power
    N_PROP = this_PROP_PAD*N
    N_GEN_HA = oversize*N_PROP
    ###################################################################
    # allocate same size of memory for many iterations

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
    
    ps_HA = np.zeros(                               # PROP,PROP
        (N_PROP, N_PROP), dtype=float)
    Q2_PROP = np.zeros(                             # PROP,PROP
        (N_PROP, N_PROP), dtype=complex)
    U_HA_at_GL = np.zeros(                          # PROP,PROP
        (N_PROP, N_PROP), dtype=complex)
    
    for index in (np.arange(len(xxx))):
        ii = xxx[index]
        jj = yyy[index]
        ll = zzz[index]

        # if it is not possible to propagate screen,
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
        del fX_PROP, fY_PROP
        Q2_PROP = np.exp(-1j*(np.pi**2*2*H_HA[ll]/wavenumber) * fsq_PROP)
        del fsq_PROP
        
        for repeat in tqdm(np.arange(REPEAT)):
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
                # HA to ground
                # scale phase screen
                ps_HA = (ps_HA_repeat
                            * (r0_HA_min/delta_max
                            * delta[ii, jj]/r0_HA[ii, kk, ll])**(5./6.))
                # now propagate
                F0_PROP[:] = np.fft.fftshift(
                    np.exp(1j*ps_HA),
                    axes=(-1, -2))
                F1_PROP[:] = np.fft.ifftshift(
                    Q2_PROP * np.fft.fftshift(
                        FF01_PROP(),
                        axes=(-1, -2)),
                    axes=(-1, -2))
                U_HA_at_GL = crop2(
                    np.fft.ifftshift(
                        FF10_PROP(),
                        axes=(-1, -2)),
                    prop_pad[ii, jj, ll])
                for repeat_GL in np.arange(REPEAT):
                    # scale phase screen
                    ps_GL = (ps_GL_all[repeat_GL, : , : ]
                            * (r0_GL_min/delta_max
                                * delta[ii, jj]/r0_GL[ii, kk, ll])**(5./6.))
                    # combine
                    U_total_at_GL = U_HA_at_GL * np.exp(1j*ps_GL)

                    ###########################################################
                    # save output ########################################
                    ###########################################################

                    with open('ps_GL/ps_GL_r{:02d}dor{:02d}rytov{:03d}h{:02d}_{:02d}{:02d}.pkl'.format(int(r0[ii]*100),int(dor0[jj]*10),int(Rytov[kk]*100),int(H_HA[ll]/1000),repeat,repeat_GL), 'wb') as file:
                        pickle.dump(ps_GL, file)
                        file.close()
                    with open('ps_HA/ps_HA_r{:02d}dor{:02d}rytov{:03d}h{:02d}_{:02d}{:02d}.pkl'.format(int(r0[ii]*100),int(dor0[jj]*10),int(Rytov[kk]*100),int(H_HA[ll]/1000),repeat,repeat_GL), 'wb') as file:
                        pickle.dump(crop2(ps_HA, prop_pad[ii, jj, ll]), file)
                        file.close()
                    with open('U_tot/U_tot_r{:02d}dor{:02d}rytov{:03d}h{:02d}_{:02d}{:02d}.pkl'.format(int(r0[ii]*100),int(dor0[jj]*10),int(Rytov[kk]*100),int(H_HA[ll]/1000),repeat,repeat_GL), 'wb') as file:
                        pickle.dump(U_total_at_GL, file)
                        file.close()

                    # if repeat == 0:
                    #     plt.imshow(np.angle(U_total_at_GL))
                    #     plt.colorbar()
                    #     plt.title('r0={:.0f}cm, d/r0={:.1f}, Rytov={:.2f}, H_HA={:.0f}km'.format(r0[ii]*100, dor0[jj], Rytov[kk], H_HA[ll]/1000))
                    #     plt.savefig('phase/phase{:d}{:d}{:d}{:d}_{:04d}.png'.format(ii,jj,kk,ll,repeat))
                    #     plt.clf()
                    #     plt.imshow(np.abs(U_total_at_GL)**2, vmin=0, vmax=2)
                    #     plt.colorbar()
                    #     plt.title('r0={:.0f}cm, d/r0={:.1f}, Rytov={:.2f}, H_HA={:.0f}km'.format(r0[ii]*100, dor0[jj], Rytov[kk], H_HA[ll]/1000))
                    #     plt.savefig('intensity/intensity{:d}{:d}{:d}{:d}_{:04d}.png'.format(ii,jj,kk,ll,repeat))
                    #     plt.clf()
                
    del F0_GEN_HA, F1_GEN_HA, FF01_GEN_HA, ps_HA, Q2_PROP, U_HA_at_GL
    del F0_PROP, F1_PROP, FF01_PROP, FF10_PROP
del ps_GL, U_total_at_GL, F0_GEN_GL, F1_GEN_GL, FF01_GEN_GL

TIME_END = time.time()
np.savetxt('totaltime{:d}it.txt'.format(REPEAT), [TIME_END - TIME_START])
