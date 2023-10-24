# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 11:37:55 2022

@author: gqmc62
"""

import numpy as np
from matplotlib import pyplot as plt
from aotools.turbulence.phasescreen import ft_sh_phase_screen
import time
from aotools.opticalpropagation import angularSpectrum
from aotools.functions.zernike import zernikeArray
import my_propagation as mprop
import my_WFS as mwfs
import my_phasescreen as mps

def get_dm_stroke_phase(phase,lenslet_pixel):
    N = len(phase)
    nl = int(N/lenslet_pixel)
    nd = nl + 1
    halflen = int(lenslet_pixel/2)
    stroke = np.zeros((nd,nd),dtype=float)
    stroke[1:-1,1:-1] = mwfs.get_val_lenslet(
        phase[halflen:-halflen,
              halflen:-halflen],
        lenslet_pixel)
    stroke[0,0]=phase[:halflen,:halflen].mean()
    stroke[0,-1]=phase[:halflen,-halflen:].mean()
    stroke[-1,0]=phase[-halflen:,:halflen].mean()
    stroke[-1,-1]=phase[-halflen:,-halflen:].mean()
    for i in np.arange(nl-1):
        stroke[0,i+1]=phase[:halflen,i*lenslet_pixel+halflen:(i+1)*lenslet_pixel+halflen].mean()
        stroke[-1,i+1]=phase[-halflen:,i*lenslet_pixel+halflen:(i+1)*lenslet_pixel+halflen].mean()
        stroke[i+1,0]=phase[i*lenslet_pixel+halflen:(i+1)*lenslet_pixel+halflen,:halflen].mean()
        stroke[i+1,-1]=phase[i*lenslet_pixel+halflen:(i+1)*lenslet_pixel+halflen,-halflen:].mean()
    return stroke

def new_slope(phase):
    """
    New better slope measurement. averaging between 4 phase pixels.
    
    x = np.arange(-16, 16)
    x = x*(x > 0) * 2.*np.pi/16.*4.
    xx, yy = np.meshgrid(x, x)
    phase = - xx + yy
    phase = np.angle(np.exp(1j*phase))
    phase_unwrap = np.unwrap(np.unwrap(phase,axis=-1),axis=-2)
    slope_phase = new_slope(phase)
    plt.pcolor(np.arange(-16,16)+0.5,
               np.arange(-16,16)+0.5,
               phase_unwrap,
               snap=True, shading='auto')
    plt.colorbar()
    plt.quiver(np.arange(-15,15+1),
               np.arange(-15,15+1),
               newslope[0],
               newslope[1],
               headwidth=5)
    plt.axis('equal')
    plt.xlabel('x (pixel)')
    plt.ylabel('y (pixel)')
    plt.title('Phase = -x + y; x>0, y>0, Phase = y; x<0, y>0\n'
              + 'Phase = -x; x>0, y<0, Phase = 0; x<0, y<0')
    plt.show()

    Parameters
    ----------
    phase : TYPE
        DESCRIPTION.

    Returns
    -------
    slope : TYPE
        DESCRIPTION.

    """
    N = len(phase)
    n = N - 1
    slope = np.zeros((2,n,n))
    for i in np.arange(n):
        for j in np.arange(n):
            A = phase[i + 1, j] - phase[i, j]
            B = phase[i + 1, j + 1] - phase[i, j + 1]
            C = phase[i, j + 1] - phase[i, j]
            D = phase[i + 1, j + 1] - phase[i + 1, j]
            temp = np.array([A,B,C,D])
            for k in np.arange(4):
                if np.abs(temp[k]) > np.pi:
                    if temp[k] > 0:
                        temp[k] = temp[k] - np.pi*2.
                    else:
                        temp[k] = temp[k] + np.pi*2.
            A, B, C, D = temp
            if np.abs(A - B) < np.pi:
                slope[1, i, j] = (A + B)/2.
            else:
                if A > B:
                    slope[1, i, j] = (A + B - 2.*np.pi)/2.
                else:
                    slope[1, i, j] = (A + B + 2.*np.pi)/2.
            if np.abs(C - D) < np.pi:
                slope[0, i, j] = (C + D)/2.
            else:
                if C > D:
                    slope[1, i, j] = (C + D - 2.*np.pi)/2.
                else:
                    slope[1, i, j] = (C + D + 2.*np.pi)/2.
    return slope

def make_pupil_of_slope(pupil):
    N = len(pupil)
    n =  N - 1
    pupil_of_slope = np.ones((n,n), dtype=float)
    for i in np.arange(n):
        for j in np.arange(n):
            if ((pupil[i, j] == 0)
                or (pupil[i, j + 1] == 0)
                or (pupil[i + 1, j] == 0)
                or (pupil[i + 1, j + 1] == 0)):
                pupil_of_slope[i, j] = 0
    return pupil_of_slope

def stroke_to_phase(stroke, lenslet_pixel):
    n = len(stroke)
    N = (n - 1)*lenslet_pixel
    phase = np.zeros((N,N), dtype=float)
    x = (np.arange(-lenslet_pixel/2,lenslet_pixel/2) + 0.5)
    x1, y1 = np.meshgrid(x, x)
    for il in np.arange(n - 1):
        for jl in np.arange(n - 1):
            slopey = ((stroke[il + 1, jl] - stroke[il, jl])
                      + stroke[il + 1, jl + 1] - stroke[il, jl + 1]) / 2. / lenslet_pixel
            slopex = ((stroke[il, jl + 1] - stroke[il, jl])
                      + stroke[il + 1, jl + 1] - stroke[il + 1, jl]) / 2. / lenslet_pixel
            subphase = x1*slopex + y1*slopey
            piston = (stroke[il + 1, jl] + stroke[il, jl]
                      + stroke[il + 1, jl + 1] + stroke[il, jl + 1]) / 4.
            phase[(il)*lenslet_pixel
                  : (il + 1)*lenslet_pixel,
                  (jl)*lenslet_pixel
                  : (jl + 1)*lenslet_pixel] = subphase + piston
    return phase

TA = time.time()

N = 2048 #1024
d_in = 3e-3#5e-3#want 1e-3
L0 = 20
l0 = 1.5e-10
wvl = 500e-9

lenslet_pixel = 16

SNR = 100
zenith = 0

# z = np.array([0.1,0.2,0.5,1,2,5,10,25], dtype=float)*1e3
# r = np.array([0.3,0.2,0.13,0.08,0.05] ,dtype=float)

# z = np.array([5,10,15,20,25])*1e3
# r = np.array([0.3,0.2,0.13,0.08,0.05])

# z = np.array([0,5,10,15,20,25])*1e3
# r = np.array([0.076,0.115,0.170,0.247,0.345,0.792])

z = np.array([0,5,10,15,20,25])*1e3/np.cos(zenith/360*2*np.pi)#([25,20,15,10,5,0])*1e3
r = np.array([0.409])#([0.05,0.08,0.13,0.2,0.3])



#
# Independent constants
#
pf_lenslet = 2
pf = 2
prN = 64 # 128
I_max = 0.3
I_min = -1.

paramdict={'N':N,'d_in':d_in,'L0':L0,'l0':l0,'wvl':wvl,
           'lenslet_pixel':lenslet_pixel,'z':z,'r':r,
           'pf_lenslet':pf_lenslet,'pf':pf,'SNR':SNR}

#
# Dependent constants
#

d_out = d_in
num_lenslet = int(N/lenslet_pixel)

d_lenslet = d_in/1000
d_detector = d_lenslet

D_lenslet = lenslet_pixel * d_lenslet
f_lenslet = d_detector/wvl*2.*D_lenslet

NN = pf*N
prNN = int(prN/lenslet_pixel)
I_norm = lenslet_pixel**2./4.

n_radial_modes = int(num_lenslet)
n_modes = (np.arange(n_radial_modes + 1) + 1).sum()

n_wfs=num_lenslet**2*2
n_dm=(num_lenslet+1)**2

#
# Checking propagation parametres #
#

# Propagation between atmospheric layers
check = True
check_condition = mprop.prop_param_check(N, d_in, d_out, wvl, z, pf,
                                        scin_blob=False)#True)#True
for i in check_condition:
    check = check * i
if check == False:
    print("One of the atm propagation condition for atm layer is failing.")
    quit

# # Propagation through SHWFS
check = True
check_condition = mprop.prop_lens_param_check(lenslet_pixel, d_lenslet, d_detector,
                                        wvl, [f_lenslet], pf_lenslet)
for i in check_condition:
    check = check * i
if check == False:
    print("One of the wfs propagation condition for lenslet is failing.")
    quit

#
# Declaration field
#
I_in_big = np.ones((NN,NN), dtype=float)          # Input amplitude.
phase_in_big = np.zeros((NN, NN), dtype=float)       # Input phase (rad).
U_in_big = np.zeros((NN,NN), dtype=complex)         # Input complex amp.

I_in = np.ones((N,N), dtype=float)          # Input amplitude.
phase_in = np.zeros((N, N), dtype=float)       # Input phase (rad).
U_in = np.zeros((N,N), dtype=complex)         # Input complex amp.

slope_phase_in = np.zeros((N, N, 2), dtype=float)

U_out_big = np.zeros((NN,NN), dtype=complex)         # Input complex amp.

I_out = np.ones((N,N), dtype=float)
phase_out = np.zeros((N, N), dtype=float)       # Input phase (rad).
U_out = np.zeros((N,N), dtype=complex)         # Input complex amp.

slope_phase_out = np.ones((N,N, 2), dtype=float)


# # curiosity
# rms_sky_to_ground = np.zeros((len(r),len(z)), dtype=float)

# # control group / measure at altitude
# # this is what we always do in simulation sense.
# rms_sky_to_sky_avg = np.zeros((len(r)),dtype=float)###
# rms_sky_to_sky_WFS = np.zeros((len(r)), dtype=float)###

# # measure at ground
# # show precision of measurement at ground
# rms_ground_to_ground_avg = np.zeros((len(r),len(z)), dtype=float)
# rms_ground_to_ground_WFS = np.zeros((len(r),len(z)), dtype=float)
# rms_ground_to_ground_weight = np.zeros((len(r),len(z)), dtype=float)

# # use ground measurement as sky shape / measure at ground assume = sky then fit dm
# # this is actually what we have always do in practical sense
# rms_sky_to_ground_avg = np.zeros((len(r),len(z)), dtype=float)
# rms_sky_to_ground_WFS = np.zeros((len(r),len(z)), dtype=float)
# rms_sky_to_ground_weight = np.zeros((len(r),len(z)), dtype=float)

# # compare wfs and wfs approx
# rms_ground_WFS_to_ground_weight = np.zeros((len(r),len(z)), dtype=float)

# # compare best dm shape with wfs
# rms_sky_avg_to_sky_WFS = np.zeros((len(r)), dtype=float)###
# rms_ground_avg_to_ground_WFS = np.zeros((len(r),len(z)), dtype=float)
# rms_ground_avg_to_ground_weight = np.zeros((len(r),len(z)), dtype=float)


#
# Calculations
#

#make zernike
# ta=time.time()

# zernike_modes = zernikeArray(n_modes, num_lenslet+1)
# pupil = zernike_modes[0,:,:]
# pupil_of_slope = make_pupil_of_slope(pupil)
# flat_zernike_modes = np.empty(n_modes, dtype=list)
# for i in np.arange(n_modes):
#     flat_zernike_modes[i] = zernike_modes[i].flatten()
# gram_schmidt_modes = np.zeros_like(zernike_modes)
# flat_gram_schmidt_modes = np.zeros_like(flat_zernike_modes)
# zernike_to_gram_schmidt = np.zeros((n_modes, n_modes))
# gram_schmidt_to_zernike = np.zeros((n_modes, n_modes))
# gram_schmidt_coefficients = np.zeros((n_modes))
# zernike_coefficients = np.zeros((n_modes))

# gram_schmidt_modes = mps.gram_schmidt_process(zernike_modes)
# for i in np.arange(n_modes):
#     flat_gram_schmidt_modes[i] = gram_schmidt_modes[i].flatten()
# for i in np.arange(n_modes):
#     for j in np.arange(n_modes):
#         zernike_to_gram_schmidt[i,j] = np.dot(flat_gram_schmidt_modes[i],
#                                               flat_zernike_modes[j])
# gram_schmidt_to_zernike = np.linalg.inv(zernike_to_gram_schmidt)
# tb=time.time()
# print('time make zernikes and gram-schmidts')
# print(tb-ta)

# IMwfsmode = np.zeros((num_lenslet**2*2,n_modes),dtype=float)
# IMmodedm = np.zeros((n_modes, (num_lenslet + 1)**2),dtype=float)
# IMwfsdm = np.zeros((num_lenslet**2*2,(num_lenslet + 1)**2),dtype=float)
# IMdmmode = np.zeros(((num_lenslet + 1)**2,n_modes),dtype=float)

# #make interaction matrix via mode
# ta=time.time()
# for imode in np.arange(zernike_modes.shape[0]):
#     phase = gram_schmidt_modes[imode]
#     slope = new_slope(phase)
#     slope = slope * np.array([pupil_of_slope, pupil_of_slope])
#     # plt.pcolor(np.arange((-(num_lenslet+1)/2.),((num_lenslet+1)/2.))+0.5,
#     #             np.arange((-(num_lenslet+1)/2.),((num_lenslet+1)/2.))+0.5,
#     #             phase,snap=True,shading='auto')
#     # plt.colorbar()
#     # plt.axis('equal')
#     # plt.quiver(np.arange(-num_lenslet/2.,num_lenslet/2.)+0.5,
#     #             np.arange(-num_lenslet/2.,num_lenslet/2.)+0.5,
#     #             slope[0],slope[1])
#     # plt.axis('equal')
#     # plt.show()
#     flat_slope = slope.flatten()
#     flat_phase = phase.flatten()
#     for i_flat_len in np.arange(num_lenslet**2*2):
#         IMwfsmode[i_flat_len,imode] = flat_slope[i_flat_len]
#     for i_actuator in np.arange((num_lenslet + 1)**2):
#         IMdmmode[i_actuator,imode] = flat_phase[i_actuator]

# #make interaction matrix
# # dm_shape = np.zeros((N + lenslet_pixel*2, N + lenslet_pixel*2))

# IMmodedm=np.linalg.pinv(IMdmmode)
# IMwfsdm=np.matmul(IMwfsmode,IMmodedm)
# IMdmwfs=np.linalg.pinv(IMwfsdm)

# plt.imshow(IMwfsdm)
# plt.colorbar()
# plt.show()

# plt.imshow(IMdmwfs)
# plt.colorbar()
# plt.show()

# tb=time.time()
# print('time making interaction matrix')
# print(tb-ta)

#test

# phasetest=gram_schmidt_modes[1]*2+gram_schmidt_modes[3]*1+gram_schmidt_modes[6]*0.5+gram_schmidt_modes[10]*0.25+gram_schmidt_modes[15]*0.1
# plt.pcolor(np.arange((-(num_lenslet+1)/2.),((num_lenslet+1)/2.))+0.5,
#              np.arange((-(num_lenslet+1)/2.),((num_lenslet+1)/2.))+0.5,
#              phasetest,snap=True,shading='auto')
# plt.axis('equal')
# plt.colorbar()
# plt.show()

# all_rytov_pixel = np.zeros((len(r),len(z)), dtype=float)

for ir in np.arange(len(r)):
    # Initialization #
    ta = time.time()
    phase_in_big = ft_sh_phase_screen(r[ir], NN, d_in, L0, l0, seed=789)
    tb = time.time()
    t_make_screen = tb - ta
    print('time make screen = {:.6f}s'.format(t_make_screen))
    
    # # Check screen condition #
    # ta = time.time()
    # check_scrn = mwfs.check_screen(phase_in_big, d_in, wvl, d_out, z, pf)
    # if check_scrn == False:
    #     print('Input phase screen is not Nyquist sampled.')
    #     print('It is too dense. aliasing happen')
    #     print('r={:.2f}m, d_in={:.2f}m'.format(r[ir], d_in))
    #     print('For this r, d_in must be smaller.')
    #     quit
    # tb = time.time()
    # t_check_screen = tb - ta
    # print('time check screen = {:.6f}s'.format(t_check_screen))

    U_in_big = np.sqrt(I_in_big)*np.exp(1j*phase_in_big)

    #
    # Reporting source conditions #
    #
    ta = time.time()
    U_in = mprop.crop2(U_in_big, pf)
    phase_in = np.angle(U_in)

    I_in = np.abs(U_in)**2
    slope_phase_in = mwfs.get_slope(phase_in, d_in, wvl)
    
    tb = time.time()
    t_analyze_input_px = tb - ta
    print('time analyze input pixely = {:.6f}s'.format(t_analyze_input_px))
    
    # ta = time.time()
    phase_in_unwrap = np.unwrap(np.unwrap(
        phase_in, period=2*np.pi, axis=-2),
        period=2*np.pi, axis=-1)
    
    #
    # Average #
    #
    ta = time.time()
    lenslet_I_in = mwfs.get_val_lenslet(I_in, lenslet_pixel)
    lenslet_slope_phase_in = (mwfs.get_vec_lenslet(slope_phase_in,
                                                    lenslet_pixel))
    tb = time.time()
    t_analyze_input_lenslet = tb - ta
    print('time anaylze input lensletly = {:.6f}s'.format(t_analyze_input_lenslet))

    # #
    # # SHWFS cheat by weighting slope with intensity #
    # #
    # ta = time.time()
    # weight_slope_phase_in = np.zeros((N,N,2), dtype=float)
    # weight_slope_phase_in[:,:,0] = slope_phase_in[:,:,0]*I_in
    # weight_slope_phase_in[:,:,1] = slope_phase_in[:,:,1]*I_in
    # lenslet_weight_slope_phase_in = mwfs.get_vec_lenslet(
    #     weight_slope_phase_in, lenslet_pixel)
    # lenslet_weight_slope_phase_in[:,:,0] = (
    #     lenslet_weight_slope_phase_in[:,:,0]/lenslet_I_in)
    # lenslet_weight_slope_phase_in[:,:,1] = (
    #     lenslet_weight_slope_phase_in[:,:,1]/lenslet_I_in)
    # tb = time.time()
    # t_analyze_input_weight = tb - ta
    # print('time analyze input weight = {:.6f}s'.format(t_analyze_input_weight))

    #
    # Proper SHWFS #
    #
    ta = time.time()
    U_WFS_detector_in = mwfs.propagate_lenslet(U_in, wvl, d_lenslet,
                                                f_lenslet, lenslet_pixel,
                                                d_detector, pf_lenslet,plot_lenslet=False)
    I_WFS_detector_in = (np.abs(U_WFS_detector_in)**2
                          * (d_detector/d_lenslet)**2)
    lenslet_I_WFS_detector_in = mwfs.get_val_lenslet(
        I_WFS_detector_in, lenslet_pixel)
    noisy_I_WFS_detector_in = np.random.poisson(
        I_WFS_detector_in/lenslet_pixel**2*SNR**2) / SNR**2*lenslet_pixel**2
    slope_WFS_in = mwfs.get_SHWFS_slope(noisy_I_WFS_detector_in, lenslet_pixel,
                                    d_detector, f_lenslet)
    projected_slope_WFS_in = slope_WFS_in * d_lenslet / d_in
    tb = time.time()
    t_analyze_input_WFS = tb - ta
    print('time to analyze WFS in = {:.6f}s'.format(t_analyze_input_WFS))

    # make perfect dm
    ta = time.time()

    tb = time.time()
    t_dm = tb - ta
    print('time make perfect dms = {:.6f}s'.format(t_dm))

    # analysis rms in
    ta = time.time()
    
    tb = time.time()
    t_analyze_rms_in = tb - ta
    print('time to analyze rms input = {:.6f}s'.format(t_analyze_rms_in))
    
TB = time.time()
t_all = TB - TA
print('time all = {:.6f}s'.format(t_all))