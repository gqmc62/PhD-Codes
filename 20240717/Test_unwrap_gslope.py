# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 15:14:50 2023

@author: gqmc62
"""
import numpy as np
from aotools.functions.zernike import zernikeArray
from matplotlib import pyplot as plt
from aotools.turbulence.phasescreen import ft_sh_phase_screen
import my_propagation as mprop

def gslope(phase):
    """
    Measure different in phase between each pixel.

    Parameters
    ----------
    phase : np.ndarray
        (N,N)
        (float, float)
        unit in radian

    Returns
    -------
    slope : np.ndarray
        (N - 1, N - 1)
        (float, float)
        unit in radian per pixel

    """
    difx = phase[1:,:] - phase[:-1,:]
    dify = phase[:,1:] - phase[:,:-1]
    slope = np.array([np.mean(difx), np.mean(dify)], dtype=float)
    return slope

# def gslope_wrap(phase_wrap):
#     """
#     Measure different in phase between each pixel, for wrapped phase.

#     Parameters
#     ----------
#     phase_wrap : np.ndarray
#         (N,N)
#         (float, float)
#         unit in radian

#     Returns
#     -------
#     slope : np.ndarray
#         (N - 1, N - 1)
#         (float, float)
#         unit in radian

#     """
#     a = phase_wrap[1:-1, 2 :  ] - phase_wrap[1:-1, 1 : -1 ]
#     b = phase_wrap[ 2 :  ,1:-1] - phase_wrap[ 1 : -1 ,1:-1]
#     atoolarge = np.where((np.abs(a) > np.pi) * (a > 0.))
#     btoolarge = np.where((np.abs(b) > np.pi) * (b > 0.))
#     atoosmall = np.where((np.abs(a) > np.pi) * (a < 0.))
#     btoosmall = np.where((np.abs(b) > np.pi) * (b < 0.))
#     a[atoolarge] = a[atoolarge] - 2*np.pi
#     b[btoolarge] = b[btoolarge] - 2*np.pi
#     a[atoosmall] = a[atoosmall] + 2*np.pi
#     b[btoosmall] = b[btoosmall] + 2*np.pi
#     slope = np.asarray([a,b])
#     return slope

def deming(x,y):
    """
    RETURN best fit parameter based on Deming orthogonal regression with delta=1.

    Parameters
    ----------
    x : x parameter
    y : y parameter

    Returns
    -------
    y = b0 + b1*x
    b0 : intercept
    b1 : slope

    """
    delta = 1
    xb = np.mean(x)
    yb = np.mean(y)
    sxx = np.mean(x**2)
    sxy = np.mean(x*y)
    syy = np.mean(y**2)
    b1 = (syy - delta*sxx + np.sqrt((syy - delta*sxx)**2 + 4*delta*sxy**2))/(2*sxy)
    b0 = yb - b1*xb
    return b0, b1

###############################################################################
# No slope case ##########################################################
###############################################################################


zero_phase = np.zeros((64,64), dtype=float)
zero_slope = gslope(zero_phase)
if (np.isclose(zero_slope[0], 0, rtol=1e-12)
    and np.isclose(zero_slope[1], 0, rtol=1e-12)):
    print('PASS the zero phase test.')
else:
    print('FAIL the zero phase test.')

###############################################################################
##########################################################################
###############################################################################

###############################################################################
# Flat slope case ########################################################
###############################################################################

N = 64
REPEAT = 10000

# provided known slopes
y, x = np.meshgrid(np.arange(-N/2 + 0.5, N/2 + 0.5),
                   np.arange(-N/2 + 0.5, N/2 + 0.5))
PASS = True

# Compare known slope with Zernike tip tilt
Z_3 = zernikeArray(3,N)
pupil = Z_3[0]
tip = Z_3[1]
tilt = Z_3[2]
NORM = np.zeros((Z_3.shape[0]), dtype=float)
for i in np.arange(Z_3.shape[0]):
    NORM[i] = np.dot(Z_3[i].flatten(), Z_3[i].flatten())
PASS1 = True
PASS2 = True
PASS3 = True

INPUT = np.zeros((REPEAT, 2), dtype=float)
GTILT = np.zeros((REPEAT, 2), dtype=float)
ZTILT = np.zeros((REPEAT, 2), dtype=float)

for repeat in np.arange(REPEAT):
    flatslopex = (np.random.rand() - 0.5)*2 * 2*np.pi
    flatslopey = (np.random.rand() - 0.5)*2 * 2*np.pi
    regen_flat_phase = np.zeros((N, N), dtype=float)
    flat_a = np.zeros((Z_3.shape[0]), dtype=float)

    flat_phase = flatslopex*x + flatslopey*y
    flat_slope = gslope(flat_phase)

    for i in np.arange(Z_3.shape[0]):
        flat_a[i] = (np.dot((flat_phase*pupil).flatten(), Z_3[i].flatten())
                / NORM[i])
        regen_flat_phase = regen_flat_phase + flat_a[i]*Z_3[i]
    zernikeflatslopex = flat_a[2]*(4/N)
    zernikeflatslopey = flat_a[1]*(4/N)

    INPUT[repeat, : ] = flatslopex, flatslopey
    GTILT[repeat, : ] = flat_slope
    ZTILT[repeat, : ] = zernikeflatslopex, zernikeflatslopey

    if not (np.isclose(flat_slope[0], flatslopex, rtol=1e-12)
        and np.isclose(flat_slope[1], flatslopey, rtol=1e-12)):
        PASS = False

    if not np.isclose(regen_flat_phase, flat_phase*pupil, rtol=1e-12).all():
        PASS1 = False
    if not (np.isclose(zernikeflatslopex, flatslopex, rtol=1e-12)
        and np.isclose(zernikeflatslopey, flatslopey, rtol=1e-12)):
        PASS2 = False

    if not (np.isclose(flat_slope[0], zernikeflatslopex, rtol=1e-12)
            and np.isclose(flat_slope[1], zernikeflatslopey, rtol=1e-12)):
        PASS3 = False

if PASS == True:
    print('PASS the flat phase test.')
else:
    print('FAIL the flat phase test.')

if PASS1 == True:
    print('PASS the Zernike regeneration')
else:
    print('FAIL the Zernike regeneration')
if PASS2 == True:
    print('PASS the Zernike flat slope test')
else:
    print('FAIL the Zernike flat slope test')

if PASS3 == True:
    print('PASS the Z-tilt and g-tilt similarity in flat slope test')
else:
    print('FAIL the Z-tilt and g-tilt similarity in flat slope test')

plt.plot(GTILT[:,0] - ZTILT[:,0], label='x', ls='', marker='.', alpha=0.1)
plt.plot(GTILT[:,1] - ZTILT[:,1], label='y', ls='', marker='.', alpha=0.1)
plt.title('Residue of tip/tilt in flat slope case')
plt.xlabel('Trial numbers')
plt.ylabel('G-tilt - Z-tilt per pixel (radian/px)')
plt.legend()
plt.show()

###############################################################################
# Kolmogorov phase case ##################################################
###############################################################################

r0 = 0.3
delta = r0/N
L0 = 20
l0 = 1e-7
oversize = 1
Zernike_MODES = int(np.floor((N/4)*(N/4 + 1)/2))
PASS3 = True
PASS4 = True
Z = zernikeArray(Zernike_MODES, N)
pupil = Z[0]
COUNT_ZERNIKE = int(0)
COUNT_G = int(0)
NORM = np.dot(pupil.flatten(), pupil.flatten())
tol = 0.2*(2*np.pi)/N
REPEAT = 10000

x1 = np.zeros((REPEAT), dtype=float)
x2 = np.zeros((REPEAT), dtype=float)
y1 = np.zeros((REPEAT), dtype=float)
y2 = np.zeros((REPEAT), dtype=float)

for repeat in np.arange(REPEAT):
    kolmogorov_a = np.zeros((Z.shape[0]), dtype=float)
    regen_kolmogorov_phase = np.zeros((N, N), dtype=float)
    
    kolmogorov_phase = mprop.crop2(
        ft_sh_phase_screen(r0, N*oversize, delta, L0, l0), oversize)
    kolmogorov_phase = kolmogorov_phase - kolmogorov_phase[int(N/2 - 1) : int(N/2 + 1),
                                                           int(N/2 - 1) : int(N/2 + 1)].mean()
    TOL = 0.2*(kolmogorov_phase*pupil).std()

    for i in np.arange(Z.shape[0]):
        kolmogorov_a[i] = (np.dot((kolmogorov_phase*pupil).flatten(), Z[i].flatten())
                / np.dot(Z[i].flatten(), Z[i].flatten()))
        regen_kolmogorov_phase = regen_kolmogorov_phase + kolmogorov_a[i]*Z[i]

    residue_Z = regen_kolmogorov_phase - kolmogorov_phase*pupil
    if residue_Z.std() > TOL:
        PASS3 = False

    # zernikekolmogorovslopex = kolmogorov_a[2]*(4/N)
    # zernikekolmogorovslopey = kolmogorov_a[1]*(4/N)
    zernikekolmogorovslopex = np.dot(kolmogorov_phase.flatten(), x.flatten())/np.dot(x.flatten(),x.flatten())
    zernikekolmogorovslopey = np.dot(kolmogorov_phase.flatten(), y.flatten())/np.dot(y.flatten(),y.flatten())

    kolmogorovslope = gslope(kolmogorov_phase)
    kolmogorovslopex = kolmogorovslope[0]
    kolmogorovslopey = kolmogorovslope[1]
    x1[repeat] = kolmogorovslopex
    x2[repeat] = zernikekolmogorovslopex
    y1[repeat] = kolmogorovslopey
    y2[repeat] = zernikekolmogorovslopey

    if((kolmogorovslopex*x + kolmogorovslopey*y - kolmogorov_phase).std()
          > (zernikekolmogorovslopex*x + zernikekolmogorovslopey*y - kolmogorov_phase).std()):
        COUNT_ZERNIKE = COUNT_ZERNIKE + 1
    else:
        COUNT_G = COUNT_G + 1

if PASS3 == True:
    print('PASS the Zernike regeneration')
else:
    print('FAIL the Zernike regeneration')

print('Zernike is {:d} occurence better.\nG is {:d} occurnece better.'.format(COUNT_ZERNIKE, COUNT_G))

Z_TILT = np.concatenate([x1,y1])
G_TILT = np.concatenate([x2,y2])
MAX = np.max([Z_TILT.max(),G_TILT.max()])
MIN = np.min([Z_TILT.min(),G_TILT.min()])

b0, b1 = deming(Z_TILT, G_TILT)
error = np.std(G_TILT - (b0 + b1*Z_TILT), ddof=1)

plt.plot(Z_TILT, G_TILT, ls='', marker='.', alpha=0.5)
plt.plot(np.linspace(MIN, MAX), np.linspace(MIN, MAX),ls='-',marker='', label=('1:1'))
plt.plot(np.linspace(MIN, MAX), b0 + b1*np.linspace(MIN, MAX),ls='-',marker='', label=('best fit'))
plt.legend()
plt.xlabel('Z-TILT (rad/px)')
plt.ylabel('G-TILT (rad/px)')
plt.title('Comparison of measurement of Kolmogorov slope\nintercept={:.3f}, slope={:.3f}'.format(b0,b1))
plt.show()

plt.plot(Z_TILT, ((b0 + b1*Z_TILT) - G_TILT)/error, ls='', marker='.', alpha=0.05)
plt.plot(np.linspace(MIN, MAX,100), np.zeros((100)),ls='-',marker='')
plt.title('Residue between measurements of Kolmogorov slope\n($\sigma$={:.3f} rad/px)'.format(error))
plt.xlabel('Z-TILT (rad/px)')
plt.ylabel('Z-TILT - G-TILT ($\sigma$)')
plt.show()

CORRELATION = (np.correlate(Z_TILT,G_TILT)[0]
      / np.sqrt((Z_TILT**2).sum() * (G_TILT**2).sum()))
if CORRELATION > 0.9:
    PASS4 == True
else:
    PASS4 == False

if ((1 - (((Z_TILT - G_TILT)**2).sum() / ((Z_TILT - Z_TILT.mean())**2).sum())) > 0.9
    and (1 - (((Z_TILT - G_TILT)**2).sum() / ((G_TILT - G_TILT.mean())**2).sum())) > 0.9):
    PASS4 == True
else:
    PASS4 == False
    
if PASS4 == True:
    print('PASS the Zernike kolmogorov slope test')
else:
    print('FAIL the Zernike kolmogorov slope test')

# plt.hist((b0 + b1*Z_TILT - G_TILT)/error,bins=100)
# plt.title('error distribution')
# plt.xlabel('sigma')
# plt.ylabel('occurences')
# plt.show()

