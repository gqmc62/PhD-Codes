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

def gslope_wrap(phase_wrap):
    """
    Measure different in phase between each pixel, for wrapped phase.

    Parameters
    ----------
    phase_wrap : np.ndarray
        (N,N)
        (float, float)
        unit in radian

    Returns
    -------
    slope : np.ndarray
        (N - 1, N - 1)
        (float, float)
        unit in radian

    """
    a = phase_wrap[1 :   , 1 : -1] - phase_wrap[  : -1, 1 : -1]
    b = phase_wrap[1 : -1, 1 :   ] - phase_wrap[1 : -1,   : -1]
    atoolarge = np.where((np.abs(a) > np.pi) * (a > 0.))
    btoolarge = np.where((np.abs(b) > np.pi) * (b > 0.))
    atoosmall = np.where((np.abs(a) > np.pi) * (a < 0.))
    btoosmall = np.where((np.abs(b) > np.pi) * (b < 0.))
    a[atoolarge] = a[atoolarge] - 2*np.pi
    b[btoolarge] = b[btoolarge] - 2*np.pi
    a[atoosmall] = a[atoosmall] + 2*np.pi
    b[btoosmall] = b[btoosmall] + 2*np.pi
    slope = np.asarray([np.mean(a),np.mean(b)])
    return slope

def gslope_wrap_px(phase_wrap):
    """
    Measure different in phase between each pixel, for wrapped phase.

    Parameters
    ----------
    phase_wrap : np.ndarray
        (N,N)
        (float, float)
        unit in radian

    Returns
    -------
    slope : np.ndarray
        (N - 1, N - 1)
        (float, float)
        unit in radian

    """
    a = phase_wrap[ 2 :  ,1:-1] - phase_wrap[ 1 : -1 ,1:-1]
    b = phase_wrap[1:-1, 2 :  ] - phase_wrap[1:-1, 1 : -1 ]
    atoolarge = np.where((np.abs(a) > np.pi) * (a > 0.))
    btoolarge = np.where((np.abs(b) > np.pi) * (b > 0.))
    atoosmall = np.where((np.abs(a) > np.pi) * (a < 0.))
    btoosmall = np.where((np.abs(b) > np.pi) * (b < 0.))
    a[atoolarge] = a[atoolarge] - 2*np.pi
    b[btoolarge] = b[btoolarge] - 2*np.pi
    a[atoosmall] = a[atoosmall] + 2*np.pi
    b[btoosmall] = b[btoosmall] + 2*np.pi
    slope = np.asarray([np.mean(a),np.mean(b)])
    return slope

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
zero_slope = gslope_wrap(zero_phase)
if (np.isclose(zero_slope[0], 0, rtol=0.001)
    and np.isclose(zero_slope[1], 0, rtol=0.001)):
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
PASS1 = True

unwrap_flat_slope = np.zeros((2), dtype=float)

INPUT = np.zeros((REPEAT, 2), dtype=float)
WRAP = np.zeros((REPEAT, 2), dtype=float)
UNWRAP = np.zeros((REPEAT, 2), dtype=float)

for repeat in np.arange(REPEAT):
    flatslopex = (np.random.rand() - 0.5)*2 * np.pi
    flatslopey = (np.random.rand() - 0.5)*2 * np.pi

    flat_phase = flatslopex*x + flatslopey*y
    flat_slope = gslope(flat_phase)
    wrap_flat_slope = gslope_wrap(np.angle(np.exp(1j*flat_phase)))

    INPUT[repeat, : ] = flatslopex, flatslopey
    UNWRAP[repeat, : ] = flat_slope
    WRAP[repeat, : ] = wrap_flat_slope

    if not (np.isclose(flatslopex, wrap_flat_slope[0], rtol=1e-6)
            and np.isclose(flatslopey, wrap_flat_slope[1], rtol=1e-6)):
        PASS = False

    if not (np.isclose(flat_slope[0], wrap_flat_slope[0], rtol=1e-6)
        and np.isclose(flat_slope[1], wrap_flat_slope[1], rtol=1e-6)):
        PASS1 = False

if PASS == True:
    print('PASS the flat phase test.')
else:
    print('FAIL the flat phase test.')

if PASS1 == True:
    print('PASS the wrap vs unwrap g-tilt similarity in flat slope test')
else:
    print('FAIL the wrap vs unwrap g-tilt similarity in flat slope test')

plt.plot(UNWRAP[:,0] - WRAP[:,0], label='x', ls='', marker='.', alpha=0.1)
plt.plot(UNWRAP[:,1] - WRAP[:,1], label='y', ls='', marker='.', alpha=0.1)
plt.title('Residue of tip/tilt in flat slope case')
plt.xlabel('Trial numbers')
plt.ylabel('UNWRAP - WRAP per pixel (radian/px)')
plt.legend()
plt.show()

###############################################################################
# Random phase case ######################################################
###############################################################################

r0 = 0.1
delta = r0/4
L0 = 20
l0 = 1e-7
oversize = 1
Zernike_MODES = int(np.floor((N/4)*(N/4 + 1)/2))
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
    kolmogorov_phase = mprop.crop2(
        ft_sh_phase_screen(r0, N*oversize, delta, L0, l0), oversize)
    kolmogorov_phase = kolmogorov_phase - kolmogorov_phase[int(N/2 - 1) : int(N/2 + 1),
                                                           int(N/2 - 1) : int(N/2 + 1)].mean()
    TOL = 0.2*(kolmogorov_phase*pupil).std()

    kolmogorovslope = gslope(kolmogorov_phase)
    wrapkolmogorovslope = gslope_wrap(np.angle(np.exp(1j*kolmogorov_phase)))

    x1[repeat] = kolmogorovslope[0]
    y1[repeat] = kolmogorovslope[1]
    x2[repeat] = wrapkolmogorovslope[0]
    y2[repeat] = wrapkolmogorovslope[1]

unwrap_TILT = np.concatenate([x1,y1])
wrap_TILT = np.concatenate([x2,y2])

MAX = np.max([wrap_TILT.max(),unwrap_TILT.max()])
MIN = np.min([wrap_TILT.min(),unwrap_TILT.min()])

b0, b1 = deming(wrap_TILT,unwrap_TILT)
error = np.std((b0 + b1*wrap_TILT) - unwrap_TILT, ddof=1)

plt.plot(wrap_TILT, unwrap_TILT, ls='', marker='.', alpha=0.5)
plt.plot(np.linspace(MIN, MAX), np.linspace(MIN, MAX),ls='-',marker='',label='1:1')
plt.plot(np.linspace(MIN, MAX), b0 + b1*np.linspace(MIN, MAX),ls='-',marker='',label='best fit')
plt.legend()
plt.xlabel('wrap-TILT (rad/px)')
plt.ylabel('unwrap-TILT (rad/px)')
plt.title('Comparison of measurement of Kolmogorov slope\nintercept={:.3f}, slope={:.3f}'.format(b0,b1))
plt.show()

plt.plot(unwrap_TILT, ((b0 + b1*wrap_TILT) - unwrap_TILT)/error, ls='', marker='.', alpha=0.5)
plt.plot(np.linspace(MIN, MAX,100), np.zeros((100)),ls='-',marker='')
plt.title('Residue between measurements of Kolmogorov slope')
plt.xlabel('unwrap-TILT (rad/px)')
plt.ylabel('wrap-TILT - unwrap-TILT ($\sigma$={:.3f} rad/px)'.format(error))
plt.show()

CORRELATION = (np.correlate(wrap_TILT,unwrap_TILT)[0]
      / np.sqrt((wrap_TILT**2).sum() * (unwrap_TILT**2).sum()))
if CORRELATION > 0.9:
    PASS4 == True
else:
    PASS4 == False

if ((1 - (((wrap_TILT - unwrap_TILT)**2).sum() / ((wrap_TILT - unwrap_TILT.mean())**2).sum())) > 0.9
    and (1 - (((wrap_TILT - unwrap_TILT)**2).sum() / ((wrap_TILT - unwrap_TILT.mean())**2).sum())) > 0.9):
    PASS4 == True
else:
    PASS4 == False
    
if PASS4 == True:
    print('PASS the wrap kolmogorov slope test')
else:
    print('FAIL the wrap kolmogorov slope test')

# plt.hist((wrap_TILT - unwrap_TILT)/error,bins=100)
# plt.title('error distribution')
# plt.xlabel('sigma')
# plt.ylabel('occurences')
# plt.show()

#how about deming fit to the comparison?