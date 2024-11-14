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

# def gslope(phase):
#     """
#     Measure different in phase between each pixel.

#     Parameters
#     ----------
#     phase : np.ndarray
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
#     difx = phase[1:,:] - phase[:-1,:]
#     dify = phase[:,1:] - phase[:,:-1]
#     slope = np.array([np.mean(difx), np.mean(dify)], dtype=float)
#     return slope

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
    difx = phase_wrap[1:,:] - phase_wrap[:-1,:]
    dify = phase_wrap[:,1:] - phase_wrap[:,:-1]
    difxtoolarge = np.where((np.abs(difx) > np.pi) * (difx > 0.))
    difytoolarge = np.where((np.abs(dify) > np.pi) * (dify > 0.))
    difxtoosmall = np.where((np.abs(difx) > np.pi) * (difx < 0.))
    difytoosmall = np.where((np.abs(dify) > np.pi) * (dify < 0.))
    difx[difxtoolarge] = difx[difxtoolarge] - 2*np.pi
    dify[difytoolarge] = dify[difytoolarge] - 2*np.pi
    difx[difxtoosmall] = difx[difxtoosmall] + 2*np.pi
    dify[difytoosmall] = dify[difytoosmall] + 2*np.pi
    slope = np.array([difx.mean(), dify.mean()], dtype=float)
    return slope

###############################################################################
# No slope case ##########################################################
###############################################################################

N = np.random.randint(16,128)

zero_phase = np.zeros((64,64), dtype=float)
zero_slope = gslope_wrap(np.angle(np.exp(1j*zero_phase)))
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

# provided known slopes
y, x = np.meshgrid(np.arange(N), np.arange(N))
flatslopex = np.random.rand()*np.pi
flatslopey = np.random.rand()*np.pi
flat_phase = flatslopex*x + flatslopey*y
flat_slope = gslope_wrap(np.angle(np.exp(1j*flat_phase)))
if (np.isclose(flat_slope[0], flatslopex, rtol=0.001)
    and np.isclose(flat_slope[1], flatslopey, rtol=0.001)):
    print('PASS the flat phase test.')
else:
    print('FAIL the flat phase test.')

# Compare known slope with Zernike tip tilt
Z_3 = zernikeArray(3,N)
pupil = Z_3[0]
flat_a = np.zeros((Z_3.shape[0]), dtype=float)
regen_flat_phase = np.zeros((N, N), dtype=float)

for i in np.arange(Z_3.shape[0]):
    flat_a[i] = (np.dot((flat_phase*pupil).flatten(), Z_3[i].flatten())
            / np.dot(Z_3[i].flatten(), Z_3[i].flatten()))
    regen_flat_phase = regen_flat_phase + flat_a[i]*Z_3[i]

if np.isclose(regen_flat_phase, flat_phase*pupil, rtol=0.001).all() == True:
    print('PASS the Zernike regeneration')
else:
    print('FAIL the Zernike regeneration')

zernikeflatslopex = flat_a[2]*(2 / (N/2))
zernikeflatslopey = flat_a[1]*(2 / (N/2))

if (np.isclose(zernikeflatslopex, flatslopex, rtol=0.001)
    and np.isclose(zernikeflatslopey, flatslopey, rtol=0.001)):
    print('PASS the Zernike flat slope test')
else:
    print('FAIL the Zernike flat slope test')

# plt.pcolor(x,y,Z[0])
# plt.colorbar()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('piston, pupil')
# plt.axis('square')
# plt.show()
# plt.clf()
# plt.pcolor(x,y,Z[1])
# plt.colorbar()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('tip, slope in y')
# plt.axis('square')
# plt.show()
# plt.clf()
# plt.pcolor(x,y,Z[2])
# plt.colorbar()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('tilt, slope in x')
# plt.axis('square')
# plt.show()
# plt.clf()

# plt.imshow(flat_phase*Z[0])
# plt.colorbar()
# plt.title('flat_phase')
# plt.show()
# plt.clf()
# plt.imshow(regen_flat_phase)
# plt.colorbar()
# plt.title('regen_flat_phase')
# plt.show()
# plt.clf()
# plt.imshow(flat_phase*Z[0] - regen_flat_phase)
# plt.colorbar()
# plt.title('residue')
# plt.show()
# plt.clf()

###############################################################################
# Random phase case ######################################################
###############################################################################

r0 = np.random.rand()*(0.3 - 0.05) + 0.05
delta = 10*r0/N
L0 = 20
l0 = 1e-7
oversize = 16
Zernike_MODES = int(np.floor((N/4)*(N/4 + 1)/2))
kolmogorov_phase = mprop.crop2(
    ft_sh_phase_screen(r0, N*oversize, delta, L0, l0), oversize)

Z = zernikeArray(Zernike_MODES, N)
pupil = Z[0]
kolmogorov_a = np.zeros((Z.shape[0]), dtype=float)
regen_kolmogorov_phase = np.zeros((N, N), dtype=float)
TOL = (kolmogorov_phase[:,:].max() - kolmogorov_phase[:,:].min()) * 0.2

for i in np.arange(Z.shape[0]):
    kolmogorov_a[i] = (np.dot((kolmogorov_phase*pupil).flatten(), Z[i].flatten())
            / np.dot(Z[i].flatten(), Z[i].flatten()))
    regen_kolmogorov_phase = regen_kolmogorov_phase + kolmogorov_a[i]*Z[i]

residue = regen_kolmogorov_phase - kolmogorov_phase*Z[0]

if residue.std() <= TOL:
    print('PASS the Zernike regeneration')
else:
    print('FAIL the Zernike regeneration')

zernikekolmogorovslopex = kolmogorov_a[2]*(2 / (N/2))
zernikekolmogorovslopey = kolmogorov_a[1]*(2 / (N/2))

kolmogorovslope = gslope_wrap(np.angle(np.exp(1j*kolmogorov_phase)))
kolmogorovslopex = kolmogorovslope[0]
kolmogorovslopey = kolmogorovslope[1]

dif_zernike_gslope = ((kolmogorovslopex - zernikekolmogorovslopex)*x
                      + (kolmogorovslopey - zernikekolmogorovslopey)*y).std()

if dif_zernike_gslope <= TOL:
    print('PASS the Zernike kolmogorov slope test')
else:
    print('FAIL the Zernike kolmogorov slope test')

# plt.imshow(np.angle(np.exp(1j*kolmogorov_phase)))
# plt.colorbar()
# plt.title('wrap slope')
# plt.show()
# plt.clf()

# plt.imshow(kolmogorov_phase*Z[0])
# plt.colorbar()
# plt.title('kolmogorov_phase')
# plt.show()
# plt.clf()
# plt.imshow(regen_kolmogorov_phase)
# plt.colorbar()
# plt.title('regen_kolmogorov_phase')
# plt.show()
# plt.clf()
# plt.imshow(residue, vmin=-TOL, vmax=TOL)
# plt.colorbar()
# plt.title('residue')
# plt.show()
# plt.clf()


