# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:36:31 2023

selecting propagation parameter

@author: gqmc62
"""

import numpy as np
from aotools.turbulence.infinitephasescreen import PhaseScreenVonKarman

ARCSEC2RAD = (1/3600)*(2*np.pi/360)

def schmidt1(wvl, z, D1, delta1, D2, delta2):
    a = wvl*z/delta1/delta2
    b = D1/delta1 + D2/delta2
    cond = ((a <= b) or np.isclose(a, b))
    if cond == False:
        print (a, b)
    return cond

def schmidt2(N, wvl, z, D1, delta1, delta2):
    a = N
    b = wvl*z/2/delta1/delta2 + D1/delta1
    cond = ((a >= b) or np.isclose(a, b))
    if cond == False:
        print (a, b)
    return cond

def schmidt3(N, wvl, z, delta1, delta2):
    a = N
    b = wvl*z/delta1/delta2
    cond = ((a >= b) or np.isclose(a, b))
    if cond == False:
        print (a, b)
    return cond

def D_FoV(pupil, FoV, z):
    D = pupil + FoV*z*ARCSEC2RAD
    return D

def solve_3(wvl, z, delta2):
    print('N*delta1 >= {:.3e}'.format(wvl*z/delta2))

def solve_2(wvl, z, delta2):
    print('N*delta1 >= {:.3e} + D1'.format(wvl*z/2/delta2))

def solve_1(wvl, z, D2, delta2):
    print('{:.3e} >= D1 + {:.3e}*delta1'.format(wvl*z/delta2, D2/delta2))

def perfect_FoV(wvl,d,r0,px):
    FWHM = 0
    if d > r0:
        FWHM = wvl/r0/px
    if r0 > d:
        FWHM = wvl/wvl/px
    drift = 4*np.sqrt(0.182*(wvl/d)**2*(d/r0)**(5/3.))/px
    room = 4*FWHM
    FoV = FWHM + room + drift
    return FoV

def param_calc_fix_linear_scale(wvl, z, pupil, pixel_scale, FoV):
    # wvl = float(wvl) # m
    # z = np.asarray(z, dtype=float) # m (from low to high z)
    # pupil = float(pupil)
    # delta1 = float(pixel_scale)
    # delta2 = float(pixel_scale)
    # FoV = float(FoV) # "
    
    delta1 = pixel_scale
    delta2 = pixel_scale
    
    n_prop = z.shape[0] - 1
    
    D = np.zeros((len(pixel_scale), len(z)), dtype=float)
    N = np.zeros((len(pixel_scale), len(z)), dtype=int)
    
    D[ : , 0] = pupil
    # delta[0] = pixel_scale
    N[ : , 0] = np.asarray(2**np.ceil(np.log(D[ : , 0]/pixel_scale)/np.log(2)), dtype=int)
    for i in np.arange(len(pixel_scale)):
        for j in np.arange(n_prop):
            dz = z[j + 1] - z[j]
            
            # delta2 = delta[i]
            D2 = D[i, j]
            
            # D1 = max(D_turb(pupil, FoV, z[i]), wvl*dz/delta2/2 + D2)
            D1 = wvl*dz/delta2[i]/2 + D2
            # delta1 = delta[i + 1]
            
            N_temp = max(D1/delta1[i] + wvl*dz/2/delta1[i]/delta2[i],
                    D2/delta2[i] + wvl*dz/2/delta1[i]/delta2[i],
                    wvl*dz/delta1[i]/delta2[i])
            
            if (schmidt1(wvl, dz, D1, delta1[i], D2, delta2[i])
                or schmidt2(N_temp, wvl, dz, D1, delta1[i], delta2[i])
                or schmidt3(N_temp, wvl, dz, delta1[i], delta2[i])) == False:
                print('fail')
                
            D[i, j + 1] = D1
            N[i, j + 1] = int(2**np.ceil(np.log(N_temp)/np.log(2)))
    
    return N, D # , delta


r0 = np.array([11, 8, 5, 2], dtype=float)*1e-2
pxXsub = 16 # px
pixel_scale = r0/pxXsub

L0 = float(20)
wvl = 500e-9
z = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40], dtype=float)*1e3
pupil = 1.5 # m
FoV = 30 # "

N, D = param_calc_fix_linear_scale(wvl, z, pupil, pixel_scale, FoV)

# print(D_FoV(pupil, FoV, z) / D)

Ngen = np.zeros_like(N)

for i in np.arange(len(r0)):
    Ngen[i] = 2**np.ceil(np.log((D[i] - pupil + D_FoV(pupil, FoV, z))/pixel_scale[i])/np.log(2))
# print(z)
# print(N) # use this big of phase screen to propagate
# print(N/N[0])
# print(Ngen) # generate this big of phase screen
# print(Ngen/N[0])



#scrn = PhaseScreenVonKarman(nx_size, pixel_scale, r0, L0)


