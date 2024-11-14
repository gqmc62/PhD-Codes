# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:05:43 2024

@author: gqmc62
"""

import numpy as np
from matplotlib import pyplot as plt
from aotools.turbulence.infinitephasescreen import PhaseScreenVonKarman

def cn22r0(cn2dz,wvl=500e-9):
    r0 = (0.423*(2.*np.pi/wvl)**2*cn2dz)**(-3./5.)
    return r0

def r02cn2(r0,wvl=500e-9):
    cn2dz = r0**(-5./3.)/(0.423*(2.*np.pi/wvl)**2)
    return cn2dz

def remove_piston_tiptilt(A):
    A -= A.mean()
    x = np.arange(A.shape[0],dtype=float)
    x -= x.mean()
    yy,xx = np.meshgrid(x,x)
    A -= xx*(A*xx).sum()/(xx**2).sum()
    A -= yy*(A*yy).sum()/(yy**2).sum()
    return A
    

D = 4.
n = 8
px_persubap = 16
d = D/n
delta = d/px_persubap

theta = 60./206265.

h = d / theta
H = D / theta

# h1 = 3*h
# h2 = 6*h

N = int(2.*D/delta)

wvl = 500e-9
L0 = 20.
r0_sum = 0.1
cn2_sum = r02cn2(0.1,wvl)
cn2_distr = np.array([0.80,0.05,0.15],dtype=float)
cn2 = cn2_sum*cn2_distr
r0 = cn22r0(cn2)

PSH0 = PhaseScreenVonKarman(N, delta, r0[0], L0)
PSH5 = PhaseScreenVonKarman(N, delta, r0[1], L0)
PSH10 = PhaseScreenVonKarman(N, delta, r0[2], L0)

for i in np.arange(N):
    PSH0.add_row()
    PSH5.add_row()
    PSH10.add_row()

H0 = remove_piston_tiptilt(PSH0.scrn)
H5 = remove_piston_tiptilt(PSH5.scrn)
H10 = remove_piston_tiptilt(PSH10.scrn)


centre = N//2
left = N//2 - N//4
right = N//2 + N//4


WFS_left_scrns = np.zeros((3,N//2,N//2),dtype=float)
WFS_right_scrns = np.zeros((3,N//2,N//2),dtype=float)

WFS_left_scrns[0] = H0[left : right, left : right]
WFS_right_scrns[0] = H0[left : right, left : right]

WFS_left_scrns[1] = H5[left - 3*px_persubap//2 : right - 3*px_persubap//2, left : right]
WFS_right_scrns[1] = H5[left + 3*px_persubap//2 : right + 3*px_persubap//2, left : right]

WFS_left_scrns[2] = H5[left - 6*px_persubap//2 : right - 6*px_persubap//2, left : right]
WFS_right_scrns[2] = H5[left + 6*px_persubap//2 : right + 6*px_persubap//2, left : right]


WFS_left = np.zeros((N,N),dtype=float)
WFS_right = np.zeros((N,N),dtype=float)
WFS_left[left:right,left:right] = WFS_left_scrns.sum(0)
WFS_right[left:right,left:right] = WFS_right_scrns.sum(0)

plt.imshow(WFS_left)
plt.title('left WFS')
plt.colorbar()
plt.show()

plt.imshow(WFS_right)
plt.title('right WFS')
plt.colorbar()
plt.show()


guess_H0 = np.zeros((N,N),dtype=float)


plt.imshow(H0)
plt.title('H0')
plt.colorbar()
plt.show()


guess_H0[left:right,left:right] += WFS_left[left : right , left : right]
guess_H0[left:right,left:right] += WFS_right[left  : right , left : right]
guess_H0[left:right,left:right] /= 2.
# guess_H0 /= 2.
# guess_H0 = remove_piston_tiptilt(guess_H0)
plt.imshow(guess_H0,vmin=H0.min(),vmax=H0.max())
plt.title('guess H0')
plt.colorbar()
plt.show()

RES_left = (WFS_left-guess_H0)
RES_right = (WFS_right-guess_H0)
# RES_left = remove_piston_tiptilt(WFS_left-guess_H0)
# RES_right = remove_piston_tiptilt(WFS_right-guess_H0)

plt.imshow(RES_left)
plt.title('left WFS res')
plt.colorbar()
plt.show()

plt.imshow(RES_right)
plt.title('right WFS res')
plt.colorbar()
plt.show()

plt.imshow(H5)
plt.title('H5')
plt.colorbar()
plt.show()

guess_H5 = np.zeros((N,N),dtype=float)
shift_subap = 3
shift_pxl = shift_subap*px_persubap//2
guess_H5[left-shift_pxl:right-shift_pxl,left:right] += (RES_left)[left : right , left : right]
guess_H5[left+shift_pxl:right+shift_pxl,left:right] += (RES_right)[left : right, left : right]
guess_H5[left+shift_pxl:right-shift_pxl,left:right] /= 2.
# guess_H5 = remove_piston_tiptilt(guess_H5)

plt.imshow(guess_H5,vmin=H5.min(),vmax=H5.max())
plt.title('guess H5')
plt.colorbar()
plt.show()

# RES_left = remove_piston_tiptilt(WFS_left-guess_H0-guess_H5)
# RES_right = remove_piston_tiptilt(WFS_right-guess_H0-guess_H5)

plt.imshow(RES_left)
plt.title('left WFS res')
plt.colorbar()
plt.show()

plt.imshow(RES_right)
plt.title('right WFS res')
plt.colorbar()
plt.show()

plt.imshow(H10)
plt.title('H10')
plt.colorbar()
plt.show()

guess_H10 = np.zeros((N,N),dtype=float)
shift_subap = 6
shift_pxl = shift_subap*px_persubap//2
guess_H10[left-shift_pxl:right-shift_pxl,left:right] += (RES_left)[left : right , left : right]
guess_H10[left+shift_pxl:right+shift_pxl,left:right] += (RES_right)[left : right, left : right]
guess_H10[left+shift_pxl:right-shift_pxl,left:right] /= 2.
# guess_H10 = remove_piston_tiptilt(guess_H10)

plt.imshow(guess_H10,vmin=H10.min(),vmax=H10.max())
plt.title('guess H10')
plt.colorbar()
plt.show()
