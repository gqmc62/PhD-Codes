# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 16:05:41 2025

@author: gqmc62
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,2*np.pi)
dx = x[1] - x[0]
shift = 2*np.pi/5.
shift_px = shift/dx

a = np.sin(x)
b = np.sin(x + shift)

plt.plot(x,a,label='a')
plt.plot(x,b,label='b')
plt.legend()
plt.show()


##############################################################################

corr_ab = np.fft.fftshift(np.fft.ifft(
    np.fft.fft(a)
    * np.conjugate(np.fft.fft(b))).real)

plt.plot(x,a,label='a')
plt.plot(x,b,label='b')
plt.plot(x,corr_ab/corr_ab.max(),label='corr')
plt.legend()
plt.show()

loc = np.where(corr_ab==corr_ab.max())[0] - corr_ab.shape[0]//2

plt.plot(x,a,label='a')
plt.plot(x + loc*dx,b,label='b + shift')
plt.legend()
plt.show()
  #############################################################################

x = np.linspace(0,2*np.pi)
dx = x[1] - x[0]
shift = np.pi/5.
shift_px = shift/dx

a = np.sin(x)
b = np.sin(x + shift)

aa = np.pad(a,(a.shape[0]//2,a.shape[0]//2),mode='constant')
bb = np.pad(b,(b.shape[0]//2,b.shape[0]//2),mode='constant')

mask = np.ones((aa.shape[0]),dtype=float)
mask[:a.shape[0]//2] = 0
mask[a.shape[0]//2*3:] = 0

plt.plot(aa,label='aa')
plt.plot(bb,label='bb')
plt.plot(mask,label='mask')
plt.legend()
plt.show()



corr_aabb = np.fft.fftshift(np.fft.ifft(
    np.fft.fft(aa)
    * np.conjugate(np.fft.fft(bb))).real)

plt.plot(aa,label='aa')
plt.plot(bb,label='bb')
plt.plot(mask,label='mask')
plt.plot(corr_aabb,label='corr')
plt.legend()
plt.show()


loc = np.where(corr_aabb==corr_aabb.max())[0] - corr_aabb.shape[0]//2
x_lazy = np.arange(aa.shape[0])

plt.plot(x_lazy,aa,label='aa')
plt.plot(x_lazy + loc,bb,label='bb')
# plt.plot(mask,label='mask')
# plt.plot(corr_aabb,label='corr')
plt.legend()
plt.show()





corr_aabb = np.fft.fftshift(np.fft.ifft(
    np.fft.fft(aa)
    * np.conjugate(np.fft.fft(bb))).real)


###############################################################################
maskcorr = np.fft.fftshift(np.fft.ifft(np.abs(np.fft.fft(mask))**2))
idx = np.zeros_like(maskcorr,dtype=bool)
# idx[~np.isclose(maskcorr,0,atol=1e-3)] = True
idx[maskcorr!=0] = True

temp_corr = np.zeros_like(corr_aabb)
temp_corr[idx] = corr_aabb[idx] / maskcorr[idx] * mask[idx]
corr_aabb = temp_corr
###############################################################################


plt.plot(aa[a.shape[0]//2:a.shape[0]//2*3],label='aa')
plt.plot(bb[a.shape[0]//2:a.shape[0]//2*3],label='bb')
# plt.plot(mask,label='mask')
plt.plot(corr_aabb[a.shape[0]//2:a.shape[0]//2*3],label='corr')
plt.legend()
plt.show()


loc = np.where(corr_aabb==corr_aabb.max())[0] - corr_aabb.shape[0]//2
# x_lazy = np.arange(aa.shape[0])

plt.plot(x,aa[a.shape[0]//2:a.shape[0]//2*3],label='aa')
plt.plot(x + loc*dx,bb[a.shape[0]//2:a.shape[0]//2*3],label='bb+shift')
plt.legend()
plt.show()


##############################################################################


N = 50
dx = 2*np.pi/N
x = np.arange(N) * dx
Y,X = np.meshgrid(x,x)

shiftx = 0
shifty = np.pi/4

A = np.sin(X) + np.sin(Y)
B = np.sin(X + shiftx) + np.sin(Y + shifty)

plt.clf()
fig,axes = plt.subplots(2,1,sharex=True,sharey=True)
axes[0].pcolor(X,Y,A)
axes[0].set_title('A')
axes[0].set_aspect('equal')
axes[1].pcolor(X,Y,B)
axes[1].set_title('B')
axes[1].set_aspect('equal')
plt.tight_layout()
plt.show()


corr_AB = np.fft.fftshift(np.fft.ifft2(
    np.fft.fft2(A)
    * np.conjugate(np.fft.fft2(B)))).real

plt.clf()
fig,axes = plt.subplots(3,1,sharex=True,sharey=True)
axes[0].pcolor(X,Y,A.T)
axes[0].set_title('A')
axes[0].set_aspect('equal')
axes[1].pcolor(X,Y,B.T)
axes[1].set_title('B')
axes[1].set_aspect('equal')
axes[2].pcolor(X,Y,corr_AB.T)
axes[2].set_title('corr')
axes[2].set_aspect('equal')
plt.tight_layout()
plt.show()


locx = np.where(corr_AB==corr_AB.max())[1] - corr_AB.shape[0]//2
locy = np.where(corr_AB==corr_AB.max())[0] - corr_AB.shape[0]//2

plt.clf()
fig,axes = plt.subplots(2,1,sharex=True,sharey=True)
axes[0].pcolor(X,Y,A.T)
axes[0].set_title('A')
axes[0].set_aspect('equal')
axes[0].set_xlim(0,2*np.pi + locx*dx)
axes[0].set_ylim(0,2*np.pi + locy*dx)
axes[1].pcolor(X+locx*dx,Y+locy*dx,B.T)
axes[1].set_title('B')
axes[1].set_aspect('equal')
axes[1].set_xlim(0,2*np.pi + locx*dx)
axes[1].set_ylim(0,2*np.pi + locy*dx)

plt.tight_layout()
plt.show()

###############################################################################

N = 50
dx = 2*np.pi/N
x = np.arange(N) * dx
Y,X = np.meshgrid(x,x)

shiftx = 0#np.pi/4
shifty = np.pi/4

A = np.sin(X) + np.sin(Y)
B = np.sin(X + shiftx) + np.sin(Y + shifty)

plt.clf()
fig,axes = plt.subplots(2,1,sharex=True,sharey=True)
axes[0].pcolor(X,Y,A.T)
axes[0].set_title('A')
axes[0].set_aspect('equal')
axes[1].pcolor(X,Y,B.T)
axes[1].set_title('B')
axes[1].set_aspect('equal')
plt.tight_layout()
plt.show()


AA = np.pad(A,((A.shape[0]//2,A.shape[0]//2),(A.shape[1]//2,A.shape[1]//2)),mode='constant')
BB = np.pad(B,((A.shape[0]//2,A.shape[0]//2),(A.shape[1]//2,A.shape[1]//2)),mode='constant')


corr_AABB = np.fft.fftshift(np.fft.ifft2(
    np.fft.fft2(AA)
    * np.conjugate(np.fft.fft2(BB)))).real#[A.shape//2:A.shape//2*3,A.shape[0]//2:A.shape[0]//2*3]

# temp = corr_AABB


window = np.zeros_like(AA)
window[A.shape[0]//2:A.shape[0]//2*3,A.shape[0]//2:A.shape[0]//2*3] = 1.
maskcorr = np.fft.fftshift(np.fft.ifft2(
    np.abs(np.fft.fft2(window))**2))

idx = (maskcorr!=0)
temp = np.zeros_like(corr_AABB)
temp[idx] = corr_AABB[idx]/maskcorr[idx]*window[idx]


corr_AB = temp[A.shape[0]//2:A.shape[0]//2*3,A.shape[0]//2:A.shape[0]//2*3]

plt.clf()
fig,axes = plt.subplots(3,1,sharex=True,sharey=True)
axes[0].pcolor(X,Y,A.T)
axes[0].set_title('A')
axes[0].set_aspect('equal')
axes[1].pcolor(X,Y,B.T)
axes[1].set_title('B')
axes[1].set_aspect('equal')
axes[2].pcolor(X,Y,corr_AB.T)
axes[2].set_title('corr')
axes[2].set_aspect('equal')
plt.tight_layout()
plt.show()


locx = np.where(corr_AB==corr_AB.max())[1] - corr_AB.shape[0]//2
locy = np.where(corr_AB==corr_AB.max())[0] - corr_AB.shape[0]//2

plt.clf()
fig,axes = plt.subplots(2,1,sharex=True,sharey=True)
axes[0].pcolor(X,Y,A.T)
axes[0].set_title('A')
axes[0].set_aspect('equal')
axes[0].set_xlim(0,2*np.pi + locx*dx)
axes[0].set_ylim(0,2*np.pi + locy*dx)
axes[1].pcolor(X+locx*dx,Y+locy*dx,B.T)
axes[1].set_title('B')
axes[1].set_aspect('equal')
axes[1].set_xlim(0,2*np.pi + locx*dx)
axes[1].set_ylim(0,2*np.pi + locy*dx)


plt.tight_layout()
plt.show()

################################################################################

N = 50
dx = 1
x = np.arange(N) * dx
Y,X = np.meshgrid(x,x)

shiftx = 0#np.pi/4
shifty = 15

tx = np.linspace(-8,8,num=2*N)
tyy,txx = np.meshgrid(tx,tx)
template = np.sinc((txx**2+tyy**2)**0.5)**2

A = template[N//2 : 3*N//2,
             N//2 : 3*N//2]
B = template[N//2 + shiftx : 3*N//2 + shiftx,
             N//2 + shifty : 3*N//2 + shifty]

plt.clf()
fig,axes = plt.subplots(2,1,sharex=True,sharey=True)
axes[0].pcolor(X,Y,A.T)
axes[0].set_title('A')
axes[0].set_aspect('equal')
axes[1].pcolor(X,Y,B.T)
axes[1].set_title('B')
axes[1].set_aspect('equal')
plt.tight_layout()
plt.show()


AA = np.pad(A,((A.shape[0]//2,A.shape[0]//2),(A.shape[1]//2,A.shape[1]//2)),mode='constant')
BB = np.pad(B,((A.shape[0]//2,A.shape[0]//2),(A.shape[1]//2,A.shape[1]//2)),mode='constant')


corr_AABB = np.fft.fftshift(np.fft.ifft2(
    np.fft.fft2(AA)
    * np.conjugate(np.fft.fft2(BB)))).real#[A.shape//2:A.shape//2*3,A.shape[0]//2:A.shape[0]//2*3]

temp = corr_AABB


window = np.zeros_like(AA)
window[A.shape[0]//2:A.shape[0]//2*3,A.shape[0]//2:A.shape[0]//2*3] = 1.
maskcorr = np.fft.fftshift(np.fft.ifft2(
    np.abs(np.fft.fft2(window))**2)).real

idx = (maskcorr!=0)
temp = np.zeros_like(corr_AABB)
temp[idx] = (corr_AABB[idx]/maskcorr[idx]*window[idx])


corr_AB = temp[A.shape[0]//2:A.shape[0]//2*3,A.shape[0]//2:A.shape[0]//2*3]

plt.clf()
fig,axes = plt.subplots(3,1,sharex=True,sharey=True)
axes[0].pcolor(X,Y,A.T,vmin=A.min(),vmax=A.max())
axes[0].set_title('A')
axes[0].set_aspect('equal')
axes[1].pcolor(X,Y,B.T,vmin=A.min(),vmax=A.max())
axes[1].set_title('B')
axes[1].set_aspect('equal')
axes[2].pcolor(X,Y,corr_AB.T)
axes[2].set_title('corr')
axes[2].set_aspect('equal')
plt.tight_layout()
plt.show()


locx = np.where(corr_AB==corr_AB.max())[1] - corr_AB.shape[0]//2
locy = np.where(corr_AB==corr_AB.max())[0] - corr_AB.shape[0]//2

plt.clf()
fig,axes = plt.subplots(2,1,sharex=True,sharey=True)
axes[0].pcolor(X,Y,A.T,vmin=A.min(),vmax=A.max())
axes[0].set_title('A')
axes[0].set_aspect('equal')
axes[0].set_xlim(0,50 + locx*dx)
axes[0].set_ylim(0,50 + locy*dx)
axes[1].pcolor(X+locx*dx,Y+locy*dx,B.T,vmin=A.min(),vmax=A.max())
axes[1].set_title('B')
axes[1].set_aspect('equal')
axes[1].set_xlim(0,50 + locx*dx)
axes[1].set_ylim(0,50 + locy*dx)


plt.tight_layout()
plt.show()




