# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:22:32 2022.

@author: gqmc62
"""
from aotools.turbulence.phasescreen import ft_sh_phase_screen
import numpy as np
from matplotlib import pyplot as plt
import my_propagation as mprop
from aotools.opticalpropagation import angularSpectrum
import my_WFS as mwfs

color_map = plt.cm.get_cmap('Blues')
my_cmap = color_map.reversed()

N = int(1024)
pixel_scale = float(5e-3)
lenslet_pixel = int(16)
r0 = float(0.05)
L0 = float(20)
l0 = float(1.5e-10)
wvl = float(500e-9)
pf = 2
NN = N*pf
z = float(25000)

prNN = int(8) # lenslet
prN = prNN*lenslet_pixel
n = int(N/lenslet_pixel)
CEN = int(N/2)
cen = int(n/2)

I_in_big = np.ones((NN,NN), dtype=float)
phase_in_big = ft_sh_phase_screen(r0, NN, pixel_scale, L0, l0)
U_in_big = np.sqrt(I_in_big)*np.exp(1j*phase_in_big)
U_in = mprop.crop2(U_in_big, pf)
phase_in = np.angle(U_in)
I_in = np.abs(U_in)**2

U_out_big = angularSpectrum(U_in_big, wvl, pixel_scale, pixel_scale, z)
U_out = mprop.crop2(U_out_big, pf)
phase_out = np.angle(U_out)
I_out = np.abs(U_out)**2

slope_phase_in = mwfs.get_slope(phase_in, pixel_scale, wvl)
lenslet_slope_phase_in = mwfs.get_vec_lenslet(slope_phase_in, lenslet_pixel)
size_lenslet_slope_phase_in = np.sqrt(lenslet_slope_phase_in[:,:,0]**2
                                + lenslet_slope_phase_in[:,:,1]**2)

slope_phase_out = mwfs.get_slope(phase_out, pixel_scale, wvl)
lenslet_slope_phase_out = mwfs.get_vec_lenslet(slope_phase_out, lenslet_pixel)
weight_slope_phase_out = slope_phase_out
lenslet_weight_slope_phase_out = lenslet_slope_phase_out
weight_slope_phase_out[:,:,0] = (slope_phase_out[:,:,0] * I_out)
weight_slope_phase_out[:,:,1] = (slope_phase_out[:,:,1] * I_out)
lenslet_weight_slope_phase_out = mwfs.get_vec_lenslet(weight_slope_phase_out,
                                                      lenslet_pixel)
lenslet_I_out = mwfs.get_val_lenslet(I_out, lenslet_pixel)
lenslet_weight_slope_phase_out[:,:,0] = (lenslet_weight_slope_phase_out[:,:,0]
                                         / lenslet_I_out)
lenslet_weight_slope_phase_out[:,:,1] = (lenslet_weight_slope_phase_out[:,:,1]
                                         / lenslet_I_out)



plt.pcolor(np.arange(-prN,prN+1)*pixel_scale,
           np.arange(-prN,prN+1)*pixel_scale,
           phase_in[CEN-prN:CEN+prN,CEN-prN:CEN+prN],
           snap=True, cmap=my_cmap)
plt.colorbar()
plt.vlines((np.arange(-prNN, prNN + 1))*pixel_scale*lenslet_pixel,
           ymin=-prNN*pixel_scale*lenslet_pixel,
           ymax=prNN*pixel_scale*lenslet_pixel,
           color='m')
plt.hlines((np.arange(-prNN, prNN + 1))*pixel_scale*lenslet_pixel,
           xmin=-prNN*pixel_scale*lenslet_pixel,
           xmax=prNN*pixel_scale*lenslet_pixel,
           color='m')
plt.quiver((np.arange(-prNN,prNN)+0.5)*lenslet_pixel*pixel_scale,
           (np.arange(-prNN,prNN)+0.5)*lenslet_pixel*pixel_scale,
           lenslet_slope_phase_in[cen-prNN:cen+prNN,cen-prNN:cen+prNN,0],
           lenslet_slope_phase_in[cen-prNN:cen+prNN,cen-prNN:cen+prNN,1],
           color='r')
plt.axis('equal')
plt.title('Phase (rad) and lenslet slope at the source '
          + '(z={:.0f}km high)\n'.format(z/1000)
          + 'r={:.0f}cm'.format(r0*100))
plt.xlabel('x (m)')
plt.ylabel('x (m)')
plt.show()

plt.pcolor(np.arange(-prN,prN+1)*pixel_scale,
           np.arange(-prN,prN+1)*pixel_scale,
           phase_out[CEN-prN:CEN+prN,CEN-prN:CEN+prN],
           snap=True, cmap=my_cmap)
plt.colorbar()
plt.vlines((np.arange(-prNN, prNN + 1))*pixel_scale*lenslet_pixel,
           ymin=-prNN*pixel_scale*lenslet_pixel,
           ymax=prNN*pixel_scale*lenslet_pixel,
           color='m')
plt.hlines((np.arange(-prNN, prNN + 1))*pixel_scale*lenslet_pixel,
           xmin=-prNN*pixel_scale*lenslet_pixel,
           xmax=prNN*pixel_scale*lenslet_pixel,
           color='m')
plt.quiver((np.arange(-prNN,prNN)+0.5)*lenslet_pixel*pixel_scale,
           (np.arange(-prNN,prNN)+0.5)*lenslet_pixel*pixel_scale,
           lenslet_slope_phase_out[cen-prNN:cen+prNN,cen-prNN:cen+prNN,0],
           lenslet_slope_phase_out[cen-prNN:cen+prNN,cen-prNN:cen+prNN,1],
           color='r')
plt.axis('equal')
plt.title('Phase (rad) and lenslet slope at ground (dz={:.0f}km)'.format(z/1000))
plt.xlabel('x (m)')
plt.ylabel('x (m)')
plt.show()

plt.pcolor(np.arange(plot_frame*lenslet_pixel+1)*pixel_scale,
           np.arange(plot_frame*lenslet_pixel+1)*pixel_scale,
           phase_out[:plot_frame*lenslet_pixel,:plot_frame*lenslet_pixel])
plt.colorbar()
plt.axis('equal')
plt.quiver(np.arange(plot_frame)*lenslet_pixel*pixel_scale,
           np.arange(plot_frame)*lenslet_pixel*pixel_scale,
           lenslet_weight_slope_phase_out[:plot_frame,:plot_frame,0],
           lenslet_weight_slope_phase_out[:plot_frame,:plot_frame,1])
plt.title('Lenslet phase (rad) and weight slope propagated to the ground')
plt.xlabel('x (m)')
plt.ylabel('x (m)')
plt.show()



dif = (lenslet_slope_phase_in - lenslet_slope_phase_out)*pixel_scale/wvl*np.pi

size_dif = np.sqrt(dif[:,:,0]**2
                   + dif[:,:,1]**2)

rms = np.sqrt(np.mean(size_dif**2))

plt.pcolor(np.arange(plot_frame*lenslet_pixel+1)*pixel_scale,
           np.arange(plot_frame*lenslet_pixel+1)*pixel_scale,
           phase_in[:plot_frame*lenslet_pixel,:plot_frame*lenslet_pixel]-phase_out[:plot_frame*lenslet_pixel,:plot_frame*lenslet_pixel])
plt.colorbar()
plt.axis('equal')
plt.quiver(np.arange(plot_frame)*lenslet_pixel*pixel_scale,
           np.arange(plot_frame)*lenslet_pixel*pixel_scale,
           lenslet_slope_phase_in[:plot_frame,:plot_frame,0] - lenslet_slope_phase_out[:plot_frame,:plot_frame,0],
           lenslet_slope_phase_in[:plot_frame,:plot_frame,1] - lenslet_slope_phase_out[:plot_frame,:plot_frame,1])
plt.title('Slope difference between actual phase'
          + ' at the source ({:.0f}m)\n'.format(z)
          + ' and actual propagated screen at ground\n'
          + 'rms={:.3f}rad'.format(rms))
plt.show()



