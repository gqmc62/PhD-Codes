# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:29:29 2023

investigate the following parameter for shwfs

1 number of input pixels
2 fov of the output
3 pixel scale of the output
4 number of output pixels

@author: gqmc62
"""

def rebin(arr, new_shape):
    """
    Direct copy from the following source.
    
    https://scipython.com/blog/binning-a-2d-array-in-numpy/

    Parameters
    ----------
    arr : TYPE
        DESCRIPTION.
    new_shape : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

import numpy as np
from matplotlib import pyplot as plt
from aotools.turbulence import infinitephasescreen
import MyWFS as MWFS
from tqdm import tqdm

N_original = int(256)
L0_original = float(20)
dor0 = float(3)
d_original = L0_original/8
r0_original = d_original/dor0
delta_original = d_original/N_original
phasescreen = infinitephasescreen.PhaseScreenVonKarman(N_original, delta_original, r0_original, L0_original)
REPEAT = 100
wvl = 500e-9

r0 = 2e-2
d = dor0*r0
delta = d/N_original

alpha = 0.5 # in wvl/d

scaling_factor = (r0_original/delta_original*delta/r0)**(5./6.)

for i in np.arange(2*N_original):
    phasescreen.add_row()
main_phasescreen = phasescreen.scrn - phasescreen.scrn.mean()

screen_sizes = (2**np.arange( 1,int( np.log(N_original)/np.log(2) + 1) ) )
FOVs = (2**np.arange(1,int(np.log(N_original)/np.log(2) + 1))) # unit of wvl/d

data = np.zeros((REPEAT,screen_sizes.shape[0],FOVs.shape[0],2), dtype = float)
dif = np.zeros((REPEAT,screen_sizes.shape[0] - 1,FOVs.shape[0] - 1,2), dtype = float)

for repeat in tqdm(np.arange(REPEAT)):
    for i in np.arange(2*N_original):
        phasescreen.add_row()
    main_phasescreen = phasescreen.scrn - phasescreen.scrn.mean()
    proper_screen = main_phasescreen * scaling_factor
    # plt.imshow(proper_screen)
    # plt.show()
    for size_i in np.arange(len(screen_sizes)):
        screen = rebin(proper_screen,(screen_sizes[size_i],screen_sizes[size_i]))
        # plt.imshow(screen)
        # plt.show()
        for fov_i in np.arange(len(FOVs)):
            WFS = MWFS.ConsecutiveAngularPropSHWFS(screen_sizes[size_i], d, alpha, FOVs[fov_i], wvl)
            data[repeat,size_i,fov_i,:] = WFS.measure_no_noise_no_threshold(np.exp(1j*screen))





most_accurate_data = data[:,-1,-1,:]

# rms_fractional_error = np.zeros((screen_sizes.shape[0]-1,FOVs.shape[0]-1),dtype=float)
rms_error = np.zeros((screen_sizes.shape[0]-1,FOVs.shape[0]-1),dtype=float)
rms_error_to_rms_data = np.zeros((screen_sizes.shape[0]-1,FOVs.shape[0]-1),dtype=float)

for repeat in np.arange(REPEAT):
    for size_i in np.arange(len(screen_sizes) - 1):
        for fov_i in np.arange(len(FOVs) - 1):
            dif[repeat,size_i,fov_i,:] = data[repeat,size_i,fov_i,:] - most_accurate_data[repeat,:]

for size_i in np.arange(len(screen_sizes) - 1):
    for fov_i in np.arange(len(FOVs) - 1):
        # info = (dif[:,size_i,fov_i,:]/most_accurate_data)
        # rms_fractional_error[size_i,fov_i] = np.sqrt(np.nanmean(info)**2)
        rms_error[size_i,fov_i] = np.sqrt(np.mean((dif[:,size_i,fov_i,:])**2))
        rms_error_to_rms_data[size_i,fov_i] = np.sqrt(np.mean((dif[:,size_i,fov_i,:])**2))/np.sqrt(np.mean((most_accurate_data)**2))

plt.pcolor(np.log(screen_sizes[:-1])/np.log(2),
           np.log(FOVs[:-1])/np.log(2),
           np.log10(rms_error_to_rms_data).T)
plt.colorbar()
plt.xlabel('screen_sizes in pixels in power of 2')
plt.ylabel('FoV in wvl/d in power of 2')
plt.title('log10 of rms error of shwfs measurement\npixel scale = 0.5wvl/d')
plt.axis('square')
plt.show()

for fov_i in np.arange(len(FOVs)-1):
    plt.plot(screen_sizes[:-1],rms_error_to_rms_data[:,fov_i],label='{:d} wvl/d fov'.format(FOVs[fov_i]),marker='o',ls='--')
    
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of input pixels')
plt.ylabel('rms error to rms data')
plt.legend()
plt.savefig('good_input_px_number.png')
plt.show()

for size_i in np.arange(len(screen_sizes)-1):
    plt.plot(FOVs[:-1],rms_error_to_rms_data[size_i,:],label='{:d} input pixels'.format(screen_sizes[size_i]),marker='o',ls='--')
    
plt.xscale('log')
plt.yscale('log')
plt.xlabel('SHWFS FoV in wvl/d')
plt.ylabel('rms error to rms data')
plt.legend()
plt.savefig('good_shwfs_fov.png')
plt.show()


accuracy_gain = rms_error_to_rms_data[-1,:-1] - rms_error_to_rms_data[-1,1:]

increase_data = screen_sizes[1:-1]**2 - screen_sizes[:-2]**2

plt.plot(screen_sizes[1:-1],accuracy_gain/increase_data,marker='o')
plt.xlabel('pixels across a screen')
plt.ylabel('accuracy gain per pixels')
plt.xscale('log')
plt.yscale('log')
plt.savefig('phase_screen_size_to_accuracy_gain_in_SHWFS.png')
plt.show()

plt.plot(screen_sizes[:-1],rms_error_to_rms_data[:,-1],marker='o')
plt.xlabel('screen size in px')
plt.ylabel('relative error')
plt.show()





plot_this = rms_error_to_rms_data

plt.pcolor(np.log(screen_sizes[:-1])/np.log(2),
           np.log(FOVs[:-1])/np.log(2),
           np.log10(plot_this).T)
plt.colorbar()
plt.xlabel('screen_sizes in pixels in power of 2')
plt.ylabel('FoV in wvl/d in power of 2')
plt.title('log10 of rms error of shwfs measurement\npixel scale = 0.5wvl/d')
plt.axis('square')
plt.show()

for fov_i in np.arange(len(FOVs)-1):
    plt.plot(screen_sizes[:-1],plot_this[:,fov_i],label='{:d} wvl/d fov'.format(FOVs[fov_i]),marker='o',ls='--')
    
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of input pixels')
plt.ylabel('rms error')
plt.legend()

plt.show()

for size_i in np.arange(len(screen_sizes)-1):
    plt.plot(FOVs[:-1],plot_this[size_i,:],label='{:d} input pixels'.format(screen_sizes[size_i]),marker='o',ls='--')
    
plt.xscale('log')
plt.yscale('log')
plt.xlabel('SHWFS FoV in wvl/d')
plt.ylabel('rms error')
plt.legend()

plt.show()


accuracy_gain = plot_this[-1,:-1] - plot_this[-1,1:]

increase_data = screen_sizes[1:-1]**2 - screen_sizes[:-2]**2

plt.plot(screen_sizes[1:-1],accuracy_gain/increase_data,marker='o')
plt.xlabel('pixels across a screen')
plt.ylabel('accuracy gain per pixels')
plt.xscale('log')
plt.yscale('log')

plt.show()

plt.plot(screen_sizes[:-1],plot_this[:,-1],marker='o')
plt.xlabel('screen size in px')
plt.ylabel('error')
plt.xscale('log')
plt.yscale('log')
plt.show()



