# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:59:42 2023

@author: gqmc62


This file should do a proper way of simulating effect of fov

it must be cropped down rather than repropagate to the new image.

or else with high enough resolution, the variance is always the same.
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:32:29 2023

@author: gqmc62
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 12:09:45 2023

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
import MyWFS as mwfs
from tqdm import tqdm

for N in np.array([16,32,64,128,256]):
    # N = 256
    Ntest = 256
    L0 = 4
    dor0 = np.array([0.5,1,2,3], dtype=float)
    r0_original = L0/dor0.max()/2
    wvl = 500e-9
    REPEAT = 10
    r0 = np.array([2], dtype=float)*1e-2
    data = np.zeros((len(dor0),len(r0), REPEAT, int(np.log(Ntest)/np.log(2)), 2),dtype=float)
    dif = np.zeros((len(dor0),len(r0), REPEAT,int(np.log(Ntest)/np.log(2) - 1),2), dtype=float)
    
    
    
    for ntest in np.arange(len(dor0)):
        
        # each image condition
        d_original = dor0[ntest]*r0_original
        pixel_scale_original = d_original/N
        
        phasescreen = infinitephasescreen.PhaseScreenVonKarman(N, pixel_scale_original, r0_original, L0)
        for i in np.arange(2*N):
            phasescreen.add_row()
        main_phasescreen = phasescreen.scrn - phasescreen.scrn.mean()
        # plt.imshow(main_phasescreen)
        # plt.show()
        
        # main_U = np.exp(1j*main_phasescreen)
        for repeat in tqdm(np.arange(REPEAT)):
            #generate each new phase screen
            for i in np.arange(2*N):
                phasescreen.add_row()
            original_main_phasescreen = phasescreen.scrn - phasescreen.scrn.mean()
            
            for ir0 in np.arange(len(r0)):
                # scale to proper r0 and scale
                pixel_scale = dor0[ntest]*r0[ir0]/N
                factor = (r0_original/pixel_scale_original
                   * pixel_scale/r0[ir0])**(5./6.)
                d = dor0[ntest]*r0[ir0]
                main_phase_screen = original_main_phasescreen * (factor)
                
                # now the experiment
                for power in (np.arange(np.log(Ntest)/np.log(2)) + 1):
                    n_out = 2**power
                    # binned_U = rebin(main_U, (int(n_binned_screen),int(n_binned_screen)))
                    # binned_phase_screen = np.angle(binned_U)
                    # binned_phase_screen = rebin(main_phasescreen, (int(n_binned_screen),int(n_binned_screen)))
                    # plt.imshow(binned_phase_screen)
                    # plt.show()
                    #                                               n_in,d_in, alpha_out, FoV_out, wvl
                    WFS = mwfs.AngularPropSHWFS(main_phase_screen.shape[0], d, 1/2, n_out, wvl)
                    data[ntest,ir0, repeat, int(power - 1), : ] = WFS.measure_no_noise_no_threshold(np.exp(1j*main_phase_screen))
                    # WFS = mwfs.AngularPropSHWFS(binned_phase_screen.shape[0], d, 0.5, 32, wvl)
                    # WFS.measure_no_noise_no_threshold(np.exp(1j*binned_phase_screen))
        
    for ntest in np.arange(len(dor0)):
        for i in np.arange(REPEAT):
            for j in np.arange(2):
                dif[ntest,ir0,i,:,j] = data[ntest,ir0,i,:-1,j] - data[ntest,ir0,i,-1,j]
    
    for ntest in np.arange(len(dor0)):
        for ir0 in np.arange(len(r0)):
            plt.errorbar(2**((np.arange(np.log(Ntest)/np.log(2) - 1) + 1)),
                     np.std(dif[ntest,ir0],axis=(0,-1))/np.std(data[ntest,ir0,:,-1,:]),
                     yerr=np.sqrt(2*np.var(dif[ntest,ir0],axis=(0,-1))**2/(dif.shape[2]*dif.shape[4] - 1)),
                     marker='o',
                     label='d/r0={:.1f}'.format(dor0[ntest]))
    # plt.title('convergence of slopes vs number of pixels in wfs image\n'
    #            + 'same input screen and output fov\n'
    #            + 'with {:d}x{:d} input pixels'.format(N,N))
    plt.xlabel('FoV ($\lambda$/d)')
    plt.ylabel('slope error ratio')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1,Ntest)
    plt.ylim(1e-4,1)
    plt.legend()
    plt.savefig('input-vs-output-pixel-number-{:d}inpixel.pdf'.format(N),bbox_inches='tight')
    plt.show()
    plt.clf()
