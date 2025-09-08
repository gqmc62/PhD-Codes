# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:06:18 2024

confirm that with 1step fresnel shwfs
there is no need to zero-pad a lot to reduce effect of discretization in fourier transform
because its effect is much less than usual theoretical sh-wfs measurement error
see thomas 2006 for comparison: doi:10.1111/j.1365-2966.2006.10661.x

this can be used as a benchmark for numerical error



@author: gqmc62
"""

import numpy as np
import aotools.functions.zernike
from matplotlib import pyplot as plt
from aotools.turbulence.infinitephasescreen import PhaseScreenVonKarman
from tqdm import tqdm

from matplotlib.lines import Line2D
import matplotlib as mpl
fontsize = 10
figsize = [4,4]
dpi = 600

mpl.rcParams['axes.titlesize'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['figure.titlesize'] = fontsize
mpl.rcParams['figure.figsize'] = figsize
mpl.rcParams['figure.dpi'] = dpi
mpl.rcParams['savefig.dpi'] = dpi
mpl.rcParams['savefig.bbox'] = 'tight'
# mpl.rcParams['savefig.bbox_inches'] = 'tight'
mpl.rcParams['savefig.format'] = 'pdf'

color_list = np.array(list(mpl.colors.TABLEAU_COLORS.keys()))

def bin_this(A, w):
    # a = numpy.nanmean(A[:(A.shape[0]//w)*w,
    #       :(A.shape[1]//w)*w].reshape(
    #           (((A.shape[0]//w),w,((A.shape[1]//w)),w))
    #           ), axis=(1,3))
    a = np.mean(A[:(A.shape[0]//w)*w,
          :(A.shape[1]//w)*w].reshape(
              (((A.shape[0]//w),w,((A.shape[1]//w)),w))
              ), axis=(1,3))
    return a


CONVERT = 500/2/np.pi
n1 = 64 # this is also fov in wvl/2D
D = 0.05
wvl = 500e-9
zero_pad_factor_range = 5
TRIAL = 10
color_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

r0_list = D/np.array([0.5,1,2,3],dtype=float)

for i_r0 in np.arange(r0_list.shape[0]):
    r0 = r0_list[i_r0]
    
    delta1 = D/n1
    
    
    
    error_list = np.zeros((TRIAL,zero_pad_factor_range - 1))
    slope_error_list = np.zeros((TRIAL,zero_pad_factor_range - 1,2))
    true_slope = np.zeros((TRIAL,zero_pad_factor_range - 1,2))
    
    X = np.arange(n1*2) - n1 + 0.5
    YY, XX = np.meshgrid(X,X)
    
    converter = wvl/2./D*D*2.*np.pi/wvl
    
    for trial in tqdm(np.arange(TRIAL)):
        
        I_list = np.zeros((zero_pad_factor_range,n1*2,n1*2),dtype=float)
        slope_list = np.zeros((zero_pad_factor_range,2),dtype=float)
        
        pupil = aotools.functions.zernike.zernikeArray(1, n1)[0]
        
        PS = PhaseScreenVonKarman(n1, delta1, r0, D*100)
        
        for i in np.arange(100*n1):
            PS.add_row()
        U1_no_pad = np.exp(1j*PS.scrn*pupil)
        
        for zero_pad_factor in np.arange(zero_pad_factor_range):
        
            N = 2*n1* 2**zero_pad_factor
            
            f = 1
            delta2 = wvl*f/2./D
            
            delta2_no_bin = wvl*f/N/delta1
            fov = delta2_no_bin*N
            
            x = (np.arange(N) - N/2 + 0.5)*delta1
            yy, xx = np.meshgrid(x,x)
            
            shift = 2.*np.pi/wvl*-delta2_no_bin/2/f*(xx + yy)
            
            
            
            
            U1 = np.zeros((N,N),dtype=complex)
            a = N//2 - n1//2
            b = N//2 + n1//2
            U1[a:b,a:b] = U1_no_pad
            
            U1 *= np.exp(1j*shift)
            
            # plt.imshow(np.abs(U1)**2/((np.abs(U1)**2).sum())*delta2_no_bin**2)
            # # plt.colorbar()
            # plt.show()
            
            
            I_no_bin = np.abs(np.fft.fftshift(np.fft.fft2(U1)))**2
            
            bin_factor = int(delta2/delta2_no_bin)
            
            I = bin_this(I_no_bin,bin_factor)
            
            # plt.imshow(I/I.sum()*delta2**2)
            # # plt.colorbar()
            # plt.show()
            
            I_list[zero_pad_factor] = I/I.max()
            slope_list[zero_pad_factor,0] = ((I*XX).sum()/I.sum())*converter    
            slope_list[zero_pad_factor,1] = ((I*YY).sum()/I.sum())*converter
        # for zero_pad_factor in np.arange(zero_pad_factor_range):
        #     plt.imshow(I_list[zero_pad_factor] - I_list[-1])
        #     plt.colorbar()
        #     plt.show()
            
        error = np.mean((I_list[:-1] - I_list[-1])**2,axis=(1,2))**0.5
        # plt.plot(error)
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.show()
        
    
        error_list[trial] = error
        
        slope_error_list[trial,:,0] = (slope_list[:-1,0] - slope_list[-1,0])
        slope_error_list[trial,:,1] = (slope_list[:-1,1] - slope_list[-1,1])
        true_slope[trial] = slope_list[-1]
        
    # plt.errorbar(np.arange(zero_pad_factor_range - 1), error_list.mean(0), yerr=error_list.std(0),capsize=5)
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.show()
    
    var_true_slope = np.std(true_slope,axis=(0,2))*CONVERT
    error_of_var_true_slope = var_true_slope/np.sqrt(var_true_slope.shape[0]*2 - 2)
    var = np.std(slope_error_list,axis=(0,2))*CONVERT
    error_of_var = var/np.sqrt(var.shape[0]*2 - 2)
    
    plt.errorbar(2**(np.arange(zero_pad_factor_range - 1) + 1),var,yerr=error_of_var,
                 capsize=5,marker='o',
                 color=color_list[i_r0])
    plt.errorbar(2**(np.arange(zero_pad_factor_range - 1) + 1),var_true_slope,yerr=error_of_var_true_slope,
                 capsize=5,marker='o',#label='slope_level{:.2}D/r$_0$'.format(D/r0),
                 ls=':',
                 color=color_list[i_r0])

legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', marker='', label='Measurement error'),
    Line2D([0], [0], color='black', linestyle='--', marker='', label='Atmospheric noise'),
    Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label='$d/r_0$=0.5'),
    Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label='$d/r_0$=1'),
    Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label='$d/r_0$=2'),
    Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label='$d/r_0$=3'),
    ]
plt.legend(handles=legend_elements, loc='best')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('RMS OPD (nm)')
plt.xlabel('Zero-padded size (x no-zero-padded)')
# plt.legend()
plt.savefig('zero_padding_wfs-{:d}px.pdf'.format(n1),bbox_inches='tight')
plt.show()
plt.clf()

