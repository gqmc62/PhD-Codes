# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:54:01 2024

@author: gqmc62
"""

from matplotlib import pyplot as plt
from soapy import soapy
# import soapy
import os
import numpy as np
from tqdm import tqdm
import yaml
import time

from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

from matplotlib.lines import Line2D
from scipy.stats import norm

import phase_screen_zernike_decomposition as z_decom

from aotools.functions import zernike

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

def angle_of_arrival(wvl,D,r0):
    alpha = np.sqrt(0.182 * wvl**2 * D**(-1./3.) * r0**(-5./3.))
    return alpha

def scrn_size_at_alt(wvl, D, r0, z):
    size = 4*wvl*z/r0 + D
    return size

def rytov_number(wvl, r0, z):
    # calculate rytov number from rytov approximation
    k = 2.*np.pi/wvl                    # float; rad/m; Wave number.
    cn2dz = r02cn2dz(r0, wvl)        # float; Turbulence strength.
    variance_theory_approx = 0.563 * k**(7./6.) * cn2dz * z**(5./6.)
    return variance_theory_approx

def cn2dz2r0(cn2dz,wvl=500e-9):
    r0 = (0.423*(2.*np.pi/wvl)**2*cn2dz)**(-3./5.)
    return r0

def r02cn2dz(r0,wvl=500e-9):
    cn2dz = r0**(-5./3.)/(0.423*(2.*np.pi/wvl)**2)
    return cn2dz

def displacement(wvl,dsub,r0,z):
    displace = angle_of_arrival(wvl,dsub,r0) * z/dsub
    return displace

def get_PSD(phase, delta):
    """
    Calculate PSD of a phase screen.

    Parameters
    ----------
    phase : numpy.ndarray
            (N,N)
            Phase screen to be measured.
            radian
    delta : float
            Pixel scale.

    Returns
    -------
    PSD : numpy.ndarray
          (N,N)
          Power Spectral Density.

    Example
    -------
    delta = 3.125e-3
    r0 = 0.05
    N = 128
    L0 = 10
    l0 = 1e-10
    f = (np.arange(N/2-1)+1)/N/delta
    RANDOM = 1000
    
    PSD_measure = np.zeros((RANDOM,int(N/2 - 1)),dtype=float)
    for random in np.arange(RANDOM):
        phase = ft_phase_screen(r0, N, delta, L0, l0)
        PSD_measure[random] += get_PSD(phase,delta)[int(N/2),int(N/2)+1:]
    mean_PSD_measure = np.mean(PSD_measure, axis=0)
    error_mean_PSD_measure = np.std(PSD_measure, axis=0, ddof=1)/np.sqrt(RANDOM)
    PSD_theory = psd_function(r0, N, delta, L0, l0)[int(N/2),int(N/2)+1:]

    plt.errorbar(f, mean_PSD_measure, yerr=error_mean_PSD_measure, label='Measured',
             marker='.',ls='',capsize=5)
    # plt.errorbar(f, np.abs(mean_PSD_measure - PSD_theory), yerr=error_mean_PSD_measure, marker='.', ls='', label='abs Residual')
    plt.plot(f, PSD_theory, label='Theory')
    plt.title('Comparison between theoretical and measured PSD')
    plt.xlabel('Spatial Frequency (1/m)')
    plt.ylabel('PSD')
    plt.xlim(f[0]/2,f[-1]*2)
    plt.ylim(1e-13,1e4)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    plt.errorbar(f, mean_PSD_measure, yerr=error_mean_PSD_measure, label='Measured',
             marker='.',ls='',capsize=5)
    plt.errorbar(f, np.abs(mean_PSD_measure - PSD_theory), yerr=error_mean_PSD_measure, marker='.', ls='', label='abs Residual')
    plt.plot(f, PSD_theory, label='Theory')
    plt.title('Comparison between theoretical and measured PSD')
    plt.xlabel('Spatial Frequency (1/m)')
    plt.ylabel('PSD')
    plt.xlim(f[0]/2,f[-1]*2)
    plt.ylim(1e-13,1e4)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    """
    dim = phase.ndim
    if dim == 2:
        N = len(phase)
        del_f = 1./(N*2.*delta)
        phase = phase - np.nanmean(phase)
        cn = my_ft2(mirror_pad2(phase), 1./N/2.)
        PSD = (np.abs(cn)**2/del_f**2)*2.   # * 2. is from the fact that
                                            # the input phase is not complex
                                            # phase are complex when generated
                                            # from cn.
        PSD[int(N/2),int(N/2)] = 0.         # PSD is 0 at the centre.
        return pick_even_index(PSD)/4
    
    if dim == 1:
        N = len(phase)
        del_f = 1/(N*2.*delta)
        phase = phase - phase.mean()
        cn = np.fft.ifftshift(np.fft.fft(np.pad(phase,(N//2,N//2),'symmetric')))*1./N/2.
        PSD = (np.abs(cn)**2/del_f)
        PSD[int(N/2)] = 0.
        return PSD[::2]/2
    
def pick_even_index(AA):
    """
    Pick only pair of even indices, applicable to 2d only.

    Parameters
    ----------
    AA : numpy.ndarray
         (N,N)
         Parent array to be wittled.

    Returns
    -------
    A : numpy.2darray
        (N/2,N/2)
        Daugther array.

    """
    return AA[tuple(np.meshgrid(np.arange(0,AA.shape[0],2),
                             np.arange(0,AA.shape[1],2)))]

def mirror_pad2(A):
    r"""
    Use AA = np.pad(A,(int(A.shape[0]/2),int(A.shape[1]/2)),'symmetric').

    Pad A with its mirrored version.

    If  /\ is the parent, then /\/\ is the daughter.
        /\                     \/\/
                               \/\/
                               /\/\
    The parent will be at the center of the daughter,
    surrounded by a half of the parrent's mirrored self,
    doubling in size.

    Parameters
    ----------
    A : numpy.ndarray
        (N,N)
        Parent array.

    Returns
    -------
    AA : numpy.ndarray
         (2N,2N)
         Daughter array.

    Examples
    --------
    x = np.arange(10)
    xx, yy = np.meshgrid(x, x)
    p = xx + yy
    plt.imshow(p)
    plt.title('Parent')
    plt.show()
    pd = mirror_pad2(p)
    plt.imshow(pd)
    plt.title('Daughter')
    plt.show()

    """

    return np.pad(A,(int(A.shape[0]/2),int(A.shape[1]/2)),'symmetric')

def my_ft2(g, delta):
    """
    Complete 2D Fourier transform.

    Parameters
    ----------
    g : numpy.ndarray
        (N,N)
        Function.
    delta : float
        Pixel scale.

    Returns
    -------
    G : numpy.ndarray
        (N,N)
        Transformed function.

    """
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(g))) * delta**2

def bin_linear_to_log10_v1(data,scale):
    delta = np.max(scale[1:] - scale[:-1])
    minx = scale[0]
    maxx = scale[-1]
    num_point = int(np.ceil((np.log10(maxx) - np.log10(minx))/np.log10(delta)))
    # num_point = scale.shape[0]
    #old_scale_log = np.log10(scale)
    new_scale_edge = np.logspace(np.log10(minx) - np.log10(delta)/2,
                            np.log10(maxx) + np.log10(delta)/2,
                            num_point + 1)
    bin_center = np.logspace(np.log10(minx),
                            np.log10(maxx),
                            num_point)
    where_belong = np.digitize(scale,new_scale_edge)
    bin_data = np.zeros((new_scale_edge.shape[0]),dtype=type(data[0]))
    bin_data_error = np.zeros((new_scale_edge.shape[0]),dtype=type(data[0]))
    for i in np.arange(new_scale_edge.shape[0]):
        in_bin = data[np.where(where_belong==i)]
        bin_data[i] = in_bin.mean()
        bin_data_error[i] = (in_bin.std()
                             / np.sqrt(in_bin.shape[0]))
    
    new_data = bin_data[1:]
    new_data_error = bin_data_error[1:]
    
    #### for showing proof
    old_integral = (delta*data).sum()
    new_integral = (bin_data[1:]*(new_scale_edge[1:]-new_scale_edge[:-1])).sum()
    plt.plot(scale,data,marker='x',label='linear')
    plt.errorbar(bin_center,bin_data[1:],yerr=new_data_error,marker='o',capsize=5,label='log')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title('old integral is {:.6f}\nnew integral is {:.6f}'.format(
        old_integral,new_integral))
    plt.show()
    
    return bin_center,new_data,new_data_error

def bin_linear_to_log10_v2(data,scale):
    # only good for presentation
    # does not conserve integral!!! beware
    delta = np.max(scale[1:] - scale[:-1])
    minx = scale[0]
    maxx = scale[-1]
    # num_point = int(np.ceil((np.log10(maxx) - np.log10(minx))/np.log10(delta)))
    num_point = scale.shape[0]
    #old_scale_log = np.log10(scale)
    new_scale_edge = np.logspace(np.log10(minx) - np.log10(delta)/2,
                            np.log10(maxx) + np.log10(delta)/2,
                            num_point + 1)
    # bin_center = np.logspace(np.log10(minx),
    #                         np.log10(maxx),
    #                         num_point)
    bin_center = 10**((np.log10(new_scale_edge[1:]) + np.log10(new_scale_edge[:-1]))/2)
    where_belong = np.digitize(scale,new_scale_edge)
    bin_data = np.zeros((new_scale_edge.shape[0]),dtype=type(data[0]))
    bin_data_error = np.zeros((new_scale_edge.shape[0]),dtype=type(data[0]))
    for i in np.arange(new_scale_edge.shape[0]):
        in_bin = data[np.where(where_belong==i)]
        bin_data[i] = in_bin.mean()
        bin_data_error[i] = (in_bin.std()
                             / np.sqrt(in_bin.shape[0]))
    
    not_nan_position = np.where(~np.isnan(bin_data[1:]))
    new_data = bin_data[1:][not_nan_position]
    new_data_error = bin_data_error[1:][not_nan_position]
    new_bin_center = bin_center[not_nan_position]
    new_bin_edge_left = new_scale_edge[not_nan_position]
    new_bin_edge_right = new_scale_edge[1:][not_nan_position]
    
    #### for showing proof
    old_integral = (delta*data).sum()
    new_integral = np.nansum(new_data*(new_bin_edge_right-new_bin_edge_left))
    plt.plot(scale,data,marker='x',label='linear')
    plt.errorbar(new_bin_center,new_data,yerr=new_data_error,marker='o',capsize=5,label='log')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title('old integral is {:.6f}\nnew integral is {:.6f}'.format(
        old_integral,new_integral))
    plt.show()
    
    return new_bin_center,new_data,new_data_error

def bin_linear_to_log10_v3(data,scale):
    # only good for presentation
    # does not conserve integral!!! beware
    delta = np.max(scale[1:] - scale[:-1])
    minx = scale[0]
    maxx = scale[-1]
    # num_point = int(np.ceil((np.log10(maxx) - np.log10(minx))/np.log10(delta)))
    num_point = scale.shape[0]
    #old_scale_log = np.log10(scale)
    new_scale_edge = np.logspace(np.log10(minx) - np.log10(delta)/2,
                            np.log10(maxx) + np.log10(delta)/2,
                            num_point + 1)
    # bin_center = np.logspace(np.log10(minx),
    #                         np.log10(maxx),
    #                         num_point)
    bin_center = 10**((np.log10(new_scale_edge[1:]) + np.log10(new_scale_edge[:-1]))/2)
    where_belong = np.digitize(scale,new_scale_edge)
    bin_data = np.zeros((new_scale_edge.shape[0]),dtype=type(data[0]))
    bin_data_error = np.zeros((new_scale_edge.shape[0]),dtype=type(data[0]))
    for i in np.arange(new_scale_edge.shape[0]):
        in_bin = data[np.where(where_belong==i)]
        bin_data[i] = in_bin.mean()
        bin_data_error[i] = (in_bin.std()
                             / np.sqrt(in_bin.shape[0])) * delta / (new_scale_edge[i] - new_scale_edge[i-1])
    
    not_nan_position = np.where(bin_data[1:]!=0)
    new_data = bin_data[1:][not_nan_position]
    new_data_error = bin_data_error[1:][not_nan_position]
    new_bin_center = bin_center[not_nan_position]
    new_bin_edge_left = new_scale_edge[not_nan_position]
    new_bin_edge_right = new_scale_edge[1:][not_nan_position]
    
    #### for showing proof
    old_integral = (delta*data).sum()
    new_integral = np.nansum(new_data*(new_bin_edge_right-new_bin_edge_left))
    plt.plot(scale,data,marker='x',label='linear')
    plt.errorbar(new_bin_center,new_data,yerr=new_data_error,marker='o',capsize=5,label='log')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title('old integral is {:.6f}\nnew integral is {:.6f}'.format(
        old_integral,new_integral))
    plt.show()
    
    return new_bin_center,new_data,new_data_error

###############################################################################
###############################################################################
###############################################################################

# setting parameter for the whole study

home = r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf'
saveto = home#r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\GLDM-IM-through-turb'

os.chdir(home)

TSTAMP = '2024-08-02-14-00-19'

os.chdir(home + '/' + TSTAMP)

SIMNAME = 'bad_IM-'

f = open('name_space.txt' ,'r')
name_list = np.asarray(f.readlines())

CASES = len(name_list)

for i in np.arange(CASES):
    name_list[i] = name_list[i].strip()

rytov = np.zeros((CASES),dtype=float)
AS = np.copy(rytov)
z = np.copy(rytov)
r0HA = np.copy(rytov)

for i in np.arange(rytov.shape[0]):
    rytov[i] = float(name_list[i][5:9])
    AS[i] = float(name_list[i][12:14])

    with open(SIMNAME + name_list[i] + '.yaml','r') as file:
        config = yaml.safe_load(file)
        z[i] = config['DM'][2]['altitude']
        r0HA[i] = config['DM'][2]['r0']
    
wvl = config['Science'][0]['wavelength']
dsub = config['Telescope']['telDiam']/config['WFS'][0]['nxSubaps']



to_read = 'IM' + '_'
hdul = fits.open(to_read + SIMNAME + name_list[0] + '.fits')
SHAPE = hdul[0].data.shape

IM = np.zeros(np.concatenate([np.array([CASES]),np.asarray(SHAPE)]),dtype=float)

for cases in np.arange(CASES):
    hdul = fits.open(to_read + SIMNAME + name_list[i] + '.fits')
    IM[cases] = hdul[0].data
    
    
    
to_read = 'perfect_IM' + '_'
hdul = fits.open(to_read + SIMNAME + name_list[0] + '.fits')
SHAPE = hdul[0].data.shape

perfect_IM = np.zeros(np.concatenate([np.array([CASES]),np.asarray(SHAPE)]),dtype=float)

for cases in np.arange(CASES):
    hdul = fits.open(to_read + SIMNAME + name_list[i] + '.fits')
    perfect_IM[cases] = hdul[0].data
    
TRIAL = IM.shape[1]
nACT = IM.shape[2]
nSUBAP = IM.shape[3]

MAX =  perfect_IM[:,2:].max()

perfect_IM /= MAX
IM /= MAX

threshold = 1/4. # selecting only the major strokes.

MASK = (np.abs(perfect_IM[0]) > threshold)
MASK[:2] = 0

# plt.imshow(perfect_IM[0]*MASK)
# plt.show()

nMAINACT = (MASK == True).sum()
DIF = np.zeros((CASES,TRIAL,nMAINACT),dtype=float)

for cases in np.arange(CASES):
    for trial in np.arange(TRIAL):
        DIF[cases,trial] = (perfect_IM[cases] - IM[cases,trial])[MASK]
        
DIF = np.asarray(DIF)
RMS = np.sqrt(np.mean(DIF**2,axis=(1,2)))









# not this part yet



to_read = 'IM' + '_'
hdul = fits.open(to_read + SIMNAME + name_list[0] + '.fits')
SHAPE = hdul[0].data.shape

IM = np.zeros(np.concatenate([np.array([CASES]),np.asarray(SHAPE)]),dtype=float)

for cases in np.arange(CASES):
    hdul = fits.open(to_read + SIMNAME + name_list[i] + '.fits')
    IM[cases] = hdul[0].data
    
    
    
to_read = 'perfect_IM' + '_'
hdul = fits.open(to_read + SIMNAME + name_list[0] + '.fits')
SHAPE = hdul[0].data.shape

perfect_IM = np.zeros(np.concatenate([np.array([CASES]),np.asarray(SHAPE)]),dtype=float)

for cases in np.arange(CASES):
    hdul = fits.open(to_read + SIMNAME + name_list[i] + '.fits')
    perfect_IM[cases] = hdul[0].data
    


