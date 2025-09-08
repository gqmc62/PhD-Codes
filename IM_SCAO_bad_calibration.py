# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:58:34 2024

@author: gqmc62
"""

# home = '/mt/cfai-temp/pkongkaew/conf'
# saveto = '/mt/cfai-temp/pkongkaew/output'
# import soapy

home = r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf'
saveto = home#r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\GLDM-IM-through-turb'
from soapy import soapy

import numpy as np

new_svd_condition = float(0.01)    

TRIAL = 50 #50    0,1,2,3,6,10,16,20
LOOP = 5 #100
z_limit = np.inf# 10e3 #5e3 #25e3 # np.inf#
CASE_STUDY = np.array([2,6,8,11,15,22])#np.array([0,9,12,14,16,19],dtype=int)#14 22 9 12 
SKIP = np.array([],dtype=int)
newIMat = False#True#
PLOT = False#True#True#
given_seed = True
# given_seed = False

original_name = 'bad_IM_A' # 10 nxsubap 0.4
# original_name = 'bad_IM_B' # 8 nxsubap 0.4
# original_name = 'bad_IM_C'# 6 nxsubap 0.4

# original_name = 'bad_IM_A2' # 10 nxsubap 0.5m
# original_name = 'bad_IM_B2' # 6 nxsubap 0.3m
# original_name = 'bad_IM_C2'# 4 nxsubap 0.2m

wvl = 500e-9
# aber_L0 = 8

GL_r0 = 0.10
MODE = 45 # 105 # 91 # 45 # 153

r0_well_sampled_limit = 0.02

from matplotlib import pyplot as plt

import os

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

def my_unwrap(wrapped_phase, period=2*np.pi):
    
    # numpy unwrap start unwrap at 0 coordinate
    # but for circular aperture there is no corner!
    # if we unwrap by each quardrant there will be a corner! noice.
    # but we may have to stitch them back together nicely
    # so things should be de-piston to the center
    
    center = wrapped_phase.shape[-1]//2
    unwrapped_phase = np.zeros_like(wrapped_phase)
    
    # ++ quardrant
    wrapped_quardrant = wrapped_phase[...,center:,center:]
    unwrapped_quardrant = np.unwrap(np.unwrap(wrapped_quardrant,axis=0),axis=1)
      
    unwrapped_quardrant -= unwrapped_quardrant[0,0]
    unwrapped_phase[...,center:,center:] = unwrapped_quardrant
    
    # +- quardrant
    wrapped_quardrant = wrapped_phase[...,center:,:center][:,::-1]
    unwrapped_quardrant = np.unwrap(np.unwrap(wrapped_quardrant,axis=0),axis=1)
      
    unwrapped_quardrant -= unwrapped_quardrant[0,0]
    unwrapped_phase[...,center:,:center] = unwrapped_quardrant[:,::-1]
    
    # -+ quardrant
    wrapped_quardrant = wrapped_phase[...,:center,center:][::-1,:]
    unwrapped_quardrant = np.unwrap(np.unwrap(wrapped_quardrant,axis=0),axis=1)
      
    unwrapped_quardrant -= unwrapped_quardrant[0,0]
    unwrapped_phase[...,:center,center:] = unwrapped_quardrant[::-1,:]
      
    # -- quardrant
    wrapped_quardrant = wrapped_phase[...,:center,:center][::-1,::-1]
    unwrapped_quardrant = np.unwrap(np.unwrap(wrapped_quardrant,axis=0),axis=1)
      
    unwrapped_quardrant -= unwrapped_quardrant[0,0]
    unwrapped_phase[...,:center,:center] = unwrapped_quardrant[::-1,::-1]
    
    return unwrapped_phase

def crop_inscribed_square(circular_data):
    N = circular_data.shape[0]
    R = int(np.floor(N//2))
    r = int(np.floor(R/np.sqrt(2)))
    square_data = circular_data[N//2 - r
                  : N//2 + r,
                  N//2 - r
                  : N//2 + r]
    return square_data

def bin_this(A, w):
    # a = numpy.nanmean(A[:(A.shape[0]//w)*w,
    #       :(A.shape[1]//w)*w].reshape(
    #           (((A.shape[0]//w),w,((A.shape[1]//w)),w))
    #           ), axis=(1,3))
    a = np.nanmean(A[:(A.shape[0]//w)*w,
          :(A.shape[1]//w)*w].reshape(
              (((A.shape[0]//w),w,((A.shape[1]//w)),w))
              ), axis=(1,3))
    return a

def angle_of_arrival(wvl,D,r0):
    alpha = np.sqrt(0.162 * wvl**2 * D**(-1./3.) * r0**(-5./3.))
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
    mean_PSD_measure = np.nanmean(PSD_measure, axis=0)
    error_mean_PSD_measure = np.nanstd(PSD_measure, axis=0, ddof=1)/np.sqrt(RANDOM)
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



os.chdir(home)

TSTAMP = time.strftime('%Y-%m-%d-%H-%M-%S')

aber_r0_list_original = cn2dz2r0(np.loadtxt('cn2dz_list_new_AS.txt')).flatten()
aber_z_list_original = np.loadtxt('z_list_new_AS.txt').flatten()

aber_r0_list = np.concatenate([np.array([np.inf]),aber_r0_list_original])
aber_z_list = np.concatenate([np.array([0]),aber_z_list_original])

if given_seed == True:
    
    hdu = fits.open('ATM_SEEDS.fits')
    ATM_SEEDS = hdu[0].data
    hdu.close()
    hdu = fits.open('aberration_seed.fits')
    aberation_seed = hdu[0].data
    hdu.close()

    # aberation_seed = np.array([ 681362132, 1912631052, 2026820200, 1483138070,  583733392, 1805998251, 1889699555,  584259759,  103018972,  899371596])



original_sim = soapy.Sim(original_name + '.yaml')
original_config = original_sim.config.configDict

original_home = home
if not os.path.exists(original_home + '/' + TSTAMP):
    os.makedirs(TSTAMP)
home = original_home + '/' + TSTAMP
os.chdir(home)

name_list=[]

FIRSTTIME = True

if not os.path.exists(original_home + '/' + 'IMCC_' + original_name):
    os.makedirs(original_home + '/' + 'IMCC_' + original_name)


for i_aber_z_list in np.arange(len(aber_z_list)):
    # print(i_aber_z_list)
    if (i_aber_z_list != CASE_STUDY).all() and (CASE_STUDY.size > 0):
        continue
    if (i_aber_z_list == SKIP).any():
        continue
    if aber_z_list[i_aber_z_list] >= z_limit:# 25e3 # too high for common dm elevation
        break
    print(i_aber_z_list)
    ##################################################################################
    # setting parameters for each case
    
    clock_in = time.time()

    aberrationHeight = aber_z_list[i_aber_z_list]
    i_aber_r0 = i_aber_z_list
    aber_r0 = aber_r0_list[i_aber_r0]
    
    if aber_r0 < r0_well_sampled_limit: # too strong turbulence to be well-sampled by my resolution

        name_list.append('skipped')
        clock_out = time.time()

        continue
    
    IM_rytov = rytov_number(wvl, aber_r0, aberrationHeight)
    dsub = (original_config['Telescope']['telDiam']/original_config['WFS'][0]['nxSubaps'])
    IM_AS = displacement(wvl, dsub, aber_r0, aberrationHeight)
    # print(IM_rytov,IM_AS)
    # continue
    # print(aber_r0,aberrationHeight,IM_rytov,IM_AS*100)

    
    
    
    
    # temp = (np.ceil((4*aberrationHeight*wvl/aber_r0
    #                   + original_config['Telescope']['telDiam'])
    #                 / (original_config['Telescope']['telDiam']
    #                     /original_config['WFS'][0]['nxSubaps'])))
    
    # temp2 = int(temp*(temp + 1) / 2)
        
    # mode_list = np.arange(temp2 - 1 - 2) + 2 + 2#np.arange(91 - 1 - 2) + 2 + 2
    # MODE = mode_list.shape[0]
    # ATM_SEEDS = np.arange(TRIAL)
    if given_seed == False:
        ATM_SEEDS = np.random.randint(0,2**31,size=TRIAL)
    # ATM_SEEDS = np.array([1457885706],dtype=int)
    

    
    new_config = dict(original_config)
    # new_config['simName'] = original_config['simName'] + '-r0{:.0f}cm-HA{:.0f}km'.format(aber_r0*100.,aberrationHeight/1000.)
    new_config['simName'] = original_config['simName'] + '-rytov{:.2f}-AS{:.0f}%'.format(IM_rytov,IM_AS*100)
    ndm = new_config['nDM']
    new_idm = 0
    new_config['Atmosphere']['r0'] = GL_r0
    # new_config['Atmosphere']['windSpeeds'] = [50]
    #new_config['Atmosphere']['scrnHeights'] = [aberrationHeight]#[15000]
    for idm in np.arange(ndm):
        if new_config['DM'][idm]['type'] == 'Aberration':
            new_config['DM'][idm]['subtype'] = 'Atmosphere_perfect_DM_no_tip_tilt'#'#'Zernike'#Zernike'#'Atmosphere'
            new_config['DM'][idm]['calibrate'] = True
            new_config['DM'][idm]['r0'] = float(aber_r0)
            # new_config['DM'][idm]['L0'] = float(aber_L0)
            #new_config['DM'][idm]['nollMode'] = int(mode_list[-1])
            new_config['nIters'] = LOOP
            new_config['DM'][idm]['altitude'] = float(aberrationHeight)
            if new_svd_condition is not None:
                new_config['Reconstructor']['svdConditioning'] = new_svd_condition
    
    new_name = new_config['simName']
    with open(new_name + '.yaml','w') as file:
        yaml.dump(new_config, file)
    name = new_name
    name_list.append('rytov{:.2f}-AS{:.0f}%'.format(IM_rytov,IM_AS*100))
    
    
    
    
    
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    
    # Calibration - perfect case

    SIM = soapy.Sim(new_name + '.yaml')
    SIM.aoinit()
    # if trial == 0:
    #     SIM.config.wfss[0].plot = True
    # else:
    #     SIM.config.wfss[0].plot = False
    # SIM.config.wfss[0].plot = True
    SIM.config.wfss[0].plot = PLOT
    
    perfect_SR = np.zeros((TRIAL,LOOP), dtype=float)
    perfect_WFE = np.zeros((TRIAL,LOOP), dtype=float)
    perfect_rytov = np.zeros((TRIAL,LOOP),dtype=float)
    
    # perfect_SR_open = np.zeros((TRIAL,LOOP), dtype=float)
    # perfect_WFE_open = np.zeros((TRIAL,LOOP), dtype=float)
    # perfect_rytov_open = np.zeros((TRIAL,LOOP),dtype=float)
    
    # perfect case don't put in aberration during calibration
    
    for idm in np.arange(SIM.config.sim.nDM):
        if SIM.dms[idm].config.type == 'Aberration' :
            SIM.dms[idm].config.calibrate = False
    # if any aberration thing exist, delete it
    
    
    # make interation matrix
    
    
    if ((not os.path.exists(original_home + '/IMCC_' + original_name + '/perfect_IM' + new_name + '.fits'))
        or newIMat):
        SIM.makeIMat(forceNew=True)
        hdu = fits.PrimaryHDU(np.asarray(SIM.recon.interaction_matrix,dtype=float))
        hdul = fits.HDUList([hdu])
        hdul.writeto(original_home + '/IMCC_' + original_name + '/perfect_IM' + new_name  + '.fits', overwrite=True)
        hdul.close()
        hdu = fits.PrimaryHDU(np.asarray(SIM.recon.control_matrix,dtype=float))
        hdul = fits.HDUList([hdu])
        hdul.writeto(original_home + '/IMCC_' + original_name + '/perfect_CC' + new_name  + '.fits', overwrite=True)
        hdul.close()
        hdu = fits.PrimaryHDU(np.asarray(SIM.recon.zero_iMat,dtype=float))
        hdul = fits.HDUList([hdu])
        hdul.writeto(original_home + '/IMCC_' + original_name + '/perfect_zero_iMat' + new_name  + '.fits', overwrite=True)
        hdul.close()
        hdu = fits.PrimaryHDU(np.asarray(SIM.recon.zero_iMat_intensity,dtype=float))
        hdul = fits.HDUList([hdu])
        hdul.writeto(original_home + '/IMCC_' + original_name + '/perfect_zero_iMat_intensity' + new_name  + '.fits', overwrite=True)
        hdul.close()
    else:
        hdul = fits.open(original_home + '/IMCC_' + original_name + '/perfect_IM' + new_name  + '.fits')
        TEMPIM = hdul[0].data
        hdul = fits.open(original_home + '/IMCC_' + original_name + '/perfect_CC' + new_name  + '.fits')
        TEMPCC = hdul[0].data
        hdul = fits.open(original_home + '/IMCC_' + original_name + '/perfect_zero_iMat' + new_name  + '.fits')
        TEMP0IM = hdul[0].data
        hdul = fits.open(original_home + '/IMCC_' + original_name + '/perfect_zero_iMat_intensity' + new_name  + '.fits')
        TEMP0IMI = hdul[0].data
        
        SIM.makeIMat(forceNew=False,imat_to_load=TEMPIM)#,cmat_to_load=TEMPCC)
        SIM.recon.zero_iMat = TEMP0IM
        SIM.recon.zero_iMat_intensity = TEMP0IMI
    
    # save data
    perfect_IM = np.copy(SIM.recon.interaction_matrix)
    perfect_CC = np.copy(SIM.recon.control_matrix)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    
    # setting for running
    
    N_calc = SIM.config.sim.pupilSize
    R_calc = int(np.floor(N_calc//2))
    r_calc = int(np.floor(R_calc/np.sqrt(2)))
    PSD_size = (2*r_calc//2) - 1
    
    IM = np.zeros((TRIAL,
                    perfect_IM.shape[0],
                    perfect_IM.shape[1]),dtype=float)
    CC = np.zeros((TRIAL,
                    perfect_IM.shape[1],
                    perfect_IM.shape[0]),dtype=float)
    
    if given_seed == False:
    # aberation_seed = np.arange(TRIAL)
        aberation_seed = np.random.randint(0,2**31,size=TRIAL)
    # aberation_seed = np.array([667518184],dtype=int)

    ABERATION_Z = np.zeros((TRIAL,
                              MODE),dtype=float)
    ABERATION_PSD_phase_frequency = np.zeros((TRIAL,
                              PSD_size),dtype=float)
    ABERATION_PSD_phase = np.zeros((TRIAL,
                              PSD_size),dtype=float)
    ZERO_IMAT = np.zeros((TRIAL,perfect_IM.shape[1]),dtype=float)
    ZERO_IMAT_intensity = np.zeros((TRIAL,perfect_IM.shape[1]),dtype=float)
    ATM_SLOPES = np.zeros((TRIAL,perfect_IM.shape[1]),dtype=float)
    
    SR = np.zeros((TRIAL,LOOP),dtype=float)
    WFE = np.zeros((TRIAL,LOOP),dtype=float)
    rytov = np.zeros((TRIAL,LOOP),dtype=float)
    # SR_open = np.zeros((TRIAL,LOOP),dtype=float)
    # WFE_open = np.zeros((TRIAL,LOOP),dtype=float)
    # rytov_open = np.zeros((TRIAL,LOOP),dtype=float)
    
    # must use aberratiion in calibration
    
    for idm in np.arange(SIM.config.sim.nDM):
        if SIM.dms[idm].config.type == 'Aberration' :
            SIM.dms[idm].config.calibrate = True
    
    A = SIM.dms[2].nx_dm_elements
    B = SIM.dms[2].pupil_size
    x = np.arange(SIM.dms[2].pupil_size) - SIM.dms[2].pupil_size//2 + 0.5
    yy, xx = np.meshgrid(x,x)
    rr = np.sqrt(xx**2 + yy**2)
    in_circle = (rr<=SIM.dms[2].pupil_size//2)
    
    # atmos_phase = np.zeros((TRIAL,SIM.atmos.scrns.shape[0],SIM.atmos.scrns.shape[1],SIM.atmos.scrns.shape[2]),dtype=float)
    SLOPES = np.zeros((TRIAL,LOOP,SIM.slopes.shape[0]),dtype=float)
    COMMANDS = np.zeros((TRIAL,LOOP,SIM.dmCommands.shape[0]),dtype=float)
    perfect_SLOPES = np.zeros((TRIAL,LOOP,SIM.slopes.shape[0]),dtype=float)
    perfect_COMMANDS = np.zeros((TRIAL,LOOP,SIM.dmCommands.shape[0]),dtype=float)
    
    # SLOPES_open = np.zeros((TRIAL,LOOP,SIM.slopes.shape[0]),dtype=float)
    # COMMANDS_open = np.zeros((TRIAL,LOOP,SIM.dmCommands.shape[0]),dtype=float)
    # perfect_SLOPES_open = np.zeros((TRIAL,LOOP,SIM.slopes.shape[0]),dtype=float)
    # perfect_COMMANDS_open = np.zeros((TRIAL,LOOP,SIM.dmCommands.shape[0]),dtype=float)
    
    RES = np.zeros((2,
                    TRIAL,
                    SIM.sciPhase[0].shape[0],
                    SIM.sciPhase[0].shape[1],
                    SIM.sciPhase[0].shape[2]),dtype=float)
    perfect_RES = np.zeros((2,
                            TRIAL,
                            SIM.sciPhase[0].shape[0],
                            SIM.sciPhase[0].shape[1],
                            SIM.sciPhase[0].shape[2]),dtype=float)
    
    # RES_open = np.zeros((2,
    #                 TRIAL,
    #                 SIM.sciPhase[0].shape[0],
    #                 SIM.sciPhase[0].shape[1],
    #                 SIM.sciPhase[0].shape[2]),dtype=float)
    # perfect_RES_open = np.zeros((2,
    #                         TRIAL,
    #                         SIM.sciPhase[0].shape[0],
    #                         SIM.sciPhase[0].shape[1],
    #                         SIM.sciPhase[0].shape[2]),dtype=float)
    
    # PSD = np.zeros((LOOP,PSD_size),dtype=float)
    perfect_PSD = np.zeros((LOOP,PSD_size),dtype=float)
    PSD_frequency = np.zeros((PSD_size),dtype=float)
    
    #####################################################################
    
    
    Zernike = zernike.zernikeArray(MODE, SIM.config.sim.pupilSize)
    GramSchmidt = z_decom.gram_schmidt_process(Zernike)
    PUPIL = Zernike[0]
    FlatZernike = Zernike.reshape((MODE, SIM.config.sim.pupilSize*SIM.config.sim.pupilSize))
    FlatGramSchmidt = GramSchmidt.reshape((MODE, SIM.config.sim.pupilSize*SIM.config.sim.pupilSize))
    GramSchmidtCoeff2ZernikeCoeff = z_decom.get_GramSchmidtCoeff2ZernikeCoeff(
        FlatZernike, FlatGramSchmidt)
    
    ATM_Z = np.zeros((TRIAL,
                              MODE),dtype=float)
    
    ATM_PSD_phase_frequency = np.zeros((TRIAL,
                              PSD_size),dtype=float)
    ATM_PSD_phase = np.zeros((TRIAL,
                              PSD_size),dtype=float)
    
    RES_phase_Z = np.zeros((TRIAL,LOOP,MODE), dtype=float)
    perfect_RES_phase_Z = np.zeros((TRIAL,LOOP,MODE), dtype=float)
    
    # RES_phase_Z_open = np.zeros((TRIAL,LOOP,MODE), dtype=float)
    # perfect_RES_phase_Z_open = np.zeros((TRIAL,LOOP,MODE), dtype=float)
    
    #####################################################################
    
    MEASURED_IM_RYTOV = np.zeros((TRIAL,SIM.dms[1].n_acts,SIM.wfss[0].n_subaps),dtype=float) # each trial, when poke each dm act, measure intensity in each wfs subap
    MEASURED_IM_SUBAP_SHIFT_influence = np.zeros((TRIAL,SIM.dms[1].n_acts,1,2),dtype=float) # each trial, when poke each dm act, for each wfs, measure shift in dm act position, in two dimension
    MEASURED_IM_SUBAP_SHIFT_slope = np.zeros((TRIAL,SIM.dms[1].n_acts,1,2),dtype=float) # each trial, 
    
    # RUNAWAY = np.zeros((TRIAL,4),dtype=bool)
    # RUNAWAY_POS = np.zeros((TRIAL,4),dtype=int)
    
    RUNAWAY = np.zeros((TRIAL,2),dtype=bool)
    RUNAWAY_POS = np.zeros((TRIAL,2),dtype=int)
    
    for trial in tqdm(np.arange(TRIAL)):
        print(i_aber_z_list,trial)
        ###############################################################################
        # run with bad calibration # with a regular case
        # if trial == 0:
        #     SIM.config.wfss[0].plot = True
        # else:
        #     SIM.config.wfss[0].plot = False
        SIM.config.wfss[0].plot = PLOT
        print('')
        
        if ((not os.path.exists(original_home + '/IMCC_' + original_name + '/IM' + new_name + '{}'.format(trial) + '.fits'))
            or newIMat) :
            
            if ((SIM.config.dms[2].subtype == 'Atmosphere')
                or (SIM.config.dms[2].subtype == 'Atmosphere_perfect_DM')
                or (SIM.config.dms[2].subtype == 'Atmosphere_perfect_DM_no_tip_tilt')):
            
                SIM.dms[2].atm_ab.random_seed = aberation_seed[trial]
                SIM.dms[2].atm_ab.make_initial_screen()
                SIM.dms[2].atm_ab.move_screen()
                
            
            if SIM.config.dms[2].subtype == 'Zernike':
                SIM.dms[2].random_seed = aberation_seed[trial]
                
            SIM.dms[2].makeDMFrame('shape')
            
            SIM.makeIMat(forceNew=True)
            hdu = fits.PrimaryHDU(np.asarray(SIM.recon.interaction_matrix,dtype=float))
            hdul = fits.HDUList([hdu])
            hdu.header['SEED'] = aberation_seed[trial]
            hdul.writeto(original_home + '/IMCC_' + original_name + '/IM' + new_name  + '{}'.format(trial) + '.fits', overwrite=True)
            hdul.close()
            hdu = fits.PrimaryHDU(np.asarray(SIM.recon.control_matrix,dtype=float))
            hdul = fits.HDUList([hdu])
            hdu.header['SEED'] = aberation_seed[trial]
            hdul.writeto(original_home + '/IMCC_' + original_name + '/CC' + new_name  + '{}'.format(trial) + '.fits', overwrite=True)
            hdul.close()
            hdu = fits.PrimaryHDU(np.asarray(SIM.recon.zero_iMat,dtype=float))
            hdul = fits.HDUList([hdu])
            hdu.header['SEED'] = aberation_seed[trial]
            hdul.writeto(original_home + '/IMCC_' + original_name + '/zero_iMat' + new_name  + '{}'.format(trial) + '.fits', overwrite=True)
            hdul.close()
            hdu = fits.PrimaryHDU(np.asarray(SIM.recon.zero_iMat_intensity,dtype=float))
            hdul = fits.HDUList([hdu])
            hdu.header['SEED'] = aberation_seed[trial]
            hdul.writeto(original_home + '/IMCC_' + original_name + '/zero_iMat_intensity' + new_name  + '{}'.format(trial) + '.fits', overwrite=True)
            hdul.close()
            hdu = fits.PrimaryHDU(np.asarray(SIM.recon.dms[1].rytov,dtype=float))
            hdul = fits.HDUList([hdu])
            hdu.header['SEED'] = aberation_seed[trial]
            hdul.writeto(original_home + '/IMCC_' + original_name + '/rytov' + new_name  + '{}'.format(trial) + '.fits', overwrite=True)
            hdul.close()
            hdu = fits.PrimaryHDU(np.asarray(SIM.recon.dms[1].subapShift_influence,dtype=float))
            hdul = fits.HDUList([hdu])
            hdu.header['SEED'] = aberation_seed[trial]
            hdul.writeto(original_home + '/IMCC_' + original_name + '/subapShift_influence' + new_name  + '{}'.format(trial) + '.fits', overwrite=True)
            hdul.close()
            hdu = fits.PrimaryHDU(np.asarray(SIM.recon.dms[1].subapShift_slope,dtype=float))
            hdul = fits.HDUList([hdu])
            hdu.header['SEED'] = aberation_seed[trial]
            hdul.writeto(original_home + '/IMCC_' + original_name + '/subapShift_slope' + new_name  + '{}'.format(trial) + '.fits', overwrite=True)
            hdul.close()
            
            # for idm in np.arange(SIM.config.sim.nDM):
            #     if not (SIM.config.dms[idm].type == 'Aberration'):
            #         hdu = fits.PrimaryHDU(np.asarray(SIM.dms[idm].valid_actuators,dtype=float))
            #         hdul = fits.HDUList([hdu])
            #         hdul.writeto(original_home + '/IMCC_' + original_name + '/perfect_active_acts_dm{}'.format(idm) + new_name  + '.fits', overwrite=True)
            #         hdul.close()
                
        else:
            hdul = fits.open(original_home + '/IMCC_' + original_name + '/IM' + new_name  + '{}'.format(trial) + '.fits')
            TEMPIM = hdul[0].data
            hdul = fits.open(original_home + '/IMCC_' + original_name + '/CC' + new_name  + '{}'.format(trial) + '.fits')
            TEMPCC = hdul[0].data
            hdul = fits.open(original_home + '/IMCC_' + original_name + '/zero_iMat' + new_name  + '{}'.format(trial) + '.fits')
            TEMP0IM = hdul[0].data
            hdul = fits.open(original_home + '/IMCC_' + original_name + '/zero_iMat_intensity' + new_name  + '{}'.format(trial) + '.fits')
            TEMP0IMI = hdul[0].data
            hdul = fits.open(original_home + '/IMCC_' + original_name + '/rytov' + new_name  + '{}'.format(trial) + '.fits')
            TEMPrytov = hdul[0].data
            try:
                hdul = fits.open(original_home + '/IMCC_' + original_name + '/subapShift_influence' + new_name  + '{}'.format(trial) + '.fits')
                subapShift_influence = hdul[0].data
                hdul = fits.open(original_home + '/IMCC_' + original_name + '/subapShift_slope' + new_name  + '{}'.format(trial) + '.fits')
                subapShift_slope = hdul[0].data
            except:
                hdul = fits.open(original_home + '/IMCC_' + original_name + '/subapShift' + new_name  + '{}'.format(trial) + '.fits')
                subapShift_influ = hdul[0].data
                subapShift_slope = np.zeros_like(subapShift_influ)*np.nan
                
            aberation_seed[trial] = hdul[0].header['SEED']
            
            if ((SIM.config.dms[2].subtype == 'Atmosphere')
                or (SIM.config.dms[2].subtype == 'Atmosphere_perfect_DM')
                or (SIM.config.dms[2].subtype == 'Atmosphere_perfect_DM_no_tip_tilt')):
            
                SIM.dms[2].atm_ab.random_seed = aberation_seed[trial]
                SIM.dms[2].atm_ab.make_initial_screen()
                SIM.dms[2].atm_ab.move_screen()
                
            
            if SIM.config.dms[2].subtype == 'Zernike':
                SIM.dms[2].random_seed = aberation_seed[trial]
                
            SIM.dms[2].makeDMFrame('shape')
            
            SIM.makeIMat(forceNew=False,imat_to_load=TEMPIM)#,cmat_to_load=TEMPCC)
            
            SIM.recon.zero_iMat = TEMP0IM
            SIM.recon.zero_iMat_intensity = TEMP0IMI
            SIM.recon.dms[1].rytov = TEMPrytov
            try:
                SIM.recon.dms[1].subapShift_influence = subapShift_influence
                SIM.recon.dms[1].subapShift_slope = subapShift_slope
            except:
                SIM.recon.dms[1].subapShift_influence = np.nan
                SIM.recon.dms[1].subapShift_slope = np.nan
                
        
        if PLOT == True:
            plt.imshow(SIM.recon.interaction_matrix)
            plt.title('IM')
            plt.colorbar()
            plt.show()
            plt.imshow(SIM.recon.control_matrix)
            plt.title('CM')
            plt.colorbar()
            plt.show()
        
        atm_size1 = (SIM.dms[2].aberration.shape[0] - SIM.config.sim.pupilSize)//2
        atm_size2 = (SIM.dms[2].aberration.shape[0] + SIM.config.sim.pupilSize)//2
        # [atm_size1:atm_size2,atm_size1:atm_size2]
        
        
        ZERO_IMAT[trial] = np.copy(SIM.recon.zero_iMat)
        ZERO_IMAT_intensity[trial] = np.copy(SIM.recon.zero_iMat_intensity)
        
        # plt.imshow(SIM.dms[2].aberration)
        # plt.title('aber')
        # plt.show()
        
        ABERATION_Z[trial] = z_decom.get_ZernikeCoeff(
            (SIM.dms[2].aberration[atm_size1:atm_size2,atm_size1:atm_size2]*PUPIL).flatten()*1e-9/wvl*2*np.pi,
            FlatGramSchmidt,
            GramSchmidtCoeff2ZernikeCoeff)
        delta = SIM.dms[2].telescope_diameter/SIM.dms[2].pupil_size
        phase = SIM.dms[2].aberration[atm_size1:atm_size2,atm_size1:atm_size2]*PUPIL * 1e-9/wvl*2*np.pi
        x = get_PSD(crop_inscribed_square(phase),delta)
        y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
        ABERATION_PSD_phase[trial] = np.copy(y)
        ABERATION_PSD_phase_frequency[trial] = ((np.arange(y.shape[-1]) + 1) * 1/(SIM.config.sim.pupilSize*delta))
        
        # try:
        #     difx = np.abs(SIM.dms[2].aberration[1:] - SIM.dms[2].aberration[:-1])[(A - B)//2 : (A + B)//2, (A - B)//2 : (A + B)//2][in_circle]
        #     dify = np.abs(SIM.dms[2].aberration[:,1:] - SIM.dms[2].aberration[:,:-1])[(A - B)//2 : (A + B)//2, (A - B)//2 : (A + B)//2][in_circle]
        
        #     if (difx/500*2.*3.14 > np.pi).any() or (dify/500*2.*3.14 > np.pi).any():
        #         # print('one of screen is not well sampled at 1')
        #         ABERATION[trial][:] = np.nan
        #         ZERO_IMAT[trial][:] = np.nan
                
        #         IM[trial][:] = np.nan
        #         CC[trial][:] = np.nan
                
        #         SR[trial][:] = np.nan
        #         WFE[trial][:] = np.nan
        #         perfect_SR[trial][:] = np.nan
        #         perfect_WFE[trial][:] = np.nan
    
        #         continue
        # except:
        #     pass
        
        IM[trial] = np.copy(SIM.recon.interaction_matrix)
        CC[trial] = np.copy(SIM.recon.control_matrix)
        
        MEASURED_IM_RYTOV[trial] = SIM.dms[1].rytov
        MEASURED_IM_SUBAP_SHIFT_influence[trial] = SIM.dms[1].subapShift_influence
        MEASURED_IM_SUBAP_SHIFT_slope[trial] = SIM.dms[1].subapShift_slope
        
        
        ###################################################################################
        
        for j in np.arange(SIM.atmos.scrns.shape[0]):
            SIM.atmos.infinite_phase_screens[j].random_seed = ATM_SEEDS[trial] + j
            SIM.atmos.infinite_phase_screens[j].make_initial_screen()
        SIM.atmos.moveScrns()
        
        # for j in np.arange(SIM.atmos.scrns.shape[0]):
        #     difx = np.abs(SIM.atmos.scrns[j,1:] - SIM.atmos.scrns[j,:-1])
        #     dify = np.abs(SIM.atmos.scrns[j,:,1:] - SIM.atmos.scrns[j,:,:-1])
        #     if (difx/500*2.*3.14 > np.pi).any() or (dify/500*2.*3.14 > np.pi).any():
        #         print('one of screen is not well sampled at 2')
        #         ABERATION[trial][:] = np.nan
        #         ZERO_IMAT[trial][:] = np.nan
                
        #         IM[trial][:] = np.nan
        #         CC[trial][:] = np.nan
                
        #         SR[trial][:] = np.nan
        #         WFE[trial][:] = np.nan
        #         perfect_SR[trial][:] = np.nan
        #         perfect_WFE[trial][:] = np.nan
        #         continue

        
        # plt.imshow(SIM.atmos.scrns[0])
        # plt.title('atmos')
        # plt.show()
            
        

        atm_size1 = (SIM.atmos.scrns[0].shape[0] - SIM.config.sim.pupilSize)//2
        atm_size2 = (SIM.atmos.scrns[0].shape[0] + SIM.config.sim.pupilSize)//2
        onpupil = SIM.atmos.scrns[0][atm_size1:atm_size2,atm_size1:atm_size2]
         
        ATM_Z[trial] = z_decom.get_ZernikeCoeff(
            (onpupil*PUPIL).flatten()*1e-9/wvl*2*np.pi,
            FlatGramSchmidt,
            GramSchmidtCoeff2ZernikeCoeff)
        
        delta = SIM.atmos.infinite_phase_screens[0].pixel_scale
        phase = onpupil * 1e-9/wvl*2*np.pi
        x = get_PSD(crop_inscribed_square(phase),delta)
        y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
        ATM_PSD_phase[trial] = np.copy(y)
        ATM_PSD_phase_frequency[trial] = ((np.arange(y.shape[-1]) + 1) * 1/(onpupil.shape[-1]*delta))
        
        #######################################################################
        # close loop with bad calibration
        for dm in range(SIM.config.sim.nDM):
            SIM.config.dms[dm].closed = True
        SIM.aoloop()
        RUNAWAY[trial,0] = SIM.run_away # close loop distort control
        RUNAWAY_POS[trial,0] = SIM.run_away_iteration
        SR[trial] = np.copy(SIM.instStrehl[0])
        WFE[trial] = np.copy(SIM.WFE[0])
        rytov[trial] = np.copy(SIM.rytov[0])
        SLOPES[trial] = np.copy(SIM.allSlopes)
        COMMANDS[trial] = np.copy(SIM.allDmCommands)
        RES[0,trial] = np.copy(SIM.sciPhase[0].real)
        RES[1,trial] = np.copy(SIM.sciPhase[0].imag)
        
        for loop in np.arange(LOOP):
        
            RES_phase = my_unwrap(np.nan_to_num(np.angle(SIM.sciPhase[0][loop])))
            
            RES_phase_Z[trial,loop] = z_decom.get_ZernikeCoeff(
                (RES_phase*PUPIL).flatten(),
                FlatGramSchmidt,
                GramSchmidtCoeff2ZernikeCoeff)
            
        # plt.imshow(SIM.recon.interaction_matrix)
        # plt.title('{:}{:}'.format(i_aber_z_list,trial))
        # plt.show()
        
        # ##########################################################################
        # # run distorted calibration again with open loop
        # # reset loop
        # SIM.reset_loop()
        # for j in np.arange(SIM.atmos.scrns.shape[0]):
        #     SIM.atmos.infinite_phase_screens[j].random_seed = ATM_SEEDS[trial] + j
        #     SIM.atmos.infinite_phase_screens[j].make_initial_screen()
        # SIM.atmos.moveScrns()
        # ##########################################################################
        # # run again with open loop
        # for dm in range(SIM.config.sim.nDM):
        #     SIM.config.dms[dm].closed = False
        # # plt.imshow(SIM.atmos.scrns[0])
        # # plt.title('atmos')
        # # plt.show()
        # SIM.aoloop()
        # RUNAWAY[trial,asdf1] = SIM.run_away # open loop distort control
        # RUNAWAY_POS[trial,asdf1] = SIM.run_away_iteration
        # SR_open[trial] = np.copy(SIM.instStrehl[0])
        # WFE_open[trial] = np.copy(SIM.WFE[0])
        # rytov_open[trial] = np.copy(SIM.rytov[0])
        # SLOPES_open[trial] = np.copy(SIM.allSlopes)
        # COMMANDS_open[trial] = np.copy(SIM.allDmCommands)
        # RES_open[0,trial] = np.copy(SIM.sciPhase[0].real)
        # RES_open[1,trial] = np.copy(SIM.sciPhase[0].imag)
        
        # for loop in np.arange(LOOP):
        
        #     RES_phase_open = my_unwrap(np.nan_to_num(np.angle(SIM.sciPhase[0][loop])))
            
        #     RES_phase_Z_open[trial,loop] = z_decom.get_ZernikeCoeff(
        #         (RES_phase_open*PUPIL).flatten(),
        #         FlatGramSchmidt,
        #         GramSchmidtCoeff2ZernikeCoeff)
        
        ##########################################################################
        # run with good calibration with close loop
        # reset loop
        SIM.reset_loop()
        for j in np.arange(SIM.atmos.scrns.shape[0]):
            SIM.atmos.infinite_phase_screens[j].random_seed = ATM_SEEDS[trial] + j
            SIM.atmos.infinite_phase_screens[j].make_initial_screen()
        SIM.atmos.moveScrns()
        
        # print('\n'+'G{:}={:}'.format(trial,SIM.atmos.scrns[0,56,23])+'\n')
        # plt.imshow(SIM.atmos.scrns[0])
        # plt.title('G{:}={:}'.format(trial,SIM.atmos.scrns[0,56,23]))
        # plt.savefig('G{:}={:}'.format(trial,SIM.atmos.scrns[0,56,23])+'.png')
        # plt.clf()
        
        # load no distortion IM
        SIM.recon.interaction_matrix = np.copy(perfect_IM)
        SIM.recon.control_matrix = np.copy(perfect_CC)
        
        # close loop
        for dm in range(SIM.config.sim.nDM):
            SIM.config.dms[dm].closed = True
        
        SIM.aoloop()
        RUNAWAY[trial,1] = SIM.run_away # close loop good control
        RUNAWAY_POS[trial,1] = SIM.run_away_iteration
        # plt.imshow(SIM.atmos.scrns[0])
        # plt.title('atmos')
        # plt.show()
        
        perfect_SR[trial] = np.copy(SIM.instStrehl[0])
        perfect_WFE[trial] = np.copy(SIM.WFE[0])

        perfect_SLOPES[trial] = np.copy(SIM.allSlopes)
        perfect_COMMANDS[trial] = np.copy(SIM.allDmCommands)
        perfect_RES[0,trial] = np.copy(SIM.sciPhase[0].real)
        perfect_RES[1,trial] = np.copy(SIM.sciPhase[0].imag)
        perfect_rytov[trial] = np.copy(SIM.rytov[0])
        
        for loop in np.arange(LOOP):
        
            perfect_RES_phase = my_unwrap(np.nan_to_num(np.angle(SIM.sciPhase[0][loop])))
            
            perfect_RES_phase_Z[trial,loop] = z_decom.get_ZernikeCoeff(
                (perfect_RES_phase*PUPIL).flatten(),
                FlatGramSchmidt,
                GramSchmidtCoeff2ZernikeCoeff)
        
        # # SLOPES[trial] = np.copy(SIM.allSlopes)
        # plt.imshow(SIM.allSlopes)
        # plt.title('perfect_slopes{:d}'.format(trial))
        # plt.colorbar()
        # plt.show()
        # # COMMANDS[trial] = np.copy(SIM.allDmCommands)
        # plt.imshow(SIM.allDmCommands)
        # plt.title('perfect_commands{:d}'.format(trial))
        # plt.colorbar()
        # plt.show()
        
        ATM_SLOPES[trial] = np.copy(SIM.allAtmSlopes[0])
        
        # ##########################################################################
        # # reset # do open loop # with no distortion IM
        
        # # reset
        # SIM.reset_loop()
        
        # # reset atm
        
        # for j in np.arange(SIM.atmos.scrns.shape[0]):
        #     SIM.atmos.infinite_phase_screens[j].random_seed = ATM_SEEDS[trial] + j
        #     SIM.atmos.infinite_phase_screens[j].make_initial_screen()
        # SIM.atmos.moveScrns()
        
        # # no need to reload IM
        
        # # do open loop
        # for dm in range(SIM.config.sim.nDM):
        #     SIM.config.dms[dm].closed = False
        
        # # redo
        
        # SIM.aoloop()
        # RUNAWAY[trial,asdf3] = SIM.run_away # open loop good control
        # RUNAWAY_POS[trial,asdf3] = SIM.run_away_iteration
        # # plt.imshow(SIM.atmos.scrns[0])
        # # plt.title('atmos')
        # # plt.show()
        
        # perfect_SR_open[trial] = np.copy(SIM.instStrehl[0])
        # perfect_WFE_open[trial] = np.copy(SIM.WFE[0])

        # perfect_SLOPES_open[trial] = np.copy(SIM.allSlopes)
        # perfect_COMMANDS_open[trial] = np.copy(SIM.allDmCommands)
        # perfect_RES_open[0,trial] = np.copy(SIM.sciPhase[0].real)
        # perfect_RES_open[1,trial] = np.copy(SIM.sciPhase[0].imag)
        # perfect_rytov_open[trial] = np.copy(SIM.rytov[0])
        
        # for loop in np.arange(LOOP):
        
        #     perfect_RES_phase_open = my_unwrap(np.nan_to_num(np.angle(SIM.sciPhase[0][loop])))
            
        #     perfect_RES_phase_Z_open[trial,loop] = z_decom.get_ZernikeCoeff(
        #         (perfect_RES_phase_open*PUPIL).flatten(),
        #         FlatGramSchmidt,
        #         GramSchmidtCoeff2ZernikeCoeff)
        
        # # # SLOPES[trial] = np.copy(SIM.allSlopes)
        # # plt.imshow(SIM.allSlopes)
        # # plt.title('perfect_slopes{:d}'.format(trial))
        # # plt.colorbar()
        # # plt.show()
        # # # COMMANDS[trial] = np.copy(SIM.allDmCommands)
        # # plt.imshow(SIM.allDmCommands)
        # # plt.title('perfect_commands{:d}'.format(trial))
        # # plt.colorbar()
        # # plt.show()
        
        ##########################################################################
        # clear data        
        SIM.reset_loop()
    
        try:
            del SIM.dms[2].aberration
        except:
            pass
        try:
            del SIM.dms[2].noll_variances
            print('i am not ready for this type of dm')
        except:
            pass
        try:
            del SIM.dms[2].aberrationStrength
            print('i am not ready for this type of dm')
        except:
            pass
    
    # A = perfect_RES_open[0] + 1j*perfect_RES_open[1]
    # delta = SIM.sciCams[0].telescope_diameter/SIM.sciCams[0].pupil_size
    # phase = np.zeros((perfect_RES_open.shape[1],
    #                   perfect_RES_open.shape[2],
    #                   perfect_RES_open.shape[3],
    #                   perfect_RES_open.shape[4]))
    # for ii in np.arange(phase.shape[0]):
    #     for jj in np.arange(phase.shape[1]):
    #         phase[ii,jj] = my_unwrap(np.angle(np.nan_to_num(A[ii,jj])))
    # x = get_PSD(crop_inscribed_square(phase[0,0]),delta)
    # y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]

    # Y = np.zeros((TRIAL,LOOP,y.shape[0]),dtype=float)

    # for trial in np.arange(TRIAL):
    #     for loop in np.arange(LOOP):
    #         x = get_PSD(crop_inscribed_square(phase[trial,loop]), delta)
    #         y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
    #         Y[trial,loop] = y

    # y = np.nanmean(Y,axis=0)
    # x = ((np.arange(y.shape[-1]) + 1) * 1/(INSCRIBED_SIZE*delta))
    # # perfect_PSD= np.copy(x)
    # perfect_PSD_phase_open = np.copy(Y)
    # # for loop in np.array([0,-1]):#np.arange(LOOP):
    # #     plt.plot(x,y[loop],label='no aberration loop#{:d}'.format(loop))
    # # plt.xscale('log')
    # # plt.yscale('log')
    # # plt.xlabel('spatial frequency (1/m)')
    # # plt.ylabel('PSD of phase (rad$^2$m$^2$)')
    # # plt.legend()
    # # plt.savefig('perfect_res_PSD' + '_' + name  + '.png', bbox_inches='tight')
    # # plt.show()
    
    # # for frequency in np.arange(x.shape[0]//10):
    # #     plt.plot(np.arange(LOOP),y[:,frequency*10],label='no aberation freq={:.2e} m$^-1$'.format(x[frequency*10]))
    # # # plt.xscale('log')
    # # plt.yscale('log')
    # # plt.xlabel('loop#')
    # # plt.ylabel('PSD of phase (rad$^2$m$^2$)')
    # # plt.legend()
    # # # plt.savefig('bad_res_PSD' + '_' + name  + '.png', bbox_inches='tight')
    # # plt.savefig('res_PSD' + '_' + name  + '.png', bbox_inches='tight')
    # # plt.show()
    
    # intensity = np.abs(np.nan_to_num(A))**2
    # x = get_PSD(crop_inscribed_square(intensity[0,0]),delta)
    # y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
    # Y = np.zeros((TRIAL,LOOP,y.shape[0]),dtype=float)
    # for trial in np.arange(TRIAL):
    #     for loop in np.arange(LOOP):
    #         x = get_PSD(crop_inscribed_square(intensity[trial,loop]), delta)
    #         y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
    #         Y[trial,loop] = y
    # y = np.nanmean(Y,axis=0)
    # perfect_PSD_intensity_open = np.copy(Y)
    
    # U = np.nan_to_num(A)
    # x = get_PSD(crop_inscribed_square(U[0,0]),delta)
    # y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
    # Y = np.zeros((TRIAL,LOOP,y.shape[0]),dtype=float)
    # for trial in np.arange(TRIAL):
    #     for loop in np.arange(LOOP):
    #         x = get_PSD(crop_inscribed_square(U[trial,loop]), delta)
    #         y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
    #         Y[trial,loop] = y
    # y = np.nanmean(Y,axis=0)
    # perfect_PSD_U_open = np.copy(Y)
    
    
    A = perfect_RES[0] + 1j*perfect_RES[1]
    delta = SIM.sciCams[0].telescope_diameter/SIM.sciCams[0].pupil_size
    phase = np.zeros((perfect_RES.shape[1],
                      perfect_RES.shape[2],
                      perfect_RES.shape[3],
                      perfect_RES.shape[4]))
    for ii in np.arange(phase.shape[0]):
        for jj in np.arange(phase.shape[1]):
            phase[ii,jj] = my_unwrap(np.angle(np.nan_to_num(A[ii,jj])))
    x = get_PSD(crop_inscribed_square(phase[0,0]),delta)
    INSCRIBED_SIZE = crop_inscribed_square(phase[0,0]).shape[-1]
    y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]

    Y = np.zeros((TRIAL,LOOP,y.shape[0]),dtype=float)

    for trial in np.arange(TRIAL):
        for loop in np.arange(LOOP):
            x = get_PSD(crop_inscribed_square(phase[trial,loop]), delta)
            y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
            Y[trial,loop] = y

    y = np.nanmean(Y,axis=0)
    x = ((np.arange(y.shape[-1]) + 1) * 1/(INSCRIBED_SIZE*delta))
    # perfect_PSD= np.copy(x)
    perfect_PSD_phase = np.copy(Y)
    # for loop in np.array([0,-1]):#np.arange(LOOP):
    #     plt.plot(x,y[loop],label='no aberration loop#{:d}'.format(loop))
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('spatial frequency (1/m)')
    # plt.ylabel('PSD of phase (rad$^2$m$^2$)')
    # plt.legend()
    # plt.savefig('perfect_res_PSD' + '_' + name  + '.png', bbox_inches='tight')
    # plt.show()
    
    # for frequency in np.arange(x.shape[0]//10):
    #     plt.plot(np.arange(LOOP),y[:,frequency*10],label='no aberation freq={:.2e} m$^-1$'.format(x[frequency*10]))
    # # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('loop#')
    # plt.ylabel('PSD of phase (rad$^2$m$^2$)')
    # plt.legend()
    # # plt.savefig('bad_res_PSD' + '_' + name  + '.png', bbox_inches='tight')
    # plt.savefig('res_PSD' + '_' + name  + '.png', bbox_inches='tight')
    # plt.show()
    
    intensity = np.abs(np.nan_to_num(A))**2
    x = get_PSD(crop_inscribed_square(intensity[0,0]),delta)
    y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
    Y = np.zeros((TRIAL,LOOP,y.shape[0]),dtype=float)
    for trial in np.arange(TRIAL):
        for loop in np.arange(LOOP):
            x = get_PSD(crop_inscribed_square(intensity[trial,loop]), delta)
            y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
            Y[trial,loop] = y
    y = np.nanmean(Y,axis=0)
    perfect_PSD_intensity = np.copy(Y)
    
    U = np.nan_to_num(A)
    x = get_PSD(crop_inscribed_square(U[0,0]),delta)
    y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
    Y = np.zeros((TRIAL,LOOP,y.shape[0]),dtype=float)
    for trial in np.arange(TRIAL):
        for loop in np.arange(LOOP):
            x = get_PSD(crop_inscribed_square(U[trial,loop]), delta)
            y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
            Y[trial,loop] = y
    y = np.nanmean(Y,axis=0)
    perfect_PSD_U = np.copy(Y)
    
    
    
    A = RES[0] + 1j*RES[1]
    delta = SIM.sciCams[0].telescope_diameter/SIM.sciCams[0].pupil_size
    phase = np.zeros((RES.shape[1],
                      RES.shape[2],
                      RES.shape[3],
                      RES.shape[4]))
    for ii in np.arange(phase.shape[0]):
        for jj in np.arange(phase.shape[1]):
            phase[ii,jj] = my_unwrap(np.angle(np.nan_to_num(A[ii,jj])))
    x = get_PSD(crop_inscribed_square(phase[0,0]),delta)
    y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]

    Y = np.zeros((TRIAL,LOOP,y.shape[0]),dtype=float)

    for trial in np.arange(TRIAL):
        for loop in np.arange(LOOP):
            x = get_PSD(crop_inscribed_square(phase[trial,loop]), delta)
            y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
            Y[trial,loop] = y

    y = np.nanmean(Y,axis=0)
    x = ((np.arange(y.shape[-1]) + 1) * 1/(INSCRIBED_SIZE*delta))
    # x = ((np.arange(y.shape[-1]) + 1) * 1/(A.shape[-1]*delta))
    # # this should not be A.shape because i use a smaller inscribed square!!!
    # # so the frequency is wrong
    # # should adjust A.shape[-1] ==> A.shape[-1]/np.sqrt(2)
    # # if too lazy to change code, or for older result than 20 nov 24
    # # multiply freq by sqrt(2)
    # # no change in psd because it only take delta which is unchange
    # # nnah that didn't really work. best to just change x = ...
    # if_you_are_running this again, fix the thing on the top throughout this file.
    PSD_phase_frequency = np.copy(x)
    PSD_phase = np.copy(Y)
    # for loop in np.array([0,-1]):#np.arange(LOOP):
    #     # scale = x
    #     data = y[loop]
    #     # new_scale, new_data, new_data_error = bin_linear_to_log10(data, scale)
    #     plt.plot(x,y[loop],label='with aberration loop#{:d}'.format(loop))
    #     # plt.errorbar(new_scale,new_data,yerr=new_data_error,label='loop#{:d}'.format(loop),capsize=5,marker='o')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('spatial frequency (1/m)')
    # plt.ylabel('PSD of phase (rad$^2$m$^2$)')
    # plt.legend()
    # # plt.savefig('bad_res_PSD' + '_' + name  + '.png', bbox_inches='tight')
    # plt.savefig('res_PSD_phase' + '_' + name  + '.png', bbox_inches='tight')
    # plt.show()
    # plt.clf()
    
    # for frequency in np.arange(x.shape[0]//10):
    #     plt.plot(np.arange(LOOP),y[:,frequency*10],label='with aberation freq={:.2e} m$^-1$'.format(x[frequency*10]))
    # # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('loop#')
    # plt.ylabel('PSD of phase (rad$^2$m$^2$)')
    # plt.legend()
    # # plt.savefig('bad_res_PSD' + '_' + name  + '.png', bbox_inches='tight')
    # plt.savefig('res_PSD' + '_' + name  + '.png', bbox_inches='tight')
    # plt.show()
    
    intensity = np.abs(np.nan_to_num(A))**2
    x = get_PSD(crop_inscribed_square(intensity[0,0]),delta)
    y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
    Y = np.zeros((TRIAL,LOOP,y.shape[0]),dtype=float)
    for trial in np.arange(TRIAL):
        for loop in np.arange(LOOP):
            x = get_PSD(crop_inscribed_square(intensity[trial,loop]), delta)
            y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
            Y[trial,loop] = y
    y = np.nanmean(Y,axis=0)
    PSD_intensity = np.copy(Y)
    
    U = np.nan_to_num(A)
    x = get_PSD(crop_inscribed_square(U[0,0]),delta)
    y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
    Y = np.zeros((TRIAL,LOOP,y.shape[0]),dtype=float)
    for trial in np.arange(TRIAL):
        for loop in np.arange(LOOP):
            x = get_PSD(crop_inscribed_square(U[trial,loop]), delta)
            y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
            Y[trial,loop] = y
    y = np.nanmean(Y,axis=0)
    PSD_U = np.copy(Y)
    
    
    
    
    
    # A = RES_open[0] + 1j*RES_open[1]
    # delta = SIM.sciCams[0].telescope_diameter/SIM.sciCams[0].pupil_size
    # phase = np.zeros((RES_open.shape[1],
    #                   RES_open.shape[2],
    #                   RES_open.shape[3],
    #                   RES_open.shape[4]))
    # for ii in np.arange(phase.shape[0]):
    #     for jj in np.arange(phase.shape[1]):
    #         phase[ii,jj] = my_unwrap(np.angle(np.nan_to_num(A[ii,jj])))
    # x = get_PSD(crop_inscribed_square(phase[0,0]),delta)
    # y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]

    # Y = np.zeros((TRIAL,LOOP,y.shape[0]),dtype=float)

    # for trial in np.arange(TRIAL):
    #     for loop in np.arange(LOOP):
    #         x = get_PSD(crop_inscribed_square(phase[trial,loop]), delta)
    #         y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
    #         Y[trial,loop] = y

    # y = np.nanmean(Y,axis=0)
    # x = ((np.arange(y.shape[-1]) + 1) * 1/(INSCRIBED_SIZE*delta))
    # PSD_phase_frequency_open = np.copy(x)
    # PSD_phase_open = np.copy(Y)
    # # for loop in np.array([0,-1]):#np.arange(LOOP):
    # #     # scale = x
    # #     data = y[loop]
    # #     # new_scale, new_data, new_data_error = bin_linear_to_log10(data, scale)
    # #     plt.plot(x,y[loop],label='with aberration loop#{:d}'.format(loop))
    # #     # plt.errorbar(new_scale,new_data,yerr=new_data_error,label='loop#{:d}'.format(loop),capsize=5,marker='o')
    # # plt.xscale('log')
    # # plt.yscale('log')
    # # plt.xlabel('spatial frequency (1/m)')
    # # plt.ylabel('PSD of phase (rad$^2$m$^2$)')
    # # plt.legend()
    # # # plt.savefig('bad_res_PSD' + '_' + name  + '.png', bbox_inches='tight')
    # # plt.savefig('res_PSD_phase' + '_' + name  + '.png', bbox_inches='tight')
    # # plt.show()
    # # plt.clf()
    
    # # for frequency in np.arange(x.shape[0]//10):
    # #     plt.plot(np.arange(LOOP),y[:,frequency*10],label='with aberation freq={:.2e} m$^-1$'.format(x[frequency*10]))
    # # # plt.xscale('log')
    # # plt.yscale('log')
    # # plt.xlabel('loop#')
    # # plt.ylabel('PSD of phase (rad$^2$m$^2$)')
    # # plt.legend()
    # # # plt.savefig('bad_res_PSD' + '_' + name  + '.png', bbox_inches='tight')
    # # plt.savefig('res_PSD' + '_' + name  + '.png', bbox_inches='tight')
    # # plt.show()
    
    # intensity = np.abs(np.nan_to_num(A))**2
    # x = get_PSD(crop_inscribed_square(intensity[0,0]),delta)
    # y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
    # Y = np.zeros((TRIAL,LOOP,y.shape[0]),dtype=float)
    # for trial in np.arange(TRIAL):
    #     for loop in np.arange(LOOP):
    #         x = get_PSD(crop_inscribed_square(intensity[trial,loop]), delta)
    #         y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
    #         Y[trial,loop] = y
    # y = np.nanmean(Y,axis=0)
    # PSD_intensity_open = np.copy(Y)
    
    # U = np.nan_to_num(A)
    # x = get_PSD(crop_inscribed_square(U[0,0]),delta)
    # y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
    # Y = np.zeros((TRIAL,LOOP,y.shape[0]),dtype=float)
    # for trial in np.arange(TRIAL):
    #     for loop in np.arange(LOOP):
    #         x = get_PSD(crop_inscribed_square(U[trial,loop]), delta)
    #         y = x[x.shape[0]//2 + 1 : ,x.shape[0]//2]
    #         Y[trial,loop] = y
    # y = np.nanmean(Y,axis=0)
    # PSD_U_open = np.copy(Y)
    
    
    

        
    hdu = fits.PrimaryHDU(PSD_phase_frequency)
    hdul = fits.HDUList([hdu])
    hdul.writeto('PSD_phase_frequency' + '_' + name  + '.fits', overwrite=True)
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('PSD_phase_frequency' + '\n')
    
    hdu = fits.PrimaryHDU(PSD_phase)
    hdul = fits.HDUList([hdu])
    hdul.writeto('PSD_phase' + '_' + name  + '.fits', overwrite=True)
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('PSD_phase' + '\n')
    
    hdu = fits.PrimaryHDU(perfect_PSD_phase)
    hdul = fits.HDUList([hdu])
    hdul.writeto('perfect_PSD_phase' + '_' + name  + '.fits', overwrite=True)
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('perfect_PSD_phase' + '\n')
    
    hdu = fits.PrimaryHDU(PSD_intensity)
    hdul = fits.HDUList([hdu])
    hdul.writeto('PSD_intensity' + '_' + name  + '.fits', overwrite=True)
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('PSD_intensity' + '\n')
    
    hdu = fits.PrimaryHDU(perfect_PSD_intensity)
    hdul = fits.HDUList([hdu])
    hdul.writeto('perfect_PSD_intensity' + '_' + name  + '.fits', overwrite=True)
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('perfect_PSD_intensity' + '\n')
    
    hdu = fits.PrimaryHDU(PSD_U)
    hdul = fits.HDUList([hdu])
    hdul.writeto('PSD_U' + '_' + name  + '.fits', overwrite=True)
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('PSD_U' + '\n')
    
    hdu = fits.PrimaryHDU(perfect_PSD_U)
    hdul = fits.HDUList([hdu])
    hdul.writeto('perfect_PSD_U' + '_' + name  + '.fits', overwrite=True)
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('perfect_PSD_U' + '\n')
          
    
    # hdu = fits.PrimaryHDU(PSD_phase_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('PSD_phase_open' + '_' + name  + '.fits', overwrite=True)
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('PSD_phase_open' + '\n')
    
    # hdu = fits.PrimaryHDU(perfect_PSD_phase_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('perfect_PSD_phase_open' + '_' + name  + '.fits', overwrite=True)
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('perfect_PSD_phase_open' + '\n')
    
    # hdu = fits.PrimaryHDU(PSD_intensity_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('PSD_intensity_open' + '_' + name  + '.fits', overwrite=True)
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('PSD_intensity_open' + '\n')
    
    # hdu = fits.PrimaryHDU(perfect_PSD_intensity_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('perfect_PSD_intensity_open' + '_' + name  + '.fits', overwrite=True)
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('perfect_PSD_intensity_open' + '\n')
    
    # hdu = fits.PrimaryHDU(PSD_U_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('PSD_U_open' + '_' + name  + '.fits', overwrite=True)
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('PSD_U_open' + '\n')
    
    # hdu = fits.PrimaryHDU(perfect_PSD_U_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('perfect_PSD_U_open' + '_' + name  + '.fits', overwrite=True)
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('perfect_PSD_U_open' + '\n')
          
          
          
          
    hdu = fits.PrimaryHDU(RES_phase_Z)
    hdul = fits.HDUList([hdu])
    hdul.writeto('RES_phase_Z' + '_' + name  + '.fits', overwrite=True)
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('RES_phase_Z' + '\n')
    
    hdu = fits.PrimaryHDU(perfect_RES_phase_Z)
    hdul = fits.HDUList([hdu])
    hdul.writeto('perfect_RES_phase_Z' + '_' + name  + '.fits', overwrite=True)
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('perfect_RES_phase_Z' + '\n')
          
    # hdu = fits.PrimaryHDU(RES_phase_Z_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('RES_phase_Z_open' + '_' + name  + '.fits', overwrite=True)
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('RES_phase_Z_open' + '\n')
    
    # hdu = fits.PrimaryHDU(perfect_RES_phase_Z_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('perfect_RES_phase_Z_open' + '_' + name  + '.fits', overwrite=True)
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('perfect_RES_phase_Z_open' + '\n')
    

    #####################################################################
    # calibration data
    # 1) aberation phase, slope
    # 2) IM with/without aberation
    # 3) CM with/without aberation # svds are not saved because actually using constant svd is better
    # 4) aberration seed # reproducibility

    # hdu = fits.PrimaryHDU(ABERATION)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('ABERATION_phase' + '_' + name  + '.fits', overwrite=True)
    
    hdu = fits.PrimaryHDU(aberation_seed)
    hdul = fits.HDUList([hdu])
    hdul.writeto('aberation_seed' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('aberation_seed' + '\n')
    
    hdu = fits.PrimaryHDU(ABERATION_Z)
    hdul = fits.HDUList([hdu])
    hdul.writeto('ABERATION_Z' + '_' + name  + '.fits', overwrite=True)
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('ABERATION_Z' + '\n')

    hdu = fits.PrimaryHDU(ABERATION_PSD_phase)
    hdul = fits.HDUList([hdu])
    hdul.writeto('ABERATION_PSD_phase' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('ABERATION_PSD_phase' + '\n')

    hdu = fits.PrimaryHDU(ABERATION_PSD_phase_frequency)
    hdul = fits.HDUList([hdu])
    hdul.writeto('ABERATION_PSD_phase_frequency' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('ABERATION_PSD_phase_frequency' + '\n')

    hdu = fits.PrimaryHDU(ZERO_IMAT)
    hdul = fits.HDUList([hdu])
    hdul.writeto('ABERATION_slope' + '_' + name  + '.fits', overwrite=True)
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('ABERATION_slope' + '\n')
          
    hdu = fits.PrimaryHDU(ZERO_IMAT_intensity)
    hdul = fits.HDUList([hdu])
    hdul.writeto('ABERATION_slope_intensity' + '_' + name  + '.fits', overwrite=True)
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('ABERATION_slope_intensity' + '\n')
          
          
    
    hdu = fits.PrimaryHDU(perfect_IM)
    hdul = fits.HDUList([hdu])
    hdul.writeto('perfect_IM' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('perfect_IM' + '\n')
    
    hdu = fits.PrimaryHDU(perfect_CC)
    hdul = fits.HDUList([hdu])
    hdul.writeto('perfect_CC' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('perfect_CC' + '\n')
          
          

    hdu = fits.PrimaryHDU(IM)
    hdul = fits.HDUList([hdu])
    hdul.writeto('IM' + '_' + name  + '.fits', overwrite=True)
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('IM' + '\n')
    
    
    hdu = fits.PrimaryHDU(CC)
    hdul = fits.HDUList([hdu])
    hdul.writeto('CC' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('CC' + '\n')
    
    
    
    
    hdu = fits.PrimaryHDU(MEASURED_IM_RYTOV)
    hdul = fits.HDUList([hdu])
    hdul.writeto('MEASURED_IM_RYTOV' + '_' + name  + '.fits', overwrite=True)
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('MEASURED_IM_RYTOV' + '\n')
    
    
    hdu = fits.PrimaryHDU(MEASURED_IM_SUBAP_SHIFT_influence)
    hdul = fits.HDUList([hdu])
    hdul.writeto('MEASURED_IM_SUBAP_SHIFT_influence' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('MEASURED_IM_SUBAP_SHIFT_influence' + '\n')
    
    hdu = fits.PrimaryHDU(MEASURED_IM_SUBAP_SHIFT_slope)
    hdul = fits.HDUList([hdu])
    hdul.writeto('MEASURED_IM_SUBAP_SHIFT_slope' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('MEASURED_IM_SUBAP_SHIFT_slope' + '\n')
    

    #######################################################################
    # 0th order ao performance data
    # 1) strehl ratio of distort and not distort
    # 2) wavefront rms error of distort and not distort

    hdu = fits.PrimaryHDU(SR)
    hdul = fits.HDUList([hdu])
    hdul.writeto('SR' + '_' + name + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('SR' + '\n')
    
    hdu = fits.PrimaryHDU(WFE)
    hdul = fits.HDUList([hdu])
    hdul.writeto('WFE' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('WFE' + '\n')
          
    hdu = fits.PrimaryHDU(rytov)
    hdul = fits.HDUList([hdu])
    hdul.writeto('rytov' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('rytov' + '\n')
          
    hdu = fits.PrimaryHDU(perfect_rytov)
    hdul = fits.HDUList([hdu])
    hdul.writeto('perfect_rytov' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('perfect_rytov' + '\n')
    
    hdu = fits.PrimaryHDU(perfect_SR)
    hdul = fits.HDUList([hdu])
    hdul.writeto('perfect_SR' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('perfect_SR' + '\n')
    
    hdu = fits.PrimaryHDU(perfect_WFE)
    hdul = fits.HDUList([hdu])
    hdul.writeto('perfect_WFE' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('perfect_WFE' + '\n')
          
          
    # hdu = fits.PrimaryHDU(SR_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('SR_open' + '_' + name + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('SR_open' + '\n')
    
    # hdu = fits.PrimaryHDU(WFE_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('WFE_open' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('WFE_open' + '\n')
          
    # hdu = fits.PrimaryHDU(rytov_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('rytov_open' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('rytov_open' + '\n')
          
    # hdu = fits.PrimaryHDU(perfect_rytov_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('perfect_rytov_open' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('perfect_rytov_open' + '\n')
    
    # hdu = fits.PrimaryHDU(perfect_SR_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('perfect_SR_open' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('perfect_SR_open' + '\n')
    
    # hdu = fits.PrimaryHDU(perfect_WFE_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('perfect_WFE_open' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('perfect_WFE_open' + '\n')

    #####################################################################
    # 1st order ao analysis
    # 1) atmosphere seed
    # 2) atmosphere phase
    # 3) atmosphere slope
    # 4) dm strokes
    # 5) wfs slopes
    # # 6) wfs detector
    # 7) residual phase
    # 8) residual slope
    # # 9) science image

    hdu = fits.PrimaryHDU(ATM_SEEDS)
    hdul = fits.HDUList([hdu])
    hdul.writeto('ATM_SEEDS' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('ATM_SEEDS' + '\n')
    
    hdu = fits.PrimaryHDU(ATM_PSD_phase)
    hdul = fits.HDUList([hdu])
    hdul.writeto('ATM_PSD_phase' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('ATM_PSD_phase' + '\n')

    hdu = fits.PrimaryHDU(ATM_PSD_phase_frequency)
    hdul = fits.HDUList([hdu])
    hdul.writeto('ATM_PSD_phase_frequency' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('ATM_PSD_phase_frequency' + '\n')

    hdu = fits.PrimaryHDU(ATM_Z)
    hdul = fits.HDUList([hdu])
    hdul.writeto('ATM_Z' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('ATM_Z' + '\n')

    hdu = fits.PrimaryHDU(SLOPES)
    hdul = fits.HDUList([hdu])
    hdul.writeto('SLOPES' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('SLOPES' + '\n')
          
    # hdu = fits.PrimaryHDU(SLOPES_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('SLOPES_open' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('SLOPES_open' + '\n')
    
    hdu = fits.PrimaryHDU(ATM_SLOPES)
    hdul = fits.HDUList([hdu])
    hdul.writeto('ATM_SLOPES' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('ATM_SLOPES' + '\n')

    hdu = fits.PrimaryHDU(COMMANDS)
    hdul = fits.HDUList([hdu])
    hdul.writeto('COMMANDS' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('COMMANDS' + '\n')
          
    # hdu = fits.PrimaryHDU(COMMANDS_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('COMMANDS_open' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('COMMANDS_open' + '\n')

    hdu = fits.PrimaryHDU(perfect_SLOPES)
    hdul = fits.HDUList([hdu])
    hdul.writeto('perfect_SLOPES' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('perfect_SLOPES' + '\n')

    hdu = fits.PrimaryHDU(perfect_COMMANDS)
    hdul = fits.HDUList([hdu])
    hdul.writeto('perfect_COMMANDS' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('perfect_COMMANDS' + '\n')
          
    # hdu = fits.PrimaryHDU(perfect_SLOPES_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('perfect_SLOPES_open' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('perfect_SLOPES_open' + '\n')

    # hdu = fits.PrimaryHDU(perfect_COMMANDS_open)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('perfect_COMMANDS_open' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('perfect_COMMANDS_open' + '\n')
          
          
    hdu = fits.PrimaryHDU(RES)
    hdul = fits.HDUList([hdu])
    hdul.writeto('RES' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('RES' + '\n')
    
    hdu = fits.PrimaryHDU(perfect_RES)
    hdul = fits.HDUList([hdu])
    hdul.writeto('perfect_RES' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('perfect_RES' + '\n')
          

    # hdu = fits.PrimaryHDU(RES[:,:,0])
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('RES0' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('RES0' + '\n')

    # hdu = fits.PrimaryHDU(perfect_RES[:,:,0])
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('perfect_RES0' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('perfect_RES0' + '\n')

    # hdu = fits.PrimaryHDU(RES[:,:,1])
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('RES1' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('RES1' + '\n')

    # hdu = fits.PrimaryHDU(perfect_RES[:,:,1])
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('perfect_RES1' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('perfect_RES1' + '\n')

    # hdu = fits.PrimaryHDU(RES[:,:,2])
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('RES2' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('RES2' + '\n')

    # hdu = fits.PrimaryHDU(perfect_RES[:,:,2])
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('perfect_RES2' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('perfect_RES2' + '\n')

    # hdu = fits.PrimaryHDU(RES[:,:,3])
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('RES3' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('RES3' + '\n')

    # hdu = fits.PrimaryHDU(perfect_RES[:,:,3])
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('perfect_RES3' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('perfect_RES3' + '\n')

    # LAST_RES = np.zeros((RES.shape[0],RES.shape[1],RES.shape[-2],RES.shape[-1]),dtype=float)
    # for i in np.arange(RES.shape[1]):
    #     LAST_RES[:,i] = RES[:,i,RUNAWAY_POS[i,0]]
    #     plt.imshow(np.angle(LAST_RES[0,i] + 1j*LAST_RES[1,i]))
    #     plt.colorbar()
    #     plt.show()
    # hdu = fits.PrimaryHDU(LAST_RES)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('RES_1' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('RES_1' + '\n')
          
    # LAST_RES = np.zeros((perfect_RES.shape[0],perfect_RES.shape[1],perfect_RES.shape[-2],perfect_RES.shape[-1]),dtype=float)
    # for i in np.arange(perfect_RES.shape[1]):
    #     LAST_RES[:,i] = perfect_RES[:,i,RUNAWAY_POS[i,1]]

    # hdu = fits.PrimaryHDU(LAST_RES)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('perfect_RES_1' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('perfect_RES_1' + '\n')
          
          
    # hdu = fits.PrimaryHDU(RES_open[:,:,0])
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('RES0_open' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('RES0_open' + '\n')

    # hdu = fits.PrimaryHDU(perfect_RES_open[:,:,0])
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('perfect_RES0_open' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('perfect_RES0_open' + '\n')

    # hdu = fits.PrimaryHDU(RES_open[:,:,1])
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('RES1_open' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('RES1_open' + '\n')

    # hdu = fits.PrimaryHDU(perfect_RES_open[:,:,1])
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('perfect_RES1_open' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('perfect_RES1_open' + '\n')

    # hdu = fits.PrimaryHDU(RES_open[:,:,2])
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('RES2_open' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('RES2_open' + '\n')

    # hdu = fits.PrimaryHDU(perfect_RES_open[:,:,2])
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('perfect_RES2_open' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('perfect_RES2_open' + '\n')

    # hdu = fits.PrimaryHDU(RES_open[:,:,3])
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('RES3_open' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('RES3_open' + '\n')

    # hdu = fits.PrimaryHDU(perfect_RES_open[:,:,3])
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('perfect_RES3_open' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('perfect_RES3_open' + '\n')

    # hdu = fits.PrimaryHDU(RES_open[:,:,-1])
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('RES_1_open' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('RES_1_open' + '\n')

    # hdu = fits.PrimaryHDU(perfect_RES_open[:,:,-1])
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('perfect_RES_1_open' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    # if FIRSTTIME == True:
    #     with open('data_space.txt', 'a') as outfile:
    #       outfile.write('perfect_RES_1_open' + '\n')
          
    hdu = fits.PrimaryHDU(np.asarray(RUNAWAY,dtype=int))
    hdul = fits.HDUList([hdu])
    hdul.writeto('RUNAWAY' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('RUNAWAY' + '\n')
    
    hdu = fits.PrimaryHDU(RUNAWAY_POS)
    hdul = fits.HDUList([hdu])
    hdul.writeto('RUNAWAY_POS' + '_' + name  + '.fits', overwrite=True)
    hdul.close()
    if FIRSTTIME == True:
        with open('data_space.txt', 'a') as outfile:
          outfile.write('RUNAWAY_POS' + '\n')
    
    # hdu = fits.PrimaryHDU(psf)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('psf' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    
    # hdu = fits.PrimaryHDU(perfect_psf)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto('psf' + '_' + name  + '.fits', overwrite=True)
    # hdul.close()
    
    ############################################################################
    



    #####################################################################
    
    # os.chdir(home)

    # plt.errorbar(np.arange(LOOP), np.nanmean(SR,axis=0),
    #               yerr=np.nanstd(SR,axis=0)#/SR.shape[0]**0.5
    #               ,capsize=5, label='bad IM')
    # plt.errorbar(np.arange(LOOP), np.nanmean(perfect_SR,axis=0),
    #               yerr=np.nanstd(perfect_SR,axis=0)#/perfect_SR.shape[0]**0.5
    #               ,capsize=5, label='good IM')
    # plt.title('aberration ao' + '_' + name)
    # plt.ylabel('strehl ratio')
    # plt.xlabel('loop frame')
    # plt.legend()
    # plt.savefig('bad_strehl' + '_' + name  + '.png', bbox_inches='tight')
    # plt.show()
    # plt.clf()
    
    # plt.errorbar(np.arange(LOOP), np.nanmean(WFE,axis=0),
    #               yerr=np.nanstd(WFE,axis=0)#/WFE.shape[0]**0.5
    #               ,capsize=5, label='bad IM')
    # plt.errorbar(np.arange(LOOP), np.nanmean(perfect_WFE,axis=0),
    #               yerr=np.nanstd(perfect_WFE,axis=0)#/perfect_WFE.shape[0]**0.5
    #               ,capsize=5, label='good IM')
    # plt.title('aberration ao' + '_' + name)
    # plt.ylabel('WFE (nm)')
    # plt.xlabel('loop frame')
    # plt.legend()
    # plt.savefig('bad_WFE' + '_' + name  + '.png', bbox_inches='tight')
    # plt.show()
    # plt.clf()
    
    
    
    clock_out = time.time()

    # print(name, aberrationHeight, clock_out - clock_in)

    with open("time_data.txt", "a") as myfile:
        myfile.write(name + 'distance {:}, use {:} time.\n'.format(aberrationHeight,clock_out-clock_in))
        
    FIRSTTIME = False
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
# os.chdir(saveto)

with open('name_space.txt', 'w') as outfile:
  outfile.write('\n'.join(str(i) for i in name_list))

#color_list=['r','g','b','c','m','y','k']

##################################################################################
# Now i need to compile one data set into one file: for easy grab
# the reason i had them saved separately earlier was to reduce memory usage.
#
