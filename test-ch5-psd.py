# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:54:13 2025

@author: gqmc62
"""


from matplotlib import pyplot as plt
# from soapy import soapy
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

from matplotlib.colors import Normalize
from scipy.interpolate import griddata

import my_phasescreen as mps

from aotools.turbulence.phasescreen import ft_sh_phase_screen
from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov

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

def old_angle_of_arrival(wvl,D,r0):
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

def old_displacement(wvl,dsub,r0,z):
    displace = angle_of_arrival(wvl,dsub,r0) * z/dsub
    return displace

def angle_of_arrival(wvl,dact_immediate_dm,r0,Dimmediate_dm,L0):
    FACTOR1 = 0.17819635611424933
    tiptilt_adjust = mps.tiptilt_variance_adjust_to_vK(dact_immediate_dm, np.min([Dimmediate_dm,L0]))
    FACTOR = (FACTOR1*tiptilt_adjust)**0.5
    alpha = np.sqrt(0.182 * wvl**2 * dact_immediate_dm**(-1./3.) * r0**(-5./3.))
    alpha *= FACTOR
    return alpha

def displacement(wvl,dact_immediate_dm,dact_next_immediate_dm,r0,z,Dimmediate_dm,L0):
    displace = angle_of_arrival(wvl,dact_immediate_dm,r0,Dimmediate_dm,L0) * z/dact_next_immediate_dm
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

def rearrange1(A, position):
    """
    rearrange soapy reported wfs value from 1d with skips into 2d

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    position : TYPE
        DESCRIPTION.

    Returns
    -------
    a : TYPE
        DESCRIPTION.

    """
    N = position.shape[0]
    size = np.max(position) - np.min(position) + 1
    DIM = position.shape[1]
    a = np.zeros((size,)*DIM, dtype=float)
    for n in np.arange(N):
        a[tuple(position[n])] = A[n]
    a[a==0] = np.nan
    return a

def measure_AS_from_IM(IM,SHWFS_position,ACT_list,ACT_position,threshold):
    debug= False
    position = SHWFS_position

    ACT_LIST = ACT_list

    DM_POS_TEMP = ACT_position

    nACT = IM.shape[0]
    nSubap = IM.shape[1]

    DM_POS = np.zeros([nACT, 2],dtype=float)
    
    j = 0
    for i in np.arange(nACT):
        if (i==ACT_LIST).any():
            DM_POS[i,0] = DM_POS_TEMP[j,1]
            DM_POS[i,1] = DM_POS_TEMP[j,0]
            j += 1
        else:
            DM_POS[i,:] = np.nan
            
    # DM_GRID = np.arange(DM_POS_TEMP.min(),DM_POS_TEMP.max() + 1)
    WFS_GRID = np.arange(position.min(),position.max() + 1)
    WFS_GRID_X,WFS_GRID_Y = np.meshgrid(WFS_GRID,WFS_GRID)

    CEN = np.zeros((nACT,2),dtype=float)
    CEN[:] = np.nan

    for act in ACT_LIST:#np.arange(83):

        
        IM /= np.abs(np.nanmax(IM))
        
        #for act in ACT_LIST:
    
        # act = 42
        WFS_MAP_X = rearrange1(IM[act][:nSubap//2], position)
        WFS_MAP_Y = rearrange1(IM[act][nSubap//2:], position)
        
        # cen_x = np.nansum(WFS_GRID_X*WFS_MAP_X**2)/np.nansum(WFS_GRID_X*WFS_MAP_X**2/WFS_GRID_X)
        # cen_y = np.nansum(WFS_GRID_Y*WFS_MAP_Y**2)/np.nansum(WFS_GRID_Y*WFS_MAP_Y**2/WFS_GRID_Y)
        
        # plt.clf()
        # fig,axes = plt.subplots(1,2,sharey=True)
        # axes[0].imshow(WFS_MAP_X,vmin=-1,vmax=1,cmap='RdBu')
        # CS = axes[1].imshow(WFS_MAP_Y,vmin=-1,vmax=1,cmap='RdBu')
        # fig.tight_layout()
        # axes[0].set_title('x_slope')
        # axes[1].set_title('y_slope')
        # cbar_ax = fig.add_axes([1,0.17,0.05,0.65])
        # axes[0].plot(DM_POS[act,0],DM_POS[act,1],marker='o',color='k')
        # axes[1].plot(DM_POS[act,0],DM_POS[act,1],marker='o',color='k')
        # fig.colorbar(CS,cax=cbar_ax)
        # fig.suptitle('act#{}'.format(act))
        # plt.show()
        
        # plt.clf()
        # # plt.plot(DM_POS[act,0],DM_POS[act,1],color='r',marker='o')
        # plt.quiver(WFS_GRID_X,WFS_GRID_Y,
        #            WFS_MAP_X,WFS_MAP_Y,scale=5)
        
        WFS_MAP = (WFS_MAP_X**2 + WFS_MAP_Y**2)**0.5
        
        WFS_MAP[WFS_MAP < threshold] = np.nan
        WFS_MAP_X[WFS_MAP < threshold] = np.nan
        WFS_MAP_Y[WFS_MAP < threshold] = np.nan
        
        MAX4 = np.sort(WFS_MAP[~np.isnan(WFS_MAP)].flatten())#[-4:]
        
        WFS_MAP_X_MAX4 = np.zeros_like(WFS_MAP)
        WFS_MAP_Y_MAX4 = np.zeros_like(WFS_MAP)
        WFS_MAP_X_MAX4[:] = np.nan
        WFS_MAP_Y_MAX4[:] = np.nan
        
        
        
        MAX4_POS = np.zeros((MAX4.shape[0],2),dtype=int)
        v_x = np.zeros((MAX4.shape[0]),dtype=float)
        v_y = np.zeros((MAX4.shape[0]),dtype=float)
        
        for i in np.arange(MAX4.shape[0]):
            loc = np.asarray(np.where(WFS_MAP == MAX4[i]))[:,0]
            MAX4_POS[i] = loc
            WFS_MAP_X_MAX4[tuple(MAX4_POS[i])] = WFS_MAP_X[tuple(MAX4_POS[i])]
            WFS_MAP_Y_MAX4[tuple(MAX4_POS[i])] = WFS_MAP_Y[tuple(MAX4_POS[i])]
            v_x[i] = WFS_MAP_X_MAX4[tuple(MAX4_POS[i])]
            v_y[i] = WFS_MAP_Y_MAX4[tuple(MAX4_POS[i])]
            
        if debug:
            plt.quiver(WFS_GRID_X,WFS_GRID_Y,
                       WFS_MAP_X_MAX4,WFS_MAP_Y_MAX4,scale=5,color='r')
            
            x_standard = np.linspace(WFS_GRID.min(),WFS_GRID.max())
            x = np.zeros((MAX4.shape[0],x_standard.shape[0]))
            y = np.zeros((MAX4.shape[0],x_standard.shape[0]),dtype=float)
            for i in np.arange(MAX4.shape[0]):
                m = v_x[i]/v_y[i]
                x[i] = x_standard
                y[i] = m*(x[i] - MAX4_POS[i][0]) + MAX4_POS[i][1]
                
                # y[i] = (x_standard*v_x[i] + MAX4_POS[i][1])
                # x[i] = (x_standard*v_y[i] + MAX4_POS[i][0])
            
                # plt.plot(y[i],x[i],ls=':',color='b',marker='',alpha=0.5)
        
        cross = np.zeros((MAX4.shape[0],MAX4.shape[0],2),dtype=float)
        cross[:] = np.nan
        trustability = np.zeros((MAX4.shape[0],MAX4.shape[0]),dtype=float)
        
        for i in np.arange(MAX4.shape[0]):
            for j in np.arange(i + 1,MAX4.shape[0]):
                mi = (v_y[i]/v_x[i])**-1
                mj = (v_y[j]/v_x[j])**-1
                xi = MAX4_POS[i][0]#x[i]
                xj = MAX4_POS[j][0]#x[j]
                yi = MAX4_POS[i][1]#y[i]
                yj = MAX4_POS[j][1]#y[j]
                
                cross[i,j,1] = ((yi - yj) + (mj*xj - mi*xi))/(mj - mi)
                cross[i,j,0] = yi + mi*(cross[i,j,1] - xi)
                
                if (
                        
                        (not (((yi - cross[i,j,0])/v_x[i] >= 0)
                              and ((xi - cross[i,j,1])/v_y[i] >= 0)
                              and ((yj - cross[i,j,0])/v_x[j] >= 0)
                              and ((xj - cross[i,j,1])/v_y[j] >= 0)))
                        
                        or
                        
                        ((cross[i,j,0] < (WFS_GRID.min() - 0.5))
                         or (cross[i,j,0] > (WFS_GRID.max() + 0.5))
                         or (cross[i,j,1] < (WFS_GRID.min() - 0.5))
                         or (cross[i,j,1] > (WFS_GRID.max() + 0.5)))
                        
                        ):

                    cross[i,j,0] = np.nan
                    cross[i,j,1] = np.nan
                    
                else:
                    if debug:
                        plt.plot(cross[i,j,0],cross[i,j,1],marker='o',color='g')
                        
                        plt.plot(y[i],x[i],ls=':',color='b',marker='')
                        plt.plot(y[j],x[j],ls=':',color='b',marker='')
                        plt.quiver(WFS_GRID_X,WFS_GRID_Y,
                                   WFS_MAP_X_MAX4,WFS_MAP_Y_MAX4,scale=5)
                        plt.plot(DM_POS[act,0],DM_POS[act,1],color='r',marker='o')
                        # plt.show()
                        # plt.plot(cen_x,cen_y,marker='x',color='b')
                        # plt.axis('square')
                        # plt.xlim(-0.5,7.5)
                        # plt.ylim(-0.5,7.5)
                        # plt.title(name_list[i_aber_z_list] + '\n'
                        #     'act#{}'.format(act)
                        #     +'\n{}{}'.format(i,j))
                        # plt.show()
                    size = ((WFS_MAP_X[tuple(MAX4_POS[i])]**2
                             + WFS_MAP_X[tuple(MAX4_POS[j])]**2)**0.5
                            * (WFS_MAP_Y[tuple(MAX4_POS[i])]**2
                               + WFS_MAP_Y[tuple(MAX4_POS[j])]**2)**0.5)
                    trustability[i,j] = size#**2
        
        cen_x = np.nansum(cross[...,0]*trustability)/np.nansum(trustability)
        cen_y = np.nansum(cross[...,1]*trustability)/np.nansum(trustability)
        
        CEN[act] = [cen_x - DM_POS[act,0], cen_y - DM_POS[act,1]]
        if debug:
            plt.plot(cen_x,cen_y,marker='o',color='b')
            plt.axis('square')
            plt.xlim(-0.5,7.5)
            plt.ylim(-0.5,7.5)
            # plt.title(name_list[i_aber_z_list] + '\n'
            #     'act#{}'.format(act))
            plt.show()

    return CEN


###############################################################################
###############################################################################
###############################################################################

# setting parameter for the whole study

# old space, no psd errorbar, wrapped phase
# home = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\2024-08-27\CombinedData'

# new space, with psd errorbar, unwrapped phase, good bad? don't know...
# home = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\2024-09-11-12-24-05'

# # new space, with psd errorbar, unwrapped phase, brief phase verification
# home = r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf\2024-10-14-14-10-19'
# os.chdir(home)

# # new space, with psd errorbar, unwrapped phase, brief phase verification
# home = r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf\2024-11-14-14-21-10'
# os.chdir(home)


as_threshold = 0.5

home = r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter5Plots\2024-11-19-10-33-59'
os.chdir(home)

SIMNAME = 'bad_IM-'

f = open('name_space.txt' ,'r')
name_list = np.asarray(f.readlines())

CASES = len(name_list)

for i in np.arange(CASES):
    name_list[i] = name_list[i].strip()

expect_rytov = np.zeros((CASES),dtype=float)
expect_AS = np.copy(expect_rytov)

z = np.copy(expect_rytov)
r0HA = np.copy(expect_rytov)

for i in np.arange(expect_rytov.shape[0]):
    expect_rytov[i] = float(name_list[i][5:9])
    expect_AS[i] = float(name_list[i][12:-1])/100.

    with open(SIMNAME + name_list[i] + '.yaml','r') as file:
        config = yaml.safe_load(file)
        z[i] = config['DM'][2]['altitude']
        r0HA[i] = config['DM'][2]['r0']

wvl = config['Science'][0]['wavelength']
dsub = config['Telescope']['telDiam']/config['WFS'][0]['nxSubaps']
D = config['Telescope']['telDiam']
# pixel_scale = dsub / config['WFS'][0]['pxlsPerSubap']

L0 = config['Atmosphere']['L0'][0]

f = open('data_space.txt' ,'r')
data_type_list = np.asarray(f.readlines())
DATATYPES = len(data_type_list)
for i in np.arange(DATATYPES):
    data_type_list[i] = data_type_list[i].strip()

try:
    IM
except:
    # print('suckers')
    for datatypes in np.arange(DATATYPES):
        for cases in np.arange(CASES):
            to_read = data_type_list[datatypes] + '_'
            # print(to_read)
            hdul = fits.open(to_read + SIMNAME + name_list[cases] + '.fits')
            if cases == 0:
                SHAPE = hdul[0].data.shape
                # print(SHAPE)
                new_SHAPE = tuple(np.append([CASES],np.asarray(SHAPE)))
                # print(new_SHAPE)
                globals()[data_type_list[datatypes]] = np.zeros((new_SHAPE),dtype=float)
            globals()[data_type_list[datatypes]][cases] = hdul[0].data
                
TRIAL = IM.shape[1]
old_expect_AS = np.copy(expect_AS)
pixel_scale = config['Telescope']['telDiam']/RES.shape[-1]


#phase = RES[:,0] + 1j*RES[:,1]

r0=0.05


# PSD_phase_new = PSD_phase # because the measured psd is from a square inscribed inside the pupil ==> 2x less area ==> scaled psd
# perfect_PSD_phase_new = perfect_PSD_phase
PSD_phase_frequency_new = PSD_phase_frequency
# PSD_phase_frequency_new = np.zeros((CASES,PSD_phase.shape[-1]),dtype=float)
# PSD_phase_frequency_new[:] = (np.arange(44) + 1) / 90. / (pixel_scale)

theory_psd = mps.psd_function(r0,(PSD_phase_frequency_new.shape[1] + 1)*2,pixel_scale,L0,1e-10)
cross_section_theory_psd = theory_psd[theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ]

# # MAX = cross_section_theory_psd.max()*10
# # MIN = cross_section_theory_psd.min()/10  

# #######################################################################################

# # control = np.zeros((100,theory_psd.shape[0],theory_psd.shape[1]),dtype=float)
# # for i in np.arange(100):
# #     control[i] = mps.get_PSD(ft_sh_phase_screen(r0,90,pixel_scale,L0,1e-6),pixel_scale)

# ###################################################################################

PS = PhaseScreenKolmogorov((PSD_phase_frequency_new.shape[1] + 1)*2, pixel_scale, r0, L0)
for i in np.arange((PSD_phase_frequency_new.shape[1] + 1)*2):
    PS.add_row()

ref = np.zeros((100,theory_psd.shape[0],theory_psd.shape[1]),dtype=float)
for i in np.arange(100):
    for j in np.arange((PSD_phase_frequency_new.shape[1] + 1)*2):
        PS.add_row()
    screen = PS.scrn
    screen -= screen.mean()
    # plt.imshow(screen)
    # plt.show()
    ref[i] = mps.get_PSD(screen,pixel_scale)

# ##########################################################################

# MAX = np.max(np.array([cross_section_theory_psd.max(),np.nanmean(perfect_PSD_phase_new,axis=1).max(),np.nanmean(PSD_phase_new,axis=1).max()]))*10
# MIN = np.min(np.array([cross_section_theory_psd.min(),np.nanmean(perfect_PSD_phase_new,axis=1).min(),np.nanmean(PSD_phase_new,axis=1).min()]))/10
# for cases in np.arange(CASES):
#     if cases != 0:
#         continue
    
#     # plt.errorbar(PSD_phase_frequency[cases],
#     #              np.nanmean(PSD_phase[cases,:,-1],axis=0),
#     #              yerr=np.nanstd(PSD_phase[cases,:,-1],axis=0)/np.sqrt(2*PSD_phase.shape[1]-2),
#     #              label='distorted control',color='r',capsize=5)
#     # plt.errorbar(PSD_phase_frequency[cases],
#     #              np.nanmean(perfect_PSD_phase[cases,:,-1],axis=0),
#     #              yerr=np.nanstd(perfect_PSD_phase[cases,:,-1],axis=0)/np.sqrt(2*perfect_PSD_phase.shape[1]-2),
#     #              label='non-distorted control',color='g',capsize=5)
#     plt.fill_between(PSD_phase_frequency_new[cases],
#                  np.nanmean(PSD_phase_new[cases,:,-1],axis=0) + np.nanstd(PSD_phase_new[cases,:,-1],axis=0,ddof=1)/np.sqrt(2*PSD_phase_new.shape[1]-2),
#                  np.nanmean(PSD_phase_new[cases,:,-1],axis=0) - np.nanstd(PSD_phase_new[cases,:,-1],axis=0,ddof=1)/np.sqrt(2*PSD_phase_new.shape[1]-2),
#                  color='r',alpha=0.2)
#     plt.fill_between(PSD_phase_frequency_new[cases],
#                  np.nanmean(perfect_PSD_phase_new[cases,:,-1],axis=0) + np.nanstd(perfect_PSD_phase_new[cases,:,-1],axis=0,ddof=1)/np.sqrt(2*perfect_PSD_phase_new.shape[1]-2),
#                  np.nanmean(perfect_PSD_phase_new[cases,:,-1],axis=0) - np.nanstd(perfect_PSD_phase_new[cases,:,-1],axis=0,ddof=1)/np.sqrt(2*perfect_PSD_phase_new.shape[1]-2),
#                  color='g',alpha=0.2)
#     plt.fill_between(PSD_phase_frequency_new[cases],
#                  np.nanmean(control[:,theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ],axis=0) + np.nanstd(control[:,theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ],axis=0)/np.sqrt(2*100-2),
#                  np.nanmean(control[:,theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ],axis=0) - np.nanstd(control[:,theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ],axis=0)/np.sqrt(2*100-2),
#                  color='b',alpha=0.2)
#     plt.plot(PSD_phase_frequency_new[cases],
#                  np.nanmean(PSD_phase_new[cases,:,-1],axis=0),
#                  label='distorted control',color='r')
#     plt.plot(PSD_phase_frequency_new[cases],
#                  np.nanmean(perfect_PSD_phase_new[cases,:,-1],axis=0),
#                  label='non-distorted control',color='g')
#     plt.plot(PSD_phase_frequency_new[cases],
#                  np.nanmean(control[:,theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ],axis=0),
#                  label='simulated atm',color='b')
#     plt.plot(PSD_phase_frequency_new[cases],cross_section_theory_psd,label='atm theory',color='k',ls=':')
#     plt.vlines(20/2,MIN,MAX,label='AO Nyquist Spatial Frequency',color='c')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.ylim(MIN,MAX)
#     plt.legend()
#     plt.title(name_list[cases])
#     plt.xlabel('spatial frequency (1/m)')
#     plt.ylabel('PSD of phase (rad$^2$ m$^2$)')
#     plt.savefig('psd-phase-{:d}.png'.format(cases))
#     plt.show()


#################################################################################

PSD_phase_frequency_new = np.zeros((CASES,44),dtype=float)
PSD_phase_frequency_new[:] = (np.arange(44) + 1) / 90 / (pixel_scale)

phase = np.angle(RES[:,0] + 1j*RES[:,1])
perfect_phase = np.angle(perfect_RES[:,0] + 1j*perfect_RES[:,1])

PSD_phase_new = np.zeros_like(PSD_phase)
perfect_PSD_phase_new = np.zeros_like(perfect_PSD_phase)
before_PSD = np.zeros((CASES,TRIAL,44),dtype=float)

for cases in np.arange(CASES):
    for trial in np.arange(TRIAL):
        PSD_phase_new[cases,trial] = mps.get_PSD(
            np.unwrap(np.unwrap(
                crop_inscribed_square(
                    phase[cases,trial,-1]
                    ),axis=-1),axis=-2), pixel_scale)[45,46:]
        perfect_PSD_phase_new[cases,trial] = mps.get_PSD(
            np.unwrap(np.unwrap(
                crop_inscribed_square(
                    perfect_phase[cases,trial,-1]
                    ),axis=-1),axis=-2), pixel_scale)[45,46:]
        before_PSD[cases,trial] = mps.get_PSD(
            np.unwrap(np.unwrap(
                crop_inscribed_square(
                    perfect_phase[cases,trial,0]
                    ),axis=-1),axis=-2), pixel_scale)[45,46:]

# theory_psd = mps.psd_function(r0,(PSD_phase_frequency_new.shape[1] + 1)*2,pixel_scale,L0,1e-10)
# cross_section_theory_psd = theory_psd[theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ]


MAX = np.max(np.array([cross_section_theory_psd.max(),np.nanmean(perfect_PSD_phase_new,axis=1).max(),np.nanmean(PSD_phase_new,axis=1).max()]))*10
MIN = np.min(np.array([cross_section_theory_psd.min(),np.nanmean(perfect_PSD_phase_new,axis=1).min(),np.nanmean(PSD_phase_new,axis=1).min()]))/10
for cases in np.arange(CASES):
    
    # if cases != 0:
    #     continue
    
    # plt.errorbar(PSD_phase_frequency[cases],
    #              np.nanmean(PSD_phase[cases,:,-1],axis=0),
    #              yerr=np.nanstd(PSD_phase[cases,:,-1],axis=0)/np.sqrt(2*PSD_phase.shape[1]-2),
    #              label='distorted control',color='r',capsize=5)
    # plt.errorbar(PSD_phase_frequency[cases],
    #              np.nanmean(perfect_PSD_phase[cases,:,-1],axis=0),
    #              yerr=np.nanstd(perfect_PSD_phase[cases,:,-1],axis=0)/np.sqrt(2*perfect_PSD_phase.shape[1]-2),
    #              label='non-distorted control',color='g',capsize=5)
    plt.fill_between(PSD_phase_frequency_new[cases],
                 np.nanmean(PSD_phase_new[cases,:,-1],axis=0) + np.nanstd(PSD_phase_new[cases,:,-1],axis=0,ddof=1)/np.sqrt(2*PSD_phase_new.shape[1]-2),
                 np.nanmean(PSD_phase_new[cases,:,-1],axis=0) - np.nanstd(PSD_phase_new[cases,:,-1],axis=0,ddof=1)/np.sqrt(2*PSD_phase_new.shape[1]-2),
                 color='r',alpha=0.2)
    plt.fill_between(PSD_phase_frequency_new[cases],
                 np.nanmean(perfect_PSD_phase_new[cases,:,-1],axis=0) + np.nanstd(perfect_PSD_phase_new[cases,:,-1],axis=0,ddof=1)/np.sqrt(2*perfect_PSD_phase_new.shape[1]-2),
                 np.nanmean(perfect_PSD_phase_new[cases,:,-1],axis=0) - np.nanstd(perfect_PSD_phase_new[cases,:,-1],axis=0,ddof=1)/np.sqrt(2*perfect_PSD_phase_new.shape[1]-2),
                 color='g',alpha=0.2)
    plt.fill_between(PSD_phase_frequency_new[cases],
                 np.nanmean(before_PSD[cases],axis=0) + np.nanstd(before_PSD[cases],axis=0)/np.sqrt(2*100-2),
                 np.nanmean(before_PSD[cases],axis=0) - np.nanstd(before_PSD[cases],axis=0)/np.sqrt(2*100-2),
                 color='b',alpha=0.2)
    plt.plot(PSD_phase_frequency_new[cases],
                 np.nanmean(PSD_phase_new[cases,:,-1],axis=0),
                 label='distorted control',color='r',marker='.')
    plt.plot(PSD_phase_frequency_new[cases],
                 np.nanmean(perfect_PSD_phase_new[cases,:,-1],axis=0),
                 label='non-distorted control',color='g',marker='.')
    plt.plot(PSD_phase_frequency_new[cases],
                 np.nanmean(before_PSD[cases],axis=0),
                 label='simulated atm',color='b',marker='.')
    plt.fill_between(PSD_phase_frequency_new[cases],
                 np.nanmean(ref[:,theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ],axis=0) + np.nanstd(ref[:,theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ],axis=0)/np.sqrt(2*100-2),
                 np.nanmean(ref[:,theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ],axis=0) - np.nanstd(ref[:,theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ],axis=0)/np.sqrt(2*100-2),
                 color='m',alpha=0.2)
    plt.plot(PSD_phase_frequency_new[cases],
                 np.nanmean(ref[:,theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ],axis=0),
                 label='ref atm',color='m')
    plt.plot(PSD_phase_frequency_new[cases],cross_section_theory_psd,label='atm theory',color='k',ls=':')
    # # plt.plot(PSD_phase_frequency_new[cases],cross_section_theory_psd*2,label='2xatm theory',color='y',ls=':')
    plt.vlines(1/dsub/2,MIN,MAX,label='AO Nyquist Spatial Frequency',color='c')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(MIN,MAX)
    plt.legend()
    plt.title(name_list[cases])
    plt.xlabel('spatial frequency (1/m)')
    plt.ylabel('PSD of phase (rad$^2$ m$^2$)')
    plt.savefig('psd-phase-{:d}.png'.format(cases))
    plt.show()



