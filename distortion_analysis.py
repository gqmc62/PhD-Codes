# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:54:01 2024

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
        
        MAX4 = np.sort(WFS_MAP[~np.isnan(WFS_MAP)].flatten())[-4:]
        
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
            
        
        # plt.quiver(WFS_GRID_X,WFS_GRID_Y,
        #            WFS_MAP_X_MAX4,WFS_MAP_Y_MAX4,scale=5,color='r')
        
        # x_standard = np.linspace(WFS_GRID.min(),WFS_GRID.max())
        # x = np.zeros((MAX4.shape[0],x_standard.shape[0]))
        # y = np.zeros((MAX4.shape[0],x_standard.shape[0]),dtype=float)
        # for i in np.arange(MAX4.shape[0]):
        #     m = v_x[i]/v_y[i]
        #     x[i] = x_standard
        #     y[i] = m*(x[i] - MAX4_POS[i][0]) + MAX4_POS[i][1]
            
        #     # y[i] = (x_standard*v_x[i] + MAX4_POS[i][1])
        #     # x[i] = (x_standard*v_y[i] + MAX4_POS[i][0])
        
        #     # plt.plot(y[i],x[i],ls=':',color='b',marker='',alpha=0.5)
        
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
                
                    # plt.plot(cross[i,j,0],cross[i,j,1],marker='o',color='g')
                    
                    # plt.plot(y[i],x[i],ls=':',color='b',marker='')
                    # plt.plot(y[j],x[j],ls=':',color='b',marker='')
                    # plt.quiver(WFS_GRID_X,WFS_GRID_Y,
                    #            WFS_MAP_X_MAX4,WFS_MAP_Y_MAX4,scale=5)
                    # # plt.show()
                    # # plt.plot(cen_x,cen_y,marker='x',color='b')
                    # plt.axis('square')
                    # plt.xlim(-0.5,7.5)
                    # plt.ylim(-0.5,7.5)
                    # plt.title(name_list[i_aber_z_list] + '\n'
                    #     'act#{}'.format(act)
                    #     +'\n{}{}'.format(i,j))
                    # plt.show()
                    size = (WFS_MAP_X[tuple(MAX4_POS[i])]**2
                            + WFS_MAP_X[tuple(MAX4_POS[j])]**2)**0.5
                    trustability[i,j] = size**2
        
        cen_x = np.nansum(cross[...,0]*trustability)/np.nansum(trustability)
        cen_y = np.nansum(cross[...,1]*trustability)/np.nansum(trustability)
        
        CEN[act] = [cen_x - DM_POS[act,0], cen_y - DM_POS[act,1]]
        
        # plt.plot(cen_x,cen_y,marker='o',color='b')
        # plt.axis('square')
        # plt.xlim(-0.5,7.5)
        # plt.ylim(-0.5,7.5)
        # plt.title(name_list[i_aber_z_list] + '\n'
        #     'act#{}'.format(act))
        # plt.show()

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

home = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\2024-11-19-10-33-59'
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
pixel_scale = dsub / config['WFS'][0]['pxlsPerSubap']

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



###############################################################################
# Sampling points
##############################################################################

plt.plot(expect_rytov,old_expect_AS,marker='o',ls='',color='r')
plt.xlabel('$\sigma_\chi^2$')
plt.ylabel('AS (subap)')
plt.title('Sampled points on previously thought spaces')
# plt.legend()
plt.xscale('log')
plt.yscale('log')
# plt.xlim(0.1,0.6)
# plt.ylim(0.1,1.0)
# plt.axis('square')
plt.show()



###############################################################################
# another method of measuring as
###############################################################################

position = np.array([[0, 3],[0, 4],
                     [1, 1],[1, 2],[1, 3],[1, 4],[1, 5],[1, 6],
                     [2, 1],[2, 2],[2, 3],[2, 4],[2, 5],[2, 6],
                     [3, 0],[3, 1],[3, 2],[3, 3],[3, 4],[3, 5],[3, 6],[3, 7],
                     [4, 0],[4, 1],[4, 2],[4, 3],[4, 4],[4, 5],[4, 6],[4, 7],
                     [5, 1],[5, 2],[5, 3],[5, 4],[5, 5],[5, 6],
                     [6, 1],[6, 2],[6, 3],[6, 4],[6, 5],[6, 6],
                     [7, 3],[7, 4]], dtype=int)

ACT_LIST = np.array([24,32,33,34,40,41,42,43,44,50,51,52,60])

DM_POS_TEMP = np.array([[1.5,3.5],
                       [2.5,2.5],[2.5,3.5],[2.5,4.5],
                       [3.5,1.5],[3.5,2.5],[3.5,3.5],[3.5,4.5],[3.5,5.5],
                       [4.5,2.5],[4.5,3.5],[4.5,4.5],
                       [5.5,3.5]])


ANOTHER_AS = np.zeros((CASES,TRIAL,83,2),dtype=float)

for cases in np.arange(CASES):
    for trial in np.arange(TRIAL):
        ANOTHER_AS[cases,trial] = (
            measure_AS_from_IM(IM[cases,trial],position,ACT_LIST,DM_POS_TEMP,as_threshold))
        
        
# ###############################################################################
# # fixing rytov measurement
# ###############################################################################   

# MASK = np.array([0,0,
#                  0,1,1,1,1,0,
#                  1,1,1,1,1,1,
#                  0,1,1,1,1,1,1,0,
#                  0,1,1,1,1,1,1,0,
#                  1,1,1,1,1,1,
#                  0,1,1,1,1,0,
#                  0,0,],dtype=bool)
# original_MEASURED_IM_RYTOV = np.copy(MEASURED_IM_RYTOV)
# MEASURED_IM_RYTOV[...,MASK==0] = np.nan

###############################################################################
# Parameter error
###############################################################################

old_MEASURED_IM_SUBAP_SHIFT = np.copy(MEASURED_IM_SUBAP_SHIFT)

temp_rytov =  np.nanmean(MEASURED_IM_RYTOV,axis=(-1,-2))
COMBINED_MEASURED_IM_RYTOV = np.nanmean(temp_rytov,axis=(-1))
AS_R = np.sqrt(np.nanmean(ANOTHER_AS**2,axis=(-1,-2)))
COMBINED_MEASURED_IM_AS = np.nanmean(AS_R,axis=(-1))

error_rytov = np.nanstd(temp_rytov,axis=(-1))#/np.sqrt(2*(np.sum(~np.isnan(temp_rytov),axis=(-1))) - 2)
error_as = np.nanstd(AS_R,axis=(-1))#/np.sqrt(2*(np.sum(~np.isnan(AS_R),axis=(-1))) - 2)

plt.plot([expect_rytov[1:],COMBINED_MEASURED_IM_RYTOV[1:]],[old_expect_AS[1:],COMBINED_MEASURED_IM_AS[1:]],ls='-',marker='',color='k')
plt.plot(expect_rytov[1:],old_expect_AS[1:],marker='o',ls='',color='r',label='Expected')
plt.errorbar(COMBINED_MEASURED_IM_RYTOV[1:],COMBINED_MEASURED_IM_AS[1:],
             xerr=error_rytov[1:],yerr=error_as[1:],capsize=5,
             ls='',marker='o',label='Measured')
# plt.arrow(expect_rytov[1:],old_expect_AS[1:],
#           (COMBINED_MEASURED_IM_RYTOV-expect_rytov)[1:],(COMBINED_MEASURED_IM_AS-old_expect_AS)[1:],
#           ls='-',marker='',color='k')
plt.xlabel('$\sigma_\chi^2$')
plt.ylabel('AS (subap)')
plt.xscale('log')
plt.yscale('log')
plt.title('before fixing error')
plt.legend()
# plt.xlim(0.1,0.6)
# plt.ylim(0.1,1.0)
plt.show()



###############################################################################
# Parameter error - Fixed
###############################################################################

NOLL = [1.0299,0.582,0.134,0.111,0.0880,0.0648,0.0587,0.0525,0.0463,0.0401,0.0377,0.0352,0.0328,0.0304,0.0279,0.0267,0.0255,0.0243,0.0232,0.0220,0.0208]

FACTOR1 = ((NOLL[0] - NOLL[2])/2.)/sum(NOLL)

tiptilt_adjust = mps.tiptilt_variance_adjust_to_vK(dsub, np.min([D,L0]))

FACTOR = (FACTOR1*tiptilt_adjust)**0.5

expect_AS = FACTOR * old_expect_AS

name_list = []
for cases in np.arange(CASES):
    name_list.append('rytov{:.2f}-AS{:.0f}%'.format(expect_rytov[cases],expect_AS[cases]*100))

# COMBINED_MEASURED_IM_RYTOV = np.nanmean(MEASURED_IM_RYTOV,axis=(-1,-2,-3))
# COMBINED_MEASURED_IM_AS = np.nanmean(np.abs(MEASURED_IM_SUBAP_SHIFT),axis=(-1,-2,-3,-4))



plt.plot([expect_rytov[1:],COMBINED_MEASURED_IM_RYTOV[1:]],[expect_AS[1:],COMBINED_MEASURED_IM_AS[1:]],ls='-',marker='',color='k')
plt.plot(expect_rytov[1:],expect_AS[1:],marker='o',ls='',color='r',label='Expected')
plt.errorbar(COMBINED_MEASURED_IM_RYTOV[1:],COMBINED_MEASURED_IM_AS[1:],
         xerr=error_rytov[1:],yerr=error_as[1:],capsize=5,
         ls='',marker='o',label='Measured')
# plt.arrow(expect_rytov[1:],expect_AS[1:],
#           (COMBINED_MEASURED_IM_RYTOV-expect_rytov)[1:],(COMBINED_MEASURED_IM_AS-expect_AS)[1:],
#           ls='-',marker='',color='k')
plt.xlabel('$\sigma_\chi^2$')
plt.ylabel('AS (subap)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title('after fixing model')
# plt.xlim(0.1,0.6)
# plt.ylim(0.01,1)
plt.show()

###############################################################################
# Fixed Sampling point actual
###############################################################################

# NOLL = [1.0299,0.582,0.134,0.111,0.0880,0.0648,0.0587,0.0525,0.0463,0.0401,0.0377,0.0352,0.0328,0.0304,0.0279,0.0267,0.0255,0.0243,0.0232,0.0220,0.0208]

# FACTOR1 = ((NOLL[0] - NOLL[2])/2.)/sum(NOLL)

# tiptilt_adjust = mps.tiptilt_variance_adjust_to_vK(dsub, np.min([D,L0]))

# FACTOR = (FACTOR1*tiptilt_adjust)**0.5

# expect_AS *= FACTOR

name_list = []
for cases in np.arange(CASES):
    name_list.append('rytov{:.2f}-AS{:.0f}%'.format(expect_rytov[cases],expect_AS[cases]*100))

# COMBINED_MEASURED_IM_RYTOV = np.nanmean(MEASURED_IM_RYTOV,axis=(-1,-2,-3))
# COMBINED_MEASURED_IM_AS = np.nanmean(np.abs(MEASURED_IM_SUBAP_SHIFT),axis=(-1,-2,-3,-4))



# plt.plot([expect_rytov[1:],COMBINED_MEASURED_IM_RYTOV[1:]],[expect_AS[1:],COMBINED_MEASURED_IM_AS[1:]],ls='-',marker='',color='k')
plt.plot(expect_rytov[1:],expect_AS[1:],marker='o',ls='',color='r',label='Expected')
# plt.plot(COMBINED_MEASURED_IM_RYTOV[1:],COMBINED_MEASURED_IM_AS[1:],ls='',marker='o',label='Measured')
# plt.arrow(expect_rytov[1:],expect_AS[1:],
#           (COMBINED_MEASURED_IM_RYTOV-expect_rytov)[1:],(COMBINED_MEASURED_IM_AS-expect_AS)[1:],
#           ls='-',marker='',color='k')
plt.xlabel('$\sigma_\chi^2$')
plt.ylabel('AS (subap)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title('after fixing model true expected sampling point')
plt.xlim(0.005,0.8)
plt.ylim(0.005,0.3)
plt.show()












# ###############################################################################
# # IM error
# ###############################################################################

# TRIAL = IM.shape[1]
# nACT = IM.shape[2]
# nSUBAP = IM.shape[3]

# MAX =  perfect_IM[:,2:].max()

# perfect_IM /= MAX
# IM /= MAX

# threshold = 1/4. # selecting only the major strokes.

# MASK = (np.abs(perfect_IM[0]) > threshold)
# MASK[:2] = 0

# # plt.imshow(perfect_IM[0]*MASK)
# # plt.show()

# nMAINACT = (MASK == True).sum()
# DIF = np.zeros((CASES,TRIAL,nMAINACT),dtype=float)

# for cases in np.arange(CASES):
#     for trial in np.arange(TRIAL):
#         DIF[cases,trial] = (perfect_IM[cases] - IM[cases,trial])[MASK]
        
# DIF = np.asarray(DIF)
# IM_RMS = np.sqrt(np.nanmean(DIF**2,axis=(1,2)))


# grid_rytov = np.unique(expect_rytov)
# grid_AS = np.unique(expect_AS)
# perfect_grid_IM_RMS = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)

# for cases in np.arange(CASES):
#     perfect_grid_IM_RMS[np.where(expect_rytov[cases]==grid_rytov),np.where(expect_AS[cases]==grid_AS)] = IM_RMS[cases]
    
# perfect_grid_IM_RMS[perfect_grid_IM_RMS==0] = np.nan

# plt.contourf(grid_rytov,grid_AS,perfect_grid_IM_RMS.T)
# # plt.colorbar()
# CS = plt.contour(grid_rytov,grid_AS,perfect_grid_IM_RMS.T,colors='k')
# # plt.grid()
# # plt.plot(expect_rytov,expect_AS,marker='o',ls='',color='r')
# # plt.xscale('log')
# # plt.yscale('log')
# plt.xlabel('$\sigma_\chi^2$')
# plt.ylabel('AS (subap)')
# plt.title('RMS error of Major Strokes of IM (normalized to max IM responses of major strokes)')
# plt.clabel(CS)
# # plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0.005,0.8)
# plt.ylim(0.005,0.3)
# # plt.axis('square')
# plt.show()

# ###############################################################################
# # CC error
# ###############################################################################


# MAX = perfect_CC.max()
# CC /= MAX
# perfect_CC /= MAX

# threshold = 1/4. # selecting only the major strokes.

# MASK = (np.abs(perfect_CC[0]) > threshold)
# nMAINACT = (MASK == True).sum()

# CC_RMS = np.zeros((CASES),dtype=float)
# DIF = np.zeros((CASES,TRIAL,nMAINACT),dtype=float)
# for cases in np.arange(CASES):
#     for trial in np.arange(TRIAL):
#         DIF[cases,trial] = (perfect_CC[cases] - CC[cases,trial])[MASK]
#     CC_RMS = (np.nanmean(DIF**2,axis=(-1,-2)))**0.5
    
# grid_rytov = np.unique(expect_rytov)
# grid_AS = np.unique(expect_AS)
# perfect_grid_CC_RMS = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)

# for cases in np.arange(CASES):
#     perfect_grid_CC_RMS[np.where(expect_rytov[cases]==grid_rytov),np.where(expect_AS[cases]==grid_AS)] = CC_RMS[cases]
    
# perfect_grid_CC_RMS[perfect_grid_CC_RMS==0] = np.nan

# plt.contourf(grid_rytov,grid_AS,perfect_grid_CC_RMS.T)
# # plt.colorbar()
# CS = plt.contour(grid_rytov,grid_AS,perfect_grid_CC_RMS.T,colors='k')
# # plt.grid()
# # plt.plot(expect_rytov,expect_AS,marker='o',ls='',color='r')
# plt.xlabel('$\sigma_\chi^2$')
# plt.ylabel('AS (subap)')
# # plt.xscale('log')
# # plt.yscale('log')
# plt.title('RMS error of Major CC Elements (normalized to max undistorted CC)')
# plt.clabel(CS)
# # plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0.005,0.8)
# plt.ylim(0.005,0.3)
# # plt.axis('square')
# plt.show()

# # for i in np.arange(CASES):
# #     for j in np.arange(TRIAL):
# #         # if RUNAWAY[i,j,0] == 1:
# #         # plt.imshow(np.unwrap(np.unwrap(
# #         #     np.angle(crop_inscribed_square(
# #         #     perfect_RES_1_open[i,0,j]+1j*perfect_RES_1_open[i,1,j]))
# #         #     , axis=0), axis=1))
# #         # plt.colorbar()
# #         # plt.title('no distortion open{},{}'.format(i,j))
# #         # plt.show()
# #         plt.imshow(np.unwrap(np.unwrap(
# #             np.angle(crop_inscribed_square(
# #             perfect_RES[i,0,j,-1]+1j*perfect_RES[i,1,j,-1]))
# #             , axis=0), axis=1))
# #         plt.colorbar()
# #         plt.title('no distortion close{},{}'.format(i,j))
# #         plt.show()
# #         # plt.imshow(np.unwrap(np.unwrap(
# #         #     np.angle(crop_inscribed_square(
# #         #     RES_1_open[i,0,j]+1j*RES_1_open[i,1,j]))
# #         #     , axis=0), axis=1))
# #         # plt.colorbar()
# #         # plt.title('distorted open{},{}'.format(i,j))
# #         # plt.show()
# #         plt.imshow(np.unwrap(np.unwrap(
# #             np.angle(crop_inscribed_square(
# #             RES[i,0,j,-1]+1j*RES[i,1,j,-1]))
# #             , axis=0), axis=1))
# #         plt.colorbar()
# #         plt.title('distorted close{},{}'.format(i,j))
# #         plt.show()

# # for i in np.arange(23):
# #     for j in np.arange(10):
# #             plt.imshow(np.unwrap(np.unwrap(np.angle(crop_inscribed_square(
# #                 RES_1_open[i,0,j]+1j*RES_1_open[i,1,j]))
# #                 ,axis=-1),axis=-2))
# #             plt.colorbar()
# #             plt.title('{},{}'.format(i,j))
# #             plt.show()
















# # ###############################################################################
# # # Control SR
# # ###############################################################################

# # perfect_avg_terminal_SR = np.nanmean(perfect_SR[:,:,-1],axis=-1)

# # grid_rytov = np.unique(expect_rytov)
# # grid_AS = np.unique(expect_AS)
# # perfect_grid_SR = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)

# # for cases in np.arange(CASES):
# #     perfect_grid_SR[np.where(expect_rytov[cases]==grid_rytov),np.where(expect_AS[cases]==grid_AS)] = perfect_avg_terminal_SR[cases]
    
# # perfect_grid_SR[perfect_grid_SR==0] = np.nan

# # plt.contourf(grid_rytov,grid_AS,perfect_grid_SR.T)
# # # plt.colorbar()
# # CS = plt.contour(grid_rytov,grid_AS,perfect_grid_SR.T,colors='k')
# # # plt.grid()
# # # plt.plot(expect_rytov,expect_AS,marker='o',ls='',color='r')
# # plt.xlabel('$\sigma_\chi^2$')
# # plt.ylabel('AS (subap)')
# # plt.title('No distortion bench mark Strehl Ratio' + '\nNo distortion has {:.2f}'.format(perfect_grid_SR[0,0]))
# # plt.clabel(CS)
# # # plt.legend()
# # # plt.xscale('log')
# # # plt.yscale('log')
# # # plt.xlim(0.1,0.6)
# # # plt.ylim(0.1,1.0)
# # # plt.axis('square')
# # plt.show()

# # perfect_avg_terminal_SR_std = np.nanstd(perfect_SR[:,:,-1],axis=-1)
# # grid_rytov = np.unique(expect_rytov)
# # grid_AS = np.unique(expect_AS)
# # perfect_grid_SR_std = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)

# # for cases in np.arange(CASES):
# #     perfect_grid_SR_std[np.where(expect_rytov[cases]==grid_rytov),np.where(expect_AS[cases]==grid_AS)] = perfect_avg_terminal_SR_std[cases]
    
# # perfect_grid_SR_std[perfect_grid_SR_std==0] = np.nan

# # plt.contourf(grid_rytov,grid_AS,perfect_grid_SR_std.T)
# # # plt.colorbar()
# # CS = plt.contour(grid_rytov,grid_AS,perfect_grid_SR_std.T,colors='k')
# # # plt.grid()
# # # plt.plot(expect_rytov,expect_AS,marker='o',ls='',color='r')
# # plt.xlabel('$\sigma_\chi^2$')
# # plt.ylabel('AS (subap)')
# # plt.title('No distortion bench mark std Strehl Ratio' + '\nNo distortion has {:.2f}'.format(perfect_grid_SR_std[0,0]))
# # plt.clabel(CS)
# # # plt.legend()
# # # plt.xscale('log')
# # # plt.yscale('log')
# # # plt.xlim(0.1,0.6)
# # # plt.ylim(0.1,1.0)
# # # plt.axis('square')
# # plt.show()



# ###############################################################################
# # Control WFE
# ###############################################################################

# perfect_avg_terminal_WFE = np.nanmean(perfect_WFE[:,:,-1],axis=-1)
# # expect_rytov
# # expect_AS

# # plt.plot(expect_rytov,avg_terminal_SR,ls='',marker='o')
# # plt.plot(expect_AS,avg_terminal_SR,ls='',marker='o')

# grid_rytov = np.unique(expect_rytov)
# grid_AS = np.unique(expect_AS)
# perfect_grid_WFE = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)

# for cases in np.arange(CASES):
#     perfect_grid_WFE[np.where(expect_rytov[cases]==grid_rytov),np.where(expect_AS[cases]==grid_AS)] = perfect_avg_terminal_WFE[cases]
    
# perfect_grid_WFE[perfect_grid_WFE==0] = np.nan

# plt.contourf(grid_rytov,grid_AS,perfect_grid_WFE.T)
# # plt.colorbar()
# CS = plt.contour(grid_rytov,grid_AS,perfect_grid_WFE.T,colors='k')
# # plt.grid()
# # plt.plot(expect_rytov,expect_AS,marker='o',ls='',color='r')
# plt.xlabel('$\sigma_\chi^2$')
# plt.ylabel('AS (subap)')
# # plt.xscale('log')
# # plt.yscale('log')
# plt.title('No distortion bench mark WFE (nm)' + '\nNo distortion has {:.2f}nm'.format(perfect_grid_WFE[0,0]))
# plt.clabel(CS)
# # plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0.005,0.8)
# plt.ylim(0.005,0.3)
# # plt.axis('square')
# plt.show()



# perfect_avg_terminal_WFE_std = np.nanstd(perfect_WFE[:,:,-1],axis=-1,ddof=1)
# # expect_rytov
# # expect_AS

# # plt.plot(expect_rytov,avg_terminal_SR,ls='',marker='o')
# # plt.plot(expect_AS,avg_terminal_SR,ls='',marker='o')

# grid_rytov = np.unique(expect_rytov)
# grid_AS = np.unique(expect_AS)
# perfect_grid_WFE_std = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)

# for cases in np.arange(CASES):
#     perfect_grid_WFE_std[np.where(expect_rytov[cases]==grid_rytov),np.where(expect_AS[cases]==grid_AS)] = perfect_avg_terminal_WFE_std[cases]
    
# perfect_grid_WFE_std[perfect_grid_WFE_std==0] = np.nan

# plt.contourf(grid_rytov,grid_AS,perfect_grid_WFE_std.T/np.nanmax(perfect_grid_WFE)*100)
# # plt.colorbar()
# CS = plt.contour(grid_rytov,grid_AS,perfect_grid_WFE_std.T/np.nanmax(perfect_grid_WFE)*100,colors='k')
# # plt.grid()
# # plt.plot(expect_rytov,expect_AS,marker='o',ls='',color='r')
# plt.xlabel('$\sigma_\chi^2$')
# plt.ylabel('AS (subap)')
# # plt.xscale('log')
# # plt.yscale('log')
# plt.title('No distortion bench mark std WFE (% of max WFE)' + '\nNo distortion has {:.2f}nm'.format(perfect_grid_WFE[0,0]))
# plt.clabel(CS)
# # plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0.005,0.8)
# plt.ylim(0.005,0.3)
# # plt.axis('square')
# plt.show()






# ###############################################################################
# # Drop in SR
# ###############################################################################



# # avg_terminal_SR = np.nanmean(SR[:,:,-1],axis=-1)

# # grid_rytov = np.unique(expect_rytov)
# # grid_AS = np.unique(expect_AS)
# # grid_SR = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)

# # for cases in np.arange(CASES):
# #     grid_SR[np.where(expect_rytov[cases]==grid_rytov),np.where(expect_AS[cases]==grid_AS)] = avg_terminal_SR[cases]
    
# # grid_SR[grid_SR==0] = np.nan

# # plt.contourf(grid_rytov,grid_AS,grid_SR.T)
# # # plt.colorbar()
# # CS = plt.contour(grid_rytov,grid_AS,grid_SR.T,colors='k')
# # # plt.grid()
# # # plt.plot(expect_rytov,expect_AS,marker='o',ls='',color='r')
# # plt.xlabel('$\sigma_\chi^2$')
# # plt.ylabel('AS (subap)')
# # # plt.xscale('log')
# # # plt.yscale('log')
# # plt.title('Strehl Ratio' + '\nNo distortion has {:.2f}'.format(grid_SR[0,0]))
# # plt.clabel(CS)
# # # plt.legend()
# # # plt.xscale('log')
# # # plt.yscale('log')
# # # plt.xlim(0.1,0.6)
# # # plt.ylim(0.1,1.0)
# # # plt.axis('square')
# # plt.show()

# # avg_terminal_SR_std = np.nanstd(SR[:,:,-1],axis=-1,ddof=1)
# # grid_rytov = np.unique(expect_rytov)
# # grid_AS = np.unique(expect_AS)
# # grid_SR_std = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)

# # for cases in np.arange(CASES):
# #     grid_SR_std[np.where(expect_rytov[cases]==grid_rytov),np.where(expect_AS[cases]==grid_AS)] = avg_terminal_SR_std[cases]
    
# # grid_SR_std[grid_SR_std==0] = np.nan

# # plt.contourf(grid_rytov,grid_AS,grid_SR_std.T)
# # # plt.colorbar()
# # CS = plt.contour(grid_rytov,grid_AS,grid_SR_std.T,colors='k')
# # # plt.grid()
# # # plt.plot(expect_rytov,expect_AS,marker='o',ls='',color='r')
# # plt.xlabel('$\sigma_\chi^2$')
# # plt.ylabel('AS (subap)')
# # # plt.xscale('log')
# # # plt.yscale('log')
# # plt.title('std Strehl Ratio' + '\nNo distortion has {:.2f}'.format(grid_SR_std[0,0]))
# # plt.clabel(CS)
# # # plt.legend()
# # # plt.xscale('log')
# # # plt.yscale('log')
# # # plt.xlim(0.1,0.6)
# # # plt.ylim(0.1,1.0)
# # # plt.axis('square')
# # plt.show()


# ###############################################################################
# # Rise in WFE
# ###############################################################################


# avg_terminal_WFE = np.nanmean(WFE[:,:,-1],axis=-1)
# # expect_rytov
# # expect_AS

# # plt.plot(expect_rytov,avg_terminal_SR,ls='',marker='o')
# # plt.plot(expect_AS,avg_terminal_SR,ls='',marker='o')

# grid_rytov = np.unique(expect_rytov)
# grid_AS = np.unique(expect_AS)
# grid_WFE = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)

# for cases in np.arange(CASES):
#     grid_WFE[np.where(expect_rytov[cases]==grid_rytov),np.where(expect_AS[cases]==grid_AS)] = avg_terminal_WFE[cases]
    
# grid_WFE[grid_WFE==0] = np.nan

# plt.contourf(grid_rytov,grid_AS,grid_WFE.T)
# # plt.colorbar()
# CS = plt.contour(grid_rytov,grid_AS,grid_WFE.T,colors='k')
# # plt.grid()
# # plt.plot(expect_rytov,expect_AS,marker='o',ls='',color='r')
# plt.xlabel('$\sigma_\chi^2$')
# plt.ylabel('AS (subap)')
# # plt.xscale('log')
# # plt.yscale('log')
# plt.title('WFE (nm)' + '\nNo distortion has {:.2f}nm'.format(grid_WFE[0,0]))
# plt.clabel(CS)
# # plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0.005,0.8)
# plt.ylim(0.005,0.3)
# # plt.axis('square')
# plt.show()


# avg_terminal_WFE_std = np.nanstd(WFE[:,:,-1],axis=-1,ddof=1)
# # expect_rytov
# # expect_AS

# # plt.plot(expect_rytov,avg_terminal_SR,ls='',marker='o')
# # plt.plot(expect_AS,avg_terminal_SR,ls='',marker='o')

# grid_rytov = np.unique(expect_rytov)
# grid_AS = np.unique(expect_AS)
# grid_WFE_std = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)

# for cases in np.arange(CASES):
#     grid_WFE_std[np.where(expect_rytov[cases]==grid_rytov),np.where(expect_AS[cases]==grid_AS)] = avg_terminal_WFE_std[cases]
    
# grid_WFE_std[grid_WFE_std==0] = np.nan

# plt.contourf(grid_rytov,grid_AS,grid_WFE_std.T/np.nanmax(grid_WFE)*100)
# # plt.colorbar()
# CS = plt.contour(grid_rytov,grid_AS,grid_WFE_std.T/np.nanmax(grid_WFE)*100,colors='k')
# # plt.grid()
# # plt.plot(expect_rytov,expect_AS,marker='o',ls='',color='r')
# plt.xlabel('$\sigma_\chi^2$')
# plt.ylabel('AS (subap)')
# # plt.xscale('log')
# # plt.yscale('log')
# plt.title('std WFE (% of max WFE)' + '\nNo distortion has {:.2f}nm'.format(grid_WFE_std[0,0]))
# plt.clabel(CS)
# # plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0.005,0.8)
# plt.ylim(0.005,0.3)
# # plt.axis('square')
# plt.show()

# ###############################################################################
# # only control error
# ###############################################################################



# avg_terminal_WFE_only_control = (np.nanmean(WFE[:,:,-1],axis=-1)**2 - np.nanmean(WFE[0,:,-1],axis=-1)**2)**0.5
# # expect_rytov
# # expect_AS



# # plt.plot(expect_rytov,avg_terminal_SR,ls='',marker='o')
# # plt.plot(expect_AS,avg_terminal_SR,ls='',marker='o')

# grid_rytov = np.unique(expect_rytov)
# grid_AS = np.unique(expect_AS)
# grid_WFE = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)

# for cases in np.arange(CASES):
#     grid_WFE[np.where(expect_rytov[cases]==grid_rytov),np.where(expect_AS[cases]==grid_AS)] = avg_terminal_WFE_only_control[cases]
    
# grid_WFE[grid_WFE==0] = np.nan

# TEMP = np.nanmax(grid_WFE)

# plt.contourf(grid_rytov,grid_AS,grid_WFE.T)
# # plt.colorbar()
# CS = plt.contour(grid_rytov,grid_AS,grid_WFE.T,colors='k')
# # plt.grid()
# # plt.plot(expect_rytov,expect_AS,marker='o',ls='',color='r')
# plt.xlabel('$\sigma_\chi^2$')
# plt.ylabel('AS (subap)')
# # plt.xscale('log')
# # plt.yscale('log')
# plt.title('WFE (nm), excluding static error')
# plt.clabel(CS)
# # plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0.005,0.8)
# plt.ylim(0.005,0.3)
# # plt.axis('square')
# plt.show()

# X = np.nanmean(WFE[:,:,-1],axis=-1)
# Y = np.nanmean(WFE[0,:,-1],axis=-1)
# sX = np.nanstd(WFE[:,:,-1],axis=-1)
# sY = np.nanstd(WFE[0,:,-1],axis=-1)
# Z = (X**2 - Y**2)**0.5
# sZ = (X**2*sX**2+Y**2+sY**2)**0.5/Z
# error_avg_terminal_WFE_only_control = sZ
# # expect_rytov
# # expect_AS



# # plt.plot(expect_rytov,avg_terminal_SR,ls='',marker='o')
# # plt.plot(expect_AS,avg_terminal_SR,ls='',marker='o')

# grid_rytov = np.unique(expect_rytov)
# grid_AS = np.unique(expect_AS)
# grid_WFE = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)

# for cases in np.arange(CASES):
#     grid_WFE[np.where(expect_rytov[cases]==grid_rytov),np.where(expect_AS[cases]==grid_AS)] = error_avg_terminal_WFE_only_control[cases]
    
# grid_WFE[grid_WFE==0] = np.nan

# plt.contourf(grid_rytov,grid_AS,grid_WFE.T/TEMP*100)
# # plt.colorbar()
# CS = plt.contour(grid_rytov,grid_AS,grid_WFE.T/TEMP*100,colors='k')
# # plt.grid()
# # plt.plot(expect_rytov,expect_AS,marker='o',ls='',color='r')
# plt.xlabel('$\sigma_\chi^2$')
# plt.ylabel('AS (subap)')
# # plt.xscale('log')
# # plt.yscale('log')
# plt.title('std of WFE (nm), excluding static error')
# plt.clabel(CS)
# # plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0.005,0.8)
# plt.ylim(0.005,0.3)
# # plt.axis('square')
# plt.show()

# ###############################################################################
# # 
# ###############################################################################

# LOOP = perfect_RES_phase_Z.shape[2]
# MODE = perfect_RES_phase_Z.shape[3]
# # radial_modes = [( 0,),
# #                 ( 1, 2),
# #                 ( 3, 4, 5,),
# #                 ( 6, 7, 8, 9,),
# #                 (10,11,12,13,14,),
# #                 (15,16,17,18,19,20,),
# #                 (21,22,23,24,25,26,27,),
# #                 (28,29,30,31,32,33,34,35,),
# #                 (36,37,38,39,40,41,42,43,44,)]

# x = []
# y = np.arange(MODE)
# number_of_radial = int(((1+8*MODE)**0.5-1)/2)
# for i in np.arange(number_of_radial):
#     x.append(tuple(y[int((i)*(i+1)//2):int((i+1)*(i+2)//2)]))

# radial_modes = x

# RADIAL_MODES = len(radial_modes) - 1



# theory_noll = mps.von_Karman_noll_variance(MODE, D/pixel_scale, pixel_scale, 0.05, 5)
# MAX = np.max(np.array([theory_noll.max(),
#                        np.nanmean(ATM_Z,axis=1).max(),
#                        np.nanmean(perfect_RES_phase_Z,axis=1).max(),
#                        np.nanmean(RES_phase_Z,axis=1).max()]))*10
# MAX = np.min(np.array([theory_noll.min(),
#                        np.nanmean(ATM_Z,axis=1).min(),
#                        np.nanmean(perfect_RES_phase_Z,axis=1).min(),
#                        np.nanmean(RES_phase_Z,axis=1).min()]))/10

# for cases in np.arange(CASES):
    
    
#     plt.plot(np.arange(MODE - 1 - 2) + 1 + 2,
#                   np.var(ATM_Z[cases,:,3:],axis=(0),ddof=1).T,
#                   color='b',label='uncorrected')
#     plt.fill_between(np.arange(MODE - 1 - 2) + 1 + 2,
#                   np.var(ATM_Z[cases,:,3:],axis=(0),ddof=1).T + 2*np.var(ATM_Z[cases,:,3:],axis=(0),ddof=1).T/np.sqrt(2*ATM_Z.shape[1] - 2),
#                   np.var(ATM_Z[cases,:,3:],axis=(0),ddof=1).T - 2*np.var(ATM_Z[cases,:,3:],axis=(0),ddof=1).T/np.sqrt(2*ATM_Z.shape[1] - 2),
#                   alpha=0.2,color='b')
    
#     plt.plot(np.arange(MODE - 1 - 2) + 1 + 2,
#                   np.var(perfect_RES_phase_Z[cases,:,-1,3:],axis=(0),ddof=1).T,
#                   color='g',label='corrected - undistorted')
#     plt.fill_between(np.arange(MODE - 1 - 2) + 1 + 2,
#                   np.var(perfect_RES_phase_Z[cases,:,-1,3:],axis=(0),ddof=1).T + 2*np.var(perfect_RES_phase_Z[cases,:,-1,3:],axis=(0),ddof=1).T/np.sqrt(2*ATM_Z.shape[1] - 2),
#                   np.var(perfect_RES_phase_Z[cases,:,-1,3:],axis=(0),ddof=1).T - 2*np.var(perfect_RES_phase_Z[cases,:,-1,3:],axis=(0),ddof=1).T/np.sqrt(2*ATM_Z.shape[1] - 2),
#                   alpha=0.2,color='g')
    
#     plt.plot(np.arange(MODE - 1 - 2) + 1 + 2,
#                   np.var(RES_phase_Z[cases,:,-1,3:],axis=(0),ddof=1).T,
#                   color='r',label='corrected - distorted')
#     plt.fill_between(np.arange(MODE - 1 - 2) + 1 + 2,
#                   np.var(RES_phase_Z[cases,:,-1,3:],axis=(0),ddof=1).T+2*np.var(RES_phase_Z[cases,:,-1,3:],axis=(0),ddof=1).T/np.sqrt(2*ATM_Z.shape[1] - 2),
#                   np.var(RES_phase_Z[cases,:,-1,3:],axis=(0),ddof=1).T-2*np.var(RES_phase_Z[cases,:,-1,3:],axis=(0),ddof=1).T/np.sqrt(2*ATM_Z.shape[1] - 2),
#                   alpha=0.2,color='r')
    
#     plt.plot(np.arange(MODE) + 1,theory_noll,ls=':',color='k',label='theory')
    
#     # plt.errorbar(np.arange(MODE - 1) + 1,
#     #              np.abs(np.var(perfect_RES_phase_Z[cases,:,-1,3:],axis=(0)).T
#     #                     - np.var(RES_phase_Z[cases,:,-1,3:],axis=(0)).T),
#     #              yerr=2*(np.var(RES_phase_Z[cases,:,-1,3:],axis=(0)).T
#     #                      + np.var(perfect_RES_phase_Z[cases,:,-1,3:],axis=(0)).T)/np.sqrt(2*ATM_Z.shape[1] - 2),
#     #              capsize=5,color='m',label='difference between distorted and undistorted')
    
#     plt.yscale('log')
#     # plt.title('terminal z-decomposition')
#     plt.legend()
#     plt.ylabel('variance of Z coefficients (rad$^2$)')
#     plt.xlabel('Z-Modes')
#     plt.title('Terminal Wavefront Residual Zernike Mode Decomposition\n' + name_list[cases])
#     plt.show()
    




# PSD_phase_new = PSD_phase
# perfect_PSD_phase_new = perfect_PSD_phase
# PSD_phase_frequency_new = PSD_phase_frequency

# theory_psd = mps.psd_function(0.05,(PSD_phase_frequency_new.shape[1] + 1)*2,pixel_scale,L0,1e-10)
# cross_section_theory_psd = theory_psd[theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ]

# # MAX = cross_section_theory_psd.max()*10
# # MIN = cross_section_theory_psd.min()/10  
# MAX = np.max(np.array([cross_section_theory_psd.max(),np.nanmean(perfect_PSD_phase_new,axis=1).max(),np.nanmean(PSD_phase_new,axis=1).max()]))*10
# MIN = np.min(np.array([cross_section_theory_psd.min(),np.nanmean(perfect_PSD_phase_new,axis=1).min(),np.nanmean(PSD_phase_new,axis=1).min()]))/10
# for cases in np.arange(CASES):
    
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
#     plt.plot(PSD_phase_frequency_new[cases],
#                  np.nanmean(PSD_phase_new[cases,:,-1],axis=0),
#                  label='distorted control',color='r')
#     plt.plot(PSD_phase_frequency_new[cases],
#                  np.nanmean(perfect_PSD_phase_new[cases,:,-1],axis=0),
#                  label='non-distorted control',color='g')
#     plt.plot(PSD_phase_frequency_new[cases],cross_section_theory_psd,label='no correction',color='k',ls=':')
#     plt.vlines(20,MIN,MAX,label='AO Spatial Frequency')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.ylim(MIN,MAX)
#     plt.legend()
#     plt.title(name_list[cases])
#     plt.xlabel('spatial frequency (1/m)')
#     plt.ylabel('PSD of phase (rad$^2$ m$^2$)')
#     plt.show()
    
    
    
    
    
    
    
# PSD_phase_new = PSD_intensity / 2.
# perfect_PSD_phase_new = perfect_PSD_intensity / 2.
# PSD_phase_frequency_new = PSD_phase_frequency *2# * np.sqrt(2)

# # theory_psd = mps.psd_function(0.05,(PSD_phase_frequency_new.shape[1] + 1)*2,pixel_scale,L0,1e-10)
# # cross_section_theory_psd = theory_psd[theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ]

# # MAX = cross_section_theory_psd.max()*10
# # MIN = cross_section_theory_psd.min()/10  
# MAX = np.max(np.array([np.nanmean(perfect_PSD_phase_new,axis=1).max(),np.nanmean(PSD_phase_new,axis=1).max()]))*10
# MIN = np.min(np.array([np.nanmean(perfect_PSD_phase_new,axis=1).min(),np.nanmean(PSD_phase_new,axis=1).min()]))/10
# for cases in np.arange(CASES):
    
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
#     plt.plot(PSD_phase_frequency_new[cases],
#                  np.nanmean(PSD_phase_new[cases,:,-1],axis=0),
#                  label='distorted control',color='r')
#     plt.plot(PSD_phase_frequency_new[cases],
#                  np.nanmean(perfect_PSD_phase_new[cases,:,-1],axis=0),
#                  label='non-distorted control',color='g')
#     # plt.plot(PSD_phase_frequency_new[cases],cross_section_theory_psd,label='no correction',color='k',ls=':')
#     plt.vlines(20,MIN,MAX,label='AO Spatial Frequency')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.ylim(MIN,MAX)
#     plt.legend()
#     plt.title(name_list[cases])
#     plt.xlabel('spatial frequency (1/m)')
#     plt.ylabel('PSD of intensity (rad$^2$ m$^2$)')
#     plt.show()
    
    
    
    
    
    
    
    
    
    
# # for cases in np.arange(CASES):
# #     for i in np.arange(3,10):#45):
# #         plt.plot(np.abs(ABERATION_Z[cases,:,i]),SR[cases,:,-1],ls='',marker='.')
# #         plt.title('intermediate dm z mode strength compare to strehl ratio' + '\n' + name_list[cases] + ', Z-{}'.format(i))
# #         plt.xlabel('z ampitude')
# #         plt.ylabel('strehl')
# #         plt.show()

# MIN = np.nanmin(WFE)
# MAX = np.nanmax(WFE)
# xMIN = np.nanmin(np.abs(ABERATION_Z))
# xMAX = np.nanmax(np.abs(ABERATION_Z))
        
# # for cases in np.arange(CASES):
# #     for i in np.arange(3,10):#45):
# #         plt.plot(np.abs(ABERATION_Z[cases,:,i]),WFE[cases,:,-1],ls='',marker='.')
# #         plt.title('intermediate dm z mode strength compare to WFE' + '\n' + name_list[cases] + ', Z-{}'.format(i))
# #         plt.xlabel('z ampitude')
# #         plt.ylabel('WFE (nm)')
# #         plt.ylim(MIN,MAX)
# #         plt.xlim(xMIN,xMAX)
# #         plt.show()        
    
    
# for i in np.arange(3,10):
#     plt.plot(np.abs(ABERATION_Z[...,i].flatten()),WFE[...,-1].flatten(),marker='.',ls='',label='{:d}'.format(i))
#     # plt.xscale('log')
#     # plt.yscale('log')
#     plt.legend()
#     plt.show()



# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax2 = ax1.twiny()

# ax1.set_xlim(1e-15,r02cn2dz(0.02))
# ax2.set_ylim(1,30)

# r0_mark = np.array([5,2,1,0.5,0.2,0.1,0.05,0.02])*100
# new_tick_locations = np.log10(r02cn2dz(r0_mark)) + 15
# ax2.set_xlim(ax1.get_xlim())
# ax2.set_xticks(new_tick_locations)
# labels = []
# for i in np.arange(r0_mark.shape[0]):
#   labels.append('{:.0f}'.format(r0_mark[i]))
# ax2.set_xticklabels(labels)

# ax2.set_xlabel('$r_0$ of turbulence layer in line of sight\nthat immediate DM is correcting (cm) at 500 nm')
# ax1.set_xlabel('$C^2_n$sec$\zeta dh$ of turbulence layer that immediate DM is correcting (m$^{1/3}$)')
# ax1.set_ylabel('Separation between immediate DM\nand its next immediate DM in a line of sight ($h$sec$\zeta$) (km)')

# x = r02cn2dz(np.logspace(np.log10(0.02),np.log10(5)))[::-1]
# y = np.logspace(np.log10(1),np.log10(30))

# yy, xx = np.meshgrid(y,x)


# ax1.set_xscale('log')
# ax1.set_yscale('log')
# #ax1.set_yscale('squareroot')

# ax1.grid()
# ax2.grid()

# # ax1.hlines(dm_separation/1000,1e-15,r02cn2dz(0.02),color='m')

# dact_immediate_dm = 0.05
# dact_next_immediate_dm = 0.05
# Dimmediate_dm = 0.4
# L0 = 5
# wvl = 500e-9

# RYTOV = rytov_number(wvl, cn2dz2r0(xx), yy*1000.)
# APSHIFT = displacement(wvl, dact_immediate_dm, dact_next_immediate_dm, cn2dz2r0(xx), yy*1000., Dimmediate_dm, L0)

# CS = ax1.contour(x,y,RYTOV.T,colors='r',levels=[0.01,0.02,0.05,0.1,0.3,0.6])
# ax1.clabel(CS)
# CS2 = ax1.contour(x,y,APSHIFT*100,colors='b',levels=[1,2,5,10,20,50])
# ax1.clabel(CS2)
# # plt.legend()

# TOPLOT = np.copy(avg_terminal_WFE_only_control)

# cmap = plt.colormaps["viridis"]
# for i in np.arange(r0HA.size):
#     ax1.plot(r02cn2dz(r0HA)[i],z[i]/1000,marker='o',markersize=20,color=cmap(TOPLOT[i]/np.nanmax(TOPLOT)))
# # ax1.tripcolor(r02cn2dz(r0HA),z/1000,TOPLOT)
# # ax1.tricontour(r02cn2dz(r0HA),z/1000,TOPLOT)


# fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,np.nanmax(TOPLOT)),cmap=cmap),ax=ax1)
# fig.suptitle('WFE from distorted control error after 5 iterations  (nm)')

# plt.gcf().set_size_inches(8,8)
# plt.show()




# # fig = plt.figure()
# # ax1 = fig.add_subplot(111)
# # ax2 = ax1.twiny()

# # ax1.set_xlim(1e-15,r02cn2dz(0.02))
# # ax2.set_ylim(1,30)

# # r0_mark = np.array([5,2,1,0.5,0.2,0.1,0.05,0.02])*100
# # new_tick_locations = np.log10(r02cn2dz(r0_mark)) + 15
# # ax2.set_xlim(ax1.get_xlim())
# # ax2.set_xticks(new_tick_locations)
# # labels = []
# # for i in np.arange(r0_mark.shape[0]):
# #   labels.append('{:.0f}'.format(r0_mark[i]))
# # ax2.set_xticklabels(labels)

# # ax2.set_xlabel('$r_0$ of turbulence layer in line of sight\nthat immediate DM is correcting (cm) at 500 nm')
# # ax1.set_xlabel('$C^2_n$sec$\zeta dh$ of turbulence layer that immediate DM is correcting (m$^{1/3}$)')
# # ax1.set_ylabel('Separation between immediate DM\nand its next immediate DM in a line of sight ($h$sec$\zeta$) (km)')

# # x = r02cn2dz(np.logspace(np.log10(0.02),np.log10(5)))[::-1]
# # y = np.logspace(np.log10(1),np.log10(30))

# # yy, xx = np.meshgrid(y,x)


# # ax1.set_xscale('log')
# # ax1.set_yscale('log')
# # #ax1.set_yscale('squareroot')

# # ax1.grid()
# # ax2.grid()

# # # ax1.hlines(dm_separation/1000,1e-15,r02cn2dz(0.02),color='m')

# # dact_immediate_dm = 0.05
# # dact_next_immediate_dm = 0.05
# # Dimmediate_dm = 0.4
# # L0 = 5
# # wvl = 500e-9

# # RYTOV = rytov_number(wvl, cn2dz2r0(xx), yy*1000.)
# # APSHIFT = displacement(wvl, dact_immediate_dm, dact_next_immediate_dm, cn2dz2r0(xx), yy*1000., Dimmediate_dm, L0)

# # CS = ax1.contour(x,y,RYTOV.T,colors='r',levels=[0.01,0.02,0.05,0.1,0.3,0.6])
# # ax1.clabel(CS)
# # CS2 = ax1.contour(x,y,APSHIFT*100,colors='b',levels=[1,2,5,10,20,50])
# # ax1.clabel(CS2)
# # # plt.legend()

# # TOPLOT = np.copy(avg_terminal_SR)#np.copy(avg_terminal_WFE_only_control)

# # cmap = plt.colormaps["viridis"]
# # for i in np.arange(r0HA.size):
# #     ax1.plot(r02cn2dz(r0HA)[i],z[i]/1000,marker='o',markersize=20,color=cmap(TOPLOT[i]/np.nanmax(TOPLOT)))


# # fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,np.nanmax(TOPLOT)),cmap=cmap),ax=ax1)
# # fig.suptitle('Terminal Strehl ratio at 500 nm')

# # plt.gcf().set_size_inches(8,8)
# # plt.show()





# # fig = plt.figure()
# # ax1 = fig.add_subplot(111)
# # ax2 = ax1.twiny()

# # ax1.set_xlim(1e-15,r02cn2dz(0.02))
# # ax2.set_ylim(1,30)

# # r0_mark = np.array([5,2,1,0.5,0.2,0.1,0.05,0.02])*100
# # new_tick_locations = np.log10(r02cn2dz(r0_mark)) + 15
# # ax2.set_xlim(ax1.get_xlim())
# # ax2.set_xticks(new_tick_locations)
# # labels = []
# # for i in np.arange(r0_mark.shape[0]):
# #   labels.append('{:.0f}'.format(r0_mark[i]))
# # ax2.set_xticklabels(labels)

# # ax2.set_xlabel('$r_0$ of turbulence layer in line of sight\nthat immediate DM is correcting (cm) at 500 nm')
# # ax1.set_xlabel('$C^2_n$sec$\zeta dh$ of turbulence layer that immediate DM is correcting (m$^{1/3}$)')
# # ax1.set_ylabel('Separation between immediate DM\nand its next immediate DM in a line of sight ($h$sec$\zeta$) (km)')

# # x = r02cn2dz(np.logspace(np.log10(0.02),np.log10(5)))[::-1]
# # y = np.logspace(np.log10(1),np.log10(30))

# # yy, xx = np.meshgrid(y,x)


# # ax1.set_xscale('log')
# # ax1.set_yscale('log')
# # #ax1.set_yscale('squareroot')

# # ax1.grid()
# # ax2.grid()

# # # ax1.hlines(dm_separation/1000,1e-15,r02cn2dz(0.02),color='m')

# # dact_immediate_dm = 0.05
# # dact_next_immediate_dm = 0.05
# # Dimmediate_dm = 0.4
# # L0 = 5
# # wvl = 500e-9

# # RYTOV = rytov_number(wvl, cn2dz2r0(xx), yy*1000.)
# # APSHIFT = displacement(wvl, dact_immediate_dm, dact_next_immediate_dm, cn2dz2r0(xx), yy*1000., Dimmediate_dm, L0)

# # CS = ax1.contour(x,y,RYTOV.T,colors='r',levels=[0.01,0.02,0.05,0.1,0.3,0.6])
# # ax1.clabel(CS)
# # CS2 = ax1.contour(x,y,APSHIFT*100,colors='b',levels=[1,2,5,10,20,50])
# # ax1.clabel(CS2)
# # # plt.legend()

# # TOPLOT = np.copy(avg_terminal_WFE)

# # cmap = plt.colormaps["viridis"]
# # for i in np.arange(r0HA.size):
# #     ax1.plot(r02cn2dz(r0HA)[i],z[i]/1000,marker='o',markersize=20,color=cmap(TOPLOT[i]/np.nanmax(TOPLOT)))

# # fig.suptitle('Terminal WFE (nm)')
# # fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,np.nanmax(TOPLOT)),cmap=cmap),ax=ax1)


# # plt.gcf().set_size_inches(8,8)
# # plt.show()



# DATA = np.copy(avg_terminal_WFE_only_control)
# # r02cn2dz(r0HA)
# # z

# unique_AS = np.sort(np.unique(expect_AS))
# unique_rytov = np.sort(np.unique(expect_rytov))


# for i in np.arange(unique_AS.shape[0]):
#     contour_val = unique_AS[i]
#     TOPLOT = DATA[expect_AS == contour_val]
#     X = expect_rytov[expect_AS == contour_val]
#     plt.plot(X,TOPLOT,label='AS = {:.1f}% of subap.'.format(contour_val*100),ls='-',marker='o')
# plt.xlabel('expect rytov')
# plt.ylabel('terminal control WFE after 5 iteration (nm)')
# plt.xscale('log')
# plt.legend()
# plt.show()


# DATA = np.copy(avg_terminal_WFE_only_control)
# # r02cn2dz(r0HA)
# # z

# unique_AS = np.sort(np.unique(expect_AS))
# unique_rytov = np.sort(np.unique(expect_rytov))


# for i in np.arange(unique_rytov.shape[0]):
#     contour_val = unique_rytov[i]
#     TOPLOT = DATA[expect_rytov == contour_val]
#     X = expect_AS[expect_rytov == contour_val]*100
#     plt.plot(X,TOPLOT,label='rytov = {:.2f}'.format(contour_val),ls='-',marker='o')
# plt.xlabel('expect aperture shift (% of subaperture)')
# plt.ylabel('terminal control WFE after 5 iteration (nm)')
# plt.xscale('log')
# plt.legend()
# plt.show()


plt.errorbar(expect_AS*100,(COMBINED_MEASURED_IM_AS)*100, error_as*100 ,ls='',marker='o',capsize=5)
plt.plot(np.linspace(1,20),np.linspace(1,20))
plt.xlabel('expected AS (% of subap)')
plt.ylabel('measured AS (% of subap)')
plt.title('comparison between measured vs expected AS : slope method')
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.5,30)
plt.ylim(0.5,30)
plt.show()

plt.errorbar(expect_AS*100,(COMBINED_MEASURED_IM_AS**2-COMBINED_MEASURED_IM_AS[0]**2)**0.5*100, error_as*100 ,ls='',marker='o',capsize=5)
plt.plot(np.linspace(1,20),np.linspace(1,20))
plt.xlabel('expected AS (% of subap)')
plt.ylabel('measured AS (% of subap)')
plt.title('comparison between measured vs expected AS : slope method - numerical error')
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.5,30)
plt.ylim(0.5,30)
plt.show()

plt.errorbar(expect_rytov,COMBINED_MEASURED_IM_RYTOV, error_rytov ,ls='',marker='o',capsize=5)
plt.plot(np.linspace(0.01,0.6),np.linspace(0.01,0.6))
plt.xlabel('expected rytov')
plt.ylabel('measured rytov')
plt.title('comparison between measured vs expected rytov')
plt.xscale('log')
plt.yscale('log')
plt.show()

plt.errorbar(expect_rytov,(COMBINED_MEASURED_IM_RYTOV-COMBINED_MEASURED_IM_RYTOV[0]), error_rytov ,ls='',marker='o',capsize=5)
plt.plot(np.linspace(0.01,0.6),np.linspace(0.01,0.6))
plt.xlabel('expected rytov')
plt.ylabel('measured rytov')
plt.title('comparison between measured vs expected rytov - numerical error')
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.005,0.8)
plt.ylim(0.005,0.8)
plt.show()


old_temp = np.sqrt(np.nanmean(MEASURED_IM_SUBAP_SHIFT**2,axis=(-1,-2,-3)))
old_COMBINED_MEASURED_IM_AS = np.nanmean(old_temp,axis=-1)
old_COMBINED_MEASURED_IM_AS_error = np.nanstd(old_temp,axis=-1)

plt.errorbar(expect_AS*100,(old_COMBINED_MEASURED_IM_AS)*100,
             yerr=old_COMBINED_MEASURED_IM_AS_error*100,
             ls='',marker='o',capsize=5)
plt.plot(np.linspace(1,20),np.linspace(1,20))
plt.xlabel('expected AS (% of subap)')
plt.ylabel('measured AS (% of subap)')
plt.title('comparison between measured vs expected AS: influence method')
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.5,30)
plt.ylim(0.5,30)
plt.show()

plt.errorbar(expect_AS*100,(old_COMBINED_MEASURED_IM_AS**2-old_COMBINED_MEASURED_IM_AS[0]**2)**0.5*100,
             yerr=old_COMBINED_MEASURED_IM_AS_error*100,
             ls='',marker='o',capsize=5)
plt.plot(np.linspace(1,20),np.linspace(1,20))
plt.xlabel('expected AS (% of subap)')
plt.ylabel('measured AS (% of subap)')
plt.title('comparison between measured vs expected AS influench method - numerical error')
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.5,30)
plt.ylim(0.5,30)
plt.show()


