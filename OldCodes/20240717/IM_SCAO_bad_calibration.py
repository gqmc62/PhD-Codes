# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:58:34 2024

@author: gqmc62
"""


from matplotlib import pyplot as plt
from soapy import soapy
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

def clean_this_IM(A):
    N = A.shape[1]//2
    
    ABS = np.sqrt(A[:,:N]**2 + A[:,N:]**2)
    MAX = np.nanmax(ABS)

    small = (ABS < MAX/10.)
    SMALL = np.array(np.concatenate([small.T,small.T]).T,dtype=bool)
    
    B = np.copy(A)
    
    B[SMALL] = np.nan
    
    # plt.pcolor(B)
    # plt.show()
    
    return B

def make_quiver_plot(position, slopes):
    #position = wfs.detector_cent_coords
    N = position.shape[0]
    
    step = (position[N//2 + 1,1]
            - position[N//2,1])
    position = position // step
    
    max_position = position.max()
    x = np.arange(max_position + 1.)
    x -= max_position/2.
    yy, xx = np.meshgrid(x,x)
    
    # slopes[slopes==0] = np.nan
    
    slopex = slopes[:N]#wfs.slopes[:N]
    slopey = slopes[N:]#wfs.slopes[N:]
    
    new_slopex = rearrange1(slopex, position)
    new_slopey = rearrange1(-slopey, position)
    
    size = np.abs(new_slopex**2 + new_slopey**2)
    max_size = np.max(size)

    new_slopex[size < max_size/10.] = np.nan
    new_slopey[size < max_size/10.] = np.nan
    
    plt.quiver(xx,-yy,
               new_slopex.T,
               new_slopey.T,
               scale=5, scale_units='inches')
    plt.title('IM Slope')
    plt.axis('square')
    plt.show()
    return

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
    # a[a==0] = np.nan
    return a

def pointing(slope,position):
    N = position.shape[0]
    slopex = slope[:N]
    slopey = slope[N:]

    # magnitude = np.sqrt(slopex**2 + slopey**2)
    # max_magnitude = np.nanmax(magnitude)
    
    clean_slopex = np.copy(slopex) #/2.#* 2.**-0.5# / max_magnitude#
    clean_slopey = np.copy(slopey) #/2.#* 2.**-0.5# / max_magnitude#
    
    # clean_slopex[np.abs(magnitude) < max_magnitude/10.] = np.nan
    # clean_slopey[np.abs(magnitude) < max_magnitude/10.] = np.nan
    
    # plt.plot(clean_slopex,ls='',marker='x')
    # plt.plot(clean_slopey,ls='',marker='x')
    # plt.plot(position[:,0] - position.max()/2.,ls='',marker='o')
    # plt.plot(position[:,1] - position.max()/2.,ls='',marker='o')
    # plt.show()
    
    size = np.max(position) - np.min(position) + 1
    DIM = position.shape[1]
    pointingx = np.zeros((size,)*DIM, dtype=float)
    pointingy = np.zeros((size,)*DIM, dtype=float)
    for n in np.arange(N):
        pointingx[tuple(position[n])] = clean_slopex[n] + position[n,1] - position.max()/2.
        pointingy[tuple(position[n])] = clean_slopey[n] + position[n,0] - position.max()/2.
    loc = np.array([pointingx,pointingy])
    loc[loc == 0] = np.nan
    
    # plt.plot(loc[0].flatten(),loc[1].flatten(),ls='',marker='o')
    # actuator_position = np.array([np.nanmean(loc[0]),np.nanmean(loc[1])])
    # plt.plot(actuator_position[0],actuator_position[1],ls='',marker='x')
    # # GUESS = np.round(actuator_position)
    # # plt.plot(GUESS[0],GUESS[1],marker='^')
    # plt.axis('square')
    # plt.xlim(-4,4)
    # plt.ylim(-4,4)
    # # plt.xlim(-8,8)
    # # plt.ylim(-8,8)
    
    # plt.show()
    
    return loc

# def get_actuator_positions(IM,position,nx_subap_pixels,norm):
#     actuator_positions = np.zeros((IM.shape[0],2),dtype=float)
    
#     clean_IM = clean_this_IM(IM)
    
#     # plt.pcolor(clean_IM)
#     # plt.colorbar()
#     # plt.show()
    
#     N = IM.shape[1]//2
    
#     M = IM.shape[0]
#     REF = np.arange(M*2)
#     REF = REF.reshape((2,M))
#     REF[0,:] = REF[0,:]%int(np.sqrt(M))
#     REF[1,:] = REF[1,:]//int(np.sqrt(M)) - int(np.sqrt(M))
#     REF = REF - np.sqrt(M)//2
    
#     plot = False
    
#     if plot == True:
       
        
#         M2 = (np.sqrt(IM.shape[0]) - 1)**2
#         REF2 = np.arange(M2*2)
#         REF2 = REF2.reshape((2,int(M2)))
#         REF2[0,:] = REF2[0,:]%int(np.sqrt(M2))
#         REF2[1,:] = REF2[1,:]//int(np.sqrt(M2)) - int(np.sqrt(M2))
#         REF2 = REF2 - np.sqrt(M2)//2 + 0.5
        
#     VALID = np.zeros((IM.shape[0]),dtype=bool)
#     for i in [13,20,21,22,23,24,29,30,31,32,33,37,38,39,40,41,42,43,47,48,49,50,51,56,57,58,59,60,67]:
#         VALID[i] = True
    
#     for i in np.arange(IM.shape[0]):
#         # slope = clean_IM[i]/float(nx_subap_pixels)
#         slope = clean_IM[i]/norm /2.
#         ABS = np.sqrt(slope[:N]**2 + slope[N:]**2)
#         CON = (ABS**2)/np.nansum(ABS**2)
#         ABS[CON<0.01] = np.nan
#         MAX = np.nanmax(ABS)
#         ABS[ABS<MAX/10.] = np.nan
        
#         slope[:N][np.isnan(ABS)] = np.nan
#         slope[N:][np.isnan(ABS)] = np.nan
        
#         # print((ABS**2)/np.nansum(ABS**2))
        
#         # plt.plot(slope,marker='x',ls='')
#         # plt.show()
#         # print(i,np.abs(REF[0,i]),np.abs(REF[1,i]),REF.max(),(np.abs(REF[0,i]) == REF.max()) or (np.abs(REF[1,i]) == REF.max()))
        
        
        
        
#         # if (np.abs(REF[0,i]) == REF.max()) or (np.abs(REF[1,i]) == REF.max()):
#         #     SKIP = True
#         # else:
#         #     SKIP = False
#         if (VALID[i] and ((np.isnan(ABS)==False).sum() >= 4)):
#             # print(i)
#             # fourth_max_slope = np.sort(np.unique(ABS[np.isnan(ABS)==False]))[::-1][3]
#             # slope[:N][ABS < fourth_max_slope] = np.nan
#             # slope[N:][ABS < fourth_max_slope] = np.nan
            
#             ends = pointing(slope,position)
            
#             actuator_positions[i] = np.array([np.nanmean(ends[0]),np.nanmean(ends[1])])
#             # actuator_positions[i,0] = np.nansum((position[:,1] - 3.5) * CON)/np.nansum(CON)
#             # actuator_positions[i,1] = np.nansum((position[:,0] - 3.5) * CON)/np.nansum(CON)
            
#             if plot == True:
#                 for j in np.arange(REF2.shape[1]):
#                     if np.isnan(ends[0].flatten()[j]) == False:
#                         plt.plot([REF2[0].flatten()[j],ends[0].flatten()[j]],
#                                   [REF2[1].flatten()[j],ends[1].flatten()[j]],
#                                   ls='-',marker='*',color='r')
#                         plt.plot(REF2[0].flatten()[j],REF2[1].flatten()[j],ls='',marker='*',color='k')
                        
#                 # plt.plot(ends[0].flatten(),ends[1].flatten(),ls='-',marker='*',color='r')
                
#                 plt.plot([REF[0,i],actuator_positions[i,0]],
#                           [REF[1,i],actuator_positions[i,1]],marker='^',ls='-',color='b')
#                 plt.plot(REF[0,i],REF[1,i],ls='',marker='^',color='g')
#                 plt.axis('square')
#                 plt.title(i)
#                 plt.xlim(-4,4)
#                 plt.ylim(-4,4)
#                 # plt.xlim(-8,8)
#                 # plt.ylim(-8,8)
                
#                 # colors=['black','red','green','blue']
#                 lines = [Line2D([0],[0],color='k',linewidth=1,linestyle='',marker='*'),
#                           Line2D([0],[0],color='r',linewidth=1,linestyle='-',marker='*'),
#                           Line2D([0],[0],color='g',linewidth=1,linestyle='',marker='^'),
#                           Line2D([0],[0],color='b',linewidth=1,linestyle='-',marker='^')]
#                 labels = ['active subaperture centre',
#                           'scaled spot drift',
#                           'active actuator original position',
#                           'active actuator displacement']
#                 plt.legend(lines,labels)
#                 plt.show()
        
#         else:
#             actuator_positions[i] = np.array([np.nan,np.nan])
        
#     return actuator_positions

def get_actuator_positions(IM,position,nx_subap_pixels,norm,plot=False):
    actuator_positions = np.zeros((IM.shape[0],2),dtype=float)
    
    clean_IM = clean_this_IM(IM)
    
    # plt.pcolor(clean_IM)
    # plt.colorbar()
    # plt.show()
    
    N = IM.shape[1]//2
    
    M = IM.shape[0]
    REF = np.arange(M*2)
    REF = REF.reshape((2,M))
    REF[0,:] = REF[0,:]%int(np.sqrt(M))
    REF[1,:] = REF[1,:]//int(np.sqrt(M)) - int(np.sqrt(M))
    REF = REF - np.sqrt(M)//2
    
    # plot = False
    
    if plot == True:
       
        
        M2 = (np.sqrt(IM.shape[0]) - 1)**2
        REF2 = np.arange(M2*2)
        REF2 = REF2.reshape((2,int(M2)))
        REF2[0,:] = REF2[0,:]%int(np.sqrt(M2))
        REF2[1,:] = REF2[1,:]//int(np.sqrt(M2)) - int(np.sqrt(M2))
        REF2 = REF2 - np.sqrt(M2)//2 + 0.5
    
    for i in np.arange(IM.shape[0]):
        # slope = clean_IM[i]/float(nx_subap_pixels)
        slope = clean_IM[i]/norm /2.
        ABS = np.sqrt(slope[:N]**2 + slope[N:]**2)
        CON = (ABS**2)/np.nansum(ABS**2)
        ABS[CON<0.01] = np.nan
        MAX = np.nanmax(ABS)
        ABS[ABS<MAX/10.] = np.nan
        
        slope[:N][np.isnan(ABS)] = np.nan
        slope[N:][np.isnan(ABS)] = np.nan
        
        # print((ABS**2)/np.nansum(ABS**2))
        
        # plt.plot(slope,marker='x',ls='')
        # plt.show()
        # print(i,np.abs(REF[0,i]),np.abs(REF[1,i]),REF.max(),(np.abs(REF[0,i]) == REF.max()) or (np.abs(REF[1,i]) == REF.max()))
        if (np.abs(REF[0,i]) == REF.max()) or (np.abs(REF[1,i]) == REF.max()):
            SKIP = True
        else:
            SKIP = False
        if ((SKIP == False) and ((np.isnan(ABS)==False).sum() >= 4)):
            # print(i)
            # fourth_max_slope = np.sort(np.unique(ABS[np.isnan(ABS)==False]))[::-1][3]
            # slope[:N][ABS < fourth_max_slope] = np.nan
            # slope[N:][ABS < fourth_max_slope] = np.nan
            
            ends = pointing(slope,position)
            
            actuator_positions[i] = np.array([np.nanmean(ends[0]),np.nanmean(ends[1])])
            
            # # actuator_positions=np.zeros((2),dtype=float)
            # actuator_positions[i,0] = REF[0,i] - np.nanmean(slope[:N])
            # actuator_positions[i,1] = REF[1,i] - np.nanmean(slope[N:])
            
            if plot == True:
                for j in np.arange(REF2.shape[1]):
                    if np.isnan(ends[0].flatten()[j]) == False:
                        plt.plot([REF2[0].flatten()[j],ends[0].flatten()[j]],
                                  [REF2[1].flatten()[j],ends[1].flatten()[j]],
                                  ls='-',marker='*',color='r')
                        plt.plot(REF2[0].flatten()[j],REF2[1].flatten()[j],ls='',marker='*',color='k')
                        
                # plt.plot(ends[0].flatten(),ends[1].flatten(),ls='-',marker='*',color='r')
                
                plt.plot([REF[0,i],actuator_positions[i,0]],
                          [REF[1,i],actuator_positions[i,1]],marker='^',ls='-',color='b')
                plt.plot(REF[0,i],REF[1,i],ls='',marker='^',color='g')
                plt.axis('square')
                plt.title(i)
                plt.xlim(-4,4)
                plt.ylim(-4,4)
                # plt.xlim(-8,8)
                # plt.ylim(-8,8)
                
                # colors=['black','red','green','blue']
                lines = [Line2D([0],[0],color='k',linewidth=1,linestyle='',marker='*'),
                          Line2D([0],[0],color='r',linewidth=1,linestyle='-',marker='*'),
                          Line2D([0],[0],color='g',linewidth=1,linestyle='',marker='^'),
                          Line2D([0],[0],color='b',linewidth=1,linestyle='-',marker='^')]
                labels = ['active subaperture centre',
                          'scaled spot drift',
                          'active actuator original position',
                          'active actuator displacement']
                plt.legend(lines,labels)
                plt.show()
        
        else:
            actuator_positions[i] = np.array([np.nan,np.nan])
        
    return actuator_positions

def compare_actuator_positions(good_IM,bad_IMs,position,nx_subap_pixels):
    norm = np.sqrt(good_IM[:,:44]**2 + good_IM[:,44:]**2).max()
    act_pos0 = get_actuator_positions(good_IM,position,nx_subap_pixels,norm)
    # plt.plot([0],[0],marker='x')
    DRIFT = []
    for trial in np.arange(bad_IMs.shape[0]):
        bad_IM = bad_IMs[trial]
        drift_act_pos = get_actuator_positions(bad_IM,position,nx_subap_pixels,norm)
        for i in np.arange(good_IM.shape[0]):
            if np.isnan(act_pos0[i].sum() + drift_act_pos[i].sum()) == False:
                # plt.plot(act_pos0[i,0],act_pos0[i,1],ls='',marker='o',color='r')
                # plt.plot(drift_act_pos[i,0],drift_act_pos[i,1],ls='',marker=',',color='b',alpha=0.05)
                driftx = drift_act_pos[i,0] - act_pos0[i,0]
                drifty = drift_act_pos[i,1] - act_pos0[i,1]
                plt.plot(driftx,
                          drifty,
                          ls='',marker=',',color='b',alpha=0.5)
                DRIFT.append([driftx,drifty])
                # plt.plot([act_pos0[i,0],drift_act_pos[i,0]],
                #          [act_pos0[i,1],drift_act_pos[i,1]],ls='-',marker='',color='k')
    plt.axis('square')
    # plt.xlim(-8,8)
    # plt.ylim(-8,8)
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.vlines(np.arange(-4,5),-4,4)
    plt.hlines(np.arange(-4,5),-4,4)
    # plt.xlim(-1,1)
    # plt.ylim(-1,1)
    plt.show()
    return np.array(DRIFT)

def compare_actuator_positionsV2(good_IM,bad_IMs,position,nx_subap_pixels):
    norm = np.sqrt(good_IM[:,:44]**2 + good_IM[:,44:]**2).max()
    act_pos0 = get_actuator_positions(good_IM,position,nx_subap_pixels,norm)
    # plt.plot([0],[0],marker='x')
    plt.plot(position[:,0] - 3.5,position[:,1] - 3.5,ls='',marker='.',color='r')
    plt.plot(act_pos0[:,0],act_pos0[:,1],ls='',marker='.',color='g')
    DRIFT = []
    for trial in np.arange(bad_IMs.shape[0]):
        bad_IM = bad_IMs[trial]
        drift_act_pos = get_actuator_positions(bad_IM,position,nx_subap_pixels,norm)
        for i in np.arange(good_IM.shape[0]):
            if np.isnan(act_pos0[i].sum() + drift_act_pos[i].sum()) == False:
                # plt.plot(act_pos0[i,0],act_pos0[i,1],ls='',marker='o',color='r')
                plt.plot(drift_act_pos[i,0],drift_act_pos[i,1],ls='',marker=',',color='b',alpha=0.5)
                driftx = drift_act_pos[i,0] - act_pos0[i,0]
                drifty = drift_act_pos[i,1] - act_pos0[i,1]
                # plt.plot(driftx,
                #          drifty,
                #          ls='',marker='.',color='b',alpha=0.05)
                DRIFT.append([driftx,drifty])
                plt.plot([act_pos0[i,0],drift_act_pos[i,0]],
                          [act_pos0[i,1],drift_act_pos[i,1]],ls='-',marker='',color='k',alpha=0.5)
    plt.axis('square')
    # plt.xlim(-8,8)
    # plt.ylim(-8,8)
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.vlines(np.arange(-4,5),-4,4)
    plt.hlines(np.arange(-4,5),-4,4)
    # plt.xlim(-1,1)
    # plt.ylim(-1,1)
    plt.show()
    return np.array(DRIFT)
    

def get_rcond_advantage_gain(A,plot=False,return_place=False):
  _,s,_ = np.linalg.svd(A)
  s /= s.max()
  energy = s**2
  energy /= energy.sum()
  accumulated_energy = np.zeros_like(energy)
  for i in np.arange(energy.shape[0]):
    accumulated_energy[i] = energy[:i].sum()
  residual = 1 - accumulated_energy
  
  metric = (residual[1:] - energy[:-1])[2:]
  if (metric > 0).all():
    place = energy.shape[0] - 1
  else:
    place = np.where(metric <= 0)[0][0] + 3
    if place > energy.shape[0] - 1:
      place = energy.shape[0] - 1
  rcond = s[place]

  if plot==True:
    # plt.plot(residual[1:],label='residual')
    # plt.plot(energy[:-1],label='energy')
    # plt.legend()
    # plt.yscale('log')
    # plt.vlines(place,0,1)
    # plt.show()

    plt.plot(s)
    plt.hlines(rcond,0,s.shape[0])
    plt.vlines(place,0,1)
    plt.yscale('log')
    plt.show()
  if return_place == True:
    return rcond, place
  else:
    return rcond

def get_rcond_energy_threshold(A,threshold,plot=False,return_place=False):
  _,s,_ = np.linalg.svd(A)
  s /= s.max()
  energy = s**2
  energy /= energy.sum()
  accumulated_energy = np.zeros_like(energy)

  for i in np.arange(energy.shape[0]):
    accumulated_energy[i] = energy[:i].sum()
  residual = 1 - accumulated_energy
  place = np.where(residual<=threshold)[0][0]
  
  if plot==True:
    # plt.plot(residual,label='residual')
    # plt.hlines(residual[place],0,s.shape[0])
    # plt.vlines(place,0,1)
    # plt.yscale('log')
    # plt.show()

    plt.plot(s)
    plt.hlines(s[place],0,s.shape[0])
    plt.vlines(place,0,1)
    plt.yscale('log')
    plt.show()
  rcond = s[place]
  if return_place == True:
    return rcond, place
  else:
    return rcond

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

def get_cndz_z_from_rytov_displace(wvl,dsub,rytov,displacement):


    A = np.log10(rytov)
    B = np.log10(displacement)
    C1 = np.log10(0.563*(2.0*np.pi/wvl)**(7./6.))
    C2 = 0.5*np.log10(0.182*0.423*(2.*np.pi)**2) - (7./6.)*np.log10(dsub)
    M = np.array([[   1., 5./6.],
                  [1./2.,    1.]])

    solve = np.linalg.inv(M) @ np.array([A - C1, B - C2])
    P = solve[0]
    Q = solve[1]

    cn2dz = 10.0**P
    z = 10.0**Q

    return cn2dz,z

###############################################################################
###############################################################################
###############################################################################

home = r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf'
saveto = home#r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\GLDM-IM-through-turb'

os.chdir(home)

TSTAMP = time.strftime('%Y-%m-%d-%H-%M-%S')

# aber_r0_list_original = np.array([0.04873389, 0.09941729, 0.05488279, 0.30776215, 0.16989848, 0.07746367,
#  0.04276341, 0.62783578, 0.34659345, 0.15802613, 0.0872375,  1.81451652,
#  0.82731232, 0.45671343, 0.25212625]) # r0 m
# aber_z_list_original = np.array([ 11.47233606,  20.78153251,   6.33324297,  53.28913203,  16.24004489,  3.37601489,
#    1.02885206,  96.53045585,  29.41798593,   6.11547315,  1.86371132, 116.87727636,
#   24.29669548,   7.40450088,   2.25654692])*1e3 # z m

# aber_r0_list_original = np.array([0.05488279, 0.16989848, 0.04276341,0.04873389]) # r0 m
# aber_z_list_original = np.array([6.33324297, 16.24004489,  1.02885206,11.47233606])*1e3 # z m

# aber_r0_list_original = np.array([0.04873389, 0.02690329, 0.00677155, 0.09941729, 0.05488279, 0.01381399, 0.30776215, 0.16989848, 0.04276341]) # r0 m
# aber_z_list_original = np.array([11.47233606,  3.49623358, 0.22149613,  20.78153251,  6.33324297,  0.40122858, 53.28913203, 16.24004489,  1.02885206])*1e3 # z m


# aber_r0_list_original = np.array([0.04873389, 0.02690329, 0.09941729, 0.05488279, 0.01381399, 0.30776215, 0.16989848, 0.04276341]) # r0 m
# aber_z_list_original = np.array([11.47233606,  3.49623358,  20.78153251,  6.33324297,  0.40122858, 53.28913203, 16.24004489,  1.02885206])*1e3 # z m

# aber_r0_list_original = np.array([0.05488279, 0.01381399, 0.16989848, 0.04276341]) # r0 m
# aber_z_list_original = np.array([6.33324297,  0.40122858, 16.24004489,  1.02885206])*1e3 # z m

# aber_r0_list_original = np.array([np.inf,0.16989848,0.05488279,0.04873389]) # r0 m
# aber_z_list_original = np.array([0,16.24004489,6.33324297,11.47233606])*1e3 # z m

# aber_r0_list_original = np.array([np.inf,0.16989848,0.05488279,0.04873389,0.01381399,0.04276341]) # r0 m # for extremes and boundaries
# aber_z_list_original = np.array([0,16.24004489,6.33324297,11.47233606,0.40122858,1.02885206])*1e3 # z m

# aber_r0_list_original = np.array([np.inf]) # r0 m
# aber_z_list_original = np.array([0])*1e3 # z m

# aber_r0_list_original = np.array([np.inf,0.01381399,0.04276341]) # r0 m
# aber_z_list_original = np.array([0,0.40122858,1.02885206])*1e3 # z m

# aber_r0_list_original = np.array([0.05488279]) # r0 m
# aber_z_list_original = np.array([6.33324297])*1e3 # z m

# aber_r0_list_original = np.array([0.05]) # r0 m # for 0.05r0 50%AS 0.32rytov
# aber_z_list_original = np.array([5.860])*1e3 # z m 

aber_r0_list_original = np.array([0.054]) # r0 m # for 50%AS 0.3rytov
aber_z_list_original = np.array([6.333])*1e3 # z m 

X = np.sort(aber_z_list_original)

aber_z_list = X

aber_r0_list = np.zeros_like(aber_r0_list_original)
for i in np.arange(aber_z_list.shape[0]):
    for j in np.arange(aber_z_list_original.shape[0]):
        if aber_z_list[i] == aber_z_list_original[j]:
            aber_r0_list[i] = aber_r0_list_original[j]
            

# aber_r0_list = aber_r0_list_original
# aber_z_list = aber_z_list_original

wvl = 500e-9
aber_L0 = 8
TRIAL = 2
LOOP = 10
GL_r0 = 0.05

ALL_PERFECT_SR_MEAN = np.zeros((aber_z_list.shape[0],LOOP))
ALL_PERFECT_SR_STD = np.zeros((aber_z_list.shape[0],LOOP))
ALL_BAD_SR_MEAN = np.zeros((aber_z_list.shape[0],LOOP))
ALL_BAD_SR_STD = np.zeros((aber_z_list.shape[0],LOOP))

ALL_PERFECT_SR = np.zeros((aber_z_list.shape[0],TRIAL,LOOP))
ALL_BAD_SR = np.zeros((aber_z_list.shape[0],TRIAL,LOOP))

original_name = 'bad_IM'
original_sim = soapy.Sim(original_name + '.yaml')
original_config = original_sim.config.configDict

original_home = home

data = {'home' : home,
        'saveto' : saveto,
        'TSTAMP' : TSTAMP,
        'aber_r0_list_original' : aber_r0_list_original,
        'aber_L0' : aber_L0,
        'aber_z_list_original' : aber_z_list_original,
        'TRIAL' : TRIAL,
        'LOOP' : LOOP,
        'GL_r0' : GL_r0,
        'original_name' : original_name}

if not os.path.exists(original_home + '\\' + TSTAMP):
    os.makedirs(TSTAMP)
home = original_home + '\\' + TSTAMP
os.chdir(home)

name_list=[]

for i_aber_z_list in np.arange(len(aber_z_list)):
    
    aberrationHeight = aber_z_list[i_aber_z_list]
    # if aberrationHeight >= 10e3:
    #     break
    
    i_aber_r0 = i_aber_z_list
    aber_r0 = aber_r0_list[i_aber_r0]
    # print(aber_r0,aberrationHeight)
    
    temp = (np.ceil((4*aberrationHeight*wvl/aber_r0
                      + original_config['Telescope']['telDiam'])
                    / (original_config['Telescope']['telDiam']
                        /original_config['WFS'][0]['nxSubaps'])))
    # temp = (np.ceil((4*aberrationHeight*wvl/aber_r0
    #                   + original_config['Telescope']['telDiam'])
    #                 / aber_r0))
    temp2 = int(temp*(temp + 1) / 2)
    
    # temp = np.ceil((np.ceil((4*aberrationHeight*wvl/aber_r0
    #                      + original_config['Telescope']['telDiam'])
    #                    / (original_config['Telescope']['telDiam']
    #                        /original_config['WFS'][0]['nxSubaps'])) + 1)**2 * np.pi/4.*2)
       
    # temp2 = 0
    # count = 0
    # while (temp2 < temp):
    #     temp2 = np.arange(count + 1).sum()
    #     count += 1
    #     if count > temp:
    #         break
       
    # temp2 = 1000
        
    mode_list = np.arange(temp2 - 1 - 2) + 2 + 2#np.arange(91 - 1 - 2) + 2 + 2
    MODE = mode_list.shape[0]
    ATM_SEEDS = np.random.randint(0,2**31,size=TRIAL)
    
    IM_rytov = rytov_number(wvl, aber_r0, aberrationHeight)
    dsub = (original_config['Telescope']['telDiam']/original_config['WFS'][0]['nxSubaps'])
    IM_AS = displacement(wvl, dsub, aber_r0, aberrationHeight)
    
    new_config = dict(original_config)
    # new_config['simName'] = original_config['simName'] + '-r0{:.0f}cm-HA{:.0f}km'.format(aber_r0*100.,aberrationHeight/1000.)
    new_config['simName'] = original_config['simName'] + '-rytov{:.0f}-AS{:.0f}%'.format(IM_rytov*100.,IM_AS*100)
    ndm = new_config['nDM']
    new_idm = 0
    new_config['Atmosphere']['r0'] = GL_r0
    # new_config['Atmosphere']['windSpeeds'] = [50]
    #new_config['Atmosphere']['scrnHeights'] = [aberrationHeight]#[15000]
    for idm in np.arange(ndm):
        if new_config['DM'][idm]['type'] == 'Aberration':
            new_config['DM'][idm]['subtype'] = 'Atmosphere_perfect_DM'#'#'Zernike'#Zernike'#'Atmosphere'
            new_config['DM'][idm]['calibrate'] = True
            new_config['DM'][idm]['r0'] = float(aber_r0)
            new_config['DM'][idm]['L0'] = float(aber_L0)
            new_config['DM'][idm]['nollMode'] = int(mode_list[-1])
            new_config['nIters'] = LOOP
            new_config['DM'][idm]['altitude'] = float(aberrationHeight)
    
    
    
    new_name = new_config['simName']
    
    name_list.append('rytov{:.2f}-AS{:.0f}%'.format(IM_rytov,IM_AS*100))
    
    with open(new_name + '.yaml','w') as file:
        yaml.dump(new_config, file)
    name = new_name
    print(name)
    

    output = open(original_name + TSTAMP + '.pkl', 'wb')
    pickle.dump(data,output)
    output.close()
    
    # print(name)
    
    # os.chdir(saveto)
    # output = open(original_name + TSTAMP + '.pkl', 'wb')
    # pickle.dump(data,output)
    # output.close()
    
    # os.chdir(home)
    

    
    
    
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    
    SIM = soapy.Sim(new_name + '.yaml')
    SIM.aoinit()
    SIM.config.wfss[0].plot = False
    
    perfect_SR = np.zeros((TRIAL,LOOP), dtype=float)
    perfect_WFE = np.zeros((TRIAL,LOOP), dtype=float)
    
    for idm in np.arange(SIM.config.sim.nDM):
        if SIM.dms[idm].config.type == 'Aberration' :
            SIM.dms[idm].config.calibrate = False
            
    # print(SIM.sciCams[0].los.wavelength,SIM.sciCams[0].los.EField_buf.shape[0],[SIM.dms[0].diameter,SIM.dms[1].diameter,SIM.dms[2].diameter],SIM.sciCams[0].los.dm_altitudes,SIM.sciCams[0].los.out_pixel_scale)
    # maxheight = max([SIM.dms[0].diameter,SIM.dms[1].diameter,SIM.dms[2].diameter,SIM.atmos.config.scrnHeights])
    # grid_diffraction_pad_per_side = SIM.sciCams[0].los.wavelength/2./SIM.sciCams[0].los.out_pixel_scale * maxheight
    # print(SIM.sciCams[0].los.EField_buf.shape[0] * SIM.sciCams[0].los.out_pixel_scale > SIM.sciCams[0].telescope_diameter + 2*grid_diffraction_pad_per_side)
    
    SIM.makeIMat(forceNew=True)
    
    # original_DRIFT_0 = SIM.dms[1].subapShift
    # original_rytov_0 = SIM.dms[1].rytov
    
    # DRIFT_0 = np.copy(original_DRIFT_0)
    
    # stable = False
    # count = 0
    # while (stable == False) and (count < 10):
    #     count += 1
    #     new_DRIFT_0 = np.copy(DRIFT_0)
    #     bound = norm.isf(0.5/((np.isnan(DRIFT_0) == False).sum() + 1))*np.nanstd(DRIFT_0)
    #     if (new_DRIFT_0[np.abs(DRIFT_0 - np.nanmean(DRIFT_0)) > bound].size == 0):
    #         stable = True
    #     else:
    #         new_DRIFT_0[np.abs(DRIFT_0 - np.nanmean(DRIFT_0)) > bound] = np.nan
    #         DRIFT_0 = np.copy(new_DRIFT_0)
    #     if count == 9:
    #         print('limit reached')
    
    
    setattr(SIM.dms[2],'aberration',np.zeros((SIM.dms[2].nx_dm_elements,
                                              SIM.dms[2].nx_dm_elements),dtype=float))
    
    perfect_IM = np.copy(SIM.recon.interaction_matrix)
    perfect_CC = np.copy(SIM.recon.control_matrix)
    perfect_SVD = np.copy(SIM.recon.config.svdConditioning)
    
    # for trial in tqdm(np.arange(TRIAL)):
    #     print('')
    #     for j in np.arange(SIM.atmos.scrns.shape[0]):
    #         SIM.atmos.infinite_phase_screens[j].random_seed = ATM_SEEDS[trial] + j
    #         SIM.atmos.infinite_phase_screens[j].make_initial_screen()
        
    #         if np.isnan(SIM.atmos.scrns[j].sum()):
    #             print('one of inf screen fail')
    #             quit
    #         difx = np.abs(SIM.atmos.scrns[j,1:] - SIM.atmos.scrns[j,:-1])
    #         dify = np.abs(SIM.atmos.scrns[j,:,1:] - SIM.atmos.scrns[j,:,:-1])
    #         if (difx > np.pi).any() or (dify > np.pi).any():
    #             print('one of screen is not well sampled')
    #             quit
            
            
            
    #     SIM.aoloop()
        
    #     perfect_SR[trial] = np.copy(SIM.instStrehl[0])
    #     perfect_WFE[trial] = np.copy(SIM.WFE[0])
        
    #     SIM.reset_loop()
    
    #del SIM.dms[2].aberrationStrength, SIM.dms[2].aberration
    # SIM.dms[2].atm_ab.make_initial_screen()
    
    
    # os.chdir(saveto)
    # plt.imshow(perfect_IM)
    # plt.title('no aberration IM')
    # plt.colorbar()
    # # plt.title('perfect IM')
    # plt.savefig(name + 'perfect_IM' + '.png', bbox_inches='tight')
    # plt.show()
    
    # plt.imshow(perfect_CC)
    # plt.title('no aberration CC')
    # plt.colorbar()
    # # plt.title('perfect IM')
    # plt.savefig(name + 'perfect_CC' + '.png', bbox_inches='tight')
    # plt.show()
    
    # plt.errorbar(np.arange(LOOP), perfect_SR.mean(0),
    #              yerr=perfect_SR.std(0)#/perfect_SR.shape[0]**0.5
    #              ,capsize=5)
    # plt.title('no aberration ao')
    # plt.ylabel('strehl ratio')
    # plt.xlabel('loop frame')
    # plt.savefig(name + 'perfect_strehl' + '.png', bbox_inches='tight')
    # plt.show()
    
    nx_subap_pixels = np.unique(SIM.wfss[0].pupil_subap_coords[1:,0] - SIM.wfss[0].pupil_subap_coords[:-1,0]).max()
    position = np.array(SIM.wfss[0].pupil_subap_coords/nx_subap_pixels,dtype=int)
    nx_subap_pixels = SIM.wfss[0].nx_subap_pixels
    
    
    hdu = fits.PrimaryHDU(perfect_IM)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + 'perfect_IM' + '.fits', overwrite=True)
    hdul.close()
    
    hdu = fits.PrimaryHDU(perfect_CC)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + 'perfect_CC' + '.fits', overwrite=True)
    hdul.close()

    hdu = fits.PrimaryHDU([perfect_SVD])
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + 'perfect_SVD' + '.fits', overwrite=True)
    hdul.close()
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    
    
    # os.chdir(home)
    
    MASTER_all_modes_IM = np.zeros((TRIAL,
                                        perfect_IM.shape[0],
                                        perfect_IM.shape[1]),dtype=float)
    MASTER_all_modes_CC = np.zeros((TRIAL,
                                        perfect_IM.shape[1],
                                        perfect_IM.shape[0]),dtype=float)
    MASTER_aberation_seed = np.random.randint(0,2**31,size=TRIAL)
    MASTER_svds = np.zeros((TRIAL),dtype=float)
    
    ALL_ABERATION = np.zeros((TRIAL,
                              SIM.dms[2].aberration.shape[0],
                              SIM.dms[2].aberration.shape[1]),dtype=float)
    ALL_ZERO_IMAT = np.zeros((TRIAL,perfect_IM.shape[1]),dtype=float)
    
    SR = np.zeros((TRIAL,LOOP),dtype=float)
    WFE = np.zeros((TRIAL,LOOP),dtype=float)
    
    # original_DRIFT = np.zeros((TRIAL,original_DRIFT_0.shape[0],original_DRIFT_0.shape[1],original_DRIFT_0.shape[2]),dtype=float)
    # original_rytov = np.zeros((TRIAL,original_rytov_0.shape[0],original_rytov_0.shape[1]),dtype=float)
    
    
    # print('expected {:.2f} subaperture shift during calibration'.format(
    #     displacement(SIM.wfss[0].wavelength,
    #                   SIM.wfss[0].subap_diam,
    #                   SIM.dms[2].config.r0,
    #                   SIM.dms[2].altitude)))
    # print('expected {:.2f} scintillation index during calibration'.format(
    #     rytov_number(SIM.wfss[0].wavelength,
    #                   SIM.dms[2].config.r0,
    #                   SIM.dms[2].altitude)))
    
    for idm in np.arange(SIM.config.sim.nDM):
        if SIM.dms[idm].config.type == 'Aberration' :
            SIM.dms[idm].config.calibrate = True
    SIM.config.wfss[0].plot = False
    
    del SIM.dms[2].aberration
    try:
        del SIM.dms[2].noll_variances
    except:
        pass
    SIM.dms[2].makeDMFrame()
    
    A = SIM.dms[2].nx_dm_elements
    B = SIM.dms[2].pupil_size
    x = np.arange(SIM.dms[2].pupil_size) - SIM.dms[2].pupil_size//2 + 0.5
    yy, xx = np.meshgrid(x,x)
    rr = np.sqrt(xx**2 + yy**2)
    in_circle = (rr<=SIM.dms[2].pupil_size//2)
    
    for trial in tqdm(np.arange(TRIAL)):
        print('')
        if SIM.config.dms[2].subtype == 'Atmosphere' or 'Atmosphere_perfect_DM':
            try:
                SIM.dms[2].atm_ab.random_seed = MASTER_aberation_seed[trial]
                SIM.dms[2].atm_ab.make_initial_screen()
            except:
                pass
        if SIM.config.dms[2].subtype == 'Zernike':
            SIM.dms[2].random_seed = MASTER_aberation_seed[trial]
            #SIM.dms[2].makeDMFrame(0)
        
        try:
            del SIM.dms[2].aberration
        except:
            pass
        try:
            del SIM.dms[2].noll_variances
        except:
            pass
        try:
            del SIM.dms[2].aberrationStrength
        except:
            pass
        
        
        SIM.makeIMat(forceNew=True)
        try:
            difx = np.abs(SIM.dms[2].aberration[1:] - SIM.dms[2].aberration[:-1])[(A - B)//2 : (A + B)//2, (A - B)//2 : (A + B)//2][in_circle]
            dify = np.abs(SIM.dms[2].aberration[:,1:] - SIM.dms[2].aberration[:,:-1])[(A - B)//2 : (A + B)//2, (A - B)//2 : (A + B)//2][in_circle]
        except:
            pass
        try:
            difx
            dify
        except:
            difx = np.array([0,0])
            dify = np.array([0,0])
        if (difx/500*2.*3.14 > np.pi).any() or (dify/500*2.*3.14 > np.pi).any():
            print('one of screen is not well sampled')
            quit
        
        # original_DRIFT[trial] = np.copy(SIM.dms[1].subapShift)
        # original_rytov[trial] = np.copy(SIM.dms[1].rytov)
        
        # compare_actuator_positions(perfect_IM[2:],
        #                            np.array([SIM.recon.interaction_matrix[2:]]),
        #                            position,
        #                            nx_subap_pixels)
        
        ALL_ABERATION[trial] = SIM.dms[2].aberration
        ALL_ZERO_IMAT[trial] = SIM.recon.zero_iMat
        
        MASTER_all_modes_IM[trial] = np.copy(SIM.recon.interaction_matrix)
        MASTER_all_modes_CC[trial] = np.copy(SIM.recon.control_matrix)
        
        
        MASTER_svds[trial] = np.copy(SIM.recon.config.svdConditioning)
        
        for j in np.arange(SIM.atmos.scrns.shape[0]):
            SIM.atmos.infinite_phase_screens[j].random_seed = ATM_SEEDS[trial] + j
            SIM.atmos.infinite_phase_screens[j].make_initial_screen()
            
            if np.isnan(SIM.atmos.scrns[j].sum()):
                print('one of inf screen fail')
                quit
            difx = np.abs(SIM.atmos.scrns[j,1:] - SIM.atmos.scrns[j,:-1])
            dify = np.abs(SIM.atmos.scrns[j,:,1:] - SIM.atmos.scrns[j,:,:-1])
            if (difx/500*2.*3.14 > np.pi).any() or (dify/500*2.*3.14 > np.pi).any():
                print('one of screen is not well sampled')
                quit
        
        # SIM.dms[2].aberrationStrength[:] = 0
        try:
            SIM.dms[2].aberration[:] = 0
        except:
            pass
        
        SIM.aoloop()
        SR[trial] = np.copy(SIM.instStrehl[0])
        WFE[trial] = np.copy(SIM.WFE[0])
    
        
        # del SIM.dms[2].aberrationStrength, SIM.dms[2].aberration
        SIM.reset_loop()
        for j in np.arange(SIM.atmos.scrns.shape[0]):
            SIM.atmos.infinite_phase_screens[j].make_initial_screen()
        
        
        SIM.recon.interaction_matrix = np.copy(perfect_IM)
        SIM.recon.control_matrix = np.copy(perfect_CC)
        
        SIM.aoloop()
        
        perfect_SR[trial] = np.copy(SIM.instStrehl[0])
        perfect_WFE[trial] = np.copy(SIM.WFE[0])
        
        SIM.reset_loop()
        
    hdu = fits.PrimaryHDU(ALL_ABERATION)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + 'all_ABERATION_phase' + '.fits', overwrite=True)
    hdu = fits.PrimaryHDU(ALL_ZERO_IMAT)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + 'all_ABERATION_slope' + '.fits', overwrite=True)
            
    
    # for trial in tqdm(np.arange(TRIAL)):
    
    #     SIM.makeIMat(forceNew=True)
        
    #     MASTER_all_modes_IM[trial] = np.copy(SIM.recon.interaction_matrix)
    #     MASTER_all_modes_CC[trial] = np.copy(SIM.recon.control_matrix)
        
    #     MASTER_strengths[trial] = np.copy(SIM.dms[2].aberrationStrength)
    #     MASTER_svds[trial] = np.copy(SIM.recon.config.svdConditioning)
    
    # os.chdir(saveto)
    
    # plt.imshow(MASTER_all_modes_IM[0])
    # plt.title('example: bad IM')
    # plt.colorbar()
    # plt.savefig(name + 'bad_IM' + '.png', bbox_inches='tight')
    # plt.show()
    
    # plt.imshow(MASTER_all_modes_CC[0])
    # plt.title('example: bad CC')
    # plt.colorbar()
    # plt.savefig(name + 'bad_CC' + '.png', bbox_inches='tight')
    # plt.show()
      
    # plt.imshow(MASTER_all_modes_IM.mean(0))
    # plt.title('mean bad IM')
    # plt.colorbar()
    # plt.savefig(name + 'mean_bad_IM' + '.png', bbox_inches='tight')
    # plt.show()
    
    # plt.imshow(MASTER_all_modes_IM.std(0))
    # plt.title('std bad IM')
    # plt.colorbar()
    # plt.savefig(name + 'std_bad_IM' + '.png', bbox_inches='tight')
    # plt.show()
    
    # plt.imshow(MASTER_all_modes_CC.mean(0))
    # plt.title('mean bad CC')
    # plt.colorbar()
    # plt.savefig(name + 'mean_bad_CC' + '.png', bbox_inches='tight')
    # plt.show()
    
    # plt.imshow(MASTER_all_modes_CC.std(0))
    # plt.title('std bad CC')
    # plt.colorbar()
    # plt.savefig(name + 'std_bad_CC' + '.png', bbox_inches='tight')
    # plt.show()
    
    
    # plt.imshow(MASTER_all_modes_IM[0] - perfect_IM)
    # plt.title('example: bad IM dif')
    # plt.colorbar()
    # plt.savefig(name + 'bad_IM' + '.png', bbox_inches='tight')
    # plt.show()
    
    # plt.imshow(MASTER_all_modes_CC[0] - perfect_CC)
    # plt.title('example: bad CC dif')
    # plt.colorbar()
    # plt.savefig(name + 'bad_CC' + '.png', bbox_inches='tight')
    # plt.show()
      
    # plt.imshow((MASTER_all_modes_IM - perfect_IM).mean(0))
    # plt.title('mean bad IM dif')
    # plt.colorbar()
    # plt.savefig(name + 'mean_bad_IM' + '.png', bbox_inches='tight')
    # plt.show()
    
    # plt.imshow((MASTER_all_modes_IM - perfect_IM).std(0))
    # plt.title('std bad IM dif')
    # plt.colorbar()
    # plt.savefig(name + 'std_bad_IM' + '.png', bbox_inches='tight')
    # plt.show()
    
    # plt.imshow((MASTER_all_modes_CC - perfect_CC).mean(0))
    # plt.title('mean bad CC dif')
    # plt.colorbar()
    # plt.savefig(name + 'mean_bad_CC' + '.png', bbox_inches='tight')
    # plt.show()
    
    # plt.imshow((MASTER_all_modes_CC - perfect_CC).std(0))
    # plt.title('std bad CC dif')
    # plt.colorbar()
    # plt.savefig(name + 'std_bad_CC' + '.png', bbox_inches='tight')
    # plt.show()
    
    # DRIFT = compare_actuator_positions(perfect_IM[2:],
    #                             MASTER_all_modes_IM[:,2:],
    #                             position,
    #                             nx_subap_pixels)
    
    
    # original_DRIFT = SIM.dms[1].subapShift
    # DRIFT = np.copy(original_DRIFT)
    
    # stable = False
    # count = 0
    # while (stable == False) and (count < 10):
    #     count += 1
    #     new_DRIFT = np.copy(DRIFT)
    #     bound = norm.isf(0.5/((np.isnan(DRIFT) == False).sum() + 1))*np.nanstd(DRIFT)
    #     if (new_DRIFT[np.abs(DRIFT - np.nanmean(DRIFT)) > bound].size == 0):
    #         stable = True
    #     else:
    #         new_DRIFT[np.abs(DRIFT - np.nanmean(DRIFT)) > bound] = np.nan
    #         DRIFT = np.copy(new_DRIFT)
    #     if count == 9:
    #         print('limit reached')
    
    # rytov = np.copy(original_rytov)
    
    # # mean_DRIFT = np.nanmean(DRIFT,axis=(0,2,3))
    # std_DRIFT = np.nanstd(DRIFT,axis=(0,2,3))
    # # weighted_mean = np.nansum(mean_DRIFT/std_DRIFT**2) / np.nansum(1./std_DRIFT**2)
    # # weighted_error = 1. / np.nansum(1./std_DRIFT**2)
    
    # print('expected {:.2f} subaperture shift during calibration'.format(
    #     displacement(SIM.wfss[0].wavelength,
    #                   SIM.wfss[0].subap_diam,
    #                   SIM.dms[2].config.r0,
    #                   SIM.dms[2].altitude)))
    # print('get {:.2f} +- {:.2f} subaperture shift during calibration'.format(
    #     np.nanmean(std_DRIFT),np.nanmean(std_DRIFT) / np.sqrt(2.*std_DRIFT.shape[0] - 2)))
    # print('where {:.2f} is numerical noise'.format(np.nanstd(DRIFT_0)))
    # print('expected {:.2f} scintillation index during calibration'.format(
    #     rytov_number(SIM.wfss[0].wavelength,
    #                   SIM.dms[2].config.r0,
    #                   SIM.dms[2].altitude)))
    # print('get {:.2f} +- {:.2f} scintillation index during calibration'.format(
    #     np.nanmean(rytov),
    #     2.*np.nanmean(rytov) / np.sqrt(2.*SIM.wfss[0].scaledMask.sum()*rytov.size - 2)))
    
    hdu = fits.PrimaryHDU(MASTER_all_modes_IM)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + 'all_IM' + '.fits', overwrite=True)
    
    
    hdu = fits.PrimaryHDU(MASTER_all_modes_CC)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + 'all_CC' + '.fits', overwrite=True)
    hdul.close()
    
    hdu = fits.PrimaryHDU(MASTER_aberation_seed)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + 'all_aberation_seed' + '.fits', overwrite=True)
    hdul.close()
    
    hdu = fits.PrimaryHDU(MASTER_svds)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + 'all_svds' + '.fits', overwrite=True)
    hdul.close()
    
    # hdu = fits.PrimaryHDU(rytov)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto(name + 'rytov' + '.fits', overwrite=True)
    # hdul.close()
    
    # hdu = fits.PrimaryHDU(DRIFT)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto(name + 'DRIFT' + '.fits', overwrite=True)
    # hdul.close()
    
    # os.chdir(home)
    # 
    # SIM = soapy.SIM()
    # SIM.aoinit()
    
    # for trial in tqdm(np.arange(TRIAL)):
    #     SIM.recon.interaction_matrix = MASTER_all_modes_IM[trial]
    #     SIM.recon.control_matrix = MASTER_all_modes_CC[trial]
    #     SIM.aoloop()
    #     SR[trial] = np.copy(SIM.instStrehl[-1])
    #     WFE[trial] = np.copy(SIM.WFE[-1])
    
        
    #     del SIM.dms[2].aberrationStrength, SIM.dms[2].aberration
    #     SIM.reset_loop()
    #     for j in np.arange(SIM.atmos.scrns.shape[0]):
    #         SIM.atmos.infinite_phase_screens[j].make_initial_screen()
    
    # os.chdir(saveto)
    
    # plt.errorbar(np.arange(LOOP), SR.mean(0),
    #               yerr=SR.std(0)#/SR.shape[0]**0.5
    #               ,capsize=5, label='bad IM')
    # plt.errorbar(np.arange(LOOP), perfect_SR.mean(0),
    #               yerr=perfect_SR.std(0)#/perfect_SR.shape[0]**0.5
    #               ,capsize=5, label='good IM')
    # plt.title('aberration ao' + name)
    # plt.ylabel('strehl ratio')
    # plt.xlabel('loop frame')
    # plt.legend()
    # plt.savefig(name + 'bad_strehl' + '.png', bbox_inches='tight')
    # plt.show()
    
    # plt.errorbar(np.arange(LOOP), WFE.mean(0),
    #               yerr=WFE.std(0)#/WFE.shape[0]**0.5
    #               ,capsize=5, label='bad IM')
    # plt.errorbar(np.arange(LOOP), perfect_WFE.mean(0),
    #               yerr=perfect_WFE.std(0)#/perfect_WFE.shape[0]**0.5
    #               ,capsize=5, label='good IM')
    # plt.title('aberration ao' + name)
    # plt.ylabel('WFE (nm)')
    # plt.xlabel('loop frame')
    # plt.legend()
    # plt.savefig(name + 'bad_WFE' + '.png', bbox_inches='tight')
    # plt.show()
    
    # plt.errorbar(np.arange(LOOP), WFE.mean(0),
    #              yerr=WFE.std(0)/WFE.shape[0]**0.5
    #              ,capsize=5, label='bad IM')
    # plt.errorbar(np.arange(LOOP), perfect_WFE.mean(0),
    #              yerr=perfect_WFE.std(0)/perfect_WFE.shape[0]**0.5
    #              ,capsize=5, label='good IM')
    # plt.title('aberration ao' + name)
    # plt.ylabel('WFE (nm)')
    # plt.xlabel('loop frame')
    # plt.legend()
    # plt.savefig(name + 'bad_WFE' + '.png', bbox_inches='tight')
    # plt.show()
    
    # for i in np.arange(WFE.shape[0]):
    #     plt.plot(np.arange(LOOP), WFE[i],alpha=0.2,color='r')
    #     plt.plot(np.arange(LOOP), perfect_WFE[i],alpha=0.2,color='b')
    # plt.title('aberration ao' + name)
    # plt.ylabel('WFE (nm)')
    # plt.xlabel('loop frame')
    # plt.legend()
    # plt.savefig(name + 'bad_WFE' + '.png', bbox_inches='tight')
    # plt.show()
    
    hdu = fits.PrimaryHDU(SR)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + 'all_SR' + '.fits', overwrite=True)
    hdul.close()
    
    hdu = fits.PrimaryHDU(WFE)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + 'all_WFE' + '.fits', overwrite=True)
    hdul.close()
    
    hdu = fits.PrimaryHDU(perfect_SR)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + 'all_perfect_SR' + '.fits', overwrite=True)
    hdul.close()
    
    hdu = fits.PrimaryHDU(perfect_WFE)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + 'all_perfect_WFE' + '.fits', overwrite=True)
    hdul.close()
    
    
    # os.chdir(home)
    
    ALL_PERFECT_SR_MEAN[i_aber_z_list] = perfect_SR.mean(0)
    ALL_PERFECT_SR_STD[i_aber_z_list] = perfect_SR.std(0)
    ALL_BAD_SR_MEAN[i_aber_z_list] = SR.mean(0)
    ALL_BAD_SR_STD[i_aber_z_list] = SR.std(0)
    
    ALL_PERFECT_SR[i_aber_z_list] = perfect_SR
    ALL_BAD_SR[i_aber_z_list] = SR
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
# os.chdir(saveto)


hdu = fits.PrimaryHDU(ALL_PERFECT_SR_MEAN)
hdul = fits.HDUList([hdu])
hdul.writeto(original_name + 'ALL_PERFECT_SR_MEAN' + '.fits', overwrite=True)
hdul.close()

hdu = fits.PrimaryHDU(ALL_PERFECT_SR_STD)
hdul = fits.HDUList([hdu])
hdul.writeto(original_name + 'ALL_PERFECT_SR_STD' + '.fits', overwrite=True)
hdul.close()

hdu = fits.PrimaryHDU(ALL_BAD_SR_MEAN)
hdul = fits.HDUList([hdu])
hdul.writeto(original_name + 'ALL_BAD_SR_MEAN' + '.fits', overwrite=True)
hdul.close()

hdu = fits.PrimaryHDU(ALL_BAD_SR_STD)
hdul = fits.HDUList([hdu])
hdul.writeto(original_name + 'ALL_BAD_SR_STD' + '.fits', overwrite=True)
hdul.close()

hdu = fits.PrimaryHDU(ALL_PERFECT_SR)
hdul = fits.HDUList([hdu])
hdul.writeto(original_name + 'ALL_PERFECT_SR' + '.fits', overwrite=True)
hdul.close()

hdu = fits.PrimaryHDU(ALL_BAD_SR)
hdul = fits.HDUList([hdu])
hdul.writeto(original_name + 'ALL_BAD_SR' + '.fits', overwrite=True)
hdul.close()

# os.chdir(home)

color_list=['r','g','b','c','m','y','k']

for i in np.arange(ALL_BAD_SR_MEAN.shape[0]):
    if True:#(i == np.array([0,3,4,5])).any():
        # plt.errorbar(np.arange(LOOP),
        #              ALL_BAD_SR_MEAN[i],
        #              yerr=ALL_BAD_SR_STD[i]/np.sqrt(TRIAL),
        #              label=name_list[i],ls='-',marker='o',capsize=5,color=color_list[i])
        # plt.errorbar(np.arange(LOOP),
        #               ALL_PERFECT_SR_MEAN[i],
        #               yerr=ALL_PERFECT_SR_STD[i]/np.sqrt(TRIAL),
        #               label=name_list[i],
        #               ls=':',marker='o',capsize=5,color=color_list[i])
        for trial in np.arange(TRIAL):
            plt.plot(np.arange(LOOP),ALL_PERFECT_SR[i,trial],color=color_list[i],alpha=0.1)
            
    plt.ylim(0,1)
    plt.title('strehl ratio of non-distorted version'+'\ndm2act=dsub,:=non-distorted,-=distorted'
              + '\n' + name_list[i])
    # plt.savefig('all_SR.png')
    plt.show()
    # plt.clf()

for i in np.arange(ALL_BAD_SR_MEAN.shape[0]):
    if True:#(i == np.array([0,3,4,5])).any():
        # plt.errorbar(np.arange(LOOP),
        #               ALL_BAD_SR_MEAN[i],
        #               yerr=ALL_BAD_SR_STD[i]/np.sqrt(TRIAL),
        #               label=name_list[i],ls='-',marker='o',capsize=5,color=color_list[i])
        for trial in np.arange(TRIAL):
            plt.plot(np.arange(LOOP),ALL_BAD_SR[i,trial],color=color_list[i],alpha=0.1)
        # plt.errorbar(np.arange(LOOP),
        #              ALL_PERFECT_SR_MEAN[i],
        #              yerr=ALL_PERFECT_SR_STD[i]/np.sqrt(TRIAL),
        #              ls=':',marker='o',capsize=5,color=color_list[i])
    plt.ylim(0,1)
    plt.title('strehl ratio of distorted version'+'\ndm2act=dsub,:=non-distorted,-=distorted'
              + '\n' + name_list[i])
    # plt.savefig('all_SR.png')
    plt.show()
    # plt.clf()

# for i in np.arange(ALL_BAD_SR_MEAN.shape[0]):
#     if True:#(i == np.array([0,3,4,5])).any():
#         plt.errorbar(np.arange(LOOP),
#                       ALL_BAD_SR_MEAN[i],
#                       yerr=ALL_BAD_SR_STD[i]/np.sqrt(TRIAL),
#                       label=name_list[i],ls='-',marker='o',capsize=5,color=color_list[i])
#         plt.errorbar(np.arange(LOOP),
#                       ALL_PERFECT_SR_MEAN[i],
#                       yerr=ALL_PERFECT_SR_STD[i]/np.sqrt(TRIAL),
#                       ls=':',marker='o',capsize=5,color=color_list[i])
#     plt.ylim(0,1)
#     plt.legend()
#     plt.title('strehl ratio comparison'+'\ndm2act=dsub,:=non-distorted,-=distorted')
#     # plt.savefig('all_SR.png')
#     plt.show()
#     # plt.clf()

# loss = np.zeros_like(ALL_BAD_SR_MEAN)
# loss_error = np.zeros_like(ALL_BAD_SR_MEAN)
# for i in np.arange(ALL_BAD_SR_MEAN.shape[0]):
#     loss[i] = ALL_BAD_SR_MEAN[i] / ALL_PERFECT_SR_MEAN[i,-1]
#     loss_error[i] = np.sqrt(loss[i]**2 * ((ALL_BAD_SR_STD[i]/np.sqrt(TRIAL)/ALL_BAD_SR_MEAN[i])**2 + (ALL_PERFECT_SR_STD[i,-1]/np.sqrt(TRIAL)/ALL_PERFECT_SR_MEAN[i,-1])**2))

# for i in np.arange(ALL_BAD_SR_MEAN.shape[0]):
#     plt.errorbar(np.arange(LOOP),
#                   loss[i],
#                   yerr=loss_error[i],
#                   label=name_list[i],ls='-',marker='o',capsize=5,color=color_list[i])
# plt.hlines(1,0,LOOP,ls=':')
# plt.ylim(0,1.1)
# plt.legend()
# plt.title('performance compare to its non-distorted version')
# plt.show()
