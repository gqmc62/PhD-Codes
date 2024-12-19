# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:12:38 2024

@author: gqmc62
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
import yaml

def cn2dz2r0(cn2dz,wvl=500e-9):
    r0 = (0.423*(2.*np.pi/wvl)**2*cn2dz)**(-3./5.)
    return r0

def r02cn2dz(r0,wvl=500e-9):
    cn2dz = r0**(-5./3.)/(0.423*(2.*np.pi/wvl)**2)
    return cn2dz


def rytov_number(wvl, r0, z):
    # calculate rytov number from rytov approximation
    k = 2.*np.pi/wvl                    # float; rad/m; Wave number.
    cn2dz = r02cn2dz(r0, wvl)        # float; Turbulence strength.
    variance_theory_approx = 0.563 * k**(7./6.) * cn2dz * z**(5./6.)
    return variance_theory_approx

def angle_of_arrival(wvl,D,r0):
    alpha = np.sqrt(0.182 * wvl**2 * D**(-1./3.) * r0**(-5./3.))
    return alpha

def displacement(wvl,dsub,r0,z):
    displace = angle_of_arrival(wvl,dsub,r0) * z/dsub
    return displace

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

    return np.nanstd(CEN)



# position = np.array([[0, 3],[0, 4],
#                      [1, 1],[1, 2],[1, 3],[1, 4],[1, 5],[1, 6],
#                      [2, 1],[2, 2],[2, 3],[2, 4],[2, 5],[2, 6],
#                      [3, 0],[3, 1],[3, 2],[3, 3],[3, 4],[3, 5],[3, 6],[3, 7],
#                      [4, 0],[4, 1],[4, 2],[4, 3],[4, 4],[4, 5],[4, 6],[4, 7],
#                      [5, 1],[5, 2],[5, 3],[5, 4],[5, 5],[5, 6],
#                      [6, 1],[6, 2],[6, 3],[6, 4],[6, 5],[6, 6],
#                      [7, 3],[7, 4]], dtype=int)

# # ACT_LIST = np.array([13,
# #                      20,21,22,23,24,
# #                      29,30,31,32,33,
# #                      37,38,39,40,41,42,43,
# #                      47,48,49,50,51,
# #                      56,57,58,59,60,
# #                      67]) + 2

# # DM_POS_TEMP = np.array([[0.5,3.5],
# #                         [1.5,1.5],[1.5,2.5],[1.5,3.5],[1.5,4.5],[1.5,5.5],
# #                         [2.5,1.5],[2.5,2.5],[2.5,3.5],[2.5,4.5],[2.5,5.5],
# #                         [3.5,0.5],[3.5,1.5],[3.5,2.5],[3.5,3.5],[3.5,4.5],[3.5,5.5],[3.5,6.5],
# #                         [4.5,1.5],[4.5,2.5],[4.5,3.5],[4.5,4.5],[4.5,5.5],
# #                         [5.5,1.5],[5.5,2.5],[5.5,3.5],[5.5,4.5],[5.5,5.5],
# #                         [6.5,3.5]])

# # ACT_LIST = np.array([21,22,23,
# #                      29,30,31,32,33,
# #                      38,39,40,41,42,
# #                      47,48,49,50,51,
# #                      57,58,59]) + 2

# # DM_POS_TEMP = np.array([[1.5,2.5],[1.5,3.5],[1.5,4.5],
# #                    [2.5,1.5],[2.5,2.5],[2.5,3.5],[2.5,4.5],[2.5,5.5],
# #                    [3.5,1.5],[3.5,2.5],[3.5,3.5],[3.5,4.5],[3.5,5.5],
# #                    [4.5,1.5],[4.5,2.5],[4.5,3.5],[4.5,4.5],[4.5,5.5],
# #                    [5.5,2.5],[5.5,3.5],[5.5,4.5]])


# ACT_LIST = np.array([24,32,33,34,40,41,42,43,44,50,51,52,60])

# DM_POS_TEMP = np.array([[1.5,3.5],
#                        [2.5,2.5],[2.5,3.5],[2.5,4.5],
#                        [3.5,1.5],[3.5,2.5],[3.5,3.5],[3.5,4.5],[3.5,5.5],
#                        [4.5,2.5],[4.5,3.5],[4.5,4.5],
#                        [5.5,3.5]])


# DM_POS = np.zeros([83,2],dtype=float)
# j = 0
# for i in np.arange(83):
#     if (i==ACT_LIST).any():
#         DM_POS[i,0] = DM_POS_TEMP[j,1]
#         DM_POS[i,1] = DM_POS_TEMP[j,0]
#         j += 1
#     else:
#         DM_POS[i,:] = np.nan
        
# DM_GRID = np.arange(DM_POS_TEMP.min(),DM_POS_TEMP.max() + 1)
# WFS_GRID = np.arange(position.min(),position.max() + 1)
# WFS_GRID_X,WFS_GRID_Y = np.meshgrid(WFS_GRID,WFS_GRID)

# TRIAL = 10 #50
# LOOP = 5 #100
# z_limit = 20e3#5e3 #25e3
# CASE_STUDY = np.array([])#np.array([0,12,16],dtype=int)#16
# SKIP = np.array([],dtype=int)
# r0_well_sampled_limit = 0.02
# wvl = 500e-9
# telDiam = 0.4
# nxSubaps = 8
# SIMNAME = 'bad_IM-'
# name_list = []

# directory_param = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\server_codes'
# os.chdir(directory_param)
# aber_r0_list_original = cn2dz2r0(np.loadtxt('cn2dz_list_new_AS.txt')).flatten()
# aber_z_list_original = np.loadtxt('z_list_new_AS.txt').flatten()

# aber_r0_list = np.concatenate([np.array([np.inf]),aber_r0_list_original])
# aber_z_list = np.concatenate([np.array([0]),aber_z_list_original])

# directory_IMCC = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\IMCC'
# os.chdir(directory_IMCC)





# for i_aber_z_list in np.arange(len(aber_z_list)):
#     if (i_aber_z_list != CASE_STUDY).all() and (CASE_STUDY.size > 0):
#         name_list.append('skipped')
#         continue
#     if (i_aber_z_list == SKIP).any():
#         name_list.append('skipped')
#         continue
#     if aber_z_list[i_aber_z_list] >= z_limit:# 25e3 # too high for common dm elevation
#         break

#     ##################################################################################
#     # setting parameters for each case
    

#     aberrationHeight = aber_z_list[i_aber_z_list]
#     i_aber_r0 = i_aber_z_list
#     aber_r0 = aber_r0_list[i_aber_r0]
    
#     # if aber_r0 < r0_well_sampled_limit: # too strong turbulence to be well-sampled by my resolution
#     #     continue
    
#     IM_rytov = rytov_number(wvl, aber_r0, aberrationHeight)
#     dsub = (telDiam/nxSubaps)
#     IM_AS = displacement(wvl, dsub, aber_r0, aberrationHeight)
    
#     name_list.append('rytov{:.2f}-AS{:.0f}%'.format(IM_rytov,IM_AS*100))
    
# to_read = 'IM'

# CEN = np.zeros((len(aber_z_list),TRIAL),dtype=float)
# CEN[:] = np.nan
    
# for i_aber_z_list in np.arange(len(aber_z_list)):
#     if (i_aber_z_list != CASE_STUDY).all() and (CASE_STUDY.size > 0):
#         continue
#     if (i_aber_z_list == SKIP).any():
#         continue
#     if aber_z_list[i_aber_z_list] >= z_limit:# 25e3 # too high for common dm elevation
#         break

#     ##################################################################################
#     # setting parameters for each case
    

#     aberrationHeight = aber_z_list[i_aber_z_list]
#     i_aber_r0 = i_aber_z_list
#     aber_r0 = aber_r0_list[i_aber_r0]
    
#     # if aber_r0 < r0_well_sampled_limit: # too strong turbulence to be well-sampled by my resolution
#     #     continue
    
#     IM_rytov = rytov_number(wvl, aber_r0, aberrationHeight)
#     dsub = (telDiam/nxSubaps)
#     IM_AS = displacement(wvl, dsub, aber_r0, aberrationHeight)
    
#     for trial in np.arange(TRIAL):
#     # trial = 0
    
#         hdul = fits.open(to_read + SIMNAME + name_list[i_aber_z_list] + '{}.fits'.format(trial))
            
#         IM = hdul[0].data
#         # plt.imshow(IM)
#         # plt.colorbar()
#         # plt.show()
        
#         IM /= np.abs(np.nanmax(IM))
        
#         # for act in ACT_LIST:#np.arange(83):
            
            
#         #     #for act in ACT_LIST:
        
#         #     # act = 42
#         #     WFS_MAP_X = rearrange1(IM[act][:44], position)
#         #     WFS_MAP_Y = rearrange1(IM[act][44:], position)
            
#         #     # cen_x = np.nansum(WFS_GRID_X*WFS_MAP_X**2)/np.nansum(WFS_GRID_X*WFS_MAP_X**2/WFS_GRID_X)
#         #     # cen_y = np.nansum(WFS_GRID_Y*WFS_MAP_Y**2)/np.nansum(WFS_GRID_Y*WFS_MAP_Y**2/WFS_GRID_Y)
            
#         #     # plt.clf()
#         #     # fig,axes = plt.subplots(1,2,sharey=True)
#         #     # axes[0].imshow(WFS_MAP_X,vmin=-1,vmax=1,cmap='RdBu')
#         #     # CS = axes[1].imshow(WFS_MAP_Y,vmin=-1,vmax=1,cmap='RdBu')
#         #     # fig.tight_layout()
#         #     # axes[0].set_title('x_slope')
#         #     # axes[1].set_title('y_slope')
#         #     # cbar_ax = fig.add_axes([1,0.17,0.05,0.65])
#         #     # axes[0].plot(DM_POS[act,0],DM_POS[act,1],marker='o',color='k')
#         #     # axes[1].plot(DM_POS[act,0],DM_POS[act,1],marker='o',color='k')
#         #     # fig.colorbar(CS,cax=cbar_ax)
#         #     # fig.suptitle('act#{}'.format(act))
#         #     # plt.show()
            
#         #     # plt.clf()
#         #     # # plt.plot(DM_POS[act,0],DM_POS[act,1],color='r',marker='o')
#         #     # plt.quiver(WFS_GRID_X,WFS_GRID_Y,
#         #     #            WFS_MAP_X,WFS_MAP_Y,scale=5)
            
#         #     WFS_MAP = (WFS_MAP_X**2 + WFS_MAP_Y**2)**0.5
            
#         #     threshold = 0.1
            
#         #     WFS_MAP[WFS_MAP < threshold] = np.nan
#         #     WFS_MAP_X[WFS_MAP < threshold] = np.nan
#         #     WFS_MAP_Y[WFS_MAP < threshold] = np.nan
            
#         #     MAX4 = np.sort(WFS_MAP[~np.isnan(WFS_MAP)].flatten())[-4:]
            
#         #     WFS_MAP_X_MAX4 = np.zeros_like(WFS_MAP)
#         #     WFS_MAP_Y_MAX4 = np.zeros_like(WFS_MAP)
#         #     WFS_MAP_X_MAX4[:] = np.nan
#         #     WFS_MAP_Y_MAX4[:] = np.nan
            
            
            
#         #     MAX4_POS = np.zeros((MAX4.shape[0],2),dtype=int)
#         #     v_x = np.zeros((MAX4.shape[0]),dtype=float)
#         #     v_y = np.zeros((MAX4.shape[0]),dtype=float)
            
#         #     for i in np.arange(MAX4.shape[0]):
#         #         loc = np.asarray(np.where(WFS_MAP == MAX4[i]))[:,0]
#         #         MAX4_POS[i] = loc
#         #         WFS_MAP_X_MAX4[tuple(MAX4_POS[i])] = WFS_MAP_X[tuple(MAX4_POS[i])]
#         #         WFS_MAP_Y_MAX4[tuple(MAX4_POS[i])] = WFS_MAP_Y[tuple(MAX4_POS[i])]
#         #         v_x[i] = WFS_MAP_X_MAX4[tuple(MAX4_POS[i])]
#         #         v_y[i] = WFS_MAP_Y_MAX4[tuple(MAX4_POS[i])]
                
            
#         #     # plt.quiver(WFS_GRID_X,WFS_GRID_Y,
#         #     #            WFS_MAP_X_MAX4,WFS_MAP_Y_MAX4,scale=5,color='r')
            
#         #     # x_standard = np.linspace(0,7)
#         #     # x = np.zeros((MAX4.shape[0],x_standard.shape[0]))
#         #     # y = np.zeros((MAX4.shape[0],x_standard.shape[0]),dtype=float)
#         #     # for i in np.arange(MAX4.shape[0]):
#         #     #     m = v_x[i]/v_y[i]
#         #     #     x[i] = x_standard
#         #     #     y[i] = m*(x[i] - MAX4_POS[i][0]) + MAX4_POS[i][1]
                
#         #     #     # y[i] = (x_standard*v_x[i] + MAX4_POS[i][1])
#         #     #     # x[i] = (x_standard*v_y[i] + MAX4_POS[i][0])
            
#         #     #     # plt.plot(y[i],x[i],ls=':',color='b',marker='',alpha=0.5)
            
#         #     cross = np.zeros((MAX4.shape[0],MAX4.shape[0],2),dtype=float)
#         #     cross[:] = np.nan
#         #     trustability = np.zeros((MAX4.shape[0],MAX4.shape[0]),dtype=float)
            
#         #     for i in np.arange(MAX4.shape[0]):
#         #         for j in np.arange(i + 1,MAX4.shape[0]):
#         #             mi = (v_y[i]/v_x[i])**-1
#         #             mj = (v_y[j]/v_x[j])**-1
#         #             xi = MAX4_POS[i][0]#x[i]
#         #             xj = MAX4_POS[j][0]#x[j]
#         #             yi = MAX4_POS[i][1]#y[i]
#         #             yj = MAX4_POS[j][1]#y[j]
                    
#         #             cross[i,j,1] = ((yi - yj) + (mj*xj - mi*xi))/(mj - mi)
#         #             cross[i,j,0] = yi + mi*(cross[i,j,1] - xi)
                    
#         #             if (
                            
#         #                     (not (((yi - cross[i,j,0])/v_x[i] >= 0)
#         #                           and ((xi - cross[i,j,1])/v_y[i] >= 0)
#         #                           and ((yj - cross[i,j,0])/v_x[j] >= 0)
#         #                           and ((xj - cross[i,j,1])/v_y[j] >= 0)))
                            
#         #                     or
                            
#         #                     ((cross[i,j,0] < -0.5)
#         #                      or (cross[i,j,0] > 7.5)
#         #                      or (cross[i,j,1] < -0.5)
#         #                      or (cross[i,j,1] > 7.5))
                            
#         #                     ):
    
#         #                 cross[i,j,0] = np.nan
#         #                 cross[i,j,1] = np.nan
                        
#         #             else:
                    
#         #                 # plt.plot(cross[i,j,0],cross[i,j,1],marker='o',color='g')
                        
#         #                 # plt.plot(y[i],x[i],ls=':',color='b',marker='')
#         #                 # plt.plot(y[j],x[j],ls=':',color='b',marker='')
#         #                 # plt.quiver(WFS_GRID_X,WFS_GRID_Y,
#         #                 #            WFS_MAP_X_MAX4,WFS_MAP_Y_MAX4,scale=5)
#         #                 # # plt.show()
#         #                 # # plt.plot(cen_x,cen_y,marker='x',color='b')
#         #                 # plt.axis('square')
#         #                 # plt.xlim(-0.5,7.5)
#         #                 # plt.ylim(-0.5,7.5)
#         #                 # plt.title(name_list[i_aber_z_list] + '\n'
#         #                 #     'act#{}'.format(act)
#         #                 #     +'\n{}{}'.format(i,j))
#         #                 # plt.show()
#         #                 size = (WFS_MAP_X[tuple(MAX4_POS[i])]**2
#         #                         + WFS_MAP_X[tuple(MAX4_POS[j])]**2)**0.5
#         #                 trustability[i,j] = size**2
            
#         #     cen_x = np.nansum(cross[...,0]*trustability)/np.nansum(trustability)
#         #     cen_y = np.nansum(cross[...,1]*trustability)/np.nansum(trustability)
            
#         #     CEN[i_aber_z_list,trial,act] = [cen_x - DM_POS[act,0], cen_y - DM_POS[act,1]]
            
#         #     # plt.plot(cen_x,cen_y,marker='o',color='b')
#         #     # plt.axis('square')
#         #     # plt.xlim(-0.5,7.5)
#         #     # plt.ylim(-0.5,7.5)
#         #     # plt.title(name_list[i_aber_z_list] + '\n'
#         #     #     'act#{}'.format(act))
#         #     # plt.show()
    
#         CEN[i_aber_z_list,trial] = measure_AS_from_IM(IM,position,ACT_LIST,DM_POS_TEMP,0.1)
        
    
# # plt.quiver(WFS_GRID,WFS_GRID,
# #                WFS_MAP_X,WFS_MAP_Y,scale=5)
# # plt.plot(DM_POS[act,0],DM_POS[act,1],color='r',marker='o')
# # plt.axis('square')
# # plt.title(name_list[i_aber_z_list] + '\n'
# #     'act#{}'.format(act))
# # plt.show()

# IM_AS_list = displacement(wvl, dsub, aber_r0_list, aber_z_list)
# plt.errorbar(IM_AS_list,
#          np.nanmean(CEN,axis=(-1)),
#          yerr=np.nanstd(CEN,axis=(-1))/np.sqrt(np.sum(~np.isnan(CEN),axis=(-1))),
#          marker='o',ls='',capsize=5)
# plt.plot(IM_AS_list,
#          IM_AS_list,
#          marker='',ls='-')
# plt.xlabel('expected_AS_spie')
# plt.ylabel('measured')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()


what you did last week
how did it fit to thesis
by end of january what you want to do

3 minutes thesis

