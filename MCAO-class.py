# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:12:23 2025

@author: gqmc62
"""

import numpy as np
import yaml
import my_phasescreen as mps
from matplotlib import pyplot as plt

def rytov_number(wvl, r0, z):
    # calculate rytov number from rytov approximation
    k = 2.*np.pi/wvl                    # float; rad/m; Wave number.
    cn2dz = r02cn2dz(r0, wvl)        # float; Turbulence strength.
    variance_theory_approx = 0.563 * k**(7./6.) * cn2dz * z**(5./6.)
    return variance_theory_approx

def cn2dz2r0(cn2dz,wvl=500e-9):
    if cn2dz == 0:
        return np.nan
    else:
        r0 = (0.423*(2.*np.pi/wvl)**2*cn2dz)**(-3./5.)
    return r0

def r02cn2dz(r0,wvl=500e-9):
    if np.isnan(r0):
        cn2dz = 0
    else:
        cn2dz = r0**(-5./3.)/(0.423*(2.*np.pi/wvl)**2)
    return cn2dz

def angle_of_arrival(wvl,dact_immediate_dm,r0,Dimmediate_dm,L0):
    FACTOR1 = 0.17819635611424933
    tiptilt_adjust = mps.tiptilt_variance_adjust_to_vK(dact_immediate_dm, np.min([Dimmediate_dm,L0]))
    FACTOR = (FACTOR1*tiptilt_adjust)**0.5
    alpha = np.sqrt(0.182 * wvl**2 * dact_immediate_dm**(-1./3.) * r0**(-5./3.))
    alpha *= FACTOR
    return alpha

# def displacement(wvl,dact_immediate_dm,dact_next_immediate_dm,r0,z,Dimmediate_dm,L0):
#     displace = angle_of_arrival(wvl,dact_immediate_dm,r0,Dimmediate_dm,L0) * z/dact_next_immediate_dm
#     return displace


class MCAO(object):
    def __init__(self,filename):
        self.filename = filename
        with open(self.filename) as file_:
            self.configDict = yaml.load(file_, Loader=yaml.SafeLoader)
        keylist = list(self.configDict.keys())
        vallist = list(self.configDict.values())
        for i_key in range(len(keylist)):
            self.__setattr__(keylist[i_key],vallist[i_key])
        del keylist
        del vallist
            
        self.calcParam()
        self.interpTurb()
        
    def interpTurb(self):
        
        turb = np.loadtxt(self.turbProfileName, delimiter=',')
        turbHeight = turb[0]
        turbProfile = turb[1]
        
        del turb
        
        old_step = turbHeight[1] - turbHeight[0]
        new_step = 50
        
        self.turbHeight = np.linspace(turbHeight.min(),
                                            turbHeight.max(),
                                            num=int(turbHeight.max() - turbHeight.min())//new_step + 1)
        
        self.turbProfile = np.interp(self.turbHeight, turbHeight, turbProfile/old_step)*new_step
        
        self.originalCn2dh = turbProfile.sum()
        
        if hasattr(self, 'scaleToR0'):
            self.turbProfile /= self.turbProfile.sum()
            self.turbProfile *= r02cn2dz(self.scaleToR0)
        
        del turbHeight, turbProfile
        
    def calcParam(self):
        self.wfsWvl = float(self.wfsWvl)
        self.mcaoFov *= 2.0*np.pi/360.0/3600.0
        self.dmHeight = np.asarray(self.dmHeight)
        self.dmProjPitch = np.asarray(self.dmProjPitch)
        # self.wfsProjPitch *= 1e-2
        self.dmProjDiam = self.pupilDiam + self.mcaoFov*self.dmHeight
        self.dmFov = np.zeros((self.nDM))
        for i in range(self.nDM):
            if self.dmHeight[i] == 0:
                self.dmFov[i] = self.mcaoFov
            else:
                self.dmFov[i] = np.min([self.pupilDiam/self.dmHeight[i],self.mcaoFov])
        self.dmCorrectionRadius = self.dmProjPitch/self.dmFov
        
    def calcControlErrorParam(self,zenithAngle,new_r0=None):
        if new_r0 is not None:
            self.scaleToR0 = new_r0
            self.interpTurb()
        dmCn2dz = self.calcDmCn2dz(zenithAngle)
        AS = self.calcActuatorShift(dmCn2dz)
        rytov = self.calcRytov(dmCn2dz)
        return AS, rytov

    def calcDmCn2dz(self,zenithAngle):
        dmCn2dz = np.zeros((self.nDM),dtype=float)
        for dm in range(self.nDM):
            dm_z = self.dmHeight[dm]
            dm_radius = self.dmCorrectionRadius[dm]
            max_z = dm_z + dm_radius
            min_z = dm_z - dm_radius
            for h in range(self.turbProfile.shape[0]):
                z = self.turbHeight[h]/np.cos(np.radians(zenithAngle))
                if (z > min_z) and (z < max_z):
                    dmCn2dz[dm] += self.turbProfile[h]
        dmCn2dz /= np.cos(np.radians(zenithAngle))
        
        return dmCn2dz

    def calcActuatorShift(self,dmCn2dz):
        AS = np.zeros((self.nDM),dtype=float)
        
        aoa = np.zeros((self.nDM),dtype=float)
        for dm in np.arange(self.nDM):
        
            aoa[dm] = angle_of_arrival(self.wfsWvl,
                                           self.dmProjPitch[dm],
                                           cn2dz2r0(dmCn2dz[dm],self.wfsWvl),
                                           self.dmProjDiam[dm],
                                           25)
        
        # print(aoa,self.dmHeight)
        for dm in np.arange(self.nDM):
            
            i_as = 0.0
            
            for i_dm in np.arange(dm): # np.arange(dm + 1) or np.arange(dm)? ???
                
                if i_dm != 0:
                    
                    # print(dm,i_dm,aoa[i_dm : dm],self.dmHeight[i_dm],self.dmHeight[i_dm - 1])
                    i_as += (aoa[i_dm : dm]**2).sum()**0.5 * np.abs(self.dmHeight[i_dm] - self.dmHeight[i_dm - 1])
                    
                if i_dm == 0:
                    
                    # print(dm,i_dm,aoa[i_dm : dm],self.dmHeight[i_dm])
                    i_as += (aoa[i_dm : dm]**2).sum()**0.5 * np.abs(self.dmHeight[i_dm])
                
            AS[dm] = i_as
            
        AS /= self.dmProjPitch
        
        return AS
    
    def calcRytov(self,dmCn2dz):
        i_rytov = np.zeros((self.nDM),dtype=float)
        rytov = np.zeros((self.nDM),dtype=float)
        for dm in range(self.nDM):
            if dm == 0:
                i_rytov[dm] = rytov_number(self.wfsWvl,
                                           cn2dz2r0(dmCn2dz[dm],self.wfsWvl),
                                           self.dmHeight[dm])
            else:
                i_rytov[dm] = rytov_number(self.wfsWvl,
                                           cn2dz2r0(dmCn2dz[dm],self.wfsWvl),
                                           np.abs(self.dmHeight[dm]))# - self.dmHeight[dm - 1]))
            

        for dm in range(self.nDM):
            rytov[dm] = i_rytov[: dm].sum() # [: dm + 1] or [: dm + 1]? ???
            
        return rytov

##################################################################################

import os
os.chdir(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter5Plots')

grid_rytov = np.loadtxt('grid_rytov.csv',delimiter=',')
grid_AS = np.loadtxt('grid_AS.csv',delimiter=',')*100
grid_WFE = np.loadtxt('grid_WFE.csv',delimiter=',')

os.chdir(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes')

obj_list = ['DKIST','EST','GeMS','MAVIS','NFIRAOS','MORFEO','MAD']
r0_list = [0.14,0.11,0.08,0.05,0.03,0.02]
zeta_list = [0,30,45,60,70,80]

color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
# color_list = color_list[::-1][3:]
# marker_list = ['o','^','>','v','<','P','x','s','D']
# marker_list = marker_list[::-1]
# marker_list = ['|','2','+','*','^','s','H']
marker_list = ['+','^','s','o','X','*']

MASTER = np.zeros((len(obj_list),len(r0_list),len(zeta_list),5,2),dtype=float)

OBJ_MASTER = []
og_AS = np.zeros((len(obj_list),len(zeta_list),5),dtype=float)
og_RY = np.zeros((len(obj_list),len(zeta_list),5),dtype=float)

for i_obj in range(len(obj_list)):
    OBJ_MASTER.append( MCAO('{}.yaml'.format(obj_list[i_obj])))
    for i_zeta in range(len(zeta_list)):
        og_AS[i_obj,i_zeta,:OBJ_MASTER[i_obj].nDM], og_RY[i_obj,i_zeta,:OBJ_MASTER[i_obj].nDM] = OBJ_MASTER[i_obj].calcControlErrorParam(zeta_list[i_zeta])

# plt.figure(figsize=(6,6))
# for i_obj in range(len(obj_list)):
#     for i_dm in range(OBJ_MASTER[i_obj].nDM):
#         if ((og_RY[i_obj,:,i_dm] > 5e-3).any()
#             and (og_RY[i_obj,:,i_dm] < 0.8).any()
#             and (og_AS[i_obj,:,i_dm]*100 > 0.5).any()
#             and (og_AS[i_obj,:,i_dm]*100 < 30).any()):
#             plt.plot(og_AS[i_obj,:,i_dm],100*og_RY[i_obj,:,i_dm],marker='o',ls='-',label='{} : DM{:.0f}km'.format(obj_list[i_obj],OBJ_MASTER[i_obj].dmHeight[i_dm]/1000))
# CS2 = plt.contourf(grid_rytov,grid_AS,grid_WFE.T)
# CS = plt.contour(grid_rytov,grid_AS,grid_WFE.T,colors='k')
# plt.clabel(CS)
# plt.colorbar(CS2)
# plt.xlim(5e-3,0.8)
# plt.ylim(0.5,30)
# plt.xscale('log')
# plt.yscale('log')
# plt.legend(loc='lower right')
# plt.xlabel('$\sigma^2_\chi$')
# plt.ylabel('actuator shift (% actuator)')
# plt.title('Current MCAO system control error at their median $r_0$')
# plt.savefig('median-r0-mcao-error.pdf')
# plt.show()

plt.figure(figsize=(6,6))
dataset = 0
for i_obj in range(len(obj_list)):
    for i_dm in range(OBJ_MASTER[i_obj].nDM):
        if ((og_RY[i_obj,:,i_dm] > 5e-3).any()
            and (og_RY[i_obj,:,i_dm] < 0.8).any()
            and (og_AS[i_obj,:,i_dm]*100 > 0.5).any()
            and (og_AS[i_obj,:,i_dm]*100 < 30).any()):
            dataset +=1
        for i_zeta in range(len(zeta_list)):
            if ((og_RY[i_obj,i_zeta,i_dm] > 5e-3)
                and (og_RY[i_obj,i_zeta,i_dm] < 0.8)
                and (og_AS[i_obj,i_zeta,i_dm]*100 > 0.5)
                and (og_AS[i_obj,i_zeta,i_dm]*100 < 30)):
                plt.plot(og_RY[i_obj,i_zeta,i_dm],
                         100*og_AS[i_obj,i_zeta,i_dm],
                         marker=marker_list[i_zeta],
                         ls='',
                         color=color_list[dataset],
                         label='{} : DM{:.0f}km : $\zeta${:.0f}$^o$'.format(
                             obj_list[i_obj],OBJ_MASTER[i_obj].dmHeight[i_dm]/1000,zeta_list[i_zeta]))
CS2 = plt.contourf(grid_rytov,grid_AS,grid_WFE.T)
CS = plt.contour(grid_rytov,grid_AS,grid_WFE.T,colors='k')
plt.clabel(CS)
plt.colorbar(CS2)
# plt.xlim(5e-3,0.8)
# plt.ylim(0.5,30)
# plt.xscale('log')
# plt.yscale('log')
plt.legend(loc='right',bbox_to_anchor=(1.8,0.5))
plt.xlabel('$\sigma^2_\chi$')
plt.ylabel('actuator shift (% actuator)')
plt.title('Current MCAO system control error at their median $r_0$')
plt.savefig('median-r0-mcao-error.pdf')
plt.show()

os.chdir(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter5Plots')
file = open('DM-Control-Error-Param-median.txt','w')
file.write('system\tdm height (km)\taperture shift (% actuator pitch)\tlog-amplitude variance \n')
for i_obj in range(len(obj_list)):
    for i_zeta in range(len(zeta_list)):
        for i_dm in range(5):
            if i_dm < OBJ_MASTER[i_obj].nDM:
                file.write('{}\t{:.0f}\t{:.0f}\t{:.1f}\t{:.3f}\n'.format(
                    obj_list[i_obj],
                    OBJ_MASTER[i_obj].dmHeight[i_dm]/1000,
                    zeta_list[i_zeta],
                    og_AS[i_obj,i_zeta,i_dm]*100,
                    og_RY[i_obj,i_zeta,i_dm]))
            else: 
                file.write('{}\t{}\t{:.0f}\t{:.1f}\t{:.3f}\n'.format(
                    obj_list[i_obj],
                    'nan',
                    zeta_list[i_zeta],
                    0,
                    0))
file.close()

os.chdir(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes')
file = open('DM-Control-Error-Param-median.txt','w')
file.write('system\tdm height (km)\tzeta (degree)\taperture shift (% actuator pitch)\tlog-amplitude variance \n')
for i_obj in range(len(obj_list)):
    for i_zeta in range(len(zeta_list)):
        for i_dm in range(5):
            if i_dm < OBJ_MASTER[i_obj].nDM:
                file.write('{}\t{:.0f}\t{:.0f}\t{:.1f}\t{:.3f}\n'.format(
                    obj_list[i_obj],
                    OBJ_MASTER[i_obj].dmHeight[i_dm]/1000,
                    zeta_list[i_zeta],
                    og_AS[i_obj,i_zeta,i_dm]*100,
                    og_RY[i_obj,i_zeta,i_dm]))
            else: 
                file.write('{}\t{}\t{:.0f}\t{:.1f}\t{:.3f}\n'.format(
                    obj_list[i_obj],
                    'nan',
                    zeta_list[i_zeta],
                    0,
                    0))
file.close()

#################################################################################
    
for i_obj in range(len(obj_list)):
    for i_r0 in range(len(r0_list)):
        for i_zeta in range(len(zeta_list)):
            AS,RYTOV = OBJ_MASTER[i_obj].calcControlErrorParam(zeta_list[i_zeta],new_r0=r0_list[i_r0])
            MASTER[i_obj,i_r0,i_zeta,:OBJ_MASTER[i_obj].nDM,0] = AS
            MASTER[i_obj,i_r0,i_zeta,:OBJ_MASTER[i_obj].nDM,1] = RYTOV

MASTER_DM_HEIGHT = np.zeros((len(obj_list),5),dtype=float)
for i in range(len(obj_list)):
    for j in range(5):
        if j < OBJ_MASTER[i].nDM:
            MASTER_DM_HEIGHT[i,j] = OBJ_MASTER[i].dmHeight[j]
        else:
            MASTER_DM_HEIGHT[i,j] = np.nan
            
INFO = {'obj_list' : obj_list,
        'r0_list' : r0_list,
        'zeta_list' : zeta_list,
        'MASTER' : MASTER,
        'MASTER_DM_HEIGHT' : MASTER_DM_HEIGHT,
        'master_dim' : 'mcao, r0, zeta, dm order, AS or Rytov'}
            
# import pickle
# with open('DM-Control-Error.pkl','wb') as file:
#     pickle.dump(INFO, file)
    

os.chdir(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter5Plots')
file = open('DM-Control-Error-Param.txt','w')
file.write('system\tr0 (cm)\tzeta (deg)\tdm height (km)\taperture shift (% actuator pitch)\tlog-amplitude variance \n')
for i_obj in range(len(obj_list)):
    for i_r0 in range(len(r0_list)):
        for i_zeta in range(len(zeta_list)):
            for i_dm in range(5):
                if i_dm < OBJ_MASTER[i_obj].nDM:
                    file.write('{}\t{:.0f}\t{:.0f}\t{:.1f}\t{:.1f}\t{:.3f}\n'.format(
                        obj_list[i_obj],
                        r0_list[i_r0]*100,
                        zeta_list[i_zeta],
                        OBJ_MASTER[i_obj].dmHeight[i_dm]/1000,
                        MASTER[i_obj,i_r0,i_zeta,i_dm,0]*100,
                        MASTER[i_obj,i_r0,i_zeta,i_dm,1]))
                else: 
                    file.write('{}\t{:.0f}\t{:.0f}\t{}\t{:.1f}\t{:.3f}\n'.format(
                        obj_list[i_obj],
                        r0_list[i_r0]*100,
                        zeta_list[i_zeta],
                        'nan',
                        0,
                        0))
file.close()

os.chdir(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes')
file = open('DM-Control-Error-Param.txt','w')
file.write('system\tr0 (cm)\tzeta (deg)\tdm height (km)\taperture shift (% actuator pitch)\tlog-amplitude variance \n')
for i_obj in range(len(obj_list)):
    for i_r0 in range(len(r0_list)):
        for i_zeta in range(len(zeta_list)):
            for i_dm in range(5):
                if i_dm < OBJ_MASTER[i_obj].nDM:
                    file.write('{}\t{:.0f}\t{:.0f}\t{:.1f}\t{:.1f}\t{:.3f}\n'.format(
                        obj_list[i_obj],
                        r0_list[i_r0]*100,
                        zeta_list[i_zeta],
                        OBJ_MASTER[i_obj].dmHeight[i_dm]/1000,
                        MASTER[i_obj,i_r0,i_zeta,i_dm,0]*100,
                        MASTER[i_obj,i_r0,i_zeta,i_dm,1]))
                else: 
                    file.write('{}\t{:.0f}\t{:.0f}\t{}\t{:.1f}\t{:.3f}\n'.format(
                        obj_list[i_obj],
                        r0_list[i_r0]*100,
                        zeta_list[i_zeta],
                        'nan',
                        0,
                        0))
file.close()



###############################################################################
###############################################################################
# best tabulate this result #
###############################################################################
###############################################################################





# ###############################################################################

# import os
# os.chdir(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter5Plots')

# grid_rytov = np.loadtxt('grid_rytov.csv',delimiter=',')
# grid_AS = np.loadtxt('grid_AS.csv',delimiter=',')*100
# grid_WFE = np.loadtxt('grid_WFE.csv',delimiter=',')

# color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
#               'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
# color_list = color_list[::-1][3:]
# # marker_list = ['o','^','>','v','<','P','x','s','D']
# marker_list = ['|','2','+','*','^','s','H']

# for i_obj in range(len(obj_list)):
#     nDM = OBJ_MASTER[i_obj].nDM
#     for iDM in range(nDM):
#         # if ((MASTER[i_obj,:,:,iDM,1] > 5e-3).any()
#         #     and (MASTER[i_obj,:,:,iDM,1] < 0.8).any()
#         #     and (MASTER[i_obj,:,:,iDM,0]*100 > 0.5).any()
#         #     and (MASTER[i_obj,:,:,iDM,0]*100 < 30).any()):
#         if ((MASTER[i_obj,:,:,iDM,1] > 0.1).any()
#             or (MASTER[i_obj,:,:,iDM,0]*100 > 4.5).any()):
#         # if ((MASTER[i_obj,:,:,iDM,1] > 0.1).any()
#         #     and (MASTER[i_obj,:,:,iDM,1] < 0.8).any()
#         #     and (MASTER[i_obj,:,:,iDM,0]*100 > 5).any()
#         #     and (MASTER[i_obj,:,:,iDM,0]*100 < 30).any()):
#             for i_r0 in range(len(r0_list)):
#                 for i_zeta in range(len(zeta_list)):
#                     plt.plot(MASTER[i_obj,i_r0,i_zeta,iDM,1],
#                               MASTER[i_obj,i_r0,i_zeta,iDM,0]*100,
#                               # marker=marker_list[i_zeta],
#                               marker='o',
#                               color=color_list[i_r0]
#                               )
#                 plt.plot(MASTER[i_obj,i_r0,:,iDM,1],
#                           MASTER[i_obj,i_r0,:,iDM,0]*100,
#                           # marker=marker_list[i_zeta],
#                           color=color_list[i_r0],
#                           ls=':',
#                           label='r0{:.0f}cm'.format(
#                               r0_list[i_r0]*100)
#                           )
#             CS2 = plt.contourf(grid_rytov,grid_AS,grid_WFE.T)
#             CS = plt.contour(grid_rytov,grid_AS,grid_WFE.T,colors='k')
#             plt.clabel(CS)
#             plt.colorbar(CS2)
#             # plt.vlines(0.1,0.5,30,ls='--',color='r')
#             # plt.hlines(4.5,8e-2,0.8,ls=':',color='r')
#             # plt.plot(np.logspace(np.log10(1.5e-2),np.log10(8e-2)),
#             #           np.logspace(np.log10(30),np.log10(4.5)),ls=':',color='r')
#             # plt.xscale('log')
#             # plt.yscale('log')
#             plt.xlabel('$\sigma^2_\chi$')
#             plt.ylabel('actuator shift (% actuator)')
#             plt.title('{} : DM{:.0f}km'.format(OBJ_MASTER[i_obj].mcaoName,
#                                                 OBJ_MASTER[i_obj].dmHeight[iDM]/1000))
#             # plt.xlim(5e-3,0.8)
#             # plt.ylim(0.5,30)
#             # plt.legend(loc='upper left')
#             plt.legend()
#             plt.show()
        

# # for i_obj in range(len(obj_list)):
# #     nDM = OBJ_MASTER[i_obj].nDM
    
# #     NFIG = (np.sum(
# #         ((MASTER[i_obj,:,:,:,1] > 5e-3)
# #         * (MASTER[i_obj,:,:,:,1] < 0.8)
# #         * (MASTER[i_obj,:,:,:,0]*100 > 0.5)
# #         * (MASTER[i_obj,:,:,:,0]*100 < 30)),
# #         axis=(0,1)) > 0).sum()
# #     print(NFIG>1)
# #     if NFIG > 1:
# #         plt.clf()
# #         fig,ax = plt.subplots(1,NFIG,figsize=(5*NFIG,5))#,width_ratios=(4,)*NFIG+(1,))
# #         nfig = 0
# #         for iDM in range(nDM):
# #             if iDM ==0:
# #                 continue
# #             if ((MASTER[i_obj,:,:,iDM,1] > 5e-3).any()
# #                 and (MASTER[i_obj,:,:,iDM,1] < 0.8).any()
# #                 and (MASTER[i_obj,:,:,iDM,0]*100 > 0.5).any()
# #                 and (MASTER[i_obj,:,:,iDM,0]*100 < 30).any()):
    
# #                 for i_r0 in range(len(r0_list)):
    
# #                     for i_zeta in range(len(zeta_list)):
    
# #                         ax[nfig].plot(MASTER[i_obj,i_r0,i_zeta,iDM,1],
# #                                   MASTER[i_obj,i_r0,i_zeta,iDM,0]*100,
# #                                   marker=marker_list[i_zeta],
# #                                   color=color_list[i_r0]
# #                                   )
# #                     ax[nfig].plot(MASTER[i_obj,i_r0,:,iDM,1],
# #                               MASTER[i_obj,i_r0,:,iDM,0]*100,
# #                               # marker=marker_list[i_zeta],
# #                               color=color_list[i_r0],
# #                               ls=':',
# #                               label='r0{:.0f}cm'.format(
# #                                   r0_list[i_r0]*100)
# #                               )
# #                 ax[nfig].legend()
# #                 CS2 = ax[nfig].contourf(grid_rytov,grid_AS,grid_WFE.T)
# #                 CS = ax[nfig].contour(grid_rytov,grid_AS,grid_WFE.T,colors='k')
# #                 ax[nfig].clabel(CS)
# #                 # fig.colorbar(CS2)
# #                 # plt.vlines(0.1,0.5,30,ls='--',color='r')
# #                 ax[nfig].hlines(4.5,8e-2,0.8,ls=':',color='r')
# #                 ax[nfig].plot(np.logspace(np.log10(1.5e-2),np.log10(8e-2)),
# #                           np.logspace(np.log10(30),np.log10(4.5)),ls=':',color='r')
# #                 ax[nfig].set_title('DM{:.0f}km'.format(OBJ_MASTER[i_obj].dmHeight[iDM]/1000))
# #                 ax[nfig].set_xlim(5e-3,0.8)
# #                 ax[nfig].set_ylim(0.5,30)
# #                 ax[nfig].set_xscale('log')
# #                 ax[nfig].set_yscale('log')
# #                 nfig += 1
        
# #         # plt.xlabel('$\sigma^2_\chi$')
# #         # plt.ylabel('actuator shift (% actuator)')
# #         # plt.legend()
        
# #         # ax[nfig].legend()
        
# #         fig.suptitle('{}'.format(OBJ_MASTER[i_obj].mcaoName))
# #         plt.savefig('mcao-error-{}.pdf'.format(OBJ_MASTER[i_obj].mcaoName))
# #         plt.show()
# #     else:

# #         for iDM in range(nDM):
# #             plt.clf()
# #             plt.figure(figsize=(5,5))
# #             if ((MASTER[i_obj,:,:,iDM,1] > 5e-3).any()
# #                 and (MASTER[i_obj,:,:,iDM,1] < 0.8).any()
# #                 and (MASTER[i_obj,:,:,iDM,0]*100 > 0.5).any()
# #                 and (MASTER[i_obj,:,:,iDM,0]*100 < 30).any()):
# #                 for i_r0 in range(len(r0_list)):
# #                     for i_zeta in range(len(zeta_list)):
# #                         plt.plot(MASTER[i_obj,i_r0,i_zeta,iDM,1],
# #                                   MASTER[i_obj,i_r0,i_zeta,iDM,0]*100,
# #                                   marker=marker_list[i_zeta],
# #                                   color=color_list[i_r0]
# #                                   )
# #                     plt.plot(MASTER[i_obj,i_r0,:,iDM,1],
# #                               MASTER[i_obj,i_r0,:,iDM,0]*100,
# #                               # marker=marker_list[i_zeta],
# #                               color=color_list[i_r0],
# #                               ls=':',
# #                               label='r0{:.0f}cm'.format(
# #                                   r0_list[i_r0]*100)
# #                               )
# #                 CS2 = plt.contourf(grid_rytov,grid_AS,grid_WFE.T)
# #                 CS = plt.contour(grid_rytov,grid_AS,grid_WFE.T,colors='k')
# #                 plt.clabel(CS)
# #                 plt.colorbar(CS2)
# #                 plt.hlines(4.5,8e-2,0.8,ls=':',color='r')
# #                 plt.plot(np.logspace(np.log10(1.5e-2),np.log10(8e-2)),
# #                           np.logspace(np.log10(30),np.log10(4.5)),ls=':',color='r')
# #                 plt.xscale('log')
# #                 plt.yscale('log')
# #                 plt.xlabel('$\sigma^2_\chi$')
# #                 plt.ylabel('actuator shift (% actuator)')
# #                 plt.title('{} : DM{:.0f}km'.format(OBJ_MASTER[i_obj].mcaoName,
# #                                                     OBJ_MASTER[i_obj].dmHeight[iDM]/1000))
# #                 plt.xlim(5e-3,0.8)
# #                 plt.ylim(0.5,30)
# #                 plt.legend(loc='upper left')
# #                 plt.savefig('mcao-error-{}.pdf'.format(OBJ_MASTER[i_obj].mcaoName))
# #                 plt.show()
        
# ###############################################################################
# sanity check for my simulation
import os
os.chdir(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes')
r0_HA = 5e-2
H_HA = 4e3
wvl = 500e-9
d = 5e-2
zeta = 0
D = 0.4
L0 = 25

test = MCAO('test.yaml')
AS,rytov = test.calcControlErrorParam(zeta,0.05)
print(wvl,test.wfsWvl)
print(d,test.dmProjPitch[1])
print(r0_HA,cn2dz2r0(test.calcDmCn2dz(zeta)[1]))
print(D,test.pupilDiam)
print(H_HA,test.dmHeight[0])
print(angle_of_arrival(wvl, d, r0_HA, D, L0)*H_HA/d*100, AS[1]*100)

print(rytov_number(wvl, r0_HA, H_HA),rytov[1])


# ###############################################################################
# # tabulate this ###############################################################
# ###############################################################################

# r0 = 0.05

# for i_zeta in range(len(zeta_list)) :
#     zeta = zeta_list[i_zeta]
    
#     wfs_gl_ha = MCAO('EST-wfs-gl-ha.yaml')
#     wfs_ha_gl = MCAO('EST-wfs-ha-gl.yaml')
    
#     a,b = wfs_gl_ha.calcControlErrorParam(zeta,r0)
#     c,d = wfs_ha_gl.calcControlErrorParam(zeta,r0)
    
#     plt.plot(b,a*100,label='wfs-gl-ha : {:.0f}'.format(zeta),marker='o',ls='-',color=color_list[i_zeta])
#     plt.plot(d,c*100,label='wfs-ha-gl : {:.0f}'.format(zeta),marker='o',ls=':',color=color_list[i_zeta])


# # plt.ylim(0.1,10)
# # plt.xlim(1e-3,3e-2)
# # plt.xscale('log')
# # plt.yscale('log')


# plt.legend()
# plt.show()


