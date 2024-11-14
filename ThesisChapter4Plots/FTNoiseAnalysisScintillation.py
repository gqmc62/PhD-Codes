# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 21:47:40 2024

@author: PuttiwatKongkaew
"""

import numpy as np
from matplotlib import pyplot as plt
import pickle
from scipy import stats

def gauss(x, mean, std, area):
    y = area*np.exp(-((x - mean)/std)**2/2.)/(std*np.sqrt(2*np.pi))*(x[1] - x[0])
    return y

Noise = SHWFSTilt - NoNoiseSHWFSTilt

STDNoise = np.std(Noise,axis=(-1,-2))
Noise[STDNoise == 0] = np.nan
STDNoise[STDNoise == 0] = np.nan

VALID = np.zeros(STDNoise.shape,dtype=bool)
SNR_list = np.zeros(STDNoise.shape,dtype=float)

ROI = ~np.isnan(STDNoise)

for i in np.arange(4):
    for j in np.arange(4):
        for k in np.arange(4):
            for l in np.arange(4):
                for m in np.arange(4):
                    for n in np.arange(6):
                        for o in np.arange(4):
                            if (np.isnan(STDNoise[i,j,k,l,m,n,o])):
                                continue
                            # if (o == 2) or (o == 3):
                            #     continue
                            
                            phot = N_phot_list[m]
                            RON = RON_list[n]
                            Threshold = Threshold_list[o]
                            
                            SNR = phot/(np.sqrt(phot + RON**2*64**2))
                            
                            noise = Noise[i,j,k,l,m,n,o].flatten()
                            expected_std = STDNoise[i,j,k,l,m,n,o]
                            # plt.hist(noise.flatten())
                            x = np.linspace(np.nanmin(noise), np.nanmax(noise),num=100)
                            area = np.sum(np.isnan(noise) == False)
        
                            counts, bins = np.histogram(noise[np.isnan(noise)==False].flatten(),bins=100)
        
                            y = gauss(x,0,expected_std,area)
                            
                            fitting_index = np.sum(np.abs(y - counts))/area
        
        
        
        
                            noise_temp = noise[np.isnan(noise) == False].flatten()
                            noise = noise_temp[noise_temp!=0]
                            res = stats.kstest(noise,np.random.normal(0,expected_std,area))
                            
                            # pvalue_list_PN[i,j,k] = res.pvalue
                            # fitting_index_list_PN[i,j,k] = fitting_index
                            
                            bincount = np.nansum(counts!=0)
                            percentage = (stats.norm.cdf(1)-stats.norm.cdf(-1))
                            expectedtobein = bincount*percentage - 2.*np.sqrt(bincount*percentage)
                            valid = ((np.nansum(((counts[counts!=0] - y[counts!=0])**2/counts[counts!=0]) < 1) / bincount) > expectedtobein/bincount)
                            
                            VALID[i,j,k,l,m,n,o] = valid
                            SNR_list[i,j,k,l,m,n,o] = SNR
                            
                            if valid == False:
                                # print(i,j,k,l,m,n,o)
                            # if j == 0:
                                plt.plot(x,y,label='noise equivalent angle model',color='k')
                                plt.stairs(counts,bins,alpha=0.5,color='k',fill=True)
                                plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
                                plt.title('No threshold = standard ; WFS slopes threshold error (rad)'
                                          + '\n{:.0f}r0, {:.1f}dor0, {:.2f}Rytov, {:.0f}H_HA'.format(r0[i]*100,dor0[j],Rytov[k],H_HA[l]/1000)
                                          + '\n{:.0f}Nphot, {:.2f}RO, threshold={:.2e}, SNR = {:.0f}'.format(phot,RON,Threshold,SNR)
                                          + '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
                                          + '\nvalid = {:}'.format(valid))
                                LIM = np.max(np.abs([x.min(),x.max()]))
                                plt.xlim(-LIM,LIM)
                                plt.show()

for i in np.arange(4):
    for j in np.arange(4):
        for k in np.arange(4):
            for l in np.arange(4):
                for m in np.arange(4):
                    for n in np.arange(6):
                        if n!=0:
                            continue
                        for o in np.arange(4):
                            if (np.isnan(STDNoise[i,j,k,l,m,n,o])):
                                continue
                            # if (o == 2) or (o == 3):
                            #     continue
                            
                            phot = N_phot_list[m]
                            RON = RON_list[n]
                            Threshold = Threshold_list[o]
                            
                            SNR = phot/(np.sqrt(phot + RON**2*64**2))
                            
                            noise = Noise[i,j,k,l,m,n,o].flatten()
                            expected_std = STDNoise[i,j,k,l,m,n,o]
                            # plt.hist(noise.flatten())
                            x = np.linspace(np.nanmin(noise), np.nanmax(noise),num=100)
                            area = np.sum(np.isnan(noise) == False)
        
                            counts, bins = np.histogram(noise[np.isnan(noise)==False].flatten(),bins=100)
        
                            y = gauss(x,0,expected_std,area)
                            
                            fitting_index = np.sum(np.abs(y - counts))/area
        
        
        
        
                            noise_temp = noise[np.isnan(noise) == False].flatten()
                            noise = noise_temp[noise_temp!=0]
                            res = stats.kstest(noise,np.random.normal(0,expected_std,area))
                            
                            # pvalue_list_PN[i,j,k] = res.pvalue
                            # fitting_index_list_PN[i,j,k] = fitting_index
                            
                            bincount = np.nansum(counts!=0)
                            percentage = (stats.norm.cdf(1)-stats.norm.cdf(-1))
                            expectedtobein = bincount*percentage - 2.*np.sqrt(bincount*percentage)
                            valid = ((np.nansum(((counts[counts!=0] - y[counts!=0])**2/counts[counts!=0]) < 1) / bincount) > expectedtobein/bincount)
                            
                            VALID[i,j,k,l,m,n,o] = valid
                            SNR_list[i,j,k,l,m,n,o] = SNR
                            
                            if valid == False:
                                # print(i,j,k,l,m,n,o)
                            # if j == 0:
                                plt.plot(x,y,label='noise equivalent angle model',color='k')
                                plt.stairs(counts,bins,alpha=0.5,color='k',fill=True)
                                plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
                                plt.title('No threshold = standard ; WFS slopes photon error (rad)'
                                          + '\n{:.0f}r0, {:.1f}dor0, {:.2f}Rytov, {:.0f}H_HA'.format(r0[i]*100,dor0[j],Rytov[k],H_HA[l]/1000)
                                          + '\n{:.0f}Nphot, {:.2f}RO, threshold={:.2e}, SNR = {:.0f}'.format(phot,RON,Threshold,SNR)
                                          + '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
                                          + '\nvalid = {:}'.format(valid))
                                LIM = np.max(np.abs([x.min(),x.max()]))
                                plt.xlim(-LIM,LIM)
                                plt.show()
                            
for i in np.arange(4):
    for j in np.arange(4):
        for k in np.arange(4):
            for l in np.arange(4):
                for m in np.arange(4):
                    for n in np.arange(6):
                        if n!=0:
                            continue
                        for o in np.arange(4):
                            if (np.isnan(STDNoise[i,j,k,l,m,n,o])):
                                continue
                            # if (o == 2) or (o == 3):
                            #     continue
                            
                            phot = N_phot_list[m]
                            RON = RON_list[n]
                            Threshold = Threshold_list[o]
                            
                            SNR = phot/(np.sqrt(phot + RON**2*64**2))
                            
                            noise = Noise[i,j,k,l,m,n,o].flatten()
                            expected_std = STDNoise[i,j,k,l,m,n,o]
                            # plt.hist(noise.flatten())
                            x = np.linspace(np.nanmin(noise), np.nanmax(noise),num=100)
                            area = np.sum(np.isnan(noise) == False)
        
                            counts, bins = np.histogram(noise[np.isnan(noise)==False].flatten(),bins=100)
        
                            y = gauss(x,0,expected_std,area)
                            
                            fitting_index = np.sum(np.abs(y - counts))/area
        
        
        
        
                            noise_temp = noise[np.isnan(noise) == False].flatten()
                            noise = noise_temp[noise_temp!=0]
                            res = stats.kstest(noise,np.random.normal(0,expected_std,area))
                            
                            # pvalue_list_PN[i,j,k] = res.pvalue
                            # fitting_index_list_PN[i,j,k] = fitting_index
                            
                            bincount = np.nansum(counts!=0)
                            percentage = (stats.norm.cdf(1)-stats.norm.cdf(-1))
                            expectedtobein = bincount*percentage - 2.*np.sqrt(bincount*percentage)
                            valid = ((np.nansum(((counts[counts!=0] - y[counts!=0])**2/counts[counts!=0]) < 1) / bincount) > expectedtobein/bincount)
                            
                            VALID[i,j,k,l,m,n,o] = valid
                            SNR_list[i,j,k,l,m,n,o] = SNR
                            
                            if valid == False:
                                # print(i,j,k,l,m,n,o)
                            # if j == 0:
                                plt.plot(x,y,label='noise equivalent angle model',color='k')
                                plt.stairs(counts,bins,alpha=0.5,color='k',fill=True)
                                plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
                                plt.title('No threshold = standard ; WFS slopes photon error (rad)'
                                          + '\n{:.0f}r0, {:.1f}dor0, {:.2f}Rytov, {:.0f}H_HA'.format(r0[i]*100,dor0[j],Rytov[k],H_HA[l]/1000)
                                          + '\n{:.0f}Nphot, {:.2f}RO, threshold={:.2e}, SNR = {:.0f}'.format(phot,RON,Threshold,SNR)
                                          + '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
                                          + '\nvalid = {:}'.format(valid))
                                LIM = np.max(np.abs([x.min(),x.max()]))
                                plt.xlim(-LIM,LIM)
                                plt.show()

for i in np.arange(4):
    if i!=0:
        continue
    for j in np.arange(4):
        for k in np.arange(4):
            for l in np.arange(4):
                for m in np.arange(4):
                    for n in np.arange(6):
                        for o in np.arange(4):
                            if (ROI[i,j,k,l,m,n,o]*~VALID[i,j,k,l,m,n,o]) == True:
                                print(r0[i],dor0[j],Rytov[k],H_HA[l],N_phot_list[m],RON_list[n],Threshold_list[o],SNR_list[i,j,k,l,m,n,o])


for i in np.arange(4):
    for j in np.arange(4):
        for k in np.arange(4):
            for l in np.arange(4):
                
                if (np.isnan(np.std(NoNoiseSHWFSTilt[i,j,k,l]))) or (0==(np.std(NoNoiseSHWFSTilt[i,j,k,l]))):
                    continue
                
                
                noise_temp = NoNoiseSHWFSTilt[i,j,k,l,0,0,0].flatten()
                noise = noise_temp[noise_temp!=0]
                expected_std = np.std(noise)
                # plt.hist(noise.flatten())
                x = np.linspace(np.nanmin(noise), np.nanmax(noise),num=100)
                area = np.sum(np.isnan(noise) == False)

                counts, bins = np.histogram(noise[np.isnan(noise)==False].flatten(),bins=100)

                y = gauss(x,0,expected_std,area)
                
                fitting_index = np.sum(np.abs(y - counts))/area

                noise = noise[np.isnan(noise) == False].flatten()
                res = stats.kstest(noise,np.random.normal(0,expected_std,area))
                
                # pvalue_list_PN[i,j,k] = res.pvalue
                # fitting_index_list_PN[i,j,k] = fitting_index
                
                bincount = np.nansum(counts!=0)
                percentage = (stats.norm.cdf(1)-stats.norm.cdf(-1))
                expectedtobein = bincount*percentage - 2.*np.sqrt(bincount*percentage)
                valid = ((np.nansum(((counts[counts!=0] - y[counts!=0])**2/counts[counts!=0]) < 1) / bincount) > expectedtobein/bincount)
                
                if valid == False:

                    plt.plot(x,y,label='noise equivalent angle model',color='k')
                    plt.stairs(counts,bins,alpha=0.5,color='k',fill=True)
                    plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
                    plt.title('No threshold = standard ; WFS slopes threshold error (rad)'
                              + '\n{:.0f}r0, {:.1f}dor0, {:.2f}Rytov, {:.0f}H_HA'.format(r0[i]*100,dor0[j],Rytov[k],H_HA[l]/1000)
                              + '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
                              + '\nvalid = {:}'.format(valid))
                    LIM = np.max(np.abs([x.min(),x.max()]))
                    plt.xlim(-LIM,LIM)
                    plt.show()
            
icase = 0
jcase = 1
# kcase = 0
# lcase = 0
for i in np.arange(4):
    # if i!= icase:
    #     continue
    for j in np.arange(4):
        # if j!= jcase:
        #     continue
        for k in np.arange(4):
            # if k!= kcase:
            #     continue
            for l in np.arange(4):
                # if l!= lcase:
                #     continue
                for m in np.arange(4):
                    for n in np.arange(6):
                        for o in np.arange(4):
                            if (np.isnan(np.std(SHWFSTilt[i,j,k,l,m,n,o]))) or (0==(np.std(SHWFSTilt[i,j,k,l,m,n,o]))):
                                continue
                            
                            
                            noise_temp = SHWFSTilt[i,j,k,l,m,n,o].flatten()
                            noise = noise_temp[noise_temp!=0]
                            expected_std = np.std(noise)
                            # plt.hist(noise.flatten())
                            x = np.linspace(np.nanmin(noise), np.nanmax(noise),num=100)
                            area = np.sum(np.isnan(noise) == False)
            
                            counts, bins = np.histogram(noise[np.isnan(noise)==False].flatten(),bins=100)
            
                            y = gauss(x,0,expected_std,area)
                            
                            fitting_index = np.sum(np.abs(y - counts))/area
            
                            noise = noise[np.isnan(noise) == False].flatten()
                            res = stats.kstest(noise,np.random.normal(0,expected_std,area))
                            
                            # pvalue_list_PN[i,j,k] = res.pvalue
                            # fitting_index_list_PN[i,j,k] = fitting_index
                            SNR = N_phot_list[m]/(np.sqrt(N_phot_list[m] + RON_list[n]**2*64**2))
                            bincount = np.nansum(counts!=0)
                            percentage = (stats.norm.cdf(1)-stats.norm.cdf(-1))
                            expectedtobein = bincount*percentage - 2.*np.sqrt(bincount*percentage)
                            valid = ((np.nansum(((counts[counts!=0] - y[counts!=0])**2/counts[counts!=0]) < 1) / bincount) > expectedtobein/bincount)
                            
                            if valid == False:
                
                                plt.plot(x,y,label='noise equivalent angle model',color='k')
                                plt.stairs(counts,bins,alpha=0.5,color='k',fill=True)
                                plt.errorbar((bins[:-1]+bins[1:])/2.,counts,yerr=np.sqrt(counts),ls='',marker='',capsize=5,color='k')
                                plt.title('No threshold = standard ; WFS slopes threshold error (rad)'
                                          + '\n{:.0f}r0, {:.1f}dor0, {:.2f}Rytov, {:.0f}H_HA'.format(r0[i]*100,dor0[j],Rytov[k],H_HA[l]/1000)
                                          + '\n{:.0f}Nphot, {:.2f}RO, threshold={:.2e}, SNR = {:.0f}'.format(N_phot_list[m],RON_list[n],Threshold_list[o],SNR)
                                          + '\np-value = {:.3f}, {:.0f}% misrepresented'.format(res.pvalue,fitting_index*100)
                                          + '\nvalid = {:}'.format(valid))
                                LIM = np.max(np.abs([x.min(),x.max()]))
                                plt.xlim(-LIM,LIM)
                                plt.show()
                        