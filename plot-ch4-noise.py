# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:36:23 2024

@author: gqmc62
"""

def my_weight_avg(data,std,axis):
    a = np.nansum(data/std**2,axis=axis)
    b = np.nansum(1/std**2,axis=axis)
    mean = a/b
    std = 1/b**0.5
    return mean,std
    

def my_avg(data,axis):
    mean = np.nanmean(data,axis=axis)
    std = np.nanstd(data,axis=axis)
    return mean,std
    


plt.plot(NEW_RON_list[:,1:].flatten(),RON_error[:,1:].flatten())
for i in np.arange(4):
    for j in np.arange(4):
        for k in np.arange(4):
            for l in np.arange(4):
                plt.plot(NEW_RON_list[:,1:].flatten(),RON_rms[i,j,k,l,:,1:].flatten(),ls='',marker='.')
plt.xscale('log')
plt.yscale('log')
plt.show()



flatten_RON = np.unique(NEW_RON_list)

flatten_RON_rms_mean = np.zeros((4,flatten_RON.shape[0]),dtype=float)
flatten_RON_rms_std = np.zeros((4,flatten_RON.shape[0]),dtype=float)

for ll in np.arange(4):
    for nnn in np.arange(flatten_RON.shape[0]):
        element = []
        for ii in np.arange(4):
            for jj in np.arange(4):
                for kk in np.arange(4):
                    for mm in np.arange(NEW_RON_list.shape[0]):
                        for nn in np.arange(NEW_RON_list.shape[1]):
                            if ((NEW_RON_list[mm,nn] == flatten_RON[nnn])
                                or np.isclose(NEW_RON_list[mm,nn],flatten_RON[nnn])):
                                element.append(RON_rms[ll,ii,jj,kk,mm,nn])
        flatten_RON_rms_mean[ll,nnn] = np.nanmean(element)
        flatten_RON_rms_std[ll,nnn] = np.nanstd(element)
    
for ii in np.arange(4):
    plt.errorbar(flatten_RON,flatten_RON_rms_mean[ii],yerr=flatten_RON_rms_std[ii],capsize=5,marker='o',ls='')
plt.plot(NEW_RON_list[:,1:].flatten(),RON_error[:,1:].flatten())
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-5,10)
plt.show()




flatten_RON = np.unique(NEW_RON_list)

flatten_RON_rms_mean = np.zeros((4,4,4,4,flatten_RON.shape[0]),dtype=float)
flatten_RON_rms_std = np.zeros((4,4,4,4,flatten_RON.shape[0]),dtype=float)

for ii in np.arange(4):
    for jj in np.arange(4):
        for kk in np.arange(4):
            for ll in np.arange(4):
                for nnn in np.arange(flatten_RON.shape[0]):
                    element = []
                    for mm in np.arange(NEW_RON_list.shape[0]):
                        for nn in np.arange(NEW_RON_list.shape[1]):
                            if ((NEW_RON_list[mm,nn] == flatten_RON[nnn])
                                or np.isclose(NEW_RON_list[mm,nn],flatten_RON[nnn])):
                                element.append(RON_rms[ii,jj,kk,ll,mm,nn])
                    flatten_RON_rms_mean[ii,jj,kk,ll,nnn] = np.nanmean(element)
                    flatten_RON_rms_std[ii,jj,kk,ll,nnn] = np.nanstd(element)

flatten_RON_rms_mean[np.where(flatten_RON_rms_mean==0)] = np.nan
flatten_RON_rms_std[np.where(flatten_RON_rms_std==0)] = np.nan




color=['c','m','y','k']
for ii in np.arange(4):
    for jj in np.arange(4):
        for kk in np.arange(4):
            for ll in np.arange(4):
                plt.errorbar(flatten_RON,
                             flatten_RON_rms_mean[ii,jj,kk,ll],
                             yerr=flatten_RON_rms_std[ii,jj,kk,ll],
                             capsize=5,marker='o',ls='',
                             color=color[ll])

plt.plot(NEW_RON_list[:,1:].flatten(),RON_error[:,1:].flatten())
plt.xscale('log')
plt.yscale('log')
plt.show()








mean,std = my_weight_avg(flatten_RON_rms_mean,flatten_RON_rms_std,(0,2,3))
for i in np.arange(4):
    plt.errorbar(flatten_RON,
                 mean[i],yerr=std[i],
                 capsize=5,marker='o',ls='',color=color[i])
plt.plot(NEW_RON_list[:,1:].flatten(),RON_error[:,1:].flatten())
plt.xscale('log')
plt.yscale('log')
plt.show()





mean,std = my_weight_avg(flatten_RON_rms_mean,flatten_RON_rms_std,(0,1,2,3))
plt.errorbar(flatten_RON,
             mean,yerr=std,
             capsize=5,marker='o',ls='')
plt.plot(NEW_RON_list[:,1:].flatten(),RON_error[:,1:].flatten())
plt.xscale('log')
plt.yscale('log')
plt.show()



