# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:04:33 2024

@author: gqmc62
"""

import os
import numpy as np
from matplotlib import pyplot as plt

os.chdir(r'C:\Users\gqmc62\Downloads\stae434_supplemental_files')

all_data = np.loadtxt('shimm_data_latest_sup_daynight_twilight.csv',skiprows=1,delimiter=',',usecols=(0,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21))

altitudes = np.array([0,4e3,12e3,20e3],dtype=float)

# 0=label,1=day,2=night,3=twilight,4=zenith,5=seeing,6=seeing error,7=theta0,8=7 error, 9=tau0
# 10=0km,11=10e,12=4km,13=12e,14=12km,15=14e,16=20km,17=16e
day_data = all_data[all_data[:,1] == True]
night_data = all_data[all_data[:,2] == True]

day_0km = day_data[:,10]
day_0km_error = day_data[:,11]

day_4km = day_data[:,12]
day_4km_error = day_data[:,13]

day_12km = day_data[:,14]
day_12km_error = day_data[:,15]

day_20km = day_data[:,16]
day_20km_error = day_data[:,17]


night_0km = night_data[:,10]
night_0km_error = night_data[:,11]

night_4km = night_data[:,12]
night_4km_error = night_data[:,13]

night_12km = night_data[:,14]
night_12km_error = night_data[:,15]

night_20km = night_data[:,16]
night_20km_error = night_data[:,17]



plt.clf()
plt.plot(altitudes,night_data[:,(10,12,14,16)].T,color='k',alpha=0.0075,marker=',')
plt.plot(altitudes,day_data[:,(10,12,14,16)].T,color='skyblue',alpha=0.0075,marker=',')
plt.yscale('log')
plt.ylim(1e-16,1e-10)
plt.show()

# for i in np.arange(day_data.shape[0]):
#     plt.errorbar(altitudes,
#                  day_data[i,(10,12,14,16)].T,
#                  yerr=day_data[i,(11,13,15,17)].T,color='r',alpha=0.005,capsize=5)
# for i in np.arange(night_data.shape[0]):
#     plt.errorbar(altitudes,
#                  night_data[i,(10,12,14,16)].T,
#                  yerr=night_data[i,(11,13,15,17)].T,color='k',alpha=0.005,capsize=5)
# plt.yscale('log')
# plt.ylim(1e-15,1e-10)
# plt.show()

