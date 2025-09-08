# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:35:16 2025

@author: gqmc62
"""

import numpy as np
import matplotlib.pyplot as plt

from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov

from tqdm import tqdm

import pickle

from matplotlib.lines import Line2D
import matplotlib as mpl

import MyWFS as MWFS

# from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

fontsize = 10
figsize = [4,4]
dpi = 600

mpl.rcParams['axes.titlesize'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['figure.titlesize'] = fontsize
mpl.rcParams['figure.figsize'] = figsize
mpl.rcParams['figure.dpi'] = dpi
mpl.rcParams['savefig.dpi'] = dpi
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.format'] = 'pdf'

color_list = np.array(list(mpl.colors.TABLEAU_COLORS.keys()))

LOAD = True
SAVE = False
PLOT = True


Threshold = [0,1e-4,1e-3,1e-2,1e-1]

AVG = 10000#10000
N = 64
n_cog = 17
TRIAL = 512#512

# partial_cog = MWFS.SHWFS_from_U_adaptive_ROI(n_cog,N)
# full_cog = MWFS.OneStepSHWFS(N)



with open('ps05_10000.pkl','rb') as file: # tip/tilt piston removed.
    data = pickle.load(file)
ps05 = np.copy(data)


ps1 = ps05*((1/0.5) * (1))**(5./6.)
ps2 = ps05*((1/0.5) * (2))**(5./6.)
ps3 = ps05*((1/0.5) * (3))**(5./6.)


CONVERT = 500/2

dif_full_turb = np.zeros((4,len(Threshold),TRIAL,AVG),dtype=float)
dif_part_turb = np.zeros((4,len(Threshold),TRIAL,AVG),dtype=float)
dif_full_dif = np.zeros((len(Threshold),TRIAL),dtype=float)
dif_part_dif = np.zeros((len(Threshold),TRIAL),dtype=float)

######################################################################################
N = 64
n = 64
X = np.arange(N) - N/2 + 0.5
YY,XX = np.meshgrid(X,X)
x = np.arange(n) - n/2 + 0.5
yy,xx = np.meshgrid(x,x)

center = np.zeros((len(Threshold),TRIAL),dtype=float)
sample = np.linspace(-N/2,N/2,TRIAL)/4

if LOAD:
    
    with open('wfs-response64-64.pkl','rb') as file:
        data = pickle.load(file)
        sample = data['sample']
        center = data['center']
        Threshold = data['Threshold']
    # mean = []
    # std = []
    # for i_threshold in range(len(Threshold)):
    #     mean.append(np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT)
    #     std.append(np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT)#)
        
    # full_05_mean = np.copy(mean)
    # full_05_std = np.copy(std)


else:
    full_cog = MWFS.OneStepSHWFS(N)
    for i_threshold in range(len(Threshold)):
        threshold = Threshold[i_threshold]
        for trial in range(TRIAL):
            i = sample[trial]
            phase = (YY + XX)*(i*(np.pi/N))
            center[i_threshold,trial] = full_cog.measure_no_noise_with_threshold(np.exp(1j*phase), threshold)[0]/np.pi

CONVERT = 500/2

if PLOT:
    
    for i_threshold in range(len(Threshold)):
        plt.plot(sample*CONVERT, (center[i_threshold] - sample)*CONVERT,label='Threshold = {:1.0e}'.format(Threshold[i_threshold]))
        dif_full_dif[i_threshold] = (center[i_threshold] - sample)*CONVERT
    
    
    # ax2.set_ylim(-20,20)
    plt.xlim(-1000,1000)
    plt.ylim(-10,10)
    plt.xlabel('Gradient OPD (nm)')
    plt.ylabel('Measurement OPD error (nm)')
    plt.legend()
    plt.savefig('wfs-response64-64.pdf',bbox_inches='tight')
    plt.show()
    
# full_dif_lim_mean = (center - sample)*CONVERT


if SAVE:
    with open('wfs-response64-64.pkl','wb') as file:
        pickle.dump({'sample' : sample,
                      'center' : center,
                      'Threshold' : Threshold,
                      # 'full_dif_lim_mean' : full_dif_lim_mean
            }, file)

###############################################################################

N = 64
n = 64
X = np.arange(N) - N/2 + 0.5
YY,XX = np.meshgrid(X,X)
x = np.arange(n) - n/2 + 0.5
yy,xx = np.meshgrid(x,x)

# Threshold = [0,1e-4,1e-3,1e-2,1e-1]

center = np.zeros((len(Threshold),TRIAL,AVG),dtype=float)
sample = np.linspace(-N/2,N/2,TRIAL)/4

wfs_type = 'full'

def measure_one(i_threshold, trial, avg, Threshold, sample, N, YY, XX, ps, wfs_type, n_cog):
    if wfs_type == 'full':
        wfs = MWFS.OneStepSHWFS(N)
    elif wfs_type == 'partial':
        wfs = MWFS.SHWFS_from_U_adaptive_ROI(n_cog,N)
    threshold = Threshold[i_threshold]
    i = sample[trial]
    phase = (YY + XX) * (i * (np.pi / N)) + ps[avg]
    value = wfs.measure_no_noise_with_threshold(np.exp(1j * phase), threshold)[0] / np.pi
    return i_threshold, trial, avg, value

if LOAD:
    with open('wfs-response64-64-turb-05.pkl','rb') as file:
        data = pickle.load(file)
        sample = data['sample']
        center = data['center']
        Threshold = data['Threshold']
    # mean = []
    # std = []
    # for i_threshold in range(len(Threshold)):
    #     mean.append(np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT)
    #     std.append(np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT)#)
        
    # full_05_mean = np.copy(mean)
    # full_05_std = np.copy(std)


else:
    
    tasks = [
        (i_threshold, trial, avg)
        for i_threshold in range(len(Threshold))
        for trial in range(TRIAL)
        for avg in range(AVG)
    ]
    
    results = Parallel(n_jobs=num_cores, backend="loky")(
    delayed(measure_one)(i_threshold, trial, avg, Threshold, sample, N, YY, XX, ps05, wfs_type, n_cog)
    for i_threshold, trial, avg in tqdm(tasks, desc="Processing", total=len(tasks))
    )

    for i_threshold, trial, avg, value in results:
        center[i_threshold, trial, avg] = value
        
CONVERT = 500/2

if PLOT:

    for i_threshold in range(len(Threshold)):
        
        plt.plot(sample*CONVERT, np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT,
                     #yerr=np.nanstd(center[i_threshold],axis=-1)*CONVERT/50**0.5,capsize=5,
                     label='Threshold = {:1.0e}'.format(Threshold[i_threshold]))
        # plt.fill_between(sample*CONVERT,
        #                   np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT - np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT,#,
        #                   np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT + np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT,#,
        #                   alpha=0.5)
        dif_full_turb[0,i_threshold] = (center[i_threshold].T - sample).T*CONVERT
    
        
        
    plt.xlim(-1000,1000)
    plt.ylim(-10,10)
    plt.xlabel('Gradient OPD (nm)')
    plt.ylabel('Measurement OPD error (nm)')
    plt.legend()
    plt.savefig('wfs-response64-64-turb-05.pdf',bbox_inches='tight')
    plt.show()

# mean = []
# std = []
# for i_threshold in range(len(Threshold)):
#     mean.append(np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT)
#     std.append(np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT)#)
    
# full_05_mean = np.copy(mean)
# full_05_std = np.copy(std)



if SAVE:

    with open('wfs-response64-64-turb-05.pkl','wb') as file:
        pickle.dump({'sample' : sample,
                      'center' : center,
                      'Threshold' : Threshold,
                      # 'full_05_mean' : full_05_mean,
                      # 'full_05_std' : full_05_std
            }, file)
    
###############################################################################

N = 64
n = 64
X = np.arange(N) - N/2 + 0.5
YY,XX = np.meshgrid(X,X)
x = np.arange(n) - n/2 + 0.5
yy,xx = np.meshgrid(x,x)

# Threshold = [0,1e-4,1e-3,1e-2,1e-1]

center = np.zeros((len(Threshold),TRIAL,AVG),dtype=float)
sample = np.linspace(-N/2,N/2,TRIAL)/4
# print('I dont think this section is correct data lol')

if LOAD:
    with open('wfs-response64-64-turb-1.pkl','rb') as file:
        data = pickle.load(file)
        sample = data['sample']
        center = data['center']
        Threshold = data['Threshold']
    # mean = []
    # std = []
    # for i_threshold in range(len(Threshold)):
    #     mean.append(np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT)
    #     std.append(np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT)#)
        
    # full_05_mean = np.copy(mean)
    # full_05_std = np.copy(std)


else:
    
    tasks = [
        (i_threshold, trial, avg)
        for i_threshold in range(len(Threshold))
        for trial in range(TRIAL)
        for avg in range(AVG)
    ]
    
    results = Parallel(n_jobs=num_cores, backend="loky")(
    delayed(measure_one)(i_threshold, trial, avg, Threshold, sample, N, YY, XX, ps1, wfs_type, n_cog)
    for i_threshold, trial, avg in tqdm(tasks, desc="Processing", total=len(tasks))
    )

    for i_threshold, trial, avg, value in results:
        center[i_threshold, trial, avg] = value
        
CONVERT = 500/2

if PLOT:

    for i_threshold in range(len(Threshold)):
        
        plt.plot(sample*CONVERT, np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT,
                     #yerr=np.nanstd(center[i_threshold],axis=-1)*CONVERT/50**0.5,capsize=5,
                     label='Threshold = {:1.0e}'.format(Threshold[i_threshold]))
        # plt.fill_between(sample*CONVERT,
        #                   np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT - np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT,#,
        #                   np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT + np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT,#,
        #                   alpha=0.5)
        dif_full_turb[1,i_threshold] = (center[i_threshold].T - sample).T*CONVERT
    
        # — Plot 2 on the right —
        
    plt.xlim(-1000,1000)
    plt.ylim(-10,10)
    plt.xlabel('Gradient OPD (nm)')
    plt.ylabel('Measurement OPD error (nm)')
    plt.legend()
    plt.savefig('wfs-response64-64-turb-1.pdf',bbox_inches='tight')
    plt.show()

# mean = []
# std = []
# for i_threshold in range(len(Threshold)):
#     mean.append(np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT)
#     std.append(np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT)#)
    
# full_1_mean = np.copy(mean)
# full_1_std = np.copy(std)


if SAVE:
    with open('wfs-response64-64-turb-1.pkl','wb') as file:
        pickle.dump({'sample' : sample,
                      'center' : center,
                      'Threshold' : Threshold,
                      # 'full_1_mean' : full_1_mean,
                      # 'full_1_std' : full_1_std
            }, file)

###############################################################################    

N = 64
n = 64
X = np.arange(N) - N/2 + 0.5
YY,XX = np.meshgrid(X,X)
x = np.arange(n) - n/2 + 0.5
yy,xx = np.meshgrid(x,x)

# Threshold = [0,1e-4,1e-3,1e-2,1e-1]

center = np.zeros((len(Threshold),TRIAL,AVG),dtype=float)
sample = np.linspace(-N/2,N/2,TRIAL)/4
# print('I dont think this section is correct data lol')

if LOAD:
    with open('wfs-response64-64-turb-2.pkl','rb') as file:
        data = pickle.load(file)
        sample = data['sample']
        center = data['center']
        Threshold = data['Threshold']
    # mean = []
    # std = []
    # for i_threshold in range(len(Threshold)):
    #     mean.append(np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT)
    #     std.append(np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT)#)
        
    # full_05_mean = np.copy(mean)
    # full_05_std = np.copy(std)


else:
    
    tasks = [
        (i_threshold, trial, avg)
        for i_threshold in range(len(Threshold))
        for trial in range(TRIAL)
        for avg in range(AVG)
    ]
    
    results = Parallel(n_jobs=num_cores, backend="loky")(
    delayed(measure_one)(i_threshold, trial, avg, Threshold, sample, N, YY, XX, ps2, wfs_type, n_cog)
    for i_threshold, trial, avg in tqdm(tasks, desc="Processing", total=len(tasks))
    )

    for i_threshold, trial, avg, value in results:
        center[i_threshold, trial, avg] = value
        
CONVERT = 500/2

if PLOT:

    for i_threshold in range(len(Threshold)):
        
        plt.plot(sample*CONVERT, np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT,
                     #yerr=np.nanstd(center[i_threshold],axis=-1)*CONVERT/50**0.5,capsize=5,
                     label='Threshold = {:1.0e}'.format(Threshold[i_threshold]))
        # plt.fill_between(sample*CONVERT,
        #                   np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT - np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT,#,
        #                   np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT + np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT,#,
        #                   alpha=0.5)
        dif_full_turb[2,i_threshold] = (center[i_threshold].T - sample).T*CONVERT
    
# — Plot 2 on the right —

    plt.xlim(-1000,1000)
    plt.ylim(-10,10)
    plt.xlabel('Gradient OPD (nm)')
    plt.ylabel('Measurement OPD error (nm)')
    plt.legend()
    
    plt.savefig('wfs-response64-64-turb-2.pdf',bbox_inches='tight')
    plt.show()

# mean = []
# std = []
# for i_threshold in range(len(Threshold)):
#     mean.append(np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT)
#     std.append(np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT)#)
    
# full_2_mean = np.copy(mean)
# full_2_std = np.copy(std)

if SAVE:
    with open('wfs-response64-64-turb-2.pkl','wb') as file:
        pickle.dump({'sample' : sample,
                      'center' : center,
                      'Threshold' : Threshold,
                      # 'full_2_mean' : full_2_mean,
                      # 'full_2_std' : full_2_std
            }, file)
#     ##########################################################

N = 64
n = 64
X = np.arange(N) - N/2 + 0.5
YY,XX = np.meshgrid(X,X)
x = np.arange(n) - n/2 + 0.5
yy,xx = np.meshgrid(x,x)


center = np.zeros((len(Threshold),TRIAL,AVG),dtype=float)
sample = np.linspace(-N/2,N/2,TRIAL)/4

if LOAD:
    with open('wfs-response64-64-turb-3.pkl','rb') as file:
        data = pickle.load(file)
        sample = data['sample']
        center = data['center']
        Threshold = data['Threshold']
    # mean = []
    # std = []
    # for i_threshold in range(len(Threshold)):
    #     mean.append(np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT)
    #     std.append(np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT)#)
        
    # full_05_mean = np.copy(mean)
    # full_05_std = np.copy(std)


else:
    
    tasks = [
        (i_threshold, trial, avg)
        for i_threshold in range(len(Threshold))
        for trial in range(TRIAL)
        for avg in range(AVG)
    ]
    
    results = Parallel(n_jobs=num_cores, backend="loky")(
    delayed(measure_one)(i_threshold, trial, avg, Threshold, sample, N, YY, XX, ps3, wfs_type, n_cog)
    for i_threshold, trial, avg in tqdm(tasks, desc="Processing", total=len(tasks))
    )

    for i_threshold, trial, avg, value in results:
        center[i_threshold, trial, avg] = value
        
CONVERT = 500/2

if PLOT:

    for i_threshold in range(len(Threshold)):
        
        plt.plot(sample*CONVERT, np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT,
                     #yerr=np.nanstd(center[i_threshold],axis=-1)*CONVERT/50**0.5,capsize=5,
                     label='Threshold = {:1.0e}'.format(Threshold[i_threshold]))
        # plt.fill_between(sample*CONVERT,
        #                   np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT - np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT,#,
        #                   np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT + np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT,#,
        #                   alpha=0.5)
        dif_full_turb[3,i_threshold] = (center[i_threshold].T - sample).T*CONVERT
    
    # — Plot 2 on the right —
    
    plt.xlim(-1000,1000)
    plt.ylim(-10,10)
    plt.xlabel('Gradient OPD (nm)')
    plt.ylabel('Measurement OPD error (nm)')
    plt.legend()
   
    plt.savefig('wfs-response64-64-turb-3.pdf',bbox_inches='tight')
    plt.show()

# mean = []
# std = []
# for i_threshold in range(len(Threshold)):
#     mean.append(np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT)
#     std.append(np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT)#)
    
# full_3_mean = np.copy(mean)
# full_3_std = np.copy(std)

if SAVE:
    with open('wfs-response64-64-turb-3.pkl','wb') as file:
        pickle.dump({'sample' : sample,
                      'center' : center,
                      'Threshold' : Threshold,
                      # 'full_3_mean' : full_3_mean,
                      # 'full_3_std' : full_3_std
            }, file)
    
###############################################################################
N = 64
n = 17
X = np.arange(N) - N/2 + 0.5
YY,XX = np.meshgrid(X,X)
x = np.arange(n) - n/2 + 0.5
yy,xx = np.meshgrid(x,x)

# Threshold = [0,1e-4,1e-3,1e-2,1e-1]

center = np.zeros((len(Threshold),TRIAL),dtype=float)
sample = np.linspace(-N/2,N/2,TRIAL)/4
partial_cog = MWFS.SHWFS_from_U_adaptive_ROI(n_cog,N)
wfs_type = 'partial'

if LOAD:

    with open('wfs-response64-16-moving-centre.pkl','rb') as file:
        data = pickle.load(file)
        sample = data['sample']
        center = data['center']
        Threshold = data['Threshold']
    # mean = []
    # std = []
    # for i_threshold in range(len(Threshold)):
    #     mean.append(np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT)
    #     std.append(np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT)#)
        
    # full_05_mean = np.copy(mean)
    # full_05_std = np.copy(std)


else:

    for i_threshold in range(len(Threshold)):
        threshold = Threshold[i_threshold]
        for trial in range(TRIAL):
            i = sample[trial]
            phase = (YY + XX)*(i*(np.pi/N))
            center[i_threshold,trial] = partial_cog.measure_no_noise_with_threshold(np.exp(1j*phase), threshold)[0]/np.pi
CONVERT = 500/2

if PLOT:

    for i_threshold in range(len(Threshold)):
        plt.plot(sample*CONVERT, (center[i_threshold] - sample)*CONVERT,label='Threshold = {:1.0e}'.format(Threshold[i_threshold]))
        dif_part_dif[i_threshold] = (center[i_threshold] - sample)*CONVERT
    
    
    # ax2.set_ylim(-20,20)
    plt.xlim(-1000,1000)
    plt.ylim(-10,10)
    plt.xlabel('Gradient OPD (nm)')
    plt.ylabel('Measurement OPD error (nm)')
    plt.legend()
    plt.savefig('wfs-response64-16-moving-centre.pdf',bbox_inches='tight')
    plt.show()
    
# partial_dif_lim_mean = (center - sample)*CONVERT

if SAVE:

    with open('wfs-response64-16-moving-centre.pkl','wb') as file:
        pickle.dump({'sample' : sample,
                      'center' : center,
                      # 'Threshold' : Threshold,
                      # 'partial_dif_lim_mean' : partial_dif_lim_mean
            }, file)
    


    
###############################################################################

    

N = 64
n = 17
X = np.arange(N) - N/2 + 0.5
YY,XX = np.meshgrid(X,X)
x = np.arange(n) - n/2 + 0.5
yy,xx = np.meshgrid(x,x)

# Threshold = [0,1e-4,1e-3,1e-2,1e-1]

center = np.zeros((len(Threshold),TRIAL,AVG),dtype=float)
sample = np.linspace(-N/2,N/2,TRIAL)/4

if LOAD:

    with open('wfs-response64-16-moving-centre-turb-05.pkl','rb') as file:
        data = pickle.load(file)
        sample = data['sample']
        center = data['center']
        Threshold = data['Threshold']
    # mean = []
    # std = []
    # for i_threshold in range(len(Threshold)):
    #     mean.append(np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT)
    #     std.append(np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT)#)
        
    # full_05_mean = np.copy(mean)
    # full_05_std = np.copy(std)


else:

    tasks = [
        (i_threshold, trial, avg)
        for i_threshold in range(len(Threshold))
        for trial in range(TRIAL)
        for avg in range(AVG)
    ]
    
    results = Parallel(n_jobs=num_cores, backend="loky")(
    delayed(measure_one)(i_threshold, trial, avg, Threshold, sample, N, YY, XX, ps05, wfs_type, n_cog)
    for i_threshold, trial, avg in tqdm(tasks, desc="Processing", total=len(tasks))
    )

    for i_threshold, trial, avg, value in results:
        center[i_threshold, trial, avg] = value
        
CONVERT = 500/2

if PLOT:

    for i_threshold in range(len(Threshold)):
        
        plt.plot(sample*CONVERT, np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT,
                     #yerr=np.nanstd(center[i_threshold],axis=-1)*CONVERT/50**0.5,capsize=5,
                     label='Threshold = {:1.0e}'.format(Threshold[i_threshold]))
        # plt.fill_between(sample*CONVERT,
        #                   np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT - np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT,#,
        #                   np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT + np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT,#,
        #                   alpha=0.5)
        dif_part_turb[0,i_threshold] = (center[i_threshold].T - sample).T*CONVERT
    
    # — Plot 2 on the right —
        
    plt.xlim(-1000,1000)
    plt.ylim(-10,10)
    plt.xlabel('Gradient OPD (nm)')
    plt.ylabel('Measurement OPD error (nm)')
    plt.legend()
    plt.savefig('wfs-response64-16-moving-centre-turb-05.pdf',bbox_inches='tight')
    plt.show()


# mean = []
# std = []
# for i_threshold in range(len(Threshold)):
#     mean.append(np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT)
#     std.append(np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT)#)
    
# partial_05_mean = np.copy(mean)
# partial_05_std = np.copy(std)



if SAVE:

    with open('wfs-response64-16-moving-centre-turb-05.pkl','wb') as file:
        pickle.dump({'sample' : sample,
                      'center' : center,
                      'Threshold' : Threshold,
                      # 'partial_05_mean' : partial_05_mean,
                      # 'partial_05_std' : partial_05_std
            }, file)    


###############################################################################

    

N = 64
n = 17
X = np.arange(N) - N/2 + 0.5
YY,XX = np.meshgrid(X,X)
x = np.arange(n) - n/2 + 0.5
yy,xx = np.meshgrid(x,x)

# Threshold = [0,1e-4,1e-3,1e-2,1e-1]

center = np.zeros((len(Threshold),TRIAL,AVG),dtype=float)
sample = np.linspace(-N/2,N/2,TRIAL)/4

if LOAD:

    with open('wfs-response64-16-moving-centre-turb-1.pkl','rb') as file:
        data = pickle.load(file)
        sample = data['sample']
        center = data['center']
        Threshold = data['Threshold']
    # mean = []
    # std = []
    # for i_threshold in range(len(Threshold)):
    #     mean.append(np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT)
    #     std.append(np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT)#)
        
    # full_05_mean = np.copy(mean)
    # full_05_std = np.copy(std)


else:

    tasks = [
        (i_threshold, trial, avg)
        for i_threshold in range(len(Threshold))
        for trial in range(TRIAL)
        for avg in range(AVG)
    ]
    
    results = Parallel(n_jobs=num_cores, backend="loky")(
    delayed(measure_one)(i_threshold, trial, avg, Threshold, sample, N, YY, XX, ps1, wfs_type, n_cog)
    for i_threshold, trial, avg in tqdm(tasks, desc="Processing", total=len(tasks))
    )

    for i_threshold, trial, avg, value in results:
        center[i_threshold, trial, avg] = value
        
CONVERT = 500/2

if PLOT:

    for i_threshold in range(len(Threshold)):
        
        plt.plot(sample*CONVERT, np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT,
                     #yerr=np.nanstd(center[i_threshold],axis=-1)*CONVERT/50**0.5,capsize=5,
                     label='Threshold = {:1.0e}'.format(Threshold[i_threshold]))
        # plt.fill_between(sample*CONVERT,
        #                   np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT - np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT,#,
        #                   np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT + np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT,#,
        #                   alpha=0.5)
        dif_part_turb[1,i_threshold] = (center[i_threshold].T - sample).T*CONVERT
    
    # — Plot 2 on the right —
    
    plt.xlim(-1000,1000)
    plt.ylim(-10,10)
    plt.xlabel('Gradient OPD (nm)')
    plt.ylabel('Measurement OPD error (nm)')
    plt.legend()
    plt.savefig('wfs-response64-16-moving-centre-turb-1.pdf',bbox_inches='tight')
    plt.show()


# mean = []
# std = []
# for i_threshold in range(len(Threshold)):
#     mean.append(np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT)
#     std.append(np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT)#)

# partial_1_mean = np.copy(mean)
# partial_1_std = np.copy(std)



if SAVE:

    with open('wfs-response64-16-moving-centre-turb-1.pkl','wb') as file:
        pickle.dump({'sample' : sample,
                      'center' : center,
                      'Threshold' : Threshold,
                      # 'partial_1_mean' : partial_1_mean,
                      # 'partial_1_std' : partial_1_std
            }, file)    
        


    
#     #################################
# #################################################################################
# ######################

    

N = 64
n = 17
X = np.arange(N) - N/2 + 0.5
YY,XX = np.meshgrid(X,X)
x = np.arange(n) - n/2 + 0.5
yy,xx = np.meshgrid(x,x)

# Threshold = [0,1e-4,1e-3,1e-2,1e-1]

center = np.zeros((len(Threshold),TRIAL,AVG),dtype=float)
sample = np.linspace(-N/2,N/2,TRIAL)/4

if LOAD:

    with open('wfs-response64-16-moving-centre-turb-2.pkl','rb') as file:
        data = pickle.load(file)
        sample = data['sample']
        center = data['center']
        Threshold = data['Threshold']
    # mean = []
    # std = []
    # for i_threshold in range(len(Threshold)):
    #     mean.append(np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT)
    #     std.append(np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT)#)
        
    # full_05_mean = np.copy(mean)
    # full_05_std = np.copy(std)


else:

    tasks = [
        (i_threshold, trial, avg)
        for i_threshold in range(len(Threshold))
        for trial in range(TRIAL)
        for avg in range(AVG)
    ]
    
    results = Parallel(n_jobs=num_cores, backend="loky")(
    delayed(measure_one)(i_threshold, trial, avg, Threshold, sample, N, YY, XX, ps2, wfs_type, n_cog)
    for i_threshold, trial, avg in tqdm(tasks, desc="Processing", total=len(tasks))
    )

    for i_threshold, trial, avg, value in results:
        center[i_threshold, trial, avg] = value
        
CONVERT = 500/2

if PLOT:

    for i_threshold in range(len(Threshold)):
        
        plt.plot(sample*CONVERT, np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT,
                     #yerr=np.nanstd(center[i_threshold],axis=-1)*CONVERT/50**0.5,capsize=5,
                     label='Threshold = {:1.0e}'.format(Threshold[i_threshold]))
        # plt.fill_between(sample*CONVERT,
        #                   np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT - np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT,#,
        #                   np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT + np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT,#,
        #                   alpha=0.5)
        dif_part_turb[2,i_threshold] = (center[i_threshold].T - sample).T*CONVERT
    
    # — Plot 2 on the right —
    
    plt.xlim(-1000,1000)
    plt.ylim(-10,10)
    plt.xlabel('Gradient OPD (nm)')
    plt.ylabel('Measurement OPD error (nm)')
    plt.legend()
    plt.savefig('wfs-response64-16-moving-centre-turb-2.pdf',bbox_inches='tight')
    plt.show()


# mean = []
# std = []
# for i_threshold in range(len(Threshold)):
#     mean.append(np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT)
#     std.append(np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT)#)
    
# partial_2_mean = np.copy(mean)
# partial_2_std = np.copy(std)



if SAVE:

    with open('wfs-response64-16-moving-centre-turb-2.pkl','wb') as file:
        pickle.dump({'sample' : sample,
                      'center' : center,
                      'Threshold' : Threshold,
                      # 'partial_2_mean' : partial_2_mean,
                      # 'partial_2_std' : partial_2_std
            }, file)    
        

N = 64
n = 17
X = np.arange(N) - N/2 + 0.5
YY,XX = np.meshgrid(X,X)
x = np.arange(n) - n/2 + 0.5
yy,xx = np.meshgrid(x,x)

# Threshold = [0,1e-4,1e-3,1e-2,1e-1]

center = np.zeros((len(Threshold),TRIAL,AVG),dtype=float)
sample = np.linspace(-N/2,N/2,TRIAL)/4

if LOAD:

    with open('wfs-response64-16-moving-centre-turb-3.pkl','rb') as file:
        data = pickle.load(file)
        sample = data['sample']
        center = data['center']
        Threshold = data['Threshold']
    # mean = []
    # std = []
    # for i_threshold in range(len(Threshold)):
    #     mean.append(np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT)
    #     std.append(np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT)#)
        
    # full_05_mean = np.copy(mean)
    # full_05_std = np.copy(std)


else:

    tasks = [
        (i_threshold, trial, avg)
        for i_threshold in range(len(Threshold))
        for trial in range(TRIAL)
        for avg in range(AVG)
    ]
    
    results = Parallel(n_jobs=num_cores, backend="loky")(
    delayed(measure_one)(i_threshold, trial, avg, Threshold, sample, N, YY, XX, ps3, wfs_type, n_cog)
    for i_threshold, trial, avg in tqdm(tasks, desc="Processing", total=len(tasks))
    )

    for i_threshold, trial, avg, value in results:
        center[i_threshold, trial, avg] = value
        
CONVERT = 500/2

if PLOT:
    
    
    for i_threshold in range(len(Threshold)):
        
        plt.plot(sample*CONVERT, np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT,
                     #yerr=np.nanstd(center[i_threshold],axis=-1)*CONVERT/50**0.5,capsize=5,
                     label='Threshold = {:1.0e}'.format(Threshold[i_threshold]))
        # plt.fill_between(sample*CONVERT,
        #                   np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT - np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT,#,
        #                   np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT + np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT,#,
        #                   alpha=0.5)
        dif_part_turb[3,i_threshold] = (center[i_threshold].T - sample).T*CONVERT
    
    # — Plot 2 on the right —

    plt.xlim(-1000,1000)
    plt.ylim(-10,10)
    plt.xlabel('Gradient OPD (nm)')
    plt.ylabel('Measurement OPD error (nm)')
    plt.legend()
    # plt.suptitle('64x64 --> 16x16 - moving centre - 3 d/r0')
    # # Tight layout to avoid overlap
    # # plt.tight_layout()
    # plt.suptitle('partial-d/r0=3\nmean={}\n+-std={}'.format(np.nanmean((center - sample)*CONVERT,axis=(-1,-2)),
    #                                                         np.nanstd((center - sample)*CONVERT,axis=(-1,-2))))
    plt.savefig('wfs-response64-16-moving-centre-turb-3.pdf',bbox_inches='tight')
    plt.show()

# mean = []
# std = []
# for i_threshold in range(len(Threshold)):
#     mean.append(np.nanmean(center[i_threshold].T - sample,axis=0)*CONVERT)
#     std.append(np.nanstd(center[i_threshold].T - sample,axis=0)*CONVERT)#)
    
# partial_3_mean = np.copy(mean)
# partial_3_std = np.copy(std)



if SAVE:

    with open('wfs-response64-16-moving-centre-turb-3.pkl','wb') as file:
        pickle.dump({'sample' : sample,
                      'center' : center,
                      'Threshold' : Threshold,
                      # 'partial_3_mean' : partial_3_mean,
                      # 'partial_3_std' : partial_3_std
            }, file)
        
        
        

# dif_full_turb = np.zeros((4,len(Threshold),TRIAL,AVG),dtype=float)
# dif_part_turb = np.zeros((4,len(Threshold),TRIAL,AVG),dtype=float)
# dif_full_dif = np.zeros((len(Threshold),TRIAL),dtype=float)
# dif_part_dif = np.zeros((len(Threshold),TRIAL),dtype=float)
        
full_rms = np.array([np.nanstd(dif_full_dif,axis=(-1)),
                np.nanstd(np.nanmean(dif_full_turb[0],axis=-1),axis=-1),
                np.nanstd(np.nanmean(dif_full_turb[1],axis=-1),axis=-1),
                np.nanstd(np.nanmean(dif_full_turb[2],axis=-1),axis=-1),
                np.nanstd(np.nanmean(dif_full_turb[3],axis=-1),axis=-1)])

part_rms = np.array([np.nanstd(dif_part_dif,axis=(-1)),
                np.nanstd(np.nanmean(dif_part_turb[0],axis=-1),axis=-1),
                np.nanstd(np.nanmean(dif_part_turb[1],axis=-1),axis=-1),
                np.nanstd(np.nanmean(dif_part_turb[2],axis=-1),axis=-1),
                np.nanstd(np.nanmean(dif_part_turb[3],axis=-1),axis=-1)])

dor0_string = ['dif. lim.','0.5','1','2','3']

# for i_threshold in range(5):
#     plt.plot(dor0_string,full_rms[:,i_threshold],ls='-',color=color_list[i_threshold])
#     plt.plot(dor0_string,part_rms[:,i_threshold],ls='--',color=color_list[i_threshold])
# plt.yscale('log')
# plt.xlabel('$d/r_0$')
# plt.ylabel('RMS of mean of OPD error (nm)')
# legend_elements = [
#     Line2D([0], [0], color='black', linestyle='-', marker='', label='Full'),
#     Line2D([0], [0], color='black', linestyle='--', marker='', label='Partial'),
#     Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label='{:.0e}'.format(Threshold[0])),
#     Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label='{:.0e}'.format(Threshold[1])),
#     Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label='{:.0e}'.format(Threshold[2])),
#     Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label='{:.0e}'.format(Threshold[3])),
#     Line2D([0], [0], color=color_list[4], linestyle='-', marker='', label='{:.0e}'.format(Threshold[4])),
#     ]
# plt.legend(handles=legend_elements, loc='best')
# plt.savefig('bench-opd-rms-mean.pdf')
# plt.show()

# Threshold_string = ['0','1e-4','1e-3','1e-2','1e-1']

# for i in range(5):
#     plt.plot(Threshold_string,full_rms[i],ls='-',color=color_list[i])
#     plt.plot(Threshold_string,part_rms[i],ls='--',color=color_list[i])
# plt.yscale('log')
# plt.xlabel('Threshold')
# plt.ylabel('RMS of mean of OPD error (nm)')
# legend_elements = [
#     Line2D([0], [0], color='black', linestyle='-', marker='', label='Full'),
#     Line2D([0], [0], color='black', linestyle='--', marker='', label='Partial'),
#     Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label='{:}'.format(dor0_string[0])),
#     Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label='{:}'.format(dor0_string[1])),
#     Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label='{:}'.format(dor0_string[2])),
#     Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label='{:}'.format(dor0_string[3])),
#     Line2D([0], [0], color=color_list[4], linestyle='-', marker='', label='{:}'.format(dor0_string[4])),
#     ]
# plt.legend(handles=legend_elements, loc='best')
# plt.savefig('bench-opd-rms-mean-2.pdf')
# plt.show()

##############################################################################


# full_rms = np.array([np.nanstd(dif_full_dif,axis=(-1)),
#                 np.nanstd(dif_full_turb[0],axis=(-1,-2)),
#                 np.nanstd(dif_full_turb[1],axis=(-1,-2)),
#                 np.nanstd(dif_full_turb[2],axis=(-1,-2)),
#                 np.nanstd(dif_full_turb[3],axis=(-1,-2))])

# part_rms = np.array([np.nanstd(dif_part_dif,axis=(-1)),
#                 np.nanstd(dif_part_turb[0],axis=(-1,-2)),
#                 np.nanstd(dif_part_turb[1],axis=(-1,-2)),
#                 np.nanstd(dif_part_turb[2],axis=(-1,-2)),
#                 np.nanstd(dif_part_turb[3],axis=(-1,-2))])

# dor0_string = ['dif. lim.','0.5','1','2','3']

# for i_threshold in range(5):
#     plt.plot(dor0_string[1:],full_rms[1:,i_threshold],ls='-',color=color_list[i_threshold],marker='x')
#     plt.plot(dor0_string[1:],part_rms[1:,i_threshold],ls='--',color=color_list[i_threshold],marker='+')
# plt.plot(dor0_string[1:],part_rms[1:,0],ls='--',color=color_list[i_threshold],marker='+')
# plt.yscale('log')
# plt.xlabel('$d/r_0$')
# plt.ylabel('RMS of OPD error (nm)')
# legend_elements = [
#     Line2D([0], [0], color='black', linestyle='-', marker='x', label='Full'),
#     Line2D([0], [0], color='black', linestyle='--', marker='+', label='Partial'),
#     Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label='{:.0e}'.format(Threshold[0])),
#     Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label='{:.0e}'.format(Threshold[1])),
#     Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label='{:.0e}'.format(Threshold[2])),
#     Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label='{:.0e}'.format(Threshold[3])),
#     Line2D([0], [0], color=color_list[4], linestyle='-', marker='', label='{:.0e}'.format(Threshold[4])),
#     ]
# plt.legend(handles=legend_elements, loc='best')
# plt.savefig('bench-opd-rms.pdf')
# plt.show()

Threshold_string = ['0','1e-4','1e-3','1e-2','1e-1']

for i in range(5):
    if i == 0:
        continue
    plt.plot(Threshold_string,full_rms[i],ls='-',color=color_list[i])
    plt.plot(Threshold_string,part_rms[i],ls='--',color=color_list[i])
plt.yscale('log')
plt.xlabel('Threshold')
plt.ylabel('RMS OPD error (nm)')
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', marker='', label='Full'),
    Line2D([0], [0], color='black', linestyle='--', marker='', label='Partial'),
    # Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label='{:}'.format(dor0_string[0])),
    Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label='{:}'.format(dor0_string[1])),
    Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label='{:}'.format(dor0_string[2])),
    Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label='{:}'.format(dor0_string[3])),
    Line2D([0], [0], color=color_list[4], linestyle='-', marker='', label='{:}'.format(dor0_string[4])),
    ]
plt.legend(handles=legend_elements, loc='upper left')
plt.savefig('bench-opd-rms-2.pdf')
plt.show()

# partial_dif_lim_std = np.zeros_like(partial_dif_lim_mean)*np.nan
# full_dif_lim_std = np.zeros_like(full_dif_lim_mean)*np.nan

# old_partial_mean = np.array([partial_dif_lim_mean,partial_05_mean,partial_1_mean,partial_2_mean,partial_3_mean])
# old_partial_std = np.array([partial_dif_lim_std,partial_05_std,partial_1_std,partial_2_std,partial_3_std])
# old_full_mean = np.array([full_dif_lim_mean,full_05_mean,full_1_mean,full_2_mean,full_3_mean])
# old_full_std = np.array([partial_dif_lim_std,full_05_std,full_1_std,full_2_std,full_3_std])

# partial_mean, partial_std = np.average(old_partial_mean,weights=old_partial_std**-2,returned=True,axis=-1)
# full_mean, full_std = np.average(old_full_mean,weights=old_full_std**-2,returned=True,axis=-1)
# partial_std = partial_std**-0.5
# full_std = full_std**-0.5

# full_mean[0] = old_full_mean[0].mean(-1)
# partial_mean[0] = old_partial_mean[0].mean(-1)

# dor0_string = ['dif. lim.','0.5','1','2','3']
    

# for i_threshold in range(len(Threshold)):
#     plt.errorbar(dor0_string,(full_mean[:,i_threshold]),
#                   yerr=full_std[:,i_threshold],
#                   capsize=5,marker='o',ls='-',color=color_list[i_threshold])

# legend_elements = [
#     Line2D([0], [0], color='black', linestyle='-', marker='', label='Full'),
#     Line2D([0], [0], color='black', linestyle='--', marker='', label='Moving ROI'),
#     Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label='{:.0e}'.format(Threshold[0])),
#     Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label='{:.0e}'.format(Threshold[1])),
#     Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label='{:.0e}'.format(Threshold[2])),
#     Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label='{:.0e}'.format(Threshold[3])),
#     Line2D([0], [0], color=color_list[4], linestyle='-', marker='', label='{:.0e}'.format(Threshold[4])),
#     ]
# plt.legend(handles=legend_elements, loc='best')
# # plt.yscale('symlog')
# plt.xlabel('$d/r_0$')
# plt.ylabel('|mean(OPD difference)| (nm)')
# plt.ylim(-4,4)
# # plt.yscale('log')
# plt.show()


# for i_threshold in range(len(Threshold)):
#     plt.errorbar(dor0_string,
#                   (partial_mean[:,i_threshold]),
#                   yerr=partial_std[:,i_threshold],
#                   capsize=5,marker='^',ls='--',color=color_list[i_threshold])
    
# legend_elements = [
#     # Line2D([0], [0], color='black', linestyle='-', marker='o', label='Full'),
#     # Line2D([0], [0], color='black', linestyle='--', marker='^', label='Moving ROI'),
#     Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label='{:.0e}'.format(Threshold[0])),
#     Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label='{:.0e}'.format(Threshold[1])),
#     Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label='{:.0e}'.format(Threshold[2])),
#     Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label='{:.0e}'.format(Threshold[3])),
#     Line2D([0], [0], color=color_list[4], linestyle='-', marker='', label='{:.0e}'.format(Threshold[4])),
#     ]
# plt.legend(handles=legend_elements, loc='best')
# # plt.yscale('symlog')
# plt.ylim(-4,4)
# # plt.yscale('log')
# plt.xlabel('$d/r_0$')
# plt.ylabel('|mean(OPD difference)| (nm)')
# plt.show()

# for i_threshold in range(len(Threshold)):
#     plt.errorbar(dor0_string,
#                   full_std[:,i_threshold],
#                   yerr=full_std[:,i_threshold]/np.sqrt(2*TRIAL-2),
#                   capsize=5,marker='o',ls='-',color=color_list[i_threshold])

# legend_elements = [
#     # Line2D([0], [0], color='black', linestyle='-', marker='', label='Full'),
#     # Line2D([0], [0], color='black', linestyle='--', marker='', label='Moving ROI'),
#     Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label='{:.0e}'.format(Threshold[0])),
#     Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label='{:.0e}'.format(Threshold[1])),
#     Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label='{:.0e}'.format(Threshold[2])),
#     Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label='{:.0e}'.format(Threshold[3])),
#     Line2D([0], [0], color=color_list[4], linestyle='-', marker='', label='{:.0e}'.format(Threshold[4])),
#     ]
# plt.legend(handles=legend_elements, loc='best')
# # plt.yscale('symlog')
# plt.xlabel('$d/r_0$')
# plt.ylabel('RMS OPD difference (nm)')
# plt.yscale('log')
# plt.ylim(1e-2,10)
# plt.show()


# for i_threshold in range(len(Threshold)):
#     plt.errorbar(dor0_string,
#                  partial_std[:,i_threshold],
#                   yerr=partial_std[:,i_threshold]/np.sqrt(2*TRIAL-2),
#                   capsize=5,marker='^',ls='--',color=color_list[i_threshold])
    
# legend_elements = [
#     # Line2D([0], [0], color='black', linestyle='-', marker='o', label='Full'),
#     # Line2D([0], [0], color='black', linestyle='--', marker='^', label='Moving ROI'),
#     Line2D([0], [0], color=color_list[0], linestyle='-', marker='', label='{:.0e}'.format(Threshold[0])),
#     Line2D([0], [0], color=color_list[1], linestyle='-', marker='', label='{:.0e}'.format(Threshold[1])),
#     Line2D([0], [0], color=color_list[2], linestyle='-', marker='', label='{:.0e}'.format(Threshold[2])),
#     Line2D([0], [0], color=color_list[3], linestyle='-', marker='', label='{:.0e}'.format(Threshold[3])),
#     Line2D([0], [0], color=color_list[4], linestyle='-', marker='', label='{:.0e}'.format(Threshold[4])),
#     ]
# plt.legend(handles=legend_elements, loc='best')
# # plt.yscale('symlog')
# plt.xlabel('$d/r_0$')
# plt.ylabel('RMS OPD difference (nm)')
# plt.yscale('log')
# plt.ylim(1e-2,10)
# plt.show()




