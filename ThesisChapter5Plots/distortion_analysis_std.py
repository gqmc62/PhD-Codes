# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:54:01 2024

@author: gqmc62
"""
from collections import defaultdict

def weighted_y_by_x_groups(x_vals, y_vals, y_errors):
    x_rounded = np.round(x_vals, 3)

    grouped = defaultdict(list)
    for i, xr in enumerate(x_rounded):
        grouped[xr].append(i)

    x_unique = []
    y_weighted_means = []
    y_weighted_errors = []

    for xr, indices in sorted(grouped.items()):
        x_group = x_vals[indices]
        y_group = y_vals[indices]
        err_group = y_errors[indices]

        weights = 1.0 / (err_group ** 2)
        # y_mean = np.nansum(y_group * weights) / np.nansum(weights)
        y_mean = np.nanmean(y_group)# / np.nansum(weights)
        # y_err = np.sqrt(1.0 / np.nansum(weights)) # error of the weighted mean
        # sigma_w2 = np.nansum(weights * (y_group - y_mean)**2)/np.nansum(weights)
        # y_err = np.sqrt(sigma_w2)
        y_err = (np.nanmean(err_group**2))**0.5

        x_unique.append(np.nanmean(x_group))  # optional: take mean of the x group
        y_weighted_means.append(y_mean)
        y_weighted_errors.append(y_err)

    return np.array(x_unique), np.array(y_weighted_means), np.array(y_weighted_errors)

import matplotlib as mpl
from scipy.signal import deconvolve

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
# mpl.rcParams['savefig.bbox_inches'] = 'tight'
mpl.rcParams['savefig.format'] = 'pdf'



###############################################################################
L0 = 4

    
###############################################################################

TYPELIST=['A','B','C'] # 10,8,6
# TYPELIST = ['A']

HOME = r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter5Plots\LastChapter5Data\conf\FinalSupreme\conf\std'

for TYPE in TYPELIST:
    
    
    slodarRefDir = r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter5Plots\LastChapter5Data\conf'
    
    if (TYPE == 'A'):
        SIMNAME = 'bad_IM_A-'
        home = r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter5Plots\LastChapter5Data\conf\FinalSupreme\conf\Data\A' # bad_IM before fix
        # slopeCovName = 'slopeCovariance_A-vk.fits'
        refFuncName = 'slodarRefFunc4cmx10subapVK4mL0.fits'
    elif TYPE == 'B':
        SIMNAME = 'bad_IM_B-'
        home = r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter5Plots\LastChapter5Data\conf\FinalSupreme\conf\Data\B' #bad_IM_B
        # home = r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf\2025-03-24-19-32-05-B-final' #bad_IM_B
        # slopeCovName = 'slopeCovariance_B-vk.fits'
        refFuncName = 'slodarRefFunc5cmx8subapVK4mL0.fits'
    elif TYPE == 'C':
        SIMNAME = 'bad_IM_C-'
        home = r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter5Plots\LastChapter5Data\conf\FinalSupreme\conf\Data\C' #bad_IM_C
        # slopeCovName = 'slopeCovariance_C-vk.fits'
        refFuncName = 'slodarRefFunc6-6cmx6subapVK4mL0.fits'
        
    dsub_A = 0.4/10
    dsub_B = 0.4/8
    dsub_C = 0.4/6
    
###############################################################################
    
    from matplotlib import pyplot as plt
    from matplotlib import colors
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
    
    # def old_angle_of_arrival(wvl,D,r0):
    #     alpha = np.sqrt(0.182 * wvl**2 * D**(-1./3.) * r0**(-5./3.))
    #     return alpha
    
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
    
   
    def clean_data(A):
        A[np.isinf(A)] = 0
        A[np.isnan(A)] = 0
        return A
    
  
    def get_PSD(phase, delta, pupil=None):
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
    
    
        """
        dim = phase.ndim
        if pupil is not None:
            if dim == 2:
                pupil = pupil
                phase = phase*pupil
                N = len(phase)
                del_f = 1./(N*2.*delta) # 2N because we pad!
                phase = phase - np.nanmean(phase)
                phase = phase*pupil
                PU = my_ft2(mirror_pad2(phase), 1./N/2.)
                P = my_ft2(mirror_pad2(pupil), 1./N/2.)
                cn = PU/(P + 1e-12 + 1j*1e-12)
                # cn = clean_data(cn)
                plt.imshow(np.abs(cn)**2)
                plt.show()
                PSD = (np.abs(cn)**2/del_f**2)*2.   # * 2. is from the fact that
                                                    # the input phase is not complex
                                                    # phase are complex when generated
                                                    # from cn.
                PSD[int(N/2),int(N/2)] = 0.         # PSD is 0 at the centre.
                # del_f now is 2*del_f ==> 1/N/delta
                return pick_even_index(PSD)/4
        else:
            if dim == 2:
                N = len(phase)
                del_f = 1./(N*2.*delta) # 2N because we pad!
                phase = phase - np.nanmean(phase)
                cn = my_ft2(mirror_pad2(phase), 1./N/2.)
                PSD = (np.abs(cn)**2/del_f**2)*2.   # * 2. is from the fact that
                                                    # the input phase is not complex
                                                    # phase are complex when generated
                                                    # from cn.
                PSD[int(N/2),int(N/2)] = 0.         # PSD is 0 at the centre.
                # del_f now is 2*del_f ==> 1/N/delta
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
        return np.fft.fftshift(np.fft.fft2((g))) * delta**2
    
    def my_ift2(G, delta_f):
        N = G.shape[-1]
        # delta = 1/N/delta_f
        return np.fft.ifft2(np.fft.ifftshift(G)) * (N * delta_f)**2
    
    
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
        # SHWFS_position = position
        # ACT_list = ACT_LIST
        # ACT_position = DM_POS_TEMP
        # threshold = as_threshold
        # IM = np.copy(IM[cases,trial])
        debug = False
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
            
            
            
            WFS_MAP = (WFS_MAP_X**2 + WFS_MAP_Y**2)**0.5
            
            WFS_MAP_X /= np.nanmax(WFS_MAP)
            WFS_MAP_Y /= np.nanmax(WFS_MAP)
            WFS_MAP /= np.nanmax(WFS_MAP)
            
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
                            
                        size = ((WFS_MAP_X[tuple(MAX4_POS[i])]**2
                                + WFS_MAP_Y[tuple(MAX4_POS[i])]**2)**0.5
                               * (WFS_MAP_X[tuple(MAX4_POS[j])]**2
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
    
    
    
    as_threshold = 0.25#0.25
    
    
    os.chdir(home)
    
    
    f = open('name_space.txt' ,'r')
    name_list = np.asarray(f.readlines())
    
    CASES = len(name_list)
    
    for i in np.arange(CASES):
        name_list[i] = name_list[i].strip()
    
    expect_rytov = np.zeros((CASES),dtype=float)
    expect_AS = np.copy(expect_rytov)
    
    z = np.copy(expect_rytov)
    r0HA = np.copy(expect_rytov)
    label_list = []
    
    for i in np.arange(expect_rytov.shape[0]):
        expect_rytov[i] = float(name_list[i][5:9])
        expect_AS[i] = float(name_list[i][12:-1])/100.
    
        with open(SIMNAME + name_list[i] + '.yaml','r') as file:
            config = yaml.safe_load(file)
            z[i] = config['DM'][2]['altitude']
            r0HA[i] = config['DM'][2]['r0']
    
        label_list.append('r0:{:.0f}cm, z:{:.3f}km'.format(r0HA[i],z[i]))
    
    wvl = float(config['Science'][0]['wavelength'])
    dsub = config['Telescope']['telDiam']/config['WFS'][0]['nxSubaps']
    D = config['Telescope']['telDiam']
    # pixel_scale = dsub / config['WFS'][0]['pxlsPerSubap']
    
    L0 = config['Atmosphere']['L0'][0]
    
    f = open('data_space.txt' ,'r')
    data_type_list = np.asarray(f.readlines())
    DATATYPES = len(data_type_list)
    for i in np.arange(DATATYPES):
        data_type_list[i] = data_type_list[i].strip()
    
    
    for datatypes in np.arange(DATATYPES):
        for cases in np.arange(CASES):
            to_read = data_type_list[datatypes] + '_'
            # print(to_read)
            if os.path.isfile(to_read + SIMNAME + name_list[cases] + '.fits'):
                hdul = fits.open(to_read + SIMNAME + name_list[cases] + '.fits')
                if cases == 0:
                    SHAPE = hdul[0].data.shape
                    # print(SHAPE)
                    new_SHAPE = tuple(np.append([CASES],np.asarray(SHAPE)))
                    # print(new_SHAPE)
                    globals()[data_type_list[datatypes]] = np.zeros((new_SHAPE),dtype=float)
                globals()[data_type_list[datatypes]][cases] = hdul[0].data
    
    expect_AS = (0.162*wvl**2*(dsub**(-1/3)-D**(-1/3))*r0HA**(-5/3))**0.5*z/dsub
    expect_rytov = rytov_number(wvl, r0HA, z)
    
    TRIAL = IM.shape[1]
    old_expect_AS = np.copy(expect_AS)
    
    # r0 = 0.05
    r0 = config['Atmosphere']['r0']
    
    pixel_scale = config['Telescope']['telDiam']/RES.shape[-1]
    
    try:
        MEASURED_IM_SUBAP_SHIFT = np.copy(MEASURED_IM_SUBAP_SHIFT_influence)
    except:
        pass
    
    MEASURED_IM_SUBAP_SHIFT = MEASURED_IM_SUBAP_SHIFT[:,:,:,0,:] # had wrong dimension, that they ended up having repeated values!!!
    

    
    
    ###############################################################################
    # another method of measuring as
    ###############################################################################
    
    if config['WFS'][0]['nxSubaps'] == 8: #8x8
        position = np.array([[0, 3],[0, 4],
                             [1, 1],[1, 2],[1, 3],[1, 4],[1, 5],[1, 6],
                             [2, 1],[2, 2],[2, 3],[2, 4],[2, 5],[2, 6],
                             [3, 0],[3, 1],[3, 2],[3, 3],[3, 4],[3, 5],[3, 6],[3, 7],
                             [4, 0],[4, 1],[4, 2],[4, 3],[4, 4],[4, 5],[4, 6],[4, 7],
                             [5, 1],[5, 2],[5, 3],[5, 4],[5, 5],[5, 6],
                             [6, 1],[6, 2],[6, 3],[6, 4],[6, 5],[6, 6],
                             [7, 3],[7, 4]], dtype=int) # wfs
        
        # ACT_LIST = np.array([24,
        #                      32,33,34,
        #                      40,41,42,43,44,
        #                      50,51,52,
        #                      60]) # dm act number
        
        ACT_LIST = np.array([13,
                             20,21,22,23,24,
                             29,30,31,32,33,
                             37,38,39,40,41,42,43,
                             47,48,49,50,51,
                             56,57,58,59,60,
                             67]) + 2
        
        DM_POS_TEMP = np.array([[0.5,3.5],
                                [1.5,1.5],[1.5,2.5],[1.5,3.5],[1.5,4.5],[1.5,5.5],
                                [2.5,1.5],[2.5,2.5],[2.5,3.5],[2.5,4.5],[2.5,5.5],
                                [3.5,0.5],[3.5,1.5],[3.5,2.5],[3.5,3.5],[3.5,4.5],[3.5,5.5],[3.5,6.5],
                                [4.5,1.5],[4.5,2.5],[4.5,3.5],[4.5,4.5],[4.5,5.5],
                                [5.5,1.5],[5.5,2.5],[5.5,3.5],[5.5,4.5],[5.5,5.5],
                                [6.5,3.5]])
    
    elif config['WFS'][0]['nxSubaps'] == 10:
        position = np.array([[0., 4.],[0., 5.],
                                 [1., 2.],[1., 3.],[1., 4.],[1., 5.],[1., 6.],[1., 7.],
                                 [2., 1.],[2., 2.],[2., 3.],[2., 4.],[2., 5.],[2., 6.],[2., 7.],[2., 8.],
                                 [3., 1.],[3., 2.],[3., 3.],[3., 4.],[3., 5.],[3., 6.],[3., 7.],[3., 8.],
                                 [4., 0.],[4., 1.],[4., 2.],[4., 3.],[4., 4.],[4., 5.],[4., 6.],[4., 7.],[4., 8.],[4., 9.],
                                 [5., 0.],[5., 1.],[5., 2.],[5., 3.],[5., 4.],[5., 5.],[5., 6.],[5., 7.],[5., 8.],[5., 9.],
                                 [6., 1.],[6., 2.],[6., 3.],[6., 4.],[6., 5.],[6., 6.],[6., 7.],[6., 8.],
                                 [7., 1.],[7., 2.],[7., 3.],[7., 4.],[7., 5.],[7., 6.],[7., 7.],[7., 8.],
                                 [8., 2.],[8., 3.],[8., 4.],[8., 5.],[8., 6.],[8., 7.],
                                 [9., 4.],[9., 5.]],dtype=int)
        ACT_LIST = np.array([17,
                                26,27,28,29,30,
                                36,37,38,39,40,41,42,
                                47,48,49,50,51,52,53,
                                57,58,59,60,61,62,63,64,65,
                                69,70,71,72,73,74,75,
                                80,81,82,83,84,85,86,
                                92,93,94,95,96,
                                116]) - 1 + 2
        DM_POS_TEMP = np.array([[0.5,4.5],
                                   [1.5,2.5],[1.5,3.5],[1.5,4.5],[1.5,5.5],[1.5,6.5],
                                   [2.5,1.5],[2.5,2.5],[2.5,3.5],[2.5,4.5],[2.5,5.5],[2.5,6.5],[2.5,7.5],
                                   [3.5,1.5],[3.5,2.5],[3.5,3.5],[3.5,4.5],[3.5,5.5],[3.5,6.5],[3.5,7.5],
                                   [4.5,0.5],[4.5,1.5],[4.5,2.5],[4.5,3.5],[4.5,4.5],[4.5,5.5],[4.5,6.5],[4.5,7.5],[4.5,8.5],
                                   [5.5,1.5],[5.5,2.5],[5.5,3.5],[5.5,4.5],[5.5,5.5],[5.5,6.5],[5.5,7.5],
                                   [6.5,1.5],[6.5,2.5],[6.5,3.5],[6.5,4.5],[6.5,5.5],[6.5,6.5],[6.5,7.5],
                                   [7.5,2.5],[7.5,3.5],[7.5,4.5],[7.5,5.5],[7.5,6.5],
                                   [8.5,4.5]])
        
    elif config['WFS'][0]['nxSubaps'] == 6:
        position = np.array([[0, 2],[0, 3], # 6x6 subx
                              [1, 1],[1, 2],[1, 3],[1, 4],
                              [2, 0],[2, 1],[2, 2],[2, 3],[2, 4],[2, 5],
                              [3, 0],[3, 1],[3, 2],[3, 3],[3, 4],[3, 5],
                              [4, 1],[4, 2],[4, 3],[4, 4],
                              [5, 2],[5, 3]], dtype=int)
        
        ACT_LIST = np.array([10,
                              16,17,18,
                              22,23,24,25,26,
                              30,31,32,
                              38],dtype=int) + 2
        
        DM_POS_TEMP = np.array([[0.5,2.5],
                                [1.5,1.5],[1.5,2.5],[1.5,3.5],
                                [2.5,0.5],[2.5,1.5],[2.5,2.5],[2.5,3.5],[2.5,4.5],
                                [3.5,1.5],[3.5,2.5],[3.5,3.5],
                                [4.5,2.5]])
    elif config['WFS'][0]['nxSubaps'] == 12:
        position = np.array([[0,5],[0,6], # 12x12 subx
                              [1,3],[1,4],[1,5],[1,6],[1,7],[1,8],
                              [2,2],[2,3],[2,4],[2,5],[2,6],[2,7],[2,8],[2,9],
                              [3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10],
                              [4,1],[4,2],[4,3],[4,4],[4,5],[4,6],[4,7],[4,8],[4,9],[4,10],
                              [5,0],[5,1],[5,2],[5,3],[5,4],[5,5],[5,6],[5,7],[5,8],[5,9],[5,10],[5,11],
                              [6,0],[6,1],[6,2],[6,3],[6,4],[6,5],[6,6],[6,7],[6,8],[6,9],[6,10],[6,11],
                              [7,1],[7,2],[7,3],[7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],
                              [8,1],[8,2],[8,3],[8,4],[8,5],[8,6],[8,7],[8,8],[8,9],[4,10],
                              [9,2],[9,3],[9,4],[9,5],[9,6],[9,7],[9,8],[9,9],
                              [10,3],[10,4],[10,5],[10,6],[10,7],[10,8],
                              [11,5],[11,6]],dtype=int)
        ACT_LIST = np.array([ 32,
                              43, 44, 45, 46, 47,
                              55, 56, 57, 58, 59, 60, 61,
                              68, 69, 70, 71, 72, 73, 74,
                              80, 81, 82, 83, 84, 85, 86, 87, 88,
                              94, 95, 96, 97, 98, 99,100,
                              107,108,109,110,111,112,113,
                              121,122,123,124,125,
                              136],dtype=int) + 2
        DM_POS_TEMP = np.array([[1.5,5.5],
                                [2.5,3.5],[2.5,4.5],[2.5,5.5],[2.5,6.5],[2.5,7.5],
                                [3.5,2.5],[3.5,3.5],[3.5,4.5],[3.5,5.5],[3.5,6.5],[3.5,7.5],[3.5,8.5],
                                [4.5,2.5],[4.5,3.5],[4.5,4.5],[4.5,5.5],[4.5,6.5],[4.5,7.5],[4.5,8.5],
                                [5.5,1.5],[5.5,2.5],[5.5,3.5],[5.5,4.5],[5.5,5.5],[5.5,6.5],[5.5,7.5],[5.5,8.5],[5.5,9.5],
                                [6.5,2.5],[6.5,3.5],[6.5,4.5],[6.5,5.5],[6.5,6.5],[6.5,7.5],[6.5,8.5],
                                [7.5,2.5],[7.5,3.5],[7.5,4.5],[7.5,5.5],[7.5,6.5],[7.5,7.5],[7.5,8.5],
                                [8.5,3.5],[8.5,4.5],[8.5,5.5],[8.5,6.5],[8.5,7.5],
                                [9.5,5.5]])
    
    elif config['WFS'][0]['nxSubaps'] == 4: # 4x4
        position = np.array([[0, 1],[0, 2],
                              [1, 0],[1, 1],[1, 2],[1, 3],
                              [2, 0],[2, 1],[2, 2],[2, 3],
                              [3, 1],[3, 2]], dtype=int) # wfs
        
        ACT_LIST = np.array([7,
                              11,12,13,
                              17]) + 2 # dm act number
        
        DM_POS_TEMP = np.array([[0.5,1.5],
                                [1.5,0.5],[1.5,1.5],[1.5,2.5],
                                [2.5,1.5]])
    
    
   
    
    
    ###############################################################################
    
    
    ANOTHER_AS = np.zeros((CASES,TRIAL,IM.shape[2],2),dtype=float)
    
    for cases in np.arange(CASES):
        for trial in np.arange(TRIAL):
            ANOTHER_AS[cases,trial] = (
                measure_AS_from_IM(np.copy(IM[cases,trial]),position,ACT_LIST,DM_POS_TEMP,as_threshold))
            
            
    # ###############################################################################
  
    # ###############################################################################
    # # Parameter error
    # ###############################################################################
    
   
    
    old_MEASURED_IM_SUBAP_SHIFT = np.copy(MEASURED_IM_SUBAP_SHIFT)
    old_COMBINED_MEASURED_IM_AS = np.nanmean(np.nanstd(MEASURED_IM_SUBAP_SHIFT,axis=(-1,-2)),axis=-1)
    old_COMBINED_MEASURED_IM_AS_error = np.nanstd(np.nanstd(MEASURED_IM_SUBAP_SHIFT,axis=(-1,-2)),axis=-1)
    
    
    
    
    COMBINED_MEASURED_IM_RYTOV = np.nanmean(np.nanmean(MEASURED_IM_RYTOV,axis=(-1,-2)),axis=-1)
    error_rytov = np.nanstd(np.nanmean(MEASURED_IM_RYTOV,axis=(-1,-2)),axis=-1)
    
    
    COMBINED_MEASURED_IM_AS = np.nanmean(np.nanstd(ANOTHER_AS[:,:,2:],axis=(-1,-2)),axis=-1)
    error_as = np.nanstd(np.nanstd(ANOTHER_AS[:,:,2:],axis=(-1,-2)),axis=-1)
    
   
    
   
    NOLL = [1.0299,0.582,0.134,0.111,0.0880,0.0648,0.0587,0.0525,0.0463,0.0401,0.0377,0.0352,0.0328,0.0304,0.0279,0.0267,0.0255,0.0243,0.0232,0.0220,0.0208]
    
    
    import os
    
    os.chdir(slodarRefDir)
    from astropy.io import fits
    from matplotlib import pyplot as plt
    import numpy as np
    
    wvlref = 500e-9
    k = 2*np.pi/wvlref
    # dsub = 0.05
    r0_ref = dsub
    
    
    hdul = fits.open(refFuncName)
    data2 = hdul[0].data
    hdul.close()
    
    mid = (data2.shape[1] - 1) // 2 
    B = (data2[0,mid,mid]) * (r0_ref)**(5.0/3.0)*dsub**(1.0/3.0)*wvlref**-2 / 206265**2
    
    
    
    new_coeff = B              #0.025501816814832663# from butterly's refFunc mid 0th layer either L or T
    
    print(B,(1-(D/dsub)**(-1.0/3.0))*0.162)
    
    # expect_AS = (new_coeff * wvl**2 * dsub**(-1/3) * r0HA**(-5/3))**0.5 * z / dsub
    expect_AS = ((1-(D/dsub)**(-1.0/3.0))*0.162 * wvl**2 * dsub**(-1.0/3.0) * r0HA**(-5.0/3.0))**0.5 * z / dsub
    
   
    os.chdir(HOME)
    
    
    ###############################################################################
    # Fixed Sampling point actual
    ###############################################################################
    
   
    
    name_list = []
    for cases in np.arange(CASES):
        name_list.append('{:.2f}'.format(expect_rytov[cases]) + '$\sigma^2_{R,DM}$' 
                         + ' - {:.0f}%'.format(expect_AS[cases]*100) + '$RMS_{AS}$')
    
   
    
    ###############################################################################
    # IM error
    ###############################################################################
    
    TRIAL = IM.shape[1]
    nACT = IM.shape[2]
    nSUBAP = IM.shape[3]
    
    MAX =  perfect_IM[:,2:].max()
    
    perfect_IM /= MAX
    IM /= MAX#IM.max()
    
    threshold = 1/4. # selecting only the major strokes.
    
    MASK = (np.abs(perfect_IM[0]) > threshold)
    MASK[:2] = 0
    
    nMAINACT = (MASK == True).sum()
    DIF = np.zeros((CASES,TRIAL,nMAINACT),dtype=float)
    
    for cases in np.arange(CASES):
        for trial in np.arange(TRIAL):
            DIF[cases,trial] = ((perfect_IM[cases] - IM[cases,trial])/perfect_IM[cases])[MASK]
            
    DIF = np.asarray(DIF)
    IM_RMS = np.nanmean(np.nanmean(np.sqrt(DIF**2),axis=-1),axis=-1)
    IM_RMS_error = np.nanstd(np.nanmean(np.sqrt(DIF**2),axis=-1),axis=-1)
    
    
    grid_rytov = np.unique(np.round(expect_rytov,decimals=3))
    grid_AS = np.unique(np.round(expect_AS,decimals=3))*100
    perfect_grid_IM_RMS = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)
    
    for cases in np.arange(CASES):
        perfect_grid_IM_RMS[np.where(np.round(expect_rytov[cases],decimals=3)==grid_rytov),
                            np.where(np.round(expect_AS[cases],decimals=3)*100==grid_AS)] = IM_RMS[cases]
        
    perfect_grid_IM_RMS[perfect_grid_IM_RMS==0] = np.nan
    
    
    ###############################################################################
    # CC error
    ###############################################################################
    
    
    MAX = perfect_CC.max()
    CC /= MAX
    perfect_CC /= MAX
    
    threshold = 1/4. # selecting only the major strokes.
    
    MASK = (np.abs(perfect_CC[0]) > threshold)
    nMAINACT = (MASK == True).sum()
    
    CC_RMS = np.zeros((CASES),dtype=float)
    DIF = np.zeros((CASES,TRIAL,nMAINACT),dtype=float)
    for cases in np.arange(CASES):
        for trial in np.arange(TRIAL):
            DIF[cases,trial] = ((perfect_CC[cases] - CC[cases,trial])/perfect_CC[cases])[MASK]
    # CC_RMS = (np.nanmean(DIF**2,axis=(-1,-2)))**0.5
    # CC_RMS_error = CC_RMS/np.sqrt(2*DIF.shape[1]*DIF.shape[2] - 2)
    CC_RMS = np.nanmean(np.nanmean(np.sqrt(DIF**2),axis=-1),axis=-1)
    CC_RMS_error = np.nanstd(np.nanmean(np.sqrt(DIF**2),axis=-1),axis=-1)
    
    # grid_rytov = np.unique(expect_rytov)
    # grid_AS = np.unique(expect_AS)
    perfect_grid_CC_RMS = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)
    
    for cases in np.arange(CASES):
        perfect_grid_CC_RMS[np.where(np.round(expect_rytov[cases],decimals=3)==grid_rytov),
                            np.where(np.round(expect_AS[cases],decimals=3)*100==grid_AS)] = CC_RMS[cases]
        
    perfect_grid_CC_RMS[perfect_grid_CC_RMS==0] = np.nan
    
   
    
    ###############################################################################
    # only control error
    ###############################################################################
    
    # !!! calculate control error
    
    
    
    avg_terminal_WFE_only_control = (np.nanmean((WFE[:,:,-1]**2 - perfect_WFE[:,:,-1]**2)**0.5,axis=-1))
    avg_terminal_WFE_only_control_error = (np.nanstd((WFE[:,:,-1]**2 - perfect_WFE[:,:,-1]**2)**0.5,axis=-1))
    grid_WFE = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)
    grid_WFE[:] = np.nan
    
    for cases in np.arange(CASES):
        grid_WFE[np.where(np.round(expect_rytov[cases],decimals=3)==grid_rytov),
                            np.where(np.round(expect_AS[cases],decimals=3)*100==grid_AS)] = avg_terminal_WFE_only_control[cases]
        
    
    
    CS2 = plt.contourf(grid_rytov,grid_AS,grid_WFE.T)
    # plt.colorbar()
    CS = plt.contour(grid_rytov,grid_AS,grid_WFE.T,colors='k')
    plt.xlabel('$\sigma^2_{R,DM}$')
    plt.ylabel('$RMS_{AS}$ (%)')
    plt.clabel(CS,fontsize=fontsize)
    plt.colorbar(CS2)
    
    plt.xlim(0.1,grid_rytov.max())
    plt.ylim(10,grid_AS.max())
    plt.savefig(TYPE + 'control_WFE.pdf')
    # plt.axis('square')
    plt.show()
    
    
    ###############################################################################
    # z modes
    ###############################################################################
    
    LOOP = perfect_RES_phase_Z.shape[2]
    MODE = perfect_RES_phase_Z.shape[3]
    # radial_modes = [( 0,),
    #                 ( 1, 2),
    #                 ( 3, 4, 5,),
    #                 ( 6, 7, 8, 9,),
    #                 (10,11,12,13,14,),
    #                 (15,16,17,18,19,20,),
    #                 (21,22,23,24,25,26,27,),
    #                 (28,29,30,31,32,33,34,35,),
    #                 (36,37,38,39,40,41,42,43,44,)]
    
    x = []
    y = np.arange(MODE)
    number_of_radial = int(((1+8*MODE)**0.5-1)/2)
    for i in np.arange(number_of_radial):
        x.append(tuple(y[int((i)*(i+1)//2):int((i+1)*(i+2)//2)]))
    
    radial_modes = x
    
    RADIAL_MODES = len(radial_modes) - 1
    
    
    
    theory_noll = mps.von_Karman_noll_variance(MODE, D/pixel_scale, pixel_scale, r0, L0)
    MAX = np.max(np.array([theory_noll.max(),
                           np.nanmean(ATM_Z,axis=1).max(),
                           np.nanmean(perfect_RES_phase_Z,axis=1).max(),
                           np.nanmean(RES_phase_Z,axis=1).max()]))*10
    MAX = np.min(np.array([theory_noll.min(),
                           np.nanmean(ATM_Z,axis=1).min(),
                           np.nanmean(perfect_RES_phase_Z,axis=1).min(),
                           np.nanmean(RES_phase_Z,axis=1).min()]))/10
    
   
    color_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple',
                  'tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan',
                  'tab:blue','tab:orange','tab:green','tab:red','tab:purple',
                  'tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan',
                  'tab:blue','tab:orange','tab:green','tab:red','tab:purple',
                  'tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    arg = [16,9,14,7,12,17,1,5,10,15,3,8,13,18,2,6,11,4,0]
    
    fig = plt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    plt.plot((np.arange(MODE))[3:], (np.var(ATM_Z,axis=(0,1)))[3:],color=color_list[0],label='No AO')
    plt.fill_between((np.arange(MODE))[3:],
                  (np.var(ATM_Z,axis=(0,1)))[3:] + 2*(np.var(ATM_Z,axis=(0,1)))[3:].T/np.sqrt(2*ATM_Z.shape[0]*ATM_Z.shape[1] - 2),
                  (np.var(ATM_Z,axis=(0,1)))[3:] - 2*(np.var(ATM_Z,axis=(0,1)))[3:].T/np.sqrt(2*ATM_Z.shape[0]*ATM_Z.shape[1] - 2),
                  color=color_list[0],
                  alpha=0.3)
    
    plt.plot(np.arange(MODE - 1-2) + 1+2,
                  np.var(perfect_RES_phase_Z[:,:,-1,1+2:],axis=(0,1),ddof=1).T,
                  color=color_list[1],
                  label='No control error',marker='.')
    plt.fill_between(np.arange(MODE - 1-2) + 1+2,
                  np.var(perfect_RES_phase_Z[:,:,-1,1+2:],axis=(0,1),ddof=1).T + 2*np.var(perfect_RES_phase_Z[:,:,-1,1+2:],axis=(0,1),ddof=1).T/np.sqrt(2*ATM_Z.shape[0]*ATM_Z.shape[1] - 2),
                  np.var(perfect_RES_phase_Z[:,:,-1,1+2:],axis=(0,1),ddof=1).T - 2*np.var(perfect_RES_phase_Z[:,:,-1,1+2:],axis=(0,1),ddof=1).T/np.sqrt(2*ATM_Z.shape[0]*ATM_Z.shape[1] - 2),
                  color=color_list[1],
                  alpha=0.3)
    count = 2
    
    for cases in np.arange(CASES)[arg][::-1]:
        # if (cases != [9,16,21]).all():
        # if (cases != [7,9,14,16,19,21]).all():
        if (cases != [7,16]).all():
            continue
        
        plt.plot(np.arange(MODE - 1-2) + 1+2,
                      np.var(RES_phase_Z[cases,:,-1,1+2:],axis=(0),ddof=1).T,
                      color=color_list[count],
                      label=name_list[cases],#'unmatch control',
                      marker='.')
        plt.fill_between(np.arange(MODE - 1-2) + 1+2,
                      np.var(RES_phase_Z[cases,:,-1,1+2:],axis=(0),ddof=1).T+2*np.var(RES_phase_Z[cases,:,-1,1+2:],axis=(0),ddof=1).T/np.sqrt(2*ATM_Z.shape[1] - 2),
                      np.var(RES_phase_Z[cases,:,-1,1+2:],axis=(0),ddof=1).T-2*np.var(RES_phase_Z[cases,:,-1,1+2:],axis=(0),ddof=1).T/np.sqrt(2*ATM_Z.shape[1] - 2),
                      color=color_list[count],
                      alpha=0.3)
        count += 1
        
    plt.yscale('log')
    # plt.title('terminal z-decomposition')
    # plt.legend()
    # plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    # plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.66, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('Variance of Z coefficients (rad$^2$)')
    plt.xlabel('Zernike Modes')
    # plt.title('Terminal Wavefront Residual Zernike Mode Decomposition\n' + name_list[cases])
    plt.savefig(TYPE + 'z_decompose-selected.pdf', bbox_inches='tight',dpi=600)#_{:d}.png'.format(cases))
    plt.show()
    
    arg = [16,9,14,7,12,17,1,5,10,15,3,8,13,18,2,6,11,4,0]
    arg2 = [16,9,14,7,1,12,5,10,3,17,15,2,8,13,6,11,4,18,0]
    
    
    
    # Get x-axis: Zernike modes 1 to MODE-1
    zernike_modes = np.arange(MODE - 1) + 1
    
    # Compute distorted and perfect variances
    distorted_var = np.var(RES_phase_Z[arg2, :, -1, 1+2:], axis=1, ddof=1)
    distorted_var_std = 2*distorted_var/np.sqrt(np.sum(~np.isnan(RES_phase_Z[arg2, :, -1, 1+2:]),axis=1))
    perfect_var = np.var(perfect_RES_phase_Z[:, :, -1, 1+2:], axis=1, ddof=1)
    perfect_var_std = 2*perfect_var/np.sqrt(np.sum(~np.isnan(perfect_RES_phase_Z[arg2, :, -1, 1+2:]),axis=1))
    
    # Ratio of variances
    var_ratio = distorted_var / perfect_var
    var_ratio_std = var_ratio * (distorted_var_std**2/distorted_var**2 + perfect_var_std**2/perfect_var**2)**0.5
    
    # Group by unique Rytov values
    rytov_values = expect_rytov[arg2]
    unique_rytov, inverse_indices = np.unique(np.round(rytov_values,decimals=3), return_inverse=True)
    
    # Average ratio rows for each unique rytov
    ratio_matrix = []
    error_matrix = []
    
    for i in range(len(unique_rytov)):
        matching_rows = var_ratio[inverse_indices == i]
        matching_rows_std = var_ratio_std[inverse_indices == i]
        ratio_matrix.append(np.mean(matching_rows, axis=0))
        error_matrix.append(np.mean(matching_rows_std**2, axis=0)**0.5)
    
    ratio_matrix = np.array(ratio_matrix)[:,:3]
    error_matrix = np.array(error_matrix)[:,:3]
    
    
    z_ratio_rytov = np.copy(ratio_matrix.mean(-1))
    z_ratio_rytov_error = np.copy(error_matrix.mean(-1))
    
    # Define X-axis: Zernike modes
    zernike_modes = np.arange(MODE - 1) + 1
    
    # Compute variances
    distorted_var = np.var(RES_phase_Z[arg, :, -1, 1+2:], axis=1, ddof=1)
    perfect_var = np.var(perfect_RES_phase_Z[:, :, -1, 1+2:], axis=1, ddof=1)
    var_ratio = distorted_var / perfect_var
    
    # AS values (as %)
    as_values = 100 * expect_AS[arg]
    unique_as, inverse_indices = np.unique(np.round(as_values,decimals=3), return_inverse=True)
    
    # Group and average var_ratios by unique AS
    ratio_matrix = []
    error_matrix = []
    
    for i in range(len(unique_as)):
        matching_rows = var_ratio[inverse_indices == i]
        matching_rows_std = var_ratio_std[inverse_indices == i]
        ratio_matrix.append(np.mean(matching_rows, axis=0))
        error_matrix.append(np.std(matching_rows_std**2, axis=0)**0.5)
    
    ratio_matrix = np.array(ratio_matrix)[:,:3]
    error_matrix = np.array(error_matrix)[:,:3]
   

    z_ratio_as = np.copy(ratio_matrix.mean(-1))
    z_ratio_as_error = np.copy(error_matrix.mean(-1))
    
    # remesure psd because inside the main code i forgot to unwrap the phase :p & i want to measure before ao as well
    
    PSD_phase_frequency_new = PSD_phase_frequency
    theory_psd = mps.psd_function(r0,(PSD_phase_frequency_new.shape[1] + 1)*2,pixel_scale,L0,1e-10)
    cross_section_theory_psd = theory_psd[theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ]
    
    
    
    PSD_phase_new = PSD_phase
    perfect_PSD_phase_new = perfect_PSD_phase
                
  
        
    
            
    ###################################################################
    
    color_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple',
                  'tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan',
                  'tab:blue','tab:orange','tab:green','tab:red','tab:purple',
                  'tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan',
                  'tab:blue','tab:orange','tab:green','tab:red','tab:purple',
                  'tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    # plt.figure(figsize=(3,3))
    arg = [16,9,14,7,12,17,1,5,10,15,3,8,13,18,2,6,11,4,0]
    MAX = np.max(np.array([cross_section_theory_psd.max(),np.nanmean(perfect_PSD_phase_new,axis=1).max(),np.nanmean(PSD_phase_new,axis=1).max()]))*10
    MIN = np.min(np.array([cross_section_theory_psd.min(),np.nanmean(perfect_PSD_phase_new,axis=1).min(),np.nanmean(PSD_phase_new,axis=1).min()]))/10
    
    fig = plt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    # plt.plot(PSD_phase_frequency_new[cases],cross_section_theory_psd,label='Turbulence',ls=':',color=color_list[0])
    plt.plot(PSD_phase_frequency_new[cases],np.mean(ATM_PSD_phase,axis=(0,1)),label='No AO',color=color_list[0])
    plt.fill_between(PSD_phase_frequency_new[cases],
                 np.mean(ATM_PSD_phase,axis=(0,1)) + np.std(ATM_PSD_phase,axis=(0,1))/np.sqrt(perfect_PSD_phase_new.shape[1]),
                 np.mean(ATM_PSD_phase,axis=(0,1)) - np.std(ATM_PSD_phase,axis=(0,1))/np.sqrt(perfect_PSD_phase_new.shape[1]),
                 color=color_list[0],
                 alpha=0.3)
    
    plt.plot(PSD_phase_frequency_new[cases],
                 np.nanmean(perfect_PSD_phase_new[cases,:,-1],axis=0),
                 label='No control error',
                 color=color_list[1],
                 marker='.')
    plt.fill_between(PSD_phase_frequency_new[cases],
                 np.nanmean(perfect_PSD_phase_new[cases,:,-1],axis=0) + np.nanstd(perfect_PSD_phase_new[cases,:,-1],axis=0,ddof=1)/np.sqrt(perfect_PSD_phase_new.shape[1]),
                 np.nanmean(perfect_PSD_phase_new[cases,:,-1],axis=0) - np.nanstd(perfect_PSD_phase_new[cases,:,-1],axis=0,ddof=1)/np.sqrt(perfect_PSD_phase_new.shape[1]),
                 color=color_list[1],
                 alpha=0.3)
    
    count = 2
    for cases in np.arange(CASES)[arg][::-1]:
        # if (cases != [9,16,21]).all():
        # if (cases != [7,9,14,16,19,21]).all():
        if (cases != [7,16]).all():
            continue
        i = count
        use_color = color_list[i]
        plt.fill_between(PSD_phase_frequency_new[cases],
                     np.nanmean(PSD_phase_new[cases,:,-1],axis=0) + np.nanstd(PSD_phase_new[cases,:,-1],axis=0,ddof=1)/np.sqrt(PSD_phase_new.shape[1]),
                     np.nanmean(PSD_phase_new[cases,:,-1],axis=0) - np.nanstd(PSD_phase_new[cases,:,-1],axis=0,ddof=1)/np.sqrt(PSD_phase_new.shape[1]),
                     color=use_color,
                     alpha=0.3)
        plt.plot(PSD_phase_frequency_new[cases],
                     np.nanmean(PSD_phase_new[cases,:,-1],axis=0),
                     label=name_list[cases],#'{:d}'.format(cases),#'distorted control',
                     color=use_color,
                     marker='.')
        count += 1
        
    plt.vlines(1/dsub/2,MIN,MAX,label='AO Nyquist Frequency',color=color_list[7])
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(MIN,MAX)
    # plt.legend()
    # plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.66, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend()
    # plt.title(name_list[cases])
    plt.xlabel('Spatial frequency (m$^{-1}$)')
    plt.ylabel('PSD of phase (rad$^2$ m$^2$)')
    plt.savefig(TYPE + 'psd-phase-selected.pdf', bbox_inches='tight',dpi=600)#-{:d}.png'.format(cases))
    plt.show()
    
    
    arg = [16,9,14,7,12,17,1,5,10,15,3,8,13,18,2,6,11,4,0]
    arg2 = [16,9,14,7,1,12,5,10,3,17,15,2,8,13,6,11,4,18,0]
    
    
   
    # Extract values
    rytov_values = expect_rytov[arg2]
    psd_values = np.nanmean((PSD_phase_new)[arg2, :, -1], axis=1)
    perfect_psd_values = np.nanmean(perfect_PSD_phase_new[arg, :, -1], axis=1)
    psd_error = np.nanstd((PSD_phase_new)[arg2, :, -1], axis=1)
    perfect_psd_error = np.nanstd(perfect_PSD_phase_new[arg, :, -1], axis=1)
    
    # Compute the ratio
    psd_ratio = psd_values / perfect_psd_values
    psd_ratio_error = psd_ratio*(psd_error**2/psd_values**2 + perfect_psd_error**2/perfect_psd_values**2)**0.5
    
    # Group by unique expect_rytov values
    unique_rytov, inverse_indices = np.unique(np.round(rytov_values,decimals=3), return_inverse=True)
    ratio_matrix = []
    error_matrix = []
    
    for i in range(len(unique_rytov)):
        matching_rows = psd_ratio[inverse_indices == i]
        matching_rows_errors = psd_ratio_error[inverse_indices == i]
        ratio_matrix.append(np.nanmean(matching_rows, axis=0))
        error_matrix.append(np.nanmean(matching_rows_errors**2, axis=0)**0.5)
    
    ratio_matrix = np.array(ratio_matrix)
    error_matrix = np.array(error_matrix)
    
   
    # psd_ratio_rytov = np.copy(ratio_matrix[:,np.where(PSD_phase_frequency_new[0]<=1/dsub/2)].mean(-1))
    psd_ratio_rytov = np.copy(ratio_matrix[:,np.where(PSD_phase_frequency_new[0]<=1/dsub/2)].mean(-1))
    psd_ratio_rytov_error = np.copy(error_matrix[:,np.where(PSD_phase_frequency_new[0]<=1/dsub/2)].mean(-1))
    
    #######
    # Compute AS values (as %)
    as_values = 100 * expect_AS[arg]
    
    # Compute the PSD ratio
    psd_values = np.nanmean((PSD_phase_new)[arg, :, -1], axis=1)
    perfect_psd_values = np.nanmean(perfect_PSD_phase_new[arg, :, -1], axis=1)
    psd_ratio = psd_values / perfect_psd_values
    
    # Group by unique AS values
    unique_as, inverse_indices = np.unique(np.round(as_values,decimals=3), return_inverse=True)
    ratio_matrix = []
    error_matrix = []
    
    for i in range(len(unique_as)):
        matching_rows = psd_ratio[inverse_indices == i]
        matching_rows_errors = psd_ratio_error[inverse_indices == i]
        ratio_matrix.append(np.nanmean(matching_rows, axis=0))
        error_matrix.append(np.nanmean(matching_rows_errors**2, axis=0)**0.5)
    
    ratio_matrix = np.array(ratio_matrix)
    error_matrix = np.array(error_matrix)
    
    
    # psd_ratio_as = np.copy(ratio_matrix[:,np.where(PSD_phase_frequency_new[0]<=1/dsub/2)].mean(-1))
    psd_ratio_as = np.copy(ratio_matrix[:,np.where(PSD_phase_frequency_new[0]<=1/dsub/2)].mean(-1))
    psd_ratio_as_error = np.copy(error_matrix[:,np.where(PSD_phase_frequency_new[0]<=1/dsub/2)].mean(-1))

    
    if TYPE == 'A':
        plt.figure(figsize=(4,4))
        plt.plot(np.arange(MODE) + 1,theory_noll,ls=':',
                 color='tab:blue',
                 label='Turbulence')
        plt.plot((np.arange(MODE))[3:], (np.var(ATM_Z,axis=(0,1)))[3:],label='No AO',color='tab:orange')
        plt.fill_between((np.arange(MODE))[3:],
                      (np.var(ATM_Z,axis=(0,1)))[3:] + 2*(np.var(ATM_Z,axis=(0,1)))[3:].T/np.sqrt(2*ATM_Z.shape[0]*ATM_Z.shape[1] - 2),
                      (np.var(ATM_Z,axis=(0,1)))[3:] - 2*(np.var(ATM_Z,axis=(0,1)))[3:].T/np.sqrt(2*ATM_Z.shape[0]*ATM_Z.shape[1] - 2),
                      color='tab:orange',
                      alpha=0.3)
        plt.plot(np.arange(MODE - 1) + 1,
                      np.var(perfect_RES_phase_Z[:,:,-1,1:],axis=(0,1),ddof=1).T,
                      color='tab:green',
                      label='No control error',marker='.')
        plt.fill_between(np.arange(MODE - 1) + 1,
                      np.var(perfect_RES_phase_Z[:,:,-1,1:],axis=(0,1),ddof=1).T + 2*np.var(perfect_RES_phase_Z[:,:,-1,1:],axis=(0,1),ddof=1).T/np.sqrt(2*ATM_Z.shape[0]*ATM_Z.shape[1] - 2),
                      np.var(perfect_RES_phase_Z[:,:,-1,1:],axis=(0,1),ddof=1).T - 2*np.var(perfect_RES_phase_Z[:,:,-1,1:],axis=(0,1),ddof=1).T/np.sqrt(2*ATM_Z.shape[0]*ATM_Z.shape[1] - 2),
                      color='tab:green',
                      alpha=0.3)
            
        plt.yscale('log')
        # plt.title('terminal z-decomposition')
        plt.legend(loc='upper right')
        plt.ylabel('Variance of Z coefficients (rad$^2$)')
        plt.xlabel('Zernike Modes')
        # plt.title('Terminal Wavefront Residual Zernike Mode Decomposition\n' + name_list[cases])
        plt.savefig(TYPE + 'theory_' + 'z_decompose.pdf', bbox_inches='tight',dpi=600)#_{:d}.png'.format(cases))
        plt.show()
        
        plt.figure(figsize=(4,4))
        MAX = np.max(np.array([cross_section_theory_psd.max(),np.nanmean(perfect_PSD_phase_new,axis=1).max(),np.nanmean(PSD_phase_new,axis=1).max()]))*10
        MIN = np.min(np.array([cross_section_theory_psd.min(),np.nanmean(perfect_PSD_phase_new,axis=1).min(),np.nanmean(PSD_phase_new,axis=1).min()]))/10
        # fig = plt.figure(figsize=(5,4))
        # ax = plt.subplot(111)
        # plt.figure(figsize=(4,4))
        plt.plot(PSD_phase_frequency_new[cases],cross_section_theory_psd,label='Turbulence',ls=':',color='tab:blue')
        
        plt.plot(PSD_phase_frequency_new[cases],np.mean(ATM_PSD_phase,axis=(0,1)),label='No AO',color='tab:orange')
        plt.fill_between(PSD_phase_frequency_new[cases],
                     np.mean(ATM_PSD_phase,axis=(0,1)) + np.std(ATM_PSD_phase,axis=(0,1))/np.sqrt(perfect_PSD_phase_new.shape[1]),
                     np.mean(ATM_PSD_phase,axis=(0,1)) - np.std(ATM_PSD_phase,axis=(0,1))/np.sqrt(perfect_PSD_phase_new.shape[1]),
                     color='tab:orange',
                     alpha=0.3)
        
        plt.plot(PSD_phase_frequency_new[cases],
                     np.nanmean(perfect_PSD_phase_new[cases,:,-1],axis=0),
                     label='No control error',
                     color='tab:green',
                     marker='.')
        plt.fill_between(PSD_phase_frequency_new[cases],
                     np.nanmean(perfect_PSD_phase_new[cases,:,-1],axis=0) + np.nanstd(perfect_PSD_phase_new[cases,:,-1],axis=0,ddof=1)/np.sqrt(perfect_PSD_phase_new.shape[1]),
                     np.nanmean(perfect_PSD_phase_new[cases,:,-1],axis=0) - np.nanstd(perfect_PSD_phase_new[cases,:,-1],axis=0,ddof=1)/np.sqrt(perfect_PSD_phase_new.shape[1]),
                     color='tab:green',
                     alpha=0.3)
        
        plt.vlines(1/dsub/2,MIN,MAX,label='AO Nyquist Freq.',color='tab:red')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(MIN,MAX)
        plt.legend(loc='upper right')
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.title(name_list[cases])
        plt.xlabel('Spatial frequency (1/m)')
        plt.ylabel('Phase power spectrum density (rad$^2$ m$^2$)')
        plt.savefig(TYPE + 'theory_' + 'psd-phase.pdf', bbox_inches='tight',dpi=600)#-{:d}.png'.format(cases))
        plt.show()
        
        
        
        
        
        
    
    from matplotlib.colors import LogNorm, Normalize
    
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    ax1.set_xlim(1e-14, r02cn2dz(0.02))
    ax2.set_ylim(1, 10)

    r0_mark = np.array([1, 0.5, 0.2, 0.1, 0.05, 0.02]) * 100
    new_tick_locations = np.log10(r02cn2dz(r0_mark / 100)) + 14
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(new_tick_locations)
    labels = ['{:.0f}'.format(val) for val in r0_mark]
    ax2.set_xticklabels(labels)

    ax2.set_xlabel("Layer Fried parameter (cm) at 500 nm",fontsize=fontsize)
    # ax1.set_xlabel("DM2's $\int C^2_n$sec$\zeta dh$ (m$^{1/3}$)")
    ax1.set_xlabel("Layer strength (m$^{1/3}$)",fontsize=fontsize)
    ax1.set_ylabel("Layer distance (km)",fontsize=fontsize)

    x = np.logspace(np.log10(1e-14),np.log10(r02cn2dz(0.02)))#r02cn2dz(np.logspace(np.log10(0.02), np.log10(5)))[::-1]
    y = np.logspace(np.log10(1), np.log10(10))
    yy, xx = np.meshgrid(y, x)
    
    dact_immediate_dm = dsub
    dact_next_immediate_dm = dsub
    Dimmediate_dm = 0.4
    L0 = 5
    wvl = 500e-9
    k = np.pi * 2 / wvl
    
    induced_rytov = (yy*1000)**(5./6.) * (0.563*k**(7./6.)*xx)
    rms_as = yy*1000*(0.162 * wvl ** 2 * (dsub ** (-1 / 3) - D ** (-1 / 3)) * cn2dz2r0(xx) ** (-5 / 3))**0.5*100/dact_next_immediate_dm

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid()
    ax2.grid()

    

    RYTOV = rytov_number(wvl, cn2dz2r0(xx), yy * 1000.)

    # color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    TOPLOT = np.copy(avg_terminal_WFE_only_control)

    cmap = plt.cm.Wistia

    # Set minimum and maximum for color normalization
    vmin = 0#TOPLOT[1:].min()#1
    vmax = 250#TOPLOT.max()#500#

    # Use LogNorm for color scaling
    norm = Normalize(vmin=vmin, vmax=vmax)

    # First plot filled markers for points >= vmin
    for i in range(r0HA.size):
        if TOPLOT[i] >= vmin:
            color_val = cmap(norm(TOPLOT[i]))
            ax1.plot(r02cn2dz(r0HA)[i], z[i] / 1000,
                     marker='o', markersize=40, color=color_val, linestyle='None')

    # Then plot hollow shell markers for points < vmin on top
    for i in range(r0HA.size):
        if TOPLOT[i] < vmin:
            ax1.plot(r02cn2dz(r0HA)[i], z[i] / 1000,
                     marker='o', markersize=40, markerfacecolor='none',
                     markeredgecolor='black', markeredgewidth=2, linestyle='None', zorder=10)

    # Add colorbar for the filled points
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1)
    cbar.ax.tick_params(labelsize=fontsize)
    
    # for count, i in enumerate([0.01, 0.02, 0.05, 0.1, 0.3, 0.6]):
    #     ax1.plot(x,
    #              (i / 0.563 / k ** (7. / 6.) / x) ** (6. / 5.) / 1000.,
    #              color=color_list[count], ls='-')
    
    cs1 = ax1.contour(x,y,induced_rytov.T,levels=[0.01, 0.02, 0.05, 0.1, 0.3, 0.6],colors='r',ls='-')
    ax1.clabel(cs1,inline=True,fontsize=fontsize)

    # for count, i in enumerate([1, 2, 5, 10, 20, 50]):
    #     ax1.plot(x,
    #              (i / 100.
    #               * dact_next_immediate_dm
    #               / (0.162 * wvl ** 2 * (dsub ** (-1 / 3) - D ** (-1 / 3)) * cn2dz2r0(x) ** (-5 / 3)) ** 0.5) / 1000,
    #              color=color_list[count], ls='--')
    cs2 = ax1.contour(x,y,rms_as.T,levels=[1, 2, 5, 10, 20, 50],colors='b',linestyles='dashed')
    ax1.clabel(cs2,inline=True,fontsize=fontsize)

    # fig.suptitle('WFE from distorted control error after 5 iterations (nm)')
    # plt.gcf().set_size_inches(8, 8)

    plt.savefig(TYPE + 'control-error-2.pdf', bbox_inches='tight',dpi=600)
    plt.show()
    
    ########################################################################
    
    
    
    DATA = np.copy(avg_terminal_WFE_only_control)
    error = np.copy(avg_terminal_WFE_only_control_error)
    
    
    # well_behaved = (expect_AS < 0.3) * (expect_rytov < 0.3)
    well_behaved = np.ones_like(expect_AS,dtype=bool)
    well_behaved[0] = False
    
    guess_AS = expect_AS
    
    
    
    adjust_COMBINED_MEASURED_IM_AS = COMBINED_MEASURED_IM_AS
    adjust_error_as = error_as
    
    
    
    if TYPE == 'A':
        expect_rytov_A = np.copy(expect_rytov)
        expect_AS_A = np.copy(expect_AS)
        AS_influ_A = np.copy(old_COMBINED_MEASURED_IM_AS)
        AS_influ_error_A = np.copy(old_COMBINED_MEASURED_IM_AS_error)
        AS_slope_A = np.copy(adjust_COMBINED_MEASURED_IM_AS)
        AS_slope_error_A = np.copy(adjust_error_as)
        rytov_A = np.copy(COMBINED_MEASURED_IM_RYTOV)
        rytov_error_A = np.copy(error_rytov)
        as_influ_a = np.copy(old_MEASURED_IM_SUBAP_SHIFT)
        as_slope_a = np.copy(ANOTHER_AS)
        WFE_A = np.copy(WFE)
        WFE_A_control = np.copy(avg_terminal_WFE_only_control)
        WFE_A_error = np.copy(avg_terminal_WFE_only_control_error)
        IM_A = np.copy(IM)
        perfect_IM_A = np.copy(perfect_IM)
        CC_A = np.copy(CC)
        perfect_CC_A = np.copy(perfect_CC)
        RES_A = np.copy(RES[:,0]+1j*RES[:,1])
        perfect_RES_A = np.copy(perfect_RES[:,0] + 1j*perfect_RES[:,1])
        psd_ratio_rytov_A = np.copy(psd_ratio_rytov)
        psd_ratio_as_A = np.copy(psd_ratio_as)
        z_ratio_rytov_A = np.copy(z_ratio_rytov)
        z_ratio_as_A = np.copy(z_ratio_as)
        unique_as_A = np.copy(unique_as)
        unique_rytov_A = np.copy(unique_rytov)
        perfect_WFE_A = np.copy(perfect_WFE)
        psd_ratio_rytov_error_A = np.copy(psd_ratio_rytov_error)
        psd_ratio_as_error_A = np.copy(psd_ratio_as_error)
        z_ratio_rytov_error_A = np.copy(z_ratio_rytov_error)
        z_ratio_as_error_A = np.copy(z_ratio_as_error)
        IM_RMS_A = np.copy(IM_RMS)
        IM_RMS_A_error = np.copy(IM_RMS_error)
        CC_RMS_A = np.copy(CC_RMS)
        CC_RMS_A_error = np.copy(CC_RMS_error)
    
    if TYPE == 'B':
        expect_rytov_B = np.copy(expect_rytov)
        expect_AS_B = np.copy(expect_AS)
        AS_influ_B = np.copy(old_COMBINED_MEASURED_IM_AS)
        AS_influ_error_B = np.copy(old_COMBINED_MEASURED_IM_AS_error)
        AS_slope_B = np.copy(adjust_COMBINED_MEASURED_IM_AS)
        AS_slope_error_B = np.copy(adjust_error_as)
        rytov_B = np.copy(COMBINED_MEASURED_IM_RYTOV)
        rytov_error_B = np.copy(error_rytov)
        as_influ_b = np.copy(old_MEASURED_IM_SUBAP_SHIFT)
        as_slope_b = np.copy(ANOTHER_AS)
        WFE_B = np.copy(WFE)
        WFE_B_control = np.copy(avg_terminal_WFE_only_control)
        WFE_B_error = np.copy(avg_terminal_WFE_only_control_error)
        IM_B = np.copy(IM)
        perfect_IM_B = np.copy(perfect_IM)
        CC_B = np.copy(CC)
        perfect_CC_B = np.copy(perfect_CC)
        RES_B = np.copy(RES[:,0]+1j*RES[:,1])
        perfect_RES_B = np.copy(perfect_RES[:,0] + 1j*perfect_RES[:,1])
        psd_ratio_rytov_B = np.copy(psd_ratio_rytov)
        psd_ratio_as_B = np.copy(psd_ratio_as)
        z_ratio_rytov_B = np.copy(z_ratio_rytov)
        z_ratio_as_B = np.copy(z_ratio_as)
        unique_as_B = np.copy(unique_as_A)
        unique_rytov_B = np.copy(unique_rytov)
        perfect_WFE_B = np.copy(perfect_WFE)
        psd_ratio_rytov_error_B = np.copy(psd_ratio_rytov_error)
        psd_ratio_as_error_B = np.copy(psd_ratio_as_error)
        z_ratio_rytov_error_B = np.copy(z_ratio_rytov_error)
        z_ratio_as_error_B = np.copy(z_ratio_as_error)
        IM_RMS_B = np.copy(IM_RMS)
        IM_RMS_B_error = np.copy(IM_RMS_error)
        CC_RMS_B = np.copy(CC_RMS)
        CC_RMS_B_error = np.copy(CC_RMS_error)
    
    if TYPE == 'C':
        expect_rytov_C = np.copy(expect_rytov)
        expect_AS_C = np.copy(expect_AS)
        AS_influ_C = np.copy(old_COMBINED_MEASURED_IM_AS)
        AS_influ_error_C = np.copy(old_COMBINED_MEASURED_IM_AS_error)
        AS_slope_C = np.copy(adjust_COMBINED_MEASURED_IM_AS)
        AS_slope_error_C = np.copy(adjust_error_as)
        rytov_C = np.copy(COMBINED_MEASURED_IM_RYTOV)
        rytov_error_C = np.copy(error_rytov)
        as_influ_c = np.copy(old_MEASURED_IM_SUBAP_SHIFT)
        as_slope_c = np.copy(ANOTHER_AS)
        WFE_C = np.copy(WFE)
        WFE_C_control = np.copy(avg_terminal_WFE_only_control)
        WFE_C_error = np.copy(avg_terminal_WFE_only_control_error)
        IM_C = np.copy(IM)
        perfect_IM_C = np.copy(perfect_IM)
        CC_C = np.copy(CC)
        perfect_CC_C = np.copy(perfect_CC)
        RES_C = np.copy(RES[:,0]+1j*RES[:,1])
        perfect_RES_C = np.copy(perfect_RES[:,0] + 1j*perfect_RES[:,1])
        psd_ratio_rytov_C = np.copy(psd_ratio_rytov)
        psd_ratio_as_C = np.copy(psd_ratio_as)
        z_ratio_rytov_C = np.copy(z_ratio_rytov)
        z_ratio_as_C = np.copy(z_ratio_as)
        unique_as_C = np.copy(unique_as)
        unique_rytov_C = np.copy(unique_rytov)
        perfect_WFE_C = np.copy(perfect_WFE)
        psd_ratio_rytov_error_C = np.copy(psd_ratio_rytov_error)
        psd_ratio_as_error_C = np.copy(psd_ratio_as_error)
        z_ratio_rytov_error_C = np.copy(z_ratio_rytov_error)
        z_ratio_as_error_C = np.copy(z_ratio_as_error)
        IM_RMS_C = np.copy(IM_RMS)
        IM_RMS_C_error = np.copy(IM_RMS_error)
        CC_RMS_C = np.copy(CC_RMS)
        CC_RMS_C_error = np.copy(CC_RMS_error)


os.chdir(HOME)



## !!! compare control error

WFE_A_control[WFE_A_control==0]=np.nan
WFE_B_control[WFE_B_control==0]=np.nan
WFE_C_control[WFE_C_control==0]=np.nan

expect_rytov_A = np.round(expect_rytov_A,decimals=2)
expect_rytov_B = np.round(expect_rytov_B,decimals=2)
expect_rytov_C = np.round(expect_rytov_C,decimals=2)


unique_expect_rytov_A, mean_control_A, error_control_A = weighted_y_by_x_groups(expect_rytov_A, WFE_A_control, WFE_A_error)
unique_expect_rytov_B, mean_control_B, error_control_B = weighted_y_by_x_groups(expect_rytov_B, WFE_B_control, WFE_B_error)
unique_expect_rytov_C, mean_control_C, error_control_C = weighted_y_by_x_groups(expect_rytov_C, WFE_C_control, WFE_C_error)




plt.errorbar(unique_expect_rytov_A,mean_control_A,yerr=error_control_A,#/np.sqrt(TRIAL),
              ls='-',marker='o',label='Subap.={:.1f}cm'.format(dsub_A*100),capsize=5)
plt.errorbar(unique_expect_rytov_B,mean_control_B,yerr=error_control_B,#/np.sqrt(TRIAL),
              ls='-',marker='o',label='Subap.={:.1f}cm'.format(dsub_B*100),capsize=5)
plt.errorbar(unique_expect_rytov_C,mean_control_C,yerr=error_control_C,#/np.sqrt(TRIAL),
              ls='-',marker='o',label='Subap.={:.1f}cm'.format(dsub_C*100),capsize=5)
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Control error (nm)')
plt.xlabel('Expected $\sigma^2_{R,DM}$')
plt.legend()
plt.ylim(1,600)
plt.savefig('compare-control-rytov-simple.pdf', bbox_inches='tight')
plt.show()




WFE_A_control[WFE_A_control==0]=np.nan
WFE_B_control[WFE_B_control==0]=np.nan
WFE_C_control[WFE_C_control==0]=np.nan

unique_expect_AS_A, mean_control_A, error_control_A = weighted_y_by_x_groups(expect_AS_A, WFE_A_control, WFE_A_error)
unique_expect_AS_B, mean_control_B, error_control_B = weighted_y_by_x_groups(expect_AS_B, WFE_B_control, WFE_B_error)
unique_expect_AS_C, mean_control_C, error_control_C = weighted_y_by_x_groups(expect_AS_C, WFE_C_control, WFE_C_error)


plt.errorbar(unique_expect_AS_A*100,mean_control_A,yerr=error_control_A,#/np.sqrt(TRIAL),
             ls='-',marker='o',label='Subap.={:.1f}cm'.format(dsub_A*100),capsize=5)
plt.errorbar(unique_expect_AS_B*100,mean_control_B,yerr=error_control_B,#/np.sqrt(TRIAL),
             ls='-',marker='o',label='Subap.={:.1f}cm'.format(dsub_B*100),capsize=5)
plt.errorbar(unique_expect_AS_C*100,mean_control_C,yerr=error_control_C,#/np.sqrt(TRIAL),
             ls='-',marker='o',label='Subap.={:.1f}cm'.format(dsub_C*100),capsize=5)
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Control error (nm)')
plt.xlabel('Expected $RMS_{AS}$ (%)')
plt.ylim(1,600)
plt.legend()
plt.savefig('compare-control-as-simple.pdf', bbox_inches='tight')
plt.show()


# # Define color normalization based on AS range
all_expect_AS = np.array([
    np.min([expect_AS_A[1:], expect_AS_B[1:], expect_AS_C[1:]]),
    np.max([expect_AS_A[1:], expect_AS_B[1:], expect_AS_C[1:]])
])
norm = colors.LogNorm(vmin=all_expect_AS[0], vmax=all_expect_AS[1])
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)


#!!! compare control error parameters



plt.clf()

# Combine all datasets
all_expect_rytov = np.concatenate([expect_rytov_A, expect_rytov_B, expect_rytov_C])
all_rytov = np.concatenate([rytov_A, rytov_B, rytov_C])
all_errors_rytov = np.concatenate([rytov_error_A, rytov_error_B, rytov_error_C])

# all_expect_rytov[all_expect_rytov==0] = np.nan
# all_rytov[all_rytov==0] = np.nan
# all_errors_rytov[all_errors_rytov==0] = np.nan

# Round x values to 3 decimals to identify unique values
rounded_expect_rytov = np.round(all_expect_rytov, 3)

# Group values by rounded x
grouped = defaultdict(list)
for x_r, x, y, err in zip(rounded_expect_rytov, all_expect_rytov, all_rytov, all_errors_rytov):
    grouped[x_r].append((x, y, err))

# Aggregate by weighted mean
x_vals = []
y_means = []
y_errors = []

for x_r in sorted(grouped.keys()):
    group = grouped[x_r]
    xs, ys, errs = zip(*group)
    
    # Mean of x (not weighted since differences are small)
    x_mean = np.mean(xs)
    
    # Convert to numpy arrays
    ys = np.array(ys)
    errs = np.array(errs)
    
    # Compute weights
    # weights = 1.0 / errs**2
    # y_mean = np.sum(ys * weights) / np.sum(weights)
    # y_err = np.sqrt(1.0 / np.sum(weights))
    y_mean = np.nanmean(ys)
    y_err = np.sqrt(np.nanmean(errs**2))

    # Store
    x_vals.append(x_mean)
    y_means.append(y_mean)
    y_errors.append(y_err)

final_expect_rytov = np.copy(x_vals)
final_measure_rytov = np.copy(y_means)
final_error_rytov = np.copy(y_errors)

# Plot
# plt.figure(figsize=(4, 4))
plt.errorbar(x_vals, y_means, yerr=y_errors, fmt='o', linestyle='none', capsize=5)
plt.plot([0.01,0.8],[0.01,0.8])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Expected $\sigma^2_{R,DM}$')
plt.ylabel('Measured $\sigma^2_{\chi,DM}$')
#plt.legend()
plt.axis('square')
plt.savefig('compare-rytov-simple.pdf', bbox_inches='tight',dpi=600)
plt.show()





# Combine all datasets
all_expect_AS_influ = np.concatenate([expect_AS_A, expect_AS_B, expect_AS_C])*100
all_AS_influ = np.concatenate([AS_influ_A, AS_influ_B, AS_influ_C])*100
all_errors_AS_influ = np.concatenate([AS_influ_error_A, AS_influ_error_B, AS_influ_error_C])*100

# all_expect_AS_influ[all_expect_AS_influ==0] = np.nan
# all_AS_influ[all_AS_influ==0] = np.nan
# all_errors_AS_influ[all_errors_AS_influ==0] = np.nan

# Round x values to 3 decimals to identify unique values
rounded_expect_AS_influ = np.round(all_expect_AS_influ, 3)

# Group values by rounded x
grouped = defaultdict(list)
for x_r, x, y, err in zip(rounded_expect_AS_influ, all_expect_AS_influ, all_AS_influ, all_errors_AS_influ):
    grouped[x_r].append((x, y, err))

# Aggregate by weighted mean
x_vals = []
y_means = []
y_errors = []

for x_r in sorted(grouped.keys()):
    group = grouped[x_r]
    xs, ys, errs = zip(*group)
    
    # Mean of x (not weighted since differences are small)
    x_mean = np.mean(xs)
    
    # Convert to numpy arrays
    ys = np.array(ys)
    errs = np.array(errs)
    
    # Compute weights
    # weights = 1.0 / errs**2
    # y_mean = np.sum(ys * weights) / np.sum(weights)
    # y_err = np.sqrt(1.0 / np.sum(weights))
    y_mean = np.nanmean(ys)
    y_err = np.sqrt(np.nanmean(errs**2))

    # Store
    x_vals.append(x_mean)
    y_means.append(y_mean)
    y_errors.append(y_err)

final_expect_AS_influ = np.copy(x_vals)
final_measure_AS_influ = np.copy(y_means)
final_error_AS_influ = np.copy(y_errors)

# Plot
# plt.figure(figsize=(4, 4))
plt.errorbar(x_vals, y_means, yerr=y_errors, fmt='o', linestyle='none', capsize=5)
plt.plot(np.unique(all_expect_AS_influ),np.unique(all_expect_AS_influ),color='tab:orange')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Expected $RMS_{AS}$ (%)')
plt.ylabel('Measured $RMS_{AS}$ : influence method (%)')
# plt.legend()
# plt.axis('square')
plt.ylim(1,120)
plt.savefig('compare-as-influ-simple.pdf', bbox_inches='tight')
plt.show()


# Combine all datasets
all_expect_AS_slope = np.concatenate([expect_AS_A, expect_AS_B, expect_AS_C])*100
all_AS_slope = np.concatenate([AS_slope_A, AS_slope_B, AS_slope_C])*100
all_errors_AS_slope = np.concatenate([AS_slope_error_A, AS_slope_error_B, AS_slope_error_C])*100

# Round x values to 3 decimals to identify unique values
rounded_expect_AS_slope = np.round(all_expect_AS_slope, 3)

# Group values by rounded x
grouped = defaultdict(list)
for x_r, x, y, err in zip(rounded_expect_AS_slope, all_expect_AS_slope, all_AS_slope, all_errors_AS_slope):
    grouped[x_r].append((x, y, err))

# Aggregate by weighted mean
x_vals = []
y_means = []
y_errors = []

for x_r in sorted(grouped.keys()):
    group = grouped[x_r]
    xs, ys, errs = zip(*group)
    
    # Mean of x (not weighted since differences are small)
    x_mean = np.mean(xs)
    
    # Convert to numpy arrays
    ys = np.array(ys)
    errs = np.array(errs)
    
    # Compute weights
    # weights = 1.0 / errs**2
    # y_mean = np.sum(ys * weights) / np.sum(weights)
    # y_err = np.sqrt(1.0 / np.sum(weights))
    y_mean = np.nanmean(ys)
    y_err = np.sqrt(np.nanmean(errs**2))

    # Store
    x_vals.append(x_mean)
    y_means.append(y_mean)
    y_errors.append(y_err)

final_expect_AS_slope = np.copy(x_vals)
final_measure_AS_slope = np.copy(y_means)
final_error_AS_slope = np.copy(y_errors)

# Plot
# plt.figure(figsize=(4, 4))
plt.errorbar(x_vals, y_means, yerr=y_errors, fmt='o', linestyle='none', capsize=5)
plt.plot(np.unique(all_expect_AS_slope),np.unique(all_expect_AS_slope),color='tab:orange')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Expected $RMS_{AS}$ (%)')
plt.ylabel('Measured $RMS_{AS}$ : slope method (%)')
# plt.legend()
plt.axis('square')
plt.savefig('compare-as-slope-simple.pdf', bbox_inches='tight')
plt.show()



plt.clf()
# plt.figure(figsize=(4,4))
plt.errorbar(final_measure_AS_influ[1:],final_measure_AS_slope[1:],
             xerr=final_error_AS_influ[1:],#/np.sqrt(TRIAL),
             yerr=final_error_AS_slope[1:],#/np.sqrt(TRIAL),
              ls='',marker='o',capsize=5,color='tab:blue',alpha=0.8)
plt.plot(np.unique(all_expect_AS_influ),np.unique(all_expect_AS_slope),color='tab:orange')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Measured $RMS_{AS}$ : influence method (%)')
plt.ylabel('Measured $RMS_{AS}$ : slope method (%)')
plt.xlim(1,120)
#plt.legend()
# plt.axis('square')
plt.savefig('compare-as-influ-slope-simple.pdf', bbox_inches='tight',dpi=600)
plt.show()

plt.clf()





#!!!! these can be combined into a point

A = np.nanvar(ABERATION_slope,axis=(-1,-2))*((config['WFS'][0]['subapFOV']/config['WFS'][0]['pxlsPerSubap'])/206265)**2 # var tip angle in radian2
B = np.nanvar(ATM_SLOPES,axis=(-1,-2))*((config['WFS'][0]['subapFOV']/config['WFS'][0]['pxlsPerSubap'])/206265)**2 # var tip angle in radian2

A /= ((wvl**2*dsub**(-1/3) * r0HA**(-5/3)))
B /= ((wvl**2*dsub**(-1/3) * r0**(-5/3)))

Ae = 2*A/np.sqrt(2*(~np.isnan(A)).sum()-2)
Be = 2*B/np.sqrt(2*(~np.isnan(B)).sum()-2)

C = (old_COMBINED_MEASURED_IM_AS*dsub/z)**2
D = (COMBINED_MEASURED_IM_AS*dsub/z)**2

C /= ((wvl**2*dsub**(-1/3) * r0HA**(-5/3)))
D /= ((wvl**2*dsub**(-1/3) * r0HA**(-5/3)))

Ce = (old_COMBINED_MEASURED_IM_AS_error*dsub/z)**2
De = (error_as*dsub/z)**2

Ce /= ((wvl**2*dsub**(-1/3) * r0HA**(-5/3)))
De /= ((wvl**2*dsub**(-1/3) * r0HA**(-5/3)))



plt.figure(figsize=(4,3))
plt.imshow(np.mean(perfect_IM_A,axis=0)[2:])
plt.colorbar()
# plt.tight_layout()
plt.savefig('undistort_IM.png',dpi=600,bbox_inches='tight')
plt.show()

plt.figure(figsize=(4,3))
plt.imshow(np.std(IM_A[16],axis=0)[2:])
plt.colorbar()
# plt.tight_layout()
plt.savefig('distort_IM_std.png',dpi=600,bbox_inches='tight')
plt.show()



# !!! SVDS

svd_A = np.zeros((19,50,123),dtype=float)
svd_B = np.zeros((19,50,83),dtype=float)
svd_C = np.zeros((19,50,48),dtype=float)

for i in range(19):
    for j in range(50):
        _,svd_A[i,j],_ = np.linalg.svd(IM_A[i,j])
        _,svd_B[i,j],_ = np.linalg.svd(IM_B[i,j])
        _,svd_C[i,j],_ = np.linalg.svd(IM_C[i,j])
        svd_A[i,j] /= svd_A[i,j].max()
        svd_B[i,j] /= svd_B[i,j].max()
        svd_C[i,j] /= svd_C[i,j].max()




plt.figure(figsize=(6,4))
for i in range(18):
    for j in range(20):
        plt.plot(svd_A[i+1,j],color='tab:blue',alpha=0.01)
        plt.plot(svd_B[i+1,j],color='tab:orange',alpha=0.01)
        plt.plot(svd_C[i+1,j],color='tab:green',alpha=0.01)
plt.plot(svd_A[0].mean(0),color='tab:blue',label='$d_{sub}$=4.0cm')
plt.plot(svd_B[0].mean(0),color='tab:orange',label='$d_{sub}$=5.0cm')
plt.plot(svd_C[0].mean(0),color='tab:green',label='$d_{sub}$=6.7cm')
plt.xlabel('Matrix mode')
plt.ylabel('Normalized SVD')
plt.yscale('log')
plt.hlines(0.05,0,120,color='k')
plt.legend()
plt.savefig('svd-distort.png',dpi=600,bbox_inches='tight')
plt.show()

A = np.copy(svd_A)
# Example array
# A.shape == (19, 50, 123)
# A = ...

# Create a boolean mask: True where A < 0.05
mask = A < 0.05

# Initialize result with a sentinel value (e.g., -1 if no such z exists)
min_z_indices = np.full((A.shape[0], A.shape[1]), -1, dtype=int)

# Check if any True values exist along the z-axis
has_values = mask.any(axis=2)

# Use argmax to get first occurrence — only where values exist
min_z_indices[has_values] = mask.argmax(axis=2)[has_values]

print(min_z_indices[0][16])
print(np.min(min_z_indices[16])) # highest number of modes lost
print(np.max(min_z_indices[16])) # lowest number of modes lost

A = np.copy(svd_B)
# Example array
# A.shape == (19, 50, 123)
# A = ...

# Create a boolean mask: True where A < 0.05
mask = A < 0.05

# Initialize result with a sentinel value (e.g., -1 if no such z exists)
min_z_indices = np.full((A.shape[0], A.shape[1]), -1, dtype=int)

# Check if any True values exist along the z-axis
has_values = mask.any(axis=2)

# Use argmax to get first occurrence — only where values exist
min_z_indices[has_values] = mask.argmax(axis=2)[has_values]

print(min_z_indices[0][16])
print(np.min(min_z_indices[16])) # highest number of modes lost
print(np.max(min_z_indices[16])) # lowest number of modes lost

A = np.copy(svd_C)
# Example array
# A.shape == (19, 50, 123)
# A = ...

# Create a boolean mask: True where A < 0.05
mask = A < 0.05

# Initialize result with a sentinel value (e.g., -1 if no such z exists)
min_z_indices = np.full((A.shape[0], A.shape[1]), -1, dtype=int)

# Check if any True values exist along the z-axis
has_values = mask.any(axis=2)

# Use argmax to get first occurrence — only where values exist
min_z_indices[has_values] = mask.argmax(axis=2)[has_values]

print(min_z_indices[0][16])
print(np.min(min_z_indices[16])) # highest number of modes lost
print(np.max(min_z_indices[16])) # lowest number of modes lost

plt.figure(figsize=(4,3))
plt.imshow(np.mean(perfect_CC_A,axis=0)[:,2:])
plt.colorbar()
plt.tight_layout()
plt.savefig('undistort_CC.png',dpi=600,bbox_inches='tight')
plt.show()

plt.figure(figsize=(4,3))
plt.imshow(np.std(CC_A[16],axis=0)[:,2:])
plt.colorbar()
plt.tight_layout()
plt.savefig('distort_CC_std.png',dpi=600,bbox_inches='tight')
plt.show()


# plt.figure(figsize=(4,4))
plt.plot(old_MEASURED_IM_SUBAP_SHIFT.flatten(),
         ANOTHER_AS[:,:,2:].flatten(),
         ls='',marker='.',alpha=0.05)
plt.axis('square')
plt.xlabel('Influence method (%)')
plt.ylabel('Slope method (%)')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.savefig('instant-influ-slope-.png',dpi=600)
plt.show()




# Plot
# plt.figure(figsize=(4,4))
plt.errorbar(unique_rytov_A,psd_ratio_rytov_A[:,0],yerr=psd_ratio_rytov_error_A[:,0],
             marker='o',label='{:.1f} cm'.format(dsub_A*100),capsize=5)
plt.errorbar(unique_rytov_B,psd_ratio_rytov_B[:,0],yerr=psd_ratio_rytov_error_B[:,0],
             marker='o',label='{:.1f} cm'.format(dsub_B*100),capsize=5)
plt.errorbar(unique_rytov_C,psd_ratio_rytov_C[:,0],yerr=psd_ratio_rytov_error_C[:,0],
             marker='o',label='{:.1f} cm'.format(dsub_C*100),capsize=5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Induced Rytov parameter')
plt.ylabel('PSD ratio (distort/undistort)')
plt.legend()
plt.savefig('PSD-ratio-rytov.pdf',bbox_inches='tight',dpi=600)
plt.show()


# plt.figure(figsize=(4,4))
plt.errorbar(unique_as_A,psd_ratio_as_A[:,0],yerr=psd_ratio_as_error_A[:,0],
             marker='o',label='{:.1f} cm'.format(dsub_A*100),capsize=5)
plt.errorbar(unique_as_B,psd_ratio_as_B[:,0],yerr=psd_ratio_as_error_B[:,0],
             marker='o',label='{:.1f} cm'.format(dsub_B*100),capsize=5)
plt.errorbar(unique_as_C,psd_ratio_as_C[:,0],yerr=psd_ratio_as_error_C[:,0],
             marker='o',label='{:.1f} cm'.format(dsub_C*100),capsize=5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('RMS actuator shift (%)')
plt.ylabel('PSD ratio (distort/undistort)')
plt.legend()
plt.savefig('PSD-ratio-as.pdf',bbox_inches='tight',dpi=600)
plt.show()


# '{:.0f}m$^-1$'.format(PSD_phase_frequency[0,0])
# Plot
# plt.figure(figsize=(4, 4))
plt.errorbar(unique_rytov_A,z_ratio_rytov_A,yerr=z_ratio_rytov_error_A,
             marker='o',label='{:.1f} cm'.format(dsub_A*100),capsize=5)
plt.errorbar(unique_rytov_B,z_ratio_rytov_B,yerr=z_ratio_rytov_error_B,
             marker='o',label='{:.1f} cm'.format(dsub_B*100),capsize=5)
plt.errorbar(unique_rytov_C,z_ratio_rytov_C,yerr=z_ratio_rytov_error_C,
             marker='o',label='{:.1f} cm'.format(dsub_C*100),capsize=5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Induced Rytov parameter')
plt.ylabel('Z power ratio (distort/undistort)')
plt.legend()
plt.savefig('z-ratio-rytov.pdf', dpi=600, bbox_inches='tight')
plt.show()

# Plot
# plt.figure(figsize=(4, 4))
plt.errorbar(unique_as_A,z_ratio_as_A,yerr=z_ratio_as_error_A,
             marker='o',label='{:.1f} cm'.format(dsub_A*100),capsize=5)
plt.errorbar(unique_as_B,z_ratio_as_B,yerr=z_ratio_as_error_B,
             marker='o',label='{:.1f} cm'.format(dsub_B*100),capsize=5)
plt.errorbar(unique_as_C,z_ratio_as_C,yerr=z_ratio_as_error_C,
             marker='o',label='{:.1f} cm'.format(dsub_C*100),capsize=5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('RMS actuator shift (%)')
plt.ylabel('Z power ratio (distort/undistort)')
plt.legend()
plt.savefig('z-ratio-as.pdf', dpi=600, bbox_inches='tight')
plt.show()

from aotools.functions import pupil
pupil_A = pupil.circle(RES_A.shape[-1]//2,RES_A.shape[-1])
pupil_A[pupil_A==0] = np.nan

x = np.arange(RES_A.shape[-1]) - RES_A.shape[-1]//2 - 0.5
x/=18
yy,xx = np.meshgrid(x,x)






# Extract the two datasets
data2 = my_unwrap(np.angle(RES_A[16, 10, -1])) * pupil_A / 2 / np.pi * 500
fig = plt.figure(figsize=(4,3))
vmax = np.nanmax(np.abs(data2))
vmin = - vmax

plt.pcolor(xx, yy, data2, vmin=vmin, vmax=vmax)
plt.hlines(np.linspace(-90/18, 90/18, 11), -90/18, 90/18, color='r')
plt.vlines(np.linspace(-90/18, 90/18, 11), -90/18, 90/18, color='r')
plt.axis('square')
plt.xlim(-90/18, 90/18)
plt.ylim(-90/18, 90/18)

plt.colorbar()
plt.savefig('waffle_mode.png', dpi=600, bbox_inches='tight')
plt.show()






# # Step 1: Combine all datasets
x_all = np.concatenate([IM_RMS_A[1:], IM_RMS_B[1:], IM_RMS_C[1:]])
y_all = np.concatenate([WFE_A_control[1:], WFE_B_control[1:], WFE_C_control[1:]])
xerr_all = np.concatenate([IM_RMS_A_error[1:], IM_RMS_B_error[1:], IM_RMS_C_error[1:]])
yerr_all = np.concatenate([WFE_A_error[1:], WFE_B_error[1:], WFE_C_error[1:]])
# Step 1: Combine all datasets
# x_all = np.concatenate([IM_RMS_B[1:], IM_RMS_C[1:]])
# y_all = np.concatenate([WFE_B_control[1:], WFE_C_control[1:]])
# xerr_all = np.concatenate([IM_RMS_B_error[1:], IM_RMS_C_error[1:]])
# yerr_all = np.concatenate([WFE_B_error[1:], WFE_C_error[1:]])

# Step 2: Define bins (e.g., logarithmic bins)
num_bins = 10
bins = np.logspace(np.log10(np.nanmin(x_all)), np.log10(np.nanmax(x_all)), num_bins + 1)

# Step 3: Bin data
bin_centers = []
y_means = []
xerr_means = []
yerr_means = []

for i in range(num_bins):
    bin_mask = (x_all >= bins[i]) & (x_all < bins[i+1])
    
    if np.any(bin_mask):  # skip empty bins
        x_bin = x_all[bin_mask]
        y_bin = y_all[bin_mask]
        xerr_bin = xerr_all[bin_mask]
        yerr_bin = yerr_all[bin_mask]

        bin_centers.append(np.nanmedian(x_bin))  # or np.mean
        y_means.append(np.nanmean(y_bin))
        xerr_means.append(np.sqrt(np.nanmean(xerr_bin**2)))  # RMS error
        yerr_means.append(np.sqrt(np.nanmean(yerr_bin**2)))  # RMS error

# Step 4: Plot
# plt.figure(figsize=(4,4))
plt.errorbar(bin_centers, y_means, xerr=xerr_means, yerr=yerr_means,
             ls='', marker='o', capsize=5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Interaction matrices error indices')
plt.ylabel('Control error (nm)')
# plt.grid(True, which='both')
# plt.savefig('im-vs-control.pdf',bbox_inches='tight',dpi=600)
plt.show()




# Step 1: Combine all datasets
x_all = np.concatenate([CC_RMS_A[1:],
                        CC_RMS_B[1:], CC_RMS_C[1:]])
y_all = np.concatenate([WFE_A_control[1:],
                        WFE_B_control[1:], WFE_C_control[1:]])
xerr_all = np.concatenate([CC_RMS_A_error[1:],
                           CC_RMS_B_error[1:], CC_RMS_C_error[1:]])
yerr_all = np.concatenate([WFE_A_error[1:],
                           WFE_B_error[1:], WFE_C_error[1:]])
# x_all = np.concatenate([CC_RMS_B[1:], CC_RMS_C[1:]])
# y_all = np.concatenate([WFE_B_control[1:], WFE_C_control[1:]])
# xerr_all = np.concatenate([CC_RMS_B_error[1:], CC_RMS_C_error[1:]])
# yerr_all = np.concatenate([WFE_B_error[1:], WFE_C_error[1:]])

# Step 2: Define bins (e.g., logarithmic bins)
num_bins = 10
bins = np.logspace(np.log10(np.nanmin(x_all)), np.log10(np.nanmax(x_all)), num_bins + 1)

# Step 3: Bin data
bin_centers = []
y_means = []
xerr_means = []
yerr_means = []

for i in range(num_bins):
    bin_mask = (x_all >= bins[i]) & (x_all < bins[i+1])
    
    if np.any(bin_mask):  # skip empty bins
        x_bin = x_all[bin_mask]
        y_bin = y_all[bin_mask]
        xerr_bin = xerr_all[bin_mask]
        yerr_bin = yerr_all[bin_mask]

        bin_centers.append(np.nanmedian(x_bin))  # or np.mean
        y_means.append(np.nanmean(y_bin))
        xerr_means.append(np.sqrt(np.nanmean(xerr_bin**2)))  # RMS error
        yerr_means.append(np.sqrt(np.nanmean(yerr_bin**2)))  # RMS error

# Step 4: Plot
# plt.figure(figsize=(4,4))
plt.errorbar(bin_centers, y_means, xerr=xerr_means, yerr=yerr_means,
             ls='', marker='o', capsize=5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Control matrices error indices')
plt.ylabel('Control error (nm)')
# plt.grid(True, which='both')
# plt.savefig('cm-vs-control.pdf',bbox_inches='tight',dpi=600)
plt.show()


def bin_and_plot(x, y, xerr, yerr,minx,maxx, label=None, color=None, num_bins=10):
    # Remove NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    xerr = xerr[mask]
    yerr = yerr[mask]

    # Define bins (logarithmic)
    bins = np.logspace(np.log10(minx), np.log10(maxx), num_bins + 1)

    # Bin data
    bin_centers = []
    y_means = []
    xerr_means = []
    yerr_means = []

    for i in range(num_bins):
        bin_mask = (x >= bins[i]) & (x < bins[i+1])
        if np.any(bin_mask):
            x_bin = x[bin_mask]
            y_bin = y[bin_mask]
            xerr_bin = xerr[bin_mask]
            yerr_bin = yerr[bin_mask]

            bin_centers.append(np.nanmedian(x_bin))  # or np.nanmean
            y_means.append(np.nanmean(y_bin))
            xerr_means.append(np.sqrt(np.nanmean(xerr_bin**2)))
            yerr_means.append(np.sqrt(np.nanmean(yerr_bin**2)))

    # Plot
    plt.errorbar(bin_centers, y_means,# xerr=xerr_means,
                 yerr=yerr_means,
                 ls='-', marker='o', capsize=5, label=label, color=color)
    
def bin_and_plot_y(x, y, yerr,minx,maxx, label=None, color=None, num_bins=10):
    # Remove NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    yerr = yerr[mask]

    # Define bins (logarithmic)
    bins = np.logspace(np.log10(minx), np.log10(maxx), num_bins + 1)

    # Bin data
    bin_centers = []
    y_means = []
    yerr_means = []

    for i in range(num_bins):
        bin_mask = (x >= bins[i]) & (x < bins[i+1])
        if np.any(bin_mask):
            x_bin = x[bin_mask]
            y_bin = y[bin_mask]
            yerr_bin = yerr[bin_mask]

            bin_centers.append(np.nanmedian(x_bin))  # or np.nanmean
            y_means.append(np.nanmean(y_bin))
            yerr_means.append(np.sqrt(np.nanmean(yerr_bin**2)))

    # Plot
    plt.errorbar(bin_centers, y_means, yerr=yerr_means,
                 ls='-', marker='o', capsize=5, label=label, color=color)

# Bin and plot A, B, C separately
# plt.figure(figsize=(4,4))
bin_and_plot(CC_RMS_A[1:]*100, WFE_A_control[1:], CC_RMS_A_error[1:]*100, WFE_A_error[1:],
             1,100,
             label='Subap.={:.1f}cm'.format(dsub_A*100))
bin_and_plot(CC_RMS_B[1:]*100, WFE_B_control[1:], CC_RMS_B_error[1:]*100, WFE_B_error[1:],
             1,100,
             label='Subap.={:.1f}cm'.format(dsub_B*100))
bin_and_plot(CC_RMS_C[1:]*100, WFE_C_control[1:], CC_RMS_C_error[1:]*100, WFE_C_error[1:],
             1,100,
             label='Subap.={:.1f}cm'.format(dsub_C*100))

# Final plot formatting
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Control matrix error (%)')
plt.ylabel('Control error (nm)')
plt.legend()
# plt.grid(True, which='both')
plt.tight_layout()
plt.savefig('cm-vs-control.pdf',bbox_inches='tight',dpi=600)
plt.show()

# plt.figure(figsize=(4,4))
bin_and_plot(IM_RMS_A[1:]*100, WFE_A_control[1:], IM_RMS_A_error[1:]*100, WFE_A_error[1:],
             1,100,
             label='Subap.={:.1f}cm'.format(dsub_A*100))
bin_and_plot(IM_RMS_B[1:]*100, WFE_B_control[1:], IM_RMS_B_error[1:]*100, WFE_B_error[1:],
             1,100,
             label='Subap.={:.1f}cm'.format(dsub_B*100))
bin_and_plot(IM_RMS_C[1:]*100, WFE_C_control[1:], IM_RMS_C_error[1:]*100, WFE_C_error[1:],
             1,100,
             label='Subap.={:.1f}cm'.format(dsub_C*100))

# Final plot formatting
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Interaction matrix error (%)')
plt.ylabel('Control error (nm)')
plt.legend()
# plt.grid(True, which='both')
plt.tight_layout()
plt.savefig('im-vs-control.pdf',bbox_inches='tight',dpi=600)
plt.show()




fix_WFE = np.zeros((5,50),dtype=float)
fix_perfect_WFE = np.zeros((5,50),dtype=float)

fix_name = ['rytov0.04-AS8%',
            'rytov0.02-AS8%',
            'rytov0.04-AS14%',
            'rytov0.01-AS8%',
            'rytov0.04-AS23%']

for i in range(len(fix_name)):
    fix_name_i = fix_name[i]
    hdul = fits.open(r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf\benchmark-rcond0.01-A-case1-2\WFE_bad_IM_A-' + fix_name_i + '.fits')
    data = hdul[0].data
    fix_WFE[i] = data[:,-1]
    hdul = fits.open(r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf\benchmark-rcond0.01-A-case1-2\perfect_WFE_bad_IM_A-' + fix_name_i + '.fits')
    data = hdul[0].data
    fix_perfect_WFE[i] = data[:,-1]


fix_avg_terminal_WFE_only_control = (np.nanmean((fix_WFE**2 - fix_perfect_WFE**2)**0.5,axis=-1))
fix_avg_terminal_WFE_only_control_error = (np.nanstd((fix_WFE**2 - fix_perfect_WFE**2)**0.5,axis=-1))

fix_arg_rytov = np.array([0,2,4],dtype=int)
fix_arg_as = np.array([1,3],dtype=int)


plt.figure(figsize=(4,4))
bin_and_plot(IM_RMS_A[1:]*100, WFE_A_control[1:], IM_RMS_A_error[1:]*100, WFE_A_error[1:],
             1,100,
             label='{:.1f} cm'.format(dsub_A*100))
bin_and_plot(IM_RMS_B[1:]*100, WFE_B_control[1:], IM_RMS_B_error[1:]*100, WFE_B_error[1:],
             1,100,
             label='{:.1f} cm'.format(dsub_B*100))
bin_and_plot(IM_RMS_C[1:]*100, WFE_C_control[1:], IM_RMS_C_error[1:]*100, WFE_C_error[1:],
             1,100,
             label='{:.1f} cm'.format(dsub_C*100))

# bin_and_plot(np.array([IM_RMS_B[1:],IM_RMS_C[1:]])*100,
#              np.array([WFE_B_control[1:],WFE_C_control[1:]]),
#              np.array([IM_RMS_B_error[1:],IM_RMS_C_error[1:]])*100,
#              np.array([WFE_B_error[1:],WFE_C_error[1:]]),label='5.0 & 6.7 cm',color=color_list[0])


plt.errorbar(IM_RMS_A[2]*100,fix_avg_terminal_WFE_only_control[0],
         xerr=IM_RMS_A_error[1]*100,yerr=fix_avg_terminal_WFE_only_control_error[0],
         marker='o',capsize=5,label='4.0 cm - 0.01 rcond')

# Final plot formatting
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Interaction matrix error (%)')
plt.ylabel('Control error (nm)')
plt.legend()
# plt.grid(True, which='both')
plt.tight_layout()
plt.savefig('fix-im-vs-control.pdf',bbox_inches='tight',dpi=600)
plt.show()





plt.errorbar(unique_expect_rytov_A,mean_control_A,yerr=error_control_A,#/np.sqrt(TRIAL),
              ls='-',marker='o',label='{:.1f} cm'.format(dsub_A*100),capsize=5)
plt.errorbar(unique_expect_rytov_B,mean_control_B,yerr=error_control_B,#/np.sqrt(TRIAL),
              ls='-',marker='o',label='{:.1f} cm'.format(dsub_B*100),capsize=5)
plt.errorbar(unique_expect_rytov_C,mean_control_C,yerr=error_control_C,#/np.sqrt(TRIAL),
              ls='-',marker='o',label='{:.1f} cm'.format(dsub_C*100),capsize=5)

# bin_and_plot_y(np.array([unique_expect_rytov_B[1:],unique_expect_rytov_C[1:]]),
#              np.array([mean_control_B[1:],mean_control_C[1:]]),
#              np.array([error_control_B[1:],error_control_C[1:]]),label='5.0 & 6.7 cm')

mean_fix,std_fix = np.average(fix_avg_terminal_WFE_only_control[fix_arg_rytov],
                              weights=1/(fix_avg_terminal_WFE_only_control_error[fix_arg_rytov])**2,returned=True)
std_fix = (1/std_fix)**0.5
plt.errorbar(unique_expect_rytov_A[1],mean_fix,
          yerr=std_fix,
          marker='o',capsize=5,label='4.0 cm - 0.01 rcond')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Control error (nm)')
plt.xlabel('Expected $\sigma^2_{R,DM}$')
plt.legend()
plt.ylim(1,600)
plt.savefig('fix-compare-control-rytov-simple.pdf', bbox_inches='tight')
plt.show()





plt.errorbar(unique_expect_AS_A*100,mean_control_A,yerr=error_control_A,#/np.sqrt(TRIAL),
              ls='-',marker='o',label='{:.1f} cm'.format(dsub_A*100),capsize=5)
plt.errorbar(unique_expect_AS_B*100,mean_control_B,yerr=error_control_B,#/np.sqrt(TRIAL),
              ls='-',marker='o',label='{:.1f} cm'.format(dsub_B*100),capsize=5)
plt.errorbar(unique_expect_AS_C*100,mean_control_C,yerr=error_control_C,#/np.sqrt(TRIAL),
              ls='-',marker='o',label='{:.1f} cm'.format(dsub_C*100),capsize=5)

# bin_and_plot_y(np.array([unique_expect_AS_B[1:],unique_expect_AS_C[1:]])*100,
#              np.array([mean_control_B[1:],mean_control_C[1:]]),
#              np.array([error_control_B[1:],error_control_C[1:]]),label='5.0 & 6.7 cm')

mean_fix,std_fix = np.average(fix_avg_terminal_WFE_only_control[fix_arg_as],
                              weights=1/(fix_avg_terminal_WFE_only_control_error[fix_arg_as])**2,returned=True)
std_fix = (1/std_fix)**0.5
plt.errorbar(unique_expect_AS_A[1]*100,mean_fix,
         yerr=std_fix,
         marker='o',capsize=5,label='4.0 cm - 0.01 rcond')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Control error (nm)')
plt.xlabel('Expected $RMS_{AS}$ (%)')
plt.ylim(1,600)
plt.legend()
plt.savefig('fix-compare-control-as-simple.pdf', bbox_inches='tight')
plt.show()



