# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:54:01 2024

@author: gqmc62
"""

###############################################################################

# # TYPE = 'C' # 12 nxsubap
# # TYPE = 'A' # 8 nxsubap
# #TYPE = 'B' # 4 nxsubap

# # TYPELIST=['A','B']
# TYPELIST=['C','A','B']
# # TYPELIST = ['A']

# HOME = r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter5Plots\FinalData'

# for TYPE in TYPELIST:
    
    
#     slodarRefDir = r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter5Plots\FinalData\RefSLODAR'
    
#     if (TYPE == 'A'):
#         SIMNAME = 'bad_IM_A-'
#         home = r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter5Plots\FinalData\rcond005\A' # bad_IM before fix
#         slopeCovName = 'slopeCovariance_A-vk.fits'
#         refFuncName = 'referenceFunctions_A-vk.fits'
#     elif TYPE == 'C':
#         SIMNAME = 'bad_IM_C-'
#         home = r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter5Plots\FinalData\rcond005\C' #bad_IM_C
#         slopeCovName = 'slopeCovariance_C-vk.fits'
#         refFuncName = 'referenceFunctions_C-vk.fits'
#     elif TYPE == 'B':
#         SIMNAME = 'bad_IM_B-'
#         home = r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter5Plots\FinalData\rcond005\B' #bad_IM_B
#         # home = r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf\2025-03-24-19-32-05-B-final' #bad_IM_B
#         slopeCovName = 'slopeCovariance_B-vk.fits'
#         refFuncName = 'referenceFunctions_B-vk.fits'
    
#     # home = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\2025-03-12-19-57-38'
#     # home = r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf\2025-03-18-20-27-50'
    
    
#     dsub_B = 0.4/4
#     dsub_A = 0.4/8
#     dsub_C = 0.4/12
    
###############################################################################

# TYPELIST=['A','B','C'] # 10,8,6

# HOME = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf'

# for TYPE in TYPELIST:
    
    
#     slodarRefDir = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf'
    
#     if (TYPE == 'A'):
#         SIMNAME = 'bad_IM_A-'
#         home = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\Final\A' # bad_IM before fix
#         # slopeCovName = 'slopeCovariance_A-vk.fits'
#         refFuncName = 'slodarRefFunc4cmx10subapVK4mL0.fits'
#     elif TYPE == 'B':
#         SIMNAME = 'bad_IM_B-'
#         home = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\Final\B' #bad_IM_B
#         # home = r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf\2025-03-24-19-32-05-B-final' #bad_IM_B
#         # slopeCovName = 'slopeCovariance_B-vk.fits'
#         refFuncName = 'slodarRefFunc5cmx8subapVK4mL0.fits'
#     elif TYPE == 'C':
#         SIMNAME = 'bad_IM_C-'
#         home = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\Final\C' #bad_IM_C
#         # slopeCovName = 'slopeCovariance_C-vk.fits'
#         refFuncName = 'slodarRefFunc6-6cmx6subapVK4mL0.fits'
        
#     dsub_A = 0.4/10
#     dsub_B = 0.4/8
#     dsub_C = 0.4/6

###############################################################################
    
TYPELIST=['A','B','C'] # 10,8,6

HOME = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\FinalSupreme\conf'

for TYPE in TYPELIST:
    
    
    slodarRefDir = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf'
    
    if (TYPE == 'A'):
        SIMNAME = 'bad_IM_A-'
        home = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\FinalSupreme\conf\A' # bad_IM before fix
        # slopeCovName = 'slopeCovariance_A-vk.fits'
        refFuncName = 'slodarRefFunc4cmx10subapVK4mL0.fits'
    elif TYPE == 'B':
        SIMNAME = 'bad_IM_B-'
        home = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\FinalSupreme\conf\B' #bad_IM_B
        # home = r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf\2025-03-24-19-32-05-B-final' #bad_IM_B
        # slopeCovName = 'slopeCovariance_B-vk.fits'
        refFuncName = 'slodarRefFunc5cmx8subapVK4mL0.fits'
    elif TYPE == 'C':
        SIMNAME = 'bad_IM_C-'
        home = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\LastChapter5Data\conf\FinalSupreme\conf\C' #bad_IM_C
        # slopeCovName = 'slopeCovariance_C-vk.fits'
        refFuncName = 'slodarRefFunc6-6cmx6subapVK4mL0.fits'
        
    dsub_A = 0.4/10
    dsub_B = 0.4/8
    dsub_C = 0.4/6


###############################################################################
    
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
    
    # def old_displacement(wvl,dsub,r0,z):
    #     displace = angle_of_arrival(wvl,dsub,r0) * z/dsub
    #     return displace
    
    # # wrong
    # def angle_of_arrival(wvl,dact_immediate_dm,r0,Dimmediate_dm,L0):
    #     FACTOR1 = 0.17819635611424933
    #     tiptilt_adjust = mps.tiptilt_variance_adjust_to_vK(dact_immediate_dm, np.min([Dimmediate_dm,L0]))
    #     FACTOR = (FACTOR1*tiptilt_adjust)**0.5
    #     alpha = np.sqrt(0.162 * wvl**2 * dact_immediate_dm**(-1./3.) * r0**(-5./3.))
    #     alpha *= FACTOR
    #     return alpha
    
    # def displacement(wvl,dact_immediate_dm,dact_next_immediate_dm,r0,z,Dimmediate_dm,L0):
    #     displace = angle_of_arrival(wvl,dact_immediate_dm,r0,Dimmediate_dm,L0) * z/dact_next_immediate_dm
    #     return displace
    
    
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
    
    
    as_threshold = 0.5#0.25
    
    
    # name_list = np.array([
    #     'rytov0.00-AS0%',
    #     'rytov0.01-AS4%',
    #     'rytov0.01-AS7%',
    #     'rytov0.02-AS7%',
    #     'rytov0.02-AS11%',
    #     'rytov0.04-AS7%',
    #     'rytov0.04-AS11%',
    #     'rytov0.04-AS18%',
    #     'rytov0.08-AS11%',
    #     'rytov0.08-AS18%',
    #     'rytov0.15-AS18%',
    #     'rytov0.15-AS29%',
    #     'rytov0.30-AS18%',
    #     'rytov0.30-AS29%',
    #     'rytov0.30-AS48%',
    #     'rytov0.60-AS48%',
    #     'rytov0.60-AS79%'
    #     ])
    
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
    
    for i in np.arange(expect_rytov.shape[0]):
        expect_rytov[i] = float(name_list[i][5:9])
        expect_AS[i] = float(name_list[i][12:-1])/100.
    
        with open(SIMNAME + name_list[i] + '.yaml','r') as file:
            config = yaml.safe_load(file)
            z[i] = config['DM'][2]['altitude']
            r0HA[i] = config['DM'][2]['r0']
    
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
    
    expect_AS = (0.162*wvl**2*dsub**(-1/3)*r0HA**(-5/3))**0.5*z/dsub
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
    
    # ###############################################################################
    # # Sampling points
    # ##############################################################################
    
    # plt.plot(expect_rytov,old_expect_AS,marker='o',ls='',color='r')
    # plt.xlabel('$\sigma_\chi^2$')
    # plt.ylabel('AS (subap)')
    # plt.title('Sampled points on previously thought spaces')
    # # plt.legend()
    # plt.xscale('log')
    # plt.yscale('log')
    # # plt.xlim(0.1,0.6)
    # # plt.ylim(0.1,1.0)
    # # plt.axis('square')
    # plt.show()
    
    
    
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
    
    
    #############################################################################
    # elif dm.n_acts == 121:
    #     FULL_ACTS = numpy.array([28,
    #                              37,38,39,40,41,
    #                              48,49,50,51,52,
    #                              58,59,60,61,62,63,64,
    #                              70,71,72,73,74,
    #                              81,82,83,84,85,
    #                              94],dtype=int) - 1
    #     position = numpy.array([[0., 4.],[0., 5.],
    #                              [1., 2.],[1., 3.],[1., 4.],[1., 5.],[1., 6.],[1., 7.],
    #                              [2., 1.],[2., 2.],[2., 3.],[2., 4.],[2., 5.],[2., 6.],[2., 7.],[2., 8.],
    #                              [3., 1.],[3., 2.],[3., 3.],[3., 4.],[3., 5.],[3., 6.],[3., 7.],[3., 8.],
    #                              [4., 0.],[4., 1.],[4., 2.],[4., 3.],[4., 4.],[4., 5.],[4., 6.],[4., 7.],[4., 8.],[4., 9.],
    #                              [5., 0.],[5., 1.],[5., 2.],[5., 3.],[5., 4.],[5., 5.],[5., 6.],[5., 7.],[5., 8.],[5., 9.],
    #                              [6., 1.],[6., 2.],[6., 3.],[6., 4.],[6., 5.],[6., 6.],[6., 7.],[6., 8.],
    #                              [7., 1.],[7., 2.],[7., 3.],[7., 4.],[7., 5.],[7., 6.],[7., 7.],[7., 8.],
    #                              [8., 2.],[8., 3.],[8., 4.],[8., 5.],[8., 6.],[8., 7.],
    #                              [9., 4.],[9., 5.]],dtype=int)
    #     ACT_LIST = numpy.array([17,
    #                             26,27,28,29,30,
    #                             36,37,38,39,40,41,42,
    #                             47,48,49,50,51,52,53,
    #                             57,58,59,60,61,62,63,64,65,
    #                             69,70,71,72,73,74,75,
    #                             80,81,82,83,84,85,86,
    #                             92,93,94,95,96,
    #                             116]) - 1
    #     DM_POS_TEMP = numpy.array([[0.5,4.5],
    #                                [1.5,2.5],[1.5,3.5],[1.5,4.5],[1.5,5.5],[1.5,6.5],
    #                                [2.5,1.5],[2.5,2.5],[2.5,3.5],[2.5,4.5],[2.5,5.5],[2.5,6.5],[2.5,7.5],
    #                                [3.5,1.5],[3.5,2.5],[3.5,3.5],[3.5,4.5],[3.5,5.5],[3.5,6.5],[3.5,7.5],
    #                                [4.5,0.5],[4.5,1.5],[4.5,2.5],[4.5,3.5],[4.5,4.5],[4.5,5.5],[4.5,6.5],[4.5,7.5],[4.5,8.5],
    #                                [5.5,1.5],[5.5,2.5],[5.5,3.5],[5.5,4.5],[5.5,5.5],[5.5,6.5],[5.5,7.5],
    #                                [6.5,1.5],[6.5,2.5],[6.5,3.5],[6.5,4.5],[6.5,5.5],[6.5,6.5],[6.5,7.5],
    #                                [7.5,2.5],[7.5,3.5],[7.5,4.5],[7.5,5.5],[7.5,6.5],
    #                                [8.5,4.5]])
        
    # elif dm.n_acts == 49:
    #     FULL_ACTS = numpy.array([17,
    #                              23,24,25,
    #                              31],dtype=int)
    #     ACT_LIST = numpy.array([10,
    #                              16,17,18,
    #                              22,23,24,25,26,
    #                              30,31,32,
    #                              38],dtype=int)
    #     position = numpy.array([[0,2],[0,3],
    #                             [1,1],[1,2],[1,3],[1,4],
    #                             [2,0],[2,1],[2,2],[2,3],[2,4],[2,5],
    #                             [3,0],[3,1],[3,2],[3,3],[3,4],[3,5],
    #                             [4,1],[4,2],[4,3],[4,4],
    #                             [5,2],[5,3]],
    #                            dtype=int)
    #     DM_POS_TEMP = numpy.array([[0.5,2.5],
    #                                [1.5,1.5],[1.5,2.5],[1.5,3.5],
    #                                [2.5,0.5],[2.5,1.5],[2.5,2.5],[2.5,3.5],[2.5,4.5],
    #                                [3.5,1.5],[3.5,2.5],[3.5,3.5],
    #                                [4.5,2.5]])
    
    
    ###############################################################################
    
    
    ANOTHER_AS = np.zeros((CASES,TRIAL,IM.shape[2],2),dtype=float)
    
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
    
    # ###############################################################################
    # # Parameter error
    # ###############################################################################
    
    # old_MEASURED_IM_SUBAP_SHIFT = np.copy(MEASURED_IM_SUBAP_SHIFT)
    # old_temp = np.nanstd(MEASURED_IM_SUBAP_SHIFT,axis=(-1,-2,-3))#np.sqrt(np.nanmean(MEASURED_IM_SUBAP_SHIFT**2,axis=(-1,-2,-3)))
    # old_COMBINED_MEASURED_IM_AS = np.nanmean(old_temp,axis=-1)
    # old_COMBINED_MEASURED_IM_AS_error = np.nanstd(old_temp,axis=-1)
    
    # temp_rytov =  np.nanmean(MEASURED_IM_RYTOV,axis=(-1,-2))
    # COMBINED_MEASURED_IM_RYTOV = np.nanmean(temp_rytov,axis=(-1))
    # AS_R = np.nanstd(ANOTHER_AS,axis=(-1,-2))#np.sqrt(np.nanmean(ANOTHER_AS**2,axis=(-1,-2)))
    # COMBINED_MEASURED_IM_AS = np.nanmean(AS_R,axis=(-1))
    
    old_MEASURED_IM_SUBAP_SHIFT = np.copy(MEASURED_IM_SUBAP_SHIFT)
    old_temp = np.nanstd(MEASURED_IM_SUBAP_SHIFT,axis=(-1,-2))#np.sqrt(np.nanmean(MEASURED_IM_SUBAP_SHIFT**2,axis=(-1,-2,-3)))
    old_COMBINED_MEASURED_IM_AS = np.nanmean(old_temp**2,axis=-1)**0.5
    old_COMBINED_MEASURED_IM_AS_error = np.nanstd(old_temp**2,axis=-1)**0.5/2.
    
    # # old_COMBINED_MEASURED_IM_AS = (old_COMBINED_MEASURED_IM_AS - old_COMBINED_MEASURED_IM_AS[0])
    # old_COMBINED_MEASURED_IM_AS = (old_COMBINED_MEASURED_IM_AS**2 - old_COMBINED_MEASURED_IM_AS[0]**2)**0.5
    
    # # old_COMBINED_MEASURED_IM_AS_error = (old_COMBINED_MEASURED_IM_AS_error**2 + old_COMBINED_MEASURED_IM_AS_error[0]**2)**0.5
    # old_COMBINED_MEASURED_IM_AS_error = old_COMBINED_MEASURED_IM_AS_error + old_COMBINED_MEASURED_IM_AS_error[0]
    
    temp_rytov =  np.nanmean(MEASURED_IM_RYTOV,axis=(-1,-2))
    COMBINED_MEASURED_IM_RYTOV = np.nanmean(temp_rytov,axis=(-1))
    error_rytov = np.nanstd(temp_rytov,axis=(-1))
    # COMBINED_MEASURED_IM_RYTOV = (COMBINED_MEASURED_IM_RYTOV - COMBINED_MEASURED_IM_RYTOV[0])
    # error_rytov = error_rytov + error_rytov[0]
    
    AS_R = np.nanstd(ANOTHER_AS[:,:,2:],axis=(-1,-2))#np.sqrt(np.nanmean(ANOTHER_AS**2,axis=(-1,-2)))#
    COMBINED_MEASURED_IM_AS = np.nanmean(AS_R**2,axis=(-1))**0.5
    error_as = np.nanstd(AS_R**2,axis=(-1))**0.5/2
    
    # COMBINED_MEASURED_IM_AS = (COMBINED_MEASURED_IM_AS**2 - COMBINED_MEASURED_IM_AS[0]**2)**0.5
    # # COMBINED_MEASURED_IM_AS = (COMBINED_MEASURED_IM_AS - COMBINED_MEASURED_IM_AS[0])
    # error_as = error_as + error_as[0]
    
    # error_rytov = np.nanstd(temp_rytov,axis=(-1))#/np.sqrt(2*(np.sum(~np.isnan(temp_rytov),axis=(-1))) - 2)
    # error_as = np.nanstd(AS_R,axis=(-1))#/np.sqrt(2*(np.sum(~np.isnan(AS_R),axis=(-1))) - 2)
    
    # plt.plot([expect_rytov[1:],COMBINED_MEASURED_IM_RYTOV[1:]],[old_expect_AS[1:],old_COMBINED_MEASURED_IM_AS[1:]],ls='-',marker='',color='k')
    # plt.plot(expect_rytov[1:],old_expect_AS[1:],marker='o',ls='',color='r',label='Expected')
    # plt.errorbar(COMBINED_MEASURED_IM_RYTOV[1:],old_COMBINED_MEASURED_IM_AS[1:],
    #              xerr=error_rytov[1:],yerr=old_COMBINED_MEASURED_IM_AS_error[1:],capsize=5,
    #              ls='',marker='o',label='Measured')
    # # plt.arrow(expect_rytov[1:],old_expect_AS[1:],
    # #           (COMBINED_MEASURED_IM_RYTOV-expect_rytov)[1:],(COMBINED_MEASURED_IM_AS-old_expect_AS)[1:],
    # #           ls='-',marker='',color='k')
    # plt.xlabel('$\sigma_\chi^2$')
    # plt.ylabel('AS (subap)')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title('before fixing error')
    # plt.legend()
    # # plt.xlim(0.1,0.6)
    # # plt.ylim(0.1,1.0)
    # plt.show()
    
    
    
    ###############################################################################
    # Parameter error - Fixed
    ###############################################################################
    
    #######################################
    # original after spie method # the one that match
    ######################################
    # FACTOR1 = ((NOLL[0] - NOLL[2])/2.)/sum(NOLL) # THIS IS WRONG
    # tiptilt_adjust = mps.tiptilt_variance_adjust_to_vK(dsub, np.min([D,L0]))
    # FACTOR = (FACTOR1*tiptilt_adjust)**0.5
    # expect_AS = FACTOR * old_expect_AS
    #######################################
    # NOLL = [1.0299,0.582,0.134,0.111,0.0880,0.0648,0.0587,0.0525,0.0463,0.0401,0.0377,0.0352,0.0328,0.0304,0.0279,0.0267,0.0255,0.0243,0.0232,0.0220,0.0208]
    # FACTOR1 = ((NOLL[0] - NOLL[2])/2.)/sum(NOLL) # THIS IS WRONG
    # tiptilt_adjust = mps.tiptilt_variance_adjust_to_vK(dsub, np.min([D,L0]))
    # tiptilt_kolmogorov_subap = old_expect_AS**2 * FACTOR1
    # tiptilt_von_karman_subap = tiptilt_kolmogorov_subap*tiptilt_adjust
    # expect_AS = (tiptilt_von_karman_subap)**0.5
    #####################################################
    
    ###############################################################################
    # more correct of old method
    ###############################################################################
    NOLL = [1.0299,0.582,0.134,0.111,0.0880,0.0648,0.0587,0.0525,0.0463,0.0401,0.0377,0.0352,0.0328,0.0304,0.0279,0.0267,0.0255,0.0243,0.0232,0.0220,0.0208]
    # kolmogorov_subap_all = old_expect_AS**2
    # kolmogorov_tiptilt_ratio = (NOLL[0] - NOLL[2])/NOLL[0]
    # kolmogorov_subap_tiptilt = kolmogorov_subap_all * kolmogorov_tiptilt_ratio
    # tiptilt_adjust = mps.tiptilt_variance_adjust_to_vK(dsub, np.min([D,L0]))
    # von_karman_subap_tiptilt = kolmogorov_subap_tiptilt * tiptilt_adjust
    # expect_AS = von_karman_subap_tiptilt**0.5
    ###############################################################################
    
    # aoa_var_subap_with_tiptilt_circ = old_expect_AS**2
    # aoa_var_subap_with_tiptilt_sqr = old_expect_AS**2 * 0.162/0.182 # i was using circular subap. it should be square
    
    # aoa_var_ap_full_sqr = aoa_var_subap_with_tiptilt_sqr * (0.4/0.05)**(-1/3)
    # aoa_var_ap_full_circ = aoa_var_ap_full_sqr / (0.162/0.182)
    
    # aoa_var_subap_no_tiptilt_sqr = aoa_var_subap_with_tiptilt_sqr - aoa_var_ap_full_circ
    # expect_AS = aoa_var_subap_no_tiptilt_sqr**0.5
    
    ###########
    #expect_AS = ( old_expect_AS**2 / 0.182 * ((1.0299 - 0.134) /2/3.14 * (0.05/0.4)**(1/3)) )**0.5
    
    import os
    
    os.chdir(slodarRefDir)
    from astropy.io import fits
    from matplotlib import pyplot as plt
    import numpy as np
    
    wvlref = 500e-9
    k = 2*np.pi/wvlref
    # dsub = 0.05
    r0_ref = dsub
    
    
    # hdul = fits.open(slopeCovName)
    # data1 = hdul[0].data
    # hdul.close()
    
    hdul = fits.open(refFuncName)
    data2 = hdul[0].data
    hdul.close()
    
    mid = (data2.shape[1] - 1) // 2 
    # A = (data1[0,mid,mid] + data1[1,mid,mid]) * (r0_ref)**(5/3)*dsub**(1/3)*wvl**-2 / 206265**2
    # B = (data2[0,mid,mid] + data2[1,mid,mid]) * (r0_ref)**(5/3)*dsub**(1/3)*wvl**-2 / 206265**2
    # A = (data1[0,mid,mid]) * (r0_ref)**(5/3)*dsub**(1/3)*wvlref**-2 / 206265**2
    B = (data2[0,mid,mid]) * (r0_ref)**(5/3)*dsub**(1/3)*wvlref**-2 / 206265**2
    
    
    
    new_coeff = B              #0.025501816814832663# from butterly's refFunc mid 0th layer either L or T
    
    expect_AS = (new_coeff * wvl**2 * dsub**(-1/3) * r0HA**(-5/3))**0.5 * z / dsub
    
    # expect_AS = np.nan
    old_name_list = np.copy(name_list)
    name_list = []
    for cases in np.arange(CASES):
        name_list.append('rytov{:.2f}-AS{:.0f}%'.format(expect_rytov[cases],expect_AS[cases]*100))
    
    # COMBINED_MEASURED_IM_RYTOV = np.nanmean(MEASURED_IM_RYTOV,axis=(-1,-2,-3))
    # COMBINED_MEASURED_IM_AS = np.nanmean(np.abs(MEASURED_IM_SUBAP_SHIFT),axis=(-1,-2,-3,-4))
    
    
    
    # plt.plot([expect_rytov[1:],COMBINED_MEASURED_IM_RYTOV[1:]],[expect_AS[1:],old_COMBINED_MEASURED_IM_AS[1:]],ls='-',marker='',color='k')
    # plt.plot(expect_rytov[1:],expect_AS[1:],marker='o',ls='',color='r',label='Expected')
    # plt.errorbar(COMBINED_MEASURED_IM_RYTOV[1:],old_COMBINED_MEASURED_IM_AS[1:],
    #          xerr=error_rytov[1:],yerr=old_COMBINED_MEASURED_IM_AS_error[1:],capsize=5,
    #          ls='',marker='o',label='Measured')
    # # plt.arrow(expect_rytov[1:],expect_AS[1:],
    # #           (COMBINED_MEASURED_IM_RYTOV-expect_rytov)[1:],(COMBINED_MEASURED_IM_AS-expect_AS)[1:],
    # #           ls='-',marker='',color='k')
    # plt.xlabel('$\sigma_\chi^2$')
    # plt.ylabel('AS (subap)')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.legend()
    # plt.title('after fixing model')
    # # plt.xlim(0.1,0.6)
    # # plt.ylim(0.01,1)
    # plt.show()
    
    # os.chdir(r'C:\Users\gqmc62\OneDrive - Durham University\Documents\PhD\Codes\ThesisChapter5Plots')
    os.chdir(HOME)
    
    
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
    
    
    
    # # plt.plot([expect_rytov[1:],COMBINED_MEASURED_IM_RYTOV[1:]],[expect_AS[1:],COMBINED_MEASURED_IM_AS[1:]],ls='-',marker='',color='k')
    # plt.plot(expect_rytov[1:],expect_AS[1:],marker='o',ls='',color='r',label='Expected')
    # # plt.plot(COMBINED_MEASURED_IM_RYTOV[1:],COMBINED_MEASURED_IM_AS[1:],ls='',marker='o',label='Measured')
    # # plt.arrow(expect_rytov[1:],expect_AS[1:],
    # #           (COMBINED_MEASURED_IM_RYTOV-expect_rytov)[1:],(COMBINED_MEASURED_IM_AS-expect_AS)[1:],
    # #           ls='-',marker='',color='k')
    # plt.xlabel('$\sigma_\chi^2$')
    # plt.ylabel('AS (subap)')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.legend()
    # plt.title('after fixing model true expected sampling point')
    # # plt.xlim(0.005,0.8)
    # # plt.ylim(0.005,0.3)
    # plt.savefig(TYPE + 'sampling_points.png')
    # plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    ###############################################################################
    # IM error
    ###############################################################################
    
    TRIAL = IM.shape[1]
    nACT = IM.shape[2]
    nSUBAP = IM.shape[3]
    
    MAX =  perfect_IM[:,2:].max()
    
    perfect_IM /= MAX
    IM /= IM.max()
    
    threshold = 1/4. # selecting only the major strokes.
    
    MASK = (np.abs(perfect_IM[0]) > threshold)
    MASK[:2] = 0
    
    # plt.imshow(perfect_IM[0]*MASK)
    # plt.show()
    
    nMAINACT = (MASK == True).sum()
    DIF = np.zeros((CASES,TRIAL,nMAINACT),dtype=float)
    
    for cases in np.arange(CASES):
        for trial in np.arange(TRIAL):
            DIF[cases,trial] = (perfect_IM[cases] - IM[cases,trial])[MASK]
            
    DIF = np.asarray(DIF)
    IM_RMS = np.sqrt(np.nanmean(DIF**2,axis=(1,2)))
    
    
    grid_rytov = np.unique(np.round(expect_rytov,decimals=3))
    grid_AS = np.unique(np.round(expect_AS,decimals=3))
    perfect_grid_IM_RMS = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)
    
    for cases in np.arange(CASES):
        perfect_grid_IM_RMS[np.where(np.round(expect_rytov[cases],decimals=3)==grid_rytov),
                            np.where(np.round(expect_AS[cases],decimals=3)==grid_AS)] = IM_RMS[cases]
        
    perfect_grid_IM_RMS[perfect_grid_IM_RMS==0] = np.nan
    
    CS2 = plt.contourf(grid_rytov,grid_AS,perfect_grid_IM_RMS.T)
    # plt.colorbar()
    CS = plt.contour(grid_rytov,grid_AS,perfect_grid_IM_RMS.T,colors='k')
    # plt.grid()
    plt.plot(expect_rytov,expect_AS,marker='o',ls='',color='r')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('$\sigma_\chi^2$')
    plt.ylabel('AS (subap)')
    plt.title('RMS error of Major Strokes of IM\n(normalized to max IM responses of major strokes)')
    plt.clabel(CS)
    # plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    # plt.xlim(0.005,0.8)
    # plt.ylim(0.005,0.3)
    plt.savefig(TYPE + 'IM_degrade.png')
    plt.colorbar(CS2)
    # plt.axis('square')
    plt.show()
    
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
            DIF[cases,trial] = (perfect_CC[cases] - CC[cases,trial])[MASK]
        CC_RMS = (np.nanmean(DIF**2,axis=(-1,-2)))**0.5
        
    # grid_rytov = np.unique(expect_rytov)
    # grid_AS = np.unique(expect_AS)
    perfect_grid_CC_RMS = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)
    
    for cases in np.arange(CASES):
        perfect_grid_CC_RMS[np.where(np.round(expect_rytov[cases],decimals=3)==grid_rytov),
                            np.where(np.round(expect_AS[cases],decimals=3)==grid_AS)] = CC_RMS[cases]
        
    perfect_grid_CC_RMS[perfect_grid_CC_RMS==0] = np.nan
    
    CS2 = plt.contourf(grid_rytov,grid_AS,perfect_grid_CC_RMS.T)
    # plt.colorbar()
    CS = plt.contour(grid_rytov,grid_AS,perfect_grid_CC_RMS.T,colors='k')
    # plt.grid()
    plt.plot(expect_rytov,expect_AS,marker='o',ls='',color='r')
    plt.xlabel('$\sigma_\chi^2$')
    plt.ylabel('AS (subap)')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.title('RMS error of Major CC Elements\n(normalized to max undistorted CC)')
    plt.clabel(CS)
    # plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    # plt.xlim(0.005,0.8)
    # plt.ylim(0.005,0.3)
    plt.colorbar(CS2)
    plt.savefig(TYPE + 'CM_degrade.png')
    # plt.axis('square')
    plt.show()
    
    # for i in np.arange(CASES):
    #     for j in np.arange(TRIAL):
    #         # if RUNAWAY[i,j,0] == 1:
    #         # plt.imshow(np.unwrap(np.unwrap(
    #         #     np.angle(crop_inscribed_square(
    #         #     perfect_RES_1_open[i,0,j]+1j*perfect_RES_1_open[i,1,j]))
    #         #     , axis=0), axis=1))
    #         # plt.colorbar()
    #         # plt.title('no distortion open{},{}'.format(i,j))
    #         # plt.show()
    #         plt.imshow(np.unwrap(np.unwrap(
    #             np.angle(crop_inscribed_square(
    #             perfect_RES[i,0,j,-1]+1j*perfect_RES[i,1,j,-1]))
    #             , axis=0), axis=1))
    #         plt.colorbar()
    #         plt.title('no distortion close{},{}'.format(i,j))
    #         plt.show()
    #         # plt.imshow(np.unwrap(np.unwrap(
    #         #     np.angle(crop_inscribed_square(
    #         #     RES_1_open[i,0,j]+1j*RES_1_open[i,1,j]))
    #         #     , axis=0), axis=1))
    #         # plt.colorbar()
    #         # plt.title('distorted open{},{}'.format(i,j))
    #         # plt.show()
    #         plt.imshow(np.unwrap(np.unwrap(
    #             np.angle(crop_inscribed_square(
    #             RES[i,0,j,-1]+1j*RES[i,1,j,-1]))
    #             , axis=0), axis=1))
    #         plt.colorbar()
    #         plt.title('distorted close{},{}'.format(i,j))
    #         plt.show()
    
    # for i in np.arange(23):
    #     for j in np.arange(10):
    #             plt.imshow(np.unwrap(np.unwrap(np.angle(crop_inscribed_square(
    #                 RES_1_open[i,0,j]+1j*RES_1_open[i,1,j]))
    #                 ,axis=-1),axis=-2))
    #             plt.colorbar()
    #             plt.title('{},{}'.format(i,j))
    #             plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # ###############################################################################
    # # Control SR
    # ###############################################################################
    
    # perfect_avg_terminal_SR = np.nanmean(perfect_SR[:,:,-1],axis=-1)
    
    # grid_rytov = np.unique(expect_rytov)
    # grid_AS = np.unique(expect_AS)
    # perfect_grid_SR = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)
    
    # for cases in np.arange(CASES):
    #     perfect_grid_SR[np.where(expect_rytov[cases]==grid_rytov),np.where(expect_AS[cases]==grid_AS)] = perfect_avg_terminal_SR[cases]
        
    # perfect_grid_SR[perfect_grid_SR==0] = np.nan
    
    # plt.contourf(grid_rytov,grid_AS,perfect_grid_SR.T)
    # # plt.colorbar()
    # CS = plt.contour(grid_rytov,grid_AS,perfect_grid_SR.T,colors='k')
    # # plt.grid()
    # # plt.plot(expect_rytov,expect_AS,marker='o',ls='',color='r')
    # plt.xlabel('$\sigma_\chi^2$')
    # plt.ylabel('AS (subap)')
    # plt.title('No distortion bench mark Strehl Ratio' + '\nNo distortion has {:.2f}'.format(perfect_grid_SR[0,0]))
    # plt.clabel(CS)
    # # plt.legend()
    # # plt.xscale('log')
    # # plt.yscale('log')
    # # plt.xlim(0.1,0.6)
    # # plt.ylim(0.1,1.0)
    # # plt.axis('square')
    # plt.show()
    
    # perfect_avg_terminal_SR_std = np.nanstd(perfect_SR[:,:,-1],axis=-1)
    # grid_rytov = np.unique(expect_rytov)
    # grid_AS = np.unique(expect_AS)
    # perfect_grid_SR_std = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)
    
    # for cases in np.arange(CASES):
    #     perfect_grid_SR_std[np.where(expect_rytov[cases]==grid_rytov),np.where(expect_AS[cases]==grid_AS)] = perfect_avg_terminal_SR_std[cases]
        
    # perfect_grid_SR_std[perfect_grid_SR_std==0] = np.nan
    
    # plt.contourf(grid_rytov,grid_AS,perfect_grid_SR_std.T)
    # # plt.colorbar()
    # CS = plt.contour(grid_rytov,grid_AS,perfect_grid_SR_std.T,colors='k')
    # # plt.grid()
    # # plt.plot(expect_rytov,expect_AS,marker='o',ls='',color='r')
    # plt.xlabel('$\sigma_\chi^2$')
    # plt.ylabel('AS (subap)')
    # plt.title('No distortion bench mark std Strehl Ratio' + '\nNo distortion has {:.2f}'.format(perfect_grid_SR_std[0,0]))
    # plt.clabel(CS)
    # # plt.legend()
    # # plt.xscale('log')
    # # plt.yscale('log')
    # # plt.xlim(0.1,0.6)
    # # plt.ylim(0.1,1.0)
    # # plt.axis('square')
    # plt.show()
    
    
    
    ###############################################################################
    # Control WFE
    ###############################################################################
    
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
    
    
    
    
    
    
    ###############################################################################
    # Drop in SR
    ###############################################################################
    
    
    
    # avg_terminal_SR = np.nanmean(SR[:,:,-1],axis=-1)
    
    # grid_rytov = np.unique(expect_rytov)
    # grid_AS = np.unique(expect_AS)
    # grid_SR = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)
    
    # for cases in np.arange(CASES):
    #     grid_SR[np.where(expect_rytov[cases]==grid_rytov),np.where(expect_AS[cases]==grid_AS)] = avg_terminal_SR[cases]
        
    # grid_SR[grid_SR==0] = np.nan
    
    # plt.contourf(grid_rytov,grid_AS,grid_SR.T)
    # # plt.colorbar()
    # CS = plt.contour(grid_rytov,grid_AS,grid_SR.T,colors='k')
    # # plt.grid()
    # # plt.plot(expect_rytov,expect_AS,marker='o',ls='',color='r')
    # plt.xlabel('$\sigma_\chi^2$')
    # plt.ylabel('AS (subap)')
    # # plt.xscale('log')
    # # plt.yscale('log')
    # plt.title('Strehl Ratio' + '\nNo distortion has {:.2f}'.format(grid_SR[0,0]))
    # plt.clabel(CS)
    # # plt.legend()
    # # plt.xscale('log')
    # # plt.yscale('log')
    # # plt.xlim(0.1,0.6)
    # # plt.ylim(0.1,1.0)
    # # plt.axis('square')
    # plt.show()
    
    # avg_terminal_SR_std = np.nanstd(SR[:,:,-1],axis=-1,ddof=1)
    # grid_rytov = np.unique(expect_rytov)
    # grid_AS = np.unique(expect_AS)
    # grid_SR_std = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)
    
    # for cases in np.arange(CASES):
    #     grid_SR_std[np.where(expect_rytov[cases]==grid_rytov),np.where(expect_AS[cases]==grid_AS)] = avg_terminal_SR_std[cases]
        
    # grid_SR_std[grid_SR_std==0] = np.nan
    
    # plt.contourf(grid_rytov,grid_AS,grid_SR_std.T)
    # # plt.colorbar()
    # CS = plt.contour(grid_rytov,grid_AS,grid_SR_std.T,colors='k')
    # # plt.grid()
    # # plt.plot(expect_rytov,expect_AS,marker='o',ls='',color='r')
    # plt.xlabel('$\sigma_\chi^2$')
    # plt.ylabel('AS (subap)')
    # # plt.xscale('log')
    # # plt.yscale('log')
    # plt.title('std Strehl Ratio' + '\nNo distortion has {:.2f}'.format(grid_SR_std[0,0]))
    # plt.clabel(CS)
    # # plt.legend()
    # # plt.xscale('log')
    # # plt.yscale('log')
    # # plt.xlim(0.1,0.6)
    # # plt.ylim(0.1,1.0)
    # # plt.axis('square')
    # plt.show()
    
    
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
    
    ###############################################################################
    # only control error
    ###############################################################################
    
    
    
    avg_terminal_WFE_only_control = (np.nanmean((WFE[:,:,-1]**2),axis=-1) - np.nanmean(WFE[0,:,-1]**2,axis=-1))**0.5
    avg_terminal_WFE_only_control[np.isnan(avg_terminal_WFE_only_control)] = 0
    # expect_rytov
    # expect_AS
    avg_terminal_WFE_only_control_error = np.nanstd(WFE[:,:,-1]**2,axis=-1)**0.5
    
    
    # plt.plot(expect_rytov,avg_terminal_SR,ls='',marker='o')
    # plt.plot(expect_AS,avg_terminal_SR,ls='',marker='o')
    
    # grid_rytov = np.unique(expect_rytov)
    # grid_AS = np.unique(expect_AS)
    grid_WFE = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)
    grid_WFE[:] = np.nan
    
    for cases in np.arange(CASES):
        # print(np.where(expect_rytov[cases]==grid_rytov),np.where(expect_AS[cases]==grid_AS))
        grid_WFE[np.where(np.round(expect_rytov[cases],decimals=3)==grid_rytov),
                            np.where(np.round(expect_AS[cases],decimals=3)==grid_AS)] = avg_terminal_WFE_only_control[cases]
        
    # grid_WFE[grid_WFE==0] = np.nan
    
    # TEMP = np.nanmax(grid_WFE)
    
    CS2 = plt.contourf(grid_rytov,grid_AS,grid_WFE.T)
    # plt.colorbar()
    CS = plt.contour(grid_rytov,grid_AS,grid_WFE.T,colors='k')
    # plt.grid()
    plt.plot(expect_rytov,expect_AS,marker='o',ls='',color='r')
    plt.xlabel('$\sigma_\chi^2$')
    plt.ylabel('AS (subap)')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.title('WFE (nm), excluding static error')
    plt.clabel(CS)
    plt.colorbar(CS2)
    # plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    # plt.xlim(0.005,0.8)
    # plt.ylim(0.005,0.3)
    plt.savefig(TYPE + 'control_WFE.png')
    # plt.axis('square')
    plt.show()
    
    np.savetxt('grid_rytov.csv',grid_rytov,delimiter=',')
    np.savetxt('grid_AS.csv',grid_AS,delimiter=',')
    np.savetxt('grid_WFE.csv',grid_WFE,delimiter=',')
    
    # X = np.nanmean(WFE[:,:,-1],axis=-1)
    # Y = np.nanmean(WFE[0,:,-1],axis=-1)
    # sX = np.nanstd(WFE[:,:,-1],axis=-1)
    # sY = np.nanstd(WFE[0,:,-1],axis=-1)
    # Z = (X**2 - Y**2)**0.5
    # sZ = (X**2*sX**2+Y**2*sY**2)**0.5/Z
    # sZ[np.isnan(sZ)] = np.inf
    # error_avg_terminal_WFE_only_control = sZ
    
    # # expect_rytov
    # # expect_AS
    
    
    
    # # plt.plot(expect_rytov,avg_terminal_SR,ls='',marker='o')
    # # plt.plot(expect_AS,avg_terminal_SR,ls='',marker='o')
    
    # grid_rytov = np.unique(expect_rytov)
    # grid_AS = np.unique(expect_AS)
    # grid_WFE = np.zeros((grid_rytov.shape[0],grid_AS.shape[0]),dtype=float)
    # grid_WFE[:] = np.nan
    
    # for cases in np.arange(CASES):
    #     grid_WFE[np.where(expect_rytov[cases]==grid_rytov),np.where(expect_AS[cases]==grid_AS)] = error_avg_terminal_WFE_only_control[cases]
        
    # # grid_WFE[grid_WFE==0] = np.nan
    
    # plt.contourf(grid_rytov,grid_AS,grid_WFE.T)
    # # plt.colorbar()
    # CS = plt.contour(grid_rytov,grid_AS,grid_WFE.T,colors='k')
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
    
    ###############################################################################
    # 
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
    
    
    
    theory_noll = mps.von_Karman_noll_variance(MODE, D/pixel_scale, pixel_scale, 0.05, 5)
    MAX = np.max(np.array([theory_noll.max(),
                           np.nanmean(ATM_Z,axis=1).max(),
                           np.nanmean(perfect_RES_phase_Z,axis=1).max(),
                           np.nanmean(RES_phase_Z,axis=1).max()]))*10
    MAX = np.min(np.array([theory_noll.min(),
                           np.nanmean(ATM_Z,axis=1).min(),
                           np.nanmean(perfect_RES_phase_Z,axis=1).min(),
                           np.nanmean(RES_phase_Z,axis=1).min()]))/10
    
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
    #     plt.savefig(TYPE + 'z_decompose_{:d}.png'.format(cases))
    #     plt.show()
        
    for cases in np.arange(CASES):
        # if (cases != [9,16,21]).all():
        if (cases != [9,14,16,19,21]).all():
            continue
        
        plt.plot(np.arange(MODE - 1) + 1,
                      np.var(RES_phase_Z[cases,:,-1,1:],axis=(0),ddof=1).T,
                      #color='r',
                      label=name_list[cases],#'unmatch control',
                      marker='.')
        plt.fill_between(np.arange(MODE - 1) + 1,
                      np.var(RES_phase_Z[cases,:,-1,1:],axis=(0),ddof=1).T+2*np.var(RES_phase_Z[cases,:,-1,1:],axis=(0),ddof=1).T/np.sqrt(2*ATM_Z.shape[1] - 2),
                      np.var(RES_phase_Z[cases,:,-1,1:],axis=(0),ddof=1).T-2*np.var(RES_phase_Z[cases,:,-1,1:],axis=(0),ddof=1).T/np.sqrt(2*ATM_Z.shape[1] - 2),
                      #color='r',
                      alpha=0.2)
        
    plt.plot(np.arange(MODE - 1) + 1,
                  np.var(perfect_RES_phase_Z[:,:,-1,1:],axis=(0,1),ddof=1).T,
                  #color='g',
                  label='no control error',marker='.')
    plt.fill_between(np.arange(MODE - 1) + 1,
                  np.var(perfect_RES_phase_Z[:,:,-1,1:],axis=(0,1),ddof=1).T + 2*np.var(perfect_RES_phase_Z[:,:,-1,1:],axis=(0,1),ddof=1).T/np.sqrt(2*ATM_Z.shape[0]*ATM_Z.shape[1] - 2),
                  np.var(perfect_RES_phase_Z[:,:,-1,1:],axis=(0,1),ddof=1).T - 2*np.var(perfect_RES_phase_Z[:,:,-1,1:],axis=(0,1),ddof=1).T/np.sqrt(2*ATM_Z.shape[0]*ATM_Z.shape[1] - 2),
                  #color='g',
                  alpha=0.2)
        
    plt.plot(np.arange(MODE) + 1,theory_noll,ls=':',
             #color='k',
             label='theory')
    
    plt.yscale('log')
    # plt.title('terminal z-decomposition')
    plt.legend()
    plt.ylabel('variance of Z coefficients (rad$^2$)')
    plt.xlabel('Z-Modes')
    plt.title('Terminal Wavefront Residual Zernike Mode Decomposition\n' + name_list[cases])
    plt.savefig(TYPE + 'z_decompose-selected.png')#_{:d}.png'.format(cases))
    plt.show()
    
    
    
    # remesure psd because inside the main code i forgot to unwrap the phase :p & i want to measure before ao as well
    
    PSD_phase_frequency_new = PSD_phase_frequency
    # PSD_phase_frequency_new = np.zeros((CASES,PSD_phase.shape[-1]),dtype=float)
    # if TYPE == 'A':
    #     PSD_phase_frequency_new[:] = (np.arange(44) + 1) / 90. / (pixel_scale) # because i have only 90 pixel in psd measuere
    # if TYPE == 'B':
    #     PSD_phase_frequency_new[:] = (np.arange(23) + 1) / 48. / (pixel_scale)
    theory_psd = mps.psd_function(r0,(PSD_phase_frequency_new.shape[1] + 1)*2,pixel_scale,L0,1e-10)
    cross_section_theory_psd = theory_psd[theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ]
    
    # PS = PhaseScreenKolmogorov((PSD_phase_frequency_new.shape[1] + 1)*2, pixel_scale, r0, L0)
    # for i in np.arange((PSD_phase_frequency_new.shape[1] + 1)*2):
    #     PS.add_row()
    # ref = np.zeros((100,theory_psd.shape[0],theory_psd.shape[1]),dtype=float)
    # for i in np.arange(100):
    #     for j in np.arange((PSD_phase_frequency_new.shape[1] + 1)*2):
    #         PS.add_row()
    #     screen = PS.scrn
    #     screen -= screen.mean()
    #     ref[i] = mps.get_PSD(screen,pixel_scale)
    
    # if not os.path.exists('PSD_phase_new' + '.fits'):
        
    # phase = np.angle(RES[:,0] + 1j*RES[:,1])
    # perfect_phase = np.angle(perfect_RES[:,0] + 1j*perfect_RES[:,1])
    
    # PSD_phase_new = np.zeros_like(PSD_phase)
    # perfect_PSD_phase_new = np.zeros_like(perfect_PSD_phase)
    # before_PSD = np.zeros((CASES,TRIAL,44),dtype=float)
    
    # for cases in np.arange(CASES):
    #     for trial in np.arange(TRIAL):
    #         PSD_phase_new[cases,trial] = mps.get_PSD(
    #             np.unwrap(np.unwrap(
    #                 crop_inscribed_square(
    #                     phase[cases,trial,-1]
    #                     ),axis=-1),axis=-2), pixel_scale)[45,46:]
    #         perfect_PSD_phase_new[cases,trial] = mps.get_PSD(
    #             np.unwrap(np.unwrap(
    #                 crop_inscribed_square(
    #                     perfect_phase[cases,trial,-1]
    #                     ),axis=-1),axis=-2), pixel_scale)[45,46:]
    #         before_PSD[cases,trial] = mps.get_PSD(
    #             np.unwrap(np.unwrap(
    #                 crop_inscribed_square(
    #                     perfect_phase[cases,trial,0]
    #                     ),axis=-1),axis=-2), pixel_scale)[45,46:]
    
    PSD_phase_new = PSD_phase
    perfect_PSD_phase_new = perfect_PSD_phase
                
    #     hdu = fits.PrimaryHDU(PSD_phase_new)
    #     hdul = fits.HDUList([hdu])
    #     hdul.writeto('PSD_phase_new' + '.fits', overwrite=True)
    #     with open('data_space.txt', 'a') as outfile:
    #         outfile.write('PSD_phase_new' + '\n')
        
    #     hdu = fits.PrimaryHDU(perfect_PSD_phase_new)
    #     hdul = fits.HDUList([hdu])
    #     hdul.writeto('perfect_PSD_phase_new'  + '.fits', overwrite=True)
    #     with open('data_space.txt', 'a') as outfile:
    #         outfile.write('perfect_PSD_phase_new' + '\n')
              
    #     hdu = fits.PrimaryHDU(before_PSD)
    #     hdul = fits.HDUList([hdu])
    #     hdul.writeto('before_PSD' + '.fits', overwrite=True)
    #     with open('data_space.txt', 'a') as outfile:
    #         outfile.write('before_PSD' + '\n')
    
    # else:
    #     hdul = fits.open('PSD_phase_new' + '.fits')
    #     SHAPE = hdul[0].data.shape
    #     globals()['PSD_phase_new'] = hdul[0].data
    #     hdul = fits.open('perfect_PSD_phase_new' + '.fits')
    #     SHAPE = hdul[0].data.shape
    #     globals()['perfect_PSD_phase_new'] = hdul[0].data
    #     hdul = fits.open('before_PSD' + '.fits')
    #     SHAPE = hdul[0].data.shape
    #     globals()['before_PSD'] = hdul[0].data
        
    
            
    ###################################################################
    
    # MAX = np.max(np.array([cross_section_theory_psd.max(),np.nanmean(perfect_PSD_phase_new,axis=1).max(),np.nanmean(PSD_phase_new,axis=1).max()]))*10
    # MIN = np.min(np.array([cross_section_theory_psd.min(),np.nanmean(perfect_PSD_phase_new,axis=1).min(),np.nanmean(PSD_phase_new,axis=1).min()]))/10
    # for cases in np.arange(CASES):
    
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
    #                  label='distorted control',color='r',marker='.')
    #     plt.plot(PSD_phase_frequency_new[cases],
    #                  np.nanmean(perfect_PSD_phase_new[cases,:,-1],axis=0),
    #                  label='non-distorted control',color='g',marker='.')
        
    #     plt.fill_between(PSD_phase_frequency_new[cases],
    #                  np.nanmean(before_PSD[cases],axis=0) + np.nanstd(before_PSD[cases],axis=0)/np.sqrt(2*100-2),
    #                  np.nanmean(before_PSD[cases],axis=0) - np.nanstd(before_PSD[cases],axis=0)/np.sqrt(2*100-2),
    #                  color='b',alpha=0.2)
    #     plt.plot(PSD_phase_frequency_new[cases],
    #                  np.nanmean(before_PSD[cases],axis=0),
    #                  label='simulated atm',color='b',marker='.')
        
    #     # plt.fill_between(PSD_phase_frequency_new[cases],
    #     #              np.nanmean(ref[:,theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ],axis=0) + np.nanstd(ref[:,theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ],axis=0)/np.sqrt(2*100-2),
    #     #              np.nanmean(ref[:,theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ],axis=0) - np.nanstd(ref[:,theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ],axis=0)/np.sqrt(2*100-2),
    #     #              color='m',alpha=0.2)
    #     # plt.plot(PSD_phase_frequency_new[cases],
    #     #              np.nanmean(ref[:,theory_psd.shape[0]//2, theory_psd.shape[0]//2 + 1: ],axis=0),
    #     #              label='ref atm',color='m')
        
    #     plt.plot(PSD_phase_frequency_new[cases],cross_section_theory_psd,label='atm theory',color='k',ls=':')
    #     plt.vlines(1/dsub/2,MIN,MAX,label='AO Nyquist Spatial Frequency',color='c')
    #     plt.xscale('log')
    #     plt.yscale('log')
    #     plt.ylim(MIN,MAX)
    #     plt.legend()
    #     plt.title(name_list[cases])
    #     plt.xlabel('spatial frequency (1/m)')
    #     plt.ylabel('PSD of phase (rad$^2$ m$^2$)')
    #     plt.savefig(TYPE + 'psd-phase-{:d}.png'.format(cases))
    #     plt.show()
    
    MAX = np.max(np.array([cross_section_theory_psd.max(),np.nanmean(perfect_PSD_phase_new,axis=1).max(),np.nanmean(PSD_phase_new,axis=1).max()]))*10
    MIN = np.min(np.array([cross_section_theory_psd.min(),np.nanmean(perfect_PSD_phase_new,axis=1).min(),np.nanmean(PSD_phase_new,axis=1).min()]))/10
    for cases in np.arange(CASES):
        # if (cases != [9,16,21]).all():
        if (cases != [9,14,16,19,21]).all():
            continue
        plt.fill_between(PSD_phase_frequency_new[cases],
                     np.nanmean(PSD_phase_new[cases,:,-1],axis=0) + np.nanstd(PSD_phase_new[cases,:,-1],axis=0,ddof=1)/np.sqrt(2*PSD_phase_new.shape[1]-2),
                     np.nanmean(PSD_phase_new[cases,:,-1],axis=0) - np.nanstd(PSD_phase_new[cases,:,-1],axis=0,ddof=1)/np.sqrt(2*PSD_phase_new.shape[1]-2),
                     #color='r',
                     alpha=0.2)
        plt.plot(PSD_phase_frequency_new[cases],
                     np.nanmean(PSD_phase_new[cases,:,-1],axis=0),
                     label=name_list[cases],#'{:d}'.format(cases),#'distorted control',
                     #color='r',
                     marker='.')
        
    plt.fill_between(PSD_phase_frequency_new[cases],
                 np.nanmean(perfect_PSD_phase_new[cases,:,-1],axis=0) + np.nanstd(perfect_PSD_phase_new[cases,:,-1],axis=0,ddof=1)/np.sqrt(2*perfect_PSD_phase_new.shape[1]-2),
                 np.nanmean(perfect_PSD_phase_new[cases,:,-1],axis=0) - np.nanstd(perfect_PSD_phase_new[cases,:,-1],axis=0,ddof=1)/np.sqrt(2*perfect_PSD_phase_new.shape[1]-2),
                 #color='g',
                 alpha=0.2)
    plt.plot(PSD_phase_frequency_new[cases],
                 np.nanmean(perfect_PSD_phase_new[cases,:,-1],axis=0),
                 label='no control error',
                 #color='g',
                 marker='.')
        
    plt.plot(PSD_phase_frequency_new[cases],cross_section_theory_psd,label='atm theory',color='k',ls=':')
    plt.vlines(1/dsub/2,MIN,MAX,label='AO Nyquist Spatial Frequency',color='c')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(MIN,MAX)
    plt.legend()
    # plt.title(name_list[cases])
    plt.xlabel('spatial frequency (1/m)')
    plt.ylabel('PSD of phase (rad$^2$ m$^2$)')
    plt.savefig(TYPE + 'psd-phase-selected.png')#-{:d}.png'.format(cases))
    plt.show()
    
        
        
    # PSD_phase_new = PSD_intensity / 2.
    # perfect_PSD_phase_new = perfect_PSD_intensity / 2.
    # PSD_phase_frequency_new = PSD_phase_frequency# * np.sqrt(2)
    
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
        
        
        
        
        
        
        
        
    ###############################################################################################
    # useful analysis on effect of zernike modes of the distortion on control performance
    ###############################################################################################        
        
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
        
    
    # from scipy.optimize import curve_fit
    # def linear(x,m,c):
    #     return m*x + c
    # # import scipy.stats.linregress as linregress
    
    # # for i in np.arange(3,10):
    # #     A = np.abs(ABERATION_Z[...,i].flatten())
    # #     B = WFE[...,-1].flatten()
    # #     plt.plot(A,B,marker='.',ls='',label='{:d}'.format(i),alpha=0.1)
    # #     # plt.xscale('log')
    # #     # plt.yscale('log')
    # #     fit_res = curve_fit(linear,A,B)
    # #     fit = linear(A,fit_res[0][0],fit_res[0][1])
    # #     plt.plot(A,fit)
    # #     plt.legend()
    # #     plt.show()
    # #     (1 - (B-fit).std() / B.std())*100
    # z_res_slope = np.zeros((ABERATION_Z.shape[-1]),dtype=float)
    # z_res_slope_error = np.zeros((ABERATION_Z.shape[-1]),dtype=float)
    # correlate = np.zeros((ABERATION_Z.shape[-1]),dtype=float)
    # for i in np.arange(3,ABERATION_Z.shape[-1]):
    #     A = np.abs(ABERATION_Z[...,i].flatten())
    #     B = WFE[...,-1].flatten()
    #     fit_res = curve_fit(linear,A,B)
    #     fit = linear(A,fit_res[0][0],fit_res[0][1])
    #     correlate[i] = 1 - ((B - fit)**2).sum() / ((B - B.mean())**2).sum()
        
    #     # measure the slope of the relation, it is misleading, because we have too noisy data!! check correlate
        
    #     z_res_slope[i] = fit_res[0][0]
    #     z_res_slope_error[i] = fit_res[1][0,0]**0.5
    #     # correlate[i] = np.mean((A - A.mean())*(B - B.mean())) / A.std() / B.std()
    #     # plt.plot(A,B,ls='',marker=',')
    #     # plt.plot(A,fit)
    #     # plt.xlabel('Zernike Coefficient')
    #     # plt.ylabel('WFE (nm)')
    #     # plt.title('{:}'.format(i))
    #     # plt.xlim(0,2)
    #     # plt.ylim(40,100)
    #     # plt.show()
        
    # # measure the slope of the relation, it is misleading, because we have too noisy data!! check correlate
    # # low correlation ! so no! dont use the slope
    # plt.plot(np.arange(3,ABERATION_Z.shape[-1]),
    #          correlate[3:ABERATION_Z.shape[-1]],
    #          marker='.',ls='',label='{:d}'.format(i))
    # plt.xlabel('zernike mode')
    # plt.ylabel('R2 of linear fit')
    # plt.title('good ness of linear fit between zernike mode strength of intermediate dm to residual WFE')
    # # plt.xscale('log')
    # # plt.yscale('log')
    # # plt.legend()
    # plt.savefig(TYPE + 'no-correlation.png')
    # plt.show()
    
    # # measure the slope of the relation, it is misleading, because we have too noisy data!! check correlate
    # # the slope have a trend as shown below.
    # # because the higher z mode in kolmogorov turbulence then to have smaller variance
    # # when the graph try to fit, the highest noisy data on the right is now closer to y axis
    # # giving an impression of having a stronger slope, lol
    # # plt.errorbar(np.arange(3,ABERATION_Z.shape[-1]),
    # #              z_res_slope[3:ABERATION_Z.shape[-1]],
    # #              yerr=z_res_slope_error[3:ABERATION_Z.shape[-1]],
    # #              capsize=5,ls='',marker='o')
    # # plt.xlabel('Zernike mode')
    # # plt.ylabel('Slope between Zernike and WFE')
    # # plt.show()
    
    ###############################################################################################
    
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    
    ax1.set_xlim(1e-15,r02cn2dz(0.02))
    ax2.set_ylim(1,30)
    
    r0_mark = np.array([5,2,1,0.5,0.2,0.1,0.05,0.02])*100
    new_tick_locations = np.log10(r02cn2dz(r0_mark)) + 15
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(new_tick_locations)
    labels = []
    for i in np.arange(r0_mark.shape[0]):
      labels.append('{:.0f}'.format(r0_mark[i]))
    ax2.set_xticklabels(labels)
    
    ax2.set_xlabel('$r_0$ of turbulence layer in line of sight\nthat immediate DM is correcting (cm) at 500 nm')
    ax1.set_xlabel('$C^2_n$sec$\zeta dh$ of turbulence layer that immediate DM is correcting (m$^{1/3}$)')
    ax1.set_ylabel('Separation between immediate DM\nand its next immediate DM in a line of sight ($h$sec$\zeta$) (km)')
    
    x = r02cn2dz(np.logspace(np.log10(0.02),np.log10(5)))[::-1]
    y = np.logspace(np.log10(1),np.log10(30))
    
    yy, xx = np.meshgrid(y,x)
    
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    #ax1.set_yscale('squareroot')
    
    ax1.grid()
    ax2.grid()
    
    # ax1.hlines(dm_separation/1000,1e-15,r02cn2dz(0.02),color='m')
    
    dact_immediate_dm = dsub
    dact_next_immediate_dm = dsub
    Dimmediate_dm = 0.4
    L0 = 5
    wvl = 500e-9
    k = np.pi*2/wvl
    
    RYTOV = rytov_number(wvl, cn2dz2r0(xx), yy*1000.)
    # APSHIFT = displacement(wvl, dact_immediate_dm, dact_next_immediate_dm, cn2dz2r0(xx), yy*1000., Dimmediate_dm, L0)
    
    # CS = ax1.contour(x,y,RYTOV.T,colors='r',levels=[0.01,0.02,0.05,0.1,0.3,0.6])
    # ax1.clabel(CS)
    # CS2 = ax1.contour(x,y,APSHIFT*100,colors='b',levels=[1,2,5,10,20,50])
    # ax1.clabel(CS2)
    # # plt.legend()
    color_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    count = 0
    for i in [0.01,0.02,0.05,0.1,0.3,0.6]:
        ax1.plot(x,
                 (i/0.563/k**(7./6.)/x)**(6./5.)/1000.,
                 #label='rytov={:}'.format(i),
                 color=color_list[count],ls='-')
        count += 1
    count = 0
    for i in [1,2,5,10,20,50]:
        ax1.plot(x,
                 (i/100.
                  *dact_next_immediate_dm
                  /(new_coeff * wvl**2 * dsub**(-1/3) * cn2dz2r0(x)**(-5/3))**0.5)/1000
                 ,
                 #label='as={:}'.format(i),
                 color=color_list[count],ls='--')
        count += 1
    
    TOPLOT = np.copy(avg_terminal_WFE_only_control)
    
    cmap = plt.colormaps["viridis"]
    for i in np.arange(r0HA.size):
        ax1.plot(r02cn2dz(r0HA)[i],z[i]/1000,marker='o',markersize=20,color=cmap(TOPLOT[i]/100))
    # ax1.tripcolor(r02cn2dz(r0HA),z/1000,TOPLOT)
    # ax1.tricontour(r02cn2dz(r0HA),z/1000,TOPLOT)
    
    
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,100),cmap=cmap),ax=ax1)
    fig.suptitle('WFE from distorted control error after 5 iterations  (nm)')
    
    # plt.legend()
    plt.gcf().set_size_inches(8,8)
    plt.savefig(TYPE + 'control-error-2.png')
    plt.show()
    
    
    
    
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
    
    # TOPLOT = np.copy(avg_terminal_SR)#np.copy(avg_terminal_WFE_only_control)
    
    # cmap = plt.colormaps["viridis"]
    # for i in np.arange(r0HA.size):
    #     ax1.plot(r02cn2dz(r0HA)[i],z[i]/1000,marker='o',markersize=20,color=cmap(TOPLOT[i]/np.nanmax(TOPLOT)))
    
    
    # fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,np.nanmax(TOPLOT)),cmap=cmap),ax=ax1)
    # fig.suptitle('Terminal Strehl ratio at 500 nm')
    
    # plt.gcf().set_size_inches(8,8)
    # plt.show()
    
    
    
    
    
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
    
    # TOPLOT = np.copy(avg_terminal_WFE)
    
    # cmap = plt.colormaps["viridis"]
    # for i in np.arange(r0HA.size):
    #     ax1.plot(r02cn2dz(r0HA)[i],z[i]/1000,marker='o',markersize=20,color=cmap(TOPLOT[i]/np.nanmax(TOPLOT)))
    
    # fig.suptitle('Terminal WFE (nm)')
    # fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,np.nanmax(TOPLOT)),cmap=cmap),ax=ax1)
    
    
    # plt.gcf().set_size_inches(8,8)
    # plt.show()
    
    
    
    DATA = np.copy(avg_terminal_WFE_only_control)
    error = np.copy(avg_terminal_WFE_only_control_error)
    # r02cn2dz(r0HA)
    # z
    
    unique_AS = np.sort(np.unique(np.round(expect_AS,decimals=3)))
    unique_rytov = np.sort(np.unique(np.round(expect_rytov,decimals=3)))
    
    
    for i in np.arange(unique_AS.shape[0]):
        contour_val = unique_AS[i]
        TOPLOT = DATA[np.round(expect_AS,decimals=3) == contour_val]
        TOPLOT_ERROR = error[np.round(expect_AS,decimals=3) == contour_val]
        X = expect_rytov[np.round(expect_AS,decimals=3) == contour_val]
        plt.errorbar(X,TOPLOT,yerr=TOPLOT_ERROR,label='AS = {:.1f}% of subap.'.format(contour_val*100),ls='-',marker='o',capsize=5)
    plt.xlabel('expect rytov')
    plt.ylabel('terminal control WFE after 5 iteration (nm)')
    plt.xscale('log')
    plt.legend()
    plt.savefig(TYPE + 'control-error-vs-rytov.png')
    plt.show()
    
    
    # DATA = np.copy(avg_terminal_WFE_only_control)
    # r02cn2dz(r0HA)
    # z
    
    # unique_AS = np.sort(np.unique(expect_AS))
    # unique_rytov = np.sort(np.unique(expect_rytov))
    
    
    for i in np.arange(unique_rytov.shape[0]):
        contour_val = unique_rytov[i]
        TOPLOT = DATA[np.round(expect_rytov,decimals=3) == contour_val]
        TOPLOT_ERROR = error[np.round(expect_rytov,decimals=3) == contour_val]
        X = expect_AS[np.round(expect_rytov,decimals=3) == contour_val]*100
        plt.errorbar(X,TOPLOT,yerr=TOPLOT_ERROR,label='rytov = {:.2f}'.format(contour_val),ls='-',marker='o',capsize=5)
    plt.xlabel('expect aperture shift (% of subaperture)')
    plt.ylabel('terminal control WFE after 5 iteration (nm)')
    plt.xscale('log')
    plt.legend()
    plt.savefig(TYPE + 'control-error-vs-as.png')
    plt.show()
    
    plt.clf()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    MAX = expect_AS.max()*100
    for i in np.arange(CASES):
        plt.errorbar(expect_rytov[i],COMBINED_MEASURED_IM_RYTOV[i], error_rytov[i] ,ls='',marker='o',capsize=5,
                     color=cmap(expect_AS[i]*100/MAX))
    plt.plot(np.linspace(0.01,0.6),np.linspace(0.01,0.6))
    plt.xlabel('expected rytov')
    plt.ylabel('measured rytov')
    plt.title('comparison between measured vs expected rytov\nexpect as (%subap) in color')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.005,0.8)
    plt.ylim(0.005,0.8)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,np.nanmax(expect_AS*100)),cmap=cmap),ax=ax1)
    plt.savefig(TYPE + 'expect-measure-rytov.png')
    plt.show()
    
    # plt.errorbar(expect_rytov,(COMBINED_MEASURED_IM_RYTOV-COMBINED_MEASURED_IM_RYTOV[0]), error_rytov ,ls='',marker='o',capsize=5)
    # plt.plot(np.linspace(0.01,0.6),np.linspace(0.01,0.6))
    # plt.xlabel('expected rytov')
    # plt.ylabel('measured rytov')
    # plt.title('comparison between measured vs expected rytov - numerical error')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlim(0.005,0.8)
    # plt.ylim(0.005,0.8)
    # plt.show()
    
    
    # well_behaved = (expect_AS < 0.3) * (expect_rytov < 0.3)
    well_behaved = np.ones_like(expect_AS,dtype=bool)
    well_behaved[0] = False
    
    guess_AS = expect_AS
    
    
    # from scipy.optimize import curve_fit
    # def func(x, intercept):
    #     y = x + intercept
    #     return y
    # FIT = curve_fit(func, np.log10(expect_AS[1:]), np.log10(COMBINED_MEASURED_IM_AS[1:]), sigma=(error_as[1:]/COMBINED_MEASURED_IM_AS[1:]/np.log(10)))
    # intercept = FIT[0][0]
    # adjust_COMBINED_MEASURED_IM_AS = COMBINED_MEASURED_IM_AS /10**intercept
    # adjust_error_as = error_as /10**intercept
    
    # adjust_COMBINED_MEASURED_IM_AS = COMBINED_MEASURED_IM_AS/2
    
    adjust_COMBINED_MEASURED_IM_AS = COMBINED_MEASURED_IM_AS
    adjust_error_as = error_as
    
    # EXPECT = np.linspace(guess_AS[1:].min(),guess_AS[1:].max())*100
    # plt.plot(EXPECT,EXPECT,label='expect')
    # plt.errorbar(guess_AS[well_behaved]*100,(adjust_COMBINED_MEASURED_IM_AS[well_behaved])*100,yerr=adjust_error_as[well_behaved]*100,
    #              ls='',marker='o',capsize=5,label='slope')
    # plt.errorbar(guess_AS[well_behaved]*100,(old_COMBINED_MEASURED_IM_AS[well_behaved])*100,yerr=old_COMBINED_MEASURED_IM_AS_error[well_behaved]*100,
    #              ls='',marker='o',capsize=5,label='influence')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('expect')
    # plt.ylabel('measure')
    # plt.legend()
    # plt.show()
    
    
    plt.clf()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    MAX = expect_rytov.max()*100
    for i in np.arange(CASES):
        plt.errorbar(expect_AS[i]*100,
                     (old_COMBINED_MEASURED_IM_AS[i])*100,
                     yerr=old_COMBINED_MEASURED_IM_AS_error[i]*100,
                     ls='',marker='o',capsize=5,
                     color=cmap(expect_rytov[i]*100/MAX))
    plt.plot(np.linspace(0.5,100),np.linspace(0.5,100))
    plt.xlabel('expected AS (% of subap)')
    plt.ylabel('measured AS (% of subap)')
    plt.title('comparison between measured vs expected AS: influence method\nexpect rytov on color')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.5,100)
    plt.ylim(0.5,100)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,np.nanmax(expect_rytov)),cmap=cmap),ax=ax1)
    plt.savefig(TYPE + 'expect-measure-as-influence.png')
    plt.show()
    
    plt.clf()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    MAX = expect_rytov.max()
    for i in np.arange(CASES):
        plt.errorbar(expect_AS[i]*100,(adjust_COMBINED_MEASURED_IM_AS[i])*100, adjust_error_as[i]*100 ,ls='',marker='o',capsize=5,
                     color=cmap(expect_rytov[i]/MAX))
    plt.plot(np.linspace(0.5,100),np.linspace(0.5,100))
    # plt.plot(np.linspace(2,50),np.linspace(2,50))
    plt.xlabel('expected AS (% of subap)')
    plt.ylabel('measured AS (% of subap)')
    plt.title('comparison between measured vs expected AS : slope method\nexpect rytov in color')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.5,100)
    plt.ylim(0.5,100)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,np.nanmax(expect_rytov)),cmap=cmap),ax=ax1)
    plt.savefig(TYPE + 'expect-measure-as-slope.png')
    plt.show()
    
    plt.errorbar(old_COMBINED_MEASURED_IM_AS.flatten(),adjust_COMBINED_MEASURED_IM_AS.flatten(),
                 xerr=old_COMBINED_MEASURED_IM_AS_error.flatten(),yerr=adjust_error_as.flatten(),
                 ls='',marker='o',capsize=5)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.01,1)
    plt.ylim(0.01,1)
    plt.plot([0.01,1],[0.01,1])
    plt.xlabel('actuator shift - influence method (DM pitch)')
    plt.ylabel('actuator shift - slope method (DM pitch)')
    plt.savefig(TYPE + 'influ-slope.png')
    plt.show()
    
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
        CC_A = np.copy(CC)
        RES_A = np.copy(RES[:,0]+1j*RES[:,1])
    
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
        CC_B = np.copy(CC)
        RES_B = np.copy(RES[:,0]+1j*RES[:,1])
    
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
        CC_C = np.copy(CC)
        RES_C = np.copy(RES[:,0]+1j*RES[:,1])


os.chdir(HOME)

plt.errorbar(expect_rytov_A,WFE_A_control,yerr=WFE_A_error,
             ls='',marker='o',label='dsub{:.2f}cm'.format(dsub_A*100),capsize=5)
plt.errorbar(expect_rytov_B,WFE_B_control,yerr=WFE_B_error,
             ls='',marker='o',label='dsub{:.2f}cm'.format(dsub_B*100),capsize=5)
plt.errorbar(expect_rytov_C,WFE_C_control,yerr=WFE_C_error,
             ls='',marker='o',label='dsub{:.2f}cm'.format(dsub_C*100),capsize=5)
# plt.yscale('log')
plt.xscale('log')
plt.ylabel('control error (nm)')
plt.xlabel('expect rytov')
plt.legend()
# plt.ylim(0,500)
plt.show()

plt.errorbar(expect_AS_A,WFE_A_control,yerr=WFE_A_error,
             ls='',marker='o',label='dsub{:.2f}cm'.format(dsub_A*100),capsize=5)
plt.errorbar(expect_AS_B,WFE_B_control,yerr=WFE_B_error,
             ls='',marker='o',label='dsub{:.2f}cm'.format(dsub_B*100),capsize=5)
plt.errorbar(expect_AS_C,WFE_C_control,yerr=WFE_C_error,
             ls='',marker='o',label='dsub{:.2f}cm'.format(dsub_C*100),capsize=5)
# plt.yscale('log')
plt.xscale('log')
plt.ylabel('control error (nm)')
plt.xlabel('expect as (dsub)')
# plt.ylim(0,500)
plt.legend()
plt.show()

# plt.errorbar(AS_influ_A,WFE_A_control,xerr=AS_influ_error_A/TRIAL**0.5,yerr=WFE_A_error/TRIAL**0.5,
#               ls='',marker='o',label='dsub{:.2f}cm'.format(dsub_A*100),capsize=5)
# plt.errorbar(AS_influ_B,WFE_B_control,xerr=AS_influ_error_B/TRIAL**0.5,yerr=WFE_B_error/TRIAL**0.5,
#               ls='',marker='o',label='dsub{:.2f}cm'.format(dsub_B*100),capsize=5)
# plt.errorbar(AS_influ_C,WFE_C_control,xerr=AS_influ_error_C/TRIAL**0.5,yerr=WFE_C_error/TRIAL**0.5,
#               ls='',marker='o',label='dsub{:.2f}cm'.format(dsub_C*100),capsize=5)
# # plt.yscale('log')
# plt.xscale('log')
# plt.ylabel('control error (nm)')
# plt.xlabel('measure as influ (dsub)')
# # plt.ylim(0,500)
# plt.legend()
# plt.show()


# plt.errorbar(AS_influ_A,WFE_A_control,xerr=AS_influ_error_A,yerr=WFE_A_error,
#               ls='',marker='o',label='dsub{:.2f}cm'.format(dsub_A*100),capsize=5)
# plt.errorbar(AS_influ_B,WFE_B_control,xerr=AS_influ_error_B,yerr=WFE_B_error,
#               ls='',marker='o',label='dsub{:.2f}cm'.format(dsub_B*100),capsize=5)
# plt.errorbar(AS_influ_C,WFE_C_control,xerr=AS_influ_error_C,yerr=WFE_C_error,
#               ls='',marker='o',label='dsub{:.2f}cm'.format(dsub_C*100),capsize=5)
# # plt.yscale('log')
# plt.xscale('log')
# plt.ylabel('control error (nm)')
# plt.xlabel('measure as influ (dsub)')
# # plt.ylim(0,500)
# plt.legend()
# plt.show()

# plt.plot(AS_influ_A,WFE_A_control,#,xerr=AS_influ_error_A,yerr=WFE_A_error,
#               ls='',marker='.',label='dsub{:.2f}cm'.format(dsub_A*100))#,capsize=5)
# plt.plot(AS_influ_B,WFE_B_control,#,xerr=AS_influ_error_B,yerr=WFE_B_error,
#               ls='',marker='.',label='dsub{:.2f}cm'.format(dsub_B*100))#,capsize=5)
# plt.plot(AS_influ_C,WFE_C_control,#,xerr=AS_influ_error_C,yerr=WFE_C_error,
#               ls='',marker='.',label='dsub{:.2f}cm'.format(dsub_C*100))#,capsize=5)
# # plt.yscale('log')
# plt.xscale('log')
# plt.ylabel('control error (nm)')
# plt.xlabel('measure as influ (dsub)')
# # plt.ylim(0,500)
# plt.legend()
# plt.show()

# plt.errorbar(AS_influ_A,WFE_A_control,yerr=WFE_A_error,
#               ls='',marker='o',label='dsub{:.2f}cm'.format(dsub_A*100),capsize=5)
# plt.errorbar(AS_influ_B,WFE_B_control,yerr=WFE_B_error,
#               ls='',marker='o',label='dsub{:.2f}cm'.format(dsub_B*100),capsize=5)
# plt.errorbar(AS_influ_C,WFE_C_control,yerr=WFE_C_error,
#               ls='',marker='o',label='dsub{:.2f}cm'.format(dsub_C*100),capsize=5)
# # plt.yscale('log')
# plt.xscale('log')
# plt.ylabel('control error (nm)')
# plt.xlabel('measure as influ (dsub)')
# # plt.ylim(0,500)
# plt.legend()
# plt.show()

# plt.clf()
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# MAX = expect_rytov.max()
# for i in np.arange(CASES):
#     plt.errorbar(expect_AS[i]*100,(adjust_COMBINED_MEASURED_IM_AS[i])*100-expect_AS[i]*100, adjust_error_as[i]*100 ,ls='',marker='o',capsize=5,
#                  color=cmap(expect_rytov[i]/MAX))
# # plt.plot(np.linspace(0.5,100),np.linspace(0.5,100))
# plt.hlines(0,0.5,100)
# # plt.plot(np.linspace(2,50),np.linspace(2,50))
# plt.xlabel('expected AS (% of subap)')
# plt.ylabel('measured AS (% of subap)')
# plt.title('comparison between measured vs expected AS : slope method\nexpect rytov in color')
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.xlim(0.5,100)
# # plt.ylim(0.5,100)
# fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,np.nanmax(expect_rytov)),cmap=cmap),ax=ax1)
# plt.savefig(TYPE + 'expect-measure-as-slope.png')
# plt.show()


# plt.errorbar(expect_AS*100,(adjust_COMBINED_MEASURED_IM_AS**2-adjust_COMBINED_MEASURED_IM_AS[0]**2)**0.5*100, adjust_error_as*100 ,ls='',marker='o',capsize=5)
# plt.plot(np.linspace(1,20),np.linspace(1,20))
# plt.xlabel('expected AS (% of subap)')
# plt.ylabel('measured AS (% of subap)')
# plt.title('comparison between measured vs expected AS : slope method - numerical error')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0.5,30)
# plt.ylim(0.5,30)
# plt.show()

# plt.errorbar(expect_AS*100,(old_COMBINED_MEASURED_IM_AS**2-old_COMBINED_MEASURED_IM_AS[0]**2)**0.5*100,
#              yerr=old_COMBINED_MEASURED_IM_AS_error*100,
#              ls='',marker='o',capsize=5)
# plt.plot(np.linspace(1,20),np.linspace(1,20))
# plt.xlabel('expected AS (% of subap)')
# plt.ylabel('measured AS (% of subap)')
# plt.title('comparison between measured vs expected AS influench method - numerical error')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0.5,30)
# plt.ylim(0.5,30)
# plt.show()


# plt.plot(AS_R.flatten(),old_temp.flatten(),ls='',marker='.')
# plt.plot(np.linspace(0.01,1),np.linspace(0.01,1),ls='-',marker='')
# plt.plot(2*np.linspace(0.01,1),np.linspace(0.01,1),ls='-',marker='')
# plt.plot(3*np.linspace(0.01,1),np.linspace(0.01,1),ls='-',marker='')
# plt.xscale('log');plt.yscale('log')
# plt.ylabel('influence function method')
# plt.xlabel('slope method')
# plt.show()

# from scipy.optimize import curve_fit
# def linear_log(x,m,c):
#     logx = np.log10(x)
#     logy = logx*m + c
#     y = 10**(logy)
#     return y



# A = np.nanmean(AS_R[1:]**2,axis=1)**0.5
# B = np.nanmean(old_temp[1:]**2,axis=1)**0.5
# C = np.zeros((CASES,TRIAL))
# # for i in np.arange(CASES):
# #     C[i,:] = expect_rytov[i]
# # C = C[0:].flatten()
# C = expect_rytov[1:]

# # C = C[~np.isnan(A)]
# # B = B[~np.isnan(A)]
# # A = A[~np.isnan(A)]

# # order = np.argsort(A)
# # A = A[order]
# # B = B[order]
# # C = C[order]

# A*=100
# B*=100

# # fit_val = curve_fit(linear_log, A,B)
# # plot_range = np.linspace(0.01,1)
# # fit = linear_log(plot_range,fit_val[0][0],fit_val[0][1])

# # r2 = 1 - (np.sum((B - linear_log(A,fit_val[0][0],fit_val[0][1]))**2))/(np.sum((B - B.mean())**2))

# plt.clf()
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# MAX = C.max()
# #for i in np.arange(A.shape[0]):
# plt.plot(A,B,ls='',marker='.',#label='measurements',
#              )#color=cmap(C[i]/MAX))
# # plt.plot(plot_range,fit)
# # plt.title('log10(y)={:.2e}*log10(x)+{:.2e}, R2={:.2f}'.format(fit_val[0][0],fit_val[0][1],r2))
# plt.plot(B,B,label='y=x')
# # plt.plot(B,B/2,label='y=x/2')
# # plt.plot(B,B/4,label='y=x/4')
# plt.xscale('log')
# plt.yscale('log')
# # plt.xlim(0.01*100,1*100)
# # plt.ylim(0.01*100,1*100)
# plt.ylabel('influence function method (%subap)')
# plt.xlabel('slope method (%subap)')
# plt.title('relation between as measurement method')#\nexpect rytov in color')
# #fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,np.nanmax(C)),cmap=cmap),ax=ax1)
# plt.legend()
# plt.grid()
# plt.savefig(TYPE + 'as-slope-vs-influence.png')
# plt.show()




# EXPECT = np.linspace(expect_AS[1:].min(),expect_AS[1:].max())*100
# plt.plot(EXPECT,EXPECT,label='expect')
# plt.errorbar(x=(adjust_COMBINED_MEASURED_IM_AS[1:])*100,
#               y=(old_COMBINED_MEASURED_IM_AS[1:])*100,
#               xerr=adjust_error_as[1:]*100,
#               yerr=old_COMBINED_MEASURED_IM_AS_error[1:]*100,
#               ls='',marker='o',capsize=5
#               )
# plt.xlabel('slope method')
# plt.ylabel('influence method')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()



# import os
# os.chdir(r'C:\Users\gqmc62\OneDrive - Durham University\Desktop')
# from astropy.io import fits
# from matplotlib import pyplot as plt
# import numpy as np

# wvl = 500e-9
# k = 2*np.pi/wvl
# D = 0.05
# r0 = D

# hdul = fits.open('slopeCovariance.fits')
# data1 = hdul[0].data
# hdul.close()

# hdul = fits.open('referenceFunctions.fits')
# data2 = hdul[0].data
# hdul.close()

# mid = data1.shape[1] //2 + 1
# # A = (data1[0,mid,mid] + data1[1,mid,mid]) * (r0)**(5/3)*dsub**(1/3)*wvl**-2 / 206265**2
# # B = (data2[0,mid,mid] + data2[1,mid,mid]) * (r0)**(5/3)*dsub**(1/3)*wvl**-2 / 206265**2
# A = (data1[0,mid,mid]) * (r0)**(5/3)*dsub**(1/3)*wvl**-2 / 206265**2
# B = (data2[0,mid,mid]) * (r0)**(5/3)*dsub**(1/3)*wvl**-2 / 206265**2


# measure_var_arcsec_slope = ((adjust_COMBINED_MEASURED_IM_AS)*dsub/z)**2*(r0HA)**(5/3)*dsub**(1/3)*wvl**-2
# err1 = (adjust_error_as*dsub/z)**2*(r0HA)**(5/3)*dsub**(1/3)*wvl**-2
# measure_var_arcsec_influ = (old_COMBINED_MEASURED_IM_AS*dsub/z)**2*(r0HA)**(5/3)*dsub**(1/3)*wvl**-2
# err2 = (old_COMBINED_MEASURED_IM_AS_error*dsub/z)**2*(r0HA)**(5/3)*dsub**(1/3)*wvl**-2
# expect_sqsubap_basic = (old_expect_AS*dsub/z)**2*(r0HA)**(5/3)*dsub**(1/3)*wvl**-2

# # argsort = np.arange(23) #np.argsort(expect_sqsubap_basic)

# # plt.hlines(A,z[1:].min()/1000,z.max()/1000,label='expect : butterly-1',color='r')
# plt.hlines(B,z[1:].min()/1000,z.max()/1000,label='expect : butterly-2',color='g')

# # show = [1,3,5,7,9,10,12,14,16,]
# show = np.arange(CASES)

# plt.errorbar(z[show]/1000,measure_var_arcsec_slope[show],yerr=err1[show],
#               label='measure : slope',ls='',marker='.',color='b',capsize=5)
# plt.errorbar(z[show]/1000,measure_var_arcsec_influ[show],yerr=err2[show],
#               label='measure : influ',ls='',marker='.',color='k',capsize=5)
# plt.plot(z[show]/1000,expect_sqsubap_basic[show],label='expect : saint-jacques',ls='-',marker='',color='c')
# plt.legend()
# plt.yscale('log')
# plt.xlabel('z (km)')
# plt.ylabel('angle-of-arrival coefficient (unitless)')
# plt.show()

# plt.hlines(A,0,CASES,label='expect : butterly-1',color='r')
# plt.hlines(B,0,CASES,label='expect : butterly-2',color='g')

# # show = [1,3,5,7,9,10,12,14,16,]
# show = np.arange(CASES)

# plt.errorbar(np.arange(CASES),measure_var_arcsec_slope[show],yerr=err1[show],
#               label='measure : slope',ls='',marker='.',color='b',capsize=5)
# plt.errorbar(np.arange(CASES),measure_var_arcsec_influ[show],yerr=err2[show],
#               label='measure : influ',ls='',marker='.',color='k',capsize=5)
# plt.plot(expect_sqsubap_basic[show],label='expect : saint-jacques',ls='-',marker='',color='c')
# plt.legend()
# plt.yscale('log')
# plt.xlabel('cases')
# plt.ylabel('angle-of-arrival coefficient (unitless)')
# plt.show()

# show = [1,3,5,7,9,10,12,14,16,]
# # show = np.arange(23)

# plt.errorbar(z[show]/1000,measure_var_arcsec_slope[show],yerr=err1[show],
#               label='measure : slope',ls='',marker='o',color='b',capsize=5)
# plt.errorbar(z[show]/1000,measure_var_arcsec_influ[show],yerr=err2[show],
#               label='measure : influ',ls='',marker='o',color='k',capsize=5)
# # plt.plot(z[show]/1000,expect_sqsubap_basic[show],label='expect : saint-jacques',ls='-',marker='',color='c')
# # plt.legend()
# plt.yscale('log')
# plt.xlabel('z (km)')
# plt.ylabel('angle-of-arrival coefficient (unitless)')
# plt.show()

# for i in np.arange(CASES):
#     plt.hist(ANOTHER_AS[i,...,1].flatten()*16,bins=50,alpha=0.5,density=True,range=(-10,10))
#     plt.hist(MEASURED_IM_SUBAP_SHIFT[i,...,1].flatten()*16,bins=50,alpha=0.5,density=True,range=(-10,10))
#     plt.title(i)
#     # plt.xlim(-10,10)
#     plt.show()


# np.average(measure_var_arcsec_influ[1:],weights=1/(err1[1:])**2)
# np.average(measure_var_arcsec_slope[1:],weights=1/(err2[1:])**2)
# B

#!!! need to adjust these following plots

all_expect_AS = np.array([np.min([expect_AS_A[1:],expect_AS_B[1:],expect_AS_C[1:]]),
                          np.max([expect_AS_A[1:],expect_AS_B[1:],expect_AS_C[1:]])])

    
plt.errorbar(expect_rytov_A[1:],rytov_A[1:].flatten(),yerr=rytov_error_A[1:].flatten(),
              ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(dsub_A*100))
plt.errorbar(expect_rytov_B[1:],rytov_B[1:].flatten(),yerr=rytov_error_B[1:].flatten(),
              ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(dsub_B*100))
plt.errorbar(expect_rytov_C[1:],rytov_C[1:].flatten(),yerr=rytov_error_C[1:].flatten(),
              ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(dsub_C*100))
plt.plot(expect_rytov_A[1:],expect_rytov_A[1:])
plt.plot(expect_rytov_B[1:],expect_rytov_B[1:])
plt.plot(expect_rytov_C[1:],expect_rytov_C[1:])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('expect rytov')
plt.ylabel('measured rytov')
plt.legend()
plt.savefig('compare-all-config-rytov.png')
plt.show()

plt.errorbar(expect_AS_A[1:]*100,AS_influ_A[1:].flatten()*100,yerr=AS_influ_error_A[1:].flatten()*100,
              ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(dsub_A*100))
plt.errorbar(expect_AS_B[1:]*100,AS_influ_B[1:].flatten()*100,yerr=AS_influ_error_B[1:].flatten()*100,
              ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(dsub_B*100))
plt.errorbar(expect_AS_C[1:]*100,AS_influ_C[1:].flatten()*100,yerr=AS_influ_error_C[1:].flatten()*100,
              ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(dsub_C*100))
# plt.plot(expect_AS_A[1:]*100,expect_AS_A[1:]*100)
# plt.plot(expect_AS_B[1:]*100,expect_AS_B[1:]*100)
# plt.plot(expect_AS_C[1:]*100,expect_AS_C[1:]*100)
plt.plot(all_expect_AS*100,all_expect_AS*100)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('expect AS (% subap)')
plt.ylabel('measured AS : influence method (% subap)')
plt.legend()
plt.savefig('compare-all-config-as-influ.png')
plt.show()

plt.errorbar(expect_AS_A[1:]*100,AS_slope_A[1:].flatten()*100,yerr=AS_slope_error_A[1:].flatten()*100,
              ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(dsub_A*100))
plt.errorbar(expect_AS_B[1:]*100,AS_slope_B[1:].flatten()*100,yerr=AS_slope_error_B[1:].flatten()*100,
              ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(dsub_B*100))
plt.errorbar(expect_AS_C[1:]*100,AS_slope_C[1:].flatten()*100,yerr=AS_slope_error_C[1:].flatten()*100,
              ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(dsub_C*100))
# plt.plot(expect_AS_A[1:]*100,expect_AS_A[1:]*100)
# plt.plot(expect_AS_B[1:]*100,expect_AS_B[1:]*100)
# plt.plot(expect_AS_C[1:]*100,expect_AS_C[1:]*100)
plt.plot(all_expect_AS*100,all_expect_AS*100)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('expect AS (% subap)')
plt.ylabel('measured AS : slope method (% subap)')
plt.legend()
plt.savefig('compare-all-config-as-slope.png')
plt.show()

plt.clf()



measured_coeff_A = AS_influ_A**2/wvl**2/dsub_A**(-1/3)/r0HA**(-5/3)/z**2*dsub_A**2
measured_coeff_error_A = (2*AS_influ_A*AS_influ_error_A)**2/wvl**2/dsub_A**(-1/3)/r0HA**(-5/3)/z**2*dsub_A**2

measured_coeff_B = AS_influ_B**2/wvl**2/dsub_B**(-1/3)/r0HA**(-5/3)/z**2*dsub_B**2
measured_coeff_error_B = (2*AS_influ_B*AS_influ_error_B)**2/wvl**2/dsub_B**(-1/3)/r0HA**(-5/3)/z**2*dsub_B**2

measured_coeff_C = AS_influ_C**2/wvl**2/dsub_C**(-1/3)/r0HA**(-5/3)/z**2*dsub_C**2
measured_coeff_error_C = (2*AS_influ_C*AS_influ_error_C)**2/wvl**2/dsub_C**(-1/3)/r0HA**(-5/3)/z**2*dsub_C**2

measured_coeff = np.concatenate([measured_coeff_A[1:],measured_coeff_B[1:],measured_coeff_C[1:]])
measured_coeff_error = np.concatenate([measured_coeff_error_A[1:],measured_coeff_error_B[1:],measured_coeff_error_C[1:]])

print('current coeff : {:.3f}'.format(np.average(measured_coeff,weights=1/measured_coeff_error**2)))






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


plt.errorbar(COMBINED_MEASURED_IM_RYTOV,A/new_coeff,
             yerr=Ae/new_coeff, capsize=5,
             label='WFS slope of HADM',marker='o',ls='')
plt.errorbar(COMBINED_MEASURED_IM_RYTOV,B/new_coeff,
         yerr=Be/new_coeff, capsize=5,
         label='WFS slope of GLATM',marker='o',ls='')
plt.errorbar(COMBINED_MEASURED_IM_RYTOV,C/new_coeff,
             yerr=Ce/new_coeff, capsize=5,
             label='measured as : influence method',ls='',marker='o')
plt.errorbar(COMBINED_MEASURED_IM_RYTOV,D/new_coeff,
             yerr=De/new_coeff, capsize=5,
             label='measured as : slope method',ls='',marker='o')
# plt.hlines(1,0,0.6,label='Butterly')
plt.legend()
plt.xlabel('measure rytov')
plt.ylabel('measure as coef (ratio to butterly)')
plt.yscale('log')
plt.ylim(0.1,10)
plt.grid()
plt.show()


plt.errorbar(z/1000,A/new_coeff,
             yerr=Ae/new_coeff, capsize=5,
             label='WFS slope of HADM',marker='o',ls='')
plt.errorbar(z/1000,B/new_coeff,
         yerr=Be/new_coeff, capsize=5,
         label='WFS slope of GLATM',marker='o',ls='')
plt.errorbar(z/1000,C/new_coeff,
             yerr=Ce/new_coeff, capsize=5,
             label='measured as : influence method',ls='',marker='o')
plt.errorbar(z/1000,D/new_coeff,
             yerr=De/new_coeff, capsize=5,
             label='measured as : slope method',ls='',marker='o')
plt.legend()
plt.xlabel('H_HA (km)')
plt.ylabel('measure as coef (ratio to butterly)')
plt.yscale('log')
plt.ylim(0.1,10)
plt.grid()
plt.show()

plt.errorbar(expect_AS,A/new_coeff,
             yerr=Ae/new_coeff, capsize=5,
             label='WFS slope of HADM',marker='o',ls='')
plt.errorbar(expect_AS,B/new_coeff,
         yerr=Be/new_coeff, capsize=5,
         label='WFS slope of GLATM',marker='o',ls='')
plt.errorbar(expect_AS,C/new_coeff,
             yerr=Ce/new_coeff, capsize=5,
             label='measured as : influence method',ls='',marker='o')
plt.errorbar(expect_AS,D/new_coeff,
             yerr=De/new_coeff, capsize=5,
             label='measured as : slope method',ls='',marker='o')
plt.legend()
plt.xlabel('expect as')
plt.ylabel('measure as coef (ratio to butterly)')
plt.yscale('log')
plt.xscale('log')
plt.ylim(0.1,10)
plt.grid()
plt.show()


# COEFFA,COEFFAe = np.average(A[~np.isnan(A)],weights=1/(Ae[~np.isnan(A)])**2,returned=True)
# COEFFAe = COEFFAe**-0.5

# COEFFB,COEFFBe = np.average(B[~np.isnan(B)],weights=1/(Be[~np.isnan(B)])**2,returned=True)
# COEFFBe = COEFFBe**-0.5

# COEFFC,COEFFCe = np.average(C[1:][~np.isnan(C[1:])],weights=1/(Ce[1:][~np.isnan(C[1:])])**2,returned=True)
# COEFFCe = COEFFCe**-0.5

# COEFFD,COEFFDe = np.average(D[1:][~np.isnan(D[1:])],weights=1/(De[1:][~np.isnan(D[1:])])**2,returned=True)
# COEFFDe = COEFFDe**-0.5

# plt.errorbar(['WFS Slope of HADM','WFS Slope of GLATM','Influence Method','Slope Method'],
#              [COEFFA,COEFFB,COEFFC,COEFFD],
#              yerr=[COEFFAe,COEFFBe,COEFFCe,COEFFDe],
#              capsize=5, marker='o',ls='')
# plt.hlines([new_coeff],0,3)
# plt.ylabel('Coefficient')

# plt.show()

COEFFA,COEFFAe = np.average(A[~np.isnan(A)],weights=1/(Ae[~np.isnan(A)])**2,returned=True)
COEFFAe = COEFFAe**-0.5

COEFFB,COEFFBe = np.average(B[~np.isnan(B)],weights=1/(Be[~np.isnan(B)])**2,returned=True)
COEFFBe = COEFFBe**-0.5

COEFFC,COEFFCe = np.average(measured_coeff,weights=1/measured_coeff_error**2,returned=True)
COEFFCe = COEFFCe**-0.5

COEFFD,COEFFDe = np.average(D[1:][~np.isnan(D[1:])],weights=1/(De[1:][~np.isnan(D[1:])])**2,returned=True)
COEFFDe = COEFFDe**-0.5

plt.errorbar(['WFS Slope of HADM','WFS Slope of GLATM','Influence Method','Slope Method'],
             [COEFFA,COEFFB,COEFFC,COEFFD],
             yerr=[COEFFAe,COEFFBe,COEFFCe,COEFFDe],
             capsize=5, marker='o',ls='')
plt.hlines([new_coeff],0,3)
plt.ylabel('Coefficient')

plt.show()





# INFLU = np.concatenate([as_influ_a.flatten(),as_influ_b.flatten()])
# SLOPE = np.concatenate([as_slope_a[:,:,2:].flatten(),as_slope_b[:,:,2:].flatten()])

# Y=SLOPE[INFLU>0.05]
# X=INFLU[INFLU>0.05]
# Y=Y[X<0.4]
# X=X[X<0.4]
# Y=Y[~np.isnan(X)]
# X=X[~np.isnan(X)]
# X=X[~np.isnan(Y)]
# Y=Y[~np.isnan(Y)]


# m, b = np.polyfit(np.log10(X), np.log10(Y), 1)

# plt.plot(as_influ_a.flatten(),as_slope_a.flatten(),ls='',marker=',',alpha=1)
# plt.plot(as_influ_b.flatten(),as_slope_b.flatten(),ls='',marker=',',alpha=1)
# plt.plot([0.01,1],[0.01**m*10**b,1**m*10**b])
# plt.xlim(0.01,1)
# plt.ylim(0.01,1)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('influence method (dm pitch)')
# plt.ylabel('slope method (dm pitch)')
# plt.title('y={:.3f}*x**{:.3f}'.format(10**b,m))
# plt.show()
# plt.clf()

# # m, b = np.polyfit(X, Y, 1)
# # for cases in range(CASES):
# cases = 1
# plt.plot(as_influ_a[cases].flatten(),as_slope_a[cases,:,2:].flatten(),ls='',marker=',',alpha=0.5)
# # plt.plot(as_influ_b[cases].flatten(),as_slope_b[cases,:,2:].flatten(),ls='',marker=',',alpha=0.5)
# # plt.plot(as_influ_c[cases].flatten(),as_slope_c[cases,:,2:].flatten(),ls='',marker=',',alpha=0.5)
# # plt.plot([0.01,1],[0.01*m+b,1*m+b])
# # plt.xlim(0.01,1)
# # plt.ylim(0.01,1)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.xlabel('influence method (dm pitch)')
# plt.ylabel('slope method (dm pitch)')
# plt.title('one measurement')
# # plt.title('y={:.3f}*x+{:.3f}'.format(m,b))
# # plt.savefig()
# plt.savefig('as_i_influ_vs_slope')
# plt.show()
# plt.clf()

# try:
#     plt.plot(as_influ_a.flatten(),as_slope_a.flatten(),ls='',marker=',')
#     plt.plot(as_influ_b.flatten(),as_slope_b.flatten(),ls='',marker=',')
#     # plt.plot(as_influ_c.flatten(),as_slope_c.flatten(),ls='',marker=',')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlim(0.01,1)
#     plt.ylim(0.01,1)
#     # plt.plot([0.01,1],[0.01,1])
#     # plt.plot([0.01,1],[0.01*1.5,1*1.5])
#     plt.xlabel('actuator shift - influence method (DM pitch)')
#     plt.ylabel('actuator shift - slope method (DM pitch)')
#     plt.savefig('as_i_influ_vs_slope')
    
    
# except:
#     plt.show()
#     plt.clf()
#     pass
# plt.show()


# P = np.concatenate([AS_influ_A,AS_influ_B])
# Pe = np.concatenate([AS_influ_error_A,AS_influ_error_B])
# Q = np.concatenate([AS_slope_A,AS_slope_B])
# Qe = np.concatenate([AS_slope_error_A,AS_slope_error_B])

# plt.errorbar(P,Q,xerr=Pe,yerr=Qe,ls='',marker='o',capsize=5)

# plt.errorbar(AS_influ_A,AS_slope_A,
#              xerr=AS_influ_error_A,yerr=AS_slope_error_A,
#              ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(0.4/8*100))
# # plt.errorbar(AS_influ_B,AS_slope_B,
# #              xerr=AS_influ_error_B,yerr=AS_slope_error_B,
# #              ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(0.4/4*100))
# # plt.errorbar(AS_influ_C,AS_slope_C,
# #              xerr=AS_influ_error_C,yerr=AS_slope_error_C,
# #              ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(0.4/12*100))
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Influence Method (DM pitch)')
# plt.ylabel('Slope Method (DM pitch)')
# plt.title('rms of an atm stat')
# plt.plot([0.01,1],[0.01,1])
# plt.xlim(0.01,1)
# plt.ylim(0.01,1)
# plt.legend()
# # plt.savefig('as_influ_vs_slope')
# plt.show()

# # plt.errorbar(AS_influ_A,AS_slope_A,
# #              xerr=AS_influ_error_A,yerr=AS_slope_error_A,
# #              ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(0.4/8*100))
# plt.errorbar(AS_influ_B,AS_slope_B,
#              xerr=AS_influ_error_B,yerr=AS_slope_error_B,
#              ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(0.4/4*100))
# # plt.errorbar(AS_influ_C,AS_slope_C,
# #              xerr=AS_influ_error_C,yerr=AS_slope_error_C,
# #              ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(0.4/12*100))
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Influence Method (DM pitch)')
# plt.ylabel('Slope Method (DM pitch)')
# plt.title('rms of an atm stat')
# plt.plot([0.01,1],[0.01,1])
# plt.xlim(0.01,1)
# plt.ylim(0.01,1)
# plt.legend()
# # plt.savefig('as_influ_vs_slope')
# plt.show()


# # plt.errorbar(AS_influ_A,AS_slope_A,
# #              xerr=AS_influ_error_A,yerr=AS_slope_error_A,
# #              ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(0.4/8*100))
# # plt.errorbar(AS_influ_B,AS_slope_B,
# #              xerr=AS_influ_error_B,yerr=AS_slope_error_B,
# #              ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(0.4/4*100))
# plt.errorbar(AS_influ_C,AS_slope_C,
#              xerr=AS_influ_error_C,yerr=AS_slope_error_C,
#              ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(0.4/12*100))
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Influence Method (DM pitch)')
# plt.ylabel('Slope Method (DM pitch)')
# plt.title('rms of an atm stat')
# plt.plot([0.01,1],[0.01,1])
# plt.xlim(0.01,1)
# plt.ylim(0.01,1)
# plt.legend()
# # plt.savefig('as_influ_vs_slope')
# plt.show()


# plt.errorbar(AS_influ_A,AS_slope_A,
#               xerr=AS_influ_error_A,yerr=AS_slope_error_A,
#               ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(0.4/8*100))
# plt.errorbar(AS_influ_B,AS_slope_B,
#               xerr=AS_influ_error_B,yerr=AS_slope_error_B,
#               ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(0.4/4*100))
# plt.errorbar(AS_influ_C,AS_slope_C,
#               xerr=AS_influ_error_C,yerr=AS_slope_error_C,
#               ls='',marker='o',capsize=5,label='subap:{:.2f}cm'.format(0.4/12*100))
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Influence Method (DM pitch)')
# plt.ylabel('Slope Method (DM pitch)')
# plt.title('rms of an aberration stat')
# plt.plot([0.01,1],[0.01,1])
# plt.xlim(0.01,1)
# plt.ylim(0.01,1)
# plt.legend()
# plt.savefig('as_influ_vs_slope')
# plt.show()


# plt.errorbar(expect_AS_A[WFE_A!=0],WFE_A[WFE_A!=0],yerr=WFE_A_error[WFE_A!=0],capsize=5,
#              ls='',marker='o',label='subap:{:.2f}cm'.format(0.4/8*100))
# plt.errorbar(expect_AS_B[WFE_B!=0],WFE_B[WFE_B!=0],yerr=WFE_B_error[WFE_B!=0],capsize=5,
#              ls='',marker='o',label='subap:{:.2f}cm'.format(0.4/4*100))
# plt.errorbar(expect_AS_C[WFE_C!=0],WFE_C[WFE_C!=0],yerr=WFE_C_error[WFE_C!=0],capsize=5,
#              ls='',marker='o',label='subap:{:.2f}cm'.format(0.4/12*100))
# plt.yscale('log')
# plt.xscale('log')
# plt.show()

# plt.errorbar(expect_rytov_A[WFE_A!=0],WFE_A[WFE_A!=0],yerr=WFE_A_error[WFE_A!=0],capsize=5,
#              ls='',marker='o',label='subap:{:.2f}cm'.format(0.4/8*100))
# plt.errorbar(expect_rytov_B[WFE_B!=0],WFE_B[WFE_B!=0],yerr=WFE_B_error[WFE_B!=0],capsize=5,
#              ls='',marker='o',label='subap:{:.2f}cm'.format(0.4/4*100))
# plt.errorbar(expect_rytov_C[WFE_C!=0],WFE_C[WFE_C!=0],yerr=WFE_C_error[WFE_C!=0],capsize=5,
#              ls='',marker='o',label='subap:{:.2f}cm'.format(0.4/12*100))
# plt.yscale('log')
# plt.xscale('log')
# plt.show()


# MAX = np.nanmax([WFE_A,WFE_B,WFE_C])

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# for i in range(CASES-1):
#     plt.plot(expect_rytov_A[1:][i],expect_AS_A[1:][i],marker='o',color=cmap(WFE_A[1:][i]/MAX))
# plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,np.nanmax(TOPLOT)),cmap=cmap),ax=ax1)
# plt.xlim(0.01/1.1,0.6*1.1)
# plt.ylim(0.01/1.1,1*1.1)
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# for i in range(CASES-1):
#     plt.plot(expect_rytov_B[1:][i],expect_AS_B[1:][i],marker='o',color=cmap(WFE_B[1:][i]/MAX))
# plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,np.nanmax(TOPLOT)),cmap=cmap),ax=ax1)
# plt.xlim(0.01/1.1,0.6*1.1)
# plt.ylim(0.01/1.1,1*1.1)
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# for i in range(CASES-1):
#     plt.plot(expect_rytov_C[1:][i],expect_AS_C[1:][i],marker='o',color=cmap(WFE_C[1:][i]/MAX))
# plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,np.nanmax(TOPLOT)),cmap=cmap),ax=ax1)
# plt.xlim(0.01/1.1,0.6*1.1)
# plt.ylim(0.01/1.1,1*1.1)
# plt.xscale('log')
# plt.yscale('log')
# plt.show()



