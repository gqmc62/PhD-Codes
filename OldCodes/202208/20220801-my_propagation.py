
###############################################################################
# Start of file ########################################################
###############################################################################
"""
Library of AO codes on Adaptive Optics (AO): propagation study.

Implemented by Puttiwat Kongkaew (Geng/Greg) during his study
for Ph.D. in Physics (Advanced Instrumentation)
at Centre for Advanced Instrumentation, Department of Physics,
Durham University,United Kingdom from October 2021 to
(ETG) September 2024.
I will try to keep everything under my universal namings.
units, etc. to keep confusions at minimal.

Updated as of 06 April 2022.

"""
###############################################################################
# List of functions collected within this file #########################
###############################################################################
# 6) Propagation report. #
#       prop_show()
#       Propagate then show optical field with 1-step angular-spectrum.
# 7) Zero pad. #
#       zero_pad2()
#       Zero pad 2D input to factor*original_size.
# 8) Undo zero pad. #
#       crop2()
#       Cropped out the zero pad of 2D input to original_size/factor.
# 9) Gaussian curve. #
#       gauss()
#       Generate Gaussian probability distribution curve.
# 10) Propagation test. #
#       prop_test()
#       Test angular-spectrum Fresnel propagation at different distances.
# 11) Variance of log amplitude over propagation distances. #
#       var_log_amp()
#       Compute variance log amplitude versus propagation distances..
# 12) Measure scintillation size. #
#       scintillation_size()
#       Measure scintillation blob size.
# 13) Laurentzian curve. #
#       lorentz()
#       Generate Lorentzian probability distribution curve.
# 14) Sinc exponentially decaying curve. #
#       sinc_exp_decay()
#       Generate sinc exponentially decaying function.
# 15) Diagonally stack a 2darray into 1darray. #
#       diagonal_stack()
#       Diagonally stack 2darray into 1darray.
# 16) Exponential decay. #
#       exp_decay()
#       Exponential decay of linear order.
# 17) Decayed sinc function for calculating Full Width Half Max. #
#       sinc_exp_decay_half()
#       Use to find FWHM of sinc exp decay function together with fsolve.
# 18) My airy disk. #
#       intensity_shape()
#       Make shape of intensity psf -ish of circular aperture.
# ...
# ...
# ...
# 22) Parameters check. #
#       prop_param_check()
#       Check propagation parameter whether or not it is good.
###############################################################################
########################################################################

#
# Imports #
#
import numpy as np
from aotools.turbulence.phasescreen import ft_sh_phase_screen
from aotools.opticalpropagation import angularSpectrum
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.special import j1

import my_WFS as mwfs

#
# Preferences #
#
color_map = plt.cm.get_cmap('viridis')
reversed_color_map = color_map.reversed()

###############################################################################
########################################################################

# 6) Propagation report. #
def prop_show(U_in, d1, wvl, d2, z, pfp):
    """
    Propagate then show optical field with 1-step angular-spectrum.

    Revision needed.

    1) Zero pad to pad_factor (adjustable within this function only).
    2) Angular-spectrum Fresnel Propagate from plane_in to plane_out.
    3) Plot input and output phase and amplitude.
    4) Return cropped U_out.

    Parameters
    ----------
    U_in : ndarray
        Dimension = square =(NN,NN).
        Element type is complex.
        Optical field at the source plane.
        Unit is not specified (ie. sqrt(Intensity)?).
    d1 : float
        Pixel scale at the source plane.
        Unit is m.
    wvl : float
        Wavelength of propagation.
        Unit is m
    d2 : float
        Pixel scale at the observation plane.
        Unit is m.
    z : float
        Propagation distance.
        Unit is m.
    pfp : int
        Padding factor of plottings.
        Actually the oversize factor, but I am too lazy to change.

    Returns
    -------
    U_out : ndarray
        Same dimension with U_in.
        Element type is complex.
        Optical field at the observation plane.
        Same Unit with U_in.

    Example
    -------
    N = 1024
    pfp = 2
    NN = pf*N
    d1 = 5e-3
    wvl = 500e-9
    d2 = 5e-3
    z = 5e3
    amp_in = np.ones((NN,NN), dtype=float)
    phase_in = np.zeros((NN,NN), dtype=float)
    U_in = np.zeros((NN,NN), dtype=complex)
    U_out = np.zeros((NN,NN),dtype=complex)
    for i in np.arange(NN):
        for j in np.arange(NN):
            phase_in[i,j] = 2*np.sin(i) + np.cos(j)**2
    U_in = amp_in*np.exp(1j*phase_in)
    U_out = prop_show(U_in, d1, wvl, d2, z, pfp)

    """
    # Inputs #
    U_in = np.asarray(U_in, dtype=complex)
                                    # (complex,complex) input complex amplitude
                                    # (NN,NN)
    d1 = float(d1)                  # float; input pixel scale (m)
    wvl = float(wvl)                # float; propagated wavelength (m)
    d2 = float(d2)                  # float; output pixel scale (m)
    z = float(z)                    # float; propagated distance (m)
    pfp = int(pfp)                  # int; (undo)padding factor for plots

    # Constants #
#    pfp = 2                    # int; (undo) padding factor for plotting
    NN = int(len(U_in))             # int; oversize size
    N = int(NN/pfp)                 # int; desired size

    check = True
    check_condition = prop_param_check(N, d1, d2, wvl, [z], pfp)
    for i in check_condition:
        check = check * i
    
    if check == False:
        print("One of the condition is failing.")
        return

    U_out = np.zeros((NN,NN), dtype=complex)
                                    # (complex,complex) output complex amp
                                    # (NN,NN)
    phase_out = np.zeros((N,N), dtype=complex)
                                    # (complex,complex) output phase (rad)
                                    # (N,N)
    I_out = np.zeros((N,N), dtype=complex)
                                    # (complex,complex) output intensity
                                    # (N,N)
    # log_amp_out = np.zeros((N,N), dtype=complex)
                                    # (complex,complex) output log amplitude
                                    # (N,N)

    x = np.arange(-N/2, N/2) * d2
    xx, yy = np.meshgrid(x,x)

    # Propagate #
    # Using aotools function
    U_out = angularSpectrum(U_in, wvl, d1, d2, z)

    # Process output #
    # Using euler notation
    # Plot cropped version
    phase_out = np.angle(crop2(U_out, pad_factor=pfp))
    I_out = np.abs(crop2(U_out, pad_factor=pfp))**2
    # log_amp_out = np.log(np.abs(crop2(U_out,pad_factor=pfp)))

    # Plots #
    plt.pcolor(xx, yy, I_out, cmap=reversed_color_map, shading='auto')
    plt.title('I out at z={:.0f}m'.format(z))
    # plt.title('log amp out {:.0f}'.format(z))
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar()
    plt.show()

    plt.pcolor(xx, yy, phase_out, cmap=reversed_color_map, shading='auto')
    plt.axis('equal')
    plt.title('Phase out at z={:.0f}m'.format(z))
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar()
    plt.show()

    # Return #
    # Return uncropped version
    return U_out

###############################################################################
########################################################################

# 7) Zero pad. #
def zero_pad2(A, pad_factor=2):
    """
    Zero pad 2D input to factor*original_size.

    Parameters
    ----------
    A : ndarray
        2D numpy array.
        Dimension = (N,N)
        Support any element type.
        Array to be zero pad to new size = old size * pad_factor.
    pad_factor : int, optional
        New size of A with the zero pad.
        Must be larger than or equal to 1.
        The default is 2.

    Returns
    -------
    zeropad : ndarray
        2D numpy array.
        Dimension = (pad_factor*N, pad_factor*N).
        Element type is the same with the input.
        Array which got zero padded
        with original input placed in the center.

    """
    # Process input #
    N = len(A)
    pad_factor = int(pad_factor)
    NN = pad_factor*N

    # Declaration field #
    zeropad = np.zeros((NN, NN), dtype=type(A[0,0]))

    # Find where old data should be #
    edge_low = int((pad_factor - 1)*N/2)
    edge_high = int((pad_factor + 1)*N/2)

    # Place old data in a new array #
    zeropad[edge_low : edge_high, edge_low : edge_high] = A

    # Return #
    return zeropad

###############################################################################
########################################################################

# 8) Undo zero pad. #
def crop2(A, pad_factor=2):
    """
    Cropped out the zero pad of 2D input to original_size/factor.

    Parameters
    ----------
    A : ndarray
        2D numpy array.
        Dimension = (NN,NN)
        Support any element type.
        Array to be undo zero pad to new size = old size / pad_factor.
    pad_factor : int, optional
        New size ratio of A when it is cropped.
        Must be larger than or equal to 1.
        The default is 2.

    Returns
    -------
    cropped : ndarray
        2D numpy array.
        Dimension = (NN/pad_factor, NN/pad_factor).
        Element type is the same with the input.
        Array which got cropped.
        It is center-aligned with the input.

    """
    NN = len(A)
    pad_factor = int(pad_factor)
    edge_low = int(NN*((pad_factor - 1)/pad_factor)/2)
    edge_high = int(NN*((pad_factor + 1)/pad_factor)/2)
    cropped = A[edge_low : edge_high, edge_low : edge_high]
    return cropped

###############################################################################
########################################################################

# 9) Gaussian curve. #
def gauss(x,a,mu,sigma,c):
    """
    Generate Gaussian probability distribution curve.

    Parameters
    ----------
    x : 1darray
        Element type = float.
        Represent 'x' of the measurements.
        'Observed value'
    a : float
        Amplitude / max events of the distribution.
        (max of 'y').
    mu : float
        Mean of the measurement.
        ('x' which have the higest 'y')
    sigma : float
        Standard deviation of the curve.
    c : float
        Base height.

    Returns
    -------
    y : 1darray
        Element type = float.
        Represent 'y' of the measurements.
        'Events'

    """
    y = a*np.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma*np.sqrt(2*np.pi)) + c
    return y

###############################################################################
########################################################################

# 10) Propagation test. #
def prop_test(N, d1, wvl, d2, z_list, pf):
    """
    Test angular-spectrum Fresnel propagation at different distances.

    Details
    -------
    1) Plane wave.
    2) Point source on edge.
    3) Does oversize with a 'pf (pad_factor)'.
        (Adjustable within this function only).
    4) Plot out propagation results at each propagation distances.

    Parameters
    ----------
    N : int
        Output screen size in number of pixel.
        Diameter.
        Unit is pixel.
    d1 : float
        Pixel scale at the source plane.
        Unit is m.
    wvl : float
        Wavelength of propagation.
        Unit is m
    d2 : float
        Pixel scale at the observation plane.
        Unit is m.
    z_list : ndarray
        Element type is float.
        Propagation distances.
        Unit is m.
    pf : int
        Padding factor.
        Actually the oversize factor, but I am too lazy to change.

    Returns
    -------
    None

    Example
    -------
    prop_test(1024, 5e-3, 500e-9, 5e-3, np.logspace(0, 2, num=10) * 1e3, 2)

    """
    # Process input.
    N = int(N)                      # int; pixel; Desired output screen size.
    d1 = float(d1)                  # float; m; Input pixel size.
    wvl = float(wvl)                # float; m; Propagated wavelength.
    d2 = float(d2)                  # float; m; Output pixel size.
    z_list = np.asarray(z_list, dtype=float)    # ndarray; (num_z); m;
                                                # Propagated distance.
    pf = int(pf)                    # int; Oversize/padding factor.

    # Global constant.
    NN = pf*N                       # int; pixel; Propagation screen size.
    
    check = True
    check_condition = prop_param_check(N, d1, d2, wvl, z_list, pf)
    for i in check_condition:
        check = check * i
    
    if check == False:
        print("One of the condition is failing.")

    # Variables declaration.
    amp_in = np.zeros((NN, NN), dtype=float)
                    # ndarray; (NN,NN); (float,float); Input amplitude.
    phase_in = np.ones((NN, NN), dtype=float)
                    # ndarray; (NN,NN); (float,float); (rad,rad); Input phase.
    U_in = np.zeros((NN, NN), dtype=complex)
                    # ndarray; (NN,NN); (complex,complex); Input complex amp.
    U_out = np.zeros((NN,NN), dtype=complex)
                    # ndarray; (NN,NN); (complex,complex); Output complex amp.
    phase_out = np.zeros((NN,NN), dtype=float)
                    # ndarray; (NN,NN); (float,float); (rad,rad); Output phase.
    I_out = np.zeros((NN,NN), dtype=float)
                    # ndarray; (NN,NN); (float,float); Output intensity.

    # Set wave at the origin plane.
    amp_in[0,0] = 10 * (N*pf)**2
    U_in = amp_in*np.exp(1j*phase_in)

    # Propagate input wave to different distances.
    for zi in z_list:
        U_out = angularSpectrum(U_in, wvl, d1, d2, zi)

        phase_out = np.angle(U_out)
        I_out = np.abs(U_out)**2

        plt.imshow(I_out, cmap=reversed_color_map)
        plt.vlines(int(N*(pf - 1)/2),
                   ymin=int(N*(pf - 1)/2), ymax=int(N*(pf + 1)/2),
                   color='k')
        plt.vlines(int(N*(pf + 1)/2),
                   ymin=int(N*(pf - 1)/2), ymax=int(N*(pf + 1)/2),
                   color='k')
        plt.hlines(int(N*(pf - 1)/2),
                   xmin=int(N*(pf - 1)/2), xmax=int(N*(pf + 1)/2),
                   color='k')
        plt.hlines(int(N*(pf + 1)/2),
                   xmin=int(N*(pf - 1)/2), xmax=int(N*(pf + 1)/2),
                   color='k')
        plt.title('I out at z={:.0f}m'.format(zi))
        plt.colorbar()
        plt.xlabel('x (px)')
        plt.ylabel('y (px)')
        plt.show()

        plt.imshow(phase_out, cmap=reversed_color_map)
        plt.vlines(int(N*(pf - 1)/2),
                   ymin=int(N*(pf - 1)/2), ymax=int(N*(pf + 1)/2),
                   color='k')
        plt.vlines(int(N*(pf + 1)/2),
                   ymin=int(N*(pf - 1)/2), ymax=int(N*(pf + 1)/2),
                   color='k')
        plt.hlines(int(N*(pf - 1)/2),
                   xmin=int(N*(pf - 1)/2), xmax=int(N*(pf + 1)/2),
                   color='k')
        plt.hlines(int(N*(pf + 1)/2),
                   xmin=int(N*(pf - 1)/2), xmax=int(N*(pf + 1)/2),
                   color='k')
        plt.title('Phase out at z={:.0f}m'.format(zi))
        plt.colorbar()
        plt.xlabel('x (px)')
        plt.ylabel('y (px)')
        plt.show()

    return None

###############################################################################
########################################################################

# 11) Variance of log amplitude over propagation distances. #
#       == Rytov number!!!
#           varied with np.cos(zenith_distance)**(-11./6.)
def var_log_amp(N, r, d_in, L0, l0, wvl, d_out, z_list, pf):
    """
    Compute variance log amplitude versus propagation distances.

    See Equation 4.71, page 111, section 4.9, chapter 4, from
    electromagnetic wave propagation in turbulence,
    evaluation and application of mellin transforms,
    second edition, by Richard J. Sasiela

    Parameters
    ----------
    N : int
        Desired output screen diameter size.
        Unit is pixel.
    r : float
        Screen strength.
        Unit is m.
    d_in : float
        Input pixel scale size.
        Unit is m.
    L0 : float
        Outer scale size.
        Unit is m.
        Suggestion: 20m.
    l0 : float
        Inner scale size.
        Unit is m.
        Suggestion: 1.5e-10m.
    wvl : float
        Propagation wavelength.
        Unit is m.
        Suggestion: 500e-9m.
    d_out : float
        Output pixel scale size.
        Unit is m.
        Suggestion: = d_in
    z_list : 1darray
        Element type is float.
        Propagation distances.
        Unit is m.
    pf : int
        Padding factor.
        Actually the oversize factor, but I am too lazy to change.

    Returns
    -------
    variance_theory_approx : 1darray
        Element type is float.
        Length = len(z).
        Unit = unitless^2.
        Theoretical approximation of
            Variance of log amplitude
            of propagated screen at each distances.
    variance_log_amp : 1darray
        Element type is float.
        Length = len(z).
        Unit = unitless^2.
        Variance of log amplitude
            of propagated screen at each distances.
    error_variance_log_amp : 1darray
        Element type is float.
        Length = len(z).
        Unit = unitless^2.
        Error of variance of log amplitude
            of propagated screen at each distances.

    Example
    -------
    var_log_amp(1024, 0.05, 5e-3, 20, 1.5e-10, 500e-9, 5e-3,
                np.logspace(np.log(1e3)/np.log(10),
                            np.log(25e3)/np.log(10), num=10), 2)

    """
    # Processing input
    N = int(N)                      # int; pixel; Desired output pixel.
    r = float(r)                    # float; m; Turbulence strength.
    d_in = float(d_in)              # float; m; Input pixel size.
    L0 = float(L0)                  # float; m; Outer scale.
    l0 = float(l0)                  # float; m; Inner scale.
    wvl = float(wvl)                # float; m; Wavelength.
    d_out = float(d_out)            # float; m; Output pixel size.
    z_list = np.asarray(z_list, dtype=float)    # ndarray; float; m;
                                                # Propagated distances.
    pf = int(pf)                    # Int; Pad/oversize factor.

    # Constants
    NN = pf*N       # Int; pixel; Padded/oversized screen size.

    check = True
    check_condition = prop_param_check(N, d_in, d_out, wvl, z_list, pf)
    for i in check_condition:
        check = check * i
    
    if check == False:
        print("One of the condition is failing.")
        return


    # Declaration field
    amp_in = np.ones((NN, NN), dtype=float)         # Input amplitude.
    phase_in = np.ones((NN, NN), dtype=float)       # Input phase (rad).
    U_in = np.zeros((NN, NN), dtype=complex)        # Input complex amp.

    U_out = np.zeros((N, N), dtype=complex)         # Output complex amp.
    log_amp_out = np.zeros((N, N), dtype = float)   # Output log amp.

    variance_log_amp = np.zeros_like(z_list, dtype=float)        # variance.
    error_variance_log_amp = np.zeros_like(z_list, dtype=float)  # err of var.
    variance_theory_approx = np.zeros_like(z_list, dtype=float)  # expect var.

    # Initialization field
    k = 2.*np.pi/wvl                    # float; rad/m; Wave number.
    cn2dz=r**(-5./3.)/0.423/k**2        # float; Turbulence strength.

    phase_in = ft_sh_phase_screen(r, NN, d_in, L0, l0)
    U_in = amp_in*np.exp(1j*phase_in)

    # Propagate input wave to different distances
    for iz in np.arange(len(z_list)):
        U_out = crop2(angularSpectrum(U_in, wvl, d_in, d_out, z_list[iz]),
                      pad_factor=pf)
        # Get log amplitude
        log_amp_out = np.log(np.abs(U_out))
        # Get variance of log amplitude
        variance_log_amp[iz] = np.var(log_amp_out.flatten(), ddof=1)

    # Calculating error of variance
    error_variance_log_amp = 2.*variance_log_amp/np.sqrt(2.*N - 2.)

    # Theoretical approximation of variance of log amplitude
    variance_theory_approx = 0.563 * k**(7./6.) * cn2dz * z_list**(5./6.)

    # Plots
    plt.errorbar(z_list, variance_log_amp, yerr=error_variance_log_amp,
                 label='observe', color='r', marker='o', ls='')
    plt.plot(z_list, variance_theory_approx, label='theory',
             color='g', marker='', ls='--')
    plt.hlines(0.5, xmin=z_list.min(), xmax=z_list.max(), label='0.5',
               color='b', ls='--')
    plt.hlines(0.25, xmin=z_list.min(), xmax=z_list.max(), label='0.25',
               color='k', ls='--')
    plt.vlines((2. * 0.563 * k**(7./6.) * cn2dz)**(-6./5.),
               ymin=variance_log_amp.min(),
               ymax=variance_log_amp.max(),
               label='variance reach 0.50', color='c', ls='--')
    plt.vlines((4. * 0.563 * k**(7./6.) * cn2dz)**(-6./5.),
               ymin=variance_log_amp.min(),
               ymax=variance_log_amp.max(),
               label='variance reach 0.25', color='m', ls='--')
    # plt.vlines((N * (pf - 1) * d_in**2) / (2.*wvl),
    #            ymin=variance_log_amp.min(),
    #            ymax=variance_log_amp.max(),
    #            label='model valid to this z', color='c', ls='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Propagation distance (m)')
    plt.ylabel('Variance of log amplitude $\sigma^2_\chi$')
    plt.title('Variance of log amplitude versus propagation distance\n$r_i$={:.1f}cm'.format(100*r))
    plt.show()

    return variance_theory_approx, variance_log_amp, error_variance_log_amp

###############################################################################
########################################################################

# 12) Measure scintillation size. #
def scintillation_size(N, r, d_in, L0, l0, wvl, d_out, z_list, pf):
    """
    Measure scintillation blob size.
    
    This method is still in a test mode.
    
    Parameters
    ----------
    N : int
        Desired output screen diameter size.
        Unit is pixel.
    r : float
        Screen strength.
        Unit is m.
    d_in : float
        Input pixel scale size.
        Unit is m.
    L0 : float
        Outer scale size.
        Unit is m.
        Suggestion: 20m.
    l0 : float
        Inner scale size.
        Unit is m.
        Suggestion: 1.5e-10m.
    wvl : float
        Propagation wavelength.
        Unit is m.
        Suggestion: 500e-9m.
    d_out : float
        Output pixel scale size.
        Unit is m.
        Suggestion: = d_in
    z_list : 1darray
        Element type is float.
        Propagation distances.
        Unit is m.
    pf : int
        Padding factor.
        Actually the oversize factor, but I am too lazy to change.

    Returns
    -------
    expected_FWHM : 1darray
        Based on Fresnel zone size.
        Currently is 1/8th. No good reason besides that it actually fits.
    FWHM_cov_I : 1darray
        DESCRIPTION.

    Example
    -------
    z_list = np.logspace(np.log(1e3)/np.log(10),
                         np.log(25e3)/np.log(10), num=10)
    d_in = float(5e-3)
    d_out = d_in
    N = int(1024)
    r = float(0.05)
    L0 = float(20)
    l0 = float(1.5e-10)
    wvl = float(500e-9)
    pf = int(2)
    k = float(2*np.pi/wvl)
    cn2dz = float(r**(-5./3.)/0.423/k**2)

    expected_FWHM, FWHM_cov_I = scintillation_size(N, r, d_in, L0, l0, wvl,
                                                   d_out, z_list, pf)

    """
    #
    # Processing input #
    #
    N = int(N)                      # int; pixel; Desired output pixel.
    r = float(r)                    # float; m; Turbulence strength.
    d_in = float(d_in)              # float; m; Input pixel size.
    L0 = float(L0)                  # float; m; Outer scale.
    l0 = float(l0)                  # float; m; Inner scale.
    wvl = float(wvl)                # float; m; Wavelength.
    d_out = float(d_out)            # float; m; Output pixel size.
    z_list = np.asarray(z_list, dtype=float)    # ndarray; float; m;
                                                # Propagated distances.
    pf = int(pf)                    # Int; Pad/oversize factor.

    #
    # Indenpendent constants #
    #
    fitr = int(64)

    #
    # Dependent constants #
    #
    NN = int(pf*N)       # Int; pixel; Padded/oversized screen size.
    k = float(2.*np.pi/wvl)                    # float; rad/m; Wave number.
    cn2dz = float(r**(-5./3.)/0.423/k**2)        # float; Turbulence strength.

    check = True
    check_condition = prop_param_check(N, d_in, d_out, wvl, z_list, pf)
    for i in check_condition:
        check = check * i
    if check == False:
        print("One of the condition is failing.")
        return

    #
    # Declaration field #
    #
    amp_in = np.ones((NN,NN), dtype=float)          # Input amplitude.
    phase_in = np.ones((NN, NN), dtype=float)       # Input phase (rad).
    U_in = np.zeros((NN,NN), dtype=complex)         # Input complex amp.

    U_out = np.zeros((NN,NN), dtype=complex)          # Output complex amp.
    I_out = np.zeros((N,N), dtype=float)            # Output intensity.

    cov_I = np.zeros((N,N), dtype=float)
    stack_cov_I = np.zeros(2*N, dtype=float)
    error_stack_cov_I = np.zeros(2*N, dtype=float)

    expected_FWHM = np.zeros(len(z_list), dtype=float)
    FWHM_cov_I = np.zeros(len(z_list), dtype=float)
    # FWHM_cov_I_error = np.zeros_like(z_list, dtype=float)

    c = float(0)
    a = float(0)

    #
    # Initialization field #
    #
    phase_in = ft_sh_phase_screen(r, NN, d_in, L0, l0)
    U_in = amp_in*np.exp(1j*phase_in)
    phase_in = np.angle(crop2(U_in, pf))
    x = np.arange(-N/2.,N/2.)*d_in
    xx, yy = np.meshgrid(x,x)
    plt.pcolor(xx, yy, phase_in, shading='auto', cmap=reversed_color_map)
    plt.title('Wrapped phase in (rad)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar()
    plt.axis('equal')
    plt.show()

    #
    # Loop over each propagation distance #
    #
    for iz in np.arange(len(z_list)):
        # Propagate
        U_out = prop_show(U_in, d_in, wvl, d_out, z_list[iz], pf)
        #U_out = angularSpectrum(U_in, wvl, d_in, d_out, z_list[iz])

        # Measure out
        I_out = np.abs(crop2(U_out, pad_factor=pf))**2
        cov_I = np.cov(I_out)
        # cov_I = np.corrcoef(I_out)
        stack_cov_I, error_stack_cov_I = diagonal_stack(cov_I)

        c = np.nanmedian(stack_cov_I)
        a = np.nanmax(stack_cov_I)

        param, cova = curve_fit(sinc_exp_decay,
                                np.arange(-fitr, fitr),
                                (stack_cov_I[N - fitr : N + fitr] - c)/a,
                                sigma=error_stack_cov_I[N - fitr : N + fitr])

        if np.abs(param[0]) < 1000.0*np.abs(param[1]):
            # For the case which the sinc still influences.
            # It is hard to get an analytic answer for FWHM of decay sinc
            # So we do numerical answer based on our glorious fit instead.
            FWHM_cov_I[iz] = np.abs(2.*fsolve(
                sinc_exp_decay_half, x0=np.array([0.0]),
                args = (param[0], param[1])))
            case = 'sinc_exp_decay'
        else:
            # The fit is baaddd.
            # Contribution of sinc is too small. We are better fit just exp
            # Do a different fit!!!!
            case = 'exp_decay'
            # Get baseline
            param, cova = curve_fit(exp_decay,
                                    np.arange(-fitr, fitr),
                                    (stack_cov_I[N - fitr : N + fitr])/a,
                                    sigma=error_stack_cov_I[N - fitr
                                                            : N + fitr])
            c = param[1]
            # Redo with a new baseline
            param, cova = curve_fit(exp_decay,
                                    np.arange(-fitr, fitr),
                                    (stack_cov_I[N - fitr : N + fitr]
                                     - a*c) / (a - a*c),
                                    sigma=error_stack_cov_I[N - fitr
                                                            : N + fitr])
            # Since we actually have a good analytic FWHM for exp
            FWHM_cov_I[iz] = 2.0*0.693147*np.abs(param[0]) # d*2*ln2
        
        # test out the fit in pixel!!!!
        plt.errorbar(np.arange(-fitr, fitr), stack_cov_I[N - fitr : N + fitr],
                      yerr=error_stack_cov_I[N - fitr : N + fitr],
                      color='r', label='Measured', marker='.', ls='')
        
        if case == 'sinc_exp_decay':
            plt.plot(np.arange(-fitr, fitr),
                      sinc_exp_decay(np.arange(-fitr, fitr),
                                      param[0], param[1])*a + c,
                      color='b', label='Sinc exp decay fit', ls='-')
        if case == 'exp_decay':
            plt.plot(np.arange(-fitr, fitr),
                      (exp_decay(np.arange(-fitr, fitr), param[0], param[1])
                        *(a-a*c) + a*c),
                      color='b', label='Exp decay fit', ls='-')
        
        plt.hlines(a/2. + c, xmin=-fitr, xmax=fitr,
                    label='Half max', ls='--', color= 'g')
        plt.legend()
        plt.xlim(-fitr, fitr)
        plt.title('Covariance of pupil intensity at'
                  + ' {:.0f}m'.format(z_list[iz]))
        plt.xlabel('pixel')
        plt.ylabel('Covariance')
        plt.show()
    
    # I have no idea though, why does it has to be 1/8th Fresnel zone.
    expected_FWHM = (np.sqrt(wvl*z_list/2.)) # radius
    FWHM_cov_I = FWHM_cov_I*d_out
    # measure FWHM of cov I correctly
    # but spot diameter is approx double this
    # so this is actually radius of spot
    plt.plot(z_list, FWHM_cov_I,
                 label='Measured fit on cov(I)',
                 marker='o', color='r', ls='')
    plt.plot(z_list, expected_FWHM, label='1/2th Fresnel zone radius',
             ls='--', color='g')
    plt.plot(z_list, (np.sqrt(wvl*z_list)), label='1st Fresnel zone radius',
             ls='--', color='k')
    #plt.plot(z_list, wvl*z_list/2./r, label='r diffraction', color='y')
    plt.hlines(2*d_out, xmin=z_list.min(), xmax=z_list.max(),
               ls='--', color='b', label='Pixel Nyquist sampling')
    plt.hlines(r, xmin=z_list.min(), xmax=z_list.max(), ls='--', color='c',
               label='r')
    plt.vlines((1./0.25 * 0.563 * k**(7./6.) * cn2dz)**(-6./5.),
               ymin=FWHM_cov_I.min(),
               ymax=FWHM_cov_I.max(),
               label='Approximation hold (var of log-amp > 0.25)',
               color='m', ls='--')
    plt.ylim(1*d_out, fitr*d_out)
    # plt.ylim(1*d_out, np.nanmax(FWHM_cov_I)*1.2)
    plt.title('FWHM of covariance of intensity, r={:.3f}m, 1px={:.3f}m'.format(r,d_out))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Propagation distance (m)')
    plt.ylabel('FWHM (m)')
    plt.legend()
    plt.show()
    
    return expected_FWHM, FWHM_cov_I

###############################################################################
########################################################################

# 13) Lorentzian curve. #
def lorentz(x, a, mu, sigma, c):
    """
    Generate Lorentzian probability distribution curve.

    Parameters
    ----------
    x : 1darray
        (len(x))
        Element is float.
        Measurements.
    a : float
        Height.
    mu : float
        Centre.
    sigma : float
        Not standard deviation.
        Width related.
    c : float
        Base height.

    Returns
    -------
    y : 1darray
        (len(x))
        Element is float.
        Frequency.

    """
    y = a*(0.5*sigma/np.pi)/((x-mu)**2 + (0.5*sigma)**2) + c
    return y

###############################################################################
########################################################################

# 14 ) Sinc exponentially decaying curve. #
def sinc_exp_decay(x, w, d):
    """
    Generate sinc exponentially decaying function.
    
    Input must be converging to 0 on both ends and centered on the peak.

    Parameters
    ----------
    x : 1darray
        Element type is float.
        Measurement.
    w : float
        Width of the sinc part.
    d : float
        Width of the decaying exponential.

    Returns
    -------
    y : 1darray
        Element type is float.
        Observation frequency.

    """
    x = np.asarray(x, dtype=float)
    w = float(w)
    d = float(d)
    y = np.zeros_like(x, dtype=float)
    for i in np.arange(len(x)):
        if x[i] == 0:
            y[i] = 1.
        else:
            y[i] = np.sin(x[i]/w)/(x[i]/w)*np.exp(-np.abs(x[i])/d)
    return y

###############################################################################
########################################################################

# 15) Diagonally stack a 2darray into 1darray. #
def diagonal_stack(A):
    """
    Diagonally stack 2darray into 1darray.
    
    Average all A[i,j] which i+j=2k together to stack_A[k]

    Parameters
    ----------
    A : 2darray
        (N,N)
        Square dimension.
        Centred around the diagonal axis -> [i,j] where i=j

    Returns
    -------
    stack_A : 1darray
        (2N,2N)
        Diagonally stacked version of a.
        Centred around N -> [N-i:N+i] is symmetric
    error_stack_A : 1darray
        (2N,2N)
        Error of the mean of the stack_A.

    """
    #
    # Read input & declare variables #
    #

    A = np.asarray(A, dtype=type(A[0]))
    N = len(A)
    stack_A = np.zeros(2*N, dtype=type(A[0,0]))
    error_stack_A = np.zeros(2*N, dtype=type(A[0,0]))

    #
    # Diagonally make summations & Average of A along the symmetrical axis #
    #

    for i in np.arange(N):
        # Elements above the symmetrical axis A[: i, i]
        # Got mapped to above centre of stack axis stack_A[N - i : N]
        # centre = A[i,i] =stack_A[N]
        stack_A[N - i : N] = stack_A[N - i : N] + A[: i, i]
        # Elements below the symmetrical axis A[i :, i]
        # Got mapped to the below the centre of stack axis stack_A[N : 2*N - i]
        stack_A[N : 2*N - i] = stack_A[N : 2*N - i] + A[i :, i]
    for i in np.arange(N):
        # Average the sum
        # Number of stacked elements is derived from counting lol.
        # Above the centre
        stack_A[i] = stack_A[i]/i
        # Below the centre
        stack_A[2*N - i - 1] = stack_A[2*N - i - 1] / (i + 1)

    #
    # Get error of the mean #
    #

    for i in np.arange(N):
        # Find error2 of the mean
        # Above the centre
        error_stack_A[N - i : N] = (error_stack_A[N - i : N]
                                        + (A[: i, i]
                                           - stack_A[N - i : N])**2)
        # Below the centre
        error_stack_A[N:2*N - i] = (error_stack_A[N : 2*N - i]
                                        + (A[i :, i]
                                           - stack_A[N : 2*N - i])**2)
    for i in np.arange(N):
        # Average error2 of the mean
        error_stack_A[i] = error_stack_A[i]/(i - 1)
        error_stack_A[2*N - i - 1] = error_stack_A[2*N - i - 1]/(i)
    # Get error of the mean
    error_stack_A = np.sqrt(error_stack_A)

    #
    # Get rid of np.NaN
    # (This is bound to happen for shape(A)=(N,N) where N is even)
    #

    stack_A[np.where(np.isnan(stack_A))] = 0.
    error_stack_A[np.where(np.isnan(error_stack_A))] = 0.

    return stack_A, error_stack_A

###############################################################################
########################################################################

# 16) Exponential decay. #
def exp_decay(x, d, c):
    """
    Exponential decay of linear order.

    Input must be converging to 0 on both ends and centered on the peak.

    Parameters
    ----------
    x : 1darray
        Element type is float.
        Measurement.
    d : float
        Width of the decay.

    Returns
    -------
    y : 1darray
        Element type is float.
        Frequency of events.

    """
    x = np.asarray(x, dtype=float)
    d = float(d)
    c = float(c)
    y = np.zeros(len(x), dtype=float)
    y = np.exp(-(np.abs(x)/d)) + c
    return y

###############################################################################
########################################################################

# 17) Decayed sinc function for calculating Full Width Half Max. #
def sinc_exp_decay_half(x, w, d):
    """
    Use to find FWHM of sinc exp decay function together with fsolve.
    
    Input must be converging to 0 on both ends and centered on the peak.

    Parameters
    ----------
    x : 1darray
        Element type is float.
        See sinc_exp_decay_half().
    w : float
        See sinc_exp_decay_half().
    d : float
        See sinc_exp_decay_half().

    Returns
    -------
    float
        See sinc_exp_decay_half.
        FWHM of sinc_exp_decay_half().

    """
    x = np.asarray(x, dtype=float)
    w = float(w)
    d = float(d)
    y = np.zeros_like(x, dtype=float)
    for i in np.arange(len(x)):
        if x[i] == 0:
            y[i] = 1.
        else:
            y[i] = np.sin(x[i]/w)/(x[i]/w)*np.exp(-np.abs(x[i])/d)
    return y - 0.5

###############################################################################
########################################################################

# 18) My airy disk. #
def intensity_shape(x, a, m, c, w):
    """
    Make shape of intensity psf -ish of circular aperture.

    I(theta) = I_0*(2*J_1(x)/x)**2.
    Please see wikipedia: https://en.wikipedia.org/wiki/Airy_disk .
    where
        I is intensity.
        theta is angle of the cone from centre of source plane to obs.
        I_0 is maximum intensity. Called a in this function
        J_1 is 1st Bessel function of the first kind.
        x = k*R*sin(theta).
        k is wavenumber = 2*np.pi/wvl.
        wvl = wavelength.
        R is radius of the aperture.
        x = (2*pi*R/wvl)*(x_obs/z). (assuming that angle is small enough).
        x_obs is actual coordinate on obs plane.
        z is distance between source and obs plane.
        To get x_obs, x_obs = x / ((2*pi*R)/(wvl*z)).
        Let w = (2*pi*R)/(wvl*z).
        Arrives at
        I(x_obs) = I_0*(2*J_1(x_obs/w)/(x_obs/w))**2.

    In this function we actually call x_obs to be x.

    Adjustments to increase flexibility of this function are
        m to be the centre of the graph
        c to be background of the graph
        a to be maximum (without background)

    Parameters
    ----------
    x : 1darray
        (len(x))
        Element type is float
        Actually x_obs.
        Coordinate of the observation plane.
    a : float
        Maximum intensity.
    m : float
        Centre of the curve.
    c : float
        Background of the curve.
    w : float
        Width of the curve.
        Actually = (2*pi*R)/(wvl*z)

    Returns
    -------
    y : 1darray
        (len(x))
        Element type is float.
        Intensity curve at the observation plane.

    """
    x = np.asarray(x, dtype=float)
    a = float(a)
    m = float(m)
    c = float(c)
    w = float(w)
    y = np.zeros(len(x), dtype=float)
    for i in np.arange(len(x)):
        if x[i] == m:
            y[i] = a + c
        else:
            y[i] = a * (2.*j1((x[i] - m)/w)/((x[i] - m)/w))**2 + c
    return y

###############################################################################
########################################################################

# 22) Parameters check. #
def prop_param_check(N, d_in, d_out, wvl, z_list, pf, scin_blob=False):
    """
    Check propagation parameter whether or not it is good.
    
    Must be printed out as True for all categories.
    **** IMPORTANT ****
    This test is inspired from page127 schmidt
    Under extra condition that D1/d1 = D2/d2 = n < N
    in this function n==>N, N==>NN
    
    schmidt1 = check
    (schmidt2 => adjusted) = check
    schmidt3 => not covered in this function.
        This requires checking input phase screen
    schmidt4 = check
    fresnel zone size = check (if wanted. totally unnecessary)

    Parameters
    ----------
    N : int
        Number of pixel of the desired output.
        pixel.
    d_in : float
        Input pixel size.
        metre.
    d_out : float
        Output pixel size.
        metre.
    wvl : float
        Propagating wavelength.
        Unit is metre.
    z_list : 1darray
        (len(z_list))
        float
        metre.
        Propagation distances.
    pf : int
        Zeropadding factor.
        Padding (N,N) array into (N*pf,N*pf)
    scin_blob : bool
        Do we care about detecting scintillation blob?
        Default is False
        False == skip scintillation blob consideration.

    Returns
    -------
    condition: 1darray
        (len(z_list))
        bool
        Pass the test or not.
        True == no problems.
        False == have problem(s).

    """
    N = int(N)
    d_in = float(d_in)
    d_out = float(d_out)
    wvl = float(wvl)
    z_list = np.asarray(z_list, dtype=float)
    pf = int(pf)
    
    condition1 = np.zeros(len(z_list), dtype=bool)
    condition2 = np.zeros(len(z_list), dtype=bool)
    condition3 = np.zeros(len(z_list), dtype=bool)
    condition4 = np.zeros(len(z_list), dtype=bool)
    condition = np.zeros(len(z_list), dtype=bool)
    
    for iz in np.arange(len(z_list)):
        # condition1 => prevent overflow. ~ schmidt2
        condition1[iz] = ((pf - 1)*N >= wvl*z_list[iz]/d_in/d_out)#checked
        # condition2 => resolve scintillation. ~ 1st fresnel zone size
        if scin_blob == True:
            condition2[iz] = (2.*d_out <= 2.*np.sqrt(wvl*z_list[iz]/8.))
            #??perhaps no 8 in this??#
        else:
            condition2[iz] = True
        # condition3 => aliasing = schmidt4
        condition3[iz] = (pf*N >= wvl*z_list[iz]/d_in/d_out)#checked
        # # condition4 => not enough input for the output
        condition4[iz] = (pf*N >= N + wvl*z_list[iz]/d_in/d_out)
        condition[iz] = (condition1[iz]*condition2[iz]*condition3[iz]
                         *condition4[iz])
        if condition1[iz] == False:
            print(N, d_in, d_out, wvl, z_list[iz], pf)
            print("Screen is not oversized enough. There is too much overflow"
                  + ", output is flooded with noise.")
            print("Wanted oversize = {:.0f}.".format(np.ceil(
                (wvl*z_list[iz]/d_in/d_out/N) + 1)))
        if condition2[iz] == False:
            print(N, d_in, d_out, wvl, z_list[iz], pf)
            print("Screen cannot detect scintillation blob.")
            print("Wanted out pixel = {:.2f}".format(
                np.sqrt(wvl*z_list[iz]/8.)))
        if condition3[iz] == False:
            print(N, d_in, d_out, wvl, z_list[iz], pf)
            print("Screen is too small to contain this big frequency."
                  + " There is aliasing.")
        if condition4[iz] == False:
            print(N, d_in, d_out, wvl, z_list[iz], pf)
            print('There is too less input for the desired output.')
            print("Wanted zeropad by {:.2f}".format(wvl*z_list[iz]/d_in/d_out/N))
        # if condition[iz] == True:
        #     print(N, d_in, d_out, wvl, z_list[iz], pf)
        #     print('Pass')
    # print(z_list, condition)
    return condition

###############################################################################
########################################################################

# 12) Measure scintillation size. #
def phase_dif_size(N, r, d_in, L0, l0, wvl, d_out, z_list, pf):
    """
    Copy from Measure scintillation blob size.
    
    This method is still in a test mode.
    
    Parameters
    ----------
    N : int
        Desired output screen diameter size.
        Unit is pixel.
    r : float
        Screen strength.
        Unit is m.
    d_in : float
        Input pixel scale size.
        Unit is m.
    L0 : float
        Outer scale size.
        Unit is m.
        Suggestion: 20m.
    l0 : float
        Inner scale size.
        Unit is m.
        Suggestion: 1.5e-10m.
    wvl : float
        Propagation wavelength.
        Unit is m.
        Suggestion: 500e-9m.
    d_out : float
        Output pixel scale size.
        Unit is m.
        Suggestion: = d_in
    z_list : 1darray
        Element type is float.
        Propagation distances.
        Unit is m.
    pf : int
        Padding factor.
        Actually the oversize factor, but I am too lazy to change.

    Returns
    -------
    expected_FWHM : 1darray
        Based on Fresnel zone size.
        Currently is 1/8th. No good reason besides that it actually fits.
    FWHM_cov_I : 1darray
        DESCRIPTION.

    Example
    -------
    z_list = np.logspace(np.log(1e3)/np.log(10),
                         np.log(25e3)/np.log(10), num=10)
    d_in = float(5e-3)
    d_out = d_in
    N = int(1024)
    r = float(0.05)
    L0 = float(20)
    l0 = float(1.5e-10)
    wvl = float(500e-9)
    pf = int(2)
    k = float(2*np.pi/wvl)
    cn2dz = float(r**(-5./3.)/0.423/k**2)

    expected_FWHM, FWHM_cov_I = phase_dif_size(N, r, d_in, L0, l0, wvl,
                                                   d_out, z_list, pf)

    """
    #
    # Processing input #
    #
    N = int(N)                      # int; pixel; Desired output pixel.
    r = float(r)                    # float; m; Turbulence strength.
    d_in = float(d_in)              # float; m; Input pixel size.
    L0 = float(L0)                  # float; m; Outer scale.
    l0 = float(l0)                  # float; m; Inner scale.
    wvl = float(wvl)                # float; m; Wavelength.
    d_out = float(d_out)            # float; m; Output pixel size.
    z_list = np.asarray(z_list, dtype=float)    # ndarray; float; m;
                                                # Propagated distances.
    pf = int(pf)                    # Int; Pad/oversize factor.

    #
    # Indenpendent constants #
    #
    fitr = int(64)

    #
    # Dependent constants #
    #
    NN = int(pf*N)       # Int; pixel; Padded/oversized screen size.
    k = float(2.*np.pi/wvl)                    # float; rad/m; Wave number.
    cn2dz = float(r**(-5./3.)/0.423/k**2)        # float; Turbulence strength.

    check = True
    check_condition = prop_param_check(N, d_in, d_out, wvl, z_list, pf)
    for i in check_condition:
        check = check * i
    if check == False:
        print("One of the condition is failing.")
        return

    #
    # Declaration field #
    #
    amp_in = np.ones((NN,NN), dtype=float)          # Input amplitude.
    phase_in = np.ones((NN, NN), dtype=float)       # Input phase (rad).
    U_in = np.zeros((NN,NN), dtype=complex)         # Input complex amp.

    U_out = np.zeros((NN,NN), dtype=complex)          # Output complex amp.
    I_out = np.zeros((N,N), dtype=float)            # Output intensity.

    cov_I = np.zeros((N,N), dtype=float)
    stack_cov_I = np.zeros(2*N, dtype=float)
    error_stack_cov_I = np.zeros(2*N, dtype=float)

    expected_FWHM = np.zeros(len(z_list), dtype=float)
    FWHM_cov_I = np.zeros(len(z_list), dtype=float)
    # FWHM_cov_I_error = np.zeros_like(z_list, dtype=float)

    c = float(0)
    a = float(0)

    #
    # Initialization field #
    #
    phase_in = ft_sh_phase_screen(r, NN, d_in, L0, l0)
    U_in = amp_in*np.exp(1j*phase_in)
    phase_in_crop = np.angle(crop2(U_in,pf))
    x = np.arange(-N/2.,N/2.)*d_in
    xx, yy = np.meshgrid(x,x)
    plt.pcolor(xx, yy, phase_in_crop, shading='auto', cmap=reversed_color_map)
    plt.title('Wrapped phase in (rad)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar()
    plt.axis('equal')
    plt.show()

    #
    # Loop over each propagation distance #
    #
    for iz in np.arange(len(z_list)):
        # Propagate
        # U_out = prop_show(U_in, d_in, wvl, d_out, z_list[iz], pf)
        U_out = angularSpectrum(U_in, wvl, d_in, d_out, z_list[iz])

        # Measure out
        #I_out = np.abs(crop2(U_out, pad_factor=pf))**2
        phase_out_crop = np.angle(crop2(U_out,pf))
        # I_out = np.unwrap(np.unwrap(phase_out_crop-phase_in_crop,period=np.pi,axis=-2),period=np.pi,axis=-1)
        I_out = phase_out_crop-phase_in_crop
        for i in np.arange(len(I_out)):
            for j in np.arange(len(I_out)):
                if np.abs(I_out[i,j]) > np.pi:
                    if I_out[i,j] > 0:
                        I_out[i,j] = I_out[i,j] - 2.*np.pi
                    else:
                        I_out[i,j] = I_out[i,j] + 2.*np.pi
        plt.imshow(I_out)
        plt.title('phase difference')
        plt.colorbar()
        plt.show()
        cov_I = np.cov(I_out)
        # cov_I = np.corrcoef(I_out)
        stack_cov_I, error_stack_cov_I = diagonal_stack(cov_I)

        c = np.nanmedian(stack_cov_I)
        a = np.nanmax(stack_cov_I)

        param, cova = curve_fit(sinc_exp_decay,
                                np.arange(-fitr, fitr),
                                (stack_cov_I[N - fitr : N + fitr] - c)/a,
                                sigma=error_stack_cov_I[N - fitr : N + fitr])

        if np.abs(param[0]) < 1000.0*np.abs(param[1]):
            # For the case which the sinc still influences.
            # It is hard to get an analytic answer for FWHM of decay sinc
            # So we do numerical answer based on our glorious fit instead.
            FWHM_cov_I[iz] = np.abs(2.*fsolve(
                sinc_exp_decay_half, x0=np.array([0.0]),
                args = (param[0], param[1])))
            case = 'sinc_exp_decay'
        else:
            # The fit is baaddd.
            # Contribution of sinc is too small. We are better fit just exp
            # Do a different fit!!!!
            case = 'exp_decay'
            # Get baseline
            param, cova = curve_fit(exp_decay,
                                    np.arange(-fitr, fitr),
                                    (stack_cov_I[N - fitr : N + fitr])/a,
                                    sigma=error_stack_cov_I[N - fitr
                                                            : N + fitr])
            c = param[1]
            # Redo with a new baseline
            param, cova = curve_fit(exp_decay,
                                    np.arange(-fitr, fitr),
                                    (stack_cov_I[N - fitr : N + fitr]
                                     - a*c) / (a - a*c),
                                    sigma=error_stack_cov_I[N - fitr
                                                            : N + fitr])
            # Since we actually have a good analytic FWHM for exp
            FWHM_cov_I[iz] = 2.0*0.693147*np.abs(param[0]) # d*2*ln2
        
        # test out the fit in pixel!!!!
        plt.errorbar(np.arange(-fitr, fitr), stack_cov_I[N - fitr : N + fitr],
                      yerr=error_stack_cov_I[N - fitr : N + fitr],
                      color='r', label='Measured', marker='.', ls='')
        
        if case == 'sinc_exp_decay':
            plt.plot(np.arange(-fitr, fitr),
                      sinc_exp_decay(np.arange(-fitr, fitr),
                                      param[0], param[1])*a + c,
                      color='b', label='Sinc exp decay fit', ls='-')
        if case == 'exp_decay':
            plt.plot(np.arange(-fitr, fitr),
                      (exp_decay(np.arange(-fitr, fitr), param[0], param[1])
                        *(a-a*c) + a*c),
                      color='b', label='Exp decay fit', ls='-')
        
        plt.hlines(a/2. + c, xmin=-fitr, xmax=fitr,
                    label='Half max', ls='--', color= 'g')
        plt.legend()
        plt.xlim(-fitr, fitr)
        plt.title('Covariance of phase difference between at'
                  + ' {:.0f}m and ground'.format(z_list[iz]))
        plt.xlabel('pixel')
        plt.ylabel('Covariance')
        plt.show()
    
    # I have no idea though, why does it has to be 1/8th Fresnel zone.
    expected_FWHM = (np.sqrt(wvl*z_list/4.))
    FWHM_cov_I = FWHM_cov_I*d_out
    plt.plot(z_list, FWHM_cov_I,
                 label='Measured fit on cov(phase out - phase in)',
                 marker='o', color='r', ls='')
    plt.plot(z_list, expected_FWHM, label='1/4th Fresnel zone size',
             ls='--', color='g')
    #plt.plot(z_list, wvl*z_list/2./r, label='r diffraction', color='y')
    plt.hlines(2*d_out, xmin=z_list.min(), xmax=z_list.max(),
               ls='--', color='b', label='Nyquist sampling')
    plt.hlines(r, xmin=z_list.min(), xmax=z_list.max(), ls='--', color='c',
               label='r')
    plt.vlines((1./0.25 * 0.563 * k**(7./6.) * cn2dz)**(-6./5.),
               ymin=FWHM_cov_I.min(),
               ymax=FWHM_cov_I.max(),
               label='Approximation hold (var of log-amp > 0.25)',
               color='m', ls='--')
    plt.ylim(1*d_out, fitr*d_out)
    # plt.ylim(1*d_out, np.nanmax(FWHM_cov_I)*1.2)
    plt.title('FWHM of covariance of phase difference, r={:.3f}m'.format(r))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Propagation distance (m)')
    plt.ylabel('FWHM (m)')
    plt.legend()
    plt.show()
    
    return expected_FWHM, FWHM_cov_I

###############################################################################
########################################################################

# 12) Measure scintillation size. #
def slopex_dif_size(N, r, d_in, L0, l0, wvl, d_out, z_list, pf):
    """
    Copy from Measure scintillation blob size.
    
    This method is still in a test mode.
    
    Parameters
    ----------
    N : int
        Desired output screen diameter size.
        Unit is pixel.
    r : float
        Screen strength.
        Unit is m.
    d_in : float
        Input pixel scale size.
        Unit is m.
    L0 : float
        Outer scale size.
        Unit is m.
        Suggestion: 20m.
    l0 : float
        Inner scale size.
        Unit is m.
        Suggestion: 1.5e-10m.
    wvl : float
        Propagation wavelength.
        Unit is m.
        Suggestion: 500e-9m.
    d_out : float
        Output pixel scale size.
        Unit is m.
        Suggestion: = d_in
    z_list : 1darray
        Element type is float.
        Propagation distances.
        Unit is m.
    pf : int
        Padding factor.
        Actually the oversize factor, but I am too lazy to change.

    Returns
    -------
    expected_FWHM : 1darray
        Based on Fresnel zone size.
        Currently is 1/8th. No good reason besides that it actually fits.
    FWHM_cov_I : 1darray
        DESCRIPTION.

    Example
    -------
    z_list = np.logspace(np.log(1e3)/np.log(10),
                         np.log(25e3)/np.log(10), num=10)
    d_in = float(5e-3)
    d_out = d_in
    N = int(1024)
    r = float(0.05)
    L0 = float(20)
    l0 = float(1.5e-10)
    wvl = float(500e-9)
    pf = int(2)
    k = float(2*np.pi/wvl)
    cn2dz = float(r**(-5./3.)/0.423/k**2)

    expected_FWHM, FWHM_cov_I = scintillation_size(N, r, d_in, L0, l0, wvl,
                                                   d_out, z_list, pf)

    """
    #
    # Processing input #
    #
    N = int(N)                      # int; pixel; Desired output pixel.
    r = float(r)                    # float; m; Turbulence strength.
    d_in = float(d_in)              # float; m; Input pixel size.
    L0 = float(L0)                  # float; m; Outer scale.
    l0 = float(l0)                  # float; m; Inner scale.
    wvl = float(wvl)                # float; m; Wavelength.
    d_out = float(d_out)            # float; m; Output pixel size.
    z_list = np.asarray(z_list, dtype=float)    # ndarray; float; m;
                                                # Propagated distances.
    pf = int(pf)                    # Int; Pad/oversize factor.

    #
    # Indenpendent constants #
    #
    fitr = int(64)

    #
    # Dependent constants #
    #
    NN = int(pf*N)       # Int; pixel; Padded/oversized screen size.
    k = float(2.*np.pi/wvl)                    # float; rad/m; Wave number.
    cn2dz = float(r**(-5./3.)/0.423/k**2)        # float; Turbulence strength.

    check = True
    check_condition = prop_param_check(N, d_in, d_out, wvl, z_list, pf)
    for i in check_condition:
        check = check * i
    if check == False:
        print("One of the condition is failing.")
        return

    #
    # Declaration field #
    #
    amp_in = np.ones((NN,NN), dtype=float)          # Input amplitude.
    phase_in = np.ones((NN, NN), dtype=float)       # Input phase (rad).
    U_in = np.zeros((NN,NN), dtype=complex)         # Input complex amp.

    U_out = np.zeros((NN,NN), dtype=complex)          # Output complex amp.
    I_out = np.zeros((N,N), dtype=float)            # Output intensity.

    cov_I = np.zeros((N,N), dtype=float)
    stack_cov_I = np.zeros(2*N, dtype=float)
    error_stack_cov_I = np.zeros(2*N, dtype=float)

    expected_FWHM = np.zeros(len(z_list), dtype=float)
    FWHM_cov_I = np.zeros(len(z_list), dtype=float)
    # FWHM_cov_I_error = np.zeros_like(z_list, dtype=float)

    c = float(0)
    a = float(0)

    #
    # Initialization field #
    #
    phase_in = ft_sh_phase_screen(r, NN, d_in, L0, l0)
    U_in = amp_in*np.exp(1j*phase_in)
    phase_in_crop = np.angle(crop2(U_in,pf))
    x = np.arange(-N/2.,N/2.)*d_in
    xx, yy = np.meshgrid(x,x)
    plt.pcolor(xx, yy, phase_in_crop, shading='auto', cmap=reversed_color_map)
    plt.title('Wrapped phase in (rad)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar()
    plt.axis('equal')
    plt.show()
    slopex_phase_in=mwfs.get_slope(phase_in_crop, d_in, wvl)[:,:,0]*d_in*2.*np.pi/wvl

    #
    # Loop over each propagation distance #
    #
    for iz in np.arange(len(z_list)):
        # Propagate
        # U_out = prop_show(U_in, d_in, wvl, d_out, z_list[iz], pf)
        U_out = angularSpectrum(U_in, wvl, d_in, d_out, z_list[iz])

        # Measure out
        #I_out = np.abs(crop2(U_out, pad_factor=pf))**2
        phase_out_crop = np.angle(crop2(U_out,pf))
        slopex_phase_out = mwfs.get_slope(phase_out_crop, d_out, wvl)[:,:,0]*d_out*2.*np.pi/wvl
        # I_out = np.unwrap(np.unwrap(slopex_phase_out-slopex_phase_in,axis=-2,period=np.pi),period=np.pi,axis=-1)
        I_out = slopex_phase_out - slopex_phase_in
        for i in np.arange(len(I_out)):
            for j in np.arange(len(I_out)):
                if np.abs(I_out[i,j]) > np.pi:
                    if I_out[i,j] > 0:
                        I_out[i,j] = I_out[i,j] - 2.*np.pi
                    else:
                        I_out[i,j] = I_out[i,j] + 2.*np.pi
        plt.imshow(I_out)
        plt.title('phase slope difference (rad/px)')
        plt.colorbar()
        plt.show()
        cov_I = np.cov(I_out)
        # cov_I = np.corrcoef(I_out)
        stack_cov_I, error_stack_cov_I = diagonal_stack(cov_I)

        c = np.nanmedian(stack_cov_I)
        a = np.nanmax(stack_cov_I)

        param, cova = curve_fit(sinc_exp_decay,
                                np.arange(-fitr, fitr),
                                (stack_cov_I[N - fitr : N + fitr] - c)/a,
                                sigma=error_stack_cov_I[N - fitr : N + fitr])

        if np.abs(param[0]) < 1000.0*np.abs(param[1]):
            # For the case which the sinc still influences.
            # It is hard to get an analytic answer for FWHM of decay sinc
            # So we do numerical answer based on our glorious fit instead.
            FWHM_cov_I[iz] = np.abs(2.*fsolve(
                sinc_exp_decay_half, x0=np.array([0.0]),
                args = (param[0], param[1])))
            case = 'sinc_exp_decay'
        else:
            # The fit is baaddd.
            # Contribution of sinc is too small. We are better fit just exp
            # Do a different fit!!!!
            case = 'exp_decay'
            # Get baseline
            param, cova = curve_fit(exp_decay,
                                    np.arange(-fitr, fitr),
                                    (stack_cov_I[N - fitr : N + fitr])/a,
                                    sigma=error_stack_cov_I[N - fitr
                                                            : N + fitr])
            c = param[1]
            # Redo with a new baseline
            param, cova = curve_fit(exp_decay,
                                    np.arange(-fitr, fitr),
                                    (stack_cov_I[N - fitr : N + fitr]
                                     - a*c) / (a - a*c),
                                    sigma=error_stack_cov_I[N - fitr
                                                            : N + fitr])
            # Since we actually have a good analytic FWHM for exp
            FWHM_cov_I[iz] = 2.0*0.693147*np.abs(param[0]) # d*2*ln2
        
        # test out the fit in pixel!!!!
        plt.errorbar(np.arange(-fitr, fitr), stack_cov_I[N - fitr : N + fitr],
                      yerr=error_stack_cov_I[N - fitr : N + fitr],
                      color='r', label='Measured', marker='.', ls='')
        
        if case == 'sinc_exp_decay':
            plt.plot(np.arange(-fitr, fitr),
                      sinc_exp_decay(np.arange(-fitr, fitr),
                                      param[0], param[1])*a + c,
                      color='b', label='Sinc exp decay fit', ls='-')
        if case == 'exp_decay':
            plt.plot(np.arange(-fitr, fitr),
                      (exp_decay(np.arange(-fitr, fitr), param[0], param[1])
                        *(a-a*c) + a*c),
                      color='b', label='Exp decay fit', ls='-')
        
        plt.hlines(a/2. + c, xmin=-fitr, xmax=fitr,
                    label='Half max', ls='--', color= 'g')
        plt.legend()
        plt.xlim(-fitr, fitr)
        plt.title('Covariance of phase slope difference between at'
                  + ' {:.0f}m and ground'.format(z_list[iz]))
        plt.xlabel('pixel')
        plt.ylabel('Covariance')
        plt.show()
    
    # I have no idea though, why does it has to be ... Fresnel zone.
    expected_FWHM = np.sqrt(np.sqrt(wvl*z_list))/16.
    FWHM_cov_I = FWHM_cov_I*d_out
    plt.plot(z_list, FWHM_cov_I,
                 label='Measured fit on cov(phase out - phase in)',
                 marker='o', color='r', ls='')
    plt.plot(z_list, expected_FWHM, label='??sqrt(Fresnel zone size)',
             ls='--', color='g')
    #plt.plot(z_list, wvl*z_list/2./r, label='r diffraction', color='y')
    plt.hlines(2*d_out, xmin=z_list.min(), xmax=z_list.max(),
               ls='--', color='b', label='Nyquist sampling')
    plt.hlines(r, xmin=z_list.min(), xmax=z_list.max(), ls='--', color='c',
               label='r')
    plt.vlines((1./0.25 * 0.563 * k**(7./6.) * cn2dz)**(-6./5.),
               ymin=FWHM_cov_I.min(),
               ymax=FWHM_cov_I.max(),
               label='Approximation hold (var of log-amp > 0.25)',
               color='m', ls='--')
    plt.ylim(1*d_out, fitr*d_out)
    # plt.ylim(1*d_out, np.nanmax(FWHM_cov_I)*1.2)
    plt.title('FWHM of covariance of phase slope difference x-axis, r={:.3f}m'.format(r))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Propagation distance (m)')
    plt.ylabel('FWHM (m)')
    plt.legend()
    plt.show()
    
    return expected_FWHM, FWHM_cov_I

def prop_test2(N, d1, wvl, d2, z_list, pf):
    """
    Test angular-spectrum Fresnel propagation at different distances.

    Details
    -------
    1) Plane wave.
    2) Point source on edge.
    3) Does oversize with a 'pf (pad_factor)'.
        (Adjustable within this function only).
    4) Plot out propagation results at each propagation distances.

    Parameters
    ----------
    N : int
        Output screen size in number of pixel.
        Diameter.
        Unit is pixel.
    d1 : float
        Pixel scale at the source plane.
        Unit is m.
    wvl : float
        Wavelength of propagation.
        Unit is m
    d2 : float
        Pixel scale at the observation plane.
        Unit is m.
    z_list : ndarray
        Element type is float.
        Propagation distances.
        Unit is m.
    pf : int
        Padding factor.
        Actually the oversize factor, but I am too lazy to change.

    Returns
    -------
    None

    Example
    -------
    prop_test(1024, 5e-3, 500e-9, 5e-3, np.logspace(0, 2, num=10) * 1e3, 2)
    prop_test2(16,3e-6,500e-9,3e-6,[5.76e-4/8.,5.76e-4/4.,5.76e-4/2.,5.76e-4,2.*5.76e-4],2)

    """
    # Process input.
    N = int(N)                      # int; pixel; Desired output screen size.
    d1 = float(d1)                  # float; m; Input pixel size.
    wvl = float(wvl)                # float; m; Propagated wavelength.
    d2 = float(d2)                  # float; m; Output pixel size.
    z_list = np.asarray(z_list, dtype=float)    # ndarray; (num_z); m;
                                                # Propagated distance.
    pf = int(pf)                    # int; Oversize/padding factor.

    # Global constant.
    NN = pf*N                       # int; pixel; Propagation screen size.
    
    check = True
    check_condition = prop_lens_param_check(N, d1, d2, wvl, z_list, pf)
    for i in check_condition:
        check = check * i
    
    if check == False:
        print("One of the condition is failing.")

    # Variables declaration.
    
    amp_in = zero_pad2(10*(N*pf)**2*np.ones((N, N), dtype=float), pf)
                    # ndarray; (NN,NN); (float,float); Input amplitude.
    phase_in = np.ones((NN, NN), dtype=float)
                    # ndarray; (NN,NN); (float,float); (rad,rad); Input phase.
    U_in = np.zeros((NN, NN), dtype=complex)
                    # ndarray; (NN,NN); (complex,complex); Input complex amp.
    U_out = np.zeros((NN,NN), dtype=complex)
                    # ndarray; (NN,NN); (complex,complex); Output complex amp.
    phase_out = np.zeros((NN,NN), dtype=float)
                    # ndarray; (NN,NN); (float,float); (rad,rad); Output phase.
    I_out = np.zeros((NN,NN), dtype=float)
                    # ndarray; (NN,NN); (float,float); Output intensity.

    # Set wave at the origin plane.
    #amp_in[0,0] = 10 * (N*pf)**2
    U_in = amp_in*np.exp(1j*phase_in)

    # Propagate input wave to different distances.
    for zi in z_list:
        U_out = angularSpectrum(U_in, wvl, d1, d2, zi)

        phase_out = np.angle(U_out)
        I_out = np.abs(U_out)**2

        plt.imshow(I_out, cmap=reversed_color_map)
        plt.vlines(int(N*(pf - 1)/2),
                   ymin=int(N*(pf - 1)/2), ymax=int(N*(pf + 1)/2),
                   color='k')
        plt.vlines(int(N*(pf + 1)/2),
                   ymin=int(N*(pf - 1)/2), ymax=int(N*(pf + 1)/2),
                   color='k')
        plt.hlines(int(N*(pf - 1)/2),
                   xmin=int(N*(pf - 1)/2), xmax=int(N*(pf + 1)/2),
                   color='k')
        plt.hlines(int(N*(pf + 1)/2),
                   xmin=int(N*(pf - 1)/2), xmax=int(N*(pf + 1)/2),
                   color='k')
        plt.title('I out at z={:.3e}m'.format(zi))
        plt.xlabel('x (px)')
        plt.ylabel('y (px)')
        plt.colorbar()
        plt.show()

        plt.imshow(phase_out, cmap=reversed_color_map)
        plt.vlines(int(N*(pf - 1)/2),
                   ymin=int(N*(pf - 1)/2), ymax=int(N*(pf + 1)/2),
                   color='k')
        plt.vlines(int(N*(pf + 1)/2),
                   ymin=int(N*(pf - 1)/2), ymax=int(N*(pf + 1)/2),
                   color='k')
        plt.hlines(int(N*(pf - 1)/2),
                   xmin=int(N*(pf - 1)/2), xmax=int(N*(pf + 1)/2),
                   color='k')
        plt.hlines(int(N*(pf + 1)/2),
                   xmin=int(N*(pf - 1)/2), xmax=int(N*(pf + 1)/2),
                   color='k')
        plt.title('Phase out at z={:.3e}m'.format(zi))
        plt.xlabel('x (px)')
        plt.ylabel('y (px)')
        plt.colorbar()
        plt.show()

    return None

###############################################################################
########################################################################

# 22) Parameters check. #
def prop_lens_param_check(N, d_in, d_out, wvl, z_list, pf, scin_blob=False):
    """
    Check propagation parameter whether or not it is good.
    
    Must be printed out as True for all categories.
    **** IMPORTANT ****
    This test is inspired from page127 schmidt
    Under extra condition that D1/d1 = D2/d2 = n < N
    in this function n==>N, N==>NN
    
    schmidt1 = check
    (schmidt2 => adjusted) = check
    schmidt3 => not covered in this function.
        This requires checking input phase screen
    schmidt4 = check
    fresnel zone size = check (if wanted. totally unnecessary)

    Parameters
    ----------
    N : int
        Number of pixel of the desired output.
        pixel.
    d_in : float
        Input pixel size.
        metre.
    d_out : float
        Output pixel size.
        metre.
    wvl : float
        Propagating wavelength.
        Unit is metre.
    z_list : 1darray
        (len(z_list))
        float
        metre.
        Propagation distances.
    pf : int
        Zeropadding factor.
        Padding (N,N) array into (N*pf,N*pf)
    scin_blob : bool
        Do we care about detecting scintillation blob?
        Default is False
        False == skip scintillation blob consideration.

    Returns
    -------
    condition: 1darray
        (len(z_list))
        bool
        Pass the test or not.
        True == no problems.
        False == have problem(s).

    """
    N = int(N)
    d_in = float(d_in)
    d_out = float(d_out)
    wvl = float(wvl)
    z_list = np.asarray(z_list, dtype=float)
    pf = int(pf)
    
    condition1 = np.zeros(len(z_list), dtype=bool)
    condition3 = np.zeros(len(z_list), dtype=bool)
    condition = np.zeros(len(z_list), dtype=bool)
    
    for iz in np.arange(len(z_list)):
        # condition1 => prevent overflow. ~ schmidt2
        condition1[iz] = ((pf - 1)*N >= wvl*z_list[iz]/2./d_in/d_out)#checked
        condition3[iz] = (pf*N >= wvl*z_list[iz]/d_in/d_out)#checked
        condition[iz] = (condition1[iz]*condition3[iz])
        if condition1[iz] == False:
            print(N, d_in, d_out, wvl, z_list[iz], pf)
            print("Screen is not oversized enough. There is too much overflow"
                  + ", output is flooded with noise.")
            print("Wanted oversize = {:.0f}.".format(np.ceil(
                (wvl*z_list[iz]/2./d_in/d_out/N + 1))))
        if condition3[iz] == False:
            print(N, d_in, d_out, wvl, z_list[iz], pf)
            print("Screen is too small to contain this big frequency."
                  + " There is aliasing.")
            print("Wanted zeropad by {:.2f}".format(wvl*z_list[iz]/d_in/d_out/N))
        # if condition[iz] == True:
        #     print(N, d_in, d_out, wvl, z_list[iz], pf)
        #     print('Pass')
    # print(z_list, condition)
    return condition

###############################################################################
########################################################################

###############################################################################
# List of other functions which I would like to write but didn't, ######
# because it will be redundant with aotools or soapy. ##################
# or with my other sections
###############################################################################
# 3) 1-step angular-spectrum Fresnel propagation
#       from aotools.opticalpropagation.angularSpectrum
# 4) WFS propagations
#       better placed within my_WFS.py
###############################################################################
########################################################################

###############################################################################
# End of File ##########################################################
###############################################################################
