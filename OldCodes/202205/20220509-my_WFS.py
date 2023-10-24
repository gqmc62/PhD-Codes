
###############################################################################
# Start of file ########################################################
###############################################################################
"""
Library of AO codes on Adaptive Optics: screen measurement study.

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
# 19) Measure slope by pixel. #
#       get_slope_pixel()
#       Measure slope/gradient (vector) of a given scalar field.
# 20) Measure slope by lenslet. #
#       get_slope_lenslet()
#       Measure average slope vector of a scalar field chopped into lenslet.
# 21) Measure intensity by lenslet. #
#       get_val_lenslet()
#       Measure average intensity field chopped into lenslet.
# ...
# ...
# ...
# 23) Normalized 1D correlation. #
#       norm_correlate1()
#       Generate normalized correlation between 2 1darrays.
# 24) Add phase caused by lens. #
#       pass_lens()
#       Pass wavefront through lens.
# 25) Add half pixel shift for SH-WFS. #
#       tilt_to_centre_image()
#       Centre SHWFS spots to be in the very corner of the pixels.
# 26) Propagating wave through SHWFS lenslet to SHWFS detector. #
#       pupil_to_focus()
#       Properly pass wavefront through SHWFS lenslet.
# 27) Check screen. #
#       check_screen()
#       Measure slope/gradient (vector) of a given scalar field.
# 28) Propagate pupil image at SHWFS lenslet array to SHWFS detector. #
#       propagate_lenslet()
#       Propagate SHWFS pupil plane/lenslet array through all lens to focus.
# 29) Measure intensity in each lenslet. #
#       get_total_I_SHWFS()
#       Measure total intensity in each SHWFS lenslet.
# 30) Measure max intensity in each lenslet. #
#       get_max_I_SHWFS()
#       Measure max intensity in each SHWFS lenslet.
# 31) Measure slopes at each lenslet. #
#       get_spot_centre_SHWFS()
#       Measure centre of each SHWFS lenslet spot, (0,0) = centre.
# 32) Get SHWFS slope. #
#       get_SHWFS_slope()
#       Get SHWFS slope in radians. This is not wavefront error.
# 33) Get SHWFS wavefront error. #
#       get_SHWFS_WFE()
#       Get SHWFS wavefront error of wavefront at WFS lenslet.
# 34) Get actual wavefront error. #
#       get_lenslet_WFE()
#       Get actual wavefront error of each lenslet.
# 35) Get cheated version of SHWFS. #
#       get_intensity_weighted_WFE()
#       Get a cheated version of SHWFS, without propagation.
# ***** IMPORTANT *****
# These 'WFE' are not really the WFE.
# ***** IMPORTANT *****
###############################################################################
########################################################################

#
# Imports #
#
import numpy as np
from aotools.opticalpropagation import angularSpectrum
from matplotlib import pyplot as plt
import my_propagation as mprop

#
# Preferences #
#
color_map = plt.cm.get_cmap('viridis')
reversed_color_map = color_map.reversed()

###############################################################################
########################################################################

# 19) Measure slope by pixel. #
def get_slope(phase, pixel_scale, wvl):
    """
    Measure slope/gradient (vector) of a given scalar field.

    Given phase does not have to be continuous.
    As long as the slope is not too high, ie >pi within 1 pixel,
    then this function will give out correct slope.
    If the given phase array is a folded phase (-pi,pi),
    this function will automatically folded phase different back
    to be less than pi.

    Parameters
    ----------
    phase : 2darray
        (N, N)
        Element type is float.
        Element unit is radian.
        Contain phase at any given point.
    pixel_scale : float
        Distance between each given element in the array.
        Unit is metre.

    Returns
    -------
    slope : 3darray
        Actually 2darray but contain a pair of vector coordinates.
        (N, N, 2)
        float
        radian.
        N is kept, even though it should be (N - 1)
        slope[N, : , : ] & slope[ : , N, : ] ORIGINALLY ARE np.NaN.
        But they are REPLACED with their closest APPROXIMATION
        which is/are their neighbor(s).
        *** IMPORTANT ***
        This slope is actually direction of the wavefront.
        NOT the phase different, etc.

    Example
    -------
    x = np.arange(-16, 16)
    x = x*(x > 0) * 2.*np.pi/16.*4.
    xx, yy = np.meshgrid(x, x)
    phase = - xx + yy
    phase = np.angle(np.exp(1j*phase))
    slope_phase = get_slope(phase, 5e-3, 500e-9)
    plt.pcolor(np.arange(-16, 16)*5e-3,
               np.arange(-16, 16)*5e-3,
               phase,
               shading='auto')
    plt.axis('equal')
    plt.colorbar()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Phase = -x + y, x>0, y>0')
    plt.show()

    plt.quiver(np.arange(-16, 16)*5e-3,
               np.arange(-16, 16)*5e-3,
               slope_phase[ : , : , 0],
               slope_phase[ : , : , 1])
    plt.title('Slope direction (phase = -x + y, x>0, y>0)')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()

    This Example should comfort the followings
        1) Axis of x, y
        2) Direction of x, y
    Between phase / slope should match
    To allow measurement to be accurate, input should be nyquist sampled,
    which means peak to valley = 2 pixels
    or peak to peak = edge to edge = 4 pixels.

    """
    # Inputs
    phase = np.asarray(phase, dtype=float)
    pixel_scale = float(pixel_scale)
    wvl = float(wvl)
    # Constants
    N = len(phase)
    # Declarations
    slope = np.zeros((N, N, 2), dtype=float)
    # Calculations
    for i in np.arange(N - 1):
        for j in np.arange(N - 1):
            # Difference
            # Order of 0,1 or i,j depends on plottings???
            # This order is perfect for pcolor and quiver.
            slope[i, j, 1] = (phase[i + 1, j] - phase[i, j]) # should be 0
            slope[i, j, 0] = (phase[i, j + 1] - phase[i, j]) # should be 1
            # Solve too large difference at the phase folding.
            for k in np.arange(2):
                # If difference size is larger than half a circle,
                # it means this is the folding.
                if np.abs(slope[i, j, k]) > np.pi:
                    # If it is too positive, subtract a full circle.
                    if slope[i, j, k] > 0:
                        slope[i, j, k] = slope[i, j, k] - 2*np.pi
                    # If it is too negative, add a full circle.
                    else:
                        slope[i, j, k] = slope[i, j, k] + 2*np.pi
                    # It is very most likely that the following statements
                    # will not ever be called, because of the following reasons
                    # Input phase here is already wrapped, so unwrapped dif.
                    # is meaning less, so we have to wrapped it. By doing so,
                    # the dif. is always less than pi.
                    if np.abs(slope[i, j, k]) > np.pi:
                        print('This screen is too dense.')
                        print(np.abs(slope[i, j, k]))
                        return False
    # Get slope from difference
    slope = slope/pixel_scale*wvl/2./np.pi
    # Replaced edges and corner data which originally reported as 0
    # with its closest approximation instead.
    # ie. with its neighbor(s).
    # The edges
    for i in np.arange(N - 1):
        for k in np.arange(2):
            slope[N - 1, i, k] = slope[N - 2, i, k]
            slope[i, N - 1, k] = slope[i, N - 2, k]
    # The corner
    slope[N - 1, N - 1, 0] = (slope[N - 2, N - 1, 0]
                              + slope[N - 1, N - 2, 0])*0.5
    slope[N - 1, N - 1, 1] = (slope[N - 2, N - 1, 1]
                              + slope[N - 1, N - 2, 1])*0.5

    return slope

###############################################################################
########################################################################

# 20) Measure slope by lenslet. #
def get_vec_lenslet(vec, lenslet_pixel):
    """
    Measure average slope vector of a scalar field chopped into lenslet.

    Parameters
    ----------
    vec : 3darray
        (N, N, 2)
        float
        Vector field to be averaged into lenslet.
    lenslet_pixel : int
        Number of pixel in each lenslet.

    Returns
    -------
    lenslet_vec : 3darray
        (N/lenslet_pixel, N/lenslet_pixel, 2)
        float
        Averaged vector field into lenslet.

    """
    vec = np.asarray(vec, dtype=float)
    lenslet_pixel = int(lenslet_pixel)
    N = len(vec)
    num_lenslet = int(N/lenslet_pixel)
    lenslet_vec = np.zeros((num_lenslet, num_lenslet, 2), dtype=float)
    ijlenslet_vec = np.zeros((lenslet_pixel, lenslet_pixel), dtype=float)

    for i_lenslet in np.arange(num_lenslet):
        for j_lenslet in np.arange(num_lenslet):
            temp_sum = np.zeros((2), dtype=float)
            ijlenslet_vec = vec[i_lenslet*lenslet_pixel
                                    : (i_lenslet + 1)*lenslet_pixel,
                                    j_lenslet*lenslet_pixel
                                    : (j_lenslet + 1)*lenslet_pixel, : ]
            for i in np.arange(lenslet_pixel):
                for j in np.arange(lenslet_pixel):
                    for k in np.arange(2):
                        temp_sum[k] = temp_sum[k] + ijlenslet_vec[i, j, k]
            lenslet_vec[i_lenslet, j_lenslet, : ] = temp_sum/lenslet_pixel**2

    return lenslet_vec

###############################################################################
########################################################################

# 21) Measure intensity by lenslet. #
def get_val_lenslet(val, lenslet_pixel):
    """
    Measure average intensity field chopped into lenslet.

    Wanna rename it: avg val lenslet
                    so it can be used for both avg intensity, avg phase

    Parameters
    ----------
    val : 2darray
        (N,N)
        Full resolution.
    lenslet_pixel : int
        Number of pixels in each row of each lenslet.

    Returns
    -------
    lenslet_val : 2darray
        (N/lenslet_pixel, N/lenslet_pixel)
        Averaged/lenslet resolution.

    """
    val = np.asarray(val)
    lenslet_pixel = int(lenslet_pixel)
    N = len(val)
    num_lenslet = int(N/lenslet_pixel)
    lenslet_val = np.zeros((num_lenslet, num_lenslet),
                           dtype=type(val[0,0]))
    ijlenslet_val = np.zeros((lenslet_pixel, lenslet_pixel),
                             dtype=type(val[0,0]))

    for i_lenslet in np.arange(num_lenslet):
        for j_lenslet in np.arange(num_lenslet):
            temp_sum = 0.0
            ijlenslet_val = val[i_lenslet*lenslet_pixel
                            : (i_lenslet + 1)*lenslet_pixel,
                            j_lenslet*lenslet_pixel
                            : (j_lenslet + 1)*lenslet_pixel]
            for i in np.arange(lenslet_pixel):
                for j in np.arange(lenslet_pixel):
                    temp_sum = temp_sum + ijlenslet_val[i, j]
            lenslet_val[i_lenslet, j_lenslet] = temp_sum/lenslet_pixel**2

    return lenslet_val

###############################################################################
########################################################################

# 23) Normalized 1D correlation. #
def norm_correlate1(a, b):
    """
    Generate normalized correlation between 2 1darrays.

    Parameters
    ----------
    a : 1darray
        (N)
        Any type
        Vector a.
    b : 1darray
        (N)
        Any type
        Vector b.

    Returns
    -------
    c : float
        Normalized correlation between a and b.

    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

#    a = a - a.mean()
#    a = a/a.std()

#    b = b - b.mean()
#    b = b/b.std()

    c = np.correlate(a,b)[0]/np.sqrt((a**2).sum()*(b**2).sum())
    return c

###############################################################################
########################################################################

# 24) Add phase caused by lens. #
def pass_lens(U_in, wvl, d, f):
    """
    Pass wavefront through lens.

    Based on Equation 4.7, Page 58, Chapter 4
    in Schmidt, Jason D., (2010)
    Numerical Simulation of Optical Wave Propagation
    with examples in MATLAB.
    BUT use angular-spectrum propagation instead of 1-step Fresnel.
    Output spacing is exactly the same with the input,
    because it barely passes through a super-duper thin lens.

    Parameters
    ----------
        Uin : 2darray
            (N, N)
            complex
            Input array of complex amplitude.
        wvl : float
            Unit is m.
            Wavelength of propagated light in metres.
        d : float
            Unit is m.
            Spacing of input plane.
        f : float
            Unit is m.
            Focal length of lens.
            Positive for a focusing lens.

    Returns
    -------
        U_after_lens : 2darray
            (N,N)
            complex
            Complex ampltitude after propagation through lens.

    """
    U_in = np.asarray(U_in, dtype=complex)
    wvl = float(wvl)
    d = float(d)
    f = float(f)

    k = 2.*np.pi/wvl
    N = len(U_in)

    U_after_lens = np.zeros_like(U_in, dtype=complex)

    x = np.arange(-N/2, N/2) * d
    x1, y1 = np.meshgrid(x, x)
    r1sq = x1**2 + y1**2 + 1e-10

    U_after_lens = U_in * np.exp(-1j*k*r1sq/2./f)

    return U_after_lens

###############################################################################
########################################################################

# 25) Add half pixel shift for SH-WFS. #
def tilt_to_centre_image(U_in, d_lenslet, d_detector, f_lenslet, wvl):
    """
    Centre SHWFS spots to be in the very corner of the pixels.

    This is applicable for 1 lenslet only.

    Parameters
    ----------
    U_in : 2darray
        (N,N)
        complex
        Actual complex amplitude at the pupil of a WFS lenslet.
    d_lenslet : float
        Input spacing at SHWFS pupil.
        Unit is metre.
    d_detector : float
        Input spacing at SHWFS sensor.
        Unit is metre.
    f_lenslet : float.
        Focal length of SHWFS lenslet.
        Unit is metre.
    wvl : float
        Propagating wavelength.
        Unit is metre.

    Returns
    -------
    U_shifted : 2darray
        (N,N)
        complex
        Rigged-to-be-centred complex amplitude.

    """
    U_in = np.asarray(U_in, dtype=complex)
    d1 = float(d_lenslet)
    d2 = float(d_detector)
    f = float(f_lenslet)
    wvl = float(wvl)

    U_shifted = np.zeros_like(U_in, dtype=complex)

    N = len(U_in)
    x = np.arange(-N/2, N/2) * d1
    x1, y1 = np.meshgrid(x, x)

    U_shifted = U_in * np.exp(1j*((-(x1 + y1) * (d2/(2.*f))) * (2.*np.pi/wvl)))

    return U_shifted

###############################################################################
########################################################################

# 26) Propagating wave through SHWFS lenslet to SHWFS detector. #
def pupil_to_focus(U_pupil, wvl, d1, f, d2, pf, plot=False):
    """
    Properly pass wavefront through SHWFS lenslet.

    It is centred perfectly on the corners of pixels.

    Parameters
    ----------
        U_pupil : 2darray
            (N, N)
            (complex, complex)
            Input array of complex amplitude at SHWFS pupil.
        wvl : float
            Unit is m.
            Wavelength of propagated light in metres.
        d1 : float
            Unit is m.
            Spacing of input plane.
        f : float
            Unit is m.
            Focal length of lens.
        d2 : float
            Unit is m.
            Spacing of output plane.
        pf : int
            Padding factor for Fourier transform.
            Size of zero padded.
        plot : bool
            For debug.

    Returns
    -------
        U_focus == U_out : 2darray
            (N,N)
            (complex, complex)
            Complex ampltitude after propagation to SHWFS detector.

    """
    U_in = np.asarray(U_pupil, dtype=complex)
    wvl = float(wvl)
    d1 = float(d1)
    f = float(f)
    d2 = float(d2)
    pf = int(pf)

    N = len(U_in)

    U_to_prop = np.zeros_like(U_in, dtype=complex)
    U_out = np.zeros_like(U_in, dtype=complex)

    if (N % 2) == 0:
        U_to_prop = tilt_to_centre_image(pass_lens(U_in, wvl, d1, f),
                                         d1, d2, f, wvl)
    else:
        U_to_prop = pass_lens(U_in, wvl, d1, f)

    U_out = mprop.crop2(
        angularSpectrum(
            mprop.zero_pad2(U_to_prop, pf),
            wvl, d1, d2, f),
        pf)

    if plot == True:
        plt.pcolor(np.arange(-len(U_in)/2, len(U_in)/2) * d1,
                   np.arange(-len(U_in)/2, len(U_in)/2) * d1,
                   np.abs(U_in)**2/np.abs((U_in)**2).max(),
                   shading='auto', cmap=reversed_color_map)
        plt.axis('equal')
        plt.colorbar()
        plt.xlabel('x (pixel)')
        plt.ylabel('y (pixel)')
        plt.title('Input normalized intensity')
        plt.show()
        plt.pcolor(np.arange(-len(U_in)/2, len(U_in)/2) * d1,
                   np.arange(-len(U_in)/2, len(U_in)/2) * d1,
                   np.angle(U_in),
                   shading='auto', cmap=reversed_color_map)
        plt.axis('equal')
        plt.colorbar()
        plt.xlabel('x (pixel)')
        plt.ylabel('y (pixel)')
        plt.title('Input phase')
        plt.show()
        # plt.plot(-0.5*d2,-0.5*d2,
        #          marker='x', color='r', ls='', markersize=10)
        plt.pcolor(np.arange(-len(U_out)/2, len(U_out)/2) * d2,
                   np.arange(-len(U_out)/2, len(U_out)/2) * d2,
                   np.abs(U_out)**2/np.abs((U_out)**2).max(),
                   shading='auto', cmap=reversed_color_map)
        plt.axis('equal')
        plt.colorbar()
        plt.xlabel('x (pixel)')
        plt.ylabel('y (pixel)')
        plt.title('Output normalized intensity')
        plt.show()
        # plt.plot(-0.5*d2,-0.5*d2,
        #          marker='x', color='r', ls='', markersize=10)
        plt.pcolor(np.arange(-len(U_out)/2, len(U_out)/2) * d2,
                   np.arange(-len(U_out)/2, len(U_out)/2) * d2,
                   np.angle(U_out),
                   shading='auto', cmap=reversed_color_map)
        plt.axis('equal')
        plt.colorbar()
        plt.xlabel('x (pixel)')
        plt.ylabel('y (pixel)')
        plt.title('Output phase')
        plt.show()
    return U_out

###############################################################################
########################################################################

# 27) check phase screen. #
def check_screen(phase, d_in, wvl, d_out, z_list, pf):
    """
    Measure slope/gradient (vector) of a given scalar field.

    No fold

    Parameters
    ----------
    phase : 2darray
        (N,N)
        float
        radian
        Contain phase at any given point.
    pixel_scale : float
        Distance between each given element in the array.
        m (preferrably)

    """
    # Inputs
    phase = mprop.crop2(phase,pf)
    d_in = float(d_in)
    d_out = float(d_out)
    # Constants
    N = len(phase)
    # Declarations
    R = N/2.*d_in
    A=np.pi*2./wvl*(d_in-d_out)
    slope=np.zeros((2), dtype=float)
    # Calculations
    for z in z_list:
        cond = np.abs(A*R/z)
        if cond > np.pi:
            print(z)
            print('The scaling difference is too high for z_list')
            print('{:.6e} is larger than pi'.format(cond))
            return False
    for i in np.arange(N - 1):
        for j in np.arange(N - 1):
            # Difference
            # Order of 0,1 or i,j depends on plottings???
            # This order is perfect for pcolor and quiver.
            slope[0] = (phase[i + 1, j] - phase[i, j]) # should be 0
            slope[1] = (phase[i, j + 1] - phase[i, j]) # should be 1
            # Solve too large difference at the phase folding.
            for k in np.arange(2):
                # If difference size is larger than half a circle,
                # it means this is the folding.
                for z in z_list:
                    criteria = np.abs(slope[k] + A*R/z)
                    if criteria > np.pi:
                        # If it is too positive, subtract a full circle.
                        print('This screen is too dense.')
                        print(np.abs(slope[k]))
                        return False
    return True

###############################################################################
########################################################################

# 28) Propagate pupil image at SHWFS lenslet array to SHWFS detector. #
def propagate_lenslet(U_WFS_pupil, wvl, d_lenslet, f_lenslet, lenslet_pixel,
                      d_detector, pf_lenslet,
                      plot_array=False, plot_lenslet=False):
    """
    Propagate SHWFS pupil plane/lenslet array through all lens to focus.

    To make this 1:1 & Nyquist,
        *** 1:1 : d_in = d_out
        *** Nyquist : wvl*f/d_in/lenslet_pixel > 2*d_out.
    d_in is the actual lenslet size
        not the projected lenslet size on the sky.

    Parameters
    ----------
    U_WFS_pupil : 2darray
        (N, N)
        (complex, complex)
        Input complex amplitude.
    wvl : float
        Units is m.
        Propagation wavelength.
    d_lenslet : float
        Unit is m.
        Input pixel scale size.
    f_lenslet : float
        Unit is m.
        Focal length of the lens.
    lenslet_pixel : float
        Unit is pixel.
        Number of pixel per lenslet.
    d_detector : float
        Unit is m.
        Output pixel scale size.
    pf_lenslet : int
        Padding factor for Fourier transform.
        Size of zero padded.
    plot_array : bool
        Plot overall image of array or not.
        Default is False.
        True == plot, False == do not plot.
    plot_lenslet : bool
        Plot image of each lenslet or not.
        Default is False.
        True == plot, False == do not plot.

    Returns
    -------
    lenslet_U_out : 2darray
        (N, N)
        (complex, complex)
        Outout complex amplitude.

    """
    U_in = np.asarray(U_WFS_pupil, dtype=complex)
    wvl = float(wvl)
    d_in = float(d_lenslet)
    f = float(f_lenslet)
    lenslet_pixel = int(lenslet_pixel)
    d_out = float(d_detector)
    pf_lenslet = int(pf_lenslet)

    N = len(U_in)
    num_lenslet = int(N/lenslet_pixel)

    ijlenslet_U_in = np.zeros((lenslet_pixel, lenslet_pixel), dtype=complex)
    ijlenslet_U_out = np.zeros((lenslet_pixel, lenslet_pixel), dtype=complex)
    lenslet_U_out = np.zeros((N, N), dtype=complex)

    for i_lenslet in np.arange(num_lenslet):
        for j_lenslet in np.arange(num_lenslet):
            ijlenslet_U_in = U_in[i_lenslet*lenslet_pixel
                            : (i_lenslet + 1)*lenslet_pixel,
                            j_lenslet*lenslet_pixel
                            : (j_lenslet + 1)*lenslet_pixel]
            ijlenslet_U_out = pupil_to_focus(ijlenslet_U_in, wvl, d_in, f,
                                             d_out, pf_lenslet,
                                             plot=plot_lenslet)
            lenslet_U_out[i_lenslet*lenslet_pixel
                         : (i_lenslet + 1)*lenslet_pixel,
                         j_lenslet*lenslet_pixel
                         : (j_lenslet + 1)*lenslet_pixel] = ijlenslet_U_out
    if plot_array == True:
        I_WFS = np.abs(lenslet_U_out)**2
        I_norm = lenslet_pixel**2./4.
        x = ((np.arange(-num_lenslet/2, num_lenslet/2) + 0.5) * lenslet_pixel
             - 0.5)
        xx, yy = np.meshgrid(x,x)
        plt.plot(xx.flatten(), yy.flatten(), color='r', marker='x', ls='',
                 markersize=1, label='lenslet centre')
        plt.legend()
        plt.pcolor(np.arange(-N/2, N/2 + 1),
                   np.arange(-N/2, N/2 + 1),
                   np.log(I_WFS/I_norm)/np.log(10.),
                   shading='auto', vmin=np.log(0.1)/np.log(10),
                   vmax=np.log(2)/np.log(10), snap=True)
        plt.axis('equal')
        plt.colorbar()
        plt.xlabel('x (pixel)')
        plt.ylabel('y (pixel)')
        plt.title('WFS log10 averaged intensity')
        plt.show()

    return lenslet_U_out

###############################################################################
########################################################################

# 29) Measure intensity in each lenslet. #
def get_total_I_SHWFS(U_detector, lenslet_pixel):
    """
    Measure total intensity in each SHWFS lenslet.

    Parameters
    ----------
    U_detector : 2darray
        (N, N)
        (complex, complex)
        Complex amplitude.
    lenslet_pixel : float
        Unit is pixel.
        Number of pixel per lenslet.

    Returns
    -------
    I_SHWFS : 2darray
        (N/lenslet_pixel, N/lenslet_pixel)
        (float, float)
        Total intensity in each lenslet.

    """
    U = np.asarray(U_detector, dtype=complex)
    lenslet_pixel = int(lenslet_pixel)
    N = len(U)
    num_lenslet = int(N/lenslet_pixel)

    ijlenslet_U = np.zeros((lenslet_pixel, lenslet_pixel), dtype=complex)
    total_I_SHWFS = np.zeros((num_lenslet, num_lenslet), dtype=float)

    for i in np.arange(num_lenslet):
        for j in np.arange(num_lenslet):
            ijlenslet_U = U[i*lenslet_pixel
                            : (i + 1)*lenslet_pixel,
                            j*lenslet_pixel
                            : (j + 1)*lenslet_pixel]
            total_I_SHWFS[i, j] = (np.abs(ijlenslet_U)**2).sum()
    return total_I_SHWFS

###############################################################################
########################################################################

# 30) Measure max intensity in each lenslet. #
def get_max_I_SHWFS(U_detector, lenslet_pixel):
    """
    Measure max intensity in each SHWFS lenslet.

    Parameters
    ----------
    U_detector : 2darray
        (N, N)
        (complex, complex)
        Complex amplitude.
    lenslet_pixel : float
        Unit is pixel.
        Number of pixel per lenslet.

    Returns
    -------
    I_SHWFS : 2darray
        (N/lenslet_pixel, N/lenslet_pixel)
        (float, float)
        Total intensity in each lenslet.

    """
    U = np.asarray(U_detector, dtype=complex)
    lenslet_pixel = int(lenslet_pixel)
    N = len(U)
    num_lenslet = int(N/lenslet_pixel)

    ijlenslet_U = np.zeros((lenslet_pixel, lenslet_pixel), dtype=complex)
    max_I_SHWFS = np.zeros((num_lenslet, num_lenslet), dtype=float)

    for i in np.arange(num_lenslet):
        for j in np.arange(num_lenslet):
            ijlenslet_U = U[i*lenslet_pixel
                            : (i + 1)*lenslet_pixel,
                            j*lenslet_pixel
                            : (j + 1)*lenslet_pixel]
            max_I_SHWFS[i, j] = (np.abs(ijlenslet_U)**2).max()
    return max_I_SHWFS

###############################################################################
########################################################################

# 31) Measure slopes at each lenslet. #
def get_spot_centre_SHWFS(I_detector, lenslet_pixel, plot=False):
    """
    Measure centre of each SHWFS lenslet spot, (0,0) = centre.

    Parameters
    ----------
    U_detector : 2darray
        (N, N)
        (complex, complex)
        Complex amplitude.
    lenslet_pixel : float
        Unit is pixel.
        Number of pixel per lenslet.
    plot : bool
        default to False.

    Returns
    -------
    slope_SHWFS : 2darray
        (N/lenslet_pixel, N/lenslet_pixel, 2)
        float
        Slope in each lenslet.

    Example
    -------
    N = 512

    wvl = 500e-9
    lenslet_pixel = 16
    D_lenslet = 150e-6
    num_lenslet = int(N/lenslet_pixel)
    d_lenslet = D_lenslet/lenslet_pixel
    f_lenslet = 2. * d_lenslet**2 * lenslet_pixel / wvl
    pf_lenslet = 8
    d_in = 5e-3
    d_out = 5e-3
    z = 25e3
    # x = np.arange(-N/2, N/2)
    # x = x*(x > 0) * 2.*np.pi/N*4.
    # xx, yy = np.meshgrid(x, x)
    # phase = - xx + yy
    
    time_N_phase_screen_generation_1 = time.time()
    phase = ft_sh_phase_screen(0.5, N, 5e-3, 20, 1.5e-10)
    time_N_phase_screen_generation_2 = time.time()

    U_in = np.exp(1j*phase)
    phase = np.angle(U_in)

    plt.pcolor(np.arange(-N/2, N/2),
               np.arange(-N/2, N/2),
               phase,
               shading='auto')
    plt.axis('equal')
    plt.colorbar()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    # plt.title('Phase = -x + y, x>0, y>0')
    plt.title('Atm phase')
    plt.show()

    time_WFS_1 = time.time()
    U_WFS = propagate_lenslet(U_in, wvl, d_lenslet, f_lenslet, lenslet_pixel,
                              d_lenslet, pf_lenslet, plot_array=False,
                              plot_lenslet=False)
    slope_SHWFS = get_spot_centre_SHWFS(U_WFS, lenslet_pixel, plot=True)
    slope_SHWFS = slope_SHWFS * d_lenslet / f_lenslet
    time_WFS_2 = time.time()

    plt.quiver(np.arange(-num_lenslet/2, num_lenslet/2),
               np.arange(-num_lenslet/2, num_lenslet/2),
               slope_SHWFS[ : , : , 0],
               slope_SHWFS[ : , : , 1])
    plt.title('Atm SHWFS - slope direction')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()

    time_slope_then_average_1 = time.time()
    full_slope = get_slope(phase, d_lenslet, wvl)
    lenslet_slope_then_average = get_vec_lenslet(full_slope, lenslet_pixel)
    time_slope_then_average_2 = time.time()

    plt.quiver(np.arange(-num_lenslet/2, num_lenslet/2),
               np.arange(-num_lenslet/2, num_lenslet/2),
               lenslet_slope_then_average[ : , : , 0],
               lenslet_slope_then_average[ : , : , 1])
    plt.title('Atm slope - take slope then average - slope direction')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()

    time_average_then_slope_1 = time.time()
    lenslet_phase = get_val_lenslet(phase, lenslet_pixel)
    lenslet_average_then_slope = get_slope(lenslet_phase,
                                           lenslet_pixel*d_lenslet, wvl)
    time_average_then_slope_2 = time.time()

    plt.quiver(np.arange(-num_lenslet/2, num_lenslet/2),
               np.arange(-num_lenslet/2, num_lenslet/2),
               lenslet_average_then_slope[ : , : , 0],
               lenslet_average_then_slope[ : , : , 1])
    plt.title('Atm slope - take average then slope - slope direction')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()
    
    plt.quiver(np.arange(-num_lenslet/2, num_lenslet/2),
               np.arange(-num_lenslet/2, num_lenslet/2),
               lenslet_slope_then_average[ : , : , 0]
               - slope_SHWFS[ : , : , 0],
               lenslet_slope_then_average[ : , : , 1]
               - slope_SHWFS[ : , : , 1])
    plt.title('Difference between measure SHWFS'
              + ' vs slope then average - slope direction')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()

    print('Atm WFS_propagation')
    print(time_WFS_2 - time_WFS_1)
    print('Atm slope then average')
    print(time_slope_then_average_2 - time_slope_then_average_1)
    print('Atm average then slope')
    print(time_average_then_slope_2 - time_average_then_slope_1)

    """
    I = np.asarray(I_detector, dtype=float)
    lenslet_pixel = int(lenslet_pixel)
    N = len(I)
    num_lenslet = int(N/lenslet_pixel)

    ijlenslet_I = np.zeros((lenslet_pixel, lenslet_pixel), dtype=float)
    position = np.zeros((num_lenslet, num_lenslet, 2), dtype=float)
    xx, yy = np.meshgrid(np.arange(lenslet_pixel), np.arange(lenslet_pixel))

    for i_lenslet in np.arange(num_lenslet):
        for j_lenslet in np.arange(num_lenslet):
            ijlenslet_I = I[i_lenslet*lenslet_pixel
                                   : (i_lenslet + 1)*lenslet_pixel,
                                   j_lenslet*lenslet_pixel
                                   : (j_lenslet + 1)*lenslet_pixel]
            temp_x = 0
            temp_y = 0
            temp = 0
            for i in np.arange(lenslet_pixel):
                for j in np.arange(lenslet_pixel):
                    temp_x = temp_x + ijlenslet_I[i,j]*xx[i,j]
                    temp_y = temp_y + ijlenslet_I[i,j]*yy[i,j]
                    temp = temp + ijlenslet_I[i,j]
            position[i_lenslet, j_lenslet, 0] = temp_x/temp
            position[i_lenslet, j_lenslet, 1] = temp_y/temp
            
            if plot == True:
                plt.pcolor(np.arange(lenslet_pixel),
                           np.arange(lenslet_pixel),
                           ijlenslet_I,
                           shading='auto')#, color=reversed_color_map)
                plt.plot(position[i_lenslet, j_lenslet, 0],
                         position[i_lenslet, j_lenslet, 1],
                         marker='o', color='r')
                plt.plot((lenslet_pixel - 1)/2., (lenslet_pixel - 1)/2.,
                         marker='x', color='r')
                plt.axis('equal')
                plt.title('centre = (x={:.2f}, y={:.2f})'.format(
                    position[i_lenslet, j_lenslet, 0],
                    position[i_lenslet, j_lenslet, 1]))
                plt.xlabel('x (pixel)')
                plt.ylabel('y (pixel)')
                plt.show()
    position = (position - (lenslet_pixel - 1)/2.)
    return position

###############################################################################
########################################################################

# 32) Get SHWFS slope. #
def get_SHWFS_slope(I_detector, lenslet_pixel, d_detector, f_lenslet):
    """
    Get SHWFS slope in radians. This is not wavefront error.

    Parameters
    ----------
    U_detector : TYPE
        DESCRIPTION.
    lenslet_pixel : TYPE
        DESCRIPTION.
    d_detector : TYPE
        DESCRIPTION.
    f_lenslet : TYPE
        DESCRIPTION.

    Returns
    -------
    SHWFS_slope : TYPE
        DESCRIPTION.

    """
    I = np.asarray(I_detector, dtype=float)
    lenslet_pixel = int(lenslet_pixel)
    d_detector = float(d_detector)
    f_lenslet = float(f_lenslet)
    N = len(I)
    SHWFS_slope = np.zeros((N, N, 2), dtype=float)
    SHWFS_slope = get_spot_centre_SHWFS(I, lenslet_pixel)
    SHWFS_slope = SHWFS_slope * d_detector / f_lenslet
    return SHWFS_slope

###############################################################################
########################################################################

# 33) Get SHWFS wavefront error. #
def get_SHWFS_WFE(U_WFS_pupil, wvl, d_lenslet, f_lenslet, lenslet_pixel,
                  d_detector, pf_lenslet):
    """
    Get SHWFS wavefront error of wavefront at WFS lenslet.

    This is the most realistic of how our SHWFS would detect.

    Parameters
    ----------
    U_WFS_pupil : TYPE
        DESCRIPTION.
    wvl : TYPE
        DESCRIPTION.
    d_lenslet : TYPE
        DESCRIPTION.
    f_lenslet : TYPE
        DESCRIPTION.
    lenslet_pixel : TYPE
        DESCRIPTION.
    d_detector : TYPE
        DESCRIPTION.
    pf_lenslet : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return get_SHWFS_slope(
        propagate_lenslet(
            U_WFS_pupil, wvl, d_lenslet, f_lenslet, lenslet_pixel,
            d_detector, pf_lenslet),
        lenslet_pixel, d_detector, f_lenslet) * d_lenslet * lenslet_pixel

###############################################################################
########################################################################

# 34) Get actual wavefront error. #
def get_lenslet_WFE(phase, pixel_scale, wvl, lenslet_pixel):
    """
    Get actual wavefront error of each lenslet.

    This is the most accurate representation of wavefront error.

    Parameters
    ----------
    phase : TYPE
        DESCRIPTION.
    pixel_scale : TYPE
        DESCRIPTION.
    wvl : TYPE
        DESCRIPTION.
    lenslet_pixel : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return get_vec_lenslet(
        get_slope(
            phase, pixel_scale, wvl),
        lenslet_pixel) * pixel_scale * lenslet_pixel

###############################################################################
########################################################################

# 35) Get cheated version of SHWFS. #
def get_intensity_weighted_WFE(phase, I, lenslet_pixel, d_lenslet, wvl):
    """
    Get a cheated version of SHWFS, without propagation.

    This is the closest approximation of SHWFS,
    without doing any propagation.

    Parameters
    ----------
    phase : TYPE
        DESCRIPTION.
    I : TYPE
        DESCRIPTION.
    lenslet_pixel : TYPE
        DESCRIPTION.
    d_lenslet : TYPE
        DESCRIPTION.

    Returns
    -------
    lenslet_weight_slope
        DESCRIPTION.

    """
    slope = get_slope(phase, d_lenslet, wvl)
    lenslet_I = get_val_lenslet(I, lenslet_pixel)

    weight_slope = np.zeros_like(slope, dtype=float)

    weight_slope[:,:,0] = I*slope[:,:,0]
    weight_slope[:,:,1] = I*slope[:,:,1]

    lenslet_weight_slope = get_vec_lenslet(weight_slope, lenslet_pixel)
    lenslet_weight_slope[:,:,0] = lenslet_weight_slope[:,:,0] / lenslet_I
    lenslet_weight_slope[:,:,1] = lenslet_weight_slope[:,:,1] / lenslet_I
    lenslet_weight_slope = lenslet_weight_slope * lenslet_pixel * d_lenslet
    return lenslet_weight_slope

###############################################################################
########################################################################




###############################################################################
# End of File ##########################################################
###############################################################################
