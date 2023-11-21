
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
from aotools import fouriertransform
import time

#
# Preferences #
#
color_map = plt.cm.get_cmap('viridis')
reversed_color_map = color_map.reversed()

def rebin(arr, new_shape):
    """
    Direct copy from the following source.
    
    https://scipython.com/blog/binning-a-2d-array-in-numpy/

    Parameters
    ----------
    arr : TYPE
        DESCRIPTION.
    new_shape : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

###############################################################################
########################################################################

# 19) Measure slope by pixel. #
def get_slope_corner(phase, search_branch_point=False):
    """
    Measure difference (vector) between neighboring pixels of any screen.

    Also detect any branch points.

    Given phase DON'T HAVE TO be continuous, i.e. unwrapable.
    As long as the slope is not too high, ie >pi within 1 pixel,
    then this function will give out correct slope.
    If the given phase array is a folded phase (-pi,pi),
    this function will automatically folded phase different back
    to be less than pi.

    Calculations
    ------------
        p  [-1 , 1]  ->  b  ->  q[ 1 , 1]
        ^                       |
        |                sy     v
        a    sx                 c
        ^                       |
        |                       v
        s  [-1 ,-1]  <-  d  <-  r[ 1 ,-1]

        0. Assume Nyquist sampling: each phase point should only be
        separated from its neighbor < pi.
        1. Measure difference between points p,q,r,s.
        2. If the difference is larger than pi, flip it.
        3. Average slope. If difference of difference is larger than pi,
        then the average is actually on the other side, ie pi away.

        Notice
        ------
        I should change this circular orientation.
        It is confusing to understand and to change, lol.

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
        (2, N - 1, N - 1)
        float
        radian.
        N is NOT kept.
        If N is needed, then write a new method.
        Yes it will be needed.
        *** IMPORTANT ***
        This 'slope' is actually phase difference.

    Example
    -------
    from aotools.turbulence import ft_phase_screen
    import time

    # N = int(32)
    timer = 0.
    N = 32
    pf = 32
    pixel_scale = 5e-3
    r0 = 0.05
    wvl = 500e-9
    x = np.arange(-N/2, N/2) + 0.5
    phase_grid = x * pixel_scale
    slope_grid = np.arange(-N/2 + 1, N/2) * pixel_scale

    # x = x*(x > 0) * 2.*np.pi/N*4.
    # xx, yy = np.meshgrid(x, x)
    # phase = - xx + yy
    # phase = np.angle(np.exp(1j*phase))

    U = mprop.crop2(
            angularSpectrum(
                np.exp(1j*ft_phase_screen(r0, N*pf, pixel_scale,
                                          20, 1e-10)),
                wvl, pixel_scale, pixel_scale, 25e3), pf)

    phase = np.angle(U)
    intensity = np.abs(U)**2
    
    # ta = time.time()
    # for i in np.arange(1000):

    slope_phase, branch_point_positive, branch_point_negative = get_slope_corner(phase,
                                                 search_branch_point=True)

    # tb = time.time()
    # print((tb - ta)/1000.)

    # slope_phase = np.asarray([np.diff(phase,axis=-1)[:-1,:],
    #                           np.diff(phase,axis=-2)[:,:-1]])
    # slope_phase = np.flip(np.asarray(np.gradient(phase)),axis=0)

    plt.pcolor(phase_grid, phase_grid, phase,
               shading='auto', snap=True)
    plt.colorbar()
    plt.quiver(slope_grid, slope_grid,
                slope_phase[0], slope_phase[1], headwidth=5, minshaft=2)
    # plt.quiver(phase_grid, phase_grid,
    #             slope_phase[0], slope_phase[1])
    # plt.quiver(phase_grid[:-1], phase_grid[:-1],
    #             slope_phase[0], slope_phase[1])
    plt.plot(pixel_scale*(branch_point_positive[0] - N/2 + 1),
             pixel_scale*(branch_point_positive[1] - N/2 + 1),
             marker='^', ls='',color='r')
    plt.plot(pixel_scale*(branch_point_negative[0] - N/2 + 1),
             pixel_scale*(branch_point_negative[1] - N/2 + 1),
             marker='v', ls='',color='r')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Phase and its slope')
    plt.show()

    plt.pcolor(phase_grid, phase_grid, intensity,
               shading='auto', snap=True)
    plt.colorbar()
    plt.quiver(slope_grid, slope_grid,
                slope_phase[0], slope_phase[1], headwidth=5, minshaft=2)
    # plt.quiver(phase_grid, phase_grid,
    #             slope_phase[0], slope_phase[1])
    # plt.quiver(phase_grid[:-1], phase_grid[:-1],
    #             slope_phase[0], slope_phase[1])
    plt.plot(pixel_scale*(branch_point_positive[0] - N/2 + 1),
             pixel_scale*(branch_point_positive[1] - N/2 + 1),
             marker='^', ls='',color='r')
    plt.plot(pixel_scale*(branch_point_negative[0] - N/2 + 1),
             pixel_scale*(branch_point_negative[1] - N/2 + 1),
             marker='v', ls='',color='r')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('intensity and its slope')
    plt.show()

    This Example should comfort the followings
        1) Axis of x, y
        2) Direction of x, y
    Between phase / slope should match
    To allow measurement to be accurate, input should be nyquist sampled,
    which means peak to valley = 2 pixels
    or peak to peak = edge to edge = 4 pixels.

    """
    a = phase[ :-1,1:  ] - phase[ :-1, :-1]
    b = phase[1:  ,1:  ] - phase[ :-1,1:  ]
    c = phase[1:  , :-1] - phase[1:  ,1:  ]
    d = phase[ :-1, :-1] - phase[1:  , :-1]
    # a = np.fmod(phase[:-1,1:] - phase[:-1,:-1], 2*np.pi)
    # b = np.fmod(phase[1:,1:] - phase[:-1,1:], 2*np.pi)
    # c = np.fmod(phase[1:,:-1] - phase[1:,1:], 2*np.pi)
    # d = np.fmod(phase[:-1,:-1] - phase[1:,:-1], 2*np.pi)

    atoolarge = np.where((np.abs(a) > np.pi) * (a > 0.))
    btoolarge = np.where((np.abs(b) > np.pi) * (b > 0.))
    ctoolarge = np.where((np.abs(c) > np.pi) * (c > 0.))
    dtoolarge = np.where((np.abs(d) > np.pi) * (d > 0.))
    atoosmall = np.where((np.abs(a) > np.pi) * (a < 0.))
    btoosmall = np.where((np.abs(b) > np.pi) * (b < 0.))
    ctoosmall = np.where((np.abs(c) > np.pi) * (c < 0.))
    dtoosmall = np.where((np.abs(d) > np.pi) * (d < 0.))

    a[atoolarge] = a[atoolarge] - 2*np.pi
    b[btoolarge] = b[btoolarge] - 2*np.pi
    c[ctoolarge] = c[ctoolarge] - 2*np.pi
    d[dtoolarge] = d[dtoolarge] - 2*np.pi
    a[atoosmall] = a[atoosmall] + 2*np.pi
    b[btoosmall] = b[btoosmall] + 2*np.pi
    c[ctoosmall] = c[ctoosmall] + 2*np.pi
    d[dtoosmall] = d[dtoosmall] + 2*np.pi

    sx = a + c
    sy = b + d

    slopex = (a - c)/2.
    slopey = (b - d)/2.

    sxtoolarge = np.where((np.abs(sx) > np.pi) * (slopex > 0.))
    sytoolarge = np.where((np.abs(sy) > np.pi) * (slopey > 0.))
    sxtoosmall = np.where((np.abs(sx) > np.pi) * (slopex < 0.))
    sytoosmall = np.where((np.abs(sy) > np.pi) * (slopey < 0.))


    slopex[sxtoolarge] = slopex[sxtoolarge] - np.pi
    slopey[sytoolarge] = slopey[sytoolarge] - np.pi
    slopex[sxtoosmall] = slopex[sxtoosmall] + np.pi
    slopey[sytoosmall] = slopey[sytoosmall] + np.pi

    # print(np.where(np.abs(slopex) > np.pi))
    # print(np.where(np.abs(slopey) > np.pi))

    # plt.hist(slopex.flatten(), bins=50, color='r', alpha=0.5)
    # plt.hist(slopex.flatten(), bins=50, color='b', alpha=0.5)
    # plt.show()

    # plt.pcolor((a+b+c+d)/2./np.pi);plt.axis('equal')
    # plt.colorbar()
    # plt.show()

    if search_branch_point == True:
        branch_point_positive = np.asarray(
            np.where(
                (np.isclose(
                    (a+b+c+d)/2/np.pi, 0, rtol=1e-1) == False)
                * (a+b+c+d>0)))
        branch_point_negative = np.asarray(
            np.where(
                (np.isclose(
                    (a+b+c+d)/2/np.pi, 0, rtol=1e-1) == False)
                * (a+b+c+d<0)))
        return np.asarray([slopex,slopey]), np.flip(branch_point_positive, axis=0), np.flip(branch_point_negative, axis=0)
    else:
        return np.asarray([slopex,slopey])

###############################################################################
########################################################################

def fast_slope(phase):
    a = phase[1:-1, 2 :  ] - phase[1:-1, 1 : -1 ]
    b = phase[ 2 :  ,1:-1] - phase[ 1 : -1 ,1:-1]
    atoolarge = np.where((np.abs(a) > np.pi) * (a > 0.))
    btoolarge = np.where((np.abs(b) > np.pi) * (b > 0.))
    atoosmall = np.where((np.abs(a) > np.pi) * (a < 0.))
    btoosmall = np.where((np.abs(b) > np.pi) * (b < 0.))
    a[atoolarge] = a[atoolarge] - 2*np.pi
    b[btoolarge] = b[btoolarge] - 2*np.pi
    a[atoosmall] = a[atoosmall] + 2*np.pi
    b[btoosmall] = b[btoosmall] + 2*np.pi
    return np.asarray([a,b])

###############################################################################
########################################################################

# Measure slope at each pixel.

def get_slope(phase):
    """
    Measure slope at each pixel using 5 stencil on wrapped screen.

    Cannot detect branch points.

    This method is not exactly slope.
    I mean the return unit is per pixel,
    so in a sense it is and also is not.
    We will see how it works out. lol.

    Left, Middle, Right <--> LMR.
    Top, Middle, Bottom <--> TMB.
    Slope = average of slope L - M and slope M - R.
    Slope = 0.5*(M - L + R - M) = (R - M)/2.
    When on the edge, it is either L - M or M - R.
    Similarly for TMB case.

    Assume Nyquist sampling,
    pi > difference per pixel. --> 2pi > difference per 2 pixel.
    If slope is measured between two pixel, there could be confusions.
    So difference has to be measured twice!!
    Adjust the above simplification to
    Slope = 0.5*((M - L) + (R - M))
    1) Get proper difference, M - L , R - M
    2) Get proper average, 0.5*((L - M) + (M - R))

    Parameters
    ----------
    phase : numpy.ndarray
            (N, N)
            Given Phase.
            unit = radian

    Returns
    -------
    slope : numpy.ndarray
            (2, N, N)
            Slope of the given phase.
            unit = wave phase difference in radian / pixel

    Example
    -------
    from aotools.turbulence import ft_phase_screen
    import time

    # N = int(32)
    timer = 0.
    N = 8
    pf = 128
    l_px = 4
    pixel_scale = 5e-3
    r0 = 0.05
    wvl = 500e-9
    x = np.arange(-N/2, N/2) + 0.5
    phase_grid = x * pixel_scale
    slope_grid = np.arange(-N/2 + 1, N/2) * pixel_scale
    n_l = int(N/l_px)
    l_grid = ((np.arange(n_l) + 0.5) * l_px - N/2) * pixel_scale
    grid_grid = np.arange(-n_l/2, n_l/2 + 1) * pixel_scale * l_px

    # x = x*(x > 0) * 2.*np.pi/N*4.
    # xx, yy = np.meshgrid(x, x)
    # phase = - xx + yy
    # phase = np.angle(np.exp(1j*phase))

    U = mprop.crop2(
            angularSpectrum(
                np.exp(1j*ft_phase_screen(r0, N*pf, pixel_scale,
                                          20, 1e-10)),
                wvl, pixel_scale, pixel_scale, 25e3), pf)

    phase = np.angle(U)
    intensity = np.abs(U)**2
    
    # ta = time.time()
    # for i in np.arange(1000):
    slope_phase = get_slope(phase)
    # tb = time.time()
    # print((tb - ta)/1000.)

    # slope_phase = np.asarray([np.diff(phase,axis=-1)[:-1,:],
    #                           np.diff(phase,axis=-2)[:,:-1]])
    # slope_phase = np.flip(np.asarray(np.gradient(phase)),axis=0)

    plt.pcolor(phase_grid, phase_grid, phase,
               shading='auto', snap=True)
    plt.colorbar()
    # plt.quiver(slope_grid, slope_grid,
    #             slope_phase[0], slope_phase[1], headwidth=5, minshaft=2)
    plt.quiver(phase_grid, phase_grid,
                slope_phase[0], slope_phase[1])
    # plt.quiver(phase_grid[:-1], phase_grid[:-1],
    #             slope_phase[0], slope_phase[1])
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Phase and its slope')
    plt.show()

    plt.pcolor(phase_grid, phase_grid, intensity,
               shading='auto', snap=True)
    plt.colorbar()
    # plt.quiver(slope_grid, slope_grid,
    #             slope_phase[0], slope_phase[1], headwidth=5, minshaft=2)
    plt.quiver(phase_grid, phase_grid,
                slope_phase[0], slope_phase[1])
    # plt.quiver(phase_grid[:-1], phase_grid[:-1],
    #             slope_phase[0], slope_phase[1])
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('intensity and its slope')
    plt.show()
    """
    slopex = np.zeros_like(phase)
    slopey = np.zeros_like(phase)
    ddifx = np.zeros_like(phase)
    ddify = np.zeros_like(phase)
    #             T = (1, 2)
    # L = (0, 1), M = (1, 1), R = (2, 1)
    #             B = (1, 0)

    # Calculate difference between pixels
    difx = phase[1 : , : ] - phase[ : -1, : ] # (N - 1, N) # R-M, M-L
    dify = phase[ : , 1 : ] - phase[ : , : -1] # (N, N - 1) # T-M, M-B

    # Apply Nyquist theorem
    # Detect where difference is too large.
    difxtoolarge = np.where((np.abs(difx) > np.pi) * (difx > 0.))
    difytoolarge = np.where((np.abs(dify) > np.pi) * (dify > 0.))
    difxtoosmall = np.where((np.abs(difx) > np.pi) * (difx < 0.))
    difytoosmall = np.where((np.abs(dify) > np.pi) * (dify < 0.))
    # 'Fix'
    difx[difxtoolarge] = difx[difxtoolarge] - 2*np.pi
    dify[difytoolarge] = dify[difytoolarge] - 2*np.pi
    difx[difxtoosmall] = difx[difxtoosmall] + 2*np.pi
    dify[difytoosmall] = dify[difytoosmall] + 2*np.pi
    # now difx, dify should be proper difference between RM,ML,TM,MB

    # Find where the difference of difference is too large,
    # that the average must be flipped before.
    ddifx[1 : -1, : ] = difx[1 : , : ] - difx[ : -1, : ]
    ddify[ : , 1 : -1] = dify[ : , 1 : ] - dify[ : , : -1]

    # non edges have to average
    slopex[1 : -1, : ] = 0.5*(difx[1 :, : ] + difx[ : -1, : ])
    slopey[ : , 1 : -1] = 0.5*(dify[ : , 1 : ] + dify[ : , : -1])
    # the edges don't need to average.
    slopex[0, : ] = difx[0, : ]
    slopex[-1, : ] = difx[-1, : ]
    slopey[ : , 0] = dify[ : , 0]
    slopey[ : , -1] = dify[ : , -1]

    ddifxtoolarge = np.where((np.abs(ddifx) > np.pi) * (slopex > 0.))
    ddifytoolarge = np.where((np.abs(ddify) > np.pi) * (slopey > 0.))
    ddifxtoosmall = np.where((np.abs(ddifx) > np.pi) * (slopex < 0.))
    ddifytoosmall = np.where((np.abs(ddify) > np.pi) * (slopey < 0.))

    slopex[ddifxtoolarge] = slopex[ddifxtoolarge] - np.pi
    slopey[ddifytoolarge] = slopey[ddifytoolarge] - np.pi
    slopex[ddifxtoosmall] = slopex[ddifxtoosmall] + np.pi
    slopey[ddifytoosmall] = slopey[ddifytoosmall] + np.pi

    return np.asarray([slopey, slopex])

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
    l_px : int
        Number of pixel in each lenslet.

    Returns
    -------
    lenslet_vec : 3darray
        (N/l_px, N/l_px, 2)
        float
        Averaged vector field into lenslet.

    Example
    -------
    from aotools.turbulence import ft_phase_screen
    import time

    # N = int(32)
    timer = 0.
    N = 64
    pf = 2
    l_px = 16
    pixel_scale = 5e-3
    r0 = 0.1
    wvl = 500e-9
    x = np.arange(-N/2, N/2) + 0.5
    phase_grid = x * pixel_scale
    slope_grid = np.arange(-N/2 + 1, N/2) * pixel_scale
    n_l = int(N/l_px)
    l_grid = ((np.arange(n_l) + 0.5) * l_px - N/2) * pixel_scale
    grid_grid = np.arange(-n_l/2, n_l/2 + 1) * pixel_scale * l_px
    D_lenslet = 150e-6
    f_lenslet = 1e-3

    x = x*(x > 0) * 2.*np.pi/N*4.
    xx, yy = np.meshgrid(x, x)
    phase = - xx + yy
    phase = np.angle(np.exp(1j*phase))

    # U = mprop.crop2(
    #         angularSpectrum(
    #             np.exp(1j*ft_phase_screen(r0, N*pf, pixel_scale,
    #                                       20, 1e-10)),
    #             wvl, pixel_scale, pixel_scale, 25e3), pf)

    # phase = np.angle(U)

    slope_phase = get_slope(phase)

    plt.pcolor(phase_grid, phase_grid, phase,
               shading='auto', snap=True)
    plt.colorbar()
    plt.quiver(phase_grid, phase_grid,
                slope_phase[0], slope_phase[1])
    plt.vlines(grid_grid,
               ymin=-N/2*pixel_scale, ymax=N/2*pixel_scale,
               color='r')
    plt.hlines(grid_grid,
               xmin=-N/2*pixel_scale, xmax=N/2*pixel_scale,
               color='r')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Phase and its slope')
    plt.show()

    # ta = time.time()
    # for i in np.arange(1000):
    l_slope = get_vec_lenslet(slope_phase, l_px)
    # l_slope = get_intensity_weighted_slope(U, l_px)
    # l_slope = get_SHWFS_slope(U, wvl, D_lenslet/l_px,
    #                                    wvl*f_lenslet/2./D_lenslet, f_lenslet,
    #                                    4, l_px, SNR=1000)
    # tb = time.time()
    # print((tb - ta)/1000.)

    plt.pcolor(phase_grid, phase_grid, phase,
               shading='auto', snap=True)
    plt.colorbar()
    plt.quiver(l_grid, l_grid,
                l_slope[0], l_slope[1])
    plt.vlines(grid_grid,
               ymin=-N/2*pixel_scale, ymax=N/2*pixel_scale,
               color='r')
    plt.hlines(grid_grid,
               xmin=-N/2*pixel_scale, xmax=N/2*pixel_scale,
               color='r')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Phase and its slope')
    plt.show()

    """
    n_l = int(vec.shape[-1]/lenslet_pixel)
    return np.moveaxis(
        np.mean(
        np.asarray(np.split(
            np.asarray(np.split(vec, n_l, axis=-1)),
            n_l, axis=-2)),
        axis=(-1,-2)),
        -1, 0)

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

    Example
    -------
    from aotools.turbulence import ft_phase_screen
    import time

    # N = int(16)
    timer = 0.
    N = 16
    pf = 2
    l_px = 4
    pixel_scale = 5e-3
    r0 = 0.05
    wvl = 500e-9
    x = np.arange(-N/2, N/2) + 0.5
    phase_grid = x * pixel_scale
    slope_grid = np.arange(-N/2 + 1, N/2) * pixel_scale
    n_l = int(N/l_px)
    l_grid = ((np.arange(n_l) + 0.5) * l_px - N/2) * pixel_scale
    grid_grid = np.arange(-n_l/2, n_l/2 + 1) * pixel_scale * l_px

    # x = x*(x > 0) * 2.*np.pi/N*4.
    # xx, yy = np.meshgrid(x, x)
    # phase = - xx + yy
    # phase = np.angle(np.exp(1j*phase))

    U = mprop.crop2(
            angularSpectrum(
                np.exp(1j*ft_phase_screen(r0, N*pf, pixel_scale,
                                          20, 1e-10)),
                wvl, pixel_scale, pixel_scale, 25e3), pf)

    phase = np.angle(U)

    # slope_phase = get_slope(phase)

    plt.pcolor(phase_grid, phase_grid, phase,
               shading='auto', snap=True)
    plt.colorbar()
    # plt.quiver(phase_grid, phase_grid,
    #             slope_phase[0], slope_phase[1])
    plt.vlines(grid_grid,
               ymin=-N/2*pixel_scale, ymax=N/2*pixel_scale,
               color='r')
    plt.hlines(grid_grid,
               xmin=-N/2*pixel_scale, xmax=N/2*pixel_scale,
               color='r')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Phase and its slope')
    plt.show()

    # ta = time.time()
    # for i in np.arange(1000):
    l_phase = get_val_lenslet(phase, l_px)
    # tb = time.time()
    # print((tb - ta)/1000.)

    plt.pcolor(l_grid, l_grid, l_phase,
               shading='auto', snap=True)
    plt.colorbar()
    # plt.quiver(l_grid, l_grid,
    #             l_slope[0], l_slope[1])
    plt.vlines(grid_grid,
               ymin=-N/2*pixel_scale, ymax=N/2*pixel_scale,
               color='r')
    plt.hlines(grid_grid,
               xmin=-N/2*pixel_scale, xmax=N/2*pixel_scale,
               color='r')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Phase and its slope')
    plt.show()

    """
    n_l = int(val.shape[-1]/lenslet_pixel)
    return np.mean(
        np.asarray(np.split(
            np.asarray(np.split(val, n_l, axis=-1)),
            n_l, axis=-2)),
        axis=(-1,-2))


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

# 27) check phase screen. #
def check_screen(phase):
    """
    Check if the unwrapped screen is Nyquist sampled or not.

    Do not use this to check an wrapped screen. It will fail.

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
    # N = phase.shape[0]
    # Declarations
    # Calculations
    # Difference
    # Solve too large difference at the phase folding.
    difx = phase[:-1,:] - phase[1:,:]
    dify = phase[:,:-1] - phase[:,:-1]
    if np.max(np.abs(difx.flatten())) > np.pi:
        return False
    if np.max(np.abs(dify.flatten())) > np.pi:
        return False
    return True

###############################################################################
########################################################################

def focus(inputComplexAmp, wvl, inputSpacing, z, pf):
    """
    Focus light. Using angular resolution method. Use for only intensity.
    
    Example
    -------
    lenslet_pixel = 16
    D_lenslet = 48e-6
    inputSpacing = D_lenslet/lenslet_pixel
    wvl = 500e-9
    f_lenslet = 0.6e-3
    outputSpacing = wvl*f_lenslet/2./D_lenslet
    z = f_lenslet
    pf = 2

    Parameters
    ----------
    inputComplexAmp : TYPE
        DESCRIPTION.
    wvl : TYPE
        DESCRIPTION.
    inputSpacing : TYPE
        DESCRIPTION.
    outputSpacing : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    pf : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    N = pf*inputComplexAmp.shape[0] #Assumes Uin is square.
    k = 2*np.pi/wvl     #optical wavevector
    
    (x1,y1) = np.meshgrid(inputSpacing*np.arange(-N/2,N/2),
                             inputSpacing*np.arange(-N/2,N/2))
    r1sq = (x1**2 + y1**2) + 1e-10
    
    #Spatial Frequencies (of source plane)
    df1 = 1. / (N*inputSpacing)
    fX,fY = np.meshgrid(df1*np.arange(-N/2,N/2),
                           df1*np.arange(-N/2,N/2))
    fsq = fX**2 + fY**2

    #Scaling Param

    #Quadratic phase factors

    Q2 = np.exp(-1j * np.pi**2 * 2 * z/k*fsq)

    #Compute propagated field
    I_focus = np.abs(mprop.crop2(fouriertransform.ift2(
        Q2 * fouriertransform.ft2(
            mprop.zero_pad2(inputComplexAmp, pf)
            * np.exp(-1j*k/2./z * (((x1 + y1) * inputSpacing) + r1sq)),
            inputSpacing), df1), pf))**2

    # OR ANOTHER METHOD?
    return I_focus

###############################################################################
########################################################################

def focus_magnify(inputComplexAmp, wvl, inputSpacing, outputSpacing, z, pf):
    """
    Focus light. Using angular resolution method. Use for only intensity.
    
    Example
    -------
    lenslet_pixel = 16
    D_lenslet = 48e-6
    inputSpacing = D_lenslet/lenslet_pixel
    wvl = 500e-9
    f_lenslet = 0.6e-3
    outputSpacing = wvl*f_lenslet/2./D_lenslet
    z = f_lenslet
    pf = 2

    Parameters
    ----------
    inputComplexAmp : TYPE
        DESCRIPTION.
    wvl : TYPE
        DESCRIPTION.
    inputSpacing : TYPE
        DESCRIPTION.
    outputSpacing : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    pf : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    N = pf*inputComplexAmp.shape[0] #Assumes Uin is square.
    k = 2*np.pi/wvl     #optical wavevector
    
    (x1,y1) = np.meshgrid(inputSpacing*np.arange(-N/2,N/2),
                             inputSpacing*np.arange(-N/2,N/2))
    r1sq = (x1**2 + y1**2) + 1e-10
    
    #Spatial Frequencies (of source plane)
    df1 = 1. / (N*inputSpacing)
    fX,fY = np.meshgrid(df1*np.arange(-N/2,N/2),
                           df1*np.arange(-N/2,N/2))
    fsq = fX**2 + fY**2

    #Scaling Param
    mag = float(outputSpacing)/inputSpacing

    #Quadratic phase factors
    Q1 = np.exp( 1j * k/2. * (1-mag)/z * r1sq)

    Q2 = np.exp(-1j * np.pi**2 * 2 * z/mag/k*fsq)

    #Compute propagated field
    I_focus = np.abs(mprop.crop2(fouriertransform.ift2(
                    Q2 * fouriertransform.ft2(
                        Q1 * mprop.zero_pad2(inputComplexAmp, pf)
                        * np.exp(-1j*k/2./z*((x1 + y1) * outputSpacing
                                             + r1sq))
                        / mag, inputSpacing), df1), pf))**2

    # OR ANOTHER METHOD?
    return I_focus

###############################################################################
########################################################################

def focus_all_size(inputComplexAmp, wvl, inputSpacing, outputSpacing, z, pf):
    """
    Focus light. Using angular resolution method. Use for only intensity.
    
    Example
    -------
    lenslet_pixel = 16
    D_lenslet = 48e-6
    inputSpacing = D_lenslet/lenslet_pixel
    wvl = 500e-9
    f_lenslet = 0.6e-3
    outputSpacing = wvl*f_lenslet/2./D_lenslet
    z = f_lenslet
    pf = 2

    Parameters
    ----------
    inputComplexAmp : TYPE
        DESCRIPTION.
    wvl : TYPE
        DESCRIPTION.
    inputSpacing : TYPE
        DESCRIPTION.
    outputSpacing : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    pf : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    N = pf*inputComplexAmp.shape[0] #Assumes Uin is square.
    k = 2*np.pi/wvl     #optical wavevector
    
    (x1,y1) = np.meshgrid(inputSpacing*np.arange(-N/2,N/2),
                             inputSpacing*np.arange(-N/2,N/2))
    r1sq = (x1**2 + y1**2) + 1e-10
    
    #Spatial Frequencies (of source plane)
    df1 = 1. / (N*inputSpacing)
    fX,fY = np.meshgrid(df1*np.arange(-N/2,N/2),
                           df1*np.arange(-N/2,N/2))
    fsq = fX**2 + fY**2

    #Scaling Param
    mag = float(outputSpacing)/inputSpacing

    #Quadratic phase factors
    Q1 = np.exp( 1j * k/2. * (1-mag)/z * r1sq)

    Q2 = np.exp(-1j * np.pi**2 * 2 * z/mag/k*fsq)

    #Compute propagated field
    if (N % 2) == 1:
        I_focus = np.abs(mprop.crop2(fouriertransform.ift2(
                        Q2 * fouriertransform.ft2(
                            Q1 * mprop.zero_pad2(inputComplexAmp, pf)
                            * np.exp(-1j*k/2./z*r1sq)
                            / mag, inputSpacing), df1), pf))**2
    else:
        I_focus = np.abs(mprop.crop2(fouriertransform.ift2(
                        Q2 * fouriertransform.ft2(
                            Q1 * mprop.zero_pad2(inputComplexAmp, pf)
                            * np.exp(-1j*k/2./z*((x1 + y1) * outputSpacing
                                                 + r1sq))
                            / mag, inputSpacing), df1), pf))**2

    # OR ANOTHER METHOD?
    return I_focus

def focus_True(inputComplexAmp, wvl, inputSpacing, outputSpacing, z, pf):
    """
    Focus light. Using angular resolution method. Accuracy prioritized.
    
    Example
    -------
    lenslet_pixel = 16
    D_lenslet = 48e-6
    inputSpacing = D_lenslet/lenslet_pixel
    wvl = 500e-9
    f_lenslet = 0.6e-3
    outputSpacing = wvl*f_lenslet/2./D_lenslet
    z = f_lenslet
    pf = 2

    Parameters
    ----------
    inputComplexAmp : TYPE
        DESCRIPTION.
    wvl : TYPE
        DESCRIPTION.
    inputSpacing : TYPE
        DESCRIPTION.
    outputSpacing : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    pf : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    N = pf*inputComplexAmp.shape[0] #Assumes Uin is square.
    k = 2*np.pi/wvl     #optical wavevector
    
    (x1,y1) = np.meshgrid(inputSpacing*np.arange(-N/2,N/2),
                             inputSpacing*np.arange(-N/2,N/2))
    r1sq = (x1**2 + y1**2) + 1e-10
    
    #Spatial Frequencies (of source plane)
    df1 = 1. / (N*inputSpacing)
    fX,fY = np.meshgrid(df1*np.arange(-N/2,N/2),
                           df1*np.arange(-N/2,N/2))
    fsq = fX**2 + fY**2
    
    #Scaling Param
    mag = float(outputSpacing)/inputSpacing
    
    #Observation Plane Co-ords
    x2,y2 = np.meshgrid( outputSpacing*np.arange(-N/2,N/2),
                            outputSpacing*np.arange(-N/2,N/2) )
    r2sq = x2**2 + y2**2
    
    #Quadratic phase factors
    Q1 = np.exp( 1j * k/2. * (1-mag)/z * r1sq)
    
    Q2 = np.exp(-1j * np.pi**2 * 2 * z/mag/k*fsq)
    
    Q3 = np.exp(1j * k/2. * (mag-1)/(mag*z) * r2sq)
    
    #Compute propagated field
    if (N % 2) == 1:
        outputComplexAmp = Q3 * fouriertransform.ift2(
                        Q2 * fouriertransform.ft2(
                            Q1 * mprop.zero_pad2(inputComplexAmp, pf)
                            * np.exp(-1j*k*r1sq/2./z)
                            / mag, inputSpacing), df1)
    else:
        outputComplexAmp = Q3 * fouriertransform.ift2(
                        Q2 * fouriertransform.ft2(
                            Q1 * mprop.zero_pad2(inputComplexAmp, pf)
                            * np.exp(1j*((-(x1 + y1)
                                          * (outputSpacing/(2.*z)))
                                         * (2.*np.pi/wvl)
                                         - k*r1sq/2./z))
                            / mag, inputSpacing), df1)

    # OR ANOTHER METHOD?
    return mprop.crop2(outputComplexAmp, pf)

def focus_1step(inputComplexAmp, wvl, inputSpacing, z, pf):
    """
    Focus light. ~one-step fresnel. NOT NYQUIST SAMPLED.

    For outputSpacing = wvl * z / N / inputSpacing ONLY.
    angular resolution per pixel = OutputSpacing/z
    If wanting to separate diffraction limited spot into spot_size pixels
    Then outputSpacing*spot_size/z = wvl/N/inputSpacing
    So: wvl*z/N/inputSpacing/spot_size = wvl*z/N/inputSpacing
      : 1/spot_size = 1
      : spot_size = 1
      : THIS IS NOT NYQUIST SAMPLED

    Example
    -------
    lenslet_pixel = 16
    D_lenslet = 48e-6
    inputSpacing = D_lenslet/lenslet_pixel
    wvl = 500e-9
    f_lenslet = 0.6e-3
    outputSpacing = wvl*f_lenslet/2./D_lenslet
    z = f_lenslet
    pf = 2

    Parameters
    ----------
    inputComplexAmp : TYPE
        DESCRIPTION.
    wvl : TYPE
        DESCRIPTION.
    inputSpacing : TYPE
        DESCRIPTION.
    outputSpacing : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    pf : TYPE
        DESCRIPTION.

    Returns
    -------
    I : np.ndarray
        (float, float)
        Intensity.

    """
    N = pf*inputComplexAmp.shape[0] #Assumes Uin is square.

    #Spatial Frequencies (of source plane)
    (x1,y1) = np.meshgrid(inputSpacing*np.arange(-N/2, N/2),
                             inputSpacing*np.arange(-N/2, N/2))
    outputSpacing = wvl * z / N / inputSpacing
    #Quadratic phase factors
    #Compute propagated field

    if (N % 2) == 1:
        I = np.abs(mprop.crop2(fouriertransform.ft2(
            mprop.zero_pad2(inputComplexAmp, pf), inputSpacing), pf))**2
    else:
        I = np.abs(mprop.crop2(fouriertransform.ft2(
            mprop.zero_pad2(inputComplexAmp, pf)
            * np.exp(1j*((-(x1 + y1)
                          * (outputSpacing/(2.*z)))
                         * (2.*np.pi/wvl))),
            inputSpacing), pf))**2

    return I

###############################################################################
########################################################################

# def lenslet_image(inputComplexAmp, wvl, inputSpacing, outputSpacing,
#                   z, pf, lenslet_pixel):
#     n_l = int(inputComplexAmp.shape[0] / lenslet_pixel)
#     for i in np.arange(n_l):
#         for j in np.arange(n_l):
#             inputComplexAmp[i*lenslet_pixel
#                             : (i + 1)*lenslet_pixel,
#                             j*lenslet_pixel
#                             : (j + 1)*lenslet_pixel] = focus(inputComplexAmp[
#                                 i*lenslet_pixel
#                                 : (i + 1)*lenslet_pixel,
#                                 j*lenslet_pixel
#                                 : (j + 1)*lenslet_pixel],
#                                 wvl, inputSpacing, outputSpacing, z, pf)
#     return inputComplexAmp

###############################################################################
########################################################################

def get_SHWFS_slope(inputComplexAmp, wvl, inputSpacing, outputSpacing,
                  z, pf, lenslet_pixel):
    """
    Simulate no noise SHWFS.

    Return unit is m per lenslet.

    Example
    -------
    from aotools.turbulence import ft_phase_screen
    import time

    # N = int(32)
    timer = 0.
    N = 128
    pf = 2
    l_px = 32
    pixel_scale = 10e-2/32
    r0 = 0.02
    wvl = 500e-9
    z = 30e3
    x = np.arange(-N/2, N/2) + 0.5
    phase_grid = x * pixel_scale
    slope_grid = np.arange(-N/2 + 1, N/2) * pixel_scale
    n_l = int(N/l_px)
    l_grid = ((np.arange(n_l) + 0.5) * l_px - N/2) * pixel_scale
    grid_grid = np.arange(-n_l/2, n_l/2 + 1) * pixel_scale * l_px
    # D_lenslet = 48e-6
    # f_lenslet = 0.6e-3
    D_lenslet = 10e-2
    spot_size = 2

    # x = x*(x > 0) * 2.*np.pi/N*4.
    # xx, yy = np.meshgrid(x, x)
    # phase = - xx + yy
    # phase = np.angle(np.exp(1j*phase))
    # intensity = np.ones_like(xx)

    U = mprop.crop2(
            angularSpectrum(
                np.exp(1j*ft_phase_screen(r0, N*pf, pixel_scale,
                                          20, 1e-10)),
                wvl, pixel_scale, pixel_scale, z), pf)
    phase = np.angle(U)
    intensity = np.abs(U)**2

    # ta = time.time()
    # for i in np.arange(1000):
    # l_slope = get_vec_lenslet(slope_phase, l_px)
    # l_slope = get_SHWFS_slope((intensity*np.exp(1j*phase)), wvl, D_lenslet/l_px,
    #                                     wvl*f_lenslet/2./D_lenslet, f_lenslet,
    #                                     4, l_px, show_one=True, show_all=True)
    l_slope = get_SHWFS_slope((intensity*np.exp(1j*phase)), wvl, D_lenslet/l_px,
                                        D_lenslet/l_px, (D_lenslet/l_px)**2*spot_size*l_px/wvl,
                                        4, l_px, show_one=True, show_all=True)
    # tb = time.time()
    # print((tb - ta)/1000.)

    plt.pcolor(phase_grid, phase_grid, phase,
               shading='auto', snap=True)
    plt.colorbar()
    plt.quiver(l_grid, l_grid,
                l_slope[0], l_slope[1])
    plt.vlines(grid_grid,
               ymin=-N/2*pixel_scale, ymax=N/2*pixel_scale,
               color='r')
    plt.hlines(grid_grid,
               xmin=-N/2*pixel_scale, xmax=N/2*pixel_scale,
               color='r')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Phase and its slope')
    plt.show()

    plt.pcolor(phase_grid, phase_grid, intensity/intensity.max(),
               shading='auto', snap=True, vmin=0.1/l_px**2, vmax=1)
    plt.colorbar()
    plt.quiver(l_grid, l_grid,
                l_slope[0], l_slope[1],color='r')
    plt.vlines(grid_grid,
               ymin=-N/2*pixel_scale, ymax=N/2*pixel_scale,
               color='r')
    plt.hlines(grid_grid,
               xmin=-N/2*pixel_scale, xmax=N/2*pixel_scale,
               color='r')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('intensity and its slope no saturation')
    plt.show()

    Parameters
    ----------
    inputComplexAmp : TYPE
        DESCRIPTION.
    wvl : TYPE
        DESCRIPTION.
    inputSpacing : TYPE
        DESCRIPTION.
    outputSpacing : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    pf : TYPE
        DESCRIPTION.
    lenslet_pixel : TYPE
        DESCRIPTION.
    plot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.
        unit = wave phase gradient in nm / lenslet @ SHWFS lenslet
        ie. what's the next dm actuator push in nm.

    """
    num_lenslet = int(inputComplexAmp.shape[-1]/lenslet_pixel)
    chop = np.asarray(np.split(
        np.asarray(np.split(
            inputComplexAmp, num_lenslet, axis=-1)),
        num_lenslet, axis=-2))
    xx, yy = np.meshgrid(np.arange(lenslet_pixel), np.arange(lenslet_pixel))
    slope = np.zeros((2, num_lenslet, num_lenslet), dtype=float)
    for i in np.arange(num_lenslet):
        for j in np.arange(num_lenslet):
            I = focus(chop[i,j], wvl,
                             inputSpacing, z, pf)
            I_total = np.sum(I.flatten())
            # this slope is deviation angle of spots
            slope[0, i, j] = np.sum((I*xx).flatten())/I_total
            slope[1, i, j] = np.sum((I*yy).flatten())/I_total

        #           recenter the spot          change px to m  get angle
        # displacement * scale / focus = px * (m/px) / (m) ==> deviation angle
    return ((slope - (lenslet_pixel - 1)/2.)
            * (outputSpacing / z * lenslet_pixel * inputSpacing))
        #   phase change per lenslet @ SHWFS lenslet
        # deviation angle * pixel in lenslet (pixel) * pixel scale (m/pixel) = path difference in (m)
###############################################################################
########################################################################

def get_SHWFS_slope_with_plots(inputComplexAmp, wvl, inputSpacing, outputSpacing,
                  z, pf, lenslet_pixel, show_one=False, show_all=False):
    """
    Simulate no noise SHWFS.

    Return unit is radian per pixel.

    Example
    -------
    from aotools.turbulence import ft_phase_screen
    import time

    # N = int(32)
    timer = 0.
    N = 128
    pf = 2
    l_px = 32
    pixel_scale = 10e-2/32
    r0 = 0.02
    wvl = 500e-9
    z = 30e3
    x = np.arange(-N/2, N/2) + 0.5
    phase_grid = x * pixel_scale
    slope_grid = np.arange(-N/2 + 1, N/2) * pixel_scale
    n_l = int(N/l_px)
    l_grid = ((np.arange(n_l) + 0.5) * l_px - N/2) * pixel_scale
    grid_grid = np.arange(-n_l/2, n_l/2 + 1) * pixel_scale * l_px
    # D_lenslet = 48e-6
    # f_lenslet = 0.6e-3
    D_lenslet = 10e-2
    spot_size = 2

    # x = x*(x > 0) * 2.*np.pi/N*4.
    # xx, yy = np.meshgrid(x, x)
    # phase = - xx + yy
    # phase = np.angle(np.exp(1j*phase))
    # intensity = np.ones_like(xx)

    U = mprop.crop2(
            angularSpectrum(
                np.exp(1j*ft_phase_screen(r0, N*pf, pixel_scale,
                                          20, 1e-10)),
                wvl, pixel_scale, pixel_scale, z), pf)
    phase = np.angle(U)
    intensity = np.abs(U)**2

    # ta = time.time()
    # for i in np.arange(1000):
    # l_slope = get_vec_lenslet(slope_phase, l_px)
    # l_slope = get_SHWFS_slope((intensity*np.exp(1j*phase)), wvl, D_lenslet/l_px,
    #                                     wvl*f_lenslet/2./D_lenslet, f_lenslet,
    #                                     4, l_px, show_one=True, show_all=True)
    l_slope = get_SHWFS_slope((intensity*np.exp(1j*phase)), wvl, D_lenslet/l_px,
                                        D_lenslet/l_px, (D_lenslet/l_px)**2*spot_size*l_px/wvl,
                                        4, l_px, show_one=True, show_all=True)
    # tb = time.time()
    # print((tb - ta)/1000.)

    plt.pcolor(phase_grid, phase_grid, phase,
               shading='auto', snap=True)
    plt.colorbar()
    plt.quiver(l_grid, l_grid,
                l_slope[0], l_slope[1])
    plt.vlines(grid_grid,
               ymin=-N/2*pixel_scale, ymax=N/2*pixel_scale,
               color='r')
    plt.hlines(grid_grid,
               xmin=-N/2*pixel_scale, xmax=N/2*pixel_scale,
               color='r')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Phase and its slope')
    plt.show()

    plt.pcolor(phase_grid, phase_grid, intensity/intensity.max(),
               shading='auto', snap=True, vmin=0.1/l_px**2, vmax=1)
    plt.colorbar()
    plt.quiver(l_grid, l_grid,
                l_slope[0], l_slope[1],color='r')
    plt.vlines(grid_grid,
               ymin=-N/2*pixel_scale, ymax=N/2*pixel_scale,
               color='r')
    plt.hlines(grid_grid,
               xmin=-N/2*pixel_scale, xmax=N/2*pixel_scale,
               color='r')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('intensity and its slope no saturation')
    plt.show()

    Parameters
    ----------
    inputComplexAmp : TYPE
        DESCRIPTION.
    wvl : TYPE
        DESCRIPTION.
    inputSpacing : TYPE
        DESCRIPTION.
    outputSpacing : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    pf : TYPE
        DESCRIPTION.
    lenslet_pixel : TYPE
        DESCRIPTION.
    plot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.
        unit = wave phase gradient in nm / lenslet @ SHWFS lenslet
        ie. what's the next dm actuator push in nm.

    """
    num_lenslet = int(inputComplexAmp.shape[-1]/lenslet_pixel)
    chop = np.asarray(np.split(
        np.asarray(np.split(
            inputComplexAmp, num_lenslet, axis=-1)),
        num_lenslet, axis=-2))
    xx, yy = np.meshgrid(np.arange(lenslet_pixel), np.arange(lenslet_pixel))
    slope = np.zeros((2, num_lenslet, num_lenslet), dtype=float)
    if (show_all == True) or (show_one == True):
        if show_all == True:
            allimage = np.zeros_like(inputComplexAmp, dtype=float)
        for i in np.arange(num_lenslet):
            for j in np.arange(num_lenslet):
                I = focus(chop[i,j], wvl,
                                 inputSpacing, z, pf)
                I_total = np.sum(I.flatten())
                slope[0, i, j] = np.sum((I*xx).flatten())/I_total
                slope[1, i, j] = np.sum((I*yy).flatten())/I_total
                if show_all == True:
                    allimage[lenslet_pixel*i
                         : lenslet_pixel*(i + 1),
                         lenslet_pixel*j
                         : lenslet_pixel*(j + 1)] = I
                if show_one == True:
                    plt.pcolor(np.arange(lenslet_pixel),
                               np.arange(lenslet_pixel),
                               I/I_total*4,
                               shading='auto', snap=True,
                               vmin=0, vmax=1)
                    plt.colorbar()
                    plt.plot(slope[0,i,j],
                             slope[1,i,j],
                             marker='x', color='r')
                    # plt.plot((lenslet_pixel - 1)/2., (lenslet_pixel - 1)/2.,
                    #          marker='o', color='r')
                    plt.axis('equal')
                    plt.title('centre = (x={:.2f}, y={:.2f})'.format(
                        slope[0,i,j],
                        slope[1,i,j]))
                    plt.xlabel('x (pixel)')
                    plt.ylabel('y (pixel)')
                    plt.savefig('WFS-{:.0f}.png'.format(time.time()))
                    plt.show()
                    plt.clf()
                    show_one=False
        if show_all == True:
            x = np.arange(-num_lenslet*lenslet_pixel/2,
                          num_lenslet*lenslet_pixel/2) + 0.5
            phase_grid = x
            l_grid = (((np.arange(num_lenslet) + 0.5)*lenslet_pixel
                       - num_lenslet*lenslet_pixel/2))
            grid_grid = (np.arange(-num_lenslet/2, num_lenslet/2 + 1)
                         * lenslet_pixel)
            marka, markb = np.meshgrid(l_grid,l_grid)
            plt.pcolor(phase_grid,
                       phase_grid,
                       allimage/(allimage.max()),
                       vmin=0, vmax=1,
                       shading='auto', snap='True')
            plt.colorbar()
            plt.plot(marka, markb, marker='x', color='r', ls='')
            plt.vlines(grid_grid,
                       ymin=-num_lenslet*lenslet_pixel/2,
                       ymax=num_lenslet*lenslet_pixel/2,
                       color='r')
            plt.hlines(grid_grid,
                       xmin=-num_lenslet*lenslet_pixel/2,
                       xmax=num_lenslet*lenslet_pixel/2,
                       color='r')
            plt.axis('equal')
            plt.xlabel('x (px)')
            plt.ylabel('y (px)')
            plt.title('SHWFS image and slope no saturation')
            plt.show()

    else:
        for i in np.arange(num_lenslet):
            for j in np.arange(num_lenslet):
                I = focus(chop[i,j], wvl,
                                 inputSpacing, z, pf)
                I_total = np.sum(I.flatten())
                # this slope is deviation angle of spots
                slope[0, i, j] = np.sum((I*xx).flatten())/I_total
                slope[1, i, j] = np.sum((I*yy).flatten())/I_total

        #           recenter the spot          change px to m  get angle
    return ((slope - (lenslet_pixel - 1)/2.) * outputSpacing / z * lenslet_pixel * inputSpacing)
        #   phase change per lenslet @ SHWFS lenslet
###############################################################################
########################################################################

# 35) Get cheated version of SHWFS. #
def get_intensity_weighted_slope(U, lenslet_pixel):
    r"""
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
        unit = wave phase difference in radian / pixel

    Example
    -------
    from aotools.turbulence import ft_phase_screen
    import time

    N = 1024
    pf = 2
    l_px = 16
    pixel_scale = 5e-3
    r0 = 0.05
    wvl = 500e-9
    x = np.arange(-N/2, N/2) + 0.5
    phase_grid = x * pixel_scale
    slope_grid = np.arange(-N/2 + 1, N/2) * pixel_scale
    n_l = int(N/l_px)
    l_grid = ((np.arange(n_l) + 0.5) * l_px - N/2) * pixel_scale
    grid_grid = np.arange(-n_l/2, n_l/2 + 1) * pixel_scale * l_px
    D_lenslet = 150e-6
    f_lenslet = 1e-3
    pf_wfs = 4
    
    Round = 1

    ta = time.time()
    for i in np.arange(Round):
        phase = ft_phase_screen(r0, N*pf, pixel_scale,20, 1e-10)
    tb = time.time()
    tgen = (tb - ta)/Round

    ta = time.time()
    for i in np.arange(Round):
        U = mprop.crop2(
                angularSpectrum(
                    np.exp(1j*phase),
                    wvl, pixel_scale, pixel_scale, 25e3), pf)
    tb = time.time()
    tprop = (tb - ta)/Round

    phase = np.angle(U)
    slope_phase = get_slope(phase)

    plt.pcolor(phase_grid, phase_grid, phase,
               shading='auto', snap=True)
    plt.colorbar()
    plt.quiver(phase_grid, phase_grid,
                slope_phase[0], slope_phase[1])
    # plt.vlines(grid_grid,
    #            ymin=-N/2*pixel_scale, ymax=N/2*pixel_scale,
    #            color='r')
    # plt.hlines(grid_grid,
    #            xmin=-N/2*pixel_scale, xmax=N/2*pixel_scale,
    #            color='r')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Phase and its slope,\n'
              + 'phase screen time={:.3f}s,\n'.format(tgen)
              + 'propagation time={:.3f}s'.format(tprop))
    plt.show()

    ta = time.time()
    for i in np.arange(Round):
        l_slope = get_vec_lenslet(get_slope(np.angle(U)), l_px)
    tb = time.time()
    tott = ((tb - ta)/Round)

    plt.pcolor(phase_grid, phase_grid, phase,
               shading='auto', snap=True)
    plt.colorbar()
    plt.quiver(l_grid, l_grid,
                l_slope[0], l_slope[1])
    # plt.vlines(grid_grid,
    #            ymin=-N/2*pixel_scale, ymax=N/2*pixel_scale,
    #            color='r')
    # plt.hlines(grid_grid,
    #            xmin=-N/2*pixel_scale, xmax=N/2*pixel_scale,
    #            color='r')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Average Slope, time={:.3f}s'.format(tott))
    plt.show()

    ta = time.time()
    for i in np.arange(Round):
        l_slope = get_intensity_weighted_slope(U, l_px)
    tb = time.time()
    tott = ((tb - ta)/Round)

    plt.pcolor(phase_grid, phase_grid, phase,
               shading='auto', snap=True)
    plt.colorbar()
    plt.quiver(l_grid, l_grid,
                l_slope[0], l_slope[1])
    # plt.vlines(grid_grid,
    #            ymin=-N/2*pixel_scale, ymax=N/2*pixel_scale,
    #            color='r')
    # plt.hlines(grid_grid,
    #            xmin=-N/2*pixel_scale, xmax=N/2*pixel_scale,
    #            color='r')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Intensity Weighted Slope, time={:.3f}s'.format(tott))
    plt.show()

    ta = time.time()
    for i in np.arange(Round):
        l_slope = get_SHWFS_slope(U, wvl, D_lenslet/l_px,
                                        wvl*f_lenslet/2./D_lenslet, f_lenslet,
                                        pf_wfs, l_px)
    tb = time.time()
    tott = ((tb - ta)/Round)

    plt.pcolor(phase_grid, phase_grid, phase,
               shading='auto', snap=True)
    plt.colorbar()
    plt.quiver(l_grid, l_grid,
                l_slope[0], l_slope[1])
    # plt.vlines(grid_grid,
    #            ymin=-N/2*pixel_scale, ymax=N/2*pixel_scale,
    #            color='r')
    # plt.hlines(grid_grid,
    #            xmin=-N/2*pixel_scale, xmax=N/2*pixel_scale,
    #            color='r')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('SHWFS Slope, time={:.3f}s'.format(tott))
    plt.show()
    """
    I = np.abs(U)**2

            # avg(intensity * (phase difference per px)) / avg(intensity)
            # sum(                                     ) / sum(         )
            # intensity-weighted phase difference per px
    return (get_vec_lenslet(get_slope(np.angle(U)) * I, lenslet_pixel)
            / get_val_lenslet(I, lenslet_pixel))

###############################################################################
########################################################################



###############################################################################
########################################################################

###############################################################################
# End of File ##########################################################
###############################################################################
