
###############################################################################
# Start of file ########################################################
###############################################################################
"""
Library of AO codes on Adaptive Optics (AO).

Implemented by Puttiwat Kongkaew (Geng/Greg) during his study
for Ph.D. in Physics (Advanced Instrumentation)
at Centre for Advanced Instrumentation, Department of Physics,
Durham University,United Kingdom from October 2021 to
(ETG) September 2024.
I will try to keep everything under my universal namings.
units, etc. to keep confusions at minimal.

"""
###############################################################################
# List of functions collected within this file #########################
###############################################################################
# 1) Power spectrum density. #
#       psd_function()
#       Create 2D power spectrum density (PSD).
# 2) Structure function. #
#       structure_function()
#       Create a structure function of the given screen.
# 3) Gram-Schmidt orthonormalization. #
#       gram_schmidt_process()
#       Orthonormalize any given set of functions with Gram-Schmidt process.
# 4) Test orthonormality. #
#       orthonormality_test()
#       Test out orthonormality of the given modesby doing dot product.
# 5) Mode strength. #
#       zernike_modes_strength()
#       Measure variance of each zernike mode off sub-harmonic von Karman.
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
# 14 ) Sinc exponentially decaying curve. #
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
# 19) Measure slope by pixel. #
#       get_slope_pixel()
#       Measure slope/gradient (vector) of a given scalar field.
# 20) Measure slope by lenslet. #
#       get_slope_lenslet()
#       Measure average slope vector of a scalar field chopped into lenslet.
# 21) Measure intensity by lenslet. #
#       get_val_lenslet()
#       Measure average intensity field chopped into lenslet.
# 22) Parameters check. #
#       prop_param_check()
#       Check propagation parameter whether or not it is good.
# 23) Normalized 1D correlation. #
#       norm_correlate1()
#       Generate normalized correlation between 2 1darrays.
###############################################################################
########################################################################

#
# Imports #
#

import numpy as np
import time
import random
from aotools.functions import zernike
from aotools.turbulence.phasescreen import ft_sh_phase_screen
from aotools.opticalpropagation import angularSpectrum
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.special import j1

#
# Preferences #
#

color_map = plt.cm.get_cmap('viridis')
reversed_color_map = color_map.reversed()

###############################################################################
########################################################################

# 1) Power spectrum density. #
def psd_function(r0, N, delta, L0, l0, seed=None):
    """
    Create 2D power spectrum density (PSD).

    It is derived almost exactly from within ft_phase_screen in
    aotools/turbulence/phasescreen.py which, in turn, based on
    Listing 9.2, Chapter 9, Page 168 of Schmidt, Jason D., (2010)
    Numerical Simulation of Optical Wave Propagation
    with examples in MATLAB.

    PSD shows turbulence distribution
    through out different spatial frequencies.

    Parameters
    ----------
    r0 : float
        r0 parameter of a turbulence layer in meters.
    N : int
        Size of phase screen in pixels.
    delta : float
        Size of each pixel in meters.
    L0 : float
        Size of outer scale in meters.
    l0 : float
        Size of inner scale in meters.
    seed : int, optional
        Specify this to obtain a static result.
        The default is None.

    Returns
    -------
    fx : 2darray
        2D numpy array representing output grid of PSD in x axis.
        Element type is float.
        Dimension is (N,N)
        Unit is 1/m.
    fy : 2darray
        2D numpy array representing output grid of PSD in y axis.
        Element type is float.
        Dimension is (N,N)
        Unit is 1/m.
    PSD_phi : 2darray
        Numpy array representing power spectrum density (PSD)
        Element type is float.
        Dimension is (N,N)
        Unit of PSD is m^2.

    Example
    -------
    fx, fy, PSD = psd_function(r0=0.05, N=1024, delta=0.2,
                               L0=20, l0=1.5e-10)
    fx, fy, PSD = psd_function(r0=0.05, N=1024*2, delta=5e-3,
                               L0=20, l0=1.5e-10)

    """
    delta = float(delta)                        # spatial pixel size (m)
    r0 = float(r0)                              # Fried parameter (m)
    L0 = float(L0)                              # outer scale (m)
    l0 = float(l0)                              # inner scale (m)
    N = int(N)

    R = random.SystemRandom(time.time())
    if seed is None:
        seed = int(R.random()*100000)    
    np.random.seed(seed)

    del_f = 1./(N*delta)                        # frequency pixel scale
    fx = np.arange(-N/2., N/2.) * del_f         # frequency 1d grid
    (fx, fy) = np.meshgrid(fx,fx)               # frequency 2d grid
    f = np.sqrt(fx**2 + fy**2)                  # freqency distance
    fm = 5.92/l0/(2.*np.pi)
    f0 = 1./L0

    # see listing 9.2 page 168 chapter 9 of Schmidt 2010
    PSD_phi = (0.023 * r0**(-5./3.)
               * np.exp(-1.*((f/fm)**2)) / (((f**2) + (f0**2))**(11./6.)))
    PSD_phi[int(N/2), int(N/2)] = 0

    # # Plot #

    plt.pcolor(fx[int(N/2) - 128 : int(N/2) + 128,
                  int(N/2) - 128 : int(N/2) + 128],
               fy[int(N/2) - 128 : int(N/2) + 128,
                  int(N/2) - 128 : int(N/2) + 128],
               np.log(PSD_phi[int(N/2) - 128 : int(N/2) + 128,
                              int(N/2) - 128 : int(N/2) + 128])/np.log(10),
               shading='auto')
    plt.title('Log_10 of power spectrum density (m$^2$)')
    plt.colorbar()
    plt.xlabel('x (1/m)')
    plt.ylabel('y (1/m)')
    plt.axis('equal')
    plt.show()

    plt.plot(fx[int(N/2), int(N/2 + 1) : ], PSD_phi[int(N/2), int(N/2 + 1) : ],
             label='Power spectrum density', color='r')
    plt.vlines(1./L0, ymin=PSD_phi[int(N/2 + 1), int(N/2)],
               ymax=PSD_phi[int(N/2 + 1), -1], color='b', label='L0')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.title('Cross section of power spectrum density (PSD)')
    plt.xlabel('Spatial frequency (1/m)')
    plt.ylabel('Power spectrum density (m$^2$)')
    plt.show()

    return fx, fy, PSD_phi

###############################################################################
########################################################################

# 2) Structure function. #
def structure_function(screen, random_radius=None, n_bin=None, n_random=None):
    """
    Create a structure function of the given screen.

    Structure function is a log-log plot of mean difference squared
    between each pixel on the screen versus their separation distance.

    Parameters
    ----------
    screen : 2darray
        SQUARE 2D numpy array to be measured.
        Size of screen should be > 200x200.
        Element type is float.
        Unit is not specified.
    random_radius : int, optional
        Max separation distance to be measured for structure function.
        Recommend screen.shape[0]/4. or less.
        The default is None = get the recommended value.
    n_bin : int, optional
        Number of bins of separation distance.
        Recommend (screen.shape[0] - 2*random_radius) / 10. or less.
        The default is None = get the recommended value. 
    n_random : int, optional
        Number of random measurement pairs.
        Recommend random_radius**2 / 10. or less.
        The default is None = get the recommended value. 

    Returns
    -------
    bin_center : 1darray
        1D numpy array of size = n_bin.
        Element type is float.
        Center of each bins of the structure function.
        Contains linear value.
        Unit is pixel.
    mean_difference_square : 1darray
        1D numpy array of size = n_bin.
        Element type is float.
        Represent mean difference squared between each pixel
        of the structure function.
        Contains linear value.
        Unit = (same unit with values in Args:screen) squared.
    error_of_mean : 1darray
        1D numpy array of size = n_bin
        Element type is float.
        Represent error bar of the mean difference squared.
        Contains linear value.
        Unit = (same unit with values in Args:screen) squared.

    Steps
    -----
    1) Random x&y of the first half of the pair from within
        random_radius centered at the center of the screen.
    2) Random distance and angle which the second half of the pair
        will be with respect to the first half of the pair.
    3) Calculate x&y of the second half of the pair from 1) & 2).
    4) Calculate difference and separation distance required
        for structure function
    5) Generate bins on separation distance in log_10 space.
    6) Sort difference into bins.
    7) Find expectation value and errorbar of each bin.
        <|a-a0|^2> <==> mean(difference**2)
    8) Return separation and structure function.

    Following this procedures guarantee that we can sample almost evenly
    linearly radially.
    Improvement can be made by sampling in log space, instead.

    Example
    -------
    d = 0.002
    r = 0.2
    Num = 1024
    L = 20
    x, y, y_error = structure_function(ft_sh_phase_screen(
        r0=r, N=Num, delta=d, L0=L, l0=1.5e-10, FFT=None, seed=None),
        random_radius=256, n_bin=50,
        n_random=100000*50)
    plt.errorbar(x*d, y, yerr=y_error, label='Measured', color='r')
    plt.plot(x*d, 6.88 * (x*d/r)**(5./3.), label='Zernike Model')
    plt.vlines(L, ymin=np.nanmin(y), ymax=np.nanmax(y), color='b', label='L0')
    plt.xlabel('Separation distance (m)')
    plt.ylabel('Structure function (rad^2)')
    plt.xlim(np.nanmin(x*d*0.9), np.nanmax(x*d*1.1))
    plt.ylim(np.nanmin(y*0.9), np.nanmax(y*1.1))
    plt.xscale('log')
    plt.yscale('log') 
    plt.legend()
    plt.title('Comparison of phase structure function'
              + ' between measurement and theory')
    plt.show()

    d = 0.2
    r = 0.2
    Num = 1024
    L = 20
    x, y, y_error = structure_function(ft_sh_phase_screen(
        r0=r, N=Num, delta=d, L0=L, l0=1.5e-10, FFT=None, seed=None),
        random_radius=256, n_bin=50,
        n_random=100000*50)
    plt.errorbar(x*d, y, yerr=y_error, label='Measured', color='r')
    plt.plot(x*d, 6.88 * (x*d/r)**(5./3.), label='Zernike Model')
    plt.vlines(L, ymin=np.nanmin(y), ymax=np.nanmax(y), color='b', label='L0')
    plt.xlabel('Separation distance (m)')
    plt.ylabel('Structure function (rad^2)')
    plt.xlim(np.nanmin(x*d*0.9), np.nanmax(x*d*1.1))
    plt.ylim(np.nanmin(y*0.9), np.nanmax(y*1.1))
    plt.xscale('log')
    plt.yscale('log') 
    plt.legend()
    plt.title('Comparison of phase structure function'
              + ' between measurement and theory')
    plt.show()

    d = 5e-3
    r = 0.05
    Num = 1024*2
    L = 20
    x, y, y_error = structure_function(ft_sh_phase_screen(
        r0=r, N=Num, delta=d, L0=L, l0=1.5e-10, FFT=None, seed=None),
        random_radius=1024*2/4, n_bin=20,
        n_random=100000*20)
    plt.errorbar(x*d, y, yerr=y_error, label='Measured', color='r')
    plt.plot(x*d, 6.88*(x*d/r)**(5./3.), ls='-',
        label='Theoretical Values',color='g')
    plt.vlines(L, ymin=(y.min() - y_error.max()),
               ymax=(y.max() + y_error.max()),
               color='b', label='L0')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Separation distance (m)')
    plt.ylabel('Structure function (rad^2)')
    plt.legend()
    plt.title('Comparison of phase structure function'
              + ' between measurement and theory')
    plt.show()

    """
    #
    # Input reading & default initialization field #
    #

    # Assume screen to be squared
    screen_size = int(screen.shape[0])                      # int, pixel
    
    # Setting defaults value
    if random_radius == None:
        random_radius = int(np.floor(screen_size/4.))
    if n_bin == None:
        n_bin = int(np.floor((screen_size - 2*random_radius) / 10.))
    if n_random == None:
        n_random = int(np.floor(random_radius**2 / 10.))
    
    random_radius = int(random_radius)                      # int, pixel
    n_bin = int(n_bin)
    n_random = int(n_random)
    
    #
    # Declaration and initialization field #
    #
    
    # Choose random pair of points
    # first point of the pair
    points_a = np.zeros((n_random, 2), dtype=int)           # ndarray
                                                            # (int,int),
                                                            # (pixel,pixel)
    # second point of the pair
    # in polar coordinate
    points_b_polar = np.zeros((n_random, 2), dtype=float)   # ndarray
                                                            # (float,float),
                                                            # (pixel,radian)
    # in cartesian coordinate
    points_b = np.zeros((n_random, 2), dtype=int)           # ndarray
                                                            # (int,int),
                                                            # (pixel,pixel)
    # Difference between pixel pair
    difference = np.zeros(n_random, dtype=float)            # float,
                                                            # unit in screen
    # Separation between pixel pair
    separation = np.zeros(n_random, dtype=float)            # float, pixel
    
    # Others
    log_minimum_separation = 0.                             # float, log10(px)
    log_maximum_separation = 0.                             # float, log10(px)
    separation_space_log = 0.                               # float, log10(px)
    bin_center = np.zeros(n_bin, dtype=list)                # float, log10(px)
    binned_difference = np.empty(n_bin, dtype=list)         # ndarray
    for i in np.arange(n_bin):                              # list
        binned_difference[i]=[]                             # unit in screen
    # binned_difference[i] contain list of difference belongs to bin_center[i]
    belong_to_group = 0                                     # int
    mean_difference_square = np.zeros(n_bin, dtype=float)   # ndarray, float,
                                                            # unit squared
    error_of_mean = np.zeros(n_bin, dtype=float)            # ndarray, float,
                                                            # unit squared 

    #
    # 1) Choose x&y of the first half of the pair #
    #

    # This point must be random_radius away from the edges.
    # Note: np.random.randint(a,b) => random includes a but not b
    #       index of np.array => includes 0 to array size - 1
    # Example: oooooooo ==> screen_size = 8, random_radius = 2
    #          index: 01234567
    #          want to random including 2 to 5 for points_a
    #          so want to do np.random.randint(2,6)
    #               which is np.random.randint(random_radius,
    #                                          screen_size-random_size)
    for i in np.arange(n_random):
        # x [-edge,edge]
        points_a[i,0] = np.random.randint(random_radius,
            (screen_size - 1) - random_radius + 1)
        # y [-edge,edge]
        points_a[i,1] = np.random.randint(random_radius,
            (screen_size - 1) - random_radius + 1)

    #
    # 2) Choose radius & angle of the second half of the pair #
    #

    # np.random.rand() gives out random float from 0 to but not include 1
    # Example: want from 0 to 2pi -> np.random.rand()*(2pi)
    #          want spanning from but not include 0 to include radius 
    #          -> radius - np.random.rand()*(radius-2)
    #          -> radius - [0,1)*(radius-2)
    #          -> radius - [0,radius-2)
    #          -> [radius,2)
    # #### adjusted to 2 instead of 0 to keep things within dimension somehow.
    # #### adjusted to 8 instead for close to circular sampling
    # flooring / ceiling doesn't have much difference in random
    for i in np.arange(n_random):
        # radius (0,edge-2]
        points_b_polar[i,0] = (random_radius
                               - np.random.rand()*(random_radius-10))
        # angle [0,2pi)
        points_b_polar[i,1] = np.random.rand() * (2*np.pi)

    #
    # 3) Calculate x&y of the second half of the pair #
    #

    for i in np.arange(n_random):
        # x = x0 + r*cos(angle)
        points_b[i,0] = int(np.floor(
            points_b_polar[i,0]*np.cos(points_b_polar[i,1]) + points_a[i,0]))
        # y = y0 + r*sin(angle)
        points_b[i,1] = int(np.floor(
            points_b_polar[i,0]*np.sin(points_b_polar[i,1]) + points_a[i,1]))

    #
    # 4) Calculate difference in value and separation #
    #

    for i in np.arange(n_random):
        difference[i] = np.abs(screen[points_a[i,0], points_a[i,1]]
                               - screen[points_b[i,0], points_b[i,1]])
        separation[i] = np.sqrt((points_a[i,0] - points_b[i,0])**2
                                + (points_a[i,1] - points_b[i,1])**2)

    #
    # 5) Generate bins of separation in log_10 space #
    #

    # lower edge of lowest bin
    log_minimum_separation = np.log(min(separation))/np.log(10)
    # higher edge of highest bin
    # need 1.1* to off-set the actual max to be less than the edge
    log_maximum_separation = np.log(1.1*max(separation))/np.log(10)
    # evenly spaced in log space
    separation_space_log = (log_maximum_separation 
                            - log_minimum_separation) / n_bin
    # bin_center = (lower edge of lowest bin) + (0.5*space)
                 # + (space * number of bin)
    # in log space
    bin_center = (separation_space_log*np.arange(n_bin)
                  + 0.5*separation_space_log + log_minimum_separation)
    # in linear space
    bin_center = 10**bin_center

    #
    # 6) Sort into bins #
    #

    for i in np.arange(n_random):
        # group# = keep integer of ((val-min)/space)
        belong_to_group = int(np.floor((np.log(separation[i])/np.log(10)
                                        - log_minimum_separation)
                                       / separation_space_log))
        binned_difference[belong_to_group].append(difference[i])

    #
    # 7) Calculate mean of (difference) squared of each bin #
    #    and       error of mean of each bin #

    for i in np.arange(n_bin):
        temp = 0.           # temporary storage for sum of square of difference
        # For successful standard deviation calculation,
        # there must be more than 1 point
        if len(binned_difference[i]) > 1:
            for j in np.arange(len(binned_difference[i])):
                temp = temp + (binned_difference[i][j]**2)
            temp = temp/len(binned_difference[i])
            mean_difference_square[i] = temp
            error_of_mean[i] = np.std(mean_difference_square, ddof=1)
        else:
            mean_difference_square[i] = np.NaN
            error_of_mean[i] = np.NaN

    #
    # 8) Return #
    #

    return bin_center, mean_difference_square, error_of_mean

###############################################################################
########################################################################

# 3) Gram-Schmidt orthonormalization. #
def gram_schmidt_process(f_in):
    """
    Orthonormalize any given set of functions with Gram-Schmidt process.

    Based on Gram-Schmidt orthonormalization.
    https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    Notations is the same with the wiki page.

    Currently any shape of f_in[i] can be any shape, which increase
    flexibility, but will reduce performance speed if the input
    is already flat.  This should not be much of problem as this
    function should only be called once.

    Parameters
    ----------
    f_in : ndarray
        Numpy array of functions to be orthonormalized.
        f_in = [f_in[0], ..., f_in[i], ... , f_in[n-1]].
        Element type is ndarray.
        Dimension is (n,i,j); where n is number of functions
            to be orthonormalized, and i,j is shape of each functions.
        f_in[i] : 
            Numpy array of function values.
            Element type is float.
            Dimension is (i,j). There is no specific requirement.
            f_in must be cropped and only have values within its
            orthonormalized region.
            Any points out off bound should be zero out prior.
            Unit is not specified.

    Returns
    -------
    f_out : ndarray
        Similar to f_in but orthonormalized.

    Example
    -------
    gram_schmidt_zernike = gram_schmidt_process(zernike.zernikeArray(10,128))
    for i in np.arange(len(gram_schmidt_zernike)):
        plt.imshow(gram_schmidt_zernike[i])
        plt.title('Orthonormalized n={:d} Zernike mode'.format(i))
        plt.xlabel('x (pixel)')
        plt.ylabel('y (pixel)')
        plt.colorbar()
        plt.show()

    """
    #
    # Declaration and initialization space #
    #

    n = len(f_in)                               # number of modes
    shape_in = f_in[0].shape                    # shape of each input modes
    flat_f_in = np.empty(n, dtype=list)         # storing input flat modes
    for i in np.arange(n):
        flat_f_in[i] = f_in[i].flatten()
    f_out = np.empty(n, dtype=list)             # output: same shape with input
    for i in np.arange(n):
        f_out[i] = np.zeros(shape_in)
        
    # now in wiki's notation
    v = flat_f_in                               # not orthogonal, not normal
    u = np.zeros_like(v)                        # is orthogonal, not normal
    e = np.zeros_like(v)                        # is orthogonal, is normal

    #
    # 1) orthogonalize #
    #

    # zeroth mode is kept.
    u[0] = v[0]
    # for other modes. index is preserved
    for i in np.arange(n - 1) + 1:
        # orthogonalized mode is initially the original modes
        u[i] = v[i]
        for j in np.arange(i):
            # then take out not orthogonal part compared to prior ones
            u[i] = u[i] - (np.dot(u[j], v[i]) / np.dot(u[j], u[j])) * u[j]

    #
    # 2) normalize #
    #

    for i in np.arange(n):
        # divide by its size
        e[i]=u[i]/np.sqrt(np.dot(u[i], u[i]))

    #
    # 3) reshape then return #
    #

    for i in np.arange(n):
        f_out[i]=e[i].reshape(shape_in)
    return f_out

###############################################################################
########################################################################

# 4) Test orthonormality. #
def orthonormality_test(modes):
    """
    Test out orthonormality of the given modesby doing dot product.

    Shows 2D array containing dot products between each given functions.

    Parameters
    ----------
    modes : ndarray
        Numpy array of functions to be tested for orthonormalization.
        modes = [modes[0], ..., modes[i], ... , modes[n-1]].
        Element type is ndarray.
        Dimension is (n,i,j); where n is number of functions
            to be tested, and i,j is shape of each functions.
        modes[i] : ndarray
            Numpy array of function values.
            Element type is float.
            Dimension is (i,j). There is no specific requirement.
            modes_in must be cropped and only have values within its
            orthonormalized region.
            Any points out off bound should be zero out prior.
            Unit is not specified.

    Returns
    -------
    orthonormal : ndarray
        Dimension is (n,n); where n is number of modes to be tested.
            orthonormal[i,j] = modes[i] dot modes[j]
            If modes are orthonormal, the diagonal should b 1,
            and other terms should be 0.

    Example
    -------
    testz = orthonormality_test(gram_schmidt_process(
        zernike.zernikeArray(55,128)))
    testz = np.abs(testz)
    testz = testz*(testz>0) + 1e-17*(testz<=0)
    plt.imshow(np.log(testz)/np.log(10))
    plt.xlabel('i (mode)')
    plt.ylabel('j (mode)')
    plt.title('Inner product between each orthonormalized mode in log_10'
              + 'with 0 replaced with 1e-17')
    plt.colorbar()
    plt.show()

    """
    n = len(modes)                              # int number of modes
    orthonormal = np.zeros((n,n))               # matrix of dot products
    for i in np.arange(n):
        for j in np.arange(n):
            # dot product between each modes
            orthonormal[i,j] = np.dot(modes[i].flatten(), modes[j].flatten())
    return orthonormal

###############################################################################
########################################################################

# 5) Mode strength. #
def zernike_modes_strength(r0, N, delta, L0, l0, n_modes, n_random,
                           FFT=None):
    """
    Measure variance of each zernike mode off sub-harmonic von Karman.

    *** Actually no gram-schmidt is needed here at all.!!!
    *** This is because gram-schmidt only helps in fitting higher order,
    *** while having no effect on lower order at all,
    *** because gram-schmidt only change the higher order!!!
    *** To fix the JUMPS, new zernike generator must be created.
    *** Or use large enough screen size.
    *** screen size * pixel size << r0 to measure good?

    Modes in theory starts from 1 which is tip/tilt.
    Modes in this function fit includes 0 which is piston.
    So when plot, this should be considered.

    Conversion between bases is based on the following link.
    https://eli.thegreenplace.net/2015/change-of-basis-in-linear-
    algebra/#:~:text=%20Change%20of%20basis%20in%20Linear%20Algebra
    %20,.%20We%27ve%20used%20the%20basis%20before%3B%20More%20

    Parameters
    ----------
    r0 : float
        r0 parameter of screen in meter.
    N : int
        Size of phase screen in pixels.
    delta : float
        Size of each pixel in meter.
    L0 : float
        Outer scale size in meter.
    l0 : float
        Inner scale size in meter.
    n_modes : int
        Number of zernike modes to measure.
    n_random : int
        Number of screens to generate to sample for the variance.
    FFT : ..., optional
        To pass on for ft_sh_phase_screen. The default is None.

    Returns
    -------
    x : ndarray
        1D Numpy array of size = n_modes.
        Element type is int.
        Contains the numbering of mode.
        Unit is the numbering of mode.
    theory : ndarray
        1D numpy array of size = n_modes.
        Element type is float.
        Contains the theoretical Noll's variance of
            Zernike coefficients.
        Unit is radian squared.
    coefficients_variance : ndarray
        1D numpy array of size = n_modes.
        Element type is float.
        Contains the measured variance of
            Zernike coefficients.
        Unit is radian squared.
    coefficients_variance_error : ndarray
        1D numpy array of size = n_modes.
        Element type is float.
        Contains the error of measured variance of
            Zernike coefficients.
        Unit is radian squared.
        
    Steps
    -----
    1) Generate n_random screens of the specified parameter
        using aotools.phasescreen.ft_sh_phase_screen().
        If other functions is desired, then create a copy of this.
    2) Fit orthonormalized n_modes Zernike modes to screens.
    3) Revert coefficient of orthonormalized Zernike modes
        to the regular Zernike modes.
    4) Find variance and error of variance of
        Zernike coefficient of each mode.
    5) Return array of mode numbering, variance, error
        of variance of each Zernike coefficient of each mode,
        and theoretical value of each mode.

    Example
    -------
    n, noll, var, err_of_var = zernike_modes_strength(
        0.2, 1024, 1e-5, 20, 1.5e-10, 32, 50)
    plt.plot(n, noll, label='Noll variance')
    plt.errorbar(n, var, yerr=err_of_var, label='Measurement', ls='',
                 marker='o')
    plt.yscale('log')
    plt.title('Comparison of variance of each Zernike mode'
              + ' between Nolls theory and measurement')
    plt.xlabel('Zernike modes')
    plt.ylabel('Variance of each Zernike mode (rad^2)')
    plt.xlim(0,33)
    plt.ylim(noll[0:32].min()*0.8, noll[0:32].max()*1.2)
    plt.grid()
    plt.legend()
    plt.show()

    n, noll, var, err_of_var = zernike_modes_strength(
        0.2, 128, 0.002, 20, 1.5e-10, 32, 50)
    plt.plot(n, noll, label='Noll variance')
    plt.errorbar(n, var, yerr=err_of_var, label='Measurement', ls='',
                 marker='o')
    plt.yscale('log')
    plt.title('Comparison of variance of each Zernike mode'
              + ' between Nolls theory and measurement')
    plt.xlabel('Zernike modes')
    plt.ylabel('Variance of each Zernike mode (rad^2)')
    plt.xlim(0,33)
    plt.ylim(noll[0:32].min()*0.8, noll[0:32].max()*1.2)
    plt.grid()
    plt.legend()
    plt.show()

    """
    #
    # Input #
    #

    r0 = float(r0)
    N = int(N)
    delta = float(delta)
    L0 = float(L0)
    l0 = float(l0)
    n_modes = int(n_modes)
    n_random = int(n_random)

    #
    # Declaration and initialization field #
    #

    zernike_modes = (zernike.zernikeArray(n_modes + 1, N)
                     / np.sqrt((np.pi * N**2 / 4.)))
    flat_zernike_modes = np.empty(n_modes + 1, dtype=list)
    for i in np.arange(n_modes + 1):
        flat_zernike_modes[i] = zernike_modes[i].flatten()
    # gram_schmidt_modes = np.zeros_like(zernike_modes)
    # flat_gram_schmidt_modes = np.zeros_like(flat_zernike_modes)
    # zernike_to_gram_schmidt = np.zeros((n_modes + 1, n_modes + 1))
    # gram_schmidt_to_zernike = np.zeros((n_modes + 1, n_modes + 1))
    phase_screen = np.zeros((N,N))
    flatten_screen = np.zeros_like(flat_zernike_modes[0])
    # gram_schmidt_coefficients = np.zeros((n_random, n_modes + 1))
    zernike_coefficients = np.zeros((n_random, n_modes + 1))
    coefficients_variance = np.zeros(n_modes + 1)
    coefficients_variance_error = np.zeros(n_modes + 1)
    x = np.arange(n_modes) + 1

    #
    # 1) Get Zernike modes #
    #
    # Already did this in the declaration and initialization field

    #
    # 2) Get Gram Schmidt / orthonormalized zernike #
    #

    # gram_schmidt_modes = gram_schmidt_process(zernike_modes)
    # for i in np.arange(n_modes + 1):
    #     flat_gram_schmidt_modes[i] = gram_schmidt_modes[i].flatten()

    #
    # 3) Get conversions between zernike and gram schmidt & reverse #
    #

    # Because gram-schmidt is orthonormalized, when dot g[i] with z[j],
    # we can write z[j] = sum of (a[ij]=<g[i],z[j]>)*g[i]
    # Comparing this with the link by, u <-> z, w <-> g, v <-> screen
    # a[ij] <-> <g[i],z[j]>
    # [v]u = [screen]zernike = c <-> what we want to return
    # [v]w = [screen]gramschmidt = a[ij]*c[j] <-> what we measured
    # a[ij] = zernike_to_gram_schmidt: convert zernike coeff to gram schmidt
    
    # for i in np.arange(n_modes + 1):
    #     for j in np.arange(n_modes + 1):
    #         zernike_to_gram_schmidt[i,j] = np.dot(flat_gram_schmidt_modes[i],
    #                                               flat_zernike_modes[j])
    # gram_schmidt_to_zernike = np.linalg.inv(zernike_to_gram_schmidt)

    #
    # 4) Generate n_random phase screens and get gram schmidt coefficients #
    #

    for i in np.arange(n_random):
        # Generate a phase screen
        phase_screen = ft_sh_phase_screen(r0, N, delta, L0, l0, FFT=FFT)
        flatten_screen = phase_screen.flatten()
        for j in np.arange(n_modes + 1):
            # gram_schmidt_coefficients[i,j] = np.dot(flatten_screen,
            #                                  flat_gram_schmidt_modes[j])
            zernike_coefficients[i,j] = np.dot(flatten_screen,
                                          flat_zernike_modes[j])

    #
    # 5) Convert gram_schmidt coefficients to zernike coefficients #
    #

    # for i in np.arange(n_random):
    #     zernike_coefficients[i] = np.matmul(gram_schmidt_to_zernike,
    #                                         gram_schmidt_coefficients[i])

    #
    # 6) Measure variance and variance error #
    #

    for i in np.arange(n_modes + 1):
        coefficients_variance[i] = (np.var(zernike_coefficients[: , i], ddof=1)
                                    / (np.pi * N**2 / 4.))
        # variance = error**2
        # variance of variance = (error of variance)**2
        #                      = (error of error)**2 (2*error)**2
        # error of variance = error of error * 2 * error
        #                   = 2 * error * error/sqrt(2n-2)
        #                   = 2 * variance/sqrt(2n-2)
        coefficients_variance_error[i] = 2*(coefficients_variance[i]
                                          / np.sqrt(2*n_random - 2))

    #
    # 7) Noll's theoretical coefficients_variance #
    #

    # D[J]-D[J+1] = residue[J], J start from 1
    A = np.array([0.4479, 0.4480, 0.0230, 0.0230, 0.0232,
                  0.0061, 0.0062, 0.0062, 0.0062, 0.0024,
                  0.0025, 0.0024, 0.0024, 0.0025, 0.0012,
                  0.0012, 0.0012, 0.0011, 0.0012, 0.0012])
    if n_modes <= 20:
        theory = A[0 : n_modes]
    if n_modes > 20:
        excess = np.arange(20, n_modes) + 1
        B = 0.2944 * ((excess)**(-np.sqrt(3)/2.) 
                      - (excess + 1)**(-np.sqrt(3)/2.))
        theory = np.append(A,B)
    theory = theory * (N*delta/r0)**(5./3.)
    coefficients_variance = coefficients_variance[1:]
    coefficients_variance_error = coefficients_variance_error[1:]
    return x, theory, coefficients_variance, coefficients_variance_error

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

    x = np.arange(-N/2, N/2) * d_out
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
        plt.show()

    return None

###############################################################################
########################################################################

# 11) Variance of log amplitude over propagation distances. #
def var_log_amp(N, r, d_in, L0, l0, wvl, d_out, z_list, pf):
    """
    Compute variance log amplitude versus propagation distances.

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
    plt.hlines(0.5, xmin=z_list.min(), xmax=z_list.max(), label='1/2',
               color='b', ls='--')
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
    plt.title('Variance of log amplitude versus propagation distance')
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
    # Constants #
    #
    NN = int(pf*N)       # Int; pixel; Padded/oversized screen size.
    fitr = int(64)
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
    phase_in = np.angle(crop2(U_in), pf)
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
    expected_FWHM = (2.*np.sqrt(wvl*z_list/8.))
    FWHM_cov_I = FWHM_cov_I*d_out
    plt.plot(z_list, FWHM_cov_I,
                 label='Measured fit on cov(I)',
                 marker='o', color='r', ls='')
    plt.plot(z_list, expected_FWHM, label='1/8th Fresnel zone size',
             ls='--', color='g')
    plt.plot(z_list, wvl*z_list/2./r, label='r diffraction', color='y')
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
    plt.title('FWHM of covariance of intensity, r={:.3f}m'.format(r))
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
        Measurement.
    w : float
        Width of the sinc part.
    d : float
        Width of the decaying exponential.

    Returns
    -------
    y : TYPE
        DESCRIPTION.

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
        float
        Square dimension.
        Centred around the diagonal axis -> [i,j] where i=j

    Returns
    -------
    stack_A : 1darray
        (2N,2N)
        float
        Diagonally stacked version of a.
        Centred around N -> [N-i:N+i] is symmetric
    error_stack_A : 1darray
        (2N,2N)
        float
        Error of the mean of the stack_A.

    """
    #
    # Read input & declare variables #
    #

    A = np.asarray(A, dtype=float)
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
    x : 1darray of float
        Measurement.
    d : float
        Width of the decay.

    Returns
    -------
    y : 1darray of float
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
        float
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
        float
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

# 19) Measure slope by pixel. #
def get_slope_pixel(phase, pixel_scale):
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
        (N,N)
        float
        radian
        Contain phase at any given point.
    pixel_scale : float
        Distance between each given element in the array.
        m (preferrably)

    Returns
    -------
    slope : 3darray
        Actually 2darray but contain a pair of vector coordinates.
        (N, N, 2)
        float
        rad/m
        N is kept even though it should be N-1.
        slope[N, : , : ] & slope[ : , N, : ] are zeros.

    Example
    -------
    x = np.arange(-16, 16)
    x = x*(x > 0) * 2.*np.pi/16.*4.
    xx, yy = np.meshgrid(x, x)
    phase = - xx + yy
    phase = np.angle(np.exp(1j*phase))
    slope_phase = get_slope_pixel(phase, 5e-3)
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
    which means peak to valley = 2 pixels or peak to peak = edge to edge = 4 pixels.

    """
    # Inputs
    phase = np.asarray(phase, dtype=float)
    pixel_scale = float(pixel_scale)
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
    # Get slope from difference
    slope = slope/pixel_scale

    return slope

###############################################################################
########################################################################

# 20) Measure slope by lenslet. #
def get_vec_lenslet(vec, lenslet_pixel):
    """
    Measure average slope vector of a scalar field chopped into lenslet.

    Parameters
    ----------
    vec : TYPE
        DESCRIPTION.
    lenslet_pixel : TYPE
        DESCRIPTION.

    Returns
    -------
    lenslet_vec : TYPE
        DESCRIPTION.

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
    val : TYPE
        DESCRIPTION.
    lenslet_pixel : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    val = np.asarray(val)
    lenslet_pixel = int(lenslet_pixel)
    N = len(val)
    num_lenslet = int(N/lenslet_pixel)
    lenslet_val = np.zeros((num_lenslet, num_lenslet), dtype=type(val[0,0]))
    ijlenslet_val = np.zeros((lenslet_pixel, lenslet_pixel), dtype=type(val[0,0]))

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

# 22) Parameters check. #
def prop_param_check(N, d_in, d_out, wvl, z_list, pf):
    """
    Check propagation parameter whether or not it is good.
    
    Must be printed out as True for all categories.

    Parameters
    ----------
    N : TYPE
        DESCRIPTION.
    d_in : TYPE
        DESCRIPTION.
    d_out : TYPE
        DESCRIPTION.
    wvl : TYPE
        DESCRIPTION.
    z_list : 1darray
        (len(z_list))
        DESCRIPTION.
    pf : TYPE
        DESCRIPTION.

    Returns
    -------
    condition: 1darray
        (len(z_list))
        bool
        Pass the test or not.

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
    condition = np.zeros(len(z_list), dtype=bool)
    
    for iz in np.arange(len(z_list)):
        condition1[iz] = ((pf - 1)*N*d_out > 2.*wvl*z_list[iz]/d_in)
        condition2[iz] = (2.*d_out < 2.*np.sqrt(wvl*z_list[iz]/8.))
        condition3[iz] = (N > wvl*z_list[iz]/d_in/d_out)
        condition[iz] = condition1[iz]*condition2[iz]*condition3[iz]
        if condition1[iz] == False:
            print(N, d_in, d_out, wvl, z_list[iz], pf)
            print("Screen is not oversized enough.")
            print("Wanted oversize = {:.0f}.".format((2.*wvl*z_list[iz]/d_in)/(N*d_out)+1))
        if condition2[iz] == False:
            print(N, d_in, d_out, wvl, z_list[iz], pf)
            print("Screen cannot detect scintillation blob.")
            print("Wanted out pixel = {:.2f}".format(np.sqrt(wvl*z_list[iz]/8.)))
        if condition3[iz] == False:
            print(N, d_in, d_out, wvl, z_list[iz], pf)
            print("Something wrong. See condition 4 of Schmidt."
                  + " N>wvl*z/d_in/d_out")
    print(z_list, condition)
    return condition

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

# 24) Propagate pupil plane to focal plane. #
def lensAgainstAng(U_in, wvl, d1, f, d2):
    """
    Propagate wave at pupil to focal plane using angular-spectrum.

    Based on Equation 4.8, Page 59, Chapter 4
    in Schmidt, Jason D., (2010)
    Numerical Simulation of Optical Wave Propagation
    with examples in MATLAB.
    BUT use angular-spectrum propagation instead of 1-step Fresnel.

    Parameters
    ----------
        Uin : 2darray
            (N, N)
            (complex, complex)
            input array of complex amplitude.
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

    Returns
    -------
        Uout : 2darray
            (N,N)
            (complex, complex)
            Complex ampltitude after propagation.

    """
    U_in = np.asarray(U_in, dtype=complex)
    wvl = float(wvl)
    d1 = float(d1)
    f = float(f)
    d2 = float(d2)
    
    U_after_lens = np.zeros_like(U_in, dtype=complex)
    U_out = np.zeros_like(U_in, dtype=complex)
    
    k = 2.*np.pi/wvl
    N = len(U_in)
    x = np.arange(-N/2, N/2)
    x1, y1 = np.meshgrid(x, x) * d_in
    r1sq = x1**2 + y1**2 + 1e-10
    shift = np.ones(N,N) / N
    U_after_lens = U_in * np.exp(-1j*k*r1sq/2./f) * np.exp(1j*np.pi*(shift))
    U_out = np.fft.fftshift(crop2(angularSpectrum(zero_pad2(U_after_lens, 2), wvl, d1, d2, f), 2))

    return U_out

###############################################################################
########################################################################

# 25) Get un-wrap slope. #
def get_slope_pixel_no_wrap(phase, pixel_scale):
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

    Returns
    -------
    slope : 3darray
        Actually 2darray but contain a pair of vector coordinates.
        (N, N, 2)
        float
        rad/m
        N is kept even though it should be N-1.
        slope[N, : , : ] & slope[ : , N, : ] are zeros.

    Example
    -------
    x = np.arange(-16, 16)
    x = x*(x > 0) * 2.*np.pi/16.*4.
    xx, yy = np.meshgrid(x, x)
    phase = - xx + yy
    phase = np.angle(np.exp(1j*phase))
    slope_phase = get_slope_pixel(phase, 5e-3)
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
    which means peak to valley = 2 pixels or peak to peak = edge to edge = 4 pixels.

    """
    # Inputs
    phase = np.asarray(phase, dtype=float)
    pixel_scale = float(pixel_scale)
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
                    print('This screen is too dense.')
                    print(np.abs(slope[i, j, k]))
                    return False
    # Get slope from difference
    slope = slope/pixel_scale

    return slope

###############################################################################
########################################################################

# 26) Propagate pupil image at SHWFS lenslet array to SHWFS detector. #
def propagate_lenslet(U_in, wvl, d_in, f, lenslet_pixel, d_out):
    """
    Propagate SHWFS pupil plane/lenslet array through all lens to focus.

    To make this 1:1 & Nyquist,
        *** 1:1 : d_in = d_out
        *** Nyquist : wvl*f/d_in/lenslet_pixel > 2*d_out.
    d_in is the actual lenslet size
        not the projected lenslet size on the sky.

    Parameters
    ----------
    U_in : 2darray
        (N, N)
        (complex, complex)
        Input complex amplitude.
    wvl : float
        Units is m.
        Propagation wavelength.
    d_in : float
        Unit is m.
        Input pixel scale size.
    f : float
        Unit is m.
        Focal length of the lens.
    lenslet_pixel : float
        Unit is pixel.
        Number of pixel per lenslet.
    d_out : float
        Unit is m.
        Output pixel scale size.

    Returns
    -------
    lenslet_U_out : 2darray
        (N, N)
        (complex, complex)
        Outout complex amplitude.

    """
    U_in = np.asarray(U_in, dtype=complex)
    wvl = float(wvl)
    d_in = float(d_in)
    f = float(f)
    lenslet_pixel = int(lenslet_pixel)
    d_out = float(d_out)

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
            ijlenslet_U_out = lensAgainstAng(ijlenslet_U_in, wvl, d_in, f, d_out)
            lenslet_U_out[i_lenslet*lenslet_pixel
                         : (i_lenslet + 1)*lenslet_pixel,
                         j_lenslet*lenslet_pixel
                         : (j_lenslet + 1)*lenslet_pixel] = ijlenslet_U_out

    return lenslet_U_out

###############################################################################
########################################################################

###############################################################################
## test region #########################################################
###############################################################################

# print('yeah')

#
# Constants / Input
#

# r0 = 5e-2
# scrnHeights = np.array([0,5,10,15])*1e3
# scrnStrengths = np.array([0.5,0.3,0.1,0.1])
# normScrnStrengths = scrnStrengths/scrnStrengths.sum()
# scrnStrengths = (r0**(-5./3) * normScrnStrengths)**(-3./5.)

N = 1024
d_in = 5e-3
L0 = 20.
l0 = 1.5e-10
wvl = 500e-9
d_out = d_in
# z = np.asarray([25000], dtype=float)
# z = np.logspace(np.log(1e3)/np.log(10), np.log(25e3)/np.log(10), num=5)
# z = np.linspace(1e3, 25e3, num=5)
z = np.array([1,5,10,25], dtype=float)*1e3
# z = np.array([25], dtype=float)*1e3
lenslet_pixel = 8
num_lenslet = int(N/lenslet_pixel)

D_lenslet = 150e-6
d_lenslet = D_lenslet/lenslet_pixel
f_lenslet = 2. * d_lenslet**2 * lenslet_pixel / wvl #???

pf = 2
NN = pf*N
prN = 8 # 32
prNN = int(prN/lenslet_pixel)


# prN = 128

#k = 2.*np.pi/wvl                    # float; rad/m; Wave number.
#cn2dz=r**(-5./3.)/0.423/k**2        # float; Turbulence strength.

# r = np.logspace(np.log(0.01)/np.log(10), np.log(0.5)/np.log(10), num=5)
# r = np.linspace(0.01, 0.5, num=5)
r = np.array([0.50, 0.20, 0.10, 0.05, 0.03], dtype=float)
# r = np.array([0.2,0.1,0.05,0.025], dtype=float)
# r = np.array([0.05])
#
# Declaration field
#

amp_in = np.ones((NN,NN), dtype=float)          # Input amplitude.
phase_in = np.zeros((NN, NN), dtype=float)       # Input phase (rad).
U_in = np.zeros((NN,NN), dtype=complex)         # Input complex amp.

U_out = np.zeros((N,N), dtype=complex)          # Output complex amp.
phase_out = np.zeros((N,N), dtype=float)
I_out = np.zeros((N,N), dtype=float)

#
# Initialization field
#

check = True
check_condition = prop_param_check(N, d_in, d_out, wvl, z, pf)
for i in check_condition:
    check = check * i

if check == False:
    print("One of the condition is failing.")
    quit()

rytov = np.array([])
cor = np.array([])

for ir in np.arange(len(r)):
    # Propagation #
    # phase_in = ft_sh_phase_screen(r[ir], NN, d_in, L0, l0)
    slope_phase_in = get_slope_pixel_no_wrap(phase_in, d_in)
    U_in = amp_in*np.exp(1j*phase_in)
    phase_in = np.angle(U_in)
    plt.imshow(phase_in)
    plt.colorbar()
    plt.show()
    if type(slope_phase_in) == bool:
        print('This phase screen is not Nyquist sampled.')
        print('It is too dense.')
        print('r={:.3f}m, d_in={:.3f}m'.format(r, d_in))
        print('Try reduce d_in, or increase r.')
        quit()
    # I_in = np.abs(U_in)**2
    # slope_phase_in = get_slope_pixel(phase_in, d_in)
    for iz in np.arange(len(z)):
        U_out = crop2(angularSpectrum(U_in, wvl, d_in, d_out, z[iz]), pf)
        phase_out = np.angle(U_out)
        I_out = np.abs(U_out)**2

        slope_phase_out = get_slope_pixel(phase_out, d_out) * wvl / (2.*np.pi)
        size_slope_phase_out = np.sqrt(slope_phase_out[ : , : , 0]**2
                                       + slope_phase_out[ : , : , 1]**2)
        weight_slope_phase_out = np.zeros_like(slope_phase_out)
        weight_slope_phase_out[ : , : , 0] = I_out[:,:]*slope_phase_out[ : , : , 0]
        weight_slope_phase_out[ : , : , 1] = I_out[:,:]*slope_phase_out[ : , : , 1]
        norm_var_I_out = (np.mean(I_out**2)/np.mean(I_out)**2) - 1.



        lenslet_U_out = get_val_lenslet(U_out, lenslet_pixel)
        phase_lenslet_out = np.angle(lenslet_U_out)
        I_lenslet_out = np.abs(lenslet_U_out)**2
        slope_phase_lenslet_out = get_slope_pixel(
            phase_lenslet_out, d_out*lenslet_pixel) * wvl / (2.*np.pi)
        size_slope_phase_lenslet_out = np.sqrt(
            slope_phase_lenslet_out[ : , : , 0]**2
            + slope_phase_lenslet_out[ : , : , 1]**2)
        weight_slope_phase_lenslet_out = np.zeros_like(slope_phase_lenslet_out)
        weight_slope_phase_lenslet_out[ : , : , 0] = (
            I_lenslet_out[:,:]*slope_phase_lenslet_out[ : , : , 0])
        weight_slope_phase_lenslet_out[ : , : , 1] = (
            I_lenslet_out[:,:]*slope_phase_lenslet_out[ : , : , 1])
        norm_var_I_lenslet_out = (
            np.mean(I_lenslet_out**2) / np.mean(
                I_lenslet_out)**2) - 1.



        plt.pcolor(np.arange(-prN, prN)*d_out,
                   np.arange(-prN, prN)*d_out,
                   phase_out[int(N/2) - prN : int(N/2) + prN,
                             int(N/2) - prN : int(N/2) + prN],
                   shading='auto')
        plt.axis('equal')
        plt.colorbar()
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Phase - z={:.0f}m, r={:.3f}m'.format(z[iz], r[ir]))
        plt.show()

        plt.quiver(np.arange(-prN, prN)*d_out,
                   np.arange(-prN, prN)*d_out,
                   slope_phase_out[int(N/2) - prN : int(N/2) + prN,
                                   int(N/2) - prN : int(N/2) + prN, 0],
                   slope_phase_out[int(N/2) - prN : int(N/2) + prN,
                                   int(N/2) - prN : int(N/2) + prN, 1])
        plt.title('Slope direction - z={:.0f}m, r={:.3f}m'.format(z[iz],
                                                                  r[ir]))
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()

        plt.pcolor(np.arange(-prN, prN)*d_in,
                   np.arange(-prN, prN)*d_in,
                   size_slope_phase_out[int(N/2) - prN : int(N/2) + prN,
                                        int(N/2) - prN : int(N/2) + prN],
                   shading='auto')
        plt.axis('equal')
        plt.colorbar()
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Phase difference size - ground'
                  + '\nr={:.3f}m, z={:.0f}m'.format(r[ir], z[iz]))
        plt.show()

        plt.pcolor(np.arange(-prN, prN)*d_out,
                   np.arange(-prN, prN)*d_out,
                   I_out[int(N/2) - prN : int(N/2) + prN,
                             int(N/2) - prN : int(N/2) + prN],
                   shading='auto')
        plt.axis('equal')
        plt.colorbar()
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Intensity - z={:.0f}m, r={:.3f}m'.format(z[iz], r[ir]))
        plt.show()

        plt.quiver(np.arange(-prN, prN)*d_out,
                   np.arange(-prN, prN)*d_out,
                   weight_slope_phase_out[int(N/2) - prN : int(N/2) + prN,
                                          int(N/2) - prN : int(N/2) + prN, 0],
                   weight_slope_phase_out[int(N/2) - prN : int(N/2) + prN,
                                          int(N/2) - prN : int(N/2) + prN, 1])
        plt.title('Weighted slope direction'
                  + ' - z={:.0f}m, r={:.3f}m'.format(z[iz], r[ir]))
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()



        plt.pcolor(np.arange(-prNN, prNN)*d_out*lenslet_pixel,
                   np.arange(-prNN, prNN)*d_out*lenslet_pixel,
                   phase_lenslet_out[int(num_lenslet/2) - prNN
                                     : int(num_lenslet/2) + prNN,
                                     int(num_lenslet/2) - prNN
                                     : int(num_lenslet/2) + prNN],
                   shading='auto')
        plt.axis('equal')
        plt.colorbar()
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Lenslet phase - z={:.0f}m, r={:.3f}m'.format(z[iz], r[ir]))
        plt.show()

        plt.quiver(np.arange(-prNN, prNN)*d_out*lenslet_pixel,
                   np.arange(-prNN, prNN)*d_out*lenslet_pixel,
                   slope_phase_lenslet_out[int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN,
                                           int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN, 0],
                   slope_phase_lenslet_out[int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN,
                                           int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN, 1])
        plt.title('Lenslet slope direction'
                  + ' - z={:.0f}m, r={:.3f}m'.format(z[iz], r[ir]))
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()

        plt.pcolor(np.arange(-prNN, prNN)*d_in*lenslet_pixel,
                   np.arange(-prNN, prNN)*d_in*lenslet_pixel,
                   size_slope_phase_lenslet_out[int(num_lenslet/2) - prNN
                                                : int(num_lenslet/2) + prNN,
                                                int(num_lenslet/2) - prNN
                                                : int(num_lenslet/2) + prNN],
                   shading='auto')
        plt.axis('equal')
        plt.colorbar()
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Lenslet phase difference size - ground'
                  + '\nr={:.3f}m, z={:.0f}m'.format(r[ir], z[iz]))
        plt.show()

        plt.pcolor(np.arange(-prNN, prNN)*d_out*lenslet_pixel,
                   np.arange(-prNN, prNN)*d_out*lenslet_pixel,
                   I_lenslet_out[int(num_lenslet/2) - prNN
                                 : int(num_lenslet/2) + prNN,
                                 int(num_lenslet/2) - prNN
                                 : int(num_lenslet/2) + prNN],
                   shading='auto')
        plt.axis('equal')
        plt.colorbar()
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Lenslet intensity - z={:.0f}m, r={:.3f}m'.format(z[iz],
                                                                    r[ir]))
        plt.show()

        plt.quiver(np.arange(-prNN, prNN)*d_out*lenslet_pixel,
                   np.arange(-prNN, prNN)*d_out*lenslet_pixel,
                   weight_slope_phase_lenslet_out[int(num_lenslet/2) - prNN
                                                  : int(num_lenslet/2) + prNN,
                                                  int(num_lenslet/2) - prNN
                                                  : int(num_lenslet/2) + prNN,
                                                  0],
                   weight_slope_phase_lenslet_out[int(num_lenslet/2) - prNN
                                                  : int(num_lenslet/2) + prNN,
                                                  int(num_lenslet/2) - prNN
                                                  : int(num_lenslet/2) + prNN,
                                                  1])
        plt.title('Lenslet weighted slope direction'
                  + ' - z={:.0f}m, r={:.2f}m'.format(z[iz], r[ir]))
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()
        
        U_WFS = propagate_lenslet(U_out, wvl, d_lenslet, f_lenslet, lenslet_pixel, d_lenslet)
        I_WFS = np.abs(U_WFS)**2
        
        plt.pcolor(np.arange(-prN, prN),
                   np.arange(-prN, prN),
                   I_WFS[int(N/2) - prN : int(N/2) + prN,
                         int(N/2) - prN : int(N/2) + prN],
                   shading='auto')
        plt.axis('equal')
        plt.colorbar()
        plt.xlabel('x (pixel)')
        plt.ylabel('y (pixel)')
        plt.title('WFS Intensity - z={:.0f}m, r={:.3f}m'.format(z[iz], r[ir]))
        plt.show()
        
        
        print(r[ir], z[iz], norm_var_I_out, norm_var_I_lenslet_out)

###############################################################################
########################################################################

###############################################################################
# List of other functions which I would like to write but didn't, ######
# because it will be redundant with aotools or soapy. ##################
###############################################################################
# 1) Regular von Karman phase screen
#    aotools.turbulence.phasescreen.ft_phase_screen
# 2) Sub-harmonic von Karman phase screen
#    aotools.turbulence.phasescreen.ft_sh_phase_screen
# 3) 1-steo angular-spectrum Fresnel propagation
###############################################################################
########################################################################

###############################################################################
# End of File ##########################################################
###############################################################################
