
###############################################################################
# Start of file ########################################################
###############################################################################
"""
Library of AO codes on Adaptive Optics (AO): study of turbulence.

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
###############################################################################
########################################################################

#
# Imports #
#
import numpy as np
from aotools.functions import zernike
from aotools.turbulence.phasescreen import ft_sh_phase_screen
from aotools.turbulence.phasescreen import ft_phase_screen
from matplotlib import pyplot as plt
import my_propagation as mprop

#
# Preferences #
#
color_map = plt.cm.get_cmap('viridis')
reversed_color_map = color_map.reversed()

###############################################################################
########################################################################

# 1) Power spectrum density. #
def psd_function(r0, N, delta, L0, l0):
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
    PSD_phi : 2darray
        Numpy array representing power spectrum density (PSD)
        Element type is float.
        Dimension is (N,N)
        Unit of PSD is m^2.
        Pixel size is 1/(N*delta). Unit of pixel size is 1/m.

    Example
    -------
    PSD = psd_function(r0=0.05, N=1024, delta=0.2,
                               L0=20, l0=1.5e-10)
    plt.plot(PSD[int(1024/2),int(1024/2):])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    PSD = psd_function(r0=0.05, N=1024*2, delta=5e-3,
                               L0=20, l0=1.5e-10)
    plt.plot(PSD[int(1024*2/2),int(1024*2/2):])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    """
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

    return PSD_phi

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

    d = 5e-2
    r = 0.05
    Num = 1024
    L = 20
    x, y, y_error = structure_function(ft_sh_phase_screen(
        r0=r, N=Num, delta=d, L0=L, l0=1.5e-10, FFT=None, seed=None),
        random_radius=1024/4, n_bin=100,
        n_random=1000000)
    plt.errorbar(x*d, y, yerr=y_error, label='Measured', color='r')
    plt.plot(x*d, 6.88*(x*d/r)**(5./3.), ls='-',
        label='Kolmogorov',color='g')
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
    
    d = 1e-2
    r = 0.1
    Num = 1024*8
    L = 20
    x, y, y_error = structure_function(ft_sh_phase_screen(
        r0=r, N=Num, delta=d, L0=L, l0=1.5e-10, FFT=None, seed=None),
        random_radius=1024*8/4, n_bin=100,
        n_random=1000000)
    plt.errorbar(x*d, y, yerr=y_error, label='Measured', color='r')
    plt.plot(x*d, 6.88*(x*d/r)**(5./3.), ls='-',
        label='Kolmogorov',color='g')
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
    # screen_size = int(screen.shape[0])                      # int, pixel
    screen_size = screen.shape[0]                      # int, pixel
    
    # Setting defaults value
    if random_radius == None:
        random_radius = int(np.floor(screen_size/4.))
    if n_bin == None:
        n_bin = int(np.floor((screen_size - 2*random_radius) / 10.))
    if n_random == None:
        n_random = int(np.floor(random_radius**2 / 10.))
    
    # random_radius = int(random_radius)                      # int, pixel
    # n_bin = int(n_bin)
    # n_random = int(n_random)
    
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
    belong_to_group = np.zeros(n_random, dtype=int)                                     # int
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
    points_a = np.random.randint(random_radius,
                                 (screen_size - 1) - random_radius + 1,
                                 size=(n_random, 2))

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
    points_b_polar[:,0] = (random_radius
                           - np.random.rand(n_random)*(random_radius - 10))
    points_b_polar[:,1] = np.random.rand(n_random) * (2*np.pi)

    #
    # 3) Calculate x&y of the second half of the pair #
    #
    # x = x0 + r*cos(angle)
    points_b[:,0] = np.asarray(np.fix(
        points_b_polar[:,0]*np.cos(points_b_polar[:,1]) + points_a[:,0]),
        dtype=int)
    # y = y0 + r*sin(angle)
    points_b[:,1] = np.asarray(np.fix(
        points_b_polar[:,0]*np.sin(points_b_polar[:,1]) + points_a[:,1]),
        dtype=int)

    #
    # 4) Calculate difference in value and separation #
    #
    difference = np.abs(screen[points_a[:,0], points_a[:,1]]
                           - screen[points_b[:,0], points_b[:,1]])
    separation = np.hypot(points_a[:,0] - points_b[:,0],
                          points_a[:,1] - points_b[:,1])

    #
    # 5) Generate bins of separation in log_10 space #
    #
    # lower edge of lowest bin
    log_minimum_separation = np.log10(separation.min())
    # higher edge of highest bin
    # need 1.1* to off-set the actual max to be less than the edge
    log_maximum_separation = np.log10(1.1*separation.max())
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
    # group# = keep integer of ((val-min)/space)
    belong_to_group = np.asarray(((np.log10(separation) - log_minimum_separation)
                       // separation_space_log), dtype=int)
    for i in np.arange(n_random):
        binned_difference[belong_to_group[i]].append(difference[i])

    #
    # 7) Calculate mean of (difference) squared of each bin #
    #    and       error of mean of each bin #
    for i in np.arange(n_bin):
        # For successful standard deviation calculation,
        # there must be more than 1 point
        temp = np.asarray(binned_difference[i])
        if temp.shape[0] > 1:
            mean_difference_square[i] = np.mean(temp**2)
            error_of_mean[i] = np.std(temp, ddof=1)**2
        else:
            mean_difference_square[i] = np.nan
            error_of_mean[i] = np.nan

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
        Dimension is (n,i); where n is number of functions
            to be orthonormalized, and i is shape of each functions.
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
    n_modes = 10
    N = 16
    zernike_modes = zernike.zernikeArray(n_modes,N)
    gram_schmidt_zernike = gram_schmidt_process(zernike_modes)
    for i in np.arange(n_modes):
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

    n = f_in.shape[0]                               # number of modes
    shape_in = f_in.shape
    if f_in.ndim == 3:
        f_in = np.reshape(f_in,(f_in.shape[0],f_in.shape[1]*f_in.shape[2]))
    f_out = np.zeros_like(f_in)
    
    # now in wiki's notation
    # v = f_in                                     # not orthogonal, not normal
    u = f_in                                       # is orthogonal, not normal
    # orthogonalized mode is initially the original modes
    e = np.zeros_like(f_in)                        # is orthogonal, is normal

    #
    # 1) orthogonalize #
    #
    # index is preserved
    e[0] = u[0]/np.sqrt(np.dot(u[0],u[0]))
    for i in np.arange(n - 1) + 1:
        for j in np.arange(i):
            # then take out not orthogonal part compared to prior ones
            u[i] = u[i] - (np.dot(u[j], f_in[i]) / np.dot(u[j], u[j])) * u[j]

        #
        # 2) normalize #
        #
        # divide by its size
        e[i]=u[i]/np.sqrt(np.dot(u[i], u[i]))

    #
    # 3) reshape then return #
    #
    f_out = e.reshape(shape_in)
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
    testz = orthonormality_test(zernike.zernikeArray(55,128))
    testz = np.abs(testz)
    testz = testz*(testz>0) + 1e-17*(testz<=0)
    plt.imshow(np.log(testz)/np.log(10))
    plt.xlabel('i (mode)')
    plt.ylabel('j (mode)')
    plt.title('dot of 2 modes in log_10 with 0 replaced with 1e-17')
    plt.colorbar()
    plt.show()
    testz = orthonormality_test(gram_schmidt_process(
        zernike.zernikeArray(55,128)))
    testz = np.abs(testz)
    testz = testz*(testz>0) + 1e-17*(testz<=0)
    plt.imshow(np.log(testz)/np.log(10))
    plt.xlabel('i (mode)')
    plt.ylabel('j (mode)')
    plt.title('dot of 2 modes in log_10 with 0 replaced with 1e-17')
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
                           ovf=1, sh=True, gram_schmidt=True, FFT=None):
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
        Number of zernike modes to measure. Non-piston
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
    import time
    ta = time.time()
    n, noll, var, err_of_var = zernike_modes_strength(
        0.10, 128, 1e-3, 20, 1.5e-10, 32, 100,
        ovf=16, sh=True, gram_schmidt=True, FFT=None)
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
    tb = time.time()
    print(tb-ta)

    """
    #
    # Declaration and initialization field #
    #
    flat_zernike_modes = np.reshape((zernike.zernikeArray(n_modes + 1, N)
                                     / np.sqrt((np.pi * N**2 / 4.))),
                                    (n_modes + 1, int(N*N)))
    if gram_schmidt == True:
        flat_gram_schmidt_modes = np.zeros_like(flat_zernike_modes)
        gram_schmidt_to_zernike = np.zeros((n_modes + 1, n_modes + 1))
        gram_schmidt_coefficients = np.zeros((n_random, n_modes + 1))
    phase_screen = np.zeros((N,N))
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
    if gram_schmidt ==True:
        flat_gram_schmidt_modes = gram_schmidt_process(flat_zernike_modes)

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
    if gram_schmidt == True:
        for i in np.arange(n_modes + 1):
            for j in np.arange(n_modes + 1):
                gram_schmidt_to_zernike[i,j] = np.dot(flat_gram_schmidt_modes[i],
                                                      flat_zernike_modes[j])
        gram_schmidt_to_zernike = np.linalg.inv(gram_schmidt_to_zernike)

    #
    # 4) Generate n_random phase screens and get gram schmidt coefficients #
    #
    for i in np.arange(n_random):
        # Generate a phase screen
        if sh == True:
            phase_screen = mprop.crop2(ft_sh_phase_screen(r0, ovf*N, delta, L0,
                                                          l0, FFT=FFT), ovf)
        else:
            phase_screen = mprop.crop2(ft_phase_screen(r0, ovf*N, delta, L0,
                                                       l0, FFT=FFT), ovf)
        if gram_schmidt == True:
            gram_schmidt_coefficients[i] = np.dot(flat_gram_schmidt_modes,
                                                    phase_screen.flatten())
            #
            # 5) Convert gram_schmidt coefficients to zernike coefficients #
            #
            zernike_coefficients[i] = np.matmul(gram_schmidt_to_zernike,
                                                gram_schmidt_coefficients[i])
        else:
            zernike_coefficients[i] = np.dot(flat_zernike_modes,
                                               phase_screen.flatten())

    #
    # 6) Measure variance and variance error #
    #
    coefficients_variance = (np.var(zernike_coefficients, ddof=1, axis=-2)
                             / (np.pi * N**2 / 4.))
    # variance = error**2
    # variance of variance = (error of variance)**2
    #                      = (error of error)**2 (2*error)**2
    # error of variance = error of error * 2 * error
    #                   = 2 * error * error/sqrt(2n-2)
    #                   = 2 * variance/sqrt(2n-2)
    coefficients_variance_error = 2*(coefficients_variance
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

def get_PSD(phase, delta):
    """
    Calculate PSD of a phase screen.

    Parameters
    ----------
    phase : numpy.ndarray
            (N,N)
            Phase screen to be measured.
    delta : float
            Pixel scale.

    Returns
    -------
    PSD : numpy.ndarray
          (N,N)
          Power Spectral Density.

    Example
    -------
    delta = 5e-3
    r0 = 0.1
    N = 1024
    L0 = 20
    l0 = 1e-10
    f = (np.arange(N/2-1)+1)/N/delta
    phase = ft_phase_screen(r0, N, delta, L0, l0)
    PSD_measure = get_PSD(phase,delta)[int(N/2),int(N/2)+1:]
    PSD_theory = psd_function(r0, N, delta, L0, l0)[int(N/2),int(N/2)+1:]

    plt.plot(f, PSD_theory, label='Theory')
    plt.plot(f, PSD_measure, label='Measured',
             marker='.',ls='')
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
    N = len(phase)
    del_f = 1./(N*2.*delta)
    phase = phase - phase.mean()
    cn = my_ft2(mirror_pad2(phase), 1./N/2.)
    PSD = (np.abs(cn)**2/del_f**2)*2.   # * 2. is from the fact that
                                        # the input phase is not complex
                                        # phase are complex when generated
                                        # from cn.
    PSD[int(N/2),int(N/2)] = 0.         # PSD is 0 at the centre.
    return pick_even_index(PSD)

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
    # G = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(g))) * delta**2
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(g))) * delta**2

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
    # A = AA[tuple(np.meshgrid(np.arange(0,AA.shape[0],2),
    #                          np.arange(0,AA.shape[1],2)))]
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
    # AA = np.pad(A,(int(A.shape[0]/2),int(A.shape[1]/2)),'symmetric')
    # ROW = A.shape[0]
    # COL = A.shape[1]
    # row = int(ROW/2.)
    # col = int(COL/2.)
    # AA = np.zeros((ROW*2,COL*2), dtype=type(A[0,0]))

    # AA[tuple(np.meshgrid(np.arange(row,row+ROW,1),
    #                      np.arange(col,col+COL,1)))] = A[
    #                          tuple(np.meshgrid(np.arange(0,ROW,1),
    #                                            np.arange(0,COL,1)))]

    # AA[tuple(np.meshgrid(np.arange(row-1,-1,-1),
    #                      np.arange(col-1,-1,-1)))] = A[
    #                          tuple(np.meshgrid(np.arange(0,row,1),
    #                                            np.arange(0,col,1)))]
    # AA[tuple(np.meshgrid(np.arange(ROW+row,ROW+2*row,1),
    #                      np.arange(COL+col,COL+2*col,1)))] = A[
    #                          tuple(np.meshgrid(np.arange(ROW-1,ROW-1-row,-1),
    #                                            np.arange(COL-1,COL-1-col,-1)))]
    # AA[tuple(np.meshgrid(np.arange(row-1,-1,-1),
    #                      np.arange(COL+col,COL+2*col,1)))] = A[
    #                          tuple(np.meshgrid(np.arange(0,row,1),
    #                                            np.arange(COL-1,COL-1-col,-1)))]
    # AA[tuple(np.meshgrid(np.arange(ROW+row,ROW+2*row,1),
    #                      np.arange(col-1,-1,-1)))] = A[
    #                          tuple(np.meshgrid(np.arange(ROW-1,ROW-1-row,-1),
    #                                            np.arange(0,col,1)))]

    # AA[tuple(np.meshgrid(np.arange(row,ROW+row,1),
    #                      np.arange(col-1,-1,-1)))] = A[
    #                          tuple(np.meshgrid(np.arange(0,ROW,1),
    #                                            np.arange(0,col,1)))]
    # AA[tuple(np.meshgrid(np.arange(row,ROW+row,1),
    #                      np.arange(COL+col,COL+2*col,1)))] = A[
    #                          tuple(np.meshgrid(np.arange(0,ROW,1),
    #                                            np.arange(COL-1,COL-1-col,-1)))]

    # AA[tuple(np.meshgrid(np.arange(row-1,-1,-1),
    #                      np.arange(col,COL+col,1)))] = A[
    #                          tuple(np.meshgrid(np.arange(0,row,1),
    #                                            np.arange(0,COL,1)))]
    # AA[tuple(np.meshgrid(np.arange(row+ROW,ROW+2*row,1),
    #                      np.arange(col,COL+col,1)))] = A[
    #                          tuple(np.meshgrid(np.arange(ROW-1,ROW-1-row,-1),
    #                                            np.arange(0,COL,1)))]
    # return AA
    return np.pad(A,(int(A.shape[0]/2),int(A.shape[1]/2)),'symmetric')

def auto_pad2(phase):
    """
    Pad with itself.

    Parameters
    ----------
    phase : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    Example
    -------
    delta = 5e-3
    r0 = 0.1
    N = 1024
    L0 = 20
    l0 = 1e-10
    phase = ft_sh_phase_screen(r0, N, delta, L0, l0)
    plt.imshow(auto_pad2(phase))
    plt.title('auto_pad of ft_sh_phase_screen')
    plt.show()
    phase = ft_phase_screen(r0, N, delta, L0, l0)
    plt.imshow(auto_pad2(phase))
    plt.title('auto_pad of ft_phase_screen')
    plt.show()
    phase = mprop.crop2(ft_sh_phase_screen(r0, N*2, delta, L0, l0),2)
    plt.imshow(auto_pad2(phase))
    plt.title('auto_pad of 2x oversize ft_sh_phase_screen')
    plt.show()
    phase = mprop.crop2(ft_phase_screen(r0, N*2, delta, L0, l0),2)
    plt.imshow(auto_pad2(phase))
    plt.title('auto_pad of 2x oversize ft_phase_screen')
    plt.show()

    """
    return np.pad(phase,(int(phase.shape[0]/2),int(phase.shape[1]/2)),'wrap')

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
###############################################################################
########################################################################

###############################################################################
# End of File ##########################################################
###############################################################################
