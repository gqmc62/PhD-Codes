###############################################################################
# Puttiwat Kongkaew 2024 ###############################################
###############################################################################

"""
Zernike mode decompositions with Gram-Schmidt process
to reduce aliasing in low sampling phase screeens

get_ZernikeCoeff() is the best function to use,
but it is badly documented on how to use it properly.

So I made another function as an example on how to use the first function.
Use zernike_mode_decomposition() as the example.
"""

###############################################################################
########################################################################
###############################################################################

import numpy as np
from aotools.functions import zernike
from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov
from matplotlib import pyplot as plt
from tqdm import tqdm

###############################################################################
########################################################################
###############################################################################


def get_ZernikeCoeff(FlatPupilPhase, FlatGramSchmidt,
        GramSchmidtCoeff2ZernikeCoeff):

    GramSchmidtCoeff = np.dot(FlatGramSchmidt, FlatPupilPhase)
    ZernikeCoeff = np.matmul(GramSchmidtCoeff2ZernikeCoeff,
                            GramSchmidtCoeff)

    return ZernikeCoeff

###############################################################################
########################################################################
###############################################################################


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

    n = f_in.shape[0]                          # number of modes
    shape_in = f_in.shape
    if f_in.ndim == 3:
        f_in = np.reshape(f_in,(f_in.shape[0],f_in.shape[1]*f_in.shape[2]))
    f_out = np.zeros_like(f_in)
    
    # now in wiki's notation
    # v = f_in                                 # not orthogonal, not normal
    u = f_in                                   # is orthogonal, not normal
    # orthogonalized mode is initially the original modes
    e = np.zeros_like(f_in)                    # is orthogonal, is normal

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
###############################################################################


def get_GramSchmidtCoeff2ZernikeCoeff(FlatZernike,FlatGramSchmidt):

    n_modes = FlatZernike.shape[0]
    GramSchmidtCoeff2ZernikeCoeff = np.zeros((n_modes, n_modes), dtype=float)

    for i in np.arange(n_modes):
        for j in np.arange(n_modes):
            GramSchmidtCoeff2ZernikeCoeff[i,j] = np.dot(FlatGramSchmidt[i],
                                                  FlatZernike[j])
    GramSchmidtCoeff2ZernikeCoeff = np.linalg.inv(
        GramSchmidtCoeff2ZernikeCoeff)

    return GramSchmidtCoeff2ZernikeCoeff

###############################################################################
########################################################################
###############################################################################


def zernike_mode_decomposition(phase, n_modes):
    """
    One stop service to do zernike_mode decomposition
    applying gram-schmidt process.
    
    This is an example of how to use 

    Parameters
    ----------
    phase : numpy.ndarray
            (N,N)
            (numpy.float,numpy.float)
            (radians,radians)
            Phase screen to be decomposed.
            Must be unwrappable,
            if not the decompostion will be in accurate.
    n_modes : numpy.int
              Number of Zernike modes to decompose, excluding piston.
              0 ==> Piston
              1 ==> Tip
              2 ==> Tilt
              Piston,Tip,Tilt = 2 n_modes.
    
    Returns
    -------
    ZernikeCoeff : numpy.ndarray
                   (n_modes)
                   (numpy.float)
                   (radians)
                   Decomposed Zernike coefficients
                   of a given phase screen in radians
    
    """
    N = phase.shape[0]
    
    Zernike = zernike.zernikeArray(n_modes, N)
    GramSchmidt = gram_schmidt_process(Zernike)
    PUPIL = Zernike[0]
    FlatZernike = Zernike.reshape((n_modes, N*N))
    FlatGramSchmidt = GramSchmidt.reshape((n_modes, N*N))
    GramSchmidtCoeff2ZernikeCoeff = get_GramSchmidtCoeff2ZernikeCoeff(
        FlatZernike, FlatGramSchmidt)

    ZernikeCoeff = get_ZernikeCoeff(
        (phase*PUPIL).flatten(),
        FlatGramSchmidt,
        GramSchmidtCoeff2ZernikeCoeff)

    return ZernikeCoeff

###############################################################################
########################################################################
###############################################################################


def test_get_ZernikeCoeff(N,D,r0,L0,RANDOM):
    """
    Test if the decomposition analyses phase screen statistics correctly.

    Assuming that the phase screen generation is correct. lol.
    
    Parameters
    ----------
    N : numpy.int
        number of pixel across the pupil.
    D : numpy.float
        metres
        Pupil size.
    r0 : numpy.float
         metres
         Fried Parameter of turbulence
    L0 : numpy.float
         metres
         Outer scale of turbulence according to von Karman Statistics
    RANDOM : int
        number of samples
        
    Returns
    -------
    plots showing Zernike mode decomposition of phase screen turbulence
    in von Karman statistics.
    
    Based on Noll (1975?) Zernike polynomials and atmospheric turbulence
    Table 4: (derived the zernike mode power/coefficient from
              difference of coefficient of table 4)
    
    The tip/tilt adjustment is according to
    Sasiela (2007) Electromagnetic wave propagation in turbulence
                   : evaluation and application of Mellin transforms
    2nd edition, Chapter 6, Page 148-149, Fig 6.2, Equation 6.8.
    
    Example
    -------
    test_get_ZernikeCoeff(40,4,0.1,20,10000)
    
    """

    n_modes = 20 + 1

    pixel_scale = D/N

    Zernike = zernike.zernikeArray(n_modes, N)
    GramSchmidt = gram_schmidt_process(Zernike)
    PUPIL = Zernike[0]
    FlatZernike = Zernike.reshape((n_modes, N*N))
    FlatGramSchmidt = GramSchmidt.reshape((n_modes, N*N))
    GramSchmidtCoeff2ZernikeCoeff = get_GramSchmidtCoeff2ZernikeCoeff(
        FlatZernike, FlatGramSchmidt)

    PhaseScreen = PhaseScreenKolmogorov(N,pixel_scale,r0,L0)

    ZernikeCoeff = np.zeros((RANDOM, n_modes), dtype=float)
    
    for random in tqdm(np.arange(RANDOM)):

        for row in np.arange(N):
            PhaseScreen.add_row()

        ZernikeCoeff[random] = get_ZernikeCoeff(
            (PhaseScreen.scrn*PUPIL).flatten(),
            FlatGramSchmidt,
            GramSchmidtCoeff2ZernikeCoeff)
        
    variance_ZernikeCoeff = np.var(ZernikeCoeff, ddof=1, axis=0)[1:]
    error_variance_ZernikeCoeff = 2*variance_ZernikeCoeff/np.sqrt(2*N - 2)
    n_modes = variance_ZernikeCoeff.shape[0]
    modes = np.arange(n_modes) + 1
    theoretical_variance_ZernikeCoeff = noll_variance(n_modes,N,pixel_scale,r0)
    e = N*pixel_scale/L0
    adjust = (1 - 1.42*e**(1./3.) + 3.70*e**2 - 4.01*e**(7./3.)
              + 4.21*e**4 - 4.00*e**(13./3.))
    L0TipTiltVariance = theoretical_variance_ZernikeCoeff[:2] * adjust
    
    plt.plot(modes, theoretical_variance_ZernikeCoeff,
             label='Kolmogorov',marker='',ls='-')
    plt.errorbar(modes, variance_ZernikeCoeff, yerr=error_variance_ZernikeCoeff,
                 label='measured', marker='o', ls='', capsize=5)
    plt.plot(modes[:2], L0TipTiltVariance, label='von Karman tip/tilt variance')
    theoretical_variance_ZernikeCoeff[:2] = (
        theoretical_variance_ZernikeCoeff[:2] * adjust)
    plt.errorbar(modes, np.abs(
        variance_ZernikeCoeff - theoretical_variance_ZernikeCoeff),
                 yerr=error_variance_ZernikeCoeff,
                 label='abs difference', marker='o', ls='', capsize=5)
    plt.yscale('log')
    plt.xlabel('zernike modes')
    plt.ylabel('variance of zernike coefficient')
    plt.legend()
    plt.ylim(np.nanmin(theoretical_variance_ZernikeCoeff)/10, np.nanmax(theoretical_variance_ZernikeCoeff)*10)
    plt.title('{:d}'.format(RANDOM)
            + 'ZernikeVariances'
            + 'D{:.0f}cm'.format(D*100)
            + 'r0{:.0f}cm'.format(r0*100)
            + 'L0{:.2f}m'.format(L0))
    plt.savefig('Zernike_coeff.png',bbox_inches='tight',dpi=300)
    plt.show()

###############################################################################
########################################################################
###############################################################################

def noll_variance(n_modes,N,delta,r0):
    A = np.array([0.4480, 0.4480, 0.0230, 0.0230, 0.0230,
                  0.0062, 0.0062, 0.0062, 0.0062, 0.0025,
                  0.0025, 0.0025, 0.0025, 0.0025, 0.0012,
                  0.0012, 0.0012, 0.0012, 0.0012, 0.0012])
    if n_modes <= 20:
        theory = A[0 : n_modes]
    if n_modes > 20:
        excess = np.arange(20, n_modes) + 1
        B = 0.2944 * ((excess)**(-np.sqrt(3)/2.) 
                      - (excess + 1)**(-np.sqrt(3)/2.))
        theory = np.append(A,B)
    theory = theory * (N*delta/r0)**(5./3.)
    return theory


test_get_ZernikeCoeff(40,4,0.1,20,10000)