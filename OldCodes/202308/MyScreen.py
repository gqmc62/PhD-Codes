"""
My faster phase screen.

This is faster than infinite screen, only for number of screen less than 7.
so let us use the infinite phasescreen.
"""

import random
import numpy as np
import time

import pyfftw
# from matplotlib import pyplot as plt

from numpy import fft
from scipy import linalg
from scipy.special import gamma, kv
import my_propagation as mprop
# import numpy
import numba
# from aotools.turbulence.phasescreen import ft_phase_screen, ft_sh_phase_screen

# from . import phasescreen, turb

class MyPhaseScreenSH():
    """Main Class."""

    def __init__(self, r0, n, oversize, delta, L0, l0, seed=None):
        self.r0 = float(r0)
        self.N = int(n*oversize)
        self.delta = float(delta)
        self.L0 = float(L0)
        self.l0 = float(l0)
        R = random.SystemRandom(time.time())
        if seed is None:
            seed = int(R.random()*100000)
        np.random.seed(seed)

        self.scrn = np.zeros((n, n), dtype=float)
        self.edgelow = int((self.N - n)/2)
        self.edgehigh = int((self.N + n)/2)

        self.D = self.N*self.delta
        self.F_in = pyfftw.empty_aligned((self.N, self.N), dtype=complex)
        self.F_out = pyfftw.empty_aligned((self.N, self.N), dtype=complex)
        self.FF_BACKWARD = pyfftw.FFTW(self.F_in, self.F_out, axes=(-1, -2), direction='FFTW_BACKWARD')

        self.del_f_hi = 1./(self.N*self.delta)
        fx = np.arange(-self.N/2., self.N/2.) * self.del_f_hi
        (fx, fy) = np.meshgrid(fx, fx)
        f = np.sqrt(fx**2. + fy**2.)
        self.fm = 5.92/l0/(2*np.pi)
        self.f0 = 1./L0
        self.PSD_phi_hi = (0.023*self.r0**(-5./3.) * np.exp(-1*((f/self.fm)**2)) / (((f**2) + (self.f0**2))**(11./6)))
        self.PSD_phi_hi[int(self.N/2), int(self.N/2)] = 0
        self.phs_hi = np.zeros((self.N, self.N), dtype=float)
        self.cn_hi = np.zeros((self.N, self.N), dtype=complex)
        

        coords = np.arange(-self.N/2, self.N/2)*self.delta
        self.x, self.y = np.meshgrid(coords, coords)
        self.fx_lo = np.arange(-1,2)
        self.fx_lo, self.fy_lo = np.meshgrid(self.fx_lo, self.fx_lo)
        self.f_lo = np.sqrt(self.fx_lo**2 +  self.fy_lo**2) # polar grid
        self.cn_lo = np.zeros((self.N, self.N), dtype=complex)
        self.phs_lo = np.zeros((self.N, self.N), dtype=complex)
        self.PSD_phi_lo = np.zeros((3, 3), dtype=complex)
        self.SH = np.zeros((self.N, self.N), dtype=complex)

        self.cn_hi = ((np.random.normal(size=(self.N, self.N))
               + 1j*np.random.normal(size=(self.N, self.N)))
              * np.sqrt(self.PSD_phi_hi)*self.del_f_hi)
        self.phs_hi = np.fft.fftshift(self.FF_BACKWARD(np.fft.fftshift(self.cn_hi)).real) * (self.N)**2

        for p in range(1,4):
            # setup the PSD
            del_f_lo = 1 / (3**p*self.D) #frequency grid spacing [1/m]

            self.PSD_phi_lo = (0.023*self.r0**(-5./3)
                        * np.exp(-1*(self.f_lo*del_f_lo/self.fm)**2) / ((self.f_lo**2*del_f_lo**2 + self.f0**2)**(11./6)))
            self.PSD_phi_lo[1,1] = 0

            # random draws of Fourier coefficients
            self.cn_lo = ( (np.random.normal(size=(3,3))
                + 1j*np.random.normal(size=(3,3)) )
                            * np.sqrt(self.PSD_phi_lo)*del_f_lo)
            self.SH = np.zeros((self.N,self.N),dtype="complex")
            # loop over frequencies on this grid
            for i in range(0, 3):
                for j in range(0, 3):
                    self.SH += self.cn_lo[i,j] * np.exp(1j*2*np.pi*(self.fx_lo[i,j]*self.x*del_f_lo + self.fy_lo[i,j]*self.y*del_f_lo))
            self.phs_lo = self.phs_lo + self.SH

        self.scrn = (self.phs_lo.real - self.phs_lo.real.mean() + self.phs_hi)[self.edgelow:self.edgehigh,self.edgelow:self.edgehigh]

    def new_screen(self):
        """Generate new phase screen with the same N."""
        R = random.SystemRandom(time.time())
        seed = int(R.random()*100000)
        np.random.seed(seed)
        self.phs_lo = np.zeros((self.N, self.N), dtype=complex)

        self.cn_hi = ((np.random.normal(size=(self.N, self.N)) + 1j*np.random.normal(size=(self.N, self.N)))*np.sqrt(self.PSD_phi_hi)*self.del_f_hi)
        self.phs_hi = np.fft.fftshift(self.FF_BACKWARD(np.fft.fftshift(self.cn_hi)).real) * (self.N)**2

        for p in range(1,4):
            # setup the PSD
            del_f_lo = 1 / (3**p*self.D) #frequency grid spacing [1/m]

            self.PSD_phi_lo = (0.023*self.r0**(-5./3)
                        * np.exp(-1*(self.f_lo*del_f_lo/self.fm)**2) / ((self.f_lo**2*del_f_lo**2 + self.f0**2)**(11./6)))
            self.PSD_phi_lo[1,1] = 0

            # random draws of Fourier coefficients
            self.cn_lo = ( (np.random.normal(size=(3,3))
                + 1j*np.random.normal(size=(3,3)) )
                            * np.sqrt(self.PSD_phi_lo)*del_f_lo)
            self.SH = np.zeros((self.N, self.N), dtype="complex")
            # loop over frequencies on this grid
            for i in range(0, 3):
                for j in range(0, 3):
                    self.SH += self.cn_lo[i,j] * np.exp(1j*2*np.pi*(self.fx_lo[i,j]*self.x*del_f_lo + self.fy_lo[i,j]*self.y*del_f_lo))
            self.phs_lo = self.phs_lo + self.SH

        self.scrn = (self.phs_lo.real - self.phs_lo.real.mean() + self.phs_hi)[self.edgelow:self.edgehigh,self.edgelow:self.edgehigh]

class MyPhaseScreenNOSH():
    """Main Class."""

    def __init__(self, r0, n, oversize, delta, L0, l0, seed=None):
        self.r0 = float(r0)
        self.N = int(n*oversize)
        self.delta = float(delta)
        self.L0 = float(L0)
        self.l0 = float(l0)
        R = random.SystemRandom(time.time())
        if seed is None:
            seed = int(R.random()*100000)
        np.random.seed(seed)

        self.scrn = np.zeros((n, n), dtype=float)
        self.edgelow = int((self.N - n)/2)
        self.edgehigh = int((self.N + n)/2)

        self.D = self.N*self.delta
        self.F_in = pyfftw.empty_aligned((self.N, self.N), dtype=complex)
        self.F_out = pyfftw.empty_aligned((self.N, self.N), dtype=complex)
        self.FF_BACKWARD = pyfftw.FFTW(self.F_in, self.F_out, axes=(-1, -2), direction='FFTW_BACKWARD')

        self.del_f_hi = 1./(self.N*self.delta)
        fx = np.arange(-self.N/2., self.N/2.) * self.del_f_hi
        (fx, fy) = np.meshgrid(fx, fx)
        f = np.sqrt(fx**2. + fy**2.)
        self.fm = 5.92/l0/(2*np.pi)
        self.f0 = 1./L0
        self.PSD_phi_hi = (0.023*self.r0**(-5./3.) * np.exp(-1*((f/self.fm)**2)) / (((f**2) + (self.f0**2))**(11./6)))
        self.PSD_phi_hi[int(self.N/2), int(self.N/2)] = 0
        self.cn_hi = np.zeros((self.N, self.N), dtype=complex)

        self.cn_hi = ((np.random.normal(size=(self.N, self.N))
               + 1j*np.random.normal(size=(self.N, self.N)))
              * np.sqrt(self.PSD_phi_hi)*self.del_f_hi)
        self.scrn = (np.fft.fftshift(self.FF_BACKWARD(np.fft.fftshift(self.cn_hi)).real) * (self.N)**2)[self.edgelow:self.edgehigh,self.edgelow:self.edgehigh]

    def new_screen(self):
        """Generate new phase screen with the same N."""
        R = random.SystemRandom(time.time())
        seed = int(R.random()*100000)
        np.random.seed(seed)
        self.phs_lo = np.zeros((self.N, self.N), dtype=complex)

        self.cn_hi = ((np.random.normal(size=(self.N, self.N)) + 1j*np.random.normal(size=(self.N, self.N)))*np.sqrt(self.PSD_phi_hi)*self.del_f_hi)
        self.scrn = (np.fft.fftshift(self.FF_BACKWARD(np.fft.fftshift(self.cn_hi)).real) * (self.N)**2)[self.edgelow:self.edgehigh,self.edgelow:self.edgehigh]


#
#
# From now on everything is copy and paste from aotools
#
#



# __all__ = ["phase_covariance"]

def phase_covariance(r, r0, L0):
    
    # Make sure everything is a float to avoid nasty surprises in division!
    r = np.float32(r)
    r0 = float(r0)
    L0 = float(L0)

    # Get rid of any zeros
    r += 1e-40

    A = (L0 / r0) ** (5. / 3)

    B1 = (2 ** (-5. / 6)) * gamma(11. / 6) / (np.pi ** (8. / 3))
    B2 = ((24. / 5) * gamma(6. / 5)) ** (5. / 6)

    C = (((2 * np.pi * r) / L0) ** (5. / 6)) * kv(5. / 6, (2 * np.pi * r) / L0)

    cov = A * B1 * B2 * C

    return cov

def ift2(G, delta_f, FFT=None):
    """
    Wrapper for inverse fourier transform

    Parameters:
        G: data to transform
        delta_f: pixel seperation
        FFT (FFT object, optional): An accelerated FFT object
    """

    N = G.shape[0]

    if FFT:
        g = np.fft.fftshift(FFT(np.fft.fftshift(G))) * (N * delta_f) ** 2
    else:
        g = fft.ifftshift(fft.ifft2(fft.fftshift(G))) * (N * delta_f) ** 2

    return g

def ft_sh_phase_screen(r0, N, delta, L0, l0, FFT=None, seed=None):

    """
    Creates a random phase screen with Von Karmen statistics with added
    sub-harmonics to augment tip-tilt modes.
    (Schmidt 2010)

    .. note::
        The phase screen is returned as a 2d array, with each element representing the phase 
        change in **radians**. This means that to obtain the physical phase distortion in nanometres, 
        it must be multiplied by (wavelength / (2*pi)), (where `wavellength` here is the same wavelength
        in which r0 is given in the function arguments)

    Args:
        r0 (float): r0 parameter of scrn in metres
        N (int): Size of phase scrn in pxls
        delta (float): size in Metres of each pxl
        L0 (float): Size of outer-scale in metres
        l0 (float): inner scale in metres
        seed (int, optional): seed for random number generator. If provided, 
            allows for deterministic screens  

    Returns:
        ndarray: numpy array representing phase screen in radians
    """
    R = np.random.default_rng(seed)

    D = N * delta
    # high-frequency screen from FFT method
    phs_hi = ft_phase_screen(r0, N, delta, L0, l0, FFT, seed=seed)

    # spatial grid [m]
    coords = np.arange(-N/2,N/2)*delta
    x, y = np.meshgrid(coords,coords)

    # initialize low-freq screen
    phs_lo = np.zeros(phs_hi.shape)

    # loop over frequency grids with spacing 1/(3^p*L)
    for p in range(1,4):
        # setup the PSD
        del_f = 1 / (3**p*D) #frequency grid spacing [1/m]
        fx = np.arange(-1,2) * del_f

        # frequency grid [1/m]
        fx, fy = np.meshgrid(fx,fx)
        f = np.sqrt(fx**2 +  fy**2) # polar grid

        fm = 5.92/l0/(2*np.pi) # inner scale frequency [1/m]
        f0 = 1./L0

        # outer scale frequency [1/m]
        # modified von Karman atmospheric phase PSD
        PSD_phi = (0.023*r0**(-5./3)
                    * np.exp(-1*(f/fm)**2) / ((f**2 + f0**2)**(11./6)) )
        PSD_phi[1,1] = 0

        # random draws of Fourier coefficients
        cn = ( (R.normal(size=(3,3))
            + 1j*R.normal(size=(3,3)) )
                        * np.sqrt(PSD_phi)*del_f )
        SH = np.zeros((N,N),dtype="complex")
        # loop over frequencies on this grid
        for i in range(0, 3):
            for j in range(0, 3):

                SH += cn[i,j] * np.exp(1j*2*np.pi*(fx[i,j]*x+fy[i,j]*y))

        phs_lo = phs_lo + SH
        # accumulate subharmonics

    phs_lo = phs_lo.real - phs_lo.real.mean()

    phs = phs_lo+phs_hi

    return phs

def ft_phase_screen(r0, N, delta, L0, l0, FFT=None, seed=None):
    """
    Creates a random phase screen with Von Karmen statistics.
    (Schmidt 2010)
    
    Parameters:
        r0 (float): r0 parameter of scrn in metres
        N (int): Size of phase scrn in pxls
        delta (float): size in Metres of each pxl
        L0 (float): Size of outer-scale in metres
        l0 (float): inner scale in metres
        seed (int, optional): seed for random number generator. If provided, 
            allows for deterministic screens  

    .. note::
        The phase screen is returned as a 2d array, with each element representing the phase 
        change in **radians**. This means that to obtain the physical phase distortion in nanometres, 
        it must be multiplied by (wavelength / (2*pi)), (where `wavellength` here is the same wavelength
        in which r0 is given in the function arguments)

    Returns:
        ndarray: numpy array representing phase screen in radians
    """
    delta = float(delta)
    r0 = float(r0)
    L0 = float(L0)
    l0 = float(l0)

    R = np.random.default_rng(seed)

    del_f = 1./(N*delta)

    fx = np.arange(-N/2., N/2.) * del_f

    (fx, fy) = np.meshgrid(fx,fx)
    f = np.sqrt(fx**2. + fy**2.)

    fm = 5.92/l0/(2*np.pi)
    f0 = 1./L0

    PSD_phi = (0.023*r0**(-5./3.) * np.exp(-1*((f/fm)**2)) / (((f**2) + (f0**2))**(11./6)))

    PSD_phi[int(N/2), int(N/2)] = 0

    cn = ((R.normal(size=(N, N))+1j * R.normal(size=(N, N))) * np.sqrt(PSD_phi)*del_f)

    phs = ift2(cn, 1, FFT).real

    return phs

# __all__ = ["PhaseScreenVonKarman", "PhaseScreenKolmogorov"]


class PhaseScreen(object):
    
    def set_X_coords(self):
        """
        Sets the coords of X, the new phase vector.
        """
        self.X_coords = np.zeros((self.nx_size, 2))
        self.X_coords[:, 0] = -1
        self.X_coords[:, 1] = np.arange(self.nx_size)
        self.X_positions = self.X_coords * self.pixel_scale

    def set_stencil_coords(self):
        """
        Sets the Z coordinates, sections of the phase screen that will be used to create new phase
        """
        self.stencil = np.zeros((self.stencil_length, self.nx_size))

        max_n = 1
        while True:
            if 2 ** (max_n - 1) + 1 >= self.nx_size:
                max_n -= 1
                break
            max_n += 1

        for n in range(0, max_n + 1):
            col = int((2 ** (n - 1)) + 1)
            n_points = (2 ** (max_n - n)) + 1

            coords = np.round(np.linspace(0, self.nx_size - 1, n_points)).astype('int32')
            self.stencil[col - 1][coords] = 1

        # Now fill in tail of stencil
        for n in range(1, self.stencil_length_factor + 1):
            col = n * self.nx_size - 1
            self.stencil[col, self.nx_size // 2] = 1

        self.stencil_coords = np.array(np.where(self.stencil == 1)).T
        self.stencil_positions = self.stencil_coords * self.pixel_scale

        self.n_stencils = len(self.stencil_coords)

    def calc_seperations(self):
        """
        Calculates the seperations between the phase points in the stencil and the new phase vector
        """
        positions = np.append(self.stencil_positions, self.X_positions, axis=0)
        self.seperations = np.zeros((len(positions), len(positions)))

        if numba:
            calc_seperations_fast(positions, self.seperations)
        else:
            for i, (x1, y1) in enumerate(positions):
                for j, (x2, y2) in enumerate(positions):
                    delta_x = x2 - x1
                    delta_y = y2 - y1

                    delta_r = np.sqrt(delta_x ** 2 + delta_y ** 2)

                    self.seperations[i, j] = delta_r

    def make_covmats(self):
        """
        Makes the covariance matrices required for adding new phase
        """
        self.cov_mat = phase_covariance(self.seperations, self.r0, self.L0)

        self.cov_mat_zz = self.cov_mat[:self.n_stencils, :self.n_stencils]
        self.cov_mat_xx = self.cov_mat[self.n_stencils:, self.n_stencils:]
        self.cov_mat_zx = self.cov_mat[:self.n_stencils, self.n_stencils:]
        self.cov_mat_xz = self.cov_mat[self.n_stencils:, :self.n_stencils]

    def makeAMatrix(self):
        """
        Calculates the "A" matrix, that uses the existing data to find a new 
        component of the new phase vector.
        """
        # Cholsky solve can fail - if so do brute force inversion
        try:
            cf = linalg.cho_factor(self.cov_mat_zz)
            inv_cov_zz = linalg.cho_solve(cf, np.identity(self.cov_mat_zz.shape[0]))
        except linalg.LinAlgError:
            # print("Cholesky solve failed. Performing SVD inversion...")
            # inv_cov_zz = np.linalg.pinv(self.cov_mat_zz)
            raise linalg.LinAlgError("Could not invert Covariance Matrix to for A and B Matrices. Try with a larger pixel scale or smaller L0")

        self.A_mat = self.cov_mat_xz.dot(inv_cov_zz)

    def makeBMatrix(self):
        """
        Calculates the "B" matrix, that turns a random vector into a component of the new phase.
        """
        # Can make initial BBt matrix first
        BBt = self.cov_mat_xx - self.A_mat.dot(self.cov_mat_zx)

        # Then do SVD to get B matrix
        u, W, ut = np.linalg.svd(BBt)

        L_mat = np.zeros((self.nx_size, self.nx_size))
        np.fill_diagonal(L_mat, np.sqrt(W))

        # Now use sqrt(eigenvalues) to get B matrix
        self.B_mat = u.dot(L_mat)

    # def make_initial_screen(self):
    #     """
    #     Makes the initial screen usign FFT method that can be extended 
    #     """

    #     # phase screen will make it *really* random if no seed at all given.
    #     # If a seed is here, screen must be repeatable wiht same seed
    #     self._R = np.random.default_rng(self.random_seed)

    #     self._scrn = ft_phase_screen(
    #         self.r0, self.stencil_length, self.pixel_scale, self.L0, 1e-10, seed=self._R
    #     )

    #     self._scrn = self._scrn[:, :self.nx_size]
    
    def make_initial_screen_nosh(self):
        """
        Makes the initial screen usign FFT method that can be extended 
        """

        # phase screen will make it *really* random if no seed at all given.
        # If a seed is here, screen must be repeatable wiht same seed
        
        self._R = np.random.default_rng(self.random_seed)
        # self._scrn = self.SCRN.scrn
        
        self._scrn = mprop.crop2(ft_phase_screen(
            self.r0, self.ovf*self.stencil_length, self.pixel_scale, self.L0, 1e-10, seed=self._R
        ), self.ovf)
        
        self._scrn = self._scrn[:, :self.nx_size]
    
    def make_initial_screen_sh(self):
        """
        Makes the initial screen usign FFT method that can be extended 
        """

        # phase screen will make it *really* random if no seed at all given.
        # If a seed is here, screen must be repeatable wiht same seed
        
        self._R = np.random.default_rng(self.random_seed)
        # self._scrn = self.SCRN.scrn
        
        self._scrn = mprop.crop2(ft_sh_phase_screen(
            self.r0, self.ovf*self.stencil_length, self.pixel_scale, self.L0, 1e-10, seed=self._R
        ), self.ovf)
        
        self._scrn = self._scrn[:, :self.nx_size]
    
    # def make_random_new_screen(self):
    #     self.SCRN.new_screen()
    #     self._scrn = self.SCRN.scrn

    def get_new_row(self):
        random_data = self._R.normal(0, 1, size=self.nx_size)

        stencil_data = self._scrn[(self.stencil_coords[:, 0], self.stencil_coords[:, 1])]
        new_row = self.A_mat.dot(stencil_data) + self.B_mat.dot(random_data)

        new_row.shape = (1, self.nx_size)
        return new_row

    def add_row(self):
        """
        Adds a new row to the phase screen and removes old ones.
        """

        new_row = self.get_new_row()

        self._scrn = np.append(new_row, self._scrn, axis=0)[:self.stencil_length, :self.nx_size]

        return self.scrn

    @property
    def scrn(self):
        """
        The current phase map held in the PhaseScreen object in radians.
        """
        return self._scrn[:self.requested_nx_size, :self.requested_nx_size]


class PhaseScreenVonKarman(PhaseScreen):
    """
    A "Phase Screen" for use in AO simulation with Von Karmon statistics.

    This represents the phase addition light experiences when passing through atmospheric
    turbulence. Unlike other phase screen generation techniques that translate a large static
    screen, this method keeps a small section of phase, and extends it as necessary for as many
    steps as required. This can significantly reduce memory consumption at the expense of more
    processing power required.

    The technique is described in a paper by Assemat and Wilson, 2006. It essentially assumes that
    there are two matrices, "A" and "B", that can be used to extend an existing phase screen.
    A single row or column of new phase can be represented by

        X = A.Z + B.b

    where X is the new phase vector, Z is some number of columns of the existing screen,
    and b is a random vector with gaussian statistics.

    This object calculates the A and B matrices using an expression of the phase covariance when it
    is initialised. Calculating A is straightforward through the relationship:

        A =  Cov_xz . (Cov_zz)^(-1).

    B is less trivial.

        BB^t = Cov_xx - A.Cov_zx

    (where B^t is the transpose of B) is a symmetric matrix, hence B can be expressed as

        B = UL,

    where U and L are obtained from the svd for BB^t

        U, w, U^t = svd(BB^t)

    L is a diagonal matrix where the diagonal elements are w^(1/2).

    On initialisation an initial phase screen is calculated using an FFT based method.
    When ``add_row`` is called, a new vector of phase is added to the phase screen using `nCols`
    columns of previous phase. Assemat & Wilson claim that two columns are adequate for good
    atmospheric statistics. The phase in the screen data is always accessed as ``<phasescreen>.scrn`` and is in radians.

        .. note::
        The phase screen is returned on each iteration as a 2d array, with each element representing the phase 
        change in **radians**. This means that to obtain the physical phase distortion in nanometres, 
        it must be multiplied by (wavelength / (2*pi)), (where `wavellength` here is the same wavelength
        in which r0 is given in the function arguments)

    Parameters:
        nx_size (int): Size of phase screen (NxN)
        pixel_scale(float): Size of each phase pixel in metres
        r0 (float): fried parameter (metres)
        L0 (float): Outer scale (metres)
        random_seed (int, optional): seed for the random number generator
        n_columns (int, optional): Number of columns to use to continue screen, default is 2
    """
    def __init__(self, nx_size, pixel_scale, r0, L0, random_seed=None, n_columns=2, sh=False, ovf=1):

        self.n_columns = n_columns

        self.requested_nx_size = nx_size
        self.nx_size = nx_size
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self.L0 = L0
        self.stencil_length_factor = 1
        self.stencil_length = self.nx_size

        self.random_seed = random_seed

        self.set_X_coords()
        self.set_stencil_coords()

        self.calc_seperations()
        self.make_covmats()

        self.makeAMatrix()
        self.makeBMatrix()
        
        self.ovf = ovf
        
        # if sh == False:
        #     self.SCRN = MyPhaseScreenNOSH(self.r0, self.stencil_length, 1, self.pixel_scale, self.L0, 1e-10)
        # else:
        #     self.SCRN = MyPhaseScreenSH(self.r0, self.stencil_length, 1, self.pixel_scale, self.L0, 1e-10)
        
        if sh == False:
            self.make_initial_screen_sh()
        else:
            self.make_initial_screen_nosh()

    def set_stencil_coords(self):
        self.stencil = np.zeros((self.stencil_length, self.nx_size))
        self.stencil[:self.n_columns] = 1

        self.stencil_coords = np.array(np.where(self.stencil==1)).T
        self.stencil_positions = self.stencil_coords * self.pixel_scale

        self.n_stencils = len(self.stencil_coords)


def find_allowed_size(nx_size):
    """
    Finds the next largest "allowed size" for the Fried Phase Screen method
    
    Parameters:
        nx_size (int): Requested size
    
    Returns:
        int: Next allowed size
    """
    n = 0
    while (2 ** n + 1) < nx_size:
        n += 1

    nx_size = 2 ** n + 1
    return nx_size


class PhaseScreenKolmogorov(PhaseScreen):
    """
    A "Phase Screen" for use in AO simulation using the Fried method for Kolmogorov turbulence.

    This represents the phase addition light experiences when passing through atmospheric 
    turbulence. Unlike other phase screen generation techniques that translate a large static 
    screen, this method keeps a small section of phase, and extends it as neccessary for as many 
    steps as required. This can significantly reduce memory consuption at the expense of more 
    processing power required.

    The technique is described in a paper by Assemat and Wilson, 2006 and expanded upon by Fried, 2008.
    It essentially assumes that there are two matrices, "A" and "B",
    that can be used to extend an existing phase screen.
    A single row or column of new phase can be represented by 

        X = A.Z + B.b

    where X is the new phase vector, Z is some data from the existing screen,
    and b is a random vector with gaussian statistics.

    This object calculates the A and B matrices using an expression of the phase covariance when it
    is initialised. Calculating A is straightforward through the relationship:

        A =  Cov_xz . (Cov_zz)^(-1).

    B is less trivial.

        BB^t = Cov_xx - A.Cov_zx

    (where B^t is the transpose of B) is a symmetric matrix, hence B can be expressed as 

        B = UL, 

    where U and L are obtained from the svd for BB^t

        U, w, U^t = svd(BB^t)

    L is a diagonal matrix where the diagonal elements are w^(1/2).    

    The Z data is taken from points in a "stencil" defined by Fried that samples the entire screen.
    An additional "reference point" is also considered, that is picked from a point separate from teh stencil 
    and applied on each iteration such that the new phase equation becomes:
    
    On initialisation an initial phase screen is calculated using an FFT based method.
    When ``add_row`` is called, a new vector of phase is added to the phase screen. The phase in the screen data
    is always accessed as ``<phasescreen>.scrn`` and is in radians.

    .. note::
        The phase screen is returned on each iteration as a 2d array, with each element representing the phase 
        change in **radians**. This means that to obtain the physical phase distortion in nanometres, 
        it must be multiplied by (wavelength / (2*pi)), (where `wavellength` here is the same wavelength
        in which r0 is given in the function arguments)

    Parameters:
        nx_size (int): Size of phase screen (NxN)
        pixel_scale(float): Size of each phase pixel in metres
        r0 (float): fried parameter (metres)
        L0 (float): Outer scale (metres)
        random_seed (int, optional): seed for the random number generator
        stencil_length_factor (int, optional): How much longer is the stencil than the desired phase? default is 4
    """
    def __init__(self, nx_size, pixel_scale, r0, L0, random_seed=None, stencil_length_factor=4):

        self.requested_nx_size = nx_size
        self.nx_size = find_allowed_size(nx_size)
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self.L0 = L0
        self.stencil_length_factor = stencil_length_factor
        self.stencil_length = stencil_length_factor * self.nx_size
        self.random_seed = random_seed

        # Coordinate of Fried's "reference point" that stops the screen diverging
        self.reference_coord = (1, 1)

        self.set_X_coords()
        self.set_stencil_coords()

        self.calc_seperations()
        self.make_covmats()

        self.makeAMatrix()
        self.makeBMatrix()
        self.make_initial_screen()

    def get_new_row(self):
        random_data = self._R.normal(0, 1, size=self.nx_size)

        stencil_data = self._scrn[(self.stencil_coords[:, 0], self.stencil_coords[:, 1])]

        reference_value = self._scrn[self.reference_coord]

        new_row = self.A_mat.dot(stencil_data - reference_value) + self.B_mat.dot(random_data) + reference_value

        new_row.shape = (1, self.nx_size)
        return new_row


    def __repr__(self):
        return str(self.scrn)
    


@numba.jit(nopython=True, parallel=True)
def calc_seperations_fast(positions, seperations):

    for i in numba.prange(len(positions)):
        x1, y1 = positions[i]
        for j in range(len(positions)):
            x2, y2 = positions[j]
            delta_x = x2 - x1
            delta_y = y2 - y1

            delta_r = np.sqrt(delta_x ** 2 + delta_y ** 2)

            seperations[i, j] = delta_r