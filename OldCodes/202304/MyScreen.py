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

class MyPhaseScreen():
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


# ss = MyPhaseScreen(0.1,64,2,0.01,20,1e-7)
# ss.new_screen()
# plt.imshow(ss.scrn)
# plt.show()