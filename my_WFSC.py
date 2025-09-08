# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:20:07 2023

@author: gqmc62
"""


"""Gather all wavefront sensor simulation and method."""

import numpy as np
from aotools.functions.zernike import zernikeArray
import pyfftw
from matplotlib import pyplot as plt

class IgTilt():
    """Ig tilt."""

    def __init__(self, N):
        self.phase = np.zeros((N, N), dtype=float)
        self.I = np.zeros((N, N), dtype=float)
        self.gx = np.zeros((N - 1, N), dtype=float)
        self.gy = np.zeros((N, N - 1), dtype=float)
        self.Ix = np.zeros((N - 1, N), dtype=float)
        self.Iy = np.zeros((N, N - 1), dtype=float)
        self.pupil = zernikeArray(1, N)[0]

    def measure(self, U):
        """Measure tilt in rad/lenslet."""
        self.phase = np.angle(U)
        self.I = np.abs(U)**2*self.pupil
        
        self.Ix = (self.I[1 : ,   : ] + self.I[ : -1, :   ])/2
        self.Iy = (self.I[  : , 1 : ] + self.I[ :   , : -1])/2

        
        self.gx = self.phase[1 : ,   : ] - self.phase[ : -1, :   ]
        self.gy = self.phase[  : , 1 : ] - self.phase[ :   , : -1]
        
        self.gx[self.gx >  np.pi] = self.gx[self.gx >  np.pi] - 2*np.pi
        self.gx[self.gx < -np.pi] = self.gx[self.gx < -np.pi] + 2*np.pi
        self.gy[self.gy >  np.pi] = self.gy[self.gy >  np.pi] - 2*np.pi
        self.gy[self.gy < -np.pi] = self.gy[self.gy < -np.pi] + 2*np.pi

        Ig0 = np.sum(self.gx * self.Ix)/self.Ix.sum()
        Ig1 = np.sum(self.gy * self.Iy)/self.Iy.sum()

        return np.asarray([Ig0, Ig1])*U.shape[0]

    def measure_with_noise(self, U, Nphot):
        """Measure tilt in rad/lenslet."""
        self.phase = np.angle(U)
        self.I = np.abs(U)**2*self.pupil
        
        self.I = self.I * Nphot/self.I.sum()
        self.I = np.random.poisson(self.I)
        
        self.Ix = (self.I[1 : ,   : ] + self.I[ : -1, :   ])/2
        self.Iy = (self.I[  : , 1 : ] + self.I[ :   , : -1])/2

        self.gx = self.phase[1 : ,   : ] - self.phase[ : -1, :   ]
        self.gy = self.phase[  : , 1 : ] - self.phase[ :   , : -1]

        self.gx[self.gx >  np.pi] = self.gx[self.gx >  np.pi] - 2*np.pi
        self.gx[self.gx < -np.pi] = self.gx[self.gx < -np.pi] + 2*np.pi
        self.gy[self.gy >  np.pi] = self.gy[self.gy >  np.pi] - 2*np.pi
        self.gy[self.gy < -np.pi] = self.gy[self.gy < -np.pi] + 2*np.pi

        Ig0 = np.sum(self.gx * self.Ix)/self.Ix.sum()
        Ig1 = np.sum(self.gy * self.Iy)/self.Iy.sum()
        
        # plt.imshow(self.I)
        # plt.colorbar()
        # plt.show()
        # plt.clf()

        return np.asarray([Ig0, Ig1]) * U.shape[0]

class Focus():
    """
    turn complex amplitude at the pupil plane into intensity profile at the focal plane
    """
    def __init__(self, N):
        """N is for the no zero-padded data."""
        self.N = N
        self.a = int(self.N/2)
        self.b = int(3*self.N/2)
        self.U = np.zeros((self.N, self.N), dtype=complex)

        self.F0 = pyfftw.empty_aligned((2*self.N, 2*self.N), dtype=complex)
        self.F1 = pyfftw.empty_aligned((2*self.N, 2*self.N), dtype=complex)
        self.FF01 = pyfftw.FFTW(self.F0, self.F1, axes=(-1,-2), direction='FFTW_FORWARD')

        (y1, x1) = np.meshgrid(np.arange(-self.N/2, self.N/2),
                               np.arange(-self.N/2, self.N/2))
        self.Q = np.exp(-1j*(x1 + y1)*np.pi/(2*self.N))

        self.Im = np.zeros((self.N, self.N), dtype=float)

        (self.ywfs, self.xwfs) = np.meshgrid(np.arange(self.N) + 0.5 - self.N/2,
                                   np.arange(self.N) + 0.5 - self.N/2)
        self.pupil = zernikeArray(1, N)[0]

        del x1, y1

    def focus(self, U):
        """Focus. return image normalized to total energy."""
        self.Im = np.abs(np.fft.fftshift(
            self.FF01(np.pad(U * self.Q * self.pupil,
                             (self.a, self.a), 'constant', constant_values=0.))
            )[self.a : self.b, self.a : self.b])**2
        self.Im = self.Im/self.Im.sum()

        return self.Im

class SHWFS_from_image_adaptive_ROI():
    """
    Run one SHWFS measurement from image of SHWFS detector plane.
    Receiving image of NxN but readout only nxn around the maximum point.
    """
    def __init__(self, N, NN):
        """N is for the no zero-padded data."""
        self.N = N # region of interest
        self.NN = NN # whole input image
        self.Im = np.zeros((self.N, self.N), dtype=float)
        self.IM = np.zeros((self.NN, self.NN), dtype=float)

        (self.ywfs, self.xwfs) = np.meshgrid(np.arange(self.N) + 0.5 - self.N/2,
                                   np.arange(self.N) + 0.5 - self.N/2)

    def measure_no_noise_no_threshold(self, IM):
        self.IM = IM

        MAX_LOC = np.where(self.IM == np.max(self.IM))
        if (len(MAX_LOC[0]) != 1) or (len(MAX_LOC[1]) != 1):
            # print('doubly max')
            positions0 = []
            positions1 = []
            for maxes in range(len(MAX_LOC[0])):
                MAX_LOC_X = MAX_LOC[0][maxes]
                MAX_LOC_Y = MAX_LOC[1][maxes]
        
                low_x = (MAX_LOC_X - self.N//2)
                high_x = (MAX_LOC_X + self.N//2)
                low_y = (MAX_LOC_Y - self.N//2)
                high_y = (MAX_LOC_Y + self.N//2)
                
                if self.N % 2 == 1:
                    high_x += 1
                    high_y += 1
        
                if ((low_x < 0) or (low_y < 0) or (high_x > (self.NN - 1)) or (high_y > (self.NN - 1))):
                    print('out of bound')
        
                self.Im = self.IM[low_x:high_x,low_y:high_y]
                Im_sum = np.sum(self.Im)
                positions0.append(np.sum(self.Im * self.xwfs)/Im_sum + MAX_LOC_X - self.NN//2)
                positions1.append(np.sum(self.Im * self.ywfs)/Im_sum + MAX_LOC_Y - self.NN//2)

            SHWFS0 = np.mean(positions0)
            SHWFS1 = np.mean(positions1)
                
        else:
            MAX_LOC_X = MAX_LOC[0][0]
            MAX_LOC_Y = MAX_LOC[1][0]
    
            low_x = (MAX_LOC_X - self.N//2)
            high_x = (MAX_LOC_X + self.N//2)
            low_y = (MAX_LOC_Y - self.N//2)
            high_y = (MAX_LOC_Y + self.N//2)
            
            if self.N %2 == 1:
                high_x += 1
                high_y += 1
    
            if ((low_x < 0) or (low_y < 0) or (high_x > (self.NN - 1)) or (high_y > (self.NN - 1))):
                print('out of bound')
    
            self.Im = self.IM[low_x:high_x,low_y:high_y]
            Im_sum = np.sum(self.Im)
            SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum + MAX_LOC_X - self.NN//2
            SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum + MAX_LOC_Y - self.NN//2
            
            # plt.imshow(self.Im)
            # plt.colorbar()
            # plt.show()
            # plt.clf()
        if self.N %2 == 1:
            SHWFS0 += 0.5
            SHWFS1 += 0.5

        return np.asarray([SHWFS0, SHWFS1])*np.pi
    
    def measure_with_noise_with_threshold(self, IM, Nphot, sigma_readout, MaxNormThreshold):
        """Measure tilt. U must not be zero-padded. In number of photons."""
        self.IM = IM

        MAX_LOC = np.where(self.IM == np.max(self.IM))
        if (len(MAX_LOC[0]) != 1) or (len(MAX_LOC[1]) != 1):
            # print('doubly max')
            positions0 = []
            positions1 = []
            for maxes in range(len(MAX_LOC[0])):
                MAX_LOC_X = MAX_LOC[0][maxes]
                MAX_LOC_Y = MAX_LOC[1][maxes]
        
                low_x = (MAX_LOC_X - self.N//2)
                high_x = (MAX_LOC_X + self.N//2)
                low_y = (MAX_LOC_Y - self.N//2)
                high_y = (MAX_LOC_Y + self.N//2)
                if self.N %2 == 1:
                    high_x += 1
                    high_y += 1
        
                if ((low_x < 0) or (low_y < 0) or (high_x > (self.NN - 1)) or (high_y > (self.NN - 1))):
                    print('out of bound')
        
                self.Im = self.IM[low_x:high_x,low_y:high_y]
                self.Im = self.Im * Nphot/self.Im.sum()
                self.Im = (np.random.poisson(self.Im)
                           + np.random.normal(0, sigma_readout, size=(self.N, self.N)))

                maxval = self.Im.max()
                self.Im = self.Im - MaxNormThreshold*maxval
                
                self.Im[self.Im < 0] = 0.
                Im_sum = np.sum(self.Im)
                positions0.append(np.sum(self.Im * self.xwfs)/Im_sum + MAX_LOC_X - self.NN//2)
                positions1.append(np.sum(self.Im * self.ywfs)/Im_sum + MAX_LOC_Y - self.NN//2)

            SHWFS0 = np.mean(positions0)
            SHWFS1 = np.mean(positions1)
                
        else:
            MAX_LOC_X = MAX_LOC[0][0]
            MAX_LOC_Y = MAX_LOC[1][0]
    
            low_x = (MAX_LOC_X - self.N//2)
            high_x = (MAX_LOC_X + self.N//2)
            low_y = (MAX_LOC_Y - self.N//2)
            high_y = (MAX_LOC_Y + self.N//2)
            if self.N %2 == 1:
                high_x += 1
                high_y += 1
    
            if ((low_x < 0) or (low_y < 0) or (high_x > (self.NN - 1)) or (high_y > (self.NN - 1))):
                print('out of bound')
    
            self.Im = self.IM[low_x:high_x,low_y:high_y]
            self.Im = self.Im * Nphot/self.Im.sum()
            self.Im = (np.random.poisson(self.Im)
                       + np.random.normal(0, sigma_readout, size=(self.N, self.N)))

            maxval = self.Im.max()
            self.Im = self.Im - MaxNormThreshold*maxval
            
            self.Im[self.Im < 0] = 0.
            Im_sum = np.sum(self.Im)
            SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum + MAX_LOC_X - self.NN//2
            SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum + MAX_LOC_Y - self.NN//2
            
            # plt.imshow(self.Im)
            # plt.colorbar()
            # plt.show()
            # plt.clf()
        if self.N %2 == 1:
            SHWFS0 += 0.5
            SHWFS1 += 0.5

        return np.asarray([SHWFS0, SHWFS1])*np.pi
    
    def measure_no_noise_with_threshold(self, IM, MaxNormThreshold):
        """Measure tilt. U must not be zero-padded. In number of photons."""
        self.IM = IM

        MAX_LOC = np.where(self.IM == np.max(self.IM))
        if (len(MAX_LOC[0]) != 1) or (len(MAX_LOC[1]) != 1):
            # print('doubly max')
            positions0 = []
            positions1 = []
            for maxes in range(len(MAX_LOC[0])):
                MAX_LOC_X = MAX_LOC[0][maxes]
                MAX_LOC_Y = MAX_LOC[1][maxes]
        
                low_x = (MAX_LOC_X - self.N//2)
                high_x = (MAX_LOC_X + self.N//2)
                low_y = (MAX_LOC_Y - self.N//2)
                high_y = (MAX_LOC_Y + self.N//2)
                if self.N %2 == 1:
                    high_x += 1
                    high_y += 1
        
                if ((low_x < 0) or (low_y < 0) or (high_x > (self.NN - 1)) or (high_y > (self.NN - 1))):
                    print('out of bound')
        
                self.Im = self.IM[low_x:high_x,low_y:high_y]
                maxval = self.Im.max()
                self.Im = self.Im - MaxNormThreshold*maxval
                
                self.Im[self.Im < 0] = 0.
                Im_sum = np.sum(self.Im)
                positions0.append(np.sum(self.Im * self.xwfs)/Im_sum + MAX_LOC_X - self.NN//2)
                positions1.append(np.sum(self.Im * self.ywfs)/Im_sum + MAX_LOC_Y - self.NN//2)

            SHWFS0 = np.mean(positions0)
            SHWFS1 = np.mean(positions1)
                
        else:
            MAX_LOC_X = MAX_LOC[0][0]
            MAX_LOC_Y = MAX_LOC[1][0]
    
            low_x = (MAX_LOC_X - self.N//2)
            high_x = (MAX_LOC_X + self.N//2)
            low_y = (MAX_LOC_Y - self.N//2)
            high_y = (MAX_LOC_Y + self.N//2)
            if self.N %2 == 1:
                high_x += 1
                high_y += 1
    
            if ((low_x < 0) or (low_y < 0) or (high_x > (self.NN - 1)) or (high_y > (self.NN - 1))):
                print('out of bound')
    
            self.Im = self.IM[low_x:high_x,low_y:high_y]
            maxval = self.Im.max()
            self.Im = self.Im - MaxNormThreshold*maxval
            
            self.Im[self.Im < 0] = 0.
            Im_sum = np.sum(self.Im)
            SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum + MAX_LOC_X - self.NN//2
            SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum + MAX_LOC_Y - self.NN//2
            
            # plt.imshow(self.Im)
            # plt.colorbar()
            # plt.show()
            # plt.clf()
        if self.N %2 == 1:
            SHWFS0 += 0.5
            SHWFS1 += 0.5

        return np.asarray([SHWFS0, SHWFS1])*np.pi


class SHWFS_from_U_adaptive_ROI():
    """
    Run one SHWFS measurement from image of SHWFS detector plane.
    Receiving image of NxN but readout only nxn around the maximum point.
    """
    def __init__(self, N, NN):
        """N is for the no zero-padded data."""
        self.N = N # region of interest
        self.NN = NN # whole input image
        self.Im = np.zeros((self.N, self.N), dtype=float)
        self.IM = np.zeros((self.NN, self.NN), dtype=float)

        (self.ywfs, self.xwfs) = np.meshgrid(np.arange(self.N) + 0.5 - self.N/2,
                                   np.arange(self.N) + 0.5 - self.N/2)
        
        self.a = int(self.NN/2)
        self.b = int(3*self.NN/2)
        self.U = np.zeros((self.NN, self.NN), dtype=complex)

        self.F0 = pyfftw.empty_aligned((2*self.NN, 2*self.NN), dtype=complex)
        self.F1 = pyfftw.empty_aligned((2*self.NN, 2*self.NN), dtype=complex)
        self.FF01 = pyfftw.FFTW(self.F0, self.F1, axes=(-1,-2), direction='FFTW_FORWARD')

        (y1, x1) = np.meshgrid(np.arange(-self.NN/2, self.NN/2),
                               np.arange(-self.NN/2, self.NN/2))
        self.Q = np.exp(-1j*(x1 + y1)*np.pi/(2*self.NN))
        
        self.pupil = zernikeArray(1, NN)[0]

        del x1, y1
    def makeIM(self,U):
        self.IM = np.abs(np.fft.fftshift(
            self.FF01(np.pad(U * self.Q * self.pupil,
                             (self.a, self.a), 'constant', constant_values=0.))
            )[self.a : self.b, self.a : self.b])**2
        self.IM = self.IM/self.IM.sum()
    def measure_no_noise_no_threshold(self, U):
        self.makeIM(U)

        MAX_LOC = np.where(self.IM == np.max(self.IM))
        if (len(MAX_LOC[0]) != 1) or (len(MAX_LOC[1]) != 1):
            # print('doubly max')
            positions0 = []
            positions1 = []
            for maxes in range(len(MAX_LOC[0])):
                MAX_LOC_X = MAX_LOC[0][maxes]
                MAX_LOC_Y = MAX_LOC[1][maxes]
        
                low_x = (MAX_LOC_X - self.N//2)
                high_x = (MAX_LOC_X + self.N//2)
                low_y = (MAX_LOC_Y - self.N//2)
                high_y = (MAX_LOC_Y + self.N//2)
                if self.N %2 == 1:
                    high_x += 1
                    high_y += 1
        
                if ((low_x < 0) or (low_y < 0) or (high_x > (self.NN - 1)) or (high_y > (self.NN - 1))):
                    print('out of bound')
        
                self.Im = self.IM[low_x:high_x,low_y:high_y]
                Im_sum = np.sum(self.Im)
                positions0.append(np.sum(self.Im * self.xwfs)/Im_sum + MAX_LOC_X - self.NN//2)
                positions1.append(np.sum(self.Im * self.ywfs)/Im_sum + MAX_LOC_Y - self.NN//2)

            SHWFS0 = np.mean(positions0)
            SHWFS1 = np.mean(positions1)
                
        else:
            MAX_LOC_X = MAX_LOC[0][0]
            MAX_LOC_Y = MAX_LOC[1][0]
    
            low_x = (MAX_LOC_X - self.N//2)
            high_x = (MAX_LOC_X + self.N//2)
            low_y = (MAX_LOC_Y - self.N//2)
            high_y = (MAX_LOC_Y + self.N//2)
            if self.N %2 == 1:
                high_x += 1
                high_y += 1
    
            if ((low_x < 0) or (low_y < 0) or (high_x > (self.NN - 1)) or (high_y > (self.NN - 1))):
                print('out of bound')
    
            self.Im = self.IM[low_x:high_x,low_y:high_y]
            Im_sum = np.sum(self.Im)
            SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum + MAX_LOC_X - self.NN//2
            SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum + MAX_LOC_Y - self.NN//2
            
            # plt.imshow(self.Im)
            # plt.colorbar()
            # plt.show()
            # plt.clf()
        
        if self.N %2 == 1:
            SHWFS0 += 0.5
            SHWFS1 += 0.5

        return np.asarray([SHWFS0, SHWFS1])*np.pi
    
    def measure_with_noise_with_threshold(self, U, Nphot, sigma_readout, MaxNormThreshold):
        """Measure tilt. U must not be zero-padded. In number of photons."""
   
        self.makeIM(U)

        MAX_LOC = np.where(self.IM == np.max(self.IM))
        if (len(MAX_LOC[0]) != 1) or (len(MAX_LOC[1]) != 1):
            # print('doubly max')
            positions0 = []
            positions1 = []
            for maxes in range(len(MAX_LOC[0])):
                MAX_LOC_X = MAX_LOC[0][maxes]
                MAX_LOC_Y = MAX_LOC[1][maxes]
        
                low_x = (MAX_LOC_X - self.N//2)
                high_x = (MAX_LOC_X + self.N//2)
                low_y = (MAX_LOC_Y - self.N//2)
                high_y = (MAX_LOC_Y + self.N//2)
                if self.N %2 == 1:
                    high_x += 1
                    high_y += 1
        
                if ((low_x < 0) or (low_y < 0) or (high_x > (self.NN - 1)) or (high_y > (self.NN - 1))):
                    print('out of bound')
        
                self.Im = self.IM[low_x:high_x,low_y:high_y]
                self.Im = self.Im * Nphot/self.Im.sum()
                self.Im = (np.random.poisson(self.Im)
                           + np.random.normal(0, sigma_readout, size=(self.N, self.N)))

                maxval = self.Im.max()
                self.Im = self.Im - MaxNormThreshold*maxval
                
                self.Im[self.Im < 0] = 0.
                Im_sum = np.sum(self.Im)
                positions0.append(np.sum(self.Im * self.xwfs)/Im_sum + MAX_LOC_X - self.NN//2)
                positions1.append(np.sum(self.Im * self.ywfs)/Im_sum + MAX_LOC_Y - self.NN//2)

            SHWFS0 = np.mean(positions0)
            SHWFS1 = np.mean(positions1)
                
        else:
            MAX_LOC_X = MAX_LOC[0][0]
            MAX_LOC_Y = MAX_LOC[1][0]
    
            low_x = (MAX_LOC_X - self.N//2)
            high_x = (MAX_LOC_X + self.N//2)
            low_y = (MAX_LOC_Y - self.N//2)
            high_y = (MAX_LOC_Y + self.N//2)
            if self.N %2 == 1:
                high_x += 1
                high_y += 1
    
            if ((low_x < 0) or (low_y < 0) or (high_x > (self.NN - 1)) or (high_y > (self.NN - 1))):
                print('out of bound')
    
            self.Im = self.IM[low_x:high_x,low_y:high_y]
            self.Im = self.Im * Nphot/self.Im.sum()
            self.Im = (np.random.poisson(self.Im)
                       + np.random.normal(0, sigma_readout, size=(self.N, self.N)))

            maxval = self.Im.max()
            self.Im = self.Im - MaxNormThreshold*maxval
            
            self.Im[self.Im < 0] = 0.
            Im_sum = np.sum(self.Im)
            SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum + MAX_LOC_X - self.NN//2
            SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum + MAX_LOC_Y - self.NN//2
            
            # plt.imshow(self.Im)
            # plt.colorbar()
            # plt.show()
            # plt.clf()
        if self.N %2 == 1:
            SHWFS0 += 0.5
            SHWFS1 += 0.5

        return np.asarray([SHWFS0, SHWFS1])*np.pi
    
    def measure_no_noise_with_threshold(self, U, MaxNormThreshold):
        """Measure tilt. U must not be zero-padded. In number of photons."""
   
        self.makeIM(U)

        MAX_LOC = np.where(self.IM == np.max(self.IM))
        if (len(MAX_LOC[0]) != 1) or (len(MAX_LOC[1]) != 1):
            # print('doubly max')
            positions0 = []
            positions1 = []
            for maxes in range(len(MAX_LOC[0])):
                MAX_LOC_X = MAX_LOC[0][maxes]
                MAX_LOC_Y = MAX_LOC[1][maxes]
        
                low_x = (MAX_LOC_X - self.N//2)
                high_x = (MAX_LOC_X + self.N//2)
                low_y = (MAX_LOC_Y - self.N//2)
                high_y = (MAX_LOC_Y + self.N//2)
                if self.N %2 == 1:
                    high_x += 1
                    high_y += 1
        
                if ((low_x < 0) or (low_y < 0) or (high_x > (self.NN - 1)) or (high_y > (self.NN - 1))):
                    print('out of bound')
        
                self.Im = self.IM[low_x:high_x,low_y:high_y]
                maxval = self.Im.max()
                self.Im = self.Im - MaxNormThreshold*maxval
                
                self.Im[self.Im < 0] = 0.
                Im_sum = np.sum(self.Im)
                positions0.append(np.sum(self.Im * self.xwfs)/Im_sum + MAX_LOC_X - self.NN//2)
                positions1.append(np.sum(self.Im * self.ywfs)/Im_sum + MAX_LOC_Y - self.NN//2)

            SHWFS0 = np.mean(positions0)
            SHWFS1 = np.mean(positions1)
                
        else:
            MAX_LOC_X = MAX_LOC[0][0]
            MAX_LOC_Y = MAX_LOC[1][0]
    
            low_x = (MAX_LOC_X - self.N//2)
            high_x = (MAX_LOC_X + self.N//2)
            low_y = (MAX_LOC_Y - self.N//2)
            high_y = (MAX_LOC_Y + self.N//2)
            if self.N %2 == 1:
                high_x += 1
                high_y += 1
    
            if ((low_x < 0) or (low_y < 0) or (high_x > (self.NN - 1)) or (high_y > (self.NN - 1))):
                print('out of bound')
    
            self.Im = self.IM[low_x:high_x,low_y:high_y]
            maxval = self.Im.max()
            self.Im = self.Im - MaxNormThreshold*maxval
            
            self.Im[self.Im < 0] = 0.
            Im_sum = np.sum(self.Im)
            SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum + MAX_LOC_X - self.NN//2
            SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum + MAX_LOC_Y - self.NN//2
            
            # plt.imshow(self.Im)
            # plt.colorbar()
            # plt.show()
            # plt.clf()
        
        if self.N %2 == 1:
            SHWFS0 += 0.5
            SHWFS1 += 0.5

        return np.asarray([SHWFS0, SHWFS1])*np.pi

class FastestFTSHWFS():
    """
    SHWFS with 1-step propagation.
    
    Works best for 1:1 input:output pixel ratio.
    Output angular scale is 0.5 wvl/d.
    """

    def __init__(self, N):
        """N is for the no zero-padded data."""
        self.N = N
        self.a = int(self.N/2)
        self.b = int(3*self.N/2)
        self.U = np.zeros((self.N, self.N), dtype=complex)

        self.F0 = pyfftw.empty_aligned((2*self.N, 2*self.N), dtype=complex)
        self.F1 = pyfftw.empty_aligned((2*self.N, 2*self.N), dtype=complex)
        self.FF01 = pyfftw.FFTW(self.F0, self.F1, axes=(-1,-2), direction='FFTW_FORWARD')

        (self.ywfs, self.xwfs) = np.meshgrid(np.arange(-self.N/2, self.N/2),
                               np.arange(-self.N/2, self.N/2))

        self.Im = np.zeros((self.N, self.N), dtype=float)

        self.pupil = zernikeArray(1, N)[0]
        
    def measure_no_noise_no_threshold(self, U):
        """Measure tilt. U must not be zero-padded."""
        self.Im = np.abs(np.fft.fftshift(
            self.FF01(np.pad(U * self.pupil,
                             (self.a, self.a), 'constant', constant_values=0.))
            )[self.a : self.b, self.a : self.b])**2
        self.Im = self.Im/self.Im.sum()

        SHWFS0 = np.sum(self.Im * self.xwfs)
        SHWFS1 = np.sum(self.Im * self.ywfs)
        
        # plt.imshow(self.Im)
        # plt.colorbar()
        # plt.show()
        # plt.clf()

        return np.asarray([SHWFS0, SHWFS1])*np.pi

class OneStepSHWFS():
    """
    SHWFS with 1-step propagation.
    
    Works best for 1:1 input:output pixel ratio.
    Output angular scale is 0.5 wvl/d.
    """

    def __init__(self, N):
        """N is for the no zero-padded data."""
        self.N = N
        self.a = int(self.N/2)
        self.b = int(3*self.N/2)
        self.U = np.zeros((self.N, self.N), dtype=complex)

        self.F0 = pyfftw.empty_aligned((2*self.N, 2*self.N), dtype=complex)
        self.F1 = pyfftw.empty_aligned((2*self.N, 2*self.N), dtype=complex)
        self.FF01 = pyfftw.FFTW(self.F0, self.F1, axes=(-1,-2), direction='FFTW_FORWARD')

        (y1, x1) = np.meshgrid(np.arange(-self.N/2, self.N/2),
                               np.arange(-self.N/2, self.N/2))
        self.Q = np.exp(-1j*(x1 + y1)*np.pi/(2*self.N))

        self.Im = np.zeros((self.N, self.N), dtype=float)

        (self.ywfs, self.xwfs) = np.meshgrid(np.arange(self.N) + 0.5 - self.N/2,
                                   np.arange(self.N) + 0.5 - self.N/2)
        self.pupil = zernikeArray(1, N)[0]

        del x1, y1
        
    def measure_no_noise_no_threshold(self, U):
        """Measure tilt. U must not be zero-padded."""
        self.Im = np.abs(np.fft.fftshift(
            self.FF01(np.pad(U * self.Q * self.pupil,
                             (self.a, self.a), 'constant', constant_values=0.))
            )[self.a : self.b, self.a : self.b])**2
        self.Im = self.Im/self.Im.sum()

        SHWFS0 = np.sum(self.Im * self.xwfs)
        SHWFS1 = np.sum(self.Im * self.ywfs)
        
        # plt.imshow(self.Im)
        # plt.colorbar()
        # plt.show()
        # plt.clf()

        return np.asarray([SHWFS0, SHWFS1])*np.pi

    def measure_with_noise_no_threshold(self, U, Nphot, sigma_readout):
        """Measure tilt. U must not be zero-padded."""
        self.Im = np.abs(np.fft.fftshift(
            self.FF01(np.pad(U * self.pupil * self.Q,
                             (self.a, self.a), 'constant', constant_values=0.))
            )[self.a : self.b, self.a : self.b])**2

        self.Im = self.Im * Nphot/self.Im.sum()
        self.Im = (np.random.poisson(self.Im)
                   + np.random.normal(0, sigma_readout, size=(self.N, self.N)))
        self.Im[self.Im < 0] = 0.

        Im_sum = np.sum(self.Im)
        SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum
        SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum
        
        # plt.imshow(self.Im)
        # plt.colorbar()
        # plt.show()
        # plt.clf()

        return np.asarray([SHWFS0, SHWFS1])*np.pi

    def measure_with_noise_with_threshold(self, U, Nphot, sigma_readout, MaxNormThreshold):
        """Measure tilt. U must not be zero-padded. In number of photons."""
        self.Im = np.abs(np.fft.fftshift(
            self.FF01(np.pad(U * self.pupil * self.Q,
                             (self.a, self.a), 'constant', constant_values=0.))
            )[self.a : self.b, self.a : self.b])**2

        self.Im = self.Im * Nphot/self.Im.sum()
        self.Im = (np.random.poisson(self.Im)
                   + np.random.normal(0, sigma_readout, size=(self.N, self.N)))
        
        maxval = self.Im.max()
        self.Im = self.Im - MaxNormThreshold*maxval
        
        self.Im[self.Im < 0] = 0.

        Im_sum = np.sum(self.Im)
        SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum
        SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum

        # plt.imshow(self.Im)
        # plt.colorbar()
        # plt.show()
        # plt.clf()

        return np.asarray([SHWFS0, SHWFS1])*np.pi
    
    def measure_no_noise_with_threshold(self, U, MaxNormThreshold):
        """Measure tilt. U must not be zero-padded. In number of photons."""
        self.Im = np.abs(np.fft.fftshift(
            self.FF01(np.pad(U * self.pupil * self.Q,
                             (self.a, self.a), 'constant', constant_values=0.))
            )[self.a : self.b, self.a : self.b])**2

        # self.Im = self.Im * Nphot/self.Im.sum()
        # self.Im = (np.random.poisson(self.Im)
        #            + np.random.normal(0, sigma_readout, size=(self.N, self.N)))
        
        maxval = self.Im.max()
        self.Im = self.Im - MaxNormThreshold*maxval
        
        self.Im[self.Im < 0] = 0.

        Im_sum = np.sum(self.Im)
        SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum
        SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum

        # plt.imshow(self.Im)
        # plt.colorbar()
        # plt.show()
        # plt.clf()

        return np.asarray([SHWFS0, SHWFS1])*np.pi

    # def measure_with_noise_with_threshold(self, U, Nphot, sigma_readout, SDThreshold):
    #     """Measure tilt. U must not be zero-padded. In number of photons."""
    #     self.Im = np.abs(np.fft.fftshift(
    #         self.FF01(np.pad(U * self.pupil * self.Q,
    #                          (self.a, self.a), 'constant', constant_values=0.))
    #         )[self.a : self.b, self.a : self.b])**2

    #     self.Im = self.Im * Nphot/self.Im.sum()
    #     self.Im = (np.random.poisson(self.Im)
    #                + np.random.normal(0, sigma_readout, size=(self.N, self.N)))
    #     background = np.std(self.Im)
    #     self.Im = self.Im - SDThreshold*background
    #     self.Im[self.Im < 0] = 0.

    #     Im_sum = np.sum(self.Im)
    #     SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum
    #     SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum

    #     # plt.imshow(self.Im)
    #     # plt.colorbar()
    #     # plt.show()
    #     # plt.clf()

    #     return np.asarray([SHWFS0, SHWFS1])*np.pi

class AngularPropSHWFS():
    """
    SHWFS with Multi-step angular propagation.
    
    Output angular scale can be adjusted.
    Input to output pixel ratio is flexible.
    
    Need proper implementation.
    """

    def __init__(self, n_in, d_in, alpha_out, FoV_out, wvl):
        """N is for the no zero-padded data. Is not yet written."""
        self.n_in = int(n_in)
        self.d_in = float(d_in)
        self.alpha_out = float(alpha_out*wvl/d_in)
        self.FoV_out = float(FoV_out*wvl/d_in)
        self.wvl = float(wvl)
        
        self.n_out = int(np.ceil(FoV_out/alpha_out))
        self.delta_in = float(d_in/n_in)
        self.delta_out = self.delta_in
        self.focal_length = self.delta_out / self.alpha_out
        self.convert = ((self.delta_out)/self.focal_length)*(self.n_in*self.delta_in)*(2*np.pi/self.wvl)
        
        wavenumber = 2*np.pi/self.wvl
        # set N to 2*max(n_in,n_out)
        # 
        self.N = int((self.n_in + self.n_out + self.wvl*self.focal_length/self.delta_in/self.delta_out)/2) # schimidt#2
        self.N = int(2**np.ceil(np.log(self.N)/np.log(2)))
        if (self.N < self.wvl*self.focal_length/self.delta_in/self.delta_out):
            if (np.isclose(self.N, self.wvl*self.focal_length/self.delta_in/self.delta_out, rtol=0.1) == False):
                raise Exception('Aliasing Problem.')

        self.U_pupil = np.zeros((self.N, self.N), dtype=complex)
        self.U_focal = np.zeros((self.N, self.N), dtype=complex)
        self.Im = np.zeros((self.n_out, self.n_out), dtype=float)
        (self.ywfs, self.xwfs) = np.meshgrid(np.arange(self.n_out) + 0.5 - self.n_out/2,
                                    np.arange(self.n_out) + 0.5 - self.n_out/2)
        self.pad_in = int((self.N - self.n_in)/2)
        self.cut_out = int((self.N - self.n_out)/2)

        (nx,ny) = np.meshgrid(np.arange(-self.N/2,self.N/2),
                                  np.arange(-self.N/2,self.N/2))
        nsq = nx**2 + ny**2

        #Spatial Frequencies (of source plane)
        df1 = 1. / (self.N*self.delta_in)

        #Quadratic phase factors
        self.Q1 = (np.exp(-1j*wavenumber*((nx + ny)*self.delta_in*self.delta_out/2/self.focal_length))
                    * np.exp(-1j*wavenumber*nsq*self.delta_in**2/2./self.focal_length))
        self.Q2 = np.exp(-1j * np.pi**2 * 2 * self.focal_length/wavenumber * nsq*df1**2)
        
        del nx, ny, nsq
        
        self.F_in = pyfftw.empty_aligned((self.N, self.N), dtype=complex)
        self.F_out = pyfftw.empty_aligned((self.N, self.N), dtype=complex)
        self.FF_FORWARD = pyfftw.FFTW(self.F_in, self.F_out, axes=(-1,-2), direction='FFTW_FORWARD')
        self.FF_BACKWARD = pyfftw.FFTW(self.F_in, self.F_out, axes=(-1,-2), direction='FFTW_BACKWARD')
        
        self.pupil = zernikeArray(1, n_in)[0]

    def measure_no_noise_no_threshold(self, U_in):
        """Measure tilt. U must not be zero-padded."""
        self.U_pupil = np.pad(U_in * self.pupil, ((self.pad_in,self.pad_in),(self.pad_in,self.pad_in)), 'constant', constant_values=0.)
        self.U_focal = np.fft.ifftshift(self.FF_BACKWARD(np.fft.ifftshift(
                        self.Q2 * np.fft.fftshift(self.FF_FORWARD(np.fft.fftshift(
                            self.Q1 * self.U_pupil
                            )))
                        )))
        self.Im = np.abs(self.U_focal[self.cut_out:self.N-self.cut_out,self.cut_out:self.N-self.cut_out])**2
        self.Im = self.Im/self.Im.sum()

        SHWFS0 = np.sum(self.Im * self.xwfs)
        SHWFS1 = np.sum(self.Im * self.ywfs)
        
        # plt.imshow(self.Im)
        # plt.colorbar()
        # plt.show()
        # plt.clf()

        return np.asarray([SHWFS0, SHWFS1])*self.convert

    def measure_no_noise_with_threshold(self, U_in, SDThreshold):
        """Measure tilt. U must not be zero-padded. Normalized to max I."""
        self.U_pupil = np.pad(U_in * self.pupil, ((self.pad_in,self.pad_in),(self.pad_in,self.pad_in)), 'constant', constant_values=0.)
        self.U_focal = np.fft.ifftshift(self.FF_BACKWARD(np.fft.ifftshift(
                        self.Q2 * np.fft.fftshift(self.FF_FORWARD(np.fft.fftshift(
                            self.Q1 * self.U_pupil
                            )))
                        )))
        self.Im = np.abs(self.U_focal[self.cut_out:self.N-self.cut_out,self.cut_out:self.N-self.cut_out])**2

        self.Im = (self.Im/self.Im.max())
        background = np.std(self.Im)
        self.Im = self.Im - SDThreshold*background
        self.Im[self.Im < 0] = 0.
        self.Im = self.Im / self.Im.max()

        Im_sum = np.sum(self.Im)
        SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum
        SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum
        
        # plt.imshow(self.Im)
        # plt.colorbar()
        # plt.show()
        # plt.clf()

        return np.asarray([SHWFS0, SHWFS1])*self.convert

    # def measure_with_noise_no_threshold(self, U_in, Nphot, sigma_readout):
    def measure_photon_noise(self, U_in, Nphot):
        """Measure tilt. U must not be zero-padded."""
        self.U_pupil = np.pad(U_in * self.pupil, ((self.pad_in,self.pad_in),(self.pad_in,self.pad_in)), 'constant', constant_values=0.)
        self.U_focal = np.fft.ifftshift(self.FF_BACKWARD(np.fft.ifftshift(
                        self.Q2 * np.fft.fftshift(self.FF_FORWARD(np.fft.fftshift(
                            self.Q1 * self.U_pupil
                            )))
                        )))
        self.Im = np.abs(self.U_focal[self.cut_out:self.N-self.cut_out,self.cut_out:self.N-self.cut_out])**2

        self.Im = self.Im * Nphot/self.Im.sum()
        # self.Im = (np.random.poisson(self.Im)
        #            + np.random.normal(0, sigma_readout, size=(self.N, self.N)))
        self.Im = (np.random.poisson(self.Im))

        Im_sum = np.sum(self.Im)
        SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum
        SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum
        
        # plt.imshow(self.Im)
        # plt.colorbar()
        # plt.show()
        # plt.clf()

        return np.asarray([SHWFS0, SHWFS1])*self.convert
    
    def measure_with_noise_no_threshold(self, U_in, Nphot, sigma_readout):
        """Measure tilt. U must not be zero-padded."""
        self.U_pupil = np.pad(U_in * self.pupil, ((self.pad_in,self.pad_in),(self.pad_in,self.pad_in)), 'constant', constant_values=0.)
        self.U_focal = np.fft.ifftshift(self.FF_BACKWARD(np.fft.ifftshift(
                        self.Q2 * np.fft.fftshift(self.FF_FORWARD(np.fft.fftshift(
                            self.Q1 * self.U_pupil
                            )))
                        )))
        self.Im = np.abs(self.U_focal[self.cut_out:self.N-self.cut_out,self.cut_out:self.N-self.cut_out])**2

        self.Im = self.Im * Nphot/self.Im.sum()
        self.Im = (np.random.poisson(self.Im)
                    + np.random.normal(0, sigma_readout, size=(self.N, self.N)))

        Im_sum = np.sum(self.Im)
        SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum
        SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum
        
        # plt.imshow(self.Im)
        # plt.colorbar()
        # plt.show()
        # plt.clf()
        return np.asarray([SHWFS0, SHWFS1])*self.convert

    def measure_with_noise_with_threshold(self, U_in, Nphot, sigma_readout, SDThreshold):
        """Measure tilt. U must not be zero-padded. In number of photons."""
        self.U_pupil = np.pad(U_in * self.pupil, ((self.pad_in,self.pad_in),(self.pad_in,self.pad_in)), 'constant', constant_values=0.)
        self.U_focal = np.fft.ifftshift(self.FF_BACKWARD(np.fft.ifftshift(
                        self.Q2 * np.fft.fftshift(self.FF_FORWARD(np.fft.fftshift(
                            self.Q1 * self.U_pupil
                            )))
                        )))
        self.Im = np.abs(self.U_focal[self.cut_out:self.N-self.cut_out,self.cut_out:self.N-self.cut_out])**2

        self.Im = self.Im * Nphot/self.Im.sum()
        self.Im = (np.random.poisson(self.Im)
                    + np.random.normal(0, sigma_readout, size=(self.N, self.N)))
        background = np.std(self.Im)
        self.Im = self.Im - background*SDThreshold
        self.Im[self.Im < 0] = 0.

        Im_sum = np.sum(self.Im)
        SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum
        SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum

        # plt.imshow(self.Im)
        # plt.colorbar()
        # plt.show()
        # plt.clf()

        return np.asarray([SHWFS0, SHWFS1])*self.convert

class WrappedGTilt():
    """Wrapped G tilt."""

    def __init__(self, N):
        self.phase = np.zeros((N, N), dtype=float)
        self.gx = np.zeros((N - 1, N), dtype=float)
        self.gy = np.zeros((N, N - 1), dtype=float)
        pupil = zernikeArray(1, N)[0]
        self.pupilx = pupil[1 : ,   : ] * pupil[ : -1, :   ]
        self.pupily = pupil[  : , 1 : ] * pupil[ :   , : -1]
        self.sumx = np.sum(self.pupilx)
        self.sumy = np.sum(self.pupily)
        del pupil

    def measure(self, U):
        """Measure tilt in rad/lenslet."""
        self.phase = np.angle(U)

        self.gx = self.phase[1 : ,   : ] - self.phase[ : -1, :   ]
        self.gy = self.phase[  : , 1 : ] - self.phase[ :   , : -1]
        
        self.gx[self.gx >  np.pi] = self.gx[self.gx >  np.pi] - 2*np.pi
        self.gx[self.gx < -np.pi] = self.gx[self.gx < -np.pi] + 2*np.pi
        self.gy[self.gy >  np.pi] = self.gy[self.gy >  np.pi] - 2*np.pi
        self.gy[self.gy < -np.pi] = self.gy[self.gy < -np.pi] + 2*np.pi

        return np.asarray([np.sum(self.gx*self.pupilx)/self.sumx, np.sum(self.gy*self.pupily)/self.sumy])*U.shape[0]

class ZTilt():
    """Z tilt. For unwrapped phase only."""

    def __init__(self, N):
        self.N = N
        Z_3 = zernikeArray(3, self.N)
        self.pupil = Z_3[0].flatten()
        self.tip = Z_3[1].flatten()
        self.tilt = Z_3[2].flatten()
        self.NORM= np.dot(self.pupil.flatten(), self.pupil.flatten())
        self.phase = np.zeros((self.N, self.N), dtype=float)
        del Z_3

    def measure_from_phase(self, unwrapped_phase):
        """Measure tilt."""
        self.phase = unwrapped_phase
        slopex = (np.dot(self.phase.flatten()*self.pupil, self.tilt)
                / self.NORM)*(4/self.N)
        slopey = (np.dot(self.phase.flatten()*self.pupil, self.tip)
                / self.NORM)*(4/self.N)
        return np.asarray([slopex, slopey])

# class ConsecutiveAngularPropSHWFS():
#     """
#     SHWFS with Multi-step angular propagation.

#     Not yet circular aperture.
    
#     Output angular scale can be adjusted.
#     Input to output pixel ratio is flexible.
    
#     Need proper implementation.
#     """

#     def __init__(self, n_in, d_in, alpha_out, FoV_out, wvl):
#         """N is for the zero-padded data. Is not yet written."""
#         self.n_in = int(n_in)
#         self.d_in = float(d_in)
#         self.alpha_out = float(alpha_out*wvl/d_in)
#         self.FoV_out = float(FoV_out*wvl/d_in)
#         self.wvl = float(wvl)

#         self.n_out = int(np.ceil(FoV_out/alpha_out))
#         self.delta_in = float(d_in/n_in)
        
#         self.delta_out = self.delta_in ### arbitrary choise ###
        
#         self.focal_length = self.delta_out / self.alpha_out
#         self.convert = ((self.delta_out)/self.focal_length)*(self.n_in*self.delta_in)*(2*np.pi/self.wvl)

#         self.wavenumber = 2*np.pi/self.wvl
#         self.N = int((self.n_in + self.n_out)) # schimidt#2
#         self.N = int(2**np.ceil(np.log(self.N)/np.log(2)))
#         self.z_max = self.N*self.delta_in**2/self.wvl
#         self.nprop = int(np.ceil(self.focal_length/self.z_max))
#         # print(self.nprop)
#         self.z = self.focal_length/self.nprop

#         if (self.N < self.wvl*self.focal_length/self.delta_in**2):
#             if (np.isclose(self.N, self.wvl*self.focal_length/self.delta_in**2, rtol=0.1) == False):
#                 raise Exception('Aliasing Problem.')

#         self.U_pupil = np.zeros((self.N, self.N), dtype=complex)
#         self.Im = np.zeros((self.n_out, self.n_out), dtype=float)
#         (self.ywfs, self.xwfs) = np.meshgrid(np.arange(self.n_out) + 0.5 - self.n_out/2,
#                                    np.arange(self.n_out) + 0.5 - self.n_out/2)
#         self.pad_in = int((self.N - self.n_in)/2)
#         self.cut_out = int((self.N - self.n_out)/2)

#         (nx, ny) = np.meshgrid(
#             np.arange(-self.N/2, self.N/2),
#             np.arange(-self.N/2, self.N/2))
#         self.nsq = nx**2 + ny**2
#         (nx_in, ny_in) = np.meshgrid(
#             np.arange(-self.n_in/2, self.n_in/2),
#             np.arange(-self.n_in/2, self.n_in/2))
#         nsq_in = nx_in**2 + ny_in**2


#         # circular
#         # self.sg = np.exp(-nsq**8/(0.41*self.N/2)**16)
#         # square # not 0.82, is 0.8 or 0.7, not 0.68 0.75?
#         self.sg = np.exp(-(((np.abs(nx) >= np.abs(ny))*np.abs(nx) + (np.abs(nx) < np.abs(ny))*np.abs(ny)))**16/(0.75*self.N/2)**16)

#         #Quadratic phase factors
#         # self.Q0 = (np.exp((-1j*self.wavenumber*self.delta_in*self.delta_out/2/self.focal_length)*(nx + ny))
#         #            * np.exp((-1j*self.wavenumber*self.delta_in**2/2./self.focal_length)*nsq))
#         # self.Q1 = np.exp((1j*self.wavenumber/2/self.focal_length)*nsq*((1 - self.delta_ip1[0]/self.delta_i[0])*self.delta_in**2))
        
#         self.Q1 = (np.exp((-1j*self.wavenumber*self.delta_in*self.delta_out/2/self.focal_length)*(nx_in + ny_in))
#                     * np.exp((-1j*self.wavenumber*self.delta_in**2/2./self.focal_length)*nsq_in))
#         self.Q2 = np.exp(-1j*np.pi**2*2*self.z/self.wavenumber*self.nsq/self.N**2/self.delta_in**2)

#         self.F_in = pyfftw.empty_aligned((self.N, self.N), dtype=complex)
#         self.F_out = pyfftw.empty_aligned((self.N, self.N), dtype=complex)
#         self.FF_FORWARD = pyfftw.FFTW(self.F_in, self.F_out, axes=(-1,-2), direction='FFTW_FORWARD')
#         self.FF_BACKWARD = pyfftw.FFTW(self.F_in, self.F_out, axes=(-1,-2), direction='FFTW_BACKWARD')

#         del nx,ny,nx_in,ny_in,nsq_in

#     def measure_no_noise_no_threshold(self, U_in):
#         """Measure tilt. U must not be zero-padded."""
#         self.U_pupil = np.pad(self.Q1*U_in,
#                               ((self.pad_in,self.pad_in),
#                                (self.pad_in,self.pad_in)),
#                               'constant', constant_values=0.)
#         for i in np.arange(self.nprop):
#             self.U_pupil = self.sg * np.fft.ifftshift(self.FF_BACKWARD(np.fft.ifftshift(
#                             self.Q2 * np.fft.fftshift(self.FF_FORWARD(np.fft.fftshift(
#                                 self.U_pupil
#                                 )))
#                             )))
#         self.Im = np.abs(self.U_pupil[self.cut_out:self.N-self.cut_out,self.cut_out:self.N-self.cut_out])**2
#         self.Im = self.Im/self.Im.sum()
#         SHWFS0 = np.sum(self.Im * self.xwfs)
#         SHWFS1 = np.sum(self.Im * self.ywfs)
        
#         # plt.imshow(self.Im)
#         # plt.colorbar()
#         # plt.show()
#         # plt.clf()
        
#         return np.asarray([SHWFS0, SHWFS1])*self.convert
    
#     def measure_no_noise_with_threshold(self, U_in, MaxNormThreshold):
#         """Measure tilt. U must not be zero-padded. Normalized to max I."""
#         self.U_pupil = np.pad(self.Q1*U_in,
#                               ((self.pad_in,self.pad_in),
#                                (self.pad_in,self.pad_in)),
#                               'constant', constant_values=0.)
#         for i in np.arange(self.nprop):
#             self.U_pupil = self.sg * np.fft.ifftshift(self.FF_BACKWARD(np.fft.ifftshift(
#                             self.Q2 * np.fft.fftshift(self.FF_FORWARD(np.fft.fftshift(
#                                 self.U_pupil
#                                 )))
#                             )))
#         self.Im = np.abs(self.U_pupil[self.cut_out:self.N-self.cut_out,self.cut_out:self.N-self.cut_out])**2

#         self.Im = (self.Im/self.Im.max() - MaxNormThreshold)
        
#         self.Im[self.Im < 0] = 0.
#         self.Im = self.Im / self.Im.max()

#         Im_sum = np.sum(self.Im)
#         SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum
#         SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum
        
#         # plt.imshow(self.Im)
#         # plt.colorbar()
#         # plt.show()
#         # plt.clf()

#         return np.asarray([SHWFS0, SHWFS1])*self.convert

#     def measure_with_noise_no_threshold(self, U_in, Nphot, sigma_readout):
#         """Measure tilt. U must not be zero-padded."""
#         self.U_pupil = np.pad(self.Q1*U_in,
#                               ((self.pad_in,self.pad_in),
#                                (self.pad_in,self.pad_in)),
#                               'constant', constant_values=0.)
#         for i in np.arange(self.nprop):
#             self.U_pupil = self.sg * np.fft.ifftshift(self.FF_BACKWARD(np.fft.ifftshift(
#                             self.Q2 * np.fft.fftshift(self.FF_FORWARD(np.fft.fftshift(
#                                 self.U_pupil
#                                 )))
#                             )))
#         self.Im = np.abs(self.U_pupil[self.cut_out:self.N-self.cut_out,self.cut_out:self.N-self.cut_out])**2

#         self.Im = self.Im * Nphot/self.Im.sum()
#         self.Im = (np.random.poisson(self.Im)
#                    + np.random.normal(0, sigma_readout, size=(self.n_out, self.n_out)))
#         self.Im[self.Im < 0] = 0.

#         Im_sum = np.sum(self.Im)
#         SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum
#         SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum
        
#         # plt.imshow(self.Im)
#         # plt.colorbar()
#         # plt.show()
#         # plt.clf()

#         return np.asarray([SHWFS0, SHWFS1])*self.convert

#     def measure_with_noise_with_threshold(self, U_in, Nphot, sigma_readout, Threshold):
#         """Measure tilt. U must not be zero-padded. In number of photons."""
#         self.U_pupil = np.pad(self.Q1*U_in,
#                               ((self.pad_in,self.pad_in),
#                                (self.pad_in,self.pad_in)),
#                               'constant', constant_values=0.)
#         for i in np.arange(self.nprop):
#             self.U_pupil = self.sg * np.fft.ifftshift(self.FF_BACKWARD(np.fft.ifftshift(
#                             self.Q2 * np.fft.fftshift(self.FF_FORWARD(np.fft.fftshift(
#                                 self.U_pupil
#                                 )))
#                             )))
#         self.Im = np.abs(self.U_pupil[self.cut_out:self.N-self.cut_out,self.cut_out:self.N-self.cut_out])**2

#         self.Im = self.Im * Nphot/self.Im.sum()
#         self.Im = (np.random.poisson(self.Im)
#                    + np.random.normal(0, sigma_readout, size=(self.n_out, self.n_out))
#                    - Threshold)
#         self.Im[self.Im < 0] = 0.

#         Im_sum = np.sum(self.Im)
#         SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum
#         SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum

#         # plt.imshow(self.Im)
#         # plt.colorbar()
#         # plt.show()
#         # plt.clf()

#         return np.asarray([SHWFS0, SHWFS1])*self.convert