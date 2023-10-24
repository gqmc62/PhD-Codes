
"""Gather all wavefront sensor simulation and method."""

import numpy as np
from aotools.functions.zernike import zernikeArray
import pyfftw
from matplotlib import pyplot as plt

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

        del x1, y1

    def measure_no_noise_no_threshold(self, U):
        """Measure tilt. U must not be zero-padded."""
        self.Im = np.abs(np.fft.fftshift(
            self.FF01(np.pad(U * self.Q,
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

    def measure_no_noise_with_threshold(self, U, MaxNormThreshold):
        """Measure tilt. U must not be zero-padded. Normalized to max I."""
        self.Im = np.abs(np.fft.fftshift(
            self.FF01(np.pad(U * self.Q,
                             (self.a, self.a), 'constant', constant_values=0.))
            )[self.a : self.b, self.a : self.b])**2

        self.Im = (self.Im/self.Im.max() - MaxNormThreshold)
        
        self.Im[self.Im < 0] = 0.
        self.Im = self.Im / self.Im.max()

        Im_sum = np.sum(self.Im)
        SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum
        SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum
        
        # plt.imshow(self.Im)
        # plt.colorbar()
        # plt.show()
        # plt.clf()

        return np.asarray([SHWFS0, SHWFS1])*np.pi

    def measure_with_noise_no_threshold(self, U, Nphot, sigma_readout):
        """Measure tilt. U must not be zero-padded."""
        self.Im = np.abs(np.fft.fftshift(
            self.FF01(np.pad(U * self.Q,
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

    def measure_with_noise_with_threshold(self, U, Nphot, sigma_readout, Threshold):
        """Measure tilt. U must not be zero-padded. In number of photons."""
        self.Im = np.abs(np.fft.fftshift(
            self.FF01(np.pad(U * self.Q,
                             (self.a, self.a), 'constant', constant_values=0.))
            )[self.a : self.b, self.a : self.b])**2

        self.Im = self.Im * Nphot/self.Im.sum()
        self.Im = (np.random.poisson(self.Im)
                   + np.random.normal(0, sigma_readout, size=(self.N, self.N))
                   - Threshold)
        self.Im[self.Im < 0] = 0.

        Im_sum = np.sum(self.Im)
        SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum
        SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum

        # plt.imshow(self.Im)
        # plt.colorbar()
        # plt.show()
        # plt.clf()

        return np.asarray([SHWFS0, SHWFS1])*np.pi

class AngularPropSHWFS():
    """
    SHWFS with angular propagation.
    
    Output angular scale can be adjusted.
    Input to output pixel ratio is flexible.
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
        self.delta_out= self.delta_in
        self.focal_length = self.delta_out / self.alpha_out
        self.convert = ((self.delta_out)/self.focal_length)*(self.n_in*self.delta_in)*(2*np.pi/self.wvl)
        
        wavenumber = 2*np.pi/self.wvl
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

        (x1,y1) = np.meshgrid(self.delta_in*np.arange(-self.N/2,self.N/2),
                                 self.delta_in*np.arange(-self.N/2,self.N/2))
        r1sq = (x1**2 + y1**2) + 1e-10

        #Spatial Frequencies (of source plane)
        df1 = 1. / (self.N*self.delta_in)
        fX,fY = np.meshgrid(df1*np.arange(-self.N/2,self.N/2),
                               df1*np.arange(-self.N/2,self.N/2))
        fsq = fX**2 + fY**2

        #Scaling Param
        self.mag = float(self.delta_out/self.delta_in)

        #Observation Plane Co-ords
        x2,y2 = np.meshgrid( self.delta_out*np.arange(-self.N/2,self.N/2),
                                self.delta_out*np.arange(-self.N/2,self.N/2) )
        r2sq = x2**2 + y2**2

        #Quadratic phase factors
        self.Q1 = (np.exp( 1j * wavenumber/2. * (1-self.mag)/self.focal_length * r1sq)
                    * np.exp(-1j*wavenumber*((x1 + y1)*self.delta_out/2/self.focal_length))
                   * np.exp(-1j*wavenumber*r1sq/2./self.focal_length))
        self.Q2 = np.exp(-1j * np.pi**2 * 2 * self.focal_length/self.mag/wavenumber * fsq)
        self.Q3 = np.exp(1j * wavenumber/2. * (self.mag-1)/(self.mag*self.focal_length) * r2sq)
        
        del x1,y1,r1sq,df1,fX,fY,fsq,x2,y2,r2sq
        
        self.F_in = pyfftw.empty_aligned((self.N, self.N), dtype=complex)
        self.F_out = pyfftw.empty_aligned((self.N, self.N), dtype=complex)
        self.FF_FORWARD = pyfftw.FFTW(self.F_in, self.F_out, axes=(-1,-2), direction='FFTW_FORWARD')
        self.FF_BACKWARD = pyfftw.FFTW(self.F_in, self.F_out, axes=(-1,-2), direction='FFTW_BACKWARD')

    # def measure(self, U_in):
    #     """Measure tilt."""
    #     self.U_pupil = np.pad(U_in, ((self.pad_in,self.pad_in),(self.pad_in,self.pad_in)), 'constant', constant_values=0.)
    #     self.U_focal = self.Q3 * np.fft.ifftshift(self.FF_BACKWARD(np.fft.ifftshift(
    #                     self.Q2 * np.fft.fftshift(self.FF_FORWARD(np.fft.fftshift(
    #                         self.Q1 * self.U_pupil/self.mag
    #                         )))
    #                     )))
    #     self.Im = np.abs(self.U_focal[self.cut_out:self.N-self.cut_out,self.cut_out:self.N-self.cut_out])**2
    #     Im_sum = np.sum(self.Im)
    #     SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum
    #     SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum
        
    #     plt.imshow(self.Im)
    #     plt.grid(which='major',axis='both')
    #     plt.colorbar()
    #     plt.show()
        
    #     return np.asarray([SHWFS0, SHWFS1])*(self.convert)
    
    def measure_no_noise_no_threshold(self, U_in):
        """Measure tilt. U must not be zero-padded."""
        self.U_pupil = np.pad(U_in, ((self.pad_in,self.pad_in),(self.pad_in,self.pad_in)), 'constant', constant_values=0.)
        self.U_focal = self.Q3 * np.fft.ifftshift(self.FF_BACKWARD(np.fft.ifftshift(
                        self.Q2 * np.fft.fftshift(self.FF_FORWARD(np.fft.fftshift(
                            self.Q1 * self.U_pupil/self.mag
                            )))
                        )))
        self.Im = np.abs(self.U_focal[self.cut_out:self.N-self.cut_out,self.cut_out:self.N-self.cut_out])**2
        self.Im = self.Im/self.Im.sum()

        SHWFS0 = np.sum(self.Im * self.xwfs)
        SHWFS1 = np.sum(self.Im * self.ywfs)
        
        plt.imshow(self.Im)
        plt.colorbar()
        plt.show()
        plt.clf()

        return np.asarray([SHWFS0, SHWFS1])*self.convert

    def measure_no_noise_with_threshold(self, U_in, MaxNormThreshold):
        """Measure tilt. U must not be zero-padded. Normalized to max I."""
        self.U_pupil = np.pad(U_in, ((self.pad_in,self.pad_in),(self.pad_in,self.pad_in)), 'constant', constant_values=0.)
        self.U_focal = self.Q3 * np.fft.ifftshift(self.FF_BACKWARD(np.fft.ifftshift(
                        self.Q2 * np.fft.fftshift(self.FF_FORWARD(np.fft.fftshift(
                            self.Q1 * self.U_pupil/self.mag
                            )))
                        )))
        self.Im = np.abs(self.U_focal[self.cut_out:self.N-self.cut_out,self.cut_out:self.N-self.cut_out])**2

        self.Im = (self.Im/self.Im.max() - MaxNormThreshold)
        
        self.Im[self.Im < 0] = 0.
        self.Im = self.Im / self.Im.max()

        Im_sum = np.sum(self.Im)
        SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum
        SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum
        
        plt.imshow(self.Im)
        plt.colorbar()
        plt.show()
        plt.clf()

        return np.asarray([SHWFS0, SHWFS1])*self.convert

    def measure_with_noise_no_threshold(self, U_in, Nphot, sigma_readout):
        """Measure tilt. U must not be zero-padded."""
        self.U_pupil = np.pad(U_in, ((self.pad_in,self.pad_in),(self.pad_in,self.pad_in)), 'constant', constant_values=0.)
        self.U_focal = self.Q3 * np.fft.ifftshift(self.FF_BACKWARD(np.fft.ifftshift(
                        self.Q2 * np.fft.fftshift(self.FF_FORWARD(np.fft.fftshift(
                            self.Q1 * self.U_pupil/self.mag
                            )))
                        )))
        self.Im = np.abs(self.U_focal[self.cut_out:self.N-self.cut_out,self.cut_out:self.N-self.cut_out])**2

        self.Im = self.Im * Nphot/self.Im.sum()
        self.Im = (np.random.poisson(self.Im)
                   + np.random.normal(0, sigma_readout, size=(self.N, self.N)))
        self.Im[self.Im < 0] = 0.

        Im_sum = np.sum(self.Im)
        SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum
        SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum
        
        plt.imshow(self.Im)
        plt.colorbar()
        plt.show()
        plt.clf()

        return np.asarray([SHWFS0, SHWFS1])*self.convert

    def measure_with_noise_with_threshold(self, U_in, Nphot, sigma_readout, Threshold):
        """Measure tilt. U must not be zero-padded. In number of photons."""
        self.U_pupil = np.pad(U_in, ((self.pad_in,self.pad_in),(self.pad_in,self.pad_in)), 'constant', constant_values=0.)
        self.U_focal = self.Q3 * np.fft.ifftshift(self.FF_BACKWARD(np.fft.ifftshift(
                        self.Q2 * np.fft.fftshift(self.FF_FORWARD(np.fft.fftshift(
                            self.Q1 * self.U_pupil/self.mag
                            )))
                        )))
        self.Im = np.abs(self.U_focal[self.cut_out:self.N-self.cut_out,self.cut_out:self.N-self.cut_out])**2

        self.Im = self.Im * Nphot/self.Im.sum()
        self.Im = (np.random.poisson(self.Im)
                   + np.random.normal(0, sigma_readout, size=(self.N, self.N))
                   - Threshold)
        self.Im[self.Im < 0] = 0.

        Im_sum = np.sum(self.Im)
        SHWFS0 = np.sum(self.Im * self.xwfs)/Im_sum
        SHWFS1 = np.sum(self.Im * self.ywfs)/Im_sum

        plt.imshow(self.Im)
        plt.colorbar()
        plt.show()
        plt.clf()

        return np.asarray([SHWFS0, SHWFS1])*self.convert


class IgTilt():
    """Ig tilt."""

    def __init__(self, N):
        self.phase = np.zeros((N, N), dtype=float)
        self.I = np.zeros((N, N), dtype=float)
        self.gx = np.zeros((N, N), dtype=float)
        self.gy = np.zeros((N, N), dtype=float)

    def measure(self, U):
        """Measure tilt in rad/lenslet."""
        self.phase = np.angle(U)
        self.I = np.abs(U)**2
        I_sum = self.I.sum()
        
        self.gx = self.phase[1 : ,   : ] - self.phase[ : -1, :   ]
        self.gy = self.phase[  : , 1 : ] - self.phase[ :   , : -1]
        
        self.gx[self.gx >  np.pi] = self.gx[self.gx >  np.pi] - 2*np.pi
        self.gx[self.gx < -np.pi] = self.gx[self.gx < -np.pi] + 2*np.pi
        self.gy[self.gy >  np.pi] = self.gy[self.gy >  np.pi] - 2*np.pi
        self.gy[self.gy < -np.pi] = self.gy[self.gy < -np.pi] + 2*np.pi

        Ig0 = np.sum(self.gx
                     * (self.I[1 : ,   : ] + self.I[ : -1, :   ])/2) / I_sum
        Ig1 = np.sum(self.gy
                     * (self.I[  : , 1 : ] + self.I[ :   , : -1])/2) / I_sum

        return np.asarray([Ig0, Ig1])*U.shape[0]

    def measure_with_noise(self, U, Nphot):
        """Measure tilt in rad/lenslet."""
        self.phase = np.angle(U)
        self.I = np.abs(U)**2
        self.I = self.I * Nphot/self.I.sum()
        
        self.I = np.random.poisson(self.I)
        I_sum = self.I.sum()
        
        self.gx = self.phase[1 : ,   : ] - self.phase[ : -1, :   ]
        self.gy = self.phase[  : , 1 : ] - self.phase[ :   , : -1]
        
        self.gx[self.gx >  np.pi] = self.gx[self.gx >  np.pi] - 2*np.pi
        self.gx[self.gx < -np.pi] = self.gx[self.gx < -np.pi] + 2*np.pi
        self.gy[self.gy >  np.pi] = self.gy[self.gy >  np.pi] - 2*np.pi
        self.gy[self.gy < -np.pi] = self.gy[self.gy < -np.pi] + 2*np.pi

        Ig0 = np.sum(self.gx
                     * (self.I[1 : ,   : ] + self.I[ : -1, :   ])/2) / I_sum
        Ig1 = np.sum(self.gy
                     * (self.I[  : , 1 : ] + self.I[ :   , : -1])/2) / I_sum

        value = np.asarray([Ig0, Ig1])*U.shape[0]
        noise = 0.0 # placeholder for noises in the future

        return value + noise

    def measure_from_phase_I(self, phase, I):
        """Measure tilt in rad/lenslet."""
        self.phase = phase
        self.I = I
        I_sum = self.I.sum()
        
        self.gx = self.phase[1 : ,   : ] - self.phase[ : -1, :   ]
        self.gy = self.phase[  : , 1 : ] - self.phase[ :   , : -1]
        
        self.gx[self.gx >  np.pi] = self.gx[self.gx >  np.pi] - 2*np.pi
        self.gx[self.gx < -np.pi] = self.gx[self.gx < -np.pi] + 2*np.pi
        self.gy[self.gy >  np.pi] = self.gy[self.gy >  np.pi] - 2*np.pi
        self.gy[self.gy < -np.pi] = self.gy[self.gy < -np.pi] + 2*np.pi

        Ig0 = np.sum(self.gx
                     * (self.I[1 : ,   : ] + self.I[ : -1, :   ])/2) / I_sum
        Ig1 = np.sum(self.gy
                     * (self.I[  : , 1 : ] + self.I[ :   , : -1])/2) / I_sum

        return np.asarray([Ig0, Ig1])*phase.shape[0]

class UnwrappedIgTilt():
    """Unwrapped Ig tilt."""

    def __init__(self, N):
        self.phase = np.zeros((N, N), dtype=float)
        self.I = np.zeros((N, N), dtype=float)
        self.gx = np.zeros((N, N), dtype=float)
        self.gy = np.zeros((N, N), dtype=float)

    def measure(self, U):
        """Measure tilt in rad/lenslet."""
        self.phase = np.angle(U)
        self.I = np.abs(U)**2
        I_sum = self.I.sum()

        self.gx = self.phase[1 : ,   : ] - self.phase[ : -1, :   ]
        self.gy = self.phase[  : , 1 : ] - self.phase[ :   , : -1]

        Ig0 = np.sum(self.gx
                     * (self.I[1 : ,   : ] + self.I[ : -1, :   ])/2) / I_sum
        Ig1 = np.sum(self.gy
                     * (self.I[  : , 1 : ] + self.I[ :   , : -1])/2) / I_sum

        return np.asarray([Ig0, Ig1])*U.shape[0]

    def measure_from_phase_I(self, phase, I):
        """Measure tilt in rad/lenslet."""
        self.phase = phase
        self.I = I
        I_sum = self.I.sum()

        self.gx = self.phase[1 : ,   : ] - self.phase[ : -1, :   ]
        self.gy = self.phase[  : , 1 : ] - self.phase[ :   , : -1]

        Ig0 = np.sum(self.gx
                     * (self.I[1 : ,   : ] + self.I[ : -1, :   ])/2) / I_sum
        Ig1 = np.sum(self.gy
                     * (self.I[  : , 1 : ] + self.I[ :   , : -1])/2) / I_sum

        return np.asarray([Ig0, Ig1])*phase.shape[0]

class WrappedGTilt():
    """Wrapped G tilt."""

    def __init__(self, N):
        self.phase = np.zeros((N, N), dtype=float)
        self.gx = np.zeros((N, N), dtype=float)
        self.gy = np.zeros((N, N), dtype=float)

    def measure(self, U):
        """Measure tilt in rad/lenslet."""
        self.phase = np.angle(U)

        self.gx = self.phase[1 : ,   : ] - self.phase[ : -1, :   ]
        self.gy = self.phase[  : , 1 : ] - self.phase[ :   , : -1]
        
        self.gx[self.gx >  np.pi] = self.gx[self.gx >  np.pi] - 2*np.pi
        self.gx[self.gx < -np.pi] = self.gx[self.gx < -np.pi] + 2*np.pi
        self.gy[self.gy >  np.pi] = self.gy[self.gy >  np.pi] - 2*np.pi
        self.gy[self.gy < -np.pi] = self.gy[self.gy < -np.pi] + 2*np.pi

        return np.asarray([np.mean(self.gx), np.mean(self.gy)])
    
    def measure_from_phase(self, phase):
        """Measure tilt in rad/lenslet."""
        self.phase = phase

        self.gx = self.phase[1 : ,   : ] - self.phase[ : -1, :   ]
        self.gy = self.phase[  : , 1 : ] - self.phase[ :   , : -1]
        
        self.gx[self.gx >  np.pi] = self.gx[self.gx >  np.pi] - 2*np.pi
        self.gx[self.gx < -np.pi] = self.gx[self.gx < -np.pi] + 2*np.pi
        self.gy[self.gy >  np.pi] = self.gy[self.gy >  np.pi] - 2*np.pi
        self.gy[self.gy < -np.pi] = self.gy[self.gy < -np.pi] + 2*np.pi

        return np.asarray([np.mean(self.gx), np.mean(self.gy)])

class UnwrappedGTilt():
    """Unwrapped G tilt."""

    def __init__(self, N):
        self.phase = np.zeros((N, N), dtype=float)

    def measure(self, U):
        """Measure tilt in rad/lenslet."""
        self.phase = np.angle(U)
        return np.asarray([np.mean(self.phase[1 : ,   : ]
                                  - self.phase[ : -1, :   ]),
                           np.mean(self.phase[  : , 1 : ]
                                  - self.phase[ :   , : -1])])
    
    def measure_from_phase(self, phase):
        """Measure tilt in rad/lenslet."""
        self.phase = phase

        return np.asarray([np.mean(self.phase[1 : ,   : ]
                                  - self.phase[ : -1, :   ]),
                           np.mean(self.phase[  : , 1 : ]
                                  - self.phase[ :   , : -1])])

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
