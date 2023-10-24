
"""Propagate."""

import pyfftw
import numpy as np

# from matplotlib import pyplot as plt

class ConsecutivePropSquare():
    """
    Propagation after 1 turbulence layer at pupil plane only.

    Best for propagation of the same grid number repetitively.
    Parameters crucial to propagation can be changed repetitively,
    except for the grid size, since it is related to the memory size.

    The input must be 2x zero-padded before putting in the class.
    Ex:    0000
        ab 0ab0  only ab  is usable
        cd 0cd0       cd
           0000
    """

    def __init__(self, wvl, delta, N, Z):
        """
        First initialization.
        
        N is the padded size.
        """
        self.N = int(N)
        self.F_in = pyfftw.empty_aligned((self.N, self.N), dtype=complex)
        self.F_out = pyfftw.empty_aligned((self.N, self.N), dtype=complex)
        self.FF_FORWARD = pyfftw.FFTW(
            self.F_in, self.F_out,axes=(-1, -2), direction='FFTW_FORWARD')
        self.FF_BACKWARD = pyfftw.FFTW(
            self.F_in, self.F_out, axes=(-1, -2), direction='FFTW_BACKWARD')
        (nx, ny) = np.meshgrid(
            np.arange(-self.N/2, self.N/2),
            np.arange(-self.N/2, self.N/2))
        nsq = nx**2 + ny**2
        # circular
        # self.sg = np.exp(-nsq**8/(0.47*self.N/2)**16)
        # square # not 0.82, is 0.8 or 0.7, not 0.68 0.75?
        self.sg = np.exp(-(((np.abs(nx) >= np.abs(ny))*np.abs(nx) + (np.abs(nx) < np.abs(ny))*np.abs(ny)))**16/(0.75*self.N/2)**16)
        
        self.wvl = float(wvl)
        self.wavenumber = float(2*np.pi/self.wvl)

        self.delta = float(delta)
        self.df = 1. / (self.N*self.delta)
        self.fsq = nsq*self.df**2

        del nsq, nx, ny

        self.Z = float(Z)
        if self.Z == 0:
            return
        self.z_max = self.delta**2*self.N/self.wvl
        self.prop_number = int(np.ceil(self.Z/self.z_max))
        self.z = self.Z/float(self.prop_number)
        if self.N < self.z*self.wvl/self.delta**2:
            raise Exception('cannot be propagate')
            return
        self.Q2 = np.exp(-1j*(np.pi**2*2*self.z/self.wavenumber) * self.fsq)

    def new_wvl(self, wvl):
        """Change wvl and reinitiate related values."""
        self.wvl = float(wvl)
        self.wavenumber = float(2*np.pi/wvl)
        self.new_Z(self, self.Z)

    def new_delta(self, delta):
        """Change delta and reinitiate related values."""
        old_delta = self.delta
        self.delta = float(delta)
        self.df = 1. / (self.N*self.delta)
        self.fsq = self.fsq*(self.delta/old_delta)**2
        self.new_Z(self, self.Z)
    
    def new_Z(self, Z):
        """Change Z and related values."""
        self.Z = float(Z)
        if self.Z == 0:
            return
        self.z_max = self.delta**2*self.N/self.wvl
        self.prop_number = int(np.ceil(self.Z/self.z_max))
        self.z = self.Z/float(self.prop_number)
        if self.N < self.z*self.wvl/self.delta**2:
            raise Exception('cannot be propagate')
            return
        self.Q2 = np.exp(-1j*(np.pi**2*2*self.z/self.wavenumber) * self.fsq)

    def prop(self, U_in):
        """
        Propagate under the initiated conditions with different input.

        The input must be double-padded
        AND zero-padded before putting in the class.

        i.e.: Want 64 px output after propagation in free space.
              128 px of turbulated phase screen should be generated.
              then padded with 128 px of zeros, resulting in 256 px.?
        """
        if self.Z == 0:
            return U_in

        U_out = U_in
        for i in np.arange(self.prop_number):
            U_out = self.sg * np.fft.ifftshift(
                self.FF_BACKWARD(
                    np.fft.ifftshift(
                        self.Q2 * np.fft.fftshift(
                            self.FF_FORWARD(
                                np.fft.fftshift(U_out, axes=(-1,-2))
                                )
                            , axes=(-1,-2))
                        , axes=(-1,-2))
                    )
                , axes=(-1,-2))

        return U_out

# class ConsecutivePropSquare2():
#     """
#     Propagation after 1 turbulence layer at pupil plane only.

#     Best for propagation of the same grid number repetitively.
#     Parameters crucial to propagation can be changed repetitively,
#     except for the grid size, since it is related to the memory size.

#     The input must be 2x zero-padded before putting in the class.
#     Ex:    0000
#         ab 0ab0  only ab  is usable
#         cd 0cd0       cd
#            0000
#     """

#     def __init__(self, wvl, delta_in, delta_out, N, Z):
#         """
#         First initialization.
        
#         N is the padded size.
#         """
#         self.N = int(N)
#         self.F_in = pyfftw.empty_aligned((self.N, self.N), dtype=complex)
#         self.F_out = pyfftw.empty_aligned((self.N, self.N), dtype=complex)
#         self.FF_FORWARD = pyfftw.FFTW(
#             self.F_in, self.F_out,axes=(-1, -2), direction='FFTW_FORWARD')
#         self.FF_BACKWARD = pyfftw.FFTW(
#             self.F_in, self.F_out, axes=(-1, -2), direction='FFTW_BACKWARD')
#         (nx, ny) = np.meshgrid(
#             np.arange(-self.N/2, self.N/2),
#             np.arange(-self.N/2, self.N/2))
#         self.nsq = nx**2 + ny**2
#         # circular
#         # self.sg = np.exp(-nsq**8/(0.47*self.N/2)**16)
#         # square # not 0.82, is 0.8 or 0.7, not 0.68 0.75?
#         self.sg = np.exp(-(((np.abs(nx) >= np.abs(ny))*np.abs(nx) + (np.abs(nx) < np.abs(ny))*np.abs(ny)))**16/(0.75*self.N/2)**16)
        
#         self.wvl = float(wvl)
#         self.wavenumber = float(2*np.pi/self.wvl)

#         self.delta_in = float(delta_in)
#         self.delta_out = float(delta_out)

#         del nx, ny

#         self.Z = float(Z)
#         if self.Z == 0:
#             return
#         self.z_max = np.min([self.delta_in,self.delta_out])**2*self.N/self.wvl
#         self.prop_number = int(np.ceil(self.Z/self.z_max))
#         self.z = self.Z/float(self.prop_number)
#         self.mag = self.delta_out/self.delta_in/self.prop_number
        
#         if self.N < self.z*self.wvl/self.delta_in/self.delta_out:
#             if (np.isclose(self.N,self.z*self.wvl/self.delta_in/self.delta_out,rtol=0.1) == False):
#                 raise Exception('cannot be propagate')
#                 return
        
#         self.Q1 = np.exp((1j*self.wavenumber/2*(1 - self.mag)/self.z*self.delta_in**2)*self.nsq)
#         self.Q2 = (np.exp((-1j*np.pi**2*2*self.z/self.mag/self.wavenumber) * self.nsq))
#         self.Q3 = np.exp((1j*self.wavenumber/2*(self.mag - 1)/self.mag/self.z*self.delta_out**2) * self.nsq)

#     def prop(self, U_in):
#         """
#         Propagate under the initiated conditions with different input.

#         The input must be double-padded
#         AND zero-padded before putting in the class.

#         i.e.: Want 64 px output after propagation in free space.
#               128 px of turbulated phase screen should be generated.
#               then padded with 128 px of zeros, resulting in 256 px.?
#         """
#         if self.Z == 0:
#             return U_in

#         U_out = U_in*self.Q1
#         for i in np.arange(self.prop_number):
#             delta_i = self.delta_in + i*((self.delta_out - self.delta_in)/(self.prop_number))
#             delta_f_i = 1. / (self.N*delta_i)
#             U_out = self.sg * np.fft.ifftshift(
#                 self.FF_BACKWARD(
#                     np.fft.ifftshift(
#                         (self.Q2**(delta_f_i**2)) * np.fft.fftshift(
#                             self.FF_FORWARD(
#                                 np.fft.fftshift(U_out/self.mag, axes=(-1,-2))
#                                 )
#                             , axes=(-1,-2))
#                         , axes=(-1,-2))
#                     )
#                 , axes=(-1,-2))
#             # plt.imshow(np.abs(U_out)**2)
#             # plt.colorbar()
#             # plt.show()
#             # plt.imshow(np.angle(U_out))
#             # plt.colorbar()
#             # plt.show()

#         return U_out*self.Q3

class ConsecutivePropCircle():
    """
    Propagation after 1 turbulence layer at pupil plane only.

    Best for propagation of the same grid number repetitively.
    Parameters crucial to propagation can be changed repetitively,
    except for the grid size, since it is related to the memory size.

    The input must be 2x zero-padded before putting in the class.
    Ex:    0000
        ab 0ab0  only ab  is usable
        cd 0cd0       cd
           0000
    """

    def __init__(self, wvl, delta, N, Z):
        """
        First initialization.
        
        N is the padded size.
        """
        self.N = int(N)
        self.F_in = pyfftw.empty_aligned((self.N, self.N), dtype=complex)
        self.F_out = pyfftw.empty_aligned((self.N, self.N), dtype=complex)
        self.FF_FORWARD = pyfftw.FFTW(
            self.F_in, self.F_out,axes=(-1, -2), direction='FFTW_FORWARD')
        self.FF_BACKWARD = pyfftw.FFTW(
            self.F_in, self.F_out, axes=(-1, -2), direction='FFTW_BACKWARD')
        (nx, ny) = np.meshgrid(
            np.arange(-self.N/2, self.N/2),
            np.arange(-self.N/2, self.N/2))
        nsq = nx**2 + ny**2
        # circular
        self.sg = np.exp(-nsq**8/(0.41*self.N)**16)
        # self.sg = np.exp(-nsq**8/(0.47*self.N)**16) original

        # self.sg = np.exp(-(((np.abs(nx) >= np.abs(ny))*np.abs(nx) + (np.abs(nx) < np.abs(ny))*np.abs(ny)))**16/(0.75*self.N/2)**16)
        
        self.wvl = float(wvl)
        self.wavenumber = float(2*np.pi/self.wvl)

        self.delta = float(delta)
        self.df = 1. / (self.N*self.delta)
        self.fsq = nsq*self.df**2

        del nsq, nx, ny

        self.Z = float(Z)
        if self.Z == 0:
            return
        self.z_max = self.delta**2*self.N/self.wvl
        self.prop_number = int(np.ceil(self.Z/self.z_max))
        self.z = self.Z/float(self.prop_number)
        if self.N < self.z*self.wvl/self.delta**2:
            raise Exception('cannot be propagate')
            return
        self.Q2 = np.exp(-1j*(np.pi**2*2*self.z/self.wavenumber) * self.fsq)

    def new_wvl(self, wvl):
        """Change wvl and reinitiate related values."""
        self.wvl = float(wvl)
        self.wavenumber = float(2*np.pi/wvl)
        self.new_Z(self, self.Z)

    def new_delta(self, delta):
        """Change delta and reinitiate related values."""
        old_delta = self.delta
        self.delta = float(delta)
        self.df = 1. / (self.N*self.delta)
        self.fsq = self.fsq*(self.delta/old_delta)**2
        self.new_Z(self, self.Z)
    
    def new_Z(self, Z):
        """Change Z and related values."""
        self.Z = float(Z)
        if self.Z == 0:
            return
        self.z_max = self.delta**2*self.N/self.wvl
        self.prop_number = int(np.ceil(self.Z/self.z_max))
        self.z = self.Z/float(self.prop_number)
        if self.N < self.z*self.wvl/self.delta**2:
            raise Exception('cannot be propagate')
            return
        self.Q2 = np.exp(-1j*(np.pi**2*2*self.z/self.wavenumber) * self.fsq)

    def prop(self, U_in):
        """
        Propagate under the initiated conditions with different input.

        The input must be double-padded
        AND zero-padded before putting in the class.

        i.e.: Want 64 px output after propagation in free space.
              128 px of turbulated phase screen should be generated.
              then padded with 128 px of zeros, resulting in 256 px.?
        """
        if self.Z == 0:
            return U_in

        U_out = U_in
        for i in np.arange(self.prop_number):
            U_out = self.sg * np.fft.ifftshift(
                self.FF_BACKWARD(
                    np.fft.ifftshift(
                        self.Q2 * np.fft.fftshift(
                            self.FF_FORWARD(
                                np.fft.fftshift(U_out, axes=(-1,-2))
                                )
                            , axes=(-1,-2))
                        , axes=(-1,-2))
                    )
                , axes=(-1,-2))

        return U_out

class OneAngularPropAtm():
    """
    One step angular propagation for free atm.
    
    N is the desired accurate output
    NN is the must have screen size
    NN/N is self.pad_factor
    """

    def __init__(self, wvl, delta, N, Z):
        self.wvl = float(wvl)
        self.wavenumber = float(2*np.pi/self.wvl)
        self.delta = float(delta)
        self.N = int(N)
        self.Z = float(Z)

        self.waste_px = (self.wvl*self.Z)/(self.delta**2)
        self.NN = self.N
        count = 0
        while ((self.NN < 2*self.waste_px)
               or (np.isclose(self.NN, 2*self.waste_px,rtol=0.1) == False)):
            count = count + 1
            self.NN = int(self.NN*2)
            if count > 16:
                raise Exception('This might be too large, or infinite loop.')
                break

        self.pad_factor = int(self.NN/self.N)

    def xtra_init(self):
        """Allocate proper memory."""
        self.F_in = pyfftw.empty_aligned((self.NN, self.NN), dtype=complex)
        self.F_out = pyfftw.empty_aligned((self.NN, self.NN), dtype=complex)
        self.FF_FORWARD = pyfftw.FFTW(
            self.F_in, self.F_out,axes=(-1, -2), direction='FFTW_FORWARD')
        self.FF_BACKWARD = pyfftw.FFTW(
            self.F_in, self.F_out, axes=(-1, -2), direction='FFTW_BACKWARD')
        (nx, ny) = np.meshgrid(
            np.arange(-self.NN/2, self.NN/2),
            np.arange(-self.NN/2, self.NN/2))
        nsq = nx**2 + ny**2
        self.df = 1. / (self.NN*self.delta)
        self.fsq = nsq*self.df**2
        del nsq, nx, ny

        if self.Z == 0:
            return
        if self.NN < self.Z*self.wvl/self.delta**2:
            raise Exception('cannot be propagate')
            return

        self.Q2 = np.exp(-1j*(np.pi**2*2*self.Z/self.wavenumber) * self.fsq)

    def prop(self, U_in):
        """Prop."""
        if self.Z == 0:
            return U_in

        U_out = np.fft.ifftshift(
            self.FF_BACKWARD(
                np.fft.ifftshift(
                    self.Q2 * np.fft.fftshift(
                        self.FF_FORWARD(
                            np.fft.fftshift(U_in, axes=(-1,-2))
                            )
                        , axes=(-1,-2))
                    , axes=(-1,-2))
                )
            , axes=(-1,-2))

        return U_out

class BlindOneAngularPropAtm():
    """
    One step angular propagation for free atm.
    
    N is the desired accurate output
    NN is the must have screen size
    NN/N is self.pad_factor
    """

    def __init__(self, wvl, delta, N, Z):
        self.wvl = float(wvl)
        self.wavenumber = float(2*np.pi/self.wvl)
        self.delta = float(delta)
        self.N = int(N)
        self.Z = float(Z)
        self.pad_factor = int(2)
        self.NN = self.pad_factor*self.N
        if self.Z == 0:
            return

        self.F_in = pyfftw.empty_aligned((self.NN, self.NN), dtype=complex)
        self.F_out = pyfftw.empty_aligned((self.NN, self.NN), dtype=complex)
        self.FF_FORWARD = pyfftw.FFTW(
            self.F_in, self.F_out,axes=(-1, -2), direction='FFTW_FORWARD')
        self.FF_BACKWARD = pyfftw.FFTW(
            self.F_in, self.F_out, axes=(-1, -2), direction='FFTW_BACKWARD')
        (nx, ny) = np.meshgrid(
            np.arange(-self.NN/2, self.NN/2),
            np.arange(-self.NN/2, self.NN/2))
        nsq = nx**2 + ny**2
        self.df = 1. / (self.NN*self.delta)
        self.fsq = nsq*self.df**2
        del nsq, nx, ny
        
        self.Q2 = np.exp(-1j*(np.pi**2*2*self.Z/self.wavenumber) * self.fsq)
        
        self.a = int(self.N/2)
        self.b = int(3*self.N/2)

    def prop(self, U_in):
        """Prop."""
        if self.Z == 0:
            return U_in

        U_out = np.fft.ifftshift(
            self.FF_BACKWARD(
                np.fft.ifftshift(
                    self.Q2 * np.fft.fftshift(
                        self.FF_FORWARD(
                            np.fft.fftshift(np.pad(U_in,
                             (self.a, self.a), 'constant', constant_values=0.), axes=(-1,-2))
                            )
                        , axes=(-1,-2))
                    , axes=(-1,-2))
                )
            , axes=(-1,-2))

        return U_out[self.a : self.b, self.a : self.b]

class OneAngularPropTele():
    """
    One step angular propagation for after telescope pupil.
    
    N is the desired accurate output
    NN is the must have screen size
    NN/N is self.pad_factor
    """

    def __init__(self, wvl, delta, N, Z):
        self.wvl = float(wvl)
        self.wavenumber = float(2*np.pi/self.wvl)
        self.delta = float(delta)
        self.N = int(N)
        self.Z = float(Z)

        self.waste_px = (self.wvl*self.Z)/(self.delta**2)/2
        self.NN = self.N
        count = 0
        while (self.NN < 2*self.waste_px):
            count = count + 1
            self.NN = int(self.NN*2)
            if count > 100:
                raise Exception('This might be too large, or infinite loop.')
                break

        self.pad_factor = int(self.NN/self.N)

    def xtra_init(self):
        """Allocate proper memory."""
        self.F_in = pyfftw.empty_aligned((self.NN, self.NN), dtype=complex)
        self.F_out = pyfftw.empty_aligned((self.NN, self.NN), dtype=complex)
        self.FF_FORWARD = pyfftw.FFTW(
            self.F_in, self.F_out,axes=(-1, -2), direction='FFTW_FORWARD')
        self.FF_BACKWARD = pyfftw.FFTW(
            self.F_in, self.F_out, axes=(-1, -2), direction='FFTW_BACKWARD')
        (nx, ny) = np.meshgrid(
            np.arange(-self.NN/2, self.NN/2),
            np.arange(-self.NN/2, self.NN/2))
        nsq = nx**2 + ny**2
        self.df = 1. / (self.NN*self.delta)
        self.fsq = nsq*self.df**2
        del nsq, nx, ny

        if self.Z == 0:
            return
        if self.NN < self.Z*self.wvl/self.delta**2:
            raise Exception('cannot be propagate')
            return

        self.Q2 = np.exp(-1j*(np.pi**2*2*self.Z/self.wavenumber) * self.fsq)

    def prop(self, U_in):
        """Prop."""
        if self.Z == 0:
            return U_in

        U_out = np.fft.ifftshift(
            self.FF_BACKWARD(
                np.fft.ifftshift(
                    self.Q2 * np.fft.fftshift(
                        self.FF_FORWARD(
                            np.fft.fftshift(U_in, axes=(-1,-2))
                            )
                        , axes=(-1,-2))
                    , axes=(-1,-2))
                )
            , axes=(-1,-2))

        return U_out