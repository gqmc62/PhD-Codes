from __future__ import print_function

"""Propagate."""

import pyfftw
import numpy as np
# from matplotlib import pyplot as plt

# from matplotlib import pyplot as plt



# class ConsecutivePropSquareV2():
#     """
#     test
    
#     don't really work somehow due to confusion in propagation criteria!!!
#     propblem start after applying q2
#     i believe there is a magnification problem.
#     because when use this version but with constant delta it is still fine
#     only when applied magnification it fails.
#     somehow after q2 things got arranged strangely.
#     tbd
#     just don't use it. even if it behave accordingly size gain = 1/2 per axis AT MAX
#     ahh might be better still though...
    
#     i found an error; not in this method; but more of criteria for this method to work
#     so schmidt did not add curvature in eqn 7.54 in a similar way to 7.48
#     the additional curvature is the fourier transform of previous curvature ...
#     Schmidt only consider for curvature in Q2 separately than the stacks
    
#     say Uin has no curvature
#     Q1 curvature is z/(1 - m1)
#     Q1*Uin curvature is z/(1 - m1)
#     F{Q1*Uin} curvature is (from fourier transform table)
#         is -(1 - m1)/wvl**2/z
#     Q2 curvature is -m1/wvl**2/z
#     Q2*F{Q1*Uin} curvature is 1/(1/-m1 + 1/-(1-m1))/wvl**2/z
#         = -m1*(1-m1)/wvl**2/z
#         ==> N*deltaf1/2/wvl/abs(curvature) < 1/2/deltaf1
#         ==> N*deltaf1**2 < abs(curvature)*wvl
#         deltaf1 = 1/N/delta1
#         ==> N/N/N/delta1**2 < abs(-m1*(1-m1))/wvl/wvl/z * wvl
#         ==> 1/N/delta1**2 < abs(-m1*(1-m1))/wvl/z
#         ==> wvl*z/delta1**2 < abs(-m1*(1-m1)) * N
    
#     """

#     def __init__(self, N, wvl, Z, n_prop, delta1, delta2):
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
#         nsq = nx**2 + ny**2
        
#         self.sg = np.exp(-(((np.abs(nx) >= np.abs(ny))*np.abs(nx) + (np.abs(nx) < np.abs(ny))*np.abs(ny)))**16/(1*self.N/2)**16)
        
#         self.wvl = float(wvl)
#         wavenumber = float(2*np.pi/self.wvl)

#         self.delta1 = float(delta1)
#         self.delta2 = float(delta2)

#         del nx, ny
        
#         self.Z = float(Z)

#         partial_z = float(Z/n_prop)
#         if self.Z == 0:
#             return
#         if (((n_prop%1) != 0) or (n_prop <= 0)):
#             raise Exception('n_prop must be integer')
#         self.prop_number = int(n_prop)
        
#         alpha = np.arange(self.prop_number + 1)/self.prop_number
#         self.delta_list = (1 - alpha)*self.delta1 + alpha*self.delta2
#         self.mag_list = self.delta_list[1:]/self.delta_list[:-1]
#         self.deltaf_list = 1 / (self.N*self.delta_list[:-1])
        
#         self.Q1 = np.exp(1j*wavenumber/2.*(1. - self.mag_list[0])/partial_z*nsq*self.delta_list[0]**2)
#         self.Q3 = np.exp(1j*wavenumber/2.*(self.mag_list[-1] - 1.)/(self.mag_list[-1]*partial_z)*nsq*self.delta_list[-1]**2)
#         self.premade_Q2 = -np.pi**2*2.*partial_z/wavenumber*nsq
#         self.Q2_conversion = self.deltaf_list**2/self.mag_list
        
#         del nsq


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
#         plt.imshow(np.angle(U_in));plt.title('input');plt.show()
#         plt.imshow(np.log(np.abs(U_in)));plt.title('input');plt.show()
#         U_out = U_in * self.Q1
#         plt.imshow(np.angle(self.Q1));plt.title('Q1');plt.show()
#         plt.imshow(np.log(np.abs(self.Q1)));plt.title('Q1');plt.show()
#         plt.imshow(np.angle(U_out));plt.title('post Q1');plt.show()
#         plt.imshow(np.log(np.abs(U_out)));plt.title('post Q1');plt.show()
#         for i in np.arange(self.prop_number):
#             U_out = self.sg * \
#                 self.FF_BACKWARD(
#                     np.fft.ifftshift(
#                         np.exp(1j*self.premade_Q2*self.Q2_conversion[i]) * np.fft.fftshift(
#                             self.FF_FORWARD(
#                                 U_out/self.mag_list[i]
#                                 )# *self.sg # no sg here!!!
#                             , axes=(-1,-2))
#                         , axes=(-1,-2))
#                     )
#             temp = np.fft.fftshift(
#                 self.FF_FORWARD(
#                     U_out/self.mag_list[i]
#                     )# *self.sg # no sg here!!!
#                 , axes=(-1,-2))
#             plt.imshow(np.angle(temp));plt.title('pre Q2 {:d}'.format(i));plt.show()
#             plt.imshow(np.log(np.abs(temp)));plt.title('pre Q2 {:d}'.format(i));plt.show()
#             plt.imshow(np.angle(np.exp(1j*self.premade_Q2*self.Q2_conversion[i])));plt.title('Q2 {:d}'.format(i));plt.show()
#             plt.imshow(np.log(np.abs(np.exp(1j*self.premade_Q2*self.Q2_conversion[i]))));plt.title('Q2 {:d}'.format(i));plt.show()
#             temp = np.exp(1j*self.premade_Q2*self.Q2_conversion[i]) * np.fft.fftshift(
#                 self.FF_FORWARD(
#                     U_out/self.mag_list[i]
#                     )# *self.sg # no sg here!!!
#                 , axes=(-1,-2))
#             plt.imshow(np.angle(temp));plt.title('post Q2 {:d}'.format(i));plt.show()
#             plt.imshow(np.log(np.abs(temp)));plt.title('post Q2 {:d}'.format(i));plt.show()
#             plt.imshow(np.angle(U_out));plt.title('intermediat step {:d}'.format(i));plt.show()
#             plt.imshow(np.log(np.abs(U_out)));plt.title('intermediat step {:d}'.format(i));plt.show()
#         U_out *= self.Q3
#         plt.imshow(np.angle(self.Q3));plt.title('Q3');plt.show()
#         plt.imshow(np.log(np.abs(self.Q3)));plt.title('Q3');plt.show()
#         plt.imshow(np.angle(U_out));plt.title('post Q3');plt.show()
#         plt.imshow(np.log(np.abs(U_out)));plt.title('post Q3');plt.show()
#         return U_out

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

    def __init__(self, wvl, delta, N, Z, nprop, alpha_max=None):
        """
        First initialization.
        
        N is the padded size.
        """
        if alpha_max == None:
            alpha_max = wvl/2/delta
        self.alpha_max = float(alpha_max)
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
        # square # not 0.82, is 0.8 or 0.7, not 0.68 0.75 0.88?
        # latest value that work? is 0.75
        # work 0.66?
        # 0.47 from schmidt is kinda good though? 1/10 by 'pupil' radius
        # or 0.71 for 1 + 1/2 radius
        
        self.sg = np.exp(-(((np.abs(nx) >= np.abs(ny))*np.abs(nx)
                            + (np.abs(nx) < np.abs(ny))*np.abs(ny)))**16
                         /(0.71*self.N/2)**16)
        
        self.wvl = float(wvl)
        self.wavenumber = float(2*np.pi/self.wvl)

        self.delta = float(delta)
        self.df = 1. / (self.N*self.delta)
        self.fsq = nsq*self.df**2
        self.fsq[np.abs(self.fsq) > (self.alpha_max/self.wvl)**2] = 0

        del nsq, nx, ny

        self.Z = float(Z)
        if self.Z == 0:
            return
        # # self.z_max = self.delta**2*self.N/self.wvl # this is the original
        # self.z_max = np.min([self.delta**2*self.N/self.wvl,1./4./self.wvl*self.delta]) # this is addition
        # self.prop_number = int(np.ceil(self.Z/self.z_max))
        self.prop_number = int(nprop)
        self.z = self.Z/float(self.prop_number)
        # if self.N < self.z*self.wvl/self.delta**2:
        #     raise Exception('cannot be propagate')
        #     return
        self.Q2 = np.exp(-1j*(np.pi**2*2*self.z/self.wavenumber) * self.fsq)

    # def new_wvl(self, wvl):
    #     """Change wvl and reinitiate related values."""
    #     self.wvl = float(wvl)
    #     self.wavenumber = float(2*np.pi/wvl)
    #     self.new_Z(self, self.Z)

    # def new_delta(self, delta):
    #     """Change delta and reinitiate related values."""
    #     old_delta = self.delta
    #     self.delta = float(delta)
    #     self.df = 1. / (self.N*self.delta)
    #     self.fsq = self.fsq*(self.delta/old_delta)**2
    #     self.new_Z(self, self.Z)
    
    # def new_Z(self, Z):
    #     """Change Z and related values."""
    #     self.Z = float(Z)
    #     if self.Z == 0:
    #         return
    #     self.z_max = self.delta**2*self.N/self.wvl
    #     self.prop_number = int(np.ceil(self.Z/self.z_max))
    #     self.z = self.Z/float(self.prop_number)
    #     if self.N < self.z*self.wvl/self.delta**2:
    #         raise Exception('cannot be propagate')
    #         return
    #     self.Q2 = np.exp(-1j*(np.pi**2*2*self.z/self.wavenumber) * self.fsq)

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
            U_out = self.sg * \
                self.FF_BACKWARD(
                    np.fft.ifftshift(
                        self.Q2 * np.fft.fftshift(
                            self.FF_FORWARD(
                                U_out
                                )# *self.sg # no sg here!!!
                            , axes=(-1,-2))
                        , axes=(-1,-2))
                    )
            # plt.imshow(np.angle(U_out));plt.title('intermediat step {:d}'.format(i));plt.show()
            # plt.imshow(np.log(np.abs(U_out)));plt.title('intermediat step {:d}'.format(i));plt.show()
        return U_out
    
    # def propv2(self, U_in):
    #     """
    #     Propagate under the initiated conditions with different input; with addition sg.

    #     The input must be double-padded
    #     AND zero-padded before putting in the class.

    #     i.e.: Want 64 px output after propagation in free space.
    #           128 px of turbulated phase screen should be generated.
    #           then padded with 128 px of zeros, resulting in 256 px.?
    #     """
    #     if self.Z == 0:
    #         return U_in

    #     U_out = U_in # * self.sg no sg here
    #     for i in np.arange(self.prop_number):
    #         U_out = self.sg * \
    #             self.FF_BACKWARD(
    #                 np.fft.ifftshift(
    #                     self.Q2 * np.fft.fftshift(
    #                         self.FF_FORWARD(
    #                             U_out
    #                             )# *self.sg # no sg here!!!
    #                         , axes=(-1,-2))
    #                     , axes=(-1,-2))
    #                 )
    #     return U_out

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
        # was 0.41
        self.sg = np.exp(-nsq**8/(0.45*self.N)**16)
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
            U_out = self.sg * \
                self.FF_BACKWARD(
                    np.fft.ifftshift(
                        self.Q2 * np.fft.fftshift(
                            self.FF_FORWARD(
                                U_out
                                )
                            , axes=(-1,-2))
                        , axes=(-1,-2))
                    )
                

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

        U_out = \
            self.FF_BACKWARD(
                np.fft.ifftshift(
                    self.Q2 * np.fft.fftshift(
                        self.FF_FORWARD(
                            U_in
                            )
                        , axes=(-1,-2))
                    , axes=(-1,-2))
                )
            

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
        self.N
        if self.Z == 0:
            return

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
        self.df = 1. / (self.N*self.delta)
        self.fsq = nsq*self.df**2
        del nsq, nx, ny
        
        self.Q2 = np.exp(-1j*(np.pi**2*2*self.Z/self.wavenumber) * self.fsq)
        

    def prop(self, U_in):
        """Prop."""
        if self.Z == 0:
            return U_in

        U_out = \
            self.FF_BACKWARD(
                np.fft.ifftshift(
                    self.Q2 * np.fft.fftshift(
                        self.FF_FORWARD(U_in)
                        , axes=(-1,-2))
                    , axes=(-1,-2))
                )
            

        return U_out

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

        U_out = \
            self.FF_BACKWARD(
                np.fft.ifftshift(
                    self.Q2 * np.fft.fftshift(
                        self.FF_FORWARD(
                            U_in
                            )
                        , axes=(-1,-2))
                    , axes=(-1,-2))
                )
            

        return U_out