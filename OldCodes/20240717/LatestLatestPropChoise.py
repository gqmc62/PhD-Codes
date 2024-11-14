# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:26:27 2023

@author: gqmc62
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

latest propagation criteria check

"""



import numpy as np
import mypropagationclass as mpc
from matplotlib import pyplot as plt
from matplotlib import colors
from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov


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

def next_2_power(n):
    N = int(2**(np.ceil(np.log(n)/np.log(2))))
    return N

def solve_quadratic(a,b,c):
    x1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    x2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    return [x1,x2]

r0 = 0.05
L0 = 10
wvl = 0.5e-6
D2_o = 0.8
delta2 = 0.05/16
# output_pixel = 16
# delta2 = D2/output_pixel
z_list = [20e3]#(np.arange(10) + 1)*5e3#

MAX_N = 4096



print('source without pupil')
for z in z_list:
    
    # # source without pupil
    
    delta1 = delta2 # doing 1:1 scale; just because
    
    D2 = D2_o# + 2*wvl*z/r0
    
    D1 = D2*delta1/delta2 + 4*wvl*z/r0 #  + wvl*z/delta2 # cond 1
    
    N = int(2**np.ceil(1 + np.log(np.max([D1/delta1,D2/delta2]))/np.log(2))) # get N # cond 0
    
    D1 = delta1*next_2_power(D1/delta1)
                                    
    input_pixel = int(np.ceil(D1/delta1)) # based on input and output pixels
    output_pixel = int(np.ceil(D2/delta2))
    if (input_pixel % 2) == 1 :
        input_pixel += 1
    if (output_pixel % 2) == 1 :
        output_pixel += 1
    
    
    
    if (N//output_pixel) > 16: # this means that only to get 1 output pixel we are using 256 pixel!!!
        print('We are wasting calculation power')
        # quit()

    
    # if input_pixel > N: # trivial but mostly forgotten and mistaken
    #     print('N is too small for input size')
    # if output_pixel > N:
    #     print('N is too small for output size')
    
    suggested_partial_z1 = (N*delta2 - D1*delta2/2./delta1 - D2/2.)/(wvl/2./delta1) # no wrapping allowed in D2 # cond 2
    suggested_partial_z2 = N*np.min([delta1,delta2])**2/wvl # no aliasing allowed in angular-spectrum transform # cond 4
    suggested_partial_z3 = (N*delta2/4.)/(wvl/delta1)
    suggested_partial_z = np.min([z,suggested_partial_z1,suggested_partial_z2,suggested_partial_z3]) # the lowest boundary
    
    nprop = int(np.ceil(z/suggested_partial_z))*2

    # print('suggest {:d} propagation'.format(nprop))
    
    partial_z = z/nprop
    
    print(z,wvl,D2,delta2,D1,delta1,N,partial_z)
    
    cond0 = (N >= D2/delta2) and (N >= D1/delta1)
    A = D2
    B = N*delta2 - 2.*wvl*partial_z/2./delta1
    # B = D1*delta2/delta1 - 2.*wvl*z/2./delta1
    cond1 = ((A <= B) or np.isclose(A,B,rtol=1e-3))
    cond2 = (N >= D1/2./delta1 + D2/2./delta2 + wvl*partial_z/2./delta1/delta2)
    cond3 = (np.abs(1 - delta2/delta1)*D1 <= wvl*z/delta1)
    cond4 = (N >= wvl*partial_z/np.min([delta1,delta2])**2)
    if not cond0:
        print('test0: simulation size canNOT accommodate input and output')
    if not cond1:
        print('test1: currently NOT all output come from source')#***
    if not cond2:
        print('test2: IS wrapping')
    if not cond3:
        print('test3: IS aliasing in Q1')#***
    if not cond4:
        print('test4: IS aliasing in Q2')

    if N < MAX_N :
        
        
        try:
            testobj2
            if ((testobj2.N != N)
                or (testobj2.wvl != wvl)
                or (testobj2.delta != delta1)
                or (testobj2.Z != z)
                or (testobj2.prop_number != nprop)):
                testobj2 = mpc.ConsecutivePropSquare(wvl, delta1, N, z, nprop, 4*wvl/r0/2)
    
        except:
            testobj2 = mpc.ConsecutivePropSquare(wvl, delta1, N, z, nprop, 4*wvl/r0/2)
    
        
        # try:
        #     fullphasescreen2
        #     if ((fullphasescreen2.nx_size != N)
        #         or (fullphasescreen2.pixel_scale != delta1)
        #         or (fullphasescreen2.r0 != r0)
        #         or (fullphasescreen2.L0 != L0)):
        #         fullphasescreen2 = PhaseScreenKolmogorov(N, delta1, r0, L0)
        # except:
        #     fullphasescreen2 = PhaseScreenKolmogorov(N, delta1, r0, L0)
        #     for row in np.arange(N):
        #         fullphasescreen2.add_row()
                
        try:
            fullphasescreen2
            if ((fullphasescreen2.nx_size != input_pixel)
                or (fullphasescreen2.pixel_scale != delta1)
                or (fullphasescreen2.r0 != r0)
                or (fullphasescreen2.L0 != L0)):
                fullphasescreen2 = PhaseScreenKolmogorov(input_pixel, delta1, r0, L0, random_seed=999)
        except:
            fullphasescreen2 = PhaseScreenKolmogorov(input_pixel, delta1, r0, L0, random_seed=999)
            for row in np.arange(input_pixel):
                fullphasescreen2.add_row()
                
        # input_pixel_show = int(D2//delta1)
        # if input_pixel_show > input_pixel :
        #     input_pixel_show = int(input_pixel)
        # output_pixel_show = int(D2//delta2)
        # if output_pixel_show > output_pixel :
        #     output_pixel_show = int(output_pixel)
        
        # input_pixel_show = N
        # output_pixel_show = N
        
        # input_pixel_show = output_pixel
        # output_pixel_show = output_pixel
        
        input_pixel_show = input_pixel
        output_pixel_show = output_pixel
        
        # prop forward
        
        screen = np.zeros((N,N),dtype=complex)
        screen[N//2 - input_pixel//2 : N//2 + input_pixel//2,
                N//2 - input_pixel//2 : N//2 + input_pixel//2] = np.exp(1j*fullphasescreen2.scrn)
        ground_from_full = testobj2.prop(screen)
        
        plt.imshow(np.angle(screen[N//2 - input_pixel_show//2
                                                            : N//2 + input_pixel_show//2,
                                                            N//2 - input_pixel_show//2
                                                            : N//2 + input_pixel_show//2]),
                    cmap='twilight_shifted',norm = colors.CenteredNorm())
        plt.colorbar()
        plt.title('source without pupil; z={:.0f}km origin phase'.format(z/1000))
        plt.show()
        
        ground_from_full = testobj2.prop(screen)
        plt.imshow(np.angle(ground_from_full[N//2 - output_pixel_show//2
                                              : N//2 + output_pixel_show//2,
                                              N//2 - output_pixel_show//2
                                              : N//2 + output_pixel_show//2]),
                    cmap='twilight_shifted',norm = colors.CenteredNorm(vcenter=0,halfrange=np.pi))
        plt.colorbar()
        plt.title('source without pupil; z={:.0f}km propagated phase'.format(z/1000))
        plt.show()
        
        plt.imshow(np.log(np.abs(ground_from_full[N//2 - output_pixel_show//2
                                              : N//2 + output_pixel_show//2,
                                              N//2 - output_pixel_show//2
                                              : N//2 + output_pixel_show//2])),
                    cmap='RdBu_r',norm = colors.CenteredNorm(vcenter=0,halfrange=np.sqrt(0.6)))
        plt.colorbar()
        plt.title('source without pupil; z={:.0f}km propagated intensity'.format(z/1000))
        plt.show()
        
        
        # prop back
        
        try:
            testobj3
            if ((testobj3.N != N)
                or (testobj3.wvl != wvl)
                or (testobj3.delta != delta1)
                or (testobj3.Z != z)
                or (testobj3.prop_number != nprop)):
                testobj3 = mpc.ConsecutivePropSquare(wvl, delta1, N, -z, nprop, 4*wvl/r0)
    
        except:
            testobj3 = mpc.ConsecutivePropSquare(wvl, delta1, N, -z, nprop, 4*wvl/r0)
        
        # pupil_ground = np.zeros_like(ground_from_full)
        # pupil_ground[N//2 - output_pixel//2
        #              : N//2 + output_pixel//2,
        #              N//2 - output_pixel//2
        #              : N//2 + output_pixel//2] \
        #     = ground_from_full[N//2 - output_pixel//2
        #                        : N//2 + output_pixel//2,
        #                        N//2 - output_pixel//2
        #                        : N//2 + output_pixel//2]
        
        pupil_ground = ground_from_full
        
        back_to_sky = testobj3.prop(pupil_ground)
        plt.imshow(np.angle(back_to_sky[N//2 - input_pixel_show//2
                                              : N//2 + input_pixel_show//2,
                                              N//2 - input_pixel_show//2
                                              : N//2 + input_pixel_show//2]),
                    cmap='twilight_shifted',norm = colors.CenteredNorm(vcenter=0,halfrange=np.pi))
        plt.colorbar()
        plt.title('source without pupil; z={:.0f}km return phase'.format(z/1000))
        plt.show()
        
        plt.imshow(np.log(np.abs(back_to_sky[N//2 - input_pixel_show//2
                                              : N//2 + input_pixel_show//2,
                                              N//2 - input_pixel_show//2
                                              : N//2 + input_pixel_show//2])),
                    cmap='RdBu_r',norm = colors.CenteredNorm(vcenter=0,halfrange=np.sqrt(0.6)))
        plt.colorbar()
        plt.title('source without pupil; z={:.0f}km return intensity'.format(z/1000))
        plt.show()
    else:
        print('We cannot afford this much memory')


# print('source with pupil')    
# for z in z_list:
        
#     # source without pupil
    
#     delta1 = delta2 # doing 1:1 scale; just because
    
#     D2 = D2_o + 2*wvl*z/r0
#     D1 = D2
    
#     input_pixel = int(np.ceil(D1/delta1)) # based on input and output pixels
#     output_pixel = int(np.ceil(D2/delta2))
#     if (input_pixel % 2) == 1 :
#         input_pixel += 1
#     if (output_pixel % 2) == 1 :
#         output_pixel += 1
#     N = int(2**np.ceil(1 + np.log(np.max([input_pixel,output_pixel]))/np.log(2))) # get N # cond 0
    
#     if (N//output_pixel) > 16: # this means that only to get 1 output pixel we are using 256 pixel!!!
#         print('We are wasting calculation power')
#         # quit()
#     if N >= MAX_N :
#         print('We cannot afford this much memory')
#         break
    
#     suggested_partial_z1 = (N*delta2 - D1*delta2/2./delta1 - D2/2.)/(wvl/2./delta1)
#     suggested_partial_z2 = N*np.min([delta1,delta2])**2/wvl
    
#     suggested_partial_z = np.min([z,suggested_partial_z1,suggested_partial_z2])
    
#     nprop = int(np.ceil(z/suggested_partial_z))

#     partial_z = z/nprop
    
#     print(z,wvl,D2,delta2,D1,delta1,N,partial_z)
    
#     cond0 = (N >= D2/delta2) and (N >= D1/delta1)
#     # cond1 = (D2 <= N*delta2 - 2.*wvl*partial_z/2./delta1)
#     cond2 = (N >= D1/2./delta1 + D2/2./delta2 + wvl*partial_z/2./delta1/delta2)
#     cond3 = (np.abs(1 - delta2/delta1)*D1 <= wvl*z/delta1)
#     cond4 = (N >= wvl*partial_z/np.min([delta1,delta2])**2)
#     if not cond0:
#         print('test0: simulation size canNOT accommodate input and output')
#     # if not cond1:
#     #     print('test1: currently NOT all output come from source')#***
#     if not cond2:
#         print('test2: IS wrapping')
#     if not cond3:
#         print('test3: IS aliasing in Q1')#***
#     if not cond4:
#         print('test4: IS aliasing in Q2')
    
    
#     try:
#         testobj2
#         if ((testobj2.N != N)
#             or (testobj2.wvl != wvl)
#             or (testobj2.delta != delta1)
#             or (testobj2.Z != z)
#             or (testobj2.prop_number != nprop)):
#             testobj2 = mpc.ConsecutivePropSquare(wvl, delta1, N, z, nprop)

#     except:
#         testobj2 = mpc.ConsecutivePropSquare(wvl, delta1, N, z, nprop)

    
#     try:
#         fullphasescreen2
#         if ((fullphasescreen2.nx_size != input_pixel)
#             or (fullphasescreen2.pixel_scale != delta1)
#             or (fullphasescreen2.r0 != r0)
#             or (fullphasescreen2.L0 != L0)):
#             fullphasescreen2 = PhaseScreenKolmogorov(input_pixel, delta1, r0, L0)
#     except:
#         fullphasescreen2 = PhaseScreenKolmogorov(input_pixel, delta1, r0, L0)
#         for row in np.arange(input_pixel):
#             fullphasescreen2.add_row()
            
#     screen = np.zeros((N,N),dtype=complex)
#     screen[N//2 - input_pixel//2 : N//2 + input_pixel//2,
#             N//2 - input_pixel//2 : N//2 + input_pixel//2] = np.exp(1j*fullphasescreen2.scrn)
#     ground_from_full = testobj2.prop(screen)
            
#     # input_pixel_show = int(D2//delta1)
#     # if input_pixel_show > input_pixel :
#     #     input_pixel_show = int(input_pixel)
#     # output_pixel_show = int(D2//delta2)
#     # if output_pixel_show > output_pixel :
#     #     output_pixel_show = int(output_pixel)
    
#     # input_pixel_show = N
#     # output_pixel_show = N
    
#     input_pixel_show = output_pixel
#     output_pixel_show = output_pixel
    
#     # prop forward
    
    
#     plt.imshow(np.angle(screen[N//2 - input_pixel_show//2
#                                                         : N//2 + input_pixel_show//2,
#                                                         N//2 - input_pixel_show//2
#                                                         : N//2 + input_pixel_show//2]),
#                 cmap='twilight_shifted',norm = colors.CenteredNorm())
#     plt.colorbar()
#     plt.title('source with pupil; z={:.0f}km origin phase'.format(z/1000))
#     plt.show()
    
#     plt.imshow(np.angle(ground_from_full[N//2 - output_pixel_show//2
#                                           : N//2 + output_pixel_show//2,
#                                           N//2 - output_pixel_show//2
#                                           : N//2 + output_pixel_show//2]),
#                 cmap='twilight_shifted',norm = colors.CenteredNorm(vcenter=0,halfrange=np.pi))
#     plt.colorbar()
#     plt.title('source with pupil; z={:.0f}km propagated phase'.format(z/1000))
#     plt.show()
    
#     plt.imshow(np.log(np.abs(ground_from_full[N//2 - output_pixel_show//2
#                                           : N//2 + output_pixel_show//2,
#                                           N//2 - output_pixel_show//2
#                                           : N//2 + output_pixel_show//2])),
#                 cmap='RdBu_r',norm = colors.CenteredNorm(vcenter=0,halfrange=np.sqrt(0.6)))
#     plt.colorbar()
#     plt.title('source with pupil; z={:.0f}km propagated intensity'.format(z/1000))
#     plt.show()
    
    
#     # prop back
    
#     try:
#         testobj3
#         if ((testobj3.N != N)
#             or (testobj3.wvl != wvl)
#             or (testobj3.delta != delta1)
#             or (testobj3.Z != z)
#             or (testobj3.prop_number != nprop)):
#             testobj3 = mpc.ConsecutivePropSquare(wvl, delta1, N, -z, nprop)

#     except:
#         testobj3 = mpc.ConsecutivePropSquare(wvl, delta1, N, -z, nprop)
    
#     # pupil_ground = np.zeros_like(ground_from_full)
#     # pupil_ground[N//2 - output_pixel//2
#     #               : N//2 + output_pixel//2,
#     #               N//2 - output_pixel//2
#     #               : N//2 + output_pixel//2] \
#     #     = ground_from_full[N//2 - output_pixel//2
#     #                         : N//2 + output_pixel//2,
#     #                         N//2 - output_pixel//2
#     #                         : N//2 + output_pixel//2]
    
#     pupil_ground = ground_from_full
    
#     back_to_sky = testobj3.prop(pupil_ground)
#     plt.imshow(np.angle(back_to_sky[N//2 - input_pixel_show//2
#                                           : N//2 + input_pixel_show//2,
#                                           N//2 - input_pixel_show//2
#                                           : N//2 + input_pixel_show//2]),
#                 cmap='twilight_shifted',norm = colors.CenteredNorm(vcenter=0,halfrange=np.pi))
#     plt.colorbar()
#     plt.title('source with pupil; z={:.0f}km return phase'.format(z/1000))
#     plt.show()
    
#     plt.imshow(np.log(np.abs(back_to_sky[N//2 - input_pixel_show//2
#                                           : N//2 + input_pixel_show//2,
#                                           N//2 - input_pixel_show//2
#                                           : N//2 + input_pixel_show//2])),
#                 cmap='RdBu_r',norm = colors.CenteredNorm(vcenter=0,halfrange=np.sqrt(0.6)))
#     plt.colorbar()
#     plt.title('source with pupil; z={:.0f}km return intensity'.format(z/1000))
#     plt.show()
    
    
    
    
    
    
    
    
# # # print('source with pupil')    
# # # for z in z_list:
    
    
# # #     # source with pupil
    
    
    
    
# # #     delta1 = delta2 # doing 1:1 scale; just because
    
# # #     D2 = D2_o# + 2*wvl*z/r0
# # #     D1 = D2
    
# # #     input_pixel = int(np.ceil(D1/delta1)) # based on input and output pixels
# # #     output_pixel = int(np.ceil(D2/delta2))
# # #     if (input_pixel % 2) == 1 :
# # #         input_pixel += 1
# # #     if (output_pixel % 2) == 1 :
# # #         output_pixel += 1
# # #     N = int(2**np.ceil(1 + np.log(np.max([input_pixel,output_pixel]))/np.log(2))) # get N # cond 0
    
# # #     if (N//output_pixel) > 16: # this means that only to get 1 output pixel we are using 256 pixel!!!
# # #         print('We are wasting calculation power')
# # #         # quit()
# # #     if N >= MAX_N :
# # #         print('We cannot afford this much memory')
# # #         break
    
# # #     suggested_partial_z1 = (N*delta2 - D1*delta2/2./delta1 - D2/2.)/(wvl/2./delta1)
# # #     suggested_partial_z2 = N*np.min([delta1,delta2])**2/wvl
    
# # #     suggested_partial_z = np.min([z,suggested_partial_z1,suggested_partial_z2])
    
# # #     nprop = int(np.ceil(z/suggested_partial_z))

# # #     partial_z = z/nprop
    
# # #     print(z,wvl,D2,delta2,D1,delta1,N,partial_z)
    
# # #     cond0 = (N >= D2/delta2) and (N >= D1/delta1)
# # #     # cond1 = (D2 <= N*delta2 - 2.*wvl*partial_z/2./delta1)
# # #     cond2 = (N >= D1/2./delta1 + D2/2./delta2 + wvl*partial_z/2./delta1/delta2)
# # #     cond3 = (np.abs(1 - delta2/delta1)*D1 <= wvl*z/delta1)
# # #     cond4 = (N >= wvl*partial_z/np.min([delta1,delta2])**2)
# # #     if not cond0:
# # #         print('test0: simulation size canNOT accommodate input and output')
# # #     # if not cond1:
# # #     #     print('test1: currently NOT all output come from source')#***
# # #     if not cond2:
# # #         print('test2: IS wrapping')
# # #     if not cond3:
# # #         print('test3: IS aliasing in Q1')#***
# # #     if not cond4:
# # #         print('test4: IS aliasing in Q2')
    
    
# # #     try:
# # #         testobj2
# # #         if ((testobj2.N != N)
# # #             or (testobj2.wvl != wvl)
# # #             or (testobj2.delta != delta1)
# # #             or (testobj2.Z != z)
# # #             or (testobj2.prop_number != nprop)):
# # #             testobj2 = mpc.ConsecutivePropSquare(wvl, delta1, N, z, nprop)

# # #     except:
# # #         testobj2 = mpc.ConsecutivePropSquare(wvl, delta1, N, z, nprop)

    
# # #     try:
# # #         fullphasescreen2
# # #         if ((fullphasescreen2.nx_size != input_pixel)
# # #             or (fullphasescreen2.pixel_scale != delta1)
# # #             or (fullphasescreen2.r0 != r0)
# # #             or (fullphasescreen2.L0 != L0)):
# # #             fullphasescreen2 = PhaseScreenKolmogorov(input_pixel, delta1, r0, L0)
# # #     except:
# # #         fullphasescreen2 = PhaseScreenKolmogorov(input_pixel, delta1, r0, L0)
# # #         for row in np.arange(input_pixel):
# # #             fullphasescreen2.add_row()
            
# # #     # input_pixel_show = input_pixel
# # #     # output_pixel_show = output_pixel
    
# # #     # input_pixel_show = N
# # #     # output_pixel_show = N
    
# # #     input_pixel_show = 64
# # #     output_pixel_show = 64
    
# # #     # forward prop
    
# # #     screen = np.zeros((N,N),dtype=complex)
# # #     screen[N//2 - input_pixel//2 : N//2 + input_pixel//2,
# # #            N//2 - input_pixel//2 : N//2 + input_pixel//2] = np.exp(1j*fullphasescreen2.scrn)
# # #     ground_from_full = testobj2.prop(screen)
    
# # #     plt.imshow(np.angle(screen[N//2 - input_pixel//2 : N//2 + input_pixel//2,
# # #            N//2 - input_pixel//2 : N//2 + input_pixel//2]),
# # #                 cmap='twilight_shifted',norm = colors.CenteredNorm())
# # #     plt.colorbar()
# # #     plt.title('source with pupil; z={:.0f}km origin phase'.format(z/1000))
# # #     plt.show()
    
    
# # #     plt.imshow(np.angle(ground_from_full[N//2 - output_pixel_show//2
# # #                                           : N//2 + output_pixel_show//2,
# # #                                           N//2 - output_pixel_show//2
# # #                                           : N//2 + output_pixel_show//2]),
# # #                 cmap='twilight_shifted',norm = colors.CenteredNorm(vcenter=0,halfrange=np.pi))
# # #     plt.colorbar()
# # #     plt.title('source with pupil; z={:.0f}km propagated phase'.format(z/1000))
# # #     plt.show()
    
# # #     plt.imshow(np.log(np.abs(ground_from_full[N//2 - output_pixel_show//2
# # #                                           : N//2 + output_pixel_show//2,
# # #                                           N//2 - output_pixel_show//2
# # #                                           : N//2 + output_pixel_show//2])),
# # #                 cmap='RdBu_r',norm = colors.CenteredNorm(vcenter=0,halfrange=np.sqrt(0.6)))
# # #     plt.colorbar()
# # #     plt.title('source with pupil; z={:.0f}km propagated intensity'.format(z/1000))
# # #     plt.show()
    
# # #     # prop back
    
# # #     try:
# # #         testobj3
# # #         if ((testobj3.N != N)
# # #             or (testobj3.wvl != wvl)
# # #             or (testobj3.delta != delta1)
# # #             or (testobj3.Z != z)
# # #             or (testobj3.prop_number != nprop)):
# # #             testobj3 = mpc.ConsecutivePropSquare(wvl, delta1, N, -z, nprop)

# # #     except:
# # #         testobj3 = mpc.ConsecutivePropSquare(wvl, delta1, N, -z, nprop)
    
    
    
# # #     back_to_sky = testobj3.prop(ground_from_full)
# # #     plt.imshow(np.angle(back_to_sky[N//2 - output_pixel_show//2
# # #                                           : N//2 + output_pixel_show//2,
# # #                                           N//2 - output_pixel_show//2
# # #                                           : N//2 + output_pixel_show//2]),
# # #                 cmap='twilight_shifted',norm = colors.CenteredNorm(vcenter=0,halfrange=np.pi))
# # #     plt.colorbar()
# # #     plt.title('source without pupil; z={:.0f}km return phase'.format(z/1000))
# # #     plt.show()
    
# # #     plt.imshow(np.log(np.abs(back_to_sky[N//2 - output_pixel_show//2
# # #                                           : N//2 + output_pixel_show//2,
# # #                                           N//2 - output_pixel_show//2
# # #                                           : N//2 + output_pixel_show//2])),
# # #                 cmap='RdBu_r',norm = colors.CenteredNorm(vcenter=0,halfrange=np.sqrt(0.6)))
# # #     plt.colorbar()
# # #     plt.title('source without pupil; z={:.0f}km return intensity'.format(z/1000))
# # #     plt.show()
    