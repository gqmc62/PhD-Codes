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




r0 = 0.05
L0 = 2

# source with pupil

wvl = 0.5e-6
delta1 = 0.05/16
D1 = 0.4

z_list = (np.arange(10) + 1)*5e3

for z in z_list:
    
    D2 = D1
    delta2 = delta1
    
    input_pixel = D1/delta1
    output_pixel = D2/delta2
    
    suggest_N = np.max([input_pixel,output_pixel]) * 2
    N = int(2**np.ceil(np.log(suggest_N)/np.log(2)))
    
    # if input_pixel > N:
    #     print('N is too small for input size')
    # if output_pixel > N:
    #     print('N is too small for output size')
    
    # max_grid_alpha = wvl/2./delta1
    # useless_alpha = (D1 + D2)/2./z
    # if max_grid_alpha > useless_alpha:
    #     print('need to crop the useless angular-spectrum.')
    # max_alpha = np.min([max_grid_alpha,useless_alpha])
    
    max_alpha = wvl/2./delta1
    
    suggested_partial_z1 = (N*delta2 - D1*delta2/2./delta1 - D2/2.)/max_alpha
    suggested_partial_z2 = N*np.min([delta1,delta2])**2/wvl
    
    suggested_partial_z = np.min([z,suggested_partial_z1,suggested_partial_z2])
    
    nprop = int(np.ceil(z/suggested_partial_z))
    
    if nprop > 1:
        print('suggest {:d} propagation'.format(nprop))
    
    partial_z = z/nprop
    
    screen_size = N*delta2
    illumination_area_with_wrapping = D1*delta2/2./delta1 + D2/2. + max_alpha*partial_z
    
    if screen_size < illumination_area_with_wrapping:
        print('contamination detected.')
    
    if N < wvl*partial_z/delta1/delta2:
        print('aliases in angular spectrum')


z_list = (np.arange(10) + 1)*5e3

for z in z_list:
    
    # source without pupil
    
    # high res
    
    wvl = 0.5e-6
    delta2 = 0.05/16
    D2 = 0.05
    # z = 10e3
    
    
    
    delta1 = delta2 # doing 1:1 scale; just because
    
    max_alpha = wvl/2./delta1
    creep = max_alpha*z # per side!
    D1 = delta1*2**(np.ceil(np.log((D2 + 2*creep)/delta1)/np.log(2))) # pick D1 that gives no creep inside D2
    
    input_pixel = D1/delta1 # based on input and output pixels
    output_pixel = D2/delta2
    
    N = int(2**np.ceil(np.log(np.max([input_pixel,output_pixel]))/np.log(2))) # get N
    
    if (N//output_pixel) > 16: # this means that only to get 1 output pixel we are using 256 pixel!!!
        print('We are wasting calculation power')
        # quit()
    
    if input_pixel > N: # trivial but mostly forgotten and mistaken
        print('N is too small for input size')
    if output_pixel > N:
        print('N is too small for output size')
    
    suggested_partial_z1 = (N*delta2 - D1*delta2/2./delta1 - D2/2.)/max_alpha # no wrapping allowed in D2
    suggested_partial_z2 = N*np.min([delta1,delta2])**2/wvl # no aliasing allowed in angular-spectrum transform
    
    suggested_partial_z = np.min([z,suggested_partial_z1,suggested_partial_z2]) # the lowest boundary
    
    nprop = int(np.ceil(z/suggested_partial_z))
    
    if nprop > 1:
        print('suggest {:d} propagation'.format(nprop))
    
    partial_z = z/nprop
    
    screen_size = N*delta2
    illumination_area_with_wrapping = D1*delta2/2./delta1 + D2/2. + max_alpha*partial_z # anti-complete-wrapping around screen
    
    if screen_size < illumination_area_with_wrapping:
        print('contamination detected.')
    if N < wvl*partial_z/delta1/delta2:
        print('aliases in angular spectrum')
    
    print(z,wvl,D2,delta2,D1,delta1,N)
    
    print(D2<=D1*delta2/delta1-wvl*z/delta1, 'test1: all output must come from source')
    print(N>=D1/2/delta1+D2/2/delta2+wvl*z/2/delta1/delta2, 'test2: no wrapping')
    print(np.abs(1-delta2/delta1)*D1<wvl*z/delta1, 'test3: no aliasing in forward ft')
    print(N>wvl*z/np.min([delta1,delta2])**2, 'test4: no aliasing in backward ft')
    
#     # print((D1*delta2/delta1+D2)/2/z>wvl/2/delta1)
    
#     try:
#         testobj
#         if ((testobj.N != N)
#             or (testobj.wvl != wvl)
#             or (testobj.delta != delta1)
#             or (testobj.Z != z)):
#             testobj = mpc.ConsecutivePropSquare(wvl, delta1, N, z)
#     except:
#         testobj = mpc.ConsecutivePropSquare(wvl, delta1, N, z)
    
#     try:
#         fullphasescreen
#         if ((fullphasescreen.nx_size != N)
#             or (fullphasescreen.pixel_scale != delta1)
#             or (fullphasescreen.r0 != r0)
#             or (fullphasescreen.L0 != L0)):
#             fullphasescreen = PhaseScreenKolmogorov(N, delta1, r0, L0)
#     except:
#         fullphasescreen = PhaseScreenKolmogorov(N, delta1, r0, L0)
#         for row in np.arange(N):
#             fullphasescreen.add_row()
        
#     plt.imshow(np.angle(np.exp(1j*fullphasescreen.scrn[N//2-8*2:N//2+8*2,N//2-8*2:N//2+8*2])),cmap='twilight_shifted',norm = colors.CenteredNorm())
#     plt.colorbar()
#     plt.title('z={:.0f}km origin phase'.format(z/1000))
#     plt.show()
    
#     ground_from_full = testobj.prop(np.exp(1j*fullphasescreen.scrn))
#     plt.imshow(np.angle(ground_from_full[N//2-8*2:N//2+8*2,N//2-8*2:N//2+8*2]),cmap='twilight_shifted',norm = colors.CenteredNorm())
#     plt.colorbar()
#     plt.title('z={:.0f}km propagated phase'.format(z/1000))
#     plt.show()
    
#     plt.imshow(np.log(np.abs(ground_from_full[N//2-8*2:N//2+8*2,N//2-8*2:N//2+8*2])),cmap='RdBu',norm = colors.CenteredNorm(vcenter=0,halfrange=1))
#     plt.colorbar()
#     plt.title('z={:.0f}km propagated intensity'.format(z/1000))
#     plt.show()
    
    
    
    
    
    
# #     # source without pupil
    
# #     wvl = 0.5e-6
# #     delta2 = 0.05/16
# #     D2 = 0.2
# #     # z = 10e3
    
    
    
# #     delta1 = delta2 # doing 1:1 scale; just because
    
# #     max_alpha = wvl/2./delta1
# #     creep = max_alpha*z # per side!
# #     D1 = delta1*2**(np.ceil(np.log((D2 + 2*creep)/delta1)/np.log(2))) # pick D1 that gives no creep inside D2
    
# #     input_pixel = D1/delta1 # based on input and output pixels
# #     output_pixel = D2/delta2
    
# #     N = int(2**np.ceil(np.log(np.max([input_pixel,output_pixel]))/np.log(2))) # get N
    
# #     if (N//output_pixel) > 16: # this means that only to get 1 output pixel we are using 256 pixel!!!
# #         print('We are wasting calculation power')
# #         # quit()
    
# #     if input_pixel > N: # trivial but mostly forgotten and mistaken
# #         print('N is too small for input size')
# #     if output_pixel > N:
# #         print('N is too small for output size')
    
# #     suggested_partial_z1 = (N*delta2 - D1*delta2/2./delta1 - D2/2.)/max_alpha # no wrapping allowed in D2
# #     suggested_partial_z2 = N*np.min([delta1,delta2])**2/wvl # no aliasing allowed in angular-spectrum transform
    
# #     suggested_partial_z = np.min([z,suggested_partial_z1,suggested_partial_z2]) # the lowest boundary
    
# #     nprop = int(np.ceil(z/suggested_partial_z))
    
# #     if nprop > 1:
# #         print('suggest {:d} propagation'.format(nprop))
    
# #     partial_z = z/nprop
    
# #     screen_size = N*delta2
# #     illumination_area_with_wrapping = D1*delta2/2./delta1 + D2/2. + max_alpha*partial_z # anti-complete-wrapping around screen
    
# #     if screen_size < illumination_area_with_wrapping:
# #         print('contamination detected.')
# #     if N < wvl*partial_z/delta1/delta2:
# #         print('aliases in angular spectrum')
    
# #     print(z,wvl,D2,delta2,D1,delta1,N)
    
# #     # print((D1*delta2/delta1+D2)/2/z>wvl/2/delta1)
    
# #     try:
# #         testobj
# #         if ((testobj.N != N)
# #             or (testobj.wvl != wvl)
# #             or (testobj.delta != delta1)
# #             or (testobj.Z != z)):
# #             testobj = mpc.ConsecutivePropSquare(wvl, delta1, N, z)
# #     except:
# #         testobj = mpc.ConsecutivePropSquare(wvl, delta1, N, z)
    
# #     # try:
# #     #     fullphasescreen
# #     #     if ((fullphasescreen.nx_size != N)
# #     #         or (fullphasescreen.pixel_scale != delta1)
# #     #         or (fullphasescreen.r0 != r0)
# #     #         or (fullphasescreen.L0 != L0)):
# #     #         fullphasescreen = PhaseScreenKolmogorov(N, delta1, r0, L0)
# #     # except:
# #     #     fullphasescreen = PhaseScreenKolmogorov(N, delta1, r0, L0)
# #     #     for row in np.arange(N):
# #     #         fullphasescreen.add_row()
    
# #     full_binned_screen = rebin(fullphasescreen.scrn,np.asarray(fullphasescreen.scrn.shape)//2)
    
# #     use_binned_screen = full_binned_screen[full_binned_screen.shape[0]//2 - N//2
# #                                            : full_binned_screen.shape[0]//2 + N//2,
# #                                            full_binned_screen.shape[0]//2 - N//2
# #                                            : full_binned_screen.shape[0]//2 + N//2]    
    
# #     plt.imshow(np.angle(np.exp(1j*use_binned_screen[N//2-8:N//2+8,N//2-8:N//2+8])),cmap='twilight_shifted',norm = colors.CenteredNorm())
# #     plt.colorbar()
# #     plt.title('z={:.0f}km origin phase'.format(z/1000))
# #     plt.show()
    
# #     ground_from_full = testobj.prop(np.exp(1j*use_binned_screen))
# #     plt.imshow(np.angle(ground_from_full[N//2-8:N//2+8,N//2-8:N//2+8]),cmap='twilight_shifted',norm = colors.CenteredNorm())
# #     plt.colorbar()
# #     plt.title('z={:.0f}km propagated phase'.format(z/1000))
# #     plt.show()
    
# #     plt.imshow(np.log(np.abs(ground_from_full[N//2-8:N//2+8,N//2-8:N//2+8])),cmap='RdBu',norm = colors.CenteredNorm(vcenter=0,halfrange=1))
# #     plt.colorbar()
# #     plt.title('z={:.0f}km propagated intensity'.format(z/1000))
# #     plt.show()
    
    
    
    
# # #     # source without pupil v2
    
# # #     wvl = 0.5e-6
# # #     delta2 = 0.05/16
# # #     D2 = 0.2
# # #     # z = 20e3
    
# # #     output_pixel = D2/delta2
# # #     N = int(2*2**np.ceil(np.log(output_pixel)/np.log(2))) # get N
    
# # #     delta1 = delta2 # doing 1:1 scale; just because
# # #     D1 = N*delta1
# # #     input_pixel = D1/delta1 # based on input and output pixels
    
# # #     max_alpha = wvl/2./delta1
    
# # #     if input_pixel > N: # trivial but mostly forgotten and mistaken
# # #         print('N is too small for input size')
# # #     if output_pixel > N:
# # #         print('N is too small for output size')
    
# # #     suggested_partial_z0 = (D1*delta2/delta1/2. - D2/2.)/max_alpha
# # #     suggested_partial_z1 = (N*delta2 - D1*delta2/2./delta1 - D2/2.)/max_alpha # no wrapping allowed in D2
# # #     suggested_partial_z2 = N*np.min([delta1,delta2])**2/wvl # no aliasing allowed in angular-spectrum transform
    
# # #     suggested_partial_z = np.min([z,suggested_partial_z1,suggested_partial_z2]) # the lowest boundary
    
# # #     nprop = int(np.ceil(z/suggested_partial_z))
    
# # #     if nprop > 1:
# # #         print('suggest {:d} propagation'.format(nprop))
    
# # #     partial_z = z/nprop
    
# # #     screen_size = N*delta2
# # #     illumination_area_with_wrapping = D1*delta2/2./delta1 + D2/2. + max_alpha*partial_z # anti-complete-wrapping around screen
    
# # #     if screen_size < illumination_area_with_wrapping:
# # #         print('contamination detected.')
# # #     if N < wvl*partial_z/delta1/delta2:
# # #         print('aliases in angular spectrum')
    
    
    
    
# # #     try:
# # #         testobj2
# # #         if ((testobj2.N != N)
# # #             or (testobj2.wvl != wvl)
# # #             or (testobj2.delta != delta1)
# # #             or (testobj2.Z != z)):
# # #             testobj2 = mpc.ConsecutivePropSquare(wvl, delta1, N, z)
# # #     except:
# # #         testobj2 = mpc.ConsecutivePropSquare(wvl, delta1, N, z)
    
# # #     try:
# # #         fullphasescreen2
# # #         if ((fullphasescreen2.nx_size != N)
# # #             or (fullphasescreen2.pixel_scale != delta1)
# # #             or (fullphasescreen2.r0 != r0)
# # #             or (fullphasescreen2.L0 != L0)):
# # #             fullphasescreen2 = PhaseScreenKolmogorov(N, delta1, r0, L0)
# # #     except:
# # #         fullphasescreen2 = PhaseScreenKolmogorov(N, delta1, r0, L0)
# # #         for row in np.arange(N):
# # #             fullphasescreen2.add_row()
        
# # #     plt.imshow(np.angle(np.exp(1j*fullphasescreen2.scrn[N//2-32:N//2+32,N//2-32:N//2+32])),cmap='twilight_shifted',norm = colors.CenteredNorm())
# # #     plt.colorbar()
# # #     plt.title('z={:.0f}km origin phase'.format(z/1000))
# # #     plt.show()
    
# # #     ground_from_full = testobj2.propv2(np.exp(1j*fullphasescreen2.scrn))
# # #     toplot = ground_from_full[N//2-32:N//2+32,N//2-32:N//2+32]
# # #     plt.imshow(np.angle(toplot),cmap='twilight_shifted',norm = colors.CenteredNorm())
# # #     plt.colorbar()
# # #     plt.title('z={:.0f}km propagated phase'.format(z/1000))
# # #     plt.show()
    
# # #     plt.imshow(np.log(np.abs(toplot)),cmap='RdBu',norm = colors.CenteredNorm(vcenter=0,halfrange=1))
# # #     plt.colorbar()
# # #     plt.title('z={:.0f}km propagated intensity'.format(z/1000))
# # #     plt.show()

# # # for i in np.arange(100):
# # #     for row in np.arange(N):
# # #         fullphasescreen2.add_row()
# # #     toplot = (fullphasescreen2.scrn) - (fullphasescreen2.scrn).mean()
# # #     plt.imshow(toplot,cmap='twilight_shifted',norm=colors.CenteredNorm())
# # #     plt.colorbar()
# # #     plt.title('z={:.0f}km origin phase'.format(z/1000))
# # #     plt.show()

    
    
#     # wvl = 0.5e-6
#     # z = 1e3
#     # D2 = 1
#     # delta2 = 0.05/16.
    
#     output_pixel = int(np.ceil(D2/delta2))
#     input_pixel = 2*output_pixel
#     N = input_pixel
    
#     mag = (N*delta2-D2)*delta2/wvl/z
#     delta1 = delta2/mag
#     D1 = D2/mag
    
    
    
    
    
#     suggested_partial_z1 = (N*delta2 - D1*delta2/2./delta1 - D2/2.)/(wvl/2/delta1) # no wrapping allowed in D2
#     suggested_partial_z2 = N*np.min([delta1,delta2])**2/wvl # no aliasing allowed in angular-spectrum transform
#     suggested_partial_z = np.min([z,suggested_partial_z1,suggested_partial_z2]) # the lowest boundary
    
#     nprop = int(np.ceil(z/suggested_partial_z))
    
#     if nprop > 1:
#         print('suggest {:d} propagation'.format(nprop))
    
#     partial_z = z/nprop
    
#     screen_size = N*delta2
#     illumination_area_with_wrapping = D1*delta2/2./delta1 + D2/2. + (wvl/2./delta1)*partial_z # anti-complete-wrapping around screen
    
#     if screen_size < illumination_area_with_wrapping:
#         print('contamination detected.')
#     if N < wvl*partial_z/delta1/delta2:
#         print('aliases in angular spectrum')
    
#     print(z,wvl,D2,delta2,D1,delta1,N)
    
    
    
#     testobj3 = mpc.ConsecutivePropSquareV2(N, wvl, z, nprop, delta1, delta2)
    

#     fullphasescreen3 = PhaseScreenKolmogorov(input_pixel, delta1, 0.2, L0)
#     for row in np.arange(input_pixel):
#         fullphasescreen3.add_row()
    
#     use_screen = np.pad(fullphasescreen3.scrn,
#                         pad_width = (((N - input_pixel)//2, (N - input_pixel)//2),
#                                       ((N - input_pixel)//2, (N - input_pixel)//2)),
#                         mode='constant',
#                         constant_values=0)
    
#     plt.imshow(np.angle(np.exp(1j*use_screen[N//2-8:N//2+8,N//2-8:N//2+8])),cmap='twilight_shifted',norm = colors.CenteredNorm())
#     plt.colorbar()
#     plt.title('z={:.0f}km origin phase'.format(z/1000))
#     plt.show()
    
#     ground_from_full = testobj3.prop(np.exp(1j*use_screen))
#     plt.imshow(np.angle(ground_from_full[N//2-8:N//2+8,N//2-8:N//2+8]),cmap='twilight_shifted',norm = colors.CenteredNorm())
#     plt.colorbar()
#     plt.title('z={:.0f}km propagated phase'.format(z/1000))
#     plt.show()
    
#     plt.imshow(np.log(np.abs(ground_from_full[N//2-8:N//2+8,N//2-8:N//2+8])),cmap='RdBu',norm = colors.CenteredNorm(vcenter=0,halfrange=1))
#     plt.colorbar()
#     plt.title('z={:.0f}km propagated intensity'.format(z/1000))
#     plt.show()
