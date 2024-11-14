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
D2_o = 0.05
delta2 = 0.05/16
# output_pixel = 16
# delta2 = D2/output_pixel
z_list = (np.arange(10) + 1)*5e3#[5e3]#

MAX_N = 2048

for z in z_list:
    
    # # eve with my best i still cannot manage tis one lol.
    
    # # for case delta1 != delta2 only & for delta1>delta2
    
    # # print('optimized version')
    # D2 = D2_o# + 2*wvl*z/r0
    
    # # solve inequality 1&3
    
    # a = D2
    # b = wvl*z-D2*delta2
    # c = -2*wvl*z*delta2
    
    # delta1max = np.max(solve_quadratic(a, b, c))
    # D1min = (D2*delta1max + wvl*z)/delta2
    
    # # pick the next power grid
    
    # suggestN = next_2_power(D1min/delta1max)
    
    # # solve inequality 1 with D1=Ndelta1
    
    # delta1 = np.max(solve_quadratic(-1, delta2, wvl*z/suggestN))
    # if delta1 > r0/4:
    #     delta1 = r0/4
    #     print('initial choice in delta1 cannot properly sample phase screen')
        
    # delta1 = delta2
    
    # D1_o = delta1*suggestN
    
    # D1 = D1_o# + 2*wvl*z/r0
    
    # input_pixel = int(np.ceil(D1/delta1)) # based on input and output pixels
    # if (input_pixel % 2) == 1 :
    #     input_pixel += 1
    # output_pixel = int(np.ceil(D2/delta2))
    # if (output_pixel % 2) == 1 :
    #     output_pixel += 1
    # N = next_2_power(np.max([input_pixel,output_pixel]))# get N
    # if (N//output_pixel) > 16: # this means that only to get 1 output pixel we are using 256 pixel!!!
    #     print('We are wasting calculation power')
    #     # quit()
    # if N >= MAX_N :
    #     print('We cannot afford this much memory')
    #     break
    
    # suggested_partial_z1 = (N - D1/2./delta1 - D2/2./delta2)/(wvl/2./delta1/delta2) # no wrapping allowed in D2
    # suggested_partial_z2 = N/(wvl/np.min([delta1,delta2])**2) # no aliasing allowed in angular-spectrum transform
    # suggested_partial_z3 = N / (wvl/delta1**2 / np.abs(delta2/delta1*(1-delta2/delta1)))
    # if suggested_partial_z3 == 0:
    #     suggested_partial_z3 = z
    # suggested_partial_z = np.min([z,suggested_partial_z1,suggested_partial_z2,suggested_partial_z3]) # the lowest boundary
    # if suggested_partial_z < 0 :
    #     print('this condition is not suitable')
    #     print(wvl,z,D2,delta2,D1,delta1)
    #     break
    # n_prop = int(np.ceil(z/suggested_partial_z))
    # print('suggest {:d} propagation'.format(n_prop))
    # partial_z = z/n_prop
    
    # print(z,wvl,D2,delta2,D1,delta1,N,partial_z)
    
    # alpha = np.arange(n_prop + 1)/n_prop
    # delta_list = (1 - alpha) * delta1 + alpha * delta2
    # mag_list = delta_list[1:]/delta_list[:-1]
    
    # N1 = N
    
    # A = (N >= D2/delta2) or np.isclose(N,D2/delta2,rtol=1e-2)
    # B = (N >= D1/delta1) or np.isclose(N,D1/delta1,rtol=1e-2)
    # print(A and B, 'test0: simulation size can accommodate input and output')
    # A = D2
    # B = D1*delta2/delta1 - 2.*wvl*z/2./delta1
    # print((A <= B) or np.isclose(A,B,rtol=1e-2), 'test1: all output must come from source')#***
    # A = N
    # B = D1/2./delta1 + D2/2./delta2 + wvl*partial_z/2./delta1/delta2
    # print((A >= B) or np.isclose(A,B,rtol=1e-2), 'test2: no wrapping')
    # A = np.abs(1 - delta2/delta1)*D1
    # B = wvl*z/delta1
    # print((A <= B) or np.isclose(A,B,rtol=1e-2), 'test3: no aliasing in Q1')#***
    # A = N
    # B = wvl*partial_z/np.min([delta1,delta2])**2
    # print((A >= B) or np.isclose(A,B,rtol=1e-2), 'test4: no aliasing in Q2')
    # A = np.abs(1 - delta1/delta2)*D2
    # B = wvl*z/delta2
    # print((A <= B) or np.isclose(A,B,rtol=1e-2), 'test3: no aliasing in Q3')
    # A = N
    # B = wvl*partial_z/delta1**2 / np.abs(delta2/delta1*(1-delta2/delta1))
    # print((A >= B) or np.isclose(A,B,rtol=1e-2), 'test5!: no aliasing in Q2*F{U1*Q1}')
    
    # try:
    #     testobj
    #     if ((testobj.N != N)
    #         or (testobj.wvl != wvl)
    #         or (testobj.delta != delta1)
    #         or (testobj.Z != z)):
    #         testobj = mpc.ConsecutivePropSquareV2(N, wvl, z, n_prop, delta1, delta2)
    # except:
    #     testobj = mpc.ConsecutivePropSquareV2(N, wvl, z, n_prop, delta1, delta2)
    
    # try:
    #     fullphasescreen
    #     if ((fullphasescreen.nx_size != N)
    #         or (fullphasescreen.pixel_scale != delta1)
    #         or (fullphasescreen.r0 != r0)
    #         or (fullphasescreen.L0 != L0)):
    #         fullphasescreen = PhaseScreenKolmogorov(N, delta1, r0, L0)
    # except:
    #     fullphasescreen = PhaseScreenKolmogorov(N, delta1, r0, L0)
    #     for row in np.arange(N):
    #         fullphasescreen.add_row()
    
    # input_pixel_show = int(D2//delta1)
    # if input_pixel_show > input_pixel :
    #     input_pixel_show = int(input_pixel)
    # output_pixel_show = int(D2_o//delta2)
    # if output_pixel_show > output_pixel :
    #     output_pixel_show = int(output_pixel)
    
    # screen = np.pad(fullphasescreen.scrn[N//2 - input_pixel//2
    #                                       : N//2 + input_pixel//2,
    #                                       N//2 - input_pixel//2
    #                                       : N//2 + input_pixel//2],
    #                 pad_width = (((N - input_pixel)//2, (N - input_pixel)//2),
    #                               ((N - input_pixel)//2, (N - input_pixel)//2)),
    #                 mode='constant',
    #                 constant_values=0)
    # # screen = np.zeros((N,N))
    
    
    
    
    # plt.imshow(np.angle(np.exp(1j*screen[N//2 - input_pixel_show//2
    #                                                     : N//2 + input_pixel_show//2,
    #                                                     N//2 - input_pixel_show//2
    #                                                     : N//2 + input_pixel_show//2])),
    #             cmap='twilight_shifted',norm = colors.CenteredNorm())
    # plt.colorbar()
    # plt.title('new; z={:.0f}km origin phase'.format(z/1000))
    # plt.show()
    
    # ground_from_full = testobj.prop(np.exp(1j*screen))
    # plt.imshow(np.angle(ground_from_full[N//2 - output_pixel_show//2
    #                                       : N//2 + output_pixel_show//2,
    #                                       N//2 - output_pixel_show//2
    #                                       : N//2 + output_pixel_show//2]),
    #             cmap='twilight_shifted',norm = colors.CenteredNorm(vcenter=0,halfrange=np.pi))
    # plt.colorbar()
    # plt.title('new; z={:.0f}km propagated phase'.format(z/1000))
    # plt.show()
    
    # plt.imshow(np.log(np.abs(ground_from_full[N//2 - output_pixel_show//2
    #                                           : N//2 + output_pixel_show//2,
    #                                           N//2 - output_pixel_show//2
    #                                           : N//2 + output_pixel_show//2])),
    #             cmap='RdBu_r',norm = colors.CenteredNorm(vcenter=0,halfrange=np.sqrt(0.6)))
    # plt.colorbar()
    # plt.title('new; z={:.0f}km propagated intensity'.format(z/1000))
    # plt.show()



    # delta = np.linspace(0.9*delta2,delta1max)
    # plt.fill_between(delta, wvl*z/np.abs(delta-delta2), wvl*z/delta2 + D2*delta/delta2,alpha=0.5,label='valid')
    # plt.plot(delta,wvl*z/np.abs(delta-delta2),label='anti-aliasing')
    # plt.plot(delta,wvl*z/delta2 + D2*delta/delta2,label='data coverage')
    
    # D = np.linspace(D1min/2*0.9,D1min*2*1.1)
    # # plt.plot(wvl*z/D+delta2,D,label='x less than this')
    # # plt.plot((D*delta2-wvl*z)/D2,D,label='x less than this')
    
    # # plt.vlines(delta2,0,16*D2,label='output scale')
    # # plt.hlines(D2,delta2,np.min([r0/4,delta1max]),label='output screen')
    # plt.plot(delta,delta*N*2,label='{:d}px'.format(N*2))
    # plt.plot(delta,delta*N,label='{:d}px'.format(N))
    # plt.plot(delta,delta*N/2,label='{:d}px'.format(N//2))
    # plt.plot(delta1,D1,marker='o',label='current choise')
    # plt.legend()
    # plt.xlim(0.9*delta2,np.min([r0/4,delta1max]))
    # plt.ylim(0.9*D1min/2,1.1*D1min*2)
    # plt.show()
    

    
    
    # # gain from special case is low.
    # # there is a problem in propagation method eventhough done exactly as schmidt said.?
    
    # print('regular version')
    
    # source without pupil
    
    delta1 = delta2 # doing 1:1 scale; just because
    
    D2 = D2_o# + 2*wvl*z/r0
    
    max_alpha = wvl/2./delta1
    creep = max_alpha*z # per side!
    D1 = delta1*2**(np.ceil(np.log((D2
                                    + 2*creep
                                    # + 2*wvl*z/r0
                                    )/delta1)/np.log(2))) # pick D1 that gives no creep inside D2
    
    input_pixel = D1/delta1 # based on input and output pixels
    output_pixel = D2/delta2
    
    N = int(2**np.ceil(np.log(np.max([input_pixel,output_pixel]))/np.log(2))) # get N
    
    if (N//output_pixel) > 16: # this means that only to get 1 output pixel we are using 256 pixel!!!
        print('We are wasting calculation power')
        # quit()
    if N >= MAX_N :
        print('We cannot afford this much memory')
        break
    
    if input_pixel > N: # trivial but mostly forgotten and mistaken
        print('N is too small for input size')
    if output_pixel > N:
        print('N is too small for output size')
    
    suggested_partial_z1 = (N*delta2 - D1*delta2/2./delta1 - D2/2.)/max_alpha # no wrapping allowed in D2
    suggested_partial_z2 = N*np.min([delta1,delta2])**2/wvl # no aliasing allowed in angular-spectrum transform
    
    suggested_partial_z = np.min([z,suggested_partial_z1,suggested_partial_z2]) # the lowest boundary
    
    nprop = int(np.ceil(z/suggested_partial_z))*4
    
    if nprop > 1:
        print('suggest {:d} propagation'.format(nprop))
    
    partial_z = z/nprop
    
    screen_size = N*delta2
    illumination_area_with_wrapping = D1*delta2/2./delta1 + D2/2. + max_alpha*partial_z # anti-complete-wrapping around screen
    
    if screen_size < illumination_area_with_wrapping:
        print('contamination detected.')
    if N < wvl*partial_z/delta1/delta2:
        print('aliases in angular spectrum')
        
    # N2 = N
    
    # print(N1/N2)
    
    print(z,wvl,D2,delta2,D1,delta1,N,partial_z)
    print((N >= D2/delta2) and (N >= D1/delta1), 'test0: simulation size can accommodate input and output')
    print(D2 <= D1*delta2/delta1 - 2.*wvl*z/2./delta1, 'test1: all output must come from source')#***
    print(N >= D1/2./delta1 + D2/2./delta2 + wvl*partial_z/2./delta1/delta2, 'test2: no wrapping')
    print(np.abs(1 - delta2/delta1)*D1 <= wvl*z/delta1, 'test3: no aliasing in forward ft')#***
    print(N >= wvl*partial_z/np.min([delta1,delta2])**2, 'test4: no aliasing in backward ft')
    
    
    try:
        testobj2
        if ((testobj2.N != N)
            or (testobj2.wvl != wvl)
            or (testobj2.delta != delta1)
            or (testobj2.Z != z)):
            testobj2 = mpc.ConsecutivePropSquare(wvl, delta1, N, z)

    except:
        testobj2 = mpc.ConsecutivePropSquare(wvl, delta1, N, z)

    
    try:
        fullphasescreen2
        if ((fullphasescreen2.nx_size != N)
            or (fullphasescreen2.pixel_scale != delta1)
            or (fullphasescreen2.r0 != r0)
            or (fullphasescreen2.L0 != L0)):
            fullphasescreen2 = PhaseScreenKolmogorov(N, delta1, r0, L0)
    except:
        fullphasescreen2 = PhaseScreenKolmogorov(N, delta1, r0, L0)
        for row in np.arange(N):
            fullphasescreen2.add_row()
            
    input_pixel_show = int(D2//delta1)
    if input_pixel_show > input_pixel :
        input_pixel_show = int(input_pixel)
    output_pixel_show = int(D2//delta2)
    if output_pixel_show > output_pixel :
        output_pixel_show = int(output_pixel)
        
    screen = np.zeros((N,N))
        
    plt.imshow(np.angle(np.exp(1j*fullphasescreen2.scrn[N//2 - input_pixel_show//2
                                                        : N//2 + input_pixel_show//2,
                                                        N//2 - input_pixel_show//2
                                                        : N//2 + input_pixel_show//2])),
                cmap='twilight_shifted',norm = colors.CenteredNorm())
    plt.colorbar()
    plt.title('standard; z={:.0f}km origin phase'.format(z/1000))
    plt.show()
    
    ground_from_full = testobj2.prop(np.exp(1j*fullphasescreen2.scrn))
    plt.imshow(np.angle(ground_from_full[N//2 - output_pixel_show//2
                                          : N//2 + output_pixel_show//2,
                                          N//2 - output_pixel_show//2
                                          : N//2 + output_pixel_show//2]),
                cmap='twilight_shifted',norm = colors.CenteredNorm(vcenter=0,halfrange=np.pi))
    plt.colorbar()
    plt.title('standard; z={:.0f}km propagated phase'.format(z/1000))
    plt.show()
    
    plt.imshow(np.log(np.abs(ground_from_full[N//2 - output_pixel_show//2
                                          : N//2 + output_pixel_show//2,
                                          N//2 - output_pixel_show//2
                                          : N//2 + output_pixel_show//2])),
                cmap='RdBu_r',norm = colors.CenteredNorm(vcenter=0,halfrange=np.sqrt(0.6)))
    plt.colorbar()
    plt.title('standard; z={:.0f}km propagated intensity'.format(z/1000))
    plt.show()
    
    