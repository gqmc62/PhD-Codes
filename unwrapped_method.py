# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 19:27:43 2024

@author: gqmc62
"""

import numpy as np
from matplotlib import pyplot as plt
from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov
from aotools import zernike
from aotools.opticalpropagation import angularSpectrum
import my_phasescreen as mps
from tqdm import tqdm
import phase_screen_zernike_decomposition as z_decom
from aotools.functions import zernike
import my_propagation as mprop


def my_unwrap(wrapped_phase, period=2*np.pi):
    
    # numpy unwrap start unwrap at 0 coordinate
    # but for circular aperture there is no corner!
    # if we unwrap by each quardrant there will be a corner! noice.
    # but we may have to stitch them back together nicely
    # so things should be de-piston to the center
    
    center = wrapped_phase.shape[-1]//2
    unwrapped_phase = np.zeros_like(wrapped_phase)
      
    # ++ quardrant
    wrapped_quardrant = wrapped_phase[...,center:,center:]
    unwrapped_quardrant = np.unwrap(np.unwrap(wrapped_quardrant,axis=0),axis=1)
      
    unwrapped_quardrant -= unwrapped_quardrant[0,0]
    unwrapped_phase[...,center:,center:] = unwrapped_quardrant
      
      
    # +- quardrant
    wrapped_quardrant = wrapped_phase[...,center:,:center][:,::-1]
    unwrapped_quardrant = np.unwrap(np.unwrap(wrapped_quardrant,axis=0),axis=1)
      
    unwrapped_quardrant -= unwrapped_quardrant[0,0]
    unwrapped_phase[...,center:,:center] = unwrapped_quardrant[:,::-1]
      
      
    # -+ quardrant
    wrapped_quardrant = wrapped_phase[...,:center,center:][::-1,:]
    unwrapped_quardrant = np.unwrap(np.unwrap(wrapped_quardrant,axis=0),axis=1)
      
    unwrapped_quardrant -= unwrapped_quardrant[0,0]
    unwrapped_phase[...,:center,center:] = unwrapped_quardrant[::-1,:]
      
      
    # -- quardrant
    wrapped_quardrant = wrapped_phase[...,:center,:center][::-1,::-1]
    unwrapped_quardrant = np.unwrap(np.unwrap(wrapped_quardrant,axis=0),axis=1)
      
    unwrapped_quardrant -= unwrapped_quardrant[0,0]
    unwrapped_phase[...,:center,:center] = unwrapped_quardrant[::-1,::-1]
      
    return unwrapped_phase

def crop_inscribed_square(circular_data):
    N = circular_data.shape[0]
    R = int(np.floor(N//2))
    r = int(np.floor(R/np.sqrt(2)))
    square_data = circular_data[N//2 - r//2
                  : N//2 + r//2,
                  N//2 - r//2
                  : N//2 + r//2]
    return square_data

def find_branch_points(wrapped_phase,pupil):
    # branch points have a property that
    # if follows slopes of phase in a circle path
    # it would have non-zero phase gain
    
    N = unwrapped_phase.shape[0]
    branch_points = np.zeros(unwrapped_phase.shape)
    period = 2*np.pi
    
    # for each corners of all set of 4 points
    for i in np.arange(N - 1):
        for j in np.arange(N - 1):
            
            # if the corner is in the pupil
            if (pupil[i:i+2,j:j+2].all()==True):
                
                # the 4 points are
                A = wrapped_phase[i    , j    ]
                B = wrapped_phase[i + 1, j    ]
                C = wrapped_phase[i + 1, j + 1]
                D = wrapped_phase[i    , j + 1]
              
                # the phase difference of paths in a circle are
                a = A - B
                b = B - C
                c = C - D
                d = D - A
              
                # if the difference is larger than half a period
                # it means the value is actually
                # on the opposite side of the circular value
                # means that it is half a period away
                if np.abs(a) > period/2.0:
                    # if it is too high
                    if a > 0:
                        # it is actually lower
                        a -= period/2.0
                    if a < 0:
                        # otherwise the opposite
                        a += period/2.0
              
                if np.abs(b) > period/2.0:
                    if b > 0:
                        b -= period/2.0
                    if b < 0:
                        b += period/2.0
              
                if np.abs(c) > period/2.0:
                    if c > 0:
                        c -= period/2.0
                    if c < 0:
                        c += period/2.0
              
                if np.abs(d) > period/2.0:
                    if d > 0:
                        d -= period/2.0
                    if d < 0:
                        d += period/2.0
              
                # encircle path phase gain
                circle = a + b + c + d
                
                # if the gain is not zero
                if (not np.isclose(np.abs(circle)/(period),0.0,atol=0.1)):
                    
                    # it is a branch point
                    branch_points[i,j] = True
    
    # position of the branch points
    branch_points_pos = np.array(np.where(branch_points == True))
    
    return branch_points_pos

N = 128
r0 = 0.05
pixel_scale = 0.05/16
L0 = 5
wvl = 500e-9
# z = 10e3
TRIAL = 10

z_list = [1e3,2e3,5e3,10e3,20e3]#[5e3]#

MODE = 45

Zernike = zernike.zernikeArray(MODE, N)
GramSchmidt = z_decom.gram_schmidt_process(Zernike)
PUPIL = Zernike[0]
FlatZernike = Zernike.reshape((MODE, N*N))
FlatGramSchmidt = GramSchmidt.reshape((MODE, N*N))
GramSchmidtCoeff2ZernikeCoeff = z_decom.get_GramSchmidtCoeff2ZernikeCoeff(
    FlatZernike, FlatGramSchmidt)
    

for z in z_list:
    
    A=[]
    B=[]
    C=[]
    D=[]
    E=[]
    # PSD_x=[]
    # PSD_y=[]
    no_scintillation_z=[]
    scintillation_z=[]
    
    for trial in tqdm(np.arange(TRIAL)):
        IPS = PhaseScreenKolmogorov(N, pixel_scale, r0, L0)
        pupil = zernike.zernike_nm(0, 0, N)
        pupil_nan = np.copy(pupil)
        pupil_nan[pupil_nan == 0] = np.nan
        for i in (np.arange(N)):
            IPS.add_row()
        U = np.exp(1j*IPS.scrn)
        U_pupil = U*pupil
        
        pad = int(np.ceil(wvl*z/2.0/pixel_scale/2)*2)
        pad = 2**int(np.ceil(np.log(pad*2 + N)/np.log(2)))
        cropA = int(pad)
        cropB = int(N + pad)
        
        propped_U = angularSpectrum(np.pad(U_pupil,
                                           ((pad,pad),
                                            (pad,pad)),
                                           mode='symmetric'),
                                    wvl, pixel_scale, pixel_scale, z
                                    )[cropA : cropB, cropA : cropB]
        
        wrapped_phase = np.angle(U)
        wrapped_phase_pupil_nan = wrapped_phase *pupil_nan
        wrapped_phase_pupil = wrapped_phase *pupil
        unwrapped_phase = np.unwrap(np.unwrap(wrapped_phase,axis=0),axis=1)
        unwrapped_phase_then_pupil_nan = unwrapped_phase * pupil_nan
        unwrapped_phase_then_pupil = unwrapped_phase * pupil
        
        
        U_pupil_nan = U*pupil_nan
        what_we_normally_get = np.unwrap(np.unwrap(np.angle(U_pupil),axis=0),axis=1)
        what_we_normally_get_nan = np.unwrap(np.unwrap(np.angle(U_pupil),axis=0),axis=1)*pupil_nan
        
        # if trial == 0:
        #     plt.imshow(wrapped_phase_pupil_nan)
        #     plt.title('wrapped phase')
        #     plt.colorbar()
        #     plt.show()
            
        #     plt.imshow(unwrapped_phase_then_pupil_nan)
        #     plt.title('what we want')
        #     plt.colorbar()
        #     plt.show()
            
        #     plt.imshow(what_we_normally_get_nan)
        #     plt.colorbar()
        #     plt.title('what we normally get')
        #     plt.show()
            
        #     my_unwrapped_phase = my_unwrap(wrapped_phase_pupil_nan)
        #     plt.imshow(my_unwrapped_phase)
        #     plt.colorbar()
        #     plt.title('my unwrapped method')
        #     plt.show()
            
        wrapped_phase_scintillation = np.angle(propped_U)*pupil#*pupil_nan#
        intensity_scintillation = np.abs(propped_U)**2*pupil#*pupil_nan#
        branch_points_pos = find_branch_points(wrapped_phase_scintillation, pupil)
        
        # if trial == 0:
        #     plt.pcolor(wrapped_phase_scintillation.T)
        #     plt.axis('square')
        #     plt.title('wrapped phase with scintillation')
        #     plt.plot(branch_points_pos[0] + 0.5, branch_points_pos[1] + 0.5,ls='',marker='.',color='r')
        #     plt.colorbar()
        #     plt.show()
            
        #     plt.pcolor(intensity_scintillation.T)
        #     plt.axis('square')
        #     plt.title('intensity with scintillation')
        #     plt.plot(branch_points_pos[0] + 0.5, branch_points_pos[1] + 0.5,ls='',marker='.',color='r')
        #     plt.colorbar()
        #     plt.show()
        
        unwrapped_phase_scintillation = my_unwrap(wrapped_phase_scintillation)*pupil
        N = unwrapped_phase_scintillation.shape[0]//2
        coord = np.arange(-N,N)
        # if trial == 0:
        plt.pcolor(#coord,coord,
                   unwrapped_phase_scintillation.T)
        plt.axis('square')
        plt.plot(branch_points_pos[0] + 0.5, branch_points_pos[1] + 0.5,ls='',marker='.',color='r')
        plt.title('unwrapped phase with scintillation')
        plt.colorbar()
        plt.show()
        
        N = unwrapped_phase_scintillation.shape[0]
        R = N//2
        square_R = int(np.floor(R*np.sqrt(2)/2)*2)
        square_range_A = (N - square_R)//2
        square_range_B = (N + square_R)//2
        square_phase_scintillation = crop_inscribed_square(pupil*unwrapped_phase_scintillation)#(pupil*unwrapped_phase_scintillation)[square_range_A:square_range_B,square_range_A:square_range_B]
        
        # PSD_x_t = np.zeros((square_phase_scintillation.shape[0],
        #                   square_phase_scintillation.shape[0]),dtype=complex)
        # PSD_y_t = np.copy(PSD_x_t)
        # for i in np.arange(square_phase_scintillation.shape[0]):
        #     PSD_y_t[i] = mps.get_PSD(square_phase_scintillation[i,:],pixel_scale)
        #     PSD_x_t[i] = mps.get_PSD(square_phase_scintillation[:,i],pixel_scale)
            
        # for i in np.arange(square_phase_scintillation.shape[0]//10):
        #     plt.plot(square_phase_scintillation[i*10,:])
        # plt.title('phase fed to PSD_y')
        # plt.show()
        
        # for i in np.arange(square_phase_scintillation.shape[0]//10):
        #     plt.plot(square_phase_scintillation[:,i*10])
        # plt.title('phase fed to PSD_x')
        # plt.show()
        
        # PSD_x.append(PSD_x_t)
        # PSD_y.append(PSD_y_t)
        
        square_PSD = mps.get_PSD(square_phase_scintillation, pixel_scale)
        
        circle_PSD = mps.get_PSD(unwrapped_phase_scintillation, pixel_scale)
        
        theory = mps.psd_function(r0,N,pixel_scale,L0,1e-10)[N//2,N//2+1:]
        no_scintillation_psd = mps.get_PSD(unwrapped_phase, pixel_scale)[N//2,N//2+1:]
        
        A.append(square_PSD[square_PSD.shape[0]//2,square_PSD.shape[0]//2 + 1:])
        B.append(square_PSD[square_PSD.shape[0]//2 + 1:,square_PSD.shape[0]//2])
        C.append(circle_PSD[circle_PSD.shape[0]//2,circle_PSD.shape[0]//2 + 1:])
        D.append(circle_PSD[circle_PSD.shape[0]//2 + 1:,circle_PSD.shape[0]//2])
        E.append(no_scintillation_psd)
        
        
        no_scintillation_z.append( z_decom.get_ZernikeCoeff(
            (unwrapped_phase).flatten(),
            FlatGramSchmidt,
            GramSchmidtCoeff2ZernikeCoeff))
        scintillation_z.append( z_decom.get_ZernikeCoeff(
            (unwrapped_phase_scintillation).flatten(),
            FlatGramSchmidt,
            GramSchmidtCoeff2ZernikeCoeff))
        
    
        
    def my_errorbar(x,y,yerr,color,alpha,label):
        plt.errorbar(x,y,color=color,label=label)
        plt.fill_between(x,y+yerr,y-yerr,color=color,alpha=alpha)
        return
        
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    D = np.asarray(D)
    E = np.asarray(E)
    no_scintillation_z = np.asarray(no_scintillation_z)
    scintillation_z = np.asarray(scintillation_z)
    
    # PSD_x = np.asarray(PSD_x)
    # PSD_y = np.asarray(PSD_y)
    
    # 1/(A.shape[-1]*delta)
    del_f = 1/2./pixel_scale/N
    delf_square = 1/2./pixel_scale/square_phase_scintillation.shape[0]
    
    alpha=0.2
    
    
    # my_errorbar((np.arange(E.shape[1]) + 1)*del_f,
    #               np.mean(E,axis=0),
    #               yerr=np.std(E,ddof=1,axis=0)/np.sqrt(2*E.shape[1]-2),
    #               color='r',alpha=alpha,
    #               label='no scintllation')
    
    # plt.plot((np.arange(theory.shape[0]) + 1)*del_f,theory,label='von Karman',ls='-',color='b')
    # plt.legend()
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.show()
    
    
    
    # my_errorbar((np.arange(D.shape[1]) + 1)*del_f,
    #               np.mean(D,axis=0),
    #               yerr=np.std(D,ddof=1,axis=0)/np.sqrt(2*D.shape[1]-2),
    #               color='k',alpha=alpha,
    #               label='circle x')
    # my_errorbar((np.arange(C.shape[1]) + 1)*del_f,
    #               np.mean(C,axis=0),
    #               yerr=np.std(C,ddof=1,axis=0)/np.sqrt(2*C.shape[1]-2),
    #               color='y',alpha=alpha,
    #               label='circle y')
    # plt.plot((np.arange(theory.shape[0]) + 1)*del_f,theory,label='von Karman',ls='-',color='b')
    # plt.legend()
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.show()
    
    
    
    # my_errorbar((np.arange(B.shape[1]) + 1)*delf_square,
    #               np.mean(B,axis=0),
    #               yerr=np.std(B,ddof=1,axis=0)/np.sqrt(2*B.shape[1]-2),
    #               color='m',alpha=alpha,
    #               label='square x')
    # # my_errorbar((np.arange(PSD_x.shape[2]//2-1) + 1)*delf_square,
    # #               np.mean(PSD_x,axis=(0,1))[PSD_x.shape[2]//2 + 1:],
    # #               yerr=np.std(PSD_x,axis=(0,1),ddof=1)[PSD_x.shape[2]//2 + 1:]/np.sqrt(2*PSD_x.shape[0]*PSD_x.shape[1]-2),
    # #               color='g',alpha=alpha,
    # #               label='1d - square x')
    # my_errorbar((np.arange(A.shape[1]) + 1)*delf_square,
    #               np.mean(A,axis=0),
    #               yerr=np.std(A,ddof=1,axis=0)/np.sqrt(2*A.shape[1]-2),
    #               color='c',alpha=alpha,
    #               label='square y')
    # # my_errorbar((np.arange(PSD_y.shape[2]//2-1) + 1)*delf_square,
    # #               np.mean(PSD_y,axis=(0,1))[PSD_y.shape[2]//2 + 1:],
    # #               yerr=np.std(PSD_y,axis=(0,1),ddof=1)[PSD_y.shape[2]//2 + 1:]/np.sqrt(2*PSD_y.shape[0]*PSD_y.shape[1]-2),
    # #               color='pink',alpha=alpha,
    # #               label='1d - square y')
    # plt.plot((np.arange(theory.shape[0]) + 1)*del_f,theory,label='von Karman',ls='-',color='b')
    # plt.legend()
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.show()
    
    
    
    
    
    # my_errorbar((np.arange(D.shape[1]) + 1)*del_f,
    #               np.mean(D,axis=0),
    #               yerr=np.std(D,ddof=1,axis=0)/np.sqrt(2*D.shape[1]-2),
    #               color='k',alpha=alpha,
    #               label='circle x')
    # my_errorbar((np.arange(B.shape[1]) + 1)*delf_square,
    #               np.mean(B,axis=0),
    #               yerr=np.std(B,ddof=1,axis=0)/np.sqrt(2*B.shape[1]-2),
    #               color='m',alpha=alpha,
    #               label='square x')
    # plt.plot((np.arange(theory.shape[0]) + 1)*del_f,theory,label='von Karman',ls='-',color='b')
    # plt.legend()
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.show()
    
    
    
    
    # my_errorbar((np.arange(C.shape[1]) + 1)*del_f,
    #               np.mean(C,axis=0),
    #               yerr=np.std(C,ddof=1,axis=0)/np.sqrt(2*C.shape[1]-2),
    #               color='y',alpha=alpha,
    #               label='circle y')
    # my_errorbar((np.arange(A.shape[1]) + 1)*delf_square,
    #               np.mean(A,axis=0),
    #               yerr=np.std(A,ddof=1,axis=0)/np.sqrt(2*A.shape[1]-2),
    #               color='c',alpha=alpha,
    #               label='square y')
    # plt.plot((np.arange(theory.shape[0]) + 1)*del_f,theory,label='von Karman',ls='-',color='b')
    # plt.legend()
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.show()
    
    rytov_index = mprop.rytov_approximation_scintillation_index(wvl,r0,z)
    
    # my_errorbar((np.arange(A.shape[1]) + 1)*delf_square,
    #               np.mean(A,axis=0),
    #               yerr=np.std(A,ddof=1,axis=0)/np.sqrt(2*A.shape[0]-2),
    #               color='c',alpha=alpha,
    #               label='sub-square-aperture y-cross-section of PSD')
    my_errorbar((np.arange(B.shape[1]) + 1)*delf_square,
                  np.mean(B,axis=0),
                  yerr=np.std(B,ddof=1,axis=0)/np.sqrt(2*B.shape[0]-2),
                  color='b',alpha=alpha,
                  label='sub-square-aperture x-cross-section of PSD')
    # my_errorbar((np.arange(C.shape[1]) + 1)*del_f,
    #               np.mean(C,axis=0),
    #               yerr=np.std(C,ddof=1,axis=0)/np.sqrt(2*C.shape[0]-2),
    #               color='y',alpha=alpha,
    #               label='circle-aperture y-cross-section of PSD')
    # my_errorbar((np.arange(D.shape[1]) + 1)*del_f,
    #               np.mean(D,axis=0),
    #               yerr=np.std(D,ddof=1,axis=0)/np.sqrt(2*D.shape[0]-2),
    #               color='k',alpha=alpha,
    #               label='circle-aperture x-cross-section of PSD')
    # my_errorbar((np.arange(E.shape[1]) + 1)*del_f,
    #               np.mean(E,axis=0),
    #               yerr=np.std(E,ddof=1,axis=0)/np.sqrt(2*E.shape[0]-2),
    #               color='r',alpha=alpha,
    #               label='no scintllation')
    my_errorbar((np.arange(E.shape[1]) + 1)*del_f,
                  np.mean(E,axis=0),
                  yerr=np.std(E,ddof=1,axis=0)/np.sqrt(2*E.shape[1]-2),
                  color='r',alpha=alpha,
                  label='no scintllation')
    
    plt.plot((np.arange(theory.shape[0]) + 1)*del_f,theory,label='von Karman',ls='-',color='k')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.title('PSD analysis before and after propagation: $\sigma^2_\chi$={:.2f}'.format(rytov_index))
    plt.show()
    
    theory_z_var = mps.von_Karman_noll_variance(MODE + 1, N, pixel_scale, r0,L0)
    plt.plot(np.arange(MODE + 1), theory_z_var,label='noll',color='k')
    my_errorbar(np.arange(MODE - 1),
                    np.var(no_scintillation_z,
                           ddof=1,axis=0)[1:],
                    yerr=2*np.var(no_scintillation_z,
                           ddof=1,axis=0)[1:]/np.sqrt(2*TRIAL-2),
                    color='r',alpha=alpha,
                    label='no scintillation')
    my_errorbar(np.arange(MODE - 1),
                    np.var(scintillation_z,
                           ddof=1,axis=0)[1:],
                    yerr=2*np.var(scintillation_z,
                           ddof=1,axis=0)[1:]/np.sqrt(2*TRIAL-2),
                    color='b',alpha=alpha,
                    label='scintillation')
    plt.legend()
    plt.yscale('log')
    plt.title('Zernike mode decomposition analysis before and after propagation: $\sigma^2_\chi$={:.2f}'.format(rytov_index))
    plt.show()
