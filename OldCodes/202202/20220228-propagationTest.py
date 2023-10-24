################################################################################
## Author:      Puttiwat Kongkaew
## Position:    Ph.D. by research student
## Affiliation: Center for Advanced Instrumentation, Department of Physics,
##              Durham University, United Kingdom
## Email:       gqmc62@durham.ac.uk
## Date:        16 February 2022
################################################################################

################################################################################
## 0.0 ## What this code should do? ############################################
################################################################################
## 1) Explore EXISTING wave propagation implementation of aotools.
##    Propagate weak wave from the top atmospheric turbulence layer
##    to the ground with familiar night-time turbulence values.
##    1.1) One-step Fresnel propagation
##         1.1.1) Enforce 1:1 input:output scale. Ignore propagation distance
##         1.1.2) Ignore input:output scale. Enforce propagation distance
##    1.2) Two-step Fresnel propagation
##    1.3) Angular Spectrum form of Fresnel propagation
## 2) Test out multi-layer propagation
##    2.1) of n layers of weak turbulence
##         a) 1 layer
##         b) 2+ layers
##    2.2) of n layers of strong turbulence
##         a) 1 layer
##         b) 2+ layers
## 3) Clear atm --> DM1 --> DM2 --> WFS.
##    DM1 is up very high in the sky, say 12-50 km.
##    DM2 is close to the ground, say 0.01-1-10 km.
##    3.1) No shape on DM1&2. --> Do IM of DM1 --> WFS
##    3.2) Strong shape on DM2. --> Do IM of DM1 --> WFS
##    3.3) Vary distance of DM2.
## 4) Go crazy on the conditions and see when things break.
################################################################################

################################################################################
## 0.1 ## Load libraries & modules & set up preferences ########################
################################################################################

from numpy import fft
import numpy as np
import random
import time
from matplotlib import pyplot as plt
from aotools import zernike
from aotools.turbulence import infinitephasescreen
from aotools import opticalpropagation
from aotools import fouriertransform

color_map = plt.cm.get_cmap('Blues')
reversed_color_map = color_map.reversed()

################################################################################
## 0.2 ## declare some useful functions ########################################
################################################################################

## almost a rip off from aotools.opticalpropagation
## the original code has an out-of-date try&except error code
## i changed it to always the '1+m' case
## i also added some plottings to show the in between
def twoStepFresnel(Uin, wvl, d1, d2, z):
    """
    Fresnel propagation using a two step Fresnel propagation method.

    Parameters:
        Uin (ndarray): A 2-d, complex, input array of complex amplitude
        wvl (float): Wavelength of propagated light in metres
        d1 (float): spacing of input plane
        d2 (float): desired output array spacing
        z (float): metres to propagate along optical axis

    Returns:
        ndarray: Complex ampltitude after propagation
    """

    N = Uin.shape[0] #Number of grid points
    k = 2*np.pi/wvl #optical wavevector

    #source plane coordinates
    x1, y1 = np.meshgrid( np.arange(-N/2,N/2) * d1,
                            np.arange(-N/2.,N/2.) * d1 )

    #magnification
    m = float(d2)/d1

    #intermediate plane
    Dz1 = z / (1+m)
    d1a = wvl * abs(Dz1) / (N*d1) #coordinates
    x1a, y1a = np.meshgrid( np.arange( -N/2.,N/2.) * d1a,
                              np.arange( -N/2.,N/2.) * d1a )

    #Evaluate Fresnel-Kirchhoff integral
    A = 1./(1j * wvl * Dz1)
    B = np.exp(1j * k/(2*Dz1) * (x1a**2 + y1a**2) )
    C = fouriertransform.ft2(Uin * np.exp(1j * k/(2*Dz1) 
                                        * (x1**2 + y1**2)), d1)
    Uitm = A*B*C
    
    ## intermediate step plots
    # scrn_size=1024
    # sample_scrn_size=128
    # edge1=int((scrn_size-sample_scrn_size)/2)
    # edge2=int((scrn_size+sample_scrn_size)/2)

    # plt.pcolor(x1a[edge1:edge2,edge1:edge2],y1a[edge1:edge2,edge1:edge2],
    #         np.angle(Uitm[edge1:edge2,edge1:edge2]),
    #         cmap=reversed_color_map,shading='auto')
    # plt.title('U_itm phase')
    # plt.axis('equal')
    # plt.xlabel('m')
    # plt.ylabel('m')
    # plt.colorbar()
    # plt.savefig('U_itm_phase')
    # plt.clf()

    # plt.pcolor(x1a[edge1:edge2,edge1:edge2],y1a[edge1:edge2,edge1:edge2],
    #         np.abs(Uitm[edge1:edge2,edge1:edge2]),
    #         cmap=reversed_color_map,shading='auto')
    # plt.title('U_itm abs')
    # plt.axis('equal')
    # plt.xlabel('m')
    # plt.ylabel('m')
    # plt.colorbar()
    # plt.savefig('U_itm_abs')
    # plt.clf()

    # plt.pcolor(x1a,y1a,np.angle(Uitm),cmap=reversed_color_map,shading='auto')
    # plt.title('U_itm phase full')
    # plt.axis('equal')
    # plt.xlabel('m')
    # plt.ylabel('m')
    # plt.colorbar()
    # plt.savefig('U_itm_phase_full')
    # plt.clf()

    # plt.pcolor(x1a,y1a,np.abs(Uitm),cmap=reversed_color_map,shading='auto')
    # plt.title('U_itm abs full')
    # plt.axis('equal')
    # plt.xlabel('m')
    # plt.ylabel('m')
    # plt.colorbar()
    # plt.savefig('U_itm_abs_full')
    # plt.clf()
    
    #Observation plane
    Dz2 = z - Dz1

    #coordinates
    x2,y2 = np.meshgrid( np.arange(-N/2., N/2.) * d2,
                            np.arange(-N/2., N/2.) * d2 )

    #Evaluate the Fresnel diffraction integral
    A = 1. / (1j * wvl * Dz2)
    B = np.exp( 1j * k/(2 * Dz2) * (x2**2 + y2**2) )
    C = fouriertransform.ft2(Uitm * np.exp( 1j * k/(2*Dz2)
                                           * (x1a**2 + y1a**2)), d1a)
    Uout = A*B*C

    return Uout

################################################################################
## 0.3 ## Set up & read in parameters ##########################################
################################################################################
## Numbering of layers follows the english floor numberings
## 0=ground
## 1=first ...
##
## ONLY declare initial/primary parameters here.
## Any secondary/derivative parameters should be in the next section.
## So we know what are the prioritized parameters.
##
## Specify units, types, dimension of parameters
################################################################################

## optical values ##############################################################

wvl=500e-9                                      #m

## atm data ####################################################################

## rip off from soapy config 'sh_8x8.yaml'
#r0=0.16                                         #m STACKED!!!
#scrnHeights=np.array([0, 5000, 10000, 15000])   #m
#scrnStrengths=np.array([0.5, 0.3, 0.1, 0.1])    #normalized cn2 ratio
#                                                #turbulence=CN2!!!!
#L0=np.array([20, 20, 20, 20])                   #m

## from Tim M
#r0=0.16
#scrnHeights=np.array([0, 3000, 12000])
#scrnStrengths=np.array([0.6, 0.15, 0.25])
#L0=np.array([20, 20, 20])

## 2layers
#scrnHeights=np.array([0, 5000])
#scrnStrengths=np.array([0.5, 0.3])
#L0=np.array([20, 20])

## Using
r0=0.05                                         #m STACKED!!!
scrnHeights=np.array([0, 5000, 10000, 15000])   #m
scrnStrengths=np.array([0.5, 0.3, 0.1, 0.1])    #normalized cn2 ratio
                                                #turbulence=CN2!!!!
L0=np.array([20, 20, 20, 20])                   #m


## simulation parameters #######################################################
random_seed_num=None#5
scrn_size=1024              #pixel
sample_scrn_size=128        #pixel
scrn_generate_L0_pad=2      #How many L0 should fit in the phase screen
                            #0=>no screen!!!
                            #1=>no oversize!!!
                            #2=>1original+1oversize

## to simulate the imaging at the focus. (sent it through fraunhofer) 
pupil_diameter=5.            #m

################################################################################
## 0.3 ## Calculate parameters from the read-in ################################
################################################################################

## optical parameter ###########################################################

k = 2*np.pi/wvl

## atm parameters ##############################################################

## number of layer
num_layer=len(scrnHeights)

## get turbulence strength of each layer : unstack r0
normScrnStrengths = scrnStrengths/(sum(scrnStrengths))
scrnStrengths = ( ((r0**(-5./3.))*normScrnStrengths)**(-3./5.) )
print(scrnStrengths)

## simulation parameter ########################################################

## get pixel scale and screen size of each layer
pixel_scale=L0*scrn_generate_L0_pad/scrn_size #m/pixel pixelscale
#sample_scrn_size=4./pixel_scale[0]

## generate grid
x,y=np.meshgrid(np.arange(-scrn_size/2.,scrn_size/2.),
                np.arange(-scrn_size/2.,scrn_size/2.))

## pupil
pupil=np.ones((scrn_size,scrn_size))                        #no stops
#pupil=(x**2+y**2<pupil_diameter**2/4./pixel_scale[-1]**2)    #pupil stop at [-1]

## initialize wave #############################################################
amp_in=pupil*np.ones((scrn_size,scrn_size))     #uniform intensity with pupil
phase_in=pupil*np.zeros((scrn_size,scrn_size))  #flat front with pupil
U_in=amp_in*np.exp(1j*phase_in)                 #euler form

## initialize data storage #####################################################
phase_scrn=[] #storing phase screens of each layer
U=[]                #U[0]= U at ground layer --> output
                    #U[-1] = U at top atm --> input
                    
for i in np.arange(num_layer+1):        #before vs after turbulent layers
    U.append(U_in)
    
for i in np.arange(num_layer):          #the turbulent layers
    phase_scrn.append(phase_in)

################################################################################
## 1 ## Explore wave propagation already implemented in SOAPY ##################
################################################################################
## test propagation
## 1) flat input
## 2) add [-1] atm
## 3) prop to [0] atm
## 4) no [0] turbulence add
## compare performance of each propagation method
################################################################################

## set up input values of THIS sub simulation ##################################

U_in=amp_in*np.exp(1j*phase_in) #actually no need to redo here lol
phase_in=np.angle(U_in)         #"-------------------------------"
amp_in=np.abs(U_in)             #"-------------------------------"
d_in=pixel_scale[-1]
z_in=scrnHeights[-1]

## add turbulence ##############################################################
edge1=int((scrn_size-sample_scrn_size)/2)
edge2=int((scrn_size+sample_scrn_size)/2)

phase_scrn=infinitephasescreen.PhaseScreenVonKarman(scrn_size,d_in,
                        scrnStrengths[-1],L0[-1],random_seed=random_seed_num)

## for self pad
## input amp & phase must be pre padded --> need for pad function
# add codes for padding input
# phase_patch=phase_scrn.scrn[edge1:edge2,edge1:edge2]
# amp_patch=amp_in[edge1:edge2,edge1:edge2]
# phase_add=phase_in
# amp_prop=amp_in
# for i in np.arange(scrn_size):
#     for j in np.arange(scrn_size):
#         phase_add[i,j]=phase_patch[int(i+sample_scrn_size/2)%sample_scrn_size,int(j+sample_scrn_size/2)%sample_scrn_size]
#         amp_prop[i,j]=amp_patch[int(i+sample_scrn_size/2)%sample_scrn_size,int(j+sample_scrn_size/2)%sample_scrn_size]
# phase_prop=phase_in+phase_add
# U_prop=amp_prop*np.exp(1j*phase_prop)

## for oversize calc & zero pad / pupil crop at [-1]
phase_add=pupil*phase_scrn.scrn
amp_prop=pupil*amp_in
phase_prop=phase_in+pupil*phase_add
U_prop=amp_prop*np.exp(1j*phase_prop)

## propagate ###################################################################

d_out=pixel_scale[0]
z_out=scrnHeights[0]

## one-step fresnel propagation: correct distance wrong scale
#U_out=opticalpropagation.oneStepFresnel(U_prop,wvl,d_in,z_in-z_out)
#d_out=wvl*(z_in-z_out)/(scrn_size*d_in)

## one-step fresnel propagation: wrong distance same scale
#z=d_in**2*scrn_size/wvl
#U_out=opticalpropagation.oneStepFresnel(U_prop,wvl,d_in,z)
#d_out=wvl*z/(scrn_size*d_in)

## two-step fresnel propagation
#U_out=twoStepFresnel(U_prop,wvl,d_in,d_out,z_in-z_out)

## angular spectrum form of fresnel propagation
U_out=opticalpropagation.angularSpectrum(U_prop,wvl,d_in,d_out,z_in-z_out)

## calculate observing values
amp_out=np.abs(U_out)
phase_out=np.angle(U_out)

## plot ########################################################################

## sample region

edge1=int((scrn_size-sample_scrn_size)/2)
edge2=int((scrn_size+sample_scrn_size)/2)

plt.pcolor(x[edge1:edge2,edge1:edge2]*d_in,y[edge1:edge2,edge1:edge2]*d_in,
           np.angle(U_prop[edge1:edge2,edge1:edge2]),
           cmap=reversed_color_map,shading='auto')
plt.title('U_prop phase')
plt.axis('equal')
plt.xlabel('m')
plt.ylabel('m')
plt.colorbar()
plt.savefig('U_prop_phase')
plt.clf()

plt.pcolor(x[edge1:edge2,edge1:edge2]*d_in,y[edge1:edge2,edge1:edge2]*d_in,
           np.abs(U_prop[edge1:edge2,edge1:edge2]),
           cmap=reversed_color_map,shading='auto')
plt.title('U_prop abs')
plt.axis('equal')
plt.xlabel('m')
plt.ylabel('m')
plt.colorbar()
plt.savefig('U_prop_abs')
plt.clf()

plt.pcolor(x*d_in,y*d_in,np.angle(U_prop),
           cmap=reversed_color_map,shading='auto')
plt.title('U_prop phase full')
plt.axis('equal')
plt.xlabel('m')
plt.ylabel('m')
plt.colorbar()
plt.savefig('U_prop_phase_full')
plt.clf()

plt.pcolor(x*d_in,y*d_in,np.abs(U_prop),
           cmap=reversed_color_map,shading='auto')
plt.title('U_prop abs full')
plt.axis('equal')
plt.xlabel('m')
plt.ylabel('m')
plt.colorbar()
plt.savefig('U_prop_abs_full')
plt.clf()

plt.pcolor(x[edge1:edge2,edge1:edge2]*d_out,y[edge1:edge2,edge1:edge2]*d_out,
           np.angle(U_out[edge1:edge2,edge1:edge2]),
           cmap=reversed_color_map,shading='auto')
plt.title('U_out phase')
plt.axis('equal')
plt.xlabel('m')
plt.ylabel('m')
plt.colorbar()
plt.savefig('U_out_phase')
plt.clf()

plt.pcolor(x[edge1:edge2,edge1:edge2]*d_out,y[edge1:edge2,edge1:edge2]*d_out,
           np.abs(U_out[edge1:edge2,edge1:edge2]),
           cmap=reversed_color_map,shading='auto')
plt.title('U_out abs')
plt.axis('equal')
plt.xlabel('m')
plt.ylabel('m')
plt.colorbar()
plt.savefig('U_out_abs')
plt.clf()

plt.pcolor(x*d_out,y*d_out,np.angle(U_out),
           cmap=reversed_color_map,shading='auto')
plt.title('U_out phase full')
plt.axis('equal')
plt.xlabel('m')
plt.ylabel('m')
plt.colorbar()
plt.savefig('U_out_phase_full')
plt.clf()

plt.pcolor(x*d_out,y*d_out,np.abs(U_out),
           cmap=reversed_color_map,shading='auto')
plt.title('U_out abs full')
plt.axis('equal')
plt.xlabel('m')
plt.ylabel('m')
plt.colorbar()
plt.savefig('U_out_abs_full')
plt.clf()

################################################################################
## xx ## n layers propagation ##################################################
################################################################################
##
################################################################################
# len(U)=num_layer+1            index from 0 to num_layer
# len(phase_scrn)=num_layer     index from 0 to num_layer-1
    
# U[num_layer-i]                Uin
# U[num_layer-i-1]              Uout
# phase_scrn[num_layer-i-1]     one_phase_scrn
    
# example
# n=4
# [0,1,2,3]
#   ground                              space
#   u0      u1      u2      u3      uspace/4
#       ps0     ps1     ps2     ps3
# perform simulation from top down
# numlayer+1 ======================> numlayer
# Uin + PS ==> Utoprop ==> propagate ==> Uout
    
# edge1=int((scrn_size-sample_scrn_size)/2)
# edge2=int((scrn_size+sample_scrn_size)/2)

# for i in np.arange(num_layer):    # only calculate between layers so..
#                                     # ref for the top layer
    
#     d_in=pixel_scale[num_layer-1-i]
#     r=scrnStrengths[num_layer-1-i]
#     L=L0[num_layer-1-i]
#     z=scrnHeights[num_layer-1-i]-scrnHeights[num_layer-2-i]
#     d_out=pixel_scale[num_layer-2-i]
    
#     # Uin
#     U_in=U[num_layer-i]
                                    
#     # get additive phase screen     # get top layer turbulence
    
#     # PS
#     one_phase_scrn=infinitephasescreen.PhaseScreenVonKarman(scrn_size,d_in,r,L)
#     # save additive phase screen    # save it
#     phase_scrn[num_layer-i-1]=one_phase_scrn.scrn
    
#     # abs & phase in
#     # get input phase screen
#     phase_in=np.angle(U_in)
#     abs_in=np.abs(U_in)
    
#     # Utoprop
#     # add additive phase screen to the input phase screen 
#                                     # add the top layer turbulence to the space 
#                                     # flat wavefront
#     phase_to_prop=phase_in+one_phase_scrn.scrn
#     U_prop=abs_in*np.exp(1j*phase_to_prop)
    
#     plt.pcolor(x[edge1:edge2,edge1:edge2]*d_in,y[edge1:edge2,edge1:edge2]*d_in,np.angle(U_prop[edge1:edge2,edge1:edge2]),cmap=reversed_color_map,shading='auto')
#     plt.title('U_prop phase_{:d}'.format(num_layer-i))
#     plt.axis('equal')
#     plt.xlabel('m')
#     plt.ylabel('m')
#     plt.colorbar()
#     plt.savefig('U_prop_phase_{:d}'.format(num_layer-i))
#     plt.clf()
    
#     plt.pcolor(x[edge1:edge2,edge1:edge2]*d_in,y[edge1:edge2,edge1:edge2]*d_in,np.abs(U_prop[edge1:edge2,edge1:edge2]),cmap=reversed_color_map,shading='auto')
#     plt.title('U_prop abs_{:d}'.format(num_layer-i))
#     plt.axis('equal')
#     plt.xlabel('m')
#     plt.ylabel('m')
#     plt.colorbar()
#     plt.savefig('U_prop_abs_{:d}'.format(num_layer-i))
#     plt.clf()
    
#     # propagate
#     # U_out
#     if z != 0:
#         U_out=opticalpropagation.angularSpectrum(U_prop,wvl,d_in,d_out,z)
#     else:
#         U_out=U_prop
#     # save it
#     U[num_layer-i-1]=U_out
    
#     plt.pcolor(x[edge1:edge2,edge1:edge2]*d_out,y[edge1:edge2,edge1:edge2]*d_out,np.angle(U_out[edge1:edge2,edge1:edge2]),cmap=reversed_color_map,shading='auto')
#     plt.title('U_out phase_{:d}'.format(num_layer-i))
#     plt.axis('equal')
#     plt.xlabel('m')
#     plt.ylabel('m')
#     plt.colorbar()
#     plt.savefig('U_out_phase_{:d}'.format(num_layer-i))
#     plt.clf()

#     plt.pcolor(x[edge1:edge2,edge1:edge2]*d_out,y[edge1:edge2,edge1:edge2]*d_out,np.abs(U_out[edge1:edge2,edge1:edge2]),cmap=reversed_color_map,shading='auto')
#     plt.title('U_out abs_{:d}'.format(num_layer-i))
#     plt.axis('equal')
#     plt.xlabel('m')
#     plt.ylabel('m')
#     plt.colorbar()
#     plt.savefig('U_out_abs_{:d}'.format(num_layer-i))
#     plt.clf()

################################################################################


