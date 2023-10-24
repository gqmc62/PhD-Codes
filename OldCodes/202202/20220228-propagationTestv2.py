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

import numpy as np
from matplotlib import pyplot as plt
from aotools.turbulence import phasescreen
from aotools import opticalpropagation

color_map = plt.cm.get_cmap('Blues')
reversed_color_map = color_map.reversed()



################################################################################
## 0.2 ## declare some useful functions ########################################
################################################################################

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

## atm data ####################################################################

## rip off from soapy config 'sh_8x8.yaml'
#r0=0.16                                         #m STACKED!!!
#scrnHeights=np.array([0, 5000, 10000, 15000])   #m
#scrnStrengths=np.array([0.5, 0.3, 0.1, 0.1])    #normalized cn2 ratio
#                                                #turbulence=CN2!!!!
#L0=np.array([20, 20, 20, 20])                   #m
#-->ri=[0.07,0.10,0.20,0.20]m

## Using
r0=0.05                                         #m STACKED!!!
l0=1.5e-10
scrnHeights=np.array([0, 5000, 10000, 15000])   #m
scrnStrengths=np.array([0.5, 0.3, 0.1, 0.1])    #normalized cn2 ratio
                                                #turbulence=CN2!!!!
L0=np.array([20, 20, 20, 20])                   #m

## simulation parameters #######################################################
random_seed_num=5
scrn_size=8192              #pixel
sample_scrn_size=8192        #pixel  #for shows only!!!
fov=10.                     #arcsecond  #fov=field diameter!!!
pupil_diameter=4.           #m      #desired pupil size of the ground
d_ground=0.3e-3     #2.5e-2             #m      #desired pixel size of the ground

## optical values ##############################################################

wvl=500e-9                                      #m



################################################################################
## 0.3 ## Calculate parameters from the read-in ################################
################################################################################

## optical parameter ###########################################################

k = 2*np.pi/wvl

## atm parameters ##############################################################

## get turbulence strength of each layer : unstack r0   ## rip off from soapy
normScrnStrengths = scrnStrengths/(sum(scrnStrengths))
scrnStrengths = ( ((r0**(-5./3.))*normScrnStrengths)**(-3./5.) )

## number of layer
num_layer=len(scrnHeights)

## pupil size on each layer
D=np.zeros(num_layer)
d=np.zeros(num_layer)
for i in np.arange(num_layer):
    D[i]=pupil_diameter+fov*(2*np.pi/360.)*(1/60./60.)*scrnHeights[i]

## simulation parameter ########################################################

d[0]=d_ground
for i in 1+np.arange(num_layer-1):
    #d[i]=wvl*(scrnHeights[i]-scrnHeights[i-1])/D[i-1]-d[i-1]*D[i]/D[i-1]
    d[i]=d_ground*D[i]/pupil_diameter

## generate grid
x,y=np.meshgrid(np.arange(-scrn_size/2.,scrn_size/2.),
                np.arange(-scrn_size/2.,scrn_size/2.))

## pupil
pupil=np.ones((scrn_size,scrn_size))                        #no stops
pupil=(x**2+y**2<pupil_diameter**2/4./d[-1]**2)    #pupil stop at [-1]

## initialize wave #############################################################

amp_in=np.ones((scrn_size,scrn_size))     #uniform intensity
phase_in=np.zeros((scrn_size,scrn_size))  #flat front
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

print('ready for simulation')

## levels to propagate between #################################################

level_in=-1
level_out=0

## in level ####################################################################

print('in step')

U_in=amp_in*np.exp(1j*phase_in) #actually no need to redo here lol
phase_in=np.angle(U_in)         #"-------------------------------"
amp_in=np.abs(U_in)             #"-------------------------------"

z_out=0.#scrnHeights[level_out]
D_out=4.#D[level_out]
d_out=0.75e-3#d[level_out]
z_in=15000.#scrnHeights[level_in]
D_in=5.#D[level_in]
#d_in=d[level_in]
d_in=0.9e-3#wvl*(z_in-z_out)/D_out-d_out*D_in/D_out
dz=z_in-z_out


## test whether or not these parameters fit with simulation requirements #######

print('double check parameters')

dinmax=wvl*dz/D_out-d_out*D_in/D_out
doutmax=wvl*dz/D_in-D_out*d_in/D_in
Dinmax=wvl*dz/d_out-D_out*d_in/d_out
Doutmax=wvl*dz/d_in-d_out*D_in/D_out

nmin1=D_in/2./d_in+D_out/2./d_out+wvl*dz/2./d_in/d_out
nmin2=wvl*dz/d_in/d_out

#the left should be less than the right.
print('input scale')
print(d_in,dinmax,d_in<=dinmax)
print('input screen size')
print(D_in,Dinmax,d_in*scrn_size,D_in<=Dinmax,D_in<=d_in*scrn_size)
print('output scale')
print(d_out,doutmax,d_out<=doutmax)
print('output screen size')
print(D_out,Doutmax,d_out*scrn_size,D_out<=Doutmax,D_out<=d_out*scrn_size)
print('number of pixel')
print(nmin1,nmin2,scrn_size,nmin1<=scrn_size,nmin2<=scrn_size)

## add turbulence / get ready to propagate #####################################

print('adding turbulence')

phase_scrn=phasescreen.ft_sh_phase_screen(scrnStrengths[level_in],
                        scrn_size,d_in,L0[level_in],l0,
                        seed=random_seed_num)
phase_add=phase_scrn
amp_prop=amp_in
phase_prop=phase_in+phase_add
U_prop=amp_prop*np.exp(1j*phase_prop)

## propagate ###################################################################

print('propagate')

U_out=opticalpropagation.angularSpectrum(U_prop,wvl,d_in,d_out,dz)

## calculate observing values ##################################################

amp_out=pupil*np.abs(U_out)
phase_out=pupil*np.angle(U_out)



## plot ########################################################################

print('plotting')

## sample region
edge1=int((scrn_size-sample_scrn_size)/2)
edge2=int((scrn_size+sample_scrn_size)/2)

## plot

# plt.pcolor(x*d_in,y*d_in,np.abs(U_prop),
#            cmap=reversed_color_map,shading='auto')
# plt.title('U_prop abs full')
# plt.axis('equal')
# plt.xlabel('m')
# plt.ylabel('m')
# plt.colorbar()
# plt.savefig('U_prop_abs_full')
# plt.clf()

# plt.pcolor(x[edge1:edge2,edge1:edge2]*d_in,y[edge1:edge2,edge1:edge2]*d_in,
#            np.abs(U_prop[edge1:edge2,edge1:edge2]),
#            cmap=reversed_color_map,shading='auto')
# plt.title('U_prop abs')
# plt.axis('equal')
# plt.xlabel('m')
# plt.ylabel('m')
# plt.colorbar()
# plt.savefig('U_prop_abs')
# plt.clf()

# plt.pcolor(x*d_in,y*d_in,np.angle(U_prop),
#            cmap=reversed_color_map,shading='auto')
# plt.title('U_prop phase full')
# plt.axis('equal')
# plt.xlabel('m')
# plt.ylabel('m')
# plt.colorbar()
# plt.savefig('U_prop_phase_full')
# plt.clf()

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

# plt.pcolor(x*d_out,y*d_out,np.angle(U_out),
#            cmap=reversed_color_map,shading='auto')
# plt.title('U_out phase full')
# plt.axis('equal')
# plt.xlabel('m')
# plt.ylabel('m')
# plt.colorbar()
# plt.savefig('U_out_phase_full')
# plt.clf()

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

# plt.pcolor(x*d_out,y*d_out,np.abs(U_out),
#            cmap=reversed_color_map,shading='auto')
# plt.title('U_out abs full')
# plt.axis('equal')
# plt.xlabel('m')
# plt.ylabel('m')
# plt.colorbar()
# plt.savefig('U_out_abs_full')
# plt.clf()

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


