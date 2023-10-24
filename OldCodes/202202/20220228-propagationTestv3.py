################################################################################
## Author:      Puttiwat Kongkaew
## Position:    Ph.D. by research student
## Affiliation: Center for Advanced Instrumentation, Department of Physics,
##              Durham University, United Kingdom
## Email:       gqmc62@durham.ac.uk
## Date:        21 February 2022
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

import random
import numpy as np
from matplotlib import pyplot as plt
from aotools.turbulence import phasescreen
from aotools import opticalpropagation
from aotools import fouriertransform

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
R0=r0
scrnHeights=np.array([0, 5000, 10000, 15000])   #m
scrnStrengths=np.array([0.5, 0.3, 0.1, 0.1])    #normalized cn2 ratio
                                                #turbulence=CN2!!!!
L0=np.array([20, 20, 20, 20])                   #m

l0=1.5e-10 #from average van der vaal radius of the earth atm

level_in=1
level_out=0

## simulation parameters #######################################################
random_seed_num=789
fov=0#10.                     #arcsecond  #fov=field diameter!!!
pupil_diameter=0.8#1.6#0.8#m      #desired pupil size of the ground

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

## simulation parameter ########################################################

## path of atmosphere which we need to have clear simulation
## -> D in Schmidt not L=d*N
fov_expand=fov*(1/3600.)*(2*np.pi/360.)*(scrnHeights[-1]-scrnHeights[0])
straylight=0#0.20 #m from 5 sigma statistics at fov_expand+pupil_diameter size
                # of how much a ray of light can deviate from its origin
D=pupil_diameter+fov_expand+2.*straylight
print(D)
        
## in case only propagations between some layer is planned
## levels to propagate between #################################################

dz=scrnHeights[level_in]-scrnHeights[level_out]
print(dz)
## get resolution & size ## from schmidt
d=0.9*wvl*dz/2./D #resolution
scrn_size=int(np.ceil(1.1*(D/d+wvl*dz/2./d/d))) #size
print(d)
print(scrn_size)
## generate grid
x=np.arange(-scrn_size/2.,scrn_size/2.) # 1d grid
xx,yy=np.meshgrid(x,x) # 2d grid

## pupil
pupil=np.ones((scrn_size,scrn_size))                        #no stops
pupil=(xx**2+yy**2<pupil_diameter**2/4./d**2)    #pupil stop at [-1]

## initialize wave #############################################################

amp_in=np.ones((scrn_size,scrn_size))     #uniform intensity
phase_in=np.zeros((scrn_size,scrn_size))  #flat front
U_in=amp_in*np.exp(1j*phase_in)                 #euler form



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

## now only supports 1 propagation

## in level ####################################################################

print('in step')

z_out=scrnHeights[level_out]
D_out=D
d_out=d
z_in=scrnHeights[level_in]
D_in=D
d_in=d
dz=z_in-z_out

## add turbulence / get ready to propagate #####################################

print('adding turbulence')

phase_scrn=phasescreen.ft_sh_phase_screen(scrnStrengths[level_in],
                        scrn_size,d_in,L0[level_in],l0,
                        seed=random_seed_num)

################################################################################
## 6 ## Make structure function ################################################
################################################################################
# 1) Choose points a and points b within the random field.
# 2) Both a&b should be random!!!
# 3) Since what matter is the distance between a and b which is radial,
#    polar coordinate is the most suitable, ensuring an even distribution of
#    distance sampling.
# 4) However, the phase screen grid is in cartesian, a conversion is needed.
# 5) Though the rounding might affect on the distribution, this shoud be minimal
# 6) calculate distance and phase difference between each two points
#    then bin them together in ***LOG SPACE***
################################################################################
print('Beginning of part 6 : make structure function')

# choose random points #########################################################
Nrandom=int((scrn_size/4)**2/10)
randomfield=int(scrn_size/4)
phase=phase_scrn
pixel_scale=d_in
nbin=int(np.sqrt(scrn_size/4))

pointsa=np.zeros((Nrandom,2),dtype=int) # [,0]=x [,1]=y
pointsb=np.zeros((Nrandom,2),dtype=int)

for i in np.arange(Nrandom):
    pointsa[i,0]=random.randint(8,randomfield-1)# less than 8 pixel is too pixelated
    pointsa[i,1]=random.randint(0,int(np.floor(2*np.pi)))
    pointsb[i,0]=random.randint(int(np.floor(randomfield+1)-2),
            int(np.floor(scrn_size-randomfield-1)))
    pointsb[i,1]=random.randint(int(np.floor(randomfield+1)-2),
            int(np.floor(scrn_size-randomfield-1)))


# convert points coordinate ####################################################

new_pointsa=np.zeros((Nrandom,2),dtype=int)
for i in np.arange(Nrandom):
    new_pointsa[i,0]=int(np.floor(pointsa[i,0]
            *np.cos(pointsa[i,1])))+pointsb[i,0]
    new_pointsa[i,1]=int(np.floor(pointsa[i,0]
            *np.sin(pointsa[i,1])))+pointsb[i,1]

# calculate distance, phase difference, etc. ###################################

distance=np.zeros(Nrandom)
phase_difference=np.zeros(Nrandom)

for i in np.arange(Nrandom):
    phase_difference[i]=np.abs(phase[int(new_pointsa[i,0]),int(new_pointsa[i,1])]
                               -phase[pointsb[i,0],pointsb[i,1]])
    distance[i]=pointsa[i,0]*pixel_scale

# creating bins ################################################################
# **binning in log space**

# calculating bin characteristics
# find range of data
distance_copy=distance[np.where((distance>0)
        &(distance<randomfield*pixel_scale))]
mindis=np.log(min(distance_copy))/np.log(10) # min
maxdis=np.log(max(distance_copy))/np.log(10) # max
space=(maxdis-mindis)/nbin              # size of each bin
distancegroup=space*np.arange(nbin+1)+mindis
                                        # edge of all bins in distance space

# Example: bin 10-20 into 5 bins
# bins: 10-12, 12-14, 14-16, 16-18, 18-20
# There are 6 edges: [10,12,16,18,20]
# nbin=5 --> np.arange(nbin+1) --> np.arange(5+1) --> [0,1,2,3,4,5]
# space=2,mindis=10 --> [0,2,4,6,8,10]+10 --> [10,12,14,16,18,20]

# empty bins in phase difference space
phasegroup=np.empty(nbin,dtype=list)
for i in np.arange(nbin):
    phasegroup[i]=[]

# sorting points into bin =*= ##################################################

# Example: (continuous from the previous subsection)
# bins: 10-12, 12-14, 14-16, 16-18, 18-20
# 12.5 should be in bin 12-14
# 12.5-10 = 2.5 --> 2.5/2 = 1.25 --> 1
# --> fall in bin number 1 which is 12-14

for i in np.arange(Nrandom):    # for all points that we random
    # find which group it belongs based on the example above
    if distance[i]>0:
        group=np.floor((np.log(distance[i])/np.log(10)-mindis)/space)
        if np.isnan(group) == False:
            if group<nbin:  # just in case that data is on the last edge
                if group>=0:
                    phasegroup[int(group)].append(phase_difference[i])
                                                            # put it in a bin

# calculate variance & error of variance of each bin ###########################

# error of error = error/sqrt(2n-2)
# based on page 17 equation 2.8
# in Measurements and Their Uncertainties
# by Ifan G Hughes & Thomas P A Hase
# which refer to Squires 2001, appendix b

# **BUT I WANT ERROR OF VARIANCE**
# by propagation of error: 
# variance = error**2
# (error of variance)**2 = (error of error)**2 * (2*error)**2
# error of variance = error of error * 2 * error
# error of variance = 2*error*error/sqrt(2n-2)
# error of variance = 2*variance/sqrt(2n-2)

var_phase=np.zeros(nbin)    # variance of phase of each bin
ye=np.zeros(nbin)           # error of variance of phase of each bin

for i in np.arange(nbin):                       # for each bin
    temp=0.                                     # average variance
    if len(phasegroup[i])>2:                   # NOT empty bin
        for j in np.arange(len(phasegroup[i])): # for every element in the bin
            temp=temp+(phasegroup[i][j]**2)     # sum difference square

        temp=temp/len(phasegroup[i])            # average difference square
        var_phase[i]=temp                       # IS variance
        ye[i]=2*temp/np.sqrt(2*len(phasegroup[i])-2)    # based on statistics
    else:                                       # if bin is empty
        var_phase[i]=np.NaN                     # no values really
        ye[i]=np.NaN

# plot #########################################################################

# calculate plot param
r0=scrnStrengths[level_in]
A=10**(distancegroup[:-1]+space/2.)
B=var_phase
C=ye
D=10**(distancegroup[:-1]+space/2.)
E=6.88*((10**(distancegroup[:-1]+space/2.))/r0)**(5./3.)

# plot
plt.errorbar(10**(distancegroup[:-1]+space/2.),var_phase,yerr=ye,ls='',
        marker='o',label='Generated Values',color='r')
plt.plot(10**(distancegroup[:-1]+space/2.),
        6.88*((10**(distancegroup[:-1]+space/2.))/r0)**(5./3.),ls='-',
        label='Theoretical Values',color='g')

# add info on plot
plt.legend()
plt.xlabel('|d| (m)')
plt.ylabel('Phase Variance (rad^2)')
plt.xscale('log')
plt.yscale('log')
#plt.xlim((1,30))
plt.ylim((0,100))
plt.grid(True,which="both", linestyle='--')
plt.title('Structure Function before Propagation')
plt.savefig('structure-function-before-propagation')
plt.clf()

################################################################################

amp_in=np.abs(U_in)
phase_in=np.angle(U_in)
log_amp_in=np.log(amp_in)

phase_add=phase_scrn
amp_prop=amp_in
phase_prop=phase_in+phase_add
U_prop=amp_prop*np.exp(1j*phase_prop)

amp_prop=np.abs(U_prop)
phase_prop=np.angle(U_prop)
log_amp_prop=np.log(amp_prop)

## propagate ###################################################################

print('propagate')

U_out=opticalpropagation.angularSpectrum(U_prop,wvl,d_in,d_out,dz)

## calculate observing values ##################################################

amp_out=np.abs(U_out)
phase_out=np.angle(U_out)
log_amp_out=np.log(amp_out)


# ## plot ########################################################################

# print('plotting')

## sample region
sample_scrn_size=int(np.ceil(pupil_diameter/d))
edge1=int((scrn_size-sample_scrn_size)/2)
edge2=int((scrn_size+sample_scrn_size)/2)
print(sample_scrn_size)

# ## plot

plt.pcolor(xx[edge1:edge2,edge1:edge2]*d_in,yy[edge1:edge2,edge1:edge2]*d_in,
           phase_prop[edge1:edge2,edge1:edge2],
           cmap=reversed_color_map,shading='auto')
plt.title('U_prop phase')
plt.axis('equal')
plt.xlabel('m')
plt.ylabel('m')
plt.colorbar()
plt.savefig('U_prop_phase')
plt.clf()

plt.pcolor(xx[edge1:edge2,edge1:edge2]*d_out,yy[edge1:edge2,edge1:edge2]*d_out,
           (pupil*phase_out)[edge1:edge2,edge1:edge2],
           cmap=reversed_color_map,shading='auto',vmin=-np.pi,vmax=np.pi)
plt.title('U_out phase')
plt.axis('equal')
plt.xlabel('m')
plt.ylabel('m')
plt.colorbar()
plt.savefig('U_out_phase')
plt.clf()

plt.pcolor(xx[edge1:edge2,edge1:edge2]*d_out,yy[edge1:edge2,edge1:edge2]*d_out,
           (pupil*log_amp_out)[edge1:edge2,edge1:edge2],
           cmap=reversed_color_map,shading='auto',vmin=-1,vmax=1)
plt.title('U_out log-amp')
plt.axis('equal')
plt.xlabel('m')
plt.ylabel('m')
plt.colorbar()
plt.savefig('U_out_log_amp')
plt.clf()



################################################################################
## 2 ## Analysis on log-amplitude ##############################################
################################################################################
## 1) histogram of the whole screen with variance
## 2) structure function
################################################################################

## histogram ###################################################################
numbin=int(np.floor((sample_scrn_size)))

log_amp=log_amp_out[edge1:edge2,edge1:edge2].flatten()
log_amp_std=np.std(log_amp)
log_amp_mean=np.mean(log_amp)
numpoint=len(log_amp)
print(log_amp_std)
print(log_amp_mean)
print(numpoint)
print(log_amp.max())
print(log_amp.min())
plt.hist(log_amp,bins=numbin)

xdum=np.arange(-3,1,0.01)
ydum=(numpoint*(log_amp.max()-log_amp.min())/numbin
      *np.exp(-0.5*(xdum-log_amp_mean)**2/log_amp_std**2)
      /log_amp_std/np.sqrt(2*np.pi))
plt.plot(xdum,ydum)

plt.xlim((-1,1))
plt.title('Whole screen histogram of log-amplitude')
plt.xlabel('log-amplitude')
plt.ylabel('frequency')
plt.savefig('log-amp-hist')
plt.clf()



## structure function ##########################################################

## find average of ... of points within each ring from the center
## ... = |A(x)-A(0)|^2

################################################################################
## 6 ## Make structure function ################################################
################################################################################
# 1) Choose points a and points b within the random field.
# 2) Both a&b should be random!!!
# 3) Since what matter is the distance between a and b which is radial,
#    polar coordinate is the most suitable, ensuring an even distribution of
#    distance sampling.
# 4) However, the phase screen grid is in cartesian, a conversion is needed.
# 5) Though the rounding might affect on the distribution, this shoud be minimal
# 6) calculate distance and phase difference between each two points
#    then bin them together in ***LOG SPACE***
################################################################################
print('Beginning of part 6 : make structure function')

# choose random points #########################################################



phase=phase_out
pixel_scale=d_out
log_amp=log_amp_out

pointsa=np.zeros((Nrandom,2),dtype=int) # [,0]=x [,1]=y
pointsb=np.zeros((Nrandom,2),dtype=int)

for i in np.arange(Nrandom):
    pointsa[i,0]=random.randint(8,randomfield-1)
    pointsa[i,1]=random.randint(0,int(np.floor(2*np.pi)))
    pointsb[i,0]=random.randint(int(np.floor(randomfield+1)-2),
            int(np.floor(scrn_size-randomfield-1)))
    pointsb[i,1]=random.randint(int(np.floor(randomfield+1)-2),
            int(np.floor(scrn_size-randomfield-1)))


# convert points coordinate ####################################################

new_pointsa=np.zeros((Nrandom,2),dtype=int)
for i in np.arange(Nrandom):
    new_pointsa[i,0]=int(np.floor(pointsa[i,0]
            *np.cos(pointsa[i,1])))+pointsb[i,0]
    new_pointsa[i,1]=int(np.floor(pointsa[i,0]
            *np.sin(pointsa[i,1])))+pointsb[i,1]

# calculate distance, phase difference, etc. ###################################

distance=np.zeros(Nrandom)
phase_difference=np.zeros(Nrandom)
log_amp_difference=np.zeros(Nrandom)

for i in np.arange(Nrandom):
    log_amp_difference[i]=np.abs(log_amp[int(new_pointsa[i,0]),int(new_pointsa[i,1])]
                               -log_amp[pointsb[i,0],pointsb[i,1]])
    tempdif=phase[int(new_pointsa[i,0]),int(new_pointsa[i,1])]-phase[pointsb[i,0],pointsb[i,1]]
    if np.abs(tempdif)<3*np.pi/2.:
        phase_difference[i]=np.abs(tempdif)
    else:
        phase_difference[i]=np.abs(tempdif-2*np.pi)
    distance[i]=pointsa[i,0]*pixel_scale

# creating bins ################################################################
# **binning in log space**

# calculating bin characteristics
# find range of data
distance_copy=distance[np.where((distance>0)
        &(distance<randomfield*pixel_scale))]
mindis=np.log(min(distance_copy))/np.log(10) # min
maxdis=np.log(max(distance_copy))/np.log(10) # max
space=(maxdis-mindis)/nbin              # size of each bin
distancegroup=space*np.arange(nbin+1)+mindis
                                        # edge of all bins in distance space

# Example: bin 10-20 into 5 bins
# bins: 10-12, 12-14, 14-16, 16-18, 18-20
# There are 6 edges: [10,12,16,18,20]
# nbin=5 --> np.arange(nbin+1) --> np.arange(5+1) --> [0,1,2,3,4,5]
# space=2,mindis=10 --> [0,2,4,6,8,10]+10 --> [10,12,14,16,18,20]

# empty bins in phase difference space
phasegroup=np.empty(nbin,dtype=list)
log_ampgroup=np.empty(nbin,dtype=list)
for i in np.arange(nbin):
    phasegroup[i]=[]
    log_ampgroup[i]=[]

# sorting points into bin =*= ##################################################

# Example: (continuous from the previous subsection)
# bins: 10-12, 12-14, 14-16, 16-18, 18-20
# 12.5 should be in bin 12-14
# 12.5-10 = 2.5 --> 2.5/2 = 1.25 --> 1
# --> fall in bin number 1 which is 12-14

for i in np.arange(Nrandom):    # for all points that we random
    # find which group it belongs based on the example above
    if distance[i]>0:
        group=np.floor((np.log(distance[i])/np.log(10)-mindis)/space)
        if np.isnan(group) == False:
            if group<nbin:  # just in case that data is on the last edge
                if group>=0:
                    phasegroup[int(group)].append(phase_difference[i])
                    log_ampgroup[int(group)].append(log_amp_difference[i])
                                                            # put it in a bin

# calculate variance & error of variance of each bin ###########################

# error of error = error/sqrt(2n-2)
# based on page 17 equation 2.8
# in Measurements and Their Uncertainties
# by Ifan G Hughes & Thomas P A Hase
# which refer to Squires 2001, appendix b

# **BUT I WANT ERROR OF VARIANCE**
# by propagation of error: 
# variance = error**2
# (error of variance)**2 = (error of error)**2 * (2*error)**2
# error of variance = error of error * 2 * error
# error of variance = 2*error*error/sqrt(2n-2)
# error of variance = 2*variance/sqrt(2n-2)

var_phase=np.zeros(nbin)    # variance of phase of each bin
var_log_amp=np.zeros(nbin)
ye=np.zeros(nbin)           # error of variance of phase of each bin
yela=np.zeros(nbin)

for i in np.arange(nbin):                       # for each bin
    temp=0.                                     # average variance
    templa=0.
    if len(phasegroup[i])>2:                   # NOT empty bin
        for j in np.arange(len(phasegroup[i])): # for every element in the bin
            temp=temp+(phasegroup[i][j]**2)     # sum difference square
            templa=templa+(log_ampgroup[i][j]**2)

        temp=temp/len(phasegroup[i])            # average difference square
        templa=templa/len(log_ampgroup[i])
        var_phase[i]=temp                       # IS variance
        var_log_amp[i]=templa
        ye[i]=2*temp/np.sqrt(2*len(phasegroup[i])-2)    # based on statistics
        yela[i]=2*templa/np.sqrt(2*len(log_ampgroup[i])-2)
    else:                                       # if bin is empty
        var_phase[i]=np.NaN                     # no values really
        var_log_amp[i]=np.NaN
        ye[i]=np.NaN
        yela[i]=np.NaN

# plot #########################################################################

# calculate plot param

# plot
plt.errorbar(10**(distancegroup[:-1]+space/2.),var_log_amp,yerr=yela,ls='',
        marker='o',label='Generated Log-Amplitude after',color='r')
plt.plot(10**(distancegroup[:-1]+space/2.),
        6.88*((10**(distancegroup[:-1]+space/2.))/r0)**(5./3.),ls='-',
        label='Kolmogorov Phase',color='g')

plt.errorbar(10**(distancegroup[:-1]+space/2.),var_phase,yerr=ye,ls='',
        marker='o',label='Generated Phase after',color='b')

plt.errorbar(A,B,yerr=C,ls='',marker='o',label='Generated Phase before',color='c')

# add info on plot
plt.legend()
plt.xlabel('|d| (m)')
plt.ylabel('Phase Variance / Log Amplitude Variance(rad^2)')
plt.xscale('log')
plt.yscale('log')
#plt.xlim((1,30))
plt.ylim((0,20))
plt.grid(True,which="both", linestyle='--')
plt.title('Structure Function of Propagated Screen')
plt.savefig('structure-function-of-propagated-screen')
plt.clf()

################################################################################
## create image ################################################################
################################################################################

def lensAgainstFT(Uin, wvl, d1,f):
    """
    Fresnel propagation using a one step Fresnel propagation method.

    Parameters:
        Uin (ndarray): A 2-d, complex, input array of complex amplitude
        wvl (float): Wavelength of propagated light in metres
        d1 (float): spacing of input plane
        z (float): metres to propagate along optical axis

    Returns:
        ndarray: Complex ampltitude after propagation
    """
    N = Uin.shape[0]    #Assume square grid
    k = 2*np.pi/wvl  #optical wavevector

    #observation plane coordinates
    fX=np.arange(-N/2.,N/2.)/(N*d1)
    x2,y2 = np.meshgrid( wvl*f*fX,wvl*f*fX )

    #evaluate Fresnel-Kirchoff integral
    A = 1/(1j*wvl*f)
    B = np.exp( 1j * k/(2*f) * (x2**2 + y2**2))
    C = fouriertransform.ft2(Uin , d1)

    Uout = A*B*C

    return Uout

################################################################################

clear=lensAgainstFT(U_in*pupil,wvl,d_out,1)
clearplot=np.abs(clear[edge1:edge2,edge1:edge2])**2/pupil.sum()**2
atm=lensAgainstFT(U_out*pupil,wvl,d_out,1)
atmplot=np.abs(atm[edge1:edge2,edge1:edge2])**2/pupil.sum()**2

plt.pcolor(clearplot,cmap=reversed_color_map)#,vmin=plotmin,vmax=plotmax)
plt.colorbar()
plt.xlabel('px')
plt.ylabel('px')
plt.axis('equal')
plt.title('Image without atmosphere')
plt.savefig('no-atm-image')
plt.clf()


plt.pcolor(atmplot,cmap=reversed_color_map)#,vmin=plotmin,vmax=plotmax)
plt.colorbar()
plt.xlabel('px')
plt.ylabel('px')
plt.axis('equal')
plt.title('Image with atmosphere')
plt.savefig('atm-image')
plt.clf()

################################################################################
## lenslet analysis ############################################################
################################################################################

shwfspupil=(U_out)[edge1:edge2,edge1:edge2]
minipupil=pupil[edge1:edge2,edge1:edge2]
n=shwfspupil.shape[0]
xshwfspupil=np.arange(n)
shwfsscale=d_out
lensletphysicaldistance=R0
numlenslet=int(np.floor(n*shwfsscale/lensletphysicaldistance))
lensletpixel=int(np.floor(lensletphysicaldistance/shwfsscale))

plt.pcolor(xshwfspupil*shwfsscale,xshwfspupil*shwfsscale,np.log(np.abs(shwfspupil)*minipupil),cmap=reversed_color_map,shading='auto',vmin=-1,vmax=1)
plt.axis('equal')
plt.vlines(np.arange(0,n,lensletpixel)*shwfsscale,ymin=0*shwfsscale,ymax=n*shwfsscale,color='k')
plt.hlines(np.arange(0,n,lensletpixel)*shwfsscale,xmin=0*shwfsscale,xmax=n*shwfsscale,color='k')
plt.title('lenslets array on pupil log-amp')
plt.xlabel('m')
plt.ylabel('m')
plt.colorbar()
plt.savefig('lenslets-array-on-pupil-log-amp')
plt.clf()

plt.pcolor(xshwfspupil*shwfsscale,xshwfspupil*shwfsscale,np.angle(shwfspupil)*minipupil,cmap=reversed_color_map,shading='auto')
plt.axis('equal')
plt.vlines(np.arange(0,n,lensletpixel)*shwfsscale,ymin=0*shwfsscale,ymax=n*shwfsscale,color='k')
plt.hlines(np.arange(0,n,lensletpixel)*shwfsscale,xmin=0*shwfsscale,xmax=n*shwfsscale,color='k')
plt.title('lenslets array on pupil phase')
plt.xlabel('m')
plt.ylabel('m')
plt.colorbar()
plt.savefig('lenslets-array-on-pupil-phase')
plt.clf()



################################################################################

lensletsumI=np.zeros((numlenslet,numlenslet))
lensletsumslopex=np.zeros((numlenslet,numlenslet))
lensletsumslopey=np.zeros((numlenslet,numlenslet))

pupilI=np.abs(shwfspupil)**2*minipupil
pupilphase=np.angle(shwfspupil)

pupilslopex=np.zeros((n,n))
pupilslopey=np.zeros((n,n))

for i in np.arange(n):
    for j in np.arange(n):
        if True:#(i-n/2)**2+(j-n/2)**2<((n-4)/2.)**2:
            tempdifx=pupilphase[i-1,j]-pupilphase[i,j]
            if np.abs(tempdifx)<np.pi/2.:
                pupilslopex[i,j]=tempdifx
            else:
                if tempdifx>0:
                    pupilslopex[i,j]=tempdifx-2*np.pi
                else:
                    pupilslopex[i,j]=tempdifx+2*np.pi
            tempdify=pupilphase[i,j-1]-pupilphase[i,j]
            if np.abs(tempdify)<np.pi/2.:
                pupilslopey[i,j]=tempdify
            else:
                if tempdify>0:
                    pupilslopey[i,j]=tempdify-2*np.pi
                else:
                    pupilslopey[i,j]=tempdify+2*np.pi
            pupilslopex[i,j]=pupilslopex[i,j]*wvl/2./np.pi/shwfsscale
            pupilslopey[i,j]=pupilslopey[i,j]*wvl/2./np.pi/shwfsscale    

for i in np.arange(numlenslet):
    for j in np.arange(numlenslet):
        lensletsumI[i,j]=pupilI[(i)*lensletpixel:(i+1)*lensletpixel,(j)*lensletpixel:(j+1)*lensletpixel].sum()
        lensletsumslopex[i,j]=(pupilslopex*minipupil)[(i)*lensletpixel:(i+1)*lensletpixel,(j)*lensletpixel:(j+1)*lensletpixel].sum()
        lensletsumslopey[i,j]=(pupilslopey*minipupil)[(i)*lensletpixel:(i+1)*lensletpixel,(j)*lensletpixel:(j+1)*lensletpixel].sum()
        
lensletsumI=lensletsumI/lensletpixel**2
lensletsumslopex=(lensletsumslopex/lensletpixel**2)*206265*1000
lensletsumslopey=(lensletsumslopey/lensletpixel**2)*206265*1000
pupilslopex=pupilslopex*206265*1000
pupilslopey=pupilslopey*206265*1000


plt.pcolor(np.arange(numlenslet)*lensletphysicaldistance,np.arange(numlenslet)*lensletphysicaldistance,
           lensletsumI,shading='auto',cmap=reversed_color_map)
plt.axis('equal')
plt.colorbar()
plt.xlabel('m')
plt.ylabel('m')
plt.title('Total intensity on each lenslet')
plt.savefig('lensletintensity')
plt.clf()

plt.quiver(np.arange(numlenslet)*lensletphysicaldistance,np.arange(numlenslet)*lensletphysicaldistance,
           lensletsumslopex,lensletsumslopey)
plt.title('lenslet slope direction')
plt.axis('equal')
plt.xlabel('m')
plt.ylabel('m')
plt.savefig('lenslet-slope-direction')
plt.clf()

plt.pcolor(np.arange(numlenslet)*lensletphysicaldistance,np.arange(numlenslet)*lensletphysicaldistance,
           np.sqrt(lensletsumslopex**2+lensletsumslopey**2),shading='auto',cmap=reversed_color_map)
plt.axis('equal')
plt.colorbar()
plt.xlabel('m')
plt.ylabel('m')
plt.title('Slope size on each lenslet (mas)')
plt.savefig('lensletslopesize')
plt.clf()

slopesize=np.sqrt(pupilslopex**2+pupilslopey**2)

plt.pcolor(np.arange(n)*shwfsscale,np.arange(n)*shwfsscale,
           slopesize*minipupil,shading='auto',cmap=reversed_color_map)
plt.axis('equal')
plt.vlines(np.arange(0,n,lensletpixel)*shwfsscale,ymin=0*shwfsscale,ymax=n*shwfsscale,color='k')
plt.hlines(np.arange(0,n,lensletpixel)*shwfsscale,xmin=0*shwfsscale,xmax=n*shwfsscale,color='k')
plt.colorbar()
plt.xlabel('m')
plt.ylabel('m')
plt.title('Slope size on pupil (mas)')
plt.savefig('pupilslopesize')
plt.clf()

image=np.zeros((n,n))
for i in np.arange(numlenslet):
    for j in np.arange(numlenslet):
        pupilplane=(shwfspupil*minipupil)[(i)*lensletpixel:(i+1)*lensletpixel,(j)*lensletpixel:(j+1)*lensletpixel]
        proppupilplane=np.zeros((2*lensletpixel,2*lensletpixel))
        proppupilplane[int(lensletpixel/2):int(lensletpixel*3./2.),int(lensletpixel/2):int(lensletpixel*3./2.)]=pupilplane
        imageplane=lensAgainstFT(proppupilplane,wvl,shwfsscale,0.05)
        image[(i)*lensletpixel:(i+1)*lensletpixel,(j)*lensletpixel:(j+1)*lensletpixel]=np.abs(imageplane[int(lensletpixel/2):int(lensletpixel*3./2.),int(lensletpixel/2):int(lensletpixel*3./2.)])**2/lensletpixel**2
        

plt.pcolor(xshwfspupil,xshwfspupil,image,
           cmap=reversed_color_map,shading='auto',vmax=np.max(image),vmin=np.median(image))
plt.vlines(np.arange(0,n,lensletpixel),ymin=0,ymax=n,color='k')
plt.hlines(np.arange(0,n,lensletpixel),xmin=0,xmax=n,color='k')
plt.colorbar()
plt.axis('equal')
plt.xlabel('pixel')
plt.ylabel('pixel')
plt.title('Intensity image of SHWFS')
plt.savefig('shwfsimage')
plt.clf()

plt.pcolor(xshwfspupil,xshwfspupil,np.log(image),
           cmap=reversed_color_map,shading='auto',vmax=np.max(np.log(image)),vmin=np.median(np.log(image)))
plt.vlines(np.arange(0,n,lensletpixel),ymin=0,ymax=n,color='k')
plt.hlines(np.arange(0,n,lensletpixel),xmin=0,xmax=n,color='k')
plt.colorbar()
plt.axis('equal')
plt.xlabel('pixel')
plt.ylabel('pixel')
plt.title('Ln intensity image of SHWFS')
plt.savefig('lnshwfsimage')
plt.clf()

from scipy import signal





slopesize=np.sqrt(pupilslopex**2+pupilslopey**2)
numbin=1000
#slopesize=signal.medfilt2d(slopesize)
slopesize=np.log(slopesize)
plt.hist(slopesize.flatten(),bins=numbin)

numpoint=len(slopesize.flatten())

xdum=np.arange(0,10,0.01)
ydum=(numpoint*(np.max(slopesize.flatten())-np.min(slopesize.flatten()))/numbin
      *np.exp(-0.5*(xdum-np.mean(slopesize.flatten()))**2/np.std(slopesize.flatten())**2)
      /np.std(slopesize.flatten())/np.sqrt(2*np.pi))
plt.plot(xdum,ydum)

plt.xlim((0,10))

plt.title('Histogram of slope size')
plt.xlabel('slope size (mas)')
plt.ylabel('frequency')
plt.savefig('slope-hist')
plt.clf()

