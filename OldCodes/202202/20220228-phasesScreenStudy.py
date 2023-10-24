################################################################################
## Author:      Puttiwat Kongkaew
## Position:    Ph.D. by research student
## Affiliation: Center for Advanced Instrumentation, Department of Physics,
##              Durham University, United Kingdom
## Email:       gqmc62@durham.ac.uk
## Date:        16 December 2021
################################################################################
## 0 ## What this code does ####################################################
################################################################################
# 1 Import packages
# 2 Read in parameters
# 3 Make circular pupil
# 4 Make power spectrum distribution
# 5 Generate phase screen (much bigger size than 'observe size')
# 6 Make structure function
# 7 Make orthonormalize zernike
# 8 Generate 10,000 phase screens
# 9 Fit zernike using orthonormalized method
# 10 Make residual error plot
################################################################################
print('Beginning of code')



################################################################################
## 1 ## Import packages and set up my preference ###############################
################################################################################
print('Beginning of part 1 : import packages and set up my preference')

# import #######################################################################

from numpy import fft
import numpy as np
import random
import time
from matplotlib import pyplot as plt
from aotools import zernike
from aotools.turbulence import infinitephasescreen

# my preference ################################################################

color_map = plt.cm.get_cmap('Blues')
reversed_color_map = color_map.reversed()

################################################################################
## 2 ## Read in parameters #####################################################
################################################################################
# n:   0 1 2 3  4  5  6   7   8   9   10   11   12   13    14    15    16
# 2^n: 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536

# ???suggested zernike=sample_screen_size/2????
# radialDegree    1   2   3    4   5   6   7   8   9  10  11  12  13  14  15  16  
# zernike(approx) 1   3   6   10  15  21  28  36  45  55  66  78  91 105 120 136 
# radialDegree    17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32
# zernike(approx)153 171 190 210 231 253 276 300 325 351 378 406 435 465 496 528
################################################################################
print('Beginning of part 2 : read in parameters')

# master phase screen ##########################################################

scrn_size=4096          # int; pixel; generated screen size 
                        # might be larger than the sample to fend off fft effect
pixel_scale=0.078125    # m/pixel 0.078125
r0=0.1                  # m
L0=20.0                 # m
l0=1e-10                # m

# structure function ###########################################################

Nrandom=int(1000000)        # int; pixel; samplings for structure function
nbin=100                    # int; pixel; binning for structure function
randomfield=512       # int; pixel; random points within ... px of the screen

# residual error ###############################################################

sample_scrn_size=64        # pixel; int;  for the 'observation'
radialDegree=8              # int; piston = 1, tip/tilt=2,
                            # focus/defocus/astigmatism=3, 
                            # 2trefoils/2comas=4, ...
Nrandomscreens=int(100)   # int; samplings for residual error
Nzernike=int(radialDegree*(radialDegree+1)/2.0) 
                            # zernike modes within given radialDegree



################################################################################
## 3 ## Make circular pupil mask ###############################################
################################################################################
# inside pupil=1=true, outside=0=false
# pupil --> 2d array
# plat_pupil --> flatten 2d array
# centered at the very center
# size is sample_scrn_size
################################################################################
print('Beginning of part 3 : make circular pupil mask')

# make coordinate grid #########################################################
                                # Example
x=np.arange(sample_scrn_size)   # [0,1,2,3]
x=x-(sample_scrn_size-1)/2.0    # [0,1,2,3]-(4-1)/2
                                # [0,1,2,3]-1.5
                                # [-1.5,-0.5,0.5,1.5]
xx,yy=np.meshgrid(x,x)          # x,y value of each point on the grid
r=np.sqrt(xx**2+yy**2)          # radius
pupil=np.zeros(r.shape)         # initialize pupil

# find points inside circle ####################################################

for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        if r[i,j]<=sample_scrn_size/2.0:    # if inside circle
            pupil[i,j]=1                    # pupil = (1 == True)
        else:                               # or else
            pupil[i,j]=0                    # pupil = (0 == false)
            
flat_pupil=pupil.flatten()                  # flat pupil



################################################################################
## 4 ## Make power spectrum density ############################################
################################################################################
# got this section from ft_phase_screen in aotools/phasescreen.py
################################################################################
print('Beginning of part 4 : make power spectrum density')

# function used by ft_phase_screen #############################################

def ift2(G, delta_f, FFT=None):
    """
    Wrapper for inverse fourier transform

    Parameters:
        G: data to transform
        delta_f: pixel seperation
        FFT (FFT object, optional): An accelerated FFT object
    """

    N = G.shape[0]

    if FFT:
        g = np.fft.fftshift(FFT(np.fft.fftshift(G))) * (N * delta_f) ** 2
    else:
        g = fft.ifftshift(fft.ifft2(fft.fftshift(G))) * (N * delta_f) ** 2

    return g

# grab some info from the class ###############################################

phase_screen=infinitephasescreen.PhaseScreenVonKarman(scrn_size,pixel_scale,r0,
                                                      L0)

# generate psd ################################################################

# set up variables as requested and defined by the original code
FFT=None
NN=phase_screen.stencil_length
delta = pixel_scale
l0 = l0
R = random.SystemRandom(time.time())
seed = int(R.random()*100000)
np.random.seed(seed)
del_f = 1./(NN*delta)

# set up grid
fx = np.arange(-NN/2, NN/2) * del_f
(fx, fy) = np.meshgrid(fx,fx)
f = np.sqrt(fx**2. + fy**2.)
fm = 5.92/l0/(2*np.pi)
f0 = 1./L0

# psd calculation
PSD_phi = (0.023*r0**(-5./3.) * np.exp(-1*((f/fm)**2)) 
           / (((f**2) + (f0**2))**(11./6)))
PSD_phi[int(NN/2), int(NN/2)] = 0

# plots ########################################################################

# plot
plt.plot(fx[int(np.ceil(scrn_size/2.0)),int(np.ceil(scrn_size/2.0))+1:-1],
         PSD_phi[int(np.ceil(scrn_size/2.0)),int(np.ceil(scrn_size/2.0))+1:-1],
         ls='-',label='PSD',color='r')
ymax=np.max(PSD_phi[int(np.ceil(scrn_size/2.0)),
        int(np.ceil(scrn_size/2.0))+1:-1])
ymin=np.min(PSD_phi[int(np.ceil(scrn_size/2.0)),
        int(np.ceil(scrn_size/2.0))+1:-1])
plt.vlines(1/L0,ymin,ymax,label='L0',color='g')
plt.vlines(1/pixel_scale,ymin,ymax,label='pixel',color='b')
plt.vlines(1/(pixel_scale*scrn_size),ymin,ymax,
        label='generated screen size',color='c')
plt.vlines(1/(pixel_scale*sample_scrn_size),ymin,ymax,
        label='sample screen size',color='m')

# add plot info
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('f (1/m)')
plt.ylabel('PSD_phi')
plt.grid(True,which="both", linestyle='--')
plt.title('PSD_phi vs f (cross section), {:d}x{:d}px, scale={:.2f}\n'.format(
    scrn_size,scrn_size,pixel_scale)
          +'r0={:.2f}m, L0={:.2f}m, l0={:.2e}m'.format(r0,L0,l0))
plt.savefig('PSD-cross-section')
plt.clf()



################################################################################
## 5 ## Generate phase screen ##################################################
################################################################################
# phase screen generation is continuous from the previous section
# same source with the previous section
# got this section from ft_phase_screen in aotools/phasescreen.py
################################################################################
print('Beginning of part 5 : generate phase screen')

# generate phase screen ########################################################

cn = ((np.random.normal(size=(NN, NN))+1j * np.random.normal(size=(NN, NN))) 
      * np.sqrt(PSD_phi)*del_f)
phase = ift2(cn, 1, FFT).real

# plots ########################################################################

# plot
plt.imshow(phase[0:sample_scrn_size,0:sample_scrn_size],cmap=reversed_color_map)

# plot info
plt.xlabel('x (pixel)')
plt.ylabel('y (pixel)')
plt.title('Masked Generated Phase Screen\nSample out of {:d}x{:d} px\n'.format(
    scrn_size,scrn_size)+'pixel size={:.2f}m, r0={:.2f}m, L0={:.2f}m'.format(
        pixel_scale,r0,L0))
plt.colorbar()
plt.savefig('sample-phase-screen')
plt.clf()



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

pointsa=np.zeros((Nrandom,2),dtype=int) # [,0]=x [,1]=y
pointsb=np.zeros((Nrandom,2),dtype=int)

for i in np.arange(Nrandom):
    pointsa[i,0]=random.randint(0,randomfield-1)
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
ymax=np.max(var_phase)
ymin=np.min(var_phase)

# plot
plt.errorbar(10**(distancegroup[:-1]+space/2.),var_phase,yerr=ye,ls='',
        marker='o',label='Generated Values',color='r')
plt.plot(10**(distancegroup[:-1]+space/2.),
        6.88*((10**(distancegroup[:-1]+space/2.))/r0)**(5./3.),ls='-',
        label='Theoretical Values',color='g')
plt.vlines(L0,0,10000,label='L0',color='b')
plt.vlines(scrn_size*pixel_scale,0,10000,label='Generated screen size',color='c')
plt.vlines(randomfield*pixel_scale,0,10000,label='Random radius',color='m')
#plt.vlines(pixel_scale,0,10000,label='Pixel size',color='y')
plt.vlines(sample_scrn_size*pixel_scale,0,10000,label='Sample screen size',color='k')

# add info on plot
plt.legend()
plt.xlabel('|d| (m)')
plt.ylabel('Phase Variance (rad^2)')
plt.xscale('log')
plt.yscale('log')
#plt.xlim((1,30))
plt.ylim((10,2000))
plt.grid(True,which="both", linestyle='--')
plt.title('Bin Structure Function')
plt.savefig('bin-structure-function')
plt.show()
plt.clf()



################################################################################
## 7 ## Make orthonormalized zernike corrections ###############################
################################################################################
# based on sample_scrn_size not scrn_size
################################################################################
print('Beginning of part 7 : make orthonormalized zernike corrections')

# make flatten zernike modes ###################################################

area=np.pi*(sample_scrn_size/2.0)**2
flat_modes=[]
modes=zernike.zernikeArray(Nzernike,sample_scrn_size)
for i in np.arange(Nzernike):
    flat_modes.append(flat_pupil*modes[i].flatten())

## Gram-Schmidt Orthogonormalized Method #######################################

# make flatten zernike modes
# flat_modes=[]
# modes=zernike.zernikeArray(Nzernike,sample_scrn_size)
# for i in np.arange(Nzernike):
#    flat_modes.append(flat_pupil*modes[i].flatten())

# Want orthonormalized bases -- z'_i which is orthonormalized z_i
# Z_i=original zernike         -->  orthonormalized but impossible to make
# z_i=replicated zernike       -->  close enough to zernike 
                                  # but not orthonormalized
# z'_i=orthonormalized z_i     -->  close enough to zernike and orthonormalized

# create a function which take in set of not orthonormal bases  -->  z_i
#                   then  return set of orthonormalized bases   -->  z'_i

#https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process

v=flat_modes    # z_i
u=[]            # orthogonalized z'_i but need to be normalized
e=[]            # orthogonalized and normalized z'_i

# orthogonalize
for i in np.arange(Nzernike):   # for each mode (vector)
    if i==0:                    # only the first need no adjustment
        u.append(v[0])
    else:
        u.append(v[i])          # ui=vi+...
        for j in np.arange(i):  # ui=vi-sum of projected v on u<i
            u[i]=u[i]-(np.dot(u[j],v[i])/np.dot(u[j],u[j]))*u[j]

# normalize            
for i in np.arange(Nzernike):
    e.append(u[i]/np.sqrt(np.dot(u[i],u[i])))
    
## TEST ########################################################################

# Make orthonormal matrix
new_orthonormality=np.zeros((Nzernike,Nzernike))

# Normality
for i in np.arange(Nzernike):
    new_orthonormality[i,i]=np.sqrt(np.dot(e[i],e[i]))

# Orthogonality
for i in np.arange(Nzernike):
    for j in np.arange(Nzernike):
        if i!=j:
            if i<j:
                new_orthonormality[i,j]=np.dot(e[i],e[j])
                new_orthonormality[j,i]=new_orthonormality[i,j]

new_orthonormality_copy=new_orthonormality
new_orthonormality_copy[np.where(new_orthonormality==0)]=1e-15

# plot
plt.imshow(np.log(np.abs(new_orthonormality_copy))/np.log(10),
           cmap=reversed_color_map)

# plot info
plt.xlabel('Zernike Mode (j)')
plt.ylabel('Zernike Mode (i)')
plt.title('New bases - orthonormality\nDot Product between each Zernike Mode\n'
          +'log10 of absolute value -inf replaced with 1e-15')
plt.colorbar()
plt.savefig('NewBasesOrthonormality')
plt.clf()



################################################################################
## 8 ## Generate 10,000 phase screens ##########################################
## 9 ## Fit zernike using orthonormalized method ###############################
################################################################################
# of sample_scrn_size which is the 'observe' range. 
# the scrn_size is the 'generate' range
################################################################################
print('Beginning of part 8 & 9 : generate 10,000 phase screens'
+'and fit zernike using orthonormaized method')

new_coeffs=np.zeros((Nrandomscreens,Nzernike)) #new_coeffs[sample,mode]

# gram-schmidt bases (new)
for i in np.arange(Nrandomscreens):
    
    # generate screens: same with section 5
    cn = ((np.random.normal(size=(NN, NN))+1j * np.random.normal(size=(NN, NN))) 
            * np.sqrt(PSD_phi)*del_f)
    phs = ift2(cn, 1, FFT).real
    
    # sample part of screen
    phase=phs[0:sample_scrn_size,0:sample_scrn_size]
    
    # flatten
    phase=phase.flatten()*flat_pupil

    # fit zirnike
    for j in range(Nzernike):
        new_coeffs[i,j]=np.dot(phase,e[j])
    if i%100==0: # to cheer up waiters
        print(i)



################################################################################
## 10 ## Make residual error plot ##############################################
################################################################################
# based on Zernike polynomials and atmospheric turbulence 1975
# by Robert J. Noll
# steps of table IV
################################################################################
print('Beginning of part 10 : make residual error plot')

# calculate variance of each mode  #############################################

# new_coeffs shape = (Nrandomscreens,Nzernike)
new_variance_cns=np.zeros(Nzernike)

for i in np.arange(Nzernike):                   # for each zernike mode
    new_variance_cns[i]=np.var(new_coeffs[:,i]) # calculate var of each mode

# plot #########################################################################

# plot theoretical values
# residual error after n zernike modes has been fitted
# table IV
# for N <= 20
temp=np.array([1.0299,0.582,0.134,0.111,0.0880,
      0.0648,0.0587,0.0525,0.0463,0.0401,
      0.0377,0.0352,0.0328,0.0304,0.0279,
      0.0267,0.0255,0.0243,0.0232,0.0220,0.0208])

res=[]                                      # difference of each residual error
for i in range(len(temp)-1):
    res.append(temp[i]-temp[i+1])
    
if Nzernike>=20:                            # if N>20 add approximation
    temp2=np.arange(22,Nzernike+1)
    res.append(temp[-1]-0.2944*temp2[0]**(-np.sqrt(3)/2.))
    for i in range(len(temp2)-2):
        res.append(0.2944*((temp2[i+1])**(-np.sqrt(3)/2.)
                           -(temp2[i+2])**(-np.sqrt(3)/2.)))
    plt.plot(np.arange(Nzernike-2)+1,
             np.array(res)*(sample_scrn_size*pixel_scale/r0)**(5./3.),
             label='Theoretical Residue',marker='x')
else:
    plt.plot(np.arange(0,Nzernike)+1,
             np.array(res)[0:Nzernike]
             *(sample_scrn_size*pixel_scale/r0)**(5./3.),
             label='Theoretical Residue',marker='x')

# plot values we have
plt.plot(new_variance_cns/area,marker='o',ls='',label='Generated Residue')

# plot info
plt.title('New Zernike-Kolmogoroff Residual Errors')
plt.xlabel('Zernike Modes')
plt.ylabel('Coefficient Variances')
plt.yscale('log')
plt.ylim(0.01,10000)
plt.xlim(0,21)
plt.grid(True,which="both", linestyle='--')
plt.legend()
plt.savefig('NewResidualErrors')
plt.clf()



################################################################################
## zernike fit test ############################################################
################################################################################

# get screens ##################################################################

# observe -- what phase screen really is
observe=phase.reshape((sample_scrn_size,sample_scrn_size))

# calculate -- what we think the phase screen is
temp_calculate=np.zeros(sample_scrn_size**2)
for i in np.arange(Nzernike):
    temp_calculate=temp_calculate+new_coeffs[-1,i]*e[i]
    
calculate=temp_calculate.reshape((sample_scrn_size,sample_scrn_size))

# residue -- what we miss
residue=observe-calculate

# calculate rms of plot ########################################################

obs_rms=0.
calc_rms=0.
res_rms=0.

obs_n=pupil.sum()
calc_n=pupil.sum()
res_n=pupil.sum()

obs_mean=(pupil*observe).sum()/obs_n
calc_mean=(pupil*calculate).sum()/calc_n
res_mean=(pupil*residue).sum()/res_n

for i in np.arange(sample_scrn_size):
    for j in np.arange(sample_scrn_size):
        if pupil[i,j]==1:
            obs_rms=obs_rms+(observe[i,j]-obs_mean)**2
            calc_rms=calc_rms+(calculate[i,j]-calc_mean)**2
            res_rms=res_rms+(residue[i,j]-res_mean)**2
            
obs_rms=np.sqrt(obs_rms/(obs_n-1))
calc_rms=np.sqrt(calc_rms/(calc_n-1))
res_rms=np.sqrt(res_rms/(res_n-1))

# plot test ####################################################################

# set plot range
mapmax=observe.max()
mapmin=observe.min()

# observe
plt.imshow(observe,cmap=reversed_color_map,vmin=mapmin,vmax=mapmax)
plt.xlabel('x (pixel)')
plt.ylabel('y (pixel)')
plt.title('Cropped Generated Phase Screen from {:d}x{:d} px\n'.format(
    int(scrn_size),int(scrn_size))+'pixel size={:.2f}, r0={:.2f},'.format(
        pixel_scale,r0)+' L0={:.2f}, rms={:.2f}rad'.format(L0,obs_rms))
plt.colorbar()
plt.savefig('PhaseScreen')
plt.clf()

# calculate
plt.imshow(calculate,cmap=reversed_color_map,vmin=mapmin,vmax=mapmax)
plt.colorbar()
plt.title('New Fitted {:d} Zernike {:d}x{:d} px\npixel size={:.2f},'.format(
    Nzernike,int(scrn_size),int(scrn_size),pixel_scale)+' r0={:.2f},'.format(
        r0)+' L0={:.2f}, rms={:.2f}rad'.format(L0,calc_rms))
plt.xlabel('x (pixel)')
plt.ylabel('y (pixel)')
plt.savefig('NewFittedZernike')
plt.clf()

# residue
plt.imshow(residue,cmap=reversed_color_map,vmin=mapmin,vmax=mapmax)
plt.colorbar()
plt.title('New Residue of {:d} Zernike from {:d}x{:d} px\n'.format(
    Nzernike,int(scrn_size),int(scrn_size))+'pixel size={:.2f},'.format(
        pixel_scale)+' r0={:.2f}, L0={:.2f}, rms={:.2f}rad'.format(
            r0,L0,res_rms))
plt.xlabel('x (pixel)')
plt.ylabel('y (pixel)')
plt.savefig('NewResidue')
plt.clf()

print('End of code')



################################################################################
## End of File #################################################################
################################################################################