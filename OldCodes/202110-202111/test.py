################################################################################
## Author:      Puttiwat Kongkaew
## Position:    Ph.D. by research student
## Affiliation: Center for Advanced Instrumentation, Department of Physics,
##              Durham University, United Kingdom
## Email:       gqmc62@durham.ac.uk
## Date:        30 November 2021
################################################################################
## 0 ## What this code does ####################################################
################################################################################
#1 Import packages
#2 Read in parameters
#3 Make circular pupil
#4 Make power spectrum distribution
#5 Generate phase screen (much bigger size than 'observe size')
#6 Make structure function
#7 Make orthonormalize zernike
#8 Generate 10,000 phase screens
#9 Fit zernike using orthonormalized method
#10 Make residual error plot
################################################################################

print('Beginning of code')

################################################################################
## 1 ## Import packages ########################################################
################################################################################

print('Beginning of part 1 : import packages')



from numpy import fft
import numpy as np
import random
import time
from matplotlib import pyplot as plt
from aotools import zernike
from aotools.turbulence import infinitephasescreen

################################################################################
## 2 ## Read in parameters #####################################################
################################################################################
#n:   0 1 2 3  4  5  6   7   8   9   10   11   12   13    14    15    16
#2^n: 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536

#???suggested zernike=sample_screen_size/2????
#radialDegree    1   2   3    4   5   6   7   8   9  10  11  12  13  14  15  16  
#zernike(approx) 1   3   6   10  15  21  28  36  45  55  66  78  91 105 120 136 
#radialDegree     17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32
#zernike(approx) 153 171 190 210 231 253 276 300 325 351 378 406 435 465 496 528
################################################################################

print('Beginning of part 2 : read in parameters')

scrn_size=2048          #pixel; int; 
                        #for the whole phase screen;
                        #total pixel number = scrn_size*scrn_size pixel
sample_scrn_size=16     #pixel; int;  for the 'observation'
pixel_scale=0.1         #m/pixel
r0=0.1                  #m
L0=20.0                 #m
l0=1e-10                #m
radialDegree=4          #int; piston = 1, tip/tilt=2,
                        #focus/defocus/astigmatism=3, 
                        #2trefoils/2comas=4, ...
Nrandomscreens=1000      #int; samplings
Nrandom=100000

nbin=100

Nzernike=int(radialDegree*(radialDegree+1)/2.0) 
                        #total number of zernike modes within given radialDegree

color_map = plt.cm.get_cmap('Blues')
reversed_color_map = color_map.reversed()



################################################################################
## 3 ## Make circular pupil ####################################################
################################################################################
#inside pupil=1=true, outside=0=false
#pupil --> 2d array
#plat_pupil --> flatten 2d array
################################################################################

print('Beginning of part 3 : make circular pupil')

x=np.arange(sample_scrn_size)
x=x-(sample_scrn_size-1)/2.0
xx,yy=np.meshgrid(x,x)
r=np.sqrt(xx**2+yy**2)
pupil=np.zeros(r.shape) #inside circle=1=True
for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        if r[i,j]<=sample_scrn_size/2.0:
            pupil[i,j]=1
        else:
            pupil[i,j]=0
            
flat_pupil=pupil.flatten()



################################################################################
## 4 ## Make power spectrum density ############################################
################################################################################
#got this section from ft_phase_screen in aotools/phasescreen.py
################################################################################

print('Beginning of part 4 : make power spectrum density')

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
        g = numpy.fft.fftshift(FFT(numpy.fft.fftshift(G))) * (N * delta_f) ** 2
    else:
        g = fft.ifftshift(fft.ifft2(fft.fftshift(G))) * (N * delta_f) ** 2

    return g

phase_screen=infinitephasescreen.PhaseScreenVonKarman(scrn_size,pixel_scale,r0,
                                                      L0)

################################################################################

FFT=None
N=phase_screen.stencil_length
delta = pixel_scale
l0 = l0
R = random.SystemRandom(time.time())
seed = int(R.random()*100000)
np.random.seed(seed)
del_f = 1./(N*delta)
fx = np.arange(-N/2, N/2) * del_f
(fx, fy) = np.meshgrid(fx,fx)
f = np.sqrt(fx**2. + fy**2.)
fm = 5.92/l0/(2*np.pi)
f0 = 1./L0
PSD_phi = (0.023*r0**(-5./3.) * np.exp(-1*((f/fm)**2)) 
           / (((f**2) + (f0**2))**(11./6)))
PSD_phi[int(N/2), int(N/2)] = 0
cn = ((np.random.normal(size=(N, N))+1j * np.random.normal(size=(N, N))) 
      * np.sqrt(PSD_phi)*del_f)
phs = ift2(cn, 1, FFT).real

# plots ########################################################################

PSD_phi_copy=PSD_phi
PSD_phi_copy[np.where(PSD_phi_copy==0)]=1e-15

plt.imshow(np.log(PSD_phi_copy),cmap=reversed_color_map)
plt.colorbar()
plt.title('log-scale PSD_phi, {:d}x{:d}px, scale={:.2f}\n'.format(
    scrn_size,scrn_size,pixel_scale)
         +'r0={:.2f}m, L0={:.2f}m, l0={:.2e}m'.format(r0,L0,l0))
plt.savefig('PSD-logscale')
plt.clf()

plt.plot(fx[int(np.ceil(scrn_size/2.0)),int(np.ceil(scrn_size/2.0))+1:-1],
         PSD_phi[int(np.ceil(scrn_size/2.0)),int(np.ceil(scrn_size/2.0))+1:-1],
         ls='-')
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

print('Beginning of part 5 : generate phase screen')

#phase screen was actually generated in the last section
#phase_screen=infinitephasescreen.PhaseScreenVonKarman(scrn_size,pixel_scale,r0,
#                                                      L0)
phase=phase_screen.scrn

#plots

plt.imshow(phase[0:sample_scrn_size,0:sample_scrn_size],cmap=reversed_color_map)
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
#choose point a to be 'center' within certain margin from edges
#choose point b to be 'another point'
################################################################################

print('Beginning of part 6 : make structure function')

phase=phase_screen.scrn
pointsa=np.zeros((Nrandom,2),dtype=int) #[,0]=x [,1]=y
pointsb=np.zeros((Nrandom,2),dtype=int)

for i in np.arange(Nrandom):
    pointsa[i,0]=random.randint(0,int((scrn_size-1)/4))
    pointsa[i,1]=random.randint(0,int((scrn_size-1)/4))
    pointsb[i,0]=random.randint(0,int((scrn_size-1)/4))
    pointsb[i,1]=random.randint(0,int((scrn_size-1)/4))

# calculate distance, phase, etc. ##############################################

distance=np.zeros(Nrandom)
phase_difference=np.zeros(Nrandom)

for i in np.arange(Nrandom):
    phase_difference[i]=np.abs(phase[int(pointsa[i,0]),int(pointsa[i,1])]
                               -phase[int(pointsb[i,0]),int(pointsb[i,1])])
    distance[i]=np.sqrt((pointsa[i,0]-pointsb[i,0])**2
                        +(pointsa[i,1]-pointsb[i,1])**2)*pixel_scale
    
# # binning data ... not exactly binning but combining data with the same distance
    
# print('not so binning')
# x=set(distance)
# x=list(x)
# x=np.asarray(x)
# x=np.sort(x)
# ye=np.zeros(len(x))

# y=[]

# for i in x: #for coverd distance
#     z=[]
#     for j in np.arange(Nrandom):
#         if distance[j]==i:
#             z.append(phase_difference[j])
#     y.append(z)

# # calculate variance??? D_phi ##################################################
    
# var_phase=np.zeros(len(y))
# for i in np.arange(len(y)):
#     temp=0.
#     if len(y[i])!=1:
#         for j in np.arange(len(y[i])):
#             temp=temp+(y[i][j]**2)
#         temp=temp/len(y[i])
#         var_phase[i]=temp
#         ye[i]=temp/np.sqrt(2*len(y[i])-2)
#     else:
#         var_phase[i]=np.NaN
#         ye[i]=np.NaN

# # plots ########################################################################
    
# plt.errorbar(x,var_phase,yerr=ye,ls='',marker='.',label='Simulated Values')
# plt.plot(x,6.88*(x/r0)**(5./3.),ls='-',label='Theoretical Values')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('|r| (m)')
# plt.ylabel('$D_\phi$ (rad$^2$)')
# plt.grid(True,which="both", linestyle='--')
# plt.legend()
# plt.title('Structure function\n{:d}x{:d} px, pixel size={:.2f}m,\n'.format(
#     scrn_size,scrn_size,pixel_scale)+'r0={:.2f}m,L0={:.2f}m'.format(r0,L0))
# plt.savefig('structure-function-log-log')
# plt.clf()

# plt.errorbar(x,var_phase,yerr=ye,ls='',marker='.',label='Simulated Values')
# plt.plot(x,6.88*(x/r0)**(5./3.),ls='-',label='Theoretical Values')
# #plt.xscale('log')
# #plt.yscale('log')
# plt.xlim((0,60))
# plt.ylim((0,10000))
# plt.xlabel('|r| (m)')
# plt.ylabel('$D_\phi$ (rad$^2$)')
# plt.grid(True,which="both", linestyle='--')
# plt.legend()
# plt.title('Structure function\n{:d}x{:d} px, pixel size={:.2f}m,\n'.format(
#     scrn_size,scrn_size,pixel_scale)+'r0={:.2f}m,L0={:.2f}m'.format(r0,L0))
# plt.savefig('structure-function')
# plt.clf()

# actual binning ###############################################################
print('really binning')
# plt.plot(distance,phase_difference,ls='',marker='.')
# plt.xscale('log')
# plt.yscale('log')
# plt.clf()

plt.hist(distance,bins=nbin)
plt.clf()
mindis=min(distance)
maxdis=max(distance)
space=(maxdis-mindis)/nbin
distancegroup=space*np.arange(nbin+1)+mindis
phasegroup=[]
ye=np.zeros(nbin)
for i in np.arange(nbin):
    box=[]
    for j in np.arange(Nrandom):
        if (distance[j]<distancegroup[i+1])&(distancegroup[i]<distance[j]):
            box.append(phase_difference[j])
    phasegroup.append(box)
            
var_phase=np.zeros(nbin)
for i in np.arange(nbin):
    temp=0.
    if len(phasegroup[i])!=0:
        for j in np.arange(len(phasegroup[i])):
            temp=temp+(phasegroup[i][j]**2)
        temp=temp/len(phasegroup[i])
        var_phase[i]=temp
        ye[i]=2*temp/np.sqrt(2*len(phasegroup[i])-2)
    else:
        var_phase[i]=np.NaN
        ye[i]=np.NaN

# plot #########################################################################

plt.errorbar(distancegroup[:-1]+space/2.,var_phase,yerr=ye,ls='',marker='o')
plt.plot(distancegroup[:-1]+space/2.,
         6.88*((distancegroup[:-1]+space/2.)/r0)**(5./3.),ls='-',
         label='Theoretical Values')
plt.xscale('log')
plt.yscale('log')
plt.xlim((0,40))
plt.ylim((0,2000))
plt.savefig('bin-structure-function')
plt.clf()



################################################################################
## 7 ## Make orthonormalized zernike corrections ###############################
################################################################################
# based on sample_scrn_size not scrn_size
################################################################################

print('Beginning of part 7 : make orthonormalized zernike corrections')

## Regular bases ## (use for old & my method)
# make flatten zernike modes

flat_modes=[]
modes=zernike.zernikeArray(Nzernike,sample_scrn_size)
for i in np.arange(Nzernike):
    flat_modes.append(flat_pupil*modes[i].flatten())

# TESTS ########################################################################
# Make orthonormal matrix
orthonormality=np.zeros((Nzernike,Nzernike))

area=np.pi*(sample_scrn_size/2.0)**2

## Gram-Schmidt Orthogonormalized Method #######################################

#make flatten zernike modes
#flat_modes=[]
#modes=zernike.zernikeArray(Nzernike,sample_scrn_size)
#for i in np.arange(Nzernike):
#    flat_modes.append(flat_pupil*modes[i].flatten())

#Want orthonormalized bases -- z'_i which is orthonormalized z_i
#Z_i=original zernike         -->  orthonormalized but impossible to make
#z_i=replicated zernike       -->  close enough to zernike 
                                  #but not orthonormalized
#z'_i=orthonormalized z_i     -->  close enough to zernike and orthonormalized

#create a function which take in set of not orthonormal bases  -->  z_i
#                  then  return set of orthonormalized bases   -->  z'_i

#https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process

v=flat_modes #z_i
N=len(v)
u=[] #orthogonalized z'_i but need to be normalized
e=[] #orthogonalized and normalized z'_i
for i in np.arange(N):
    if i==0:
        u.append(v[0])
    else:
        u.append(v[i])
        for j in np.arange(i):
            u[i]=u[i]-(np.dot(u[j],v[i])/np.dot(u[j],u[j]))*u[j]
            
for i in np.arange(N):
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

plt.imshow(np.log(np.abs(new_orthonormality_copy))/np.log(10),
           cmap=reversed_color_map)
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

print('Beginning of part 8 & 9 : generate 10,000 phase screens'
+'and fit zernike using orthonormaized method')

#of sample_scrn_size which is the 'observe' range. 
#the scrn_size is the 'generate' range
phase_screen=infinitephasescreen.PhaseScreenVonKarman(scrn_size,pixel_scale,r0,
                                                      L0)
#coeffs=np.zeros((Nrandomscreens,Nzernike)) #coeffs[sample,mode]
new_coeffs=np.zeros((Nrandomscreens,Nzernike)) #new_coeffs[sample,mode]

# gram-schmidt bases (new)
for i in np.arange(Nrandomscreens):
    phase_screen.make_initial_screen()
    phase=phase_screen.scrn[0:sample_scrn_size,0:sample_scrn_size]
    phase=phase.flatten()*flat_pupil

    for j in range(Nzernike):
        new_coeffs[i,j]=np.dot(phase,e[j])
    if i%100==0:
        print(i)



################################################################################
## 10 ## Make residual error plot ##############################################
################################################################################

print('Beginning of part 10 : make residual error plot')

#coeffs shape = (Nrandomscreens,Nzernike)
new_variance_cns=np.zeros(Nzernike)

for i in np.arange(Nzernike): # for each zernike mode
    new_variance_cns[i]=np.var(new_coeffs[:,i])

# new bases - plot #############################################################

temp=np.array([1.0299,0.582,0.134,0.111,0.0880,
      0.0648,0.0587,0.0525,0.0463,0.0401,
      0.0377,0.0352,0.0328,0.0304,0.0279,
      0.0267,0.0255,0.0243,0.0232,0.0220,0.0208])

res=[]
for i in range(len(temp)-1):
    res.append(temp[i]-temp[i+1])
    
if Nzernike>=20:
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

plt.plot(new_variance_cns/area,marker='o',ls='',label='Generated Residue')
plt.title('New Zernike-Kolmogoroff Residual Errors')
plt.xlabel('Zernike Modes')
plt.ylabel('Coefficient Variances')
plt.yscale('log')
plt.ylim(0.1,1000)
plt.xlim(0,55)
plt.grid(True,which="both", linestyle='--')
plt.legend()
plt.savefig('NewResidualErrors')
plt.clf()



################################################################################
## zernike fit test ############################################################
################################################################################

## New method

observe=phase.reshape((sample_scrn_size,sample_scrn_size))
temp_calculate=np.zeros(sample_scrn_size**2)
for i in np.arange(Nzernike):
    temp_calculate=temp_calculate+new_coeffs[-1,i]*e[i]
    
calculate=temp_calculate.reshape((sample_scrn_size,sample_scrn_size))
residue=observe-calculate

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

mapmax=observe.max()
mapmin=observe.min()

plt.imshow(observe,cmap=reversed_color_map,vmin=mapmin,vmax=mapmax)
plt.xlabel('x (pixel)')
plt.ylabel('y (pixel)')
plt.title('Cropped Generated Phase Screen from {:d}x{:d} px\n'.format(
    int(scrn_size),int(scrn_size))+'pixel size={:.2f}, r0={:.2f},'.format(
        pixel_scale,r0)+' L0={:.2f}, rms={:.2f}rad'.format(L0,obs_rms))
plt.colorbar()
plt.savefig('PhaseScreen')
plt.clf()

plt.imshow(calculate,cmap=reversed_color_map,vmin=mapmin,vmax=mapmax)
plt.colorbar()
plt.title('New Fitted {:d} Zernike {:d}x{:d} px\npixel size={:.2f},'.format(
    Nzernike,int(scrn_size),int(scrn_size),pixel_scale)+' r0={:.2f},'.format(
        r0)+' L0={:.2f}, rms={:.2f}rad'.format(L0,calc_rms))
plt.xlabel('x (pixel)')
plt.ylabel('y (pixel)')
plt.savefig('NewFittedZernike')
plt.clf()

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