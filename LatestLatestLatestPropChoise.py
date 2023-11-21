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
from soapy import soapy

def next_2_power(n):
    N = int(2**(np.ceil(np.log(n)/np.log(2))))
    return N

RAD2ARCSEC = 360*60*60/(2*np.pi)
ARCSEC2RAD = 1/RAD2ARCSEC
TURBULENCE_SPREADING_FACTOR = 2
HALFTURBULENCEHEIGHT = 5e3
MAX_N = 2048

r0 = 0.05           # fried parameter
L0 = 4              # outer scale
wvl = 0.5e-6        # wavelength
D = 0.4            # telescope pupil
d = 0.05            # wfs subaperture
subapPxl = 16
wfssSeparation = 10*ARCSEC2RAD


subapFocalPlateScale = wvl/2./d
subapFoV = subapPxl*subapFocalPlateScale
delta = d/subapPxl

numSubap = int(np.floor(D//d))
numLayer = numSubap
numProp = numLayer - 1
layerSeparation = d/wfssSeparation

turbulenceStrength = 1/2**(np.arange(numLayer)*layerSeparation/HALFTURBULENCEHEIGHT)
turbulenceStrength[-1]=(2-turbulenceStrength[:-1].sum()*layerSeparation/HALFTURBULENCEHEIGHT)

# turbulenceStrength = np.ones(numLayer)
# turbulenceStrength[0] = (numLayer - 2)
# turbulenceStrength[-1] = (numLayer - 2)

turbulenceStrength /= turbulenceStrength.sum()
distributedr0 = (turbulenceStrength*r0**(-5./3.))**(-3./5.)

HA = layerSeparation*np.arange(numLayer)

oneWFSField = D + HA*subapFoV

oneWFSFieldWithDiffusion = np.zeros(numLayer)
oneWFSFieldWithDiffusion[0] = D
alpha_max = subapFoV/2.
alpha_max_list = np.zeros(numLayer)
alpha_max_list[0] = subapFoV/2.
for ilayer in np.arange(numProp):
    alpha_max += TURBULENCE_SPREADING_FACTOR*wvl/distributedr0[ilayer + 1]/2.
    alpha_max_list[ilayer + 1] = alpha_max
    oneWFSFieldWithDiffusion[ilayer + 1] += (oneWFSFieldWithDiffusion[ilayer]
                                             + 2.*alpha_max_list[ilayer + 1]*layerSeparation)
    # print(alpha_max, alpha_max_list, oneWFSFieldWithDiffusion)

if (alpha_max_list/wvl >= 1/2./delta).sum() > 0 :
    print('current pixel scale cannot contain enough frequency')
    print('suggest pixel scale to be smaller than {}'.format(wvl/2./alpha_max_list.max()))
    quit()
    
NList = np.zeros(numLayer, dtype=int)
NList[0] = next_2_power(oneWFSFieldWithDiffusion[0]/delta)
for ilayer in np.arange(numProp):
    NList[ilayer + 1] = next_2_power(oneWFSFieldWithDiffusion[ilayer + 1]/delta)

D1 = oneWFSFieldWithDiffusion[1:]
D2 = oneWFSFieldWithDiffusion[:-1]

suggestZList = np.zeros(numProp)
suggestZList = np.min([(NList[1:]*delta - D1/2. - D2/2.)/(alpha_max_list[1:]),
                       NList[1:]/(wvl/delta**2)], axis=0)
partialPropSteps = np.array(np.ceil(layerSeparation/suggestZList), dtype=int)
partialZList = layerSeparation/partialPropSteps


print('enough pixel to host input', NList[1:] >= D1/delta)
print('enough pixel to host output', NList[1:] >= D2/delta)
print('enough input to cover output', D1 >= D2 + 2.*alpha_max_list[1:]*layerSeparation)
print('no wrapping', NList[1:]*delta >= D1/2. + D2/2. + alpha_max_list[1:]*partialZList)
print('no aliasing', NList[1:] >= wvl*partialZList/delta**2)
print('finalized propagation parameters\n'
      + 'N={}\n'.format(NList[1:])
      + 'delta={}\n'.format(delta)
      + 'wvl={}\n'.format(wvl)
      + 'partial propagation distance={}\n'.format(partialZList)
      + 'number of partial propagation={}\n'.format(partialPropSteps)
      + 'D1={}\n'.format(D1)
      + 'D2={}\n'.format(D2))

if (NList >= MAX_N).sum() > 0 :
    print('we cannot afford this much memory')
    quit()

twoWFSFieldWithDiffusion = oneWFSFieldWithDiffusion + wfssSeparation*HA

twoPhasescreenSize = np.array(np.ceil(twoWFSFieldWithDiffusion/delta/2)*2, dtype=int)
onePhasescreenSize = np.array(np.ceil(oneWFSFieldWithDiffusion/delta/2)*2, dtype=int)

try:
    phasescreenList
    for ilayer in np.arange(numLayer):
        if ((phasescreenList[ilayer].nx_size != twoPhasescreenSize[ilayer])
            or (phasescreenList[ilayer].pixelscale != delta)
            or (phasescreenList[ilayer].r0 != distributedr0[ilayer])
            or (phasescreenList[ilayer].L0 != L0)):
            phasescreen = PhaseScreenKolmogorov(twoPhasescreenSize[ilayer],
                                            delta,
                                            distributedr0[ilayer],
                                            L0)
            for row in np.arange(phasescreen.scrn.shape[0]):
                phasescreen.add_row()
            phasescreenList[ilayer] = phasescreen
except:
    phasescreenList = []
    for ilayer in np.arange(numLayer):
        phasescreen = PhaseScreenKolmogorov(twoPhasescreenSize[ilayer],
                                        delta,
                                        distributedr0[ilayer],
                                        L0)
        for row in np.arange(phasescreen.scrn.shape[0]):
            phasescreen.add_row()
        phasescreenList.append(phasescreen)
    for ilayer in np.arange(len(phasescreenList)):
        plt.pcolor(np.arange(phasescreenList[ilayer].scrn.shape[0])*delta,
                    np.arange(phasescreenList[ilayer].scrn.shape[1])*delta,
                    np.angle(np.exp(1j*phasescreenList[ilayer].scrn)),
                    cmap='twilight',
                    norm=colors.CenteredNorm())
        plt.axis('square')
        plt.xticks(np.linspace(0,phasescreenList[ilayer].scrn.shape[0]*delta,num=4))
        plt.yticks(np.linspace(0,phasescreenList[ilayer].scrn.shape[1]*delta,num=4))
        plt.colorbar()
        plt.show()

Ns2Phasescreen = np.array(np.ceil(twoWFSFieldWithDiffusion/delta/2)*2,dtype=int)
Ns1Phasescreen = np.array(np.ceil(oneWFSFieldWithDiffusion/delta/2)*2,dtype=int)
centerPoints = np.array(HA*wfssSeparation/2.//delta, dtype=int)


lower_bound1 = (twoPhasescreenSize - onePhasescreenSize)//2
upper_bound1 = (twoPhasescreenSize + onePhasescreenSize)//2
lower_bound2 = (twoPhasescreenSize - onePhasescreenSize)//2
upper_bound2 = (twoPhasescreenSize + onePhasescreenSize)//2
onAxisSize = int(np.ceil(D/delta))

WFS1ScreenList = []
WFS2ScreenList = []
for ilayer in np.arange(numLayer):
    screen1 = phasescreenList[ilayer].scrn[lower_bound1[ilayer] + centerPoints[ilayer]
                                            : upper_bound1[ilayer] + centerPoints[ilayer],
                                            lower_bound1[ilayer]
                                            : upper_bound1[ilayer]]
    screen2 = phasescreenList[ilayer].scrn[lower_bound2[ilayer] - centerPoints[ilayer]
                                            : upper_bound2[ilayer] - centerPoints[ilayer],
                                            lower_bound2[ilayer]
                                            : upper_bound2[ilayer]]
    WFS1ScreenList.append(screen1)
    WFS2ScreenList.append(screen2)
    
for ilayer in np.arange(len(phasescreenList)):
    plt.pcolor(np.arange(Ns2Phasescreen[ilayer])*delta,
                np.arange(Ns2Phasescreen[ilayer])*delta,
                np.angle(np.exp(1j*phasescreenList[ilayer].scrn)),
                cmap='twilight',
                norm=colors.CenteredNorm())
    plt.axis('square')
    plt.xticks(np.linspace(0,Ns2Phasescreen[ilayer]*delta,num=4))
    plt.yticks(np.linspace(0,Ns2Phasescreen[ilayer]*delta,num=4))
    plt.colorbar()
    plt.hlines([(lower_bound1[ilayer] + centerPoints[ilayer])*delta,
                (upper_bound1[ilayer] + centerPoints[ilayer])*delta],
                xmin=(lower_bound1[ilayer])*delta,
                xmax=(upper_bound1[ilayer])*delta,
                label='wfs1',color='g')
    plt.vlines([(lower_bound1[ilayer])*delta,
                (upper_bound1[ilayer])*delta],
                ymin=(lower_bound1[ilayer] + centerPoints[ilayer])*delta,
                ymax=(upper_bound1[ilayer] + centerPoints[ilayer])*delta,
                color='g')
    plt.hlines([(lower_bound1[ilayer] - centerPoints[ilayer])*delta,
                (upper_bound1[ilayer] - centerPoints[ilayer])*delta],
                xmin=(lower_bound1[ilayer])*delta,
                xmax=(upper_bound1[ilayer])*delta,
                label='wfs2',color='r')
    plt.vlines([(lower_bound1[ilayer])*delta,
                (upper_bound1[ilayer])*delta],
                ymin=(lower_bound1[ilayer] - centerPoints[ilayer])*delta,
                ymax=(upper_bound1[ilayer] - centerPoints[ilayer])*delta,
                color='r')
    plt.hlines([(twoPhasescreenSize[ilayer]//2 + onAxisSize//2 + centerPoints[ilayer])*delta,
                (twoPhasescreenSize[ilayer]//2 - onAxisSize//2 + centerPoints[ilayer])*delta],
                xmin=(twoPhasescreenSize[ilayer]//2 - onAxisSize//2)*delta,
                xmax=(twoPhasescreenSize[ilayer]//2 + onAxisSize//2)*delta,
                label='on-axis wfss1',color='c')
    plt.vlines([(twoPhasescreenSize[ilayer]//2 + onAxisSize//2)*delta,
                (twoPhasescreenSize[ilayer]//2 - onAxisSize//2)*delta],
                ymin=(twoPhasescreenSize[ilayer]//2 - onAxisSize//2 + centerPoints[ilayer])*delta,
                ymax=(twoPhasescreenSize[ilayer]//2 + onAxisSize//2 + centerPoints[ilayer])*delta,
                color='c')
    plt.hlines([(twoPhasescreenSize[ilayer]//2 + onAxisSize//2 - centerPoints[ilayer])*delta,
                (twoPhasescreenSize[ilayer]//2 - onAxisSize//2 - centerPoints[ilayer])*delta],
                xmin=(twoPhasescreenSize[ilayer]//2 - onAxisSize//2)*delta,
                xmax=(twoPhasescreenSize[ilayer]//2 + onAxisSize//2)*delta,
                label='on-axis wfss2',color='m')
    plt.vlines([(twoPhasescreenSize[ilayer]//2 + onAxisSize//2)*delta,
                (twoPhasescreenSize[ilayer]//2 - onAxisSize//2)*delta],
                ymin=(twoPhasescreenSize[ilayer]//2 - onAxisSize//2 - centerPoints[ilayer])*delta,
                ymax=(twoPhasescreenSize[ilayer]//2 + onAxisSize//2 - centerPoints[ilayer])*delta,
                color='m')
    plt.hlines([(twoPhasescreenSize[ilayer] + centerPoints[ilayer])*delta/2. - oneWFSField[ilayer]/2.,
                (twoPhasescreenSize[ilayer] + centerPoints[ilayer])*delta/2. + oneWFSField[ilayer]/2.],
                xmin=(twoPhasescreenSize[ilayer])*delta/2. - oneWFSField[ilayer]/2.,
                xmax=(twoPhasescreenSize[ilayer])*delta/2. + oneWFSField[ilayer]/2.,
                label='wfs1 no scattering FoV',color='b')
    plt.vlines([(twoPhasescreenSize[ilayer])*delta/2. - oneWFSField[ilayer]/2.,
                (twoPhasescreenSize[ilayer])*delta/2. + oneWFSField[ilayer]/2.],
                ymin=(twoPhasescreenSize[ilayer] + centerPoints[ilayer])*delta/2. - oneWFSField[ilayer]/2.,
                ymax=(twoPhasescreenSize[ilayer] + centerPoints[ilayer])*delta/2. + oneWFSField[ilayer]/2.,
                color='b')
    plt.hlines([(twoPhasescreenSize[ilayer] - centerPoints[ilayer])*delta/2. - oneWFSField[ilayer]/2.,
                (twoPhasescreenSize[ilayer] - centerPoints[ilayer])*delta/2. + oneWFSField[ilayer]/2.],
                xmin=(twoPhasescreenSize[ilayer])*delta/2. - oneWFSField[ilayer]/2.,
                xmax=(twoPhasescreenSize[ilayer])*delta/2. + oneWFSField[ilayer]/2.,
                label='wfs2 no scattering FoV',color='y')
    plt.vlines([(twoPhasescreenSize[ilayer])*delta/2. - oneWFSField[ilayer]/2.,
                (twoPhasescreenSize[ilayer])*delta/2. + oneWFSField[ilayer]/2.],
                ymin=(twoPhasescreenSize[ilayer] - centerPoints[ilayer])*delta/2. - oneWFSField[ilayer]/2.,
                ymax=(twoPhasescreenSize[ilayer] - centerPoints[ilayer])*delta/2. + oneWFSField[ilayer]/2.,
                color='y')
    # plt.legend()
    plt.title('generated phase screen')
    plt.show()
    # plt.pcolor(np.arange(Ns1Phasescreen[ilayer])*delta,
    #             np.arange(Ns1Phasescreen[ilayer])*delta,
    #             np.angle(np.exp(1j*WFS1ScreenList[ilayer])),
    #             cmap='twilight',
    #             norm=colors.CenteredNorm())
    # plt.axis('square')
    # plt.xticks(np.linspace(0,Ns1Phasescreen[ilayer]*delta,num=4))
    # plt.yticks(np.linspace(0,Ns1Phasescreen[ilayer]*delta,num=4))
    # plt.colorbar()
    # plt.title('feed to wfs1')
    # plt.show()
    # plt.pcolor(np.arange(Ns1Phasescreen[ilayer])*delta,
    #             np.arange(Ns1Phasescreen[ilayer])*delta,
    #             np.angle(np.exp(1j*WFS2ScreenList[ilayer])),
    #             cmap='twilight',
    #             norm=colors.CenteredNorm())
    # plt.axis('square')
    # plt.xticks(np.linspace(0,Ns1Phasescreen[ilayer]*delta,num=4))
    # plt.yticks(np.linspace(0,Ns1Phasescreen[ilayer]*delta,num=4))
    # plt.colorbar()
    # plt.title('feed to wfs2')
    # plt.show()

try:
    PROPAGATOR_LIST
    for iprop in np.flip(np.arange(numProp) + 1):
        if ((PROPAGATOR_LIST[iprop - 1].wvl != wvl)
            or (PROPAGATOR_LIST[iprop - 1].delta != delta)
            or (PROPAGATOR_LIST[iprop - 1].N != NList[iprop])
            or (PROPAGATOR_LIST[iprop - 1].Z != layerSeparation)
            or (PROPAGATOR_LIST[iprop - 1].prop_number != partialPropSteps[iprop - 1])
            or (PROPAGATOR_LIST[iprop - 1].alpha_max != alpha_max_list[iprop])):
            PROPAGATOR = mpc.ConsecutivePropSquare(wvl,
                                                   delta,
                                                   NList[iprop],
                                                   layerSeparation,
                                                   partialPropSteps[iprop - 1],
                                                   alpha_max_list[iprop])
            PROPAGATOR_LIST[iprop - 1] = PROPAGATOR
        # print(wvl,delta,NList[iprop],layerSeparation,partialPropSteps[iprop - 1],alpha_max_list[iprop])
except:
    PROPAGATOR_LIST = []
    for iprop in (np.arange(numProp) + 1):
        PROPAGATOR = mpc.ConsecutivePropSquare(wvl,
                                               delta,
                                               NList[iprop],
                                               layerSeparation,
                                               partialPropSteps[iprop - 1],
                                               alpha_max_list[iprop])
        PROPAGATOR_LIST.append(PROPAGATOR)


TIME = 10
WIND = NList.min()//4
for time in np.arange(TIME):
    no_prop_screen1 = np.zeros((NList[0],NList[0]), dtype=float)
    no_prop_screen2 = np.zeros((NList[0],NList[0]), dtype=float)
    for ilayer in np.arange(numLayer):
        for row in np.arange(WIND):
            phasescreenList[ilayer].add_row()
        no_prop_screen1 += phasescreenList[ilayer].scrn[
            (twoPhasescreenSize[ilayer]//2 - onAxisSize//2)
            : (twoPhasescreenSize[ilayer]//2 + onAxisSize//2),
            (twoPhasescreenSize[ilayer]//2 - onAxisSize//2 + centerPoints[ilayer])
            : (twoPhasescreenSize[ilayer]//2 + onAxisSize//2 + centerPoints[ilayer])]
        no_prop_screen2 += phasescreenList[ilayer].scrn[
            (twoPhasescreenSize[ilayer]//2 - onAxisSize//2)
            : (twoPhasescreenSize[ilayer]//2 + onAxisSize//2),
            (twoPhasescreenSize[ilayer]//2 - onAxisSize//2 - centerPoints[ilayer])
            : (twoPhasescreenSize[ilayer]//2 + onAxisSize//2 - centerPoints[ilayer])]
    previous_output_screen1 = np.ones(NList.max(), dtype=complex)
    previous_output_screen2 = np.ones(NList.max(), dtype=complex)
    
    for iprop in np.flip(np.arange(numProp) + 1):
        N = NList[iprop]
        inputScreen1 = np.zeros((N,N), dtype=complex)
        output_screen1 = np.zeros((N,N), dtype=complex)
        inputScreen2 = np.zeros((N,N), dtype=complex)
        output_screen2 = np.zeros((N,N), dtype=complex)
        inputScreen1 = np.pad(np.exp(
            1j*phasescreenList[iprop].scrn[lower_bound1[iprop] + centerPoints[iprop]
                                            : upper_bound1[iprop] + centerPoints[iprop],
                                            lower_bound1[iprop]
                                            : upper_bound1[iprop]]),
            pad_width=(((N - Ns1Phasescreen[iprop])//2,(N - Ns1Phasescreen[iprop])//2),
                       ((N - Ns1Phasescreen[iprop])//2,(N - Ns1Phasescreen[iprop])//2)),
            mode='constant', constant_values=(0,0))
        inputScreen2 = np.pad(np.exp(
            1j*phasescreenList[iprop].scrn[lower_bound1[iprop] - centerPoints[iprop]
                                            : upper_bound1[iprop] - centerPoints[iprop],
                                            lower_bound1[iprop]
                                            : upper_bound1[iprop]]),
            pad_width=(((N - Ns1Phasescreen[iprop])//2,(N - Ns1Phasescreen[iprop])//2),
                       ((N - Ns1Phasescreen[iprop])//2,(N - Ns1Phasescreen[iprop])//2)),
            mode='constant', constant_values=(0,0))
        
        inputScreen1 *= previous_output_screen1
        inputScreen2 *= previous_output_screen2
        
        # plt.imshow(np.angle(inputScreen1))
        # plt.show()
        # plt.imshow(np.log(np.abs(inputScreen1)))
        # plt.show()
        # plt.imshow(np.angle(inputScreen2))
        # plt.show()
        # plt.imshow(np.log(np.abs(inputScreen2)))
        # plt.show()
        
        output_screen1 = PROPAGATOR_LIST[iprop - 1].prop(inputScreen1)
        previous_output_screen1 = output_screen1[(N - NList[iprop - 1])//2
                                           : (N + NList[iprop - 1])//2,
                                           (N - NList[iprop - 1])//2
                                           : (N + NList[iprop - 1])//2]
        
        # plt.imshow(np.angle(previous_output_screen1))
        # plt.show()
        # plt.imshow(np.log(np.abs(previous_output_screen1)))
        # plt.show()
        
        output_screen2 = PROPAGATOR_LIST[iprop - 1].prop(inputScreen2)
        previous_output_screen2 = output_screen2[(N - NList[iprop - 1])//2
                                           : (N + NList[iprop - 1])//2,
                                           (N - NList[iprop - 1])//2
                                           : (N + NList[iprop - 1])//2]
        
        # plt.imshow(np.angle(previous_output_screen2))
        # plt.show()
        # plt.imshow(np.log(np.abs(previous_output_screen2)))
        # plt.show()
    
    plt.imshow(np.angle(previous_output_screen1),cmap='twilight')
    plt.title('WFS1 phase {}'.format(time))
    plt.colorbar()
    plt.show()
    plt.imshow((np.abs(previous_output_screen1)**2),vmin=0,vmax=2,cmap='Greys_r')
    plt.title('WFS1 intensity {}'.format(time))
    plt.colorbar()
    plt.show()
    plt.imshow(np.angle(previous_output_screen2),cmap='twilight')
    plt.title('WFS2 phase {}'.format(time))
    plt.colorbar()
    plt.show()
    plt.imshow((np.abs(previous_output_screen2)**2),vmin=0,vmax=2,cmap='Greys_r')
    plt.title('WFS2 intensity {}'.format(time))
    plt.colorbar()
    plt.show()
    
    # shift = np.angle(previous_output_screen1[NList[0]//2,NList[0]//2])
    
    plt.imshow(np.angle(np.exp(1j*(no_prop_screen1))),cmap='twilight')
    plt.title('WFS1 no propagation phase {}'.format(time))
    plt.colorbar()
    plt.show()
    
    # shift = np.angle(previous_output_screen2[NList[0]//2,NList[0]//2])
    
    plt.imshow(np.angle(np.exp(1j*(no_prop_screen2))),cmap='twilight')
    plt.title('WFS2 no propagation phase {}'.format(time))
    plt.colorbar()
    plt.show()
