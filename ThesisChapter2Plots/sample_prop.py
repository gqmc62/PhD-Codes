# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:44:11 2024

@author: gqmc62
"""


from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov
from aotools.opticalpropagation import angularSpectrum
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import pickle


D = 1
delta = 0.05/16
r0 = 0.05
L0 = 5
z = 5e3
wvl = 500e-9

name = 'sample_prop_D{:.0f}m_delta{:.0f}mm_r0{:.0f}cm_L0{:.0f}m_z{:.0f}km_wvl{:.0f}nm'.format(
    D,delta*1e3,r0*1e2,L0,z*1e-3,wvl*1e9)

N = int(np.ceil(D/delta))

if os.path.isfile(name + '.pkl'):
    print('read old file')
    
    pkl_file = open(name + '.pkl', 'rb')
    data = pickle.load(pkl_file)
    
    x = data['x']
    phase_GL = data['phase_GL']
    phase_HA = data['phase_HA']
    intensity_HA = data['intensity_HA']
    pkl_file.close()

else:
    print('make a new one')
    
    x = (np.arange(N)- N//2) * delta
    atm_pad = int(np.ceil(2*wvl*z/r0/delta))
    prop_pad = int(np.ceil(wvl*z/2/delta**2))
    # print(2*atm_pad,prop_pad)

    IF3 = PhaseScreenKolmogorov(N+2*atm_pad, delta, r0, L0,random_seed=999)
    for i in np.arange(N+2*atm_pad):
        IF3.add_row()
    SCRN = IF3.scrn
    SCRN -= SCRN.mean()
    
    plt.imshow(SCRN)
    plt.show()
    
    U_GL = np.exp(1j*SCRN[atm_pad:-atm_pad,atm_pad:-atm_pad])
    U_GL /= U_GL.mean()
    phase_GL = np.angle(U_GL)
    
    atm_pad_scrn = SCRN
    atm_pad_Uin = np.exp(1j*np.pad(atm_pad_scrn,
                                    ((prop_pad-atm_pad,prop_pad-atm_pad),
                                      (prop_pad-atm_pad,prop_pad-atm_pad)),'symmetric'))#'symmetric',reflect_type='even'))
    atm_pad_Uout = angularSpectrum(atm_pad_Uin, wvl, delta, delta, z)[prop_pad:-prop_pad,
                                                                      prop_pad:-prop_pad]
    mean = atm_pad_Uout.mean()
    mean /= np.abs(mean)
    atm_pad_Uout /= mean

    intensity_HA = np.abs(atm_pad_Uout)**2
    phase_HA = np.angle(atm_pad_Uout)
    
    data = {'x' : x,
            'phase_GL' : phase_GL,
            'phase_HA' : phase_HA,
            'intensity_HA' : intensity_HA}
    
    output = open(name + '.pkl', 'wb')
    pickle.dump(data,output)
    output.close()
    
plt.pcolor(x[::2],x[::2],phase_GL[::2,::2])
plt.axis('square')
plt.colorbar()
plt.xlabel('position (m)')
plt.title('phase @ z=0km (rad)')
plt.savefig(name + '_phase0' + '.eps',bbox_inches='tight',dpi=100)
plt.savefig(name + '_phase0' + '.png',bbox_inches='tight',dpi=100)
plt.show()

plt.pcolor(x[::2],x[::2],phase_HA[::2,::2])
plt.axis('square')
plt.colorbar()
plt.xlabel('position (m)')
plt.title('phase @ z={:.0f}km (rad)'.format(z/1e3))
plt.savefig(name + '_phase_z' + '.eps',bbox_inches='tight',dpi=100)
plt.savefig(name + '_phase_z' + '.png',bbox_inches='tight',dpi=100)
plt.show()

plt.pcolor(x[::2],x[::2],intensity_HA[::2,::2])
plt.axis('square')
plt.colorbar()
plt.xlabel('position (m)')
plt.title('intensity @ z={:.0f}km (rad)'.format(z/1e3))
plt.savefig(name + '_intensity_z' + '.eps',bbox_inches='tight',dpi=100)
plt.savefig(name + '_intensity_z' + '.png',bbox_inches='tight',dpi=100)
plt.show()





k = 2*np.pi/wvl
cn2dz = r0**(-5./3.)/0.423/k**2
log_amplitude_variance = 0.563 * k**(7./6.) * cn2dz * z**(5./6.)
true_log_amplitude_variance = np.var(np.log(np.sqrt(intensity_HA)))

plt.clf()
fig,axs = plt.subplots(1,3, sharex=True, sharey=True, squeeze=True)

im = axs[0].pcolor(x[::2],x[::2],phase_GL[::2,::2])
# fig[0].colorbar()
axs[0].axis('square')
# axs[0].set_xlabel('position (m)')
# axs[0].set_title('$\phi$ @ z=0km (rad)')
axs[0].set_xlabel('(a)')
axs[0].set_xticks([])
axs[0].set_yticks([])

divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="5%", pad=0.05)   
plt.colorbar(im, cax=cax)

im = axs[1].pcolor(x[::2],x[::2],phase_HA[::2,::2])
# fig[0].colorbar()
axs[1].axis('square')
# axs[1].set_xlabel('position (m)')
# axs[1].set_title('$\phi$ @ z={:.0f}km (rad)'.format(z/1000))
axs[1].set_xlabel('(b)')
axs[1].set_xticks([])
axs[1].set_yticks([])

divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.05)   
plt.colorbar(im, cax=cax)

im = axs[2].pcolor(x[::2],x[::2],intensity_HA[::2,::2],vmax=2,vmin=0)
# fig[1].colorbar()
# axs[2].set_xlabel('position (m)')
# axs[2].set_title('I @ z={:.0f}km (norm)'.format(z/1000))
axs[2].axis('square')
axs[2].set_xlabel('(c)')
axs[2].set_xticks([])
axs[2].set_yticks([])

axs[1].set_title('$\sigma_\chi^2$={:.2f} rad$^2$'.format(true_log_amplitude_variance))

divider = make_axes_locatable(axs[2])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

plt.tight_layout()

# plt.gcf ().set_size_inches (1.8, 5.4)

plt.savefig(name + '_combined' + '.eps',bbox_inches='tight')
plt.savefig(name + '_combined' + '.png',bbox_inches='tight')

print(log_amplitude_variance,np.var(np.log(np.sqrt(intensity_HA))))





# # atm_pad = int(np.ceil(2*wvl*z/r0/delta))
# # prop_pad = int(np.ceil(wvl*z/2/delta**2))


# # NN = N - 2*atm_pad
# # x = (np.arange(NN)- NN//2) * delta

# # IF = PhaseScreenKolmogorov(N, delta, r0, L0,random_seed=999)
# # for i in np.arange(N):
# #     IF.add_row()
# # scrn = IF.scrn
# # scrn -= scrn.mean()

# # Uout = angularSpectrum(np.pad(np.exp(1j*scrn),
# #                               ((prop_pad,prop_pad),
# #                                (prop_pad,prop_pad)),'symmetric'),
# #                         wvl, delta, delta, z)[prop_pad:-prop_pad,
# #                                               prop_pad:-prop_pad][atm_pad:-atm_pad,
# #                                                                   atm_pad:-atm_pad]

# # intensity = np.abs(Uout)**2
# # phase = np.angle(Uout)
# # init_phase = np.angle(np.exp(1j*scrn[atm_pad:-atm_pad,atm_pad:-atm_pad]))

# # plt.pcolor(x,x,init_phase)
# # plt.colorbar()
# # plt.xlabel('position (m)')
# # plt.axis('square')
# # plt.title('phase (rad)')
# # plt.savefig('sample_init_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.svg'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
# # plt.show()

# # plt.pcolor(x,x,phase)
# # plt.colorbar()
# # plt.xlabel('position (m)')
# # plt.axis('square')
# # plt.title('phase (rad)')
# # plt.savefig('sample_prop_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.svg'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
# # plt.show()

# # plt.pcolor(x,x,intensity,vmax=2,vmin=0)
# # plt.colorbar()
# # plt.axis('square')
# # plt.xlabel('position (m)')
# # plt.title('normalized intensity')
# # plt.savefig('sample_prop_intensity_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.svg'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
# # plt.show()


# # plt.clf()
# # fig,axs = plt.subplots(1,3, sharex=True, sharey=True)

# # im = axs[0].pcolor(x,x,init_phase)
# # # fig[0].colorbar()
# # axs[0].axis('square')
# # # axs[0].set_xlabel('position (m)')
# # # axs[0].set_title('$\phi$ @ z=0km (rad)')
# # axs[0].set_xlabel('(a)')
# # axs[0].set_xticks([])
# # axs[0].set_yticks([])

# # divider = make_axes_locatable(axs[0])
# # cax = divider.append_axes("right", size="5%", pad=0.05)   
# # plt.colorbar(im, cax=cax)

# # im = axs[1].pcolor(x,x,phase)
# # # fig[0].colorbar()
# # axs[1].axis('square')
# # # axs[1].set_xlabel('position (m)')
# # # axs[1].set_title('$\phi$ @ z={:.0f}km (rad)'.format(z/1000))
# # axs[1].set_xlabel('(b)')
# # axs[1].set_xticks([])
# # axs[1].set_yticks([])

# # divider = make_axes_locatable(axs[1])
# # cax = divider.append_axes("right", size="5%", pad=0.05)   
# # plt.colorbar(im, cax=cax)

# # im = axs[2].pcolor(x,x,intensity,vmax=2,vmin=0)
# # # fig[1].colorbar()
# # # axs[2].set_xlabel('position (m)')
# # # axs[2].set_title('I @ z={:.0f}km (norm)'.format(z/1000))
# # axs[2].axis('square')
# # axs[2].set_xlabel('(c)')
# # axs[2].set_xticks([])
# # axs[2].set_yticks([])

# # divider = make_axes_locatable(axs[2])
# # cax = divider.append_axes("right", size="5%", pad=0.05)
# # plt.colorbar(im, cax=cax)

# # plt.savefig('sample_prop_intensity_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.svg'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
# # plt.savefig('sample_prop_intensity_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.png'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')


# # plt.show()

# # print(np.var(np.log(np.sqrt(intensity/intensity.mean()))))


# # pad = int(np.ceil(wvl*z/delta**2))
# # IF2 = PhaseScreenKolmogorov(N+pad+pad, delta, r0, L0,random_seed=999)
# # for i in np.arange(N+pad+pad):
# #     IF2.add_row()
# # scrn = IF2.scrn
# # scrn -= scrn.mean()

# # x = (np.arange(N)- N//2) * delta

# # Uout = angularSpectrum(np.exp(1j*scrn), wvl, delta, delta, z)[pad:-pad,pad:-pad]

# # intensity = np.abs(Uout)**2
# # phase = np.angle(Uout)

# # plt.pcolor(x,x,phase)
# # plt.colorbar()
# # plt.xlabel('position (m)')
# # plt.axis('square')
# # plt.title('phase (rad)')
# # # plt.savefig('sample_prop_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
# # plt.show()

# # plt.pcolor(x,x,intensity,vmax=2,vmin=0)
# # plt.colorbar()
# # plt.axis('square')
# # plt.xlabel('position (m)')
# # plt.title('normalized intensity')
# # # plt.savefig('sample_prop_intensity_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
# # plt.show()


# #to show that the mirror pad works fine
# x = (np.arange(N)- N//2) * delta
# atm_pad = int(np.ceil(2*wvl*z/r0/delta))
# prop_pad = int(np.ceil(wvl*z/2/delta**2))
# # print(2*atm_pad,prop_pad)

# IF3 = PhaseScreenKolmogorov(N+2*prop_pad, delta, r0, L0,random_seed=999)
# for i in np.arange(N+2*prop_pad):
#     IF3.add_row()
# scrn = IF3.scrn
# scrn -= scrn.mean()

# plt.imshow(scrn[prop_pad:-prop_pad,prop_pad:-prop_pad])
# plt.show()

# ##reduce infinite phase screen size & reduce propagation size = not working
# # atm_pad_scrn = scrn[prop_pad-atm_pad:-(prop_pad-atm_pad),prop_pad-atm_pad:-(prop_pad-atm_pad)]
# # atm_pad_Uin = np.exp(1j*np.pad(atm_pad_scrn,
# #                                 ((atm_pad,atm_pad),
# #                                   (atm_pad,atm_pad)),'edge'))#'symmetric',reflect_type='even'))
# # atm_pad_Uout = angularSpectrum(atm_pad_Uin, wvl, delta, delta, z)[2*atm_pad:-2*atm_pad,
# #                                                                   2*atm_pad:-2*atm_pad]

# # intensity_atm_pad = np.abs(atm_pad_Uout)**2
# # phase_atm_pad = np.angle(atm_pad_Uout)

# # plt.pcolor(x,x,phase_atm_pad)
# # plt.colorbar()
# # plt.xlabel('position (m)')
# # plt.axis('square')
# # plt.title('phase_atm_pad (rad)')
# # # plt.savefig('sample_prop_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
# # plt.show()

# # plt.pcolor(x,x,intensity_atm_pad,vmax=2,vmin=0)
# # plt.colorbar()
# # plt.axis('square')
# # plt.xlabel('position (m)')
# # plt.title('normalized intensity_atm_pad')
# # # plt.savefig('sample_prop_intensity_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
# # plt.show()



# #reduce infinite phase screen size # pad to prop size
# atm_pad_scrn = scrn[prop_pad-atm_pad:-(prop_pad-atm_pad),prop_pad-atm_pad:-(prop_pad-atm_pad)]
# atm_pad_Uin = np.exp(1j*np.pad(atm_pad_scrn,
#                                 ((prop_pad-atm_pad,prop_pad-atm_pad),
#                                   (prop_pad-atm_pad,prop_pad-atm_pad)),'symmetric'))#'symmetric',reflect_type='even'))
# atm_pad_Uout = angularSpectrum(atm_pad_Uin, wvl, delta, delta, z)[prop_pad:-prop_pad,
#                                                                   prop_pad:-prop_pad]

# intensity_atm_pad = np.abs(atm_pad_Uout)**2
# phase_atm_pad = np.angle(atm_pad_Uout)

# plt.pcolor(x,x,phase_atm_pad)
# plt.colorbar()
# plt.xlabel('position (m)')
# plt.axis('square')
# plt.title('phase_atm_pad (rad)')
# # plt.savefig('sample_prop_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
# plt.show()

# plt.pcolor(x,x,intensity_atm_pad,vmax=2,vmin=0)
# plt.colorbar()
# plt.axis('square')
# plt.xlabel('position (m)')
# plt.title('normalized intensity_atm_pad')
# # plt.savefig('sample_prop_intensity_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
# plt.show()



# #perfect math
# prop_pad_Uin = np.exp(1j*scrn)
# prop_pad_Uout = angularSpectrum(prop_pad_Uin, wvl, delta, delta, z)[prop_pad:-prop_pad,prop_pad:-prop_pad]

# intensity_prop_pad = np.abs(prop_pad_Uout)**2
# phase_prop_pad = np.angle(prop_pad_Uout)

# plt.pcolor(x,x,phase_prop_pad)
# plt.colorbar()
# plt.xlabel('position (m)')
# plt.axis('square')
# plt.title('phase_prop_pad (rad)')
# # plt.savefig('sample_prop_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
# plt.show()

# plt.pcolor(x,x,intensity_prop_pad,vmax=2,vmin=0)
# plt.colorbar()
# plt.axis('square')
# plt.xlabel('position (m)')
# plt.title('normalized intensity_prop_pad')
# # plt.savefig('sample_prop_intensity_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
# plt.show()

# import my_phasescreen as mps
# PSDUatm = mps.get_PSD(atm_pad_Uout,delta)[N//2,N//2+1:]
# PSDUprop = mps.get_PSD(prop_pad_Uout,delta)[N//2,N//2+1:]
# f=(np.arange(N//2-1) + 1)/N/delta
# plt.plot(f,PSDUatm,label='atm pad')
# plt.plot(f,PSDUprop,label='prop pad')
# # plt.vlines([1/np.sqrt(wvl*z)],
# #            np.min([PSDUprop,PSDUatm]),
# #            np.max([PSDUprop,PSDUatm]),label='1/$\sqrt{\lambda z}$',color='k')
# plt.legend()
# plt.title('psd U')
# plt.xlabel('f (1/m)')
# plt.ylabel('PSD_U (normI m$^2$)')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

# PSDphaseatm = mps.get_PSD(phase_atm_pad,delta)[N//2,N//2+1:]
# PSDphaseprop = mps.get_PSD(phase_prop_pad,delta)[N//2,N//2+1:]
# f=(np.arange(N//2-1) + 1)/N/delta
# plt.plot(f,PSDphaseatm,label='atm pad')
# plt.plot(f,PSDphaseprop,label='prop pad')
# plt.legend()
# plt.title('psd phase')
# plt.xlabel('f (1/m)')
# plt.ylabel('PSD_$\phi$ (rad$^2$ m$^2$)')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

# PSDintensityatm = mps.get_PSD(intensity_atm_pad,delta)[N//2,N//2+1:]
# PSDintensityprop = mps.get_PSD(intensity_prop_pad,delta)[N//2,N//2+1:]
# f=(np.arange(N//2-1) + 1)/N/delta
# plt.plot(f,PSDintensityatm,label='atm pad')
# plt.plot(f,PSDintensityprop,label='prop pad')
# plt.vlines([1/np.sqrt(wvl*z)],
#            np.min([PSDintensityatm,PSDintensityprop]),
#            np.max([PSDintensityatm,PSDintensityprop]),label='1/$\sqrt{\lambda z}$',color='k')
# plt.legend()
# plt.title('psd intensity')
# plt.xlabel('f (1/m)')
# plt.ylabel('PSD_I (normI m$^2$)')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

# # SFUatm=mps.get_structure_function(atm_pad_Uout)
# # SFUprop=mps.get_structure_function(prop_pad_Uout)
# # plt.plot(x,SFUatm,label='atm')
# # plt.plot(x,SFUprop,label='prop')
# # plt.legend()
# # plt.title('structure function U')
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.xlabel('separation (m)')
# # plt.ylabel('$\sigma^2_U$ (normI)')
# # plt.show()

# # SFphaseatm=mps.get_structure_function_ft(phase_atm_pad)
# # SFphaseprop=mps.get_structure_function_ft(phase_prop_pad)

# # SFphaseatm=mps.get_structure_function(phase_atm_pad)
# # SFphaseprop=mps.get_structure_function(phase_prop_pad)

# SFphaseatm=mps.get_structure_function(phase_atm_pad)
# SFphaseprop=mps.get_structure_function(phase_prop_pad)

# for i in np.arange(N - 1):
#     SFphaseatm+=mps.get_structure_function(np.roll(phase_atm_pad,(i,i),axis=(0,1)))
#     SFphaseprop+=mps.get_structure_function(np.roll(phase_prop_pad,(i,i),axis=(0,1)))
# SFphaseatm/=N
# SFphaseprop/=N

# # plt.plot(np.arange(N)*delta,SFphaseatm,label='atm')
# # plt.plot(np.arange(N)*delta,SFphaseprop,label='prop')
# plt.plot(np.arange(N//2)*delta,SFphaseatm[:N//2],label='atm')
# plt.plot(np.arange(N//2)*delta,SFphaseprop[:N//2],label='prop')
# plt.legend()
# plt.title('structure function $\phi$')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('separation (m)')
# plt.ylabel('$\sigma^2_\phi$ (rad$^2$)')
# plt.show()

# SFintensityatm=mps.get_structure_function(intensity_atm_pad)
# SFintensityprop=mps.get_structure_function(intensity_prop_pad)

# for i in np.arange(N - 1):
#     SFintensityatm+=mps.get_structure_function(np.roll(intensity_atm_pad,(i,i),axis=(0,1)))
#     SFintensityprop+=mps.get_structure_function(np.roll(intensity_prop_pad,(i,i),axis=(0,1)))
# SFintensityatm/=N
# SFintensityprop/=N

# # plt.plot(np.arange(N)*delta,SFphaseatm,label='atm')
# # plt.plot(np.arange(N)*delta,SFphaseprop,label='prop')
# plt.plot(np.arange(N//2)*delta,SFintensityatm[:N//2],label='atm')
# plt.plot(np.arange(N//2)*delta,SFintensityprop[:N//2],label='prop')
# plt.vlines([np.sqrt(wvl*z)],
#            np.min([SFintensityatm,SFintensityprop]),
#            np.max([SFintensityatm,SFintensityprop]),label='$\sqrt{\lambda z}$',color='k')
# plt.legend()
# plt.title('structure function I')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('separation (m)')
# plt.ylabel('$\sigma^2_I$ (normI$^2$)')
# plt.show()

