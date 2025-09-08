import numpy as np


datapoint = 10

from matplotlib.lines import Line2D
import matplotlib as mpl
from tqdm import tqdm
fontsize = 10
figsize = [4,4]
dpi = 600

mpl.rcParams['axes.titlesize'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['figure.titlesize'] = fontsize
mpl.rcParams['figure.figsize'] = figsize
mpl.rcParams['figure.dpi'] = dpi
mpl.rcParams['savefig.dpi'] = dpi
mpl.rcParams['savefig.bbox'] = 'tight'
# mpl.rcParams['savefig.bbox_inches'] = 'tight'
mpl.rcParams['savefig.format'] = 'pdf'

color_list = np.array(list(mpl.colors.TABLEAU_COLORS.keys()))

r0_list = [0.02,0.05,0.1]

rms = np.zeros((datapoint,3),dtype=float)
rms_error = np.copy(rms)
for i_z in tqdm(range(datapoint)):
    z = np.logspace(2,4,datapoint)[i_z]
    for i_r0 in range(len(r0_list)):
        r0 = r0_list[i_r0]
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
        import my_phasescreen as mps
        
        """
        
        SF, SFerror = mps.get_structure_function_rolling(phase)
        PSD, PSDerror = 
        
        """
        
        D = 0.4
        delta = 0.05/16
        # r0 = 0.05
        L0 = 4
        # z = 10e3
        wvl = 500e-9
        
        # name = 'sample_prop_D{:.0f}m_delta{:.0f}mm_r0{:.0f}cm_L0{:.0f}m_z{:.0f}km_wvl{:.0f}nm'.format(
        #     D,delta*1e3,r0*1e2,L0,z*1e-3,wvl*1e9)
        
        N = int(np.ceil(D/delta))
        
        # if os.path.isfile(name + '.pkl'):
        #     print('read old file')
            
        #     pkl_file = open(name + '.pkl', 'rb')
        #     data = pickle.load(pkl_file)
            
        #     x = data['x']
        #     phase_GL = data['phase_GL']
        #     phase_HA = data['phase_HA']
        #     intensity_HA = data['intensity_HA']
        #     pkl_file.close()
        
        # else:
        #     print('make a new one')
            
        #     x = (np.arange(N)- N//2) * delta
        #     atm_pad = int(np.ceil(2*wvl*z/r0/delta))
        #     prop_pad = int(np.ceil(wvl*z/2/delta**2))
        #     # print(2*atm_pad,prop_pad)
        
        #     IF3 = PhaseScreenKolmogorov(N+2*atm_pad, delta, r0, L0,random_seed=999)
        #     for i in np.arange(N+2*atm_pad):
        #         IF3.add_row()
        #     SCRN = IF3.scrn
        #     SCRN -= SCRN.mean()
            
        #     plt.imshow(SCRN)
        #     plt.show()
            
        #     U_GL = np.exp(1j*SCRN[atm_pad:-atm_pad,atm_pad:-atm_pad])
        #     U_GL /= U_GL.mean()
        #     phase_GL = np.angle(U_GL)
            
        #     atm_pad_scrn = SCRN
        #     atm_pad_Uin = np.exp(1j*np.pad(atm_pad_scrn,
        #                                     ((prop_pad-atm_pad,prop_pad-atm_pad),
        #                                       (prop_pad-atm_pad,prop_pad-atm_pad)),'symmetric'))#'symmetric',reflect_type='even'))
        #     atm_pad_Uout = angularSpectrum(atm_pad_Uin, wvl, delta, delta, z)[prop_pad:-prop_pad,
        #                                                                       prop_pad:-prop_pad]
        #     mean = atm_pad_Uout.mean()
        #     mean /= np.abs(mean)
        #     atm_pad_Uout /= mean
        
        #     intensity_HA = np.abs(atm_pad_Uout)**2
        #     phase_HA = np.angle(atm_pad_Uout)
            
        #     data = {'x' : x,
        #             'phase_GL' : phase_GL,
        #             'phase_HA' : phase_HA,
        #             'intensity_HA' : intensity_HA}
            
        #     output = open(name + '.pkl', 'wb')
        #     pickle.dump(data,output)
        #     output.close()
            
        # plt.pcolor(x[::2],x[::2],phase_GL[::2,::2])
        # plt.axis('square')
        # plt.colorbar()
        # plt.xlabel('position (m)')
        # plt.title('phase @ z=0km (rad)')
        # plt.savefig(name + '_phase0' + '.eps',bbox_inches='tight',dpi=100)
        # plt.savefig(name + '_phase0' + '.png',bbox_inches='tight',dpi=100)
        # plt.show()
        
        # plt.pcolor(x[::2],x[::2],phase_HA[::2,::2])
        # plt.axis('square')
        # plt.colorbar()
        # plt.xlabel('position (m)')
        # plt.title('phase @ z={:.0f}km (rad)'.format(z/1e3))
        # plt.savefig(name + '_phase_z' + '.eps',bbox_inches='tight',dpi=100)
        # plt.savefig(name + '_phase_z' + '.png',bbox_inches='tight',dpi=100)
        # plt.show()
        
        # plt.pcolor(x[::2],x[::2],intensity_HA[::2,::2])
        # plt.axis('square')
        # plt.colorbar()
        # plt.xlabel('position (m)')
        # plt.title('intensity @ z={:.0f}km (rad)'.format(z/1e3))
        # plt.savefig(name + '_intensity_z' + '.eps',bbox_inches='tight',dpi=100)
        # plt.savefig(name + '_intensity_z' + '.png',bbox_inches='tight',dpi=100)
        # plt.show()
        
        
        
        
        
        # k = 2*np.pi/wvl
        # cn2dz = r0**(-5./3.)/0.423/k**2
        # log_amplitude_variance = 0.563 * k**(7./6.) * cn2dz * z**(5./6.)
        # true_log_amplitude_variance = np.var(np.log(np.sqrt(intensity_HA)))
        
        # plt.clf()
        # fig,axs = plt.subplots(1,3, sharex=True, sharey=True, squeeze=True)
        
        # im = axs[0].pcolor(x[::2],x[::2],phase_GL[::2,::2])
        # # fig[0].colorbar()
        # axs[0].axis('square')
        # # axs[0].set_xlabel('position (m)')
        # # axs[0].set_title('$\phi$ @ z=0km (rad)')
        # axs[0].set_xlabel('(a)')
        # axs[0].set_xticks([])
        # axs[0].set_yticks([])
        
        # divider = make_axes_locatable(axs[0])
        # cax = divider.append_axes("right", size="5%", pad=0.05)   
        # plt.colorbar(im, cax=cax)
        
        # im = axs[1].pcolor(x[::2],x[::2],phase_HA[::2,::2])
        # # fig[0].colorbar()
        # axs[1].axis('square')
        # # axs[1].set_xlabel('position (m)')
        # # axs[1].set_title('$\phi$ @ z={:.0f}km (rad)'.format(z/1000))
        # axs[1].set_xlabel('(b)')
        # axs[1].set_xticks([])
        # axs[1].set_yticks([])
        
        # divider = make_axes_locatable(axs[1])
        # cax = divider.append_axes("right", size="5%", pad=0.05)   
        # plt.colorbar(im, cax=cax)
        
        # im = axs[2].pcolor(x[::2],x[::2],intensity_HA[::2,::2],vmax=2,vmin=0)
        # # fig[1].colorbar()
        # # axs[2].set_xlabel('position (m)')
        # # axs[2].set_title('I @ z={:.0f}km (norm)'.format(z/1000))
        # axs[2].axis('square')
        # axs[2].set_xlabel('(c)')
        # axs[2].set_xticks([])
        # axs[2].set_yticks([])
        
        # axs[1].set_title('$\sigma_\chi^2$={:.2f} rad$^2$'.format(true_log_amplitude_variance))
        
        # divider = make_axes_locatable(axs[2])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, cax=cax)
        
        # plt.tight_layout()
        
        # # plt.gcf ().set_size_inches (1.8, 5.4)
        
        # plt.savefig(name + '_combined' + '.eps',bbox_inches='tight')
        # plt.savefig(name + '_combined' + '.png',bbox_inches='tight')
        
        # print(log_amplitude_variance,np.var(np.log(np.sqrt(intensity_HA))))
        
        
        
        
        
        
        
        
        # atm_pad = int(np.ceil(2*wvl*z/r0/delta))
        # prop_pad = int(np.ceil(wvl*z/2/delta**2))
        
        
        # NN = N - 2*atm_pad
        # x = (np.arange(NN)- NN//2) * delta
        
        # IF = PhaseScreenKolmogorov(N, delta, r0, L0,random_seed=999)
        # for i in np.arange(N):
        #     IF.add_row()
        # scrn = IF.scrn
        # scrn -= scrn.mean()
        
        # Uout = angularSpectrum(np.pad(np.exp(1j*scrn),
        #                               ((prop_pad,prop_pad),
        #                                (prop_pad,prop_pad)),'symmetric'),
        #                         wvl, delta, delta, z)[prop_pad:-prop_pad,
        #                                               prop_pad:-prop_pad][atm_pad:-atm_pad,
        #                                                                   atm_pad:-atm_pad]
        
        # intensity = np.abs(Uout)**2
        # phase = np.angle(Uout)
        # init_phase = np.angle(np.exp(1j*scrn[atm_pad:-atm_pad,atm_pad:-atm_pad]))
        
        # plt.pcolor(x,x,init_phase)
        # plt.colorbar()
        # plt.xlabel('position (m)')
        # plt.axis('square')
        # plt.title('phase (rad)')
        # plt.savefig('sample_init_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.svg'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
        # plt.show()
        
        # plt.pcolor(x,x,phase)
        # plt.colorbar()
        # plt.xlabel('position (m)')
        # plt.axis('square')
        # plt.title('phase (rad)')
        # plt.savefig('sample_prop_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.svg'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
        # plt.show()
        
        # plt.pcolor(x,x,intensity,vmax=2,vmin=0)
        # plt.colorbar()
        # plt.axis('square')
        # plt.xlabel('position (m)')
        # plt.title('normalized intensity')
        # plt.savefig('sample_prop_intensity_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.svg'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
        # plt.show()
        
        
        # plt.clf()
        # fig,axs = plt.subplots(1,3, sharex=True, sharey=True)
        
        # im = axs[0].pcolor(x,x,init_phase)
        # # fig[0].colorbar()
        # axs[0].axis('square')
        # # axs[0].set_xlabel('position (m)')
        # # axs[0].set_title('$\phi$ @ z=0km (rad)')
        # axs[0].set_xlabel('(a)')
        # axs[0].set_xticks([])
        # axs[0].set_yticks([])
        
        # divider = make_axes_locatable(axs[0])
        # cax = divider.append_axes("right", size="5%", pad=0.05)   
        # plt.colorbar(im, cax=cax)
        
        # im = axs[1].pcolor(x,x,phase)
        # # fig[0].colorbar()
        # axs[1].axis('square')
        # # axs[1].set_xlabel('position (m)')
        # # axs[1].set_title('$\phi$ @ z={:.0f}km (rad)'.format(z/1000))
        # axs[1].set_xlabel('(b)')
        # axs[1].set_xticks([])
        # axs[1].set_yticks([])
        
        # divider = make_axes_locatable(axs[1])
        # cax = divider.append_axes("right", size="5%", pad=0.05)   
        # plt.colorbar(im, cax=cax)
        
        # im = axs[2].pcolor(x,x,intensity,vmax=2,vmin=0)
        # # fig[1].colorbar()
        # # axs[2].set_xlabel('position (m)')
        # # axs[2].set_title('I @ z={:.0f}km (norm)'.format(z/1000))
        # axs[2].axis('square')
        # axs[2].set_xlabel('(c)')
        # axs[2].set_xticks([])
        # axs[2].set_yticks([])
        
        # divider = make_axes_locatable(axs[2])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, cax=cax)
        
        # plt.savefig('sample_prop_intensity_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.svg'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
        # plt.savefig('sample_prop_intensity_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.png'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
        
        
        # plt.show()
        
        # print(np.var(np.log(np.sqrt(intensity/intensity.mean()))))
        
        
        # pad = int(np.ceil(wvl*z/delta**2))
        # IF2 = PhaseScreenKolmogorov(N+pad+pad, delta, r0, L0,random_seed=999)
        # for i in np.arange(N+pad+pad):
        #     IF2.add_row()
        # scrn = IF2.scrn
        # scrn -= scrn.mean()
        
        # x = (np.arange(N)- N//2) * delta
        
        # Uout = angularSpectrum(np.exp(1j*scrn), wvl, delta, delta, z)[pad:-pad,pad:-pad]
        
        # intensity = np.abs(Uout)**2
        # phase = np.angle(Uout)
        
        # plt.pcolor(x,x,phase)
        # plt.colorbar()
        # plt.xlabel('position (m)')
        # plt.axis('square')
        # plt.title('phase (rad)')
        # # plt.savefig('sample_prop_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
        # plt.show()
        
        # plt.pcolor(x,x,intensity,vmax=2,vmin=0)
        # plt.colorbar()
        # plt.axis('square')
        # plt.xlabel('position (m)')
        # plt.title('normalized intensity')
        # # plt.savefig('sample_prop_intensity_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
        # plt.show()
        
        
        #to show that the mirror pad works fine
        x = (np.arange(N)- N//2) * delta
        atm_pad = int(np.ceil(2*wvl*z/r0/delta))
        # atm pad is good with wvl*z/r0/delta 2*wvl*z/r0/delta is already overkill
        prop_pad = int(np.ceil(wvl*z/2/delta**2))
        # print(2*atm_pad,prop_pad)
        
        IF3 = PhaseScreenKolmogorov(N+2*prop_pad, delta, r0, L0)
        for i in np.arange(N+2*prop_pad):
            IF3.add_row()
        scrn = IF3.scrn
        scrn -= scrn.mean()
        
        if z == 10000:
            plt.figure(figsize=(5,4))
            plt.pcolor(x,x,np.angle(np.exp(1j*scrn[prop_pad:-prop_pad,prop_pad:-prop_pad])))
            plt.colorbar()
            plt.axis('square')
            plt.xlabel('Position (m)')
            # plt.title('phase before propagation')
            plt.show()
        
        
        ###############################################################################
        ###############################################################################
        
        #perfect math
        prop_pad_Uin = np.exp(1j*scrn)
        prop_pad_Uout = angularSpectrum(prop_pad_Uin, wvl, delta, delta, z)[prop_pad:-prop_pad,prop_pad:-prop_pad]
        
        intensity_prop_pad = np.abs(prop_pad_Uout)**2
        phase_prop_pad = np.angle(prop_pad_Uout)
        
        x = (np.arange(N)- N//2) * delta
        if z == 10000:
            plt.figure(figsize=(5,4))
            plt.pcolor(x,x,phase_prop_pad)
            plt.colorbar()
            plt.xlabel('Position (m)')
            plt.axis('square')
            # plt.title('phase_prop_pad (rad)')
            # plt.savefig('sample_prop_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
            plt.show()
        
            plt.figure(figsize=(5,4))
            plt.pcolor(x,x,intensity_prop_pad,vmax=2,vmin=0)
            plt.colorbar()
            plt.axis('square')
            plt.xlabel('Position (m)')
            # plt.title('normalized intensity_prop_pad')
            # plt.savefig('sample_prop_intensity_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
            plt.show()
        
        ###############################################################################
        
        if z == 10000:
        
            PSDUprop = np.zeros((N,N),dtype=float)
            for i in np.arange(len(prop_pad_Uout)):
                PSDUprop[i] = mps.get_PSD(prop_pad_Uout[i],delta)
            PSDUprop[:] += PSDUprop[:,::-1]
            PSDUprop /= 2.
            
            PSDphaseprop = np.zeros((N,N),dtype=float)
            for i in np.arange(len(phase_prop_pad)):
                PSDphaseprop[i] = mps.get_PSD(phase_prop_pad[i],delta)
            PSDphaseprop[:] += PSDphaseprop[:,::-1]
            PSDphaseprop /= 2.
            
            PSDintensityprop = np.zeros((N,N),dtype=float)
            for i in np.arange(len(intensity_prop_pad)):
                PSDintensityprop[i] = mps.get_PSD(intensity_prop_pad[i],delta)
            PSDintensityprop[:] += PSDintensityprop[:,::-1]
            PSDintensityprop /= 2.
            
            SFphaseprop = np.zeros((N,N), dtype=float)
            # for i in np.arange(N):
            #     SFphaseprop[i] = mps.get_structure_function(np.roll(phase_prop_pad,(i),axis=(0)))
            # SFphaseprop = (phase_prop_pad[:,0] - phase_prop_pad[:,:])**2
            for i in np.arange(N):
                SFphaseprop[i] = np.angle((prop_pad_Uout[i,i] / np.roll(prop_pad_Uout[i],(-i),axis=(0))))**2
                
            
            
            
            SFintensityprop = np.zeros((N,N), dtype=float)
            # for i in np.arange(N):
            #     SFintensityprop[i] = mps.get_structure_function(np.roll(intensity_prop_pad,(i),axis=(0)))
            # SFintensityprop = (intensity_prop_pad[:,0] - intensity_prop_pad[:,:])**2
            for i in np.arange(N):
                SFintensityprop[i] = (intensity_prop_pad[i,i] - np.roll(intensity_prop_pad[i],(-i),axis=(0)))**2
                
        
        
        ###############################################################################
        ###############################################################################
        
        # for atm_dif_ratio in np.arange(5):
        # for extra_atm_pad in np.arange(4):
        
        # atm_pad = int(np.ceil(atm_dif_ratio*wvl*z/r0/delta))
        
        # ###############################################################################
        
        # ##reduce infinite phase screen size & reduce propagation size = not working
        # # PSD intensity not match at high frequency
        # TAG = 'prop screen size = ATM'
        # atm_pad_scrn = scrn[prop_pad-atm_pad:-(prop_pad-atm_pad),prop_pad-atm_pad:-(prop_pad-atm_pad)]
        # atm_pad_Uin = np.exp(1j*atm_pad_scrn)#'symmetric',reflect_type='even'))
        # atm_pad_Uout = angularSpectrum(atm_pad_Uin, wvl, delta, delta, z)[atm_pad:-atm_pad,
        #                                                                   atm_pad:-atm_pad]
        
        # intensity_atm_pad = np.abs(atm_pad_Uout)**2
        # phase_atm_pad = np.angle(atm_pad_Uout)
        # x = (np.arange(N)- N//2) * delta
        # plt.pcolor(x,x,phase_atm_pad)
        # plt.colorbar()
        # plt.xlabel('position (m)')
        # plt.axis('square')
        # plt.title('phase_atm_pad (rad)' + '\n' + TAG)
        # # plt.savefig('sample_prop_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
        # plt.show()
        
        # plt.pcolor(x,x,intensity_atm_pad,vmax=2,vmin=0)
        # plt.colorbar()
        # plt.axis('square')
        # plt.xlabel('position (m)')
        # plt.title('normalized intensity_atm_pad'  + '\n' + TAG)
        # # plt.savefig('sample_prop_intensity_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
        # plt.show()
        
        # PSDUatm = np.zeros((N,N),dtype=float)
        # for i in np.arange(len(atm_pad_Uout)):
        #     PSDUatm[i] = mps.get_PSD(atm_pad_Uout[i],delta)
        # PSDUatm[:] += PSDUatm[:,::-1]
        # PSDUatm /= 2.
        
        # f=(np.arange(N//2-1) + 1)/N/delta
        # A = np.mean(PSDUatm[:,N//2+1:],axis=0)
        # Ae = np.std(PSDUatm[:,N//2+1:],axis=0)/np.sqrt(N)
        # B = np.mean(PSDUprop[:,N//2+1:],axis=0)
        # Be = np.std(PSDUprop[:,N//2+1:],axis=0)/np.sqrt(N)
        
        # plt.plot(f,A,color='r',label='atm pad')
        # plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(f,B,color='b',label='prop pad')
        # plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
        # plt.legend()
        # plt.ylim(B.min()*0.9,B.max()*1.1)
        # plt.title('psd U' + '\n' + TAG)
        # plt.xlabel('f (1/m)')
        # plt.ylabel('PSD_U (I$_{norm}$ m$^2$)')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        
        
        
        # PSDphaseatm = np.zeros((N,N),dtype=float)
        # for i in np.arange(len(phase_atm_pad)):
        #     PSDphaseatm[i] = mps.get_PSD(phase_atm_pad[i],delta)
        # PSDphaseatm[:] += PSDphaseatm[:,::-1]
        # PSDphaseatm /= 2.
        
        # f=(np.arange(N//2-1) + 1)/N/delta
        # A = np.mean(PSDphaseatm[:,N//2+1:],axis=0)
        # Ae = np.std(PSDphaseatm[:,N//2+1:],axis=0)/np.sqrt(N)
        # B = np.mean(PSDphaseprop[:,N//2+1:],axis=0)
        # Be = np.std(PSDphaseprop[:,N//2+1:],axis=0)/np.sqrt(N)
        
        # plt.plot(f,A,color='r',label='atm pad')
        # plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(f,B,color='b',label='prop pad')
        # plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
        # plt.legend()
        # plt.ylim(B.min()*0.9,B.max()*1.1)
        # plt.title('psd phase' + '\n' + TAG)
        # plt.xlabel('f (1/m)')
        # plt.ylabel('PSD_$\phi$ (rad$^2$ m$^2$)')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        
        
        
        # PSDintensityatm = np.zeros((N,N),dtype=float)
        # for i in np.arange(len(intensity_atm_pad)):
        #     PSDintensityatm[i] = mps.get_PSD(intensity_atm_pad[i],delta)
        # PSDintensityatm[:] += PSDintensityatm[:,::-1]
        # PSDintensityatm /= 2.
        
        # f = (np.arange(N//2-1) + 1)/N/delta
        # A = np.mean(PSDintensityatm[:,N//2+1:],axis=0)
        # Ae = np.std(PSDintensityatm[:,N//2+1:],axis=0)/np.sqrt(N)
        # B = np.mean(PSDintensityprop[:,N//2+1:],axis=0)
        # Be = np.std(PSDintensityprop[:,N//2+1:],axis=0)/np.sqrt(N)
        
        # plt.plot(f,A,color='r',label='atm pad')
        # plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(f,B,color='b',label='prop pad')
        # plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
        # plt.vlines([1/np.sqrt(wvl*z)],B.min(),B.max(),label='1/$\sqrt{\lambda z}$',color='k')
        # plt.legend()
        # plt.ylim(B.min()*0.9,B.max()*1.1)
        # plt.title('psd intensity' + '\n' + TAG)
        # plt.xlabel('f (1/m)')
        # plt.ylabel('PSD_I (I$_{norm}$ m$^2$)')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        
        
        
        # SFphaseatm = np.zeros((N,N), dtype=float)
        # # for i in np.arange(N):
        # #     SFphaseatm[i] = mps.get_structure_function(np.roll(phase_atm_pad,(i),axis=(0)))
        # # SFphaseatm = (phase_atm_pad[:,0] - phase_atm_pad[:,:])**2
        # for i in np.arange(N):
        #     SFphaseatm[i] = np.angle((atm_pad_Uout[i,i] / np.roll(atm_pad_Uout[i],(-i),axis=(0))))**2
        
        
        # x = np.arange(N//2)*delta
        # A = np.mean(SFphaseatm[:,:N//2],axis=0)
        # Ae = np.std(SFphaseatm[:,:N//2],axis=0)/np.sqrt(N)
        # B = np.mean(SFphaseprop[:,:N//2],axis=0)
        # Be = np.std(SFphaseprop[:,:N//2],axis=0)/np.sqrt(N)
        
        # plt.plot(x,A,color='r',label='atm pad')
        # plt.fill_between(x,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(x,B,color='b',label='prop pad')
        # plt.fill_between(x,B-Be,B+Be,alpha=0.2,color='b')
        # plt.legend()
        # plt.ylim(B[1:].min()*0.9,B.max()*1.1)
        # plt.title('structure function $\phi$' + '\n' + TAG)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel('separation (m)')
        # plt.ylabel('$\sigma^2_\phi$ (rad$^2$)')
        # plt.show()
        
        
        
        # SFintensityatm = np.zeros((N,N), dtype=float)
        # # for i in np.arange(N):
        # #     SFintensityatm[i] = mps.get_structure_function(np.roll(intensity_atm_pad,(i),axis=(0)))
        # # SFintensityatm = (intensity_atm_pad[:,0] - intensity_atm_pad[:,:])**2
        # for i in np.arange(N):
        #     SFintensityatm[i] = (intensity_atm_pad[i,i] - np.roll(intensity_atm_pad[i],(-i),axis=(0)))**2
        
        # x = np.arange(N//2)*delta
        # A = np.mean(SFintensityatm[:,:N//2],axis=0)
        # Ae = np.std(SFintensityatm[:,:N//2],axis=0)/np.sqrt(N)
        # B = np.mean(SFintensityprop[:,:N//2],axis=0)
        # Be = np.std(SFintensityprop[:,:N//2],axis=0)/np.sqrt(N)
        
        # plt.plot(x,A,color='r',label='atm pad')
        # plt.fill_between(x,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(x,B,color='b',label='prop pad')
        # plt.fill_between(x,B-Be,B+Be,alpha=0.2,color='b')
        # plt.vlines([np.sqrt(wvl*z)],
        #             B.min(),B.max(),label='$\sqrt{\lambda z}$',color='k')
        # plt.legend()
        # plt.title('structure function I' + '\n' + TAG)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel('separation (m)')
        # plt.ylabel('$\sigma^2_I$ (I$_{norm}$$^2$)')
        # plt.show()
        
        # ###############################################################################
        
        
        # ##reduce infinite phase screen size & reduce propagation size + 1 atm edge pad = not working
        # # psd U not matching at high frequency
        # TAG = 'prop screen size = ATM + 2 edge pad'
        # atm_pad_scrn = scrn[prop_pad-atm_pad:-(prop_pad-atm_pad),prop_pad-atm_pad:-(prop_pad-atm_pad)]
        # atm_pad_Uin = np.exp(1j*np.pad(atm_pad_scrn,
        #                                 ((atm_pad,atm_pad),
        #                                   (atm_pad,atm_pad)),'edge'))#'symmetric',reflect_type='even'))
        # atm_pad_Uout = angularSpectrum(atm_pad_Uin, wvl, delta, delta, z)[2*atm_pad:-2*atm_pad,
        #                                                                   2*atm_pad:-2*atm_pad]
        
        # intensity_atm_pad = np.abs(atm_pad_Uout)**2
        # phase_atm_pad = np.angle(atm_pad_Uout)
        # x = (np.arange(N)- N//2) * delta
        # plt.pcolor(x,x,phase_atm_pad)
        # plt.colorbar()
        # plt.xlabel('position (m)')
        # plt.axis('square')
        # plt.title('phase_atm_pad (rad)' + '\n' + TAG)
        # # plt.savefig('sample_prop_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
        # plt.show()
        
        # plt.pcolor(x,x,intensity_atm_pad,vmax=2,vmin=0)
        # plt.colorbar()
        # plt.axis('square')
        # plt.xlabel('position (m)')
        # plt.title('normalized intensity_atm_pad' + '\n' + TAG)
        # # plt.savefig('sample_prop_intensity_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
        # plt.show()
        
        # PSDUatm = np.zeros((N,N),dtype=float)
        # for i in np.arange(len(atm_pad_Uout)):
        #     PSDUatm[i] = mps.get_PSD(atm_pad_Uout[i],delta)
        # PSDUatm[:] += PSDUatm[:,::-1]
        # PSDUatm /= 2.
        
        # f=(np.arange(N//2-1) + 1)/N/delta
        # A = np.mean(PSDUatm[:,N//2+1:],axis=0)
        # Ae = np.std(PSDUatm[:,N//2+1:],axis=0)/np.sqrt(N)
        # B = np.mean(PSDUprop[:,N//2+1:],axis=0)
        # Be = np.std(PSDUprop[:,N//2+1:],axis=0)/np.sqrt(N)
        
        # plt.plot(f,A,color='r',label='atm pad')
        # plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(f,B,color='b',label='prop pad')
        # plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
        # plt.legend()
        # plt.ylim(B.min()*0.9,B.max()*1.1)
        # plt.title('psd U' + '\n' + TAG)
        # plt.xlabel('f (1/m)')
        # plt.ylabel('PSD_U (I$_{norm}$ m$^2$)')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        
        
        
        # PSDphaseatm = np.zeros((N,N),dtype=float)
        # for i in np.arange(len(phase_atm_pad)):
        #     PSDphaseatm[i] = mps.get_PSD(phase_atm_pad[i],delta)
        # PSDphaseatm[:] += PSDphaseatm[:,::-1]
        # PSDphaseatm /= 2.
        
        # f=(np.arange(N//2-1) + 1)/N/delta
        # A = np.mean(PSDphaseatm[:,N//2+1:],axis=0)
        # Ae = np.std(PSDphaseatm[:,N//2+1:],axis=0)/np.sqrt(N)
        # B = np.mean(PSDphaseprop[:,N//2+1:],axis=0)
        # Be = np.std(PSDphaseprop[:,N//2+1:],axis=0)/np.sqrt(N)
        
        # plt.plot(f,A,color='r',label='atm pad')
        # plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(f,B,color='b',label='prop pad')
        # plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
        # plt.legend()
        # plt.ylim(B.min()*0.9,B.max()*1.1)
        # plt.title('psd phase' + '\n' + TAG)
        # plt.xlabel('f (1/m)')
        # plt.ylabel('PSD_$\phi$ (rad$^2$ m$^2$)')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        
        
        
        # PSDintensityatm = np.zeros((N,N),dtype=float)
        # for i in np.arange(len(intensity_atm_pad)):
        #     PSDintensityatm[i] = mps.get_PSD(intensity_atm_pad[i],delta)
        # PSDintensityatm[:] += PSDintensityatm[:,::-1]
        # PSDintensityatm /= 2.
        
        # f = (np.arange(N//2-1) + 1)/N/delta
        # A = np.mean(PSDintensityatm[:,N//2+1:],axis=0)
        # Ae = np.std(PSDintensityatm[:,N//2+1:],axis=0)/np.sqrt(N)
        # B = np.mean(PSDintensityprop[:,N//2+1:],axis=0)
        # Be = np.std(PSDintensityprop[:,N//2+1:],axis=0)/np.sqrt(N)
        
        # plt.plot(f,A,color='r',label='atm pad')
        # plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(f,B,color='b',label='prop pad')
        # plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
        # plt.vlines([1/np.sqrt(wvl*z)],B.min(),B.max(),label='1/$\sqrt{\lambda z}$',color='k')
        # plt.legend()
        # plt.ylim(B.min()*0.9,B.max()*1.1)
        # plt.title('psd intensity' + '\n' + TAG)
        # plt.xlabel('f (1/m)')
        # plt.ylabel('PSD_I (I$_{norm}$ m$^2$)')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        
        
        
        # SFphaseatm = np.zeros((N,N), dtype=float)
        # # for i in np.arange(N):
        # #     SFphaseatm[i] = mps.get_structure_function(np.roll(phase_atm_pad,(i),axis=(0)))
        # # SFphaseatm = (phase_atm_pad[:,0] - phase_atm_pad[:,:])**2
        # for i in np.arange(N):
        #     SFphaseatm[i] = np.angle((atm_pad_Uout[i,i] / np.roll(atm_pad_Uout[i],(-i),axis=(0))))**2
        
        
        # x = np.arange(N//2)*delta
        # A = np.mean(SFphaseatm[:,:N//2],axis=0)
        # Ae = np.std(SFphaseatm[:,:N//2],axis=0)/np.sqrt(N)
        # B = np.mean(SFphaseprop[:,:N//2],axis=0)
        # Be = np.std(SFphaseprop[:,:N//2],axis=0)/np.sqrt(N)
        
        # plt.plot(x,A,color='r',label='atm pad')
        # plt.fill_between(x,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(x,B,color='b',label='prop pad')
        # plt.fill_between(x,B-Be,B+Be,alpha=0.2,color='b')
        # plt.legend()
        # plt.ylim(B[1:].min()*0.9,B.max()*1.1)
        # plt.title('structure function $\phi$' + '\n' + TAG)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel('separation (m)')
        # plt.ylabel('$\sigma^2_\phi$ (rad$^2$)')
        # plt.show()
        
        
        
        # SFintensityatm = np.zeros((N,N), dtype=float)
        # # for i in np.arange(N):
        # #     SFintensityatm[i] = mps.get_structure_function(np.roll(intensity_atm_pad,(i),axis=(0)))
        # # SFintensityatm = (intensity_atm_pad[:,0] - intensity_atm_pad[:,:])**2
        # for i in np.arange(N):
        #     SFintensityatm[i] = (intensity_atm_pad[i,i] - np.roll(intensity_atm_pad[i],(-i),axis=(0)))**2
        
        # x = np.arange(N//2)*delta
        # A = np.mean(SFintensityatm[:,:N//2],axis=0)
        # Ae = np.std(SFintensityatm[:,:N//2],axis=0)/np.sqrt(N)
        # B = np.mean(SFintensityprop[:,:N//2],axis=0)
        # Be = np.std(SFintensityprop[:,:N//2],axis=0)/np.sqrt(N)
        
        # plt.plot(x,A,color='r',label='atm pad')
        # plt.fill_between(x,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(x,B,color='b',label='prop pad')
        # plt.fill_between(x,B-Be,B+Be,alpha=0.2,color='b')
        # plt.vlines([np.sqrt(wvl*z)],
        #             B.min(),B.max(),label='$\sqrt{\lambda z}$',color='k')
        # plt.legend()
        # plt.title('structure function I' + '\n' + TAG)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel('separation (m)')
        # plt.ylabel('$\sigma^2_I$ (I$_{norm}$$^2$)')
        # plt.show()
        
        
        # ###############################################################################
        
        # ##reduce infinite phase screen size & reduce propagation size + 1 atm symmetric pad = not working
        # # psd intensity not matching at high frequency
        # TAG = 'prop screen size = ATM + 2 symmetric pad'
        # atm_pad_scrn = scrn[prop_pad-atm_pad:-(prop_pad-atm_pad),prop_pad-atm_pad:-(prop_pad-atm_pad)]
        # atm_pad_Uin = np.exp(1j*np.pad(atm_pad_scrn,
        #                                 ((atm_pad,atm_pad),
        #                                   (atm_pad,atm_pad)),'symmetric'))#'symmetric',reflect_type='even'))
        # atm_pad_Uout = angularSpectrum(atm_pad_Uin, wvl, delta, delta, z)[2*atm_pad:-2*atm_pad,
        #                                                                   2*atm_pad:-2*atm_pad]
        # # atm_pad_Uin = np.exp(1j*np.pad(atm_pad_scrn,
        # #                                 ((extra_atm_pad*atm_pad,extra_atm_pad*atm_pad),
        # #                                   (extra_atm_pad*atm_pad,extra_atm_pad*atm_pad)),'symmetric'))#'symmetric',reflect_type='even'))
        # # atm_pad_Uout = angularSpectrum(atm_pad_Uin, wvl, delta, delta, z)[(extra_atm_pad+1)*atm_pad:-(extra_atm_pad+1)*atm_pad,
        # #                                                                   (extra_atm_pad+1)*atm_pad:-(extra_atm_pad+1)*atm_pad]
        # # not working
        
        # intensity_atm_pad = np.abs(atm_pad_Uout)**2
        # phase_atm_pad = np.angle(atm_pad_Uout)
        # x = (np.arange(N)- N//2) * delta
        # plt.pcolor(x,x,phase_atm_pad)
        # plt.colorbar()
        # plt.xlabel('position (m)')
        # plt.axis('square')
        # plt.title('phase_atm_pad (rad)' + '\n' + TAG)
        # # plt.savefig('sample_prop_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
        # plt.show()
        # x = (np.arange(N)- N//2) * delta
        # plt.pcolor(x,x,intensity_atm_pad,vmax=2,vmin=0)
        # plt.colorbar()
        # plt.axis('square')
        # plt.xlabel('position (m)')
        # plt.title('normalized intensity_atm_pad' + '\n' + TAG)
        # # plt.savefig('sample_prop_intensity_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
        # plt.show()
        
        # PSDUatm = np.zeros((N,N),dtype=float)
        # for i in np.arange(len(atm_pad_Uout)):
        #     PSDUatm[i] = mps.get_PSD(atm_pad_Uout[i],delta)
        # PSDUatm[:] += PSDUatm[:,::-1]
        # PSDUatm /= 2.
        
        # f=(np.arange(N//2-1) + 1)/N/delta
        # A = np.mean(PSDUatm[:,N//2+1:],axis=0)
        # Ae = np.std(PSDUatm[:,N//2+1:],axis=0)/np.sqrt(N)
        # B = np.mean(PSDUprop[:,N//2+1:],axis=0)
        # Be = np.std(PSDUprop[:,N//2+1:],axis=0)/np.sqrt(N)
        
        # plt.plot(f,A,color='r',label='atm pad')
        # plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(f,B,color='b',label='prop pad')
        # plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
        # plt.legend()
        # plt.ylim(B.min()*0.9,B.max()*1.1)
        # plt.title('psd U' + '\n' + TAG)
        # plt.xlabel('f (1/m)')
        # plt.ylabel('PSD_U (I$_{norm}$ m$^2$)')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        
        
        
        # PSDphaseatm = np.zeros((N,N),dtype=float)
        # for i in np.arange(len(phase_atm_pad)):
        #     PSDphaseatm[i] = mps.get_PSD(phase_atm_pad[i],delta)
        # PSDphaseatm[:] += PSDphaseatm[:,::-1]
        # PSDphaseatm /= 2.
        
        # f=(np.arange(N//2-1) + 1)/N/delta
        # A = np.mean(PSDphaseatm[:,N//2+1:],axis=0)
        # Ae = np.std(PSDphaseatm[:,N//2+1:],axis=0)/np.sqrt(N)
        # B = np.mean(PSDphaseprop[:,N//2+1:],axis=0)
        # Be = np.std(PSDphaseprop[:,N//2+1:],axis=0)/np.sqrt(N)
        
        # plt.plot(f,A,color='r',label='atm pad')
        # plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(f,B,color='b',label='prop pad')
        # plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
        # plt.legend()
        # plt.ylim(B.min()*0.9,B.max()*1.1)
        # plt.title('psd phase' + '\n' + TAG)
        # plt.xlabel('f (1/m)')
        # plt.ylabel('PSD_$\phi$ (rad$^2$ m$^2$)')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        
        
        
        # PSDintensityatm = np.zeros((N,N),dtype=float)
        # for i in np.arange(len(intensity_atm_pad)):
        #     PSDintensityatm[i] = mps.get_PSD(intensity_atm_pad[i],delta)
        # PSDintensityatm[:] += PSDintensityatm[:,::-1]
        # PSDintensityatm /= 2.
        
        # f = (np.arange(N//2-1) + 1)/N/delta
        # A = np.mean(PSDintensityatm[:,N//2+1:],axis=0)
        # Ae = np.std(PSDintensityatm[:,N//2+1:],axis=0)/np.sqrt(N)
        # B = np.mean(PSDintensityprop[:,N//2+1:],axis=0)
        # Be = np.std(PSDintensityprop[:,N//2+1:],axis=0)/np.sqrt(N)
        
        # plt.plot(f,A,color='r',label='atm pad')
        # plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(f,B,color='b',label='prop pad')
        # plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
        # plt.vlines([1/np.sqrt(wvl*z)],B.min(),B.max(),label='1/$\sqrt{\lambda z}$',color='k')
        # plt.legend()
        # plt.ylim(B.min()*0.9,B.max()*1.1)
        # plt.title('psd intensity' + '\n' + TAG)
        # plt.xlabel('f (1/m)')
        # plt.ylabel('PSD_I (I$_{norm}$ m$^2$)')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        
        
        
        # SFphaseatm = np.zeros((N,N), dtype=float)
        # # for i in np.arange(N):
        # #     SFphaseatm[i] = mps.get_structure_function(np.roll(phase_atm_pad,(i),axis=(0)))
        # # SFphaseatm = (phase_atm_pad[:,0] - phase_atm_pad[:,:])**2
        # for i in np.arange(N):
        #     SFphaseatm[i] = np.angle((atm_pad_Uout[i,i] / np.roll(atm_pad_Uout[i],(-i),axis=(0))))**2
        
        
        # x = np.arange(N//2)*delta
        # A = np.mean(SFphaseatm[:,:N//2],axis=0)
        # Ae = np.std(SFphaseatm[:,:N//2],axis=0)/np.sqrt(N)
        # B = np.mean(SFphaseprop[:,:N//2],axis=0)
        # Be = np.std(SFphaseprop[:,:N//2],axis=0)/np.sqrt(N)
        
        # plt.plot(x,A,color='r',label='atm pad')
        # plt.fill_between(x,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(x,B,color='b',label='prop pad')
        # plt.fill_between(x,B-Be,B+Be,alpha=0.2,color='b')
        # plt.legend()
        # plt.ylim(B[1:].min()*0.9,B.max()*1.1)
        # plt.title('structure function $\phi$' + '\n' + TAG)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel('separation (m)')
        # plt.ylabel('$\sigma^2_\phi$ (rad$^2$)')
        # plt.show()
        
        
        
        # SFintensityatm = np.zeros((N,N), dtype=float)
        # # for i in np.arange(N):
        # #     SFintensityatm[i] = mps.get_structure_function(np.roll(intensity_atm_pad,(i),axis=(0)))
        # # SFintensityatm = (intensity_atm_pad[:,0] - intensity_atm_pad[:,:])**2
        # for i in np.arange(N):
        #     SFintensityatm[i] = (intensity_atm_pad[i,i] - np.roll(intensity_atm_pad[i],(-i),axis=(0)))**2
        
        # x = np.arange(N//2)*delta
        # A = np.mean(SFintensityatm[:,:N//2],axis=0)
        # Ae = np.std(SFintensityatm[:,:N//2],axis=0)/np.sqrt(N)
        # B = np.mean(SFintensityprop[:,:N//2],axis=0)
        # Be = np.std(SFintensityprop[:,:N//2],axis=0)/np.sqrt(N)
        
        # plt.plot(x,A,color='r',label='atm pad')
        # plt.fill_between(x,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(x,B,color='b',label='prop pad')
        # plt.fill_between(x,B-Be,B+Be,alpha=0.2,color='b')
        # plt.vlines([np.sqrt(wvl*z)],
        #             B.min(),B.max(),label='$\sqrt{\lambda z}$',color='k')
        # plt.legend()
        # plt.title('structure function I' + '\n' + TAG)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel('separation (m)')
        # plt.ylabel('$\sigma^2_I$ (I$_{norm}$$^2$)')
        # plt.show()
        
        # ###############################################################################
        
        
        # # reduce infinite phase screen size # pad to prop size # edge pad
        # # intensity and U psd bad
        # TAG = 'prop screen size = edge_pad --> prop'
        # atm_pad_scrn = scrn[prop_pad-atm_pad:-(prop_pad-atm_pad),prop_pad-atm_pad:-(prop_pad-atm_pad)]
        # atm_pad_Uin = np.exp(1j*np.pad(atm_pad_scrn,
        #                                 ((prop_pad-atm_pad,prop_pad-atm_pad),
        #                                   (prop_pad-atm_pad,prop_pad-atm_pad)),'edge'))#'symmetric',reflect_type='even'))
        # atm_pad_Uout = angularSpectrum(atm_pad_Uin, wvl, delta, delta, z)[prop_pad:-prop_pad,
        #                                                                   prop_pad:-prop_pad]
        
        # intensity_atm_pad = np.abs(atm_pad_Uout)**2
        # phase_atm_pad = np.angle(atm_pad_Uout)
        # x = (np.arange(N)- N//2) * delta
        # plt.pcolor(x,x,phase_atm_pad)
        # plt.colorbar()
        # plt.xlabel('position (m)')
        # plt.axis('square')
        # plt.title('phase_atm_pad (rad)' + '\n' + TAG)
        # # plt.savefig('sample_prop_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
        # plt.show()
        
        # plt.pcolor(x,x,intensity_atm_pad,vmax=2,vmin=0)
        # plt.colorbar()
        # plt.axis('square')
        # plt.xlabel('position (m)')
        # plt.title('normalized intensity_atm_pad' + '\n' + TAG)
        # # plt.savefig('sample_prop_intensity_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
        # plt.show()
        
        # PSDUatm = np.zeros((N,N),dtype=float)
        # for i in np.arange(len(atm_pad_Uout)):
        #     PSDUatm[i] = mps.get_PSD(atm_pad_Uout[i],delta)
        # PSDUatm[:] += PSDUatm[:,::-1]
        # PSDUatm /= 2.
        
        # f=(np.arange(N//2-1) + 1)/N/delta
        # A = np.mean(PSDUatm[:,N//2+1:],axis=0)
        # Ae = np.std(PSDUatm[:,N//2+1:],axis=0)/np.sqrt(N)
        # B = np.mean(PSDUprop[:,N//2+1:],axis=0)
        # Be = np.std(PSDUprop[:,N//2+1:],axis=0)/np.sqrt(N)
        
        # plt.plot(f,A,color='r',label='atm pad')
        # plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(f,B,color='b',label='prop pad')
        # plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
        # plt.legend()
        # plt.ylim(B.min()*0.9,B.max()*1.1)
        # plt.title('psd U' + '\n' + TAG)
        # plt.xlabel('f (1/m)')
        # plt.ylabel('PSD_U (I$_{norm}$ m$^2$)')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        
        
        
        # PSDphaseatm = np.zeros((N,N),dtype=float)
        # for i in np.arange(len(phase_atm_pad)):
        #     PSDphaseatm[i] = mps.get_PSD(phase_atm_pad[i],delta)
        # PSDphaseatm[:] += PSDphaseatm[:,::-1]
        # PSDphaseatm /= 2.
        
        # f=(np.arange(N//2-1) + 1)/N/delta
        # A = np.mean(PSDphaseatm[:,N//2+1:],axis=0)
        # Ae = np.std(PSDphaseatm[:,N//2+1:],axis=0)/np.sqrt(N)
        # B = np.mean(PSDphaseprop[:,N//2+1:],axis=0)
        # Be = np.std(PSDphaseprop[:,N//2+1:],axis=0)/np.sqrt(N)
        
        # plt.plot(f,A,color='r',label='atm pad')
        # plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(f,B,color='b',label='prop pad')
        # plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
        # plt.legend()
        # plt.ylim(B.min()*0.9,B.max()*1.1)
        # plt.title('psd phase' + '\n' + TAG)
        # plt.xlabel('f (1/m)')
        # plt.ylabel('PSD_$\phi$ (rad$^2$ m$^2$)')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        
        
        
        # PSDintensityatm = np.zeros((N,N),dtype=float)
        # for i in np.arange(len(intensity_atm_pad)):
        #     PSDintensityatm[i] = mps.get_PSD(intensity_atm_pad[i],delta)
        # PSDintensityatm[:] += PSDintensityatm[:,::-1]
        # PSDintensityatm /= 2.
        
        # f = (np.arange(N//2-1) + 1)/N/delta
        # A = np.mean(PSDintensityatm[:,N//2+1:],axis=0)
        # Ae = np.std(PSDintensityatm[:,N//2+1:],axis=0)/np.sqrt(N)
        # B = np.mean(PSDintensityprop[:,N//2+1:],axis=0)
        # Be = np.std(PSDintensityprop[:,N//2+1:],axis=0)/np.sqrt(N)
        
        # plt.plot(f,A,color='r',label='atm pad')
        # plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(f,B,color='b',label='prop pad')
        # plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
        # plt.vlines([1/np.sqrt(wvl*z)],B.min(),B.max(),label='1/$\sqrt{\lambda z}$',color='k')
        # plt.legend()
        # plt.ylim(B.min()*0.9,B.max()*1.1)
        # plt.title('psd intensity' + '\n' + TAG)
        # plt.xlabel('f (1/m)')
        # plt.ylabel('PSD_I (I$_{norm}$ m$^2$)')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        
        
        
        # SFphaseatm = np.zeros((N,N), dtype=float)
        # # for i in np.arange(N):
        # #     SFphaseatm[i] = mps.get_structure_function(np.roll(phase_atm_pad,(i),axis=(0)))
        # # SFphaseatm = (phase_atm_pad[:,0] - phase_atm_pad[:,:])**2
        # for i in np.arange(N):
        #     SFphaseatm[i] = np.angle((atm_pad_Uout[i,i] / np.roll(atm_pad_Uout[i],(-i),axis=(0))))**2
        
        
        # x = np.arange(N//2)*delta
        # A = np.mean(SFphaseatm[:,:N//2],axis=0)
        # Ae = np.std(SFphaseatm[:,:N//2],axis=0)/np.sqrt(N)
        # B = np.mean(SFphaseprop[:,:N//2],axis=0)
        # Be = np.std(SFphaseprop[:,:N//2],axis=0)/np.sqrt(N)
        
        # plt.plot(x,A,color='r',label='atm pad')
        # plt.fill_between(x,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(x,B,color='b',label='prop pad')
        # plt.fill_between(x,B-Be,B+Be,alpha=0.2,color='b')
        # plt.legend()
        # plt.ylim(B[1:].min()*0.9,B.max()*1.1)
        # plt.title('structure function $\phi$' + '\n' + TAG)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel('separation (m)')
        # plt.ylabel('$\sigma^2_\phi$ (rad$^2$)')
        # plt.show()
        
        
        
        # SFintensityatm = np.zeros((N,N), dtype=float)
        # # for i in np.arange(N):
        # #     SFintensityatm[i] = mps.get_structure_function(np.roll(intensity_atm_pad,(i),axis=(0)))
        # # SFintensityatm = (intensity_atm_pad[:,0] - intensity_atm_pad[:,:])**2
        # for i in np.arange(N):
        #     SFintensityatm[i] = (intensity_atm_pad[i,i] - np.roll(intensity_atm_pad[i],(-i),axis=(0)))**2
        
        # x = np.arange(N//2)*delta
        # A = np.mean(SFintensityatm[:,:N//2],axis=0)
        # Ae = np.std(SFintensityatm[:,:N//2],axis=0)/np.sqrt(N)
        # B = np.mean(SFintensityprop[:,:N//2],axis=0)
        # Be = np.std(SFintensityprop[:,:N//2],axis=0)/np.sqrt(N)
        
        # plt.plot(x,A,color='r',label='atm pad')
        # plt.fill_between(x,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(x,B,color='b',label='prop pad')
        # plt.fill_between(x,B-Be,B+Be,alpha=0.2,color='b')
        # plt.vlines([np.sqrt(wvl*z)],
        #             B.min(),B.max(),label='$\sqrt{\lambda z}$',color='k')
        # plt.legend()
        # plt.title('structure function I' + '\n' + TAG)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel('separation (m)')
        # plt.ylabel('$\sigma^2_I$ (I$_{norm}$$^2$)')
        # plt.show()
        
        
        ###############################################################################
        
        
        # reduce infinite phase screen size # pad to prop size # symmetric pad
        # everything matching; 0 & 1 wvl*z/r0/delta is not enough. Must be >=2
        TAG = 'prop screen size = symmetric_pad --> prop'
        atm_pad_scrn = scrn[prop_pad-atm_pad:-(prop_pad-atm_pad),
                            prop_pad-atm_pad:-(prop_pad-atm_pad)]
        atm_pad_Uin = np.exp(1j*np.pad(atm_pad_scrn,
                                        ((prop_pad-atm_pad,prop_pad-atm_pad),
                                          (prop_pad-atm_pad,prop_pad-atm_pad)),'symmetric'))#'symmetric',reflect_type='even'))
        atm_pad_Uout = angularSpectrum(atm_pad_Uin, wvl, delta, delta, z)[prop_pad:-prop_pad,
                                                                          prop_pad:-prop_pad]
        
        intensity_atm_pad = np.abs(atm_pad_Uout)**2
        phase_atm_pad = np.angle(atm_pad_Uout)
        
        if z == 10000:
        
            x = (np.arange(N)- N//2) * delta
            plt.figure(figsize=(5,4))
            plt.pcolor(x,x,phase_atm_pad)
            plt.colorbar()
            plt.xlabel('Position (m)')
            plt.axis('square')
            # plt.title('phase_atm_pad (rad)' + '\n' + TAG)
            # plt.savefig('sample_prop_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
            plt.savefig('phase-shortcut.png',dpi=600,bbox_inches='tight')
            plt.show()
            
            plt.figure(figsize=(5,4))
            plt.pcolor(x,x,intensity_atm_pad,vmax=2,vmin=0)
            plt.colorbar()
            plt.axis('square')
            plt.xlabel('Position (m)')
            plt.savefig('intensity-shortcut.png',dpi=600,bbox_inches='tight')
            # plt.title('normalized intensity_atm_pad' + '\n' + TAG)
            # plt.savefig('sample_prop_intensity_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
            plt.show()
        
            # PSDUatm = np.zeros((N,N),dtype=float)
            # for i in np.arange(len(atm_pad_Uout)):
            #     PSDUatm[i] = mps.get_PSD(atm_pad_Uout[i],delta)
            # PSDUatm[:] += PSDUatm[:,::-1]
            # PSDUatm /= 2.
            
            # f=(np.arange(N//2-1) + 1)/N/delta
            # A = np.mean(PSDUatm[:,N//2+1:],axis=0)
            # Ae = np.std(PSDUatm[:,N//2+1:],axis=0)/np.sqrt(N)
            # B = np.mean(PSDUprop[:,N//2+1:],axis=0)
            # Be = np.std(PSDUprop[:,N//2+1:],axis=0)/np.sqrt(N)
            
            
            # plt.plot(f,B,color='r',label='Full')
            # plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='r')
            # plt.plot(f,A,color='b',label='Reduced')
            # plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='b')
            # plt.legend()
            # plt.ylim(B.min()*0.9,B.max()*1.1)
            # # plt.title('psd U' + '\n' + TAG)
            # plt.xlabel('Spatial frequency (1/m)')
            # plt.ylabel('$PSD_U$ (I$_{norm}$ m)')
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.savefig('1DPSD_PROP_U{:.0f}.pdf'.format(z/1000),bbox_inches='tight')
            # plt.show()
            
        
        
            PSDphaseatm = np.zeros((N,N),dtype=float)
            for i in np.arange(len(phase_atm_pad)):
                PSDphaseatm[i] = mps.get_PSD(phase_atm_pad[i],delta)
            PSDphaseatm[:] += PSDphaseatm[:,::-1]
            PSDphaseatm /= 2.
        
            f=(np.arange(N//2-1) + 1)/N/delta
            A = np.mean(PSDphaseatm[:,N//2+1:],axis=0)
            Ae = np.std(PSDphaseatm[:,N//2+1:],axis=0)/np.sqrt(N)
            B = np.mean(PSDphaseprop[:,N//2+1:],axis=0)
            Be = np.std(PSDphaseprop[:,N//2+1:],axis=0)/np.sqrt(N)
        
        
            
            plt.plot(f,B,color='r',label='Full')
            plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='r')
            plt.plot(f,A,color='b',label='Reduced')
            plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='b')
            plt.legend()
            plt.ylim(B.min()*0.9,B.max()*1.1)
            # plt.title('psd phase' + '\n' + TAG)
            plt.xlabel('Spatial frequency (1/m)')
            plt.ylabel('Phase power spectrum density (rad$^2$ m)')
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig('1DPSD_PROP_phase{:.0f}.pdf'.format(z/1000),bbox_inches='tight')
            plt.show()
        
        
        
            PSDintensityatm = np.zeros((N,N),dtype=float)
            for i in np.arange(len(intensity_atm_pad)):
                PSDintensityatm[i] = mps.get_PSD(intensity_atm_pad[i],delta)
            PSDintensityatm[:] += PSDintensityatm[:,::-1]
            PSDintensityatm /= 2.
            
            f = (np.arange(N//2-1) + 1)/N/delta
            A = np.mean(PSDintensityatm[:,N//2+1:],axis=0)
            Ae = np.std(PSDintensityatm[:,N//2+1:],axis=0)/np.sqrt(N)
            B = np.mean(PSDintensityprop[:,N//2+1:],axis=0)
            Be = np.std(PSDintensityprop[:,N//2+1:],axis=0)/np.sqrt(N)
        
        
            plt.plot(f,B,color='r',label='Full')
            plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='r')
            plt.plot(f,A,color='b',label='Reduced')
            plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='b')
            
            # plt.vlines([1/np.sqrt(wvl*z)],B.min(),B.max(),label='1/$\sqrt{\lambda z}$',color='k')
            plt.legend()
            plt.ylim(B.min()*0.9,B.max()*1.1)
            # plt.title('psd intensity' + '\n' + TAG)
            plt.xlabel('Spatial frequency (1/m)')
            plt.ylabel('Intensity power spectrum density')
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig('1DPSD_PROP_I{:.0f}.pdf'.format(z/1000),bbox_inches='tight')
            plt.show()
            
        
        
            SFphaseatm = np.zeros((N,N), dtype=float)
            # for i in np.arange(N):
            #     SFphaseatm[i] = mps.get_structure_function(np.roll(phase_atm_pad,(i),axis=(0)))
            # SFphaseatm = (phase_atm_pad[:,0] - phase_atm_pad[:,:])**2
            for i in np.arange(N):
                SFphaseatm[i] = np.angle((atm_pad_Uout[i,i] / np.roll(atm_pad_Uout[i],(-i),axis=(0))))**2
        
            
            x = np.arange(N//2)*delta
            A = np.mean(SFphaseatm[:,:N//2],axis=0)
            Ae = np.std(SFphaseatm[:,:N//2],axis=0)/np.sqrt(N)
            B = np.mean(SFphaseprop[:,:N//2],axis=0)
            Be = np.std(SFphaseprop[:,:N//2],axis=0)/np.sqrt(N)
        
        
            
            plt.plot(x,B,color='r',label='Full')
            plt.fill_between(x,B-Be,B+Be,alpha=0.2,color='r')
            plt.plot(x,A,color='b',label='Reduced')
            plt.fill_between(x,A-Ae,A+Ae,alpha=0.2,color='b')
            plt.legend()
            plt.ylim(B[1:].min()*0.9,B.max()*1.1)
            # plt.title('structure function $\phi$' + '\n' + TAG)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Separation (m)')
            plt.ylabel('Phase structure function (rad$^2$)')
            plt.savefig('SF_PROP_phase{:.0f}.pdf'.format(z/1000),bbox_inches='tight')
            plt.show()
        
        
        
            SFintensityatm = np.zeros((N,N), dtype=float)
            # for i in np.arange(N):
            #     SFintensityatm[i] = mps.get_structure_function(np.roll(intensity_atm_pad,(i),axis=(0)))
            # SFintensityatm = (intensity_atm_pad[:,0] - intensity_atm_pad[:,:])**2
            for i in np.arange(N):
                SFintensityatm[i] = (intensity_atm_pad[i,i] - np.roll(intensity_atm_pad[i],(-i),axis=(0)))**2
            
            x = np.arange(N//2)*delta
            A = np.mean(SFintensityatm[:,:N//2],axis=0)
            Ae = np.std(SFintensityatm[:,:N//2],axis=0)/np.sqrt(N)
            B = np.mean(SFintensityprop[:,:N//2],axis=0)
            Be = np.std(SFintensityprop[:,:N//2],axis=0)/np.sqrt(N)
        
        
            plt.plot(x,B,color='r',label='Full')
            plt.fill_between(x,B-Be,B+Be,alpha=0.2,color='r')
            plt.plot(x,A,color='b',label='Reduced')
            plt.fill_between(x,A-Ae,A+Ae,alpha=0.2,color='b')
            # plt.vlines([np.sqrt(wvl*z)],
            #             B.min(),B.max(),label='$\sqrt{\lambda z}$',color='k')
            plt.legend()
            # plt.title('structure function I' + '\n' + TAG)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Separation (m)')
            plt.ylabel('Intensity structure function')
            plt.savefig('SF_PROP_I{:.0f}.pdf'.format(z/1000),bbox_inches='tight')
            plt.show()
            
        
            SFUatm = np.zeros((N,N), dtype=float)
            for i in np.arange(N):
                SFUatm[i] = np.abs(atm_pad_Uout[i,i] - np.roll(atm_pad_Uout[i],(-i),axis=(0)))**2
            SFUprop = np.zeros((N,N), dtype=float)
            for i in np.arange(N):
                SFUprop[i] = np.abs(prop_pad_Uout[i,i] - np.roll(prop_pad_Uout[i],(-i),axis=(0)))**2
            
            x = np.arange(N//2)*delta
            A = np.mean(SFUatm[:,:N//2],axis=0)
            Ae = np.std(SFUatm[:,:N//2],axis=0)/np.sqrt(N)
            B = np.mean(SFUprop[:,:N//2],axis=0)
            Be = np.std(SFUprop[:,:N//2],axis=0)/np.sqrt(N)
        
        dif = np.angle(atm_pad_Uout/prop_pad_Uout)#*intensity_prop_pad**0.5
        weighted_rms = np.sum(dif**2*intensity_prop_pad**2)/np.sum(intensity_prop_pad**2)*500/2/np.pi
        #rms_error_i = np.sqrt(np.mean(weighted_rms)**2 * (np.std(dif**2)/np.mean(dif**2))**2 + (np.std(intensity_prop_pad)/np.mean(intensity_prop_pad))**2)
        rms_error_i = (1/np.sum(intensity_prop_pad**2))**0.5#(np.sum(intensity_prop_pad**2)/np.sum(intensity_prop_pad)**2)**0.5*weighted_rms
        
        if z == 10000:
        
            # plt.plot(x,A,color='r',label='shortcut')
            # plt.fill_between(x,A-Ae,A+Ae,alpha=0.2,color='r')
            # plt.plot(x,B,color='b',label='proper')
            # plt.fill_between(x,B-Be,B+Be,alpha=0.2,color='b')
            # # plt.vlines([np.sqrt(wvl*z)],
            # #             B.min(),B.max(),label='$\sqrt{\lambda z}$',color='k')
            # plt.legend()
            # # plt.title('structure function U' + '\n' + TAG)
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.xlabel('separation (m)')
            # plt.ylabel('$D_U$ (I$_{norm}$)')
            # plt.savefig('SF_PROP_U{:.0f}.png'.format(z/1000),bbox_inches='tight')
            # plt.show()
            
            plt.figure(figsize=(5,4))
            
            plt.imshow(dif*intensity_prop_pad)
            plt.colorbar()
            
            plt.xlabel('Position (m)')
            # plt.title('OPD intensity-weighted RMS difference {:.0f} (nm)'.format(weighted_rms))
            # plt.savefig('OPD_RMS_DIF{:.0f}.pdf'.format(z/1000),bbox_inches='tight')
            plt.savefig('OPD_IWRMS.png'.format(z/1000),bbox_inches='tight')
            plt.show()
        
        rms[i_z,i_r0] = weighted_rms
        rms_error[i_z,i_r0] = rms_error_i
        
        ###############################################################################
        ###############################################################################
        
        
        # ###############################################################################
        
        
        # # reduce infinite phase screen size # pad to 2x(ATM) # symmetric pad
        # # everything matching!!!
        # TAG = 'prop screen size = symmetric_pad --> 2*ATM'
        # atm_pad_scrn = scrn[prop_pad-atm_pad:-(prop_pad-atm_pad),prop_pad-atm_pad:-(prop_pad-atm_pad)]
        # atm_pad_Uin = np.exp(1j*np.pad(atm_pad_scrn,
        #                                 ((atm_pad_scrn.shape[0]//2,atm_pad_scrn.shape[0]//2),
        #                                   (atm_pad_scrn.shape[0]//2,atm_pad_scrn.shape[0]//2)),'symmetric'))#'symmetric',reflect_type='even'))
        # atm_pad_Uout = angularSpectrum(atm_pad_Uin, wvl, delta, delta, z)[(N+4*atm_pad)//2:-(N+4*atm_pad)//2,
        #                                                                   (N+4*atm_pad)//2:-(N+4*atm_pad)//2]
        
        # intensity_atm_pad = np.abs(atm_pad_Uout)**2
        # phase_atm_pad = np.angle(atm_pad_Uout)
        
        # x = (np.arange(N)- N//2) * delta
        # plt.pcolor(x,x,phase_atm_pad)
        # plt.colorbar()
        # plt.xlabel('position (m)')
        # plt.axis('square')
        # plt.title('phase_atm_pad (rad)' + '\n' + TAG)
        # # plt.savefig('sample_prop_phase_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
        # plt.show()
        
        # plt.pcolor(x,x,intensity_atm_pad,vmax=2,vmin=0)
        # plt.colorbar()
        # plt.axis('square')
        # plt.xlabel('position (m)')
        # plt.title('normalized intensity_atm_pad' + '\n' + TAG)
        # # plt.savefig('sample_prop_intensity_L0{:.0f}m_r0{:.0f}cm_wvl{:.0f}nm_z{:.0f}km.pdf'.format(L0,r0*100,wvl/1e-9,z/1e3),bbox_inches='tight')
        # plt.show()
        
        # PSDUatm = np.zeros((N,N),dtype=float)
        # for i in np.arange(len(atm_pad_Uout)):
        #     PSDUatm[i] = mps.get_PSD(atm_pad_Uout[i],delta)
        # PSDUatm[:] += PSDUatm[:,::-1]
        # PSDUatm /= 2.
        
        # f=(np.arange(N//2-1) + 1)/N/delta
        # A = np.mean(PSDUatm[:,N//2+1:],axis=0)
        # Ae = np.std(PSDUatm[:,N//2+1:],axis=0)/np.sqrt(N)
        # B = np.mean(PSDUprop[:,N//2+1:],axis=0)
        # Be = np.std(PSDUprop[:,N//2+1:],axis=0)/np.sqrt(N)
        
        # plt.plot(f,A,color='r',label='atm pad')
        # plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(f,B,color='b',label='prop pad')
        # plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
        # plt.legend()
        # plt.ylim(B.min()*0.9,B.max()*1.1)
        # plt.title('psd U' + '\n' + TAG)
        # plt.xlabel('f (1/m)')
        # plt.ylabel('PSD_U (I$_{norm}$ m$^2$)')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        
        
        
        # PSDphaseatm = np.zeros((N,N),dtype=float)
        # for i in np.arange(len(phase_atm_pad)):
        #     PSDphaseatm[i] = mps.get_PSD(phase_atm_pad[i],delta)
        # PSDphaseatm[:] += PSDphaseatm[:,::-1]
        # PSDphaseatm /= 2.
        
        # f=(np.arange(N//2-1) + 1)/N/delta
        # A = np.mean(PSDphaseatm[:,N//2+1:],axis=0)
        # Ae = np.std(PSDphaseatm[:,N//2+1:],axis=0)/np.sqrt(N)
        # B = np.mean(PSDphaseprop[:,N//2+1:],axis=0)
        # Be = np.std(PSDphaseprop[:,N//2+1:],axis=0)/np.sqrt(N)
        
        # plt.plot(f,A,color='r',label='atm pad')
        # plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(f,B,color='b',label='prop pad')
        # plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
        # plt.legend()
        # plt.ylim(B.min()*0.9,B.max()*1.1)
        # plt.title('psd phase' + '\n' + TAG)
        # plt.xlabel('f (1/m)')
        # plt.ylabel('PSD_$\phi$ (rad$^2$ m$^2$)')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        
        
        
        # PSDintensityatm = np.zeros((N,N),dtype=float)
        # for i in np.arange(len(intensity_atm_pad)):
        #     PSDintensityatm[i] = mps.get_PSD(intensity_atm_pad[i],delta)
        # PSDintensityatm[:] += PSDintensityatm[:,::-1]
        # PSDintensityatm /= 2.
        
        # f = (np.arange(N//2-1) + 1)/N/delta
        # A = np.mean(PSDintensityatm[:,N//2+1:],axis=0)
        # Ae = np.std(PSDintensityatm[:,N//2+1:],axis=0)/np.sqrt(N)
        # B = np.mean(PSDintensityprop[:,N//2+1:],axis=0)
        # Be = np.std(PSDintensityprop[:,N//2+1:],axis=0)/np.sqrt(N)
        
        # plt.plot(f,A,color='r',label='atm pad')
        # plt.fill_between(f,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(f,B,color='b',label='prop pad')
        # plt.fill_between(f,B-Be,B+Be,alpha=0.2,color='b')
        # plt.vlines([1/np.sqrt(wvl*z)],B.min(),B.max(),label='1/$\sqrt{\lambda z}$',color='k')
        # plt.legend()
        # plt.ylim(B.min()*0.9,B.max()*1.1)
        # plt.title('psd intensity' + '\n' + TAG)
        # plt.xlabel('f (1/m)')
        # plt.ylabel('PSD_I (I$_{norm}$ m$^2$)')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        
        
        
        # SFphaseatm = np.zeros((N,N), dtype=float)
        # # for i in np.arange(N):
        # #     SFphaseatm[i] = mps.get_structure_function(np.roll(phase_atm_pad,(i),axis=(0)))
        # # SFphaseatm = (phase_atm_pad[:,0] - phase_atm_pad[:,:])**2
        # for i in np.arange(N):
        #     SFphaseatm[i] = np.angle((atm_pad_Uout[i,i] / np.roll(atm_pad_Uout[i],(-i),axis=(0))))**2
        
        
        # x = np.arange(N//2)*delta
        # A = np.mean(SFphaseatm[:,:N//2],axis=0)
        # Ae = np.std(SFphaseatm[:,:N//2],axis=0)/np.sqrt(N)
        # B = np.mean(SFphaseprop[:,:N//2],axis=0)
        # Be = np.std(SFphaseprop[:,:N//2],axis=0)/np.sqrt(N)
        
        # plt.plot(x,A,color='r',label='atm pad')
        # plt.fill_between(x,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(x,B,color='b',label='prop pad')
        # plt.fill_between(x,B-Be,B+Be,alpha=0.2,color='b')
        # plt.legend()
        # plt.ylim(B[1:].min()*0.9,B.max()*1.1)
        # plt.title('structure function $\phi$' + '\n' + TAG)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel('separation (m)')
        # plt.ylabel('$\sigma^2_\phi$ (rad$^2$)')
        # plt.show()
        
        
        
        # SFintensityatm = np.zeros((N,N), dtype=float)
        # # for i in np.arange(N):
        # #     SFintensityatm[i] = mps.get_structure_function(np.roll(intensity_atm_pad,(i),axis=(0)))
        # # SFintensityatm = (intensity_atm_pad[:,0] - intensity_atm_pad[:,:])**2
        # for i in np.arange(N):
        #     SFintensityatm[i] = (intensity_atm_pad[i,i] - np.roll(intensity_atm_pad[i],(-i),axis=(0)))**2
        
        # x = np.arange(N//2)*delta
        # A = np.mean(SFintensityatm[:,:N//2],axis=0)
        # Ae = np.std(SFintensityatm[:,:N//2],axis=0)/np.sqrt(N)
        # B = np.mean(SFintensityprop[:,:N//2],axis=0)
        # Be = np.std(SFintensityprop[:,:N//2],axis=0)/np.sqrt(N)
        
        # plt.plot(x,A,color='r',label='atm pad')
        # plt.fill_between(x,A-Ae,A+Ae,alpha=0.2,color='r')
        # plt.plot(x,B,color='b',label='prop pad')
        # plt.fill_between(x,B-Be,B+Be,alpha=0.2,color='b')
        # plt.vlines([np.sqrt(wvl*z)],
        #             B.min(),B.max(),label='$\sqrt{\lambda z}$',color='k')
        # plt.legend()
        # plt.title('structure function I' + '\n' + TAG)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel('separation (m)')
        # plt.ylabel('$\sigma^2_I$ (I$_{norm}$$^2$)')
        # plt.show()
        

for i in range(len(r0_list)):
    plt.errorbar(np.logspace(2,4,datapoint),
                 rms[:,i],
                 yerr=rms_error[:,i],
                 capsize=5,marker='o',
                 label='$r_0$={:.0f}cm'.format(r0_list[i]*100))
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Propagation distance (m)')
plt.ylabel('Intensity-weighted RMS OPD (nm)')
plt.savefig('IW-OPD-RMS-dif-cheat.pdf',bbox_inches='tight')
plt.show()





