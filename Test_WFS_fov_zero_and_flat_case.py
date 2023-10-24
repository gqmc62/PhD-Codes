"""Test SHWFS against fov parameter."""

import numpy as np
import pyfftw
from aotools.functions.zernike import zernikeArray
from matplotlib import pyplot as plt

N = int(64)
oversize = int(16)
zero_pad = int(2)
wvl = float(500e-9)
L0 = float(20)
l0 = float(1e-7)
r0 = float(5e-2)
dor0 = np.array([0.5,1,2,3], dtype=float)
REPEAT1 = int(12)
REPEAT2 =  int(10000)

d = r0*dor0
delta = d/N
f_lenslet = delta**2*2*N/wvl
CEN = int(N/2)

# NN = int(oversize*N)
# edge_low = int(NN*((oversize - 1)/oversize)/2)
# edge_high = int(NN*((oversize + 1)/oversize)/2)
# add_each_side = int((oversize - 1)/2*N)

N_zero_pad = N * zero_pad
edge_low0 = int(N_zero_pad*((zero_pad - 1)/zero_pad)/2)
edge_high0 = int(N_zero_pad*((zero_pad + 1)/zero_pad)/2)
add_each_side0 = int((zero_pad - 1)*N/2)

F0 = pyfftw.empty_aligned((zero_pad*N, zero_pad*N), dtype=complex)
F1 = pyfftw.empty_aligned((zero_pad*N, zero_pad*N), dtype=complex)
FF01 = pyfftw.FFTW(F0, F1, axes=(-1,-2), direction='FFTW_FORWARD')

Threshold = 1e-4
wavenumber = 2*np.pi/wvl
pupil, tip, tilt = zernikeArray(3,N)

CEN = int(N/2)

# ###############################################################################
# # Zero slope #############################################################
# ###############################################################################

zero_phase = np.zeros((N, N), dtype=float)
g_slope_zero = np.zeros((REPEAT1, 2), dtype=float)
real_SHWFS_slope_zero = np.zeros((REPEAT1, CEN, 2), dtype=float)
dif_SHWFS_slope_zero = np.zeros((REPEAT1, CEN, 2), dtype=float)

for num_dor in np.arange(len(dor0)):
    (y1, x1) = np.meshgrid(delta[num_dor]*np.arange(-N/2,N/2),
                          delta[num_dor]*np.arange(-N/2,N/2))
    Q = np.exp(-1j*wavenumber/2./f_lenslet[num_dor] * (((x1 + y1) * delta[num_dor])))
    convert_real = ((delta[num_dor]/f_lenslet[num_dor]) # convert pixel drift to angle drift
                    * (delta[num_dor] * N) # convert angle drift to wavefront error
                    * (2*np.pi/wvl)) # convert wavefront error to phase error
    for repeat in np.arange(REPEAT1):
        U_pupil = np.exp(1j*zero_phase)

        # unit is rad per lenslet
        g_slope_zero[repeat,0] = np.mean(zero_phase[1:,:] - zero_phase[:-1,:]) * N
        g_slope_zero[repeat,1] = np.mean(zero_phase[:,1:] - zero_phase[:,:-1]) * N

        F0[:,:] = np.pad(U_pupil * Q,
                          (add_each_side0, add_each_side0),
                          'constant', constant_values=0.)
        # F0[:,:] = np.pad(U_pupil,
        #                   (add_each_side,add_each_side),
        #                   'constant', constant_values=0.) * QQ
        I_focus = (np.abs(
            np.fft.fftshift(FF01())[edge_low0 : edge_high0,
                                    edge_low0 : edge_high0])**2
            / (2*N**2)**2)

        # unit of half fov is pixel! = fov in wvl/d
        for half_fov in (np.arange(CEN) + 1):
            yy, xx = np.meshgrid(np.arange(-half_fov, half_fov) + 0.5,
                                  np.arange(-half_fov, half_fov) + 0.5)

            I_fov = I_focus[CEN - half_fov : CEN + half_fov,
                            CEN - half_fov : CEN + half_fov]

            I_fov = I_fov/I_fov.max()
            I_fov = I_fov - np.max([I_fov[0,:].max(),
                                        I_fov[-1,:].max(),
                                        I_fov[:,0].max(),
                                        I_fov[:,-1].max()])
            I_fov[np.where(I_fov < 0)] = 0.0
            I_fov = I_fov/np.sum(I_fov)

            # unit is rad per lenslet
            real_SHWFS_slope_zero[repeat,half_fov - 1, 0] = (np.sum(I_fov*xx)
                                                              * convert_real)
            real_SHWFS_slope_zero[repeat,half_fov - 1, 1] = (np.sum(I_fov*yy)
                                                              * convert_real)

            dif_SHWFS_slope_zero[repeat,half_fov - 1,:] = (g_slope_zero[repeat,:]
                                                            - real_SHWFS_slope_zero[repeat,half_fov - 1,:])


plt.errorbar(x=np.arange(CEN) + 1,
              y=np.mean(real_SHWFS_slope_zero[:,:,0],axis=0),
              yerr=np.std(real_SHWFS_slope_zero[:,:,0],axis=0,ddof=1),
              label='x-axis slope', marker='.')
plt.errorbar(x=np.arange(CEN) + 1,
              y=np.mean(real_SHWFS_slope_zero[:,:,1],axis=0),
              yerr=np.std(real_SHWFS_slope_zero[:,:,1],axis=0,ddof=1),
              label='y-axis slope', marker='.')
plt.legend()
plt.title('SHWFS zero slope measurement')
plt.xlabel('FoV (wvl/d)')
plt.ylabel('Slope (rad)')
# plt.xscale('log')
# plt.yscale('log')
plt.show()

###############################################################################
# fov analysis Flat slope no threshold ###################################
###############################################################################

flat_phase = np.zeros((N, N), dtype=float)
g_slope_flat = np.zeros((REPEAT1, 2), dtype=float)
real_SHWFS_slope_flat = np.zeros((REPEAT1, CEN, 2), dtype=float)
dif_SHWFS_slope_flat = np.zeros((REPEAT1, CEN, 2), dtype=float)

for num_dor in np.arange(len(dor0)):
    (y1, x1) = np.meshgrid(delta[num_dor]*np.arange(-N/2,N/2),
                          delta[num_dor]*np.arange(-N/2,N/2))
    Q = np.exp(-1j*wavenumber/2./f_lenslet[num_dor] * (((x1 + y1) * delta[num_dor])))
    convert_real = ((delta[num_dor]/f_lenslet[num_dor]) # convert pixel drift to angle drift
                    * (delta[num_dor] * N) # convert angle drift to wavefront error
                    * (2*np.pi/wvl)) # convert wavefront error to phase error
    for repeat in np.arange(REPEAT1):
        # flat_slope_x = (np.random.rand() - 0.5)*2 * (N/4) * (np.pi/N/delta[num_dor])
        # flat_slope_y = (np.random.rand() - 0.5)*2 * (N/4) * (np.pi/N/delta[num_dor])
        flat_slope_x = ((repeat+1)*N/4/REPEAT1) / delta[num_dor] / N * np.pi
        flat_slope_y = ((repeat+1)*N/4/REPEAT1) / delta[num_dor] / N * np.pi
        flat_phase = (flat_slope_x*x1 + flat_slope_y*y1)
        U_pupil = np.exp(1j*flat_phase)
        
        # unit is rad per lenslet
        g_slope_flat[repeat,0] = np.mean(flat_phase[1:,:] - flat_phase[:-1,:]) * N
        g_slope_flat[repeat,1] = np.mean(flat_phase[:,1:] - flat_phase[:,:-1]) * N
    
        F0[:,:] = np.pad(U_pupil * Q,
                          (add_each_side0, add_each_side0),
                          'constant', constant_values=0.)
        I_focus = (np.abs(
            np.fft.fftshift(FF01())[edge_low0 : edge_high0,
                                    edge_low0 : edge_high0])**2
            / (2*N**2)**2)

        I_focus = I_focus/I_focus.max()
        # I_focus = I_focus - np.max([I_focus[0,:].max(),
        #                             I_focus[-1,:].max(),
        #                             I_focus[:,0].max(),
        #                             I_focus[:,-1].max()])
        # I_focus = I_focus - Threshold
        # I_focus[np.where(I_focus < 0)] = 0.0
        # I_focus = I_focus/np.sum(I_focus)

        # unit of half fov is pixel! = fov in wvl/d
        for half_fov in (np.arange(CEN) + 1):
            yy, xx = np.meshgrid(np.arange(-half_fov, half_fov) + 0.5,
                                  np.arange(-half_fov, half_fov) + 0.5)
            # x = np.arange(-half_fov, half_fov) + 0.5
    
            I_fov = I_focus[CEN - half_fov : CEN + half_fov,
                            CEN - half_fov : CEN + half_fov]

            # I_fov = I_fov/I_fov.max()
            # I_fov = I_fov - np.max([I_fov[0,:].max(),
            #                             I_fov[-1,:].max(),
            #                             I_fov[:,0].max(),
            #                             I_fov[:,-1].max()])
            # I_fov = I_fov - 1e-4
            # I_fov[np.where(I_fov < 0)] = 0.0
            I_fov = I_fov/np.sum(I_fov)

            # unit is rad per lenslet
            # real_SHWFS_slope_flat[repeat,half_fov - 1, 0] = (np.sum(I_fov*xx)
            #                                                  / np.sum(I_fov)
            #                                                  * convert_real)
            # real_SHWFS_slope_flat[repeat,half_fov - 1, 1] = (np.sum(I_fov*yy)
            #                                                 / np.sum(I_fov)
            #                                                   * convert_real)
            real_SHWFS_slope_flat[repeat, half_fov - 1, 0] = np.sum(I_fov * xx) / np.sum(I_fov) * convert_real
            real_SHWFS_slope_flat[repeat, half_fov - 1, 1] = np.sum(I_fov * yy) / np.sum(I_fov) * convert_real
    
            dif_SHWFS_slope_flat[repeat,half_fov - 1, : ] = (g_slope_flat[repeat,:]
                                                            - real_SHWFS_slope_flat[repeat,half_fov - 1,:])

# plt.imshow(I_focus[CEN-4:CEN+4,CEN-4:CEN+4].T)
# plt.xlabel('x = +1px')
# plt.ylabel('y = +2px')
# plt.vlines(3.5,ymin=-0.5,ymax=7.5)
# plt.hlines(3.5,xmin=-0.5,xmax=7.5)
# plt.show()

for repeat in np.arange(REPEAT1):
    plt.plot(np.arange(CEN) + 1,
                  np.abs(dif_SHWFS_slope_flat[repeat,:,0]),
                  label='{:.1f}'.format((repeat+1)/REPEAT1*N/4),marker='.')
plt.legend()
plt.title('Absolute difference between\nSHWFS and g_tilt in flat slope case\nwithout threshold for varying centroid shift in pixel')
plt.xlabel('FoV (wvl/d)')
plt.ylabel('Mean of slope difference (rad)')
plt.xscale('log')
plt.yscale('log')
plt.xlim((1,64))
plt.show()

###############################################################################
# fov and threshold analysis Flat slope with threshold ###################
###############################################################################

flat_phase = np.zeros((N, N), dtype=float)
g_slope_flat = np.zeros((REPEAT1, 2), dtype=float)
real_SHWFS_slope_flat = np.zeros((REPEAT1, CEN, 2), dtype=float)
dif_SHWFS_slope_flat = np.zeros((REPEAT1, CEN, 2), dtype=float)

for num_dor in np.arange(len(dor0)):
    (y1, x1) = np.meshgrid(delta[num_dor]*np.arange(-N/2,N/2),
                          delta[num_dor]*np.arange(-N/2,N/2))
    Q = np.exp(-1j*wavenumber/2./f_lenslet[num_dor] * (((x1 + y1) * delta[num_dor])))
    convert_real = ((delta[num_dor]/f_lenslet[num_dor]) # convert pixel drift to angle drift
                    * (delta[num_dor] * N) # convert angle drift to wavefront error
                    * (2*np.pi/wvl)) # convert wavefront error to phase error
    for repeat in np.arange(REPEAT1):
        # flat_slope_x = (np.random.rand() - 0.5)*2 * (N/4) * (np.pi/N/delta[num_dor])
        # flat_slope_y = (np.random.rand() - 0.5)*2 * (N/4) * (np.pi/N/delta[num_dor])
        flat_slope_x = ((repeat+1)/REPEAT1*N/4) / delta[num_dor] / N * np.pi
        flat_slope_y = ((repeat+1)/REPEAT1*N/4) / delta[num_dor] / N * np.pi
        flat_phase = (flat_slope_x*x1 + flat_slope_y*y1)
        U_pupil = np.exp(1j*flat_phase)
        
        # unit is rad per lenslet
        g_slope_flat[repeat,0] = np.mean(flat_phase[1:,:] - flat_phase[:-1,:]) * N
        g_slope_flat[repeat,1] = np.mean(flat_phase[:,1:] - flat_phase[:,:-1]) * N
    
        F0[:,:] = np.pad(U_pupil * Q,
                          (add_each_side0, add_each_side0),
                          'constant', constant_values=0.)
        I_focus = (np.abs(
            np.fft.fftshift(FF01())[edge_low0 : edge_high0,
                                    edge_low0 : edge_high0])**2
            / (2*N**2)**2)

        I_focus = I_focus/I_focus.max()
        # I_focus = I_focus - np.max([I_focus[0,:].max(),
        #                             I_focus[-1,:].max(),
        #                             I_focus[:,0].max(),
        #                             I_focus[:,-1].max()])
        # I_focus = I_focus - Threshold
        # I_focus[np.where(I_focus < 0)] = 0.0
        # I_focus = I_focus/np.sum(I_focus)

        # unit of half fov is pixel! = fov in wvl/d
        for half_fov in (np.arange(CEN) + 1):
            yy, xx = np.meshgrid(np.arange(-half_fov, half_fov) + 0.5,
                                  np.arange(-half_fov, half_fov) + 0.5)
            # x = np.arange(-half_fov, half_fov) + 0.5
    
            I_fov = I_focus[CEN - half_fov : CEN + half_fov,
                            CEN - half_fov : CEN + half_fov]

            # I_fov = I_fov/I_fov.max()
            I_fov = I_fov - np.max([I_fov[0,:].max(),
                                        I_fov[-1,:].max(),
                                        I_fov[:,0].max(),
                                        I_fov[:,-1].max()])
            # I_fov = I_fov - 1e-4
            I_fov[np.where(I_fov < 0)] = 0.0
            I_fov = I_fov/np.sum(I_fov)

            # unit is rad per lenslet
            # real_SHWFS_slope_flat[repeat,half_fov - 1, 0] = (np.sum(I_fov*xx)
            #                                                  / np.sum(I_fov)
            #                                                  * convert_real)
            # real_SHWFS_slope_flat[repeat,half_fov - 1, 1] = (np.sum(I_fov*yy)
            #                                                 / np.sum(I_fov)
            #                                                   * convert_real)
            real_SHWFS_slope_flat[repeat, half_fov - 1, 0] = np.sum(I_fov * xx) / np.sum(I_fov) * convert_real
            real_SHWFS_slope_flat[repeat, half_fov - 1, 1] = np.sum(I_fov * yy) / np.sum(I_fov) * convert_real
    
            dif_SHWFS_slope_flat[repeat,half_fov - 1, : ] = (g_slope_flat[repeat,:]
                                                            - real_SHWFS_slope_flat[repeat,half_fov - 1,:])

# plt.imshow(I_focus[CEN-  4:CEN + 4,CEN - 4:CEN + 4].T)
# plt.xlabel('x = +1px')
# plt.ylabel('y = +2px')
# plt.vlines(3.5,ymin=-0.5,ymax=7.5)
# plt.hlines(3.5,xmin=-0.5,xmax=7.5)
# plt.show()

for repeat in np.arange(REPEAT1):
    plt.plot(np.arange(CEN) + 1,
                  np.abs(dif_SHWFS_slope_flat[repeat,:,0]),
                  label='{:.1f}'.format((repeat+1)/REPEAT1*N/4),marker='.')
plt.legend()
plt.title('Absolute difference between\nSHWFS and g_tilt in flat slope case\nwith threshold for varying centroid shift in px')
plt.xlabel('FoV (wvl/d)')
plt.ylabel('Mean of slope difference (rad)')
plt.xscale('log')
plt.yscale('log')
plt.xlim((1,64))
plt.ylim((1e-15,1e-13))
plt.show()



# do error between shwfs to g-tilt

# next do kolmogorov terbulence
# how about deming fit to the comparison?

