# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 10:13:41 2025

@author: gqmc62
"""

import numpy as np
from matplotlib import pyplot as plt

from aotools.opticalpropagation import angularSpectrum
from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov
from soapy.soapy import interp
import matplotlib as mpl
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


def remove_pistontiptilt(scrn):
    tilt = (scrn[1:]-scrn[:-1]).mean()
    tip = (scrn[:,1:]-scrn[:,:-1]).mean()

    x1 = np.arange(scrn.shape[0])
    yy1 , xx1 = np.meshgrid(x1,x1)

    scrn -= tilt*xx1
    scrn -= tip*yy1
    scrn -= scrn.mean()
    
    return scrn

def perfect_dm(phase,delta,dm_pitch):
    N = phase.shape[0]
    PHASE = np.pad(phase,(N//2,N//2),'symmetric')
    NN = N*2
    del_f = 1./(NN*delta)
    spatial_frequency = np.fft.fftshift(np.fft.fft2(PHASE))
    f = (np.arange(NN) - NN/2)*del_f
    fy, fx = np.meshgrid(f,f)
    FILTER = ((np.abs(fx) < 1./2./dm_pitch) * (np.abs(fy) < 1./2./dm_pitch))
    DM_PHASE = (np.fft.ifft2(np.fft.fftshift(spatial_frequency * FILTER)).real)[N//2:3*N//2,N//2:3*N//2]
    return DM_PHASE

D = 0.4
n_subap_x_ap = 8
wvl = 500e-9
d = D / n_subap_x_ap
pixel_x_subap = 16
delta = d/pixel_x_subap
n1 = int(D/delta)
subap_fov = wvl/2/d*pixel_x_subap

dor0 = 1
r0 = d/dor0
L0 = 2

z = 10e3
c_expand = 4


##############################################################################
# tel --> DM1 --> DM2 --> WFS
# G --> A --> G --> G
##############################################################################


IPS = PhaseScreenKolmogorov(n1, delta, r0, L0)
for row in range(3*n1):
    IPS.add_row()
scrn = IPS.scrn

scrn = remove_pistontiptilt(scrn)
scrn = perfect_dm(scrn, delta, d)
x = np.arange(n1) - n1/2 + 0.5
yy,xx = np.meshgrid(x,x)
scrn[xx**2+yy**2 > (n1/2)**2] = 0



n_HA = int(np.ceil((D + z*(4*wvl/r0 + subap_fov))/delta))
# n_HA = int(np.ceil((D + z*(wvl/delta))/delta))
D_HA = n_HA*delta

N = int(np.max([D/delta, D_HA/delta,
                wvl*z/delta**2,
                D/2/delta+D_HA/2/delta+wvl*z/2/delta**2]))
if N%2 == 1:
    N += 1

pupil_N = np.zeros((N,N),dtype=int)
# pupil_N[(N-n1)//2:(N+n1)//2,(N-n1)//2:(N+n1)//2] = 1
x = np.arange(N) - N/2 + 0.5
yy,xx = np.meshgrid(x,x)
pupil_N[xx**2+yy*2 <= (n1/2)**2] = 1



actGrid = np.zeros((n_subap_x_ap + 1,n_subap_x_ap + 1),dtype=float)
actGrid[n_subap_x_ap//2,n_subap_x_ap//2] = 1

n12n_HA = int((n_HA-n1)//2)

DM = np.pad(interp.zoom_rbs(
        actGrid, n1, order=2),
    ((n12n_HA,n12n_HA),(n12n_HA,n12n_HA)),
    mode='constant', constant_values=0)


n_HA2N = int((N-n_HA)//2)

U_pupil = np.asarray(pupil_N,dtype=complex)

pupil_HA = angularSpectrum(U_pupil, wvl, delta, delta, -z)

HA_U = np.pad(np.exp(1j*DM),
              ((n_HA2N,n_HA2N),(n_HA2N,n_HA2N)),
              mode='constant',constant_values=1)*pupil_HA
G_U = angularSpectrum(HA_U, wvl, delta, delta, z)[(N-n1)//2:(N+n1)//2,(N-n1)//2:(N+n1)//2]

U_WFS = G_U*np.exp(1j*scrn)


undistort_act = G_U
distort_act = U_WFS

no_poke_undistort = np.ones((n1,n1),dtype=complex)
no_poke_distort = np.exp(1j*scrn)

undistort_poke_intensity = np.log(np.abs(undistort_act/no_poke_undistort))
distort_poke_intensity = np.log(np.abs(distort_act/no_poke_distort))

plt.imshow(np.angle(undistort_act/no_poke_undistort))
plt.colorbar()
plt.title('undistort effective poke phase')
plt.show()
plt.imshow(np.angle(distort_act/no_poke_distort))
plt.colorbar()
plt.title('distort effective poke phase')
plt.show()

plt.imshow(undistort_poke_intensity,vmin=-1,vmax=1)
plt.colorbar()
plt.title('undistort effective poke logamp')
plt.show()
plt.imshow(distort_poke_intensity,vmin=-1,vmax=1)
plt.colorbar()
plt.title('distort effective poke logamp')
plt.show()
plt.imshow(distort_poke_intensity - undistort_poke_intensity,vmin=-1,vmax=1)
plt.title('effective poke logamp change by distortion')
plt.colorbar()
plt.show()




##############################################################################
# tel --> DM1 --> DM2 --> WFS
# G --> G --> A --> G
##############################################################################

n_HA = int(np.ceil((D + z*(4*wvl/r0 + subap_fov))/delta))
# n_HA = int(np.ceil((D + z*(wvl/delta))/delta))
D_HA = n_HA*delta

N = int(np.max([D/delta, D_HA/delta,
                wvl*z/delta**2,
                D/2/delta+D_HA/2/delta+wvl*z/2/delta**2]))
if N%2 == 1:
    N += 1

pupil_N = np.zeros((N,N),dtype=int)
# pupil_N[(N-n1)//2:(N+n1)//2,(N-n1)//2:(N+n1)//2] = 1
x = np.arange(N) - N/2 + 0.5
yy,xx = np.meshgrid(x,x)
pupil_N[xx**2+yy**2 <= (n1/2)**2] = 1


IPS = PhaseScreenKolmogorov(n_HA, delta, r0, L0)
for row in range(3*n1):
    IPS.add_row()
scrn = IPS.scrn
scrn = remove_pistontiptilt(scrn)
scrn = perfect_dm(scrn, delta, d)
x = np.arange(n_HA) - n_HA/2 + 0.5
yy,xx = np.meshgrid(x,x)
scrn[xx**2+yy**2 > (n1/2)**2] = 0


actGrid = np.zeros((n_subap_x_ap + 1,n_subap_x_ap + 1),dtype=float)
actGrid[n_subap_x_ap//2,n_subap_x_ap//2] = 1

n12N = int((N-n1)//2)

DM = np.pad(interp.zoom_rbs(
        actGrid, n1, order=2),
    ((n12N,n12N),(n12N,n12N)),
    mode='constant', constant_values=0)

# no distortion

undistort_act = (pupil_N*np.exp(1j*DM))[(N-n1)//2:(N+n1)//2,(N-n1)//2:(N+n1)//2]
no_poke_undistort = pupil_N[(N-n1)//2:(N+n1)//2,(N-n1)//2:(N+n1)//2]

# distort
# poke
U_pupil = np.asarray(pupil_N,dtype=complex)
U_pupil_poke = (pupil_N*np.exp(1j*DM))

U_pupil_poke_at_HA = angularSpectrum(U_pupil_poke, wvl, delta, delta, -z)

HA_U = np.pad(np.exp(1j*scrn),
              ((n_HA2N,n_HA2N),(n_HA2N,n_HA2N)),
              mode='constant',constant_values=1)*U_pupil_poke_at_HA
U_WFS = angularSpectrum(HA_U, wvl, delta, delta, z)[(N-n1)//2:(N+n1)//2,(N-n1)//2:(N+n1)//2]

distort_act = U_WFS

#nopoke
U_pupil = np.asarray(pupil_N,dtype=complex)
U_pupil_nopoke = (pupil_N)

U_pupil_nopoke_at_HA = angularSpectrum(U_pupil_nopoke, wvl, delta, delta, -z)

HA_U = np.pad(np.exp(1j*scrn),
              ((n_HA2N,n_HA2N),(n_HA2N,n_HA2N)),
              mode='constant',constant_values=1)*U_pupil_nopoke_at_HA
U_WFS = angularSpectrum(HA_U, wvl, delta, delta, z)[(N-n1)//2:(N+n1)//2,(N-n1)//2:(N+n1)//2]

no_poke_distort = U_WFS

undistort_poke_intensity = np.log(np.abs(undistort_act/no_poke_undistort))
distort_poke_intensity = np.log(np.abs(distort_act/no_poke_distort))


plt.imshow(np.angle(undistort_act/no_poke_undistort))
plt.colorbar()
plt.title('undistort effective poke phase')
plt.show()
plt.imshow(np.angle(distort_act/no_poke_distort),
           vmin=np.angle(undistort_act/no_poke_undistort).min(),
           vmax=np.angle(undistort_act/no_poke_undistort).max())
plt.colorbar()
plt.title('distort effective poke phase')
plt.show()


plt.imshow(undistort_poke_intensity,vmin=-1,vmax=1)
plt.colorbar()
plt.title('undistort effective poke logamp')
plt.show()
plt.imshow(distort_poke_intensity,vmin=-1,vmax=1)
plt.colorbar()
plt.title('distort effective poke logamp')
plt.show()
plt.imshow(distort_poke_intensity - undistort_poke_intensity,vmin=-1,vmax=1)
plt.title('effective poke logamp change by distortion')
plt.colorbar()
plt.show()



mask = np.asarray(pupil_N[(N-n1)//2:(N+n1)//2,(N-n1)//2:(N+n1)//2],dtype=float)
mask[mask==0] = np.nan

plt.figure(figsize=(4,3))
plt.pcolor(np.arange(128)*delta,np.arange(128)*delta,
           (np.abs(no_poke_distort*mask))**2)
plt.xlabel('m')
plt.ylabel('m')
plt.axis('square')
plt.colorbar()
plt.savefig('dm-scin-induced-5cm-10km.png',dpi=600,bbox_inches='tight')

plt.figure(figsize=(4,3))
plt.pcolor(np.arange(128)*delta,np.arange(128)*delta,scrn[256-64:256+64,256-64:256+64]*mask)
plt.xlabel('m')
plt.ylabel('m')
plt.axis('square')
plt.colorbar()
plt.savefig('dm-apply-phase.png',dpi=600,bbox_inches='tight')

