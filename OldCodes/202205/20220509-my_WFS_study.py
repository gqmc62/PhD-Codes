
###############################################################################
# Start of file ########################################################
###############################################################################
"""Playground for WFS."""
###############################################################################
########################################################################
#
# Imports #
#

import numpy as np
from matplotlib import pyplot as plt

from aotools.turbulence.phasescreen import ft_sh_phase_screen
from aotools.turbulence.phasescreen import ft_phase_screen
from aotools.opticalpropagation import angularSpectrum

import my_propagation as mprop
import my_WFS as mwfs
import my_phasescreen as mps
import time
#
# Preferences #
#

color_map = plt.cm.get_cmap('viridis')
reversed_color_map = color_map.reversed()

###############################################################################
########################################################################
def show_slope_vs_phase(lenslet_slope_phase, phase, plot_radius,
                        d, lenslet_pixel, dz, r, rytov):
    """
    Show how well is our slopes compare to actual phase screen.

    Parameters
    ----------
    lenslet_slope_phase : TYPE
        DESCRIPTION.
    phase : TYPE
        DESCRIPTION.
    plot_radius : TYPE
        DESCRIPTION.
    d : TYPE
        DESCRIPTION.
    lenslet_pixel : TYPE
        DESCRIPTION.
    dz : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.
    rytov : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    slope = np.asarray(lenslet_slope_phase, dtype=float)
    phase = np.asarray(phase, dtype=float)
    prN = int(plot_radius)
    d = float(d)
    lenslet_pixel = int(lenslet_pixel)
    dz = float(dz)
    r = float(r)
    rytov = float(rytov)

    N = len(phase)
    prNN = int(prN/lenslet_pixel)
    num_lenslet = int(N/lenslet_pixel)

    plt.pcolor(np.arange(-prN, prN + 1)*d,
               np.arange(-prN, prN + 1)*d,
               phase[int(N/2) - prN
                     : int(N/2) + prN,
                     int(N/2) - prN
                     : int(N/2) + prN],
               shading='auto', snap=True)
    plt.colorbar()
    plt.vlines((np.arange(-prNN, prNN + 1))*d*lenslet_pixel,
               ymin=-prNN*d*lenslet_pixel,
               ymax=prNN*d*lenslet_pixel,
               color='m')
    plt.hlines((np.arange(-prNN, prNN + 1))*d*lenslet_pixel,
               xmin=-prNN*d*lenslet_pixel,
               xmax=prNN*d*lenslet_pixel,
               color='m')
    plt.quiver((np.arange(-prNN, prNN) + 0.5)*d*lenslet_pixel,
               (np.arange(-prNN, prNN) + 0.5)*d*lenslet_pixel,
               slope[int(num_lenslet/2) - prNN
                     : int(num_lenslet/2) + prNN,
                     int(num_lenslet/2) - prNN
                     : int(num_lenslet/2) + prNN, 0],
               slope[int(num_lenslet/2) - prNN
                     : int(num_lenslet/2) + prNN,
                     int(num_lenslet/2) - prNN
                     : int(num_lenslet/2) + prNN, 1],
               color='r')
    plt.title('Phase (rad) over lenslet phase slope,\n'
              + 'dz=0m, r={:.2f}m, Rytov={:.2f}'.format(r, rytov))
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()
###############################################################################
########################################################################

##

# measure rms between two same screen size

def get_rms2_lenslet(A_pixel, B_lenslet):
    A_pixel = np.asarray(A_pixel)
    B_lenslet = np.asarray(B_lenslet)
    N = len(A_pixel)
    n = len(B_lenslet)
    lenslet_pixel = int(N/n)
    rms = float(0)
    for ilenslet in np.arange(n):
        for jlenslet in np.arange(n):
            for i in np.arange(lenslet_pixel):
                for j in np.arange(lenslet_pixel):
                    ii = i + ilenslet*lenslet_pixel
                    jj = j + ilenslet*lenslet_pixel
                    rms = rms + (A_pixel[ii,jj]
                                 - B_lenslet[ilenslet,jlenslet])**2
    rms = np.sqrt(rms / A_pixel.size)
    return rms

def get_rms3_lenslet(A_pixel, B_lenslet):
    A_pixel = np.asarray(A_pixel)
    B_lenslet = np.asarray(B_lenslet)
    N = len(A_pixel)
    n = len(B_lenslet)
    lenslet_pixel = int(N/n)
    rms = float(0)
    for ilenslet in np.arange(n):
        for jlenslet in np.arange(n):
            for i in np.arange(lenslet_pixel):
                for j in np.arange(lenslet_pixel):
                    ii = i + ilenslet*lenslet_pixel
                    jj = j + jlenslet*lenslet_pixel
                    for k in np.arange(A_pixel.shape[2]):
                        rms = rms + (A_pixel[ii,jj,k]
                                     - B_lenslet[ilenslet,jlenslet,k])**2
    rms = np.sqrt(rms / A_pixel.size)
    return rms

def get_dif_lenslet(A_pixel, B_lenslet):
    A_pixel = np.asarray(A_pixel)
    B_lenslet = np.asarray(B_lenslet)
    N = len(A_pixel)
    n = len(B_lenslet)
    lenslet_pixel = int(N/n)
    C_pixel = np.zeros((N,N),dtype=type(A_pixel[0,0]))
    for ilenslet in np.arange(n):
        for jlenslet in np.arange(n):
            for i in np.arange(lenslet_pixel):
                for j in np.arange(lenslet_pixel):
                    ii = i + ilenslet*lenslet_pixel
                    jj = j + jlenslet*lenslet_pixel
                    C_pixel[ii,jj] = (A_pixel[ii,jj]
                                 - B_lenslet[ilenslet,jlenslet])
    return C_pixel

def get_PSD(phase, delta):
    N = len(phase)
    del_f = 1./(N*delta)
    phase = phase - phase.mean()
    cn = my_ft2(phase, 1./N)
    PSD = (np.abs(cn)**2/del_f**2)*2.   # * 2. is from the fact that
                                        # the input phase is not complex
                                        # phase are complex when generated
                                        # from cn.
    PSD[int(N/2),int(N/2)] = 0.         # PSD is 0 at the centre.
    return PSD

def PSD_to_phase(PSD, delta_f):
    cn = ((np.random.normal(size=(len(PSD),len(PSD)))
           + 1j*np.random.normal(size=(len(PSD),len(PSD))))
          * np.sqrt(PSD*delta_f**2))
    phase = my_ift2(cn, 1).real
    return phase

def my_ft2(g, delta):
    G = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(g))) * delta**2
    return G

def my_ift2(G, delta_f):
    N = len(G)
    g = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(G))) * (N*delta_f)**2
    return g

def auto_pad2(A, pf):
    AA = np.zeros((A.shape[0]*pf, A.shape[1]*pf), dtype=type(A[0,0]))
    for ipf in np.arange(pf):
        for jpf in np.arange(pf):
            for i in np.arange(int(A.shape[0])):
                for j in np.arange(int(A.shape[1])):
                    row = int(ipf*A.shape[0] + i + A.shape[0]/2)%(A.shape[0]*pf)
                    col = int(jpf*A.shape[1] + j + A.shape[1]/2)%(A.shape[1]*pf)
                    AA[row,col]=A[i,j]
    return AA

def mirror_pad2(A):
    ROW = A.shape[0]
    COL = A.shape[1]
    row = int(ROW/2.)
    col = int(COL/2.)
    AA = np.zeros((ROW*2,COL*2), dtype=type(A[0,0]))
    for i in np.arange(ROW):
        for j in np.arange(COL):
            AA[row+i,col+j]=A[i,j]#correct
    for i in np.arange(row):
        for j in np.arange(col):
            AA[row-i-1,col-j-1]=A[i,j]#correct
            AA[ROW+row+i,COL+col+j]=A[ROW-1-i,COL-1-j]#correct
            AA[row-i-1,COL+col+j]=A[i,COL-1-j]#correct
            AA[ROW+row+i,col-j-1]=A[ROW-i-1,j]#correct
    for i in np.arange(ROW):
        for j in np.arange(col):
            AA[row+i,col-j-1]=A[i,j]#correct
            AA[row+i,COL+col+j]=A[i,COL-j-1]#correct
    for i in np.arange(row):
        for j in np.arange(COL):
            AA[row-i-1,col+j]=A[i,j]#correct
            AA[row+ROW+i,col+j]=A[ROW-i-1,j]
    return AA

def outcrop2(AA, pf):
    ROW=AA.shape[0]
    COL=AA.shape[1]
    row=int(ROW/pf)
    col=int(COL/pf)
    r1=int(row/2.)
    r2=int(ROW-row/2.)
    c1=int(col/2.)
    c2=int(COL-col/2.)
    A=np.zeros((row,col),dtype=type(AA[0,0]))
    for i in np.arange(r1):
        for j in np.arange(c1):
            A[i,j]=AA[i,j]
            A[i+r1,j+c1]=AA[i+r2,j+c2]
            A[i,j+c1]=AA[i,j+c2]
            A[i+r1,j]=AA[i+r2,j]
    return A

def pick_even_index(AA):
    ROW=AA.shape[0]
    COL=AA.shape[1]
    row=int(ROW/2)
    col=int(COL/2)
    A=np.zeros((row,col),dtype=type(AA[0,0]))
    for i in np.arange(row):
        for j in np.arange(col):
            A[i,j]=AA[2*i,2*j]
    return A

# not good
def make_dm_edge(lenslet_slope_phase,lenslet_pixel,pixel_scale):
    N=lenslet_pixel*len(lenslet_slope_phase)
    x=np.arange(-lenslet_pixel/2,lenslet_pixel/2) * pixel_scale
    x1,y1=np.meshgrid(x,x)
    dm=np.zeros((N,N),dtype=float)
    lenslet_pixel=int(lenslet_pixel)
    nl=int(np.floor(N/lenslet_pixel))
    piston = 0
    for i in np.arange(nl):
        for j in np.arange(nl):
            tiptilt = (((x1*lenslet_slope_phase[i,j,0]
                      + y1*lenslet_slope_phase[i,j,1]) * (2.*np.pi/wvl)))
            if ((i != 0) and (j != 0)):
                neighbor_edge = (dm[i*lenslet_pixel-1,
                             j*lenslet_pixel:(j+1)*lenslet_pixel].mean()
                          + dm[i*lenslet_pixel:(i+1)*lenslet_pixel,
                               j*lenslet_pixel-1].mean()) / 2.
                my_edge = (tiptilt[0,:] + tiptilt[:,0])*0.5
            else:
                if ((i != 0) and (j == 0)):
                    neighbor_edge = dm[i*lenslet_pixel-1,
                                j*lenslet_pixel:(j+1)*lenslet_pixel].mean()
                    my_edge = (tiptilt[0,:])
                else:
                    if ((i == 0) and (j != 0)):
                        neighbor_edge = dm[i*lenslet_pixel:(i+1)*lenslet_pixel,
                                    j*lenslet_pixel-1].mean()
                        my_edge = (tiptilt[:,0])
                    else:
                        neighbor_edge = 0.
                        my_edge = 0.
            piston = neighbor_edge - my_edge
            correct = tiptilt + piston
            dm[i*lenslet_pixel:(i+1)*lenslet_pixel,
               j*lenslet_pixel:(j+1)*lenslet_pixel] = correct
    return dm-dm.mean()

# best if unwrap phase is given
def make_dm_perfect_piston(phase_unwrap,lenslet_slope_phase,lenslet_pixel,pixel_scale):
    N=len(phase_unwrap)
    x=np.arange(-lenslet_pixel/2,lenslet_pixel/2) * pixel_scale
    x1,y1=np.meshgrid(x,x)
    dm=np.zeros((N,N),dtype=float)
    lenslet_pixel=int(lenslet_pixel)
    nl=int(np.floor(N/lenslet_pixel))
    for i in np.arange(nl):
        for j in np.arange(nl):
            piston=phase_unwrap[i*lenslet_pixel:(i+1)*lenslet_pixel,
                            j*lenslet_pixel:(j+1)*lenslet_pixel].mean()
            correct=(((x1*lenslet_slope_phase[i,j,0]
                      + y1*lenslet_slope_phase[i,j,1]) * (2.*np.pi/wvl))
            + piston)
            dm[i*lenslet_pixel:(i+1)*lenslet_pixel,
               j*lenslet_pixel:(j+1)*lenslet_pixel]=correct
    return np.angle(np.exp(1j*dm))

# not good if the lenslet is close to half-half on the wrapping phsae
def make_dm_perfect_piston2(phase,lenslet_slope_phase,lenslet_pixel,pixel_scale):
    N=len(phase)
    x=np.arange(-lenslet_pixel/2,lenslet_pixel/2) * pixel_scale
    x1,y1=np.meshgrid(x,x)
    dm=np.zeros((N,N),dtype=float)
    lenslet_pixel=int(lenslet_pixel)
    nl=int(np.floor(N/lenslet_pixel))
    for i in np.arange(nl):
        for j in np.arange(nl):
            tiptilt=(((x1*lenslet_slope_phase[i,j,0]
                      + y1*lenslet_slope_phase[i,j,1]) * (2.*np.pi/wvl)))
            piston=dif_unwrap(phase[i*lenslet_pixel:(i+1)*lenslet_pixel,
                                    j*lenslet_pixel:(j+1)*lenslet_pixel],
                                  tiptilt).mean()
            correct=tiptilt+piston
            dm[i*lenslet_pixel:(i+1)*lenslet_pixel,
               j*lenslet_pixel:(j+1)*lenslet_pixel]=correct
    return np.angle(np.exp(1j*dm))

#promising but no
def make_dm_perfect_piston3(phase,lenslet_slope_phase,lenslet_pixel,pixel_scale):
    N=len(phase)
    x=np.arange(-lenslet_pixel/2,lenslet_pixel/2) * pixel_scale
    x1,y1=np.meshgrid(x,x)
    dm=np.zeros((N,N),dtype=float)
    lenslet_pixel=int(lenslet_pixel)
    nl=int(np.floor(N/lenslet_pixel))
    for i in np.arange(nl):
        for j in np.arange(nl):
            tiptilt=(((x1*lenslet_slope_phase[i,j,0]
                      + y1*lenslet_slope_phase[i,j,1]) * (2.*np.pi/wvl)))
            piston=(phase[int(i + 0.5)*lenslet_pixel,
                          int(j + 0.5)*lenslet_pixel]
                    - tiptilt[int(0.5*lenslet_pixel),
                              int(0.5*lenslet_pixel)])
            correct = tiptilt + piston
            dm[i*lenslet_pixel:(i+1)*lenslet_pixel,
               j*lenslet_pixel:(j+1)*lenslet_pixel]=correct
    return np.angle(np.exp(1j*dm))

#looks great but wrong global tip/tilt
#perhaps an upgrad of this works
def make_dm_corner(lenslet_slope_phase,lenslet_pixel,pixel_scale):
    N=lenslet_pixel*len(lenslet_slope_phase)
    x=np.arange(-lenslet_pixel/2,lenslet_pixel/2) * pixel_scale
    x1,y1=np.meshgrid(x,x)
    dm=np.zeros((N,N),dtype=float)
    lenslet_pixel=int(lenslet_pixel)
    nl=int(np.floor(N/lenslet_pixel))
    piston = 0
    for i in np.arange(nl):
        for j in np.arange(nl):
            tiptilt = (((x1*lenslet_slope_phase[i,j,0]
                      + y1*lenslet_slope_phase[i,j,1]) * (2.*np.pi/wvl)))
            if ((i != 0) and (j != 0)):
                neighbor_corner = (dm[i*lenslet_pixel-1,j*lenslet_pixel]
                                   +dm[i*lenslet_pixel,j*lenslet_pixel-1]
                                   +dm[i*lenslet_pixel-1,j*lenslet_pixel-1])/3.
            else:
                if ((i != 0) and (j == 0)):
                    neighbor_corner = (dm[i*lenslet_pixel-1,j*lenslet_pixel])
                else:
                    if ((i == 0) and (j != 0)):
                        neighbor_corner = (dm[i*lenslet_pixel,j*lenslet_pixel-1])
                    else:
                        neighbor_corner = 0.
            my_corner = tiptilt[0,0]
            piston = neighbor_corner - my_corner
            correct = tiptilt + piston
            dm[i*lenslet_pixel:(i+1)*lenslet_pixel,
               j*lenslet_pixel:(j+1)*lenslet_pixel] = correct
    return np.angle(np.exp(1j*(dm-dm.mean())))

#not work if the center is split
def make_dm_perfect_corners(phase,lenslet_slope_phase,lenslet_pixel,pixel_scale):
    N=lenslet_pixel*len(lenslet_slope_phase)
    x=np.arange(-lenslet_pixel/2,lenslet_pixel/2) * pixel_scale
    x1,y1=np.meshgrid(x,x)
    dm=np.zeros((N,N),dtype=float)
    lenslet_pixel=int(lenslet_pixel)
    nl=int(np.floor(N/lenslet_pixel))
    piston = 0
    for i in np.arange(nl):
        for j in np.arange(nl):
            tiptilt = (((x1*lenslet_slope_phase[i,j,0]
                      + y1*lenslet_slope_phase[i,j,1]) * (2.*np.pi/wvl)))
            neighbor_corners = (phase[i*lenslet_pixel,
                                      j*lenslet_pixel]
                                + phase[(i+1)*lenslet_pixel - 1,
                                        j*lenslet_pixel]
                                + phase[i*lenslet_pixel,
                                        (j+1)*lenslet_pixel - 1]
                                + phase[(i+1)*lenslet_pixel - 1,
                                        (j+1)*lenslet_pixel - 1]) / 4.
            my_corners = (tiptilt[0,0] + tiptilt[0,-1]
                          + tiptilt[-1,0] + tiptilt[-1,-1]) / 4.
            piston = neighbor_corners - my_corners
            correct = tiptilt + piston
            dm[i*lenslet_pixel:(i+1)*lenslet_pixel,
               j*lenslet_pixel:(j+1)*lenslet_pixel] = correct
    return np.angle(np.exp(1j*(dm)))

#not as great as promised
def make_dm_perfect_corner(phase,lenslet_slope_phase,lenslet_pixel,pixel_scale):
    N=lenslet_pixel*len(lenslet_slope_phase)
    x=(np.arange(-lenslet_pixel/2,lenslet_pixel/2) + 0.5) * pixel_scale
    x1,y1=np.meshgrid(x,x)
    dm=np.zeros((N,N),dtype=float)
    lenslet_pixel=int(lenslet_pixel)
    nl=int(np.floor(N/lenslet_pixel))
    piston = 0
    for i in np.arange(nl):
        for j in np.arange(nl):
            tiptilt = (((x1*lenslet_slope_phase[i,j,0]
                      + y1*lenslet_slope_phase[i,j,1]) * (2.*np.pi/wvl)))
            if ((i != 0) and (j != 0)):
                neighbor_corner = (phase[i*lenslet_pixel-1,j*lenslet_pixel]
                                   +phase[i*lenslet_pixel,j*lenslet_pixel-1]
                                   +phase[i*lenslet_pixel-1,j*lenslet_pixel-1])/3.
            else:
                if ((i != 0) and (j == 0)):
                    neighbor_corner = (phase[i*lenslet_pixel-1,j*lenslet_pixel])
                else:
                    if ((i == 0) and (j != 0)):
                        neighbor_corner = (phase[i*lenslet_pixel,j*lenslet_pixel-1])
                    else:
                        neighbor_corner = phase[0,0]
            my_corner = tiptilt[0,0]
            piston = neighbor_corner - my_corner
            correct = tiptilt + piston
            dm[i*lenslet_pixel:(i+1)*lenslet_pixel,
               j*lenslet_pixel:(j+1)*lenslet_pixel] = correct
    return np.angle(np.exp(1j*(dm)))

def dif_unwrap(A,B):
    C = np.zeros_like(A)
    N=len(A)
    period=np.pi*2.
    temp=period/2.
    for i in np.arange(N):
        for j in np.arange(N):
            dif=A[i,j]-B[i,j]
            if np.abs(dif) < temp:
                C[i,j]=dif
            else:
                if dif>0:
                    C[i,j]=dif-period
                else:
                    C[i,j]=dif+period
    return (C)

def simple_dm(lenslet_slope_phase,lenslet_pixel,pixel_scale):
    N=lenslet_pixel*len(lenslet_slope_phase)
    x=(np.arange(-lenslet_pixel/2,lenslet_pixel/2) + 0.5) * pixel_scale
    x1,y1=np.meshgrid(x,x)
    dm=np.zeros((N,N),dtype=float)
    lenslet_pixel=int(lenslet_pixel)
    nl=int(np.floor(N/lenslet_pixel))
    for i in np.arange(nl):
        for j in np.arange(nl):
            tiptilt = (((x1*lenslet_slope_phase[i,j,0]
                      + y1*lenslet_slope_phase[i,j,1]) * (2.*np.pi/wvl)))
            dm[i*lenslet_pixel:(i+1)*lenslet_pixel,
               j*lenslet_pixel:(j+1)*lenslet_pixel] = tiptilt
    return np.angle(np.exp(1j*(dm)))

# measure rms between two screen with different size

##

TA = time.time()

N = 1024 #1024
d_in = 3e-3#5e-3#want 1e-3
L0 = 20
l0 = 1.5e-10
wvl = 500e-9
d_out = d_in
lenslet_pixel = 16
D_lenslet = 150e-6
SNR = 100
#z = np.array([5,10,25], dtype=float)*1e3
#r = np.array([0.3,0.2,0.1,0.05], dtype=float)
#z = np.array([1,2,5,10,25], dtype=float)*1e3
#r = np.array([0.3,0.2,0.13,0.08,0.05] ,dtype=float)

#z = np.array([0,5,10,15], dtype=float)*1e3
#r = np.array([0.07,0.10,0.19,0.19] ,dtype=float)

#z = np.array([0,5,10,15,20,25], dtype=float)*1e3
#r = np.array([0.07,0.11,0.17,0.27,0.40,0.40] ,dtype=float)

z = np.array([0.1,0.2,0.5,1,2,5,10,25], dtype=float)*1e3
r = np.array([0.3,0.2,0.13,0.08,0.05] ,dtype=float)

fov_wfs = 5 # arcsecond # diameter of the field

#
# Independent constants
#
pf_lenslet = 8
pf = 4
prN = 128 # 128
I_max = 0.3
I_min = -1.
paramdict={'N':N,'d_in':d_in,'L0':L0,'l0':l0,'wvl':wvl,'d_out':d_out,
           'lenslet_pixel':lenslet_pixel,'D_lenslet':D_lenslet,'z':z,'r':r,
           'fov_wfs':fov_wfs,'pf_lenslet':pf_lenslet,'pf':pf,'SNR':SNR}

#
# Dependent constants
#
fov_wfs = fov_wfs/3600.*2.*np.pi/360. # converted to rad
num_lenslet = int(N/lenslet_pixel)
d_lenslet = D_lenslet/lenslet_pixel
# f_lenslet = 2. * d_lenslet**2 * lenslet_pixel / wvl
f_lenslet = D_lenslet / fov_wfs
NN = pf*N
prNN = int(prN/lenslet_pixel)
I_norm = lenslet_pixel**2./4.
# d_detector = d_lenslet
d_detector = wvl*f_lenslet/2./D_lenslet
#k = 2.*np.pi/wvl                    # float; rad/m; Wave number.
#cn2dz=r**(-5./3.)/0.423/k**2        # float; Turbulence strength.

#
# Checking propagation parametres #
#

# # Propagation between atmospheric layers
check = True
check_condition = mprop.prop_param_check(N, d_in, d_out, wvl, z, pf,
                                        scin_blob=True)#True)#True
for i in check_condition:
    check = check * i
if check == False:
    print("One of the propagation condition for atm layer is failing.")
    #quit

# # Propagation through SHWFS
check = True
check_condition = mprop.prop_param_check(lenslet_pixel, d_lenslet, d_detector,
                                        wvl, [f_lenslet], pf_lenslet)
for i in check_condition:
    check = check * i
if check == False:
    print("One of the propagation condition for lenslet is failing.")
    #quit

#
# Declaration field
#
I_in_big = np.ones((NN,NN), dtype=float)          # Input amplitude.
phase_in_big = np.zeros((NN, NN), dtype=float)       # Input phase (rad).
U_in_big = np.zeros((NN,NN), dtype=complex)         # Input complex amp.

I_in = np.ones((N,N), dtype=float)          # Input amplitude.
phase_in = np.zeros((N, N), dtype=float)       # Input phase (rad).
U_in = np.zeros((N,N), dtype=complex)         # Input complex amp.

slope_phase_in = np.zeros((N, N, 2), dtype=float)

U_out_big = np.zeros((NN,NN), dtype=complex)         # Input complex amp.

I_out = np.ones((N,N), dtype=float)
phase_out = np.zeros((N, N), dtype=float)       # Input phase (rad).
U_out = np.zeros((N,N), dtype=complex)         # Input complex amp.

slope_phase_out = np.ones((N,N, 2), dtype=float)

#
# Calculations
#

# curiosity
rms_sky_to_ground = np.zeros((len(r),len(z)), dtype=float)

# control group / measure at altitude
# this is what we always do in simulation sense.
rms_sky_to_sky_avg = np.zeros((len(r)),dtype=float)###
rms_sky_to_sky_WFS = np.zeros((len(r)), dtype=float)###

# measure at ground
# show precision of measurement at ground
rms_ground_to_ground_avg = np.zeros((len(r),len(z)), dtype=float)
rms_ground_to_ground_WFS = np.zeros((len(r),len(z)), dtype=float)
rms_ground_to_ground_weight = np.zeros((len(r),len(z)), dtype=float)

# use ground measurement as sky shape / measure at ground assume = sky then fit dm
# this is actually what we have always do in practical sense
rms_sky_to_ground_avg = np.zeros((len(r),len(z)), dtype=float)
rms_sky_to_ground_WFS = np.zeros((len(r),len(z)), dtype=float)
rms_sky_to_ground_weight = np.zeros((len(r),len(z)), dtype=float)

# compare wfs and wfs approx
rms_ground_WFS_to_ground_weight = np.zeros((len(r),len(z)), dtype=float)

# compare best dm shape with wfs
rms_sky_avg_to_sky_WFS = np.zeros((len(r)), dtype=float)###
rms_ground_avg_to_ground_WFS = np.zeros((len(r),len(z)), dtype=float)
rms_ground_avg_to_ground_weight = np.zeros((len(r),len(z)), dtype=float)


all_rytov_pixel = np.zeros((len(r),len(z)), dtype=float)
# all_rytov_lenslet = np.zeros((len(r),len(z)), dtype=float)

# std_slope_in = np.zeros((len(r)), dtype=float)
# std_slope_out = np.zeros((len(r),len(z)), dtype=float)

for ir in np.arange(len(r)):
    # Initialization #
    ta = time.time()
    phase_in_big = ft_sh_phase_screen(r[ir], NN, d_in, L0, l0, seed=99)
    tb = time.time()
    t_make_screen = tb - ta
    print('time make screen = {:.6f}s'.format(t_make_screen))
    
    # Check screen condition #
    # ta = time.time()
    # check_scrn = mwfs.check_screen(phase_in_big, d_in, wvl, d_out, z, pf)
    # if check_scrn == False:
    #     print('Input phase screen is not Nyquist sampled.')
    #     print('It is too dense. aliasing happen')
    #     print('r={:.2f}m, d_in={:.2f}m'.format(r[ir], d_in))
    #     print('For this r, d_in must be smaller.')
    #     #quit
    # tb = time.time()
    # t_check_screen = tb - ta
    # print('time check screen = {:.6f}s'.format(t_check_screen))

    U_in_big = np.sqrt(I_in_big)*np.exp(1j*phase_in_big)
    
    # tempscrn = angularSpectrum(U_in_big, wvl, d_in, d_out, z.max())
    # U_in_big = angularSpectrum(tempscrn, wvl, d_out, d_in, -z.max())

    #
    # Reporting source conditions #
    #
    ta = time.time()
    U_in = mprop.crop2(U_in_big, pf)
    phase_in = np.angle(U_in)

    I_in = np.abs(U_in)**2
    slope_phase_in = mwfs.get_slope(phase_in, d_in, wvl)
    # std_slope_in[ir] = slope_phase_in.std()
    # size_slope_phase_in = np.sqrt(slope_phase_in[:,:,0]**2
    #                               + slope_phase_in[:,:,1]**2)
    tb = time.time()
    t_analyze_input_px = tb - ta
    print('time analyze input pixely = {:.6f}s'.format(t_analyze_input_px))
    
    # ta = time.time()
    phase_in_unwrap = np.unwrap(np.unwrap(
        phase_in, period=2*np.pi, axis=-2),
        period=2*np.pi, axis=-1)
    # plt.imshow(phase_in[:16,:16])
    # plt.title('Sampled phase in - looped')
    # plt.colorbar()
    # plt.show()
    # plt.imshow(phase_in_unwrap[:16,:16])
    # plt.title('Sampled phase in unwrap')
    # plt.colorbar()
    # plt.show()
    # PSD_in = pick_even_index(get_PSD(mirror_pad2(phase_in_unwrap), d_in))
    # PSD = mps.psd_function(r[ir], N, d_in, L0, l0)
    # tb = time.time()
    # t_PSD_in = tb - ta
    # print('time making PSD / measuring PSD in = {:.6f}s'.format(t_PSD_in))
    
    #
    # Average #
    #
    ta = time.time()
    lenslet_I_in = mwfs.get_val_lenslet(I_in, lenslet_pixel)
    lenslet_slope_phase_in = (mwfs.get_vec_lenslet(slope_phase_in,
                                                    lenslet_pixel))
    # size_lenslet_slope_phase_in = np.sqrt(
    #     lenslet_slope_phase_in[ : , : , 0]**2
    #     + lenslet_slope_phase_in[ : , : , 1]**2)
    # rytov_lenslet_in = ((np.mean(lenslet_I_in**2) 
    #                       / np.mean(lenslet_I_in)**2) - 1.) / 4.
    tb = time.time()
    t_analyze_input_lenslet = tb - ta
    print('time anaylze input lensletly = {:.6f}s'.format(t_analyze_input_lenslet))

    #
    # SHWFS cheat by weighting slope with intensity #
    #
    ta = time.time()
    weight_slope_phase_in = np.zeros((N,N,2), dtype=float)
    weight_slope_phase_in[:,:,0] = slope_phase_in[:,:,0]*I_in
    weight_slope_phase_in[:,:,1] = slope_phase_in[:,:,1]*I_in
    lenslet_weight_slope_phase_in = mwfs.get_vec_lenslet(
        weight_slope_phase_in, lenslet_pixel)
    lenslet_weight_slope_phase_in[:,:,0] = (
        lenslet_weight_slope_phase_in[:,:,0]/lenslet_I_in)
    lenslet_weight_slope_phase_in[:,:,1] = (
        lenslet_weight_slope_phase_in[:,:,1]/lenslet_I_in)
    # size_lenslet_weight_slope_phase_in = np.sqrt(
    #     lenslet_weight_slope_phase_in[ : , : , 0]**2
    #     + lenslet_weight_slope_phase_in[ : , : , 1]**2)
    tb = time.time()
    t_analyze_input_weight = tb - ta
    print('time analyze input weight = {:.6f}s'.format(t_analyze_input_weight))

    #
    # Proper SHWFS #
    #
    ta = time.time()
    U_WFS_detector_in = mwfs.propagate_lenslet(U_in, wvl, d_lenslet,
                                                f_lenslet, lenslet_pixel,
                                                d_detector, pf_lenslet,)
    I_WFS_detector_in = (np.abs(U_WFS_detector_in)**2
                         * (d_detector/d_lenslet)**2)
    lenslet_I_WFS_detector_in = mwfs.get_val_lenslet(
        I_WFS_detector_in, lenslet_pixel)
    noisy_I_WFS_detector_in = np.random.poisson(
        I_WFS_detector_in/lenslet_pixel**2*SNR**2) / SNR**2*lenslet_pixel**2
    slope_WFS_in = mwfs.get_SHWFS_slope(noisy_I_WFS_detector_in, lenslet_pixel,
                                    d_detector, f_lenslet)
    projected_slope_WFS_in = slope_WFS_in * d_lenslet / d_in
    # size_projected_slope_WFS_in = np.sqrt(projected_slope_WFS_in[:,:,0]**2
    #                                       + projected_slope_WFS_in[:,:,1]**2)
    # rytov_WFS_in = (
    #     (np.mean(lenslet_I_WFS_detector_in**2) / np.mean(
    #         lenslet_I_WFS_detector_in)**2) - 1.) / 4.
    tb = time.time()
    t_analyze_input_WFS = tb - ta
    print('time to analyze WFS in = {:.6f}s'.format(t_analyze_input_WFS))

    # make perfect dm
    ta = time.time()
    # dm_in_avg = make_dm_perfect_piston3(phase_in_unwrap,lenslet_slope_phase_in, lenslet_pixel, d_in)
    # dm_in_WFS = make_dm_perfect_piston3(phase_in_unwrap,projected_slope_WFS_in, lenslet_pixel, d_in)
    # dm_in_avg = make_dm_perfect_piston2(phase_in_unwrap,lenslet_slope_phase_in, lenslet_pixel, d_in)
    # dm_in_WFS = make_dm_perfect_piston2(phase_in_unwrap,projected_slope_WFS_in, lenslet_pixel, d_in)
    # dm_in_avg = make_dm_perfect_piston(phase_in_unwrap,lenslet_slope_phase_in, lenslet_pixel, d_in)
    # dm_in_WFS = make_dm_perfect_piston(phase_in_unwrap,projected_slope_WFS_in, lenslet_pixel, d_in)
    # dm_in_avg = make_dm_corner(lenslet_slope_phase_in, lenslet_pixel, d_in)
    # dm_in_WFS = make_dm_corner(projected_slope_WFS_in, lenslet_pixel, d_in)
    dm_in_avg = simple_dm(lenslet_slope_phase_in[:1,:1,:], lenslet_pixel, d_in)
    dm_in_WFS = simple_dm(projected_slope_WFS_in[:1,:1,:], lenslet_pixel, d_in)
    res_in_avg = dif_unwrap(phase_in[:lenslet_pixel,:lenslet_pixel], dm_in_avg)
    res_in_avg = res_in_avg-res_in_avg.mean()
    res_in_WFS = dif_unwrap(phase_in[:lenslet_pixel,:lenslet_pixel], dm_in_WFS)
    res_in_WFS = res_in_avg-res_in_WFS.mean()
    tb = time.time()
    t_dm = tb - ta
    print('time make perfect dms = {:.6f}s'.format(t_dm))

    # analysis rms in
    ta = time.time()
    rms_sky_to_sky_avg[ir] = res_in_avg.std()
    rms_sky_to_sky_WFS[ir] = res_in_WFS.std()
    rms_sky_avg_to_sky_WFS[ir] = dif_unwrap(dm_in_avg, dm_in_WFS).std()
    # rms_sky_to_sky_avg[ir] = get_rms3_lenslet(slope_phase_in,
    #                                           lenslet_slope_phase_in)
    # rms_sky_to_sky_WFS[ir] = get_rms3_lenslet(slope_phase_in,
    #                                           projected_slope_WFS_in)
    # rms_sky_to_sky_weight[ir] = get_rms3_lenslet(slope_phase_in,
    #                                              lenslet_weight_slope_phase_in)
    # rms_sky_avg_to_sky_WFS[ir] = (lenslet_slope_phase_in
    #                               - projected_slope_WFS_in).std()
    # rms_sky_avg_to_sky_weight[ir] = (lenslet_slope_phase_in
    #                                  - lenslet_weight_slope_phase_in).std()
    # rms_sky_WFS_to_sky_weight[ir] = (projected_slope_WFS_in
    #                                  - lenslet_weight_slope_phase_in).std()
    tb = time.time()
    t_analyze_rms_in = tb - ta
    print('time to analyze rms input = {:.6f}s'.format(t_analyze_rms_in))

    # #
    # # Plot input analysis #
    # #
    # ta = time.time()
    # # Actual value
    # plt.pcolor(np.arange(-prN, prN + 1)*d_in,
    #             np.arange(-prN, prN + 1)*d_in,
    #             phase_in[int(N/2) - prN
    #                               : int(N/2) + prN,
    #                               int(N/2) - prN
    #                               : int(N/2) + prN],
    #             shading='auto', snap=True)
    # plt.colorbar()
    # plt.vlines((np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
    #             ymin=-prNN*d_in*lenslet_pixel,
    #             ymax=prNN*d_in*lenslet_pixel,
    #             color='m')
    # plt.hlines((np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
    #             xmin=-prNN*d_in*lenslet_pixel,
    #             xmax=prNN*d_in*lenslet_pixel,
    #             color='m')
    # plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
    #             (np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
    #             lenslet_slope_phase_in[int(num_lenslet/2) - prNN
    #                                     : int(num_lenslet/2) + prNN,
    #                                     int(num_lenslet/2) - prNN
    #                                     : int(num_lenslet/2) + prNN, 0],
    #             lenslet_slope_phase_in[int(num_lenslet/2) - prNN
    #                                     : int(num_lenslet/2) + prNN,
    #                                     int(num_lenslet/2) - prNN
    #                                     : int(num_lenslet/2) + prNN, 1],
    #             color='r')
    # plt.title('Phase (rad) over lenslet phase slope,\n'
    #           + 'dz=0m, r={:.2f}m, Rytov=0'.format(r[ir]))
    # plt.axis('equal')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    # plt.show()
    
    plt.pcolor(np.arange(-8,9)*d_in,
                np.arange(-8,9)*d_in,
                phase_in[:16,:16],
                shading='auto', snap=True)
    plt.colorbar()
    plt.vlines((np.arange(-0.5,1.5))*d_in*lenslet_pixel,
                ymin=-0.5*d_in*lenslet_pixel,
                ymax=0.5*d_in*lenslet_pixel,
                color='m')
    plt.hlines((np.arange(-0.5,1.5))*d_in*lenslet_pixel,
                xmin=-0.5*d_in*lenslet_pixel,
                xmax=0.5*d_in*lenslet_pixel,
                color='m')
    plt.quiver((np.arange(1))*d_in*lenslet_pixel,
                (np.arange(1))*d_in*lenslet_pixel,
                lenslet_slope_phase_in[:1,:1,0],
                lenslet_slope_phase_in[:1,:1,1],
                color='r')
    plt.title('Phase (rad) over lenslet phase slope,\n'
              + 'dz=0m, r={:.2f}m, Rytov=0'.format(r[ir]))
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()

    # # # Average
    # # plt.pcolor((np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
    # #             (np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
    # #             size_lenslet_slope_phase_in[int(num_lenslet/2) - prNN
    # #                                         : int(num_lenslet/2) + prNN,
    # #                                         int(num_lenslet/2) - prNN
    # #                                         : int(num_lenslet/2) + prNN],
    # #             shading='auto', snap=True)
    # # plt.colorbar()
    # # plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
    # #             (np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
    # #             lenslet_slope_phase_in[int(num_lenslet/2) - prNN
    # #                                     : int(num_lenslet/2) + prNN,
    # #                                     int(num_lenslet/2) - prNN
    # #                                     : int(num_lenslet/2) + prNN, 0],
    # #             lenslet_slope_phase_in[int(num_lenslet/2) - prNN
    # #                                     : int(num_lenslet/2) + prNN,
    # #                                     int(num_lenslet/2) - prNN
    # #                                     : int(num_lenslet/2) + prNN, 1],
    # #             color='r')
    # # plt.axis('equal')
    # # plt.xlabel('x (m)')
    # # plt.ylabel('y (m)')
    # # plt.title('Lenslet phase slope size (rad),\n' 
    # #           + 'dz=0m, r={:.2f}m,\n'.format(r[ir])
    # #           + 'Rytov={:.2f}, lenslet Rytov={:.2f}'.format(
    # #               rytov_in, rytov_lenslet_in))
    # # plt.show()

    # # # cheat SHWFS #
    # # plt.pcolor((np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
    # #             (np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
    # #             size_lenslet_weight_slope_phase_in[int(num_lenslet/2) - prNN
    # #                                         : int(num_lenslet/2) + prNN,
    # #                                         int(num_lenslet/2) - prNN
    # #                                         : int(num_lenslet/2) + prNN],
    # #             shading='auto', snap=True)
    # # plt.colorbar()
    # # plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
    # #             (np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
    # #             lenslet_weight_slope_phase_in[int(num_lenslet/2) - prNN
    # #                                     : int(num_lenslet/2) + prNN,
    # #                                     int(num_lenslet/2) - prNN
    # #                                     : int(num_lenslet/2) + prNN, 0],
    # #             lenslet_weight_slope_phase_in[int(num_lenslet/2) - prNN
    # #                                     : int(num_lenslet/2) + prNN,
    # #                                     int(num_lenslet/2) - prNN
    # #                                     : int(num_lenslet/2) + prNN, 1],
    # #             color='r')
    # # plt.axis('equal')
    # # plt.xlabel('x (m)')
    # # plt.ylabel('y (m)')
    # # plt.title('Lenslet weighted phase slope (rad),\n'
    # #           + 'dz=0m, r={:.2f}m,\n'.format(r[ir])
    # #           +'Rytov={:.2f}, lenslet Rytov={:.2f}'.format(
    # #               rytov_in, rytov_lenslet_in))
    # # plt.show()

    # # proper SHWFS #
    # # plt.plot(lenslet_I_WFS_detector_in.flatten(),
    # #           size_projected_slope_WFS_in.flatten(),
    # #           ls='', marker=',')
    # # plt.xscale('log')
    # # plt.xlabel('Intensity')
    # # plt.ylabel('Slope size (rad)')
    # # plt.title('WFS intensity vs phase slope size (rad),\n'
    # #           + 'dz=0m, r={:.2f}m,\nRytov={:.2f}, WFS Rytov={:.2f}'.format(
    # #               r[ir], rytov_in, rytov_WFS_in))
    # # plt.show()

    # # plt.pcolor(np.arange(-prN, prN + 1)*d_in,
    # #             np.arange(-prN, prN + 1)*d_in,
    # #             np.log(I_WFS_detector_in[int(N/2) - prN : int(N/2) + prN,
    # #                   int(N/2) - prN : int(N/2) + prN]/I_norm)/np.log(10.),
    # #             shading='auto', vmin=I_min, vmax=I_max, snap=True)
    # # plt.colorbar()
    # # plt.vlines((np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
    # #             ymin=-prNN*d_in*lenslet_pixel,
    # #             ymax=prNN*d_in*lenslet_pixel,
    # #             color='m')
    # # plt.hlines((np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
    # #             xmin=-prNN*d_in*lenslet_pixel,
    # #             xmax=prNN*d_in*lenslet_pixel,
    # #             color='m')
    # # plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
    # #             (np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
    # #             projected_slope_WFS_in[int(num_lenslet/2) - prNN
    # #                                     : int(num_lenslet/2) + prNN,
    # #                                     int(num_lenslet/2) - prNN
    # #                                     : int(num_lenslet/2) + prNN, 0],
    # #             projected_slope_WFS_in[int(num_lenslet/2) - prNN
    # #                                     : int(num_lenslet/2) + prNN,
    # #                                     int(num_lenslet/2) - prNN
    # #                                     : int(num_lenslet/2) + prNN, 1],
    # #             color='r')
    # # plt.axis('equal')
    # # plt.xlabel('x (pixel)')
    # # plt.ylabel('y (pixel)')
    # # plt.title('WFS log10 image over phase slope,\n'
    # #           + 'dz=0m, r={:.2f}m, '.format(r[ir])
    # #           + 'Rytov={:.2f}, '.format(rytov_in)
    # #           + 'WFS Rytov={:.2f},\n'.format(rytov_WFS_in)
    # #           + 'max intensity {:.3f}, '.format(
    # #               lenslet_I_WFS_detector_in.max())
    # #           + 'min intensity {:.3f}'.format(
    # #               lenslet_I_WFS_detector_in.min()))
    # # plt.show()

    # # plt.pcolor((np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
    # #             (np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
    # #             size_projected_slope_WFS_in[int(num_lenslet/2) - prNN
    # #                                         : int(num_lenslet/2) + prNN,
    # #                                         int(num_lenslet/2) - prNN
    # #                                         : int(num_lenslet/2) + prNN],
    # #             shading='auto', snap=True)
    # # plt.colorbar()
    # # plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
    # #             (np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
    # #             projected_slope_WFS_in[int(num_lenslet/2) - prNN
    # #                                     : int(num_lenslet/2) + prNN,
    # #                                     int(num_lenslet/2) - prNN
    # #                                     : int(num_lenslet/2) + prNN, 0],
    # #             projected_slope_WFS_in[int(num_lenslet/2) - prNN
    # #                                     : int(num_lenslet/2) + prNN,
    # #                                     int(num_lenslet/2) - prNN
    # #                                     : int(num_lenslet/2) + prNN, 1],
    # #             color='r')
    # # plt.axis('equal')
    # # plt.xlabel('x (m)')
    # # plt.ylabel('y (m)')
    # # plt.title('WFS phase slope (rad),\n'
    # #           + 'dz=0m, r={:.2f}m,\n'.format(r[ir])
    # #           + 'Rytov={:.2f}, WFS Rytov={:.2f}'.format(
    # #               rytov_in, rytov_WFS_in))
    # # plt.show()
    
    # # perfect DM
    # plt.pcolor(np.arange(-prN, prN + 1)*d_in,
    #             np.arange(-prN, prN + 1)*d_in,
    #             dm_in_avg[int(N/2) - prN
    #                               : int(N/2) + prN,
    #                               int(N/2) - prN
    #                               : int(N/2) + prN],
    #             shading='auto', snap=True)
    # plt.colorbar()
    # plt.vlines((np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
    #             ymin=-prNN*d_in*lenslet_pixel,
    #             ymax=prNN*d_in*lenslet_pixel,
    #             color='m')
    # plt.hlines((np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
    #             xmin=-prNN*d_in*lenslet_pixel,
    #             xmax=prNN*d_in*lenslet_pixel,
    #             color='m')
    # plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
    #             (np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
    #             lenslet_slope_phase_in[int(num_lenslet/2) - prNN
    #                                     : int(num_lenslet/2) + prNN,
    #                                     int(num_lenslet/2) - prNN
    #                                     : int(num_lenslet/2) + prNN, 0],
    #             lenslet_slope_phase_in[int(num_lenslet/2) - prNN
    #                                     : int(num_lenslet/2) + prNN,
    #                                     int(num_lenslet/2) - prNN
    #                                     : int(num_lenslet/2) + prNN, 1],
    #             color='r')
    # plt.title('DM phase (rad) from lenslet phase slope,\n'
    #           + 'dz=0m, r={:.2f}m, Rytov=0'.format(r[ir]))
    # plt.axis('equal')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    # plt.show()
    
    # plt.pcolor(np.arange(-prN, prN + 1)*d_in,
    #             np.arange(-prN, prN + 1)*d_in,
    #             dm_in_WFS[int(N/2) - prN
    #                               : int(N/2) + prN,
    #                               int(N/2) - prN
    #                               : int(N/2) + prN],
    #             shading='auto', snap=True)
    # plt.colorbar()
    # plt.vlines((np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
    #             ymin=-prNN*d_in*lenslet_pixel,
    #             ymax=prNN*d_in*lenslet_pixel,
    #             color='m')
    # plt.hlines((np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
    #             xmin=-prNN*d_in*lenslet_pixel,
    #             xmax=prNN*d_in*lenslet_pixel,
    #             color='m')
    # plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
    #             (np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
    #             projected_slope_WFS_in[int(num_lenslet/2) - prNN
    #                                     : int(num_lenslet/2) + prNN,
    #                                     int(num_lenslet/2) - prNN
    #                                     : int(num_lenslet/2) + prNN, 0],
    #             projected_slope_WFS_in[int(num_lenslet/2) - prNN
    #                                     : int(num_lenslet/2) + prNN,
    #                                     int(num_lenslet/2) - prNN
    #                                     : int(num_lenslet/2) + prNN, 1],
    #             color='r')
    # plt.title('DM phase (rad) from WFS slope,\n'
    #           + 'dz=0m, r={:.2f}m, Rytov=0'.format(r[ir]))
    # plt.axis('equal')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    # plt.show()

    # plt.pcolor(np.arange(-prN, prN + 1)*d_in,
    #             np.arange(-prN, prN + 1)*d_in,
    #             res_in_avg[int(N/2) - prN
    #                               : int(N/2) + prN,
    #                               int(N/2) - prN
    #                               : int(N/2) + prN],
    #             shading='auto', snap=True,cmap='bwr')
    # plt.title('Residual phase (rad):\n'
    #           + 'altitude phase vs dm from altitude averaged slope\n'
    #           + 'dz=0m, r={:.2f}m, Rytov=0'.format(r[ir]))
    # plt.axis('equal')
    # plt.colorbar()
    # plt.show()

    # plt.pcolor(np.arange(-prN, prN + 1)*d_in,
    #             np.arange(-prN, prN + 1)*d_in,
    #             res_in_WFS[int(N/2) - prN
    #                               : int(N/2) + prN,
    #                               int(N/2) - prN
    #                               : int(N/2) + prN],
    #             shading='auto', snap=True,cmap='bwr')
    # plt.title('Residual phase (rad):\n'
    #           + 'altitude phase vs dm from altitude WFS slope\n'
    #           + 'dz=0m, r={:.2f}m, Rytov=0'.format(r[ir]))
    # plt.axis('equal')
    # plt.colorbar()
    # plt.show()
    
    plt.pcolor(np.arange(-8,9)*d_in,
                np.arange(-8,9)*d_in,
                dm_in_avg[:16,:16],
                shading='auto', snap=True)
    plt.colorbar()
    plt.vlines((np.arange(-0.5, 1.5))*d_in*lenslet_pixel,
                ymin=-0.5*d_in*lenslet_pixel,
                ymax=0.5*d_in*lenslet_pixel,
                color='m')
    plt.hlines((np.arange(-0.5, 1.5))*d_in*lenslet_pixel,
                xmin=-0.5*d_in*lenslet_pixel,
                xmax=0.5*d_in*lenslet_pixel,
                color='m')
    plt.quiver((np.arange(1))*d_in*lenslet_pixel,
                (np.arange(1))*d_in*lenslet_pixel,
                lenslet_slope_phase_in[:1,:1,0],
                lenslet_slope_phase_in[:1,:1,1],
                color='r')
    plt.title('DM phase (rad) from lenslet phase slope,\n'
              + 'dz=0m, r={:.2f}m, Rytov=0'.format(r[ir]))
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()
    
    plt.pcolor(np.arange(-8,9)*d_in,
                np.arange(-8,9)*d_in,
                dm_in_WFS[:16,:16],
                shading='auto', snap=True)
    plt.colorbar()
    plt.vlines((np.arange(-0.5, 1.5))*d_in*lenslet_pixel,
                ymin=-0.5*d_in*lenslet_pixel,
                ymax=0.5*d_in*lenslet_pixel,
                color='m')
    plt.hlines((np.arange(-0.5, 1.5))*d_in*lenslet_pixel,
                xmin=-0.5*d_in*lenslet_pixel,
                xmax=0.5*d_in*lenslet_pixel,
                color='m')
    plt.quiver((np.arange(1))*d_in*lenslet_pixel,
                (np.arange(1))*d_in*lenslet_pixel,
                projected_slope_WFS_in[:1,:1,0],
                projected_slope_WFS_in[:1,:1,1],
                color='r')
    plt.title('DM phase (rad) from WFS slope,\n'
              + 'dz=0m, r={:.2f}m, Rytov=0'.format(r[ir]))
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()

    plt.pcolor(np.arange(-8,9)*d_in,
            np.arange(-8,9)*d_in,
                res_in_avg[:16,:16],
                shading='auto', snap=True,cmap='bwr')
    plt.title('Residual phase (rad):\n'
              + 'altitude phase vs dm from altitude averaged slope\n'
              + 'dz=0m, r={:.2f}m, Rytov=0'.format(r[ir]))
    plt.axis('equal')
    plt.colorbar()
    plt.show()

    plt.pcolor(np.arange(-8,9)*d_in,
                np.arange(-8,9)*d_in,
                res_in_WFS[:16,:16],
                shading='auto', snap=True,cmap='bwr')
    plt.title('Residual phase (rad):\n'
              + 'altitude phase vs dm from altitude WFS slope\n'
              + 'dz=0m, r={:.2f}m, Rytov=0'.format(r[ir]))
    plt.axis('equal')
    plt.colorbar()
    plt.show()

    # tb = time.time()
    # t_plot_in = tb - ta
    # print('time plotting input = {:.6f}s'.format(t_plot_in))

    #
    # Propagations #
    #
    for iz in np.arange(len(z)):
        ta = time.time()
        U_out_big = angularSpectrum(U_in_big, wvl, d_in, d_out, z[iz])
        tb = time.time()
        t_prop = tb - ta
        print('time prop = {:.6f}s'.format(t_prop))
    
        #
        # Direct output
        #
        ta = time.time()
        U_out = mprop.crop2(U_out_big, pf)
        phase_out = np.angle(U_out)
        I_out = np.abs(U_out)**2
        slope_phase_out = mwfs.get_slope(phase_out, d_out, wvl)
        # size_slope_phase_out = np.sqrt(slope_phase_out[:,:,0]**2
        #                                 + slope_phase_out[:,:,1]**2)
        rytov_out = ((np.mean(I_out**2) / np.mean(I_out)**2) - 1.) / 4.
        tb = time.time()
        t_out_px = tb - ta
        print('time analyze output pixely = {:.6f}s'.format(t_out_px))
        
        ta = time.time()
        phase_out_unwrap = np.unwrap(np.unwrap(
            phase_out, period=2*np.pi, axis=-2),
            period=2*np.pi, axis=-1)
        # PSD_out = pick_even_index(get_PSD(mirror_pad2(phase_out_unwrap), d_out))
        # phase_dif = (np.unwrap(np.unwrap(phase_in-phase_out,axis=-2),axis=-1))
        # PSD_dif = pick_even_index(get_PSD(mirror_pad2(phase_dif), d_out))
        tb = time.time()
        t_PSD_out = tb - ta
        print('time measuring psd out and psd dif = {:.6f}s'.format(t_PSD_out))
    
        #
        # Averaging lenslet
        #
        ta = time.time()
        lenslet_I_out = mwfs.get_val_lenslet(I_out, lenslet_pixel)
        lenslet_slope_phase_out = mwfs.get_vec_lenslet(
            slope_phase_out, lenslet_pixel)
        # size_lenslet_slope_phase_out = np.sqrt(
        #     lenslet_slope_phase_out[ : , : , 0]**2
        #     + lenslet_slope_phase_out[ : , : , 1]**2)
        # rytov_lenslet_out = ((np.mean(lenslet_I_out**2) 
        #                       / np.mean(lenslet_I_out)**2) - 1.) / 4.
        tb = time.time()
        t_out_lenslet = tb - ta
        print('time analyze output lensletly = {:.6f}s'.format(t_out_lenslet))
    
        #
        # SHWFS cheat by weighting slope with intensity #
        #
        ta = time.time()
        weight_slope_phase_out = np.zeros((N,N,2), dtype=float)
        weight_slope_phase_out[:,:,0] = slope_phase_out[:,:,0]*I_out
        weight_slope_phase_out[:,:,1] = slope_phase_out[:,:,1]*I_out
        lenslet_weight_slope_phase_out = mwfs.get_vec_lenslet(
            weight_slope_phase_out, lenslet_pixel)
        lenslet_weight_slope_phase_out[:,:,0] = (
            lenslet_weight_slope_phase_out[:,:,0]/lenslet_I_out)
        lenslet_weight_slope_phase_out[:,:,1] = (
            lenslet_weight_slope_phase_out[:,:,1]/lenslet_I_out)
        # size_lenslet_weight_slope_phase_out = np.sqrt(
        #     lenslet_weight_slope_phase_out[ : , : , 0]**2
        #     + lenslet_weight_slope_phase_out[ : , : , 1]**2)
        tb = time.time()
        t_out_weight = tb - ta
        print('time analyze output weight = {:.6f}s'.format(t_out_weight))
    
        #
        # Proper SHWFS
        #
        ta = time.time()
        U_WFS_detector_out = mwfs.propagate_lenslet(U_out, wvl, d_lenslet,
                                                f_lenslet, lenslet_pixel,
                                                d_detector, pf_lenslet)
        I_WFS_detector_out = (np.abs(U_WFS_detector_out)**2
                              * (d_detector/d_lenslet)**2)
        lenslet_I_WFS_detector_out = mwfs.get_val_lenslet(
            I_WFS_detector_out, lenslet_pixel)
        noisy_I_WFS_detector_out = (np.random.poisson(
            I_WFS_detector_out/lenslet_pixel**2*SNR**2)
            / SNR**2*lenslet_pixel**2)
        slope_WFS_out = mwfs.get_SHWFS_slope(noisy_I_WFS_detector_out,
                                              lenslet_pixel, d_detector,
                                              f_lenslet)
        projected_slope_WFS_out = slope_WFS_out * d_lenslet / d_out
        # size_projected_slope_WFS_out = np.sqrt(
        #     projected_slope_WFS_out[:,:,0]**2
        #     + projected_slope_WFS_out[:,:,1]**2)
        # rytov_WFS_out = (
        #     (np.mean(lenslet_I_WFS_detector_out**2) / np.mean(
        #         lenslet_I_WFS_detector_out)**2) - 1.) / 4.
        tb = time.time()
        t_WFS_out = tb - ta
        print('time analyze WFS out = {:.6f}s'.format(t_WFS_out))
    
        # Store data
        all_rytov_pixel[ir,iz] = rytov_out
        # all_rytov_lenslet[ir,iz] = rytov_lenslet_out
        # std_slope_out[ir, iz] = slope_phase_out.std()
        
        # make perfect dm
        ta = time.time()
        # dm_out_avg = make_dm_corner(lenslet_slope_phase_out, lenslet_pixel, d_out)
        # dm_out_WFS = make_dm_corner(projected_slope_WFS_out, lenslet_pixel, d_out)
        # dm_out_weight = make_dm_corner(lenslet_weight_slope_phase_out, lenslet_pixel, d_out)
        dm_out_avg = simple_dm(lenslet_slope_phase_out[:1,:1,:], lenslet_pixel, d_out)
        dm_out_WFS = simple_dm(projected_slope_WFS_out[:1,:1,:], lenslet_pixel, d_out)
        dm_out_weight = simple_dm(lenslet_weight_slope_phase_out[:1,:1,:], lenslet_pixel, d_out)
        # dm_out_avg = make_dm_perfect_piston2(phase_out,lenslet_slope_phase_out, lenslet_pixel, d_out)
        # dm_out_WFS = make_dm_perfect_piston2(phase_out,projected_slope_WFS_out, lenslet_pixel, d_out)
        # dm_out_weight = make_dm_perfect_piston2(phase_out,lenslet_weight_slope_phase_out, lenslet_pixel, d_out)
        # dm_out_avg = make_dm_perfect_piston3(phase_out,lenslet_slope_phase_out, lenslet_pixel, d_out)
        # dm_out_WFS = make_dm_perfect_piston3(phase_out,projected_slope_WFS_out, lenslet_pixel, d_out)
        # dm_out_weight = make_dm_perfect_piston3(phase_out,lenslet_weight_slope_phase_out, lenslet_pixel, d_out)
        
        res_out_avg = dif_unwrap(phase_out[:lenslet_pixel,:lenslet_pixel], dm_out_avg)
        res_out_avg = res_out_avg-res_out_avg.mean()
        res_out_WFS = dif_unwrap(phase_out[:lenslet_pixel,:lenslet_pixel], dm_out_WFS)
        res_out_WFS = res_out_WFS-res_out_WFS.mean()
        res_out_weight = dif_unwrap(phase_out[:lenslet_pixel,:lenslet_pixel], dm_out_weight)
        res_out_weight = res_out_weight-res_out_weight.mean()
        res_sg_avg = dif_unwrap(phase_in[:lenslet_pixel,:lenslet_pixel], dm_out_avg)
        res_sg_avg = res_sg_avg-res_sg_avg.mean()
        res_sg_WFS = dif_unwrap(phase_in[:lenslet_pixel,:lenslet_pixel], dm_out_WFS)
        res_sg_WFS = res_sg_WFS-res_sg_WFS.mean()
        res_sg_weight = dif_unwrap(phase_in[:lenslet_pixel,:lenslet_pixel], dm_out_weight)
        res_sg_weight = res_sg_weight-res_sg_weight.mean()
        
        tb = time.time()
        t_dm_out = tb - ta
        print('time make perfect dms = {:.6f}s'.format(t_dm_out))
        
        # rms calculation
        ta = time.time()
        rms_sky_to_ground[ir,iz] = dif_unwrap(phase_in,phase_out).std()
    
        rms_sky_to_ground_avg[ir,iz] = res_sg_avg.std()
        rms_sky_to_ground_WFS[ir,iz] = res_sg_WFS.std()
        rms_sky_to_ground_weight[ir,iz] = res_sg_weight.std()
    
        rms_ground_to_ground_avg[ir,iz] = res_out_avg.std()
        rms_ground_to_ground_WFS[ir,iz] = res_out_WFS.std()
        rms_ground_to_ground_weight[ir,iz] = res_out_weight.std()
    
        rms_ground_avg_to_ground_WFS[ir,iz] = dif_unwrap(dm_out_avg,dm_out_WFS).std()
        rms_ground_avg_to_ground_weight[ir,iz] = dif_unwrap(dm_out_avg,dm_out_weight).std()
        rms_ground_WFS_to_ground_weight[ir,iz] = dif_unwrap(dm_out_WFS, dm_out_weight).std()
        
        
        # rms_sky_to_ground[ir,iz] = (slope_phase_in - slope_phase_out).std()
    
        # rms_sky_to_ground_avg[ir,iz] = get_rms3_lenslet(slope_phase_in,
        #                                                 lenslet_slope_phase_out)
        # rms_sky_to_ground_WFS[ir,iz] = get_rms3_lenslet(slope_phase_in,
        #                                                 projected_slope_WFS_out)
        # rms_sky_to_ground_weight[ir,iz] = (
        #     get_rms3_lenslet(slope_phase_in, lenslet_weight_slope_phase_out))
    
        # rms_ground_to_ground_avg[ir,iz] = (
        #     get_rms3_lenslet(slope_phase_out, lenslet_slope_phase_out))
        # rms_ground_to_ground_WFS[ir,iz] = (
        #     get_rms3_lenslet(slope_phase_out, projected_slope_WFS_out))
        # rms_ground_to_ground_weight[ir,iz] = (
        #     get_rms3_lenslet(slope_phase_out, lenslet_weight_slope_phase_out))
    
        # rms_ground_avg_to_ground_WFS[ir,iz] = (
        #     (lenslet_slope_phase_out - projected_slope_WFS_out)).std()
        # rms_ground_avg_to_ground_weight[ir,iz] = (
        #     (lenslet_slope_phase_out - lenslet_weight_slope_phase_out)).std()
        # rms_ground_WFS_to_ground_weight[ir,iz] = (
        #     (projected_slope_WFS_out - lenslet_weight_slope_phase_out)).std()
        tb = time.time()
        t_rms_out = tb - ta
        print('time analyze rms out = {:.6f}s'.format(t_rms_out))
    
    
    
        # # #
        # # # Plot output analysis
        # # #
        
        # ta = time.time()
        # # Actual values
        # # plt.plot(I_out.flatten(), size_slope_phase_out.flatten(),
        # #           ls='', marker=',', alpha=0.01)
        # # plt.xscale('log')
        # # plt.xlabel('Intensity')
        # # plt.ylabel('Slope size (rad)')
        # # plt.title('Intensity vs phase slope size (rad),\n'
        # #           + 'dz={:.0f}m, r={:.2f}m, Rytov={:.2f},'.format(
        # #               z[iz], r[ir], rytov_out))
        # # plt.show()
    
        # # plt.pcolor(np.arange(-prN, prN)*d_out,
        # #             np.arange(-prN, prN)*d_out,
        # #             np.log(I_out[int(N/2) - prN : int(N/2) + prN,
        # #                       int(N/2) - prN : int(N/2) + prN])/np.log(10),
        # #             shading='auto', vmin=I_min, vmax=I_max)
        # # plt.colorbar()
        # # plt.vlines((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
        # #             ymin=-prNN*d_out*lenslet_pixel,
        # #             ymax=prNN*d_out*lenslet_pixel,
        # #             color='m')
        # # plt.hlines((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
        # #             xmin=-prNN*d_out*lenslet_pixel,
        # #             xmax=prNN*d_out*lenslet_pixel,
        # #             color='m')
        # # plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        # #             (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        # #             lenslet_slope_phase_out[int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN,
        # #                                     int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN, 0],
        # #             lenslet_slope_phase_out[int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN,
        # #                                     int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN, 1],
        # #             color='r')
        # # plt.title('Log10 intensity over lenslet phase slope,\n'
        # #           + 'dz={:.0f}m, r={:.2f}m, '.format(z[iz], r[ir])
        # #           + 'Rytov={:.2f}, lenslet Rytov={:.2f}\n'.format(
        # #               rytov_out, rytov_lenslet_out)
        # #           + 'max intensity {:.3f}, min intensity {:.3f}'.format(
        # #               I_out.max(), I_out.min()))
        # # plt.axis('equal')
        # # plt.xlabel('x (m)')
        # # plt.ylabel('y (m)')
        # # plt.show()
    
        # plt.pcolor(np.arange(-prN, prN + 1)*d_out,
        #             np.arange(-prN, prN + 1)*d_out,
        #             phase_out[int(N/2) - prN
        #                               : int(N/2) + prN,
        #                               int(N/2) - prN
        #                               : int(N/2) + prN],
        #             shading='auto', snap=True)
        # plt.colorbar()
    
    
        # plt.vlines((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
        #             ymin=-prNN*d_out*lenslet_pixel,
        #             ymax=prNN*d_out*lenslet_pixel,
        #             color='m')
        # plt.hlines((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
        #             xmin=-prNN*d_out*lenslet_pixel,
        #             xmax=prNN*d_out*lenslet_pixel,
        #             color='m')
        # plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        #             (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        #             lenslet_slope_phase_out[int(num_lenslet/2) - prNN
        #                                     : int(num_lenslet/2) + prNN,
        #                                     int(num_lenslet/2) - prNN
        #                                     : int(num_lenslet/2) + prNN, 0],
        #             lenslet_slope_phase_out[int(num_lenslet/2) - prNN
        #                                     : int(num_lenslet/2) + prNN,
        #                                     int(num_lenslet/2) - prNN
        #                                     : int(num_lenslet/2) + prNN, 1],
        #             color='r')
        # plt.title('Phase (rad) over lenslet phase slope,\n'
        #           + 'dz={:.0f}m, r={:.2f}m,\n'.format(z[iz], r[ir])
        #           + 'Rytov={:.2f}'.format(rytov_out))
        # plt.axis('equal')
        # plt.xlabel('x (m)')
        # plt.ylabel('y (m)')
        # plt.show()
        
        plt.pcolor(np.arange(-8,9)*d_in,
                    np.arange(-8,9)*d_in,
                    phase_out[:16,:16],
                    shading='auto', snap=True)
        plt.colorbar()
        plt.vlines((np.arange(-0.5, 1.5))*d_in*lenslet_pixel,
                    ymin=-0.5*d_in*lenslet_pixel,
                    ymax=0.5*d_in*lenslet_pixel,
                    color='m')
        plt.hlines((np.arange(-0.5, 1.5))*d_in*lenslet_pixel,
                    xmin=-0.5*d_in*lenslet_pixel,
                    xmax=0.5*d_in*lenslet_pixel,
                    color='m')
        plt.quiver((np.arange(1))*d_in*lenslet_pixel,
                    (np.arange(1))*d_in*lenslet_pixel,
                    lenslet_slope_phase_out[:1,:1,0],
                    lenslet_slope_phase_out[:1,:1,1],
                    color='r')
        plt.title('Phase (rad) over lenslet phase slope,\n'
                  + 'dz={:.0f}m, r={:.2f}m,\n'.format(z[iz], r[ir])
                  + 'Rytov={:.2f}'.format(rytov_out))
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()
    
        # # # Averaged
        # # plt.plot(lenslet_I_out.flatten(),
        # #           size_lenslet_slope_phase_out.flatten(),
        # #           ls='', marker=',')
        # # plt.xscale('log')
        # # plt.xlabel('Intensity')
        # # plt.ylabel('Slope size (rad)')
        # # plt.title('Lenslet intensity vs phase slope size (rad),\n'
        # #           + 'dz={:.0f}m, r={:.2f}m,\n'.format(z[iz], r[ir])
        # #           + 'Rytov={:.2f}, lenslet Rytov={:.2f}'.format(
        # #               rytov_out, rytov_lenslet_out))
        # # plt.show()
    
        # # plt.pcolor(np.arange(-prNN, prNN + 1)*d_out*lenslet_pixel,
        # #             np.arange(-prNN, prNN + 1)*d_out*lenslet_pixel,
        # #             np.log(lenslet_I_out[int(num_lenslet/2) - prNN
        # #                           : int(num_lenslet/2) + prNN,
        # #                           int(num_lenslet/2) - prNN
        # #                           : int(num_lenslet/2) + prNN])/np.log(10.),
        # #             shading='auto', vmin=I_min, vmax=I_max, snap=True)
        # # plt.colorbar()
        # # plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        # #             (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        # #             lenslet_slope_phase_out[int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN,
        # #                                     int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN, 0],
        # #             lenslet_slope_phase_out[int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN,
        # #                                     int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN, 1],
        # #             color='r')
        # # plt.axis('equal')
        # # plt.xlabel('x (m)')
        # # plt.ylabel('y (m)')
        # # plt.title('Lenslet log10 intensity over lenslet phase slope,\n'
        # #           + 'dz={:.0f}m, r={:.2f}m, '.format(z[iz], r[ir])
        # #           + 'Rytov={:.2f}, lenslet Rytov={:.2f}\n'.format(
        # #               rytov_out, rytov_lenslet_out)
        # #           + 'max intensity {:.3f}, '.format(
        # #               lenslet_I_out.max())
        # #           + 'min intensity {:.3f}'.format(
        # #               lenslet_I_out.min()))
        # # plt.show()
    
        # # plt.pcolor((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
        # #             (np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
        # #             size_lenslet_slope_phase_out[int(num_lenslet/2) - prNN
        # #                                         : int(num_lenslet/2) + prNN,
        # #                                         int(num_lenslet/2) - prNN
        # #                                         : int(num_lenslet/2) + prNN],
        # #             shading='auto', snap=True)
        # # plt.colorbar()
        # # plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        # #             (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        # #             lenslet_slope_phase_out[int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN,
        # #                                     int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN, 0],
        # #             lenslet_slope_phase_out[int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN,
        # #                                     int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN, 1],
        # #             color='r')
        # # plt.axis('equal')
        # # plt.xlabel('x (m)')
        # # plt.ylabel('y (m)')
        # # plt.title('Lenslet phase slope size (rad),\n' 
        # #           + 'dz={:.0f}m, r={:.2f}m,\n'.format(z[iz], r[ir])
        # #           + 'Rytov={:.2f}, lenslet Rytov={:.2f}'.format(
        # #               rytov_out, rytov_lenslet_out))
        # # plt.show()
    
        # # # Cheat SHWFS
        # # plt.pcolor(np.arange(-prNN, prNN + 1)*d_out*lenslet_pixel,
        # #             np.arange(-prNN, prNN + 1)*d_out*lenslet_pixel,
        # #             np.log(lenslet_I_out[int(num_lenslet/2) - prNN :
        # #                                 int(num_lenslet/2) + prNN,
        # #                                 int(num_lenslet/2) - prNN :
        # #                                     int(num_lenslet/2) + prNN])
        # #                 /np.log(10.),
        # #             shading='auto', vmin=I_min, vmax=I_max, snap=True)
        # # plt.colorbar()
        # # plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        # #             (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        # #             lenslet_weight_slope_phase_out[int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN,
        # #                                     int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN, 0],
        # #             lenslet_weight_slope_phase_out[int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN,
        # #                                     int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN, 1],
        # #             color='r')
        # # plt.axis('equal')
        # # plt.xlabel('x (pixel)')
        # # plt.ylabel('y (pixel)')
        # # plt.title('Lenslet log10 intensity over '
        # #           +'weighted phase slope,\n'
        # #           + 'dz={:.0f}m, r={:.2f}m, '.format(z[iz], r[ir])
        # #           + 'Rytov={:.2f}, '.format(rytov_out)
        # #           + 'lenslet Rytov={:.2f},\n'.format(rytov_lenslet_out)
        # #           + 'max intensity {:.3f}, '.format(
        # #               lenslet_I_out.max())
        # #           + 'min intensity {:.3f}'.format(
        # #               lenslet_I_out.min()))
        # # plt.show()
    
        # # plt.pcolor((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
        # #             (np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
        # #             size_lenslet_weight_slope_phase_out[int(num_lenslet/2)
        # #                                                 - prNN
        # #                                                 : int(num_lenslet/2)
        # #                                                 + prNN,
        # #                                                 int(num_lenslet/2)
        # #                                                 - prNN
        # #                                                 : int(num_lenslet/2)
        # #                                                 + prNN],
        # #             shading='auto', snap=True)
        # # plt.colorbar()
        # # plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        # #             (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        # #             lenslet_weight_slope_phase_out[int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN,
        # #                                     int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN, 0],
        # #             lenslet_weight_slope_phase_out[int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN,
        # #                                     int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN, 1],
        # #             color='r')
        # # plt.axis('equal')
        # # plt.xlabel('x (m)')
        # # plt.ylabel('y (m)')
        # # plt.title('Lenslet weighted phase slope (rad),\n'
        # #           + 'dz={:.0f}m, r={:.2f}m,\n'.format(z[iz], r[ir])
        # #           +'Rytov={:.2f}, lenslet Rytov={:.2f}'.format(
        # #               rytov_out, rytov_lenslet_out))
        # # plt.show()
    
        # # # Proper SHWFS #
        # # plt.plot(lenslet_I_WFS_detector_out.flatten(),
        # #           size_projected_slope_WFS_out.flatten(),
        # #           ls='', marker=',')
        # # plt.xscale('log')
        # # plt.xlabel('Intensity')
        # # plt.ylabel('Slope size (rad)')
        # # plt.title('WFS intensity vs phase slope size (rad),\n'
        # #           + 'dz={:.0f}m, r={:.2f}m,\nRytov={:.2f}, WFS Rytov={:.2f}'.format(
        # #               z[iz], r[ir], rytov_out, rytov_WFS_out))
        # # plt.show()
    
        # # plt.pcolor(np.arange(-prN, prN + 1)*d_out,
        # #             np.arange(-prN, prN + 1)*d_out,
        # #             np.log(I_WFS_detector_out[int(N/2) - prN : int(N/2) + prN,
        # #                   int(N/2) - prN : int(N/2) + prN]/I_norm)/np.log(10.),
        # #             shading='auto', vmin=I_min, vmax=I_max, snap=True)
        # # plt.colorbar()
        # # plt.vlines((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
        # #             ymin=-prNN*d_out*lenslet_pixel,
        # #             ymax=prNN*d_out*lenslet_pixel,
        # #             color='m')
        # # plt.hlines((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
        # #             xmin=-prNN*d_out*lenslet_pixel,
        # #             xmax=prNN*d_out*lenslet_pixel,
        # #             color='m')
        # # plt.axis('equal')
        # # plt.xlabel('x (pixel)')
        # # plt.ylabel('y (pixel)')
        # # plt.title('WFS log10 image over phase slope,\n'
        # #           + 'dz={:.0f}m, r={:.2f}m, '.format(z[iz], r[ir])
        # #           + 'Rytov={:.2f}, '.format(rytov_out)
        # #           + 'WFS Rytov={:.2f},\n'.format(rytov_WFS_out)
        # #           + 'max intensity {:.3f}, '.format(
        # #               lenslet_I_WFS_detector_out.max())
        # #           + 'min intensity {:.3f}'.format(
        # #               lenslet_I_WFS_detector_out.min()))
        # # plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        # #             (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        # #             slope_WFS_out[int(num_lenslet/2) - prNN
        # #                           : int(num_lenslet/2) + prNN,
        # #                           int(num_lenslet/2) - prNN
        # #                           : int(num_lenslet/2) + prNN, 0],
        # #             slope_WFS_out[int(num_lenslet/2) - prNN
        # #                           : int(num_lenslet/2) + prNN,
        # #                           int(num_lenslet/2) - prNN
        # #                           : int(num_lenslet/2) + prNN, 1],
        # #             color='r')
        # # plt.show()
    
        # # plt.pcolor(np.arange(-prNN, prNN + 1)*d_out*lenslet_pixel,
        # #             np.arange(-prNN, prNN + 1)*d_out*lenslet_pixel,
        # #             np.log(lenslet_I_WFS_detector_out[int(num_lenslet/2)
        # #                                               - prNN
        # #                                               : int(num_lenslet/2)
        # #                                               + prNN,
        # #                                               int(num_lenslet/2)
        # #                                               - prNN
        # #                                               : int(num_lenslet/2)
        # #                                               + prNN])
        # #                 /np.log(10.),
        # #             shading='auto', vmin=I_min, vmax=I_max, snap=True)
        # # plt.colorbar()
        # # plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        # #             (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        # #             projected_slope_WFS_out[int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN,
        # #                                     int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN, 0],
        # #             projected_slope_WFS_out[int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN,
        # #                                     int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN, 1],
        # #             color='r')
        # # plt.axis('equal')
        # # plt.xlabel('x (pixel)')
        # # plt.ylabel('y (pixel)')
        # # plt.title('WFS log10 total intensity over phase slope,\n'
        # #           + 'dz={:.0f}m, r={:.2f}m, '.format(z[iz], r[ir])
        # #           + 'Rytov={:.2f}, '.format(rytov_out)
        # #           + 'WFS Rytov={:.2f},\n'.format(rytov_WFS_out)
        # #           + 'max intensity {:.3f}, '.format(
        # #               lenslet_I_WFS_detector_out.max())
        # #           + 'min intensity {:.3f}'.format(
        # #               lenslet_I_WFS_detector_out.min()))
        # # plt.show()
    
        # # plt.pcolor((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
        # #             (np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
        # #             size_projected_slope_WFS_out[int(num_lenslet/2) - prNN
        # #                                         : int(num_lenslet/2) + prNN,
        # #                                         int(num_lenslet/2) - prNN
        # #                                         : int(num_lenslet/2) + prNN],
        # #             shading='auto', snap=True)
        # # plt.colorbar()
        # # plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        # #             (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        # #             projected_slope_WFS_out[int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN,
        # #                                     int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN, 0],
        # #             projected_slope_WFS_out[int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN,
        # #                                     int(num_lenslet/2) - prNN
        # #                                     : int(num_lenslet/2) + prNN, 1],
        # #             color='r')
        # # plt.axis('equal')
        # # plt.xlabel('x (m)')
        # # plt.ylabel('y (m)')
        # # plt.title('WFS phase slope (rad),\n'
        # #           + 'dz={:.0f}m, r={:.2f}m,\n'.format(z[iz], r[ir])
        # #           + 'Rytov={:.2f}, WFS Rytov={:.2f}'.format(
        # #               rytov_out, rytov_WFS_out))
        # # plt.show()
        
        # # PSD analysis result
        # # plt.imshow(phase_out[:16,:16])
        # # plt.colorbar()
        # # plt.title('Sampled phase out\n'
        # #           + 'r={:.0f}cm, dz={:.0f}km'.format(r[ir]*100,z[iz]/1000))
        # # plt.show()
        # # plt.imshow(phase_out_unwrap[:16,:16])
        # # plt.colorbar()
        # # plt.title('Sampled unwrapped phase out\n'
        # #           + 'r={:.0f}cm, dz={:.0f}km'.format(r[ir]*100,z[iz]/1000))
        # # plt.show()
        # # plt.imshow(phase_dif[:128,:128])
        # # plt.vlines(np.arange(9)*16,0,128,color='r')
        # # plt.hlines(np.arange(9)*16,0,128,color='r')
        # # plt.colorbar()
        # # plt.title('Sampled unwrapped phase in-out\n'
        # #           + 'r={:.0f}cm, dz={:.0f}km'.format(r[ir]*100,z[iz]/1000))
        # # plt.show()
        
        # # plt.plot((np.arange(int(N/2-1))+1)/(N*d_in),
        # #           PSD[int(N/2),int(N/2)+1:],
        # #           ls='-', marker='',
        # #           color='r', label='theoretical PSD')
        # # plt.plot((np.arange(int(N/2-1))+1)/(N*d_in),
        # #           PSD_in[int(N/2),int(N/2)+1:],
        # #           ls='', marker='v', alpha=0.25,
        # #           color='g', label='PSD altitude')
        # # plt.plot((np.arange(int(N/2)-1)+1)/(N*d_out),
        # #           PSD_out[int(N/2),int(N/2)+1:],
        # #           ls='', marker='^', alpha=0.25,
        # #           color='b', label='PSD ground')
        # # plt.plot((np.arange(int(N/2-1))+1)/(N*d_in),
        # #           PSD_dif[int(N/2),int(N/2)+1:],
        # #           ls='', marker='<', alpha=0.25,
        # #           color='c', label='PSD of phase difference')
        # # plt.vlines(1./(2.*lenslet_pixel*d_out),1e-10,1e2,
        # #            label='lenslet detectable',color='m')
        # # plt.vlines(1./(2.*np.sqrt(wvl*z[iz])),1e-10,1e2,
        # #            label='1st Fresnel zone size',color='k')
        # # plt.xscale('log')
        # # plt.yscale('log')
        # # plt.xlabel('Spatial frequency (1/m)')
        # # plt.ylabel('PSD (m^2)')
        # # plt.legend()
        # # plt.title('Comparison of phase screen power spectrum density\n'
        # #           + 'before and after propagation, Rytov={:.2f}\n'.format(rytov_out)
        # #           + 'r={:.0f}cm, dz={:.0f}km, '.format(r[ir]*100,z[iz]/1000)
        # #           + 'accurate screen pixel={:d}px\n'.format(N)
        # #           + 'pixel={:.0f}mm, L0={:.0f}m'.format(d_in*1000,L0))
        # # plt.show()
        
        # # plt.plot((np.arange(int(N/2-1))+1)/(N*d_in),
        # #           PSD_out[int(N/2),int(N/2)+1:] / PSD_in[int(N/2),int(N/2)+1:],
        # #           ls='', marker='^', alpha=0.25,
        # #           color='b', label='PSD ground / PSD altitude')
        # # plt.xscale('log')
        # # plt.yscale('log')
        # # plt.xlabel('Spatial frequency (1/m)')
        # # plt.ylabel('PSD (m^2)')
        # # plt.hlines(1,1e-1,1e2,label='equal',color='r')
        # # plt.vlines(1./(2.*lenslet_pixel*d_out),1e-3,1e3,
        # #            label='lenslet detectable',color='m')
        # # plt.vlines(1./(2.*np.sqrt(wvl*z[iz])),1e-3,1e3,
        # #            label='frequency of 1st Fresnel zone size',color='k')
        # # plt.legend()
        # # plt.title('Comparison of phase screen power spectrum density\n'
        # #           + 'before - after propagation, Rytov={:.2f}\n'.format(rytov_out)
        # #           + 'r={:.0f}cm, dz={:.0f}km, '.format(r[ir]*100,z[iz]/1000)
        # #           + 'accurate screen pixel={:d}px\n'.format(N)
        # #           + 'pixel={:.0f}mm, L0={:.0f}m'.format(d_in*1000,L0))
        # # plt.show()
        
        
        # #dm
        
        # plt.pcolor(np.arange(-prN, prN + 1)*d_out,
        #             np.arange(-prN, prN + 1)*d_out,
        #             dm_out_avg[int(N/2) - prN
        #                               : int(N/2) + prN,
        #                               int(N/2) - prN
        #                               : int(N/2) + prN],
        #             shading='auto', snap=True)
        # plt.colorbar()
        # plt.vlines((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
        #             ymin=-prNN*d_out*lenslet_pixel,
        #             ymax=prNN*d_out*lenslet_pixel,
        #             color='m')
        # plt.hlines((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
        #             xmin=-prNN*d_out*lenslet_pixel,
        #             xmax=prNN*d_out*lenslet_pixel,
        #             color='m')
        # plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        #             (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        #             lenslet_slope_phase_out[int(num_lenslet/2) - prNN
        #                                     : int(num_lenslet/2) + prNN,
        #                                     int(num_lenslet/2) - prNN
        #                                     : int(num_lenslet/2) + prNN, 0],
        #             lenslet_slope_phase_out[int(num_lenslet/2) - prNN
        #                                     : int(num_lenslet/2) + prNN,
        #                                     int(num_lenslet/2) - prNN
        #                                     : int(num_lenslet/2) + prNN, 1],
        #             color='r')
        # plt.title('DM phase (rad) from lenslet phase slope,\n'
        #           + 'dz={:.0f}m, r={:.2f}m, Rytov={:.2f}'.format(z[iz],r[ir], rytov_out))
        # plt.axis('equal')
        # plt.xlabel('x (m)')
        # plt.ylabel('y (m)')
        # plt.show()
        
        # plt.pcolor(np.arange(-prN, prN + 1)*d_out,
        #             np.arange(-prN, prN + 1)*d_out,
        #             dm_out_WFS[int(N/2) - prN
        #                               : int(N/2) + prN,
        #                               int(N/2) - prN
        #                               : int(N/2) + prN],
        #             shading='auto', snap=True)
        # plt.colorbar()
        # plt.vlines((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
        #             ymin=-prNN*d_out*lenslet_pixel,
        #             ymax=prNN*d_out*lenslet_pixel,
        #             color='m')
        # plt.hlines((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
        #             xmin=-prNN*d_out*lenslet_pixel,
        #             xmax=prNN*d_out*lenslet_pixel,
        #             color='m')
        # plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        #             (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        #             projected_slope_WFS_out[int(num_lenslet/2) - prNN
        #                                     : int(num_lenslet/2) + prNN,
        #                                     int(num_lenslet/2) - prNN
        #                                     : int(num_lenslet/2) + prNN, 0],
        #             projected_slope_WFS_out[int(num_lenslet/2) - prNN
        #                                     : int(num_lenslet/2) + prNN,
        #                                     int(num_lenslet/2) - prNN
        #                                     : int(num_lenslet/2) + prNN, 1],
        #             color='r')
        # plt.title('DM phase (rad) from WFS slope,\n'
        #           + 'dz={:.0f}m, r={:.2f}m, Rytov={:.2f}'.format(z[iz],r[ir], rytov_out))
        # plt.axis('equal')
        # plt.xlabel('x (m)')
        # plt.ylabel('y (m)')
        # plt.show()
        
        # plt.pcolor(np.arange(-prN, prN + 1)*d_out,
        #             np.arange(-prN, prN + 1)*d_out,
        #             dm_out_weight[int(N/2) - prN
        #                               : int(N/2) + prN,
        #                               int(N/2) - prN
        #                               : int(N/2) + prN],
        #             shading='auto', snap=True)
        # plt.colorbar()
        # plt.vlines((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
        #             ymin=-prNN*d_out*lenslet_pixel,
        #             ymax=prNN*d_out*lenslet_pixel,
        #             color='m')
        # plt.hlines((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
        #             xmin=-prNN*d_out*lenslet_pixel,
        #             xmax=prNN*d_out*lenslet_pixel,
        #             color='m')
        # plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        #             (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
        #             lenslet_weight_slope_phase_out[int(num_lenslet/2) - prNN
        #                                     : int(num_lenslet/2) + prNN,
        #                                     int(num_lenslet/2) - prNN
        #                                     : int(num_lenslet/2) + prNN, 0],
        #             lenslet_weight_slope_phase_out[int(num_lenslet/2) - prNN
        #                                     : int(num_lenslet/2) + prNN,
        #                                     int(num_lenslet/2) - prNN
        #                                     : int(num_lenslet/2) + prNN, 1],
        #             color='r')
        # plt.title('DM phase (rad) from WFS approximation,\n'
        #           + 'dz={:.0f}m, r={:.2f}m, Rytov={:.2f}'.format(z[iz],r[ir], rytov_out))
        # plt.axis('equal')
        # plt.xlabel('x (m)')
        # plt.ylabel('y (m)')
        # plt.show()
    
        # plt.pcolor(np.arange(-prN, prN + 1)*d_in,
        #             np.arange(-prN, prN + 1)*d_in,
        #             res_out_avg[int(N/2) - prN
        #                               : int(N/2) + prN,
        #                               int(N/2) - prN
        #                               : int(N/2) + prN],
        #             shading='auto', snap=True,cmap='bwr')
        # plt.title('Residual phase (rad):\n'
        #           + 'ground phase vs dm from ground averaged slope\n'
        #           + 'dz={:.0f}m, r={:.2f}m, Rytov={:.2f}'.format(z[iz],r[ir], rytov_out))
        # plt.axis('equal')
        # plt.colorbar()
        # plt.show()
    
        # plt.pcolor(np.arange(-prN, prN + 1)*d_in,
        #             np.arange(-prN, prN + 1)*d_in,
        #             res_out_WFS[int(N/2) - prN
        #                               : int(N/2) + prN,
        #                               int(N/2) - prN
        #                               : int(N/2) + prN],
        #             shading='auto', snap=True,cmap='bwr')
        # plt.title('Residual phase (rad):\n'
        #           + 'ground phase vs dm from ground WFS slope\n'
        #           + 'dz={:.0f}m, r={:.2f}m, Rytov={:.2f}'.format(z[iz],r[ir], rytov_out))
        # plt.axis('equal')
        # plt.colorbar()
        # plt.show()
        
        # plt.pcolor(np.arange(-prN, prN + 1)*d_in,
        #             np.arange(-prN, prN + 1)*d_in,
        #             res_out_weight[int(N/2) - prN
        #                               : int(N/2) + prN,
        #                               int(N/2) - prN
        #                               : int(N/2) + prN],
        #             shading='auto', snap=True,cmap='bwr')
        # plt.title('Residual phase (rad):\n'
        #           + 'ground phase vs dm from ground WFS approximation slope\n'
        #           + 'dz={:.0f}m, r={:.2f}m, Rytov={:.2f}'.format(z[iz],r[ir], rytov_out))
        # plt.axis('equal')
        # plt.colorbar()
        # plt.show()
        
        # plt.pcolor(np.arange(-prN, prN + 1)*d_in,
        #             np.arange(-prN, prN + 1)*d_in,
        #             res_sg_avg[int(N/2) - prN
        #                               : int(N/2) + prN,
        #                               int(N/2) - prN
        #                               : int(N/2) + prN],
        #             shading='auto', snap=True,cmap='bwr')
        # plt.title('Residual phase (rad):\n'
        #           + 'altitude phase vs dm from ground averaged slope\n'
        #           + 'dz={:.0f}m, r={:.2f}m, Rytov={:.2f}'.format(z[iz],r[ir], rytov_out))
        # plt.axis('equal')
        # plt.colorbar()
        # plt.show()
        
        
        
        # plt.pcolor(np.arange(-prN, prN + 1)*d_in,
        #             np.arange(-prN, prN + 1)*d_in,
        #             res_sg_WFS[int(N/2) - prN
        #                               : int(N/2) + prN,
        #                               int(N/2) - prN
        #                               : int(N/2) + prN],
        #             shading='auto', snap=True,cmap='bwr')
        # plt.title('Residual phase (rad):\n'
        #           + 'altitude phase vs dm from ground WFS slope\n'
        #           + 'dz={:.0f}m, r={:.2f}m, Rytov={:.2f}'.format(z[iz],r[ir], rytov_out))
        # plt.axis('equal')
        # plt.colorbar()
        # plt.show()
        
        # plt.pcolor(np.arange(-prN, prN + 1)*d_in,
        #             np.arange(-prN, prN + 1)*d_in,
        #             res_sg_weight[int(N/2) - prN
        #                               : int(N/2) + prN,
        #                               int(N/2) - prN
        #                               : int(N/2) + prN],
        #             shading='auto', snap=True,cmap='bwr')
        # plt.title('Residual phase (rad):\n'
        #           + 'altitude phase vs dm from ground WFS approximation slope\n'
        #           + 'dz={:.0f}m, r={:.2f}m, Rytov={:.2f}'.format(z[iz],r[ir], rytov_out))
        # plt.axis('equal')
        # plt.colorbar()
        # plt.show()
        
        plt.pcolor(np.arange(-8,9)*d_in,
                    np.arange(-8,9)*d_in,
                    dm_out_avg[:16,:16],
                    shading='auto', snap=True)
        plt.colorbar()
        plt.vlines((np.arange(-0.5, 1.5))*d_in*lenslet_pixel,
                    ymin=-0.5*d_in*lenslet_pixel,
                    ymax=0.5*d_in*lenslet_pixel,
                    color='m')
        plt.hlines((np.arange(-0.5, 1.5))*d_in*lenslet_pixel,
                    xmin=-0.5*d_in*lenslet_pixel,
                    xmax=0.5*d_in*lenslet_pixel,
                    color='m')
        plt.quiver((np.arange(1))*d_in*lenslet_pixel,
                    (np.arange(1))*d_in*lenslet_pixel,
                    lenslet_slope_phase_out[:1,:1,0],
                    lenslet_slope_phase_out[:1,:1,1],
                    color='r')
        plt.title('DM phase (rad) from lenslet phase slope,\n'
                  + 'dz={:.0f}m, r={:.2f}m, Rytov={:.2f}'.format(z[iz],r[ir], rytov_out))
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()
        
        plt.pcolor(np.arange(-8,9)*d_in,
                    np.arange(-8,9)*d_in,
                    dm_out_WFS[:16,:16],
                    shading='auto', snap=True)
        plt.colorbar()
        plt.vlines((np.arange(-0.5, 1.5))*d_in*lenslet_pixel,
                    ymin=-0.5*d_in*lenslet_pixel,
                    ymax=0.5*d_in*lenslet_pixel,
                    color='m')
        plt.hlines((np.arange(-0.5, 1.5))*d_in*lenslet_pixel,
                    xmin=-0.5*d_in*lenslet_pixel,
                    xmax=0.5*d_in*lenslet_pixel,
                    color='m')
        plt.quiver((np.arange(1))*d_in*lenslet_pixel,
                    (np.arange(1))*d_in*lenslet_pixel,
                    projected_slope_WFS_out[:1,:1,0],
                    projected_slope_WFS_out[:1,:1,1],
                    color='r')
        plt.title('DM phase (rad) from WFS slope,\n'
                  + 'dz={:.0f}m, r={:.2f}m, Rytov={:.2f}'.format(z[iz],r[ir], rytov_out))
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()
        
        plt.pcolor(np.arange(-8,9)*d_in,
                    np.arange(-8,9)*d_in,
                    dm_out_weight[:16,:16],
                    shading='auto', snap=True)
        plt.colorbar()
        plt.vlines((np.arange(-0.5, 1.5))*d_in*lenslet_pixel,
                    ymin=-0.5*d_in*lenslet_pixel,
                    ymax=0.5*d_in*lenslet_pixel,
                    color='m')
        plt.hlines((np.arange(-0.5, 1.5))*d_in*lenslet_pixel,
                    xmin=-0.5*d_in*lenslet_pixel,
                    xmax=0.5*d_in*lenslet_pixel,
                    color='m')
        plt.quiver((np.arange(1))*d_in*lenslet_pixel,
                    (np.arange(1))*d_in*lenslet_pixel,
                    lenslet_weight_slope_phase_out[:1,:1,0],
                    lenslet_weight_slope_phase_out[:1,:1,1],
                    color='r')
        plt.title('DM phase (rad) from WFS approximation,\n'
                  + 'dz={:.0f}m, r={:.2f}m, Rytov={:.2f}'.format(z[iz],r[ir], rytov_out))
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()
    
        plt.pcolor(np.arange(-8,9)*d_in,
                    np.arange(-8,9)*d_in,
                    res_out_avg[:16,:16],
                    shading='auto', snap=True,cmap='bwr')
        plt.title('Residual phase (rad):\n'
                  + 'ground phase vs dm from ground averaged slope\n'
                  + 'dz={:.0f}m, r={:.2f}m, Rytov={:.2f}'.format(z[iz],r[ir], rytov_out))
        plt.axis('equal')
        plt.colorbar()
        plt.show()
    
        plt.pcolor(np.arange(-8,9)*d_in,
                    np.arange(-8,9)*d_in,
                    res_out_WFS[:16,:16],
                    shading='auto', snap=True,cmap='bwr')
        plt.title('Residual phase (rad):\n'
                  + 'ground phase vs dm from ground WFS slope\n'
                  + 'dz={:.0f}m, r={:.2f}m, Rytov={:.2f}'.format(z[iz],r[ir], rytov_out))
        plt.axis('equal')
        plt.colorbar()
        plt.show()
        
        plt.pcolor(np.arange(-8,9)*d_in,
                    np.arange(-8,9)*d_in,
                    res_out_weight[:16,:16],
                    shading='auto', snap=True,cmap='bwr')
        plt.title('Residual phase (rad):\n'
                  + 'ground phase vs dm from ground WFS approximation slope\n'
                  + 'dz={:.0f}m, r={:.2f}m, Rytov={:.2f}'.format(z[iz],r[ir], rytov_out))
        plt.axis('equal')
        plt.colorbar()
        plt.show()
        
        plt.pcolor(np.arange(-8,9)*d_in,
                    np.arange(-8,9)*d_in,
                    res_sg_avg[:16,:16],
                    shading='auto', snap=True,cmap='bwr')
        plt.title('Residual phase (rad):\n'
                  + 'altitude phase vs dm from ground averaged slope\n'
                  + 'dz={:.0f}m, r={:.2f}m, Rytov={:.2f}'.format(z[iz],r[ir], rytov_out))
        plt.axis('equal')
        plt.colorbar()
        plt.show()
        
        
        
        plt.pcolor(np.arange(-8,9)*d_in,
                    np.arange(-8,9)*d_in,
                    res_sg_WFS[:16,:16],
                    shading='auto', snap=True,cmap='bwr')
        plt.title('Residual phase (rad):\n'
                  + 'altitude phase vs dm from ground WFS slope\n'
                  + 'dz={:.0f}m, r={:.2f}m, Rytov={:.2f}'.format(z[iz],r[ir], rytov_out))
        plt.axis('equal')
        plt.colorbar()
        plt.show()
        
        plt.pcolor(np.arange(-8,9)*d_in,
                    np.arange(-8,9)*d_in,
                    res_sg_weight[:16,:16],
                    shading='auto', snap=True,cmap='bwr')
        plt.title('Residual phase (rad):\n'
                  + 'altitude phase vs dm from ground WFS approximation slope\n'
                  + 'dz={:.0f}m, r={:.2f}m, Rytov={:.2f}'.format(z[iz],r[ir], rytov_out))
        plt.axis('equal')
        plt.colorbar()
        plt.show()
        
        tb = time.time()
        t_plot_out = tb - ta
        print('time plot output analysis  = {:.6f}s'.format(t_plot_out))









# Rytov plot
for ir in np.arange(len(r)):
    plt.plot(z/1000., all_rytov_pixel[ir,:],
              label='r={:.0f}cm'.format(r[ir]*100), marker='o')
plt.legend()
plt.xlabel('z (km)')
plt.ylabel('Rytov number (unitless)')
plt.title('True Rytov number')
plt.xscale('log')
plt.yscale('log')
plt.show()
# for ir in np.arange(len(r)):
#     plt.plot(z/1000., all_rytov_pixel[ir,:],
#               label='r={:.0f}cm'.format(r[ir]*100), marker='o')
#     plt.legend()
#     plt.xlabel('z (km)')
#     plt.ylabel('Rytov number (unitless)')
#     plt.title('True Rytov number')
#     #plt.xscale('log')
#     #plt.yscale('log')
#     plt.show()


# # lenslet rytov
# for ir in np.arange(len(r)):
#     plt.plot(z/1000., all_rytov_lenslet[ir,:],
#               label='r={:.0f}cm'.format(r[ir]*100), marker='o')
# plt.legend()
# plt.xlabel('z (km)')
# plt.ylabel('Ln amplitude variance (unitless)')
# plt.title('(Fake) Rytov number')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
# # for ir in np.arange(len(r)):
# #     plt.plot(z/1000., all_rytov_lenslet[ir,:],
# #               label='r={:.0f}cm'.format(r[ir]*100), marker='o')
# #     plt.legend()
# #     plt.xlabel('z (km)')
# #     plt.ylabel('Ln amplitude variance (unitless)')
# #     plt.title('(Fake) Rytov number')
# #     #plt.xscale('log')
# #     #plt.yscale('log')
# #     plt.show()


# curiosity compare ground to sky
for ir in np.arange(len(r)):
    plt.plot(all_rytov_pixel[ir,:], rms_sky_to_ground[ir,:]*wvl/2./np.pi/1e-9,
              marker='o', ls='', label='r0={:.0f}cm'.format(r[ir]*100))
plt.legend()
plt.xlabel('Rytov number (unitless)')
plt.ylabel('RMS phase difference (nm)')
plt.title('Comparison of reconstructed phase\nbetween altitude and ground')
plt.xscale('log')
plt.yscale('log')
plt.show()
# for ir in np.arange(len(r)):
#     plt.plot(all_rytov_pixel[ir,:], rms_sky_to_ground[ir,:]*d_out/1e-9,
#               marker='o', ls='', label='r0={:.0f}cm'.format(r[ir]*100))
#     plt.legend()
#     plt.xlabel('Rytov number (unitless)')
#     plt.ylabel('RMS slope difference (nm)')
#     plt.title('Comparison between altitude and ground')
#     #plt.xscale('log')
#     #plt.yscale('log')
#     plt.show()


# compare sky and sky measurement
# this should be similar to usual simulation
plt.plot(r*100, rms_sky_to_sky_avg*wvl/2/np.pi/1.e-9,
              label='sky vs sky lenslet average', marker='x')
plt.plot(r*100, rms_sky_to_sky_WFS*wvl/2/np.pi/1.e-9,
              label='sky vs sky WFS', marker='o')
plt.plot(r*100,#0.0377#0.5304#0.134
         np.sqrt(0.134*(lenslet_pixel*d_in/r)**(5./3.))*wvl/2./np.pi/1e-9,
         label='Noll tip/tilt-removed residual')
plt.legend()
plt.xlabel('r (cm)')
plt.ylabel('rms phase difference (nm)')
plt.title('Comparison of reconstructed phase\nat altitude and altitude measurement')
plt.xscale('log')
plt.yscale('log')
plt.show()


# compare sky to ground measurements
# this should be similar to what actually happen when we do ao
# where we think what we measure on ground is on the sky
# and use it to fit dm
# for ir in np.arange(len(r)):
#     plt.plot(np.append(0,z)/1000.,
#              np.append(rms_sky_to_sky_avg[ir],
#                        rms_sky_to_ground_avg[ir,:])*d_out/1.e-9,
#               label='r={:.0f}cm'.format(r[ir]*100), marker='o')
# plt.legend()
# plt.xlabel('z (km)')
# plt.ylabel('rms slope difference (nm)')
# plt.title('Comparison of altitude and ground lenslet average')
# #plt.xscale('log')
# #plt.yscale('log')
# plt.show()
# for ir in np.arange(len(r)):
#     plt.plot(np.append(0,z)/1000.,
#              np.append(rms_sky_to_sky_avg[ir],
#                        rms_sky_to_ground_avg[ir,:])*d_out/1.e-9,
#               label='r={:.0f}cm'.format(r[ir]*100), marker='o')
#     plt.legend()
#     plt.xlabel('z (km)')
#     plt.ylabel('rms slope difference (nm)')
#     plt.title('Comparison of altitude and ground lenslet average')
#     #plt.xscale('log')
#     #plt.yscale('log')
#     plt.show()

# for ir in np.arange(len(r)):
#     plt.plot(np.append(0,z)/1000.,
#              np.append(rms_sky_to_sky_WFS[ir],
#                        rms_sky_to_ground_WFS[ir,:])*d_out/1.e-9,
#               label='r={:.0f}cm'.format(r[ir]*100), marker='o')
# plt.legend()
# plt.xlabel('z (km)')
# plt.ylabel('rms slope difference (nm)')
# plt.title('Comparison of altitude and ground WFS measurement')
# #plt.xscale('log')
# #plt.yscale('log')
# plt.show()
# for ir in np.arange(len(r)):
#     plt.plot(np.append(0,z)/1000.,
#              np.append(rms_sky_to_sky_WFS[ir],
#                        rms_sky_to_ground_WFS[ir,:])*d_out/1.e-9,
#               label='r={:.0f}cm'.format(r[ir]*100), marker='o')
#     plt.legend()
#     plt.xlabel('z (km)')
#     plt.ylabel('rms slope difference (nm)')
#     plt.title('Comparison of altitude and ground WFS measurement')
#     #plt.xscale('log')
#     #plt.yscale('log')
#     plt.show()

# for ir in np.arange(len(r)):
#     plt.plot(np.append(0,z)/1000.,
#              np.append(rms_sky_to_sky_weight[ir],
#                        rms_sky_to_ground_weight[ir,:])*d_out/1.e-9,
#               label='r={:.0f}cm'.format(r[ir]*100), marker='o')
# plt.legend()
# plt.xlabel('z (km)')
# plt.ylabel('rms slope difference (nm)')
# plt.title('Comparison of altitude and ground WFS approximation')
# #plt.xscale('log')
# #plt.yscale('log')
# plt.show()
# for ir in np.arange(len(r)):
#     plt.plot(np.append(0,z)/1000.,
#              np.append(rms_sky_to_sky_weight[ir],
#                        rms_sky_to_ground_weight[ir,:])*d_out/1.e-9,
#               label='r={:.0f}cm'.format(r[ir]*100), marker='o')
#     plt.legend()
#     plt.xlabel('z (km)')
#     plt.ylabel('rms slope difference (nm)')
#     plt.title('Comparison of altitude and ground WFS approximation')
#     #plt.xscale('log')
#     #plt.yscale('log')
#     plt.show()

for ir in np.arange(len(r)):
    plt.plot(np.append(0,z)/1000.,
             np.append(rms_sky_to_sky_avg[ir],
                       rms_sky_to_ground_avg[ir,:])*wvl/2/np.pi/1.e-9,
              label='sky to ground lenslet average', marker='<')
    plt.plot(np.append(0,z)/1000.,
             np.append(rms_sky_to_sky_WFS[ir],
                       rms_sky_to_ground_WFS[ir,:])*wvl/2/np.pi/1.e-9,
              label='sky to ground WFS', marker='>')
    plt.plot(np.append(0,z)/1000.,
             np.append(rms_sky_to_sky_avg[ir],
                       rms_sky_to_ground_weight[ir,:])*wvl/2/np.pi/1.e-9,
              label='sky to ground WFS approximation', marker='^')
#plt.legend()
plt.xlabel('z (km)')
plt.ylabel('rms phase difference (nm)')
plt.title('Comparison reconstructed phase\nof altitude and ground measurements, '
          + 'r={:.0f}cm'.format(r[ir]*100))
#plt.xscale('log')
#plt.yscale('log')
#plt.show()

# compare the ground to ground measurements
# check accuracy
# for ir in np.arange(len(r)):
#     plt.plot(np.append(0, z)/1000.,
#              np.append(rms_sky_to_sky_avg[ir],
#                        rms_ground_to_ground_avg[ir,:])*d_out/1.e-9,
#               label='r={:.0f}cm'.format(r[ir]*100), marker='o')
# plt.legend()
# plt.xlabel('z (km)')
# plt.ylabel('rms slope difference (nm)')
# plt.title('Comparison of ground and ground lenslet average')
# #plt.xscale('log')
# #plt.yscale('log')
# plt.show()

# for ir in np.arange(len(r)):
#     plt.plot(np.append(0,z)/1000.,
#              np.append(rms_sky_to_sky_WFS[ir],
#                        rms_ground_to_ground_WFS[ir,:])*d_out/1.e-9,
#               label='r={:.0f}cm'.format(r[ir]*100), marker='o')
# plt.legend()
# plt.xlabel('z (km)')
# plt.ylabel('rms slope difference (nm)')
# plt.title('Comparison of ground and ground WFS measurement')
# #plt.xscale('log')
# #plt.yscale('log')
# plt.show()

# for ir in np.arange(len(r)):
#     plt.plot(np.append(0,z)/1000.,
#              np.append(rms_sky_to_sky_weight[ir],
#                        rms_ground_to_ground_weight[ir,:])*d_out/1.e-9,
#               label='r={:.0f}cm'.format(r[ir]*100), marker='o')
# plt.legend()
# plt.xlabel('z (km)')
# plt.ylabel('rms slope difference (nm)')
# plt.title('Comparison of ground and ground WFS approximation')
# #plt.xscale('log')
# #plt.yscale('log')
# plt.show()

for ir in np.arange(len(r)):
    plt.plot(np.append(0,z)/1000.,
             np.append(rms_sky_to_sky_avg[ir],
                       rms_ground_to_ground_avg[ir,:])*wvl/2/np.pi/1.e-9,
              label='ground to ground lenslet average', marker='<')
    plt.plot(np.append(0,z)/1000.,
             np.append(rms_sky_to_sky_WFS[ir],
                       rms_ground_to_ground_WFS[ir,:])*wvl/2/np.pi/1.e-9,
              label='ground to ground WFS', marker='>')
    plt.plot(np.append(0,z)/1000.,
             np.append(rms_sky_to_sky_avg[ir],
                       rms_ground_to_ground_weight[ir,:])*wvl/2/np.pi/1.e-9,
              label='ground to ground WFS approximation', marker='^')
    plt.legend()
    plt.xlabel('z (km)')
    plt.ylabel('rms phase difference (nm)')
    plt.title('Comparison of reconstructed phase\nfrom ground and ground measurements, '
              + 'r={:.0f}cm'.format(r[ir]*100))
    plt.xscale('log')
    plt.yscale('log')
    plt.show()



# compare wfs and wfs approximation
# plt.plot(r*100, rms_sky_WFS_to_sky_weight*5e-3/1.e-9,
#               label='sky WFS vs sky WFS approximation', marker='o')
# plt.legend()
# plt.xlabel('r (cm)')
# plt.ylabel('rms slope difference (nm)')
# plt.title('Comparison of WFS and WFS approximation at altitude')
# #plt.xscale('log')
# #plt.yscale('log')
# plt.show()
for ir in np.arange(len(r)):
    plt.plot(np.append(0,z)/1000.,
             np.append(rms_sky_avg_to_sky_WFS[ir],
                       rms_ground_WFS_to_ground_weight[ir,:])*wvl/2/np.pi/1.e-9,
              label='r={:.0f}cm'.format(r[ir]*100), marker='o')
plt.legend()
plt.xlabel('z (km)')
plt.ylabel('rms phase difference (nm)')
plt.title('Comparison of reconstructed phase from\n altitude WFS measurement and ground WFS approximation')
plt.xscale('log')
plt.yscale('log')
plt.show()




# for ir in np.arange(len(r)):
#     plt.plot(z/1000., rms_ground_avg_to_ground_WFS[ir,:]*d_out/1.e-9,
#              label='r0={:.0f}cm'.format(r[ir]*100), marker='o')
# plt.legend()
# plt.xlabel('z (km)')
# plt.ylabel('rms slope difference (nm)')
# plt.title('Comparison of ground lenslet average and ground WFS measurement')
# #plt.xscale('log')
# #plt.yscale('log')
# plt.show()

# for ir in np.arange(len(r)):
#     plt.plot(z/1000., rms_ground_avg_to_ground_weight[ir,:]*d_out/1.e-9,
#               label='r0={:.0f}cm'.format(r[ir]*100), marker='o')
# plt.legend()
# plt.xlabel('z (km)')
# plt.ylabel('rms slope difference (nm)')
# plt.title('Comparison of ground lenslet average and ground WFS approximation')
# #plt.xscale('log')
# #plt.yscale('log')
# plt.show()




# plt.plot((rms_ground_to_ground_avg.flatten()
#           - rms_ground_to_ground_WFS.flatten())/std_slope_out.flatten(),
#          marker='o', ls='',
#          label='A: avg, B: WFS')
# plt.plot((rms_ground_to_ground_avg.flatten()
#           - rms_ground_to_ground_weight.flatten())/std_slope_out.flatten(),
#          marker='o', ls='',
#          label='A: avg, B: WFS approx')
# plt.plot((rms_ground_to_ground_weight.flatten()
#           - rms_ground_to_ground_WFS.flatten())/std_slope_out.flatten(),
#          marker='o', ls='',
#          label='A: WFS approx, B: WFS')
# plt.plot(np.arange(len(z)*len(r)),np.zeros(len(z)*len(r)),marker='', ls='-')
# plt.xlabel('arbitrary element index')
# plt.ylabel('RMS difference / ground RMS')
# plt.title('(RMS(A-C)-RMS(B-C))/std(C); C=ground')
# plt.legend()
# plt.show()







# ## Checking if measuring PSD of a cropped screen is valid

# N = 256
# NN = N*2
# delta = 0.005
# r0 = 0.08
# L0 = 20
# l0 = 1.5e-10
# del_ff = 1/NN/delta
# del_f = 1/N/delta

# PSD_theory_big = mps.psd_function(r0, NN, delta, L0, l0)
# PSD_theory_small = mps.psd_function(r0, N, delta, L0, l0)

# # phase_big = PSD_to_phase(PSD_theory_big, del_ff)
# # phase_small = PSD_to_phase(PSD_theory_small, del_f)
# # phase_big = mprop.crop2(ft_phase_screen(r0, NN*zpf, delta, L0, l0),zpf)
# # phase_small = mprop.crop2(ft_phase_screen(r0, N*zpf, delta, L0, l0),zpf)
# phase_big = ft_sh_phase_screen(r0, NN, delta, L0, l0)
# phase_small = ft_sh_phase_screen(r0, N, delta, L0, l0)

# phase_crop = mprop.crop2(phase_big,int(NN/N))

# PSD_full_big = get_PSD(phase_big, delta)
# PSD_full_small = get_PSD(phase_small, delta)
# PSD_crop = get_PSD(phase_crop, delta)

# PSD_full_big_auto_pad = pick_even_index(get_PSD(mirror_pad2(phase_big), delta))
# PSD_full_small_auto_pad = pick_even_index(get_PSD(mirror_pad2(phase_small), delta))
# PSD_crop_auto_pad = pick_even_index(get_PSD(mirror_pad2(phase_crop), delta))

# plt.plot((np.arange(int(NN/2-1))+1)*del_ff,
#           PSD_theory_big[int(NN/2),int(NN/2)+1:],
#           ls='-',marker='',
#           label='Theoretical PSD of full big screen')
# plt.plot((np.arange(int(N/2-1))+1)*del_f,
#           PSD_theory_small[int(N/2)+1,int(N/2)+1:],
#           ls='-',marker='',
#           label='Theoretical PSD of full small screen')
# # plt.plot((np.arange(int(NN/2-1))+1)*del_ff,
# #           PSD_full_big[int(NN/2),int(NN/2)+1:],
# #           ls='', marker='.', markersize=2,
# #           label='Measured PSD of full big screen')
# # plt.plot((np.arange(int(N/2-1))+1)*del_f,
# #           PSD_full_small[int(N/2),int(N/2)+1:],
# #           ls='',marker='.', markersize=2,
# #           label='Measured PSD of full small screen')
# # plt.plot((np.arange(int(N/2-1))+1)*del_f,
# #           PSD_crop[int(N/2),int(N/2)+1:],
# #           ls='',marker='.', markersize=2,
# #           label='Measured PSD of cropped big screen')
# plt.plot((np.arange(int(NN/2-1))+1)*del_ff,
#           PSD_full_big_auto_pad[int(NN/2),int(NN/2)+1:],
#           ls='',marker='.', markersize=2,
#           label='Measured PSD of small screen with auto pad')
# plt.plot((np.arange(int(N/2-1))+1)*del_f,
#           PSD_full_small_auto_pad[int(N/2),int(N/2)+1:],
#           ls='',marker='.', markersize=2,
#           label='Measured PSD of small screen with auto pad')
# plt.plot((np.arange(int(N/2-1))+1)*del_f,
#           PSD_crop_auto_pad[int(N/2),int(N/2)+1:],
#           ls='',marker='.', markersize=2,
#           label='Measured PSD of cropped big screen with auto pad')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Spatial frequency (1/m)')
# plt.ylabel('PSD (m$^2$)')
# plt.legend()
# plt.title('Pixel size = {:.0f}mm, L0 = {:.0f}m, '.format(delta*1e3,L0)
#           + 'r0 = {:.0f}cm, full N = {:d}'.format(r0*1e2,NN))
# plt.show()




# a_big,b_big,c_big=mps.structure_function(phase_big)
# a_small,b_small,c_small=mps.structure_function(phase_small)
# a_crop,b_crop,c_crop=mps.structure_function(phase_crop)

# plt.plot(a_big*delta, 6.88 * (a_big*delta/r0)**(5./3.), label='Kolmogorov Model')
# plt.errorbar(a_big*delta, b_big, yerr=c_big, label='Measured full big screen')
# plt.errorbar(a_small*delta, b_small, yerr=c_small, 
# label='Measured full small screen')
# plt.errorbar(a_crop*delta, b_crop, yerr=c_crop, label='Measured cropped big screen')
# plt.xlabel('Separation distance (m)')
# plt.ylabel('Structure function (rad^2)')
# plt.xlim(np.nanmin(a_big*delta*0.9), np.nanmax(a_big*delta*1.1))
# plt.ylim(np.nanmin(b_big*0.9), np.nanmax(b_big*1.1))
# plt.xscale('log')
# plt.yscale('log') 
# plt.legend()
# plt.title('Comparison of phase structure function'
#           + ' between measurement and theory\n'
#           + 'Pixel size = {:.0f}mm, L0 = {:.0f}m, '.format(delta*1e3,L0)
#           + 'r0 = {:.0f}cm, full N = {:d}'.format(r0*1e2,NN))
# plt.show()

# plt.imshow(phase_big)
# plt.title('phase big')
# plt.show()
# plt.imshow(phase_crop)
# plt.title('phase crop')
# plt.show()


# # case 1 regular
# PSD=mps.psd_function(r0, N, delta, L0, l0)
# phase=PSD_to_phase(PSD, del_f)
# plt.imshow(phase)
# plt.colorbar()
# plt.title('regular')
# plt.show()

# # case 2 oversize
# PSD=mps.psd_function(r0, N*4, delta, L0, l0)
# phase=mprop.crop2(PSD_to_phase(PSD, del_f/4.),4)
# plt.imshow(phase)
# plt.colorbar()
# plt.title('oversize')
# plt.show()

# # case 3 zeropad
# PSD=mps.psd_function(r0, N, delta, L0, l0)
# phase=mprop.crop2(PSD_to_phase(mprop.zero_pad2(PSD,4), del_f/4.),4)
# plt.imshow(phase)
# plt.colorbar()
# plt.title('zero pad')
# plt.show()

# # case 4 oversize and zeropad
# PSD=mps.psd_function(r0, N*4, delta, L0, l0)
# phase=mprop.crop2(PSD_to_phase(mprop.zero_pad2(PSD,4), del_f/4./4.),4)
# plt.imshow(phase)
# plt.colorbar()
# plt.title('oversize & zero pad')
# plt.show()

###############################################################################
########################################################################

TB = time.time()
t_all = TB - TA
print('time all = {:.6f}s'.format(t_all))

###############################################################################
# End of File ##########################################################
###############################################################################