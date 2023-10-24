
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
from aotools.opticalpropagation import angularSpectrum

import my_propagation as mprop
import my_WFS as mwfs
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

N = 1024
d_in = 5e-3
L0 = 20.
l0 = 1.5e-10
wvl = 500e-9
d_out = d_in
z = np.array([1,5,10,25], dtype=float)*1e3
lenslet_pixel = 16
D_lenslet = 150e-6
r = np.array([0.50, 0.20, 0.10, 0.05], dtype=float)

#
# Independent constants
#
pf_lenslet = 8
pf = 2
prN = 64 # 128
I_max = 0.3
I_min = -1.

#
# Dependent constants
#
num_lenslet = int(N/lenslet_pixel)
d_lenslet = D_lenslet/lenslet_pixel
f_lenslet = 2. * d_lenslet**2 * lenslet_pixel / wvl
NN = pf*N
prNN = int(prN/lenslet_pixel)
I_norm = lenslet_pixel**2./4.
d_detector = d_lenslet
#k = 2.*np.pi/wvl                    # float; rad/m; Wave number.
#cn2dz=r**(-5./3.)/0.423/k**2        # float; Turbulence strength.

#
# Declaration field
#
amp_in_big = np.ones((NN,NN), dtype=float)          # Input amplitude.
phase_in_big = np.zeros((NN, NN), dtype=float)       # Input phase (rad).
U_in_big = np.zeros((NN,NN), dtype=complex)         # Input complex amp.

amp_in = np.ones((N,N), dtype=float)          # Input amplitude.
phase_in = np.zeros((N, N), dtype=float)       # Input phase (rad).
U_in = np.zeros((N,N), dtype=complex)         # Input complex amp.

#
# Checking propagation parametres #
#

# Propagation between atmospheric layers
check = True
check_condition = mprop.prop_param_check(N, d_in, d_out, wvl, z, pf,
                                        scin_blob=True)
for i in check_condition:
    check = check * i
if check == False:
    print("One of the propagation condition for atm layer is failing.")
    quit()

# Propagation through SHWFS
check = True
check_condition = mprop.prop_param_check(lenslet_pixel, d_lenslet, d_lenslet,
                                        wvl, [f_lenslet], pf_lenslet)
for i in check_condition:
    check = check * i
if check == False:
    print("One of the propagation condition for lenslet is failing.")
    quit()

#
# Calculations
#

for ir in np.arange(len(r)):
    # Initialization #
    phase_in_big = ft_sh_phase_screen(r[ir], NN, d_in, L0, l0)
    
    # Check another condition #
    slope_phase_in_big = mwfs.get_slope_no_wrap(phase_in_big, d_in, wvl)
    if type(slope_phase_in_big) == bool:
        print('Input phase screen is not Nyquist sampled.')
        print('It is too dense.')
        print('r={:.2f}m, d_in={:.2f}m'.format(r, d_in))
        print('For this r, d_in must be smaller.')
        quit()

    U_in_big = amp_in_big*np.exp(1j*phase_in_big)

    #
    # Reporting source conditions #
    #
    U_in = mprop.crop2(U_in_big, pf)
    phase_in = np.angle(U_in)
    I_in = np.abs(U_in)**2
    slope_phase_in = mwfs.get_slope(phase_in, d_in, wvl)
    size_slope_phase_in = np.sqrt(slope_phase_in[:,:,0]**2
                                  + slope_phase_in[:,:,1]**2)
    rytov_in = ((np.mean(I_in**2) / np.mean(I_in)**2) - 1.) / 4.

    #
    # Average #
    #
    lenslet_I_in = mwfs.get_val_lenslet(I_in, lenslet_pixel)
    lenslet_slope_phase_in = (mwfs.get_vec_lenslet(slope_phase_in,
                                                   lenslet_pixel))
    size_lenslet_slope_phase_in = np.sqrt(
        lenslet_slope_phase_in[ : , : , 0]**2
        + lenslet_slope_phase_in[ : , : , 1]**2)
    rytov_lenslet_in = ((np.mean(lenslet_I_in**2) 
                         / np.mean(lenslet_I_in)**2) - 1.) / 4.
    #
    # SHWFS cheat by weighting slope with intensity #
    #
    weight_slope_phase_in = np.zeros((N,N,2), dtype=float)
    weight_slope_phase_in[:,:,0] = slope_phase_in[:,:,0]*I_in
    weight_slope_phase_in[:,:,1] = slope_phase_in[:,:,1]*I_in
    lenslet_weight_slope_phase_in = mwfs.get_vec_lenslet(
        weight_slope_phase_in, lenslet_pixel)
    lenslet_weight_slope_phase_in[:,:,0] = (
        lenslet_weight_slope_phase_in[:,:,0]/lenslet_I_in)
    lenslet_weight_slope_phase_in[:,:,1] = (
        lenslet_weight_slope_phase_in[:,:,1]/lenslet_I_in)
    size_lenslet_weight_slope_phase_in = np.sqrt(
        lenslet_weight_slope_phase_in[ : , : , 0]**2
        + lenslet_weight_slope_phase_in[ : , : , 1]**2)

    #
    # Proper SHWFS #
    #
    U_WFS_detector_in = mwfs.propagate_lenslet(U_in, wvl, d_lenslet,
                                                f_lenslet, lenslet_pixel,
                                                d_lenslet, pf_lenslet,)
    I_WFS_detector_in = np.abs(U_WFS_detector_in)**2
    lenslet_I_WFS_detector_in = mwfs.get_val_lenslet(
        I_WFS_detector_in, lenslet_pixel)
    slope_WFS_in = mwfs.get_SHWFS_slope(U_WFS_detector_in, lenslet_pixel,
                                    d_detector, f_lenslet)
    projected_slope_WFS_in = slope_WFS_in *d_detector / d_in
    size_projected_slope_WFS_in = np.sqrt(projected_slope_WFS_in[:,:,0]**2
                                          + projected_slope_WFS_in[:,:,1]**2)
    rytov_WFS_in = (
        (np.mean(lenslet_I_WFS_detector_in**2) / np.mean(
            lenslet_I_WFS_detector_in)**2) - 1.) / 4.

    #
    # Plot input analysis #
    #
    # Actual value

    plt.pcolor(np.arange(-prN, prN + 1)*d_in,
               np.arange(-prN, prN + 1)*d_in,
               phase_in[int(N/2) - prN
                                 : int(N/2) + prN,
                                 int(N/2) - prN
                                 : int(N/2) + prN],
               shading='auto', snap=True)
    plt.colorbar()
    plt.vlines((np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
               ymin=-prNN*d_in*lenslet_pixel,
               ymax=prNN*d_in*lenslet_pixel,
               color='m')
    plt.hlines((np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
               xmin=-prNN*d_in*lenslet_pixel,
               xmax=prNN*d_in*lenslet_pixel,
               color='m')
    plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
               (np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
               lenslet_slope_phase_in[int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN,
                                       int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN, 0],
               lenslet_slope_phase_in[int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN,
                                       int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN, 1],
               color='r')
    plt.title('Phase (rad) over lenslet phase slope,\n'
              + 'dz=0m, r={:.2f}m, Rytov={:.2f}'.format(r[ir], rytov_in))
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()

    # Average
    plt.plot(lenslet_I_in.flatten(), size_lenslet_slope_phase_in.flatten(),
             ls='', marker=',')
    plt.xscale('log')
    plt.xlabel('Intensity')
    plt.ylabel('Slope size (rad)')
    plt.title('Lenslet intensity vs phase slope size (rad),\n'
              + 'dz=0m, r={:.2f}m,\nRytov={:.2f}, lenslet Rytov={:.2f}'.format(
                  r[ir], rytov_in, rytov_lenslet_in))
    plt.show()

    plt.pcolor((np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
               (np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
               size_lenslet_slope_phase_in[int(num_lenslet/2) - prNN
                                            : int(num_lenslet/2) + prNN,
                                            int(num_lenslet/2) - prNN
                                            : int(num_lenslet/2) + prNN],
               shading='auto', snap=True)
    plt.colorbar()
    plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
               (np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
               lenslet_slope_phase_in[int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN,
                                       int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN, 0],
               lenslet_slope_phase_in[int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN,
                                       int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN, 1],
               color='r')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Lenslet phase slope size (rad),\n' 
              + 'dz=0m, r={:.2f}m,\n'.format(r[ir])
              + 'Rytov={:.2f}, lenslet Rytov={:.2f}'.format(
                  rytov_in, rytov_lenslet_in))
    plt.show()

    # cheat SHWFS #
    plt.pcolor(np.arange(-prNN, prNN + 1)*d_in*lenslet_pixel,
               np.arange(-prNN, prNN + 1)*d_in*lenslet_pixel,
               np.log(lenslet_I_in[int(num_lenslet/2) - prNN :
                                                int(num_lenslet/2) + prNN,
                     int(num_lenslet/2) - prNN : int(num_lenslet/2) + prNN])
                   /np.log(10.),
               shading='auto', vmin=I_min, vmax=I_max, snap=True)
    plt.colorbar()
    plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
               (np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
               lenslet_weight_slope_phase_in[int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN,
                                       int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN, 0],
               lenslet_weight_slope_phase_in[int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN,
                                       int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN, 1],
               color='r')
    plt.axis('equal')
    plt.xlabel('x (pixel)')
    plt.ylabel('y (pixel)')
    plt.title('Lenslet log10 intensity over '
              +'weighted phase slope,\n'
              + 'dz=0m, r={:.2f}m, '.format(r[ir])
              + 'Rytov={:.2f}, '.format(rytov_in)
              + 'lenslet Rytov={:.2f},\n'.format(rytov_lenslet_in)
              + 'max intensity {:.2f}, '.format(
                  lenslet_I_in.max())
              + 'min intensity {:.2f}'.format(
                  lenslet_I_in.min()))
    plt.show()

    plt.pcolor((np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
               (np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
               size_lenslet_weight_slope_phase_in[int(num_lenslet/2) - prNN
                                            : int(num_lenslet/2) + prNN,
                                            int(num_lenslet/2) - prNN
                                            : int(num_lenslet/2) + prNN],
               shading='auto', snap=True)
    plt.colorbar()
    plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
               (np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
               lenslet_weight_slope_phase_in[int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN,
                                       int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN, 0],
               lenslet_weight_slope_phase_in[int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN,
                                       int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN, 1],
               color='r')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Lenslet weighted phase slope (rad),\n'
              + 'dz=0m, r={:.2f}m,\n'.format(r[ir])
              +'Rytov={:.2f}, lenslet Rytov={:.2f}'.format(
                  rytov_in, rytov_lenslet_in))
    plt.show()

    # proper SHWFS #
    plt.plot(lenslet_I_WFS_detector_in.flatten(),
             size_projected_slope_WFS_in.flatten(),
             ls='', marker=',')
    plt.xscale('log')
    plt.xlabel('Intensity')
    plt.ylabel('Slope size (rad)')
    plt.title('WFS intensity vs phase slope size (rad),\n'
              + 'dz=0m, r={:.2f}m,\nRytov={:.2f}, WFS Rytov={:.2f}'.format(
                  r[ir], rytov_in, rytov_WFS_in))
    plt.show()

    plt.pcolor(np.arange(-prN, prN + 1)*d_in,
               np.arange(-prN, prN + 1)*d_in,
               np.log(I_WFS_detector_in[int(N/2) - prN : int(N/2) + prN,
                     int(N/2) - prN : int(N/2) + prN]/I_norm)/np.log(10.),
               shading='auto', vmin=I_min, vmax=I_max, snap=True)
    plt.colorbar()
    plt.vlines((np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
               ymin=-prNN*d_in*lenslet_pixel,
               ymax=prNN*d_in*lenslet_pixel,
               color='m')
    plt.hlines((np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
               xmin=-prNN*d_in*lenslet_pixel,
               xmax=prNN*d_in*lenslet_pixel,
               color='m')
    plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
               (np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
               projected_slope_WFS_in[int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN,
                                       int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN, 0],
               projected_slope_WFS_in[int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN,
                                       int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN, 1],
               color='r')
    plt.axis('equal')
    plt.xlabel('x (pixel)')
    plt.ylabel('y (pixel)')
    plt.title('WFS log10 image over phase slope,\n'
              + 'dz=0m, r={:.2f}m, '.format(r[ir])
              + 'Rytov={:.2f}, '.format(rytov_in)
              + 'WFS Rytov={:.2f},\n'.format(rytov_WFS_in)
              + 'max intensity {:.2f}, '.format(
                  lenslet_I_WFS_detector_in.max())
              + 'min intensity {:.2f}'.format(
                  lenslet_I_WFS_detector_in.min()))
    plt.show()

    plt.pcolor(np.arange(-prNN, prNN + 1)*d_in*lenslet_pixel,
               np.arange(-prNN, prNN + 1)*d_in*lenslet_pixel,
               np.log(lenslet_I_WFS_detector_in[int(num_lenslet/2) - prNN :
                                                int(num_lenslet/2) + prNN,
                     int(num_lenslet/2) - prNN : int(num_lenslet/2) + prNN])
                   /np.log(10.),
               shading='auto', vmin=I_min, vmax=I_max, snap=True)
    plt.colorbar()
    plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
               (np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
               projected_slope_WFS_in[int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN,
                                       int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN, 0],
               projected_slope_WFS_in[int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN,
                                       int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN, 1],
               color='r')
    plt.axis('equal')
    plt.xlabel('x (pixel)')
    plt.ylabel('y (pixel)')
    plt.title('WFS log10 total intensity over phase slope,\n'
              + 'dz=0m, r={:.2f}m, '.format(r[ir])
              + 'Rytov={:.2f}, '.format(rytov_in)
              + 'WFS Rytov={:.2f},\n'.format(rytov_WFS_in)
              + 'max intensity {:.2f}, '.format(
                  lenslet_I_WFS_detector_in.max())
              + 'min intensity {:.2f}'.format(
                  lenslet_I_WFS_detector_in.min()))
    plt.show()

    plt.pcolor((np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
               (np.arange(-prNN, prNN + 1))*d_in*lenslet_pixel,
               size_projected_slope_WFS_in[int(num_lenslet/2) - prNN
                                            : int(num_lenslet/2) + prNN,
                                            int(num_lenslet/2) - prNN
                                            : int(num_lenslet/2) + prNN],
               shading='auto', snap=True)
    plt.colorbar()
    plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
               (np.arange(-prNN, prNN) + 0.5)*d_in*lenslet_pixel,
               projected_slope_WFS_in[int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN,
                                       int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN, 0],
               projected_slope_WFS_in[int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN,
                                       int(num_lenslet/2) - prNN
                                       : int(num_lenslet/2) + prNN, 1],
               color='r')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('WFS phase slope (rad),\n'
              + 'dz=0m, r={:.2f}m,\n'.format(r[ir])
              + 'Rytov={:.2f}, WFS Rytov={:.2f}'.format(
                  rytov_in, rytov_WFS_in))
    plt.show()

    print(rytov_in, rytov_lenslet_in, rytov_WFS_in)

    #
    # Propagations #
    #
    for iz in np.arange(len(z)):
        U_out_big = angularSpectrum(U_in_big, wvl, d_in, d_out, z[iz])
        #
        # Direct output
        #
        U_out = mprop.crop2(U_out_big, pf)
        phase_out = np.angle(U_out)
        I_out = np.abs(U_out)**2
        slope_phase_out = mwfs.get_slope(phase_out, d_out, wvl)
        size_slope_phase_out = np.sqrt(slope_phase_out[:,:,0]**2
                                       + slope_phase_out[:,:,1]**2)
        rytov_out = ((np.mean(I_out**2) / np.mean(I_out)**2) - 1.) / 4.

        #
        # Averaging lenslet
        #
        lenslet_I_out = mwfs.get_val_lenslet(I_out, lenslet_pixel)
        lenslet_slope_phase_out = mwfs.get_vec_lenslet(
            slope_phase_out, lenslet_pixel)
        size_lenslet_slope_phase_out = np.sqrt(
            lenslet_slope_phase_out[ : , : , 0]**2
            + lenslet_slope_phase_out[ : , : , 1]**2)
        rytov_lenslet_out = ((np.mean(lenslet_I_out**2) 
                              / np.mean(lenslet_I_out)**2) - 1.) / 4.

        #
        # SHWFS cheat by weighting slope with intensity #
        #
        weight_slope_phase_out = np.zeros((N,N,2), dtype=float)
        weight_slope_phase_out[:,:,0] = slope_phase_out[:,:,0]*I_out
        weight_slope_phase_out[:,:,1] = slope_phase_out[:,:,1]*I_out
        lenslet_weight_slope_phase_out = mwfs.get_vec_lenslet(
            weight_slope_phase_out, lenslet_pixel)
        lenslet_weight_slope_phase_out[:,:,0] = (
            lenslet_weight_slope_phase_out[:,:,0]/lenslet_I_out)
        lenslet_weight_slope_phase_out[:,:,1] = (
            lenslet_weight_slope_phase_out[:,:,1]/lenslet_I_out)
        size_lenslet_weight_slope_phase_out = np.sqrt(
            lenslet_weight_slope_phase_out[ : , : , 0]**2
            + lenslet_weight_slope_phase_out[ : , : , 1]**2)

        #
        # Proper SHWFS
        #
        U_WFS_detector_out = mwfs.propagate_lenslet(U_out, wvl, d_lenslet,
                                               f_lenslet, lenslet_pixel,
                                               d_lenslet, pf_lenslet)
        I_WFS_detector_out = np.abs(U_WFS_detector_out)**2
        lenslet_I_WFS_detector_out = mwfs.get_val_lenslet(
            I_WFS_detector_out, lenslet_pixel)
        slope_WFS_out = mwfs.get_SHWFS_slope(U_WFS_detector_out, lenslet_pixel,
                                               d_detector, f_lenslet)
        projected_slope_WFS_out = slope_WFS_out * d_detector / d_out
        size_projected_slope_WFS_out = np.sqrt(
            projected_slope_WFS_out[:,:,0]**2
            + projected_slope_WFS_out[:,:,1]**2)
        rytov_WFS_out = (
            (np.mean(lenslet_I_WFS_detector_out**2) / np.mean(
                lenslet_I_WFS_detector_out)**2) - 1.) / 4.

        #
        # Plot output analysis
        #
        # Actual values
        plt.plot(I_out.flatten(), size_slope_phase_out.flatten(),
                  ls='', marker=',')
        plt.xscale('log')
        plt.xlabel('Intensity')
        plt.ylabel('Slope size (rad)')
        plt.title('Intensity vs phase slope size (rad),\n'
                  + 'dz={:.2f}m, r={:.2f}m, Rytov={:.2f},'.format(
                      z[iz], r[ir], rytov_in))
        plt.show()

        plt.pcolor(np.arange(-prN, prN)*d_out,
                    np.arange(-prN, prN)*d_out,
                    np.log(I_out[int(N/2) - prN : int(N/2) + prN,
                              int(N/2) - prN : int(N/2) + prN])/np.log(10),
                    shading='auto', vmin=I_min, vmax=I_max)
        plt.colorbar()
        plt.vlines((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
                   ymin=-prNN*d_out*lenslet_pixel,
                   ymax=prNN*d_out*lenslet_pixel,
                   color='m')
        plt.hlines((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
                   xmin=-prNN*d_out*lenslet_pixel,
                   xmax=prNN*d_out*lenslet_pixel,
                   color='m')
        plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
                   (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
                   lenslet_slope_phase_out[int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN,
                                           int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN, 0],
                   lenslet_slope_phase_out[int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN,
                                           int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN, 1],
                   color='r')
        plt.title('Log10 intensity over lenslet phase slope,\n'
                  + 'dz={:.0f}m, r={:.2f}m, '.format(z[iz], r[ir])
                  + 'Rytov={:.2f}, lenslet Rytov={:.2f}\n'.format(
                      rytov_out, rytov_lenslet_out)
                  + 'max intensity {:.2f}, min intensity {:.2f}'.format(
                      I_out.max(), I_out.min()))
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()

        plt.pcolor(np.arange(-prN, prN + 1)*d_out,
                   np.arange(-prN, prN + 1)*d_out,
                   phase_out[int(N/2) - prN
                                     : int(N/2) + prN,
                                     int(N/2) - prN
                                     : int(N/2) + prN],
                   shading='auto', snap=True)
        plt.colorbar()
        plt.vlines((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
                   ymin=-prNN*d_out*lenslet_pixel,
                   ymax=prNN*d_out*lenslet_pixel,
                   color='m')
        plt.hlines((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
                   xmin=-prNN*d_out*lenslet_pixel,
                   xmax=prNN*d_out*lenslet_pixel,
                   color='m')
        plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
                   (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
                   lenslet_slope_phase_out[int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN,
                                           int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN, 0],
                   lenslet_slope_phase_out[int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN,
                                           int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN, 1],
                   color='r')
        plt.title('Phase (rad) over lenslet phase slope,\n'
                  + 'dz={:.0f}m, r={:.2f}m,\n'.format(z[iz], r[ir])
                  + 'Rytov={:.2f}, lenslet Rytov={:.2f}'.format(
                      rytov_out, rytov_lenslet_out))
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()

        # Averaged
        plt.plot(lenslet_I_out.flatten(),
                 size_lenslet_slope_phase_out.flatten(),
                 ls='', marker=',')
        plt.xscale('log')
        plt.xlabel('Intensity')
        plt.ylabel('Slope size (rad)')
        plt.title('Lenslet intensity vs phase slope size (rad),\n'
                  + 'dz={:.2f}m, r={:.2f}m,\n'.format(z[iz], r[ir])
                  + 'Rytov={:.2f}, lenslet Rytov={:.2f}'.format(
                      rytov_out, rytov_lenslet_out))
        plt.show()

        plt.pcolor(np.arange(-prNN, prNN + 1)*d_out*lenslet_pixel,
                   np.arange(-prNN, prNN + 1)*d_out*lenslet_pixel,
                   np.log(lenslet_I_WFS_detector_out[int(num_lenslet/2) - prNN
                                 : int(num_lenslet/2) + prNN,
                                 int(num_lenslet/2) - prNN
                                 : int(num_lenslet/2) + prNN])/np.log(10.),
                   shading='auto', vmin=I_min, vmax=I_max, snap=True)
        plt.colorbar()
        plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
                   (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
                   lenslet_slope_phase_out[int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN,
                                           int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN, 0],
                   lenslet_slope_phase_out[int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN,
                                           int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN, 1],
                   color='r')
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Lenslet log10 intensity over lenslet phase slope,\n'
                  + 'dz={:.0f}m, r={:.2f}m, '.format(z[iz], r[ir])
                  + 'Rytov={:.2f}, lenslet Rytov={:.2f}\n'.format(
                      rytov_out, rytov_lenslet_out)
                  + 'max intensity {:.2f}, '.format(
                      lenslet_I_WFS_detector_out.max())
                  + 'min intensity {:.2f}'.format(
                      lenslet_I_WFS_detector_out.min()))
        plt.show()

        plt.pcolor((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
                   (np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
                   size_lenslet_slope_phase_out[int(num_lenslet/2) - prNN
                                                : int(num_lenslet/2) + prNN,
                                                int(num_lenslet/2) - prNN
                                                : int(num_lenslet/2) + prNN],
                   shading='auto', snap=True)
        plt.colorbar()
        plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
                   (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
                   lenslet_slope_phase_out[int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN,
                                           int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN, 0],
                   lenslet_slope_phase_out[int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN,
                                           int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN, 1],
                   color='r')
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Lenslet phase slope size (rad),\n' 
                  + 'dz={:.0f}m, r={:.2f}m,\n'.format(z[iz], r[ir])
                  + 'Rytov={:.2f}, lenslet Rytov={:.2f}'.format(
                      rytov_out, rytov_lenslet_out))
        plt.show()

        # Cheat SHWFS
        plt.pcolor(np.arange(-prNN, prNN + 1)*d_out*lenslet_pixel,
                   np.arange(-prNN, prNN + 1)*d_out*lenslet_pixel,
                   np.log(lenslet_I_out[int(num_lenslet/2) - prNN :
                                        int(num_lenslet/2) + prNN,
                                        int(num_lenslet/2) - prNN :
                                            int(num_lenslet/2) + prNN])
                       /np.log(10.),
                   shading='auto', vmin=I_min, vmax=I_max, snap=True)
        plt.colorbar()
        plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
                   (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
                   lenslet_weight_slope_phase_out[int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN,
                                           int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN, 0],
                   lenslet_weight_slope_phase_out[int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN,
                                           int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN, 1],
                   color='r')
        plt.axis('equal')
        plt.xlabel('x (pixel)')
        plt.ylabel('y (pixel)')
        plt.title('Lenslet log10 intensity over '
                  +'weighted phase slope,\n'
                  + 'dz={:.2f}m, r={:.2f}m, '.format(z[iz], r[ir])
                  + 'Rytov={:.2f}, '.format(rytov_out)
                  + 'lenslet Rytov={:.2f},\n'.format(rytov_lenslet_out)
                  + 'max intensity {:.2f}, '.format(
                      lenslet_I_out.max())
                  + 'min intensity {:.2f}'.format(
                      lenslet_I_out.min()))
        plt.show()

        plt.pcolor((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
                   (np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
                   size_lenslet_weight_slope_phase_out[int(num_lenslet/2)
                                                       - prNN
                                                       : int(num_lenslet/2)
                                                       + prNN,
                                                       int(num_lenslet/2)
                                                       - prNN
                                                       : int(num_lenslet/2)
                                                       + prNN],
                   shading='auto', snap=True)
        plt.colorbar()
        plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
                   (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
                   lenslet_weight_slope_phase_out[int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN,
                                           int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN, 0],
                   lenslet_weight_slope_phase_out[int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN,
                                           int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN, 1],
                   color='r')
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Lenslet weighted phase slope (rad),\n'
                  + 'dz={:.2f}m, r={:.2f}m,\n'.format(z[iz], r[ir])
                  +'Rytov={:.2f}, lenslet Rytov={:.2f}'.format(
                      rytov_out, rytov_lenslet_out))
        plt.show()

        # Proper SHWFS #
        plt.plot(lenslet_I_WFS_detector_out.flatten(),
                 size_projected_slope_WFS_out.flatten(),
                 ls='', marker=',')
        plt.xscale('log')
        plt.xlabel('Intensity')
        plt.ylabel('Slope size (rad)')
        plt.title('WFS intensity vs phase slope size (rad),\n'
                  + 'dz=0m, r={:.2f}m,\nRytov={:.2f}, WFS Rytov={:.2f}'.format(
                      r[ir], rytov_in, rytov_WFS_in))
        plt.show()

        plt.pcolor(np.arange(-prN, prN + 1)*d_out,
                   np.arange(-prN, prN + 1)*d_out,
                   np.log(I_WFS_detector_out[int(N/2) - prN : int(N/2) + prN,
                         int(N/2) - prN : int(N/2) + prN]/I_norm)/np.log(10.),
                   shading='auto', vmin=I_min, vmax=I_max, snap=True)
        plt.colorbar()
        plt.vlines((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
                   ymin=-prNN*d_out*lenslet_pixel,
                   ymax=prNN*d_out*lenslet_pixel,
                   color='m')
        plt.hlines((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
                   xmin=-prNN*d_out*lenslet_pixel,
                   xmax=prNN*d_out*lenslet_pixel,
                   color='m')
        plt.axis('equal')
        plt.xlabel('x (pixel)')
        plt.ylabel('y (pixel)')
        plt.title('WFS log10 image over phase slope,\n'
                  + 'dz={:.0f}m, r={:.2f}m, '.format(z[iz], r[ir])
                  + 'Rytov={:.2f}, '.format(rytov_out)
                  + 'WFS Rytov={:.2f},\n'.format(rytov_WFS_out)
                  + 'max intensity {:.2f}, '.format(
                      lenslet_I_WFS_detector_out.max())
                  + 'min intensity {:.2f}'.format(
                      lenslet_I_WFS_detector_out.min()))
        plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
                   (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
                   slope_WFS_out[int(num_lenslet/2) - prNN
                                 : int(num_lenslet/2) + prNN,
                                 int(num_lenslet/2) - prNN
                                 : int(num_lenslet/2) + prNN, 0],
                   slope_WFS_out[int(num_lenslet/2) - prNN
                                 : int(num_lenslet/2) + prNN,
                                 int(num_lenslet/2) - prNN
                                 : int(num_lenslet/2) + prNN, 1],
                   color='r')
        plt.show()

        plt.pcolor(np.arange(-prNN, prNN + 1)*d_out*lenslet_pixel,
                   np.arange(-prNN, prNN + 1)*d_out*lenslet_pixel,
                   np.log(lenslet_I_WFS_detector_out[int(num_lenslet/2)
                                                     - prNN
                                                     : int(num_lenslet/2)
                                                     + prNN,
                                                     int(num_lenslet/2)
                                                     - prNN
                                                     : int(num_lenslet/2)
                                                     + prNN])
                       /np.log(10.),
                   shading='auto', vmin=I_min, vmax=I_max, snap=True)
        plt.colorbar()
        plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
                   (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
                   projected_slope_WFS_out[int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN,
                                           int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN, 0],
                   projected_slope_WFS_out[int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN,
                                           int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN, 1],
                   color='r')
        plt.axis('equal')
        plt.xlabel('x (pixel)')
        plt.ylabel('y (pixel)')
        plt.title('WFS log10 total intensity over phase slope,\n'
                  + 'dz=0m, r={:.2f}m, '.format(r[ir])
                  + 'Rytov={:.2f}, '.format(rytov_out)
                  + 'WFS Rytov={:.2f},\n'.format(rytov_WFS_out)
                  + 'max intensity {:.2f}, '.format(
                      lenslet_I_WFS_detector_out.max())
                  + 'min intensity {:.2f}'.format(
                      lenslet_I_WFS_detector_out.min()))
        plt.show()

        plt.pcolor((np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
                   (np.arange(-prNN, prNN + 1))*d_out*lenslet_pixel,
                   size_projected_slope_WFS_out[int(num_lenslet/2) - prNN
                                                : int(num_lenslet/2) + prNN,
                                                int(num_lenslet/2) - prNN
                                                : int(num_lenslet/2) + prNN],
                   shading='auto', snap=True)
        plt.colorbar()
        plt.quiver((np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
                   (np.arange(-prNN, prNN) + 0.5)*d_out*lenslet_pixel,
                   projected_slope_WFS_out[int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN,
                                           int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN, 0],
                   projected_slope_WFS_out[int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN,
                                           int(num_lenslet/2) - prNN
                                           : int(num_lenslet/2) + prNN, 1],
                   color='r')
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('WFS phase slope (rad),\n'
                  + 'dz=0m, r={:.2f}m,\n'.format(r[ir])
                  + 'Rytov={:.2f}, WFS Rytov={:.2f}'.format(
                      rytov_out, rytov_WFS_out))
        plt.show()

        print(r[ir], z[iz], rytov_out, rytov_WFS_out)

###############################################################################
########################################################################

###############################################################################
# End of File ##########################################################
###############################################################################