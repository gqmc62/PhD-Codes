
###############################################################################
# Start of file ########################################################
###############################################################################
"""
Contains list of plotting functions for the sake of phasescreen study.

Implemented by Puttiwat Kongkaew (Geng/Greg) during his study
for Ph.D. in Physics (Advanced Instrumentation)
at Centre for Advanced Instrumentation, Department of Physics,
Durham University,United Kingdom from October 2021 to
(ETG) September 2024.
I will try to keep everything under my universal namings.
units, etc. to keep confusions at minimal.

This file contains
    1) Functions that only does plotting of certain study purpose.
    2) Active codes that procure all data for plottings (if needed).
"""
###############################################################################
# List of plots collected within this file #########################
###############################################################################
# 1) Power spectrum density. #
#       psd_function()
#       Create 2D power spectrum density (PSD).
# 2) Structure function. #
#       structure_function()
#       Create a structure function of the given screen.
# 3) Gram-Schmidt orthonormalization. #
#       gram_schmidt_process()
#       Orthonormalize any given set of functions with Gram-Schmidt process.
# 4) Test orthonormality. #
#       orthonormality_test()
#       Test out orthonormality of the given modesby doing dot product.
# 5) Mode strength. #
#       zernike_modes_strength()
#       Measure variance of each zernike mode off sub-harmonic von Karman.
###############################################################################
########################################################################

#
# Imports #
#
import my_phasescreen as mps
import numpy as np
from matplotlib import pyplot as plt

#
# Preferences #
#
color_map = plt.cm.get_cmap('viridis')
reversed_color_map = color_map.reversed()

###############################################################################
########################################################################

# 1) Plot Power spectrum density
def plot1(fx, fy, PSD, N):
    """Show log10 of Power Spectrum Density in colormap."""
    # # Plot #
    plt.pcolor(fx[int(N/2) - 128 : int(N/2) + 128,
                  int(N/2) - 128 : int(N/2) + 128],
               fy[int(N/2) - 128 : int(N/2) + 128,
                  int(N/2) - 128 : int(N/2) + 128],
               np.log(PSD[int(N/2) - 128 : int(N/2) + 128,
                              int(N/2) - 128 : int(N/2) + 128])/np.log(10),
               shading='auto')
    plt.title('Log_10 of power spectrum density (m$^2$)')
    plt.colorbar()
    plt.xlabel('x (1/m)')
    plt.ylabel('y (1/m)')
    plt.axis('equal')
    plt.show()

    return

###############################################################################
########################################################################

def plot2(fx, PSD, N, L0):
    """Show cross section of power spectrum density in log-log plot."""
    plt.plot(fx[int(N/2), int(N/2 + 1) : ], PSD[int(N/2), int(N/2 + 1) : ],
             label='Power spectrum density', color='r')
    plt.vlines(1./L0, ymin=PSD[int(N/2 + 1), int(N/2)],
               ymax=PSD[int(N/2 + 1), -1], color='b', label='L0')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.title('Cross section of power spectrum density (PSD)')
    plt.xlabel('Spatial frequency (1/m)')
    plt.ylabel('Power spectrum density (m$^2$)')
    plt.show()
    return

###############################################################################
########################################################################

r0 = 0.05
N = 1024*2
delta = 5e-3
L0 = 20
l0 = 1.5e-10
fx, fy, PSD = mps.psd_function(r0, N, delta, L0, l0)

plot1(fx, fy, PSD, N)
plot2(fx, PSD, N, L0)

###############################################################################
########################################################################

###############################################################################
# End of File ##########################################################
###############################################################################
